#!/usr/bin/env python3
# inspect_gate_gradients.py
# Diagnostic: soft-oracle KL loss gradient on the gate (single forward/backward).
#
# Usage:
#   python inspect_gate_gradients.py \
#       --config <exp11_config.yaml> \
#       --ce_path <ce_ckpt> --la_path <la_ckpt> --bs_path <bs_ckpt> \
#       --gate_ckpt <gate_checkpoint_*.pth>

import os
import sys
import argparse
import re

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

# 1. Parse our custom arguments FIRST and remove them from sys.argv
custom_parser = argparse.ArgumentParser(add_help=False)
custom_parser.add_argument('--ce_path', type=str, required=True)
custom_parser.add_argument('--la_path', type=str, required=True)
custom_parser.add_argument('--bs_path', type=str, required=True)
custom_parser.add_argument('--gate_ckpt', type=str, required=True)
custom_args, remaining_argv = custom_parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_argv

# 2. NOW import and call get_args() (handles --config)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from imbalanceddl.utils.config import get_args
from imbalanceddl.dataset.imbalance_dataset import ImbalancedDataset
from imbalanceddl.utils.debug.models import ExpertEnsemble, GateMLP
from imbalanceddl.utils.gate_features import (
    calibrate_expert_probs, gate_input_dim,
)


class GateGradientInspector:
    """Inspects soft-oracle KL gradient pressure on the gate router.

    Runs one batch through the exact ``train_one_epoch`` forward pass
    (probability+feature embeddings at T=1.0, soft-oracle target at the
    checkpoint temperature) and backprops the soft-oracle KL loss so the
    ``gate.fc.weight`` gradient norm and routing weights can be reported.
    """

    NUM_EXPERTS = 3

    def __init__(self, cfg, ce_path, la_path, bs_path, gate_ckpt_path,
                 device):
        self.cfg = cfg
        self.device = device
        self.la_tau = self._resolve_la_tau(cfg, la_path)

        self.model = ExpertEnsemble(
            cfg, device, {'CE': ce_path, 'LA': la_path, 'BS': bs_path}
        )
        self.gate = GateMLP(input_dim=gate_input_dim(cfg.num_classes),
                            num_experts=3).to(device)
        gate_ckpt = torch.load(gate_ckpt_path, map_location='cpu',
                               weights_only=False)
        self.gate.load_state_dict(gate_ckpt['gate_state_dict'])
        self.T = gate_ckpt.get('temperature', 1.0)
        # Replicate training-time forward: BN uses batch statistics.
        self.gate.train()

    def _resolve_la_tau(self, cfg, la_path):
        """LA prior temperature: config value, else LA filename, else 1.5."""
        tau = getattr(cfg, 'la_tau', None)
        if tau is not None:
            return tau
        match_tau = re.search(r't([\d\.]+)', os.path.basename(la_path))
        return float(match_tau.group(1)) if match_tau else 1.5

    def _get_probs(self, logits_list, T):
        """Calibrated expert posteriors (shared math)."""
        return calibrate_expert_probs(
            logits_list, self.cfg.cls_num_list, self.la_tau, T
        )

    def run(self, images, labels):
        """Forward one batch, backprop the soft-oracle KL loss."""
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        with torch.no_grad():
            logits_list, embeddings = self.model(images)
        probs = self._get_probs(logits_list, self.T)

        gate_logits = self.gate(embeddings)
        weights = F.softmax(gate_logits, dim=1)

        B = labels.size(0)
        true_probs = torch.stack(
            [p[torch.arange(B), labels] for p in probs], dim=1
        )
        tau_oracle = getattr(self.cfg, 'gate_oracle_tau', 0.2)
        soft_target = F.softmax(true_probs / tau_oracle, dim=1)
        log_weights = F.log_softmax(gate_logits, dim=1)
        loss = F.kl_div(log_weights, soft_target, reduction='batchmean')

        self.gate.zero_grad()
        loss.backward()
        grad_norm = self.gate.fc.weight.grad.norm().item()

        self._print_report(loss.item(), grad_norm, weights)

    def _print_report(self, loss_value, grad_norm, weights):
        fc_weight = self.gate.fc.weight.detach()
        avg_weights = weights.mean(dim=0).tolist()

        print("\n" + "=" * 80)
        print("GATE GRADIENT INSPECTION: SOFT-ORACLE KL LOSS")
        print("=" * 80)
        print(f"Gate temperature T      : {self.T}")
        print(f"LA tau                  : {self.la_tau}")
        print(f"Batch size              : {weights.size(0)}")
        print("-" * 80)
        print(f"soft-oracle KL loss     : {loss_value:.6f}")
        print(f"||grad|| (fc.weight)    : {grad_norm:.6f}")
        print("-" * 80)
        print(f"fc.weight mean          : {fc_weight.mean().item():+.6f}")
        print(f"fc.weight std           : {fc_weight.std().item():.6f}")
        print(f"avg routing weights     : CE={avg_weights[0]:.4f} "
              f"| LA={avg_weights[1]:.4f} | BS={avg_weights[2]:.4f}")
        print("=" * 80)


def main():
    cfg = get_args()
    if cfg.dataset == 'cifar100':
        cfg.num_classes = 100

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    dataset = ImbalancedDataset(cfg, cfg.dataset, augmentation='none')
    train_dataset, val_dataset = dataset.train_val_sets

    train_targets = np.array(train_dataset.targets)
    cfg.cls_num_list = np.bincount(
        train_targets, minlength=cfg.num_classes).tolist()

    val_targets = np.array(val_dataset.targets)
    val_indices = np.arange(len(val_targets))
    _, test_idx = train_test_split(val_indices, test_size=0.8,
                                   stratify=val_targets,
                                   random_state=cfg.seed)
    test_loader = DataLoader(Subset(val_dataset, test_idx), batch_size=128,
                             shuffle=False, num_workers=4)

    print("\n[INFO] Loading expert ensemble and gate...")
    inspector = GateGradientInspector(
        cfg, custom_args.ce_path, custom_args.la_path,
        custom_args.bs_path, custom_args.gate_ckpt, device
    )

    images, labels = next(iter(test_loader))
    print(f"[INFO] Evaluating gradients on one batch "
          f"({images.size(0)} samples)...\n")
    inspector.run(images, labels)


if __name__ == "__main__":
    main()
