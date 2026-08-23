#!/usr/bin/env python3
# inspect_gate_gradients.py
# Exp 11 diagnostic: NLL vs Switch-aux loss gradient balance on the gate.
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


class GateGradientInspector:
    """Compares NLL vs Switch-aux gradient pressure on the gate router.

    Runs one batch through the exact ``train_one_epoch`` forward pass
    (probability-space embeddings at T=1.0, mixture at the checkpoint
    temperature) and performs two separate backward passes so the L2 grad
    norms of ``gate.fc.weight`` can be attributed to each loss individually.
    """

    AUX_ALPHA = 0.01
    NUM_EXPERTS = 3

    def __init__(self, cfg, ce_path, la_path, bs_path, gate_ckpt_path,
                 device):
        self.cfg = cfg
        self.device = device
        self.la_tau = self._resolve_la_tau(cfg, la_path)

        self.model = ExpertEnsemble(
            cfg, device, {'CE': ce_path, 'LA': la_path, 'BS': bs_path}
        )
        self.gate = GateMLP(input_dim=300, num_experts=3).to(device)
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
        """Calibrated expert posteriors (same math as get_probs)."""
        p_ce = F.softmax(logits_list[0] / T, dim=1)

        cls_num_list = torch.tensor(
            self.cfg.cls_num_list, device=self.device, dtype=torch.float32
        )
        log_prior = torch.log(cls_num_list / cls_num_list.sum() + 1e-12)
        p_la = F.softmax((logits_list[1] + self.la_tau * log_prior) / T, dim=1)

        log_spc = torch.log(cls_num_list + 1e-12)
        p_bs = F.softmax((logits_list[2] + log_spc) / T, dim=1)

        return [p_ce, p_la, p_bs]

    def run(self, images, labels):
        """Forward one batch, backprop each loss, print gradient norms."""
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        with torch.no_grad():
            logits_list, embeddings = self.model(images)
        probs = self._get_probs(logits_list, self.T)

        gate_logits = self.gate(embeddings)
        weights = F.softmax(gate_logits, dim=1)

        mix_prob = torch.zeros_like(probs[0])
        for i in range(self.NUM_EXPERTS):
            mix_prob += weights[:, i:i+1] * probs[i]

        nll_loss = F.nll_loss(torch.log(mix_prob + 1e-8), labels)
        aux_loss = (self.AUX_ALPHA * self.NUM_EXPERTS
                    * torch.sum(weights.mean(dim=0) ** 2))

        # --- Separate backward passes, zeroing grads in between ---
        self.gate.zero_grad()
        nll_loss.backward(retain_graph=True)
        nll_grad_norm = self.gate.fc.weight.grad.norm().item()

        self.gate.zero_grad()
        aux_loss.backward()
        aux_grad_norm = self.gate.fc.weight.grad.norm().item()

        self._print_report(nll_loss.item(), aux_loss.item(),
                           nll_grad_norm, aux_grad_norm, weights)

    def _print_report(self, nll_value, aux_value, nll_grad, aux_grad, weights):
        fc_weight = self.gate.fc.weight.detach()
        avg_weights = weights.mean(dim=0).tolist()
        ratio = aux_grad / (nll_grad + 1e-12)

        print("\n" + "=" * 80)
        print("GATE GRADIENT INSPECTION (Exp 11): NLL vs AUX")
        print("=" * 80)
        print(f"Gate temperature T      : {self.T}")
        print(f"LA tau                  : {self.la_tau}")
        print(f"Batch size              : {weights.size(0)}")
        print("-" * 80)
        print(f"nll_loss (mixture NLL)  : {nll_value:.6f}")
        print(f"aux_loss (load-balance) : {aux_value:.6f}")
        print("-" * 80)
        print(f"||grad nll|| (fc.weight): {nll_grad:.6f}")
        print(f"||grad aux|| (fc.weight): {aux_grad:.6f}")
        print(f"aux/nll grad ratio      : {ratio:.3f}x")
        print("-" * 80)
        print(f"fc.weight mean          : {fc_weight.mean().item():+.6f}")
        print(f"fc.weight std           : {fc_weight.std().item():.6f}")
        print(f"avg routing weights     : CE={avg_weights[0]:.4f} "
              f"| LA={avg_weights[1]:.4f} | BS={avg_weights[2]:.4f}")
        print("[INFO] fc.weight std -> 0 and weights ~ 0.33/0.33/0.33 "
              "indicate collapse to uniform routing.")
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
