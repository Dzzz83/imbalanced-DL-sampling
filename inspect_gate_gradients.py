#!/usr/bin/env python3
# inspect_gate_gradients.py
# Diagnostic: replicate the checkpoint's training loss and inspect gate
# gradients (single forward/backward).
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
from sklearn.isotonic import IsotonicRegression
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
    calibrate_expert_probs, gate_input_dim, build_mixture,
    build_oracle_target,
)


class GateGradientInspector:
    """Replicates the checkpoint's training loss and reports gradient pressure.

    The loss is rebuilt from the checkpoint's metadata (target mode, tau,
    per-expert temperatures, k, mixture space) so the reported gradient is
    the one the gate actually trained with.
    """

    NUM_EXPERTS = 3

    def __init__(self, cfg, ce_path, la_path, bs_path, gate_ckpt_path,
                 device):
        self.cfg = cfg
        self.device = device
        self.la_tau = self._resolve_la_tau(cfg, la_path)

        gate_ckpt = torch.load(gate_ckpt_path, map_location='cpu',
                               weights_only=False)
        # Recipe metadata (with safe defaults for old checkpoints).
        self.target_mode = gate_ckpt.get('target_mode', 'logprob')
        self.tau = gate_ckpt.get('tau', 0.2)
        self.T = gate_ckpt.get('temperature', 1.0)
        self.expert_T = list(gate_ckpt.get('expert_temps', [1.0, 1.0, 1.0]))
        self.k = gate_ckpt.get('k', getattr(cfg, 'routing_sparsity', 2))
        self.space = gate_ckpt.get('mix_space', getattr(cfg, 'mix_space', 'logit'))
        self.weight_floor = gate_ckpt.get('weight_floor', 0.0)
        self.gate_temp = gate_ckpt.get('gate_temp', 1.0)
        self.norm_blocks = gate_ckpt.get('norm_blocks', True)
        self.calibrators = None  # fitted from a loader when target is correctness

        self.model = ExpertEnsemble(
            cfg, device, {'CE': ce_path, 'LA': la_path, 'BS': bs_path},
            expert_T=self.expert_T, normalize_blocks=self.norm_blocks,
        )
        self.gate = GateMLP(input_dim=gate_input_dim(cfg.num_classes),
                            num_experts=3).to(device)
        self.gate.load_state_dict(gate_ckpt['gate_state_dict'])
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
        """Calibrated expert posteriors (shared math, per-expert temps)."""
        return calibrate_expert_probs(
            logits_list, self.cfg.cls_num_list, self.la_tau,
            T, per_expert_T=self.expert_T,
        )

    def fit_calibrators(self, loader, max_batches=8):
        """Fit per-expert correctness calibrators (target_mode='correctness')."""
        confs = [[], [], []]
        corrects = [[], [], []]
        with torch.no_grad():
            for i, (images, labels) in enumerate(loader):
                if i >= max_batches:
                    break
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                logits_list, _ = self.model(images)
                probs = self._get_probs(logits_list, 1.0)
                for j, p in enumerate(probs):
                    confs[j].append(p.max(dim=1).values.cpu().numpy())
                    corrects[j].append(
                        (p.argmax(dim=1) == labels).cpu().numpy().astype(float))
        self.calibrators = []
        for j in range(3):
            conf = np.concatenate(confs[j])
            correct = np.concatenate(corrects[j])
            if correct.sum() < 20:
                self.calibrators.append(lambda c: np.clip(c, 0.05, 0.95))
            else:
                iso = IsotonicRegression(out_of_bounds='clip',
                                         y_min=0.02, y_max=0.98)
                iso.fit(conf, correct)
                self.calibrators.append(iso)

    def _correctness_target(self, probs):
        confs = torch.stack([p.max(dim=1).values for p in probs], dim=1)
        t = torch.zeros_like(confs)
        for j, cal in enumerate(self.calibrators):
            vals = cal(confs[:, j].cpu().numpy())
            t[:, j] = torch.from_numpy(np.asarray(vals, dtype=np.float32)).to(confs.device)
        return t / t.sum(dim=1, keepdim=True)

    def run(self, images, labels):
        """Forward one batch, backprop the checkpoint's training loss."""
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        with torch.no_grad():
            logits_list, embeddings = self.model(images)
        probs = self._get_probs(logits_list, self.T)

        gate_logits = self.gate(embeddings)
        weights = F.softmax(gate_logits, dim=1)
        B = labels.size(0)

        if self.target_mode == 'mix_nll':
            p_mix = build_mixture(
                logits_list, weights, self.cfg.cls_num_list, self.la_tau,
                T=self.T, per_expert_T=self.expert_T, k=self.k,
                space=self.space, weight_floor=self.weight_floor,
            )
            loss = F.nll_loss(torch.log(p_mix.clamp_min(1e-12)), labels)
            loss_name = 'mixture NLL (logit space)'
        elif self.target_mode == 'logprob':
            true_probs = torch.stack(
                [p[torch.arange(B), labels] for p in probs], dim=1
            )
            soft_target = build_oracle_target(true_probs, self.tau,
                                              space='logprob')
            log_weights = F.log_softmax(gate_logits, dim=1)
            loss = F.kl_div(log_weights, soft_target, reduction='batchmean')
            loss_name = 'soft-oracle KL (log-space target)'
        else:  # correctness
            if self.calibrators is None:
                raise RuntimeError(
                    "target_mode='correctness' requires fit_calibrators(loader)"
                    " before run().")
            soft_target = self._correctness_target(probs)
            log_weights = F.log_softmax(gate_logits, dim=1)
            loss = F.kl_div(log_weights, soft_target, reduction='batchmean')
            loss_name = 'correctness KL (L2D-style)'

        self.gate.zero_grad()
        loss.backward()
        grad_norm = self.gate.fc.weight.grad.norm().item()

        self._print_report(loss.item(), grad_norm, weights, loss_name)

    def _print_report(self, loss_value, grad_norm, weights, loss_name):
        fc_weight = self.gate.fc.weight.detach()
        avg_weights = weights.mean(dim=0).tolist()

        print("\n" + "=" * 80)
        print("GATE GRADIENT INSPECTION")
        print("=" * 80)
        print(f"Loss                   : {loss_name}")
        print(f"Gate temperature T      : {self.T}")
        print(f"LA tau                  : {self.la_tau}")
        print(f"Per-expert temps        : {[f'{t:.3f}' for t in self.expert_T]}")
        print(f"k / mix_space           : {self.k} / {self.space}")
        print(f"Batch size              : {weights.size(0)}")
        print("-" * 80)
        print(f"loss                    : {loss_value:.6f}")
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
    if inspector.target_mode == 'correctness':
        print("[INFO] Fitting correctness calibrators on the test set...")
        inspector.fit_calibrators(test_loader)

    images, labels = next(iter(test_loader))
    print(f"[INFO] Evaluating gradients on one batch "
          f"({images.size(0)} samples)...\n")
    inspector.run(images, labels)


if __name__ == "__main__":
    main()
