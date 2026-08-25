#!/usr/bin/env python3
"""
Feature Collinearity Diagnostic for MoE Gate Routing.

This script independently verifies the central hypothesis of Exp 14:
the three calibrated 100-dim probability distributions (CE, LA, BS)
are near-collinear, making per-sample routing impossible regardless of
target or architecture choice.

It computes:
  1. Pairwise cross-block correlation (CE vs LA, CE vs BS, LA vs BS)
     — the key collinearity metric.
  2. Within-block singular-value / effective-rank analysis.
  3. PCA variance explained by the top components.
  4. Block-wise mean probability profiles to show how similar the three
     distributions actually are at the per-class level.

Usage:
  cd /home/dzzz83/Documents/code/imbalanced-DL-sampling
  .venv/bin/python3 diagnose_feature_collinearity.py
"""

import os
import sys
import argparse
import re

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

# Project imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from imbalanceddl.utils.config import get_args
from imbalanceddl.dataset.imbalance_dataset import ImbalancedDataset
from imbalanceddl.utils.gate_features import (
    calibrate_expert_probs, build_gate_input, gate_input_dim,
)
from imbalanceddl.net.network import build_model


# ────────────────────────────────────────────── #
#  Minimal ExpertEnsemble (inference only)       #
# ────────────────────────────────────────────── #
class FrozenExpertEnsemble(torch.nn.Module):
    """Loads the three frozen experts and computes gate features."""

    def __init__(self, cfg, device, ckpt_paths, la_tau=1.5):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.la_tau = la_tau
        self.experts = torch.nn.ModuleList()
        for name, path in ckpt_paths.items():
            print(f"  Loading expert {name} from {path}")
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            has_bias = ckpt.get('bias', False)
            model = build_model(cfg)
            actual = model.module if isinstance(model, torch.nn.DataParallel) else model
            actual.classifier = torch.nn.Linear(
                actual.feature_len, actual.num_classes, bias=has_bias
            ).to(device)
            state = {k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()}
            actual.load_state_dict(state)
            for p in actual.parameters():
                p.requires_grad = False
            actual.eval()
            self.experts.append(actual.to(device))

    @torch.no_grad()
    def forward(self, x):
        logits_list = [expert(x)[0] for expert in self.experts]
        probs = calibrate_expert_probs(
            logits_list, self.cfg.cls_num_list, self.la_tau,
            T=1.0, per_expert_T=[1.5, 1.2, 1.5],  # Exp 14 balanced temps
        )
        # Build features WITH freq_features=True (the 316-dim config)
        embeddings = build_gate_input(
            probs, normalize_blocks=True,
            cls_num_list=self.cfg.cls_num_list,
        )
        return logits_list, embeddings, probs


# ────────────────────────────────────────────── #
#  Correlation Analysis                           #
# ────────────────────────────────────────────── #
def block_correlation(embeddings, block_size=100, num_blocks=3):
    """Pairwise correlation between each pair of 100-dim probability blocks."""
    blocks = [embeddings[:, b*block_size:(b+1)*block_size] for b in range(num_blocks)]
    results = {}
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            # For each sample, compute the correlation between its two 100-dim vectors
            corrs = []
            for s in range(embeddings.size(0)):
                vi = blocks[i][s].cpu().numpy()
                vj = blocks[j][s].cpu().numpy()
                if np.std(vi) > 1e-8 and np.std(vj) > 1e-8:
                    c = np.corrcoef(vi, vj)[0, 1]
                else:
                    c = 0.0
                corrs.append(c)
            name = f"Block{i}_vs_Block{j}"
            results[name] = {
                'mean_corr': float(np.mean(corrs)),
                'std_corr': float(np.std(corrs)),
                'min_corr': float(np.min(corrs)),
                'max_corr': float(np.max(corrs)),
                'pct_gt_09': float(np.mean(np.array(corrs) > 0.9) * 100),
                'pct_gt_095': float(np.mean(np.array(corrs) > 0.95) * 100),
            }
    return results


def feature_covariance_analysis(embeddings):
    """SVD-based analysis of the full feature covariance."""
    # Center the features
    X = embeddings - embeddings.mean(dim=0, keepdim=True)
    # Compute SVD
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    S = S.cpu().numpy()
    var_explained = S ** 2 / (S ** 2).sum()
    cum_var = np.cumsum(var_explained)

    # Effective rank (participation ratio)
    eff_rank = (S.sum() ** 2) / (S ** 2).sum()

    return {
        'singular_values': S[:20].tolist(),
        'var_explained_top5': float(var_explained[:5].sum()),
        'var_explained_top10': float(var_explained[:10].sum()),
        'var_explained_top20': float(var_explained[:20].sum()),
        'effective_rank': float(eff_rank),
        'condition_number': float(S[0] / S[-1]) if S[-1] > 1e-10 else float('inf'),
    }


def per_expert_mean_profile(probs, cls_num_list):
    """Average probability profile per expert across all samples."""
    profiles = {}
    for name, p in zip(['CE', 'LA', 'BS'], probs):
        mean_p = p.mean(dim=0).cpu().numpy()
        profiles[name] = mean_p
    return profiles


def cross_block_mean_corr(embeddings, block_size=100):
    """Simpler: compute the matrix correlation of the block MEANS."""
    blocks = [embeddings[:, b*block_size:(b+1)*block_size] for b in range(3)]
    names = ['CE_prob_block', 'LA_prob_block', 'BS_prob_block']
    # Concatenate over samples and compute correlation of the full matrices
    # This gives us the correlation of the 100-dimensional distributions
    # across all samples jointly.
    matrix_corrs = {}
    for i in range(3):
        for j in range(i + 1, 3):
            # Flatten to (n_samples*100,) vectors
            vi = blocks[i].reshape(-1).cpu().numpy()
            vj = blocks[j].reshape(-1).cpu().numpy()
            c = float(np.corrcoef(vi, vj)[0, 1])
            matrix_corrs[f"{names[i]}_vs_{names[j]}"] = c
    return matrix_corrs


def within_block_svd(embeddings, block_size=100):
    """SVD within each probability block to measure its intrinsic dim."""
    results = {}
    for b in range(3):
        block = embeddings[:, b*block_size:(b+1)*block_size]
        X = block - block.mean(dim=0, keepdim=True)
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        S = S.cpu().numpy()
        var_ex = S ** 2 / (S ** 2).sum()
        eff_rank = (S.sum() ** 2) / (S ** 2).sum()
        results[f'Block{b}_effective_rank'] = float(eff_rank)
        results[f'Block{b}_top5_var'] = float(var_ex[:5].sum())
        results[f'Block{b}_top1_var'] = float(var_ex[0])
    return results


# ────────────────────────────────────────────── #
#  Main                                           #
# ────────────────────────────────────────────── #
def main():
    # Minimal arg setup
    parser = argparse.ArgumentParser(description='Feature collinearity diagnostic.')
    parser.add_argument('--ce_path', type=str, default=None)
    parser.add_argument('--la_path', type=str, default=None)
    parser.add_argument('--bs_path', type=str, default=None)
    parser.add_argument('--config', type=str, default='config/what_to_train/cifar100/_gate_train.yaml')
    parser.add_argument('--n_samples', type=int, default=500,
                        help='Number of test samples to process (for speed on CPU)')
    args = parser.parse_args()

    # Load config via project's get_args
    sys.argv = [sys.argv[0], '--config', args.config]
    cfg = get_args()
    if cfg.dataset == 'cifar100':
        cfg.num_classes = 100

    device = torch.device('cpu')
    print(f"Device: {device}")
    print(f"Config: {args.config}")

    # Resolve expert paths
    ckpt_dir = 'checkpoint/experts_sweep_cifar100_calib'
    default_paths = {
        'CE': os.path.join(ckpt_dir, 'expert_CE_biasFalse_ls0.0_epoch162.pth'),
        'LA': os.path.join(ckpt_dir, 'expert_LA_biasFalse_ls0.0_t1.5_epoch161.pth'),
        'BS': os.path.join(ckpt_dir, 'expert_BS_biasFalse_ls0.0_epoch161.pth'),
    }
    ce_path = args.ce_path or default_paths['CE']
    la_path = args.la_path or default_paths['LA']
    bs_path = args.bs_path or default_paths['BS']
    ckpt_paths = {'CE': ce_path, 'LA': la_path, 'BS': bs_path}

    # Parse LA tau from filename
    la_tau = 1.5
    mt = re.search(r't([\d\.]+)', os.path.basename(la_path))
    if mt:
        la_tau = float(mt.group(1))
    print(f"LA tau = {la_tau}")

    # Load dataset (test set only — no augmentation)
    print("\nLoading CIFAR-100-LT test set...")
    dataset = ImbalancedDataset(cfg, cfg.dataset, augmentation='none')
    train_dataset, val_dataset = dataset.train_val_sets

    # Get class counts
    train_targets = np.array(train_dataset.targets)
    cfg.cls_num_list = np.bincount(train_targets, minlength=cfg.num_classes).tolist()

    # Split val into tune + test (same as GateTrainer)
    val_targets = np.array(val_dataset.targets)
    val_indices = np.arange(len(val_targets))
    tune_idx, test_idx = train_test_split(
        val_indices, test_size=0.5, stratify=val_targets, random_state=cfg.seed
    )
    test_dataset = Subset(val_dataset, test_idx)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    # Load expert ensemble
    print("\nLoading frozen expert ensemble...")
    model = FrozenExpertEnsemble(cfg, device, ckpt_paths, la_tau=la_tau)

    # Extract features from a subset of test samples
    print(f"\nExtracting gate features from up to {args.n_samples} test samples...")
    all_embeddings = []
    all_probs = [[], [], []]
    sample_count = 0
    for images, labels in test_loader:
        if sample_count >= args.n_samples:
            break
        images = images.to(device)
        logits_list, embeddings, probs = model(images)
        all_embeddings.append(embeddings.cpu())
        for i in range(3):
            all_probs[i].append(probs[i].cpu())
        sample_count += images.size(0)

    embeddings = torch.cat(all_embeddings, dim=0)[:args.n_samples]
    probs = [torch.cat(p, dim=0)[:args.n_samples] for p in all_probs]
    N = embeddings.size(0)
    D = embeddings.size(1)
    print(f"Feature matrix: {N} samples x {D} dimensions")

    # ── 1. Per-sample pairwise block correlations ──
    print("\n" + "=" * 80)
    print("1. PER-SAMPLE BLOCK CORRELATION ANALYSIS")
    print("=" * 80)
    print("Three 100-dim probability blocks (CE, LA, BS) after L2 normalization.")
    print("If mean per-sample correlation > 0.95, the blocks are near-collinear.\n")

    block_corrs = block_correlation(embeddings, block_size=100, num_blocks=3)
    for pair, stats in block_corrs.items():
        print(f"  {pair}:")
        print(f"    mean corr = {stats['mean_corr']:.4f} ± {stats['std_corr']:.4f}")
        print(f"    min = {stats['min_corr']:.4f}, max = {stats['max_corr']:.4f}")
        print(f"    % with corr > 0.90 = {stats['pct_gt_09']:.1f}%")
        print(f"    % with corr > 0.95 = {stats['pct_gt_095']:.1f}%")

    # ── 2. Cross-block matrix correlation ──
    print("\n" + "-" * 80)
    print("2. CROSS-BLOCK MATRIX CORRELATION (flattened vectors)")
    print("-" * 80)
    matrix_corrs = cross_block_mean_corr(embeddings, block_size=100)
    for pair, corr in matrix_corrs.items():
        print(f"  {pair}: r = {corr:.4f}")

    # ── 3. Within-block SVD ──
    print("\n" + "-" * 80)
    print("3. WITHIN-BLOCK EFFECTIVE RANK (SVD)")
    print("-" * 80)
    wb = within_block_svd(embeddings, block_size=100)
    for k, v in wb.items():
        print(f"  {k}: {v:.4f}")

    # ── 4. Full feature covariance ──
    print("\n" + "-" * 80)
    print("4. FULL FEATURE COVARIANCE ANALYSIS (316-dim)")
    print("-" * 80)
    cov_analysis = feature_covariance_analysis(embeddings)
    print(f"  Effective rank (participation ratio): {cov_analysis['effective_rank']:.2f}")
    print(f"  Condition number: {cov_analysis['condition_number']:.2f}")
    print(f"  Top-5 singular values: {[f'{s:.2f}' for s in cov_analysis['singular_values'][:5]]}")
    print(f"  Variance explained by top 5 components: {cov_analysis['var_explained_top5']*100:.1f}%")
    print(f"  Variance explained by top 10 components: {cov_analysis['var_explained_top10']*100:.1f}%")
    print(f"  Variance explained by top 20 components: {cov_analysis['var_explained_top20']*100:.1f}%")

    # ── 5. Per-expert mean probability profile ──
    print("\n" + "-" * 80)
    print("5. PER-EXPERT MEAN PROBABILITY PROFILE (CLASS-WISE)")
    print("-" * 80)
    profiles = per_expert_mean_profile(probs, cfg.cls_num_list)
    for name, profile in profiles.items():
        top5 = np.argsort(profile)[-5:][::-1]
        print(f"  {name}: top-5 classes = {top5.tolist()}, "
              f"top-5 mean probs = {[f'{profile[c]:.4f}' for c in top5]}")

    # ── 6. Cross-block mean correlation of the PROFILE (class-level) ──
    print("\n" + "-" * 80)
    print("6. CLASS-LEVEL PROFILE CORRELATION")
    print("-" * 80)
    for i, ni in enumerate(['CE', 'LA', 'BS']):
        for j, nj in enumerate(['CE', 'LA', 'BS']):
            if j <= i:
                continue
            r = float(np.corrcoef(profiles[ni], profiles[nj])[0, 1])
            print(f"  {ni} vs {nj}: r = {r:.4f}")

    # ── 7. Feature variance breakdown by block ──
    print("\n" + "-" * 80)
    print("7. FEATURE VARIANCE BY BLOCK")
    print("-" * 80)
    block_names = ['CE_probs (0-99)', 'LA_probs (100-199)', 'BS_probs (200-299)',
                   'Stats (300-308)', 'Agreement (309-311)', 'Freq (312-315)']
    block_ranges = [(0, 100), (100, 200), (200, 300), (300, 309), (309, 312), (312, 316)]
    for name, (start, end) in zip(block_names, block_ranges):
        block = embeddings[:, start:end]
        print(f"  {name:<22}: mean={block.mean():.4f}, std={block.std():.4f}, "
              f"min={block.min():.4f}, max={block.max():.4f}")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
