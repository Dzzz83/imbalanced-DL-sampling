#!/usr/bin/env python3
# ultra_debug.py
# Pipeline Verification & Paper Comparison (Tables 2, 3 & Diagnostics)

import os
import sys
import argparse
import torch
import numpy as np
import re
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

# 2. NOW import and call get_args()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from imbalanceddl.utils.config import get_args
from imbalanceddl.dataset.imbalance_dataset import ImbalancedDataset
from imbalanceddl.utils.plugin_rule import define_groups_2
from imbalanceddl.utils.debug.models import ExpertEnsemble, GateMLP
from imbalanceddl.utils.gate_features import gate_input_dim
from imbalanceddl.utils.debug.evaluation import (
    extract_data, run_metric_comparisons, run_temperature_comparison,
    run_saves_the_day_checks, recipe_from_checkpoint,
    run_raw_prob_inspection, run_oracle_diagnostic
)
from imbalanceddl.utils.debug.metrics import compute_all_metrics
from imbalanceddl.utils.debug.diagnostics import print_stage3_plugin_params, print_expert_agreement, print_per_class_extreme_routing


class LinearWeightPeakAnalyzer:
    """Diagnoses whether the gate acts as a naive probability peak-detector.

    Two views into the router:
    1. The GateMLP's first linear layer weight matrix (Linear(300, 64)),
       split into the three 100-dim input blocks per expert: CE = input
       cols 0-99, LA = 100-199, BS = 200-299. Near-uniform weights mean
       the gate is tracking overall input magnitude; extreme weights on a
       few classes mean it is overfitting to spurious per-class signals.
    2. How often each expert owns the highest per-sample maximum
       probability ("peak") across the test set, which reveals whether an
       expert is starved simply because it rarely produces the largest
       peak.
    """

    EXPERT_NAMES = ("CE", "LA", "BS")
    EXPERT_BLOCKS = ((0, 100), (100, 200), (200, 300))

    def __init__(self, gate, expert_probs):
        # Mini-MLP fc layer: (hidden_dim, 300) = hidden units x logit inputs.
        # The per-expert blocks live on the 300 input columns (CE 0-99, LA
        # 100-199, BS 200-299), so the column slicing below is unchanged.
        self.weight = gate.fc.weight.detach().cpu()
        self.expert_probs = expert_probs

    def run(self):
        """Print both diagnostics of the gate's routing behaviour."""
        self._print_linear_weight_analysis()
        self._print_peak_probability_frequency()

    def _print_linear_weight_analysis(self):
        print("\n" + "=" * 80)
        print("LINEAR WEIGHT & PEAK LOGIT ANALYSIS")
        print("=" * 80)
        print(f"GateMLP fc.weight shape: {tuple(self.weight.shape)} "
              "(hidden units x D gate inputs: 3x100 probs + 9 conf-stats "
              "+ 3 agreement)")
        print(f"{'Expert':<6} | {'Input block':<12} | {'Mean':<10} | "
              f"{'Std':<10} | {'Min':<10} | {'Max':<10}")
        print("-" * 70)
        for name, (start, end) in zip(self.EXPERT_NAMES, self.EXPERT_BLOCKS):
            block = self.weight[:, start:end]
            print(f"{name:<6} | {start}-{end - 1:<9} | "
                  f"{block.mean():+.6f} | {block.std():.6f} | "
                  f"{block.min():+.6f} | {block.max():+.6f}")
        print("-" * 70)
        print("[INFO] Uniform weights ~ tracking overall logit magnitude;")
        print("[INFO] extreme per-class weights ~ overfitting to spurious "
              "class signals.")

    def _print_peak_probability_frequency(self):
        peaks = torch.stack(
            [probs.max(dim=1).values for probs in self.expert_probs],
            dim=1,
        )
        peak_winner = torch.argmax(peaks, dim=1)
        total = peak_winner.numel()
        print("-" * 80)
        print(f"Max Probability Frequency ({total} test samples):")
        for i, name in enumerate(self.EXPERT_NAMES):
            count = int((peak_winner == i).sum().item())
            print(f"  {name}: {count}/{total} ({count / total * 100:.1f}%) | "
                  f"mean peak probability {peaks[:, i].mean():+.4f}")
        print("=" * 80)


def main():
    cfg = get_args()
    if cfg.dataset == 'cifar100':
        cfg.num_classes = 100
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        
    print("\n" + "="*80)
    print("ULTRA DEBUG: PIPELINE & PAPER COMPARISON")
    print("="*80)

    dataset = ImbalancedDataset(cfg, cfg.dataset, augmentation='none')
    train_dataset, val_dataset = dataset.train_val_sets
    
    train_targets = np.array(train_dataset.targets)
    cfg.cls_num_list = np.bincount(train_targets, minlength=cfg.num_classes).tolist()
    
    val_targets = np.array(val_dataset.targets)
    val_indices = np.arange(len(val_targets))
    tune_idx, test_idx = train_test_split(val_indices, test_size=0.8, stratify=val_targets, random_state=cfg.seed)
    
    tune_dataset = Subset(val_dataset, tune_idx)
    test_dataset = Subset(val_dataset, test_idx)
    
    tune_loader = DataLoader(tune_dataset, batch_size=128, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    ckpt_paths = {'CE': custom_args.ce_path, 'LA': custom_args.la_path, 'BS': custom_args.bs_path}

    # FIX: Added weights_only=False
    gate_ckpt = torch.load(custom_args.gate_ckpt, map_location='cpu', weights_only=False)

    la_tau = 1.5
    match_tau = re.search(r't([\d\.]+)', os.path.basename(custom_args.la_path))
    if match_tau:
        la_tau = float(match_tau.group(1))
    print(f"[INFO] Using LA Tau = {la_tau} parsed from filename")

    # Reconstruct the exact mixture recipe the checkpoint was trained with
    # (per-expert temperatures, k, mixture space, gate/mixture temperatures).
    recipe = recipe_from_checkpoint(gate_ckpt, cfg, la_tau=la_tau)
    print(f"[INFO] Recipe: T={recipe['T']} | expert_temps={recipe['expert_temps']} | "
          f"k={recipe['k']} | space={recipe['space']} | gate_temp={recipe['gate_temp']:.3f} | "
          f"mix_temp={recipe['mix_temp']:.3f}")

    model = ExpertEnsemble(cfg, device, ckpt_paths,
                           expert_T=recipe['expert_temps'],
                           normalize_blocks=recipe['norm_blocks'],
                           freq_features=recipe['freq_features']).to(device)

    gate = GateMLP(input_dim=gate_input_dim(cfg.num_classes,
                                            freq_features=recipe['freq_features']),
                   num_experts=3,
                   linear_router=recipe.get('linear_router', False)).to(device)
    print(f"[INFO] Loading Gate from {custom_args.gate_ckpt}")

    gate.load_state_dict(gate_ckpt['gate_state_dict'])
    gate.eval()

    print("\n[INFO] Extracting posteriors...")
    (p_mix_tune, p_unif_tune, p_ce_tune, p_la_tune, p_bs_tune,
     l_ce_tune, l_la_tune, l_bs_tune, w_tune, labels_tune,
     gate_logits_tune) = extract_data(model, gate, tune_loader, device, recipe)

    (p_mix_test, p_unif_test, p_ce_test, p_la_test, p_bs_test,
     l_ce_test, l_la_test, l_bs_test, w_test, labels_test,
     gate_logits_test) = extract_data(model, gate, test_loader, device, recipe)

    group_ids_2 = define_groups_2(cfg.cls_num_list)

    # 0. Tensor-Level Health Check
    # Earliest signal: are the 300-dim logit inputs (CE, LA, BS) on wildly
    # different scales, and are the gate's pre-softmax activations collapsing
    # toward zero? Either would bias the router.
    print("\n" + "=" * 80)
    print("TENSOR-LEVEL HEALTH CHECK (Scale & Collapse Diagnosis)")
    print("=" * 80)

    logit_stds = []
    for name, logits in (("CE", l_ce_test), ("LA", l_la_test),
                         ("BS", l_bs_test)):
        logit_stds.append(logits.std().item())
        print(f"{name} raw logits: mean={logits.mean().item():+.4f} | "
              f"std={logits.std().item():.4f}")
        print(f"  [min={logits.min().item():+.3f}, "
              f"max={logits.max().item():+.3f}]")
    print(f"Logit scale ratio (max std / min std): "
          f"{max(logit_stds) / min(logit_stds):.2f}x")

    print("-" * 80)
    print("Gate pre-softmax activations (gate_logits):")
    print(f"  overall: mean={gate_logits_test.mean().item():+.6f} | "
          f"std={gate_logits_test.std().item():.6f} | "
          f"max_abs={gate_logits_test.abs().max().item():.6f}")
    for i, name in enumerate(["CE", "LA", "BS"]):
        col = gate_logits_test[:, i]
        print(f"  expert {name}: mean={col.mean().item():+.6f} | "
              f"std={col.std().item():.6f}")
    if gate_logits_test.abs().max().item() < 1e-3:
        print("[WARN] Gate pre-softmax activations collapsed toward zero.")
    else:
        print("[INFO] Gate pre-softmax activations are not collapsed "
              "(healthy scale).")
    print("=" * 80)

    # 1. Linear Weight & Peak Logit Analysis
    # Inspects the gate's learned weights and how often each expert wins the
    # max-probability peak race (is BS starved because it rarely peaks
    # highest?). The gate routes on calibrated probabilities, so pass the
    # T-calibrated posteriors (as torch tensors) rather than raw logits.
    peak_probs = (torch.from_numpy(p_ce_test), torch.from_numpy(p_la_test),
                  torch.from_numpy(p_bs_test))
    LinearWeightPeakAnalyzer(gate, peak_probs).run()

    # 2. Metrics & Comparisons
    run_metric_comparisons(p_mix_tune, p_unif_tune, p_ce_tune, p_la_tune, p_mix_test, p_unif_test, p_ce_test, p_la_test, p_bs_test, l_ce_test, l_la_test, l_bs_test, labels_tune, labels_test, group_ids_2, cfg, train_dataset)
    
    # 3. Temperature Comparison
    # m_unif / m_method = metrics of the Uniform and Gate-routed Method
    # posteriors under the checkpoint recipe (p_unif_test / p_mix_test were
    # extracted by extract_data). The T=1.0 columns are computed inside
    # run_temperature_comparison (with the corrected la_tau bias).
    m_unif = compute_all_metrics(p_unif_test, labels_test, None, cfg, train_dataset)
    m_method = compute_all_metrics(p_mix_test, labels_test, None, cfg, train_dataset)
    run_temperature_comparison(recipe, l_ce_test, l_la_test, l_bs_test,
                               gate_logits_test, labels_test, cfg,
                               train_dataset, m_unif, m_method)
    
    # 4. Routing Statistics
    label_groups_test = group_ids_2[labels_test]
    head_mask = (label_groups_test == 0)
    tail_mask = (label_groups_test == 1)
    print_per_class_extreme_routing(w_test, labels_test, cfg)
    
    # 5. LA Saves the Day & Raw Prob Inspection
    la_saves_day_indices = run_saves_the_day_checks(p_ce_test, p_la_test, p_bs_test, w_test, labels_test, label_groups_test, recipe['k'])
    run_raw_prob_inspection(la_saves_day_indices, p_ce_test, p_la_test, p_bs_test, w_test, labels_test)
    
    # 6. Oracle Diagnostic
    run_oracle_diagnostic(p_ce_test, p_la_test, p_bs_test, p_mix_test, labels_test, head_mask, tail_mask, cfg, train_dataset)
    
    # 7. Stage 3 Plugin Parameters
    print_stage3_plugin_params(p_mix_tune, labels_tune, group_ids_2, cfg)
    
    # 8. Expert Correlation & Sharpening Check
    print_expert_agreement(p_mix_test, np.argmax(p_ce_test, axis=1), np.argmax(p_la_test, axis=1), np.argmax(p_bs_test, axis=1), labels_test)
    
    agreement = np.mean((np.argmax(p_ce_test, axis=1) == np.argmax(p_la_test, axis=1)) & (np.argmax(p_la_test, axis=1) == np.argmax(p_bs_test, axis=1)))
    print(f"Expert Prediction Agreement: {agreement*100:.2f}%")
    
    unif_max_conf = np.max(p_unif_test, axis=1)
    method_max_conf = np.max(p_mix_test, axis=1)
    print(f"Uniform Avg Max Confidence:  {np.mean(unif_max_conf):.4f}")
    print(f"My Method Avg Max Confidence: {np.mean(method_max_conf):.4f}")

if __name__ == "__main__":
    main()