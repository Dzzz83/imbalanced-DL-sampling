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
custom_parser.add_argument('--diagnose_confident_wrong', action='store_true',
                           help='Run only the DaWin confidence-wrong diagnostic '
                                'and exit.')
custom_parser.add_argument('--diagnose_embeddings', action='store_true',
                           help='Run only the 192-dim embedding correlation '
                                'diagnostic (Exp 18, Phase A) and exit.')
custom_args, remaining_argv = custom_parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_argv

# 2. NOW import and call get_args()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from imbalanceddl.utils.config import get_args
from imbalanceddl.dataset.imbalance_dataset import ImbalancedDataset
from imbalanceddl.utils.plugin_rule import define_groups_2
from imbalanceddl.utils.debug.models import ExpertEnsemble, GateMLP
from imbalanceddl.utils.debug.evaluation import (
    extract_data, run_metric_comparisons, run_temperature_comparison,
    run_saves_the_day_checks, recipe_from_checkpoint,
    run_raw_prob_inspection, run_oracle_diagnostic
)
from imbalanceddl.utils.debug.metrics import compute_all_metrics
from imbalanceddl.utils.debug.diagnostics import print_stage3_plugin_params, print_expert_agreement, print_per_class_extreme_routing


class LinearWeightPeakAnalyzer:
    """Diagnoses whether the gate acts as a naive peak-detector.

    Two views into the router:
    1. The GateMLP's first linear layer weight matrix split into the per-
       expert input blocks. In penultimate mode each expert occupies 64
       columns (embedding dim); in probability mode each expert occupies
       C columns (class-probability dim). Near-uniform weights mean the
       gate is tracking overall input magnitude; extreme weights on a few
       features mean it is overfitting to spurious per-class signals.
    2. How often each expert owns the highest per-sample maximum
       probability ("peak") across the test set, which reveals whether an
       expert is starved simply because it rarely produces the largest
       peak.
    """

    EXPERT_NAMES = ("CE", "LA", "BS")

    def __init__(self, gate, expert_probs, gate_input_mode='probability',
                 num_classes=100):
        self.weight = gate.fc.weight.detach().cpu()
        self.expert_probs = expert_probs
        self.gate_input_mode = gate_input_mode
        self.num_classes = num_classes
        # Determine per-expert block size from the actual weight shape.
        in_features = self.weight.shape[1]
        if gate_input_mode == 'penultimate':
            # 3 experts × 64-dim penultimate embeddings = 192
            block_size = in_features // 3  # typically 64
            self._input_desc = (
                f"3x{block_size} embeddings (penultimate mode)")
        else:
            # Probability mode: first 3*C columns are C-dim prob blocks
            block_size = num_classes
            extra = in_features - 3 * num_classes  # stats + agreements
            self._input_desc = (
                f"3x{num_classes} probs + {extra} stats/agree")
        # Guard against empty slices when the tensor is smaller than expected.
        if 3 * block_size > in_features:
            # Fall back to splitting evenly if the assumed block size is
            # larger than the actual feature count.
            block_size = in_features // 3
            self._input_desc += f" [fallback: {block_size}-dim blocks]"
        self._block_size = block_size
        self.EXPERT_BLOCKS = tuple(
            (i * block_size, (i + 1) * block_size)
            for i in range(3)
        )

    def run(self):
        """Print both diagnostics of the gate's routing behaviour."""
        self._print_linear_weight_analysis()
        self._print_peak_probability_frequency()

    def _print_linear_weight_analysis(self):
        print("\n" + "=" * 80)
        print("LINEAR WEIGHT & PEAK LOGIT ANALYSIS")
        print("=" * 80)
        print(f"GateMLP fc.weight shape: {tuple(self.weight.shape)} "
              f"(hidden units x D gate inputs: {self._input_desc})")
        print(f"{'Expert':<6} | {'Input block':<12} | {'Mean':<10} | "
              f"{'Std':<10} | {'Min':<10} | {'Max':<10}")
        print("-" * 70)
        for name, (start, end) in zip(self.EXPERT_NAMES, self.EXPERT_BLOCKS):
            block = self.weight[:, start:end]
            if block.numel() == 0:
                print(f"{name:<6} | {start}-{end - 1:<9} | "
                      f"{'EMPTY':>10} | {'EMPTY':>10} | "
                      f"{'EMPTY':>10} | {'EMPTY':>10}")
                continue
            print(f"{name:<6} | {start}-{end - 1:<9} | "
                  f"{block.mean():+.6f} | {block.std():.6f} | "
                  f"{block.min():+.6f} | {block.max():+.6f}")
        print("-" * 70)
        print("[INFO] Uniform weights ~ tracking overall input magnitude;")
        print("[INFO] extreme per-class weights ~ overfitting to spurious "
              "feature signals.")

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


def run_confident_wrong_diagnostic(p_ce, p_la, p_bs, labels, cls_num_list,
                                   recipe, p_ce_tune=None, p_la_tune=None,
                                   p_bs_tune=None, labels_tune=None,
                                   logits_test=None):
    """DaWin assumption verification: measure how often the most confident
    expert is wrong.

    DaWin routes by ``w_j = softmax(max_k p_j(k|x) / T)``. This works when
    high-confidence experts are usually correct. If the confidently-wrong rate
    is high, DaWin will amplify errors and underperform the uniform baseline.

    Parameters
    ----------
    p_ce, p_la, p_bs : np.ndarray, shape (N, C)
        Per-expert calibrated probabilities on the test set.
    labels : np.ndarray, shape (N,)
        Ground-truth class labels.
    cls_num_list : list of int
        Per-class training counts (used for head/mid/tail grouping).
    recipe : dict
        Mixture recipe (used only for logit calibration in DaWin simulation).
    p_ce_tune, p_la_tune, p_bs_tune : np.ndarray or None, shape (M, C)
        Tune-set probabilities for fitting T̂. If None, T̂=1.0 is assumed.
    labels_tune : np.ndarray or None
        Tune-set labels.
    logits_test : list of torch.Tensor or None, shape (N, C) each
        Raw test logits for building mixtures in DaWin simulation. If None,
        the mixture is built from probabilities (approximate).

    Returns
    -------
    dict with keys:
        - 'confidently_wrong_rate': float (overall fraction)
        - 'confidently_wrong_rate_by_group': dict per group
        - 'avg_conf_when_correct': float
        - 'avg_conf_when_wrong': float
        - 'dawin_bal_acc': float or None
        - 'uniform_bal_acc': float
        - 'gate_bal_acc': float or None
        - 'verdict': str
    """
    print("\n" + "=" * 80)
    print("DAWIN ASSUMPTION DIAGNOSTIC: Confidently-Wrong Analysis")
    print("=" * 80)
    print("[INFO] DaWin routes by w_j = softmax(max_k p_j(k|x) / T).")
    print("[INFO] This diagnostic measures how often the most confident")
    print("[INFO] expert is WRONG — the key assumption behind DaWin.\n")

    N = len(labels)
    confidences = np.stack([
        p_ce.max(axis=1), p_la.max(axis=1), p_bs.max(axis=1)
    ], axis=1)  # (N, 3)
    predictions = np.stack([
        p_ce.argmax(axis=1), p_la.argmax(axis=1), p_bs.argmax(axis=1)
    ], axis=1)  # (N, 3)
    expert_correct = (predictions == labels.reshape(-1, 1))  # (N, 3)

    # ---- Which expert is most confident per sample? ----
    most_confident_idx = confidences.argmax(axis=1)
    most_confident_conf = confidences[np.arange(N), most_confident_idx]
    most_confident_correct = expert_correct[np.arange(N), most_confident_idx]

    confidently_wrong_rate = 1.0 - most_confident_correct.mean()
    n_confidently_wrong = int((~most_confident_correct).sum())

    # ---- Per-group breakdown (head/mid/tail) ----
    cls_num_arr = np.array(cls_num_list)
    group_ids = np.full(len(cls_num_arr), 1, dtype=np.int64)  # default mid
    group_ids[cls_num_arr > 100] = 0   # head (many-shot)
    group_ids[cls_num_arr < 20] = 2    # tail (low-shot)
    label_groups = group_ids[labels]   # (N,)
    group_names = {0: "Head", 1: "Mid", 2: "Tail"}

    per_group = {}
    for gid in [0, 1, 2]:
        mask = (label_groups == gid)
        if mask.sum() == 0:
            per_group[group_names[gid]] = {'count': 0, 'conf_wrong_rate': 0.0,
                                           'avg_conf_wrong': 0.0,
                                           'avg_conf_correct': 0.0}
            continue
        g_correct = most_confident_correct[mask]
        g_conf = most_confident_conf[mask]
        g_wrong_mask = ~g_correct
        g_correct_mask = g_correct
        per_group[group_names[gid]] = {
            'count': int(mask.sum()),
            'conf_wrong_rate': float(g_wrong_mask.mean()),
            'avg_conf_wrong': float(g_conf[g_wrong_mask].mean()
                                    if g_wrong_mask.sum() > 0 else 0.0),
            'avg_conf_correct': float(g_conf[g_correct_mask].mean()
                                      if g_correct_mask.sum() > 0 else 0.0),
        }

    # ---- Average confidence when correct vs wrong ----
    wrong_mask = ~most_confident_correct
    correct_mask = most_confident_correct
    avg_conf_correct = float(most_confident_conf[correct_mask].mean()
                             if correct_mask.sum() > 0 else 0.0)
    avg_conf_wrong = float(most_confident_conf[wrong_mask].mean()
                           if wrong_mask.sum() > 0 else 0.0)

    # ---- Print results ----
    print(f"\n{'Group':<8} | {'Samples':<8} | {'Conf-Wrong Rate':<16} "
          f"| {'Avg Conf (Wrong)':<16} | {'Avg Conf (Correct)':<18}")
    print("-" * 75)
    print(f"{'ALL':<8} | {N:<8} | {confidently_wrong_rate:<16.2%} "
          f"| {avg_conf_wrong:<16.4f} | {avg_conf_correct:<18.4f}")
    for gname in ["Head", "Mid", "Tail"]:
        pg = per_group[gname]
        if pg['count'] == 0:
            continue
        print(f"{gname:<8} | {pg['count']:<8} | {pg['conf_wrong_rate']:<16.2%} "
              f"| {pg['avg_conf_wrong']:<16.4f} | {pg['avg_conf_correct']:<18.4f}")
    print("-" * 75)

    # ---- Distribution of confidence when wrong ----
    if wrong_mask.sum() > 0:
        bins = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
        labels_bins = ['0-0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9', '0.9-1.0']
        conf_when_wrong = most_confident_conf[wrong_mask]
        hist, _ = np.histogram(conf_when_wrong, bins=bins)
        hist_pct = hist / hist.sum() * 100
        print("\nConfidence distribution when most-confident expert is WRONG:")
        for lb, h, hp in zip(labels_bins, hist, hist_pct):
            bar = '#' * int(hp / 2)
            print(f"  conf ∈ {lb:<9} : {h:>4d} samples ({hp:5.1f}%) {bar}")

    # ---- Who is the most confident expert? ----
    print("\nMost-confident expert identity:")
    for i, name in enumerate(["CE", "LA", "BS"]):
        count = int((most_confident_idx == i).sum())
        correct_count = int(((most_confident_idx == i) & expert_correct[:, i]).sum())
        wrong_count = count - correct_count
        print(f"  {name}: most-confident on {count:>4d}/{N} samples "
              f"({100*count/N:5.1f}%) — correct {correct_count}, wrong {wrong_count}")

    # ---- Confidently-wrong pairs ----
    # On samples where the most-confident expert is wrong, is there ANOTHER
    # expert that is correct? This tells us whether confidence correlates with
    # correctness at the expert level.
    if n_confidently_wrong > 0:
        other_experts_correct = expert_correct[wrong_mask].sum(axis=1) - 0  # subtract 0 b/c most-confident is counted
        # Actually, we need: of the OTHER two experts (not the most-confident one),
        # how many are correct?
        other_correct = np.zeros(n_confidently_wrong, dtype=int)
        for i in range(n_confidently_wrong):
            idx_most = most_confident_idx[wrong_mask][i]
            other_mask = np.array([0, 1, 2]) != idx_most
            other_correct[i] = expert_correct[wrong_mask][i][other_mask].sum()
        at_least_one_other_correct = (other_correct >= 1).sum()
        both_other_correct = (other_correct == 2).sum()
        print(f"\nOn {n_confidently_wrong} confidently-wrong samples:")
        print(f"  At least one other expert correct: {at_least_one_other_correct} "
              f"({100*at_least_one_other_correct/n_confidently_wrong:.1f}%)")
        print(f"  Both other experts correct:        {both_other_correct} "
              f"({100*both_other_correct/n_confidently_wrong:.1f}%)")
        print(f"  All experts wrong:                 {n_confidently_wrong - at_least_one_other_correct} "
              f"({100*(n_confidently_wrong - at_least_one_other_correct)/n_confidently_wrong:.1f}%)")

    # ---- Uniform baseline comparison on confidently-wrong samples ----
    # On these samples, what would uniform averaging predict?
    if n_confidently_wrong > 0:
        p_unif_conf_wrong = (p_ce[wrong_mask] + p_la[wrong_mask]
                             + p_bs[wrong_mask]) / 3.0
        unif_preds_conf_wrong = p_unif_conf_wrong.argmax(axis=1)
        unif_correct_conf_wrong = (unif_preds_conf_wrong
                                   == labels[wrong_mask])
        unif_acc_conf_wrong = unif_correct_conf_wrong.mean()
        print(f"\nUniform baseline on confidently-wrong samples: "
              f"{unif_acc_conf_wrong:.2%} "
              f"({int(unif_correct_conf_wrong.sum())}/{n_confidently_wrong})")
        # How does this compare to DaWin (simulated)?
        # For each sample, if we used only the most-confident expert:
        most_conf_preds = predictions[np.arange(N), most_confident_idx]
        most_conf_acc_on_wrong = (most_conf_preds[wrong_mask]
                                  == labels[wrong_mask]).mean()
        print(f"Most-confident expert's own accuracy on these samples: "
              f"{most_conf_acc_on_wrong:.2%} "
              f"(= {confidently_wrong_rate:.2%} wrong rate inverted)")

    # ---- Simulated DaWin accuracy (grid-search T on tune set) ----
    dawin_bal_acc = None
    dawin_bal_acc_on_conf_wrong = None
    if (p_ce_tune is not None and p_la_tune is not None
            and p_bs_tune is not None and labels_tune is not None):
        print("\n" + "-" * 75)
        print("DaWin Simulation: Grid-Search Temperature on Tune Set")
        print("-" * 75)
        confidences_tune = np.stack([
            p_ce_tune.max(axis=1), p_la_tune.max(axis=1),
            p_bs_tune.max(axis=1)
        ], axis=1)

        # Balanced accuracy function for tuning
        def balanced_acc(preds, true_labels):
            """Compute per-class balanced accuracy."""
            classes = np.unique(true_labels)
            accs = []
            for c in classes:
                mask = (true_labels == c)
                if mask.sum() > 0:
                    accs.append((preds[mask] == c).mean())
            return np.mean(accs) * 100.0 if accs else 0.0

        best_T, best_bal = 1.0, 0.0
        T_candidates = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
        for T_hat in T_candidates:
            w_dawin = np.exp(confidences_tune / T_hat)
            w_dawin /= w_dawin.sum(axis=1, keepdims=True)

            # Build mixture in prob space (simplified: average probabilities
            # weighted by DaWin weights). For exact logit-space mixing we
            # would need the raw logits.
            p_dawin_tune = (w_dawin[:, 0:1] * p_ce_tune
                            + w_dawin[:, 1:2] * p_la_tune
                            + w_dawin[:, 2:3] * p_bs_tune)
            dawin_preds = p_dawin_tune.argmax(axis=1)
            bal = balanced_acc(dawin_preds, labels_tune)
            if bal > best_bal:
                best_bal = bal
                best_T = T_hat

        print(f"  Best T̂ = {best_T:.2f} with tune bal acc = {best_bal:.2f}%")

        # Evaluate DaWin on test set
        confidences_test = np.stack([
            p_ce.max(axis=1), p_la.max(axis=1), p_bs.max(axis=1)
        ], axis=1)
        w_dawin_test = np.exp(confidences_test / best_T)
        w_dawin_test /= w_dawin_test.sum(axis=1, keepdims=True)
        p_dawin_test = (w_dawin_test[:, 0:1] * p_ce
                        + w_dawin_test[:, 1:2] * p_la
                        + w_dawin_test[:, 2:3] * p_bs)
        dawin_preds_test = p_dawin_test.argmax(axis=1)
        dawin_bal_acc = balanced_acc(dawin_preds_test, labels)

        # Also compute overall accuracy (not just balanced)
        dawin_overall_acc = (dawin_preds_test == labels).mean() * 100.0

        # On confidently-wrong subset
        if n_confidently_wrong > 0:
            p_dawin_conf_wrong = p_dawin_test[wrong_mask]
            dawin_preds_cw = p_dawin_conf_wrong.argmax(axis=1)
            dawin_bal_acc_on_conf_wrong = (
                dawin_preds_cw == labels[wrong_mask]
            ).mean() * 100.0
        else:
            dawin_bal_acc_on_conf_wrong = 0.0

        # Uniform baseline for comparison
        p_unif_test = (p_ce + p_la + p_bs) / 3.0
        unif_preds_test = p_unif_test.argmax(axis=1)
        unif_bal_acc = balanced_acc(unif_preds_test, labels)
        unif_overall_acc = (unif_preds_test == labels).mean() * 100.0

        # Gate baseline (if available — p_mix_test is computed by extract_data)
        # We don't have it here, so we'll skip reporting gate bal acc from
        # this function.

        print(f"\n  Test set results (T̂ = {best_T:.2f}):")
        print(f"  {'Method':<20} | {'Bal Acc':<8} | {'Overall Acc':<12}")
        print(f"  {'-'*42}")
        print(f"  {'DaWin':<20} | {dawin_bal_acc:<8.2f} | {dawin_overall_acc:<12.2f}")
        print(f"  {'Uniform':<20} | {unif_bal_acc:<8.2f} | {unif_overall_acc:<12.2f}")
        if dawin_bal_acc_on_conf_wrong > 0:
            print(f"\n  On confidently-wrong samples only:")
            print(f"  {'DaWin accuracy':<30} : {dawin_bal_acc_on_conf_wrong:.2f}%")
            print(f"  {'Uniform accuracy':<30} : {unif_acc_conf_wrong*100:.2f}%")

    # ---- Verdict ----
    print("\n" + "=" * 80)
    print("DAWIN VERDICT")
    print("=" * 80)
    if confidently_wrong_rate < 0.15:
        verdict = "PASS — DaWin assumption holds."
        print(f"  Confidently-wrong rate = {confidently_wrong_rate:.2%} (< 15%).")
        print("  The most confident expert is almost always correct.")
        print("  DaWin is safe to proceed as planned.")
    elif confidently_wrong_rate > 0.30:
        verdict = "FAIL — DaWin assumption violated."
        print(f"  Confidently-wrong rate = {confidently_wrong_rate:.2%} (> 30%).")
        print("  The most confident expert is often wrong.")
        print("  DaWin will likely amplify errors. Recommend skipping to")
        print("  penultimate feature routing (#2) or SADE (#4).")
    else:
        verdict = "INCONCLUSIVE — requires temperature verification."
        print(f"  Confidently-wrong rate = {confidently_wrong_rate:.2%} (15-30%).")
        print("  DaWin may still work if temperature scaling suppresses")
        print("  overconfident-wrong experts, but the margin is narrow.")
        if dawin_bal_acc is not None:
            print(f"\n  Empirical check: DaWin bal acc = {dawin_bal_acc:.2f}% "
                  f"vs Uniform = {unif_bal_acc:.2f}%.")
            delta = dawin_bal_acc - unif_bal_acc
            if delta > 1.0:
                print(f"  DaWin beats uniform by {delta:.2f} pp — "
                      f"temperature scaling mitigates the issue.")
                verdict = "PASS (empirical) — DaWin assumption holds with T scaling."
            elif delta > 0.0:
                print(f"  DaWin narrowly beats uniform by {delta:.2f} pp — "
                      f"marginal but positive.")
            else:
                print(f"  DaWin underperforms uniform by {-delta:.2f} pp — "
                      f"confidence routing is unreliable.")
                verdict = ("INCONCLUSIOUS — DaWin trails uniform; "
                           "skip to penultimate routing (#2).")
    print("=" * 80)

    return {
        'confidently_wrong_rate': float(confidently_wrong_rate),
        'confidently_wrong_rate_by_group': {
            gname: pg['conf_wrong_rate']
            for gname, pg in per_group.items()
        },
        'avg_conf_correct': avg_conf_correct,
        'avg_conf_wrong': avg_conf_wrong,
        'dawin_bal_acc': dawin_bal_acc,
        'uniform_bal_acc': (unif_bal_acc if dawin_bal_acc is not None
                            else None),
        'gate_bal_acc': None,
        'verdict': verdict,
    }


# ═══════════════════════════════════════════════════════════════ #
#  Embedding Correlation Diagnostic (Exp 18, Phase A)            #
# ═══════════════════════════════════════════════════════════════ #

def _per_sample_block_correlation(embeddings, block_size, num_blocks=3):
    """Pairwise per-sample Pearson correlation between embedding blocks.

    Same methodology as ``diagnose_feature_collinearity.py`` but operates on
    arbitrary block sizes (64 for penultimate embeddings, 100 for probs).

    Parameters
    ----------
    embeddings : np.ndarray, shape (N, D)
        Concatenated per-expert blocks.
    block_size : int
        Dimension of each expert block (e.g. 64 for penultimate, 100 for probs).
    num_blocks : int
        Number of experts (3).

    Returns
    -------
    dict of ``{(i,j): {'mean_corr': float, 'std_corr': float, ...}}``
    """
    N = embeddings.shape[0]
    blocks = [embeddings[:, b*block_size:(b+1)*block_size] for b in range(num_blocks)]
    results = {}
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            corrs = np.zeros(N)
            for s in range(N):
                vi, vj = blocks[i][s], blocks[j][s]
                if vi.std() > 1e-8 and vj.std() > 1e-8:
                    corrs[s] = np.corrcoef(vi, vj)[0, 1]
            results[(i, j)] = {
                'mean_corr': float(np.mean(corrs)),
                'std_corr': float(np.std(corrs)),
                'min_corr': float(np.min(corrs)),
                'max_corr': float(np.max(corrs)),
                'pct_gt_09': float(np.mean(corrs > 0.9) * 100),
                'pct_gt_095': float(np.mean(corrs > 0.95) * 100),
            }
    return results


def _feature_covariance_analysis(embeddings):
    """SVD-based analysis of feature covariance (effective rank, var explained).

    Parameters
    ----------
    embeddings : np.ndarray, shape (N, D)

    Returns
    -------
    dict with effective_rank, condition_number, var_explained_top5/10/20.
    """
    X = embeddings - embeddings.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var_ex = S ** 2 / (S ** 2).sum()
    eff_rank = (S.sum() ** 2) / (S ** 2).sum()
    return {
        'effective_rank': float(eff_rank),
        'condition_number': float(S[0] / S[-1]) if S[-1] > 1e-10 else float('inf'),
        'var_explained_top5': float(var_ex[:5].sum()),
        'var_explained_top10': float(var_ex[:10].sum()),
        'var_explained_top20': float(var_ex[:20].sum()),
    }


def _within_block_svd(embeddings, block_size, num_blocks=3):
    """SVD within each embedding block to measure its intrinsic dimension."""
    results = {}
    for b in range(num_blocks):
        block = embeddings[:, b*block_size:(b+1)*block_size]
        X = block - block.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        var_ex = S ** 2 / (S ** 2).sum()
        eff_rank = (S.sum() ** 2) / (S ** 2).sum()
        results[f'Block{b}_effective_rank'] = float(eff_rank)
        results[f'Block{b}_top5_var'] = float(var_ex[:5].sum())
    return results


def compute_embedding_correlation_diagnostic(model, loader, device,
                                             cls_num_list, n_samples=500):
    """Extract per-expert penultimate embeddings and probability features,
    then compute cross-expert correlation metrics for both spaces.

    This directly answers: are the 192-dim penultimate embeddings more diverse
    (less correlated across experts) than the 316-dim probability features?

    Parameters
    ----------
    model : ExpertEnsemble
        Frozen expert ensemble (from ``imbalanceddl.utils.debug.models``).
    loader : DataLoader
        Test-set loader (any batch size).
    device : torch.device
    cls_num_list : list of int
        Per-class training counts (for gate input construction).
    n_samples : int
        Max number of samples to process (for speed on CPU).

    Returns
    -------
    dict with keys for both embedding-space and prob-space metrics,
    plus a 'recommendation' string.
    """
    print("\n" + "=" * 80)
    print("EMBEDDING CORRELATION DIAGNOSTIC (Exp 18, Phase A)")
    print("=" * 80)
    print("[INFO] Comparing 192-dim penultimate embeddings vs 316-dim")
    print("[INFO] calibrated probability features for per-expert diversity.\n")

    # Collect data
    all_hidden = [[], [], []]   # 3 × 64-dim per-expert embeddings
    all_probs = [[], [], []]    # 3 × 100-dim calibrated probabilities
    sample_count = 0

    for images, labels in loader:
        if sample_count >= n_samples:
            break
        images = images.to(device)
        batch_size = images.size(0)
        # Use forward_with_hidden to get per-expert penultimate embeddings
        logits_list, embeddings_316, hidden_list = model.forward_with_hidden(images)

        # Calibrate probabilities (same recipe as the model's forward)
        from imbalanceddl.utils.gate_features import calibrate_expert_probs
        probs = calibrate_expert_probs(
            logits_list, cls_num_list, model.la_tau,
            T=1.0, per_expert_T=model.expert_T,
        )

        for i in range(3):
            all_hidden[i].append(hidden_list[i].cpu().numpy())
            all_probs[i].append(probs[i].cpu().numpy())
        sample_count += batch_size

    # Concatenate and trim
    hidden = [np.concatenate(h, axis=0)[:n_samples] for h in all_hidden]
    probs = [np.concatenate(p, axis=0)[:n_samples] for p in all_probs]
    emb_192 = np.concatenate(hidden, axis=1)   # (N, 192)
    emb_316 = np.concatenate(probs, axis=1)    # (N, 300) — no stats/freq here
    # For fair comparison, also build the full 316-dim gate input
    # (with L2 normalization + stats + freq features)
    from imbalanceddl.utils.gate_features import build_gate_input
    probs_t = [torch.from_numpy(p) for p in probs]
    gate_input = build_gate_input(
        probs_t, normalize_blocks=True,
        cls_num_list=torch.tensor(cls_num_list, dtype=torch.float32),
    ).numpy()  # (N, 316)

    N = emb_192.shape[0]
    print(f"  Samples: {N}")
    print(f"  192-dim embedding space: {emb_192.shape[1]} dims")
    print(f"  316-dim probability space: {gate_input.shape[1]} dims")
    print()

    # ── 1. Per-sample pairwise block correlations ──
    print("-" * 75)
    print("1. PER-SAMPLE PAIRWISE BLOCK CORRELATION")
    print("-" * 75)

    # 192-dim embeddings (3 × 64 blocks)
    print("\n  192-dim penultimate embeddings (3 × 64-dim blocks):")
    emb_corrs = _per_sample_block_correlation(emb_192, block_size=64)
    expert_names = ['CE', 'LA', 'BS']
    for (i, j), stats in emb_corrs.items():
        print(f"    {expert_names[i]} vs {expert_names[j]}: "
              f"mean r = {stats['mean_corr']:.4f} ± {stats['std_corr']:.4f}")

    # 316-dim probabilities (3 × 100 blocks, L2-normalized)
    print("\n  316-dim probability features (3 × 100-dim blocks, L2-normed):")
    prob_corrs = _per_sample_block_correlation(gate_input, block_size=100)
    for (i, j), stats in prob_corrs.items():
        print(f"    {expert_names[i]} vs {expert_names[j]}: "
              f"mean r = {stats['mean_corr']:.4f} ± {stats['std_corr']:.4f}")

    # ── 2. Within-block effective rank ──
    print("\n" + "-" * 75)
    print("2. WITHIN-BLOCK EFFECTIVE RANK (SVD)")
    print("-" * 75)
    print("\n  192-dim embeddings:")
    emb_wb = _within_block_svd(emb_192, block_size=64)
    for k, v in emb_wb.items():
        print(f"    {k}: {v:.2f}")

    print("\n  316-dim probabilities:")
    prob_wb = _within_block_svd(gate_input, block_size=100)
    for k, v in prob_wb.items():
        print(f"    {k}: {v:.2f}")

    # ── 3. Full covariance analysis ──
    print("\n" + "-" * 75)
    print("3. FULL COVARIANCE ANALYSIS")
    print("-" * 75)

    print("\n  192-dim embedding space:")
    emb_cov = _feature_covariance_analysis(emb_192)
    print(f"    Effective rank: {emb_cov['effective_rank']:.2f} / {emb_192.shape[1]}")
    print(f"    Condition number: {emb_cov['condition_number']:.2f}")
    print(f"    Top 5 PCs explain: {emb_cov['var_explained_top5']*100:.1f}%")
    print(f"    Top 10 PCs explain: {emb_cov['var_explained_top10']*100:.1f}%")
    print(f"    Top 20 PCs explain: {emb_cov['var_explained_top20']*100:.1f}%")

    print("\n  316-dim probability space (gate input):")
    prob_cov = _feature_covariance_analysis(gate_input)
    print(f"    Effective rank: {prob_cov['effective_rank']:.2f} / {gate_input.shape[1]}")
    print(f"    Condition number: {prob_cov['condition_number']:.2f}")
    print(f"    Top 5 PCs explain: {prob_cov['var_explained_top5']*100:.1f}%")
    print(f"    Top 10 PCs explain: {prob_cov['var_explained_top10']*100:.1f}%")
    print(f"    Top 20 PCs explain: {prob_cov['var_explained_top20']*100:.1f}%")

    # ── 4. Head-to-head comparison ──
    print("\n" + "=" * 75)
    print("4. HEAD-TO-HEAD COMPARISON: EMBEDDINGS vs PROBABILITIES")
    print("=" * 75)

    emb_mean_corr = np.mean([s['mean_corr'] for s in emb_corrs.values()])
    prob_mean_corr = np.mean([s['mean_corr'] for s in prob_corrs.values()])

    print(f"\n  {'Metric':<45} | {'192-emb':<10} | {'316-prob':<10} | {'Delta':<10}")
    print(f"  {'-'*77}")
    print(f"  {'Mean pairwise block correlation':<45} | {emb_mean_corr:<10.4f} | "
          f"{prob_mean_corr:<10.4f} | {emb_mean_corr - prob_mean_corr:<+10.4f}")
    print(f"  {'Effective rank (full space)':<45} | {emb_cov['effective_rank']:<10.2f} | "
          f"{prob_cov['effective_rank']:<10.2f} | "
          f"{emb_cov['effective_rank'] - prob_cov['effective_rank']:<+10.2f}")
    print(f"  {'Condition number':<45} | {emb_cov['condition_number']:<10.1f} | "
          f"{prob_cov['condition_number']:<10.1f} | "
          f"{emb_cov['condition_number'] - prob_cov['condition_number']:<+10.1f}")
    print(f"  {'Top-5 PC variance explained (%)':<45} | "
          f"{emb_cov['var_explained_top5']*100:<10.1f} | "
          f"{prob_cov['var_explained_top5']*100:<10.1f} | "
          f"{(emb_cov['var_explained_top5'] - prob_cov['var_explained_top5'])*100:<+10.1f}")
    print(f"  {'Top-10 PC variance explained (%)':<45} | "
          f"{emb_cov['var_explained_top10']*100:<10.1f} | "
          f"{prob_cov['var_explained_top10']*100:<10.1f} | "
          f"{(emb_cov['var_explained_top10'] - prob_cov['var_explained_top10'])*100:<+10.1f}")

    # ── 5. Recommendation ──
    print("\n" + "=" * 75)
    print("5. RECOMMENDATION")
    print("=" * 75)

    if emb_mean_corr < 0.5:
        rec = ("PASS — Penultimate feature routing is viable.\n"
               f"  Mean embedding correlation = {emb_mean_corr:.4f} (< 0.5).\n"
               "  The 192-dim embeddings are substantially more diverse than\n"
               "  the 316-dim probability features. Proceed to Exp 19:\n"
               "  implement Linear(192,3) penultimate feature routing.")
        print(f"\n  ✅ {rec}")
    elif emb_mean_corr < 0.6:
        rec = ("BORDERLINE — Embeddings are somewhat more diverse.\n"
               f"  Mean embedding correlation = {emb_mean_corr:.4f} (0.5–0.6).\n"
               "  Penultimate routing may still help but gains will be modest.\n"
               "  Proceed to Exp 19 but set expectations accordingly.")
        print(f"\n  ⚠️  {rec}")
    else:
        rec = ("FAIL — Embeddings are as correlated as probabilities.\n"
               f"  Mean embedding correlation = {emb_mean_corr:.4f} (≥ 0.6).\n"
               "  The 192-dim embeddings share the same cross-expert\n"
               "  correlation problem as the 316-dim probabilities.\n"
               "  **Pivot to expert diversification strategies**\n"
               "  (RIDE-style diversity losses, different backbone\n"
               "  architectures, or non-parametric SADE routing).")
        print(f"\n  ❌ {rec}")

    # Show relative improvement
    rel_improvement = (prob_mean_corr - emb_mean_corr) / prob_mean_corr * 100
    print(f"\n  Relative improvement: correlation dropped by "
          f"{rel_improvement:.1f}% (from {prob_mean_corr:.4f} to {emb_mean_corr:.4f}).")
    print("=" * 75)

    return {
        'embedding_mean_corr': emb_mean_corr,
        'prob_mean_corr': prob_mean_corr,
        'embedding_cov': emb_cov,
        'prob_cov': prob_cov,
        'recommendation': rec,
        'n_samples': N,
    }


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
                           freq_features=recipe['freq_features'],
                           gate_input_mode=recipe['gate_input_mode']).to(device)

    gate = GateMLP(input_dim=recipe['input_dim'],
                   num_experts=3,
                   linear_router=recipe['linear_router']).to(device)
    print(f"[INFO] Loading Gate from {custom_args.gate_ckpt}")

    try:
        gate.load_state_dict(gate_ckpt['gate_state_dict'])
    except RuntimeError as e:
        print(f"[ERROR] Gate architecture mismatch.\n"
              f"  Recipe: freq_features={recipe['freq_features']}, "
              f"linear_router={recipe['linear_router']}\n"
              f"  GateMLP input_dim={gate._input_dim}\n"
              f"  Checkpoint fc.weight shape: "
              f"{gate_ckpt['gate_state_dict']['fc.weight'].shape}\n"
              f"  Error: {e}")
        sys.exit(1)
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
    LinearWeightPeakAnalyzer(
        gate, peak_probs,
        gate_input_mode=recipe['gate_input_mode'],
        num_classes=cfg.num_classes,
    ).run()

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

    # 9. DaWin Assumption Diagnostic (Confidently-Wrong Analysis)
    if custom_args.diagnose_confident_wrong:
        print("\n[INFO] Running in --diagnose_confident_wrong mode only.\n")
    run_confident_wrong_diagnostic(
        p_ce_test, p_la_test, p_bs_test, labels_test,
        cfg.cls_num_list, recipe,
        p_ce_tune=p_ce_tune, p_la_tune=p_la_tune,
        p_bs_tune=p_bs_tune, labels_tune=labels_tune,
    )
    if custom_args.diagnose_confident_wrong:
        print("\n[INFO] --diagnose_confident_wrong complete. Exiting.\n")
        return

    # 10. Embedding Correlation Diagnostic (Exp 18, Phase A)
    # Verifies whether 192-dim penultimate embeddings are more diverse than
    # the 316-dim probability features. This determines whether penultimate
    # feature routing (#2) is viable.
    if custom_args.diagnose_embeddings:
        print("\n[INFO] Running in --diagnose_embeddings mode only.\n")
    compute_embedding_correlation_diagnostic(
        model, test_loader, device, cfg.cls_num_list, n_samples=500,
    )
    if custom_args.diagnose_embeddings:
        print("\n[INFO] --diagnose_embeddings complete. Exiting.\n")
        return

if __name__ == "__main__":
    main()