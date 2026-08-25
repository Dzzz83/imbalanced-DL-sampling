#!/usr/bin/env python3
"""debug_routing_signal.py — verify WHY routing hurts (round-2 diagnosis).

Runs entirely on an EXISTING gate checkpoint + the frozen experts (no
training). Answers, with numbers:

  H1 (top-2 truncation): does dropping the 3rd expert cause the head-class
     damage?  -> per-group accuracy of the gated mixture for k in {1, 2, 3}.
  H2 (noise routing):   are the gate's weight deviations net-harmful?
     -> accuracy when interpolating weights toward uniform, t in {0, .5, 1}.
  H3 (tune wants uniform): -> balanced accuracy vs gate temperature on tune.
  H4 (agreement):       routing cannot matter when experts agree ->
     per-subset accuracy (gated vs uniform must be IDENTICAL on agree set).
  H5 (signal quality):  gate-vs-oracle match per group on disagree samples,
     and the correctness-forecasting ceiling (logistic on gate features).

Usage:
  python debug_routing_signal.py \
      --ce_path <CE.pth> --la_path <LA.pth> --bs_path <BS.pth> \
      --gate_ckpt <gate_checkpoint_*.pth> -c <config.yaml>
"""
import os
import sys
import argparse
import re

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

custom_parser = argparse.ArgumentParser(add_help=False)
custom_parser.add_argument('--ce_path', type=str, required=True)
custom_parser.add_argument('--la_path', type=str, required=True)
custom_parser.add_argument('--bs_path', type=str, required=True)
custom_parser.add_argument('--gate_ckpt', type=str, required=True)
custom_args, remaining_argv = custom_parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_argv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from imbalanceddl.utils.config import get_args
from imbalanceddl.dataset.imbalance_dataset import ImbalancedDataset
from imbalanceddl.utils.plugin_rule import define_groups
from imbalanceddl.utils.debug.models import ExpertEnsemble, GateMLP
from imbalanceddl.utils.debug.evaluation import recipe_from_checkpoint
from imbalanceddl.utils.gate_features import (
    gate_input_dim, calibrate_expert_probs, build_gate_input,
    build_mixture, uniform_weights, expert_disagreement,
)

GROUP_NAMES = {0: 'Head', 1: 'Med', 2: 'Tail'}


def bal_acc(p_mix, labels, classes=None):
    preds = p_mix.argmax(dim=1).numpy()
    labels = labels.numpy()
    if classes is None:
        classes = range(p_mix.size(1))
    accs = [np.mean(preds[labels == c] == c)
            for c in classes if np.sum(labels == c) > 0]
    return float(np.mean(accs)) * 100 if accs else 0.0


def group_accs(p_mix, labels, group_ids):
    return {
        g: bal_acc(p_mix, labels, classes=np.where(group_ids == g)[0])
        for g in np.unique(group_ids)
    }


def fmt_group(accs):
    return " | ".join(f"{GROUP_NAMES[g]}: {accs[g]:6.2f}" for g in sorted(accs))


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
    cfg.cls_num_list = np.bincount(train_targets, minlength=cfg.num_classes).tolist()

    val_targets = np.array(val_dataset.targets)
    val_indices = np.arange(len(val_targets))
    tune_idx, test_idx = train_test_split(val_indices, test_size=0.8,
                                          stratify=val_targets,
                                          random_state=cfg.seed)
    tune_loader = DataLoader(Subset(val_dataset, tune_idx), batch_size=128,
                             shuffle=False, num_workers=4)
    test_loader = DataLoader(Subset(val_dataset, test_idx), batch_size=128,
                             shuffle=False, num_workers=4)

    gate_ckpt = torch.load(custom_args.gate_ckpt, map_location='cpu',
                           weights_only=False)
    la_tau = 1.5
    match_tau = re.search(r't([\d\.]+)',
                          os.path.basename(custom_args.la_path))
    if match_tau:
        la_tau = float(match_tau.group(1))
    recipe = recipe_from_checkpoint(gate_ckpt, cfg, la_tau=la_tau)
    T = recipe['T']
    expert_temps = recipe['expert_temps']
    space = recipe['space']
    weight_floor = recipe['weight_floor']
    gate_temp = recipe['gate_temp']
    mix_temp = recipe['mix_temp']
    k_recipe = recipe['k']
    print(f"[INFO] Recipe: T={T} expert_temps={expert_temps} k={k_recipe} "
          f"space={space} gate_temp={gate_temp:.3f} mix_temp={mix_temp:.3f}")

    ckpt_paths = {'CE': custom_args.ce_path, 'LA': custom_args.la_path,
                  'BS': custom_args.bs_path}
    model = ExpertEnsemble(cfg, device, ckpt_paths,
                           expert_T=expert_temps,
                           normalize_blocks=recipe['norm_blocks'],
                           freq_features=recipe['freq_features']).to(device)
    gate = GateMLP(input_dim=gate_input_dim(cfg.num_classes,
                                            freq_features=recipe['freq_features']),
                   num_experts=3,
                   linear_router=recipe.get('linear_router', False)).to(device)
    gate.load_state_dict(gate_ckpt['gate_state_dict'])
    gate.eval()

    group_ids = define_groups(cfg.cls_num_list)  # 0=Head, 1=Med, 2=Tail

    def cache(loader):
        logits = [[], [], []]
        labels = []
        with torch.no_grad():
            for images, lab in loader:
                images = images.to(device)
                ll, _ = model(images)
                for i in range(3):
                    logits[i].append(ll[i].cpu())
                labels.append(lab)
        return [torch.cat(l, dim=0) for l in logits], torch.cat(labels)

    print("[INFO] Caching tune/test expert logits...")
    tune_logits, tune_labels = cache(tune_loader)
    test_logits, test_labels = cache(test_loader)

    @torch.no_grad()
    def get_weights(logits, tg):
        probs = calibrate_expert_probs(logits, cfg.cls_num_list, la_tau,
                                       T=1.0, per_expert_T=expert_temps)
        emb = build_gate_input(
            probs, normalize_blocks=recipe['norm_blocks'],
            cls_num_list=cfg.cls_num_list if recipe['freq_features'] else None,
        )
        g = gate(emb.to(device)).cpu()
        return F.softmax(g / tg, dim=1), probs

    def mixture(logits, weights, k):
        return build_mixture(logits, weights, cfg.cls_num_list, la_tau,
                             T=T, per_expert_T=expert_temps, k=k,
                             space=space, weight_floor=weight_floor,
                             mix_temperature=mix_temp)

    w_test, probs_test = get_weights(test_logits, gate_temp)
    w_tune, probs_tune = get_weights(tune_logits, gate_temp)
    u_test = uniform_weights(w_test.size(0), 3)
    u_tune = uniform_weights(w_tune.size(0), 3)

    print("\n" + "=" * 90)
    print("H1: TOP-K TRUNCATION — per-group acc of the gated mixture, k in {1,2,3}")
    print("=" * 90)
    print(f"{'k':<4} | {'Bal':<7} | {'Head':<7} | {'Med':<7} | {'Tail':<7} | note")
    print("-" * 90)
    p_unif = mixture(test_logits, u_test, None)
    for k in [1, 2, 3]:
        p_g = mixture(test_logits, w_test, k)
        accs = group_accs(p_g, test_labels, group_ids)
        bal = bal_acc(p_g, test_labels)
        note = "= uniform (truncation free)" if k >= 3 else ""
        print(f"{k:<4} | {bal:<7.2f} | {accs[0]:<7.2f} | {accs[1]:<7.2f} | "
              f"{accs[2]:<7.2f} | {note}")
    accs_u = group_accs(p_unif, test_labels, group_ids)
    print(f"{'U':<4} | {bal_acc(p_unif, test_labels):<7.2f} | {accs_u[0]:<7.2f} | "
          f"{accs_u[1]:<7.2f} | {accs_u[2]:<7.2f} | uniform (k=all)")
    print("[READ] If k=3 recovers the head loss vs k=2 -> truncation is the "
          "damage mechanism; set routing_sparsity: 3.")

    print("\n" + "=" * 90)
    print("H2: WEIGHT INTERPOLATION — acc(w -> uniform), t in {0, 0.5, 1}")
    print("=" * 90)
    print(f"{'t':<4} | {'k':<4} | {'Bal':<7} | {'Head':<7} | {'Med':<7} | {'Tail':<7}")
    print("-" * 90)
    for t in [0.0, 0.5, 1.0]:
        w_t = (1 - t) * w_test + t * u_test
        for k in [k_recipe, 3]:
            p_g = mixture(test_logits, w_t, k)
            accs = group_accs(p_g, test_labels, group_ids)
            print(f"{t:<4.1f} | {k:<4} | {bal_acc(p_g, test_labels):<7.2f} | "
                  f"{accs[0]:<7.2f} | {accs[1]:<7.2f} | {accs[2]:<7.2f}")
    print("[READ] If bal acc rises as t->1, the gate's deviations are "
          "net-harmful -> add gate_kl_uniform (and/or disagree weighting).")

    print("\n" + "=" * 90)
    print("H3: GATE TEMPERATURE — tune bal acc vs T_gate (routing sharpness)")
    print("=" * 90)
    grid = [0.3, 0.5, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0, 4.0, 6.0, 8.0]
    print(f"{'T_gate':<7} | {'Bal (tune)':<11} | {'Bal (test)':<11}")
    print("-" * 90)
    for tg in grid:
        w_tg, _ = get_weights(tune_logits, tg)
        bal_tune = bal_acc(mixture(tune_logits, w_tg, k_recipe), tune_labels)
        w_tg_t, _ = get_weights(test_logits, tg)
        bal_test = bal_acc(mixture(test_logits, w_tg_t, k_recipe), test_labels)
        star = "  <-- fitted" if abs(tg - gate_temp) < 1e-6 else ""
        print(f"{tg:<7.1f} | {bal_tune:<11.2f} | {bal_test:<11.2f}{star}")
    print("[READ] Monotone rise with T_gate (and fitted value at the soft "
          "edge) => routing is net-negative; regularize or go k=3.")

    print("\n" + "=" * 90)
    print("H4: AGREEMENT SPLIT — routing cannot matter when experts agree")
    print("=" * 90)
    disagree = expert_disagreement(probs_test).numpy()
    p_unif_t = mixture(test_logits, u_test, None)
    p_gated = mixture(test_logits, w_test, k_recipe)
    for name, mask in [('agree', ~disagree), ('disagree', disagree)]:
        n = int(mask.sum())
        if n == 0:
            continue
        bal_u = bal_acc(p_unif_t[mask], test_labels[mask])
        bal_g = bal_acc(p_gated[mask], test_labels[mask])
        print(f"{name:<9} | n={n:<6} | uniform: {bal_u:6.2f} | "
              f"gated(k={k_recipe}): {bal_g:6.2f} | "
              f"delta: {bal_g - bal_u:+.2f}")
    print("[READ] On the agree subset, gated == uniform (sanity, delta ~ 0). "
          "The gate's value (or damage) lives entirely in the disagree subset.")

    print("\n" + "=" * 90)
    print("H5a: GATE vs ORACLE — on disagree samples, does the gate pick the "
         "right expert? (chance = 1/3)")
    print("=" * 90)
    B = test_labels.size(0)
    true_probs = torch.stack(
        [p[torch.arange(B), test_labels] for p in probs_test], dim=1)
    oracle = true_probs.argmax(dim=1).numpy()
    w_np = w_test.numpy()
    for g in np.unique(group_ids):
        mask = disagree & (group_ids[test_labels.numpy()] == g)
        if mask.sum() == 0:
            continue
        top1 = w_np[mask].argmax(1)
        top2 = np.argsort(w_np[mask], axis=1)[:, ::-1][:, :2]
        m1 = np.mean(top1 == oracle[mask]) * 100
        m2 = np.mean(np.any(top2 == oracle[mask][:, None], axis=1)) * 100
        print(f"{GROUP_NAMES[g]:<5} | n={mask.sum():<6} | top-1 match: "
              f"{m1:5.1f}% | top-2 match: {m2:5.1f}%")
    print("[READ] top-1 match < 33% => anti-predictive; ~33% => pure noise; "
          ">40% => real signal the gate is failing to exploit.")

    print("\n" + "=" * 90)
    print("H5b: CONFIDENCE vs CORRECTNESS (per expert, per group, test)")
    print("=" * 90)
    print(f"{'Expert':<6} | {'Group':<5} | {'mean max-prob | correct':<24} | "
          f"{'mean max-prob | wrong':<22} | delta")
    print("-" * 90)
    for j, name in enumerate(['CE', 'LA', 'BS']):
        conf = probs_test[j].max(dim=1).values.numpy()
        pred = probs_test[j].argmax(dim=1).numpy()
        correct = pred == test_labels.numpy()
        for g in np.unique(group_ids):
            mask = group_ids[test_labels.numpy()] == g
            if mask.sum() == 0 or correct[mask].sum() == 0:
                continue
            c_c = conf[mask & correct].mean()
            c_w = conf[mask & ~correct].mean()
            print(f"{name:<6} | {GROUP_NAMES[g]:<5} | {c_c:<24.4f} | "
                  f"{c_w:<22.4f} | {c_c - c_w:+.4f}")
    print("[READ] If delta is ~0 or negative (confident-wrong), confidence "
          "features are anti-predictive on that group — the gate is routing "
          "on a misleading signal.")

    print("\n" + "=" * 90)
    print("H5c: CORRECTNESS-FORECASTING CEILING (logistic on gate features)")
    print("=" * 90)
    with torch.no_grad():
        probs_tune = calibrate_expert_probs(
            tune_logits, cfg.cls_num_list, la_tau, T=1.0,
            per_expert_T=expert_temps)
        emb_tune = build_gate_input(probs_tune,
                                    normalize_blocks=recipe['norm_blocks'])
        emb_test = build_gate_input(probs_test,
                                    normalize_blocks=recipe['norm_blocks'])
    X_t, X_e = emb_tune.numpy(), emb_test.numpy()
    y_t, y_e = tune_labels.numpy(), test_labels.numpy()
    aucs, ceiling_w = [], []
    for j in range(3):
        c_t = (probs_tune[j].argmax(1).numpy() == y_t).astype(float)
        c_e = (probs_test[j].argmax(1).numpy() == y_e).astype(float)
        lr = LogisticRegression(C=0.1, max_iter=2000)
        lr.fit(X_t, c_t)
        p_hat = lr.predict_proba(X_e)[:, 1]
        ceiling_w.append(p_hat)
        for g in np.unique(group_ids):
            mask = group_ids[y_e] == g
            try:
                auc = roc_auc_score(c_e[mask], p_hat[mask])
            except ValueError:
                auc = float('nan')
            if g == 0:
                aucs.append((j, auc))
            print(f"  expert {'CE' if j == 0 else 'LA' if j == 1 else 'BS'} "
                  f"| AUC {GROUP_NAMES[g]}: {auc:.3f}")
    ceil_w = np.stack(ceiling_w, axis=1)
    ceil_w = ceil_w / ceil_w.sum(1, keepdims=True)
    for k in [k_recipe, 3]:
        p_c = build_mixture(test_logits,
                            torch.from_numpy(ceil_w).float(),
                            cfg.cls_num_list, la_tau, T=T,
                            per_expert_T=expert_temps, k=k, space=space,
                            weight_floor=0.0, mix_temperature=1.0)
        accs = group_accs(p_c, test_labels, group_ids)
        print(f"  ceiling routing (k={k}): bal {bal_acc(p_c, test_labels):.2f} "
              f"| {fmt_group(accs)}")
    print(f"  uniform baseline      : bal {bal_acc(p_unif, test_labels):.2f} "
          f"| {fmt_group(accs_u)}")
    print("[READ] The ceiling is the best a *well-trained* router on these "
          "features could do. If it is only ~1 pp above uniform, the gate is "
          "already near its feature limit and gains must come from features "
          "(penultimate) or policy (two-stage), not more training.")

    print("\n" + "=" * 90)
    print("H5d: WEIGHT DEVIATION STATS (|w - uniform|, test)")
    print("=" * 90)
    dev = np.abs(w_np - 1.0 / 3).sum(1)
    labels_np = test_labels.numpy()
    for g in np.unique(group_ids):
        mask = group_ids[labels_np] == g
        d = dev[mask]
        frac = np.mean(d > 0.1) * 100
        print(f"{GROUP_NAMES[g]:<5} | mean |w-u|_1: {d.mean():.4f} | "
              f"frac deviating >0.1: {frac:5.1f}%")
    print("[READ] Large mean deviation on Head with small top-1 oracle match "
          "=> confident noise routing on head.")

    print("\n" + "=" * 90)
    print("H6: CLASS/GROUP-CONDITIONAL RULE BENCHMARK (no MLP, tune-estimated)")
    print("=" * 90)
    print("[INFO] Rule: for each frequency group, weights = softmax of the "
          "group's per-expert accuracy on tune; applied per sample from the "
          "uniform mixture's predicted class. LA alone gets Low 12.21 — if "
          "this rule beats the MLP gate, per-class priors are the better "
          "policy.")
    with torch.no_grad():
        probs_tune = calibrate_expert_probs(
            tune_logits, cfg.cls_num_list, la_tau, T=1.0,
            per_expert_T=expert_temps)
    y_t = tune_labels.numpy()
    p_tune_np = [p.numpy() for p in probs_tune]
    # Per-expert accuracy per group on tune.
    g_t = group_ids[y_t]
    acc_g = np.zeros((3, len(np.unique(group_ids))))
    for j in range(3):
        preds_j = p_tune_np[j].argmax(1)
        for g in np.unique(group_ids):
            m = g_t == g
            acc_g[j, g] = np.mean(preds_j[m] == y_t[m]) if m.sum() else 0.0
    for g in np.unique(group_ids):
        print(f"  tune acc {GROUP_NAMES[g]}: CE={acc_g[0, g]:.3f} "
              f"LA={acc_g[1, g]:.3f} BS={acc_g[2, g]:.3f}")
    w_group = np.exp(acc_g.T)          # (G, 3) softmax over experts
    w_group = w_group / w_group.sum(1, keepdims=True)
    # Apply on test via the uniform mixture's predicted class.
    p_unif_np = p_unif.numpy()
    pred_unif = p_unif_np.argmax(1)
    g_pred = group_ids[pred_unif]
    w_rule = w_group[g_pred]
    for k in [3, k_recipe]:
        p_r = build_mixture(test_logits,
                            torch.from_numpy(w_rule).float(),
                            cfg.cls_num_list, la_tau, T=T,
                            per_expert_T=expert_temps, k=k, space=space,
                            weight_floor=0.0, mix_temperature=1.0)
        accs = group_accs(p_r, test_labels, group_ids)
        print(f"  group-rule (k={k}): bal {bal_acc(p_r, test_labels):.2f} "
              f"| {fmt_group(accs)}")
    print(f"  MLP gate           : bal {bal_acc(p_gated, test_labels):.2f} "
          f"| {fmt_group(group_accs(p_gated, test_labels, group_ids))}")
    print("[READ] If the group-rule beats the MLP gate, replace per-sample "
          "routing with group-conditional weights (or add class-frequency "
          "features so the MLP can learn the same rule — round-3 fix).")

    print("\n" + "=" * 90)
    print("SUMMARY / DECISION RULES")
    print("=" * 90)
    print("1. H1: k=3 better than k=2 on Head  -> set routing_sparsity: 3")
    print("2. H2: bal acc rises as t->1         -> set gate_kl_uniform > 0")
    print("3. H3: tune bal acc rises with T_gate-> same as (2); soft gate")
    print("4. H5a: top-1 match < ~33% on Head   -> gate is noise/anti-predictive;")
    print("     regularize (2) or switch target to 'correctness'")
    print("5. H5c: ceiling close to uniform     -> stop tuning the gate; invest")
    print("     in features (penultimate) or two-stage routing instead")
    print("6. H6: group-rule beats the MLP      -> use group-conditional weights")
    print("     and/or enable gate_freq_features (round-3 fix)")


if __name__ == "__main__":
    main()
