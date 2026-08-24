#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import glob
import re

custom_parser = argparse.ArgumentParser(add_help=False)
custom_parser.add_argument('--ce_path', type=str, required=True)
custom_parser.add_argument('--la_path', type=str, required=True)
custom_parser.add_argument('--bs_path', type=str, required=True)
custom_parser.add_argument('--gate_dir', type=str, required=True)
custom_args, remaining_argv = custom_parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_argv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from imbalanceddl.utils.config import get_args
from imbalanceddl.dataset.imbalance_dataset import ImbalancedDataset
from imbalanceddl.utils.metrics import shot_acc
from imbalanceddl.utils.debug.models import ExpertEnsemble, GateMLP
from imbalanceddl.utils.gate_features import (
    gate_input_dim, calibrate_expert_probs, build_gate_input,
    build_mixture, uniform_weights,
)
from imbalanceddl.utils.debug.evaluation import recipe_from_checkpoint
from torch.utils.data import DataLoader

# ExpertEnsemble / GateMLP are imported from imbalanceddl.utils.debug.models,
# which build the same 312-dim calibrated-probability + confidence/agreement
# feature vector as the training path (see imbalanceddl.utils.gate_features).

def compute_ece(confidences, preds, labels, n_bins=15):
    accs = (preds == labels)
    bin_lowers = np.linspace(0, 1, n_bins + 1)[:-1]
    bin_uppers = np.linspace(0, 1, n_bins + 1)[1:]
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accs[in_bin])
            avg_conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_conf_in_bin - acc_in_bin) * prop_in_bin
    return ece

def get_accs(probs, labels, cfg, train_dataset):
    preds = np.argmax(probs, axis=1)
    bal = np.mean([np.mean(preds[labels == c] == c) for c in range(cfg.num_classes) if np.sum(labels == c) > 0]) * 100
    many, med, low = shot_acc(cfg, preds, labels, train_dataset, acc_per_cls=False)
    return bal, many*100, med*100, low*100

def get_calib(probs, labels, cfg):
    preds = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    
    cls_num_list = np.array(cfg.cls_num_list)
    priors = cls_num_list / cls_num_list.sum()
    sample_weights = priors[labels]
    sample_weights = sample_weights / sample_weights.sum()

    true_probs = probs[np.arange(len(labels)), labels]
    nll = -np.sum(sample_weights * np.log(true_probs + 1e-8))
    
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    brier = np.sum(sample_weights * np.sum((probs - one_hot)**2, axis=1))
    
    ece_all = compute_ece(conf, preds, labels)
    
    tail_mask = cls_num_list[labels] <= 20
    head_mask = ~tail_mask
    
    ece_tail = compute_ece(conf[tail_mask], preds[tail_mask], labels[tail_mask]) if np.sum(tail_mask) > 0 else 0.0
    ece_head = compute_ece(conf[head_mask], preds[head_mask], labels[head_mask]) if np.sum(head_mask) > 0 else 0.0
    
    return nll, brier, ece_all, ece_head, ece_tail

def main():
    cfg = get_args()
    if cfg.dataset == 'cifar100':
        cfg.num_classes = 100

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*100)
    print("CRISP STAGE 2 GATE VERIFICATION (FOLDER SCAN) - PAPER k=2 ROUTING")
    print("="*100)

    dataset = ImbalancedDataset(cfg, cfg.dataset, augmentation='none')
    train_dataset, val_dataset = dataset.train_val_sets
    
    train_targets = np.array(train_dataset.targets)
    cfg.cls_num_list = np.bincount(train_targets, minlength=cfg.num_classes).tolist()

    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

    ckpt_paths = {'CE': custom_args.ce_path, 'LA': custom_args.la_path, 'BS': custom_args.bs_path}
    model = ExpertEnsemble(cfg, device, ckpt_paths).to(device)

    print("\n[INFO] Caching raw expert logits on test set...")
    all_logits = [[], [], []]
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits_list, _ = model(images)
            for i in range(3):
                all_logits[i].append(logits_list[i].cpu())
            all_labels.append(labels)

    all_logits = [torch.cat(l, dim=0) for l in all_logits]
    labels = torch.cat(all_labels, dim=0).numpy()

    gate_files = sorted(glob.glob(os.path.join(custom_args.gate_dir, "*.pth")))
    if not gate_files:
        print(f"[ERROR] No checkpoints found in {custom_args.gate_dir}")
        sys.exit(1)

    print(f"[INFO] Found {len(gate_files)} gate checkpoints to evaluate.")

    la_tau = 1.5
    match_tau = re.search(r't([\d\.]+)', os.path.basename(custom_args.la_path))
    if match_tau:
        la_tau = float(match_tau.group(1))
    print(f"[INFO] Using LA Tau = {la_tau} parsed from filename")

    # Uniform baseline: first checkpoint's recipe with equal weights over all
    # experts (the fair "did routing help" comparison).
    first_fname = os.path.basename(gate_files[0])
    first_ckpt = torch.load(gate_files[0], map_location='cpu', weights_only=False)
    recipe0 = recipe_from_checkpoint(first_ckpt, cfg, la_tau=la_tau)
    print(f"[INFO] Computing Uniform Ensemble baseline (recipe of {first_fname}): "
          f"T={recipe0['T']}, expert_temps={recipe0['expert_temps']}, "
          f"space={recipe0['space']}")
    with torch.no_grad():
        unif_w = uniform_weights(labels.shape[0], 3)
        p_unif = build_mixture(
            all_logits, unif_w, cfg.cls_num_list, la_tau,
            T=recipe0['T'], per_expert_T=recipe0['expert_temps'], k=None,
            space=recipe0['space'], mix_temperature=1.0,
        )
    accs_unif = get_accs(p_unif.numpy(), labels, cfg, train_dataset)
    cal_unif = get_calib(p_unif.numpy(), labels, cfg)

    results = []

    for g_path in gate_files:
        fname = os.path.basename(g_path)
        clean_name = fname.replace("gate_checkpoint_", "").replace(".pth", "")

        # FIX: Added weights_only=False
        ckpt = torch.load(g_path, map_location='cpu', weights_only=False)

        # Reconstruct this checkpoint's exact mixture recipe (training/eval
        # consistency; see imbalanceddl.utils.debug.evaluation).
        recipe = recipe_from_checkpoint(ckpt, cfg, la_tau=la_tau)
        T = recipe['T']
        expert_temps = recipe['expert_temps']
        k = recipe['k']
        space = recipe['space']
        weight_floor = recipe['weight_floor']
        gate_temp = recipe['gate_temp']
        mix_temp = recipe['mix_temp']

        # Gate input dim depends on the checkpoint's freq_features flag
        # (round-3: 316 vs 312 dims) — build per checkpoint.
        gate = GateMLP(
            input_dim=gate_input_dim(cfg.num_classes,
                                     freq_features=recipe['freq_features']),
            num_experts=3,
        ).to(device)
        try:
            gate.load_state_dict(ckpt['gate_state_dict'])
        except RuntimeError:
            print(f"[WARNING] Skipping checkpoint {fname} due to "
                  f"architecture mismatch (stale 300-dim gate?).")
            continue
        gate.eval()

        with torch.no_grad():
            # Gate embeddings depend on the checkpoint's per-expert temps and
            # block-normalization, so build them per checkpoint from raw logits.
            probs = calibrate_expert_probs(
                all_logits, cfg.cls_num_list, la_tau, T=1.0,
                per_expert_T=expert_temps,
            )
            embeddings = build_gate_input(probs, normalize_blocks=recipe['norm_blocks'])

            gate_logits = gate(embeddings.to(device))
            weights = F.softmax(gate_logits / gate_temp, dim=1)

            p_mix = build_mixture(
                all_logits, weights.cpu(), cfg.cls_num_list, la_tau,
                T=T, per_expert_T=expert_temps, k=k, space=space,
                weight_floor=weight_floor, mix_temperature=mix_temp,
            )
            # Fair per-checkpoint uniform baseline (same recipe, equal weights).
            unif_w = uniform_weights(weights.size(0), 3)
            p_unif_cur = build_mixture(
                all_logits, unif_w, cfg.cls_num_list, la_tau,
                T=T, per_expert_T=expert_temps, k=None, space=space,
                mix_temperature=1.0,
            )

            p_mix_np = p_mix.numpy()
            weights_np = weights.cpu().numpy()

            accs_mix = get_accs(p_mix_np, labels, cfg, train_dataset)
            cal_mix = get_calib(p_mix_np, labels, cfg)
            accs_unif_cur = get_accs(p_unif_cur.numpy(), labels, cfg, train_dataset)

            avg_w_ce = np.mean(weights_np[:, 0])
            avg_w_la = np.mean(weights_np[:, 1])
            avg_w_bs = np.mean(weights_np[:, 2])

            beats_unif = "✅" if accs_mix[0] >= accs_unif_cur[0] else "❌"

            results.append({
                'name': clean_name,
                'bal_acc': accs_mix[0], 'many': accs_mix[1], 'med': accs_mix[2], 'low': accs_mix[3],
                'nll': cal_mix[0], 'brier': cal_mix[1], 'ece': cal_mix[2],
                'ece_head': cal_mix[3], 'ece_tail': cal_mix[4],
                'w_ce': avg_w_ce, 'w_la': avg_w_la, 'w_bs': avg_w_bs,
                'beats_unif': beats_unif
            })
            print(f"  Evaluated {clean_name:<25} | Bal Acc: {accs_mix[0]:.2f}% | {beats_unif}")

    print("\n" + "="*180)
    print("STAGE 2 METRICS SUMMARY (Gate Sweep vs Uniform Baseline) vs. PAPER (TABLE 3)")
    print("="*180)
    print(f"{'Checkpoint':<25} | {'Bal Acc':<7} | {'Many':<6} | {'Med':<6} | {'Low':<6} | {'NLL':<8} | {'Brier':<8} | {'ECE All':<8} | {'ECE Head':<8} | {'ECE Tail':<8} | {'w_CE':<6} | {'w_LA':<6} | {'w_BS':<6} | {'Beats Unif?':<12}")
    print("-"*180)
    
    print(f"{'PAPER CRISP':<25} | {'N/A':<7} | {'N/A':<6} | {'N/A':<6} | {'N/A':<6} | {'1.18':<8} | {'0.403':<8} | {'N/A':<8} | {'N/A':<8} | {'0.088':<8} | {'N/A':<6} | {'N/A':<6} | {'N/A':<6} | {'N/A':<12}")
    print("-"*180)
    
    print(f"{'UNIFORM BASELINE':<25} | {accs_unif[0]:<7.2f} | {accs_unif[1]:<6.2f} | {accs_unif[2]:<6.2f} | {accs_unif[3]:<6.2f} | {cal_unif[0]:<8.3f} | {cal_unif[1]:<8.3f} | {cal_unif[2]:<8.3f} | {cal_unif[3]:<8.3f} | {cal_unif[4]:<8.3f} | {'0.33':<6} | {'0.33':<6} | {'0.34':<6} | {'---':<12}")
    print("-"*180)
    
    results.sort(key=lambda x: -x['bal_acc'])
    
    for r in results:
        print(f"{r['name']:<25} | {r['bal_acc']:<7.2f} | {r['many']:<6.2f} | {r['med']:<6.2f} | {r['low']:<6.2f} | {r['nll']:<8.3f} | {r['brier']:<8.3f} | {r['ece']:<8.3f} | {r['ece_head']:<8.3f} | {r['ece_tail']:<8.3f} | {r['w_ce']:<6.3f} | {r['w_la']:<6.3f} | {r['w_bs']:<6.3f} | {r['beats_unif']:<12}")
    print("="*180)

if __name__ == "__main__":
    main()