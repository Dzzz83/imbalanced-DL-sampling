#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
from sklearn.model_selection import train_test_split

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
from imbalanceddl.utils.plugin_rule import define_groups_2, tune_plugin_for_rho, evaluate_plugin_for_rho
from imbalanceddl.utils.debug.models import ExpertEnsemble, GateMLP
from imbalanceddl.utils.gate_features import build_mixture
from imbalanceddl.utils.debug.extraction import recipe_from_checkpoint
from torch.utils.data import DataLoader, Subset

def chows_rule_risk_balanced(p_tune, labels_tune, p_test, labels_test, group_ids, rho, mode='bal'):
    confs = np.max(p_tune, axis=1)
    threshold = np.percentile(confs, rho * 100)
    
    test_confs = np.max(p_test, axis=1)
    accepted = test_confs >= threshold
    coverage = np.mean(accepted)
    
    preds = np.argmax(p_test, axis=1)
    label_groups = group_ids[labels_test]
    K = len(np.unique(group_ids))
    
    risks_k = []
    for k in range(K):
        mask = (label_groups == k) & accepted
        if np.sum(mask) == 0:
            risks_k.append(1.0)
        else:
            err = np.sum(preds[mask] != labels_test[mask])
            risks_k.append(err / np.sum(mask))
            
    risk = np.max(risks_k) if mode == 'worst' else np.mean(risks_k)
    return coverage, risk

def main():
    cfg = get_args()
    
    if cfg.dataset == 'cifar100':
        cfg.num_classes = 100
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*100)
    print("CRISP STAGE 3 PLUG-IN RULE VERIFICATION (FIXED)")
    print("="*100)

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

    print(f"[INFO] Loading Gate from {custom_args.gate_ckpt}")
    gate_ckpt = torch.load(custom_args.gate_ckpt, map_location='cpu', weights_only=False)

    # Reconstruct the checkpoint's exact mixture recipe (per-expert temps,
    # k, mixture space, gate/mixture temperatures).
    la_tau = 1.5
    match_tau = re.search(r't([\d\.]+)', os.path.basename(custom_args.la_path))
    if match_tau:
        la_tau = float(match_tau.group(1))
    print(f"[INFO] Using LA Tau = {la_tau} parsed from filename")

    recipe = recipe_from_checkpoint(gate_ckpt, cfg, la_tau=la_tau)
    T = recipe['T']
    expert_temps = recipe['expert_temps']
    k = recipe['k']
    space = recipe['space']
    weight_floor = recipe['weight_floor']
    gate_temp = recipe['gate_temp']
    mix_temp = recipe['mix_temp']
    print(f"[INFO] Recipe: T={T}, expert_temps={expert_temps}, k={k}, "
          f"space={space}, gate_temp={gate_temp:.3f}, mix_temp={mix_temp:.3f}")

    model = ExpertEnsemble(cfg, device, ckpt_paths,
                           expert_T=expert_temps,
                           normalize_blocks=recipe['norm_blocks'],
                           freq_features=recipe['freq_features'],
                           gate_input_mode=recipe['gate_input_mode']).to(device)

    gate = GateMLP(input_dim=recipe['input_dim'],
                   num_experts=3,
                   linear_router=recipe['linear_router']).to(device)
    try:
        gate.load_state_dict(gate_ckpt['gate_state_dict'])
    except RuntimeError as e:
        print(f"[ERROR] Gate architecture mismatch for {custom_args.gate_ckpt}.\n"
              f"  Recipe: freq_features={recipe['freq_features']}, "
              f"linear_router={recipe['linear_router']}\n"
              f"  GateMLP input_dim={gate._input_dim}\n"
              f"  Checkpoint fc.weight shape: "
              f"{gate_ckpt['gate_state_dict']['fc.weight'].shape}\n"
              f"  Error: {e}")
        sys.exit(1)
    gate.eval()

    def extract_posteriors(loader):
        all_p_mix = []
        all_labels = []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                logits_list, embeddings = model(images)

                gate_logits = gate(embeddings) / gate_temp
                weights = F.softmax(gate_logits, dim=1)

                p_mix = build_mixture(
                    logits_list, weights, cfg.cls_num_list, la_tau,
                    T=T, per_expert_T=expert_temps, k=k, space=space,
                    weight_floor=weight_floor, mix_temperature=mix_temp,
                )

                all_p_mix.append(p_mix.cpu().numpy())
                all_labels.append(labels.numpy())
        return np.concatenate(all_p_mix, axis=0), np.concatenate(all_labels, axis=0)

    print("\n[INFO] Extracting posteriors...")
    p_tune, labels_tune = extract_posteriors(tune_loader)
    p_test, labels_test = extract_posteriors(test_loader)

    group_ids = define_groups_2(cfg.cls_num_list)
    
    print("\n" + "="*80)
    print("1. PARAMETER TUNING CHECK (at 20% Rejection Rate)")
    print("="*80)
    
    alpha_bal, mu_bal = tune_plugin_for_rho(p_tune, labels_tune, group_ids, rho=0.2, mode='bal')
    alpha_wst, mu_wst = tune_plugin_for_rho(p_tune, labels_tune, group_ids, rho=0.2, mode='worst')
    
    print(f"Tuned Alpha (Balanced): {alpha_bal}")
    print(f"Tuned Mu (Balanced):    {mu_bal}")
    print(f"Tuned Alpha (Worst):    {alpha_wst}")
    print(f"Tuned Mu (Worst):       {mu_wst}")

    print("\n" + "="*80)
    print("2. RISK VS. COVERAGE COMPARISON (Lower Risk is Better)")
    print("="*80)
    print(f"{'Target Rej':<12} | {'Method':<15} | {'Coverage':<10} | {'Risk (Bal)':<12} | {'Risk (Wst)':<12}")
    print("-"*70)
    
    for rho in [0.0, 0.2, 0.4, 0.6]:
        cov_chow, risk_chow_bal = chows_rule_risk_balanced(p_tune, labels_tune, p_test, labels_test, group_ids, rho, mode='bal')
        _, risk_chow_wst = chows_rule_risk_balanced(p_tune, labels_tune, p_test, labels_test, group_ids, rho, mode='worst')
        print(f"{rho*100:<12.0f} | {'Chow':<15} | {cov_chow:<10.2f} | {risk_chow_bal:<12.4f} | {risk_chow_wst:<12.4f}")
        
        cov_bal, risk_bal = evaluate_plugin_for_rho(p_test, labels_test, group_ids, alpha_bal, mu_bal, rho, mode='bal')
        print(f"{'':<12} | {'Plug-in[Bal]':<15} | {cov_bal:<10.2f} | {risk_bal:<12.4f} | {'N/A':<12}")
        
        cov_wst, risk_wst = evaluate_plugin_for_rho(p_test, labels_test, group_ids, alpha_wst, mu_wst, rho, mode='worst')
        print(f"{'':<12} | {'Plug-in[Wst]':<15} | {cov_wst:<10.2f} | {'N/A':<12} | {risk_wst:<12.4f}")
        print("-"*70)

if __name__ == "__main__":
    main()