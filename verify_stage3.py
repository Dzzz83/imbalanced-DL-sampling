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
from imbalanceddl.utils.gate_features import gate_input_dim
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
    model = ExpertEnsemble(cfg, device, ckpt_paths).to(device)
    
    gate = GateMLP(input_dim=gate_input_dim(cfg.num_classes), num_experts=3).to(device)
    print(f"[INFO] Loading Gate from {custom_args.gate_ckpt}")
    gate_ckpt = torch.load(custom_args.gate_ckpt, map_location='cpu', weights_only=False)
    gate.load_state_dict(gate_ckpt['gate_state_dict'])
    gate.eval()

    T = gate_ckpt.get('temperature', 1.0)
    print(f"[INFO] Using Temperature T={T} from gate checkpoint")

    la_tau = 1.5
    match_tau = re.search(r't([\d\.]+)', os.path.basename(custom_args.la_path))
    if match_tau:
        la_tau = float(match_tau.group(1))
    print(f"[INFO] Using LA Tau = {la_tau} parsed from filename")
    
    cls_num_list = torch.tensor(cfg.cls_num_list, device=device, dtype=torch.float32)
    log_prior = torch.log(cls_num_list / cls_num_list.sum() + 1e-12)
    log_spc = torch.log(cls_num_list + 1e-12)

    def extract_posteriors(loader):
        all_p_mix = []
        all_labels = []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                logits_list, embeddings = model(images)
                
                p_ce = F.softmax(logits_list[0] / T, dim=1)
                p_la = F.softmax((logits_list[1] + la_tau * log_prior) / T, dim=1)
                p_bs = F.softmax((logits_list[2] + log_spc) / T, dim=1)
                probs = [p_ce, p_la, p_bs]
                
                # Use 192-dim embeddings directly
                gate_logits = gate(embeddings)
                weights = F.softmax(gate_logits, dim=1)
                
                k = getattr(cfg, 'routing_sparsity', 2)
                topk_weights, topk_indices = torch.topk(weights, k, dim=1)
                topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)
                
                stacked_probs = torch.stack(probs, dim=1)
                mix_prob = torch.zeros_like(stacked_probs[:, 0, :])
                for i in range(k):
                    idx = topk_indices[:, i]
                    w = topk_weights[:, i].unsqueeze(1)
                    expert_probs = stacked_probs[torch.arange(images.size(0)), idx, :]
                    mix_prob += w * expert_probs

                all_p_mix.append(mix_prob.cpu().numpy())
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