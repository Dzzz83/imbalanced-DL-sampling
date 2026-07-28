#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
from imbalanceddl.net.network import build_model
from imbalanceddl.dataset.imbalance_dataset import ImbalancedDataset
from imbalanceddl.utils.gate_features import compute_gate_features
from imbalanceddl.utils.plugin_rule import define_groups_2, tune_plugin_for_rho, evaluate_plugin_for_rho
from torch.utils.data import DataLoader, Subset

class ExpertEnsemble(nn.Module):
    def __init__(self, cfg, device, ckpt_paths):
        super().__init__()
        self.experts = nn.ModuleList()
        for name, path in ckpt_paths.items():
            print(f"[INFO] Loading expert {name} from {path}")
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            has_bias = ckpt.get('bias', False)
            
            model = build_model(cfg)
            actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            actual_model.classifier = nn.Linear(actual_model.feature_len, actual_model.num_classes, bias=has_bias).to(device)
            model.load_state_dict(ckpt['state_dict'])
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
            self.experts.append(model.to(device))

    @torch.no_grad()
    def forward(self, x):
        logits_list = []
        for expert in self.experts:
            logits, _ = expert(x)
            logits_list.append(logits)
        return logits_list, None

class GateMLP(nn.Module):
    def __init__(self, input_dim=24, hidden1=256, hidden2=128, num_experts=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_experts)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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
            risks_k.append(0.0)
        else:
            err = np.sum(preds[mask] != labels_test[mask])
            risks_k.append(err / np.sum(mask))
            
    risk = np.max(risks_k) if mode == 'worst' else np.mean(risks_k)
    return coverage, risk

def main():
    cfg = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*100)
    print("CRISP STAGE 3 PLUG-IN RULE VERIFICATION (FIXED)")
    print("="*100)

    dataset = ImbalancedDataset(cfg, cfg.dataset, augmentation='none')
    train_dataset, val_dataset = dataset.train_val_sets
    
    train_targets = np.array(train_dataset.targets)
    train_indices = np.arange(len(train_targets))
    _, gate_idx = train_test_split(
        train_indices,
        test_size=1 - cfg.gate_split_ratio,   # 0.1 (10%)
        stratify=train_targets,
        random_state=cfg.seed
    )
    gate_dataset = Subset(train_dataset, gate_idx)
    gate_loader = DataLoader(gate_dataset, batch_size=128, shuffle=False, num_workers=4)

    # ---- Use the full validation set as test set ----
    test_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

    # Load experts and gate
    ckpt_paths = {'CE': custom_args.ce_path, 'LA': custom_args.la_path, 'BS': custom_args.bs_path}
    model = ExpertEnsemble(cfg, device, ckpt_paths).to(device)
    
    gate = GateMLP(input_dim=24, hidden1=cfg.gate_hidden_size, hidden2=cfg.gate_hidden_size2).to(device)
    print(f"[INFO] Loading Gate from {custom_args.gate_ckpt}")
    gate_ckpt = torch.load(custom_args.gate_ckpt, map_location='cpu')
    gate.load_state_dict(gate_ckpt['gate_state_dict'])
    gate.eval()

    T = gate_ckpt.get('temperature', 1.0)
    print(f"[INFO] Using Temperature T={T} from gate checkpoint")

    def extract_posteriors(loader):
        all_p_mix = []
        all_labels = []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                logits_list, _ = model(images)
                probs = [F.softmax(l / T, dim=1) for l in logits_list]
                phi = compute_gate_features(logits_list, probs)
                
                gate_logits = gate(phi)
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
    p_tune, labels_tune = extract_posteriors(gate_loader)   # tuning on gating split
    p_test, labels_test = extract_posteriors(test_loader)   # test on full validation set

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