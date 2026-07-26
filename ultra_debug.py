#!/usr/bin/env python3
# ultra_debug.py
# CRISP Pipeline Verification & Paper Comparison (Tables 2, 3 & Diagnostics)

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split

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
from imbalanceddl.net.network import build_model
from imbalanceddl.dataset.imbalance_dataset import ImbalancedDataset
from imbalanceddl.utils.gate_features import compute_gate_features
from imbalanceddl.utils.plugin_rule import define_groups, define_groups_2, compute_aurc_metrics
from imbalanceddl.utils.metrics import shot_acc
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

def compute_chow_aurc(p_tune, labels_tune, p_test, labels_test, group_ids, mode='bal'):
    rho_grid = np.arange(0.0, 1.1, 0.1)
    coverages = []
    risks = []
    for rho in rho_grid:
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
        coverages.append(coverage)
        risks.append(risk)
    sort_idx = np.argsort(coverages)
    coverages = np.array(coverages)[sort_idx]
    risks = np.array(risks)[sort_idx]
    if coverages[0] > 0:
        coverages = np.insert(coverages, 0, 0.0)
        # FIX: Risk at 0 coverage is 1.0 (or undefined, but 1.0 prevents AURC deflation)
        risks = np.insert(risks, 0, 1.0) 
    return np.trapz(risks, coverages)

def compute_all_metrics(probs, labels, logits=None, cfg=None, train_dataset=None):
    """Helper to compute all calibration and accuracy metrics with LT re-weighting."""
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    
    bal_acc = np.mean([np.mean(preds[labels == c] == c) for c in range(cfg.num_classes) if np.sum(labels == c) > 0]) * 100
    many, med, low = shot_acc(cfg, preds, labels, train_dataset, acc_per_cls=False)
    
    # L2R/CRISP Protocol: Re-weight the balanced test set to mimic the long-tailed training distribution
    cls_num_list = np.array(cfg.cls_num_list)
    priors = cls_num_list / cls_num_list.sum()
    sample_weights = priors[labels]
    sample_weights = sample_weights / sample_weights.sum()

    true_probs = probs[np.arange(len(labels)), labels]
    nll = -np.sum(sample_weights * np.log(true_probs + 1e-8))
    
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    brier = np.sum(sample_weights * np.sum((probs - one_hot)**2, axis=1))
    
    # Overall ECE
    accs = (preds == labels)
    bin_lowers = np.linspace(0, 1, 16)[:-1]
    bin_uppers = np.linspace(0, 1, 16)[1:]
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accs[in_bin])
            avg_conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_conf_in_bin - acc_in_bin) * prop_in_bin

    # Tail-ECE (Matches Paper Table 3)
    group_ids = define_groups_2(cfg.cls_num_list)
    label_groups = group_ids[labels]
    tail_mask = (label_groups == 1)
    tail_conf = confidences[tail_mask]
    tail_correct = accs[tail_mask]
    tail_ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (tail_conf > bin_lower) & (tail_conf <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(tail_correct[in_bin])
            avg_conf_in_bin = np.mean(tail_conf[in_bin])
            tail_ece += np.abs(avg_conf_in_bin - acc_in_bin) * prop_in_bin

    metrics = {
        'bal_acc': bal_acc, 'many': many * 100, 'med': med * 100, 'low': low * 100,
        'nll': nll, 'brier': brier, 'ece': ece, 'tail_ece': tail_ece
    }
    
    if logits is not None:
        max_logits = logits.max(dim=1)[0].numpy()
        metrics['mean_logit'] = np.mean(max_logits)
        metrics['sat_10'] = np.mean(max_logits > 10.0) * 100
        metrics['sat_20'] = np.mean(max_logits > 20.0) * 100
        
    return metrics

def main():
    cfg = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "="*80)
    print("ULTRA DEBUG: CRISP PIPELINE & PAPER COMPARISON")
    print("="*80)

    dataset = ImbalancedDataset(cfg, cfg.dataset, augmentation='none')
    train_dataset, val_dataset = dataset.train_val_sets
    
    # Paper Protocol: 80/20 split of the test set for tuning and evaluation
    val_targets = np.array(val_dataset.targets)
    val_indices = np.arange(len(val_targets))
    tune_idx, test_idx = train_test_split(val_indices, test_size=0.8, stratify=val_targets, random_state=cfg.seed)
    
    tune_dataset = Subset(val_dataset, tune_idx)
    test_dataset = Subset(val_dataset, test_idx)
    
    tune_loader = DataLoader(tune_dataset, batch_size=128, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    ckpt_paths = {'CE': custom_args.ce_path, 'LA': custom_args.la_path, 'BS': custom_args.bs_path}
    model = ExpertEnsemble(cfg, device, ckpt_paths).to(device)
    
    gate = GateMLP(input_dim=24, hidden1=cfg.gate_hidden_size, hidden2=cfg.gate_hidden_size2).to(device)
    print(f"[INFO] Loading Gate from {custom_args.gate_ckpt}")
    gate_ckpt = torch.load(custom_args.gate_ckpt, map_location='cpu')
    gate.load_state_dict(gate_ckpt['gate_state_dict'])
    gate.eval()

    # FIX: Extract the exact Temperature used during gate training from the checkpoint
    T = gate_ckpt.get('temperature', 1.0)
    print(f"[INFO] Using Temperature T={T} extracted from gate checkpoint")

    def extract_data(loader):
        all_logits = [[], [], []]
        all_labels = []
        all_weights = []
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                logits_list, _ = model(images)
                for i in range(3):
                    all_logits[i].append(logits_list[i])
                all_labels.append(labels)
                
            all_logits = [torch.cat(l, dim=0) for l in all_logits]
            labels = torch.cat(all_labels, dim=0)
            
            adj_probs = [
                F.softmax(all_logits[0] / T, dim=1),
                F.softmax(all_logits[1] / T, dim=1),
                F.softmax(all_logits[2] / T, dim=1)
            ]
            
            # Gate routing
            phi = compute_gate_features(all_logits, adj_probs)
            gate_logits = gate(phi)
            weights = F.softmax(gate_logits, dim=1)
            all_weights.append(weights.cpu().numpy())
            
            k = getattr(cfg, 'routing_sparsity', 2)
            topk_weights, topk_indices = torch.topk(weights, k, dim=1)
            topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)
            
            stacked_probs = torch.stack(adj_probs, dim=1)
            p_mix = torch.zeros_like(stacked_probs[:, 0, :])
            
            N = stacked_probs.size(0)
            for i in range(k):
                idx = topk_indices[:, i]
                w = topk_weights[:, i].unsqueeze(1)
                expert_probs = stacked_probs[torch.arange(N), idx, :]
                p_mix += w * expert_probs
                
            p_uniform = (adj_probs[0] + adj_probs[1] + adj_probs[2]) / 3.0
            
        avg_weights = np.mean(np.concatenate(all_weights, axis=0), axis=0)
            
        return (p_mix.cpu().numpy(), p_uniform.cpu().numpy(), 
                adj_probs[0].cpu().numpy(), adj_probs[1].cpu().numpy(), adj_probs[2].cpu().numpy(), 
                all_logits[0].cpu(), all_logits[1].cpu(), all_logits[2].cpu(), 
                avg_weights, labels.cpu().numpy())

    print("\n[INFO] Extracting posteriors...")
    (p_mix_tune, p_unif_tune, p_ce_tune, p_la_tune, p_bs_tune, 
     l_ce_tune, l_la_tune, l_bs_tune, w_tune, labels_tune) = extract_data(tune_loader)
     
    (p_mix_test, p_unif_test, p_ce_test, p_la_test, p_bs_test, 
     l_ce_test, l_la_test, l_bs_test, w_test, labels_test) = extract_data(test_loader)

    group_ids_2 = define_groups_2(cfg.cls_num_list)
    
    print("\n[INFO] Computing AURC & Calibration metrics...")
    # AURC Calculations
    chow_bal = compute_chow_aurc(p_mix_tune, labels_tune, p_mix_test, labels_test, group_ids_2, mode='bal')
    chow_wst = compute_chow_aurc(p_mix_tune, labels_tune, p_mix_test, labels_test, group_ids_2, mode='worst')
    
    la_bal = compute_aurc_metrics(p_la_tune, labels_tune, p_la_test, labels_test, group_ids_2, cls_num_list=cfg.cls_num_list, mode='bal')
    la_wst = compute_aurc_metrics(p_la_tune, labels_tune, p_la_test, labels_test, group_ids_2, cls_num_list=cfg.cls_num_list, mode='worst')
    
    unif_bal = compute_aurc_metrics(p_unif_tune, labels_tune, p_unif_test, labels_test, group_ids_2, cls_num_list=cfg.cls_num_list, mode='bal')
    unif_wst = compute_aurc_metrics(p_unif_tune, labels_tune, p_unif_test, labels_test, group_ids_2, cls_num_list=cfg.cls_num_list, mode='worst')
    
    crisp_bal = compute_aurc_metrics(p_mix_tune, labels_tune, p_mix_test, labels_test, group_ids_2, cls_num_list=cfg.cls_num_list, mode='bal')
    crisp_wst = compute_aurc_metrics(p_mix_tune, labels_tune, p_mix_test, labels_test, group_ids_2, cls_num_list=cfg.cls_num_list, mode='worst')

    # Diagnostic Metric Calculations
    m_ce = compute_all_metrics(p_ce_test, labels_test, l_ce_test, cfg, train_dataset)
    m_la = compute_all_metrics(p_la_test, labels_test, l_la_test, cfg, train_dataset)
    m_bs = compute_all_metrics(p_bs_test, labels_test, l_bs_test, cfg, train_dataset)
    m_unif = compute_all_metrics(p_unif_test, labels_test, None, cfg, train_dataset)
    m_crisp = compute_all_metrics(p_mix_test, labels_test, None, cfg, train_dataset)

    # --- Print Table 1: Paper vs Yours (Truthful Comparison) ---
    print("\n" + "="*100)
    print("TABLE 1: CRISP PAPER vs. YOUR IMPLEMENTATION (CIFAR-100-LT)")
    print("="*100)
    print(f"{'Metric':<25} | {'Paper (Top)':<20} | {'Yours (Bottom)':<20}")
    print("-"*70)
    
    print(f"{'Chow Bal AURC':<25} | {'0.509':<20} | {chow_bal:<20.4f}")
    print(f"{'Chow Wst AURC':<25} | {'0.883':<20} | {chow_wst:<20.4f}")
    print("-"*70)
    print(f"{'Single LA Bal AURC':<25} | {'0.287':<20} | {la_bal['AURC']:<20.4f}")
    print(f"{'Single LA Wst AURC':<25} | {'0.321':<20} | {la_wst['AURC']:<20.4f}")
    print("-"*70)
    print(f"{'Uniform Bal AURC':<25} | {'0.254':<20} | {unif_bal['AURC']:<20.4f}")
    print(f"{'Uniform Wst AURC':<25} | {'0.261':<20} | {unif_wst['AURC']:<20.4f}")
    print("-"*70)
    print(f"{'CRISP Bal AURC':<25} | {'0.253':<20} | {crisp_bal['AURC']:<20.4f}")
    print(f"{'CRISP Wst AURC':<25} | {'0.248':<20} | {crisp_wst['AURC']:<20.4f}")
    print("-"*70)
    print(f"{'CRISP NLL':<25} | {'1.18':<20} | {m_crisp['nll']:<20.4f}")
    print(f"{'CRISP Brier':<25} | {'0.403':<20} | {m_crisp['brier']:<20.4f}")
    print(f"{'CRISP tail-ECE':<25} | {'0.088':<20} | {m_crisp['tail_ece']:<20.4f}")
    print("="*100)

    # --- Print Table 2: Full Diagnostic Breakdown ---
    print("\n" + "="*140)
    print("TABLE 2: FULL DIAGNOSTIC BREAKDOWN (TEST SET)")
    print("="*140)
    print(f"{'Method':<10} | {'Bal Acc':<7} | {'Many':<6} | {'Med':<6} | {'Low':<6} | {'NLL':<8} | {'Brier':<8} | {'ECE':<8} | {'Tail ECE':<8} | {'Mean Logit':<10} | {'%>10':<6} | {'%>20':<6}")
    print("-"*140)
    
    def print_row(name, m):
        print(f"{name:<10} | {m['bal_acc']:<7.2f} | {m['many']:<6.2f} | {m['med']:<6.2f} | {m['low']:<6.2f} | {m['nll']:<8.3f} | {m['brier']:<8.3f} | {m['ece']:<8.3f} | {m['tail_ece']:<8.3f} | {m.get('mean_logit', 0):<10.2f} | {m.get('sat_10', 0):<6.1f} | {m.get('sat_20', 0):<6.1f}")

    print_row("CE", m_ce)
    print_row("LA", m_la)
    print_row("BS", m_bs)
    print_row("Uniform", m_unif)
    print_row("CRISP", m_crisp)
    print("="*140)

    # --- Print Table 3: Gate Routing ---
    print("\n" + "="*80)
    print("TABLE 3: GATE ROUTING STATISTICS (TEST SET)")
    print("="*80)
    print(f"{'Metric':<25} | {'Value':<20}")
    print("-"*50)
    print(f"{'Avg Weight CE':<25} | {w_test[0]:<20.4f}")
    print(f"{'Avg Weight LA':<25} | {w_test[1]:<20.4f}")
    print(f"{'Avg Weight BS':<25} | {w_test[2]:<20.4f}")
    print("="*80)
    
    print("\n[INFO] Analysis:")
    print("1. If CRISP Bal AURC < Uniform Bal AURC, the Gate is successfully adding value.")
    print("2. If CRISP NLL/ECE < CE NLL/ECE, the posterior is successfully repaired.")
    print("3. If Mean Logit > 15.0 or %>20 > 50%, Stage 1 suffers from logit saturation.")

if __name__ == "__main__":
    main()