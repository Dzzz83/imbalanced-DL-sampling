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
from imbalanceddl.utils.debug.evaluation import (
    extract_data, run_metric_comparisons, run_temperature_comparison,
    run_sample_by_sample_output, run_saves_the_day_checks, 
    run_raw_prob_inspection, run_oracle_diagnostic
)
from imbalanceddl.utils.debug.diagnostics import print_stage3_plugin_params, print_expert_agreement, print_per_class_extreme_routing

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
    model = ExpertEnsemble(cfg, device, ckpt_paths).to(device)
    
    gate = GateMLP(input_dim=192, hidden1=cfg.gate_hidden_size, hidden2=cfg.gate_hidden_size2).to(device)
    print(f"[INFO] Loading Gate from {custom_args.gate_ckpt}")
    
    # FIX: Added weights_only=False
    gate_ckpt = torch.load(custom_args.gate_ckpt, map_location='cpu', weights_only=False)
    gate.load_state_dict(gate_ckpt['gate_state_dict'])
    gate.eval()

    T = gate_ckpt.get('temperature', 1.0)
    print(f"[INFO] Using Temperature T={T} extracted from gate checkpoint")

    la_tau = 1.5
    match_tau = re.search(r't([\d\.]+)', os.path.basename(custom_args.la_path))
    if match_tau:
        la_tau = float(match_tau.group(1))
    print(f"[INFO] Using LA Tau = {la_tau} parsed from filename")
    
    cls_num_list = torch.tensor(cfg.cls_num_list, device=device, dtype=torch.float32)
    log_prior = torch.log(cls_num_list / cls_num_list.sum() + 1e-12)
    log_spc = torch.log(cls_num_list + 1e-12)
    k = getattr(cfg, 'routing_sparsity', 2)

    print("\n[INFO] Extracting posteriors...")
    (p_mix_tune, p_unif_tune, p_ce_tune, p_la_tune, p_bs_tune, 
     l_ce_tune, l_la_tune, l_bs_tune, w_tune, labels_tune) = extract_data(model, gate, tune_loader, T, la_tau, log_prior, log_spc, k, device)
     
    (p_mix_test, p_unif_test, p_ce_test, p_la_test, p_bs_test, 
     l_ce_test, l_la_test, l_bs_test, w_test, labels_test) = extract_data(model, gate, test_loader, T, la_tau, log_prior, log_spc, k, device)

    group_ids_2 = define_groups_2(cfg.cls_num_list)
    
    # 1. Metrics & Comparisons
    run_metric_comparisons(p_mix_tune, p_unif_tune, p_ce_tune, p_la_tune, p_mix_test, p_unif_test, p_ce_test, p_la_test, p_bs_test, l_ce_test, l_la_test, l_bs_test, labels_tune, labels_test, group_ids_2, cfg, train_dataset)
    
    # 2. Temperature Comparison
    run_temperature_comparison(T, l_ce_test, l_la_test, l_bs_test, w_test, k, log_prior, log_spc, labels_test, cfg, train_dataset)
    
    # 3. Routing Statistics
    print_per_class_extreme_routing(w_test, labels_test, cfg)
    
    # 4. Sample-by-Sample Output
    label_groups_test = group_ids_2[labels_test]
    head_mask = (label_groups_test == 0)
    tail_mask = (label_groups_test == 1)
    run_sample_by_sample_output(head_mask, tail_mask, p_mix_test, p_ce_test, p_la_test, p_bs_test, w_test, labels_test, label_groups_test, k)
    
    # 5. LA Saves the Day & Raw Prob Inspection
    la_saves_day_indices = run_saves_the_day_checks(p_ce_test, p_la_test, p_bs_test, w_test, labels_test, label_groups_test, k)
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