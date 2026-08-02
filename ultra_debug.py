#!/usr/bin/env python3
# ultra_debug.py
# Pipeline Verification & Paper Comparison (Tables 2, 3 & Diagnostics)

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import re
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
from imbalanceddl.dataset.imbalance_dataset import ImbalancedDataset
from imbalanceddl.utils.gate_features import compute_gate_features
from imbalanceddl.utils.plugin_rule import define_groups, define_groups_2, compute_aurc_metrics
from imbalanceddl.utils.debug import (
    ExpertEnsemble, GateMLP, compute_chow_aurc, compute_all_metrics, 
    print_uniform_comparison, print_method_vs_uniform_comparison, print_ce_comparison, print_final_method_comparison,
    print_gate_feature_importance, print_expert_agreement, print_stage3_plugin_params, print_per_class_extreme_routing
)
from torch.utils.data import DataLoader, Subset

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
    
    gate = GateMLP(input_dim=24, hidden1=cfg.gate_hidden_size, hidden2=cfg.gate_hidden_size2).to(device)
    print(f"[INFO] Loading Gate from {custom_args.gate_ckpt}")
    gate_ckpt = torch.load(custom_args.gate_ckpt, map_location='cpu')
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
            
            p_ce = F.softmax(all_logits[0] / T, dim=1)
            p_la = F.softmax((all_logits[1] + la_tau * log_prior) / T, dim=1)
            p_bs = F.softmax((all_logits[2] + log_spc) / T, dim=1)
            adj_probs = [p_ce, p_la, p_bs]
            
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
            
        return (p_mix.cpu().numpy(), p_uniform.cpu().numpy(), 
                adj_probs[0].cpu().numpy(), adj_probs[1].cpu().numpy(), adj_probs[2].cpu().numpy(), 
                all_logits[0].cpu(), all_logits[1].cpu(), all_logits[2].cpu(), 
                np.concatenate(all_weights, axis=0), labels.cpu().numpy())

    print("\n[INFO] Extracting posteriors...")
    (p_mix_tune, p_unif_tune, p_ce_tune, p_la_tune, p_bs_tune, 
     l_ce_tune, l_la_tune, l_bs_tune, w_tune, labels_tune) = extract_data(tune_loader)
     
    (p_mix_test, p_unif_test, p_ce_test, p_la_test, p_bs_test, 
     l_ce_test, l_la_test, l_bs_test, w_test, labels_test) = extract_data(test_loader)

    group_ids_2 = define_groups_2(cfg.cls_num_list)
    
    print("\n[INFO] Computing AURC & Calibration metrics...")
    chow_bal = compute_chow_aurc(p_mix_tune, labels_tune, p_mix_test, labels_test, group_ids_2, mode='bal')
    chow_wst = compute_chow_aurc(p_mix_tune, labels_tune, p_mix_test, labels_test, group_ids_2, mode='worst')
    
    la_bal = compute_aurc_metrics(p_la_tune, labels_tune, p_la_test, labels_test, group_ids_2, cls_num_list=cfg.cls_num_list, mode='bal')
    la_wst = compute_aurc_metrics(p_la_tune, labels_tune, p_la_test, labels_test, group_ids_2, cls_num_list=cfg.cls_num_list, mode='worst')

    ce_bal = compute_aurc_metrics(p_ce_tune, labels_tune, p_ce_test, labels_test, group_ids_2, cls_num_list=cfg.cls_num_list, mode='bal')
    ce_wst = compute_aurc_metrics(p_ce_tune, labels_tune, p_ce_test, labels_test, group_ids_2, cls_num_list=cfg.cls_num_list, mode='worst')
    
    unif_bal = compute_aurc_metrics(p_unif_tune, labels_tune, p_unif_test, labels_test, group_ids_2, cls_num_list=cfg.cls_num_list, mode='bal')
    unif_wst = compute_aurc_metrics(p_unif_tune, labels_tune, p_unif_test, labels_test, group_ids_2, cls_num_list=cfg.cls_num_list, mode='worst')
    
    method_bal = compute_aurc_metrics(p_mix_tune, labels_tune, p_mix_test, labels_test, group_ids_2, cls_num_list=cfg.cls_num_list, mode='bal')
    method_wst = compute_aurc_metrics(p_mix_tune, labels_tune, p_mix_test, labels_test, group_ids_2, cls_num_list=cfg.cls_num_list, mode='worst')

    m_ce = compute_all_metrics(p_ce_test, labels_test, l_ce_test, cfg, train_dataset)
    m_la = compute_all_metrics(p_la_test, labels_test, l_la_test, cfg, train_dataset)
    m_bs = compute_all_metrics(p_bs_test, labels_test, l_bs_test, cfg, train_dataset)
    m_unif = compute_all_metrics(p_unif_test, labels_test, None, cfg, train_dataset)
    m_method = compute_all_metrics(p_mix_test, labels_test, None, cfg, train_dataset)

    # --- RAW T=1.0 vs GATE T={} COMPARISON ---
    print("\n" + "="*110)
    print("RAW T=1.0 vs GATE T={} COMPARISON".format(T))
    print("="*110)
    
    p_ce_T1 = F.softmax(l_ce_test, dim=1)
    p_la_T1 = F.softmax(l_la_test + la_tau * log_prior.cpu(), dim=1)
    p_bs_T1 = F.softmax(l_bs_test + log_spc.cpu(), dim=1)
    
    p_unif_T1 = (p_ce_T1 + p_la_T1 + p_bs_T1) / 3.0
    
    k = getattr(cfg, 'routing_sparsity', 2)
    w_test_tensor = torch.tensor(w_test)
    topk_weights_T1, topk_indices_T1 = torch.topk(w_test_tensor, k, dim=1)
    topk_weights_T1 = topk_weights_T1 / topk_weights_T1.sum(dim=1, keepdim=True)
    
    stacked_probs_T1 = torch.stack([p_ce_T1, p_la_T1, p_bs_T1], dim=1)
    p_mix_T1 = torch.zeros_like(stacked_probs_T1[:, 0, :])
    N = stacked_probs_T1.size(0)
    for i in range(k):
        idx = topk_indices_T1[:, i]
        w = topk_weights_T1[:, i].unsqueeze(1)
        expert_probs = stacked_probs_T1[torch.arange(N), idx, :]
        p_mix_T1 += w * expert_probs

    m_unif_T1 = compute_all_metrics(p_unif_T1.numpy(), labels_test, None, cfg, train_dataset)
    m_method_T1 = compute_all_metrics(p_mix_T1.numpy(), labels_test, None, cfg, train_dataset)
    
    print(f"{'Metric':<35} | {'Unif @ T=1.0':<15} | {'Unif @ T={}'.format(T):<15} | {'Method @ T=1.0':<15} | {'Method @ T={}'.format(T):<15}")
    print("-"*110)
    
    def print_T_row(name, val_u1, val_uT, val_m1, val_mT):
        print(f"{name:<35} | {val_u1:<15.4f} | {val_uT:<15.4f} | {val_m1:<15.4f} | {val_mT:<15.4f}")
        
    print_T_row("NLL (lower is better)", m_unif_T1['nll'], m_unif['nll'], m_method_T1['nll'], m_method['nll'])
    print_T_row("Brier (lower is better)", m_unif_T1['brier'], m_unif['brier'], m_method_T1['brier'], m_method['brier'])
    print_T_row("ECE All (lower is better)", m_unif_T1['ece'], m_unif['ece'], m_method_T1['ece'], m_method['ece'])
    print_T_row("tail-ECE (lower is better)", m_unif_T1['tail_ece'], m_unif['tail_ece'], m_method_T1['tail_ece'], m_method['tail_ece'])
    print_T_row("Bal Acc (higher is better)", m_unif_T1['bal_acc'], m_unif['bal_acc'], m_method_T1['bal_acc'], m_method['bal_acc'])
    print("="*110)
    print("[INFO] If Method @ T=1.0 is drastically worse than Method @ T={}, it proves temperature".format(T))
    print("       scaling is doing the calibration work, not the dynamic routing.")

    # --- COMPARISON TABLES ---
    print_uniform_comparison(m_unif, m_unif['nll'], m_unif['tail_ece'], m_unif['brier'], unif_bal['AURC'], unif_wst['AURC'])
    print_method_vs_uniform_comparison(m_method, m_unif, method_bal['AURC'], unif_bal['AURC'], method_wst['AURC'], unif_wst['AURC'])
    print_ce_comparison(m_ce, ce_bal['AURC'], ce_wst['AURC'])

    print_final_method_comparison(m_method, method_bal['AURC'], method_wst['AURC'])

    print("\n" + "="*140)
    print("TABLE 2: FULL DIAGNOSTIC BREAKDOWN (TEST SET) - PAPER BASELINES INCLUDED")
    print("="*140)
    print(f"{'Method':<15} | {'Bal Acc':<7} | {'Many':<6} | {'Med':<6} | {'Low':<6} | {'NLL':<8} | {'Brier':<8} | {'ECE':<8} | {'Tail ECE':<8} | {'Mean Logit':<10} | {'%>10':<6} | {'%>20':<6}")
    print("-"*140)
    
    def print_row(name, m):
        print(f"{name:<15} | {m['bal_acc']:<7.2f} | {m['many']:<6.2f} | {m['med']:<6.2f} | {m['low']:<6.2f} | {m['nll']:<8.3f} | {m['brier']:<8.3f} | {m['ece']:<8.3f} | {m['tail_ece']:<8.3f} | {m.get('mean_logit', 0):<10.2f} | {m.get('sat_10', 0):<6.1f} | {m.get('sat_20', 0):<6.1f}")

    def print_paper_row(name, bal, many, med, low, nll, brier, ece, tail_ece):
        print(f"{name:<15} | {bal:<7} | {many:<6} | {med:<6} | {low:<6} | {nll:<8} | {brier:<8} | {ece:<8} | {tail_ece:<8} | {'N/A':<10} | {'N/A':<6} | {'N/A':<6}")

    print_paper_row("Paper's Method", "N/A", "N/A", "N/A", "N/A", "1.18", "0.403", "N/A", "0.088")
    print_paper_row("Paper Unif", "N/A", "N/A", "N/A", "N/A", "1.30", "0.442", "N/A", "0.171")
    print("-"*140)
    print_row("YOUR CE", m_ce)
    print_row("YOUR LA", m_la)
    print_row("YOUR BS", m_bs)
    print_row("YOUR Uniform", m_unif)
    print_row("My Method", m_method)
    print("="*140)

    # --- PROBABILITY SATURATION VERIFICATION ---
    print("\n" + "="*80)
    print(f"PROBABILITY SATURATION VERIFICATION (AT GATE TEMPERATURE T={T})")
    print("="*80)
    def check_saturation(probs, name):
        max_probs = np.max(probs, axis=1)
        print(f"{name} Avg Max Prob: {np.mean(max_probs):.4f}")
        print(f"{name} % Max Prob > 0.90: {np.mean(max_probs > 0.90) * 100:.2f}%")
        print(f"{name} % Max Prob > 0.99: {np.mean(max_probs > 0.99) * 100:.2f}%")
        
    check_saturation(p_ce_test, "CE")
    print("-"*40)
    check_saturation(p_la_test, "LA")
    print("-"*40)
    check_saturation(p_bs_test, "BS")
    print("="*80)

    print("\n" + "="*80)
    print("TABLE 3: GATE ROUTING STATISTICS (TEST SET)")
    print("="*80)
    
    label_groups_test = group_ids_2[labels_test]
    head_mask = (label_groups_test == 0)
    tail_mask = (label_groups_test == 1)
    
    print(f"{'Metric':<25} | {'Value':<20}")
    print("-"*50)
    print(f"{'Avg Weight CE (All)':<25} | {np.mean(w_test[:, 0]):<20.4f}")
    print(f"{'Avg Weight LA (All)':<25} | {np.mean(w_test[:, 1]):<20.4f}")
    print(f"{'Avg Weight BS (All)':<25} | {np.mean(w_test[:, 2]):<20.4f}")
    print("-"*50)
    print(f"{'Avg Weight CE (Head)':<25} | {np.mean(w_test[head_mask, 0]):<20.4f}")
    print(f"{'Avg Weight LA (Head)':<25} | {np.mean(w_test[head_mask, 1]):<20.4f}")
    print(f"{'Avg Weight BS (Head)':<25} | {np.mean(w_test[head_mask, 2]):<20.4f}")
    print("-"*50)
    print(f"{'Avg Weight CE (Tail)':<25} | {np.mean(w_test[tail_mask, 0]):<20.4f}")
    print(f"{'Avg Weight LA (Tail)':<25} | {np.mean(w_test[tail_mask, 1]):<20.4f}")
    print(f"{'Avg Weight BS (Tail)':<25} | {np.mean(w_test[tail_mask, 2]):<20.4f}")
    print("="*80)
    
    # --- EXPERT CHOICE COUNTS (TOP-K ROUTING) ---
    print("\n" + "="*80)
    print("EXPERT CHOICE COUNTS (TOP-K ROUTING)")
    print("="*80)
    
    topk_indices_all = np.argsort(w_test, axis=1)[:, ::-1][:, :k]
    
    head_choices = topk_indices_all[head_mask]
    tail_choices = topk_indices_all[tail_mask]
    
    head_total = len(head_choices)
    tail_total = len(tail_choices)
    
    head_ce_count = np.sum(head_choices == 0)
    head_la_count = np.sum(head_choices == 1)
    head_bs_count = np.sum(head_choices == 2)
    
    tail_ce_count = np.sum(tail_choices == 0)
    tail_la_count = np.sum(tail_choices == 1)
    tail_bs_count = np.sum(tail_choices == 2)
    
    print(f"Head Classes ({head_total} samples, choosing {k} experts per sample):")
    print(f"  CE chosen in {head_ce_count}/{head_total} samples ({head_ce_count/head_total*100:.1f}%)")
    print(f"  LA chosen in {head_la_count}/{head_total} samples ({head_la_count/head_total*100:.1f}%)")
    print(f"  BS chosen in {head_bs_count}/{head_total} samples ({head_bs_count/head_total*100:.1f}%)")
    
    print(f"\nTail Classes ({tail_total} samples, choosing {k} experts per sample):")
    print(f"  CE chosen in {tail_ce_count}/{tail_total} samples ({tail_ce_count/tail_total*100:.1f}%)")
    print(f"  LA chosen in {tail_la_count}/{tail_total} samples ({tail_la_count/tail_total*100:.1f}%)")
    print(f"  BS chosen in {tail_bs_count}/{tail_total} samples ({tail_bs_count/tail_total*100:.1f}%)")
    print("="*80)

    # --- 4. PER-CLASS EXTREME ROUTING ---
    print_per_class_extreme_routing(w_test, labels_test, cfg)

    # --- GATE INPUT FEATURE TABLE (10 SAMPLES) ---
    print("\n" + "="*100)
    print("GATE INPUT FEATURE TABLE (10 SAMPLES: 5 Head, 5 Tail)")
    print("="*100)
    
    feat_phi = compute_gate_features(
        [l_ce_test, l_la_test, l_bs_test], 
        [torch.tensor(p_ce_test), torch.tensor(p_la_test), torch.tensor(p_bs_test)]
    )
    
    feat_names = [
        "CE_Ent", "CE_Max", "CE_Marg", "CE_Top5", "CE_Tail", "CE_Cos", "CE_KL",
        "LA_Ent", "LA_Max", "LA_Marg", "LA_Top5", "LA_Tail", "LA_Cos", "LA_KL",
        "BS_Ent", "BS_Max", "BS_Marg", "BS_Top5", "BS_Tail", "BS_Cos", "BS_KL",
        "Glb_MeanEnt", "Glb_ClassVar", "Glb_ConfDisp"
    ]
    
    head_idxs_feat = np.where(head_mask)[0][:5]
    tail_idxs_feat = np.where(tail_mask)[0][:5]
    
    for i in np.concatenate([head_idxs_feat, tail_idxs_feat]):
        group_name = "Head" if label_groups_test[i] == 0 else "Tail"
        print(f"\n--- Sample {i} ({group_name}) | True Label: {labels_test[i]} | Prediction: {np.argmax(p_mix_test[i])} ---")
        for name, val in zip(feat_names, feat_phi[i]):
            print(f"  {name:<15} | {val.item():.6f}")

    # --- 1. GATE MLP FEATURE IMPORTANCE ---
    print_gate_feature_importance(gate)

    # --- SAMPLE-LEVEL GATE OUTPUTS ---
    print("\n" + "="*100)
    print("SAMPLE-BY-SAMPLE GATE OUTPUTS (10 Head, 10 Tail)")
    print("="*100)
    
    head_idxs = np.where(head_mask)[0][:10]
    tail_idxs = np.where(tail_mask)[0][:10]
    
    print(f"{'Idx':<6} | {'Group':<5} | {'True Label':<4} | {'Prediction':<8} | {'CE_Pred':<8} | {'LA_Pred':<8} | {'BS_Pred':<8} | {'w_CE':<6} | {'w_LA':<6} | {'w_BS':<6} | Top-k Chosen")
    print("-"*100)
    
    for i in np.concatenate([head_idxs, tail_idxs]):
        w = w_test[i]
        topk_idx = np.argsort(w)[::-1][:k]
        
        mix_pred = np.argmax(p_mix_test[i])
        ce_pred = np.argmax(p_ce_test[i])
        la_pred = np.argmax(p_la_test[i])
        bs_pred = np.argmax(p_bs_test[i])
        
        true_label = labels_test[i]
        group_name = "Head" if label_groups_test[i] == 0 else "Tail"
        
        experts_chosen = []
        if 0 in topk_idx: experts_chosen.append("CE")
        if 1 in topk_idx: experts_chosen.append("LA")
        if 2 in topk_idx: experts_chosen.append("BS")
        experts_chosen_str = ",".join(experts_chosen)
        
        print(f"{i:<6} | {group_name:<5} | {true_label:<4} | {mix_pred:<8} | {ce_pred:<8} | {la_pred:<8} | {bs_pred:<8} | {w[0]:<6.3f} | {w[1]:<6.3f} | {w[2]:<6.3f} | {experts_chosen_str}")
    print("="*100)

    # --- LA 'SAVES THE DAY' ROUTING CHECK ---
    print("\n" + "="*100)
    print("LA 'SAVES THE DAY' ROUTING CHECK (Tail Samples where LA is right, CE & BS are wrong)")
    print("="*100)
    
    # Recompute preds just to be safe
    ce_preds_test = np.argmax(p_ce_test, axis=1)
    la_preds_test = np.argmax(p_la_test, axis=1)
    bs_preds_test = np.argmax(p_bs_test, axis=1)
    
    # Mask: Tail samples
    tail_mask_check = (label_groups_test == 1)
    
    # Mask: LA is correct
    la_correct_mask = (la_preds_test == labels_test)
    
    # Mask: CE and BS are incorrect
    ce_bs_wrong_mask = (ce_preds_test != labels_test) & (bs_preds_test != labels_test)
    
    # Combined mask
    la_saves_day_mask = tail_mask_check & la_correct_mask & ce_bs_wrong_mask
    
    # Get the indices
    la_saves_day_indices = np.where(la_saves_day_mask)[0]
    
    total_la_saves_day = len(la_saves_day_indices)
    
    if total_la_saves_day == 0:
        print("[INFO] No samples found where LA was the sole correct expert on Tail classes.")
    else:
        print(f"[INFO] Found {total_la_saves_day} samples where LA was the sole correct expert on Tail classes.")
        
        # Average routing weights for these specific samples
        avg_w_la_saves = np.mean(w_test[la_saves_day_indices], axis=0)
        print(f"Average Routing Weights for these samples: CE={avg_w_la_saves[0]:.4f} | LA={avg_w_la_saves[1]:.4f} | BS={avg_w_la_saves[2]:.4f}")
        
        # How many times was LA actually chosen in top-k?
        topk_indices_la_saves = np.argsort(w_test[la_saves_day_indices], axis=1)[:, ::-1][:, :k]
        la_chosen_count = np.sum(topk_indices_la_saves == 1)
        print(f"LA was chosen in Top-{k} routing for {la_chosen_count}/{total_la_saves_day} of these samples ({la_chosen_count/total_la_saves_day*100:.1f}%)")
        
        # Print a few examples
        print(f"\n{'Idx':<6} | {'True':<5} | {'CE_Pred':<8} | {'LA_Pred':<8} | {'BS_Pred':<8} | {'w_CE':<6} | {'w_LA':<6} | {'w_BS':<6} | Top-k Chosen")
        print("-"*100)
        
        # Print up to 15 examples
        for i in la_saves_day_indices[:15]:
            w = w_test[i]
            topk_idx = np.argsort(w)[::-1][:k]
            
            mix_pred = np.argmax(p_mix_test[i])
            ce_pred = ce_preds_test[i]
            la_pred = la_preds_test[i]
            bs_pred = bs_preds_test[i]
            
            true_label = labels_test[i]
            
            experts_chosen = []
            if 0 in topk_idx: experts_chosen.append("CE")
            if 1 in topk_idx: experts_chosen.append("LA")
            if 2 in topk_idx: experts_chosen.append("BS")
            experts_chosen_str = ",".join(experts_chosen)
            
            print(f"{i:<6} | {true_label:<5} | {ce_pred:<8} | {la_pred:<8} | {bs_pred:<8} | {w[0]:<6.3f} | {w[1]:<6.3f} | {w[2]:<6.3f} | {experts_chosen_str}")
    print("="*100)

    # --- 3. STAGE 3 PLUG-IN PARAMETERS ---
    print_stage3_plugin_params(p_mix_tune, labels_tune, group_ids_2, cfg)

    print("\n[INFO] Analysis:")
    print("1. If Method Bal AURC < Uniform Bal AURC, the Gate is successfully adding value.")
    print("2. If Method NLL/ECE < CE NLL/ECE, the posterior is successfully repaired.")
    print("3. If Mean Logit > 15.0 or %>20 > 50%, Stage 1 suffers from logit saturation.")
    print("4. If CE(Head) > CE(Tail) and LA/BS(Tail) > LA/BS(Head), the gate is routing correctly.")

    print("\n" + "="*80)
    print("EXPERT CORRELATION & SHARPENING CHECK")
    print("="*80)
    
    ce_preds = np.argmax(p_ce_test, axis=1)
    la_preds = np.argmax(p_la_test, axis=1)
    bs_preds = np.argmax(p_bs_test, axis=1)
    
    # --- 2. EXPERT AGREEMENT ON CORRECT VS INCORRECT ---
    print_expert_agreement(p_mix_test, ce_preds, la_preds, bs_preds, labels_test)
    
    agreement = np.mean((ce_preds == la_preds) & (la_preds == bs_preds))
    print(f"Expert Prediction Agreement: {agreement*100:.2f}%  (If >90%, experts are too similar)")
    
    unif_max_conf = np.max(p_unif_test, axis=1)
    method_max_conf = np.max(p_mix_test, axis=1)
    print(f"Uniform Avg Max Confidence:  {np.mean(unif_max_conf):.4f}")
    print(f"My Method Avg Max Confidence: {np.mean(method_max_conf):.4f}  (If > Uniform, gate is sharpening)")
    print("="*80)

if __name__ == "__main__":
    main()