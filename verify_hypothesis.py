#!/usr/bin/env python3
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

def get_tail_acc(preds, labels, group_ids):
    tail_mask = group_ids[labels] == 2
    if np.sum(tail_mask) == 0: return 0.0
    return np.mean(preds[tail_mask] == labels[tail_mask]) * 100

def main():
    cfg = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "="*80)
    print("HYPOTHESIS VERIFICATION SCRIPT (Flawed Routing Policy)")
    print("="*80)

    dataset = ImbalancedDataset(cfg, cfg.dataset, augmentation='none')
    train_dataset, val_dataset = dataset.train_val_sets
    
    val_targets = np.array(val_dataset.targets)
    val_indices = np.arange(len(val_targets))
    _, test_idx = train_test_split(val_indices, test_size=0.8, stratify=val_targets, random_state=cfg.seed)
    test_dataset = Subset(val_dataset, test_idx)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    ckpt_paths = {'CE': custom_args.ce_path, 'LA': custom_args.la_path, 'BS': custom_args.bs_path}
    model = ExpertEnsemble(cfg, device, ckpt_paths).to(device)
    
    gate = GateMLP(input_dim=24, hidden1=cfg.gate_hidden_size, hidden2=cfg.gate_hidden_size2).to(device)
    gate_ckpt = torch.load(custom_args.gate_ckpt, map_location='cpu')
    gate.load_state_dict(gate_ckpt['gate_state_dict'])
    gate.eval()
    T = gate_ckpt.get('temperature', 1.0)

    group_ids_3 = define_groups(cfg.cls_num_list)
    group_ids_2 = define_groups_2(cfg.cls_num_list)

    all_logits = [[], [], []]
    all_labels = []
    all_weights = []

    print("[INFO] Extracting logits and gate weights on test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits_list, _ = model(images)
            for i in range(3):
                all_logits[i].append(logits_list[i].cpu())
            all_labels.append(labels)
            
        all_logits = [torch.cat(l, dim=0) for l in all_logits]
        labels = torch.cat(all_labels, dim=0).numpy()

        adj_probs = [
            F.softmax(all_logits[0] / T, dim=1),
            F.softmax(all_logits[1] / T, dim=1),
            F.softmax(all_logits[2] / T, dim=1)
        ]
        
        phi = compute_gate_features(all_logits, adj_probs).to(device)
        gate_logits = gate(phi)
        weights = F.softmax(gate_logits, dim=1)
        all_weights.append(weights.cpu().numpy())
        
        k = getattr(cfg, 'routing_sparsity', 2)
        topk_weights, topk_indices = torch.topk(weights, k, dim=1)
        topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)
        
        stacked_probs = torch.stack(adj_probs, dim=1).to(device)
        p_mix = torch.zeros_like(stacked_probs[:, 0, :])
        N = stacked_probs.size(0)
        for i in range(k):
            idx = topk_indices[:, i]
            w = topk_weights[:, i].unsqueeze(1)
            expert_probs = stacked_probs[torch.arange(N), idx, :]
            p_mix += w * expert_probs
            
        p_uniform = (adj_probs[0] + adj_probs[1] + adj_probs[2]) / 3.0

    weights_np = np.concatenate(all_weights, axis=0)
    p_mix_np = p_mix.cpu().numpy()
    p_unif_np = p_uniform.numpy()

    preds_ce = np.argmax(adj_probs[0].numpy(), axis=1)
    preds_la = np.argmax(adj_probs[1].numpy(), axis=1)
    preds_bs = np.argmax(adj_probs[2].numpy(), axis=1)
    preds_unif = np.argmax(p_unif_np, axis=1)
    preds_crisp = np.argmax(p_mix_np, axis=1)

    # =========================================================================
    # VERIFICATION: Flawed Routing Policy on Tail Samples
    # =========================================================================
    print("\n" + "="*80)
    print("VERIFICATION: Flawed Routing Policy on Tail Samples")
    print("="*80)
    
    # Find tail samples where CE is WRONG but LA is CORRECT
    tail_mask = group_ids_3[labels] == 2
    ce_wrong = preds_ce != labels
    la_right = preds_la == labels
    critical_mask = tail_mask & ce_wrong & la_right
    
    num_critical = np.sum(critical_mask)
    print(f"Found {num_critical} tail samples where CE is WRONG but LA is CORRECT.")
    print("(On these samples, a perfect gate would assign 100% weight to LA and 0% to CE)\n")
    
    if num_critical > 0:
        # 1. Average weights on these critical samples
        avg_w_ce_crit = np.mean(weights_np[critical_mask, 0])
        avg_w_la_crit = np.mean(weights_np[critical_mask, 1])
        avg_w_bs_crit = np.mean(weights_np[critical_mask, 2])
        
        print(f"Average Gate Weights on these critical samples:")
        print(f"  CE (Wrong): {avg_w_ce_crit:.3f}")
        print(f"  LA (Right): {avg_w_la_crit:.3f}")
        print(f"  BS:         {avg_w_bs_crit:.3f}")
        
        # 2. How many did CRISP mixture get wrong?
        crisp_wrong_on_crit = np.sum(preds_crisp[critical_mask] != labels[critical_mask])
        unif_wrong_on_crit = np.sum(preds_unif[critical_mask] != labels[critical_mask])
        
        print(f"\nPrediction Outcome on these critical samples:")
        print(f"  Uniform Baseline got {unif_wrong_on_crit}/{num_critical} wrong.")
        print(f"  CRISP Gate got {crisp_wrong_on_crit}/{num_critical} wrong.")
        
        if avg_w_ce_crit > 0.2 and crisp_wrong_on_crit > unif_wrong_on_crit:
            print("\n  -> VERIFIED: The gate assigns a high baseline weight to CE even when it is wrong")
            print("               on tail samples. This dilutes the correct LA signal, causing the")
            print("               CRISP mixture to make more mistakes than the Uniform baseline.")
            
        # 3. Print 3 specific examples
        print("\n  Examples of Flawed Routing:")
        crit_indices = np.where(critical_mask)[0][:3]
        for idx in crit_indices:
            print(f"\n  Sample {idx} (True Label: {labels[idx]}, Tail Class)")
            print(f"    CE pred: {preds_ce[idx]} (Conf: {np.max(adj_probs[0][idx].numpy()):.4f}) | Weight: {weights_np[idx, 0]:.4f}")
            print(f"    LA pred: {preds_la[idx]} (Conf: {np.max(adj_probs[1][idx].numpy()):.4f}) | Weight: {weights_np[idx, 1]:.4f}")
            print(f"    BS pred: {preds_bs[idx]} (Conf: {np.max(adj_probs[2][idx].numpy()):.4f}) | Weight: {weights_np[idx, 2]:.4f}")
            print(f"    -> CRISP Mixture Pred: {preds_crisp[idx]} ({'CORRECT' if preds_crisp[idx] == labels[idx] else 'WRONG'})")
            print(f"    -> Uniform Baseline Pred: {preds_unif[idx]} ({'CORRECT' if preds_unif[idx] == labels[idx] else 'WRONG'})")

    # =========================================================================
    # OVERALL IMPACT
    # =========================================================================
    print("\n" + "="*80)
    print("OVERALL IMPACT ON TAIL ACCURACY AND AURC")
    print("="*80)
    
    acc_la = get_tail_acc(preds_la, labels, group_ids_3)
    acc_unif = get_tail_acc(preds_unif, labels, group_ids_3)
    acc_crisp = get_tail_acc(preds_crisp, labels, group_ids_3)
    
    print(f"Tail Accuracy:")
    print(f"  LA Expert alone:      {acc_la:.2f}%")
    print(f"  Uniform Baseline:     {acc_unif:.2f}%")
    print(f"  CRISP Gate:           {acc_crisp:.2f}%")
    
    unif_aurc = compute_aurc_metrics(p_unif_np, labels, p_unif_np, labels, group_ids_2, cls_num_list=cfg.cls_num_list, mode='bal')
    crisp_aurc = compute_aurc_metrics(p_mix_np, labels, p_mix_np, labels, group_ids_2, cls_num_list=cfg.cls_num_list, mode='bal')
    
    print(f"\nBal AURC (Selective Risk):")
    print(f"  Uniform Baseline:     {unif_aurc['AURC']:.4f}")
    print(f"  CRISP Gate:           {crisp_aurc['AURC']:.4f}")
    
    if acc_crisp < acc_unif and crisp_aurc['AURC'] > unif_aurc['AURC']:
        print("\n  -> VERIFIED: The flawed routing policy directly caused CRISP to underperform")
        print("               the Uniform baseline in both Tail Accuracy and Selective Risk.")

if __name__ == "__main__":
    main()