#!/usr/bin/env python3
# verify_spillover.py
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
from imbalanceddl.net.network import build_model
from imbalanceddl.dataset.imbalance_dataset import ImbalancedDataset
from imbalanceddl.utils.gate_features import compute_gate_features
from imbalanceddl.utils.plugin_rule import define_groups_2
from torch.utils.data import DataLoader, Subset

class ExpertEnsemble(nn.Module):
    def __init__(self, cfg, device, ckpt_paths):
        super().__init__()
        self.experts = nn.ModuleList()
        for name, path in ckpt_paths.items():
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            has_bias = ckpt.get('bias', False)
            model = build_model(cfg)
            actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            actual_model.classifier = nn.Linear(actual_model.feature_len, actual_model.num_classes, bias=has_bias).to(device)
            state_dict = ckpt['state_dict']
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            actual_model.load_state_dict(new_state_dict)
            for param in actual_model.parameters():
                param.requires_grad = False
            actual_model.eval()
            self.experts.append(actual_model.to(device))

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
    la_tau = 1.5
    match_tau = re.search(r't([\d\.]+)', os.path.basename(custom_args.la_path))
    if match_tau:
        la_tau = float(match_tau.group(1))
    
    cls_num_list = torch.tensor(cfg.cls_num_list, device=device, dtype=torch.float32)
    log_prior = torch.log(cls_num_list / cls_num_list.sum() + 1e-12)
    log_spc = torch.log(cls_num_list + 1e-12)

    all_logits = [[], [], []]
    all_labels = []
    
    print("\n[INFO] Extracting posteriors...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits_list, _ = model(images)
            for i in range(3):
                all_logits[i].append(logits_list[i])
            all_labels.append(labels)
            
        all_logits = [torch.cat(l, dim=0) for l in all_logits]
        labels = torch.cat(all_labels, dim=0).cpu().numpy()
        
        p_ce = F.softmax(all_logits[0] / T, dim=1)
        p_la = F.softmax((all_logits[1] + la_tau * log_prior) / T, dim=1)
        p_bs = F.softmax((all_logits[2] + log_spc) / T, dim=1)
        adj_probs = [p_ce, p_la, p_bs]
        
        phi = compute_gate_features(all_logits, adj_probs)
        gate_logits = gate(phi)
        weights = F.softmax(gate_logits, dim=1)
        
        # Use dense routing (k=3) to match training
        k = 3 
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
            
    p_mix_np = p_mix.cpu().numpy()
    p_unif_np = p_uniform.cpu().numpy()
    p_ce_np = p_ce.cpu().numpy()
    p_la_np = p_la.cpu().numpy()
    p_bs_np = p_bs.cpu().numpy()

    group_ids_2 = define_groups_2(cfg.cls_num_list)
    tail_mask = (group_ids_2[labels] == 1)
    head_mask = ~tail_mask
    
    # 1. Expert Similarity (L2 Distance between probability vectors)
    def avg_l2(p1, p2, mask):
        return np.mean(np.linalg.norm(p1[mask] - p2[mask], axis=1))
        
    dist_ce_la_head = avg_l2(p_ce_np, p_la_np, head_mask)
    dist_ce_la_tail = avg_l2(p_ce_np, p_la_np, tail_mask)
    dist_ce_bs_head = avg_l2(p_ce_np, p_bs_np, head_mask)
    dist_ce_bs_tail = avg_l2(p_ce_np, p_bs_np, tail_mask)

    # 2. Confidence Spillover (Ratio of CRISP conf to Uniform conf)
    def avg_conf(probs, mask):
        return np.mean(np.max(probs[mask], axis=1))
        
    unif_conf_head = avg_conf(p_unif_np, head_mask)
    crisp_conf_head = avg_conf(p_mix_np, head_mask)
    ratio_head = crisp_conf_head / unif_conf_head
    
    unif_conf_tail = avg_conf(p_unif_np, tail_mask)
    crisp_conf_tail = avg_conf(p_mix_np, tail_mask)
    ratio_tail = crisp_conf_tail / unif_conf_tail

    print("\n" + "="*90)
    print("VERIFICATION: EXPERT SIMILARITY & CONFIDENCE SPILLOVER")
    print("="*90)
    print(f"{'Metric':<45} | {'Head':<12} | {'Tail':<12}")
    print("-"*75)
    print(f"{'L2 Dist (CE vs LA)':<45} | {dist_ce_la_head:<12.4f} | {dist_ce_la_tail:<12.4f}")
    print(f"{'L2 Dist (CE vs BS)':<45} | {dist_ce_bs_head:<12.4f} | {dist_ce_bs_tail:<12.4f}")
    print("-"*75)
    print(f"{'Uniform Avg Max Confidence':<45} | {unif_conf_head:<12.4f} | {unif_conf_tail:<12.4f}")
    print(f"{'CRISP Avg Max Confidence':<45} | {crisp_conf_head:<12.4f} | {crisp_conf_tail:<12.4f}")
    print("-"*75)
    print(f"{'Sharpening Ratio (CRISP / Uniform)':<45} | {ratio_head:<12.4f} | {ratio_tail:<12.4f}")
    print("="*90)

    print("\n[INFO] Analysis:")
    print("1. If L2 Distances are small (< 0.5), experts are highly correlated.")
    print("2. If Sharpening Ratio for Head and Tail are nearly identical, the gate is applying a global sharpening filter (spillover).")

if __name__ == "__main__":
    main()