#!/usr/bin/env python3
# verify_feature_compression.py
# Verifies if logit saturation compresses gate features, blinding the gate to Head vs Tail differences.

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
            print(f"[INFO] Loading expert {name} from {path}")
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
    print("\n" + "="*80)
    print("FEATURE COMPRESSION VERIFICATION (Head vs Tail)")
    print("="*80)

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
    
    cls_num_list = torch.tensor(cfg.cls_num_list, device=device, dtype=torch.float32)
    log_prior = torch.log(cls_num_list / cls_num_list.sum() + 1e-12)
    log_spc = torch.log(cls_num_list + 1e-12)

    all_logits = [[], [], []]
    all_labels = []
    all_phi = []
    all_weights = []
    
    print("\n[INFO] Extracting features on test set...")
    with torch.no_grad():
        for images, labels in test_loader:
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
        
        all_phi.append(phi.cpu().numpy())
        all_weights.append(weights.cpu().numpy())
            
    all_phi_np = np.concatenate(all_phi, axis=0)
    all_weights_np = np.concatenate(all_weights, axis=0)
    labels_np = labels.cpu().numpy()
    all_logits_np = [l.cpu().numpy() for l in all_logits]
    all_probs_np = [p.cpu().numpy() for p in adj_probs]

    group_ids_2 = define_groups_2(cfg.cls_num_list)
    tail_mask = (group_ids_2[labels_np] == 1)
    head_mask = ~tail_mask
    
    # 1. Logit and Prob Stats
    ce_logit_head = np.mean(np.max(all_logits_np[0][head_mask], axis=1))
    ce_logit_tail = np.mean(np.max(all_logits_np[0][tail_mask], axis=1))
    ce_prob_head = np.mean(np.max(all_probs_np[0][head_mask], axis=1))
    ce_prob_tail = np.mean(np.max(all_probs_np[0][tail_mask], axis=1))
    
    # 2. Feature Stats
    phi_head = np.mean(all_phi_np[head_mask], axis=0)
    phi_tail = np.mean(all_phi_np[tail_mask], axis=0)
    
    # Cosine Similarity (1.0 = identical, 0.0 = orthogonal)
    cos_sim = np.dot(phi_head, phi_tail) / (np.linalg.norm(phi_head) * np.linalg.norm(phi_tail) + 1e-8)
    # L2 Distance (0.0 = identical)
    l2_dist = np.linalg.norm(phi_head - phi_tail)
    
    # 3. Gate Routing Stats
    w_head = np.mean(all_weights_np[head_mask], axis=0)
    w_tail = np.mean(all_weights_np[tail_mask], axis=0)
    w_l2_dist = np.linalg.norm(w_head - w_tail)
    
    print("\n" + "="*90)
    print("VERIFICATION RESULTS: FEATURE COMPRESSION & GATE BLINDNESS")
    print("="*90)
    print(f"{'Metric':<45} | {'Head':<12} | {'Tail':<12}")
    print("-"*75)
    print(f"{'CE Expert Mean Max Logit':<45} | {ce_logit_head:<12.2f} | {ce_logit_tail:<12.2f}")
    print(f"{'CE Expert Mean Max Prob (Softmax)':<45} | {ce_prob_head:<12.4f} | {ce_prob_tail:<12.4f}")
    print("-"*75)
    print(f"{'Gate Routing Weights (CE/LA/BS)':<45} | {f'{w_head[0]:.2f}/{w_head[1]:.2f}/{w_head[2]:.2f}':<12} | {f'{w_tail[0]:.2f}/{w_tail[1]:.2f}/{w_tail[2]:.2f}':<12}")
    print("="*90)
    
    print("\n" + "="*90)
    print("FEATURE DISTINCTNESS (Head vs Tail)")
    print("="*90)
    print(f"{'Cosine Similarity of Mean Feature Vectors':<45} | {cos_sim:<12.4f}  (1.0 = identical/blind)")
    print(f"{'L2 Distance of Mean Feature Vectors':<45} | {l2_dist:<12.4f}  (0.0 = identical/blind)")
    print(f"{'L2 Distance of Gate Routing Weights':<45} | {w_l2_dist:<12.4f}  (0.0 = identical/blind)")
    print("="*90)

    print("\n[INFO] Analysis:")
    print("1. If Mean Max Logit is > 10.0, the expert is severely saturated.")
    print("2. If Cosine Similarity is > 0.95 and L2 Distance is near 0, the features are compressed.")
    print("3. If features are compressed, the Gate Routing L2 Distance will be near 0, proving the gate is blind to Head vs Tail.")

if __name__ == "__main__":
    main()