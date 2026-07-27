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
custom_parser.add_argument('--ce0', type=str, required=True)
custom_parser.add_argument('--la0', type=str, required=True)
custom_parser.add_argument('--bs0', type=str, required=True)
custom_parser.add_argument('--gate0', type=str, required=True)

custom_parser.add_argument('--ce1', type=str, required=True)
custom_parser.add_argument('--la1', type=str, required=True)
custom_parser.add_argument('--bs1', type=str, required=True)
custom_parser.add_argument('--gate1', type=str, required=True)
custom_args, remaining_argv = custom_parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_argv

# 2. NOW import and call get_args()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from imbalanceddl.utils.config import get_args
from imbalanceddl.net.network import build_model
from imbalanceddl.dataset.imbalance_dataset import ImbalancedDataset
from imbalanceddl.utils.gate_features import compute_gate_features
from imbalanceddl.utils.plugin_rule import define_groups
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
            model.load_state_dict(ckpt['state_dict'])
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

def analyze_run(run_name, cfg, device, ckpt_paths, gate_ckpt_path, test_loader, group_ids_3, labels_np):
    print(f"\n{'='*80}")
    print(f"ANALYZING RUN: {run_name}")
    print(f"{'='*80}")
    
    model = ExpertEnsemble(cfg, device, ckpt_paths).to(device)
    gate = GateMLP(input_dim=24, hidden1=cfg.gate_hidden_size, hidden2=cfg.gate_hidden_size2).to(device)
    
    gate_ckpt = torch.load(gate_ckpt_path, map_location='cpu')
    gate.load_state_dict(gate_ckpt['gate_state_dict'])
    gate.eval()
    T = gate_ckpt.get('temperature', 1.0)

    all_logits = [[], [], []]
    
    print("[INFO] Extracting logits...")
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            logits_list, _ = model(images)
            for i in range(3):
                all_logits[i].append(logits_list[i].cpu())
            
        all_logits = [torch.cat(l, dim=0) for l in all_logits]
        
        adj_probs = [
            F.softmax(all_logits[0] / T, dim=1),
            F.softmax(all_logits[1] / T, dim=1),
            F.softmax(all_logits[2] / T, dim=1)
        ]
        
        # FIX: Ensure gate inference is inside no_grad
        phi = compute_gate_features(all_logits, adj_probs).to(device)
        gate_logits = gate(phi)
        weights = F.softmax(gate_logits, dim=1).cpu().numpy()
    
    ce_preds = np.argmax(adj_probs[0].numpy(), axis=1)
    la_preds = np.argmax(adj_probs[1].numpy(), axis=1)
    
    # Find Critical Tail Samples: Tail class, CE WRONG, LA CORRECT
    tail_mask = group_ids_3[labels_np] == 2
    ce_wrong = ce_preds != labels_np
    la_right = la_preds == labels_np
    critical_mask = tail_mask & ce_wrong & la_right
    
    num_critical = np.sum(critical_mask)
    print(f"Found {num_critical} critical tail samples (CE wrong, LA right).")
    
    if num_critical == 0:
        print("No critical samples found.")
        return
        
    # 1. Extract Max Probabilities
    max_p_ce = adj_probs[0].max(dim=1)[0].numpy()[critical_mask]
    max_p_la = adj_probs[1].max(dim=1)[0].numpy()[critical_mask]
    max_p_bs = adj_probs[2].max(dim=1)[0].numpy()[critical_mask]
    
    print(f"\nAverage Max Probability on Critical Samples:")
    print(f"  CE (Wrong): {np.mean(max_p_ce):.4f}")
    print(f"  LA (Right): {np.mean(max_p_la):.4f}")
    print(f"  BS:         {np.mean(max_p_bs):.4f}")
    
    # 2. Extract Gate Weights
    w_ce = weights[critical_mask, 0]
    w_la = weights[critical_mask, 1]
    w_bs = weights[critical_mask, 2]
    
    print(f"\nAverage Gate Weights on Critical Samples:")
    print(f"  CE (Wrong): {np.mean(w_ce):.4f}")
    print(f"  LA (Right): {np.mean(w_la):.4f}")
    print(f"  BS:         {np.mean(w_bs):.4f}")

def main():
    cfg = get_args()
    # FIX: Explicitly use cuda:0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    dataset = ImbalancedDataset(cfg, cfg.dataset, augmentation='none')
    _, val_dataset = dataset.train_val_sets
    
    val_targets = np.array(val_dataset.targets)
    val_indices = np.arange(len(val_targets))
    _, test_idx = train_test_split(val_indices, test_size=0.8, stratify=val_targets, random_state=cfg.seed)
    test_dataset = Subset(val_dataset, test_idx)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    
    all_labels = []
    for _, labels in test_loader:
        all_labels.append(labels)
    labels_np = torch.cat(all_labels).numpy()

    group_ids_3 = define_groups(cfg.cls_num_list)

    # Run 1: ls=0.0
    ckpt_paths_0 = {'CE': custom_args.ce0, 'LA': custom_args.la0, 'BS': custom_args.bs0}
    analyze_run("ls=0.0 (Old Run)", cfg, device, ckpt_paths_0, custom_args.gate0, test_loader, group_ids_3, labels_np)

    # Run 2: ls=0.1
    ckpt_paths_1 = {'CE': custom_args.ce1, 'LA': custom_args.la1, 'BS': custom_args.bs1}
    analyze_run("ls=0.1 (New Run)", cfg, device, ckpt_paths_1, custom_args.gate1, test_loader, group_ids_3, labels_np)

if __name__ == "__main__":
    main()