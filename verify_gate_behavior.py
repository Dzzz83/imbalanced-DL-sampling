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
from imbalanceddl.utils.plugin_rule import define_groups, define_groups_2
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

def main():
    cfg = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = ImbalancedDataset(cfg, cfg.dataset, augmentation='none')
    train_dataset, val_dataset = dataset.train_val_sets
    
    val_targets = np.array(val_dataset.targets)
    val_indices = np.arange(len(val_targets))
    tune_idx, test_idx = train_test_split(val_indices, test_size=0.8, stratify=val_targets, random_state=cfg.seed)
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
    
    head_weights = []
    med_weights = []
    tail_weights = []
    
    print("[INFO] Extracting routing weights across entire test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits_list, _ = model(images)
            adj_probs = [F.softmax(l / T, dim=1) for l in logits_list]
            phi = compute_gate_features(logits_list, adj_probs)
            gate_logits = gate(phi)
            weights = F.softmax(gate_logits, dim=1)
            
            for i in range(len(labels)):
                grp = group_ids_3[labels[i].item()]
                if grp == 0: 
                    head_weights.append(weights[i].cpu().numpy())
                elif grp == 1: 
                    med_weights.append(weights[i].cpu().numpy())
                else: 
                    tail_weights.append(weights[i].cpu().numpy())

    head_weights = np.array(head_weights)
    med_weights = np.array(med_weights)
    tail_weights = np.array(tail_weights)

    print("\n" + "="*80)
    print("AVERAGE GATE ROUTING WEIGHTS PER GROUP")
    print("="*80)
    print(f"{'Group':<10} | {'# Samples':<10} | {'CE Weight':<10} | {'LA Weight':<10} | {'BS Weight':<10}")
    print("-"*80)
    
    if len(head_weights) > 0:
        avg_ce = np.mean(head_weights[:, 0])
        avg_la = np.mean(head_weights[:, 1])
        avg_bs = np.mean(head_weights[:, 2])
        print(f"{'Head':<10} | {len(head_weights):<10} | {avg_ce:<10.4f} | {avg_la:<10.4f} | {avg_bs:<10.4f}")
        
    if len(med_weights) > 0:
        avg_ce = np.mean(med_weights[:, 0])
        avg_la = np.mean(med_weights[:, 1])
        avg_bs = np.mean(med_weights[:, 2])
        print(f"{'Medium':<10} | {len(med_weights):<10} | {avg_ce:<10.4f} | {avg_la:<10.4f} | {avg_bs:<10.4f}")
        
    if len(tail_weights) > 0:
        avg_ce = np.mean(tail_weights[:, 0])
        avg_la = np.mean(tail_weights[:, 1])
        avg_bs = np.mean(tail_weights[:, 2])
        print(f"{'Tail':<10} | {len(tail_weights):<10} | {avg_ce:<10.4f} | {avg_la:<10.4f} | {avg_bs:<10.4f}")
    print("="*80)

if __name__ == "__main__":
    main()