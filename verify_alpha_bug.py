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
from imbalanceddl.utils.plugin_rule import define_groups_2, tune_plugin_for_rho
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
    # FIX: Explicitly use cuda:0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    print("\n" + "="*80)
    print("VERIFICATION: ALPHA TUNING BUG ON BALANCED SPLIT")
    print("="*80)

    dataset = ImbalancedDataset(cfg, cfg.dataset, augmentation='none')
    _, val_dataset = dataset.train_val_sets
    
    val_targets = np.array(val_dataset.targets)
    val_indices = np.arange(len(val_targets))
    tune_idx, _ = train_test_split(val_indices, test_size=0.8, stratify=val_targets, random_state=cfg.seed)
    tune_dataset = Subset(val_dataset, tune_idx)
    tune_loader = DataLoader(tune_dataset, batch_size=128, shuffle=False, num_workers=4)

    ckpt_paths = {'CE': custom_args.ce_path, 'LA': custom_args.la_path, 'BS': custom_args.bs_path}
    model = ExpertEnsemble(cfg, device, ckpt_paths).to(device)
    
    gate = GateMLP(input_dim=24, hidden1=cfg.gate_hidden_size, hidden2=cfg.gate_hidden_size2).to(device)
    gate_ckpt = torch.load(custom_args.gate_ckpt, map_location='cpu')
    gate.load_state_dict(gate_ckpt['gate_state_dict'])
    gate.eval()
    T = gate_ckpt.get('temperature', 1.0)

    # 1. Check Tune Set Distribution
    tune_labels = val_targets[tune_idx]
    group_ids_2 = define_groups_2(cfg.cls_num_list)
    tune_groups = group_ids_2[tune_labels]
    
    num_head = np.sum(tune_groups == 0)
    num_tail = np.sum(tune_groups == 1)
    print(f"\nTune Set Class Distribution:")
    print(f"  Head Samples: {num_head}")
    print(f"  Tail Samples: {num_tail}")
    if num_head == num_tail:
        print("  -> VERIFIED: The tune set is perfectly balanced (50/50).")

    # 2. Extract Posteriors
    print("\n[INFO] Extracting posteriors for tune set...")
    all_p_mix = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tune_loader:
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
            
    p_mix_tune = np.concatenate(all_p_mix, axis=0)
    labels_tune = np.concatenate(all_labels, axis=0)

    # 3. Tune Alpha at 20% Rejection
    print("\n[INFO] Tuning Plug-in rule at 20% rejection rate...")
    alpha_bal, mu_bal = tune_plugin_for_rho(p_mix_tune, labels_tune, group_ids_2, rho=0.2, mode='bal')
    
    print(f"\nTuned Alpha (Balanced): {alpha_bal}")
    print(f"Tuned Mu (Balanced):    {mu_bal}")
    
    if abs(alpha_bal[0] - alpha_bal[1]) < 0.1:
        print("\n  -> VERIFIED: Alpha_Head and Alpha_Tail are roughly equal.")
        print("     The algorithm is blind to the long-tailed distribution.")
        print("     It cannot boost tail class probabilities, resulting in poor AURC.")

if __name__ == "__main__":
    main()