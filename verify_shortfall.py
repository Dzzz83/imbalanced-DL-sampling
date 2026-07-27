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

def main():
    cfg = get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    print("\n" + "="*80)
    print("VERIFICATION: WHY WE FALL SHORT OF THE PAPER")
    print("="*80)

    dataset = ImbalancedDataset(cfg, cfg.dataset, augmentation='none')
    _, val_dataset = dataset.train_val_sets
    
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

    weights_np = np.concatenate(all_weights, axis=0)

    # =========================================================================
    # CLAIM 1: Stage 1 Logit Saturation
    # =========================================================================
    print("\n" + "="*80)
    print("CLAIM 1: Stage 1 Logit Saturation")
    print("="*80)
    
    for i, name in enumerate(['CE', 'LA', 'BS']):
        max_logits = all_logits[i].max(dim=1)[0].numpy()
        max_probs = adj_probs[i].max(dim=1)[0].numpy()
        
        mean_logit = np.mean(max_logits)
        sat_10 = np.mean(max_logits > 10.0) * 100
        sat_20 = np.mean(max_logits > 20.0) * 100
        prob_1 = np.mean(max_probs >= 0.999) * 100
        
        print(f"Expert {name}: Mean Logit={mean_logit:.2f} | %>10={sat_10:.1f}% | %>20={sat_20:.1f}% | Prob~=1.0={prob_1:.1f}%")
    print("-> If Prob~=1.0 is high, probabilities are binary, causing NLL gradients to vanish.")

    # =========================================================================
    # CLAIM 2: Stage 2 Pseudo-Uniform Collapse
    # =========================================================================
    print("\n" + "="*80)
    print("CLAIM 2: Stage 2 Gate Routing Collapse (Pseudo-Uniform)")
    print("="*80)
    
    avg_w_ce = np.mean(weights_np[:, 0])
    avg_w_la = np.mean(weights_np[:, 1])
    avg_w_bs = np.mean(weights_np[:, 2])
    
    std_w_ce = np.std(weights_np[:, 0])
    std_w_la = np.std(weights_np[:, 1])
    std_w_bs = np.std(weights_np[:, 2])
    
    print(f"Average Weights: CE={avg_w_ce:.3f} | LA={avg_w_la:.3f} | BS={avg_w_bs:.3f}")
    print(f"Std Dev Weights: CE={std_w_ce:.3f} | LA={std_w_la:.3f} | BS={std_w_bs:.3f}")
    print("-> If Std Dev is near 0, the gate is static (pseudo-uniform), not sample-dependent.")

    # =========================================================================
    # CLAIM 3: Failure to Route on Critical Tail Samples
    # =========================================================================
    print("\n" + "="*80)
    print("CLAIM 3: Failure to Route on Critical Tail Samples")
    print("="*80)
    
    ce_preds = np.argmax(adj_probs[0].numpy(), axis=1)
    la_preds = np.argmax(adj_probs[1].numpy(), axis=1)
    
    tail_mask = group_ids_3[labels] == 2
    ce_wrong = ce_preds != labels
    la_right = la_preds == labels
    critical_mask = tail_mask & ce_wrong & la_right
    
    num_critical = np.sum(critical_mask)
    print(f"Found {num_critical} critical tail samples (CE WRONG, LA RIGHT).")
    
    if num_critical > 0:
        avg_w_ce_crit = np.mean(weights_np[critical_mask, 0])
        avg_w_la_crit = np.mean(weights_np[critical_mask, 1])
        avg_w_bs_crit = np.mean(weights_np[critical_mask, 2])
        
        print(f"Average Gate Weights on these critical samples:")
        print(f"  CE (Wrong): {avg_w_ce_crit:.3f}")
        print(f"  LA (Right): {avg_w_la_crit:.3f}")
        print(f"  BS:         {avg_w_bs_crit:.3f}")
        print("-> If LA weight is ~0.3 instead of ~0.9, the gate failed to route.")
        print("   This dilutes the correct LA signal, causing tail accuracy to collapse.")

if __name__ == "__main__":
    main()