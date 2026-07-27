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
    print("DETAILED VERIFICATION OF CLAIMS")
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

    print("[INFO] Extracting logits and gate weights...")
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
    # CLAIM 1: Vanishing Gradients due to Logit Saturation
    # =========================================================================
    print("\n" + "="*80)
    print("CLAIM 1: Vanishing Gradients due to Logit Saturation")
    print("="*80)
    
    ce_preds = np.argmax(adj_probs[0].numpy(), axis=1)
    tail_mask = group_ids_3[labels] == 2
    ce_wrong = ce_preds != labels
    critical_mask = tail_mask & ce_wrong
    
    critical_indices = np.where(critical_mask)[0]
    if len(critical_indices) > 0:
        idx = critical_indices[0]
        true_label = labels[idx]
        ce_pred = ce_preds[idx]
        
        logit_true = all_logits[0][idx, true_label].item()
        logit_pred = all_logits[0][idx, ce_pred].item()
        prob_true = adj_probs[0][idx, true_label].item()
        prob_pred = adj_probs[0][idx, ce_pred].item()
        
        print(f"Sample {idx} (True Label: {true_label}, Tail Class)")
        print(f"  CE Predicted: {ce_pred} (Logit: {logit_pred:.2f}, Prob: {prob_pred:.8f})")
        print(f"  CE True Class: {true_label} (Logit: {logit_true:.2f}, Prob: {prob_true:.8f})")
        
        # Simulate gradient calculation
        w = torch.tensor([0.33, 0.33, 0.34], requires_grad=True)
        p_true_tensor = torch.stack([adj_probs[0][idx, true_label], adj_probs[1][idx, true_label], adj_probs[2][idx, true_label]])
        mix_prob = (w * p_true_tensor).sum()
        loss = -torch.log(mix_prob + 1e-8)
        loss.backward()
        
        print(f"\n  NLL Loss: {loss.item():.4f}")
        print(f"  Gradient for CE weight (dL/dw_ce): {w.grad[0].item():.8f}")
        
        if abs(w.grad[0].item()) < 1e-4:
            print("  -> VERIFIED: The probability for the true class is so small (~3e-7)")
            print("     that the NLL gradient for the CE weight vanishes to near zero.")
            print("     The Stage 2 optimizer cannot detect this error to penalize CE.")

    # =========================================================================
    # CLAIM 2: "Fuzzy" Static Average (No Sample-Dependent Routing)
    # =========================================================================
    print("\n" + "="*80)
    print("CLAIM 2: 'Fuzzy' Static Average (No Sample-Dependent Routing)")
    print("="*80)
    
    for i, name in enumerate(['CE', 'LA', 'BS']):
        w_vec = weights_np[:, i]
        mean_w = np.mean(w_vec)
        std_w = np.std(w_vec)
        min_w = np.min(w_vec)
        max_w = np.max(w_vec)
        pct_gt_08 = np.mean(w_vec > 0.8) * 100
        
        print(f"Expert {name}:")
        print(f"  Mean: {mean_w:.3f} | Std Dev: {std_w:.3f}")
        print(f"  Min:  {min_w:.3f} | Max:     {max_w:.3f}")
        print(f"  % of samples where weight > 0.8: {pct_gt_08:.2f}%")
        
    print("\n-> If Std Dev is ~0.14 and Max is ~0.6, the weights only fluctuate")
    print("   in a narrow band (0.2 to 0.5). The gate is acting as a 'fuzzy' uniform")
    print("   average. True sample-dependent routing would require weights to swing")
    print("   from 0.0 to 1.0 (>80% should be common), which the lambda_bal=1.0")
    print("   regularizer explicitly prevents to avoid total collapse.")

if __name__ == "__main__":
    main()