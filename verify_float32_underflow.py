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
custom_args, remaining_argv = custom_parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_argv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from imbalanceddl.utils.config import get_args
from imbalanceddl.net.network import build_model
from imbalanceddl.dataset.imbalance_dataset import ImbalancedDataset
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

def main():
    cfg = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = ImbalancedDataset(cfg, cfg.dataset, augmentation='none')
    train_dataset, val_dataset = dataset.train_val_sets
    
    val_targets = np.array(val_dataset.targets)
    val_indices = np.arange(len(val_targets))
    _, test_idx = train_test_split(val_indices, test_size=0.8, stratify=val_targets, random_state=cfg.seed)
    test_dataset = Subset(val_dataset, test_idx)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    ckpt_paths = {'CE': custom_args.ce_path, 'LA': custom_args.la_path, 'BS': custom_args.bs_path}
    model = ExpertEnsemble(cfg, device, ckpt_paths).to(device)
    
    group_ids_3 = define_groups(cfg.cls_num_list)
    
    print("[INFO] Extracting logits...")
    all_logits = [[], [], []]
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits_list, _ = model(images)
            for i in range(3):
                all_logits[i].append(logits_list[i].cpu())
            all_labels.append(labels)
            
    all_logits = [torch.cat(l, dim=0) for l in all_logits]
    labels = torch.cat(all_labels, dim=0).numpy()

    print("\n" + "="*80)
    print("VERIFICATION: FLOAT32 UNDERFLOW & VANISHING GRADIENTS")
    print("="*80)

    # Check T=1.0 (Raw logits)
    T = 1.0
    print(f"\n--- Testing with Temperature T = {T} ---")
    p_ce = F.softmax(all_logits[0] / T, dim=1)
    p_la = F.softmax(all_logits[1] / T, dim=1)
    p_bs = F.softmax(all_logits[2] / T, dim=1)
    
    ce_preds = np.argmax(p_ce.numpy(), axis=1)
    tail_mask = group_ids_3[labels] == 2
    ce_wrong = ce_preds != labels
    critical_mask = tail_mask & ce_wrong
    
    # Extract probabilities for the true class
    p_ce_true = p_ce[torch.arange(len(labels)), labels]
    
    # Count how many are exactly 0.0 in float32
    critical_p_ce_true = p_ce_true[critical_mask]
    num_zeros = torch.sum(critical_p_ce_true == 0.0).item()
    num_critical = len(critical_p_ce_true)
    
    print(f"Found {num_critical} tail samples where CE is wrong.")
    print(f"Of those, {num_zeros} have p_ce[true_class] EXACTLY equal to 0.0 in float32.")
    
    if num_zeros > 0:
        # Find the first one
        zero_indices = torch.where(critical_p_ce_true == 0.0)[0]
        idx = np.where(critical_mask)[0][zero_indices[0].item()]
        
        print(f"\nAnalyzing Sample {idx} (True Label: {labels[idx]}, Tail Class)")
        print(f"  CE pred: {ce_preds[idx]} (Raw Logit: {all_logits[0][idx, ce_preds[idx]].item():.4f})")
        print(f"  CE true logit: {all_logits[0][idx, labels[idx]].item():.4f}")
        print(f"  CE prob for true class: {p_ce[idx, labels[idx]].item():.10f} (Underflowed to 0.0!)")
        
        # Simulate Gate NLL Gradient
        weights = torch.tensor([0.33, 0.33, 0.34], requires_grad=True)
        probs_true = torch.stack([p_ce[idx, labels[idx]], p_la[idx, labels[idx]], p_bs[idx, labels[idx]]])
        
        mix_prob = (weights * probs_true).sum()
        mix_nll = -torch.log(mix_prob + 1e-8)
        
        mix_nll.backward()
        
        print(f"\n  Gate NLL Loss: {mix_nll.item():.4f}")
        print(f"  Gate Weight Gradients (dL/dw):")
        print(f"    CE grad: {weights.grad[0].item():.10f}")
        print(f"    LA grad: {weights.grad[1].item():.10f}")
        print(f"    BS grad: {weights.grad[2].item():.10f}")
        
        if weights.grad[0].item() == 0.0:
            print("\n  -> VERIFIED: Because p_ce underflowed to 0.0, the CE gradient is EXACTLY 0.0.")
            print("     The gate receives zero signal to penalize CE for this wrong tail prediction.")

if __name__ == "__main__":
    main()