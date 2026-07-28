#!/usr/bin/env python3
# verify_feature_squashing.py
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

    la_tau = 1.5
    match_tau = re.search(r't([\d\.]+)', os.path.basename(custom_args.la_path))
    if match_tau:
        la_tau = float(match_tau.group(1))
    
    cls_num_list = torch.tensor(cfg.cls_num_list, device=device, dtype=torch.float32)
    log_prior = torch.log(cls_num_list / cls_num_list.sum() + 1e-12)
    log_spc = torch.log(cls_num_list + 1e-12)

    all_logits = [[], [], []]
    all_labels = []
    
    print("\n[INFO] Extracting features on test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits_list, _ = model(images)
            for i in range(3):
                all_logits[i].append(logits_list[i])
            all_labels.append(labels)
            
        all_logits = [torch.cat(l, dim=0) for l in all_logits]
        labels = torch.cat(all_labels, dim=0).cpu().numpy()
        
        p_ce = F.softmax(all_logits[0], dim=1)
        p_la = F.softmax((all_logits[1] + la_tau * log_prior), dim=1)
        p_bs = F.softmax((all_logits[2] + log_spc), dim=1)
        adj_probs = [p_ce, p_la, p_bs]
        
        phi = compute_gate_features(all_logits, adj_probs).cpu().numpy()
        probs_np = [p.cpu().numpy() for p in adj_probs]

    group_ids_2 = define_groups_2(cfg.cls_num_list)
    tail_mask = (group_ids_2[labels] == 1)
    head_mask = ~tail_mask
    
    print("\n" + "="*90)
    print("FEATURE SQUASHING VERIFICATION (Head vs Tail)")
    print("="*90)
    print(f"{'Feature':<20} | {'Head Mean':<10} | {'Head Std':<10} | {'Tail Mean':<10} | {'Tail Std':<10}")
    print("-"*65)
    
    # 0: Ent, 1: Max, 2: Marg, 6: KL for each expert
    feats = {
        "CE Max Prob": 1,
        "CE Entropy": 0,
        "CE Margin": 2,
        "CE KL": 6,
        "LA Max Prob": 8,
        "LA Entropy": 7,
        "LA Margin": 9,
        "LA KL": 13,
        "BS Max Prob": 15,
        "BS Entropy": 14,
        "BS Margin": 16,
        "BS KL": 20
    }
    
    for name, idx in feats.items():
        h_mean = np.mean(phi[head_mask, idx])
        h_std = np.std(phi[head_mask, idx])
        t_mean = np.mean(phi[tail_mask, idx])
        t_std = np.std(phi[tail_mask, idx])
        print(f"{name:<20} | {h_mean:<10.4f} | {h_std:<10.4f} | {t_mean:<10.4f} | {t_std:<10.4f}")
    print("="*90)

    print("\n[INFO] Analysis:")
    print("1. If Max Prob is > 0.95, the expert is 100% confident (squashed).")
    print("2. If Entropy is near 0.0, the probability distribution is one-hot (squashed).")
    print("3. If Margin is near 1.0, the top-1 and top-2 probabilities are far apart (squashed).")
    print("4. If KL is near 0.0, the expert distribution matches the uniform mean (squashed).")

if __name__ == "__main__":
    main()