#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import glob

# 1. Parse our custom arguments FIRST and remove them from sys.argv
custom_parser = argparse.ArgumentParser(add_help=False)
custom_parser.add_argument('--ckpt_dir', type=str, default='checkpoint/experts_sweep_cifar100_calib', help="Directory containing expert checkpoints")
custom_args, remaining_argv = custom_parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_argv

# 2. NOW import and call get_args()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from imbalanceddl.utils.config import get_args
from imbalanceddl.net.network import build_model
from imbalanceddl.dataset.imbalance_dataset import ImbalancedDataset
from imbalanceddl.utils.metrics import shot_acc
from torch.utils.data import DataLoader

def compute_ece(confidences, preds, labels, n_bins=15):
    accs = (preds == labels)
    bin_lowers = np.linspace(0, 1, n_bins + 1)[:-1]
    bin_uppers = np.linspace(0, 1, n_bins + 1)[1:]
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accs[in_bin])
            avg_conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_conf_in_bin - acc_in_bin) * prop_in_bin
    return ece

def load_expert(cfg, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    has_bias = ckpt.get('bias', False)
    
    # Set the default GPU to the requested device before building the model
    if 'cuda' in str(device):
        torch.cuda.set_device(device)
        
    model = build_model(cfg)
    
    # Unwrap DataParallel just in case
    actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    actual_model.classifier = nn.Linear(actual_model.feature_len, actual_model.num_classes, bias=has_bias)
    
    # Load weights and move to the single target device
    actual_model.load_state_dict(ckpt['state_dict'])
    actual_model = actual_model.to(device)
    actual_model.eval()
    
    return actual_model

def main():
    cfg = get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    print("="*100)
    print("CRISP STAGE 1 EXPERT VERIFICATION (FOLDER SCAN vs PAPER)")
    print("="*100)

    dataset = ImbalancedDataset(cfg, cfg.dataset, augmentation='none')
    train_dataset, val_dataset = dataset.train_val_sets
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

    ckpt_files = sorted(glob.glob(os.path.join(custom_args.ckpt_dir, "*.pth")))
    if not ckpt_files:
        print(f"[ERROR] No checkpoints found in {custom_args.ckpt_dir}")
        sys.exit(1)

    print(f"[INFO] Found {len(ckpt_files)} checkpoints to evaluate.")

    print("[INFO] Caching test set into memory...")
    all_images = []
    all_labels = []
    for images, labels in val_loader:
        all_images.append(images)
        all_labels.append(labels)
    all_images = torch.cat(all_images).to(device)
    all_labels = torch.cat(all_labels).cpu().numpy()

    train_targets = np.array(train_dataset.targets)
    cls_counts = np.bincount(train_targets, minlength=cfg.num_classes)
    
    priors = cls_counts / cls_counts.sum()
    sample_weights = priors[all_labels]
    sample_weights = sample_weights / sample_weights.sum()

    head_mask = cls_counts > 100
    tail_mask = cls_counts < 20
    test_head_mask = head_mask[all_labels]
    test_tail_mask = tail_mask[all_labels]

    results = []

    print("\n[INFO] Evaluating experts...")
    for ckpt_path in ckpt_files:
        fname = os.path.basename(ckpt_path)
        clean_name = fname.replace("expert_", "").replace(".pth", "").replace("bias", "b").replace("tau", "t")
        
        model = load_expert(cfg, ckpt_path, device)
        
        with torch.no_grad():
            batch_size = 256
            logits_list = []
            for i in range(0, len(all_images), batch_size):
                batch_imgs = all_images[i:i+batch_size]
                logits, _ = model(batch_imgs)
                logits_list.append(logits.cpu())
            
            logits = torch.cat(logits_list, dim=0)
            
            # --- FIXED PARSING & ADJUSTMENT BLOCK ---
            expert_name = clean_name.upper()
            log_prior = torch.log(torch.tensor(priors, device=logits.device) + 1e-12)
            
            if "LA" in expert_name:
                # Safely parse tau from filename parts
                tau = 1.0
                parts = clean_name.split('_')
                for p in parts:
                    if p.startswith('t'):
                        try:
                            tau = float(p[1:])
                        except ValueError:
                            pass
                adj_logits = logits + tau * log_prior
            elif "BS" in expert_name:
                log_spc = torch.log(torch.tensor(cls_counts, device=logits.device, dtype=torch.float32) + 1e-12)
                adj_logits = logits + log_spc
            else: # CE
                adj_logits = logits
                
            probs = F.softmax(adj_logits, dim=1).numpy()
            # ----------------------------------------
            preds = np.argmax(probs, axis=1)
            confidences = np.max(probs, axis=1)
            
            bal_acc = np.mean([np.mean(preds[all_labels == c] == c) for c in range(cfg.num_classes) if np.sum(all_labels == c) > 0]) * 100
            many, med, low = shot_acc(cfg, preds, all_labels, train_dataset, acc_per_cls=False)
            
            nll = -np.sum(sample_weights * np.log(probs[np.arange(len(all_labels)), all_labels] + 1e-8))
            brier = np.sum(sample_weights * np.sum((probs - np.eye(cfg.num_classes)[all_labels])**2, axis=1))
            
            ece_overall = compute_ece(confidences, preds, all_labels)
            ece_head = compute_ece(confidences[test_head_mask], preds[test_head_mask], all_labels[test_head_mask])
            ece_tail = compute_ece(confidences[test_tail_mask], preds[test_tail_mask], all_labels[test_tail_mask])
            
            max_logits = logits.max(dim=1)[0].numpy()
            
            results.append({
                'name': clean_name,
                'bal_acc': bal_acc,
                'many': many * 100,
                'med': med * 100,
                'low': low * 100,
                'nll': nll,
                'brier': brier,
                'ece_overall': ece_overall,
                'ece_head': ece_head,
                'ece_tail': ece_tail,
                'mean_max_logit': np.mean(max_logits),
                'sat_10': np.mean(max_logits > 10.0) * 100,
                'sat_20': np.mean(max_logits > 20.0) * 100,
            })
            
            print(f"  Evaluated {clean_name:<20} | Bal Acc: {bal_acc:.2f}% | Tail: {low*100:.2f}% | NLL: {nll:.3f}")
            
        del model
        torch.cuda.empty_cache()

    print("\n" + "="*150)
    print("STAGE 1 METRICS SUMMARY (LT-Weighted) vs. PAPER BASELINE (TABLE 3)")
    print("="*150)
    header = f"{'Expert Config':<20} | {'Bal Acc':<7} | {'Many':<6} | {'Med':<6} | {'Low':<6} | {'NLL':<6} | {'Brier':<6} | {'ECE All':<7} | {'ECE Head':<8} | {'ECE Tail':<8} | {'Mean Logit':<10} | {'%>10':<5} | {'%>20':<5}"
    print(header)
    print("-"*150)
    
    print(f"{'PAPER CE-only':<20} | {'~38.9':<7} | {'~65':<6} | {'~37':<6} | {'~10':<6} | {'1.78':<6} | {'0.531':<6} | {'N/A':<7} | {'N/A':<8} | {'0.520':<8} | {'N/A':<10} | {'N/A':<5} | {'N/A':<5}")
    print("-"*150)
    
    results.sort(key=lambda x: (x['name'][:2], -x['bal_acc']))
    
    for r in results:
        print(f"{r['name']:<20} | {r['bal_acc']:<7.2f} | {r['many']:<6.2f} | {r['med']:<6.2f} | {r['low']:<6.2f} | {r['nll']:<6.3f} | {r['brier']:<6.3f} | {r['ece_overall']:<7.3f} | {r['ece_head']:<8.3f} | {r['ece_tail']:<8.3f} | {r['mean_max_logit']:<10.2f} | {r['sat_10']:<5.1f} | {r['sat_20']:<5.1f}")
    print("="*150)
    
    print("\n[INFO] CRITICAL CHECK: Compare your 'NLL' and 'ECE Tail' against the 'PAPER CE-only' row.")

if __name__ == "__main__":
    main()