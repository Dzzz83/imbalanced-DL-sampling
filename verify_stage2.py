#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import glob
import re

custom_parser = argparse.ArgumentParser(add_help=False)
custom_parser.add_argument('--ce_path', type=str, required=True)
custom_parser.add_argument('--la_path', type=str, required=True)
custom_parser.add_argument('--bs_path', type=str, required=True)
custom_parser.add_argument('--gate_dir', type=str, required=True)
custom_args, remaining_argv = custom_parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_argv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from imbalanceddl.utils.config import get_args
from imbalanceddl.net.network import build_model
from imbalanceddl.dataset.imbalance_dataset import ImbalancedDataset
from imbalanceddl.utils.metrics import shot_acc
from imbalanceddl.utils.gate_features import compute_gate_features
from torch.utils.data import DataLoader

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

def get_accs(probs, labels, cfg, train_dataset):
    preds = np.argmax(probs, axis=1)
    bal = np.mean([np.mean(preds[labels == c] == c) for c in range(cfg.num_classes) if np.sum(labels == c) > 0]) * 100
    many, med, low = shot_acc(cfg, preds, labels, train_dataset, acc_per_cls=False)
    return bal, many*100, med*100, low*100

def get_calib(probs, labels, cfg):
    preds = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    nll = -np.mean(np.log(probs[np.arange(len(labels)), labels] + 1e-8))
    brier = np.mean(np.sum((probs - np.eye(cfg.num_classes)[labels])**2, axis=1))
    ece = compute_ece(conf, preds, labels)
    return nll, brier, ece

def main():
    cfg = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*100)
    print("CRISP STAGE 2 GATE VERIFICATION (FOLDER SCAN)")
    print("="*100)

    dataset = ImbalancedDataset(cfg, cfg.dataset, augmentation='none')
    train_dataset, val_dataset = dataset.train_val_sets
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

    ckpt_paths = {'CE': custom_args.ce_path, 'LA': custom_args.la_path, 'BS': custom_args.bs_path}
    model = ExpertEnsemble(cfg, device, ckpt_paths).to(device)
    
    gate = GateMLP(input_dim=24, hidden1=cfg.gate_hidden_size, hidden2=cfg.gate_hidden_size2).to(device)
    
    print("\n[INFO] Caching expert logits on test set...")
    all_logits = [[], [], []]
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits_list, _ = model(images)
            for i in range(3):
                all_logits[i].append(logits_list[i].cpu())
            all_labels.append(labels)
            
    all_logits = [torch.cat(l, dim=0) for l in all_logits]
    labels = torch.cat(all_labels, dim=0).numpy()
    
    print("[INFO] Computing Uniform Ensemble baseline...")
    probs_unif = (F.softmax(all_logits[0], dim=1) + F.softmax(all_logits[1], dim=1) + F.softmax(all_logits[2], dim=1)) / 3.0
    accs_unif = get_accs(probs_unif.numpy(), labels, cfg, train_dataset)
    cal_unif = get_calib(probs_unif.numpy(), labels, cfg)

    gate_files = sorted(glob.glob(os.path.join(custom_args.gate_dir, "*.pth")))
    if not gate_files:
        print(f"[ERROR] No checkpoints found in {custom_args.gate_dir}")
        sys.exit(1)
        
    print(f"[INFO] Found {len(gate_files)} gate checkpoints to evaluate.")
    
    results = []

    for g_path in gate_files:
        fname = os.path.basename(g_path)
        clean_name = fname.replace("gate_checkpoint_", "").replace(".pth", "")
        
        match = re.search(r'T([\d\.]+)', fname)
        T = float(match.group(1)) if match else 1.0
        
        ckpt = torch.load(g_path, map_location='cpu')
        gate.load_state_dict(ckpt['gate_state_dict'])
        gate.to(device)
        gate.eval()
        
        with torch.no_grad():
            adj_probs = [
                F.softmax(all_logits[0] / T, dim=1),
                F.softmax(all_logits[1] / T, dim=1),
                F.softmax(all_logits[2] / T, dim=1)
            ]
            
            phi = compute_gate_features(all_logits, adj_probs).to(device)
            gate_logits = gate(phi)
            weights = F.softmax(gate_logits, dim=1)
            
            k = getattr(cfg, 'routing_sparsity', 2)
            topk_weights, topk_indices = torch.topk(weights, k, dim=1)
            topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)
            
            stacked_probs = torch.stack(adj_probs, dim=1).to(device)
            p_mix = torch.zeros_like(stacked_probs[:, 0, :])
            N = stacked_probs.size(0)
            for i in range(k):
                idx = topk_indices[:, i]
                w = topk_weights[:, i].unsqueeze(1)
                expert_probs = stacked_probs[torch.arange(N), idx, :]
                p_mix += w * expert_probs
                
            p_mix_np = p_mix.cpu().numpy()
            
            accs_mix = get_accs(p_mix_np, labels, cfg, train_dataset)
            cal_mix = get_calib(p_mix_np, labels, cfg)
            
            beats_unif = "✅" if accs_mix[0] >= accs_unif[0] else "❌"
            
            results.append({
                'name': clean_name,
                'bal_acc': accs_mix[0], 'many': accs_mix[1], 'med': accs_mix[2], 'low': accs_mix[3],
                'nll': cal_mix[0], 'brier': cal_mix[1], 'ece': cal_mix[2],
                'beats_unif': beats_unif
            })
            print(f"  Evaluated {clean_name:<25} | Bal Acc: {accs_mix[0]:.2f}% | {beats_unif}")

    print("\n" + "="*130)
    print("STAGE 2 METRICS SUMMARY (Gate Sweep vs Uniform Baseline)")
    print("="*130)
    print(f"{'Checkpoint':<25} | {'Bal Acc':<7} | {'Many':<6} | {'Med':<6} | {'Low':<6} | {'NLL':<8} | {'Brier':<8} | {'ECE':<8} | {'Beats Unif?':<12}")
    print("-"*130)
    
    print(f"{'UNIFORM BASELINE':<25} | {accs_unif[0]:<7.2f} | {accs_unif[1]:<6.2f} | {accs_unif[2]:<6.2f} | {accs_unif[3]:<6.2f} | {cal_unif[0]:<8.3f} | {cal_unif[1]:<8.3f} | {cal_unif[2]:<8.3f} | {'---':<12}")
    print("-"*130)
    
    results.sort(key=lambda x: -x['bal_acc'])
    
    for r in results:
        print(f"{r['name']:<25} | {r['bal_acc']:<7.2f} | {r['many']:<6.2f} | {r['med']:<6.2f} | {r['low']:<6.2f} | {r['nll']:<8.3f} | {r['brier']:<8.3f} | {r['ece']:<8.3f} | {r['beats_unif']:<12}")
    print("="*130)

if __name__ == "__main__":
    main()