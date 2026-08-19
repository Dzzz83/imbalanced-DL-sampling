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
        embeddings_list = []
        for expert in self.experts:
            logits, hidden = expert(x)
            logits_list.append(logits)
            embeddings_list.append(hidden)
        embeddings = torch.cat(embeddings_list, dim=1)
        return logits_list, embeddings

class GateMLP(nn.Module):
    def __init__(self, input_dim=192, hidden1=256, hidden2=128, num_experts=3, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_experts)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
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
    
    cls_num_list = np.array(cfg.cls_num_list)
    priors = cls_num_list / cls_num_list.sum()
    sample_weights = priors[labels]
    sample_weights = sample_weights / sample_weights.sum()

    true_probs = probs[np.arange(len(labels)), labels]
    nll = -np.sum(sample_weights * np.log(true_probs + 1e-8))
    
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    brier = np.sum(sample_weights * np.sum((probs - one_hot)**2, axis=1))
    
    ece_all = compute_ece(conf, preds, labels)
    
    tail_mask = cls_num_list[labels] <= 20
    head_mask = ~tail_mask
    
    ece_tail = compute_ece(conf[tail_mask], preds[tail_mask], labels[tail_mask]) if np.sum(tail_mask) > 0 else 0.0
    ece_head = compute_ece(conf[head_mask], preds[head_mask], labels[head_mask]) if np.sum(head_mask) > 0 else 0.0
    
    return nll, brier, ece_all, ece_head, ece_tail

def main():
    cfg = get_args()
    if cfg.dataset == 'cifar100':
        cfg.num_classes = 100

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*100)
    print("CRISP STAGE 2 GATE VERIFICATION (FOLDER SCAN) - PAPER k=2 ROUTING")
    print("="*100)

    dataset = ImbalancedDataset(cfg, cfg.dataset, augmentation='none')
    train_dataset, val_dataset = dataset.train_val_sets
    
    train_targets = np.array(train_dataset.targets)
    cfg.cls_num_list = np.bincount(train_targets, minlength=cfg.num_classes).tolist()

    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

    ckpt_paths = {'CE': custom_args.ce_path, 'LA': custom_args.la_path, 'BS': custom_args.bs_path}
    model = ExpertEnsemble(cfg, device, ckpt_paths).to(device)
    
    gate = GateMLP(input_dim=192, hidden1=cfg.gate_hidden_size, hidden2=cfg.gate_hidden_size2).to(device)
    
    print("\n[INFO] Caching expert logits and embeddings on test set...")
    all_logits = [[], [], []]
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits_list, embeddings = model(images)
            for i in range(3):
                all_logits[i].append(logits_list[i].cpu())
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)
            
    all_logits = [torch.cat(l, dim=0) for l in all_logits]
    all_embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy()

    gate_files = sorted(glob.glob(os.path.join(custom_args.gate_dir, "*.pth")))
    if not gate_files:
        print(f"[ERROR] No checkpoints found in {custom_args.gate_dir}")
        sys.exit(1)
        
    print(f"[INFO] Found {len(gate_files)} gate checkpoints to evaluate.")
    
    la_tau = 1.0
    match_tau = re.search(r't([\d\.]+)', os.path.basename(custom_args.la_path))
    if match_tau:
        la_tau = float(match_tau.group(1))
    print(f"[INFO] Using LA Tau = {la_tau} parsed from filename")
    
    cls_num_list = torch.tensor(cfg.cls_num_list, device=device, dtype=torch.float32)
    log_prior = torch.log(cls_num_list / cls_num_list.sum() + 1e-12)
    log_spc = torch.log(cls_num_list + 1e-12)

    first_fname = os.path.basename(gate_files[0])
    match = re.search(r'T([\d\.]+)', first_fname)
    T_unif = float(match.group(1)) if match else 1.0
    
    print(f"[INFO] Computing Uniform Ensemble baseline (T={T_unif})...")
    p_ce = F.softmax(all_logits[0] / T_unif, dim=1)
    p_la = F.softmax((all_logits[1] + la_tau * log_prior.cpu()) / T_unif, dim=1)
    p_bs = F.softmax((all_logits[2] + log_spc.cpu()) / T_unif, dim=1)
    probs_unif = (p_ce + p_la + p_bs) / 3.0
    
    accs_unif = get_accs(probs_unif.numpy(), labels, cfg, train_dataset)
    cal_unif = get_calib(probs_unif.numpy(), labels, cfg)
    
    results = []

    for g_path in gate_files:
        fname = os.path.basename(g_path)
        clean_name = fname.replace("gate_checkpoint_", "").replace(".pth", "")
        
        match = re.search(r'T([\d\.]+)', fname)
        T = float(match.group(1)) if match else 1.0
        
        # FIX: Added weights_only=False for PyTorch 2.6+ compatibility
        ckpt = torch.load(g_path, map_location='cpu', weights_only=False)
        gate.load_state_dict(ckpt['gate_state_dict'])
        gate.to(device)
        gate.eval()
        
        with torch.no_grad():
            adj_probs = [
                F.softmax(all_logits[0] / T, dim=1),
                F.softmax((all_logits[1] + la_tau * log_prior.cpu()) / T, dim=1),
                F.softmax((all_logits[2] + log_spc.cpu()) / T, dim=1)
            ]
            
            gate_logits = gate(all_embeddings.to(device))
            weights = F.softmax(gate_logits, dim=1)
            
            k = 2  
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
            weights_np = weights.cpu().numpy()
            
            accs_mix = get_accs(p_mix_np, labels, cfg, train_dataset)
            cal_mix = get_calib(p_mix_np, labels, cfg)
            
            avg_w_ce = np.mean(weights_np[:, 0])
            avg_w_la = np.mean(weights_np[:, 1])
            avg_w_bs = np.mean(weights_np[:, 2])
            
            beats_unif = "✅" if accs_mix[0] >= accs_unif[0] else "❌"
            
            results.append({
                'name': clean_name,
                'bal_acc': accs_mix[0], 'many': accs_mix[1], 'med': accs_mix[2], 'low': accs_mix[3],
                'nll': cal_mix[0], 'brier': cal_mix[1], 'ece': cal_mix[2],
                'ece_head': cal_mix[3], 'ece_tail': cal_mix[4],
                'w_ce': avg_w_ce, 'w_la': avg_w_la, 'w_bs': avg_w_bs,
                'beats_unif': beats_unif
            })
            print(f"  Evaluated {clean_name:<25} | Bal Acc: {accs_mix[0]:.2f}% | {beats_unif}")

    print("\n" + "="*180)
    print("STAGE 2 METRICS SUMMARY (Gate Sweep vs Uniform Baseline) vs. PAPER (TABLE 3)")
    print("="*180)
    print(f"{'Checkpoint':<25} | {'Bal Acc':<7} | {'Many':<6} | {'Med':<6} | {'Low':<6} | {'NLL':<8} | {'Brier':<8} | {'ECE All':<8} | {'ECE Head':<8} | {'ECE Tail':<8} | {'w_CE':<6} | {'w_LA':<6} | {'w_BS':<6} | {'Beats Unif?':<12}")
    print("-"*180)
    
    print(f"{'PAPER CRISP':<25} | {'N/A':<7} | {'N/A':<6} | {'N/A':<6} | {'N/A':<6} | {'1.18':<8} | {'0.403':<8} | {'N/A':<8} | {'N/A':<8} | {'0.088':<8} | {'N/A':<6} | {'N/A':<6} | {'N/A':<6} | {'N/A':<12}")
    print("-"*180)
    
    print(f"{'UNIFORM BASELINE':<25} | {accs_unif[0]:<7.2f} | {accs_unif[1]:<6.2f} | {accs_unif[2]:<6.2f} | {accs_unif[3]:<6.2f} | {cal_unif[0]:<8.3f} | {cal_unif[1]:<8.3f} | {cal_unif[2]:<8.3f} | {cal_unif[3]:<8.3f} | {cal_unif[4]:<8.3f} | {'0.33':<6} | {'0.33':<6} | {'0.34':<6} | {'---':<12}")
    print("-"*180)
    
    results.sort(key=lambda x: -x['bal_acc'])
    
    for r in results:
        print(f"{r['name']:<25} | {r['bal_acc']:<7.2f} | {r['many']:<6.2f} | {r['med']:<6.2f} | {r['low']:<6.2f} | {r['nll']:<8.3f} | {r['brier']:<8.3f} | {r['ece']:<8.3f} | {r['ece_head']:<8.3f} | {r['ece_tail']:<8.3f} | {r['w_ce']:<6.3f} | {r['w_la']:<6.3f} | {r['w_bs']:<6.3f} | {r['beats_unif']:<12}")
    print("="*180)

if __name__ == "__main__":
    main()