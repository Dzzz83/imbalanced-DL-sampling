# ultra_debug.py
import sys
import os

# 1. Pre-process sys.argv before any project imports
experts_ckpt = None
gate_ckpt = None
config_path = None

argv = sys.argv[1:]
new_argv = []
i = 0
while i < len(argv):
    if argv[i] == '--experts_ckpt' and i + 1 < len(argv):
        experts_ckpt = argv[i+1]
        i += 2
    elif argv[i] == '--gate_ckpt' and i + 1 < len(argv):
        gate_ckpt = argv[i+1]
        i += 2
    elif argv[i] == '--config' and i + 1 < len(argv):
        config_path = argv[i+1]
        i += 2
    else:
        new_argv.append(argv[i])
        i += 1

# Reconstruct sys.argv so get_args() only sees the config file
sys.argv = ['main.py', '-c', config_path] if config_path else ['main.py']

# 2. Now import project modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
import numpy as np
from imbalanceddl.utils.config import get_args
from imbalanceddl.net.network import build_model
from imbalanceddl.dataset.imbalance_dataset import ImbalancedDataset
from imbalanceddl.utils.gate_features import compute_gate_features
from imbalanceddl.utils.plugin_rule import define_groups, tune_plugin_bal, tune_plugin_worst, compute_paper_metrics
from torch.utils.data import DataLoader

def get_group_masks(labels, cls_num_list):
    cls_num_list = np.array(cls_num_list)
    group_ids = define_groups(cls_num_list)
    label_groups = group_ids[labels]
    return {
        'Head': label_groups == 0,
        'Medium': label_groups == 1,
        'Tail': label_groups == 2
    }

def main():
    if not config_path:
        print("ERROR: Please provide --config <path>")
        sys.exit(1)
    if not experts_ckpt:
        print("ERROR: Please provide --experts_ckpt <path>")
        sys.exit(1)

    cfg = get_args()
    if cfg.dataset == 'cifar100':
        cfg.num_classes = 100

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "="*80)
    print("ULTRA DEBUG: CRISP PIPELINE DIAGNOSTICS")
    print("="*80)

    print("\n[1] LOADING EXPERTS")
    cfg.strategy = 'Experts'
    model = build_model(cfg)
    checkpoint = torch.load(experts_ckpt, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    print("\n[2] LOADING DATA")
    dataset = ImbalancedDataset(cfg, dataset_name=cfg.dataset, augmentation='none')
    _, val_dataset = dataset.train_val_sets
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

    cls_num_list = torch.FloatTensor(cfg.cls_num_list).to(device)
    log_prior = (cls_num_list / cls_num_list.sum()).log()

    all_logits = [[], [], []]
    all_labels = []

    print("\n[3] EXTRACTING LOGITS...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits_list, _ = model(images)
            for i in range(3):
                all_logits[i].append(logits_list[i])
            all_labels.append(labels)

    all_logits = [torch.cat(l, dim=0) for l in all_logits]
    all_labels = torch.cat(all_labels, dim=0).to(device)
    labels_np = all_labels.cpu().numpy()
    masks = get_group_masks(labels_np, cfg.cls_num_list)

    print("\n" + "="*80)
    print("DIAGNOSTIC 1: EXPERT LOGIT HEALTH (Saturation Check)")
    print("="*80)
    for i in range(3):
        max_logits = all_logits[i].max(dim=1)[0]
        mean_max = max_logits.mean().item()
        sat_count = (max_logits > 15.0).sum().item()
        print(f"Expert {i}: Mean Max Logit = {mean_max:.4f} | Saturated (>15.0) = {sat_count}/{len(labels_np)}")

    print("\n" + "="*80)
    print("DIAGNOSTIC 2: STRUCTURAL DIVERSITY (Per-Group Accuracy)")
    print("="*80)
    
    adj_probs = [
        F.softmax(all_logits[0], dim=1),
        F.softmax(all_logits[1] - log_prior, dim=1),
        F.softmax(all_logits[2] + log_prior, dim=1)
    ]
    
    print(f"{'Expert':<10} | {'Overall':<10} | {'Head':<10} | {'Medium':<10} | {'Tail':<10}")
    print("-"*60)
    for i in range(3):
        preds = adj_probs[i].argmax(dim=1).cpu().numpy()
        acc_overall = np.mean(preds == labels_np) * 100
        acc_head = np.mean(preds[masks['Head']] == labels_np[masks['Head']]) * 100 if np.sum(masks['Head']) > 0 else 0
        acc_med = np.mean(preds[masks['Medium']] == labels_np[masks['Medium']]) * 100 if np.sum(masks['Medium']) > 0 else 0
        acc_tail = np.mean(preds[masks['Tail']] == labels_np[masks['Tail']]) * 100 if np.sum(masks['Tail']) > 0 else 0
        print(f"Exp {i}     | {acc_overall:<10.2f} | {acc_head:<10.2f} | {acc_med:<10.2f} | {acc_tail:<10.2f}")
    
    print("\n" + "="*80)
    print("DIAGNOSTIC 3: GATE FEATURE VARIANCE (Blind Gate Check)")
    print("="*80)
    
    phi = compute_gate_features(all_logits, adj_probs)
    feat_std = phi.std(dim=0)
    print(f"Feature Vector Shape: {phi.shape}")
    print(f"Mean Std Dev across batch: {feat_std.mean().item():.6f}")
    if feat_std.mean().item() < 0.01:
        print("[CRITICAL BUG] Gate features have near-zero variance. Gate is completely blind.")
    else:
        print("[INFO] Gate features have healthy variance.")

    print("\n" + "="*80)
    print("DIAGNOSTIC 4: GATE ROUTING & MIXTURE HEALTH")
    print("="*80)
    
    if gate_ckpt and os.path.exists(gate_ckpt):
        print(f"[INFO] Loading gate from {gate_ckpt}")
        from imbalanceddl.strategy._gate_trainer import GateMLP
        gate = GateMLP(input_dim=24, hidden1=256, hidden2=128, num_experts=3).to(device)
        gate_ckpt_data = torch.load(gate_ckpt, map_location=device)
        gate.load_state_dict(gate_ckpt_data['gate_state_dict'])
        gate.eval()
        
        with torch.no_grad():
            gate_logits = gate(phi)
            weights = F.softmax(gate_logits, dim=1)
            
            prob_true = torch.stack([adj_probs[i][torch.arange(len(labels_np)), all_labels] for i in range(3)], dim=1)
            mix_prob_true = (weights * prob_true).sum(dim=1)
            mix_nll = -torch.log(mix_prob_true + 1e-8).mean().item()
            
            ent_reg = -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean().item()
            avg_weights = weights.mean(dim=0)
            bal_reg = ((avg_weights - 1.0 / 3.0) ** 2).sum().item()
            
            print(f"Gate Loss Components:")
            print(f"  Mix NLL   = {mix_nll:.4f}")
            print(f"  Ent Reg   = {ent_reg:.4f} (Weighted: {cfg.lambda_ent * ent_reg:.4f})")
            print(f"  Bal Reg   = {bal_reg:.4f} (Weighted: {cfg.lambda_bal * bal_reg:.4f})")
            
            print(f"\nAverage Gate Weights (Pre-TopK): CE={avg_weights[0]:.4f}, LA={avg_weights[1]:.4f}, BS={avg_weights[2]:.4f}")
            if np.max(avg_weights.cpu().numpy()) > 0.95:
                print("[CRITICAL BUG] Gate has collapsed onto a single expert!")
                
            k = getattr(cfg, 'routing_sparsity', 2)
            topk_weights, topk_indices = torch.topk(weights, k, dim=1)
            topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)
            
            p_mix = torch.zeros_like(adj_probs[0])
            for i in range(k):
                idx = topk_indices[:, i]
                w = topk_weights[:, i].unsqueeze(1)
                expert_probs = torch.stack(adj_probs, dim=1)[torch.arange(len(labels_np)), idx, :]
                p_mix += w * expert_probs
    else:
        print("[INFO] No gate provided. Using uniform ensemble (Structured-DE).")
        p_mix = torch.stack(adj_probs, dim=0).mean(dim=0)

    print("\n" + "="*80)
    print("DIAGNOSTIC 5: PLUG-IN RULE & FINAL METRICS")
    print("="*80)
    
    p_mix_np = p_mix.cpu().numpy()
    group_ids = define_groups(cfg.cls_num_list)
    
    params_bal = tune_plugin_bal(p_mix_np, labels_np, group_ids)
    params_worst = tune_plugin_worst(p_mix_np, labels_np, group_ids)
    
    print(f"Tuned Parameters (Bal):   Alpha = {params_bal['alpha']} | Mu = {params_bal['mu']}")
    print(f"Tuned Parameters (Worst): Alpha = {params_worst['alpha']} | Mu = {params_worst['mu']}")
    
    metrics_bal = compute_paper_metrics(p_mix_np, labels_np, group_ids, params_bal['alpha'], params_bal['mu'])
    metrics_worst = compute_paper_metrics(p_mix_np, labels_np, group_ids, params_worst['alpha'], params_worst['mu'])
    
    print("\n" + "-"*80)
    print(f"{'Method':<25} | {'AURCbal':<10} | {'AURCwst':<10} | {'NLL':<10} | {'Brier':<10} | {'tail-ECE':<10}")
    print("-"*80)
    print(f"{'CRISP+Plug-in[Bal]':<25} | {metrics_bal['AURCbal']:<10.4f} | {metrics_bal['AURCwst']:<10.4f} | {metrics_bal['NLL']:<10.4f} | {metrics_bal['Brier']:<10.4f} | {metrics_bal['tail-ECE']:<10.4f}")
    print(f"{'CRISP+Plug-in[Worst]':<25} | {metrics_worst['AURCbal']:<10.4f} | {metrics_worst['AURCwst']:<10.4f} | {metrics_worst['NLL']:<10.4f} | {metrics_worst['Brier']:<10.4f} | {metrics_worst['tail-ECE']:<10.4f}")
    print("-"*80)
    print(f"{'Paper Reference (Bal)':<25} | 0.253      | 0.302      | 1.18       | 0.403      | 0.088      ")
    print(f"{'Paper Reference (Worst)':<25} | 0.233      | 0.248      | 1.18       | 0.403      | 0.088      ")
    print("="*80)

if __name__ == "__main__":
    main()