# ultra_debug.py
import sys
import os

# Pre-process sys.argv
experts_dir = None
gate_ckpt = None
config_path = None

argv = sys.argv[1:]
new_argv = []
i = 0
while i < len(argv):
    if argv[i] == '--experts_dir' and i + 1 < len(argv):
        experts_dir = argv[i+1]; i += 2
    elif argv[i] == '--gate_ckpt' and i + 1 < len(argv):
        gate_ckpt = argv[i+1]; i += 2
    elif argv[i] == '--config' and i + 1 < len(argv):
        config_path = argv[i+1]; i += 2
    else:
        new_argv.append(argv[i]); i += 1

sys.argv = ['main.py', '-c', config_path] if config_path else ['main.py']

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
    return {'Head': label_groups == 0, 'Medium': label_groups == 1, 'Tail': label_groups == 2}

def main():
    cfg = get_args()
    if cfg.dataset == 'cifar100': cfg.num_classes = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "="*80)
    print("ULTRA DEBUG: CRISP PIPELINE (ORACLE & DIVERSITY CHECK)")
    print("="*80)

    print("\n[1] LOADING 3 INDEPENDENT EXPERTS")
    models = []
    for i in range(3):
        m = build_model(cfg)
        if i == 0: 
            m.classifier = torch.nn.Linear(m.feature_len, m.num_classes, bias=True).to(device)
        else:      
            m.classifier = torch.nn.Linear(m.feature_len, m.num_classes, bias=False).to(device)
            
        ckpt = torch.load(os.path.join(experts_dir, f"expert_{i}.pth"), map_location=device)
        m.load_state_dict(ckpt['state_dict'])
        m.eval()
        models.append(m)

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
            for i in range(3):
                logits, _ = models[i](images)
                all_logits[i].append(logits)
            all_labels.append(labels)

    all_logits = [torch.cat(l, dim=0) for l in all_logits]
    all_labels = torch.cat(all_labels, dim=0).to(device)
    labels_np = all_labels.cpu().numpy()
    masks = get_group_masks(labels_np, cfg.cls_num_list)

    print("\n" + "="*80)
    print("DIAGNOSTIC 1: EXPERT LOGIT HEALTH")
    print("="*80)
    for i in range(3):
        max_logits = all_logits[i].max(dim=1)[0]
        print(f"Expert {i}: Mean Max Logit = {max_logits.mean().item():.4f} | Saturated (>15.0) = {(max_logits > 15.0).sum().item()}/{len(labels_np)}")

    print("\n" + "="*80)
    print("DIAGNOSTIC 2: STRUCTURAL DIVERSITY & ORACLE ACCURACY")
    print("="*80)
    
    T = 3.0
    adj_probs = [
        F.softmax(all_logits[0] / T, dim=1),
        F.softmax((all_logits[1] - log_prior) / T, dim=1),
        F.softmax((all_logits[2] + log_prior) / T, dim=1)
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
        
    # ORACLE ACCURACY: If we pick the correct expert for every sample
    oracle_correct = 0
    for i in range(len(labels_np)):
        # An expert is "correct" if its top prediction matches the label
        if any(adj_probs[e][i].argmax().item() == labels_np[i] for e in range(3)):
            oracle_correct += 1
    print(f"\n[INFO] Oracle Ensemble Accuracy (Max Possible): {oracle_correct / len(labels_np) * 100:.2f}%")
    print("[INFO] If Oracle Acc is < 45%, the experts are too correlated and CRISP cannot work.")

    print("\n" + "="*80)
    print("DIAGNOSTIC 3: INDIVIDUAL EXPERT AURC (Baseline)")
    print("="*80)
    p_mix_np_list = [adj_probs[i].cpu().numpy() for i in range(3)]
    group_ids = define_groups(cfg.cls_num_list)
    
    for i in range(3):
        params = tune_plugin_bal(p_mix_np_list[i], labels_np, group_ids)
        metrics = compute_paper_metrics(p_mix_np_list[i], labels_np, group_ids, params['alpha'], params['mu'])
        print(f"Expert {i} (Alone) AURCbal: {metrics['AURCbal']:.4f}")

    print("\n" + "="*80)
    print("DIAGNOSTIC 4: GATE ROUTING & MIXTURE")
    print("="*80)
    phi = compute_gate_features(all_logits, adj_probs)
    print(f"Gate Feature Mean Std Dev: {phi.std(dim=0).mean().item():.6f}")
    
    if gate_ckpt and os.path.exists(gate_ckpt):
        from imbalanceddl.strategy._gate_trainer import GateMLP
        gate = GateMLP(input_dim=24, hidden1=256, hidden2=128, num_experts=3).to(device)
        gate.load_state_dict(torch.load(gate_ckpt, map_location=device)['gate_state_dict'])
        gate.eval()
        with torch.no_grad():
            weights = F.softmax(gate(phi), dim=1)
            avg_weights = weights.mean(dim=0)
            print(f"Avg Gate Weights: CE={avg_weights[0]:.4f}, LA={avg_weights[1]:.4f}, BS={avg_weights[2]:.4f}")
            
            k = 2
            topk_weights, topk_indices = torch.topk(weights, k, dim=1)
            topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)
            p_mix = torch.zeros_like(adj_probs[0])
            for i in range(k):
                idx = topk_indices[:, i]
                w = topk_weights[:, i].unsqueeze(1)
                expert_probs = torch.stack(adj_probs, dim=1)[torch.arange(len(labels_np)), idx, :]
                p_mix += w * expert_probs
    else:
        print("[INFO] No gate. Using uniform ensemble.")
        p_mix = torch.stack(adj_probs, dim=0).mean(dim=0)

    print("\n" + "="*80)
    print("DIAGNOSTIC 5: FINAL CRISP METRICS")
    print("="*80)
    p_mix_np = p_mix.cpu().numpy()
    params_bal = tune_plugin_bal(p_mix_np, labels_np, group_ids)
    metrics_bal = compute_paper_metrics(p_mix_np, labels_np, group_ids, params_bal['alpha'], params_bal['mu'])
    
    print(f"{'Method':<25} | {'AURCbal':<10} | {'AURCwst':<10} | {'NLL':<10} | {'Brier':<10} | {'tail-ECE':<10}")
    print("-"*80)
    print(f"{'CRISP+Plug-in[Bal]':<25} | {metrics_bal['AURCbal']:<10.4f} | {metrics_bal['AURCwst']:<10.4f} | {metrics_bal['NLL']:<10.4f} | {metrics_bal['Brier']:<10.4f} | {metrics_bal['tail-ECE']:<10.4f}")
    print(f"{'Paper Reference':<25} | 0.253      | 0.302      | 1.18       | 0.403      | 0.088      ")

if __name__ == "__main__":
    main()