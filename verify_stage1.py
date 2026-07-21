import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from imbalanceddl.utils.config import get_args
from imbalanceddl.net.network import build_model
from imbalanceddl.dataset.imbalance_dataset import ImbalancedDataset
from imbalanceddl.utils.metrics import shot_acc
from torch.utils.data import DataLoader

def accuracy(probs, labels):
    preds = np.argmax(probs, axis=1)
    return np.mean(preds == labels) * 100

def per_class_acc(probs, labels, num_classes):
    preds = np.argmax(probs, axis=1)
    accs = []
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() > 0:
            accs.append(np.mean(preds[mask] == c))
        else:
            accs.append(0.0)
    return np.array(accs)

def load_expert(cfg, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    # Use bias=True as trained
    has_bias = True 
    model = build_model(cfg)
    model.classifier = nn.Linear(model.feature_len, model.num_classes, bias=has_bias).to(device)
    model = model.to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model

def main():
    cfg = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*80)
    print("STAGE 1 VERIFICATION – CRISP PAPER METRICS")
    print("="*80)

    dataset = ImbalancedDataset(cfg, cfg.dataset, augmentation='none')
    _, val_dataset = dataset.train_val_sets
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

    experts = []
    for i in range(3):
        ckpt_path = os.path.join(cfg.root_model, f"expert_{i}.pth")
        if not os.path.exists(ckpt_path):
            print(f"[ERROR] Checkpoint {ckpt_path} not found!")
            sys.exit(1)
        model = load_expert(cfg, ckpt_path, device)
        experts.append(model)
        print(f"Loaded expert {i} from {ckpt_path}")

    all_logits = [[], [], []]
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels_np = labels.numpy()
            all_labels.extend(labels_np)
            for i, model in enumerate(experts):
                logits, _ = model(images)
                all_logits[i].append(logits.cpu())
                
    all_labels = np.array(all_labels)
    all_logits = [torch.cat(logs, dim=0) for logs in all_logits] 

    cls_num_list = torch.FloatTensor(cfg.cls_num_list)
    log_spc = cls_num_list.log()

    # CE requires subtraction of log_spc for balanced inference
    probs_ce_bal = F.softmax(all_logits[0] - log_spc, dim=1).numpy()
    # LA and BS raw logits are already balanced
    probs_la_bal = F.softmax(all_logits[1], dim=1).numpy()
    probs_bs_bal = F.softmax(all_logits[2], dim=1).numpy()

    results = {}
    results['CE'] = {'bal_acc': accuracy(probs_ce_bal, all_labels), 'many': 0, 'med': 0, 'low': 0}
    results['LA'] = {'bal_acc': accuracy(probs_la_bal, all_labels), 'many': 0, 'med': 0, 'low': 0}
    results['BS'] = {'bal_acc': accuracy(probs_bs_bal, all_labels), 'many': 0, 'med': 0, 'low': 0}

    train_dataset = dataset.train_val_sets[0]
    for name, probs in [('CE', probs_ce_bal), ('LA', probs_la_bal), ('BS', probs_bs_bal)]:
        preds = np.argmax(probs, axis=1)
        many, med, low = shot_acc(cfg, preds, all_labels, train_dataset, acc_per_cls=False)
        results[name]['many'] = many * 100
        results[name]['med'] = med * 100
        results[name]['low'] = low * 100

    logit_stats = []
    for i, logits in enumerate(all_logits):
        max_vals = logits.max(dim=1)[0]
        logit_stats.append({
            'mean_max': max_vals.mean().item(),
            'saturated': (max_vals > 15.0).sum().item() / len(all_labels) * 100,
        })

    print("\n" + "="*80)
    print("SUMMARY TABLE – STAGE 1 METRICS")
    print("="*80)
    print(f"{'Expert':<6} | {'Bal Acc':<8} | {'Many':<6} | {'Med':<6} | {'Low':<6} | {'Mean Max Logit':<12} | {'Saturated %':<10}")
    print("-"*80)
    for i, name in enumerate(['CE', 'LA', 'BS']):
        r = results[name]
        print(f"{name:<6} | {r['bal_acc']:<8.2f} | {r['many']:<6.2f} | {r['med']:<6.2f} | {r['low']:<6.2f} | {logit_stats[i]['mean_max']:<12.4f} | {logit_stats[i]['saturated']:<10.2f}")

    # Corrected thresholds based on official LA/BALMS papers for CIFAR-100-LT (imb_factor=0.01)
    thresholds = {
        'CE_bal': 38.0, 'LA_bal': 42.0, 'BS_bal': 42.0,
        'CE_tail': 10.0, 'LA_tail': 15.0, 'BS_tail': 15.0,
    }

    ce_bal = results['CE']['bal_acc']
    la_bal = results['LA']['bal_acc']
    bs_bal = results['BS']['bal_acc']
    ce_tail = results['CE']['low']
    la_tail = results['LA']['low']
    bs_tail = results['BS']['low']

    checks = []
    checks.append(('CE Balanced Acc >= 38%', ce_bal >= thresholds['CE_bal']))
    checks.append(('LA Balanced Acc >= 42%', la_bal >= thresholds['LA_bal']))
    checks.append(('BS Balanced Acc >= 42%', bs_bal >= thresholds['BS_bal']))
    checks.append(('CE Tail Acc >= 10%', ce_tail >= thresholds['CE_tail']))
    checks.append(('LA Tail Acc >= 15%', la_tail >= thresholds['LA_tail']))
    checks.append(('BS Tail Acc >= 15%', bs_tail >= thresholds['BS_tail']))
    diversity_ok = (la_tail > ce_tail * 1.2) and (bs_tail > ce_tail * 1.2)
    checks.append(('Diversity (LA,BS tail > 1.2x CE tail)', diversity_ok))
    
    saturated_warning = any(s['saturated'] > 80 for s in logit_stats)

    all_pass = True
    print("\n" + "="*80)
    print("PASS/FAIL CHECKS (vs. Paper Expectations)")
    print("="*80)
    for desc, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        if not passed: all_pass = False
        print(f"{desc:<40} : {status}")
        
    if saturated_warning:
        print("⚠️  Warning: Logits are highly saturated (>80%). Check weight decay.")
    else:
        print("✅ Logits are reasonably calibrated (saturation < 80%).")

    print("\n" + "="*80)
    if all_pass:
        print("✅ STAGE 1 VERDICT: PASS – All metrics meet paper expectations.")
    else:
        print("❌ STAGE 1 VERDICT: FAIL – Some metrics are below expectations.")
    print("="*80)

if __name__ == "__main__":
    main()