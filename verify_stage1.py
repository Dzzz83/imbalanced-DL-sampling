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

# ---------- Helper functions ----------
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
# --------------------------------------

def load_expert(cfg, ckpt_path, device):
    """Load an expert model with the correct bias setting based on checkpoint."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    has_bias = 'classifier.bias' in ckpt['state_dict']
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

    # 1. Load dataset (no augmentation)
    dataset = ImbalancedDataset(cfg, cfg.dataset, augmentation='none')
    _, val_dataset = dataset.train_val_sets
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

    # 2. Load the three experts with auto bias detection
    experts = []
    for i in range(3):
        ckpt_path = os.path.join(cfg.root_model, f"expert_{i}.pth")
        if not os.path.exists(ckpt_path):
            print(f"[ERROR] Checkpoint {ckpt_path} not found!")
            sys.exit(1)
        model = load_expert(cfg, ckpt_path, device)
        experts.append(model)
        print(f"Loaded expert {i} from {ckpt_path}")

    # 3. Extract logits (keep on CPU)
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
    all_logits = [torch.cat(logs, dim=0) for logs in all_logits]  # each [N, C] on CPU

    # 4. Class priors (CPU)
    cls_num_list = torch.FloatTensor(cfg.cls_num_list)
    log_prior = (cls_num_list / cls_num_list.sum()).log()

    # 5. Compute CORRECTED balanced posteriors per paper
    probs_ce_bal = F.softmax(all_logits[0] - log_prior, dim=1).numpy()   # CE: subtract prior
    probs_la_bal = F.softmax(all_logits[1], dim=1).numpy()               # LA: raw (already balanced)
    probs_bs_bal = F.softmax(all_logits[2], dim=1).numpy()               # BS: raw (already balanced)

    # Also compute raw for comparison
    probs_ce_raw = F.softmax(all_logits[0], dim=1).numpy()
    probs_la_raw = F.softmax(all_logits[1], dim=1).numpy()
    probs_bs_raw = F.softmax(all_logits[2], dim=1).numpy()

    # 6. Compute all metrics
    results = {}
    results['CE'] = {
        'bal_acc': accuracy(probs_ce_bal, all_labels),
        'raw_acc': accuracy(probs_ce_raw, all_labels),
        'many': 0, 'med': 0, 'low': 0,
    }
    results['LA'] = {
        'bal_acc': accuracy(probs_la_bal, all_labels),
        'raw_acc': accuracy(probs_la_raw, all_labels),
        'many': 0, 'med': 0, 'low': 0,
    }
    results['BS'] = {
        'bal_acc': accuracy(probs_bs_bal, all_labels),
        'raw_acc': accuracy(probs_bs_raw, all_labels),
        'many': 0, 'med': 0, 'low': 0,
    }

    train_dataset = dataset.train_val_sets[0]
    for name, probs in [('CE', probs_ce_bal), ('LA', probs_la_bal), ('BS', probs_bs_bal)]:
        preds = np.argmax(probs, axis=1)
        many, med, low = shot_acc(cfg, preds, all_labels, train_dataset, acc_per_cls=False)
        results[name]['many'] = many * 100
        results[name]['med'] = med * 100
        results[name]['low'] = low * 100

    # Per-class accuracies (balanced)
    num_classes = cfg.num_classes
    ce_per = per_class_acc(probs_ce_bal, all_labels, num_classes)
    la_per = per_class_acc(probs_la_bal, all_labels, num_classes)
    bs_per = per_class_acc(probs_bs_bal, all_labels, num_classes)

    # Logit stats
    logit_stats = []
    for i, logits in enumerate(all_logits):
        max_vals = logits.max(dim=1)[0]
        logit_stats.append({
            'mean_max': max_vals.mean().item(),
            'mean_logit': logits.mean().item(),
            'std_logit': logits.std().item(),
            'saturated': (max_vals > 15.0).sum().item() / len(all_labels) * 100,
        })

    # Weight norms
    weight_norms = [model.classifier.weight.norm().item() for model in experts]

    # -------- PRINT SUMMARY TABLE --------
    print("\n" + "="*80)
    print("SUMMARY TABLE – STAGE 1 METRICS")
    print("="*80)
    print(f"{'Expert':<6} | {'Bal Acc':<8} | {'Many':<6} | {'Med':<6} | {'Low':<6} | {'Mean Max Logit':<12} | {'Weight Norm':<10}")
    print("-"*80)
    for i, name in enumerate(['CE', 'LA', 'BS']):
        r = results[name]
        print(f"{name:<6} | {r['bal_acc']:<8.2f} | {r['many']:<6.2f} | {r['med']:<6.2f} | {r['low']:<6.2f} | {logit_stats[i]['mean_max']:<12.4f} | {weight_norms[i]:<10.4f}")

    # -------- PASS/FAIL CHECKS --------
    print("\n" + "="*80)
    print("PASS/FAIL CHECKS (vs. Paper Expectations)")
    print("="*80)

    # Expected thresholds (based on literature)
    thresholds = {
        'CE_bal': 42.0,
        'LA_bal': 43.0,
        'BS_bal': 45.0,
        'CE_tail': 20.0,
        'LA_tail': 25.0,
        'BS_tail': 28.0,
    }

    ce_bal = results['CE']['bal_acc']
    la_bal = results['LA']['bal_acc']
    bs_bal = results['BS']['bal_acc']
    ce_tail = results['CE']['low']
    la_tail = results['LA']['low']
    bs_tail = results['BS']['low']

    checks = []
    checks.append(('CE Balanced Acc >= 42%', ce_bal >= thresholds['CE_bal']))
    checks.append(('LA Balanced Acc >= 43%', la_bal >= thresholds['LA_bal']))
    checks.append(('BS Balanced Acc >= 45%', bs_bal >= thresholds['BS_bal']))
    checks.append(('CE Tail Acc >= 20%', ce_tail >= thresholds['CE_tail']))
    checks.append(('LA Tail Acc >= 25%', la_tail >= thresholds['LA_tail']))
    checks.append(('BS Tail Acc >= 28%', bs_tail >= thresholds['BS_tail']))
    # Diversity: LA/BS tail should be > CE tail * 1.5
    diversity_ok = (la_tail > ce_tail * 1.5) and (bs_tail > ce_tail * 1.5)
    checks.append(('Diversity (LA,BS tail > 1.5× CE tail)', diversity_ok))
    # Logit health: no NaN/Inf (already handled during training)
    # Logit saturation warning (not a failure)
    saturated_warning = any(s['saturated'] > 50 for s in logit_stats)

    all_pass = True
    for desc, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_pass = False
        print(f"{desc:<40} : {status}")
    if saturated_warning:
        print("⚠️  Warning: Logits are highly saturated (>50%). Consider temperature scaling in Stage 2.")
    else:
        print("✅ Logits are reasonably calibrated (saturation < 50%).")

    # -------- FINAL VERDICT --------
    print("\n" + "="*80)
    if all_pass:
        print("✅ STAGE 1 VERDICT: PASS – All metrics meet paper expectations.")
        print("   Experts are correctly trained and structurally diverse.")
        print("   Proceed to Stage 2 (gate training) with temperature scaling.")
    else:
        print("❌ STAGE 1 VERDICT: FAIL – Some metrics are below expectations.")
        print("   Consider retraining the underperforming expert(s) with adjusted hyperparameters.")
        # Provide specific advice
        if ce_bal < thresholds['CE_bal']:
            print("   - CE balanced accuracy is low. Check training logs for CE.")
        if la_bal < thresholds['LA_bal']:
            print("   - LA balanced accuracy is low. Check training logs for LA.")
        if bs_bal < thresholds['BS_bal']:
            print("   - BS balanced accuracy is low. Consider retraining with bias=True, LR=0.2, wd=5e-4.")
        if not diversity_ok:
            print("   - Diversity condition not met. Check if BS/LA tail accuracies are sufficiently higher than CE.")
    print("="*80)

if __name__ == "__main__":
    main()