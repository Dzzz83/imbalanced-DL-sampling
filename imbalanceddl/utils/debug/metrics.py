import torch
import numpy as np
from imbalanceddl.utils.metrics import shot_acc
from imbalanceddl.utils.plugin_rule import define_groups_2

def compute_chow_aurc(p_tune, labels_tune, p_test, labels_test, group_ids, mode='bal'):
    rho_grid = np.arange(0.0, 1.1, 0.1)
    coverages = []
    risks = []
    for rho in rho_grid:
        confs = np.max(p_tune, axis=1)
        threshold = np.percentile(confs, rho * 100)
        test_confs = np.max(p_test, axis=1)
        accepted = test_confs >= threshold
        coverage = np.mean(accepted)
        preds = np.argmax(p_test, axis=1)
        label_groups = group_ids[labels_test]
        K = len(np.unique(group_ids))
        risks_k = []
        for k in range(K):
            mask = (label_groups == k) & accepted
            if np.sum(mask) == 0:
                risks_k.append(1.0)
            else:
                err = np.sum(preds[mask] != labels_test[mask])
                risks_k.append(err / np.sum(mask))
        risk = np.max(risks_k) if mode == 'worst' else np.mean(risks_k)
        coverages.append(coverage)
        risks.append(risk)
    sort_idx = np.argsort(coverages)
    coverages = np.array(coverages)[sort_idx]
    risks = np.array(risks)[sort_idx]
    if coverages[0] > 0:
        coverages = np.insert(coverages, 0, 0.0)
        risks = np.insert(risks, 0, 1.0) 
    return np.trapezoid(risks, coverages)

def compute_all_metrics(probs, labels, logits=None, cfg=None, train_dataset=None):
    """Helper to compute all calibration and accuracy metrics with LT re-weighting."""
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    
    bal_acc = np.mean([np.mean(preds[labels == c] == c) for c in range(cfg.num_classes) if np.sum(labels == c) > 0]) * 100
    many, med, low = shot_acc(cfg, preds, labels, train_dataset, acc_per_cls=False)
    
    cls_num_list = np.array(cfg.cls_num_list)
    priors = cls_num_list / cls_num_list.sum()
    sample_weights = priors[labels]
    sample_weights = sample_weights / sample_weights.sum()

    true_probs = probs[np.arange(len(labels)), labels]
    nll = -np.sum(sample_weights * np.log(true_probs + 1e-8))
    
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    brier = np.sum(sample_weights * np.sum((probs - one_hot)**2, axis=1))
    
    accs = (preds == labels)
    bin_lowers = np.linspace(0, 1, 16)[:-1]
    bin_uppers = np.linspace(0, 1, 16)[1:]
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accs[in_bin])
            avg_conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_conf_in_bin - acc_in_bin) * prop_in_bin

    group_ids = define_groups_2(cfg.cls_num_list)
    label_groups = group_ids[labels]
    tail_mask = (label_groups == 1)
    tail_conf = confidences[tail_mask]
    tail_correct = accs[tail_mask]
    tail_ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (tail_conf > bin_lower) & (tail_conf <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(tail_correct[in_bin])
            avg_conf_in_bin = np.mean(tail_conf[in_bin])
            tail_ece += np.abs(avg_conf_in_bin - acc_in_bin) * prop_in_bin

    metrics = {
        'bal_acc': bal_acc, 'many': many * 100, 'med': med * 100, 'low': low * 100,
        'nll': nll, 'brier': brier, 'ece': ece, 'tail_ece': tail_ece
    }
    
    if logits is not None:
        max_logits = logits.max(dim=1)[0].numpy()
        metrics['mean_logit'] = np.mean(max_logits)
        metrics['sat_10'] = np.mean(max_logits > 10.0) * 100
        metrics['sat_20'] = np.mean(max_logits > 20.0) * 100
        
    return metrics

def print_uniform_comparison(unif_metrics, unif_nll, unif_tail_ece, unif_brier, unif_bal_aurc, unif_wst_aurc):
    print("\n" + "="*110)
    print("UNIFORM BASELINE COMPARISON: Paper's Method vs My Method")
    print("="*110)
    print(f"{'Metric':<40} | {'Paper Method':<20} | {'My Method':<20} | {'Difference':<15}")
    print("-"*110)
    
    def print_diff(name, paper_val, your_val, is_percentage_point=False):
        diff = your_val - paper_val
        sign = "+" if diff >= 0 else ""
        if is_percentage_point:
            print(f"{name:<40} | {paper_val:<20.4f} | {your_val:<20.4f} | {sign}{diff:.2f}%")
        else:
            print(f"{name:<40} | {paper_val:<20.4f} | {your_val:<20.4f} | {sign}{diff*100:.2f}%")

    print_diff("Uniform Bal AURC (lower is better)", 0.254, unif_bal_aurc)
    print_diff("Uniform Wst AURC (lower is better)", 0.261, unif_wst_aurc)
    print_diff("Uniform NLL (lower is better)", 1.30, unif_nll)
    print_diff("Uniform Brier (lower is better)", 0.442, unif_brier)
    print_diff("Uniform tail-ECE (lower is better)", 0.171, unif_tail_ece)
    print_diff("Uniform Bal Acc (higher is better)", 43.28, unif_metrics['bal_acc'], is_percentage_point=True)
    print_diff("Uniform Many Acc (higher is better)", 71.06, unif_metrics['many'], is_percentage_point=True)
    print_diff("Uniform Med Acc (higher is better)", 42.74, unif_metrics['med'], is_percentage_point=True)
    print_diff("Uniform Low Acc (higher is better)", 11.47, unif_metrics['low'], is_percentage_point=True)
    print("="*110)

def print_method_vs_uniform_comparison(method_metrics, unif_metrics, method_bal_aurc, unif_bal_aurc, method_wst_aurc, unif_wst_aurc):
    print("\n" + "="*110)
    print("MY UNIFORM vs MY METHOD (Baseline = Uniform)")
    print("="*110)
    print(f"{'Metric':<40} | {'My Uniform':<20} | {'My Method':<20} | {'Difference':<15}")
    print("-"*110)
    
    def print_diff(val1, val2, is_pp=False):
        diff = val2 - val1
        sign = "+" if diff >= 0 else ""
        if is_pp:
            return f"{sign}{diff:.2f}%"
        else:
            return f"{sign}{diff*100:.2f}%"

    print(f"{'Bal AURC (lower is better)':<40} | {unif_bal_aurc:<20.4f} | {method_bal_aurc:<20.4f} | {print_diff(unif_bal_aurc, method_bal_aurc):<15}")
    print(f"{'Wst AURC (lower is better)':<40} | {unif_wst_aurc:<20.4f} | {method_wst_aurc:<20.4f} | {print_diff(unif_wst_aurc, method_wst_aurc):<15}")
    print(f"{'NLL (lower is better)':<40} | {unif_metrics['nll']:<20.4f} | {method_metrics['nll']:<20.4f} | {print_diff(unif_metrics['nll'], method_metrics['nll']):<15}")
    print(f"{'Brier (lower is better)':<40} | {unif_metrics['brier']:<20.4f} | {method_metrics['brier']:<20.4f} | {print_diff(unif_metrics['brier'], method_metrics['brier']):<15}")
    print(f"{'tail-ECE (lower is better)':<40} | {unif_metrics['tail_ece']:<20.4f} | {method_metrics['tail_ece']:<20.4f} | {print_diff(unif_metrics['tail_ece'], method_metrics['tail_ece']):<15}")
    print(f"{'Bal Acc (higher is better)':<40} | {unif_metrics['bal_acc']:<20.4f} | {method_metrics['bal_acc']:<20.4f} | {print_diff(unif_metrics['bal_acc'], method_metrics['bal_acc'], is_pp=True):<15}")
    print(f"{'Many Acc (higher is better)':<40} | {unif_metrics['many']:<20.4f} | {method_metrics['many']:<20.4f} | {print_diff(unif_metrics['many'], method_metrics['many'], is_pp=True):<15}")
    print(f"{'Med Acc (higher is better)':<40} | {unif_metrics['med']:<20.4f} | {method_metrics['med']:<20.4f} | {print_diff(unif_metrics['med'], method_metrics['med'], is_pp=True):<15}")
    print(f"{'Low Acc (higher is better)':<40} | {unif_metrics['low']:<20.4f} | {method_metrics['low']:<20.4f} | {print_diff(unif_metrics['low'], method_metrics['low'], is_pp=True):<15}")
    print("="*110)

def print_ce_comparison(ce_metrics, ce_bal_aurc, ce_wst_aurc):
    print("\n" + "="*110)
    print("CE EXPERT COMPARISON: Paper's Method vs My Method")
    print("="*110)
    print(f"{'Metric':<40} | {'Paper Method':<20} | {'My Method':<20} | {'Difference':<15}")
    print("-"*110)
    
    def print_diff(name, paper_val, your_val, is_percentage_point=False):
        diff = your_val - paper_val
        sign = "+" if diff >= 0 else ""
        if is_percentage_point:
            print(f"{name:<40} | {paper_val:<20.4f} | {your_val:<20.4f} | {sign}{diff:.2f}%")
        else:
            print(f"{name:<40} | {paper_val:<20.4f} | {your_val:<20.4f} | {sign}{diff*100:.2f}%")

    # Paper's CE-only metrics from Table 3
    print_diff("CE Bal AURC (lower is better)", 0.297, ce_bal_aurc)
    print_diff("CE Wst AURC (lower is better)", 0.321, ce_wst_aurc)
    print_diff("CE NLL (lower is better)", 1.78, ce_metrics['nll'])
    print_diff("CE Brier (lower is better)", 0.531, ce_metrics['brier'])
    print_diff("CE tail-ECE (lower is better)", 0.520, ce_metrics['tail_ece'])
    print_diff("CE Bal Acc (higher is better)", 38.9, ce_metrics['bal_acc'], is_percentage_point=True)
    print_diff("CE Many Acc (higher is better)", 65.0, ce_metrics['many'], is_percentage_point=True)
    print_diff("CE Med Acc (higher is better)", 37.0, ce_metrics['med'], is_percentage_point=True)
    print_diff("CE Low Acc (higher is better)", 10.0, ce_metrics['low'], is_percentage_point=True)
    print("="*110)


def print_final_method_comparison(method_metrics, method_bal_aurc, method_wst_aurc):
    print("\n" + "="*110)
    print("FINAL METHOD COMPARISON: Paper's Method vs My Method")
    print("="*110)
    print(f"{'Metric':<40} | {'Paper Method':<20} | {'My Method':<20} | {'Difference':<15}")
    print("-"*110)
    
    def print_diff(name, paper_val, your_val, is_percentage_point=False):
        diff = your_val - paper_val
        sign = "+" if diff >= 0 else ""
        if is_percentage_point:
            print(f"{name:<40} | {paper_val:<20.4f} | {your_val:<20.4f} | {sign}{diff:.2f}%")
        else:
            print(f"{name:<40} | {paper_val:<20.4f} | {your_val:<20.4f} | {sign}{diff*100:.2f}%")

    # Paper's CRISP (Method) metrics from Table 2 & 3
    print_diff("Method Bal AURC (lower is better)", 0.233, method_bal_aurc)
    print_diff("Method Wst AURC (lower is better)", 0.248, method_wst_aurc)
    print_diff("Method NLL (lower is better)", 1.18, method_metrics['nll'])
    print_diff("Method Brier (lower is better)", 0.403, method_metrics['brier'])
    print_diff("Method tail-ECE (lower is better)", 0.088, method_metrics['tail_ece'])
    print_diff("Method Bal Acc (higher is better)", 42.33, method_metrics['bal_acc'], is_percentage_point=True)
    print_diff("Method Many Acc (higher is better)", 70.30, method_metrics['many'], is_percentage_point=True)
    print_diff("Method Med Acc (higher is better)", 41.40, method_metrics['med'], is_percentage_point=True)
    print_diff("Method Low Acc (higher is better)", 10.80, method_metrics['low'], is_percentage_point=True)
    print("="*110)