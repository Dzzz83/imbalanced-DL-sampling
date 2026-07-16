import numpy as np

def define_groups(cls_num_list, num_groups=3):
    cls_num_list = np.array(cls_num_list)
    group_ids = np.zeros(len(cls_num_list), dtype=int)
    group_ids[cls_num_list < 20] = 2
    group_ids[(cls_num_list >= 20) & (cls_num_list <= 100)] = 1
    group_ids[cls_num_list > 100] = 0
    return group_ids

def compute_plugin_metrics(p_mix, labels, group_ids, alpha, mu, c, beta=None):
    N, C = p_mix.shape
    K = len(alpha)
    if beta is None:
        beta = np.ones(K) / K

    alpha_safe = np.clip(alpha, 1e-6, 1.0)
    
    W = beta[group_ids] / alpha_safe[group_ids]
    M = mu[group_ids]

    Wp = p_mix * W
    S_max = np.max(Wp, axis=1)
    preds = np.argmax(Wp, axis=1)

    S_sum = np.sum(p_mix * (W - M), axis=1)
    margin = S_sum - S_max
    
    reject = margin > c
    coverage = np.mean(~reject)
    
    label_groups = group_ids[labels]
    
    risks = np.zeros(K)
    for k in range(K):
        idx_k = (label_groups == k)
        accepted_k = idx_k & ~reject
        if np.sum(accepted_k) == 0:
            risks[k] = 1.0
        else:
            err = np.sum((preds != labels) & accepted_k)
            risks[k] = err / np.sum(accepted_k)
            
    bal_risk = np.mean(risks)
    wst_risk = np.max(risks)
        
    return preds, reject, coverage, bal_risk, wst_risk

def tune_plugin_bal(p_mix, labels, group_ids, target_rejections):
    K = len(np.unique(group_ids))
    beta = np.ones(K) / K
    mu_delta_grid = [-5, -2, -1, 0, 1, 2, 3, 5, 6, 8, 11, 15, 20]
    best_params = {}
    
    for mu_delta in mu_delta_grid:
        mu = np.zeros(K)
        mu[0] = 0
        if K > 1:
            mu[1:] = mu_delta
            
        for rho in target_rejections:
            alpha = np.ones(K) / K * (1.0 - rho)
            
            for _ in range(20):
                alpha_safe = np.clip(alpha, 1e-6, 1.0)
                W = beta[group_ids] / alpha_safe[group_ids]
                M = mu[group_ids]
                Wp = p_mix * W
                S_max = np.max(Wp, axis=1)
                S_sum = np.sum(p_mix * (W - M), axis=1)
                margin = S_sum - S_max
                
                c = np.percentile(margin, 100.0 * (1.0 - rho))
                
                reject = margin > c
                preds = np.argmax(Wp, axis=1)
                
                label_groups = group_ids[labels]
                alpha_new = np.zeros(K)
                for k in range(K):
                    alpha_new[k] = np.mean((~reject) & (label_groups == k))
                alpha_new = np.clip(alpha_new, 1e-6, 1.0)
                alpha = 0.5 * alpha + 0.5 * alpha_new
                # FIX: Clamp alpha to prevent W from exploding and causing a feedback loop
                alpha = np.clip(alpha, 0.01 * (1.0 - rho), 1.0)
                
            _, _, cov, risk, _ = compute_plugin_metrics(p_mix, labels, group_ids, alpha, mu, c, beta)
            
            key = (rho, mu_delta)
            if key not in best_params or risk < best_params[key]['risk']:
                best_params[key] = {'alpha': alpha.copy(), 'mu': mu.copy(), 'c': c, 'risk': risk, 'coverage': cov}
                
    final_params = {}
    for rho in target_rejections:
        best_risk = 1e9
        best_p = None
        for mu_delta in mu_delta_grid:
            p = best_params.get((rho, mu_delta))
            if p and p['risk'] < best_risk:
                best_risk = p['risk']
                best_p = p
        if best_p:
            final_params[rho] = best_p
    return final_params

def tune_plugin_worst(p_mix, labels, group_ids, target_rejections):
    K = len(np.unique(group_ids))
    beta = np.ones(K) / K
    mu_delta_grid = [1, 6, 11]
    best_params = {}
    
    for mu_delta in mu_delta_grid:
        mu = np.zeros(K)
        mu[0] = 0
        if K > 1:
            mu[1:] = mu_delta
            
        for rho in target_rejections:
            alpha = np.ones(K) / K * (1.0 - rho)
            
            for _ in range(25):
                alpha_safe = np.clip(alpha, 1e-6, 1.0)
                W = beta[group_ids] / alpha_safe[group_ids]
                M = mu[group_ids]
                Wp = p_mix * W
                S_max = np.max(Wp, axis=1)
                S_sum = np.sum(p_mix * (W - M), axis=1)
                margin = S_sum - S_max
                
                c = np.percentile(margin, 100.0 * (1.0 - rho))
                
                reject = margin > c
                preds = np.argmax(Wp, axis=1)
                
                label_groups = group_ids[labels]
                grad = np.zeros(K)
                for k in range(K):
                    idx_k = (label_groups == k) & (~reject)
                    if np.sum(idx_k) > 0:
                        err = np.sum(preds[idx_k] != labels[idx_k])
                        grad[k] = err / np.sum(idx_k)
                    else:
                        grad[k] = 1.0
                        
                alpha = alpha * np.exp(1.0 * grad)
                alpha = np.clip(alpha, 1e-6, 1.0)
                alpha = alpha / alpha.sum() * (1.0 - rho)
                # FIX: Clamp alpha to prevent W from exploding
                alpha = np.clip(alpha, 0.01 * (1.0 - rho), 1.0)
                
            _, _, cov, _, wst_risk = compute_plugin_metrics(p_mix, labels, group_ids, alpha, mu, c, beta)
            
            key = (rho, mu_delta)
            if key not in best_params or wst_risk < best_params[key]['risk']:
                best_params[key] = {'alpha': alpha.copy(), 'mu': mu.copy(), 'c': c, 'risk': wst_risk, 'coverage': cov}
                
    final_params = {}
    for rho in target_rejections:
        best_risk = 1e9
        best_p = None
        for mu_delta in mu_delta_grid:
            p = best_params.get((rho, mu_delta))
            if p and p['risk'] < best_risk:
                best_risk = p['risk']
                best_p = p
        if best_p:
            final_params[rho] = best_p
    return final_params

def compute_paper_metrics(p_mix, labels, group_ids, alpha, mu, beta=None):
    N, C = p_mix.shape
    K = len(alpha)
    if beta is None:
        beta = np.ones(K) / K

    true_probs = p_mix[np.arange(N), labels]
    nll = -np.mean(np.log(true_probs + 1e-8))

    one_hot = np.zeros_like(p_mix)
    one_hot[np.arange(N), labels] = 1.0
    brier = np.mean(np.sum((p_mix - one_hot)**2, axis=1))

    confidences = np.max(p_mix, axis=1)
    preds = np.argmax(p_mix, axis=1)
    correct = (preds == labels).astype(int)
    label_groups = group_ids[labels]
    tail_mask = (label_groups == 2)
    
    tail_conf = confidences[tail_mask]
    tail_correct = correct[tail_mask]
    n_bins = 15
    bin_lowers = np.linspace(0, 1, n_bins + 1)[:-1]
    bin_uppers = np.linspace(0, 1, n_bins + 1)[1:]
    ece = 0.0
    n_tail = len(tail_conf)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (tail_conf > bin_lower) & (tail_conf <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(tail_correct[in_bin])
            avg_conf_in_bin = np.mean(tail_conf[in_bin])
            ece += np.abs(avg_conf_in_bin - acc_in_bin) * prop_in_bin

    alpha_safe = np.clip(alpha, 1e-6, 1.0)
    W = beta[group_ids] / alpha_safe[group_ids]
    M = mu[group_ids]
    Wp = p_mix * W
    S_max = np.max(Wp, axis=1)
    S_sum = np.sum(p_mix * (W - M), axis=1)
    margin = S_sum - S_max
    
    sorted_idx = np.argsort(margin)
    sorted_labels = labels[sorted_idx]
    sorted_preds = np.argmax(Wp[sorted_idx], axis=1)
    sorted_label_groups = label_groups[sorted_idx]
    err = (sorted_preds != sorted_labels).astype(float)
    
    cum_err = np.zeros((N, K))
    cum_count = np.zeros((N, K))
    for k in range(K):
        mask = (sorted_label_groups == k).astype(float)
        cum_err[:, k] = np.cumsum(err * mask)
        cum_count[:, k] = np.cumsum(mask)
        
    risk = np.zeros((N, K))
    for k in range(K):
        valid = cum_count[:, k] > 0
        risk[valid, k] = cum_err[valid, k] / cum_count[valid, k]
        # FIX: Use NaN for missing groups so they don't artificially inflate the average
        risk[~valid, k] = np.nan 
        
    # FIX: Use nanmean and nanmax to ignore missing groups
    with np.errstate(invalid='ignore'):
        bal_risks = np.nanmean(risk, axis=1)
        wst_risks = np.nanmax(risk, axis=1)
        
    # FIX: Prepend coverage 0 with risk 0 for proper AURC integration
    coverages = np.arange(1, N + 1) / N
    coverages = np.insert(coverages, 0, 0.0)
    bal_risks = np.insert(bal_risks, 0, 0.0)
    wst_risks = np.insert(wst_risks, 0, 0.0)
    
    # Fill any NaNs that might occur at the very first index if all groups are missing
    bal_risks = np.nan_to_num(bal_risks, nan=0.0)
    wst_risks = np.nan_to_num(wst_risks, nan=0.0)
    
    aurc_bal = np.trapz(bal_risks, coverages)
    aurc_wst = np.trapz(wst_risks, coverages)
    
    return {'NLL': nll, 'Brier': brier, 'tail-ECE': ece, 'AURCbal': aurc_bal, 'AURCwst': aurc_wst}