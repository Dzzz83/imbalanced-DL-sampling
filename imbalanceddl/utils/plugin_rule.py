import numpy as np

def define_groups(cls_num_list, num_groups=3):
    """
    Partition classes into K groups based on sample counts.
    Group 0: Head (>100), Group 1: Medium (20-100), Group 2: Tail (<20)
    """
    cls_num_list = np.array(cls_num_list)
    group_ids = np.zeros(len(cls_num_list), dtype=int)
    
    group_ids[cls_num_list < 20] = 2
    group_ids[(cls_num_list >= 20) & (cls_num_list <= 100)] = 1
    group_ids[cls_num_list > 100] = 0
    
    return group_ids

def compute_plugin_metrics(p_mix, labels, group_ids, alpha, mu, c, beta=None):
    """
    Compute predictions, rejections, and metrics for the Plug-in Rule.
    """
    N, C = p_mix.shape
    K = len(alpha)
    if beta is None:
        beta = np.ones(K) / K

    # Map class labels to their group weights
    W = beta[group_ids] / alpha[group_ids]  # Shape: (C,)
    M = mu[group_ids]                       # Shape: (C,)

    # S_max = max_y (beta_[y] / alpha_[y]) * p_y
    Wp = p_mix * W  # Shape: (N, C)
    S_max = np.max(Wp, axis=1)  # Shape: (N,)
    preds = np.argmax(Wp, axis=1)  # Shape: (N,)

    # S_sum = sum_y' ( (beta_[y'] / alpha_[y']) - mu_[y'] ) * p_y'
    S_sum = np.sum(p_mix * (W - M), axis=1)  # Shape: (N,)

    # Reject if S_max < S_sum - c  =>  margin > c
    margin = S_sum - S_max
    reject = margin > c

    # Coverage = P(r(x) = 0)
    coverage = np.mean(~reject)
    
    # Fix mapping: true label group
    label_groups = group_ids[labels]
    
    bal_risk = 0.0
    for k in range(K):
        idx_k = (label_groups == k)
        if np.sum(idx_k & ~reject) == 0:
            risk_k = 0.0
        else:
            # Risk is error rate among non-rejected samples in group k
            err = np.sum((preds != labels) & idx_k & ~reject)
            risk_k = err / np.sum(idx_k & ~reject)
        bal_risk += beta[k] * risk_k
        
    return preds, reject, coverage, bal_risk

def tune_plugin_bal(p_mix, labels, group_ids, target_rejections):
    """
    Algorithm 1 of Narasimhan et al. [26] (Plug-in [Bal])
    Power iteration with 20 iterations and damping 0.5.
    """
    K = len(np.unique(group_ids))
    beta = np.ones(K) / K
    
    mu_delta_grid = [-5, -2, -1, 0, 1, 2, 3, 5, 6, 8, 11, 15, 20]
    
    best_params = {}
    
    for mu_delta in mu_delta_grid:
        mu = np.zeros(K)
        mu[0] = 0
        if K > 1:
            mu[1:] = mu_delta
            
        alpha = np.ones(K) / K
        
        for _ in range(20):
            preds, reject, _, _ = compute_plugin_metrics(p_mix, labels, group_ids, alpha, mu, c=1e9, beta=beta)
            
            label_groups = group_ids[labels]
            alpha_new = np.zeros(K)
            for k in range(K):
                alpha_new[k] = np.mean((~reject) & (label_groups == k))
                
            alpha_new = np.clip(alpha_new, 1e-6, 1.0)
            alpha_new = alpha_new / alpha_new.sum()
            
            # Damping 0.5
            alpha = 0.5 * alpha + 0.5 * alpha_new
            
        W = beta[group_ids] / alpha[group_ids]
        M = mu[group_ids]
        Wp = p_mix * W
        S_max = np.max(Wp, axis=1)
        S_sum = np.sum(p_mix * (W - M), axis=1)
        margin = S_sum - S_max
        
        for rho in target_rejections:
            # FIX: To reject 'rho' fraction of samples, we need the (1 - rho) percentile
            c = np.percentile(margin, 100.0 * (1.0 - rho))
            _, _, cov, risk = compute_plugin_metrics(p_mix, labels, group_ids, alpha, mu, c, beta)
            
            key = (rho, mu_delta)
            if key not in best_params or risk < best_params[key]['risk']:
                best_params[key] = {
                    'alpha': alpha.copy(),
                    'mu': mu.copy(),
                    'c': c,
                    'risk': risk,
                    'coverage': cov
                }
                
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
    """
    Algorithm 2 of Narasimhan et al. [26] (Plug-in [Worst])
    Exponentiated gradient with 25 outer iterations, step size 1.0.
    """
    K = len(np.unique(group_ids))
    beta = np.ones(K) / K
    
    mu_delta_grid = [1, 6, 11]
    
    best_params = {}
    
    for mu_delta in mu_delta_grid:
        mu = np.zeros(K)
        mu[0] = 0
        if K > 1:
            mu[1:] = mu_delta
            
        alpha = np.ones(K) / K
        
        for _ in range(25):
            preds, reject, _, _ = compute_plugin_metrics(p_mix, labels, group_ids, alpha, mu, c=1e9, beta=beta)
            
            label_groups = group_ids[labels]
            
            grad = np.zeros(K)
            for k in range(K):
                idx_k = (label_groups == k) & (~reject)
                if np.sum(idx_k) > 0:
                    err = np.sum(preds[idx_k] != labels[idx_k])
                    grad[k] = err / np.sum(idx_k)
                    
            alpha = alpha * np.exp(1.0 * grad)
            alpha = np.clip(alpha, 1e-6, 1.0)
            alpha = alpha / alpha.sum()
            
        W = beta[group_ids] / alpha[group_ids]
        M = mu[group_ids]
        Wp = p_mix * W
        S_max = np.max(Wp, axis=1)
        S_sum = np.sum(p_mix * (W - M), axis=1)
        margin = S_sum - S_max
        
        for rho in target_rejections:
            # FIX: To reject 'rho' fraction of samples, we need the (1 - rho) percentile
            c = np.percentile(margin, 100.0 * (1.0 - rho))
            _, _, cov, risk = compute_plugin_metrics(p_mix, labels, group_ids, alpha, mu, c, beta)
            
            key = (rho, mu_delta)
            if key not in best_params or risk < best_params[key]['risk']:
                best_params[key] = {
                    'alpha': alpha.copy(),
                    'mu': mu.copy(),
                    'c': c,
                    'risk': risk,
                    'coverage': cov
                }
                
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