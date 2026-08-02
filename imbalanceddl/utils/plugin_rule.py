import numpy as np

def define_groups(cls_num_list, num_groups=3):
    cls_num_list = np.array(cls_num_list)
    group_ids = np.zeros(len(cls_num_list), dtype=int)
    group_ids[cls_num_list < 20] = 2
    group_ids[(cls_num_list >= 20) & (cls_num_list <= 100)] = 1
    group_ids[cls_num_list > 100] = 0
    return group_ids

def define_groups_2(cls_num_list):
    cls_num_list = np.array(cls_num_list)
    group_ids = np.zeros(len(cls_num_list), dtype=int)
    group_ids[cls_num_list <= 20] = 1
    return group_ids

def _get_per_class_weights(group_ids, beta, alpha, mu, num_classes):
    """
    Construct per-class weight matrices W and M of shape (N, C).
    W[:, y] = beta[group_ids[y]] / alpha[group_ids[y]]
    M[:, y] = mu[group_ids[y]]
    """
    # Create a mapping from class index to group index
    class_to_group = group_ids  # Shape (C,)
    
    # Compute per-class scalars
    w_per_class = beta[class_to_group] / np.clip(alpha[class_to_group], 1e-6, 1.0)  # Shape (C,)
    m_per_class = mu[class_to_group]  # Shape (C,)
    
    # No need to broadcast to (N, C) explicitly if we rely on numpy broadcasting later,
    # but for clarity we return the per-class vectors.
    return w_per_class, m_per_class

def power_iteration_alpha(p_mix, labels, group_ids, beta, mu, rho, sample_weights=None, max_iter=20, damp=0.5, kappa=1e-4):
    N, C = p_mix.shape
    K = len(beta)
    
    if sample_weights is None:
        sample_weights = np.ones(N) / N
    total_weight = np.sum(sample_weights)
    
    alpha = np.ones(K) / K * (1.0 - rho)
    label_groups = group_ids[labels]
    
    # Precompute per-class weight vectors
    w_per_class, m_per_class = _get_per_class_weights(group_ids, beta, alpha, mu, C)
    
    for _ in range(max_iter):
        alpha_safe = np.clip(alpha, kappa, 1.0)
        # Update per-class weights with current alpha
        w_per_class = beta[group_ids] / alpha_safe[group_ids]  # Shape (C,)
        
        # Broadcast to (N, C) and multiply
        Wp = p_mix * w_per_class  # Shape (N, C)
        S_max = np.max(Wp, axis=1)  # Shape (N,)
        
        # S_sum = sum_y (beta_[y]/alpha_[y] * p_y(x))
        S_sum = np.sum(p_mix * w_per_class, axis=1)  # Shape (N,)
        
        # mu_p = sum_y (mu_[y] * p_y(x))
        mu_p = np.sum(p_mix * m_per_class, axis=1)  # Shape (N,)
        
        margin = S_sum - S_max - mu_p
        
        if rho >= 1.0:
            reject = np.ones(N, dtype=bool)
        else:
            c = np.percentile(margin, 100.0 * (1.0 - rho))
            reject = margin > c
        
        alpha_new = np.zeros(K)
        for k in range(K):
            mask = (label_groups == k)
            accepted = mask & (~reject)
            alpha_new[k] = np.sum(sample_weights[accepted]) / total_weight
            
        alpha_new = np.clip(alpha_new, kappa, 1.0)
        alpha = damp * alpha + (1 - damp) * alpha_new
        alpha = np.clip(alpha, kappa, 1.0)
        
        # Update m_per_class if alpha changed (though mu is fixed in the inner loop)
        # Actually mu is fixed, only alpha changes, which affects w_per_class
        # m_per_class remains constant across iterations since mu is fixed
    return alpha

def tune_plugin_for_rho(p_mix, labels, group_ids, rho, mode='bal', cls_num_list=None, sample_weights=None):
    K = len(np.unique(group_ids))
    beta_init = np.ones(K) / K
    
    if mode == 'bal':
        mu_delta_grid = [-5, -2, -1, 0, 1, 2, 3, 5, 6, 8, 11, 15, 20]
        best_risk = 1e9
        best_alpha = np.ones(K) / K * (1.0 - rho)
        best_mu = np.zeros(K)
        
        for mu_delta in mu_delta_grid:
            mu = np.zeros(K)
            mu[0] = 0
            if K > 1:
                mu[1:] = mu_delta
            
            alpha = power_iteration_alpha(p_mix, labels, group_ids, beta_init, mu, rho=rho, sample_weights=sample_weights)
            _, risk = evaluate_plugin_for_rho(p_mix, labels, group_ids, alpha, mu, rho, beta_init, mode=mode, sample_weights=sample_weights)
            if risk < best_risk:
                best_risk = risk
                best_alpha = alpha
                best_mu = mu
        return best_alpha, best_mu
    else:
        beta = np.ones(K) / K
        best_risk = 1e9
        best_alpha = np.ones(K) / K * (1.0 - rho)
        best_mu = np.zeros(K)
        
        for t in range(25):
            mu_delta_grid = [1, 6, 11]
            inner_best_risk = 1e9
            inner_alpha, inner_mu = best_alpha, best_mu
            
            for mu_delta in mu_delta_grid:
                mu = np.zeros(K)
                mu[0] = 0
                if K > 1:
                    mu[1:] = mu_delta
                alpha = power_iteration_alpha(p_mix, labels, group_ids, beta, mu, rho=rho, sample_weights=sample_weights)
                _, risk = evaluate_plugin_for_rho(p_mix, labels, group_ids, alpha, mu, rho, beta, mode='worst', sample_weights=sample_weights)
                if risk < inner_best_risk:
                    inner_best_risk = risk
                    inner_alpha, inner_mu = alpha, mu
                    
            _, _, risks_k = evaluate_plugin_for_rho(p_mix, labels, group_ids, inner_alpha, inner_mu, rho, beta, mode='worst', return_risks=True, sample_weights=sample_weights)
            beta = beta * np.exp(1.0 * np.array(risks_k))
            beta = beta / (beta.sum() + 1e-12)
            
            if inner_best_risk < best_risk:
                best_risk = inner_best_risk
                best_alpha, best_mu = inner_alpha, inner_mu
                
        return best_alpha, best_mu

def evaluate_plugin_for_rho(p_mix, labels, group_ids, alpha, mu, rho, beta=None, mode='bal', return_risks=False, sample_weights=None):
    N, C = p_mix.shape
    K = len(alpha)
    if beta is None:
        beta = np.ones(K) / K
        
    if sample_weights is None:
        sample_weights = np.ones(N) / N
    total_weight = np.sum(sample_weights)
        
    label_groups = group_ids[labels]
    alpha_safe = np.clip(alpha, 1e-6, 1.0)
    
    # FIX: Construct per-class weight vectors of shape (C,)
    w_per_class = beta[group_ids] / alpha_safe[group_ids]  # Shape (C,)
    m_per_class = mu[group_ids]  # Shape (C,)
    
    # Multiply p_mix (N, C) by per-class weights (C,)
    Wp = p_mix * w_per_class  # Shape (N, C)
    S_max = np.max(Wp, axis=1)  # Shape (N,)
    
    # S_sum = sum_y (beta_[y]/alpha_[y] * p_y(x))
    S_sum = np.sum(p_mix * w_per_class, axis=1)  # Shape (N,)
    
    # mu_p = sum_y (mu_[y] * p_y(x))
    mu_p = np.sum(p_mix * m_per_class, axis=1)  # Shape (N,)
    
    margin = S_sum - S_max - mu_p
    preds = np.argmax(Wp, axis=1)
    
    if rho == 0.0:
        accepted = np.ones(N, dtype=bool)
    elif rho >= 1.0:
        accepted = np.zeros(N, dtype=bool)
    else:
        c = np.percentile(margin, 100.0 * (1.0 - rho))
        accepted = margin <= c  
    
    coverage = np.sum(sample_weights[accepted]) / total_weight
    risks_k = []
    for k in range(K):
        mask = (label_groups == k) & accepted
        group_total_weight = np.sum(sample_weights[mask])
        if group_total_weight == 0:
            risks_k.append(1.0)
        else:
            err_mask = mask & (preds != labels)
            err_weight = np.sum(sample_weights[err_mask])
            risks_k.append(err_weight / group_total_weight)
    
    risk = np.max(risks_k) if mode == 'worst' else np.mean(risks_k)
    
    if return_risks:
        return coverage, risk, risks_k
    return coverage, risk

def compute_aurc_metrics(p_mix_val, labels_val, p_mix_test, labels_test, group_ids, cls_num_list=None, mode='bal'):
    N_test, C = p_mix_test.shape
    K = len(np.unique(group_ids))
    beta = np.ones(K) / K
    rho_grid = np.arange(0.0, 1.1, 0.1)
    
    if cls_num_list is not None:
        cls_num_list = np.array(cls_num_list)
        priors = cls_num_list / cls_num_list.sum()
        sample_weights_val = priors[labels_val]
        sample_weights_val = sample_weights_val / sample_weights_val.sum()
        
        sample_weights_test = priors[labels_test]
        sample_weights_test = sample_weights_test / sample_weights_test.sum()
    else:
        sample_weights_val = None
        sample_weights_test = None
    
    coverages = []
    risks = []
    
    for rho in rho_grid:
        alpha, mu = tune_plugin_for_rho(p_mix_val, labels_val, group_ids, rho, mode=mode, sample_weights=sample_weights_val)
        coverage, risk = evaluate_plugin_for_rho(p_mix_test, labels_test, group_ids, alpha, mu, rho, beta, mode=mode, sample_weights=sample_weights_test)
        coverages.append(coverage)
        risks.append(risk)
    
    sort_idx = np.argsort(coverages)
    coverages = np.array(coverages)[sort_idx]
    risks = np.array(risks)[sort_idx]
    
    if coverages[0] > 0:
        coverages = np.insert(coverages, 0, 0.0)
        risks = np.insert(risks, 0, 1.0)
        
    aurc = np.trapezoid(risks, coverages)
    
    true_probs = p_mix_test[np.arange(N_test), labels_test]
    if cls_num_list is not None:
        priors = cls_num_list / cls_num_list.sum()
        sample_weights = priors[labels_test]
        sample_weights = sample_weights / sample_weights.sum()
    else:
        sample_weights = np.ones(N_test) / N_test
        
    nll = -np.sum(sample_weights * np.log(true_probs + 1e-8))
    
    one_hot = np.zeros_like(p_mix_test)
    one_hot[np.arange(N_test), labels_test] = 1.0
    brier = np.sum(sample_weights * np.sum((p_mix_test - one_hot)**2, axis=1))
    
    confidences = np.max(p_mix_test, axis=1)
    preds = np.argmax(p_mix_test, axis=1)
    correct = (preds == labels_test).astype(int)
    label_groups = group_ids[labels_test]
    tail_mask = (label_groups == 2) | (label_groups == 1)
    tail_conf = confidences[tail_mask]
    tail_correct = correct[tail_mask]
    
    n_bins = 15
    bin_lowers = np.linspace(0, 1, n_bins + 1)[:-1]
    bin_uppers = np.linspace(0, 1, n_bins + 1)[1:]
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (tail_conf > bin_lower) & (tail_conf <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(tail_correct[in_bin])
            avg_conf_in_bin = np.mean(tail_conf[in_bin])
            ece += np.abs(avg_conf_in_bin - acc_in_bin) * prop_in_bin

    return {
        'AURC': aurc,
        'NLL': nll,
        'Brier': brier,
        'tail-ECE': ece
    }