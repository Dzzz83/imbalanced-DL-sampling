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

def power_iteration_alpha(p_mix, labels, group_ids, beta, mu, rho, max_iter=20, damp=0.5, kappa=1e-4):
    K = len(beta)
    N = len(labels)
    alpha = np.ones(K) / K * (1.0 - rho)
    label_groups = group_ids[labels]
    
    for _ in range(max_iter):
        alpha_safe = np.clip(alpha, kappa, 1.0)
        W = beta[group_ids] / alpha_safe[group_ids]
        M = mu[group_ids]
        Wp = p_mix * W
        S_max = np.max(Wp, axis=1)
        S_sum = np.sum(p_mix * W, axis=1)
        mu_p = np.sum(p_mix * M, axis=1)
        margin = S_sum - S_max - mu_p
        
        if rho >= 1.0:
            reject = np.ones(N, dtype=bool)
        else:
            c = np.percentile(margin, 100.0 * rho)
            reject = margin < c
        
        alpha_new = np.zeros(K)
        for k in range(K):
            mask = (label_groups == k)
            accepted = mask & (~reject)
            alpha_new[k] = np.sum(accepted) / N
            
        alpha_new = np.clip(alpha_new, kappa, 1.0)
        alpha = damp * alpha + (1 - damp) * alpha_new
        alpha = np.clip(alpha, kappa, 1.0)
    return alpha

def tune_plugin_for_rho(p_mix, labels, group_ids, rho, mode='bal', cls_num_list=None):
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
            
            alpha = power_iteration_alpha(p_mix, labels, group_ids, beta_init, mu, rho=rho)
            _, risk, _ = evaluate_plugin_for_rho(p_mix, labels, group_ids, alpha, mu, rho, beta_init, mode=mode)
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
                alpha = power_iteration_alpha(p_mix, labels, group_ids, beta, mu, rho=rho)
                _, risk, _ = evaluate_plugin_for_rho(p_mix, labels, group_ids, alpha, mu, rho, beta, mode='worst')
                if risk < inner_best_risk:
                    inner_best_risk = risk
                    inner_alpha, inner_mu = alpha, mu
                    
            _, _, risks_k = evaluate_plugin_for_rho(p_mix, labels, group_ids, inner_alpha, inner_mu, rho, beta, mode='worst')
            beta = beta * np.exp(1.0 * np.array(risks_k))
            beta = beta / (beta.sum() + 1e-12)
            
            if inner_best_risk < best_risk:
                best_risk = inner_best_risk
                best_alpha, best_mu = inner_alpha, inner_mu
                
        return best_alpha, best_mu

def evaluate_plugin_for_rho(p_mix, labels, group_ids, alpha, mu, rho, beta=None, mode='bal', return_risks=False):
    N, _ = p_mix.shape
    K = len(alpha)
    if beta is None:
        beta = np.ones(K) / K
        
    label_groups = group_ids[labels]
    alpha_safe = np.clip(alpha, 1e-6, 1.0)
    W = beta[group_ids] / alpha_safe[group_ids]
    M = mu[group_ids]
    Wp = p_mix * W
    S_max = np.max(Wp, axis=1)
    S_sum = np.sum(p_mix * W, axis=1)
    mu_p = np.sum(p_mix * M, axis=1)
    margin = S_sum - S_max - mu_p
    preds = np.argmax(Wp, axis=1)
    
    if rho == 0.0:
        accepted = np.ones(N, dtype=bool)
    elif rho >= 1.0:
        accepted = np.zeros(N, dtype=bool)
    else:
        c = np.percentile(margin, 100.0 * rho)
        accepted = margin >= c
    
    coverage = np.mean(accepted)
    risks_k = []
    for k in range(K):
        mask = (label_groups == k) & accepted
        if np.sum(mask) == 0:
            risks_k.append(1.0)
        else:
            err_k = np.sum((preds[mask] != labels[mask]))
            risks_k.append(err_k / np.sum(mask))
    
    risk = np.max(risks_k) if mode == 'worst' else np.mean(risks_k)
    
    if return_risks:
        return coverage, risk, risks_k
    return coverage, risk

def compute_aurc_metrics(p_mix_val, labels_val, p_mix_test, labels_test, group_ids, cls_num_list=None, mode='bal'):
    N_test, C = p_mix_test.shape
    K = len(np.unique(group_ids))
    beta = np.ones(K) / K
    rho_grid = np.arange(0.0, 1.1, 0.1)
    
    coverages = []
    risks = []
    
    for rho in rho_grid:
        alpha, mu = tune_plugin_for_rho(p_mix_val, labels_val, group_ids, rho, mode=mode)
        coverage, risk = evaluate_plugin_for_rho(p_mix_test, labels_test, group_ids, alpha, mu, rho, beta, mode=mode)
        coverages.append(coverage)
        risks.append(risk)
    
    sort_idx = np.argsort(coverages)
    coverages = np.array(coverages)[sort_idx]
    risks = np.array(risks)[sort_idx]
    
    if coverages[0] > 0:
        coverages = np.insert(coverages, 0, 0.0)
        risks = np.insert(risks, 0, 1.0)
        
    aurc = np.trapz(risks, coverages)
    
    true_probs = p_mix_test[np.arange(N_test), labels_test]
    nll = -np.mean(np.log(true_probs + 1e-8))
    
    one_hot = np.zeros_like(p_mix_test)
    one_hot[np.arange(N_test), labels_test] = 1.0
    brier = np.mean(np.sum((p_mix_test - one_hot)**2, axis=1))
    
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