import numpy as np

def define_groups(cls_num_list, num_groups=3):
    cls_num_list = np.array(cls_num_list)
    group_ids = np.zeros(len(cls_num_list), dtype=int)
    group_ids[cls_num_list < 20] = 2
    group_ids[(cls_num_list >= 20) & (cls_num_list <= 100)] = 1
    group_ids[cls_num_list > 100] = 0
    return group_ids

def power_iteration_alpha(p_mix, labels, group_ids, beta, mu, rho, max_iter=20, damp=0.5):
    K = len(beta)
    alpha = np.ones(K) / K * (1.0 - rho)
    for _ in range(max_iter):
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
            mask = (label_groups == k)
            accepted = mask & (~reject)
            total = np.sum(mask)
            if total > 0:
                alpha_new[k] = np.sum(accepted) / total
            else:
                alpha_new[k] = 0.0
        alpha_new = np.clip(alpha_new, 1e-6, 1.0)
        alpha = damp * alpha + (1 - damp) * alpha_new
        alpha = np.clip(alpha, 0.01 * (1.0 - rho), 1.0)
    return alpha

def exponentiated_gradient_alpha(p_mix, labels, group_ids, beta, mu, rho, outer_iter=25, step=1.0):
    K = len(beta)
    alpha = np.ones(K) / K * (1.0 - rho)
    for _ in range(outer_iter):
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
        risks = np.zeros(K)
        for k in range(K):
            mask = (label_groups == k)
            accepted = mask & (~reject)
            if np.sum(accepted) == 0:
                risks[k] = 1.0
            else:
                err = np.sum((preds[accepted] != labels[accepted]))
                risks[k] = err / np.sum(accepted)
        alpha = alpha * np.exp(step * risks)
        alpha = alpha / alpha.sum() * (1.0 - rho)
        alpha = np.clip(alpha, 0.01 * (1.0 - rho), 1.0)
    return alpha

def tune_plugin_bal(p_mix, labels, group_ids):
    K = len(np.unique(group_ids))
    beta = np.ones(K) / K
    mu_delta_grid = [-5, -2, -1, 0, 1, 2, 3, 5, 6, 8, 11, 15, 20]
    best_aurc = 1e9
    best_params = None
    for mu_delta in mu_delta_grid:
        mu = np.zeros(K)
        mu[0] = 0
        if K > 1:
            mu[1:] = mu_delta
        alpha = power_iteration_alpha(p_mix, labels, group_ids, beta, mu, rho=0.5)
        metrics = compute_paper_metrics(
            p_mix, labels, group_ids, alpha, mu, beta,
            tune_alpha_per_threshold=True,
            alpha_tuner=power_iteration_alpha
        )
        if metrics['AURCbal'] < best_aurc:
            best_aurc = metrics['AURCbal']
            best_params = {'alpha': alpha.copy(), 'mu': mu.copy()}
    return best_params

def tune_plugin_worst(p_mix, labels, group_ids):
    K = len(np.unique(group_ids))
    beta = np.ones(K) / K
    mu_delta_grid = [1, 6, 11]
    best_aurc = 1e9
    best_params = None
    for mu_delta in mu_delta_grid:
        mu = np.zeros(K)
        mu[0] = 0
        if K > 1:
            mu[1:] = mu_delta
        alpha = exponentiated_gradient_alpha(p_mix, labels, group_ids, beta, mu, rho=0.5)
        metrics = compute_paper_metrics(
            p_mix, labels, group_ids, alpha, mu, beta,
            tune_alpha_per_threshold=True,
            alpha_tuner=exponentiated_gradient_alpha
        )
        if metrics['AURCwst'] < best_aurc:
            best_aurc = metrics['AURCwst']
            best_params = {'alpha': alpha.copy(), 'mu': mu.copy()}
    return best_params

def compute_paper_metrics(p_mix, labels, group_ids, alpha, mu, beta=None,
                          tune_alpha_per_threshold=False, alpha_tuner=None):
    N, C = p_mix.shape
    K = len(alpha)
    if beta is None:
        beta = np.ones(K) / K

    # NLL, Brier, tail‑ECE
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
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (tail_conf > bin_lower) & (tail_conf <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(tail_correct[in_bin])
            avg_conf_in_bin = np.mean(tail_conf[in_bin])
            ece += np.abs(avg_conf_in_bin - acc_in_bin) * prop_in_bin

    # AURC using paper's exact 11 target rejection rates
    if tune_alpha_per_threshold and alpha_tuner is not None:
        rates = np.arange(0.0, 1.1, 0.1)
        coverages = []
        risk_balanced = []
        risk_worst = []
        for rho in rates:
            if rho == 0.0:
                coverages.append(0.0)
                risk_balanced.append(0.0)
                risk_worst.append(0.0)
                continue
            if rho == 1.0:
                alpha_rho = np.ones(K) * 1.0
                accepted = np.ones(N, dtype=bool)
            else:
                alpha_rho = alpha_tuner(p_mix, labels, group_ids, beta, mu, rho=rho)
                alpha_safe = np.clip(alpha_rho, 1e-6, 1.0)
                W = beta[group_ids] / alpha_safe[group_ids]
                M = mu[group_ids]
                Wp = p_mix * W
                S_max = np.max(Wp, axis=1)
                S_sum = np.sum(p_mix * (W - M), axis=1)
                margin = S_sum - S_max
                c = np.percentile(margin, 100.0 * (1.0 - rho))
                accepted = margin <= c

            alpha_safe = np.clip(alpha_rho, 1e-6, 1.0)
            W = beta[group_ids] / alpha_safe[group_ids]
            M = mu[group_ids]
            Wp = p_mix * W
            preds_rho = np.argmax(Wp, axis=1)
            risks_k = []
            for k in range(K):
                mask = (label_groups == k) & accepted
                if np.sum(mask) == 0:
                    risks_k.append(1.0)
                else:
                    err_k = np.sum((preds_rho[mask] != labels[mask]))
                    risks_k.append(err_k / np.sum(mask))
            risk_bal = np.mean(risks_k)
            risk_wst = np.max(risks_k)
            risk_balanced.append(risk_bal)
            risk_worst.append(risk_wst)
            coverages.append(np.mean(accepted))

        sort_idx = np.argsort(coverages)
        coverages = np.array(coverages)[sort_idx]
        risk_balanced = np.array(risk_balanced)[sort_idx]
        risk_worst = np.array(risk_worst)[sort_idx]
        aurc_bal = np.trapz(risk_balanced, coverages)
        aurc_wst = np.trapz(risk_worst, coverages)
    else:
        aurc_bal = 0.0
        aurc_wst = 0.0

    return {'NLL': nll, 'Brier': brier, 'tail-ECE': ece,
            'AURCbal': aurc_bal, 'AURCwst': aurc_wst}