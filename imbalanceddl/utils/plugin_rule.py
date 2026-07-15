import numpy as np

def define_groups(cls_num_list, num_groups=3):
    """
    Partition classes into K groups based on sample counts.
    Group 0: Head (>100), Group 1: Medium (20-100), Group 2: Tail (<20)
    
    Example:
    cls_num_list = [500, 50, 5] (3 classes)
    Returns: [0, 1, 2]  (Class 0 is Head, Class 1 is Medium, Class 2 is Tail)
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
    
    Example Setup:
    - 3 classes (C=3). Group 0 (Head), Group 1 (Medium), Group 2 (Tail).
    - group_ids = [0, 1, 2]
    - alpha = [0.8, 0.15, 0.05] (Model currently answers 80% Head, 5% Tail)
    - mu = [0, 0, 0] (No penalty for this example)
    - beta = [1/3, 1/3, 1/3] (All groups equally important)
    
    Sample Image 1 (Clear Head): p_mix = [0.9, 0.05, 0.05]
    Sample Image 2 (Ambiguous) : p_mix = [0.4, 0.3, 0.3]
    """
    N, C = p_mix.shape
    K = len(alpha)
    if beta is None:
        beta = np.ones(K) / K

    # 1. Create the "Affirmative Action" Multiplier (W) and Penalty (M)
    # W = beta / alpha. If a group has low coverage (small alpha), W is huge.
    # Example W = [1/3/0.8, 1/3/0.15, 1/3/0.05] = [0.416, 2.22, 6.66]
    # The Tail class (Group 2) gets a massive 6.66x boost to its probabilities!
    W = beta[group_ids] / alpha[group_ids]  # Shape: (C,)
    M = mu[group_ids]                       # Shape: (C,)

    # 2. Calculate the Signal (S_max)
    # Multiply probabilities by W, then find the max for each image.
    # Image 1 Wp = [0.9*0.416, 0.05*2.22, 0.05*6.66] = [0.374, 0.111, 0.333]
    # Image 2 Wp = [0.4*0.416, 0.3*2.22, 0.3*6.66]   = [0.166, 0.666, 1.998]
    Wp = p_mix * W  # Shape: (N, C)
    
    # Image 1 S_max = 0.374 (Class 0 wins)
    # Image 2 S_max = 1.998 (Class 2 wins! The boost helped the Tail class win)
    S_max = np.max(Wp, axis=1)  # Shape: (N,)
    preds = np.argmax(Wp, axis=1)  # Shape: (N,)

    # 3. Calculate the Noise (S_sum)
    # Sum of all boosted probabilities minus the mu penalty.
    # Image 1 S_sum = 0.374 + 0.111 + 0.333 = 0.818
    # Image 2 S_sum = 0.166 + 0.666 + 1.998 = 2.830
    S_sum = np.sum(p_mix * (W - M), axis=1)  # Shape: (N,)

    # 4. The Rejection Rule: Reject if Signal < Noise - Cost
    # Margin = Noise - Signal.
    # Image 1 margin = 0.818 - 0.374 = 0.444
    # Image 2 margin = 2.830 - 1.998 = 0.832 (Much noisier, harder to predict)
    margin = S_sum - S_max
    
    # If c = 0.5: Image 1 (0.444 < 0.5) -> Keep. Image 2 (0.832 > 0.5) -> Reject!
    reject = margin > c

    # Coverage = P(r(x) = 0) -> Percentage of images not rejected
    coverage = np.mean(~reject)
    
    # 5. Calculate Balanced Risk
    # Map true labels to their groups (0=Head, 1=Medium, 2=Tail)
    # E.g., if labels = [0, 99, 5], label_groups becomes [0, 2, 1]
    label_groups = group_ids[labels]
    
    bal_risk = 0.0
    # Loop through each group: k=0 (Head), k=1 (Medium), k=2 (Tail)
    for k in range(K):
        # Create a boolean mask: True for images that belong to group k, False otherwise
        idx_k = (label_groups == k)
        
        # Check if the model answered ANY images from this group.
        # idx_k & ~reject means "belongs to group k" AND "was NOT rejected"
        if np.sum(idx_k & ~reject) == 0:
            # Safety check: If the model rejected 100% of this group, 
            # we can't divide by zero, so error rate is set to 0.0
            risk_k = 0.0
        else:
            # Count the mistakes: 
            # (preds != labels) -> Model predicted the wrong class
            # & idx_k           -> AND the image belongs to group k
            # & ~reject         -> AND the model chose to answer it (not rejected)
            err = np.sum((preds != labels) & idx_k & ~reject)
            
            # Calculate the error rate for this specific group:
            # Mistakes / Total number of answered images in this group
            risk_k = err / np.sum(idx_k & ~reject)
            
        # Weight risk equally across all groups (beta = [1/3, 1/3, 1/3])
        # This forces Head, Medium, and Tail groups to contribute exactly 33.3% 
        # to the final score, preventing Head classes from dominating the metric.
        bal_risk += beta[k] * risk_k
        
    return preds, reject, coverage, bal_risk

def tune_plugin_bal(p_mix, labels, group_ids, target_rejections):
    """
    Algorithm 1 of Narasimhan et al. [26] (Plug-in [Bal])
    Power iteration with 20 iterations and damping 0.5.
    
    Logic: Alpha depends on what we reject, but what we reject depends on Alpha.
    We solve this chicken-and-egg problem by looping 20 times until Alpha stabilizes.
    """
    K = len(np.unique(group_ids))
    beta = np.ones(K) / K
    
    # Try different penalty values to see which yields the lowest risk
    mu_delta_grid = [-5, -2, -1, 0, 1, 2, 3, 5, 6, 8, 11, 15, 20]
    
    best_params = {}
    
    for mu_delta in mu_delta_grid:
        mu = np.zeros(K)
        mu[0] = 0
        if K > 1:
            mu[1:] = mu_delta
            
        # Start with equal coverage (33% for each group)
        alpha = np.ones(K) / K
        
        for _ in range(20):
            # Evaluate predictions with NO rejection (c = 1e9 means reject nothing)
            # We just want to see the model's natural behavior to update Alpha
            preds, reject, _, _ = compute_plugin_metrics(p_mix, labels, group_ids, alpha, mu, c=1e9, beta=beta)
            
            # Measure the ACTUAL natural coverage of each group
            label_groups = group_ids[labels]
            alpha_new = np.zeros(K)
            for k in range(K):
                # How many non-rejected samples belong to group k?
                alpha_new[k] = np.mean((~reject) & (label_groups == k))
                
            # Clip to avoid division by zero later
            alpha_new = np.clip(alpha_new, 1e-6, 1.0)
            alpha_new = alpha_new / alpha_new.sum()
            
            # Damping 0.5: Mix old and new to prevent bouncing wildly back and forth
            # alpha = 0.5 * previous_alpha + 0.5 * measured_alpha
            alpha = 0.5 * alpha + 0.5 * alpha_new
            
        # After Alpha is stable, calculate the margins for all images
        W = beta[group_ids] / alpha[group_ids]
        M = mu[group_ids]
        Wp = p_mix * W
        S_max = np.max(Wp, axis=1)
        S_sum = np.sum(p_mix * (W - M), axis=1)
        margin = S_sum - S_max
        
        # Now find the threshold 'c' that achieves the target rejection rates
        for rho in target_rejections:
            # If we want to reject 60% (rho=0.6), we need the 40th percentile of margins.
            # All images with a margin below this 40th percentile will be rejected.
            c = np.percentile(margin, 100.0 * (1.0 - rho))
            _, _, cov, risk = compute_plugin_metrics(p_mix, labels, group_ids, alpha, mu, c, beta)
            
            key = (rho, mu_delta)
            # Save the best parameters that minimize Balanced Risk for this rejection rate
            if key not in best_params or risk < best_params[key]['risk']:
                best_params[key] = {
                    'alpha': alpha.copy(),
                    'mu': mu.copy(),
                    'c': c,
                    'risk': risk,
                    'coverage': cov
                }
                
    # Filter the grid to find the absolute best mu_delta for each rejection rate
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
    
    Logic: Instead of balancing average risk, this minimizes the WORST group's risk.
    It uses exponentiated gradient descent to update Alpha.
    """
    K = len(np.unique(group_ids))
    beta = np.ones(K) / K
    
    # Smaller grid for the inner mu tuning
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
            
            # Calculate the gradient (error rate) for each group
            grad = np.zeros(K)
            for k in range(K):
                idx_k = (label_groups == k) & (~reject)
                if np.sum(idx_k) > 0:
                    err = np.sum(preds[idx_k] != labels[idx_k])
                    # Gradient is the error rate for this group
                    grad[k] = err / np.sum(idx_k)
                    
            # Exponentiated gradient update
            # If a group has high error (bad), grad is high. exp(grad) is high.
            # alpha increases, which INCREASES the boost W = beta/alpha for that group.
            # Wait, if alpha increases, beta/alpha DECREASES. 
            # Actually, in Narasimhan's dual formulation, increasing the dual variable 
            # penalizes the noise side, making it harder to reject that group.
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