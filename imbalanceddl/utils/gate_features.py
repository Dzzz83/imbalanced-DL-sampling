import torch
import torch.nn.functional as F
import numpy as np

def compute_gate_features(logits_list):
    """
    Args:
        logits_list: list of 3 tensors, each shape (B, C)
    Returns:
        features: tensor of shape (B, 24)
    """
    # Convert to probabilities
    probs = [F.softmax(logits, dim=1) for logits in logits_list]
    B = probs[0].size(0)
    
    # Per-expert features (7 each)
    per_expert = []
    for p in probs:
        # 1. max prob
        max_prob, _ = p.max(dim=1)
        # 2. entropy
        entropy = -(p * torch.log(p + 1e-8)).sum(dim=1)
        # 3. margin
        top2 = torch.topk(p, 2, dim=1)[0]
        margin = top2[:, 0] - top2[:, 1]
        # 4. top-1 prob (same as max)
        top1 = max_prob
        # 5. top-2 prob
        top2_prob = top2[:, 1]
        # 6. L2 norm of logits
        logits_norm = torch.norm(logits_list[i], dim=1)
        # 7. variance of logits
        logits_var = torch.var(logits_list[i], dim=1)
        per_expert.extend([max_prob, entropy, margin, top1, top2_prob, logits_norm, logits_var])
    # per_expert has 7*3 = 21 features

    # Global features (3)
    # 8. average entropy
    avg_entropy = torch.stack([per_expert[1], per_expert[8], per_expert[15]]).mean(dim=0)
    # 9. variance of max probabilities
    max_probs = torch.stack([per_expert[0], per_expert[7], per_expert[14]])  # indices of max prob for each expert
    var_max = torch.var(max_probs, dim=0)
    # 10. average pairwise KL divergence
    kl_01 = (probs[0] * (torch.log(probs[0] + 1e-8) - torch.log(probs[1] + 1e-8))).sum(dim=1)
    kl_02 = (probs[0] * (torch.log(probs[0] + 1e-8) - torch.log(probs[2] + 1e-8))).sum(dim=1)
    kl_12 = (probs[1] * (torch.log(probs[1] + 1e-8) - torch.log(probs[2] + 1e-8))).sum(dim=1)
    avg_kl = (kl_01 + kl_02 + kl_12) / 3.0

    global_feat = torch.stack([avg_entropy, var_max, avg_kl], dim=1)  # (B, 3)

    # Combine: per_expert is a list of 21 tensors, each (B,)
    per_expert_tensor = torch.stack(per_expert, dim=1)  # (B, 21)
    features = torch.cat([per_expert_tensor, global_feat], dim=1)  # (B, 24)
    return features