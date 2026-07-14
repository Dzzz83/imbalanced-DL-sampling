import torch
import torch.nn.functional as F
from .debug_logger import get_debug_logger

def compute_gate_features(logits_list, debug=False):
    """
    Args:
        logits_list: list of 3 tensors, each shape (B, C)
    Returns:
        features: tensor of shape (B, 24)
    """
    if debug:
        logger = get_debug_logger(debug=True)
        logger.debug("Computing gate features...")
        
    probs = [F.softmax(logits, dim=1) for logits in logits_list]
    B = probs[0].size(0)
    mean_prob = torch.stack(probs, dim=0).mean(dim=0) # B, C

    # Per-expert features (7 each)
    per_expert = []
    for i, p in enumerate(probs):
        max_prob, _ = p.max(dim=1)
        entropy = -(p * torch.log(p + 1e-8)).sum(dim=1)
        top2 = torch.topk(p, 2, dim=1)[0]
        margin = top2[:, 0] - top2[:, 1]
        top2_prob = top2[:, 1]
        logits_norm = torch.norm(logits_list[i], dim=1)
        logits_var = torch.var(logits_list[i], dim=1)
        
        # Fix duplicate feature: use KL divergence to mean instead of top1
        kl_to_mean = (p * (torch.log(p + 1e-8) - torch.log(mean_prob + 1e-8))).sum(dim=1)
        
        per_expert.extend([max_prob, entropy, margin, top2_prob, logits_norm, logits_var, kl_to_mean])

    # Global features (3)
    avg_entropy = torch.stack([per_expert[1], per_expert[8], per_expert[15]]).mean(dim=0)
    max_probs = torch.stack([per_expert[0], per_expert[7], per_expert[14]])
    var_max = torch.var(max_probs, dim=0)
    
    kl_01 = (probs[0] * (torch.log(probs[0] + 1e-8) - torch.log(probs[1] + 1e-8))).sum(dim=1)
    kl_02 = (probs[0] * (torch.log(probs[0] + 1e-8) - torch.log(probs[2] + 1e-8))).sum(dim=1)
    kl_12 = (probs[1] * (torch.log(probs[1] + 1e-8) - torch.log(probs[2] + 1e-8))).sum(dim=1)
    avg_kl = (kl_01 + kl_02 + kl_12) / 3.0

    global_feat = torch.stack([avg_entropy, var_max, avg_kl], dim=1)  # (B, 3)

    per_expert_tensor = torch.stack(per_expert, dim=1)  # (B, 21)
    features = torch.cat([per_expert_tensor, global_feat], dim=1)  # (B, 24)

    return features