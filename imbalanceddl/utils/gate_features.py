import torch
import torch.nn.functional as F
from .debug_logger import get_debug_logger

def compute_gate_features(logits_list, probs_list=None, debug=False):
    """
    Computes the exact 7E+3 feature vector from App. F of the CRISP paper.
    """
    if probs_list is None:
        probs_list = [F.softmax(logits, dim=1) for logits in logits_list]
        
    probs = probs_list
    E = len(probs)
    B = probs[0].size(0)
    mean_prob = torch.stack(probs, dim=0).mean(dim=0) # B, C

    # Per-expert features (7 each)
    per_expert = []
    for i, p in enumerate(probs):
        # 1. predictive entropy
        entropy = -(p * torch.log(p + 1e-8)).sum(dim=1)
        # 2. maximum confidence
        max_prob, _ = p.max(dim=1)
        # 3. top-1/top-2 margin
        top2 = torch.topk(p, 2, dim=1)[0]
        margin = top2[:, 0] - top2[:, 1]
        # 4. top-k cumulative probability mass (using k=5 as standard for 100 classes)
        topk_mass = torch.topk(p, 5, dim=1)[0].sum(dim=1)
        # 5. tail residual
        tail_residual = 1.0 - topk_mass
        
        # 6. cosine similarity to uniform expert mean
        p_norm = F.normalize(p, dim=1)
        mean_norm = F.normalize(mean_prob, dim=1)
        cos_sim = (p_norm * mean_norm).sum(dim=1)
        
        # 7. KL divergence to expert mean
        kl = (p * (torch.log(p + 1e-8) - torch.log(mean_prob + 1e-8))).sum(dim=1)
        
        # Maintain exact paper order
        per_expert.extend([entropy, max_prob, margin, topk_mass, tail_residual, cos_sim, kl])

    # Global features (3)
    # 1. mean entropy
    mean_entropy = torch.stack([per_expert[0], per_expert[7], per_expert[14]]).mean(dim=0)
    
    # 2. class-wise posterior variance: L^{-1} sum_y Var_e[p_e(y|x)]
    stacked_probs = torch.stack(probs, dim=0) # E, B, C
    class_var = torch.var(stacked_probs, dim=0).mean(dim=1) # B
    
    # 3. confidence dispersion: Var_e[p_e^max(x)]
    max_probs = torch.stack([per_expert[1], per_expert[8], per_expert[15]], dim=0) # E, B
    conf_disp = torch.var(max_probs, dim=0) # B
    
    global_feat = torch.stack([mean_entropy, class_var, conf_disp], dim=1)  # (B, 3)
    per_expert_tensor = torch.stack(per_expert, dim=1)  # (B, 21)
    features = torch.cat([per_expert_tensor, global_feat], dim=1)  # (B, 24)

    return features