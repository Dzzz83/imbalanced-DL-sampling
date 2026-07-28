import torch
import torch.nn.functional as F

def compute_gate_features(logits_list, probs_list=None, top_k_mass=5):
    """
    Computes the exact 7E+3 feature vector from App. F of the CRISP paper.
    """
    if probs_list is None:
        probs_list = [F.softmax(logits, dim=1) for logits in logits_list]
        
    E = len(probs_list)
    mean_prob = torch.stack(probs_list, dim=0).mean(dim=0) # (B, C)

    per_expert_feats = []
    for p in probs_list:
        # 1. predictive entropy
        entropy = -(p * torch.log(p + 1e-8)).sum(dim=1)
        # 2. maximum confidence
        max_prob, _ = p.max(dim=1)
        # 3. top-1/top-2 margin
        top2 = torch.topk(p, 2, dim=1)[0]
        margin = top2[:, 0] - top2[:, 1]
        # 4. top-k cumulative probability mass
        topk_mass = torch.topk(p, top_k_mass, dim=1)[0].sum(dim=1)
        # 5. tail residual
        tail_residual = 1.0 - topk_mass
        # 6. cosine similarity to uniform expert mean
        p_norm = F.normalize(p, p=2, dim=1)
        mean_norm = F.normalize(mean_prob, p=2, dim=1)
        cos_sim = (p_norm * mean_norm).sum(dim=1)
        # 7. KL divergence to expert mean
        kl = (p * (torch.log(p + 1e-8) - torch.log(mean_prob + 1e-8))).sum(dim=1)
        
        # Stack features for this expert: shape (7, B)
        expert_feats = torch.stack([entropy, max_prob, margin, topk_mass, tail_residual, cos_sim, kl], dim=0)
        per_expert_feats.append(expert_feats)
    
    per_expert_tensor = torch.stack(per_expert_feats, dim=0).permute(2, 0, 1).reshape(per_expert_feats[0].shape[1], -1)

    # Global features (3)
    entropies = torch.stack([f[0] for f in per_expert_feats], dim=0) # (E, B)
    mean_entropy = entropies.mean(dim=0) # (B,)
    
    stacked_probs = torch.stack(probs_list, dim=0) # (E, B, C)
    class_var = torch.var(stacked_probs, dim=0, unbiased=False).mean(dim=1) # (B,)
    
    max_probs = torch.stack([f[1] for f in per_expert_feats], dim=0) # (E, B)
    conf_disp = torch.var(max_probs, dim=0, unbiased=False) # (B,)
    
    global_feat = torch.stack([mean_entropy, class_var, conf_disp], dim=1)  # (B, 3)
    
    # Final concatenation: (B, 7E) + (B, 3) -> (B, 7E+3)
    features = torch.cat([per_expert_tensor, global_feat], dim=1)

    # FIX: Z-score normalize the features across the batch dimension
    # This prevents the MLP from ignoring the tiny variance features (0.001) 
    # and saturating on the large entropy features (0.7)
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True) + 1e-8
    features = (features - mean) / std

    return features