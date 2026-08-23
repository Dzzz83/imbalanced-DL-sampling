"""Shared gate-input feature construction for MoE expert routing.

This module is the **single source of truth** for what the gate consumes and
how expert logits are calibrated. It is imported by BOTH the training path
(``imbalanceddl.strategy._gate_trainer``) and the evaluation / diagnostic path
(``imbalanceddl.utils.debug.models``) so the gate is always trained and
evaluated on the exact same representation.

Why probability space instead of raw logits
-------------------------------------------
Raw logit magnitude is anti-correlated with "is this expert correct" in the
overconfident-but-wrong regime (a wrong CE/LA expert can have a *larger* max
logit than a right BS/LA expert). Routing on magnitude therefore degenerates
to naive peak-detection. Routing on bias-adjusted, temperature-scaled
probabilities, plus explicit confidence / margin / entropy / agreement
features, lets the gate detect "sharp one-hot but disagrees with the others"
and route to the trustworthy expert.

Feature layout (num_experts=3, num_classes=100 -> 312 dims)
-----------------------------------------------------------
  0-99   : CE probability distribution (100)
  100-199: LA probability distribution (100)
  200-299: BS probability distribution (100)
  300-308: per-expert [confidence, margin, entropy] x 3 (9)
  309-311: pairwise agreement dot-products CE*LA, CE*BS, LA*BS (3)

The full probability distributions are kept first so the existing
per-expert 100-dim column blocks in the diagnostic scripts remain valid.
"""
import torch
import torch.nn.functional as F


def calibrate_expert_probs(logits_list, cls_num_list, la_tau, T=1.0):
    """Bias-adjust + temperature-scale the three experts' raw logits.

    Parameters
    ----------
    logits_list : list of 3 tensors ``[z_ce, z_la, z_bs]``, each ``(B, C)``
        Raw, *unadjusted* scorers output by the frozen experts. CE has no
        bias; LA/BS apply their prior bias at this stage (matching their
        training losses), so the returned probabilities are the calibrated
        posteriors actually used for mixing.
    cls_num_list : array-like of length C, per-class training counts.
    la_tau : float, logit-adjustment temperature for the LA expert.
    T : float, softmax temperature.

    Returns
    -------
    list of 3 tensors ``[p_ce, p_la, p_bs]``, each ``(B, C)``.
    """
    device = logits_list[0].device
    cls_num_list = torch.as_tensor(
        cls_num_list, dtype=torch.float32, device=device
    )
    log_prior = torch.log(cls_num_list / cls_num_list.sum() + 1e-12)
    log_spc = torch.log(cls_num_list + 1e-12)

    p_ce = F.softmax(logits_list[0] / T, dim=1)
    p_la = F.softmax((logits_list[1] + la_tau * log_prior) / T, dim=1)
    p_bs = F.softmax((logits_list[2] + log_spc) / T, dim=1)
    return [p_ce, p_la, p_bs]


def build_gate_input(probs):
    """Build the gate feature vector from calibrated expert probabilities.

    Parameters
    ----------
    probs : list of tensors ``[p_ce, p_la, p_bs]``, each ``(B, C)``.

    Returns
    -------
    ``(B, num_experts*(C+3) + C(num_experts,2))`` feature tensor. For 3
    experts and C=100 this is 312 columns.
    """
    # 1) Full probability distributions first, so each expert keeps a
    #    contiguous C-dim block (used by the diagnostic weight analyzer).
    dist_feats = torch.cat(probs, dim=1)

    # 2) Per-expert confidence statistics.
    stat_feats = []
    for p in probs:
        stat_feats.append(p.max(dim=1, keepdim=True).values)      # confidence
        top2 = p.topk(2, dim=1).values                            # (B, 2)
        stat_feats.append((top2[:, 0] - top2[:, 1]).unsqueeze(1))  # margin
        stat_feats.append(                                        # entropy
            -(p * p.clamp_min(1e-8).log()).sum(1, keepdim=True)
        )

    # 3) Pairwise agreement (dot product of probability vectors). High value
    #    -> experts agree; disagreement + high confidence -> overconfident.
    agree_feats = []
    for i in range(len(probs)):
        for j in range(i + 1, len(probs)):
            agree_feats.append((probs[i] * probs[j]).sum(1, keepdim=True))

    return torch.cat([dist_feats] + stat_feats + agree_feats, dim=1)


def gate_input_dim(num_classes, num_experts=3):
    """Column count produced by :func:`build_gate_input`."""
    per_expert = num_classes + 3
    pairwise = num_experts * (num_experts - 1) // 2
    return num_experts * per_expert + pairwise
