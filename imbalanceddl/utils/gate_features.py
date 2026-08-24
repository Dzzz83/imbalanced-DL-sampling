"""Shared gate-input feature construction and mixture math for MoE expert routing.

This module is the **single source of truth** for:

1. how expert logits are calibrated (bias-adjusted, per-expert temperature),
2. what the gate consumes (``build_gate_input``),
3. **how the expert mixture is built** (``build_mixture``) — the *same* recipe
   (top-k, logit-vs-probability space, weight floor, mixture temperature) is
   used by training, checkpoint selection, and every evaluation script, so the
   gate can never again be trained on a different mixture than it is scored on,
4. the soft-oracle routing target (``build_oracle_target``) with log-space
   sharpening so tail samples get a decisive target (see the post-mortem in
   ``literature_review_moe_routing.md``, RC1/RC2).

Why probability space for the gate, logit space for the mixture
---------------------------------------------------------------
- The gate consumes **calibrated probabilities** (bias-adjusted,
  temperature-scaled) plus explicit confidence / margin / entropy / agreement
  features, so it can detect "sharp one-hot but disagrees with the others"
  instead of doing naive peak-detection on raw logit magnitude.
- The *mixture* defaults to **logit space** (``space='logit'``): averaging
  calibrated logits = product-of-experts (BalPoE), which is better calibrated
  than averaging probabilities and whose mixture-NLL gradients do not vanish
  when the mixture is already confident (RC4).

Feature layout (num_experts=3, num_classes=100 -> 312 dims)
-----------------------------------------------------------
  0-99   : CE probability distribution (100), per-block L2-normalized (optional)
  100-199: LA probability distribution (100), per-block L2-normalized (optional)
  200-299: BS probability distribution (100), per-block L2-normalized (optional)
  300-308: per-expert [confidence, margin, entropy] x 3 (9)
  309-311: pairwise agreement dot-products CE*LA, CE*BS, LA*BS (3)

The statistics/agreement features are always computed on the *unnormalized*
probabilities (max-prob, entropy, agreement are only meaningful on real
probabilities); only the 300 distribution dims are normalized.
"""
import torch
import torch.nn.functional as F


def calibrate_expert_logits(logits_list, cls_num_list, la_tau, T=1.0,
                            per_expert_T=None):
    """Bias-adjust + temperature-scale the three experts' raw logits.

    Parameters
    ----------
    logits_list : list of 3 tensors ``[z_ce, z_la, z_bs]``, each ``(B, C)``
        Raw, *unadjusted* scorers output by the frozen experts. CE has no
        bias; LA/BS apply their prior bias at this stage (matching their
        training losses: LA uses ``+ tau*log(pi)``, BS uses ``+ log(n_y)``).
    cls_num_list : array-like of length C, per-class training counts.
    la_tau : float, logit-adjustment temperature of the LA expert.
    T : float, global softmax temperature (sweep parameter).
    per_expert_T : optional list/tensor of 3 per-expert temperatures; the
        effective temperature of expert i is ``T * per_expert_T[i]``. If None,
        all experts use ``T``.

    Returns
    -------
    list of 3 tensors ``[z_ce, z_la, z_bs]``, each ``(B, C)``, calibrated.
    """
    device = logits_list[0].device
    cls_num_list = torch.as_tensor(
        cls_num_list, dtype=torch.float32, device=device
    )
    log_prior = torch.log(cls_num_list / cls_num_list.sum() + 1e-12)
    log_spc = torch.log(cls_num_list + 1e-12)
    biases = [0.0, la_tau * log_prior, log_spc]

    if per_expert_T is None:
        per_expert_T = [1.0, 1.0, 1.0]

    out = []
    for z, bias, t_j in zip(logits_list, biases, per_expert_T):
        out.append((z + bias) / (T * t_j))
    return out


def calibrate_expert_probs(logits_list, cls_num_list, la_tau, T=1.0,
                           per_expert_T=None):
    """Calibrated expert posteriors (softmax of :func:`calibrate_expert_logits`)."""
    return [
        F.softmax(z, dim=1)
        for z in calibrate_expert_logits(
            logits_list, cls_num_list, la_tau, T, per_expert_T
        )
    ]


def build_gate_input(probs, normalize_blocks=True):
    """Build the gate feature vector from calibrated expert probabilities.

    Parameters
    ----------
    probs : list of tensors ``[p_ce, p_la, p_bs]``, each ``(B, C)``.
    normalize_blocks : bool, L2-normalize each C-dim probability block before
        concatenation (removes overall probability-mass magnitude, which the
        gate otherwise latches onto as a trivial signal).

    Returns
    -------
    ``(B, num_experts*(C+3) + C(num_experts,2))`` feature tensor. For 3
    experts and C=100 this is 312 columns.
    """
    # 1) Statistics on the *raw* probabilities (meaningful only unnormalized).
    stat_feats = []
    for p in probs:
        stat_feats.append(p.max(dim=1, keepdim=True).values)      # confidence
        top2 = p.topk(2, dim=1).values                            # (B, 2)
        stat_feats.append((top2[:, 0] - top2[:, 1]).unsqueeze(1))  # margin
        stat_feats.append(                                        # entropy
            -(p * p.clamp_min(1e-8).log()).sum(1, keepdim=True)
        )

    # 2) Pairwise agreement (dot product of probability vectors).
    agree_feats = []
    for i in range(len(probs)):
        for j in range(i + 1, len(probs)):
            agree_feats.append((probs[i] * probs[j]).sum(1, keepdim=True))

    # 3) The C-dim distributions, optionally per-block L2-normalized.
    if normalize_blocks:
        dist_feats = torch.cat([F.normalize(p, p=2, dim=1) for p in probs],
                               dim=1)
    else:
        dist_feats = torch.cat(probs, dim=1)

    return torch.cat([dist_feats] + stat_feats + agree_feats, dim=1)


def gate_input_dim(num_classes, num_experts=3):
    """Column count produced by :func:`build_gate_input`."""
    per_expert = num_classes + 3
    pairwise = num_experts * (num_experts - 1) // 2
    return num_experts * per_expert + pairwise


def apply_weight_floor(weights, weight_floor):
    """Clip each expert's weight to at least ``weight_floor`` and renormalize.

    Guarantees a rare tail class can never be starved by a degenerate gate
    (expert-choice-routing style capacity floor, §1.7 of the literature review).
    """
    if weight_floor is None or weight_floor <= 0.0:
        return weights
    weights = torch.clamp(weights, min=weight_floor)
    return weights / weights.sum(dim=1, keepdim=True)


def build_mixture(logits_list, weights, cls_num_list, la_tau, T=1.0,
                  per_expert_T=None, k=None, space='logit',
                  weight_floor=0.0, mix_temperature=1.0):
    """Build the routed mixture posterior from raw expert logits + gate weights.

    This is the **single mixture recipe** used everywhere (training loss,
    validation/checkpoint selection, ``extract_posteriors``, and all verify /
    debug scripts), so training and evaluation can never drift apart (RC2).

    Parameters
    ----------
    logits_list : list of 3 raw logit tensors ``(B, C)``.
    weights : tensor ``(B, num_experts)`` of gate weights (need not sum to 1;
        normalized internally when truncated/floored).
    cls_num_list, la_tau, T, per_expert_T : calibration params (see
        :func:`calibrate_expert_logits`).
    k : top-k truncation (after renormalization). ``k >= num_experts`` or
        ``None`` keeps the full mixture.
    space : ``'logit'`` (default; product-of-experts: softmax of weighted
        calibrated logits) or ``'prob'`` (mixture of probabilities).
    weight_floor : minimum weight per expert (capacity floor, §1.7).
    mix_temperature : temperature applied to the *mixture* logits before the
        final softmax (logit space only; fit on the tune set to calibrate the
        final posterior).

    Returns
    -------
    ``p_mix`` tensor ``(B, C)``, differentiable w.r.t. ``weights``.
    """
    num_experts = len(logits_list)
    weights = apply_weight_floor(weights, weight_floor)

    if k is not None and k < num_experts:
        sel_weights, sel_idx = torch.topk(weights, k, dim=1)
        sel_weights = sel_weights / sel_weights.sum(dim=1, keepdim=True)
    else:
        sel_weights, sel_idx = weights, None

    B = weights.size(0)
    if space == 'logit':
        stacked = torch.stack(
            calibrate_expert_logits(logits_list, cls_num_list, la_tau, T,
                                    per_expert_T),
            dim=1,
        )  # (B, E, C)
    else:
        stacked = torch.stack(
            calibrate_expert_probs(logits_list, cls_num_list, la_tau, T,
                                   per_expert_T),
            dim=1,
        )  # (B, E, C)

    if sel_idx is not None:
        rows = torch.arange(B, device=weights.device).unsqueeze(1)
        stacked = stacked[rows, sel_idx]          # (B, k, C)
        w = sel_weights.unsqueeze(2)              # (B, k, 1)
    else:
        w = sel_weights.unsqueeze(2)              # (B, E, 1)

    combined = (w * stacked).sum(dim=1)           # (B, C)
    if space == 'logit':
        return F.softmax(mix_temperature * combined, dim=1)
    return combined


def build_oracle_target(true_probs_experts, tau=0.2, space='logprob'):
    """Soft-oracle routing target from per-expert true-class probabilities.

    ``true_probs_experts`` : ``(B, 3)`` tensor of ``p_i(y|x)``.

    ``space='logprob'`` (default): ``softmax((log p - max log p)/tau)`` — log
    compression keeps small tail probabilities *contrasted*, so tail samples
    get a sharp, decisive target (RC1 fix). Example with ``p=[0.05, 0.02,
    0.08]``: probability-space target ≈ [0.33, 0.29, 0.38] (flat), log-space
    target ≈ [0.08, 0.001, 0.92] (sharp).

    ``space='prob'``: legacy ``softmax(p/tau)`` (flat on tail — do not use).
    """
    if space == 'prob':
        return F.softmax(true_probs_experts / tau, dim=1)
    lp = torch.log(true_probs_experts.clamp_min(1e-12))
    lp = lp - lp.max(dim=1, keepdim=True).values
    return F.softmax(lp / tau, dim=1)


def uniform_weights(batch_size, num_experts, device='cpu'):
    """Equal-weight routing (the uniform baseline) as a ``(B, E)`` tensor."""
    return torch.full((batch_size, num_experts), 1.0 / num_experts,
                      device=device)


def expert_disagreement(probs):
    """Per-sample mask: do the experts disagree on the argmax?

    When all experts predict the same class ``k``, the routed mixture
    (any convex combination, in prob *or* logit space) also argmaxes to
    ``k`` — routing cannot change the prediction, so its learning signal
    on those samples is noise. Returns ``True`` where routing can matter.
    """
    preds = torch.stack([p.argmax(dim=1) for p in probs], dim=1)  # (B, E)
    agree = (preds == preds[:, :1]).all(dim=1)
    return ~agree
