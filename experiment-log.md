# Experiment Log: MoE Gate Routing — Imbalanced-DL-Sampling

## ⚠️ INSTRUCTIONS TO AI MODELS — DO NOT MODIFY THIS SECTION

This file is the permanent research memory for this project.
Follow these rules exactly.

### Purpose
1. Track every experiment attempted (successful or failed)
2. Prevent re-suggesting approaches that already failed
3. Provide evidence-based context for proposing new experiments

### Rules for AI Models Reading This File
- **NEVER suggest** any approach listed in the FAILED APPROACHES table
- **Check the IDEAS NOT YET TRIED table** before proposing anything — it may already be planned
- When analyzing results, reference experiment numbers (e.g., "as seen in Exp #N")
- Base new hypotheses on the evidence in this log, not generic knowledge
- If tables or entries contain [FILL IN] placeholders, treat that data as unknown — do not invent values

### Rules for Logging New Experiments (After Each Training Run)
When a new experiment completes, append a new entry to EXPERIMENT HISTORY
using the exact format below. Place newest entries at the BOTTOM of that
section. Then update the FAILED APPROACHES or IDEAS tables if applicable.

**Required entry format:**

### Experiment N: [Short Descriptive Name]
**Date:** YYYY-MM-DD
**Hypothesis:** [One sentence: what you believed would happen]
**Changes:** [Bullet list: exactly what differed from prior run]
**Results:**
- Overall Acc: XX.X%
- Head: XX.X% | Middle: XX.X% | Tail: XX.X%
- NLL: X.XX
- Gate routing distribution: CE XX% / LA XX% / BS XX%
**Outcome:** [✅ SUCCESS / ❌ FAILED / ⏳ INCONCLUSIVE]
**Root Cause / Why It Worked:** [1-3 sentences grounded in the data]
**Next Step:** [What this result implies should be tried next]

### Rules for Updating Tables
- **FAILED APPROACHES:** Add an entry ONLY when an experiment conclusively fails. Include the root cause so future models understand *why* it failed, not just *that* it failed.
- **IDEAS NOT YET TRIED:** When proposing a new idea, first add it here with status "Proposed". After testing, update status to "Tested — see Exp #N" and move it to the appropriate section if it failed.
- **SUCCESSFUL APPROACHES:** Add only when an experiment beats the uniform-averaging baseline.
- **BASELINE REFERENCE:** Update whenever new baseline numbers are established.

═══════════════════════════════════════════════════════════
## END OF INSTRUCTIONS — LOG ENTRIES BEGIN BELOW
═══════════════════════════════════════════════════════════

# EXPERIMENT HISTORY
---

### Experiment 12: Diagnostic Sweep — Mixture NLL in Logit Space (current config)
**Date:** 2025-07-15
**Hypothesis:** Running the full diagnostic suite (`verify_stage2.py`, `ultra_debug.py`, `inspect_gate_gradients.py`) would reveal the root cause of gate uniform collapse.
**Changes:** No training run. All three diagnostic scripts executed on existing checkpoint `gate_checkpoint_bs128_T1.0_epoch30.pth` with expert temps [1.5, 1.5, 1.5], k=3 training (k=2 eval), mix_nll target, disagree_weight + kl_uniform=3.0 regularizers.
**Results:**
- Best gate bal acc: 43.59% vs uniform 43.35% (+0.24 pp)
- Tail acc: 12.00% vs uniform 11.00% (+1.00 pp)
- Head acc: 70.29% vs uniform 70.69% (−0.40 pp, regression)
- NLL: 1.214 vs uniform 1.202 (worse)
- Brier: 0.430 vs uniform 0.427 (worse)
- Tail-ECE: 0.287 vs uniform 0.291 (same)
- Gate routing: CE=33.1% / LA=36.4% / BS=30.6% (near-uniform)
- Gradient norm: 0.2508 (low — confirms flat loss landscape)
- Oracle gap: 8.61 pp on bal acc, 6.75 pp on tail
- Per-class routing: only ONE tail class (99, n=5) deviates from uniform (w_LA=0.5445)
- LA-saves-the-day: 136 tail samples where LA is sole correct expert → avg w_LA=0.5464 (gate works when signal exists)
**Outcome:** ❌ FAILED (but diagnosis of failure is ✅ SUCCESSFUL)
**Root Cause / Why It Worked:** The mixture NLL gradient `∂L/∂g_j = w_j · (p_mix(y) − p_j(y))` vanishes on tail samples where all `p_j(y|x)` are tiny. The gate receives no gradient for ~2,344/2,480 tail samples. Additionally: (1) the train/eval k mismatch (k=3 training vs k=2 eval) wastes the 3rd expert's weight gradient, (2) the disagree_weight + kl_uniform=3.0 regularizers actively fight the remaining gradient on disagree samples, and (3) the equal expert_temps=[1.5, 1.5, 1.5] nullifies per-expert calibration. The 136 LA-saves-the-day samples prove the gate *can* specialize when the gradient is non-zero, making this a supervision problem, not an architecture problem.
**Next Step:** Switch to logprob target (`gate_target_mode: logprob`), align train/eval k, remove disagree_weight/kl_uniform for logprob mode, and increase tune set for expert temp fitting. Then re-train.

---

### Experiment 13: Logprob Target + [1.5, 1.5, 1.5] Temps (Gradient Starvation Fix)
**Date:** 2025-07-16
**Hypothesis:** Switching to logprob target (soft-oracle KL with log-space sharpening) gives non-zero gradient on all samples, fixing the gradient starvation that prevented per-sample routing.
**Changes:** `gate_target_mode: logprob` (was mix_nll), `gate_disagree_weight: false`, `gate_kl_uniform: 0.0`, `test_size=0.5` in `_gate_trainer.py` (larger tune set). Expert temps still [1.5, 1.5, 1.5] (fitting produced all-equal values). Gate is MLP (316→64→3).
**Results:**
- Best bal acc: 44.16% vs uniform 43.59% (+0.57 pp)
- Tail acc: 13.67% vs uniform 12.57% (+1.10 pp)
- Head acc: 70.04% vs uniform 70.51% (−0.47 pp, regression)
- NLL: 1.255 vs uniform 1.210 (worse)
- Gradient norm: 0.892 (3.6× larger than Exp 12's 0.251 — gradient starvation fixed)
- fc.weight blocks still indistinguishable: CE=0.074, LA=0.073, BS=0.073 std
- Gate pre-softmax std: 1.57 (healthy)
- LA peak-probability frequency: 31.9% (balanced)
**Outcome:** ❌ FAILED (gradient starvation fixed, but performance still near-uniform)
**Root Cause / Why It Worked:** The logprob target fixed gradient starvation (gradient norm 3.6× larger). The fc.weight std is tiny (0.074) because the MLP's 64 hidden units are dead — the ReLU collapses near-identical input blocks to near-identical activations. The MLP (20k params on 1,125 samples) overparameterizes the problem, causing the hidden layer to memorize noise at the original position in the 316-dim space rather than developing expert-specific selectivity.
**Next Step:** Replace MLP with linear router (fewer params, no ReLU collapse), and balance expert temps to prevent any single expert from dominating the input features.

---

### Experiment 14: Linear Router + Balanced Temps [1.5, 1.2, 1.5]
**Date:** 2025-07-17
**Hypothesis:** A linear router (316→3, no hidden layer, 951 params) matched to the ~1,125-sample gate training set will learn per-expert weight selectivity, and balanced temperatures [1.5, 1.2, 1.5] will prevent LA from dominating the gate input features.
**Changes:** `gate_linear_router: true` (linear 316→3), `expert_temperatures: [1.5, 1.2, 1.5]` (was [1.5, 1.5, 1.5]). GateMLP rewritten to support both linear and MLP modes. All 7 eval scripts updated for linear_router flag. Logprob target retained, k=3, gate_temp fitted on tune.
**Results:**
- Best bal acc: **43.97%** (epoch 92) vs Uniform **43.53%** (+0.44 pp)
- Tail acc: **12.83%** vs Uniform **11.60%** (+1.23 pp)
- Head acc: **69.89%** vs Uniform **70.80%** (−0.91 pp, regression)
- Med acc: **44.74%** vs Uniform **43.63%** (+1.11 pp)
- NLL: **1.231** vs Uniform **1.200** (worse)
- Brier: **0.435** vs Uniform **0.427** (worse)
- Tail-ECE: **0.341** vs Uniform **0.320** (worse)
- Gradient norm: **0.370** (down from 0.892 with MLP + same temps)
- Gate weights: w_CE=0.292, w_LA=0.382, w_BS=0.325 (near-uniform)
- fc.weight block std: CE=**0.201**, LA=**0.203**, BS=**0.200** (healthy but **indistinguishable**)
- LA peak-probability frequency: **51.2%** (balanced, was 72.4% with [2.0,1.0,1.5])
- Oracle bal acc: **52.09%** (gap: **8.12%**)
- Oracle distribution: CE=30.3%, LA=40.0%, BS=29.7%
- LA-saves-the-day w_LA: **0.583** (136 samples where LA is sole correct expert)
- Gate pre-softmax std: **0.441** (healthy)
- NLL gap to paper: +4.22% (1.222 vs 1.18)
- 30 checkpoints evaluated, best epoch 92 out of 92
**Outcome:** ❌ FAILED (linear router didn't improve over MLP)
**Root Cause / Why It Worked:**
The fc.weight blocks have healthy std (~0.20, 2.7× larger than MLP's 0.074) but are **statistically indistinguishable across experts** (0.201, 0.203, 0.200). This confirms the **feature bottleneck**: the 316-dim input (three 100-dim calibrated probability distributions + statistics) is near-collinear because all three experts share the same ResNet32 backbone. The gate can learn per-class biases (class 0 w_LA=0.177, class 99 w_LA=0.455) but cannot learn per-sample routing because the probability distributions are too similar. The gradient norm dropped from 0.892 (MLP) to 0.370 (linear) because the linear router has fewer params and converges to a near-uniform solution faster.

**Root cause chain (now complete):**
1. Exp 12: mixture NLL → gradient starvation on tail (∂L/∂g_j ≈ 0)
2. Exp 13: logprob target → fixed gradient starvation, but MLP overparameterized (20k params / 1,125 samples → ReLU collapse → fc.weight std=0.074)
3. Exp 14: linear router → fixed overparameterization (951 params, fc.weight std=0.20), but **features are the bottleneck** — three calibrated probability distributions are near-collinear
4. Exp 15: correctness targets → **failed worse** — isotonic calibrators on 625 tune samples produce near-uniform targets; routing completely collapses (gate_temp=2.200, LA-saves-the-day w_LA=0.311)

**Empirical proof of feature bottleneck:**
- fc.weight blocks indistinguishable despite healthy std (0.201 vs 0.203 vs 0.200)
- Per-class routing shows class-level biases (head→CE, tail→LA) but no per-sample specialization
- Gradient norm is healthy (0.370) yet weights remain near-uniform
- LA-saves-the-day shows the gate CAN specialize when signal is clear (w_LA=0.583 on 136 samples), confirming the gate works — it's the features that don't support per-sample routing
**Next Step:** Revert to Exp 13 best config (logprob + MLP, 44.16% bal acc). The linear router and correctness targets both failed to improve over this. The fundamental bottleneck is **feature collinearity** which no target or architecture change can fix with 316-dim calibrated probability features. Next line of attack: change the gate input features to penultimate features (192-dim embeddings), which are richer and less correlated.

---

### Experiment 15: Correctness Targets (L2D) — Failed
**Date:** 2025-07-18
**Hypothesis:** Correctness targets `t_j = P(expert j correct | x)` would give non-zero gradient on all samples (even when p_j(y|x) is tiny), and the binary correctness signal would be less corrupted by feature collinearity than the logprob target's softmax over true-class probabilities.
**Changes:** `gate_target_mode: correctness` (was logprob). All other settings identical to Exp 14: linear router, [1.5, 1.2, 1.5] temps, k=3, logit mixing. No code changes needed (correctness target was already implemented via `_fit_correctness_calibrators` + `_correctness_target`).
**Results:**
- Best bal acc: **43.34%** (epoch 3) vs Uniform **43.53%** (−0.19 pp — **BELOW BASELINE**)
- Tail acc: **11.43%** vs Uniform **11.60%** (−0.17 pp — **BELOW BASELINE**)
- Head acc: **70.83%** vs Uniform **70.80%** (+0.03 pp, flat)
- NLL: **1.201** (same as uniform 1.200)
- Gate routing weights: w_CE=0.320, w_LA=0.345, w_BS=0.335 (completely uniform)
- fc.weight block std: CE=0.080, LA=0.081, BS=0.081 (collapsed from Exp 14's 0.20)
- **Gate temperature fitted to 2.200** (highest ever — tune set wants gate dead)
- LA-saves-the-day w_LA: **0.311** (completely uniform — down from 0.583 in Exp 14)
- Gate pre-softmax std: 0.219 (healthy scale but uniform direction)
- Only 7 epochs evaluated; training stopped because all checkpoints below baseline
**Outcome:** ❌ FAILED (worse than baseline — first experiment to regress below uniform)
**Root Cause / Why It Failed:** The isotonic regression calibrators (fit on only ~625 tune samples) produce `P(correct | max-prob)` values with extremely limited dynamic range (0.05–0.25 for most samples). After normalization to a simplex, the target is near-uniform [~0.333, ~0.333, ~0.334] for all samples. The KL gradient `∂L/∂g = w − t` therefore pushes `w` toward uniform for every sample, collapsing the routing. The tune set confirms this by selecting gate_temp=2.200 (making the softmax output as near-uniform as possible). This is a practical failure of the L2D approach under data-limited conditions: the correctness calibrators need much more tune data to learn a stable mapping for 100 classes.
**Next Step:** Revert to Exp 13 best config (logprob + MLP, 44.16% bal acc). The correctness target joins the logprob linear router in the FAILED list. The fundamental bottleneck is **feature collinearity** — next line of attack must change the gate input features.

---

# FAILED APPROACHES — DO NOT SUGGEST

| # | Approach | Root Cause of Failure | Exp # | Date |
|---|----------|----------------------|-------|------|
| 1 | Mini-MLP with BatchNorm on Raw Logits | Non-linear architecture did not fix the uniform weight distribution. The gate still acted as a peak-detector because raw logit magnitude spikes dominate the dot product math regardless of gate depth. | 10 | 2024-05-29 |
| 2 | Probability Routing + Switch Load Balancing Loss | Switch loss successfully balanced routing weights (fixing BS starvation), but probability-space mixing caused the gate to collapse to uniform routing (~33% for all experts), failing to beat the baseline. | 11 | 2024-05-30 |
| 3 | Mixture NLL in logit space + disagree_weight + kl_uniform=3.0 (current config before diagnosis) | Gradient `∂L/∂g_j = w_j · (p_mix(y) − p_j(y))` vanishes on tail samples (all `p_j(y)` tiny). The gate receives zero gradient on ~95% of tail samples. Per-expert calibration also failed (all temps = 1.5), and train/eval k mismatch wasted the 3rd expert's gradient. Regularizers (disagree_weight + kl_uniform) actively fought the remaining signal. | 12 | 2025-07-15 |
| 4 | Logprob target + linear router + balanced temps [1.5, 1.2, 1.5] | **Feature bottleneck**: three calibrated probability distributions from the same ResNet32 backbone are near-collinear. The fc.weight blocks have healthy std (0.20, 2.7× MLP) but are statistically indistinguishable across experts (CE=0.201, LA=0.203, BS=0.200). The gate learns per-class biases (head→CE, tail→LA) but cannot learn per-sample routing because the input features don't contain enough per-sample discriminative signal. Linear router (43.97%) performs essentially identically to MLP (44.16%), confirming architecture is not the bottleneck. | 14 | 2025-07-17 |
| 5 | Correctness targets (L2D) + linear router + balanced temps [1.5, 1.2, 1.5] | Isotonic regression calibrators on only ~625 tune samples produce `P(correct | max-prob)` targets with negligible dynamic range. After normalization, targets are near-uniform [~1/3] for all samples. KL gradient `∂L/∂g = w − t` therefore pushes `w` toward uniform, collapsing routing. Gate temp=2.200 (highest ever) confirms tune set wants gate dead. LA-saves-the-day w_LA=0.311 (completely uniform). First experiment to regress below uniform baseline. | 15 | 2025-07-18 |

---

# IDEAS NOT YET TRIED

| # | Idea | Status | Priority | Notes |
|---|------|--------|----------|-------|
| 1 | Disagreement Gating | Proposed | MED | Only train/invoke the gate when experts disagree. Removes easy samples, allowing gate to focus on hard, tail-heavy samples. |
| 2 | Log-Space Sharpened Target (build_oracle_target with space='logprob') | **Tested — see Exp 13, 14** | FAILED | Fixes gradient starvation but features are the bottleneck. The logprob target gives non-zero gradients, but the calibrated probability features are near-collinear. |
| 3 | Correctness Targets (L2D family) | **Tested — see Exp 15** | FAILED | Fit per-expert isotonic correctness calibrators on tune set; replace target with P(expert correct \| x). **Practical failure**: isotonic calibrators on only ~625 tune samples produce near-uniform targets. Gate temp=2.200 (highest ever). Routing collapsed below uniform baseline (43.34% vs 43.53%). Theoretical guarantees don't hold under data-limited (625 samples, 100-class) conditions. |
| 4 | Penultimate Feature Routing (192-dim) | Proposed | **HIGH** | Replace 316-dim calibrated probability features with 192-dim penultimate features (3 × 64-dim embeddings from frozen experts). These features are richer and less correlated. Requires pipeline change: `ExpertEnsemble.forward()` currently returns only last expert's embeddings; needs per-expert embeddings. |
| 5 | DaWin-Style Confidence Routing (3-dim) | Proposed | **HIGH** | Training-free: route by `softmax(conf_j / T)` where `conf_j = max_k p_j(y=k\|x)`. Just 3 numbers, eliminates feature collinearity problem entirely. Matches DaWin (Oh et al., NeurIPS 2024) which proves this beats learned routers for frozen experts. Can be evaluated as a post-hoc baseline without training. |
| 6 | Two-Stage Routing (RIDE-style) | Proposed | LOW | Default = uniform mixture. Only activate per-sample router when expert disagreement is high. Reduces routing noise on easy/ambiguous samples. |
| 7 | Max-KL Routing (track the third expert) | Proposed | LOW | Change verify_stage2.py to use recipe['k'] instead of hard-coded top-2. Ultra_debug already does this correctly. |

---

# SUCCESSFUL APPROACHES (Beats Uniform Baseline)

| # | Approach | Improvement over Uniform | Exp # | Date |
|---|----------|--------------------------|-------|------|
| — | [FILL IN — none yet] | — | — | — |

---

# BASELINE REFERENCE NUMBERS

| Metric | Uniform Avg (Floor) | Oracle (Ceiling) | Gap (Routing Opportunity) |
|--------|--------------------|------------------|---------------------------|
| Overall Acc | 42.78% | 52.50% | +9.72% |
| Head Acc | 70.54% | 80.14% | +9.60% |
| Middle Acc | 42.77% | 54.21% | +11.44% |
| Tail Acc | 10.40% | 18.25% | +7.85% |
| NLL | 1.268 | [FILL IN] | [FILL IN] |

**Updated Baseline (2025-07-17 diagnostic sweep — Exp 14):**

| Metric | Uniform Avg (Floor) | Best Gate | Oracle (Ceiling) | Gap Captured |
|--------|:-------------------:|:---------:|:----------------:|:------------:|
| Balanced Acc | 43.53% | **43.97%** (exp14) | 52.09% | **+0.44 pp / 8.12 pp** |
| Head Acc | 70.80% | 69.89% (exp14) | 79.79% | −0.91 pp |
| Med Acc | 43.63% | 44.74% (exp14) | 53.75% | +1.11 pp |
| Tail Acc | 11.60% | **12.83%** (exp14) | 17.83% | **+1.23 pp / 6.17 pp** |
| NLL | 1.200 | 1.231 (exp14) | — | +0.031 (worse) |
| Brier | 0.427 | 0.435 (exp14) | — | +0.008 (worse) |
| Tail-ECE | 0.320 | 0.341 (exp14) | 0.088* | +0.021 (worse) |

*Paper reports 0.088 tail-ECE for CRISP method. Our oracle achieves this for the *oracle-picked* expert, not the mixture — listed as reference ceiling only.

**Diagnostic Trace (Root Cause Progression):**

| Exp | Config | Bottleneck Identified | Key Evidence |
|-----|--------|----------------------|-------------|
| 12 | mix_nll + MLP | **Gradient starvation** on tail | ∂L/∂g_j ≈ 0 for 95% of tail samples. Gradient norm = 0.251. |
| 13 | logprob + MLP | **Overparameterization** (20k params / 1,125 samples) | fc.weight std = 0.074 (dead hidden layer). ReLU collapses near-identical features. Gradient norm = 0.892 (fixed). |
| 14 | logprob + linear | **Feature collinearity** (probabilities from same backbone) | fc.weight blocks indistinguishable (0.201, 0.203, 0.200). Linear (43.97%) ≈ MLP (44.16%). Gate learns class bias, not per-sample routing. |
| 15 | correctness + linear | **L2D calibrators need more data** | Isotonic fit on 625 tune samples → near-uniform targets → gate_temp=2.200 (highest ever) → routing collapsed below baseline (43.34%). |
