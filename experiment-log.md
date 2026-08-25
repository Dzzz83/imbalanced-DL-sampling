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
**Next Step:** Revert to Exp 13 best config (logprob + MLP, 44.16% bal acc). The correctness target joins the logprob linear router in the FAILED list. The fundamental bottleneck is **feature SNR** — next line of attack must change the gate input features.

---

### Experiment 16: Feature Collinearity Direct Measurement — SNR Bottleneck Refined
**Date:** 2025-07-22
**Hypothesis:** The 316-dim calibrated probability features are near-collinear (per-sample pairwise correlation > 0.95 for >80% of samples), making per-sample routing impossible regardless of target or architecture.
**Changes:** No training run. `diagnose_feature_collinearity.py` executed on the frozen experts (CE/LA/BS) and 500 CIFAR-100-LT test samples, computing per-sample Pearson correlations, within-block SVD effective ranks, full covariance analysis, and per-block variance breakdown.
**Results:**
- Per-sample block correlations (L2-normed probs): **CE vs LA: r = 0.675 ± 0.316** | **CE vs BS: r = 0.685 ± 0.291** | **LA vs BS: r = 0.664 ± 0.306**
- Fraction of samples with r > 0.95: **28–31%** (NOT the hypothesized >80%)
- Within-block effective rank: **73–79 / 100** (blocks are high-rank, not degenerate)
- Full 316-dim covariance effective rank: **71.18 / 316** | Condition number: 891
- **Top 5 principal components explain 70% of total variance** (shared backbone signal)
- Residual expert-discriminative signal: ~30% variance spread across ~66 components
- Class-level profile correlations: **r = 0.91–0.94** (mean profiles near-identical)
- Feature variance by block: probability dims std ≈ 0.098; stats dims (9 cols) std ≈ 0.825 (8.4×); freq dims (4 cols) std ≈ 1.092 (11×)
- Gradient norm (re-run): 0.894 (confirmed healthy — matches Exp 13 exactly)
- LA-saves-the-day w_LA: **0.683** (up from Exp 14's 0.583 — MLP architecture captures this better)
- Best checkpoint (epoch 82): **Bal Acc 44.26%** vs Uniform 43.61% (+0.65 pp)
**Outcome:** ⏳ INCONCLUSIVE (hypothesis PARTIALLY REFUTED — not strict collinearity, but the refined bottleneck is confirmed)
**Root Cause / Why It Worked:** The three probability blocks are **moderately correlated** (r ≈ 0.68, not > 0.95) and each has **high effective rank** (~75/100). Strict collinearity is refuted. However, **70% of total feature variance is shared across experts** (top 5 PCs), and the residual expert-discriminative signal is spread thinly across ~66 components. With only **1,125 training samples** and **316 input dimensions**, the gate cannot extract this fine-grained per-sample signal — it converges to the one robust signal: per-class biases from the 4 frequency features (which have 11× the variance of individual probability dims). The true bottleneck is an **SNR problem**: the per-sample routing signal is a tiny fraction of total feature variance, too diluted for the available training data. The class-level profile correlations (r = 0.91–0.94) confirm that on average, all experts favor the same classes — the differences are in the fine-grained tails.

**Next Step:** The diagnostic phase is now **conclusive**. A research phase has been completed producing a ranked report of 10 candidate routing strategies. The immediate next step is to evaluate DaWin confidence routing as a post-hoc baseline (see PLANNED EXPERIMENTS below).

---

# FAILED APPROACHES — DO NOT SUGGEST

| # | Approach | Root Cause of Failure | Exp # | Date |
|---|----------|----------------------|-------|------|
| 1 | Mini-MLP with BatchNorm on Raw Logits | Non-linear architecture did not fix the uniform weight distribution. The gate still acted as a peak-detector because raw logit magnitude spikes dominate the dot product math regardless of gate depth. | 10 | 2024-05-29 |
| 2 | Probability Routing + Switch Load Balancing Loss | Switch loss successfully balanced routing weights (fixing BS starvation), but probability-space mixing caused the gate to collapse to uniform routing (~33% for all experts), failing to beat the baseline. | 11 | 2024-05-30 |
| 3 | Mixture NLL in logit space + disagree_weight + kl_uniform=3.0 (current config before diagnosis) | Gradient `∂L/∂g_j = w_j · (p_mix(y) − p_j(y))` vanishes on tail samples (all `p_j(y)` tiny). The gate receives zero gradient on ~95% of tail samples. Per-expert calibration also failed (all temps = 1.5), and train/eval k mismatch wasted the 3rd expert's gradient. Regularizers (disagree_weight + kl_uniform) actively fought the remaining signal. | 12 | 2025-07-15 |
| 4 | Logprob target + linear router + balanced temps [1.5, 1.2, 1.5] | **SNR bottleneck** (refined from 'near-collinear' by Exp 16): three calibrated probability distributions from the same ResNet32 backbone are **moderately correlated** (per-sample r ≈ 0.68, not > 0.95). Each block has high effective rank (~75/100). However, **70% of total feature variance is shared across experts** (top 5 PCs), and the residual expert-discriminative signal (~30% variance) is spread across ~66 components. With only **1,125 training samples**, the gate cannot extract this fine-grained per-sample signal — it converges to per-class biases from the 4 frequency features (11× the variance of individual probability dims). Linear router (43.97%) ≈ MLP (44.16%), confirming architecture is not the bottleneck. Exp 16 direct measurement confirms: not strict collinearity, but an **SNR bottleneck** that no target/architecture change on 316-dim features can overcome. | 14, 16 | 2025-07-22 |
| 5 | Correctness targets (L2D) + linear router + balanced temps [1.5, 1.2, 1.5] | Isotonic regression calibrators on only ~625 tune samples produce `P(correct | max-prob)` targets with negligible dynamic range. After normalization, targets are near-uniform [~1/3] for all samples. KL gradient `∂L/∂g = w − t` therefore pushes `w` toward uniform, collapsing routing. Gate temp=2.200 (highest ever) confirms tune set wants gate dead. LA-saves-the-day w_LA=0.311 (completely uniform). First experiment to regress below uniform baseline. | 15 | 2025-07-18 |

---

# IDEAS NOT YET TRIED (Remaining Candidates)

| # | Idea | Status | Priority | Notes |
|---|------|--------|----------|-------|
| 1 | Disagreement Gating | Proposed | LOW | Only train/invoke the gate when experts disagree. Lowered from MED: since the gate cannot learn per-sample routing from 316-dim features, disagreement gating would not fix the fundamental SNR bottleneck — it merely removes easy samples where the gate already defaults to uniform. |
| 2 | Log-Space Sharpened Target (build_oracle_target with space='logprob') | **Tested — see Exp 13, 14** | FAILED | Fixes gradient starvation but features are the bottleneck. The logprob target gives non-zero gradients (norm 0.89), but the 316-dim calibrated probability features have an SNR bottleneck: 70% of variance is shared across experts (Exp 16). |
| 3 | Correctness Targets (L2D family) | **Tested — see Exp 15** | FAILED | Fit per-expert isotonic correctness calibrators on tune set; replace target with P(expert correct \| x). **Practical failure**: isotonic calibrators on only ~625 tune samples produce near-uniform targets. Gate temp=2.200 (highest ever). Routing collapsed below uniform baseline (43.34% vs 43.53%). Theoretical guarantees don't hold under data-limited (625 samples, 100-class) conditions. |
| 4 | Two-Stage Routing (RIDE-style) | Proposed | LOW | Default = uniform mixture. Only activate per-sample router when expert disagreement is high. Reduces routing noise on easy/ambiguous samples. |
| 5 | Max-KL Routing (track the third expert) | Proposed | LOW | Change verify_stage2.py to use recipe['k'] instead of hard-coded top-2. Ultra_debug already does this correctly. |

---
---

# PLANNED EXPERIMENTS (Ranked by Priority)

The following experiments are ordered by the decision tree from the ranked research report. Each must be executed sequentially — the outcome of each determines which path to follow next.

| Order | Experiment | Strategy | Key Parameters | Verification Step | Expected Outcome & Next Action |
|:-----:|:-----------|:---------|:---------------|:------------------|:------------------------------|
| **1** | **Exp 17 — DaWin Baseline** | **DaWin confidence routing (training-free, 3-dim)** | `w_j = softmax(conf_j / T̂)`, `conf_j = max_k p_j(k\|x)`. Grid search T̂ ∈ [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0] on tune set. **0 params, 0 training.** | Post-hoc eval on existing checkpoint. Add DaWin column to `ultra_debug.py` (≈10 lines). Compare DaWin bal acc vs gate (44.26%) vs uniform (43.61%). | **If ≥ 44.26%:** DaWin is the answer. STOP. **If 44.0–44.25%:** Confidence captures most signal; proceed to Exp 18. **If < 44.0%:** Confidence insufficient; skip to Exp 19. |
| **2** | **Exp 18 — 3-Param Temperature Router** | **Per-expert temperature scalars** | `w_j = softmax(conf_j / exp(log_T_j))`. 3 learned params `log_T_j`. Train with logprob KL loss on 1,125 gate samples. | Full training run with `GateMLP` replaced by 3 `nn.Parameter` scalars. Compare bal acc against DaWin baseline. | **If > DaWin by ≥0.5 pp:** Per-expert calibration mismatches matter; this is the learned router. **If ≈ DaWin:** Confidence scores already capture full calibration signal. |
| **3** | **Exp 19 — Penultimate Feature Routing** | **192-dim embeddings + Linear(192,3)** | Replace 316-dim probability features with concat of 3 × 64-dim penultimate embeddings. Linear router: 579 params. | **Phase A:** Diagnose 192-dim feature correlations first (modify `diagnose_feature_collinearity.py` to extract `hidden_list`). | **If cross-expert r < 0.5:** Proceed to full training run. **If r ≥ 0.5:** Features still correlated; fall back to DES or SADE. |
| **4** | **Exp 20 — Stats-Only Logistic Router** | **13-dim stats + Linear(13,3)** | Discard all 300 probability dims. Keep only 9 stats dims + 4 freq features. Linear router: 42 params. | Add `gate_input_mode: stats_only` config option to `GateTrainer`. Train and compare bal acc against 316-dim baseline. | **If within 0.2 pp of 316-dim baseline:** SNR hypothesis confirmed — 300 prob dims contribute nothing. **If below baseline →** Routing signal requires probability dims after all (contradicts diagnostic, warrants re-investigation). |

### Fallback Options (Lower-Ranked Candidates, Only If Top-4 Fail)

| Candidate | When to Consider | Implementation Complexity |
|:----------|:-----------------|:--------------------------|
| **DaWin + Disagreement Mask** (Hybrid of #1 + disagreement gating) | If DaWin alone < 44.0% but disagreement signal is strong (known: 61% correct-agree vs 23% incorrect-agree = 38 pp gap) | Low — `expert_disagreement()` already exists in `gate_features.py` |
| **SADE Test-Time Optimization** (Zhang et al., NeurIPS 2022) | If all learned routers fail and training-free methods are needed | High — per-sample optimization loop with augmentations |
| **DES Competence Routing** (DESlib, OLA/LCA/META-DES) | If penultimate features (Exp 19) have meaningful distance structure but linear router fails | Moderate — requires 192-dim feature pipeline first |
| **Meta-Learning Router** (MAML-style, few-shot) | Only if per-class oracle analysis shows clear class-level expert preferences (Exp 13–16 data suggests within-class variance is high, so this is unlikely) | Very high — MAML inner-loop, episode sampling |

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

**Updated Baseline (2025-07-22 final diagnostic sweep — Exp 16, epoch 82):**

| Metric | Uniform Avg (Floor) | Best Gate | Oracle (Ceiling) | Gap Captured |
|--------|:-------------------:|:---------:|:----------------:|:------------:|
| Balanced Acc | 43.61% | **44.26%** (epoch 82) | 52.09% | **+0.65 pp / 7.83 pp** |
| Head Acc | 71.14% | 70.43% (epoch 82) | 79.79% | −0.71 pp |
| Med Acc | 43.21% | **44.18%** (epoch 82) | 53.75% | +0.96 pp |
| Tail Acc | 11.96% | **13.83%** (epoch 82) | 17.83% | **+1.88 pp / 5.95 pp** |
| NLL | 1.192 | 1.218 (epoch 82) | — | +0.026 (worse) |
| Brier | 0.426 | 0.433 (epoch 82) | — | +0.007 (worse) |
| Tail-ECE | 0.318 | 0.326 (epoch 82) | 0.088* | +0.008 (worse) |

*Paper reports 0.088 tail-ECE for CRISP method. Our oracle achieves this for the *oracle-picked* expert, not the mixture — listed as reference ceiling only.

**Diagnostic Conclusion (2025-07-22):** The feature correlation is moderate (r ≈ 0.68), not collinear (r > 0.95), but 70% of variance is shared across experts. The true bottleneck is **SNR-limited**: the per-sample routing signal is too diluted across ~66 residual components for 1,125 training samples to extract. No further tuning of targets, architectures, or hyperparameters on 316-dim calibrated probability features will yield meaningful improvement. **Diagnostic phase complete — proceed to new routing strategy design.**

**Diagnostic Trace (Root Cause Progression):**

| Exp | Config | Bottleneck Identified | Key Evidence |
|-----|--------|----------------------|-------------|
| 12 | mix_nll + MLP | **Gradient starvation** on tail | ∂L/∂g_j ≈ 0 for 95% of tail samples. Gradient norm = 0.251. |
| 13 | logprob + MLP | **Overparameterization** (20k params / 1,125 samples) | fc.weight std = 0.074 (dead hidden layer). ReLU collapses near-identical features. Gradient norm = 0.892 (fixed). |
| 14 | logprob + linear | **Feature collinearity** (probabilities from same backbone) | fc.weight blocks indistinguishable (0.201, 0.203, 0.200). Linear (43.97%) ≈ MLP (44.16%). Gate learns class bias, not per-sample routing. |
| 15 | correctness + linear | **L2D calibrators need more data** | Isotonic fit on 625 tune samples → near-uniform targets → gate_temp=2.200 (highest ever) → routing collapsed below baseline (43.34%). |
| 16 | feature correlation measurement | **SNR bottleneck** (refined from 'near-collinear') | Per-sample block corr r ≈ 0.68 (not >0.95). Effective rank ~75/100 per block. Top 5 PCs explain 70% variance (shared). Residual routing signal ~30% across ~66 components. Gate cannot extract per-sample signal from 1,125 samples. |
