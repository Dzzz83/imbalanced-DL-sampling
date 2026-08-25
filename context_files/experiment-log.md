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

See `experiment-log.md` (root) for the full experiment history (Exp 10–16) with complete entries, diagnostic trace, and baseline tables.

---

# FAILED APPROACHES — DO NOT SUGGEST

| # | Approach | Root Cause of Failure | Exp # | Date |
|---|----------|----------------------|-------|------|
| 1 | Mini-MLP with BatchNorm on Raw Logits | Non-linear architecture did not fix the uniform weight distribution. The gate still acted as a peak-detector because raw logit magnitude spikes dominate the dot product math regardless of gate depth. | 10 | 2024-05-29 |
| 2 | Probability Routing + Switch Load Balancing Loss | Switch loss successfully balanced routing weights (fixing BS starvation), but probability-space mixing caused the gate to collapse to uniform routing (~33% for all experts), failing to beat the baseline. | 11 | 2024-05-30 |
| 3 | Mixture NLL in logit space + disagree_weight + kl_uniform=3.0 | Gradient `∂L/∂g_j = w_j · (p_mix(y) − p_j(y))` vanishes on tail samples (all `p_j(y)` tiny). Regularizers fight remaining signal. | 12 | 2025-07-15 |
| 4 | Logprob target + linear router + balanced temps [1.5, 1.2, 1.5] | **SNR bottleneck** (refined from 'near-collinear' by Exp 16): 70% of total feature variance is shared across experts; residual routing signal too diluted for 1,125 training samples. | 14, 16 | 2025-07-22 |
| 5 | Correctness targets (L2D) + linear router + balanced temps [1.5, 1.2, 1.5] | Isotonic calibrators on only ~625 tune samples produce near-uniform targets; gate temp=2.200; routing collapsed below uniform baseline. | 15 | 2025-07-18 |

---

# IDEAS NOT YET TRIED (Remaining Candidates)

| # | Idea | Status | Priority | Notes |
|---|------|--------|----------|-------|
| 1 | Disagreement Gating | Proposed | LOW | Only train/invoke the gate when experts disagree. |
| 2 | Two-Stage Routing (RIDE-style) | Proposed | LOW | Default = uniform mixture. Only activate per-sample router when disagreement is high. |
| 3 | Max-KL Routing (track the third expert) | Proposed | LOW | Change verify_stage2.py to use recipe['k'] instead of hard-coded top-2. |

---

# PLANNED EXPERIMENTS (Ranked by Priority)

The following experiments are ordered by the decision tree from the ranked research report. Each must be executed sequentially.

| Order | Experiment | Strategy | Key Parameters | Verification Step | Expected Outcome & Next Action |
|:-----:|:-----------|:---------|:---------------|:------------------|:------------------------------|
| **1** | **Exp 17 — DaWin Baseline** | **DaWin confidence routing (training-free, 3-dim)** | `w_j = softmax(conf_j / T̂)`, `conf_j = max_k p_j(k\|x)`. Grid search T̂ on tune set. **0 params, 0 training.** | Post-hoc eval on existing checkpoint. Add DaWin column to `ultra_debug.py` (≈10 lines). Compare vs gate (44.26%) vs uniform (43.61%). | **If ≥ 44.26%:** DaWin is the answer. STOP. **If 44.0–44.25%:** Proceed to Exp 18. **If < 44.0%:** Skip to Exp 19. |
| **2** | **Exp 18 — 3-Param Temperature Router** | **Per-expert temperature scalars** | `w_j = softmax(conf_j / exp(log_T_j))`. 3 learned params. Train with logprob KL loss. | Full training run. Compare bal acc against DaWin baseline. | **If > DaWin by ≥0.5 pp:** Calibration matters. **If ≈ DaWin:** Confidence captures full calibration. |
| **3** | **Exp 19 — Penultimate Feature Routing** | **192-dim embeddings + Linear(192,3)** | Replace 316-dim probs with 3 × 64-dim embeddings. Linear router: 579 params. | **Phase A:** Diagnose 192-dim feature correlations first. | **If r < 0.5:** Proceed. **If r ≥ 0.5:** Fall back to DES or SADE. |
| **4** | **Exp 20 — Stats-Only Logistic Router** | **13-dim stats + Linear(13,3)** | Keep 9 stats dims + 4 freq features. Linear router: 42 params. | Add `gate_input_mode: stats_only` config. Train and compare. | **If within 0.2 pp of 316-dim baseline:** SNR hypothesis confirmed. |

### Fallback Options (Lower-Ranked Candidates, Only If Top-4 Fail)

| Candidate | When to Consider | Implementation Complexity |
|:----------|:-----------------|:--------------------------|
| **DaWin + Disagreement Mask** | If DaWin alone < 44.0% but disagreement signal is strong | Low |
| **SADE Test-Time Optimization** | If all learned routers fail | High |
| **DES Competence Routing** | If penultimate features have meaningful distance structure | Moderate |
| **Meta-Learning Router** | Only if per-class oracle shows clear expert preferences (unlikely given Exp 13–16 data) | Very high |

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
