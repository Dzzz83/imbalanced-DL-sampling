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

# FAILED APPROACHES — DO NOT SUGGEST

| # | Approach | Root Cause of Failure | Exp # | Date |
|---|----------|----------------------|-------|------|
| 1 | Mini-MLP with BatchNorm on Raw Logits | Non-linear architecture did not fix the uniform weight distribution. The gate still acted as a peak-detector because raw logit magnitude spikes dominate the dot product math regardless of gate depth. | 10 | 2024-05-29 |
| 2 | Probability Routing + Switch Load Balancing Loss | Switch loss successfully balanced routing weights (fixing BS starvation), but probability-space mixing caused the gate to collapse to uniform routing (~33% for all experts), failing to beat the baseline. | 11 | 2024-05-30 |
| 3 | Mixture NLL in logit space + disagree_weight + kl_uniform=3.0 (current config before diagnosis) | Gradient `∂L/∂g_j = w_j · (p_mix(y) − p_j(y))` vanishes on tail samples (all `p_j(y)` tiny). The gate receives zero gradient on ~95% of tail samples. Per-expert calibration also failed (all temps = 1.5), and train/eval k mismatch wasted the 3rd expert's gradient. Regularizers (disagree_weight + kl_uniform) actively fought the remaining signal. | 12 | 2025-07-15 |

---

# IDEAS NOT YET TRIED

| # | Idea | Status | Priority | Notes |
|---|------|--------|----------|-------|
| 1 | Disagreement Gating | Proposed | MED | Only train/invoke the gate when experts disagree. Removes easy samples, allowing gate to focus on hard, tail-heavy samples. |
| 2 | Log-Space Sharpened Target (build_oracle_target with space='logprob') | **Ready to test** | **HIGH** | Change config to `gate_target_mode: logprob`. The `build_oracle_target` function with `space='logprob'` is already implemented. Gradient `∂L/∂g = w − t` with log-sharpened `t` is non-zero on tail samples. This directly addresses RC4a. |
| 3 | Correctness Targets (L2D family) | Proposed | MED | Fit per-expert isotonic correctness calibrators on tune set; replace target with P(expert correct | x). Consistent (Mozannar–Sontag), calibrated (Cao), tail-safe. Backup if #2 underperforms. |
| 4 | Max-KL Routing (track the third expert) | Proposed | LOW | Change verify_stage2.py to use recipe['k'] instead of hard-coded top-2. Ultra_debug already does this correctly. |

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

**Updated Baseline (2025-07-15 diagnostic sweep):**

| Metric | Uniform Avg (Floor) | Best Gate | Oracle (Ceiling) | Gap Captured |
|--------|:-------------------:|:---------:|:----------------:|:------------:|
| Balanced Acc | 43.35% | **43.59%** | 51.96% | **+0.24 pp / 8.61 pp** |
| Head Acc | 70.69% | 70.29% | 79.68% | −0.40 pp |
| Med Acc | 43.74% | 43.97% | 53.57% | +0.23 pp |
| Tail Acc | 11.00% | **12.00%** | 17.75% | **+1.00 pp / 6.75 pp** |
| NLL | 1.202 | 1.214 | — | +0.012 (worse) |
| Brier | 0.427 | 0.430 | — | +0.003 (worse) |
| Tail-ECE | 0.291 | 0.287 | 0.088 | −0.004 (same) |
