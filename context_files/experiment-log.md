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

See `experiment-log.md` (root) for the full experiment history (Exp 10–17) with complete entries, diagnostic trace, and baseline tables.

---

### Experiment 17: DaWin Assumption Verification — Confidently-Wrong Diagnostic
**Date:** 2025-07-22
**Hypothesis:** DaWin confidence routing (routing by `softmax(conf_j / T̂)`) is safe for this ensemble because the most confident expert is usually correct.
**Changes:** No training run. The `run_confident_wrong_diagnostic` function (added to `ultra_debug.py`) was executed on the existing epoch-82 checkpoint (MLP, logprob target, [1.5, 1.2, 1.5] temps).
**Results:**
- Confidently-wrong rate (overall): **57.96%** — the most confident expert is wrong more often than correct
- By group: Head 30.00% | Mid 58.75% | **Tail 89.67%**
- Avg conf when correct: 0.8464 | Avg conf when wrong: 0.6245
- Most-confident expert identity: CE=27.6% (39.3% correct), LA=51.2% (45.4% correct), BS=21.1% (37.5% correct)
- **On 81.6% of confidently-wrong samples, all three experts are wrong together** (3,784/4,637 samples)
- DaWin simulation: best T̂ = 1.00 (tune bal acc 42.95%)
- **DaWin test bal acc: 42.75% vs Uniform: 42.88% vs Gate MLP: 44.26%** — DaWin underperforms uniform
- Expert individual bal acc: CE=38.94% | LA=40.70% | BS=39.35% | Uniform=43.61%
- Uniform ensemble (43.61%) beats paper's reported uniform baseline (43.28%) — experts are well-trained
- Oracle bal acc: 52.09% (confirming expert diversity exists but is hidden in features)
**Outcome:** ❌ FAILED (DaWin assumption conclusively violated)
**Root Cause / Why It Worked:** The most confident expert is wrong 58% of the time because all three experts share the same backbone and training data — their probability outputs are too correlated for confidence to be a reliable routing signal. On tail classes (89.67% confidently-wrong), no expert has learned reliable features (5–6 samples/class). The 81.6% "all experts wrong" rate reveals a hard ceiling: on ~48% of test samples, no expert knows the correct answer.
**Next Step:** Abandon DaWin. Proceed to 192-dim embedding correlation check (Exp 18, Phase A) to determine viability of penultimate feature routing.

---

# FAILED APPROACHES — DO NOT SUGGEST

| # | Approach | Root Cause of Failure | Exp # | Date |
|---|----------|----------------------|-------|------|
| 1 | Mini-MLP with BatchNorm on Raw Logits | Non-linear architecture did not fix the uniform weight distribution. | 10 | 2024-05-29 |
| 2 | Probability Routing + Switch Load Balancing Loss | Switch loss balanced weights but probability-space mixing caused uniform routing collapse. | 11 | 2024-05-30 |
| 3 | Mixture NLL in logit space + disagree_weight + kl_uniform=3.0 | Gradient vanishes on tail samples. Regularizers fight remaining signal. | 12 | 2025-07-15 |
| 4 | Logprob target + linear router + balanced temps [1.5, 1.2, 1.5] | **SNR bottleneck**: 70% shared variance; routing signal too diluted for 1,125 samples. | 14, 16 | 2025-07-22 |
| 5 | Correctness targets (L2D) + linear router + balanced temps [1.5, 1.2, 1.5] | Isotonic calibrators on 625 tune samples produce near-uniform targets; gate temp=2.200. | 15 | 2025-07-18 |
| 6 | **DaWin confidence routing** (training-free, 3-dim) | Confidently-wrong rate = 57.96% (89.67% on tail). 81.6% of conf-wrong samples have all 3 experts wrong. DaWin (42.75%) < Uniform (42.88%). Assumption violated. | 17 | 2025-07-22 |

---

# IDEAS NOT YET TRIED (Remaining Candidates)

| # | Idea | Status | Priority | Notes |
|---|------|--------|----------|-------|
| 1 | Disagreement Gating | Proposed | LOW | Only train/invoke the gate when experts disagree. |
| 2 | Two-Stage Routing (RIDE-style) | Proposed | LOW | Default = uniform mixture. Only activate per-sample router when disagreement is high. |
| 3 | Max-KL Routing (track the third expert) | Proposed | LOW | Change verify_stage2.py to use recipe['k'] instead of hard-coded top-2. |

---

# PLANNED EXPERIMENTS (Ranked by Priority)

The following experiments are ordered after the completion of Exp 17 (DaWin — failed assumption). The next step is to verify whether penultimate feature routing is viable.

| Order | Experiment | Strategy | Verification Step | Expected Outcome & Next Action |
|:-----:|:-----------|:---------|:------------------|:------------------------------|
| **1** | **Exp 18 — Embedding Correlation Check** | **Diagnose per-expert penultimate feature diversity (192-dim)** | Extract 3 × 64-dim hidden_list from `ExpertEnsemble.forward()`. Compute per-sample correlations, effective rank, variance decomposition. | **If r < 0.5:** Proceed to Exp 19. **If r ≥ 0.5:** Pivot to expert diversification. |
| **2** | **Exp 19 — Penultimate Feature Routing** | **192-dim embeddings + Linear(192,3)** | Train linear router (579 params) on penultimate embeddings. Compare vs gate (44.26%) and uniform (43.61%). | **If > 44.26%:** Penultimate features help. **If ≈ 44.26%:** Diversity is the fundamental limit. |
| **3** | **Exp 20 — Stats-Only Logistic Router** | **13-dim stats + Linear(13,3)** | Add `gate_input_mode: stats_only`. Train 42-param router. Compare vs 316-dim baseline. | **If within 0.2 pp:** SNR hypothesis confirmed. |

### Fallback Options

| Candidate | When to Consider |
|:----------|:-----------------|
| **SADE Test-Time Optimization** | If all learned routers fail |
| **DES Competence Routing** | If penultimate features have meaningful distance structure |
| **Expert Diversification** (RIDE-style losses) | If Exp 18 shows embeddings also correlated (r ≥ 0.5) |

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
