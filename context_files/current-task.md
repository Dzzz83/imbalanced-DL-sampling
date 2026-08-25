# Current Task Context (Daily Update)

## 1. ACTIVE TASK
- **Task**: Research phase: evaluate routing candidates. The ranked research report is complete; transition to implementing the recommended strategy.
- **Objective**: Evaluate and implement a new routing strategy that bypasses the 316-dim calibrated probability feature space, starting with the highest-confidence, lowest-risk candidate.
- **Deadline/Phase**: Core Research Phase (Stage 2 of 3) — Diagnostic sub-phase ✅ COMPLETE; Design sub-phase ✅ COMPLETE; **Candidate evaluation sub-phase — IN PROGRESS**.

## 2. BACKGROUND
- **Why This Task**: Six experiments (Exp 10–15) plus a dedicated feature-correlation diagnostic script (Exp 16) have traced the routing failure to its root cause: an **SNR-to-sample-size mismatch** where 70% of 316-dim feature variance is shared across experts and the residual expert-discriminative signal is too diluted for 1,125 training samples. No target function or architecture can overcome this within the 316-dim calibrated probability feature space.
- **Previous Work**: Diagnostic completed (Exp 12–16). Research phase completed (ranked report of 10 candidate routing strategies, evaluated against the confirmed SNR bottleneck). Top-3 strategies identified: (1) DaWin confidence routing (training-free, 3-dim), (2) Penultimate feature routing (192-dim embeddings), (3) Stats-only logistic routing (13-dim, 42 params).
- **Related Issues**: See `experiment-log.md` for full experimental history. See `context_files/project-context.md` for architecture overview and candidate design notes.

## 3. IMPLEMENTATION PLAN
- **Approach**: **Stop debugging. Stop designing. Evaluate empirically.** The ranked research report identified DaWin-style confidence routing (3-dim, training-free) as the strongest candidate. The immediate next step is to implement a DaWin baseline in `ultra_debug.py` as a post-hoc evaluation — no training pipeline changes needed. The decision tree below governs follow-up actions based on the DaWin result.
- **Coding Constraints**: All code must strictly follow Object-Oriented Programming (OOP) and modular design principles.

## 4. CURRENT STATUS
- **Progress**: **100%** (Diagnostic phase complete — 6 experiments, 4 bottlenecks identified, all verified). **Milestone**: Research phase: candidate evaluation.
- **Blockers**: The SNR bottleneck is confirmed and understood (70% shared variance, 30% routing signal diluted across ~66 components). No further blocking issues remain — the path forward is clear: evaluate DaWin, then iterate.
- **Next Step**: **Implement DaWin baseline on existing checkpoint.**

### Immediate Action: DaWin Confidence Routing (Post-hoc Baseline)

Modify `ultra_debug.py` to add a DaWin-routed mixture column:

1. Extract per-expert max-prob from per-expert probability arrays: `confidences = np.stack([p_ce.max(axis=1), p_la.max(axis=1), p_bs.max(axis=1)], axis=1)` → shape `(N, 3)`.
2. Grid search temperature T̂ on the tune set (625 samples) to maximize balanced accuracy. Candidate T values: [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0].
3. Compute DaWin weights: `w_dawin = softmax(confidences / T̂, axis=1)`.
4. Build mixture: `p_dawin = build_mixture(logits, w_dawin, ...)`.
5. Compare: DaWin bal acc vs Gate MLP bal acc (44.26%) vs Uniform bal acc (43.61%).

**Cost:** ~10 lines of Python. Zero parameters. Zero training. No GateTrainer changes.

### Decision Tree

```
Run DaWin baseline on existing checkpoint (Exp 17)
├── DaWin ≥ 44.26% → DaWin is the routing mechanism. STOP.
│   Total gain: +0.65 pp over uniform. No learned router needed.
│
├── 44.0% ≤ DaWin < 44.26% → Confidence captures most signal.
│   └── Train 3-param per-expert temperature router (Exp 18)
│       ├── If > DaWin by ≥0.5 pp → calibration mismatches matter
│       └── If ≈ DaWin → confidence already captures calibration
│
└── DaWin < 44.0% → Confidence routing insufficient.
    └── Implement penultimate feature routing (Exp 19, 192-dim)
        ├── First: diagnose 192-dim feature correlations
        │   ├── If r < 0.5 → proceed with penultimate routing
        │   └── If r ≥ 0.5 → features still correlated; explore
        │       DES competence routing or SADE test-time optimization
        └── If all learned routers fail → DaWin + disagreement mask (prior-free hybrid)
```

## 5. TECHNICAL DECISIONS & WORKFLOW RULES
- **Decision Log**:
  - [COMPLETED] **Diagnostic phase completed on 2025-07-22.** All bottlenecks conclusively identified:
    1. Gradient starvation (Exp 12) — FIXED
    2. Overparameterization (Exp 13) — FIXED
    3. SNR bottleneck (Exp 14→16) — **FUNDAMENTAL** — not fixable within 316-dim probability feature space
    4. L2D calibrators data-limited (Exp 15) — FAILED
  - [COMPLETED] **Research phase completed.** Ranked report of 10 candidate routing strategies produced, each evaluated against the confirmed SNR bottleneck. Top-3 strategies identified and documented below.

### Candidate Strategies (Ranked Top-3)

#### #1 — DaWin-Style Confidence Routing (Training-Free, 3-Dim)
- **Core idea:** Route by `w_j(x) = softmax(conf_j(x) / T̂)` where `conf_j(x) = max_k p_j(k|x)`.
- **Parameters:** 0 trainable (just T̂ scalar from tune set grid search).
- **Rationale:** Completely bypasses the 316-dim feature space. The 3 confidence scores carry the strongest expert-discriminative signal (diagnostic: stats dims have 8.4× higher variance than prob dims). LA-saves-the-day samples show w_LA = 0.683 when LA's confidence is high — exactly DaWin's mechanism. DaWin (Oh et al., NeurIPS 2024) proves this beats learned routers for frozen experts.
- **Verification:** Compare DaWin bal acc vs current gate MLP bal acc (44.26%). If ≥ 44.26%, DaWin is the answer.

#### #2 — Penultimate Feature Routing (192-Dim, Linear Router)
- **Core idea:** Replace 316-dim probability features with 192-dim per-expert penultimate embeddings (3 × 64-dim). Route with Linear(192,3) — 579 params.
- **Parameters:** 579 — well under 2K limit.
- **Rationale:** Penultimate features carry richer, less-correlated information than L2-normalized probabilities. The embedding space is not constrained by softmax over the same 100 classes, so cross-expert correlation should be substantially lower than r ≈ 0.68.
- **Verification:** Run feature correlation diagnostic on 192-dim embeddings first. If cross-expert r < 0.5, proceed. Requires pipeline change to `ExpertEnsemble.forward()` (hidden_list already collected but discarded).

#### #3 — Stats-Only Logistic Router (13-Dim, 42 Params)
- **Core idea:** Discard all 300 probability dims. Route using only the 9 high-variance stats dims + 4 frequency features (13 total). Linear(13,3) — 42 params.
- **Parameters:** 42 — virtually immune to overfitting.
- **Rationale:** The diagnostic showed stats dims have 8.4× higher variance than prob dims, and freq dims have 11× higher variance. If the 13-dim logistic router matches the 316-dim MLP (44.16%), it proves the 300 probability dims contribute no routing signal.
- **Verification:** Modify `GateTrainer` config to accept `gate_input_mode: stats_only`. Compare bal acc against 316-dim baseline. If within 0.2 pp, SNR hypothesis is confirmed.

### Root Cause Chain (Complete)
1. Gradient starvation (Exp 12) → FIXED by logprob target
2. Overparameterization (Exp 13) → FIXED by linear router
3. **SNR bottleneck** (Exp 14→16) → **FUNDAMENTAL** — 70% shared variance, routing signal too diluted for 1,125 samples
4. L2D calibrators data-limited (Exp 15) → FAILED

**Best known config reverted:**
- `gate_target_mode: logprob`
- MLP architecture
- `expert_temperatures: [1.5, 1.2, 1.5]`
- k=3, logit mixing

## 6. RESOURCES
- **Reference Code**: `imbalanceddl/strategy/_gate_trainer.py`, `ultra_debug.py`, `inspect_gate_gradients.py`, `diagnose_feature_collinearity.py`
- **Documentation**: `experiment-log.md` (Exp 12-16), ranked research report (top-10 candidates), DaWin (Oh et al., NeurIPS 2024).
- **Related PRs/Issues**: Local development branch `expert`.
