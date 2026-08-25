# Current Task Context (Daily Update)

## 1. ACTIVE TASK
- **Task**: Diagnostic extension completed: DaWin assumption verified and violated. Transitioning to next verification step.
- **Objective**: Verify whether 192-dim penultimate embeddings carry more diverse per-expert information than the 316-dim probability features. If yes, implement penultimate feature routing. If no, pivot to expert diversification strategies.
- **Deadline/Phase**: Core Research Phase (Stage 2 of 3) — Diagnostic sub-phase ✅ COMPLETE; Design sub-phase ✅ COMPLETE; Candidate evaluation sub-phase: **DaWin tested — FAILED**; **Embedding correlation check — NEXT**.

## 2. BACKGROUND
- **Why This Task**: Six experiments (Exp 10–15) plus a dedicated feature-correlation diagnostic (Exp 16) traced the routing failure to an **SNR-to-sample-size mismatch** in the 316-dim probability feature space. DaWin confidence routing was proposed as the top candidate (training-free, 3-dim). Exp 17 tested the DaWin assumption and found it **violated**: the confidently-wrong rate is 57.96% overall (89.67% on tail), and DaWin underperforms the uniform baseline (42.75% vs 42.88%). The remaining candidate is penultimate feature routing, but it requires a correlation check on the 192-dim embedding space first.
- **Previous Work**: Diagnostic completed (Exp 12–16). DaWin assumption tested and failed (Exp 17). Experts confirmed individually well-trained (CE=38.94%, LA=40.70%, BS=39.35%) and collectively beat the paper's uniform baseline (43.61% vs 43.28%). The 81.6% "all experts wrong" rate on confidently-wrong samples reveals a hard ceiling of ~48% of test samples where no routing method can help.
- **Related Issues**: See `experiment-log.md` for full experimental history. See `context_files/project-context.md` for architecture overview and diagnostic findings.

## 3. IMPLEMENTATION PLAN
- **Approach**: **Stop designing. Verify remaining candidate before implementing.** DaWin is ruled out. The next candidate (penultimate feature routing) depends on whether the 192-dim embedding space is more diverse than the probability space. This must be measured before any training. The decision tree below governs next actions.
- **Coding Constraints**: All code must strictly follow Object-Oriented Programming (OOP) and modular design principles.

## 4. CURRENT STATUS
- **Progress**: **60%** (Diagnostic phase: Exp 12–16 complete ✅; DaWin assumption verification complete — failed ❌; Remaining candidate identification: penultimate feature routing — pending embedding correlation check).
- **Blockers**: **Two confirmed bottlenecks that cannot be fixed within the current feature spaces:**
  1. **316-dim probability features have an SNR bottleneck** (70% shared variance, routing signal too diluted for 1,125 samples). No target/architecture change on these features can work.
  2. **DaWin assumption violated** (confidently-wrong rate 57.96%). Confidence is not a reliable routing signal because experts share the same backbone and training data, producing correlated probability outputs.
  3. **On 81.6% of confidently-wrong samples, all three experts are wrong together** — this is a hard ceiling: ~48% of test samples have no correct expert, so no routing method can salvage them. The remaining routing opportunity is at most ~8.5 pp (oracle ceiling 52.09% - uniform 43.61%), and the current gate already captures 0.65 pp of that.
- **Next Step**: **Run 192-dim embedding correlation check (Exp 18, Phase A).**

### Immediate Action: Penultimate Embedding Correlation Check

Modify `ExpertEnsemble.forward()` to return per-expert `hidden_list` (3 × 64-dim embeddings) — currently the `hidden_list` is collected but discarded after averaging. Then run the same correlation diagnostics as `diagnose_feature_collinearity.py` on the 192-dim embedding space:

1. Add a `return_hidden_list=True` option to `ExpertEnsemble.forward()`.
2. On 500+ test samples, extract the three 64-dim embeddings per sample.
3. Compute per-sample pairwise correlations between the three embedding blocks.
4. Compute effective rank and variance decomposition (top-5 PC variance share).
5. Compare against the 316-dim probability space baselines (r ≈ 0.68, 70% shared variance).

**Cost:** ~50 lines of Python (modify `ExpertEnsemble`, write diagnostic loop). No training. Uses existing checkpoint infrastructure.

### Decision Tree

```
Run 192-dim embedding correlation check (Exp 18, Phase A)
├── Cross-expert embedding correlation r < 0.5
│   └── Embeddings ARE more diverse than probabilities.
│       └── Implement penultimate feature routing (Exp 19, Linear(192,3), 579 params)
│           ├── If bal acc > 44.26% → Penultimate features help. Continue.
│           └── If bal acc ≈ 44.26% → Features not the issue; diversity is the limit.
│
└── Cross-expert embedding correlation r ≥ 0.5
    └── Embeddings share same correlation problem as probabilities.
        ├── Pivot to expert diversification (RIDE-style diversity losses)
        │   Requires re-training Stage 1 experts with diversity objectives.
        └── Or use non-parametric SADE test-time optimization (no training, high inference cost)
```

## 5. TECHNICAL DECISIONS & WORKFLOW RULES
- **Decision Log**:
  - [COMPLETED] **Diagnostic phase completed on 2025-07-22.** All bottlenecks conclusively identified:
    1. Gradient starvation (Exp 12) — FIXED
    2. Overparameterization (Exp 13) — FIXED
    3. SNR bottleneck (Exp 14→16) — **FUNDAMENTAL** — not fixable within 316-dim probability feature space
    4. L2D calibrators data-limited (Exp 15) — FAILED
  - [COMPLETED] **Research phase completed.** Ranked report of 10 candidate routing strategies produced, top-3 identified.
  - [COMPLETED] **Exp 17 — DaWin assumption tested.** Confidently-wrong rate = 57.96%. 81.6% of confidently-wrong samples have all 3 experts wrong. DaWin (42.75%) < Uniform (42.88%). **DaWin ruled out — not viable for this ensemble.**
  - Expert training quality verified: individual experts (CE=38.94%, LA=40.70%, BS=39.35%) are within expected range. Uniform ensemble (43.61%) beats paper baseline (43.28%). **Stage 1 training is correct.**

### Candidate Strategies (Updated After Exp 17)

#### ❌ TESTED — DaWin Confidence Routing (Training-Free, 3-Dim) — FAILED
- **Status:** Tested in Exp 17. Confidently-wrong rate 57.96% (89.67% on tail). DaWin (42.75%) underperforms uniform (42.88%). Assumption violated.
- **Root cause:** Confidence is not a reliable routing signal for this ensemble because experts share the same backbone and training data, producing correlated probability outputs. On tail classes, no expert has reliable knowledge (5–6 training samples/class), so confidence is essentially random.

#### #1 (Pending) — Penultimate Feature Routing (192-Dim, Linear Router)
- **Core idea:** Replace 316-dim probability features with 192-dim per-expert penultimate embeddings (3 × 64-dim). Route with Linear(192,3) — 579 params.
- **Parameters:** 579 — well under 2K limit.
- **Rationale:** Penultimate features carry richer, less-correlated information than L2-normalized probabilities. The embedding space is not constrained by softmax over the same 100 classes, so cross-expert correlation should be substantially lower than r ≈ 0.68.
- **Prerequisite verification required (Exp 18, Phase A):** Run feature correlation diagnostic on 192-dim embeddings first. If cross-expert r < 0.5, proceed. Requires pipeline change to `ExpertEnsemble.forward()` (hidden_list already collected but discarded).

#### #2 (Pending) — Stats-Only Logistic Router (13-Dim, 42 Params)
- **Core idea:** Discard all 300 probability dims. Route using only the 9 high-variance stats dims + 4 frequency features (13 total). Linear(13,3) — 42 params.
- **Parameters:** 42 — virtually immune to overfitting.
- **Rationale:** The diagnostic showed stats dims have 8.4× higher variance than prob dims, and freq dims have 11× higher variance. If the 13-dim logistic router matches the 316-dim MLP (44.16%), it proves the 300 probability dims contribute no routing signal.
- **Verification:** Modify `GateTrainer` config to accept `gate_input_mode: stats_only`. Compare bal acc against 316-dim baseline. If within 0.2 pp, SNR hypothesis is confirmed.

### Root Cause Chain (Complete — Updated)
1. Gradient starvation (Exp 12) → FIXED by logprob target
2. Overparameterization (Exp 13) → FIXED by linear router
3. **SNR bottleneck** (Exp 14→16) → **FUNDAMENTAL** — 70% shared variance, routing signal too diluted for 1,125 samples
4. L2D calibrators data-limited (Exp 15) → FAILED
5. **DaWin assumption violated** (Exp 17) → ❌ FAILED — confidently-wrong rate 57.96%, DaWin < Uniform

**Best known config reverted:**
- `gate_target_mode: logprob`
- MLP architecture
- `expert_temperatures: [1.5, 1.2, 1.5]`
- k=3, logit mixing

## 6. RESOURCES
- **Reference Code**: `imbalanceddl/strategy/_gate_trainer.py` (ExpertEnsemble.forward — hidden_list already collected), `ultra_debug.py` (Exp 17 diagnostic function), `diagnose_feature_collinearity.py` (methodology to reuse for embedding correlation), `inspect_gate_gradients.py`
- **Documentation**: `experiment-log.md` (Exp 12-17), `context_files/project-context.md` (Diagnostic Findings table), ranked research report (top-10 candidates).
- **Related PRs/Issues**: Local development branch `expert`.
