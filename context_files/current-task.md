# Current Task Context (Daily Update)

## 1. ACTIVE TASK
- **Task**: Debugging why the current method failed.
- **Objective**: Run a debugging script, obtain debugging data, diagnose the problem.
- **Deadline/Phase**: Core Research Phase (Stage 2 of 3)

## 2. BACKGROUND
- **Why This Task**: The routing mechanism fails to route correctly as the balanced accuracy and tail accuracy is not much better than the uniform baseline and is far from Oracle Accuracy.
- **Previous Work**: Stage 1 is complete. Three independent experts (CE, LA tau=1.5, BS) were trained.
- **Related Issues**: We are strictly in the Diagnostic Phase. No fixes will be proposed until empirical evidence is gathered.

## 3. IMPLEMENTATION PLAN
- **Approach**: Institute the "Diagnose -> Hypothesis -> Verification -> Targeted Fix" workflow.
- **Coding Constraints**: All code must strictly follow Object-Oriented Programming (OOP) and modular design principles.

## 4. CURRENT STATUS
- **Progress**: **90%** (Diagnostic phase complete — 5 experiments, all bottlenecks mapped).
- **Blockers**: **Feature collinearity** — the 316-dim calibrated probability features from the same ResNet32 backbone are near-collinear. No target (logprob, correctness) or architecture (MLP, linear) can extract per-sample routing signal from these features. The gate can only learn per-class biases.
- **Next Step**: Change gate input features to penultimate features (192-dim embeddings), or implement DaWin-style confidence routing (3-dim).

## 5. TECHNICAL DECISIONS & WORKFLOW RULES
- **Decision Log**:
  - Decided to abandon "black-box" random architecture search.
  - Decided to implement the "Autopsy -> Hypothesis -> Verification" workflow.
  - Decided to write diagnostic scripts on existing checkpoints before any new training runs.
  - Decided to implement Probability Routing + Switch Loss (Failed in Exp 11).
  - Reverted from premature Logit-Space Mixing fix to respect the Diagnostic Phase. Must verify gradient magnitudes before proposing a fix for the uniform collapse.
  - **Exp 12** (mix_nll): Confirmed gradient starvation. Gradient norm = 0.251. Fixed by logprob target.
  - **Exp 13** (logprob + MLP): Fixed gradient starvation (norm = 0.892), but MLP overparameterized (20k params / 1,125 samples, fc.weight std = 0.074). Fixed by linear router.
  - **Exp 14** (logprob + linear): Fixed overparameterization (951 params, fc.weight std = 0.20), but **feature collinearity** confirmed: three calibrated probability distributions from same backbone are near-collinear. Blocks indistinguishable (0.201, 0.203, 0.200). Linear (43.97%) ≈ MLP (44.16%).
  - **Exp 15** (correctness + linear): L2D correctness targets failed — isotonic calibrators on 625 tune samples produce near-uniform targets. Gate temp = 2.200 (highest ever). Routing collapsed below baseline (43.34%). Theoretical guarantees don't hold under data-limited conditions.

**Root cause chain (complete — all bottlenecks identified and tested):**
1. Gradient starvation (Exp 12) → FIXED by logprob target
2. Overparameterization (Exp 13) → FIXED by linear router
3. Feature collinearity (Exp 14) → **PERSISTS** — no target/architecture change can fix this
4. L2D calibrators data-limited (Exp 15) → FAILED — additional bottleneck

**Best known config reverted:**
- `gate_target_mode: logprob` (not correctness)
- MLP architecture (not linear router) — though MLP (44.16%) ≈ linear (43.97%)
- `expert_temperatures: [1.5, 1.2, 1.5]`
- k=3, logit mixing

## 6. RESOURCES
- **Reference Code**: `imbalanceddl/strategy/_gate_trainer.py`, `ultra_debug.py`, `inspect_gate_gradients.py`
- **Documentation**: "Mixture of Experts for Long-Tailed Visual Recognition" (MoE-LT), "Sparsely-Gated MoE", "BalPoE", "SADE".
- **Related PRs/Issues**: Local development branch `expert`.