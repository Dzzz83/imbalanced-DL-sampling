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
- **Progress**: **85%** (Root cause chain fully mapped — see below).
- **Blockers**: None — root cause identified. Next fix is ready.
- **Next Step**: Switch to correctness targets (`gate_target_mode: correctness`).

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

**Root cause chain (complete):**
1. Gradient starvation (Exp 12) → FIXED by logprob target
2. Overparameterization (Exp 13) → FIXED by linear router
3. Feature collinearity (Exp 14) → NEXT FIX: correctness targets (L2D)

**Current diagnosis evidence (Exp 14):**
- fc.weight blocks: CE=0.201, LA=0.203, BS=0.200 — statistically identical means/std
- Per-class routing shows class-level biases (class 0: w_LA=0.177; class 99: w_LA=0.455) but no per-sample specialization
- LA-saves-the-day routing works (w_LA=0.583 on 136 samples) — proves gate can specialize when signal exists
- Oracle gap: 8.12 pp remaining
- Paper gap: our NLL (1.222) is 4.22% worse than paper (1.18); our tail-ECE (0.336) is 0.248 worse than paper (0.088)

## 6. RESOURCES
- **Reference Code**: `imbalanceddl/strategy/_gate_trainer.py`, `ultra_debug.py`, `inspect_gate_gradients.py`
- **Documentation**: "Mixture of Experts for Long-Tailed Visual Recognition" (MoE-LT), "Sparsely-Gated MoE", "BalPoE", "SADE".
- **Related PRs/Issues**: Local development branch `expert`.