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
- **Progress**: 10% (Diagnostic phase initiated).
- **Blockers**: Haven't have enough data to identify the blockers.
- **Next Steps**: Diagnose the problem.

## 5. TECHNICAL DECISIONS & WORKFLOW RULES
- **Decision Log**:
  - Decided to abandon "black-box" random architecture search.
  - Decided to implement the "Autopsy -> Hypothesis -> Verification" workflow.
  - Decided to write diagnostic scripts on existing checkpoints before any new training runs.
  - Decided to implement Probability Routing + Switch Loss (Failed in Exp 11).
  - Reverted from premature Logit-Space Mixing fix to respect the Diagnostic Phase. Must verify gradient magnitudes before proposing a fix for the uniform collapse.

## 6. RESOURCES
- **Reference Code**: `imbalanceddl/strategy/_gate_trainer.py`, `ultra_debug.py`, `inspect_gate_gradients.py`
- **Documentation**: "Mixture of Experts for Long-Tailed Visual Recognition" (MoE-LT), "Sparsely-Gated MoE", "BalPoE", "SADE".
- **Related PRs/Issues**: Local development branch `expert`.