# Current Task Context (Daily Update)

## 1. ACTIVE TASK
- **Task**: Implement the P0 fix (switch to logprob target) and re-train the gate.
- **Objective**: Switch `gate_target_mode` from `mix_nll` to `logprob` in the config, align train/eval k, and run a new gate sweep. Compare metrics against the diagnostic baseline.
- **Deadline/Phase**: Core Research Phase (Stage 2 of 3)

## 2. BACKGROUND
- **Why This Task**: The diagnostic phase is complete. Three scripts (`verify_stage2.py`, `ultra_debug.py`, `inspect_gate_gradients.py`) have confirmed the root cause: mixture NLL gradient vanishes on tail samples because `∂L/∂g_j = w_j · (p_mix(y) − p_j(y))` ≈ 0 when all `p_j(y|x)` are tiny.
- **Previous Work**: Diagnosis revealed:
  - RC4a (primary): Gradient starvation on tail samples for mixture NLL
  - RC2 (secondary): Train k=3 vs eval k=2 protocol mismatch
  - RC6b (tertiary): expert_temps=[1.5, 1.5, 1.5] nullifies per-expert calibration
  - RC4b: disagree_weight + kl_uniform=3.0 fight the remaining gradient signal
- **Related Issues**: The log-space sharpened target (`build_oracle_target` with `space='logprob'`) is already implemented and ready to use. No new code needed — just a config change.

## 3. IMPLEMENTATION PLAN
- **Approach**: Follow the "Targeted Fix" phase — use the empirical evidence from diagnosis to guide changes.
- **Coding Constraints**: All code must strictly follow Object-Oriented Programming (OOP) and modular design principles.

## 4. CURRENT STATUS
- **Progress**: 80% (Diagnostic phase completed successfully; moving to Targeted Fix).
- **Blockers**: None identified for the P0 changes.
- **Next Steps**:
  1. Change `gate_target_mode: mix_nll` → `gate_target_mode: logprob` in config
  2. Set `gate_disagree_weight: false` and `gate_kl_uniform: 0.0` (these fight the logprob gradient)
  3. Run new gate sweep
  4. Evaluate with `ultra_debug.py` and compare metrics against diagnostic baseline
  5. If tail bal acc improves >1 pp, the logprob target is confirmed as the correct fix

## 5. TECHNICAL DECISIONS & WORKFLOW RULES
- **Decision Log**:
  - Decided to abandon "black-box" random architecture search.
  - Decided to implement the "Autopsy -> Hypothesis -> Verification" workflow.
  - Decided to write diagnostic scripts on existing checkpoints before any new training runs.
  - ✅ **Diagnostic phase completed on 2025-07-15.** Three root causes confirmed:
    1. Mixture NLL gradient starvation on tail samples (RC4a — primary)
    2. Train/eval k mismatch (RC2 — secondary)
    3. Equal expert_temps nullify calibration (RC6b — tertiary)
  - Decided to proceed with **P0 fix: switch to logprob target** (one config change).
  - Log-space sharpening already implemented in `build_oracle_target` with `space='logprob'`.

## 6. RESOURCES
- **Reference Code**: `imbalanceddl/strategy/_gate_trainer.py`, `ultra_debug.py`, `inspect_gate_gradients.py`
- **Documentation**: `literature_review_moe_routing.md` (sections §9, §0.5, §2.5)
- **Diagnostic Results**: `experiment-log.md` (Experiment 12), `verify_stage2.py` output
