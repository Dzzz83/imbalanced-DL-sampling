# Round-3 Diagnosis & Fixes — The Gate Now Beats Uniform (barely); What Remains (8/26 run)

**Setup that produced these results:** round-2 fixes active — `k=3`, `mix_nll` logit-space loss,
`gate_disagree_weight: true`, `gate_dropout: 0.1`, `gate_kl_uniform: 0.0` (not yet enabled),
per-expert temps `[1.5, 1.5, 1.5]`, fitted gate temperature.

---

## 0. Executive summary

**The round-2 fixes worked.** k=3 removed the top-2 truncation damage (Many −2.04 → −0.82),
disagreement weighting + dropout cut the noise, and the method now **beats its own uniform
baseline** on both success criteria: bal 43.50 vs 43.47 (+0.03), Low 12.08 vs 11.42 (+0.67) on
test; best checkpoint 43.38 vs 43.35 ✅ on full val.

**But the win is fragile (+0.03 pp) and two problems remain:**

- **P1 — the gate's raw output is still bad; a fitted temperature is doing all the work.**
  "Method @ T=1.0" (gate at its natural sharpness) gets **42.375** bal vs 43.500 at the fitted
  (soft) gate temperature — a 1.1 pp swing from softening alone. The tune set still wants the gate
  as soft as the grid allows. `gate_kl_uniform` was 0.0 in this run — the regularization that
  would internalize this softening was not enabled.
- **P2 — the tail signal is real but diluted.** **LA alone gets Low 12.21 — higher than the
  method's 12.08.** The uniform mixture gets 11.42. On the 136 LA-saves-day samples, the gate
  gives the top weight to a *wrong* expert on roughly half of them (samples 156/414/538: CE gets
  0.54–0.79; 477/646/751: BS gets 0.44–0.71), because its confidence-driven features are
  anti-predictive there and it has **no explicit head/tail frequency signal** — it tries to infer
  "this looks like a tail sample" from 100-dim distributions learned on ~11 samples per class.

Additionally, code review found three smaller issues (fixed): a `torch.tensor` warning in the
T=1.0 comparison table, the verify scripts building the gate with a fixed input dim (old and new
checkpoints can no longer coexist in one sweep folder), and a **config interaction**: with
`gate_disagree_weight: true` and `gate_kl_uniform: 0.0`, the ~40% of samples where all experts
agree get **zero loss**, so the gate's weights on those samples freeze at their random init —
harmless for accuracy (the prediction is fixed) but a real NLL/calibration penalty.

---

## 1. What the numbers show (test split)

| | Uniform | Method (round-2) | Delta | Round-1 delta |
|---|---|---|---|---|
| Bal | 43.47 | 43.50 | **+0.03** | −0.69 |
| Many | 71.04 | 70.21 | −0.82 | −2.04 |
| Med | 43.39 | 43.71 | +0.32 | −0.86 |
| Low | 11.42 | 12.08 | **+0.67** | +1.08 |
| NLL | 1.196 | 1.223 | +0.027 | +0.075 |

- The head loss is now *only* the gate's weight deviations (k=3 keeps all votes); it costs −0.82
  Many, still the largest single drag.
- Tail: 12.08 vs LA-alone 12.21 vs oracle-tail 17.75. The gate captures part of the LA signal but
  dilutes it with wrong-expert weight on ~half the saves-day samples.
- Oracle bal 51.96 — 8.5 pp of headroom remains, most of it per-sample structure the current
  features cannot express.

---

## 2. Diagnosis

### P1 — the gate learns overconfident noise; softening is a crutch, not a fix

Evidence chain:
1. `gate_temp` was fitted at the *soft edge of the grid* in the round-1 run (3.0); the extended
   grid in round-2 exists precisely because of that. "Method @ T=1.0" = 42.375 vs 43.500 at the
   fitted temp: **1.1 pp of the method's score is a temperature knob, not routing**.
2. The gate trains on ~1.1k class-balanced samples (~11/class): per-class preferences are noise
   (class 0 → BS 0.45 — BS is the weakest head expert at 65.79 Many vs CE 66.75).
3. With `gate_kl_uniform: 0.0`, nothing during training penalizes confident deviations; the
   fitted temperature has to clean up after the fact.

**Fix:** `gate_kl_uniform: 3.0` — `λ·KL(w(x)‖uniform)` in the loss makes deviation cost real
gradient pressure, so the gate learns to be *inherently* conservative and the fitted temperature
should come back toward 1. (Loss-scale note: the KL term is ~0.005·λ per sample vs NLL ~1.2, but
its *gradient* `λ·(w_j − u_j)` is the right order of magnitude to counter the mixture gradient on
noisy samples. λ=3 is the starting point; H2 in `debug_routing_signal.py` measures the optimum.)

### P2 — the tail signal exists but the gate cannot see the head/tail distinction

Evidence:
- LA alone: Low 12.21 > method 12.08 — a trivial "always LA on tail" rule already beats the
  learned gate on tail.
- Saves-day samples where the gate misroutes: 156 (CE 0.79, wrong, max-prob 0.40), 414 (CE 0.54,
  wrong), 538 (CE 0.64, wrong), 477 (BS 0.59, wrong), 646 (BS 0.71, wrong), 751 (BS 0.44, wrong).
  The gate routes by confidence/entropy patterns that are **anti-predictive on exactly these
  samples** (LA is right with *lower* confidence than the wrong experts).
- The gate input has no explicit "which frequency group does this sample belong to" signal. The
  calibrated probs carry frequency *implicitly* (LA/BS biases), but extracting it requires
  learning a 100-dim pattern per class from ~11 samples.

**Fix (round-3): class-frequency features.** Append the log-prior of each expert's predicted
class and of the uniform mixture's predicted class (+4 dims). "Predicted class is a tail class →
boost LA" becomes a single learnable weight instead of a pattern-matching problem. This is the
frequency-aware-router idea (LTDA-Router, arXiv:2507.01351) applied to a frozen-expert gate. The
gate still decides per-sample (it sees confidence, agreement, entropy), but it now has the one
feature that carries the largest, most reliable routing signal in this dataset.

### P3 — config interaction found during code review (fixed)

`gate_disagree_weight: true` + `gate_kl_uniform: 0.0` ⇒ agree samples (where routing provably
cannot change the prediction) have **zero loss**, so their gate outputs stay at the random
initialization. Prediction is unaffected (all experts share the argmax), but the mixture's
NLL/calibration on those samples uses random weights. The trainer now **warns** when this
combination is detected, and the recommended config keeps `kl_uniform > 0` whenever disagreement
weighting is on.

### Code bugs found and fixed (round-3)

1. `run_temperature_comparison`: `torch.tensor(gate_logits_test)` on an already-detached tensor
   (warning + needless copy) → `gate_logits_test.detach().clone()`.
2. `verify_stage1.py` / `verify_stage2.py`: the gate was constructed once outside the checkpoint
   loop with a fixed input dim — with round-3's variable input dim (312 vs 316), checkpoints from
   different configs in the same sweep folder would fail to load. The gate is now built
   **per-checkpoint** from `recipe['freq_features']`.
3. `recipe_from_checkpoint` now carries `freq_features`; `ultra_debug`, `verify_stage3`,
   `inspect_gate_gradients`, and `debug_routing_signal` all construct the model/gate from the
   recipe, so old (312-dim) and new (316-dim) checkpoints are each evaluated with their own
   architecture.

---

## 3. Implemented fixes (round-3)

| # | Change | Config | Evidence |
|---|---|---|---|
| F1 | **Class-frequency features** (+4 dims: per-expert predicted-class log-prior + uniform-mixture predicted-class log-prior) | `gate_freq_features: true` | P2: LA-alone Low 12.21 > method 12.08; gate misroutes half the saves-day samples |
| F2 | **KL-to-uniform regularization enabled** | `gate_kl_uniform: 3.0` | P1: T=1.0 method = 42.375 vs 43.500 at fitted temp — the gate must learn to be conservative, not be softened afterwards |
| F3 | **Warning** when `disagree_weight` is on with `kl_uniform == 0` | code | P3 |
| F4 | **Per-checkpoint gate construction** in verify scripts + recipe carries `freq_features` | code | bug fix |
| F5 | **H6 benchmark** in `debug_routing_signal.py`: tune-estimated group-conditional routing rule vs the MLP gate | debug script | tells us if per-class/group priors beat the MLP |

Code defaults are unchanged (312-dim, kl=0, freq off) — the behavior only changes when the config
says so. `_gate_train.yaml` now carries the recommended round-3 settings:
`routing_sparsity: 3, gate_dropout: 0.1, gate_disagree_weight: true, gate_kl_uniform: 3.0,
gate_freq_features: true`.

---

## 4. Verification plan for the next iteration

**Step 1 (no training) — run `debug_routing_signal.py` on the round-2 checkpoint.** Key outputs
to confirm the diagnosis:
- **H2 interpolation**: if bal acc rises as weights → uniform, F2 (KL) is confirmed as needed;
  the curve also brackets the right λ.
- **H3 T_gate sweep**: if tune bal acc still rises at the soft end of the grid, the gate is still
  overconfident — with `kl_uniform: 3.0` it should flatten out.
- **H5a oracle match per group**: top-1 match ≈ 1/3 on Head ⇒ noise (KL + freq features are the
  right fix); > 40% on Tail ⇒ the tail signal is real and freq features should amplify it.
- **H6 group-rule benchmark**: if the tune-estimated group rule (tail→LA-heavy, head/med→uniform)
  beats the MLP gate, the group-conditional policy is the safer architecture and the MLP must at
  least match it.

**Step 2 (training) — next Kaggle run with the recommended config.** Success indicators:
- Fitted `gate_temp` should move back toward 1 (no more grid-edge warning).
- Train-vs-val mixture-acc gap should shrink (KL + dropout + disagreement weighting).
- Low acc should approach LA-alone's 12.2+ (freq features let the gate boost LA on predicted-tail
  samples without learning per-class noise); Many should recover toward uniform's 71 (KL pins
  head samples to uniform where the signal is noise).
- NLL/Brier of the method should approach the uniform baseline's (agree-sample weights pinned to
  uniform by the KL term).

**Expected outcome.** The method's score is now "uniform + tail-conditional LA boost − residual
head noise". F1 attacks the tail side (the largest clean gain: Low 11.42 → 12.2+), F2 attacks the
head side (Many −0.82 → ~0). Together they should move bal from +0.03 to roughly +0.3–0.5 pp over
uniform, and make the win robust across checkpoints instead of a single-epoch fluke. The oracle
gap (8.5 pp) will then be dominated by per-sample structure the current features cannot express —
at which point the documented next step is penultimate-feature inputs (RIDE-style) or accepting
the group-conditional policy as the final method.

**Files changed (round-3):** `imbalanceddl/utils/gate_features.py` (freq features),
`imbalanceddl/strategy/_gate_trainer.py` (flag, warning, per-checkpoint gate rebuild, metadata),
`imbalanceddl/utils/debug/models.py`, `imbalanceddl/utils/debug/evaluation.py` (recipe +
`torch.tensor` fix), `imbalanceddl/utils/config.py`, `verify_stage1.py`, `verify_stage2.py`,
`verify_stage3.py`, `ultra_debug.py`, `inspect_gate_gradients.py`, `debug_routing_signal.py` (H6
benchmark), `config/what_to_train/cifar100/_gate_train.yaml`, `smoke_test_gate.py` (freq + kl
paths; all 120+ checks passing).
