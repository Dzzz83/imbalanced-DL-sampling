# Round-2 Diagnosis — Why the Fixed Gate Still Loses to Uniform (8/25 run)

**Setup that produced these results:** the 8/24 fixes active — `mix_nll` loss, logit-space
mixture, per-expert temperatures fit to `[1.5, 1.5, 1.5]`, gate temperature fit to `3.0`, top-2
(`k=2`) truncation, 100 epochs, ~1.1k-sample class-balanced gate split.

---

## 0. Executive summary

The 8/24 fixes **worked mechanically**: the gate now routes per-sample (weights deviate,
`w_LA ≈ 0.38–0.44`, LA-saves-day weight rose 0.46 → 0.53 with 86.8% top-2 inclusion), and the
**uniform baseline is much better calibrated** (bal 42.74 → 43.35, NLL 1.257 → 1.202, tail-ECE
0.369 → 0.291 — the per-expert temperatures + logit-space recipe did their job).

The problem moved: **the gate's routing is now net-negative**. It helps tail (+1.08 pp Low) but
damages head/med (−2.04 pp Many, −0.86 pp Med), so the method loses to its own uniform baseline
on balanced accuracy (42.79 vs 43.47 test / 42.75 vs 43.35 full-val) — every checkpoint is ❌.

Critically, **the head damage is structural, not a regression**: the previous run also lost on
Many (−1.46 pp). The tail gain is real but small (+1.1 pp both runs). Three mechanisms explain it
(H1–H3 below), and the decisive evidence is already in the output:

> **The tune set fit `gate_temp = 3.0` — the softest value the grid allowed.** The validation
> split is telling us it wants the gate as close to uniform as possible. That is the single
> clearest symptom that the gate's per-sample deviations are doing more harm than good.

---

## 1. What the numbers show (new run, test split unless noted)

| | Uniform (same recipe) | Method (k=2) | Delta |
|---|---|---|---|
| Bal Acc | 43.47 | 42.79 | **−0.69** |
| Many | 71.04 | 69.00 | **−2.04** |
| Med | 43.39 | 42.54 | −0.86 |
| Low | 11.42 | 12.50 | **+1.08** |
| NLL | 1.196 | 1.271 | +0.075 |

Previous run for comparison: Many **−1.46**, Med **+1.39**, Low **+1.17**, Bal **+0.33** vs a
weaker uniform (42.74). The head loss existed both times; the med gain disappeared because the
new uniform baseline is much stronger (43.39 vs 42.18).

Other observations:
- Gate logits are healthy (std 2.26, LA mean +0.68) — not collapsed; the gate is *confidently
  routing*, just badly.
- Per-class routing on head is erratic: class 0 → **BS 0.634** (CE 0.18, LA 0.19) — BS is the
  *weakest* head expert (65.79 Many vs CE 66.75, LA 66.32); class 1 → LA 0.43; class 2 → LA 0.42;
  class 3 → BS 0.36; class 4 → CE 0.44. There is no coherent head policy — the signature of
  per-class preferences fit to ~11 training samples per class.
- Tail classes get LA 0.36–0.57 — a coherent, useful policy (LA is the best tail expert, Low
  12.21 vs CE 8.75 / BS 8.88).
- LA-saves-day samples: avg `w_LA = 0.53`, top-2 in 86.8% — the gate *can* route on tail.
- Oracle ceiling: 51.96 bal (Head LA 1796/5520, Tail LA 1029/2480).
- Expert agreement: 63.54% when the mixture is correct, 22.46% when wrong.

---

## 2. Diagnosis — three mechanisms, ranked by evidence

### H1 (primary): top-2 truncation discards the 3rd expert's vote — the head damage

The uniform 3-way mixture achieves 71.04 Many even though no expert exceeds 66.75 — that extra
~4.5 pp is the **majority-vote / complementarity effect** of combining three partially
independent experts. The gated method uses only the top-2 experts (k=2): with near-uniform
weights the dropped expert carries ~1/3 of the mass, and whenever that expert was on the correct
side, its vote is lost. On head classes (where the uniform mixture is already near-optimal) this
is pure loss. This explains why **both runs** lose on Many (−1.46, −2.04) regardless of the
loss/target used, and why the loss grew once the gate started deviating more.

Mechanism check (provable): when all experts predict the same class, any convex mixture argmaxes
to that class — routing is irrelevant there; it only matters on disagreeing samples, and on those
the k=2 truncation throws away a full vote.

**Fix: `routing_sparsity: 3`** — the gate re-weights all three experts instead of dropping one.
The mixture becomes "uniform ± learned weight shifts", so the head damage mechanism disappears
while the tail LA-bias is retained (LA's 0.53 weight still dominates the tail mixture).
*This deviates from the paper's k=2 protocol deliberately, on evidence; k=2 remains available.*

### H2 (primary): the gate overfits a tiny training split — confident noise routing

The gate trains on 10% of CIFAR-100-LT ≈ **1,100 samples** (~11 per class, class-balanced). A
312→64→3 MLP fits that easily, and the mixture-NLL gradient on head samples is weak (all experts
are similar there), so the gate latches onto per-class noise. Evidence:
- erratic per-class preferences on head (class 0 → BS 0.63 — the weakest head expert);
- `gate_temp` fit at the soft grid edge (3.0) — the tune set rewards uniformity;
- all 15 checkpoints ❌ vs uniform across epochs 0–88 with no trend — nothing is being learned
  that transfers.

**Fixes (implemented, configurable):**
- `gate_kl_uniform` (λ): add `λ·KL(w(x) ‖ uniform)` to the loss — the gate may deviate from
  uniform only where the mixture gradient consistently beats the pull. Soft implementation of
  RIDE's "default = the collective; add experts only when uncertain".
- `gate_disagree_weight: true`: weight each sample's loss by expert disagreement (0 where all
  experts agree and routing provably cannot change the prediction). Removes the noisy gradient
  those samples inject through `p_mix(y)`.
- `gate_dropout: 0.1`: the config advertised dropout but `GateMLP` never used it; the gate now
  applies it after the ReLU.
- New training log: train-vs-val mixture accuracy gap per epoch (overfitting detector), and a
  warning when the fitted `gate_temp` hits the grid edge.

### H3 (secondary): logit-space mixing amplifies a confidently-wrong expert

Logit mixing is unbounded (raw logits reach ±38; calibrated ±25 at T=1.5): one
confident-but-wrong expert can swamp the mixture. The LA-saves-day inspection shows the gate
routing to the *most confident* expert and losing:
- sample 156 (true 94): CE wrong at max-prob 0.402 → gate gives CE **0.667**; LA right at 0.214
  → 0.18. Mixture dominated by CE's wrong peak → wrong.
- sample 207 (true 89): BS wrong at max-prob 0.680 → gate gives BS **0.463**; LA right → 0.316.
  Mixture dominated by BS's wrong peak → wrong.

The gate's confidence features are **anti-predictive on exactly the samples where routing could
rescue the prediction** (the miscalibration pattern from the round-1 report, now visible at the
mixture level). The H2 fixes make deviations conservative, which caps this damage; switching
`gate_target_mode: correctness` (L2D targets) is the principled A/B for the underlying
signal, because it trains the gate on *correctness* rather than probability mass.

---

## 3. Proposed fixes (implemented; recommended config in `_gate_train.yaml`)

| # | Change | Config | Rationale (evidence) |
|---|---|---|---|
| F1 | **k=3, no truncation** | `routing_sparsity: 3` | H1: majority vote is worth ~4.5 pp on head; both runs lost on Many |
| F2 | **Disagreement-weighted loss** | `gate_disagree_weight: true` | H2: routing cannot change predictions on agree samples; their gradient is noise |
| F3 | **KL(w‖uniform) regularizer** | `gate_kl_uniform: 0.0` (tune after debug) | H2: deviate only with evidence; `gate_temp=3.0` edge-fit says the tune wants uniform |
| F4 | **Gate dropout** | `gate_dropout: 0.1` | H2: ~1.1k training samples; config flag existed but was never wired |
| F5 | **Extended gate-temp grid + edge warning + train/val gap log** | code | H2/H3 diagnostics; `gate_temp=3.0` hit the old grid edge |
| F6 | **A/B target: correctness** | `gate_target_mode: correctness` | H3: train on P(expert correct), not probability mass |

All are one-config-line changes; code defaults keep the current behavior so nothing changes
unless the config says so. The recommended next run: `routing_sparsity: 3`,
`gate_disagree_weight: true`, `gate_dropout: 0.1`, `gate_kl_uniform: 0.0` initially.

---

## 4. Debugging to verify the diagnosis (run on the EXISTING checkpoint, no training)

**`python debug_routing_signal.py --ce_path <CE.pth> --la_path <LA.pth> --bs_path <BS.pth> --gate_ckpt <best.pth> -c config/what_to_train/cifar100/_gate_train.yaml`**

Every section tests one hypothesis with inference-only math on the current checkpoint:

| Section | Tests | Decision rule |
|---|---|---|
| **H1: k-sweep** (k ∈ {1,2,3}, per-group acc) | top-2 truncation damage | If k=3 recovers most of the Head loss vs k=2 → set `routing_sparsity: 3` (expected) |
| **H2: weight interpolation** (w → uniform, t ∈ {0, .5, 1}) | are deviations net-harmful | If bal acc rises as t→1 → add `gate_kl_uniform` |
| **H3: T_gate sweep on tune** | does tune prefer uniform | Monotone rise / fitted value at the soft edge → routing is net-negative; regularize |
| **H4: agreement split** | routing irrelevance on agree samples | Gated == uniform on the agree subset (sanity); all damage/value lives in the disagree subset |
| **H5a: gate-vs-oracle match** (per group, disagree samples) | noise vs anti-predictive | top-1 match < 33% → anti-predictive; ≈33% → noise; >40% → real signal being wasted |
| **H5b: confidence vs correctness** (per expert/group) | is confidence anti-predictive | delta ≈ 0 or negative on a group → the gate is routing on a misleading signal |
| **H5c: correctness ceiling** (logistic on gate features) | how much headroom exists | If the ceiling is ~1 pp above uniform, stop tuning the gate; invest in features (penultimate) or two-stage policy |
| **H5d: weight deviation stats** | confident noise on head | Large mean deviation on Head + low oracle match → H2 confirmed |

Plus two diagnostics already added to training: the per-epoch **train-vs-val mixture accuracy
gap** (overfitting detector) and the **gate-temp grid-edge warning**.

---

## 5. Why this is the right direction

1. **It follows the evidence, not the paper.** The paper's k=2 protocol is the *only* reason the
   method truncates; your own numbers show truncation costs ~2 pp on head in two independent
   runs, while the gate's learned weights are near-uniform (so k=3 ≈ uniform + tail bias — a
   strictly better trade).
2. **It follows the literature's "route only when uncertain" consensus** — RIDE's sequential
   router (start from the collective, deploy experts only when uncertain), Divide-Weight-Route's
   difficulty-aware fusion, and classic dynamic-ensemble-selection competence rules (DESlib).
   `gate_disagree_weight` and `gate_kl_uniform` are the soft implementations of that principle.
3. **It is low-risk and reversible.** All fixes are config flags; the recommended run keeps
   `gate_kl_uniform: 0.0` until the debug script quantifies it; k=2 remains one line away.
4. **The ceiling is honest.** The oracle is 51.96 bal / 17.75 Low; the gate's real, learnable
   tail signal is LA (top-2 in 86.8% of saves-day samples). Expect the next run to land
   somewhere between "uniform + tail bias" (≈ 44 bal / 12–13 Low) and the oracle, and use H5c
   to decide whether further gate investment is worthwhile at all.

**Files changed this round:** `imbalanceddl/utils/gate_features.py` (`expert_disagreement`),
`imbalanceddl/strategy/_gate_trainer.py` (dropout, KL-to-uniform, disagreement weighting,
extended T_gate grid + edge warning, train/val gap log, metadata),
`imbalanceddl/utils/debug/models.py` (GateMLP dropout), `imbalanceddl/utils/config.py` (2 new
flags), `config/what_to_train/cifar100/_gate_train.yaml` (recommended round-2 settings),
`debug_routing_signal.py` (new verification script), `smoke_test_gate.py` (round-2 paths, all
checks passing).
