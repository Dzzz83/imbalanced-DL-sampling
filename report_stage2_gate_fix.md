# Stage-2 Gate Routing Fix — Problems, Fixes, Literature, and Justification

**Project:** CIFAR-100-LT MoE (3 frozen experts: CE / LA(τ=1.5) / BS, one trained gate, top-2
routing, plug-in rejection rule). **Date:** 2025-08-24.
**Scope of this report:** why the 8/23 gate sweep produced a near-uniform gate with marginal gains
(43.06 vs 42.74 bal acc, 11.75 vs 10.58 tail, but NLL 1.383 vs 1.257, tail-ECE 0.454 vs 0.369),
which fixes were implemented in code, the literature each fix rests on, and why the fixes are the
correct ones. Companion document: `literature_review_moe_routing.md` (full annotated review).

---

## 0. Executive summary

The gate is not broken because it is too small, too shallow, or badly initialized. It is broken
because **the supervision it was trained with is flat exactly where routing matters, and because
it was trained on a different mixture than the one it is scored with**. Concretely:

1. The soft-oracle KL target `softmax([p_CE(y), p_LA(y), p_BS(y)]/τ)` is **sharp for head classes
   and ~uniform for tail classes**, because all `p_i(y)` are tiny when all experts are wrong. The
   KL gradient w.r.t. gate logits is exactly `w − t` (verified to 1e-8), so a flat target means
   **zero supervision**. The class-balanced sampler makes this worse: most gate batches are tail
   samples with flat targets, so the *average* gradient pushes the gate to uniform. The observed
   equilibrium `w* ≈ [0.31, 0.35, 0.35]` is precisely `E[target]`.
2. Training/checkpoint selection used the **full 3-way mixture**, final evaluation used the
   **top-2 renormalized mixture** — the gate was never trained on the operation it is scored on.
3. The target is a **probability-ratio proxy for the wrong quantity**: "which expert has the
   highest `p(y|x)`" is not "which expert will be correct", because the experts are miscalibrated
   (tail-ECE 0.47–0.52). The consistent alternative is a *correctness* target
   `P(expert correct | x)` (learning-to-defer literature).
4. **Probability-space mixing starves tail gradients twice** (different mechanisms for the two
   losses; both formulas below were verified numerically).
5. Secondary issues: 312-dim near-collinear features with no learned block selectivity, a dropped
   `la_tau` factor in one eval table, gate features built at fixed T=1.0 while the mixture used the
   sweep T, checkpoint selection on a 20% tune slice, and no final calibration of the mixture
   (explaining the NLL/ECE gap).

**The fixes** (all implemented, all verified by a CPU smoke test, none touching the frozen experts
or the plug-in rule):

| # | Fix | File | Fixes |
|---|---|---|---|
| F1 | One shared mixture recipe used by loss, selection, and every eval script (`build_mixture`: k, logit/prob space, weight floor, mixture temperature) | `imbalanceddl/utils/gate_features.py` | RC2 |
| F2 | Default loss = **mixture NLL in logit space** (`gate_target_mode='mix_nll'`, `mix_space='logit'`) — gradient depends on logit *regret*, never vanishes | `_gate_trainer.py` | RC4 |
| F3 | **Log-space sharpened oracle target** (`logprob` mode): `softmax((log p − max log p)/τ)` | `gate_features.py` | RC1 |
| F4 | **Correctness targets** (`correctness` mode): isotonic-calibrated `P(expert correct \| max-prob)` fit on the tune set | `_gate_trainer.py` | RC3 |
| F5 | **Per-expert temperatures** fit on tune (BalPoE-style), applied before features *and* mixture | `_gate_trainer.py` | RC6b/calibration |
| F6 | **Gate temperature + final mixture temperature** fit on tune, saved in checkpoint metadata | `_gate_trainer.py` | calibration gap |
| F7 | Per-block L2 normalization of the 300 probability dims in gate features | `gate_features.py` | RC5 |
| F8 | Optional per-expert **weight floor** (capacity guarantee) | `gate_features.py` | RC7/starvation |
| F9 | Eval scripts read the recipe from checkpoint metadata; uniform baseline = same recipe, equal weights; `la_tau` bug fixed; T-consistency enforced | `debug/evaluation.py`, `verify_stage1/2/3.py`, `ultra_debug.py`, `inspect_gate_gradients.py` | RC2/RC6 |

Everything is configurable (`gate_target_mode`, `mix_space`, `gate_weight_floor`,
`gate_norm_blocks`, `fit_expert_temps`, `fit_gate_temp`, `fit_mix_temp`, `expert_temperatures`) so
the next run can A/B the alternatives; the config default is the recommended combination.

---

## 1. The problems, with evidence from the 8/23 sweep

### RC1 — the soft-oracle target is flat on tail classes (the primary bug)

Target: `t = softmax([p_i(y|x)]/τ)`, τ = `gate_oracle_tau` = 0.2. For `F.kl_div(log_weights, t)` the
gradient w.r.t. gate logits is (verified to machine precision):

```
∂L/∂g_j = w_j − t_j
```

When all three experts are wrong on a tail sample, `p_i(y|x) ≈ [0.02, 0.01, 0.03]`; after `/τ`
they are `[0.10, 0.05, 0.15]` and `softmax` returns `≈ [0.33, 0.31, 0.36]` — a target
indistinguishable from uniform, i.e. **zero gradient**. Head samples (`p_y ≈ 0.3–0.9`) produce
sharp targets, so the only strong supervision the gate ever sees comes from head classes.

Evidence from your own run:

- Average gate weights `CE=0.3085 | LA=0.3457 | BS=0.3458` on every checkpoint — statistically
  uniform. This matches `E[target]` over the class-balanced gate data almost exactly.
- The per-class routing table shows deviations **only** for head classes 0–4 (toward CE); the tail
  rows are 5–6-sample noise.
- LA is the sole correct expert on 136 tail samples and receives avg `w_LA = 0.46` there — the
  *only* place the gate visibly deviates, because those are the only tail samples where the target
  is moderately sharp.
- The gradient-inspection run reports soft-oracle KL loss 0.339 — *small*, because both the
  target and the gate output are near-uniform: a low KL here is the collapse itself, not a sign of
  learning. (For comparison, a gate that was actually learning sharp per-sample routing would show
  a much larger loss during training.)

The class-balanced `WeightedRandomSampler` compounds this: it upsamples tail classes, so the
majority of training batches carry flat targets and the dominant gradient direction is "stay
uniform".

### RC2 — training and evaluation use different mixtures (protocol mismatch)

- `train_one_epoch` / `validate()` (checkpoint selection): **full 3-way** probability mixture.
- Final metrics (`extract_posteriors`, `verify_stage2`, `ultra_debug`): **top-2 renormalized**
  probability mixture (`routing_sparsity=2`).

With near-uniform weights, top-2 renormalization compresses `[0.31, 0.35, 0.34]` into `≈ [0.51,
0.49]` — the effective test-time mixture is "average of two experts chosen by a weak ranking".
The gate therefore optimizes a different object than it is graded on, and the weight the loss
*does* shape for the 3rd expert is discarded at test time.

### RC3 — the supervision proxies the wrong quantity

`argmax_i p_i(y|x)` (and its soft version) is a noisy stand-in for "expert i will be correct".
The experts are miscalibrated (tail-ECE: CE 0.521, LA 0.471, BS 0.495), and combining
*uncalibrated* expert probabilities is provably biased (BalPoE's central theorem). The oracle
diagnostic quantifies the noise: oracle choice splits CE 32.4 / LA 35.8 / BS 31.8%, and the oracle
itself only reaches 52.50% bal acc (vs 42.74 uniform). The gate must predict "who is right" from
features; a target built from miscalibrated probability *ratios* is the wrong label.

### RC4 — probability-space losses starve tail gradients twice

**Mixture NLL (Exp ≤ 11)** — gradient w.r.t. gate logit `j` (verified numerically):

```
∂L/∂g_j = w_j · (p_mix(y) − p_j(y)) / p_mix(y)
```

This vanishes whenever the mixture is already confident (`p_mix(y) → 1`) *and* whenever the
experts' true-class probabilities are small (tail). Signal exists only where the mixture is wrong
*and* some expert is confident — a thin slice of the data. This is the mathematical reason Exp 11
collapsed to uniform even with the Switch balancing loss attached.

**Soft-oracle KL (current)** — gradient `w − t` is well-formed except the target is flat on tail
(RC1). Both losses concentrate their signal on "experts disagree with high confidence", which is a
small slice of tail data (136/2480 tail samples have LA as sole correct expert).

**Logit-space mixture NLL** (the fix) — verified numerically to 6e-7:

```
∂L/∂g_j = w_j · ( regret_j − Σ_i w_i·regret_i ),   regret_i = E_{p_mix}[ẑ_i] − ẑ_i(y)
```

`regret_i` is a logit-scale quantity: it does **not** shrink when the mixture is confident (logits
are not crushed by softmax) and stays well-conditioned on tail samples (logit differences, not
probability residuals). The gate is pushed toward experts whose logits are "more aligned with the
current mixture than with the true class" — exactly the routing signal.

### RC5 — the feature representation wastes gate capacity

The 312-dim input is three near-collinear 100-dim distributions. The fc-weight analysis shows the
first layer never develops block selectivity (mean ≈ 0, std ≈ 0.074 on all three expert blocks) —
the gate degrades to a fixed random projection over BN-whitened probability mass ("tracking
overall magnitude", per the diagnostic's own note). With ~4,500 gate-training samples and a noisy
target, a 312→64→3 MLP can only fit the *average* expert preference, not per-sample structure.

### RC6 — concrete code bugs (all verified by reading the code)

(a) `run_temperature_comparison` recomputed the T=1.0 baselines as `softmax(l_la + log_prior)` —
**the `la_tau` factor was dropped** (should be `+ la_tau·log_prior`, τ=1.5). The "Unif @ T=1.0 /
Method @ T=1.0" columns in the RAW T=1.0 table therefore used the wrong LA calibration; the
0.02–0.05 differences in that table are artifacts of this bug.

(b) `ExpertEnsemble.forward` built gate features at a **fixed T=1.0** while the mixture and the
soft target used the sweep temperature — for the T=2/5/10 rows the gate reasoned about T=1.0
features while being mixed at T (those rows are degenerate anyway: NLL ≈ 3.4).

(c) Checkpoint selection used a 20% tune slice with the full mixture; reported numbers are the 80%
test slice with top-2. Selection noise is visible in the sweep table (epochs 94/75/86 all within
0.2 pp).

(d) The NLL/Brier/ECE gap vs uniform (1.383 vs 1.257, 0.470 vs 0.434, 0.454 vs 0.369) is
structural: the method uses the **top-2 truncated** mixture while the uniform baseline averages
**all 3** probabilities, and no final temperature calibration is applied to the mixture.

### RC7 — with near-uniform weights, the achievable gain is tiny by construction

Top-2 of `≈ [0.31, 0.35, 0.34]` renormalizes to `≈ [0.51, 0.49]`: the test-time mixture is ~50/50
of two experts, so the only real lever is the *rank* (which expert to drop), and the gate's rank
decisions are only modestly better than chance. Even a perfect weight *magnitude* would not help
under this protocol; the gate must get the ranking right.

---

## 2. The fixes, and how each one addresses a root cause

### F1 — one mixture recipe everywhere (`build_mixture`)

`imbalanceddl/utils/gate_features.py` now exports the single mixture constructor used by the
training loss, `validate()`/checkpoint selection, `extract_posteriors()`, and all verify/debug
scripts:

```python
build_mixture(logits_list, weights, cls_num_list, la_tau, T, per_expert_T,
              k, space, weight_floor, mix_temperature) -> p_mix
```

Semantics: (i) optional per-expert weight floor (`clip + renormalize`, F8); (ii) optional top-k
truncation with renormalization (k ≥ 3 ⇒ full mixture); (iii) `space='logit'` ⇒
`p_mix = softmax(T_mix · Σ_i w_i ẑ_i)` (product-of-experts), `space='prob'` ⇒
`p_mix = Σ_i w_i p̂_i` (mixture of experts). `ẑ_i` come from `calibrate_expert_logits`
(bias-adjusted, per-expert temperature). This kills RC2 by construction: there is no second
mixture implementation left to drift.

### F2 — default loss: mixture NLL in logit space (`gate_target_mode='mix_nll'`)

The training loss is now `−log p_mix(y)` where `p_mix` is built by F1 with the *same k and space
used at evaluation* (top-k indices are treated as constants; the renormalized weights carry the
gradient). In logit space the gradient is the regret formula of RC4 — it never vanishes when the
mixture is confident, and it is well-conditioned on tail samples. This simultaneously fixes RC4
and makes the training objective identical to the evaluation metric's loss (RC2).

Why not keep the oracle-KL at all? It remains available (`logprob`, F3) because it is a valid
alternative — but mixture NLL is the *direct* objective (the gate is trained to maximize the
quality of the exact mixture it will be scored on), whereas the oracle-KL optimizes a proxy
(true-class-probability ranking).

### F3 — log-space sharpened oracle target (`logprob` mode)

```python
t = softmax( (log p_j(y) − max_i log p_i(y)) / τ )
```

Log compression keeps small probabilities *contrasted*. Verified example with `p = [0.05, 0.02,
0.08]`, τ = 0.2: probability-space target `≈ [0.331, 0.285, 0.384]` (flat) vs log-space target
`≈ [0.087, 0.001, 0.912]` (decisive). The shift by the max makes the target invariant to
per-expert calibration constants. This restores gradient signal on tail samples (RC1) with a
one-line change; the loss is still the same KL.

### F4 — correctness targets (`correctness` mode, learning-to-defer)

For each expert, fit on the **tune set** (never the gate split) an isotonic map
`max-prob_j → P(expert j correct)` (sklearn `IsotonicRegression`, `out_of_bounds='clip'`, with a
clipped-confidence fallback if a group has < 20 correct samples). The target is the normalized
per-expert correctness probability vector; the gate is trained with the same KL. This is the
learning-to-defer (L2D) supervision family: the target is the *right quantity*
("will expert j be right?"), it is consistent (the Bayes-optimal router for the surrogate equals
the oracle router, Mozannar–Sontag), and it is tail-safe (correctness is a binary quantity whose
expectation is estimable even where `p(y|x)` is tiny — unlike probability ratios, which are
crushed to ~0 on tail).

### F5 — per-expert temperatures (`fit_expert_temps`, BalPoE-style)

Grid search per expert over `{0.5, 0.75, 1.0, 1.5, 2.0, 3.0}` on the tune set, minimizing the
prior-weighted NLL of that expert's calibrated posterior. Effective temperature of expert i is
`T_sweep · T_i`; the temperatures are applied *before* the gate features and *before* the
mixture. Rationale: mixing heterogeneous experts requires commensurate logit scales (BalPoE's
theorem — averaging uncalibrated experts is biased); a single global temperature cannot reconcile
CE's sharp head-peaked logits with BS's prior-suppressed ones. Manual override available via
`expert_temperatures`.

### F6 — gate temperature and mixture temperature fit on tune

- `fit_gate_temp`: during validation, grid search `T_gate ∈ {0.3, …, 3.0}` maximizing the tune
  balanced accuracy of the *final mixture* (the gate's softmax is divided by `T_gate`). This is a
  post-hoc sharpening/softening knob that the old pipeline lacked.
- `fit_mix_temp`: for logit space, grid search `T_mix ∈ {0.6, …, 2.0}` minimizing the
  prior-weighted NLL of the final mixture on tune. `T_mix` does not change the argmax (softmax is
  monotone) but calibrates NLL/Brier/ECE — directly targeting the calibration gap (RC6d).

Both values are stored in the checkpoint metadata and used by every eval script.

### F7 — per-block L2 normalization of gate features (`gate_norm_blocks`)

`build_gate_input` L2-normalizes each expert's 100-dim probability block before concatenation
(the 9 confidence/entropy/margin and 3 agreement features stay on *real* probabilities).
Rationale (LogitNorm, τ-norm): the gate otherwise latches onto overall probability mass — the
diagnostic's own interpretation of the near-uniform fc weights (RC5). Dimension count is
unchanged, so all existing tooling still works.

### F8 — per-expert weight floor (`gate_weight_floor`)

Optional `clip(weights, min=floor)` + renormalization inside `build_mixture`. This is the
expert-choice-routing idea (Zhou et al. 2022) transferred to a soft mixture: a structural
guarantee that a degenerate gate can never starve the expert that is right on a rare tail class.
Default 0.0 (off); intended as a safeguard once the gate starts routing harder.

### F9 — evaluation protocol consistency

- `recipe_from_checkpoint` (in `debug/evaluation.py`) reconstructs each checkpoint's exact recipe
  (`T`, `la_tau`, `expert_temps`, `k`, `space`, `weight_floor`, `gate_temp`, `mix_temp`,
  `norm_blocks`) from checkpoint metadata; all verify/debug scripts consume it.
- The **uniform baseline is now the same recipe with equal weights over all experts**
  (`uniform_weights`, full mixture) — the fair "did routing help?" comparison.
- The `la_tau` omission in `run_temperature_comparison` is fixed by construction (the bias now
  lives inside `calibrate_expert_logits`).
- `inspect_gate_gradients.py` rebuilds the *checkpoint's actual loss* (target mode, τ, recipe) so
  the reported gradient is the one the gate trained with.
- The gate-temperature sweep `[1.0, 2.0, 5.0, 10.0]` is dropped from the config (those rows were
  degenerate: NLL ≈ 3.4); per-expert temperatures now do the calibration work.

### F10 — infra: tune-set caching and correctness fitting

`GateTrainer` caches the tune set's raw logits once at init; validation, temperature fitting, and
correctness calibration reuse them (no per-epoch ensemble forward passes). Checkpoints store the
full recipe so post-hoc evaluation is deterministic.

---

## 3. Literature behind the fixes

### 3.1 Mixture construction, calibration, and product-of-experts (F1, F2, F5, F6)

- **Balanced Product of Calibrated Experts (BalPoE), Aimar et al., CVPR 2023** —
  https://arxiv.org/abs/2206.05260. The single most relevant paper. Its expert set `λ = {1, 0,
  −1}` (forward/uniform/inverse bias) is *exactly* our CE/BS/LA trio. Central theorem: averaging
  **uncalibrated** experts is biased; with calibrated experts, logit averaging is
  Fisher-consistent for the balanced error. Two of our fixes are direct transfers: per-expert
  calibration (F5) and logit-space mixing (F2's `mix_space='logit'`, i.e. `softmax(Σ wᵢ ẑᵢ)` =
  product-of-experts).
- **SADE, Zhang et al., NeurIPS 2022** — https://arxiv.org/abs/2107.09249. Shows that for 3
  skill-diverse experts a tiny aggregation rule beats fragile per-sample deep routers, and that
  logit-average combination is the strongest default. Supports choosing a *simple, well-supervised*
  router over a bigger one.
- **RIDE, Wang et al., ICLR 2021** — https://arxiv.org/abs/2010.01809. Routes on L2-normalized
  features + top-s ranked logits (anti-magnitude input design; our F7 is the same idea in
  probability space) and reframes routing as "start from the collective, add experts only when
  uncertain" (our documented future work, two-stage router).
- **Temperature scaling, Guo et al., ICML 2017** — https://arxiv.org/abs/1706.04599. The
  standard post-hoc calibration device; F5/F6 are its per-expert and per-mixture instantiations.
- **τ-norm, Kang et al., 2020** — https://arxiv.org/abs/1910.09217. Normalizing classifier/logit
  scale to make experts commensurable; the ancestor of per-expert temperature handling.
- **Logit Adjustment, Menon et al., ICLR 2021** — https://arxiv.org/abs/2007.07314 and
  **Balanced Meta-Softmax, Ren et al., NeurIPS 2020** — https://arxiv.org/abs/2007.10740. Define
  the bias adjustments (`+τ log π`, `+log n_y`) that our calibration applies before mixing; our
  three experts are three points on this τ-sweep.

### 3.2 Routing losses and targets (F2, F3, F4)

- **Learning to Defer — consistent estimators, Mozannar & Sontag, ICML 2020** —
  https://icml.cc/media/icml-2020/Slides/6448.pdf (arXiv:2006.01808). The key consistency result:
  a softmax over (predict ∪ defer-to-j) trained with the augmented cross-entropy routes like the
  oracle router at its Bayes optimum. Our `correctness` mode and `mix_nll` mode are instances of
  this family (the "predict" arm is the ensemble baseline). This is why we expect the *target
  quantity* to be "P(expert j correct)", not a probability ratio (RC3).
- **Predict Responsibly (learning to defer), Madras et al., NeurIPS 2018** —
  https://mlanthology.org/neurips/2018/madras2018neurips-predict/ (arXiv:1806.07866). Introduced
  the defer action space and the coupled loss.
- **In Defense of Softmax Parametrization for Calibrated and Consistent Learning to Defer, Cao et
  al., NeurIPS 2023** — https://papers.nips.cc/paper_files/paper/2023/hash/791d3337291b2c574545aeecfa75484c-Abstract-Conference.html.
  Softmax-parametrized deferral losses are not only consistent but **calibrated**: the softmax
  output is interpretable as `P(expert j beats the system | x)`. Legitimizes reading the gate's
  output as a correctness probability and validating its calibration on the tune set.
- **Mastering Multiple-Expert Routing: Realizable h-Consistency and Strong Guarantees for
  Learning to Defer, Mao et al., ICML 2025** —
  https://research.google/pubs/mastering-multiple-expert-routing-realizable-h-consistency-and-strong-guarantees-for-learning-to-defer/.
  Extends L2D guarantees to routing among E experts with a reject option — exactly our setting
  (3 experts + plug-in rejection).
- **LogitNorm, Wei et al., ICML 2022** — https://arxiv.org/abs/2205.09310. Logit normalization
  bounds overconfidence; its "optimize direction, not magnitude" philosophy underlies both the
  log-space target (F3: only log-ratios matter) and the block normalization (F7).
- **Knowledge distillation temperature, Hinton et al., 2015** — https://arxiv.org/abs/1503.02531.
  The τ in soft targets is a sharpness knob; our log-space target applies the same idea to the
  oracle-probability ranking.

### 3.3 Router collapse, load balancing, and capacity (F8; context for why the gate stayed uniform)

- **Sparsely-Gated MoE, Shazeer et al., 2017** — https://arxiv.org/abs/1701.06538. Noisy top-k
  gating exists precisely because clean gates collapse; the gate needs exploration or a strong
  signal. Our diagnosis: the signal was missing (RC1/RC4), so no amount of noise would have helped.
- **Switch Transformers, Fedus et al., 2021** — https://arxiv.org/abs/2101.03961. The
  load-balancing loss; Exp 11 showed it *causes* uniform collapse when it fights a target that is
  already near-uniform — consistent with our RC1/RC4 analysis (the auxiliary loss was not the bug;
  the supervision was).
- **On the Representation Collapse of Sparse Mixture of Experts, Chi et al., NeurIPS 2022** —
  https://mlanthology.org/neurips/2022/chi2022neurips-representation/. Routers collapse to
  degenerate policies; for frozen experts the analog is our uniform collapse — the "lazy routing"
  optimum of a flat target.
- **Routers in Vision Mixture of Experts: An Empirical Study, Liu et al., TMLR 2024** —
  https://mlanthology.org/tmlr/2024/liu2024tmlr-routers/ (arXiv:2401.15969). Simple linear routers
  match MLP routers in vision MoE — evidence that the 312→64→3 MLP is not the bottleneck; the
  features and loss are.
- **Mixture-of-Experts with Expert Choice Routing, Zhou et al., NeurIPS 2022** —
  https://papers.nips.cc/paper_files/paper/2022/hash/2f00ecd787b432c1d36f3de9800728eb-Abstract-Conference.html.
  Capacity guarantees make starvation structurally impossible; our F8 weight floor is the soft
  mixture analog.
- **V-MoE, Riquelme et al., 2021** — https://arxiv.org/abs/2106.05974. Batch-priority routing and
  capacity limits: hard structural guarantees complement soft losses (same idea as F8).
- **LightGBM-MoE collapse notes** —
  https://github.com/kyo219/LightGBM-MoE/blob/master/docs/moe/advanced-collapse.md. Practical
  catalogue showing uniform collapse is a known mode of soft routers trained on weak targets.
- **Dynamic ensemble selection (DESlib / MCS literature)** — https://deslib.readthedocs.io/.
  Classic per-sample classifier selection via local competence measures; confirms that
  correctness-based competence, not confidence magnitude, is the routing signal.

### 3.4 The target paper and training-free alternatives (context)

- **Learning to Reject Meets Long-Tail Learning, Narasimhan et al., ICLR 2024** —
  https://proceedings.iclr.cc/paper_files/paper/2024/hash/c4f129179494c1ea14b63fc0019f3095-Abstract-Conference.html —
  the paper this repo replicates (top-2 routing + plug-in rejection, Bal/Worst AURC). Its
  headline numbers (NLL 1.18, Brier 0.403, tail-ECE 0.088) come from a **calibrated** pipeline;
  our F5/F6 restore that calibration step.
- **DaWin: Training-free Dynamic Weight Interpolation, Oh et al., 2024** —
  https://ar5iv.labs.arxiv.org/html/2410.03782. A zero-training confidence-weighted router; the
  cheapest possible baseline to compare the learned gate against (documented as a benchmark in the
  review, not yet implemented).
- **Divide, Weight, and Route (PRCV 2025)** — https://ar5iv.labs.arxiv.org/html/2508.19630 and
  **Long-Tailed Distribution-Aware Router (2025)** — https://ar5iv.labs.arxiv.org/html/2507.01351.
  Recent long-tail routing work; their "difficulty-conditioned routing" and
  "frequency-aware router" ideas are the documented next step (two-stage router) rather than part
  of this fix.

---

## 4. Why these are the correct fixes

The argument is fourfold: **evidence**, **theory**, **literature consensus**, and **risk**.

### 4.1 Evidence — every fix maps to an observed symptom

| Symptom (your 8/23 output) | Explanation | Fix |
|---|---|---|
| `w* ≈ [0.31, 0.35, 0.35]` on all 40+ checkpoints | gate output ≈ `E[target]`; target flat on tail (RC1) | F3/F4 (sharp/correct targets), F2 (direct objective) |
| Only classes 0–4 deviate from uniform | only head samples carry sharp targets | F3/F4 |
| LA-saves-day: w_LA = 0.46, top-2 in 82.4% | the only tail samples with sharp targets | F3/F4 (should push this higher) |
| Bal acc +0.33 pp, tail +1.17 pp only | RC7: top-2 of near-uniform weights ≈ 50/50 averaging | F2 (train on the real mixture), F6 (gate temp), F8 (floor) |
| NLL 1.383 vs 1.257, tail-ECE 0.454 vs 0.369 | RC6d: uncalibrated top-2 truncation vs full-3 baseline | F5/F6 (calibration), F1/F9 (same recipe comparisons) |
| fc weights: mean ≈ 0, std ≈ 0.074 on all blocks | RC5: no learned block selectivity | F7 (kill magnitude signal) |
| "Unif @ T=1.0" columns differ from "Unif @ T" for T=1.0 | RC6a: `la_tau` dropped | F9 (bias inside shared calibration) |
| oracle 52.50 vs method 43.06 | ~9.8 pp recoverable; supervision never taught the gate the right signal | F2/F3/F4 (train on the right objective) |

### 4.2 Theory — the gradients and consistency arguments are verified, not assumed

The three gradient formulas in this report were **numerically verified** against autograd
(prob-space mixture NLL, KL `w − t` to 1e-8, logit-space regret formula to 6e-7). The
consequences are exact: probability-space mixture NLL has zero gradient on confident mixtures and
tiny gradient on tail samples; the KL has zero gradient on flat targets; the logit-space loss has
neither failure mode. The correctness target (F4) inherits the L2D consistency theorem
(Mozannar–Sontag) and the calibration property (Cao et al.): at its Bayes optimum the router
routes exactly like the oracle router, and the softmax output is a calibrated probability of
"expert j is better". That is a *provable* property of the target choice, not a heuristic.

### 4.3 Literature consensus

Every method that survives contact with CIFAR-100-LT either (a) averages *calibrated logits*
(BalPoE, SADE, RIDE) rather than routing raw outputs, (b) trains the router against
*correctness/deferral* (L2D family), or (c) guarantees expert usage structurally (expert-choice,
V-MoE capacity). The previous pipeline violated (a) (probability-space top-2 without
calibration), used a proxy target instead of (b), and had no (c). The fixes bring the pipeline
into the consensus region: calibrated logit-space mixing (a), direct mixture NLL with
correctness/log-space targets available (b), and an optional weight floor (c).

### 4.4 Risk — minimal-change, reversible, and verifiable

- **Nothing outside the gate is touched**: the frozen experts, the plug-in rejection rule, the
  GateMLP architecture, and the checkpoint file format (extended, not replaced) are unchanged.
- **Backward compatibility**: eval scripts fall back to safe defaults for old checkpoints;
  `mix_space='prob'` + `gate_target_mode='logprob'` reproduces the old protocol (with the bugs
  fixed), so any regression is attributable.
- **A/B-able**: `gate_target_mode` (mix_nll / logprob / correctness) and `mix_space` (logit /
  prob) are config flags; the next run can compare them directly with identical evaluation code.
- **Verification**: `smoke_test_gate.py` (CPU venv) checks the calibration math against manual
  formulas, both mixture spaces against hand-computed values, top-k and weight-floor behavior,
  target sharpness, isotonic direction, and full end-to-end training + validation + checkpoint
  metadata + posterior extraction + plug-in eval for all six (target × space) combinations —
  110+ checks, all passing.

### 4.5 What "correct" does NOT mean here

These fixes guarantee that the gate *can* learn per-sample routing (healthy gradients, right
target quantity, consistent protocol, calibrated mixture). They do **not** guarantee a large win:
the recoverable signal is bounded by how much the experts' outputs reveal correctness (oracle
ceiling 52.50 vs uniform 42.74; the achievable fraction depends on the features). If the next run
still shows a small gain, the documented next steps are: correctness-forecasting AUC to measure
the ceiling, RIDE-style penultimate-feature inputs, the two-stage ambiguity router, and
per-class routing priors (head→CE, tail→LA) — all in `literature_review_moe_routing.md` §7 (P1–P3).

---

## 5. Verification and how to read the next run

Run `./.venv/bin/python smoke_test_gate.py` locally to re-verify the implementation (CPU only, no
training). On Kaggle, train with the existing command — `config/what_to_train/cifar100/_gate_train.yaml`
now carries the new defaults — and evaluate with `verify_stage2.py` / `ultra_debug.py`, which
reconstruct each checkpoint's recipe automatically.

Signs of a *working* fix in the next sweep:

1. `Avg Weights` during training deviate from [1/3, 1/3, 1/3] per-sample (and per-class table shows
   tail classes getting real weight mass, not just head classes).
2. `oracle_match` in training logs rises above the old ~40% plateau.
3. NLL/Brier/tail-ECE of "My Method" move toward or below the uniform baseline (mixture
   temperature calibration, F6).
4. The RAW T=1.0 vs RECIPE table shows near-identical columns for the uniform rows (the `la_tau`
   bug is gone) and the method rows show the effect of the fitted gate/mixture temperatures.
5. If `gate_target_mode` is A/B'd: `mix_nll` should show the healthiest gradients
   (`inspect_gate_gradients.py` reports the checkpoint's actual loss now), `correctness` should
   show the most tail-sensitive weights, `logprob` sits between.

---

## 6. References

1. Shazeer et al., 2017 — Sparsely-Gated MoE — https://arxiv.org/abs/1701.06538
2. Fedus et al., 2021 — Switch Transformers — https://arxiv.org/abs/2101.03961
3. Riquelme et al., 2021 — V-MoE — https://arxiv.org/abs/2106.05974
4. Wang et al., 2021 — RIDE (ICLR) — https://arxiv.org/abs/2010.01809
5. Zhang et al., 2022 — SADE (NeurIPS) — https://arxiv.org/abs/2107.09249
6. Aimar et al., 2023 — BalPoE (CVPR) — https://arxiv.org/abs/2206.05260
7. Menon et al., 2021 — Logit Adjustment (ICLR) — https://arxiv.org/abs/2007.07314
8. Ren et al., 2020 — Balanced Meta-Softmax (NeurIPS) — https://arxiv.org/abs/2007.10740
9. Wei et al., 2022 — LogitNorm (ICML) — https://arxiv.org/abs/2205.09310
10. Kang et al., 2020 — τ-norm / decoupling — https://arxiv.org/abs/1910.09217
11. Guo et al., 2017 — Temperature scaling (ICML) — https://arxiv.org/abs/1706.04599
12. Hinton et al., 2015 — Knowledge distillation — https://arxiv.org/abs/1503.02531
13. Zhou et al., 2022 — Expert Choice Routing (NeurIPS) — https://papers.nips.cc/paper_files/paper/2022/hash/2f00ecd787b432c1d36f3de9800728eb-Abstract-Conference.html
14. Chi et al., 2022 — Representation Collapse of Sparse MoE (NeurIPS) — https://mlanthology.org/neurips/2022/chi2022neurips-representation/
15. Liu et al., 2024 — Routers in Vision MoE (TMLR) — https://mlanthology.org/tmlr/2024/liu2024tmlr-routers/ (arXiv:2401.15969)
16. Madras et al., 2018 — Learning to Defer (NeurIPS) — https://mlanthology.org/neurips/2018/madras2018neurips-predict/ (arXiv:1806.07866)
17. Mozannar & Sontag, 2020 — Consistent Estimators for Learning to Defer (ICML) — https://icml.cc/media/icml-2020/Slides/6448.pdf (arXiv:2006.01808)
18. Cao et al., 2023 — Softmax Parametrization for Calibrated & Consistent L2D (NeurIPS) — https://papers.nips.cc/paper_files/paper/2023/hash/791d3337291b2c574545aeecfa75484c-Abstract-Conference.html
19. Mao et al., 2025 — Mastering Multiple-Expert Routing (ICML) — https://research.google/pubs/mastering-multiple-expert-routing-realizable-h-consistency-and-strong-guarantees-for-learning-to-defer/
20. Narasimhan et al., 2024 — Learning to Reject Meets Long-Tail Learning (ICLR) — https://proceedings.iclr.cc/paper_files/paper/2024/hash/c4f129179494c1ea14b63fc0019f3095-Abstract-Conference.html
21. Oh et al., 2024 — DaWin — https://ar5iv.labs.arxiv.org/html/2410.03782
22. Cai et al., 2025 — LTDA-Router — https://ar5iv.labs.arxiv.org/html/2507.01351
23. Wei & Yi, 2025 — Divide, Weight, and Route (PRCV) — https://ar5iv.labs.arxiv.org/html/2508.19630
24. DESlib — Dynamic Ensemble Selection — https://deslib.readthedocs.io/
25. LightGBM-MoE — collapse notes — https://github.com/kyo219/LightGBM-MoE/blob/master/docs/moe/advanced-collapse.md
