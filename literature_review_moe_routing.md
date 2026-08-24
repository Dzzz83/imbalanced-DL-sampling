# Routing Mechanisms for MoE & Ensembles in Long-Tailed Recognition

**A targeted literature review for Stage 2 (gate routing) of the CIFAR-100-LT MoE project.**
Setup under review: 3 frozen ResNet-32 experts (CE = head-biased, LA = tail-biased, BS = intermediate),
a lightweight gate consuming the **312-dim calibrated-probability features** (3×100 probs + 9
confidence/margin/entropy stats + 3 pairwise agreements, `build_gate_input`), trained with the
**soft-oracle KL** objective, and evaluated with the paper's **top-2 renormalized mixture** +
plug-in rejection rule.

**Status (8/23 sweep):** the gate is still ~uniform (`w ≈ [0.31, 0.35, 0.34]` on every
checkpoint). Bal acc 43.06 vs 42.74 uniform (+0.33 pp) and Low acc 11.75 vs 10.58 (+1.17 pp) —
the success criterion is met only *marginally* — while NLL/Brier/ECE are far **worse** than the
uniform baseline (1.383 vs 1.257 / 0.470 vs 0.434 / 0.454 vs 0.369). The oracle gap is ~9.8 pp
(oracle 52.50) and the gate captures ~0.3 pp of it. §0.5 is the full post-mortem; §6 maps each
root cause to a fix; §7 is the updated, prioritized recommendation list; §8 lists references
(new papers: learning-to-defer family, DaWin, LTDA-Router, Divide-Weight-Route, expert-choice
routing, router-collapse empirical studies).

---

## 0. TL;DR — the failure mode, named

"Naive peak-detection" is a known failure of **uncalibrated logit-space routing**. The literature
converges on three independent remedies, which map 1:1 onto the four questions below:

1. **Don't route on raw logit magnitude** — route on *normalized* logits, *probabilities*, or
   *calibrated* scores (LogitNorm, τ-norm, temperature scaling, BalPoE).
2. **Don't let the gate learn a one-hot winner-take-all** — add load-balancing / diversity /
   entropy regularization (Shazeer, Switch, RIDE) or a "deploy-more-experts-only-if-ambiguous"
   objective (RIDE's sequential router).
3. **Give the gate a *soft* teacher, not a hard argmax label** — the ensemble itself is the
   best teacher (RIDE routing distillation, SADE prediction stability, BalPoE's logit averaging).
4. **Align the expert scales *before* mixing** — a single global temperature is not enough when
   CE/LA/BS live on different magnitude ranges (BalPoE's per-expert calibration, RIDE's class-wise
   temperature, per-expert standardization).

> **Critical observation specific to this repo (updated for the 8/23 sweep).** The gate no longer
> sees raw logits — it is fed **312-dim calibrated probabilities** (`build_gate_input`) and trained
> with the **soft-oracle KL** loss `KL(softmax(gate) ‖ softmax([p_CE(y), p_LA(y), p_BS(y)]/τ))`.
> The failure mode *moved*: the gate is still ~uniform, but the cause is now the **supervision,
> not the representation**. The soft-oracle target is **sharp for head classes and flat for tail
> classes** (all `p_i(y)` are tiny when all experts are wrong), so the gate is literally taught to
> output uniform weights exactly where per-sample routing would matter most (§0.5 RC1, §2.5). The
> fix is a sharpness-corrected or correctness-based target (§2.4, §2.5), not a bigger gate.

---

## 0.5 STAGE-2 POST-MORTEM — WHY THE GATE IS STILL (ALMOST) UNIFORM

**Pipeline state that produced the 8/23 sweep.** Gate = `GateMLP` (BatchNorm(312) → Linear(64) →
ReLU → Linear(3)), fed calibrated-probability features at T=1.0, trained with the soft-oracle KL
loss (τ = `gate_oracle_tau` = 0.2) on the class-balanced gate split, validated on a 20% tune slice
with the **full 3-way mixture**, and finally evaluated on the 80% test slice with the **top-2
renormalized mixture** (`extract_posteriors` / `verify_stage2`, `routing_sparsity=2`).

**What the numbers say.**
- Gate weights are statistically indistinguishable from uniform on every one of 40+ checkpoints
  (`w_CE ≈ 0.31–0.35`, `w_LA ≈ 0.33–0.37`, `w_BS ≈ 0.30–0.34`); the gradient-inspection run shows
  `CE=0.3085 | LA=0.3457 | BS=0.3458`.
- Best checkpoint: Bal 43.06 vs 42.74 uniform (+0.33 pp), Low 11.75 vs 10.58 (+1.17 pp) — the
  user's success criterion is *marginally* met, but NLL/Brier/ECE are much worse than uniform
  (1.383 vs 1.257, 0.470 vs 0.434, 0.454 vs 0.369).
- Oracle bal acc is 52.50 vs 42.74 uniform: a ~9.8 pp recoverable gap; the gate captures ~0.3 pp.
- LA is the sole correct expert on 136 tail samples and receives avg `w_LA = 0.46` there (top-2 in
  82.4%) — the *only* place the gate visibly deviates from uniform.

**RC1 — the soft-oracle target is sharp on head classes and flat on tail classes; the gate is
taught to be uniform where routing matters most.**
`t = softmax([p_i(y|x)]/τ)`. When all three experts are wrong on a tail sample, all `p_i(y|x)` are
small (≈ [0.02, 0.01, 0.03]); after `/τ` they are ≈ [0.10, 0.05, 0.15] and `softmax` returns
≈ [0.33, 0.31, 0.36] — a near-uniform target. For `F.kl_div(log_weights, soft_target)` the gradient
w.r.t. gate logits is exactly **`w − t`**, i.e. ~zero whenever the target is uniform. Head samples
(p_y ≈ 0.3–0.9) produce sharp targets, so the *only* strong supervision the gate ever receives
comes from head classes — exactly the pattern in the per-class routing table (only classes 0–4
deviate, toward CE; the tail rows are 5–6-sample noise). The class-balanced `WeightedRandomSampler`
compounds this: gate batches are dominated by tail samples (flat targets), so the *average*
gradient pushes toward uniform. The observed equilibrium `w* ≈ [0.31, 0.35, 0.35]` is precisely
`E[target]` over the class-balanced gate data — the gate found no conditional structure because
the target carries almost none on the samples where it matters.

**RC2 — the gate optimizes a different mixture than the one it is scored with.**
Training loss and checkpoint selection use the **full 3-way mixture** (`validate()`), but final
metrics use the **top-2 renormalized mixture**. With near-uniform weights, top-2 renormalization
compresses [0.31, 0.35, 0.34] into ≈ [0.51, 0.49] — the effective test-time mixture is "average of
two experts chosen by a weak ranking". The third expert's weight (which the loss *does* shape) is
discarded at test time, and the gate is never trained on the operation it is scored on.

**RC3 — the supervision is a proxy for the wrong thing.**
`argmax_i p_i(y|x)` (or its soft version) is a noisy stand-in for "expert i will be correct": the
experts are badly miscalibrated (tail-ECE 0.37–0.52), and BalPoE's theorem (§1.6) says combining
*uncalibrated* expert probabilities is biased. The consistent alternative from the literature is a
**correctness target** `t_i = P(expert i correct | x)` — the learning-to-defer family (§2.4). The
oracle diagnostic quantifies the noise: oracle choice splits CE 32.4 / LA 35.8 / BS 31.8% and the
oracle itself only reaches 52.5% bal acc.

**RC4 — probability-space mixing starves tail gradients *twice* (two different mechanisms).**
- Mixture NLL (Exp ≤ 11): `∂L/∂g_j = w_j·(p_mix(y) − p_j(y))/p_mix(y)` — vanishes whenever the
  mixture is already confident, and on tail samples the numerator is tiny too. Hence Exp 11's
  uniform collapse.
- Soft-oracle KL (current): gradient `w − t` is perfect *except* the target is flat on tail (RC1).
Both losses concentrate their signal on "experts disagree with high confidence" — which for this
dataset is a small slice of tail data (only 136/2480 tail samples have LA as sole correct expert).

**RC5 — the feature representation wastes the gate's capacity.**
The 312-dim input is three near-collinear 100-dim distributions. The fc-weight analysis shows the
first layer never develops block selectivity (mean ≈ 0, std ≈ 0.074 on all three expert blocks) —
the gate degrades to a fixed random projection over BN-whitened mass ("tracking overall magnitude",
per the diagnostic's own note). With ~4,500 training samples and a noisy target, a 312→64→3 MLP can
only fit the *average* expert preference (the small per-class biases), not per-sample structure.

**RC6 — smaller protocol bugs (all verifiable in the code).**
(a) `run_temperature_comparison` recomputes the T=1.0 baselines as `softmax(l_la + log_prior)` and
**drops the `la_tau` factor** (should be `+ la_tau·log_prior`, τ=1.5): the "Unif @ T=1.0 / Method @
T=1.0" columns in the RAW T=1.0 table use the wrong LA calibration, so the 0.02–0.05 differences in
that table are artifacts of this bug, not real temperature effects.
(b) `ExpertEnsemble.forward` builds gate features at a **fixed T=1.0** while the mixture and the
soft target use the sweep temperature — for the T=2/5/10 rows the gate reasons about T=1.0 features
while being mixed at T (those rows are degenerate anyway: NLL ≈ 3.4).
(c) Checkpoint selection uses a 20% tune slice with the full mixture; reported numbers are the 80%
test slice with top-2. Selection noise is visible in the sweep table (epochs 94/75/86 all within
0.2 pp of each other).

> **Bottom line.** This is no longer an *architecture* problem (raw-logit peak-detection is gone);
> it is a **supervision-sharpness + protocol-mismatch** problem. The gate learned the *average*
> expert preference and cannot see per-sample signal because (i) the target is flat wherever
> per-sample routing would matter, and (ii) it is scored with a different mixture than it is
> trained on. Fix the target and the protocol first (§7 P0/P1); anything else (architecture,
> regularization) is second-order until those are done.

---

## 1. Gate Architectures — and how they avoid magnitude tracking

### 1.1 Sparsely-Gated MoE — noisy top-k gating (Shazeer et al., 2017)
[Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer](https://arxiv.org/abs/1701.06538)

- **Architecture.** A linear router `H(x) = W_g · x` over the *embedding*, then `G(x) = softmax(TopK(H(x)))`.
- **The anti-collapse trick is in the *noise*, not the depth.**
  `H(x)` is replaced by `H(x) + ε · softplus(W_noise · x) · 𝒩(0,1)` **during training only**.
  This Gaussian noise forces the gate to *explore* all experts early and prevents it from committing
  to whichever expert happens to have the largest logit from initialization. The paper's own ablation
  shows clean top-k gating without noise collapses to a single expert.
- **Takeaway for us:** depth (the Mini-MLP) helps expressivity but does **not** by itself stop
  peak-tracking. The decisive mechanisms are (a) noisy/exploratory gating and (b) an explicit
  balancing penalty (Section 3). Consider training-time Gaussian noise on the gate logits, or the
  equivalent exploration via the auxiliary loss.

### 1.2 Switch Transformer — top-1 with a single balancing loss (Fedus et al., 2021)
[Switch Transformers: Scaling to Trillion Parameter Models](https://arxiv.org/abs/2101.03961)

- **Architecture.** Simplifies Shazeer to *hard* top-1 routing (no noise, no `TopK>1`), but keeps an
  auxiliary load-balancing loss `L_aux = α · N · Σᵢ fᵢ · Pᵢ` (`fᵢ` = fraction of samples routed to
  expert i, `Pᵢ` = mean router probability for expert i). Balanced routing ⇒ `fᵢ = Pᵢ = 1/N` ⇒ loss
  minimized. This is the cheapest "use every expert" regularizer in the literature.
- **Takeaway for us:** with only 3 experts, a Switch-style balancing term is a one-line fix to the
  BS-starvation symptom and can be bolted onto the existing Mini-MLP unchanged.

### 1.3 V-MoE — vision MoE, linear router + capacity (Riquelme et al., 2021)
[Scaling Vision with Sparse Mixture of Experts](https://arxiv.org/abs/2106.05974)

- **Architecture.** Linear router over ViT token embeddings, top-k selection with **Batch Priority
  Routing (BPR)** and per-expert **capacity limits** (an expert that is over-subscribed simply stops
  accepting tokens). Capacity is a *hard* structural guarantee against starvation, complementing the
  soft auxiliary loss.
- **Takeaway for us:** a softmax weight can asymptotically ignore an expert even with a balancing
  loss; a **capacity floor** (e.g., enforce each expert's mean weight ≥ a floor via the loss, or use
  a top-k mixture) is the stronger guarantee. This is why RIDE/SADE-style *ensembling* usually beats
  hard routing for 3 experts.

### 1.4 RIDE — routing *features*, not raw logits; sequential binary router (Wang et al., ICLR 2021)
[Long-tailed Recognition by Routing Diverse Distribution-Aware Experts](https://arxiv.org/abs/2010.01809)

- **Architecture.** The router is a *binary* classifier (2 FC layers), not a 3-way softmax. It decides
  "deploy expert k+1 or stop?" It consumes **`f_θ(x)/‖f_θ(x)‖` (L2-normalized features)** concatenated
  with the **top-s ranked mean logits** of the experts already deployed — *not* the full raw logits.
  Rank-ordering and top-s selection strip out absolute magnitude; L2-normalization strips out feature
  scale. Both are deliberate anti-magnitude choices.
- **Why this beats our 3-way softmax for the same goal:** RIDE reframes routing from "pick the one
  best expert" to "start with a cheap answer and *add experts only when uncertain*". This never
  starves an expert because the *default* answer is the collective, and extra experts are additive.
- **Takeaway for us:** feeding the gate **normalized, rank/thresholded** inputs — or better, making
  the gate output a *mixture weight over all 3* with a strong baseline of uniform averaging — is
  closer to what works than a pure argmax gate.

### 1.5 SADE — the gate is *just a 3-vector of learnable weights* (Zhang et al., NeurIPS 2022)
[Self-Supervised Aggregation of Diverse Experts for Test-Agnostic Long-Tailed Recognition](https://arxiv.org/abs/2107.09249)

- **Architecture.** The "router" is a single learnable weight `w = [w₁,w₂,w₃]`, softmax-normalized,
  producing `ŷ = softmax(w₁v₁ + w₂v₂ + w₃v₃)` — i.e. a **weighted average of logits**. Crucially, this
  weight is *not* per-sample and *not* trained with labels; it is optimized at test time to maximize
  **prediction stability** `Σ ŷ¹·ŷ²` between two augmented views (Section 3.3).
- **Why this is relevant:** SADE demonstrates that for 3 skill-diverse experts, a *per-sample deep
  router is overkill and fragile*; a tiny weight learned with the right objective matches or beats it.
  It also shows logit-*average* combination (equivalent to product-of-softmax) is the strongest
  default ensemble.

### 1.6 BalPoE — no learned gate at all; average logits (Aimar et al., CVPR 2023)
[Balanced Product of Calibrated Experts for Long-Tailed Recognition](https://arxiv.org/abs/2206.05260)

- **Architecture.** "During inference, we **average the logits of all experts before softmax**
  normalization." Averaging logits = product of softmax distributions = the *joint* expert
  probability under an independence assumption. No learned router. What makes it work is that each
  expert is trained with a **generalized logit-adjusted (gLA) loss** `sᵧ = fᵧ + τᵧ log P_train(y)` and
  **calibrated via mixup** so their logits are commensurate (Sections 3.3–3.4).
- **Why this is the single most relevant paper for this project:** BalPoE's expert set `λ = {1, 0, −1}`
  (forward / uniform / inverse bias) is *exactly* our CE / BS / LA trio. Its central theorem is that
  logit-averaging over **uncalibrated** experts is biased — which is the same pathology as our gate
  being fooled by overconfident CE/LA. The fix is calibration, not a fancier gate.

### 1.7 Expert-Choice Routing — capacity guarantees instead of balancing losses (Zhou et al., NeurIPS 2022)
[Mixture-of-Experts with Expert Choice Routing](https://papers.nips.cc/paper_files/paper/2022/hash/2f00ecd787b432c1d36f3de9800728eb-Abstract-Conference.html) · [Google Research blog](https://research.google/blog/mixture-of-experts-with-expert-choice-routing/)

- Inverts the assignment: instead of *tokens choosing experts* (which collapses or starves experts
  when the router is bad), each **expert chooses its top-k tokens** from the batch. Every expert is
  guaranteed exactly k tokens per batch — starvation and collapse are *structurally impossible*
  rather than merely penalized.
- **Takeaway for us:** with 3 frozen experts and a soft mixture, the analog is a **per-batch weight
  floor**: clip/renormalize so each expert keeps a minimum share `w_i ≥ ε` (e.g. ε = 0.05–0.1), or
  a top-2-with-random-backup, guaranteeing the mixture can never silently ignore the expert that is
  right on a rare tail class. Cheaper and stronger than a soft balancing penalty (§3.2).

### 1.8 What the MoE-router literature says about collapse and router capacity
- **[On the Representation Collapse of Sparse Mixture of Experts](https://mlanthology.org/neurips/2022/chi2022neurips-representation/) (Chi et al., NeurIPS 2022):** routers trained jointly with experts collapse toward using a few experts; entropy regularization is the standard countermeasure. For **frozen** experts the analog is the gate collapsing to a degenerate policy — ours collapses to *uniform*, the "lazy-routing" optimum: with a near-flat target, uniform minimizes expected KL (§0.5 RC1).
- **[Routers in Vision Mixture of Experts: An Empirical Study](https://mlanthology.org/tmlr/2024/liu2024tmlr-routers/) (Liu et al., TMLR 2024, arXiv:2401.15969):** deep MLP routers do **not** beat simple linear routers in vision MoE, and router decisions are entropic/unstable. Implication: our 312→64→3 MLP is unlikely to be the bottleneck *or* the fix — the features and the loss are. If a linear router (or a tuned per-class bias vector) matches the MLP, the MLP's capacity is wasted.
- **[A Provably Effective Method for Pruning Experts in Fine-Tuned Sparse MoE](https://mlanthology.org/icml/2024/chowdhury2024icml-provably/) (Chowdhury et al., ICML 2024):** pruning "useless" experts is often lossless. Analog: under the top-2 protocol the real question is not the mixture weights but **which expert to drop** — treat routing as a *ranking* problem, not a regression problem.
- **[LightGBM-MoE — advanced collapse notes](https://github.com/kyo219/LightGBM-MoE/blob/master/docs/moe/advanced-collapse.md):** a practical catalogue of collapse modes (uniform collapse, single-expert collapse) and how load-balancing losses interact with router entropy. Our "uniform collapse" is a known failure mode of soft routers trained on weak/entropic targets — not a bug in AdamW or initialization (gate logits std is a healthy 1.42).

---

## 2. Routing Losses — training a gate when the target expert is unknown

There are three families; the literature's answer is "avoid hard argmax labels, use the ensemble as
the soft teacher."

### 2.1 Hard-label cross-entropy against an *oracle* expert
- Requires a per-sample "which expert is best" label. We already compute this as `target_expert =
  argmaxᵢ pᵢ(y|x)` in `_gate_trainer.py` (the "Oracle Match Diagnostic").
- **RIDE's variant is smarter and self-supervised:** the router label is `y_on = 1` if *adding the
  next expert corrects the current expert's mistake*, `0` otherwise. The label is derived from the
  experts themselves — no external annotation. Trained with **weighted binary CE**
  (`ω_on = 100` to bias toward deploying the extra expert).
- **Caveat:** a hard argmax target is exactly what biases the gate toward the most-confident (CE/LA)
  experts. Use it as a *diagnostic* (as we do) rather than the training signal.

### 2.2 Mixture NLL / soft mixture (what Exp ≤ 11 used; superseded by the soft-oracle KL)
`L = −log Σᵢ wᵢ · pᵢ(y|x)` — train the gate to minimize CE of the weighted expert mixture.
- **RIDE explicitly warns against this when experts are *trainable*** (their "collaborative loss"):
  it makes experts *correlated* rather than *complementary*. **This objection does not apply to us** —
  our experts are frozen, so mixture NLL cannot collapse them into each other; it only risks the *gate*
  over-trusting the highest-confidence expert.
- **The fix within mixture NLL:** add the balancing/entropy regularizer (Section 3) and calibrate the
  inputs (Section 4). **But see §2.5:** mixture NLL's gradient also vanishes whenever the mixture is
  already confident and on tail samples (small `p_i(y)`), which is exactly why Exp 11 collapsed to
  uniform even with regularization. The objective is *not* as sound as this section used to claim —
  that is the main lesson of the 8/23 post-mortem (§0.5 RC4).

### 2.3 Knowledge distillation / self-distillation — the ensemble teaches the gate
- **RIDE routing distillation:** distill a many-expert (6) teacher into a few-expert (2–4) student via
  `L_KD = T² · KL(logits_teacher/T ‖ logits_student/T)`. The teacher's *soft* distribution is a richer
  target than any hard expert label and does not require knowing the optimal expert.
- **RIDE's router itself** is trained against a distillation-style soft target (the collective
  prediction), not a hard class label.
- **SADE prediction-stability maximization** is a *self-supervised* distillation: the gate is trained
  so two augmented views of the same sample agree (`Σ ŷ¹·ŷ²`), which provably maximizes mutual
  information with the test distribution and implicitly down-weights overconfident-but-unstable experts.
- **Takeaway for us:** if mixture NLL plateaus, replace/augment the target with
  (a) the **ensemble teacher's soft probability** as a distillation target for the gate, and/or
  (b) a **consistency loss** between two augmented views — both remove the need for a hard "correct
  expert" label and both are robust to overconfident experts.

### 2.4 Learning to Defer — the consistent loss family for "route to the right expert"
This family is the theoretically grounded answer to "what should the gate's target be", and is the
single most useful new literature for this project.

- **[Predict Responsibly: Improving Fairness and Accuracy by Learning to Defer](https://mlanthology.org/neurips/2018/madras2018neurips-predict/) (Madras et al., NeurIPS 2018, arXiv:1806.07866):** the system either predicts itself or defers to one of E experts; the loss couples the deferral decision with the expert losses. Introduces the standard "defer" action space.
- **[Consistent Estimators for Learning to Defer to an Expert](https://icml.cc/media/icml-2020/Slides/6448.pdf) (Mozannar & Sontag, ICML 2020, arXiv:2006.01808):** the key consistency result: train a `softmax` over (predict ∪ defer-to-j) with the *augmented* cross-entropy — target = "predict y" if the system can get y right, else "defer to the expert j that gets y right". This surrogate is **consistent**: its Bayes optimum routes exactly like the oracle router. Our gate's 3-way softmax is precisely this structure (with the "predict" arm replaced by the uniform ensemble baseline).
- **[In Defense of Softmax Parametrization for Calibrated and Consistent Learning to Defer](https://papers.nips.cc/paper_files/paper/2023/hash/791d3337291b2c574545aeecfa75484c-Abstract-Conference.html) (Cao et al., NeurIPS 2023):** softmax-parametrized deferral losses are not only consistent but **calibrated**: the softmax output is interpretable as `P(expert j beats the system | x)`. This legitimizes reading the gate's softmax as a *correctness probability* and suggests validating its calibration on the tune set.
- **[Mastering Multiple-Expert Routing: Realizable h-Consistency and Strong Guarantees for Learning to Defer](https://research.google/pubs/mastering-multiple-expert-routing-realizable-h-consistency-and-strong-guarantees-for-learning-to-defer/) (Mao et al., ICML 2025):** extends L2D guarantees to **routing among E experts** (exactly our case) with a reject option — h-consistency bounds for a router trained with a softmax-CE surrogate over expert selection.

**Takeaway for us:** replace `softmax(p_i(y|x)/τ)` — a *probability-ratio* target that is flat on
tail (RC1) — with a **correctness target** `t_i = P(expert i correct | x)`, estimated on the tune
set via per-expert calibration of `max-prob → P(correct)` (isotonic regression or per-expert
temperature), optionally per group. Train the same gate with KL or CE against this target. It is
consistent (Mozannar–Sontag), calibrated (Cao), and tail-safe: correctness is a binary quantity
whose expectation is estimable even where `p(y|x)` is tiny.

### 2.5 Why BOTH current losses starve the tail signal (gradient math)
- **Mixture NLL** (Exp ≤ 11): `∂L/∂g_j = w_j·(p_mix(y) − p_j(y))/p_mix(y)`. Vanishes when the
  mixture is already confident (`p_mix(y) → 1`); on tail samples all `p_j(y)` are small so the
  numerator is small too. Signal exists only where the mixture is wrong *and* some expert is
  confident — a thin slice of the data. This is Exp 11's uniform collapse.
- **Soft-oracle KL** (current): `∂L/∂g_j = w_j − t_j` with `t = softmax(p_y/τ)`. The gradient is
  well-formed *except* that `t` is near-uniform whenever all `p_y` are small — exactly on the tail
  samples where routing matters. The gate is *instructed* to be uniform there (RC1).
- **Fixes that restore tail gradients:**
  (a) **Log-space target:** `softmax((log p_j(y) − max_i log p_i(y))/τ)` — log compression keeps
      small probabilities contrasted. Example with `p = [0.05, 0.02, 0.08]`: probability-space
      target ≈ [0.33, 0.29, 0.38] (flat, useless), log-space target ≈ [0.08, 0.001, 0.92] (sharp,
      decisive). One line in `train_one_epoch`.
  (b) **Correctness targets** (§2.4).
  (c) **Logit-space mixing** (§5): the mixture NLL gradient no longer vanishes when the mixture is
      confident, because `∂L/∂g_j` then depends on logit *differences*, not probability residuals.

---

## 3. Auxiliary Losses — forcing the gate to use all experts dynamically

### 3.1 Shazeer CV² load-balancing loss (Shazeer et al., 2017)
`L_balance = α · ( CV(Importance)² + CV(Load)² )`
- `Importance(X) = Σ_{x∈X} G(x)` (sum of gate probabilities per expert over the batch),
  `Load(X) = Σ_{x∈X} P(select expert i)` (fraction of samples where expert i is in top-k).
- `CV = std/mean` (coefficient of variation). Minimizing `CV²` drives both the *soft* routing mass
  and the *hard* selection frequency toward uniformity — directly the anti-starvation term.

### 3.2 Switch load-balancing loss (Fedus et al., 2021)
`L_aux = α · N · Σᵢ fᵢ · Pᵢ` — the standard, cheapest form. For us: `fᵢ = mean(wᵢ)` and
`Pᵢ = mean(softmax(gate_logits)ᵢ)`. One line in PyTorch.

### 3.3 RIDE distribution-aware diversity loss (Wang et al., ICLR 2021)
`L_diverse = − 1/(n−1) Σ_{j≠i} KL( p⁽ⁱ⁾(x,y) ‖ p⁽ʲ⁾(x,y) )` — i.e. **maximize** the KL divergence
between expert predictions, with a **class-wise temperature** `T_k = α(β_k + 1 − max β_j)` that is
*lower for tail classes* (making the loss more sensitive to tail disagreement).
- This regularizes the **experts** (trainable in RIDE), so it cannot be applied verbatim to our frozen
  experts. But its *spirit* transfers to the gate: add a term that rewards the gate for **combining
  experts whose predictions disagree** (high pairwise KL among the experts the gate up-weights),
  discouraging the gate from collapsing onto the single most confident expert.

### 3.4 Entropy / concentration penalties
- A cheap, widely-used gate regularizer is `−H(w)` (encourage soft, non-one-hot weights) or its
  reverse, temperature-annealed entropy *minimization* for confident-but-calibrated routing. RIDE's
  `ω_on` and SADE's uniform-init + softmax normalization are both instances of "keep weights from
  collapsing to a corner."

### 3.5 SADE prediction-stability maximization (test-time)
The auxiliary objective *is* the training signal (Section 2.3): it is a self-supervised regularizer
that also serves as the routing loss. This blurring of "loss" and "regularizer" is characteristic of
the most robust 3-expert methods.

> **Summary table**

| Regularizer | Paper | What it penalizes | Applies to frozen experts? |
|---|---|---|---|
| CV² importance+load | Shazeer 2017 | unequal routing mass / selection freq | ✅ directly |
| `N Σ fᵢ Pᵢ` balancing | Switch 2021 | unequal routing | ✅ directly |
| Max-KL diversity | RIDE 2021 | experts agreeing (correlated) | ⚠️ gate-level re-interpretation |
| Prediction stability | SADE 2022 | unstable / overconfident aggregation | ✅ directly |

---

## 4. Logit Normalization — stopping overconfident experts from dominating

This is the question that most directly explains the BS-starvation symptom, and where the literature
is clearest.

### 4.1 LogitNorm — L2-normalize logits before *anything* (Wei et al., ICML 2022)
[Mitigating Neural Network Overconfidence with Logit Normalization](https://arxiv.org/abs/2205.09310)

- **Mechanism.** Replace `f` with `f̂ = f / ‖f‖₂`, then optimize CE. This removes logit *magnitude* as
  a free parameter and forces the model to optimize only the *direction* (angular) of the logit vector,
  directly bounding overconfidence. Empirically it fixes overconfidence under imbalance/OOD.
- **For the gate:** normalize each expert's 100-dim logit to unit L2 norm (or standardize to zero
  mean/unit std) **before concatenation**. This is the single most direct fix for "CE/LA spikes
  dominate, BS soft peaks get starved" — it removes the very quantity the naive gate latches onto.

### 4.2 Temperature scaling / calibration (Guo et al., 2017; BalPoE 2023)
[On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)

- **Mechanism.** Divide logits by a temperature `T` before softmax; `T` is fit (per model or per
  expert) on a validation set to minimize ECE. BalPoE makes this *the* linchpin: it proves that
  logit-averaging heterogeneous experts is Fisher-consistent for the balanced error **only if each
  expert is calibrated**, and achieves calibration with mixup (ECE 31.5% → 4.1% on CIFAR-100-LT).
- **For the gate:** our single global `T` (swept in `do_train_val`) is insufficient — it cannot
  reconcile CE's sharp, head-peaked logits with BS's soft, prior-suppressed logits. Use **per-expert
  temperatures** `T_ce, T_la, T_bs` (fit or learned), applied to the logits *before* they enter the
  gate (not only before mixing).

### 4.3 τ-norm / logit re-normalization (Kang et al., 2020)
[Decoupling Representation and Classifier for Long-Tailed Recognition](https://arxiv.org/abs/1910.09217)

- **Mechanism.** Normalize the classifier weights (and optionally features) to unit L2 norm and
  re-scale the logits, removing scale differences between the classifier's magnitude and the
  feature's magnitude — a standard trick that equalizes logit scales across classes *and* across
  separately-trained experts.

### 4.4 Bias/prior re-alignment (logit adjustment & balanced softmax)
[Logit Adjustment — Menon et al., ICLR 2021](https://arxiv.org/abs/2007.07314) ·
[Balanced Meta-Softmax — Ren et al., NeurIPS 2020](https://arxiv.org/abs/2007.10740)

- **Mechanism.** `sᵧ = fᵧ + τ·log πᵧ`. Our three experts are exactly three points on this τ-sweep
  (CE: τ=0, BS: τ=1, LA: τ≈1.5). The gate *now* sees the **bias-adjusted probabilities** (which
  include these biases — this specific leak from the raw-logit era is fixed). What remains is the
  **contrast problem** (§2.5): the biases enter through `softmax`, which crushes tail values to ~0
  in probability space, so the *differences* between experts' tail behavior are nearly invisible to
  the gate and flat in the soft-oracle target. Log-space features/targets restore that contrast.
- BalPoE's gLA loss generalizes exactly this (`sᵧ^λ = fᵧ + τᵧ log P_train(y)`, `λᵧ = 1 − τᵧ`), and
  its `λ = {1, 0, −1}` set is our CE/BS/LA trio. Its proof that the *average* of these bias-adjusted
  logits is the unbiased predictor is the theoretical grounding for "calibrate + average" over
  "route by argmax."

### 4.5 RIDE class-wise temperature (Wang et al., 2021)
`T_k = α(β_k + 1 − max_j β_j)`, `β_k = γ·n_k/mean(n) + (1−γ)` — temperature *per class*, lower for
tail classes, used to make the diversity signal tail-sensitive. A reminder that a scalar temperature
is rarely enough; class- or expert-conditioned scaling is the norm.

### 4.6 DaWin — training-free confidence-weighted routing (Oh et al., 2024)
[DaWin: Training-free Dynamic Weight Interpolation for Robust Adaptation](https://ar5iv.labs.arxiv.org/html/2410.03782) (arXiv:2410.03782)

- Computes per-sample interpolation weights over the ensemble **without any training**:
  `w_j(x) ∝ exp(conf_j(x)/T̂)`, where `conf_j` is a temperature-scaled confidence of model j and
  `T̂` is fit once on the validation set.
- Empirically beats fixed ensembling and even some learned routers under distribution shift
  (CIFAR-C, ImageNet-C).
- **Takeaway for us:** this is a *zero-training baseline* that directly competes with the MLP gate:
  replace the gate with `softmax over experts of (max-prob_j / T̂)`, `T̂` fit on the tune set. If
  the MLP gate cannot beat DaWin on bal + tail acc, the routing signal is in the confidence
  *magnitudes* and the MLP is only adding noise. (DaWin's oracle analysis — `λ_j(x) ≥ 1/M` for
  experts that are correct — is the same ceiling as our oracle diagnostic.)

### 4.7 Long-tail-aware routers — conditioning the router on class frequency
- **[Long-Tailed Distribution-Aware Router for Mixture-of-Experts in Large Vision-Language Models](https://ar5iv.labs.arxiv.org/html/2507.01351) (Cai et al., 2025, arXiv:2507.01351):** standard routers trained on (near-)balanced data misroute tail samples; proposes a router that is aware of class frequency / the long-tailed distribution. Transfer: give our gate explicit frequency-aware features (per-class training counts, per-class expert-accuracy vector) or a class-conditioned router head.
- **[Divide, Weight, and Route: Difficulty-Aware Optimization with Dynamic Expert Fusion for Long-Tailed Recognition](https://ar5iv.labs.arxiv.org/html/2508.19630) (Wei & Yi, PRCV 2025, arXiv:2508.19630):** sample *difficulty* (head/medium/tail) modulates how experts are weighted and routed, with a difficulty-aware objective. Transfer: a two-stage gate — classify the sample's difficulty/ambiguity first, then route with a difficulty-conditioned policy. This matches RIDE's "deploy more only when uncertain" idea and our own statistics: when the mixture is wrong, all three experts disagree on 77% of samples, so *ambiguity is the strongest routing signal available*.

---

## 5. Training-time routing vs. inference-time ensembling (the key distinction)

| | Training-time routing | Inference-time ensembling |
|---|---|---|
| **Question** | how to *learn* the gate | how to *combine* the experts |
| **Shazeer/Switch/V-MoE** | noisy top-k + CV²/load-balancing loss | hard top-k selection (capacity-limited) |
| **RIDE** | diversity loss on experts; binary router trained with weighted CE + KD | *average logits* of the experts the router activated (softmax of average = product of probs) |
| **SADE** | skill-diverse expert training (different losses/sampling) | test-time-learned 3-vector weight; **logit average** |
| **BalPoE** | gLA loss per expert + mixup calibration | **average logits** (no learned gate at all) |
| **Learning to Defer** (Madras 2018; Mozannar–Sontag 2020; Cao 2023; Mao 2025) | consistency-calibrated deferral/router losses; correctness targets | softmax over experts, read as `P(expert correct)`; optionally + reject option |
| **DaWin** (2024) | none — training-free | `softmax(conf_j / T̂)` per sample |
| **This repo** | `soft-oracle KL` on a `Mini-MLP` gate (312-dim prob features) | top-2 renormalized weighted average of *probabilities* (`Σ wᵢ pᵢ`) |

**The literature's consistent verdict:** for a *small number of frozen, heterogeneous* experts, the
best "routing" is usually **learned re-weighting of logits (ensembling), not hard per-sample
selection**. Hard selection discards complementary signal and, on raw logits, degenerates to
peak-detection. Every method that survives contact with CIFAR-100-LT either (a) averages logits
(RIDE, SADE, BalPoE) or (b) uses explicit balancing/diversity terms (Shazeer, Switch), and the
newer learning-to-defer line (c) trains the router against **correctness**, not probability ratios
(Mozannar–Sontag, Cao, Mao), or (d) skips training entirely and weights by calibrated confidence
(DaWin).

> **Protocol note for this repo:** the "My Method vs Uniform" comparisons currently mix recipes —
> the method is the **top-2 renormalized** mixture while the uniform baseline averages **all 3**
> probabilities. Top-2 truncation sharpens the mixture and is a structural disadvantage for
> NLL/Brier/ECE (the paper's NLL 1.18 comes from a *calibrated* top-2 pipeline). Compare like with
> like: same recipe for both, and calibrate the final mixture (fit one temperature on the tune set)
> before computing calibration metrics.

---

## 6. Diagnosis → fix map (8/23 sweep)

| # | Root cause (evidence) | Fix (section) | Priority |
|---|---|---|---|
| RC1 | Soft-oracle target flat on tail → gate taught uniform where routing matters (`w* ≈ E[target]`, only head classes deviate) | Log-space target / correctness target (§2.5a, §2.4; rec. 1–2) | **P0** |
| RC2 | Train/select on full 3-way mixture, score on top-2 renormalized (`validate` vs `extract_posteriors`) | One mixture recipe everywhere (rec. 4) | **P0** |
| RC3 | Target = probability-ratio proxy for correctness; experts miscalibrated (tail-ECE 0.37–0.52) | Correctness targets, L2D family (§2.4; rec. 2) | **P0** |
| RC4 | Probability-space losses starve tail gradients (mixture NLL vanishes when confident; KL flat on tail) | Logit-space mixing (§5; rec. 3) | P1 |
| RC5 | 312-dim collinear features; fc has no block selectivity; ~4.5k samples | Normalize / shrink features (rec. 6–7) | P1 |
| RC6 | `la_tau` dropped in `run_temperature_comparison`; gate features fixed at T=1.0; 20% tune selection noise | Bug fixes (rec. 5) | P1 |
| RC7 | Near-uniform weights ⇒ top-2 ≈ average of weakly-ranked pair ⇒ tiny achievable gain | Two-stage router, per-class priors, capacity floors (rec. 9–11; §1.7) | P2 |

---

## 7. Concrete recommendations for this PyTorch setup (updated, prioritized)

**Status of the previous list:** (1) calibrated inputs — ✅ done (312-dim prob features);
(3) Switch balancing — ⚠️ tried in Exp 11, *caused* uniform collapse because it fights a target
that is already near-uniform (the collapse was supervision-driven, RC1, not balancing-driven);
(5) soft teacher — ⚠️ the current soft-oracle KL is the right *family* but the wrong *sharpness*;
(6) uniform init — ✅ the gate starts uniform by construction (Xavier + zero bias); the end state
is uniform because the signal is, not the init.

### P0 — fix the supervision (highest leverage, do these first)

1. **Log-space sharpened target (one line).** In `train_one_epoch`, replace
   `softmax(true_probs_experts / tau)` with
   `softmax((log(true_probs_experts) − log(true_probs_experts).max(1, keepdim=True)) / tau)`.
   Log compression makes tail targets sharp (example in §2.5a); keep τ ≈ 0.1–0.5 or anneal it.
2. **Correctness targets (L2D-consistent).** On the tune set, fit per-expert (optionally
   per-group) maps `f_j: max-prob_j → P(expert j correct)` (isotonic regression or a per-expert
   temperature); target `t = normalize([f_j(p_j(ŷ_j))])`; train the gate with the same KL (or CE).
   Consistent (Mozannar–Sontag 2020), calibrated (Cao 2023), tail-safe (§2.4).
3. **Mix in logit space.** `ŷ = argmax Σ_i w_i·(z_i + bias_i)/T_i`, with per-expert temperatures
   fit on the tune set (BalPoE §1.6/§4.2). This removes RC4's gradient starvation and matches the
   product-of-experts optimum. Keep the gate *features* in probability space (they are
   informative); only the mixture changes.
4. **One mixture recipe everywhere.** Decide full-3 vs top-2 and use it in the loss (with
   stop-gradient through the top-k selection), in `validate()`, in checkpoint selection, and in
   `extract_posteriors`. If the paper protocol is top-2, make the training loss *see* the top-2
   renormalization.
5. **Fix the eval bugs.** Add `la_tau` in `run_temperature_comparison` (RC6a); build gate features
   at the same T as the mixture (RC6b); select checkpoints on the full val split (or average over
   seeds) on a composite of bal + tail acc (RC6c).

### P1 — give the gate signal it can actually use

6. **Normalize the 300 probability dims** per expert (L2-normalize each 100-dim block, or
   per-block standardization) before concatenation — kills magnitude tracking (LogitNorm §4.1;
   RC5).
7. **Shrink the input to what predicts correctness:** RIDE-style top-s logits per expert +
   per-expert penultimate features (64-dim, L2-normalized) + entropy/margin/agreement stats
   (§1.4). Or start with a **linear router** (§1.8) — if it matches the MLP, keep the simple one.
8. **Per-expert temperatures** `T_ce, T_la, T_bs` fit on the tune set (BalPoE §4.2), applied
   before the mixture *and* before the gate features.

### P2 — policy structure

9. **Two-stage routing (RIDE-style):** default = uniform ensemble; a binary "ambiguous?" decision
   (agreement low / mixture entropy high) activates the per-sample router (§4.7, §1.4). Evidence:
   when the mixture is correct, all three experts agree 63% of the time; when wrong, only 23% —
   disagreement is the routing signal.
10. **Per-class/group priors:** initialize or bias the gate with tune-estimated per-class expert
    preference (head → CE, tail → LA; the oracle's group split is head CE 35.4% / tail LA 41.7%).
    The MLP then only learns the *residual* per-sample structure (RC7).
11. **Capacity floors** (expert-choice §1.7): clip weights to `w_i ≥ ε` (e.g. 0.05–0.1) so a rare
    tail class can never be starved by a degenerate gate.

### P3 — baselines and evaluation

12. **Compute cheap anchors the MLP must beat:** (a) per-group fixed routing (head→CE, tail→LA);
    (b) per-class fixed routing (tune-estimated); (c) DaWin confidence weighting (§4.6);
    (d) best single expert. Report bal + tail acc for each. The current MLP gain (+0.33 pp bal /
    +1.17 pp low) must be compared against these — if a fixed per-group rule matches it, the MLP is
    adding noise, not routing.
13. **Calibrate the final mixture** (one temperature on the top-2 mixture, fit on tune) before
    computing NLL/Brier/ECE; compare with the same recipe as the uniform baseline (§5 protocol
    note). The paper's NLL 1.18 comes from a calibrated top-2 pipeline.
14. **Measure the achievable ceiling:** train a logistic "correctness forecaster" per expert on the
    gate split (`features → P(correct_j)`) and report its AUC plus the bal acc of routing by its
    argmax. If even this trained ceiling is only ~1–2 pp above uniform, per-sample routing on these
    features is near its limit and the gains must come from new features (penultimate features,
    rec. 7) or from the policy (P2), not from more gate capacity.

---

## 8. References

1. Shazeer et al., 2017 — *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer* — https://arxiv.org/abs/1701.06538
2. Fedus et al., 2021 — *Switch Transformers* — https://arxiv.org/abs/2101.03961
3. Riquelme et al., 2021 — *Scaling Vision with Sparse Mixture of Experts* (V-MoE) — https://arxiv.org/abs/2106.05974
4. Wang et al., 2021 — *Long-tailed Recognition by Routing Diverse Distribution-Aware Experts* (RIDE, ICLR) — https://arxiv.org/abs/2010.01809
5. Zhang et al., 2022 — *Self-Supervised Aggregation of Diverse Experts for Test-Agnostic Long-Tailed Recognition* (SADE, NeurIPS) — https://arxiv.org/abs/2107.09249
6. Aimar et al., 2023 — *Balanced Product of Calibrated Experts for Long-Tailed Recognition* (BalPoE, CVPR) — https://arxiv.org/abs/2206.05260
7. Menon et al., 2021 — *Long-tail Learning via Logit Adjustment* (ICLR) — https://arxiv.org/abs/2007.07314
8. Ren et al., 2020 — *Balanced Meta-Softmax for Long-Tailed Visual Recognition* (NeurIPS) — https://arxiv.org/abs/2007.10740
9. Wei et al., 2022 — *Mitigating Neural Network Overconfidence with Logit Normalization* (LogitNorm, ICML) — https://arxiv.org/abs/2205.09310
10. Zhu et al., 2022 — *Balanced Contrastive Learning for Long-Tailed Visual Recognition* (BCL, CVPR) — https://arxiv.org/abs/2205.14085
11. Cui et al., 2021 — *Parametric Contrastive Learning* (PaCo, ICCV) — https://arxiv.org/abs/2010.16079
12. Kang et al., 2020 — *Decoupling Representation and Classifier for Long-Tailed Recognition* (τ-norm/cRT/LWS) — https://arxiv.org/abs/1910.09217
13. Guo et al., 2017 — *On Calibration of Modern Neural Networks* (temperature scaling, ICML) — https://arxiv.org/abs/1706.04599
14. Hinton et al., 2015 — *Distilling the Knowledge in a Neural Network* — https://arxiv.org/abs/1503.02531
15. Zhou et al., 2022 — *Mixture-of-Experts with Expert Choice Routing* (NeurIPS) — https://papers.nips.cc/paper_files/paper/2022/hash/2f00ecd787b432c1d36f3de9800728eb-Abstract-Conference.html · blog: https://research.google/blog/mixture-of-experts-with-expert-choice-routing/
16. Chi et al., 2022 — *On the Representation Collapse of Sparse Mixture of Experts* (NeurIPS) — https://mlanthology.org/neurips/2022/chi2022neurips-representation/
17. Liu et al., 2024 — *Routers in Vision Mixture of Experts: An Empirical Study* (TMLR) — https://mlanthology.org/tmlr/2024/liu2024tmlr-routers/ · arXiv:2401.15969
18. Chowdhury et al., 2024 — *A Provably Effective Method for Pruning Experts in Fine-Tuned Sparse Mixture-of-Experts* (ICML) — https://mlanthology.org/icml/2024/chowdhury2024icml-provably/
19. Madras et al., 2018 — *Predict Responsibly: Improving Fairness and Accuracy by Learning to Defer* (NeurIPS) — https://mlanthology.org/neurips/2018/madras2018neurips-predict/ · arXiv:1806.07866
20. Mozannar & Sontag, 2020 — *Consistent Estimators for Learning to Defer to an Expert* (ICML) — https://icml.cc/media/icml-2020/Slides/6448.pdf · arXiv:2006.01808
21. Cao et al., 2023 — *In Defense of Softmax Parametrization for Calibrated and Consistent Learning to Defer* (NeurIPS) — https://papers.nips.cc/paper_files/paper/2023/hash/791d3337291b2c574545aeecfa75484c-Abstract-Conference.html
22. Mao et al., 2025 — *Mastering Multiple-Expert Routing: Realizable h-Consistency and Strong Guarantees for Learning to Defer* (ICML) — https://research.google/pubs/mastering-multiple-expert-routing-realizable-h-consistency-and-strong-guarantees-for-learning-to-defer/
23. Narasimhan et al., 2024 — *Learning to Reject Meets Long-Tail Learning* (ICLR) — https://proceedings.iclr.cc/paper_files/paper/2024/hash/c4f129179494c1ea14b63fc0019f3095-Abstract-Conference.html (the "CRISP" paper this repo replicates: top-2 routing + plug-in reject rule, Bal/Worst AURC)
24. Oh et al., 2024 — *DaWin: Training-free Dynamic Weight Interpolation for Robust Adaptation* — https://ar5iv.labs.arxiv.org/html/2410.03782 (arXiv:2410.03782)
25. Cai et al., 2025 — *Long-Tailed Distribution-Aware Router for Mixture-of-Experts in Large Vision-Language Models* — https://ar5iv.labs.arxiv.org/html/2507.01351 (arXiv:2507.01351)
26. Wei & Yi, 2025 — *Divide, Weight, and Route: Difficulty-Aware Optimization with Dynamic Expert Fusion for Long-Tailed Recognition* (PRCV) — https://ar5iv.labs.arxiv.org/html/2508.19630 (arXiv:2508.19630)
27. DESlib — *Dynamic Ensemble Selection library* (KNORA, OLA/LCA competence measures) — https://deslib.readthedocs.io/
28. LightGBM-MoE — *Advanced collapse notes for mixture-of-experts training* — https://github.com/kyo219/LightGBM-MoE/blob/master/docs/moe/advanced-collapse.md
