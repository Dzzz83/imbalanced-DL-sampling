# Routing Mechanisms for MoE & Ensembles in Long-Tailed Recognition

**A targeted literature review for Stage 2 (gate routing) of the CIFAR-100-LT MoE project.**
Setup under review: 3 frozen ResNet-32 experts (CE = head-biased, LA = tail-biased, BS = intermediate),
a lightweight gate consuming the **300-dim concatenated raw logits**, and a `Mixture NLL` training objective.

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

> **Critical observation specific to this repo.** In `_gate_trainer.py`, the gate is fed the
> **raw 300-dim logits** (`embeddings`), but the *mixing* happens in **probability space**
> (`mix_prob = Σ wᵢ · softmax((zᵢ + biasᵢ)/T)`). The gate therefore reasons about a representation
> (raw, magnitude-spiky logits) that is *different* from the representation it is ultimately asked
> to combine (calibrated, bias-adjusted probabilities). The SOTA answer to "why does the gate track
> max-logit spikes?" is largely *this*: it is being shown the wrong, un-normalized representation.

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

### 2.2 Mixture NLL / soft mixture (what we currently use)
`L = −log Σᵢ wᵢ · pᵢ(y|x)` — train the gate to minimize CE of the weighted expert mixture.
- **RIDE explicitly warns against this when experts are *trainable*** (their "collaborative loss"):
  it makes experts *correlated* rather than *complementary*. **This objection does not apply to us** —
  our experts are frozen, so mixture NLL cannot collapse them into each other; it only risks the *gate*
  over-trusting the highest-confidence expert.
- **The fix within mixture NLL:** add the balancing/entropy regularizer (Section 3) and calibrate the
  inputs (Section 4). The objective is sound; the *input representation and regularization* are what
  are missing.

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
  (CE: τ=0, BS: τ=1, LA: τ≈1.5). The gate sees raw `f` (τ=0 representation) for all three, so it
  cannot "see" the prior biases that differentiate them. **This is a real information leak in our
  pipeline:** the gate should see the *bias-adjusted* logits (`fᵧ + τᵢ log πᵧ` per expert) or the
  *probabilities*, not the raw unadjusted scorers, otherwise BS looks artificially "flat" to the gate.
- BalPoE's gLA loss generalizes exactly this (`sᵧ^λ = fᵧ + τᵧ log P_train(y)`, `λᵧ = 1 − τᵧ`), and
  its `λ = {1, 0, −1}` set is our CE/BS/LA trio. Its proof that the *average* of these bias-adjusted
  logits is the unbiased predictor is the theoretical grounding for "calibrate + average" over
  "route by argmax."

### 4.5 RIDE class-wise temperature (Wang et al., 2021)
`T_k = α(β_k + 1 − max_j β_j)`, `β_k = γ·n_k/mean(n) + (1−γ)` — temperature *per class*, lower for
tail classes, used to make the diversity signal tail-sensitive. A reminder that a scalar temperature
is rarely enough; class- or expert-conditioned scaling is the norm.

---

## 5. Training-time routing vs. inference-time ensembling (the key distinction)

| | Training-time routing | Inference-time ensembling |
|---|---|---|
| **Question** | how to *learn* the gate | how to *combine* the experts |
| **Shazeer/Switch/V-MoE** | noisy top-k + CV²/load-balancing loss | hard top-k selection (capacity-limited) |
| **RIDE** | diversity loss on experts; binary router trained with weighted CE + KD | *average logits* of the experts the router activated (softmax of average = product of probs) |
| **SADE** | skill-diverse expert training (different losses/sampling) | test-time-learned 3-vector weight; **logit average** |
| **BalPoE** | gLA loss per expert + mixup calibration | **average logits** (no learned gate at all) |
| **This repo** | `Mixture NLL` on a `Mini-MLP` gate | weighted average of *probabilities* (`Σ wᵢ pᵢ`) |

**The literature's consistent verdict:** for a *small number of frozen, heterogeneous* experts, the
best "routing" is usually **learned re-weighting of logits (ensembling), not hard per-sample
selection**. Hard selection discards complementary signal and, on raw logits, degenerates to
peak-detection. Every method that survives contact with CIFAR-100-LT either (a) averages logits
(RIDE, SADE, BalPoE) or (b) uses explicit balancing/diversity terms (Shazeer, Switch).

---

## 6. Concrete recommendations for this PyTorch setup (3 frozen experts)

Ordered by expected impact / effort. All are compatible with the existing `GateMLP` and `Mixture NLL`.

1. **Feed the gate calibrated inputs, not raw logits.**
   Replace `embeddings = cat(logits)` with `cat([normalize(zᵢ); …])` where each expert's logits are
   (a) bias-adjusted (`zᵢ + τᵢ·log π`) and (b) per-expert standardized or L2-normalized
   (LogitNorm §4.1, logit-adjustment §4.4). This is the highest-leverage change and directly kills
   the "max-logit spike" signal.

2. **Per-expert temperature (not one global T).** Make `T_ce, T_la, T_bs` learnable scalars or fit
   them on the tune split (BalPoE §4.2), applied *before* the gate sees the logits.

3. **Add a Switch-style balancing loss.** `L_aux = α·3·Σᵢ mean(wᵢ)·mean(softmax(gate)ᵢ)` with a small
   α. One line, directly targets BS starvation (§3.2).

4. **Switch the mixing from probabilities to logits.** `ŷ = softmax(Σᵢ wᵢ·(zᵢ + biasᵢ)/Tᵢ)` — logit
   averaging is what RIDE/SADE/BalPoE all converge on and is the product-of-experts optimum (§5).

5. **Soft teacher instead of (or in addition to) mixture NLL.** Distill the gate against the
   equally-weighted ensemble's soft probability, or add SADE's prediction-stability term on two
   augmented views (§2.3). Removes the need for a hard "correct expert."

6. **If you keep a 3-way softmax gate:** (a) initialize the output bias so the gate starts at
   uniform (≈ equal weights) rather than argmax; (b) add training-time Gaussian noise to gate logits
   (Shazeer §1.1); (c) report `mean(wᵢ)` per epoch as the collapse monitor (already logged).

7. **Do not use the oracle argmax (`target_expert`) as a training target** — keep it purely as the
   diagnostic it currently is (§2.1).

---

## 7. References

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
