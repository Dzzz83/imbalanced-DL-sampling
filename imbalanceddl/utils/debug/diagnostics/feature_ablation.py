"""FeatureAblationRunner — test gate routing with different input feature
subsets (penultimate-only, probability-only, full) and linear probes."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from imbalanceddl.utils.gate_features import (build_mixture,
                                               uniform_weights)
from .base import DiagnosticBase, DiagnosticResult


class FeatureAblationRunner(DiagnosticBase):
    """Evaluate routing performance under different gate-input configurations:

    1. Penultimate features only (zero out prob columns in weight matrix)
    2. Probability features only (zero out embedding columns)
    3. Full (baseline — current gate)
    4. Linear(192, 3) probe trained on tune set
    5. Linear(316, 3) probe trained on tune set

    This directly answers: "would routing work if we changed the input
    features?"
    """

    name = "Gate Input Feature Ablation"
    depends_on = ["gate", "hidden_ce", "hidden_la", "hidden_bs",
                  "gate_input_probability", "gate_input_penultimate",
                  "l_ce", "l_la", "l_bs", "labels", "labels_tune",
                  "p_mix_tune", "p_mix", "p_unif", "recipe",
                  "cls_num_list", "group_ids"]

    def run(self) -> DiagnosticResult:
        d = self.data

        # We need penultimate hidden states for ablation
        if d.hidden_ce is None or d.gate_input_penultimate is None:
            return DiagnosticResult(
                title="Gate Input Feature Ablation",
                summary=("Penultimate features not available. "
                         "Run with penultimate-mode gate checkpoint."),
                metrics={},
    verdict=None,
                recommendation=None,
            )

        results = {}

        # ── 1. Clone gate and zero out specific input columns ──
        # Strategy: create weight masks and re-run forward pass

        N = len(d.labels)
        raw_logits = [d.l_ce, d.l_la, d.l_bs]
        cls_list = d.cls_num_list
        la_tau = d.recipe.get("la_tau", 1.5)
        T = d.recipe.get("T", 1.0)
        expert_temps = d.recipe.get("expert_temps", [1.0, 1.0, 1.0])
        k = d.recipe.get("k", 3)
        space = d.recipe.get("space", "logit")
        weight_floor = d.recipe.get("weight_floor", 0.0)
        gate_temp = d.recipe.get("gate_temp", 1.0)

        gate_input_dim = d.gate.fc.weight.shape[1]  # expected input dim

        # Helper: eval gate with a given input tensor
        def eval_gate_with_input(gate_input):
            with torch.no_grad():
                # Guard: the gate's weight matrix fixes the input dimension;
                # if the provided features have a different dimension, we
                # cannot pass them through this gate.
                if gate_input.shape[-1] != gate_input_dim:
                    return None, None
                gl = d.gate(gate_input.to(d.device))
                w = F.softmax(gl / gate_temp, dim=1)
                # Move raw logits to the same device as weights
                raw_logits_dev = [r.to(d.device) for r in raw_logits]
                pm = build_mixture(
                    raw_logits_dev, w, cls_list, la_tau,
                    T=T, per_expert_T=expert_temps,
                    k=k, space=space, weight_floor=weight_floor,
                    mix_temperature=1.0,
                ).cpu().numpy()
                preds = pm.argmax(axis=1)
                bal = np.mean([
                    np.mean(preds[d.labels == c] == c)
                    for c in range(int(preds.max()) + 1)
                    if (d.labels == c).sum() > 0
                ]) * 100
                tail_mask = d.group_ids == 2
                tail_acc = (np.mean(preds[tail_mask]
                                    == d.labels[tail_mask]) * 100
                            if tail_mask.sum() > 0 else 0.0)
                return bal, tail_acc

        # --- A) Penultimate features only ---
        # Use the cached gate_input_penultimate
        bal, tail = eval_gate_with_input(d.gate_input_penultimate)
        results["Penultimate Only (gate)"] = {"bal_acc": bal, "tail_acc": tail}

        # --- B) Probability features only ---
        if d.gate_input_probability is not None:
            bal, tail = eval_gate_with_input(d.gate_input_probability)
            if bal is not None:
                results["Probability Only (gate)"] = {"bal_acc": bal,
                                                  "tail_acc": tail}

        # --- C) Full (baseline) ---
        # This is the standard p_mix; compute from existing data
        preds = d.p_mix.argmax(axis=1)
        bal_full = np.mean([
            np.mean(preds[d.labels == c] == c)
            for c in range(int(preds.max()) + 1)
            if (d.labels == c).sum() > 0
        ]) * 100
        tail_mask = d.group_ids == 2
        tail_full = (np.mean(preds[tail_mask] == d.labels[tail_mask]) * 100
                     if tail_mask.sum() > 0 else 0.0)
        results["Full (baseline)"] = {"bal_acc": bal_full, "tail_acc": tail_full}

        # --- D) Linear(192, 3) probe ---
        if d.emb_192_tune is not None and len(d.emb_192_tune) > 0:
            bal_probe, tail_probe = self._train_linear_probe_and_eval(
                d.emb_192_tune, d.labels_tune,
                d.gate_input_penultimate.numpy(), d.labels,
                input_dim=192, d=d,
            )
            results["Linear(192,3) probe"] = {"bal_acc": bal_probe,
                                               "tail_acc": tail_probe}

        # --- E) Linear(316, 3) probe ---
        if d.gate_input_probability_tune is not None and len(d.gate_input_probability_tune) > 0:
            bal_probe, tail_probe = self._train_linear_probe_and_eval(
                d.gate_input_probability_tune, d.labels_tune,
                d.gate_input_probability, d.labels,
                input_dim=d.gate_input_probability.shape[1], d=d,
            )
            results["Linear(316,3) probe"] = {"bal_acc": bal_probe,
                                               "tail_acc": tail_probe}

        rows = [[name, f"{v['bal_acc']:.2f}%", f"{v['tail_acc']:.2f}%"]
                for name, v in results.items()]

        # Find best config
        best_config = max(results, key=lambda k: results[k]["bal_acc"])
        best_bal = results[best_config]["bal_acc"]
        baseline_bal = results.get("Full (baseline)", {}).get("bal_acc", 0)

        return DiagnosticResult(
            title="Gate Input Feature Ablation",
            summary=f"Best config: {best_config} ({best_bal:.2f}%). "
                    f"Baseline: {baseline_bal:.2f}%.",
            metrics={k.replace(" ", "_").replace("(", "").replace(")", "")
                     .lower(): v["bal_acc"]
                     for k, v in results.items()},
            tables=[{"headers": ["Config", "Bal Acc", "Tail Acc"],
                     "rows": rows}],
            verdict=None,
            recommendation=None,
        )

    def _train_linear_probe_and_eval(self, X_train, y_train, X_test, y_test,
                                     input_dim, d, lr=0.01, epochs=50):
        """Train a quick linear probe and evaluate on test set.

        The probe learns to predict the **oracle expert** (0/1/2) — the
        expert with the highest true-class probability — NOT the class
        label.  That is why ``y_train`` / ``y_test`` are converted to
        oracle targets before training.
        """
        import torch.optim as optim

        # Convert class labels to oracle expert targets (0, 1, 2)
        # using the tune-set probabilities.
        def _to_oracle_targets(probs_ce, probs_la, probs_bs, labels):
            N = len(labels)
            true_probs = np.stack([
                probs_ce[np.arange(N), labels],
                probs_la[np.arange(N), labels],
                probs_bs[np.arange(N), labels],
            ], axis=1)  # (N, 3)
            return np.argmax(true_probs, axis=1).astype(np.int64)

        oracle_train = _to_oracle_targets(
            d.p_ce_tune, d.p_la_tune, d.p_bs_tune, y_train)
        oracle_test = _to_oracle_targets(
            d.p_ce, d.p_la, d.p_bs, y_test)

        Xt = torch.from_numpy(X_train).float()
        yt = torch.from_numpy(oracle_train).long()
        Xe = torch.from_numpy(X_test).float()
        ye = torch.from_numpy(y_test).long()  # keep for bal-acc eval

        probe = nn.Linear(input_dim, 3)
        opt = optim.Adam(probe.parameters(), lr=lr)

        for _ in range(epochs):
            opt.zero_grad()
            logits = probe(Xt)
            loss = F.cross_entropy(logits, yt)
            loss.backward()
            opt.step()

        with torch.no_grad():
            w = F.softmax(probe(Xe) / d.recipe.get("gate_temp", 1.0), dim=1)
            pm = build_mixture(
                [d.l_ce, d.l_la, d.l_bs], w, d.cls_num_list,
                d.recipe.get("la_tau", 1.5),
                T=d.recipe.get("T", 1.0),
                per_expert_T=d.recipe.get("expert_temps", [1.0, 1.0, 1.0]),
                k=d.recipe.get("k", 3), space=d.recipe.get("space", "logit"),
                weight_floor=d.recipe.get("weight_floor", 0.0),
                mix_temperature=1.0,
            ).numpy()
            preds = pm.argmax(axis=1)
            bal = np.mean([
                np.mean(preds[ye.numpy() == c] == c)
                for c in range(int(preds.max()) + 1)
                if (ye.numpy() == c).sum() > 0
            ]) * 100
            tail_mask = np.array(d.group_ids) == 2
            tail_acc = (np.mean(preds[tail_mask] == ye.numpy()[tail_mask]) * 100
                        if tail_mask.sum() > 0 else 0.0)
        return bal, tail_acc
