"""PenultimateRoutingSimulator — quick evaluation of a Linear(192,3)
router trained on tune-set features."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from imbalanceddl.utils.gate_features import build_mixture
from .base import DiagnosticBase, DiagnosticResult


class PenultimateRoutingSimulator(DiagnosticBase):
    """Train a simple Linear(192, 3) router on the tune set using the
    penultimate embeddings and evaluate on the test set.

    This directly tests the recommendation from the feature correlation
    diagnostic: "switch to penultimate feature routing."
    """

    name = "Penultimate Routing Simulation"
    depends_on = ["emb_192_tune", "emb_192", "labels_tune", "labels",
                  "l_ce", "l_la", "l_bs", "recipe", "cls_num_list",
                  "group_ids"]

    def run(self) -> DiagnosticResult:
        d = self.data

        if d.emb_192_tune is None or d.emb_192 is None:
            return DiagnosticResult(
                title="Penultimate Routing Simulation",
                summary="Penultimate embeddings not available. "
                        "Run with penultimate-mode gate.",
                metrics={},
    verdict=None,
                recommendation=None,
            )

        # Train linear probe — predict oracle expert (0/1/2), not class label
        def _oracle_targets(probs_ce, probs_la, probs_bs, labels):
            N = len(labels)
            true_probs = np.stack([
                probs_ce[np.arange(N), labels],
                probs_la[np.arange(N), labels],
                probs_bs[np.arange(N), labels],
            ], axis=1)
            return np.argmax(true_probs, axis=1).astype(np.int64)

        oracle_train = _oracle_targets(
            d.p_ce_tune, d.p_la_tune, d.p_bs_tune, d.labels_tune)

        X_train = torch.from_numpy(d.emb_192_tune).float()
        y_train = torch.from_numpy(oracle_train).long()
        X_test = torch.from_numpy(d.emb_192).float()

        probe = nn.Linear(192, 3)
        optimizer = torch.optim.Adam(probe.parameters(), lr=0.01)

        for epoch in range(100):
            optimizer.zero_grad()
            logits = probe(X_train)
            loss = F.cross_entropy(logits, y_train)
            loss.backward()
            optimizer.step()

        # Evaluate
        with torch.no_grad():
            w = F.softmax(probe(X_test) / d.recipe.get("gate_temp", 1.0),
                          dim=1)
            pm = build_mixture(
                [d.l_ce, d.l_la, d.l_bs], w, d.cls_num_list,
                d.recipe.get("la_tau", 1.5),
                T=d.recipe.get("T", 1.0),
                per_expert_T=d.recipe.get("expert_temps", [1.0, 1.0, 1.0]),
                k=d.recipe.get("k", 3),
                space=d.recipe.get("space", "logit"),
                weight_floor=d.recipe.get("weight_floor", 0.0),
                mix_temperature=1.0,
            ).numpy()
            preds = pm.argmax(axis=1)
            bal = np.mean([
                np.mean(preds[d.labels == c] == c)
                for c in range(int(preds.max()) + 1)
                if (d.labels == c).sum() > 0
            ]) * 100
            tail_mask = d.group_ids == 2
            tail_acc = (np.mean(preds[tail_mask] == d.labels[tail_mask]) * 100
                        if tail_mask.sum() > 0 else 0.0)

        # Baseline comparison
        unif_preds = ((d.p_ce + d.p_la + d.p_bs) / 3.0).argmax(axis=1)
        unif_bal = np.mean([
            np.mean(unif_preds[d.labels == c] == c)
            for c in range(int(unif_preds.max()) + 1)
            if (d.labels == c).sum() > 0
        ]) * 100
        unif_tail = (np.mean(unif_preds[tail_mask] == d.labels[tail_mask])
                     * 100 if tail_mask.sum() > 0 else 0.0)

        delta_bal = bal - unif_bal
        delta_tail = tail_acc - unif_tail

        return DiagnosticResult(
            title="Penultimate Routing Simulation",
            summary=(f"Linear(192,3) Bal Acc = {bal:.2f}% vs "
                     f"Uniform = {unif_bal:.2f}% "
                     f"(Δ = {delta_bal:+.2f} pp). "
                     f"Tail Acc = {tail_acc:.2f}% vs {unif_tail:.2f}% "
                     f"(Δ = {delta_tail:+.2f} pp)."),
            metrics={
                "penultimate_sim_bal_acc": bal,
                "penultimate_sim_tail_acc": tail_acc,
                "penultimate_sim_delta_bal": delta_bal,
                "penultimate_sim_delta_tail": delta_tail,
            },
            tables=[{"headers": ["Method", "Bal Acc", "Tail Acc"],
                     "rows": [
                         ("Linear(192,3)", f"{bal:.2f}%", f"{tail_acc:.2f}%"),
                         ("Uniform", f"{unif_bal:.2f}%", f"{unif_tail:.2f}%"),
                         ("Delta", f"{delta_bal:+.2f}pp",
                          f"{delta_tail:+.2f}pp"),
                     ]}],
            verdict=None,
            recommendation=None,
        )
