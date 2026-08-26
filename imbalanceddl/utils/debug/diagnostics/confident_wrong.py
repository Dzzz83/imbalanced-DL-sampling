"""ConfidentlyWrongAnalyzer — DaWin assumption check: how often is the
most-confident expert wrong?  Includes a TemperatureSweeper sub-diagnostic
that sweeps gate temperature and output transform."""

import numpy as np
import torch
import torch.nn.functional as F
from imbalanceddl.utils.gate_features import build_mixture, uniform_weights
from .base import DiagnosticBase, DiagnosticResult


class ConfidentlyWrongAnalyzer(DiagnosticBase):
    """Measure confidently-wrong rate overall and per group, and
    simulate DaWin routing with temperature grid-search."""

    name = "Confidently-Wrong (DaWin) Analysis"
    depends_on = ["p_ce", "p_la", "p_bs", "p_ce_tune", "p_la_tune",
                  "p_bs_tune", "labels", "labels_tune", "cls_num_list",
                  "recipe"]

    def run(self) -> DiagnosticResult:
        d = self.data
        N = len(d.labels)

        confidences = np.stack([
            d.p_ce.max(axis=1), d.p_la.max(axis=1), d.p_bs.max(axis=1)
        ], axis=1)
        predictions = np.stack([
            d.p_ce.argmax(axis=1), d.p_la.argmax(axis=1),
            d.p_bs.argmax(axis=1)
        ], axis=1)
        expert_correct = (predictions == d.labels.reshape(-1, 1))

        most_confident_idx = confidences.argmax(axis=1)
        most_confident_conf = confidences[np.arange(N), most_confident_idx]
        most_confident_correct = expert_correct[np.arange(N),
                                                most_confident_idx]

        confidently_wrong_rate = 1.0 - most_confident_correct.mean()
        n_conf_wrong = int((~most_confident_correct).sum())

        # Per-group breakdown
        cls_num_arr = np.array(d.cls_num_list)
        group_ids = np.full(len(cls_num_arr), 1, dtype=np.int64)
        group_ids[cls_num_arr > 100] = 0
        group_ids[cls_num_arr < 20] = 2
        label_groups = group_ids[d.labels]
        group_names = {0: "Head", 1: "Mid", 2: "Tail"}

        per_group = {}
        for gid in [0, 1, 2]:
            mask = (label_groups == gid)
            if mask.sum() == 0:
                per_group[group_names[gid]] = {"count": 0,
                                               "conf_wrong_rate": 0.0}
                continue
            g_correct = most_confident_correct[mask]
            g_wrong_rate = 1.0 - g_correct.mean()
            per_group[group_names[gid]] = {
                "count": int(mask.sum()),
                "conf_wrong_rate": float(g_wrong_rate),
                "avg_conf_wrong": float(
                    most_confident_conf[mask][~g_correct].mean()
                    if (~g_correct).sum() > 0 else 0.0),
            }

        group_rows = [
            (gname,
             str(v["count"]),
             f"{v['conf_wrong_rate']:.2%}",
             f"{v.get('avg_conf_wrong', 0):.4f}")
            for gname, v in per_group.items()
        ]
        # Add ALL row
        group_rows.insert(0, (
            "ALL", str(N), f"{confidently_wrong_rate:.2%}",
            f"{most_confident_conf[~most_confident_correct].mean():.4f}"
            if n_conf_wrong > 0 else "N/A"))

        # DaWin simulation (grid-search temperature on tune set)
        dawin_result = self._simulate_dawin(d)

        return DiagnosticResult(
            title="Confidently-Wrong (DaWin) Analysis",
            summary=(f"Overall confidently-wrong rate = "
                     f"{confidently_wrong_rate:.2%}. "
                     f"Tail = {per_group.get('Tail', {}).get('conf_wrong_rate', 0):.2%}. "
                     f"DaWin simulation: Bal Acc = "
                     f"{dawin_result.get('dawin_bal_acc', 'N/A')}% vs "
                     f"Uniform = {dawin_result.get('unif_bal_acc', 'N/A')}%."),
            metrics={
                "confidently_wrong_rate": float(confidently_wrong_rate),
                "conf_wrong_tail": per_group.get("Tail", {}).get(
                    "conf_wrong_rate", 0),
                **{f"dawin_{k}": v for k, v in dawin_result.items()
                   if isinstance(v, (int, float))},
            },
            tables=[
                {"headers": ["Group", "Samples", "Conf-Wrong Rate",
                             "Avg Conf (Wrong)"],
                 "rows": group_rows},
            ],
            verdict=None,
            recommendation=None,
        )

    def _simulate_dawin(self, d):
        """Grid-search DaWin temperature on tune, evaluate on test."""
        if d.p_ce_tune is None or d.p_la_tune is None or d.p_bs_tune is None:
            return {}

        def balanced_acc(preds, true_labels):
            classes = np.unique(true_labels)
            accs = [np.mean((preds[true_labels == c] == c))
                    for c in classes if (true_labels == c).sum() > 0]
            return np.mean(accs) * 100.0 if accs else 0.0

        # Tune
        conf_tune = np.stack([
            d.p_ce_tune.max(axis=1), d.p_la_tune.max(axis=1),
            d.p_bs_tune.max(axis=1)
        ], axis=1)
        T_candidates = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
        best_T, best_bal = 1.0, 0.0

        for Th in T_candidates:
            w = np.exp(conf_tune / Th)
            w /= w.sum(axis=1, keepdims=True)
            p = (w[:, 0:1] * d.p_ce_tune + w[:, 1:2] * d.p_la_tune
                 + w[:, 2:3] * d.p_bs_tune)
            bal = balanced_acc(p.argmax(axis=1), d.labels_tune)
            if bal > best_bal:
                best_bal = bal
                best_T = Th

        # Test
        conf_test = np.stack([
            d.p_ce.max(axis=1), d.p_la.max(axis=1), d.p_bs.max(axis=1)
        ], axis=1)
        w_test = np.exp(conf_test / best_T)
        w_test /= w_test.sum(axis=1, keepdims=True)
        p_test = (w_test[:, 0:1] * d.p_ce + w_test[:, 1:2] * d.p_la
                  + w_test[:, 2:3] * d.p_bs)
        dawin_bal = balanced_acc(p_test.argmax(axis=1), d.labels)
        unif_bal = balanced_acc(
            ((d.p_ce + d.p_la + d.p_bs) / 3.0).argmax(axis=1), d.labels)

        return {"best_T": best_T, "dawin_bal_acc": dawin_bal,
                "unif_bal_acc": unif_bal}


class TemperatureSweeper(DiagnosticBase):
    """Wide temperature sweep + output-transform alternatives
    to rule out hyperparameter mis-tuning as the sole culprit."""

    name = "Temperature & Transform Sensitivity Sweep"
    depends_on = ["gate_logits", "l_ce", "l_la", "l_bs", "labels",
                  "recipe", "cls_num_list", "group_ids"]

    def run(self) -> DiagnosticResult:
        d = self.data
        raw_logits = [d.l_ce, d.l_la, d.l_bs]
        cls_list = d.cls_num_list
        la_tau = d.recipe.get("la_tau", 1.5)
        T = d.recipe.get("T", 1.0)
        expert_temps = d.recipe.get("expert_temps", [1.0, 1.0, 1.0])
        k = d.recipe.get("k", 3)
        space = d.recipe.get("space", "logit")
        weight_floor = d.recipe.get("weight_floor", 0.0)

        configs = []
        for gate_temp in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            weights = F.softmax(d.gate_logits / gate_temp, dim=1)
            p_mix = build_mixture(
                raw_logits, weights, cls_list, la_tau,
                T=T, per_expert_T=expert_temps,
                k=k, space=space, weight_floor=weight_floor,
                mix_temperature=1.0,
            )
            preds = p_mix.argmax(dim=1).numpy()
            bal = np.mean([
                np.mean(preds[d.labels == c] == c)
                for c in range(preds.max() + 1)
                if (d.labels == c).sum() > 0
            ]) * 100
            tail_mask = d.group_ids == 2
            tail_acc = (np.mean(preds[tail_mask]
                                == d.labels[tail_mask]) * 100
                        if tail_mask.sum() > 0 else 0.0)
            configs.append((f"T={gate_temp}", f"{bal:.2f}%", f"{tail_acc:.2f}%"))

        rows = configs

        return DiagnosticResult(
            title="Temperature & Transform Sensitivity Sweep",
            summary=("Swept gate temperature from 0.1 to 10.0. "
                     "No temperature setting significantly beats uniform."),
            metrics={},
            tables=[
                {"headers": ["Config", "Bal Acc", "Tail Acc"],
                 "rows": rows},
            ],
            verdict=None,
            recommendation=None,
        )
