"""ExpertPerformanceSummary — compact single-table overview of all methods."""

import numpy as np
from ..metrics import compute_all_metrics
from .base import DiagnosticBase, DiagnosticResult


class ExpertPerformanceSummary(DiagnosticBase):
    """Balanced accuracy, per-group accuracy, NLL, Brier, ECE, tail-ECE
    for each expert, the uniform ensemble, and the gate-routed method."""

    name = "Expert Performance Summary"
    depends_on = ["p_ce", "p_la", "p_bs", "p_mix", "p_unif",
                  "l_ce", "l_la", "l_bs", "labels", "cfg"]

    def run(self) -> DiagnosticResult:
        d = self.data
        rows = []
        best_per_group = {"bal_acc": ("", 0.0), "many": ("", 0.0),
                          "med": ("", 0.0), "low": ("", 0.0)}

        for label, probs, logits in [
            ("CE", d.p_ce, d.l_ce),
            ("LA", d.p_la, d.l_la),
            ("BS", d.p_bs, d.l_bs),
        ]:
            m = compute_all_metrics(probs, d.labels, logits, d.cfg,
                                    getattr(d.cfg, "train_dataset", None))
            rows.append(self._row(label, m, best_per_group))

        m_unif = compute_all_metrics(d.p_unif, d.labels, None, d.cfg,
                                     getattr(d.cfg, "train_dataset", None))
        rows.append(self._row("Uniform", m_unif, best_per_group))

        m_gate = compute_all_metrics(d.p_mix, d.labels, None, d.cfg,
                                     getattr(d.cfg, "train_dataset", None))
        rows.append(self._row("Gate", m_gate, best_per_group))

        # Delta row
        delta = {}
        for k in ["bal_acc", "many", "med", "low", "nll", "tail_ece"]:
            delta[k] = m_gate[k] - m_unif[k]
        delta_row = (
            f"Δ (Gate−Unif)",
            f"{delta['bal_acc']:+6.2f}",
            f"{delta['many']:+6.2f}",
            f"{delta['med']:+6.2f}",
            f"{delta['low']:+6.2f}",
            f"{delta['nll']:+7.4f}",
            f"{delta['tail_ece']:+7.4f}",
        )

        headers = ["Method", "Bal Acc", "Many", "Med", "Low",
                    "NLL", "Tail ECE"]
        all_rows = rows + [delta_row]

        # Verdict
        if m_gate["bal_acc"] > m_unif["bal_acc"] + 0.5:
            verdict = "PASS"
            rec = "Routing improves over uniform. Proceed to hyperparameter tuning."
        elif m_gate["bal_acc"] > m_unif["bal_acc"]:
            verdict = "WARN"
            rec = ("Routing marginally beats uniform (Δ < 0.5 pp). "
                   "Investigate tail gain vs head loss.")
        else:
            verdict = "FAIL"
            rec = ("Routing does NOT improve over uniform. "
                   "See Sections 3–6 for root causes.")

        best_str = " ◆ Best expert: " + ", ".join(
            f"{k}={v[0]}" for k, v in best_per_group.items()
        )

        return DiagnosticResult(
            title="Expert Performance Summary",
            summary=f"Gate Bal Acc = {m_gate['bal_acc']:.2f}% vs "
                    f"Uniform = {m_unif['bal_acc']:.2f}% "
                    f"(Δ = {m_gate['bal_acc'] - m_unif['bal_acc']:+.2f} pp)."
                    + best_str,
            metrics={
                "gate_bal_acc": m_gate["bal_acc"],
                "uniform_bal_acc": m_unif["bal_acc"],
                "gate_tail_acc": m_gate["low"],
                "uniform_tail_acc": m_unif["low"],
                "delta_bal_acc": m_gate["bal_acc"] - m_unif["bal_acc"],
                "delta_tail_acc": m_gate["low"] - m_unif["low"],
            },
            tables=[{"headers": headers, "rows": all_rows}],
            verdict=verdict,
            recommendation=rec,
        )

    @staticmethod
    def _row(label, m, best):
        bal = f"{m['bal_acc']:.2f}"
        many = f"{m['many']:.2f}"
        med = f"{m['med']:.2f}"
        low = f"{m['low']:.2f}"
        nll = f"{m['nll']:.4f}"
        tece = f"{m['tail_ece']:.4f}"
        # Track best per group
        for k, field in [("bal_acc", "bal_acc"), ("many", "many"),
                          ("med", "med"), ("low", "low")]:
            cur_best_name, cur_best_val = best[k]
            if m[field] > cur_best_val:
                best[k] = (label, m[field])
        return (label, bal, many, med, low, nll, tece)
