"""CalibrationAnalyzer — per-expert calibration metrics (ECE, tail-ECE)."""

import numpy as np
from ..metrics import compute_all_metrics
from .base import DiagnosticBase, DiagnosticResult


class CalibrationAnalyzer(DiagnosticBase):
    """Compact per-expert and ensemble calibration overview."""

    name = "Calibration Analysis"
    depends_on = ["p_ce", "p_la", "p_bs", "p_mix", "p_unif",
                  "l_ce", "l_la", "l_bs", "labels", "cfg"]

    def run(self) -> DiagnosticResult:
        d = self.data
        rows = []
        for label, probs, logits in [
            ("CE", d.p_ce, d.l_ce),
            ("LA", d.p_la, d.l_la),
            ("BS", d.p_bs, d.l_bs),
            ("Uniform", d.p_unif, None),
            ("Gate", d.p_mix, None),
        ]:
            m = compute_all_metrics(probs, d.labels, logits, d.cfg,
                                    getattr(d.cfg, "train_dataset", None))
            rows.append((label, f"{m['ece']:.4f}", f"{m['tail_ece']:.4f}",
                         f"{m['nll']:.4f}", f"{m['brier']:.4f}"))

        return DiagnosticResult(
            title="Calibration Analysis",
            summary="ECE and tail-ECE per method.",
            metrics={},
            tables=[{"headers": ["Method", "ECE", "Tail ECE", "NLL", "Brier"],
                     "rows": rows}],
            verdict=None,
            recommendation=None,
        )
