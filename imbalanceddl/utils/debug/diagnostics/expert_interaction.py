"""ExpertAgreementTracker — agreement rates when gate is correct/incorrect,
and the correctness overlap matrix (how often 0/1/2/3 experts are correct)."""

import numpy as np
from .base import DiagnosticBase, DiagnosticResult


class ExpertAgreementTracker(DiagnosticBase):
    """Measure expert prediction agreement on correct vs incorrect gate
    predictions, and compute the correctness overlap matrix."""

    name = "Expert Agreement & Correctness Overlap"
    depends_on = ["p_mix", "p_ce", "p_la", "p_bs", "labels"]

    def run(self) -> DiagnosticResult:
        d = self.data
        method_preds = np.argmax(d.p_mix, axis=1)
        ce_preds = np.argmax(d.p_ce, axis=1)
        la_preds = np.argmax(d.p_la, axis=1)
        bs_preds = np.argmax(d.p_bs, axis=1)

        correct_mask = (method_preds == d.labels)
        incorrect_mask = ~correct_mask

        # Agreement rate when correct vs incorrect
        def agreement(p1, p2, p3, mask):
            if mask.sum() == 0:
                return 0.0
            return np.mean((p1[mask] == p2[mask])
                           & (p2[mask] == p3[mask]))

        agree_correct = agreement(ce_preds, la_preds, bs_preds, correct_mask)
        agree_incorrect = agreement(ce_preds, la_preds, bs_preds, incorrect_mask)

        # Overall agreement
        overall_agree = np.mean((ce_preds == la_preds) & (la_preds == bs_preds))

        # Correctness overlap matrix: for each sample, count experts correct
        expert_correct = np.stack([
            ce_preds == d.labels,
            la_preds == d.labels,
            bs_preds == d.labels,
        ], axis=1)  # (N, 3)
        n_correct = expert_correct.sum(axis=1)
        overlap_counts = {}
        for k in range(4):
            overlap_counts[f"{k} correct"] = int((n_correct == k).sum())

        return DiagnosticResult(
            title="Expert Agreement & Correctness Overlap",
            summary=(f"Overall prediction agreement: {overall_agree*100:.2f}%. "
                     f"When gate is CORRECT: {agree_correct*100:.2f}% agree. "
                     f"When gate is INCORRECT: {agree_incorrect*100:.2f}% agree. "
                     f"Overlap: {overlap_counts}"),
            metrics={
                "overall_agreement": float(overall_agree),
                "agree_when_correct": float(agree_correct),
                "agree_when_incorrect": float(agree_incorrect),
                **{f"n_{k}_correct": v for k, v in overlap_counts.items()},
            },
            tables=[
                {"headers": ["Metric", "Value"],
                 "rows": [
                     ("Overall agreement", f"{overall_agree*100:.2f}%"),
                     ("Agreement when CORRECT", f"{agree_correct*100:.2f}%"),
                     ("Agreement when INCORRECT", f"{agree_incorrect*100:.2f}%"),
                 ]},
                {"headers": ["# Experts Correct", "Samples"],
                 "rows": [[k, str(v)] for k, v in overlap_counts.items()]},
            ],
            verdict=("FAIL" if agree_incorrect < 0.3
                     else "PASS" if agree_incorrect > 0.5
                     else "WARN"),
            recommendation=(
                "Low incorrect agreement → gate fails to pick the correct expert, "
                "not that experts share blind spots."),
        )
