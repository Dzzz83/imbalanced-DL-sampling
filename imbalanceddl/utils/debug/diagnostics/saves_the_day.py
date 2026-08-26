"""SavesTheDayAnalyzer — for tail samples where LA is the sole correct
expert, does the gate assign it sufficient weight?"""

import numpy as np
from .base import DiagnosticBase, DiagnosticResult


class SavesTheDayAnalyzer(DiagnosticBase):
    """Check if LA (the tail-specialist) gets enough weight when it is
    the only correct expert on tail samples."""

    name = "LA 'Saves the Day' Analysis"
    depends_on = ["p_ce", "p_la", "p_bs", "w", "labels", "group_ids",
                  "recipe"]

    def run(self) -> DiagnosticResult:
        d = self.data
        ce_preds = np.argmax(d.p_ce, axis=1)
        la_preds = np.argmax(d.p_la, axis=1)
        bs_preds = np.argmax(d.p_bs, axis=1)

        # Tail samples where LA is correct and both CE and BS are wrong
        tail_mask = d.group_ids == 2  # assuming group_ids uses head=0, mid=1, tail=2
        la_correct = la_preds == d.labels
        ce_bs_wrong = (ce_preds != d.labels) & (bs_preds != d.labels)
        saves_day = tail_mask & la_correct & ce_bs_wrong
        saves_idx = np.where(saves_day)[0]
        n_saves = len(saves_idx)

        if n_saves == 0:
            return DiagnosticResult(
                title="LA 'Saves the Day' Analysis",
                summary="No tail samples found where LA was the sole correct expert.",
                metrics={},
                verdict=None,
                recommendation=None,
            )

        avg_w = np.mean(d.w[saves_idx], axis=0)
        k = d.recipe.get("k", 3)
        topk_indices = np.argsort(d.w[saves_idx], axis=1)[:, ::-1][:, :k]
        la_chosen = np.sum(topk_indices == 1)
        la_chosen_pct = la_chosen / n_saves * 100

        # Also compute average true-class probability
        true_probs = np.stack([
            d.p_ce[np.arange(len(d.labels)), d.labels],
            d.p_la[np.arange(len(d.labels)), d.labels],
            d.p_bs[np.arange(len(d.labels)), d.labels],
        ], axis=1)
        avg_true_prob = true_probs[saves_idx].mean(axis=0)

        return DiagnosticResult(
            title="LA 'Saves the Day' Analysis",
            summary=(f"Found {n_saves} tail samples where LA was sole correct. "
                     f"Avg weights: CE={avg_w[0]:.4f}, LA={avg_w[1]:.4f}, "
                     f"BS={avg_w[2]:.4f}. "
                     f"LA in top-{k}: {la_chosen}/{n_saves} ({la_chosen_pct:.1f}%)."),
            metrics={
                "n_la_saves_day": n_saves,
                "avg_w_ce_when_sole_correct": float(avg_w[0]),
                "avg_w_la_when_sole_correct": float(avg_w[1]),
                "avg_w_bs_when_sole_correct": float(avg_w[2]),
                "la_chosen_in_topk_pct": la_chosen_pct,
                "avg_true_prob_ce": float(avg_true_prob[0]),
                "avg_true_prob_la": float(avg_true_prob[1]),
                "avg_true_prob_bs": float(avg_true_prob[2]),
            },
            tables=[
                {"headers": ["When LA sole correct", "Value"],
                 "rows": [
                     ("Samples found", str(n_saves)),
                     ("Avg w_CE", f"{avg_w[0]:.4f}"),
                     ("Avg w_LA", f"{avg_w[1]:.4f}"),
                     ("Avg w_BS", f"{avg_w[2]:.4f}"),
                     ("LA in top-k", f"{la_chosen}/{n_saves} ({la_chosen_pct:.1f}%)"),
                     ("Avg true prob CE", f"{avg_true_prob[0]:.4f}"),
                     ("Avg true prob LA", f"{avg_true_prob[1]:.4f}"),
                     ("Avg true prob BS", f"{avg_true_prob[2]:.4f}"),
                 ]},
            ],
            verdict=None,
            recommendation=None,
        )
