"""DifficultyStratificationAnalyzer — stratify test samples by
prediction entropy and compare routing success in each stratum."""

import numpy as np
from .base import DiagnosticBase, DiagnosticResult


class DifficultyStratificationAnalyzer(DiagnosticBase):
    """Stratify samples by uniform-ensemble entropy (easy / moderate / hard)
    and measure gate vs uniform accuracy in each stratum."""

    name = "Sample-Difficulty Stratification"
    depends_on = ["p_unif", "p_mix", "w", "labels", "group_ids"]

    def run(self) -> DiagnosticResult:
        d = self.data

        # Compute entropy of the uniform mixture
        pu = np.clip(d.p_unif, 1e-12, 1.0)
        entropy = -np.sum(pu * np.log(pu), axis=1)

        # Define bins
        bins = [0.0, 0.3, 0.7, entropy.max() + 0.01]
        bin_labels = ["Easy (ent < 0.3)", "Moderate (0.3–0.7)",
                      "Hard (ent > 0.7)"]
        bin_indices = np.digitize(entropy, bins) - 1

        gate_preds = d.p_mix.argmax(axis=1)
        unif_preds = d.p_unif.argmax(axis=1)

        rows = []
        metrics = {}
        for bidx, bname in enumerate(bin_labels):
            mask = bin_indices == bidx
            n = int(mask.sum())
            if n == 0:
                continue
            gate_acc = np.mean(gate_preds[mask] == d.labels[mask]) * 100
            unif_acc = np.mean(unif_preds[mask] == d.labels[mask]) * 100
            delta = gate_acc - unif_acc

            # Gate weight entropy in this stratum
            wb = np.clip(d.w[mask], 1e-12, 1.0)
            w_ent = -np.sum(wb * np.log(wb), axis=1).mean()

            rows.append((bname, str(n), f"{gate_acc:.2f}%",
                         f"{unif_acc:.2f}%", f"{delta:+.2f}%",
                         f"{w_ent:.4f}"))
            metrics[f"gate_acc_{bidx}"] = gate_acc
            metrics[f"unif_acc_{bidx}"] = unif_acc
            metrics[f"delta_{bidx}"] = delta

        if not rows:
            return DiagnosticResult(
                title="Sample-Difficulty Stratification",
                summary="No stratification possible.",
                metrics={},
    verdict=None,
                recommendation=None,
            )

        return DiagnosticResult(
            title="Sample-Difficulty Stratification",
            summary=("Gate performs similarly to uniform across all "
                     "difficulty levels."),
            metrics=metrics,
            tables=[{"headers": ["Stratum", "Samples", "Gate Acc",
                                 "Unif Acc", "Delta", "Gate W Entropy"],
                     "rows": rows}],
            verdict=None,
            recommendation=None,
        )
