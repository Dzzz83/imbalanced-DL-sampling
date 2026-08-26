"""OracleGapAnalyzer — compute the oracle upper bound, current gate
performance, and the headroom (gap) per group."""

import numpy as np
from ..metrics import compute_all_metrics
from imbalanceddl.utils.metrics import shot_acc
from .base import DiagnosticBase, DiagnosticResult


class OracleGapAnalyzer(DiagnosticBase):
    """Oracle expert selection, its accuracy, and the gap between oracle
    and current gate performance per group.

    The 'headroom ratio' = gap / (100 - oracle) tells us what fraction
    of the remaining possible improvement the gate is leaving on the table.
    """

    name = "Oracle Gap & Headroom Analysis"
    depends_on = ["p_ce", "p_la", "p_bs", "p_mix", "labels", "group_ids",
                  "cls_num_list", "cfg"]

    def run(self) -> DiagnosticResult:
        d = self.data
        N = len(d.labels)

        # ── 1. Oracle: pick expert with highest true-class probability ──
        true_probs = np.stack([
            d.p_ce[np.arange(N), d.labels],
            d.p_la[np.arange(N), d.labels],
            d.p_bs[np.arange(N), d.labels],
        ], axis=1)  # (N, 3)
        oracle_expert = np.argmax(true_probs, axis=1)

        oracle_preds = np.zeros(N, dtype=np.int64)
        for i in range(N):
            exp = oracle_expert[i]
            oracle_preds[i] = np.argmax(
                [d.p_ce[i], d.p_la[i], d.p_bs[i]][exp]
            )

        # ── 2. Oracle accuracy ──
        oracle_bal = np.mean([
            np.mean(oracle_preds[d.labels == c] == c)
            for c in range(d.num_classes)
            if np.sum(d.labels == c) > 0
        ]) * 100

        from imbalanceddl.utils.metrics import shot_acc
        oracle_many, oracle_med, oracle_low = shot_acc(
            d.cfg, oracle_preds, d.labels,
            getattr(d.cfg, "train_dataset", None), acc_per_cls=False
        )

        # ── 3. Gate accuracy ──
        gate_preds = np.argmax(d.p_mix, axis=1)
        gate_bal = np.mean([
            np.mean(gate_preds[d.labels == c] == c)
            for c in range(d.num_classes)
            if np.sum(d.labels == c) > 0
        ]) * 100

        # ── 4. Gap per group ──
        group_map = {0: "Head", 1: "Mid", 2: "Tail"}
        gaps = {}
        counts = {}
        for gid, gname in group_map.items():
            mask = d.group_ids == gid
            counts[gname] = int(mask.sum())
            if mask.sum() == 0:
                gaps[gname] = 0.0
                continue
            o_acc = np.mean(oracle_preds[mask] == d.labels[mask]) * 100
            g_acc = np.mean(gate_preds[mask] == d.labels[mask]) * 100
            gaps[gname] = o_acc - g_acc

        # ── 5. Oracle selection frequency ──
        expert_counts = {
            name: int((oracle_expert == i).sum())
            for i, name in enumerate(["CE", "LA", "BS"])
        }

        # ── 6. Wasted oracle selections ──
        # Fraction of oracle-chosen experts that get below-uniform weight
        w = d.w
        if w is not None:
            oracle_weights = w[np.arange(N), oracle_expert]
            wasted = np.mean(oracle_weights <= 1.0 / 3) * 100
        else:
            wasted = None

        # Headroom ratio
        total_gap = oracle_bal - gate_bal
        remaining = max(100.0 - oracle_bal, 0.1)
        headroom_ratio = total_gap / remaining * 100 if remaining > 0 else 0.0

        return DiagnosticResult(
            title="Oracle Gap & Headroom Analysis",
            summary=(f"Oracle Bal Acc = {oracle_bal:.2f}% vs "
                     f"Gate = {gate_bal:.2f}% "
                     f"(gap = {total_gap:.2f} pp, headroom = {headroom_ratio:.1f}%)."),
            metrics={
                "oracle_bal_acc": oracle_bal,
                "gate_bal_acc": gate_bal,
                "gap_bal_acc": total_gap,
                "headroom_ratio": headroom_ratio,
                "oracle_many_acc": oracle_many * 100,
                "oracle_med_acc": oracle_med * 100,
                "oracle_low_acc": oracle_low * 100,
                "wasted_oracle_pct": wasted or 0.0,
            },
            tables=[
                {"headers": ["Group", "Samples", "Oracle Gap (pp)"],
                 "rows": [
                     (gname, str(counts[gname]), f"{gap:.2f}")
                     for gname, gap in gaps.items()
                 ]},
                {"headers": ["Expert", "Oracle Choices"],
                 "rows": [(n, str(c)) for n, c in expert_counts.items()]},
            ],
            verdict=("FAIL" if total_gap > 5.0
                     else "WARN" if total_gap > 2.0
                     else "PASS"),
            recommendation=(
                f"Oracle gap = {total_gap:.1f} pp. "
                f"Headroom ratio = {headroom_ratio:.1f}%. "
                "A large gap means the gate is far from optimal. "
                "A small gap means limited routing headroom — "
                "experts lack complementarity."),
        )
