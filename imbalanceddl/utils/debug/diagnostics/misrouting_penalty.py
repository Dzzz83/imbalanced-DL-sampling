"""MisroutingPenaltyAnalyzer — when the gate does NOT pick the best
expert, measure the cost in accuracy and confidence, broken down by
group and severity."""

import numpy as np
from .base import DiagnosticBase, DiagnosticResult


class MisroutingPenaltyAnalyzer(DiagnosticBase):
    """For each sample where the gate's chosen expert != oracle-chosen
    expert, measure:
    - Δ accuracy (oracle correct, gate wrong → 1.0 drop)
    - Δ confidence (oracle's max prob − gate-chosen's max prob)
    - Δ true-class probability
    Aggregated by head/mid/tail and severity."""

    name = "Misrouting Penalty Breakdown"
    depends_on = ["p_ce", "p_la", "p_bs", "w", "labels", "group_ids"]

    def run(self) -> DiagnosticResult:
        d = self.data
        N = len(d.labels)

        # Per-expert predictions
        expert_preds = np.stack([
            np.argmax(d.p_ce, axis=1),
            np.argmax(d.p_la, axis=1),
            np.argmax(d.p_bs, axis=1),
        ], axis=1)  # (N, 3)
        expert_correct = (expert_preds == d.labels.reshape(-1, 1))  # (N, 3)

        # True-class probability per expert
        true_probs = np.stack([
            d.p_ce[np.arange(N), d.labels],
            d.p_la[np.arange(N), d.labels],
            d.p_bs[np.arange(N), d.labels],
        ], axis=1)  # (N, 3)

        # Max confidence per expert
        max_confs = np.stack([
            d.p_ce.max(axis=1),
            d.p_la.max(axis=1),
            d.p_bs.max(axis=1),
        ], axis=1)  # (N, 3)

        # Oracle = expert with highest true-class prob
        oracle_expert = np.argmax(true_probs, axis=1)
        # Gate's chosen expert
        gate_expert = np.argmax(d.w, axis=1)

        misrouted = (gate_expert != oracle_expert)
        n_misrouted = int(misrouted.sum())

        if n_misrouted == 0:
            return DiagnosticResult(
                title="Misrouting Penalty Breakdown",
                summary="Gate always picks the oracle expert. No misrouting.",
                metrics={"n_misrouted": 0, "misrouting_rate": 0.0},
                tables=[],
                verdict="PASS",
                recommendation=None,
            )

        # Penalty metrics
        oracle_correct = expert_correct[np.arange(N), oracle_expert]
        gate_correct = expert_correct[np.arange(N), gate_expert]
        oracle_conf = max_confs[np.arange(N), oracle_expert]
        gate_conf = max_confs[np.arange(N), gate_expert]
        oracle_true = true_probs[np.arange(N), oracle_expert]
        gate_true = true_probs[np.arange(N), gate_expert]

        # On misrouted samples
        om = misrouted
        acc_drop = 1.0 * (oracle_correct[om] & ~gate_correct[om])
        conf_drop = oracle_conf[om] - gate_conf[om]
        true_drop = oracle_true[om] - gate_true[om]

        catastrophic = int(acc_drop.sum())
        safe_misroute = n_misrouted - catastrophic
        avg_conf_drop = float(conf_drop.mean())
        avg_true_drop = float(true_drop.mean())

        # Per-group breakdown
        group_map = {0: "Head", 1: "Mid", 2: "Tail"}
        group_breakdown = {}
        for gid, gname in group_map.items():
            mask = om & (d.group_ids == gid)
            if mask.sum() == 0:
                continue
            group_breakdown[gname] = {
                "n": int(mask.sum()),
                "catastrophic": int((acc_drop[mask]).sum()),
                "avg_conf_drop": float(conf_drop[mask].mean()),
            }

        group_rows = [
            (gname, str(v["n"]), str(v["catastrophic"]),
             f"{v['avg_conf_drop']:.4f}")
            for gname, v in group_breakdown.items()
        ]

        misrouting_rate = n_misrouted / N * 100

        return DiagnosticResult(
            title="Misrouting Penalty Breakdown",
            summary=(f"{n_misrouted}/{N} samples misrouted "
                     f"({misrouting_rate:.1f}%). "
                     f"Of those, {catastrophic} are catastrophic "
                     f"(gate wrong while oracle correct). "
                     f"Avg confidence drop = {avg_conf_drop:.4f}."),
            metrics={
                "n_misrouted": n_misrouted,
                "misrouting_rate": misrouting_rate,
                "catastrophic_misroutes": catastrophic,
                "safe_misroutes": safe_misroute,
                "avg_conf_drop": avg_conf_drop,
                "avg_true_prob_drop": avg_true_drop,
            },
            tables=[
                {"headers": ["Group", "Misrouted", "Catastrophic",
                             "Avg Conf Drop"],
                 "rows": group_rows},
            ],
            verdict=("FAIL" if catastrophic / max(n_misrouted, 1) > 0.3
                     else "WARN" if catastrophic > 0 else "PASS"),
            recommendation=(
                f"{catastrophic}/{n_misrouted} misroutes are catastrophic "
                f"({catastrophic / max(n_misrouted, 1) * 100:.1f}%). "
                "High catastrophic rate → gate makes expensive mistakes."),
        )
