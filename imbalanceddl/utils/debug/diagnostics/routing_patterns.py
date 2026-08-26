"""RoutingPatternAnalyzer — per-class weight analysis, group-level
gate-vs-uniform breakdown, and weight entropy per group."""

import numpy as np
from .base import DiagnosticBase, DiagnosticResult


class RoutingPatternAnalyzer(DiagnosticBase):
    """Analyse how gate weights vary across classes and groups,
    and measure the gate-vs-uniform accuracy trade-off per group."""

    name = "Routing Pattern Analysis"
    depends_on = ["w", "w_tune", "labels", "labels_tune", "p_mix",
                  "p_unif", "p_mix_tune", "p_unif_tune",
                  "cls_num_list", "group_ids", "cfg"]

    def run(self) -> DiagnosticResult:
        d = self.data
        cls_num_arr = np.array(d.cls_num_list)

        # ── 1. Average routing weights ──
        avg_w = np.mean(d.w, axis=0)
        avg_w_row = (f"CE={avg_w[0]:.4f}", f"LA={avg_w[1]:.4f}",
                     f"BS={avg_w[2]:.4f}")

        # ── 2. Per-class extreme routing (top-5 head, top-5 tail) ──
        sorted_classes = np.argsort(cls_num_arr)
        top5_head = sorted_classes[-5:]
        top5_tail = sorted_classes[:5]

        head_rows = []
        for c in top5_head:
            mask = d.labels == c
            if mask.sum() > 0:
                w_avg = np.mean(d.w[mask], axis=0)
                head_rows.append((str(c), str(cls_num_arr[c]),
                                  f"{w_avg[0]:.4f}", f"{w_avg[1]:.4f}",
                                  f"{w_avg[2]:.4f}"))

        tail_rows = []
        for c in top5_tail:
            mask = d.labels == c
            if mask.sum() > 0:
                w_avg = np.mean(d.w[mask], axis=0)
                tail_rows.append((str(c), str(cls_num_arr[c]),
                                  f"{w_avg[0]:.4f}", f"{w_avg[1]:.4f}",
                                  f"{w_avg[2]:.4f}"))

        # ── 3. Weight entropy per group ──
        w = np.clip(d.w, 1e-12, 1.0)
        entropy_all = -np.sum(w * np.log(w), axis=1)
        group_ids = d.group_ids

        entropy_rows = []
        for gid, gname in [(0, "Head"), (1, "Mid"), (2, "Tail")]:
            mask = group_ids == gid
            if mask.sum() > 0:
                e = entropy_all[mask].mean()
            else:
                e = 0.0
            entropy_rows.append((gname, f"{e:.4f}"))

        # ── 4. Gate vs uniform per-group accuracy breakdown ──
        def per_group_acc(preds, labels, gids):
            acc = {}
            for gid, gname in [(0, "Head"), (1, "Mid"), (2, "Tail")]:
                m = gids == gid
                if m.sum() > 0:
                    acc[gname] = (preds[m] == labels[m]).mean() * 100
            return acc

        gate_preds = np.argmax(d.p_mix, axis=1)
        unif_preds = np.argmax(d.p_unif, axis=1)
        gate_acc = per_group_acc(gate_preds, d.labels, group_ids)
        unif_acc = per_group_acc(unif_preds, d.labels, group_ids)

        group_acc_rows = []
        for gname in ["Head", "Mid", "Tail"]:
            if gname in gate_acc:
                delta = gate_acc[gname] - unif_acc[gname]
                group_acc_rows.append((
                    gname,
                    f"{unif_acc[gname]:.2f}%",
                    f"{gate_acc[gname]:.2f}%",
                    f"{delta:+.2f}%",
                ))

        return DiagnosticResult(
            title="Routing Pattern Analysis",
            summary=(f"Average weights: CE={avg_w[0]:.3f}, "
                     f"LA={avg_w[1]:.3f}, BS={avg_w[2]:.3f} "
                     f"(uniform baseline: 0.333 each). "
                     f"Weight entropy = {entropy_all.mean():.4f} "
                     f"(max = {(3).__float__().log():.4f})."),
            metrics={
                "avg_w_ce": float(avg_w[0]),
                "avg_w_la": float(avg_w[1]),
                "avg_w_bs": float(avg_w[2]),
                "weight_entropy": float(entropy_all.mean()),
            },
            tables=[
                {"headers": ["Group", "Gate Acc", "Uniform Acc", "Delta"],
                 "rows": group_acc_rows},
                {"headers": ["Group", "Entropy"], "rows": entropy_rows},
            ],
            verdict="FAIL" if abs(avg_w[0] - 1 / 3) < 0.02 else "PASS",
            recommendation=("Weights are near-uniform. "
                            "Gate is not learning meaningful routing."),
        )
