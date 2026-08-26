"""Stage3PluginAnalyzer — selective prediction plugin parameters
(alpha, mu) for Stage 3 evaluation."""

from .base import DiagnosticBase, DiagnosticResult
from imbalanceddl.utils.plugin_rule import tune_plugin_for_rho


class Stage3PluginAnalyzer(DiagnosticBase):
    """Compute Stage 3 plug-in parameters (alpha, mu) at 50% rejection.
    These reveal whether selective prediction could salvage tail accuracy."""

    name = "Stage 3 Plugin Parameters"
    depends_on = ["p_mix_tune", "labels_tune", "cls_group_ids",
                  "cls_num_list", "cfg"]

    def run(self) -> DiagnosticResult:
        d = self.data

        alpha_bal, mu_bal = tune_plugin_for_rho(
            d.p_mix_tune, d.labels_tune, d.cls_group_ids,
            rho=0.5, mode="bal", cls_num_list=d.cls_num_list
        )
        alpha_wst, mu_wst = tune_plugin_for_rho(
            d.p_mix_tune, d.labels_tune, d.cls_group_ids,
            rho=0.5, mode="worst", cls_num_list=d.cls_num_list
        )

        rows = [
            ("Balanced", f"{alpha_bal[0]:.4f}", f"{alpha_bal[1]:.4f}",
             f"{mu_bal[0]:.4f}", f"{mu_bal[1]:.4f}"),
            ("Worst-case", f"{alpha_wst[0]:.4f}", f"{alpha_wst[1]:.4f}",
             f"{mu_wst[0]:.4f}", f"{mu_wst[1]:.4f}"),
        ]

        tail_alpha_near_zero = alpha_bal[1] < 0.15 or alpha_wst[1] < 0.15

        return DiagnosticResult(
            title="Stage 3 Plugin Parameters",
            summary=("At 50% rejection rate." +
                     (" Tail alpha near 0 → rejector kills tail group."
                      if tail_alpha_near_zero else "")),
            metrics={
                "alpha_bal_head": float(alpha_bal[0]),
                "alpha_bal_tail": float(alpha_bal[1]),
                "alpha_wst_head": float(alpha_wst[0]),
                "alpha_wst_tail": float(alpha_wst[1]),
            },
            tables=[{"headers": ["Mode", "α Head", "α Tail", "μ Head",
                                  "μ Tail"],
                     "rows": rows}],
            verdict=None,
            recommendation=None,
        )
