"""ReportGenerator — hierarchical output formatting for the diagnostics
framework.  Takes a list of DiagnosticResult objects and produces the
final structured report."""

from typing import Optional
from .diagnostics.base import DiagnosticResult


class ReportGenerator:
    """Assembles all DiagnosticResult objects into a hierarchical report."""

    def __init__(self, results: list[DiagnosticResult],
                 uniform_bal: Optional[float] = None,
                 gate_bal: Optional[float] = None):
        self.results = results
        self.uniform_bal = uniform_bal
        self.gate_bal = gate_bal

    def generate(self) -> str:
        """Render the complete report."""
        lines = []
        lines.append("=" * 80)
        lines.append("DIAGNOSTIC REPORT: GATE ROUTING FAILURE ANALYSIS")
        lines.append("=" * 80)

        section_map = self._assign_section_ids()

        for section_id, title, result_list in section_map:
            lines.append("")
            lines.append(f"--- {section_id}: {title} ---")
            for r in result_list:
                lines.append(r.summary)

        # ── Synthesis ──
        lines.append("")
        lines.append("=" * 80)
        lines.append("SYNTHESIS & RECOMMENDATION")
        lines.append("=" * 80)
        lines.extend(self._synthesize())
        lines.append("=" * 80)

        return "\n".join(lines)

    def _assign_section_ids(self) -> list:
        """Group results into sections and assign hierarchical ids."""
        sections = [
            ("2", "Expert Performance Summary", []),
            ("3", "Gate Input Quality", []),
            ("4", "Gate Decision Analysis", []),
            ("5", "Oracle Gap & Misrouting Penalty", []),
            ("6", "Sensitivity & Ablation", []),
            ("7", "Supplementary", []),
        ]

        for result in self.results:
            title = result.title
            if "Expert Performance" in title:
                sections[0][2].append(result)
            elif "Logit Scale" in title or "Feature Correlation" in title \
                    or "Gate Weight" in title or "Gradient" in title \
                    or "Feature Ablation" in title:
                sections[1][2].append(result)
            elif "Routing Pattern" in title or "Expert Agreement" in title \
                    or "Saves the Day" in title:
                sections[2][2].append(result)
            elif "Oracle" in title or "Misrouting" in title:
                sections[3][2].append(result)
            elif "Temperature" in title or "Difficulty" in title \
                    or "Confidently" in title or "Penultimate" in title:
                sections[4][2].append(result)
            else:
                sections[5][2].append(result)

        result = []
        for i, (sid, stitle, rlist) in enumerate(sections):
            if rlist:
                for j, r in enumerate(rlist):
                    r.section_id = f"{sid}.{j + 1}"
                result.append((sid, stitle, rlist))
        return result

    def _synthesize(self) -> list[str]:
        """Cross-reference results and produce ranked root-cause list."""
        lines = []

        # Collect key metrics
        metrics = {}
        for r in self.results:
            metrics.update(r.metrics)

        lines.append("")
        lines.append("ROOT CAUSE HIERARCHY (empirically ranked):")
        lines.append("")

        findings = []

        # 1. Feature correlation
        emb_corr = metrics.get("embedding_mean_corr", None)
        prob_corr = metrics.get("prob_mean_corr", None)
        if prob_corr is not None and prob_corr > 0.5:
            findings.append((
                "PRIMARY",
                "Probability features are collinear "
                f"(r = {prob_corr:.2f})",
                "Gate cannot distinguish experts in probability space. "
                "See Feature Correlation Analysis.",
                "CRITICAL"
            ))

        # 2. Weight collapse
        w_entropy = metrics.get("weight_entropy", None)
        if w_entropy is not None and w_entropy > 1.5:
            findings.append((
                "PRIMARY",
                f"Gate weight entropy = {w_entropy:.3f} "
                f"(near max {1.099:.3f})",
                "Gate weights are near-uniform; gate is not routing.",
                "CRITICAL"
            ))

        # 3. Confidently wrong on tail
        cw_tail = metrics.get("conf_wrong_tail", None)
        if cw_tail is not None and cw_tail > 0.5:
            findings.append((
                "SECONDARY",
                f"Tail confidently-wrong rate = {cw_tail:.1%}",
                "Confidence routing catastrophically fails on tail. "
                "See Confidently-Wrong Analysis.",
                "CRITICAL"
            ))

        # 4. Oracle gap
        gap = metrics.get("gap_bal_acc", None)
        headroom = metrics.get("headroom_ratio", None)
        if gap is not None and gap > 2.0:
            findings.append((
                "TERTIARY",
                f"Oracle gap = {gap:.1f} pp "
                f"(headroom = {headroom:.1f}%)",
                "Even perfect routing leaves significant error. "
                "Limited routing headroom.",
                "SIGNIFICANT"
            ))

        # 5. Misrouting
        catastrophic = metrics.get("catastrophic_misroutes", None)
        misrouted = metrics.get("n_misrouted", None)
        if catastrophic is not None and misrouted and misrouted > 0:
            cat_rate = catastrophic / misrouted * 100
            if cat_rate > 20:
                findings.append((
                    "SECONDARY",
                    f"{catastrophic}/{misrouted} misroutes are "
                    f"catastrophic ({cat_rate:.0f}%)",
                    "Gate makes expensive mistakes when it routes. "
                    "See Misrouting Penalty.",
                    "SIGNIFICANT"
                ))

        # 6. Gradient sensitivity (blindness)
        grad_sens = {k: v for k, v in metrics.items()
                     if k.startswith("grad_sensitivity_")}
        if grad_sens:
            min_sens = min(grad_sens.values())
            max_sens = max(grad_sens.values())
            if max_sens > 0 and min_sens / max_sens < 0.1:
                blind = [k.replace("grad_sensitivity_", "")
                         for k, v in grad_sens.items()
                         if v < 0.1 * max_sens]
                findings.append((
                    "PRIMARY",
                    f"Gate is BLIND to {blind} experts "
                    f"(gradient sensitivity near-zero)",
                    "Gate cannot utilize some experts' features.",
                    "CRITICAL"
                ))

        # 7. Ablation result
        pen_sim_bal = metrics.get("penultimate_sim_bal_acc", None)
        baseline_bal = metrics.get("gate_bal_acc",
                                   metrics.get("uniform_bal_acc", None))
        if pen_sim_bal is not None and baseline_bal is not None:
            if pen_sim_bal > baseline_bal + 0.5:
                findings.append((
                    "INSIGHT",
                    f"Penultimate routing sim = {pen_sim_bal:.2f}% "
                    f"(+{pen_sim_bal - baseline_bal:.1f} pp vs baseline)",
                    "Penultimate features carry useful signal. "
                    "Worth implementing Linear(192,3).",
                    "POSITIVE"
                ))

        # Sort by severity
        severity_order = {"CRITICAL": 0, "SIGNIFICANT": 1, "POSITIVE": 2}
        findings.sort(key=lambda x: severity_order.get(x[3], 99))

        for severity, title, detail, _ in findings:
            icon = {"CRITICAL": "🔴", "SIGNIFICANT": "🟡", "POSITIVE": "🟢"}
            lines.append(
                f"  {icon.get(severity, '⚪')} [{severity}] {title}")
            lines.append(f"     {detail}")
            lines.append("")

        # Recommendation
        lines.append("RECOMMENDED NEXT STEPS:")
        lines.append("")
        if any("CRITICAL" in f[3] for f in findings):
            lines.append("  [1] 🎯 Pivot to Linear(192,3) penultimate router")
            lines.append("  [2] 🎯 Train with mixture-BCE loss "
                         "(not soft-oracle KL)")
            lines.append("  [3] 🎯 Add learnable gate temperature")
            lines.append("  [4] 📊 Fallback: uniform ensemble + "
                         "Stage 3 selective prediction")
        else:
            lines.append("  [1] ✅ Continue with current architecture, "
                         "focus on hyperparameter tuning")
            lines.append("  [2] ✅ Fine-tune gate temperature on validation set")

        return lines
