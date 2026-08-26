"""ReportGenerator — hierarchical output formatting for the diagnostics
framework.  Takes a list of DiagnosticResult objects and produces the
final structured report."""
# This module outputs raw diagnostic results only. No synthesis.
# Verification and interpretation are the caller's responsibility.

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
        """Render the complete report with raw results only."""
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
                for table in r.tables:
                    self._append_table(lines, table.get("headers", []),
                                       table.get("rows", []))

        return "\n".join(lines)

    def _append_table(self, lines, headers, rows):
        if not headers or not rows:
            return
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row[:len(headers)]):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        sep = "  ".join("-" * w for w in col_widths)
        header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        row_lines = []
        for row in rows:
            padded = [str(c).ljust(w) for c, w in zip(row, col_widths)]
            row_lines.append("  ".join(padded))
        lines.append(f"\n{header_line}\n{sep}")
        lines.extend(row_lines)

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
