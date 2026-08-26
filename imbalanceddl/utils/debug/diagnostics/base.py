"""Base classes for all routing diagnostics."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DiagnosticResult:
    """Structured output from one diagnostic run.

    The ``.metrics`` dict collects all scalar findings so that
    the ``ReportGenerator`` can cross-reference them in the final
    synthesis section.
    """

    title: str
    """Short section title, e.g. "Oracle Gap & Headroom Analysis"."""

    summary: str
    """Single-sentence headline capturing the essential finding."""

    metrics: dict[str, Any] = field(default_factory=dict)
    """All scalar/array results keyed by name for cross-referencing."""

    tables: list[dict] = field(default_factory=list)
    """Each table dict: ``{'headers': [...], 'rows': [[...], ...]}``."""

    verdict: Optional[str] = None
    """"PASS" / "WARN" / "FAIL" or None."""

    recommendation: Optional[str] = None
    """Actionable next step based on this diagnostic alone."""

    section_id: str = ""
    """Hierarchical id, set by the ReportGenerator, e.g. "3.2"."""


class DiagnosticBase(ABC):
    """Every diagnostic component inherits from this.

    Subclasses override :meth:`run` to perform pure analysis and
    return a :class:`DiagnosticResult`.  They must **never** print
    directly — formatting is handled by the caller (usually
    ``ReportGenerator`` or the diagnostic's own ``report()`` method).
    """

    name: str = ""
    """Human-readable name, used as the section heading."""

    depends_on: list[str] = []
    """Keys in ``DiagnosticData`` that this diagnostic requires."""

    def __init__(self, data: Any) -> None:
        self.data = data

    @abstractmethod
    def run(self) -> DiagnosticResult:
        """Execute the diagnostic and return structured results.

        This method must be pure — no side effects, no printing.
        """

    def report(self, result: DiagnosticResult) -> str:
        """Default formatter.

        Override in subclasses that need custom narrative output.
        """
        lines = [f"\n{'=' * 80}", f"{result.title}", f"{'=' * 80}"]
        if result.summary:
            lines.append(f"[INFO] {result.summary}")
        for table in result.tables:
            lines.append(_format_table(table.get("headers", []),
                                       table.get("rows", [])))
        if result.verdict:
            lines.append(f"  Verdict: {result.verdict}")
        if result.recommendation:
            lines.append(f"  → {result.recommendation}")
        lines.append("=" * 80)
        return "\n".join(lines)


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Simple column-aligned table formatter."""
    if not headers or not rows:
        return ""
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
    return f"\n{header_line}\n{sep}\n" + "\n".join(row_lines)
