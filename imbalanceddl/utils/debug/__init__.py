"""Debug utilities for gate routing diagnostics.

This package provides a modular, OOP-based diagnostic framework for
analysing why a learned gate router fails to outperform uniform
averaging in long-tailed classification.

Key components:
- ``runner.PipelineOrchestrator`` — one-call entry point
- ``extraction.DataExtractor`` — populates the ``DiagnosticData`` container
- ``diagnostics.*`` — individual diagnostic classes
- ``reporting.ReportGenerator`` — hierarchical report assembly
"""

from .models import ExpertEnsemble, GateMLP
from .extraction import DataExtractor, extract_data, recipe_from_checkpoint, DiagnosticData
from .reporting import ReportGenerator

# Re-export diagnostics base
from .diagnostics.base import DiagnosticBase, DiagnosticResult
