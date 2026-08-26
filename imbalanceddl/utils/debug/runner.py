"""PipelineOrchestrator — loads models, extracts data, runs all
diagnostics, and generates the final report."""

from __future__ import annotations

import os
import sys
import re
import argparse
from typing import Any, Optional

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from imbalanceddl.utils.config import get_args
from imbalanceddl.dataset.imbalance_dataset import ImbalancedDataset
from imbalanceddl.utils.plugin_rule import define_groups_2
from imbalanceddl.utils.debug.models import ExpertEnsemble, GateMLP

from .extraction import (
    DataExtractor,
    recipe_from_checkpoint,
    DiagnosticData,
)
from .reporting import ReportGenerator


class DiagnosticRegistry:
    """Ordered list of diagnostic classes to run."""

    def __init__(self):
        self._diagnostics: list[type] = []

    def register(self, diag_cls: type) -> None:
        self._diagnostics.append(diag_cls)

    def build_all(self, data: DiagnosticData) -> list:
        return [cls(data) for cls in self._diagnostics]


class PipelineOrchestrator:
    """Main orchestrator: parse → load → extract → diagnose → report."""

    def __init__(self):
        # Parse custom args from sys.argv BEFORE get_args() sees them.
        # Must mirror the old ultra_debug.py behaviour: parse_known_args
        # strips --ce_path/--la_path/--bs_path/--gate_ckpt/--diagnose_*
        # and leaves the rest for get_args().
        self.custom_args, remaining_argv = self._parse_custom_args()
        sys.argv = [sys.argv[0]] + remaining_argv

    def _parse_custom_args(self):
        """Parse our custom arguments and return (args, remaining_argv).

        These are stripped from sys.argv so that ``get_args()`` does not
        choke on them.
        """
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--ce_path", type=str, required=True)
        parser.add_argument("--la_path", type=str, required=True)
        parser.add_argument("--bs_path", type=str, required=True)
        parser.add_argument("--gate_ckpt", type=str, required=True)
        parser.add_argument("--diagnose_confident_wrong", action="store_true",
                            help="Run only DaWin diagnostic and exit.")
        parser.add_argument("--diagnose_embeddings", action="store_true",
                            help="Run only embedding correlation diagnostic "
                                 "and exit.")
        return parser.parse_known_args()

    def run(self) -> list:
        """Execute the full diagnostic pipeline.

        Returns the list of DiagnosticResult objects for programmatic access.
        """
        # ── Phase 0: Config & Device ──
        cfg = get_args()
        if cfg.dataset == "cifar100":
            cfg.num_classes = 100

        device = torch.device("cuda:0" if torch.cuda.is_available()
                              else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)

        self._log("=" * 80)
        self._log("DIAGNOSTIC REPORT: GATE ROUTING FAILURE ANALYSIS")
        self._log("=" * 80)
        self._log("")

        # ── Phase 1: Data Loading ──
        self._log("--- Section 1: Data & Model Load ---")
        dataset = ImbalancedDataset(cfg, cfg.dataset, augmentation="none")
        train_dataset, val_dataset = dataset.train_val_sets

        train_targets = np.array(train_dataset.targets)
        cfg.cls_num_list = np.bincount(
            train_targets, minlength=cfg.num_classes
        ).tolist()
        cfg.train_dataset = train_dataset  # needed by compute_all_metrics → shot_acc

        val_targets = np.array(val_dataset.targets)
        val_indices = np.arange(len(val_targets))
        tune_idx, test_idx = train_test_split(
            val_indices, test_size=0.8, stratify=val_targets,
            random_state=cfg.seed,
        )
        tune_dataset = Subset(val_dataset, tune_idx)
        test_dataset = Subset(val_dataset, test_idx)

        tune_loader = DataLoader(
            tune_dataset, batch_size=128, shuffle=False, num_workers=4
        )
        test_loader = DataLoader(
            test_dataset, batch_size=128, shuffle=False, num_workers=4
        )

        self._log("  ✓ Dataset loaded")
        self._log(f"  ✓ Tune: {len(tune_dataset)} samples, "
                  f"Test: {len(test_dataset)} samples")

        # ── Phase 2: Load Models ──
        ckpt_paths = {
            "CE": self.custom_args.ce_path,
            "LA": self.custom_args.la_path,
            "BS": self.custom_args.bs_path,
        }

        gate_ckpt = torch.load(
            self.custom_args.gate_ckpt, map_location="cpu",
            weights_only=False,
        )

        la_tau = 1.5
        match_tau = re.search(
            r"t([\d\.]+)", os.path.basename(self.custom_args.la_path)
        )
        if match_tau:
            la_tau = float(match_tau.group(1))
        self._log(f"  ✓ LA Tau = {la_tau} (parsed from filename)")

        recipe = recipe_from_checkpoint(gate_ckpt, cfg, la_tau=la_tau)
        self._log(
            f"  ✓ Recipe: T={recipe['T']} | "
            f"expert_temps={recipe['expert_temps']} | "
            f"k={recipe['k']} | space={recipe['space']} | "
            f"gate_temp={recipe['gate_temp']:.3f} | "
            f"mix_temp={recipe['mix_temp']:.3f}"
        )

        model = ExpertEnsemble(
            cfg, device, ckpt_paths,
            expert_T=recipe["expert_temps"],
            normalize_blocks=recipe["norm_blocks"],
            freq_features=recipe["freq_features"],
            gate_input_mode=recipe["gate_input_mode"],
        ).to(device)

        gate = GateMLP(
            input_dim=recipe["input_dim"],
            num_experts=3,
            linear_router=recipe["linear_router"],
        ).to(device)

        self._log(f"  ✓ Loading Gate from {self.custom_args.gate_ckpt}")
        try:
            gate.load_state_dict(gate_ckpt["gate_state_dict"])
        except RuntimeError as e:
            self._log(
                f"  ✗ Gate architecture mismatch.\n"
                f"    Recipe: freq_features={recipe['freq_features']}, "
                f"linear_router={recipe['linear_router']}\n"
                f"    GateMLP input_dim={gate._input_dim}\n"
                f"    Checkpoint fc.weight shape: "
                f"{gate_ckpt['gate_state_dict']['fc.weight'].shape}\n"
                f"    Error: {e}"
            )
            sys.exit(1)
        gate.eval()
        self._log("  ✓ Models loaded successfully")

        # ── Phase 3: Extract Data ──
        self._log("")
        self._log("[INFO] Extracting posteriors and features...")
        extractor = DataExtractor(model, gate, device, recipe, cfg=cfg)
        data = extractor.extract_all(tune_loader, test_loader)
        self._log("  ✓ Data extraction complete")

        # ── Phase 4: Run Diagnostics ──
        registry = self._build_default_registry()
        diagnostics = registry.build_all(data)
        results = []

        for diag in diagnostics:
            self._log("")
            result = diag.run()
            results.append(result)
            # Print the diagnostic output
            print(diag.report(result))

        # ── Phase 5: Synthesis Report ──
        report_gen = ReportGenerator(results)
        print(report_gen.generate())

        return results

    def _build_default_registry(self) -> DiagnosticRegistry:
        """Ordered list of diagnostics to run."""
        from .diagnostics.expert_summary import ExpertPerformanceSummary
        from .diagnostics.logit_scale import LogitScaleAndActivationChecker
        from .diagnostics.feature_correlation import \
            FeatureCorrelationAnalyzer
        from .diagnostics.gate_weights import GateWeightAnalyzer
        from .diagnostics.routing_patterns import RoutingPatternAnalyzer
        from .diagnostics.expert_interaction import ExpertAgreementTracker
        from .diagnostics.saves_the_day import SavesTheDayAnalyzer
        from .diagnostics.oracle_gap import OracleGapAnalyzer
        from .diagnostics.misrouting_penalty import MisroutingPenaltyAnalyzer
        from .diagnostics.confident_wrong import (
            ConfidentlyWrongAnalyzer,
            TemperatureSweeper,
        )
        from .diagnostics.gradient_sensitivity import \
            GradientSensitivityAnalyzer
        from .diagnostics.feature_ablation import FeatureAblationRunner
        from .diagnostics.difficulty_stratification import \
            DifficultyStratificationAnalyzer
        from .diagnostics.penultimate_sim import \
            PenultimateRoutingSimulator
        from .diagnostics.calibration import CalibrationAnalyzer
        from .diagnostics.plugin_params import Stage3PluginAnalyzer

        registry = DiagnosticRegistry()
        # Section 2
        registry.register(ExpertPerformanceSummary)
        # Section 3
        registry.register(LogitScaleAndActivationChecker)
        registry.register(FeatureCorrelationAnalyzer)
        registry.register(GateWeightAnalyzer)
        registry.register(GradientSensitivityAnalyzer)
        registry.register(FeatureAblationRunner)
        # Section 4
        registry.register(RoutingPatternAnalyzer)
        registry.register(ExpertAgreementTracker)
        registry.register(SavesTheDayAnalyzer)
        # Section 5
        registry.register(OracleGapAnalyzer)
        registry.register(MisroutingPenaltyAnalyzer)
        # Section 6
        registry.register(TemperatureSweeper)
        registry.register(DifficultyStratificationAnalyzer)
        registry.register(ConfidentlyWrongAnalyzer)
        registry.register(PenultimateRoutingSimulator)
        # Section 7
        registry.register(CalibrationAnalyzer)
        registry.register(Stage3PluginAnalyzer)

        return registry

    def _log(self, msg: str) -> None:
        """Print a log message (preserved from original behaviour)."""
        print(msg)
