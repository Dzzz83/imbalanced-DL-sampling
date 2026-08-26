#!/usr/bin/env python3
"""ultra_debug.py — Refactored entry point for gate-routing diagnostics.

This is now a thin wrapper around the modular diagnostic framework.
Run it to get a hierarchical report pinpointing why the gate router
fails to beat uniform averaging.

Usage:
    python ultra_debug.py \
        --ce_path checkpoint/experts_sweep_cifar100_calib/expert_CE_*.pth \
        --la_path checkpoint/experts_sweep_cifar100_calib/expert_LA_*.pth \
        --bs_path checkpoint/experts_sweep_cifar100_calib/expert_BS_*.pth \
        --gate_ckpt checkpoint/gate_cifar100_checkpoint/gate_checkpoint_*.pth \
        -c config/what_to_train/cifar100/_gate_train_penultimate.yaml
"""

import sys
import os

# Ensure the project root is on the path for Kaggle compatibility
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from imbalanceddl.utils.debug.runner import PipelineOrchestrator


def main():
    runner = PipelineOrchestrator()
    results = runner.run()
    # results is a list of DiagnosticResult objects for programmatic access
    return results


if __name__ == "__main__":
    main()
