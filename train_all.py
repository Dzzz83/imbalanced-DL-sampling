#!/usr/bin/env python3
"""
Run train.py on every YAML config file in a specified directory.
If a config fails, log the error and continue with the next file.
Usage: python train_all.py --config_dir /path/to/configs [--ratios 0.9 0.7 0.5 ...]
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime
from pathlib import Path

# ========== ENVIRONMENT DETECTION ==========
def get_project_root():
    """Determine project root based on environment (local or Kaggle)."""
    if os.path.exists('/kaggle/working'):
        kaggle_project = '/kaggle/working/imbalanced-DL-sampling'
        if os.path.exists(kaggle_project):
            return kaggle_project
        else:
            return os.getcwd()
    else:
        return os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = get_project_root()

# ========== CONFIGURATION ==========
DEFAULT_CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
DEFAULT_RATIOS = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1]

def log_error(error_log_path, config_name, error_message):
    """Append error details to the log file with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(error_log_path, 'a') as f:
        f.write(f"[{timestamp}] CONFIG: {config_name}\n")
        f.write(f"ERROR: {error_message}\n")
        f.write("-" * 80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Run ratio sweep on all YAML configs in a directory.")
    parser.add_argument('--config_dir', type=str, default=DEFAULT_CONFIG_DIR,
                        help='Directory containing YAML config files')
    parser.add_argument('--ratios', type=float, nargs='+', default=DEFAULT_RATIOS,
                        help='List of selection ratios to sweep (passed to train.py)')
    parser.add_argument('--train_script', type=str, default='train.py',
                        help='Path to train.py script')
    parser.add_argument('--error_log', type=str, default='train_all_errors.log',
                        help='Log file for errors (relative to project root)')
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    if not config_dir.is_absolute():
        config_dir = PROJECT_ROOT / config_dir

    if not config_dir.exists():
        print(f"Config directory not found: {config_dir}")
        sys.exit(1)

    # Find all YAML files in the directory (recursively)
    yaml_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))
    yaml_files = sorted(yaml_files)  # deterministic order

    if not yaml_files:
        print(f"No YAML files found in {config_dir}")
        sys.exit(0)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Config directory: {config_dir}")
    print(f"Found {len(yaml_files)} config files.")
    print(f"Ratios to sweep: {args.ratios}")
    print(f"Error log: {args.error_log}")
    print("=" * 80)

    error_log_path = Path(args.error_log)
    if not error_log_path.is_absolute():
        error_log_path = PROJECT_ROOT / error_log_path

    # Clear previous error log (start fresh)
    with open(error_log_path, 'w') as f:
        f.write(f"TrainAll error log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")

    # Path to train.py
    train_script = Path(args.train_script)
    if not train_script.is_absolute():
        train_script = PROJECT_ROOT / train_script

    if not train_script.exists():
        print(f"train.py not found at {train_script}")
        sys.exit(1)

    total_configs = len(yaml_files)
    failed_configs = []

    for idx, config_file in enumerate(yaml_files, 1):
        print(f"\n>>> Processing config [{idx}/{total_configs}]: {config_file.name}")
        print(f"    Full path: {config_file}")

        # Build command: python train.py --config <file> --ratios <ratios>
        cmd = [
            sys.executable, str(train_script),
            '--config', str(config_file),
            '--ratios', *map(str, args.ratios)
        ]

        print(f"    Command: {' '.join(cmd)}\n")
        try:
            # Run train.py and stream output
            process = subprocess.Popen(cmd, cwd=PROJECT_ROOT,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       universal_newlines=True, bufsize=1)
            for line in process.stdout:
                print(line, end='')
            returncode = process.wait()

            if returncode != 0:
                error_msg = f"train.py returned non-zero exit code {returncode}"
                log_error(error_log_path, config_file.name, error_msg)
                failed_configs.append(config_file.name)
                print(f"\n[ERROR] {config_file.name} failed. Skipping to next config.")
            else:
                print(f"\n[SUCCESS] {config_file.name} completed.")

        except Exception as e:
            error_msg = f"Exception: {str(e)}"
            log_error(error_log_path, config_file.name, error_msg)
            failed_configs.append(config_file.name)
            print(f"\n[EXCEPTION] {config_file.name} failed: {e}. Skipping to next config.")

    # Final summary
    print("\n" + "=" * 80)
    print(f"Completed {total_configs} configs.")
    print(f"Successful: {total_configs - len(failed_configs)}")
    print(f"Failed: {len(failed_configs)}")
    if failed_configs:
        print("Failed configs:")
        for name in failed_configs:
            print(f"  - {name}")
    print(f"Errors logged to {error_log_path}")

if __name__ == "__main__":
    main()