#!/usr/bin/env python3
"""
Run multiple experiments with different selection_ratio values for a single YAML config.
Works on both local machine and Kaggle.
Prints training output in real time.
Usage: python train.py --config /path/to/config.yaml [--ratios 0.9 0.7 0.5]
"""

import os
import sys
import yaml
import subprocess
import argparse
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

# ========== DEFAULT CONFIGURATION ==========
DEFAULT_CONFIG_FILE = os.path.join(PROJECT_ROOT, "config1", "cifar10_noisy", "model_1.yaml")
DEFAULT_RATIOS = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1]

# ========== HELPER FUNCTIONS ==========
def setup_directories(temp_dir):
    os.makedirs(temp_dir, exist_ok=True)

def log_error(error_log, message):
    with open(error_log, "a") as f:
        f.write(message + "\n")
    print(f"\n[ERROR LOGGED] {message.splitlines()[0]}...")

def run_command_stream_output(cmd, cwd):
    """
    Run a command and stream its stdout/stderr to the terminal in real time.
    Returns the return code.
    """
    process = subprocess.Popen(cmd, shell=True, cwd=cwd,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               universal_newlines=True, bufsize=1)
    for line in process.stdout:
        print(line, end='')
    process.wait()
    return process.returncode

def modify_config_for_ratio(original_config_path, ratio, temp_dir):
    with open(original_config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['selection_ratio'] = ratio
    original_store_name = config.get('store_name', 'model')
    config['store_name'] = f"{original_store_name}_ratio{ratio}"

    base_name = os.path.basename(original_config_path).replace('.yaml', '')
    # Add timestamp (and optional process ID) for uniqueness
    timestamp = int(time.time() * 1000)  # milliseconds
    pid = os.getpid()
    temp_filename = f"{base_name}_ratio{ratio}_{timestamp}_{pid}.yaml"
    temp_path = os.path.join(temp_dir, temp_filename)

    with open(temp_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return temp_path

def main():
    parser = argparse.ArgumentParser(description="Run ratio sweep for a YAML config.")
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--ratios', type=float, nargs='+', help='List of selection ratios to sweep')
    parser.add_argument('--temp_dir', type=str, default='temp_ratio_configs', help='Directory for temporary configs')
    parser.add_argument('--error_log', type=str, default='ratio_sweep_errors.log', help='Error log file path')
    args = parser.parse_args()

    # Determine config file
    if args.config:
        config_file = args.config
        if not os.path.isabs(config_file):
            config_file = os.path.join(PROJECT_ROOT, config_file)
    else:
        config_file = DEFAULT_CONFIG_FILE

    # Determine ratios
    ratios = args.ratios if args.ratios else DEFAULT_RATIOS

    temp_dir = os.path.join(PROJECT_ROOT, args.temp_dir)
    error_log = os.path.join(PROJECT_ROOT, args.error_log)

    setup_directories(temp_dir)
    if os.path.exists(error_log):
        os.remove(error_log)

    if not os.path.isfile(config_file):
        print(f"Config file not found: {config_file}")
        sys.exit(1)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Config file: {config_file}")
    print(f"Will run for ratios: {ratios}")
    print(f"Temporary configs stored in: {temp_dir}")
    print(f"Errors logged to: {error_log}")
    print("=" * 80)

    total_experiments = len(ratios)
    experiment_counter = 0
    config_name = os.path.basename(config_file)

    print(f"\n>>> Processing config: {config_name}")

    for ratio in ratios:
        experiment_counter += 1
        print(f"\n--- [{experiment_counter}/{total_experiments}] Running ratio={ratio} ---")

        try:
            temp_config = modify_config_for_ratio(config_file, ratio, temp_dir)
            cmd = f"python main.py --config {temp_config}"
            print(f"Command: {cmd}\n")

            returncode = run_command_stream_output(cmd, PROJECT_ROOT)

            if returncode != 0:
                error_msg = (f"ERROR in {config_name} with ratio={ratio}\n"
                             f"Return code: {returncode}\n"
                             f"{'-'*60}")
                log_error(error_log, error_msg)
                print(f"Failed with return code {returncode}")
            else:
                print(f"Success")

            # Optional: remove temp config file (uncomment if desired)
            # os.remove(temp_config)

        except Exception as e:
            error_msg = (f"EXCEPTION in {config_name} with ratio={ratio}\n"
                         f"Exception: {str(e)}\n"
                         f"{'-'*60}")
            log_error(error_log, error_msg)
            print(f"Exception: {e}")

    print("\n" + "=" * 80)
    print("All experiments finished.")
    print(f"Check {error_log} for any errors.")

if __name__ == "__main__":
    main()