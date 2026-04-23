#!/usr/bin/env python3
"""
Run multiple experiments with different selection_ratio values for a single YAML config.
Works on both local machine and Kaggle.
Prints training output in real time.
Usage: python run_ratio_sweep.py
"""

import os
import sys
import yaml
import subprocess
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
# Only this specific config file will be used
CONFIG_FILE = os.path.join(PROJECT_ROOT, "config1", "cifar10", "model_2.yaml")
TEMP_CONFIG_DIR = os.path.join(PROJECT_ROOT, "temp_ratio_configs")
ERROR_LOG = os.path.join(PROJECT_ROOT, "ratio_sweep_errors.log")
RATIOS = [0.9, 0.5, 0.3, 0.1]

# ========== HELPER FUNCTIONS ==========
def setup_directories():
    os.makedirs(TEMP_CONFIG_DIR, exist_ok=True)

def log_error(message):
    with open(ERROR_LOG, "a") as f:
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
    temp_filename = f"{base_name}_ratio{ratio}.yaml"
    temp_path = os.path.join(temp_dir, temp_filename)

    with open(temp_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return temp_path

def main():
    setup_directories()
    if os.path.exists(ERROR_LOG):
        os.remove(ERROR_LOG)

    # Check if the specific config file exists
    if not os.path.isfile(CONFIG_FILE):
        print(f"Config file not found: {CONFIG_FILE}")
        sys.exit(1)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Config file: {CONFIG_FILE}")
    print(f"Will run for ratios: {RATIOS}")
    print(f"Temporary configs stored in: {TEMP_CONFIG_DIR}")
    print(f"Errors logged to: {ERROR_LOG}")
    print("=" * 80)

    total_experiments = len(RATIOS)
    experiment_counter = 0

    config_name = os.path.basename(CONFIG_FILE)
    print(f"\n>>> Processing config: {config_name}")

    for ratio in RATIOS:
        experiment_counter += 1
        print(f"\n--- [{experiment_counter}/{total_experiments}] Running ratio={ratio} ---")

        try:
            temp_config = modify_config_for_ratio(CONFIG_FILE, ratio, TEMP_CONFIG_DIR)
            cmd = f"python main.py --config {temp_config}"
            print(f"Command: {cmd}\n")

            returncode = run_command_stream_output(cmd, PROJECT_ROOT)

            if returncode != 0:
                error_msg = (f"ERROR in {config_name} with ratio={ratio}\n"
                             f"Return code: {returncode}\n"
                             f"{'-'*60}")
                log_error(error_msg)
                print(f"Failed with return code {returncode}")
            else:
                print(f"Success")

            # Optional: remove temp config file (uncomment if desired)
            # os.remove(temp_config)

        except Exception as e:
            error_msg = (f"EXCEPTION in {config_name} with ratio={ratio}\n"
                         f"Exception: {str(e)}\n"
                         f"{'-'*60}")
            log_error(error_msg)
            print(f"Exception: {e}")

    print("\n" + "=" * 80)
    print("All experiments finished.")
    print(f"Check {ERROR_LOG} for any errors.")

if __name__ == "__main__":
    main()