#!/usr/bin/env python3
"""
Run multiple experiments with different selection_ratio values for each YAML config.
Usage: python run_ratio_sweep.py
"""

import os
import sys
import yaml
import subprocess
import shutil
from pathlib import Path

# ========== CONFIGURATION ==========
CONFIG_DIR = "/home/phatht/phat/imbalanced-DL-sampling/config1/cifar10"
PROJECT_ROOT = "/home/phatht/phat/imbalanced-DL-sampling"   # where main.py lives
TEMP_CONFIG_DIR = os.path.join(PROJECT_ROOT, "temp_ratio_configs")
ERROR_LOG = os.path.join(PROJECT_ROOT, "ratio_sweep_errors.log")
RATIOS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]   # you can modify this list

# ========== HELPER FUNCTIONS ==========
def setup_directories():
    """Create temporary config directory if it doesn't exist."""
    os.makedirs(TEMP_CONFIG_DIR, exist_ok=True)

def log_error(message):
    """Append error message to the log file."""
    with open(ERROR_LOG, "a") as f:
        f.write(message + "\n")
    print(message)

def run_command(cmd, cwd):
    """Run a shell command and return (returncode, stdout, stderr)."""
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr

def modify_config_for_ratio(original_config_path, ratio, temp_dir):
    """
    Load the original YAML, modify selection_ratio and store_name,
    write a temporary config file, and return its path.
    """
    with open(original_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Modify selection_ratio
    config['selection_ratio'] = ratio

    # Modify store_name to include the ratio (avoid overwriting)
    original_store_name = config.get('store_name', 'model')
    config['store_name'] = f"{original_store_name}_ratio{ratio}"

    # Optionally adjust root_log or root_model to separate runs (optional)
    # config['root_log'] = f"experiments/train/ratio_{ratio}"
    # config['root_model'] = f"./checkpoints/ratio_{ratio}"

    # Create a unique filename for this ratio
    base_name = os.path.basename(original_config_path).replace('.yaml', '')
    temp_filename = f"{base_name}_ratio{ratio}.yaml"
    temp_path = os.path.join(temp_dir, temp_filename)

    with open(temp_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return temp_path

def main():
    setup_directories()
    # Clear old error log
    if os.path.exists(ERROR_LOG):
        os.remove(ERROR_LOG)

    # Find all YAML files in CONFIG_DIR
    yaml_files = list(Path(CONFIG_DIR).glob("*.yaml"))
    if not yaml_files:
        print(f"No YAML files found in {CONFIG_DIR}")
        return

    print(f"Found {len(yaml_files)} config files.")
    print(f"Will run for ratios: {RATIOS}")
    print(f"Temporary configs will be stored in: {TEMP_CONFIG_DIR}")
    print(f"Errors logged to: {ERROR_LOG}")
    print("=" * 80)

    total_experiments = len(yaml_files) * len(RATIOS)
    experiment_counter = 0

    for config_file in yaml_files:
        config_name = config_file.stem
        print(f"\n>>> Processing config: {config_file.name}")

        for ratio in RATIOS:
            experiment_counter += 1
            print(f"  [{experiment_counter}/{total_experiments}] Running ratio={ratio} ...")

            try:
                # Create temporary config with this ratio
                temp_config = modify_config_for_ratio(config_file, ratio, TEMP_CONFIG_DIR)

                # Build the command
                cmd = f"python main.py --config {temp_config}"
                print(f"    Command: {cmd}")

                # Run the command from the project root
                returncode, stdout, stderr = run_command(cmd, PROJECT_ROOT)

                if returncode != 0:
                    error_msg = (f"ERROR in {config_file.name} with ratio={ratio}\n"
                                 f"Return code: {returncode}\n"
                                 f"STDERR:\n{stderr}\n"
                                 f"STDOUT (last 500 chars):\n{stdout[-500:]}\n"
                                 f"{'-'*60}")
                    log_error(error_msg)
                    print(f"    ❌ Failed (see log for details)")
                else:
                    print(f"    ✅ Success")

                # Clean up the temporary config file (optional)
                # os.remove(temp_config)

            except Exception as e:
                error_msg = (f"EXCEPTION in {config_file.name} with ratio={ratio}\n"
                             f"Exception: {str(e)}\n"
                             f"{'-'*60}")
                log_error(error_msg)
                print(f"    ❌ Exception: {e}")

    print("\n" + "=" * 80)
    print("All experiments finished.")
    print(f"Check {ERROR_LOG} for any errors.")

if __name__ == "__main__":
    main()