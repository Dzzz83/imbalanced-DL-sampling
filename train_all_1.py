import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_experiment(config_path):
    cmd = [
        sys.executable,
        "main.py",
        "--config",
        str(config_path)
    ]

    print("\n" + "=" * 80)
    print(f"Running: {config_path.name}")
    print("=" * 80)

    start_time = time.time()

    result = subprocess.run(cmd)

    elapsed = time.time() - start_time

    if result.returncode == 0:
        print(f"✓ Finished in {elapsed/60:.2f} minutes")
        return True
    else:
        print(f"✗ Failed (return code {result.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run all configs in a directory"
    )

    parser.add_argument(
        "--config_dir",
        type=str,
        required=True,
        help="Directory containing config files"
    )

    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop immediately if one experiment fails"
    )

    args = parser.parse_args()

    config_dir = Path(args.config_dir)

    if not config_dir.exists():
        print(f"Directory does not exist: {config_dir}")
        sys.exit(1)

    config_files = []

    for ext in ("*.yaml", "*.yml", "*.json"):
        config_files.extend(config_dir.glob(ext))

    config_files = sorted(config_files)

    if not config_files:
        print(f"No config files found in {config_dir}")
        sys.exit(1)

    print(f"Found {len(config_files)} config files")

    total_start = time.time()

    success = 0
    failed = 0

    for config_path in config_files:
        ok = run_experiment(config_path)

        if ok:
            success += 1
        else:
            failed += 1

            if args.stop_on_error:
                print("\nStopping because a run failed.")
                break

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Success: {success}")
    print(f"Failed : {failed}")
    print(f"Total Time: {total_elapsed/3600:.2f} hours")


if __name__ == "__main__":
    main()