#!/usr/bin/env python3
"""
CLI wrapper for running the packaged null baseline shell pipeline.

This command delegates to `run_null_baseline_analysis.sh` and keeps the
existing environment-variable interface unchanged.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the wrapper."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the full null baseline attribution pipeline "
            "(create NULL data, train, explain, compare)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--print-script-path",
        action="store_true",
        help="Print the path to run_null_baseline_analysis.sh and exit.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point used by `sieve-run-null-baseline`."""
    args = parse_args()
    script_path = Path(__file__).with_name("run_null_baseline_analysis.sh")

    if args.print_script_path:
        print(script_path)
        return

    if not script_path.exists():
        print(
            f"ERROR: packaged script not found at {script_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    completed = subprocess.run(["bash", str(script_path)], check=False)
    sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
