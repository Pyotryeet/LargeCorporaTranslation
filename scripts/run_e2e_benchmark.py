#!/usr/bin/env python3
"""E2E benchmark runner — thin wrapper around the main benchmark harness.

This replaces the old tests/test_e2e.py which was a 368-line script (not a
test) that downloaded multi-GB models at module import time.

Usage:
    python scripts/run_e2e_benchmark.py [--model MODEL_ID] [--duration SECONDS]

The heavy lifting is delegated to scripts/run_one_model.py for single-model
runs and scripts/benchmark_all_models.py for multi-model comparison runs.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="E2E benchmark — run a model through translation + quality",
    )
    parser.add_argument(
        "--model",
        default="google/translategemma-4b-it",
        help="Model ID to benchmark (default: %(default)s)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=120,
        help="Benchmark duration in seconds (default: %(default)ss)",
    )
    parser.add_argument(
        "--quality",
        action="store_true",
        default=True,
        help="Also run quality benchmark (default: True)",
    )
    parser.add_argument(
        "--no-quality",
        action="store_false",
        dest="quality",
        help="Skip quality benchmark",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: auto-generated under data/output/)",
    )
    args = parser.parse_args()

    # Delegate to the single-model runner
    runner_path = PROJECT_ROOT / "scripts" / "run_one_model.py"
    if runner_path.exists():
        # Run via subprocess so the sub-script gets its own clean environment
        import subprocess

        cmd = [
            sys.executable,
            str(runner_path),
            "--model", args.model,
            "--duration", str(args.duration),
        ]
        if args.output:
            cmd.extend(["--output", args.output])
        if not args.quality:
            cmd.append("--no-quality")

        print(f"[e2e] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    else:
        print(f"[e2e] ERROR: {runner_path} not found — cannot run benchmark")
        sys.exit(1)


if __name__ == "__main__":
    main()
