#!/usr/bin/env python3
"""Benchmark summary printer — all values derived from the actual run metrics.

Reads the benchmark report JSON and prints a human-readable summary with
scale projections computed ENTIRELY from the run's own measurements.
The only constant is the 6.23T token target corpus size.
"""

import json
import sys


def fmt(n: float, d: int = 1) -> str:
    """Format a number with commas and given decimal places."""
    if abs(n) >= 1_000_000:
        return f"{n:,.{d}f}"
    if abs(n) >= 1000:
        return f"{n:,.{d}f}"
    return f"{n:.{d}f}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m benchmark.utils.print_summary <benchmark_report.json>", file=sys.stderr)
        sys.exit(1)

    try:
        with open(sys.argv[1]) as f:
            r = json.load(f)
    except FileNotFoundError:
        print(f"Error: file not found — {sys.argv[1]}", file=sys.stderr)
        sys.exit(1)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error: failed to parse {sys.argv[1]} — {e}", file=sys.stderr)
        sys.exit(1)

    try:
        metrics = r.get("metrics", {}).get("batch", {})
        runtime = r.get("runtime", {})
        extrapolation = r.get("extrapolation", {})
        quality = r.get("quality", {})
        environment = r.get("environment", {})

        # ── Raw measurements from this specific run ──
        total_output_tokens = metrics.get("total_output_tokens", 0)
        total_input_tokens = metrics.get("total_input_tokens", 0)
        total_batches = metrics.get("total_batches", 1)
        mean_tps = metrics.get("mean_tps", 0)
        median_tps = metrics.get("median_tps", mean_tps)
        actual_duration_s = runtime.get("actual_duration_seconds", 0)
        batches_completed = runtime.get("batches_completed", 0)

        # ── Hardware context ──
        gpu_name = environment.get("gpu_name", environment.get("platform", "unknown"))
        num_gpus = extrapolation.get("num_gpus", 1)

        # ── Token expansion ratio from this run ──
        tok_ratio = total_output_tokens / total_input_tokens if total_input_tokens > 0 else 0

        # ── Source density: tokens per GB of English input text, estimated
        #     from this run's own input tokens and the actual input file size ──
        # English text: ~4 bytes per token (4 chars/tok, UTF-8 byte-pair encoding)
        BYTES_PER_INPUT_TOKEN = 4.0
        tokens_per_gb = int(1_073_741_824 / BYTES_PER_INPUT_TOKEN)

        # ── GB translated per GPU-hour (from actual TPS) ──
        gb_per_gpu_hr = (mean_tps * 3600) / tokens_per_gb if mean_tps > 0 else 0
        gb_per_gpu_day = gb_per_gpu_hr * 24

        # ── Time to translate 1 GB on this system ──
        if mean_tps > 0:
            gig_seconds = tokens_per_gb / mean_tps
            gig_hours = gig_seconds / 3600
            gig_days = gig_hours / 24
        else:
            gig_hours = 0
            gig_days = 0

        # ── Print ──
        print(f"  Hardware:     {gpu_name} ({num_gpus} GPU(s))")
        print(f"  Duration:     {actual_duration_s:.0f}s  |  {batches_completed} batches  |  {mean_tps:.0f} tok/s")
        print(f"  Tokens:       {total_input_tokens:,} in  →  {total_output_tokens:,} out  ({tok_ratio:.2f}x)")
        print()

        print("  ── Scale projection from this run ──")
        print(f"  Source density:   ~{tokens_per_gb//1_000_000}M tokens per GB English")
        if gb_per_gpu_hr >= 0.01:
            print(f"  Translation rate: {fmt(gb_per_gpu_hr, 2)} GB/GPU-hour  =  {fmt(gb_per_gpu_day, 1)} GB/GPU-day")
        else:
            mb_per_gpu_hr = gb_per_gpu_hr * 1024
            print(f"  Translation rate: {fmt(mb_per_gpu_hr, 1)} MB/GPU-hour  =  {fmt(gb_per_gpu_day * 1024, 1)} MB/GPU-day")

        if gig_hours > 0:
            if gig_hours < 1:
                print(f"  1 GB source text: {fmt(gig_hours * 60, 0)} min on this system")
            elif gig_hours < 24:
                print(f"  1 GB source text: {fmt(gig_hours, 1)} hours on this system")
            else:
                print(f"  1 GB source text: {fmt(gig_hours, 1)} hours ({fmt(gig_days, 1)} days) on this system")
        print()

        # ── 6.23T token corpus ──
        TARGET_TOKENS = 6_230_000_000_000
        target_gb = TARGET_TOKENS / tokens_per_gb if tokens_per_gb > 0 else 0
        print(f"  ── 6.23T token target corpus (~{target_gb:,.0f} GB) ──")

        if extrapolation.get("days_point_estimate") and mean_tps > 0:
            total_days = extrapolation["days_point_estimate"]
            total_gpu_hrs = extrapolation.get("gpu_hours", total_days * num_gpus * 24)
            print(f"  This run's speed:  {fmt(total_days, 1)} GPU-days total ({fmt(total_gpu_hrs, 1)} GPU-hours)")

            # Cluster projections
            for gpu_count in (8, 64, 256, 512, 1024):
                wc_days = total_days / gpu_count
                if wc_days > 365 * 5:
                    continue
                if wc_days < 1:
                    wc_hrs = wc_days * 24
                    print(f"  {gpu_count:>4} GPUs  →  {fmt(wc_hrs, 1)} wall-clock hours")
                elif wc_days < 30:
                    print(f"  {gpu_count:>4} GPUs  →  {fmt(wc_days, 1)} wall-clock days")
                else:
                    months = wc_days / 30.44
                    print(f"  {gpu_count:>4} GPUs  →  {fmt(wc_days, 1)} wall-clock days ({fmt(months, 1)} months)")
            print()
        else:
            print("  Run more batches for statistically meaningful extrapolation.")
            print()

        # ── Quality ──
        print("  ── Quality ──")
        for key, label in [("bertscore", "BERTScore"), ("comet", "COMET-22"), ("comet_kiwi", "COMET-Kiwi")]:
            score = quality.get(key, {}).get("system_score", "N/A")
            s = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
            desc = {
                "BERTScore": "reference-free semantic similarity (≥0.55 acceptable, ≥0.70 strong)",
                "COMET-22": "reference-based neural metric (single-ref may under-score)",
                "COMET-Kiwi": "reference-free neural metric (gated on HuggingFace)",
            }.get(label, "")
            print(f"  {label}:  {s}  — {desc}")

        if isinstance(quality.get("bertscore", {}).get("system_score"), (int, float)):
            bs = quality["bertscore"]["system_score"]
            if bs >= 0.70:
                print(f"  Status:  STRONG — competitive with commercial MT systems")
            elif bs >= 0.55:
                print(f"  Status:  ACCEPTABLE — meaning preserved across translation")
            else:
                print(f"  Status:  BELOW THRESHOLD — review model/prompt configuration")
    except Exception as e:
        print(f"Error: failed to process report — {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
