#!/usr/bin/env python3
r"""Scientific TPS benchmark — measure every optimization in isolation.

Runs a matrix of controlled experiments on H200 GPUs:
  - Causal models (TranslateGemma-4B) evaluate the full 2³ factorial combinations of
    FP8, torch.compile, and Flash attention, as well as speculative decoding,
    PagedAttention, and continuous batching.
  - Encoder-decoder models (NLLB, MADLAD) evaluate BF16 baseline vs eager attention.
    Invalid combinations (FP8/compile) are skipped to avoid crashes and false labeling.

Output: data/output/scientific_tps_matrix.csv

Usage:
    python3 scripts/benchmark_scientific_tps.py
"""
import json, csv, os, sys, time, tempfile, atexit, subprocess, fcntl
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT = ROOT / "data" / "output" / f"scientific_tps_matrix_{TIMESTAMP}.csv"
DEFAULT_OUTPUT = ROOT / "data" / "output" / "scientific_tps_matrix.csv"

# ── Models to benchmark ──────────────────────────────────────────
# Format: (model_id, huggingface_path, architecture_type)
MODELS = [
    ("nllb_600m",    "facebook/nllb-200-distilled-600M", "nllb"),
    ("nllb_1.3b",    "facebook/nllb-200-distilled-1.3B", "nllb"),
    ("nllb_3.3b",    "facebook/nllb-200-3.3B",           "nllb"),
    ("madlad_3b",    "google/madlad400-3b-mt",           "madlad"),
    ("gemma_4b",     "google/translategemma-4b-it",      "gemma"),
]

# ── Optimization Matrix Definitions ──────────────────────────────
EXPERIMENTS = [
    {
        "label": "baseline_bf16",
        "fp8": False,
        "compile": False,
        "flash": True,
        "speculative": False,
        "paged": False,
        "continuous": False,
        "safe_mode": False,
        "description": "BF16 baseline with Flash SDPA, no compile, no FP8",
    },
    {
        "label": "fp8_smoothquant",
        "fp8": True,
        "compile": False,
        "flash": True,
        "speculative": False,
        "paged": False,
        "continuous": False,
        "safe_mode": False,
        "description": "SmoothQuant + static FP8, no compile",
    },
    {
        "label": "torch_compile",
        "fp8": False,
        "compile": True,
        "flash": True,
        "speculative": False,
        "paged": False,
        "continuous": False,
        "safe_mode": False,
        "description": "torch.compile only, BF16, Flash SDPA on",
    },
    {
        "label": "no_flash_sdpa",
        "fp8": False,
        "compile": False,
        "flash": False,
        "speculative": False,
        "paged": False,
        "continuous": False,
        "safe_mode": False,  # Changed to False so we only measure attention delta, not cudaMallocAsync
        "description": "Baseline (BF16, Flash SDPA disabled, eager attention)",
    },
    {
        "label": "eager_safe_mode",
        "fp8": False,
        "compile": False,
        "flash": False,
        "speculative": False,
        "paged": False,
        "continuous": False,
        "safe_mode": True,  # Disables both Flash attention and cudaMallocAsync for a true unaccelerated baseline
        "description": "Safe-mode baseline (BF16, eager attention, cudaMallocAsync disabled)",
    },
    {
        "label": "fp8_plus_compile",
        "fp8": True,
        "compile": True,
        "flash": True,
        "speculative": False,
        "paged": False,
        "continuous": False,
        "safe_mode": False,
        "description": "FP8 + torch.compile + Flash SDPA (all core optimizations)",
    },
    {
        "label": "fp8_no_flash",
        "fp8": True,
        "compile": False,
        "flash": False,
        "speculative": False,
        "paged": False,
        "continuous": False,
        "safe_mode": False,
        "description": "FP8 + eager attention (isolate FP8 without Flash SDPA)",
    },
    {
        "label": "compile_no_flash",
        "fp8": False,
        "compile": True,
        "flash": False,
        "speculative": False,
        "paged": False,
        "continuous": False,
        "safe_mode": False,
        "description": "torch.compile + eager attention (isolate compile without Flash SDPA)",
    },
    {
        "label": "fp8_compile_no_flash",
        "fp8": True,
        "compile": True,
        "flash": False,
        "speculative": False,
        "paged": False,
        "continuous": False,
        "safe_mode": False,
        "description": "FP8 + torch.compile, Flash SDPA off (all opts except flash)",
    },
    {
        "label": "speculative_decoding",
        "fp8": False,
        "compile": False,
        "flash": True,
        "speculative": True,
        "paged": False,
        "continuous": False,
        "safe_mode": False,
        "description": "Speculative decoding (K=3, self-speculative draft layers)",
    },
    {
        "label": "paged_attention",
        "fp8": False,
        "compile": False,
        "flash": True,
        "speculative": False,
        "paged": True,
        "continuous": False,
        "safe_mode": False,
        "description": "PagedAttention KV-cache optimization",
    },
    {
        "label": "continuous_batching",
        "fp8": False,
        "compile": False,
        "flash": True,
        "speculative": False,
        "paged": True,
        "continuous": True,
        "safe_mode": False,
        "description": "Continuous Batching (with PagedAttention)",
    },
]

GPU_COUNTS = [1, 2]
DURATION_SECONDS = 60

FIELD_NAMES = [
    "timestamp", "model_id", "model_path", "model_type",
    "experiment", "gpus", "dp",
    "mean_tps", "median_tps", "std_tps",
    "total_tokens", "batches", "duration_s",
    "flash_sdpa", "torch_compile", "fp8_smoothquant",
    "speculative", "paged_attention", "continuous_batching",
    "batch_size",
]

# Track temp files for cleanup
_temp_files: list[str] = []


def _cleanup_temp_files():
    for f in _temp_files:
        try:
            os.unlink(f)
        except OSError:
            pass


atexit.register(_cleanup_temp_files)


def _write_temp_config(cfg: dict) -> str:
    """Write a YAML config to a temp file, return its path."""
    import yaml
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="sci_tps_")
    with os.fdopen(fd, "w") as f:
        yaml.dump(cfg, f)
    _temp_files.append(path)
    return path


def _build_config(
    model_path: str,
    model_type: str,
    experiment: dict,
    num_gpus: int,
) -> dict:
    """Build a benchmark config dict for one experiment cell."""
    backend_type = "encoder_decoder" if model_type in ("nllb", "madlad") else "auto"

    cfg = {
        "backend": "auto",
        "model": {
            "model_path": model_path,
            "tokenizer_path": model_path,
            "max_input_tokens": 512,
            "max_new_tokens": 256,
            "temperature": 0.0,
            "do_sample": False,
            "num_beams": 1,
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "use_flash_attention": experiment["flash"],
            "backend_type": backend_type,
            "plugin_name": "",
            "plugin_config": {},
            "use_speculative": experiment.get("speculative", False),
            "speculative_mode": "self",
            "speculative_num_tokens": 3,
            "speculative_draft_model": "",
            "speculative_num_draft_layers": 0,
            "use_paged_attention": experiment.get("paged", False),
            "use_continuous_batching": experiment.get("continuous", False),
            "quantization": "bf16",
            "data_parallel_size": num_gpus,
            "nllb_source_lang": "eng_Latn",
            "nllb_target_lang": "tur_Latn",
        },
        "runtime": {
            "target_duration_seconds": DURATION_SECONDS,
            "checkpoint_interval_seconds": 300,
            "heartbeat_interval_seconds": 30,
            "metrics_sample_rate_hz": 1,
            "seed": 42,
        },
        "data": {
            "input_paths": ["./data/input/*.jsonl.gz"],
            "output_dir": "./output/scientific",
            "reference_set_path": "./data/references/golden_en_tr.jsonl",
            "prefetch_workers": 4,
            "shuffle": False,
            "min_chunk_tokens": 10,
            "max_garbage_ratio": 0.95,
        },
        "extrapolation": {
            "total_clearnet_non_tr_tokens": 200_000_000_000,
            "gpu_cost_per_hour_usd": None,
        },
    }
    return cfg


def run_experiment(
    model_id: str,
    model_path: str,
    model_type: str,
    experiment: dict,
    num_gpus: int,
) -> dict | None:
    """Run one benchmark cell. Returns a dict row for the CSV, or None on failure."""
    exp_label = experiment["label"]
    dp = num_gpus

    print(f"\n{'=' * 72}")
    print(f"  {model_id} | {exp_label} | dp={dp} ({num_gpus} GPU(s))")
    print(f"  {experiment['description']}")
    print(f"{'=' * 72}")

    cfg = _build_config(model_path, model_type, experiment, num_gpus)
    config_path = _write_temp_config(cfg)

    # ── Build environment ──
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))

    # FP8/SmoothQuant control
    if experiment["fp8"]:
        env.pop("TR_SKIP_FP8", None)
        env.pop("TR_SKIP_SMOOTHQUANT", None)
    else:
        env["TR_SKIP_FP8"] = "1"
        env["TR_SKIP_SMOOTHQUANT"] = "1"

    # Force batch sizes explicitly to bypass the BatchSizeTuner search
    # This prevents tuner time-waste and CUDA/MPS OOMs during exploration runs.
    base_bs = 16 if experiment.get("continuous") else (256 if num_gpus == 2 else 128)
    current_bs = base_bs

    while True:
        # ── Build command ──
        cmd = [
            sys.executable, "-m", "benchmark",
            "--config", config_path,
            "--translate-only",
            "--duration", str(DURATION_SECONDS),
        ]

        if not experiment["compile"]:
            cmd.append("--no-compile")

        if experiment.get("safe_mode"):
            cmd.append("--safe-mode")

        cmd.extend(["--batch-size", str(current_bs)])

        # For NLLB/MADLAD, tell the CLI it's an encoder-decoder model
        if model_type in ("nllb", "madlad"):
            cmd.append("--nllb")

        print(f"  CMD: {' '.join(cmd)}")
        print(f"  FP8={'ON' if experiment['fp8'] else 'OFF'} | "
              f"Compile={'ON' if experiment['compile'] else 'OFF'} | "
              f"Flash={'ON' if experiment['flash'] else 'OFF'} | "
              f"DP={dp} | BatchSize={current_bs}")

        # ── Execute ──
        t_start = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 min hard timeout
                env=env,
                cwd=str(ROOT),
            )
        except subprocess.TimeoutExpired:
            print("  ⚠ TIMEOUT (10 min) — skipping this cell")
            return None

        elapsed = time.time() - t_start
        print(f"  Completed in {elapsed:.0f}s (exit={result.returncode})")

        # Check for CUDA OOM / OutOfMemoryError in stdout or stderr
        is_oom = False
        if result.returncode != 0:
            output_for_oom_check = (result.stdout or "") + (result.stderr or "")
            if any(kw in output_for_oom_check for kw in ["OutOfMemoryError", "out of memory", "OOM"]):
                is_oom = True

        if is_oom:
            next_bs = current_bs // 2
            if next_bs >= 8:
                print(f"  ⚠ CUDA OOM detected at batch size {current_bs}! Retrying with batch size {next_bs}...")
                current_bs = next_bs
                continue
            else:
                print(f"  ⚠ CUDA OOM detected at batch size {current_bs}, but cannot reduce batch size further (minimum is 8).")
                print("--- STDOUT ---")
                print(result.stdout)
                print("--- STDERR ---")
                print(result.stderr)
                return None

        if result.returncode != 0:
            print(f"  ⚠ Subprocess failed with exit code {result.returncode}")
            print("--- STDOUT ---")
            print(result.stdout)
            print("--- STDERR ---")
            print(result.stderr)
            return None

        # Successfully ran. Break out to parse results.
        break

    # Parse exact run directory from stdout to avoid race conditions or picking up old reports
    report_dir = None
    if result.stdout:
        for line in result.stdout.split("\n"):
            if "Run dir:" in line:
                parts = line.split("Run dir:")
                if len(parts) > 1:
                    report_dir = Path(parts[1].strip())
                    break

    if not report_dir:
        print("  ⚠ Could not parse Run dir from stdout")
        print("--- STDOUT ---")
        print(result.stdout)
        print("--- STDERR ---")
        print(result.stderr)
        return None

    # Support both relative and absolute report directory paths
    if report_dir.is_absolute():
        report_path = report_dir / "report" / "benchmark_report.json"
    else:
        report_path = ROOT / report_dir / "report" / "benchmark_report.json"

    if not report_path.exists():
        print(f"  ⚠ Report file does not exist: {report_path}")
        return None

    print(f"  Report: {report_path}")

    with open(report_path) as f:
        report = json.load(f)

    # Extract TPS metrics
    batch_metrics = report.get("metrics", {}).get("batch", {}) or {}
    extrapolation = report.get("extrapolation", {}) or {}
    runtime = report.get("runtime", {}) or {}

    mean_tps = batch_metrics.get("mean_tps")
    if mean_tps is None:
        mean_tps = extrapolation.get("mean_tokens_per_second")
    median_tps = batch_metrics.get("median_tps")
    std_tps = batch_metrics.get("std_tps")

    if mean_tps is not None:
        print(f"  ✓ TPS: mean={mean_tps:.1f}, median={median_tps if median_tps is not None else 0.0:.1f}, std={std_tps if std_tps is not None else 0.0:.1f}")
    else:
        print(f"  ⚠ No TPS data found in report")

    # Safe lookup of optional runtime duration
    actual_duration = runtime.get("actual_duration_seconds")
    duration_s = round(actual_duration, 1) if actual_duration is not None else None

    return {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_id": model_id,
        "model_path": model_path.split("/")[-1],
        "model_type": model_type,
        "experiment": exp_label,
        "gpus": num_gpus,
        "dp": num_gpus,
        "mean_tps": round(mean_tps, 2) if mean_tps is not None else None,
        "median_tps": round(median_tps, 2) if median_tps is not None else None,
        "std_tps": round(std_tps, 2) if std_tps is not None else None,
        "total_tokens": runtime.get("total_tokens_translated"),
        "batches": runtime.get("batches_completed"),
        "duration_s": duration_s,
        "flash_sdpa": "✓" if experiment["flash"] else "✗",
        "torch_compile": "✓" if experiment["compile"] else "✗",
        "fp8_smoothquant": "✓" if experiment["fp8"] else "✗",
        "speculative": "✓" if experiment.get("speculative") else "✗",
        "paged_attention": "✓" if experiment.get("paged") else "✗",
        "continuous_batching": "✓" if experiment.get("continuous") else "✗",
        "batch_size": batch_metrics.get("batch_size", ""),
    }


def _write_csv_row(row: dict):
    """Append a single row to the CSV file incrementally in a concurrency-safe, race-free manner.

    Uses POSIX fcntl.flock to acquire an exclusive write lock on the file.
    Checks if the file is empty inside the lock to safely decide whether to write the header.
    Writes with UTF-8-SIG (BOM) for correct rendering in Windows Excel.
    """
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Open in read/write/append mode to check size and append safely
        with open(OUTPUT, "a+", newline="", encoding="utf-8-sig") as f:
            # Acquire exclusive lock (blocks until acquired)
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                # Seek to start to check size, then seek to end to append
                f.seek(0, os.SEEK_END)
                is_empty = f.tell() == 0
                w = csv.DictWriter(f, fieldnames=FIELD_NAMES)
                if is_empty:
                    w.writeheader()
                w.writerow(row)
            finally:
                # Release lock
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        print(f"  ⚠ Failed incremental write to CSV with lock: {e}")


def _get_active_runs() -> list[tuple]:
    """Generate the exact list of valid runs to execute.

    Filters experiments based on architectural compatibility:
      - Gemma (Causal): supports all experiments.
      - NLLB/MADLAD (Enc-Dec): supports baseline, eager attention, and FP8 weight quantization.
        torch.compile is skipped to avoid DynamicCache shape crashes, and causal-only
        optimizations (speculative, paged, continuous) are excluded.
    """
    runs = []
    for model_id, model_path, model_type in MODELS:
        for experiment in EXPERIMENTS:
            # Skip invalid optimizations for encoder-decoder models
            is_invalid_for_enc_dec = (
                experiment["compile"] or
                experiment.get("speculative") or
                experiment.get("paged") or
                experiment.get("continuous")
            )
            if is_invalid_for_enc_dec and model_type != "gemma":
                continue

            for num_gpus in GPU_COUNTS:
                runs.append((model_id, model_path, model_type, experiment, num_gpus))
    return runs


def main() -> int:
    active_runs = _get_active_runs()
    total = len(active_runs)

    print(f"Scientific TPS Benchmark Matrix")
    print(f"  Models:      {len(MODELS)}")
    print(f"  Experiments: {len(EXPERIMENTS)} (some causal-only)")
    print(f"  GPU counts:  {GPU_COUNTS}")
    print(f"  Total cells: {total}")
    print(f"  Duration:    {DURATION_SECONDS}s per cell")
    print(f"  Output:      {OUTPUT}")
    print()

    rows: list[dict] = []
    completed = 0
    failures = 0

    for model_id, model_path, model_type, experiment, num_gpus in active_runs:
        completed += 1
        print(f"\n[{completed}/{total}]", end="")

        row = run_experiment(
            model_id, model_path, model_type,
            experiment, num_gpus,
        )

        if row:
            rows.append(row)
            _write_csv_row(row)
            print(f"  → Saved ({len(rows)} rows so far)")
        else:
            print("  ⚠ Failed cell run recorded as None")
            failures += 1

        # Thermal cooldown: pause to prevent cumulative GPU heat and thermal throttling
        # between consecutive runs, ensuring scientific accuracy of TPS measurements.
        if completed < total:
            print("  Cooldown: sleeping for 5s to allow GPU temperature to stabilize...")
            time.sleep(5)

    # Copy the final timestamped CSV results to the default unversioned file path
    if OUTPUT.exists():
        try:
            import shutil
            shutil.copyfile(OUTPUT, DEFAULT_OUTPUT)
            print(f"  → Copied versioned results to: {DEFAULT_OUTPUT}")
        except Exception as e:
            print(f"  ⚠ Failed to copy versioned CSV: {e}")

    print(f"\n{'=' * 72}")
    print(f"DONE — {len(rows)}/{total} experiments completed (failures={failures})")
    print(f"  - Versioned results: {OUTPUT}")
    print(f"  - Default symlink:    {DEFAULT_OUTPUT}")
    print(f"{'=' * 72}")

    return 1 if failures > 0 or len(rows) == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
