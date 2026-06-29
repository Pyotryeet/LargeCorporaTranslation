#!/usr/bin/env python3
r"""Scientific TPS benchmark — measure every optimization in isolation.

Runs a matrix of controlled experiments:
  Optimization: BF16, FP8+SmoothQuant, torch.compile, Flash SDPA off, dp=1, dp=2
  Models:       NLLB-600M, NLLB-1.3B, NLLB-3.3B, MADLAD-3B
  GPUs:         1, 2

Each cell runs --translate-only --duration 60 to capture steady-state TPS.
Config is mutated per-run to isolate the variable under test.

Output: data/output/scientific_tps_matrix.csv

Usage:
    python3 scripts/benchmark_scientific_tps.py
"""
import json, csv, os, subprocess, time, tempfile, shutil
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data" / "output" / "scientific_tps_matrix.csv"

MODELS = [
    "facebook/nllb-200-distilled-600M",
    "facebook/nllb-200-distilled-1.3B",
    "facebook/nllb-200-3.3B",
    "google/madlad400-3b-mt",
]

# ── Optimization matrix ──────────────────────────────────────────
# Each row = one experiment cell
# NLLB-600M gets both GPU counts tested; larger models get only 1 GPU
# (they won't fit 2 copies in VRAM) but we can still test 2-GPU DP for them
EXPERIMENTS = [
    # (label, model_idx, gpus, config_overrides)
    # ── Baseline BF16 ──
    ("bf16_1gpu",     None, 1, {"dtype": "bfloat16", "use_flash_attention": True, "data_parallel_size": 1}),
    ("bf16_2gpu",     None, 2, {"dtype": "bfloat16", "use_flash_attention": True, "data_parallel_size": 2}),
    # ── Flash SDPA off ──
    ("no_flash_1gpu",  None, 1, {"dtype": "bfloat16", "use_flash_attention": False, "data_parallel_size": 1}),
    # ── FP8 + SmoothQuant ──
    ("fp8_1gpu",       None, 1, {"dtype": "auto", "use_flash_attention": True, "data_parallel_size": 1}),
    ("fp8_2gpu",       None, 2, {"dtype": "auto", "use_flash_attention": True, "data_parallel_size": 2}),
    # ── torch.compile off ──
    ("no_compile_1gpu", None, 1, {"dtype": "bfloat16", "use_flash_attention": True, "data_parallel_size": 1}),
    # ── All-on (FP8+compile+flash+DP) ──
    ("all_on_1gpu",     None, 1, {"dtype": "auto", "use_flash_attention": True, "data_parallel_size": 1}),
    ("all_on_2gpu",     None, 2, {"dtype": "auto", "use_flash_attention": True, "data_parallel_size": 2}),
]

FIELD_NAMES = [
    "timestamp", "model", "experiment", "gpus", "mean_tps",
    "total_tokens", "batches", "duration_s",
    "flash_attn", "compile", "fp8_smoothquant", "dp",
    "dtype", "batch_size_reported",
]


def run_experiment(model_path, experiment_label, num_gpus, overrides, model_idx):
    """Run one benchmark cell. Returns a dict row for the CSV."""
    print(f"\n{'='*70}")
    print(f"  {experiment_label} | {model_path} | {num_gpus} GPU(s)")
    print(f"{'='*70}")

    model_short = model_path.split("/")[-1]

    # Build runtime config
    cfg = {
        "backend": "auto",
        "model": {
            "model_path": model_path,
            "tokenizer_path": model_path,
            "max_input_tokens": 512,
            "max_new_tokens": 512,
            "temperature": 0.0,
            "do_sample": False,
            "num_beams": 1,
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "use_flash_attention": True,
            "backend_type": "encoder_decoder",
            "plugin_name": "",
            "plugin_config": {},
            "use_speculative": False,
            "speculative_mode": "self",
            "speculative_num_tokens": 3,
            "speculative_draft_model": "",
            "speculative_num_draft_layers": 0,
            "use_paged_attention": False,
            "use_continuous_batching": False,
            "quantization": "bf16",
            "data_parallel_size": 1,
            "nllb_source_lang": "eng_Latn",
            "nllb_target_lang": "tur_Latn",
        },
        "runtime": {
            "target_duration_seconds": 60,
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
            "total_clearnet_non_tr_tokens": 200000000000,
            "gpu_cost_per_hour_usd": None,
        },
    }

    # Apply overrides
    for k, v in overrides.items():
        cfg["model"][k] = v

    # Write temp config
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    import yaml
    yaml.dump(cfg, tmp)
    tmp.close()

    # Set CUDA_VISIBLE_DEVICES based on GPU count
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0" if num_gpus == 1 else "0,1"
    env["TR_SKIP_FP8"] = "0"  # Enable FP8 by default
    env["TR_SKIP_SMOOTHQUANT"] = "0"  # Enable SmoothQuant by default

    # Override compile based on experiment
    no_compile = "--no-compile" if "no_compile" in experiment_label else ""

    # For FP8 experiment, enable it; for BF16 baseline, skip it
    if "fp8" in experiment_label or "all_on" in experiment_label:
        env.pop("TR_SKIP_FP8", None)
        env.pop("TR_SKIP_SMOOTHQUANT", None)
    else:
        env["TR_SKIP_FP8"] = "1"
        env["TR_SKIP_SMOOTHQUANT"] = "1"

    cmd = [
        "python3", "-m", "benchmark",
        "--config", tmp.name,
        "--translate-only",
        "--duration", "60",
    ]
    if no_compile:
        cmd.append(no_compile)

    print(f"  Running: {' '.join(cmd)}")
    print(f"  FP8={'ON' if env.get('TR_SKIP_FP8')=='0' else 'OFF'} "
          f"SmoothQuant={'ON' if env.get('TR_SKIP_SMOOTHQUANT')=='0' else 'OFF'} "
          f"Compile={'ON' if not no_compile else 'OFF'}")

    t_start = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            env=env, cwd=str(ROOT),
        )
    except subprocess.TimeoutExpired:
        print("  TIMEOUT (10min)")
        return None

    elapsed = time.time() - t_start
    print(f"  Completed in {elapsed:.0f}s (exit={result.returncode})")

    # Find the report
    output_dir = ROOT / "output" / "scientific"
    reports = sorted(output_dir.glob("**/benchmark_report.json"))
    report = None
    if reports:
        report = reports[-1]  # newest

    if not report or not report.exists():
        print(f"  WARNING: No report found in {output_dir}")
        return None

    with open(report) as f:
        r = json.load(f)

    batch = (r.get("metrics", {}) or {}).get("batch", {}) or {}
    ext = r.get("extrapolation", {}) or {}
    rt = r.get("runtime", {}) or {}
    tps = ext.get("mean_tokens_per_second") or batch.get("mean_tps")

    return {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": model_short,
        "experiment": experiment_label,
        "gpus": num_gpus,
        "mean_tps": round(tps, 1) if tps else None,
        "total_tokens": rt.get("total_tokens_translated"),
        "batches": rt.get("batches_completed"),
        "duration_s": round(rt.get("actual_duration_seconds", 0), 0) if rt.get("actual_duration_seconds") else None,
        "flash_attn": "✓" if overrides.get("use_flash_attention", True) else "✗",
        "compile": "✗" if "no_compile" in experiment_label else "✓",
        "fp8_smoothquant": "✓" if ("fp8" in experiment_label or "all_on" in experiment_label) else "✗",
        "dp": f"✓ ({num_gpus}×GPU)" if overrides.get("data_parallel_size", 1) > 1 else "✗",
        "dtype": overrides.get("dtype", "bfloat16"),
        "batch_size_reported": "",
    }


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for exp_label, model_idx, num_gpus, overrides in EXPERIMENTS:
        # Which models to test for this experiment?
        if model_idx is not None:
            model_list = [MODELS[model_idx]]
        else:
            model_list = MODELS  # All models

        for model_path in model_list:
            row = run_experiment(model_path, exp_label, num_gpus, overrides, model_idx)
            if row:
                rows.append(row)
                # Write incrementally for crash safety
                with open(OUTPUT, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=FIELD_NAMES)
                    w.writeheader()
                    w.writerows(rows)
            # Brief pause between runs for GPU cooldown
            time.sleep(5)

    print(f"\n{'='*70}")
    print(f"DONE — {len(rows)} experiments → {OUTPUT}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
