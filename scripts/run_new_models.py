#!/usr/bin/env python3
"""
Sequential 5-Minute Benchmark Runner — New Model Integration Test (v3.4).

Benchmarks all 5 new models on MPS for 300s each and produces a comparison
report.  Designed for the H200Research benchmark harness.

Usage:
    python scripts/run_new_models.py
    python scripts/run_new_models.py --models e2b_qat e4b_qat
    python scripts/run_new_models.py --duration 600 --dry-run
"""

import gc
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Add project root to path ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

import psutil
import torch


def _mps_mem_info() -> dict:
    """Return current MPS memory state."""
    gc.collect()
    if hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    proc = psutil.Process()
    rss_gb = proc.memory_info().rss / (1024**3)
    drv_gb = torch.mps.driver_allocated_memory() / (1024**3) if hasattr(
        torch.mps, "driver_allocated_memory"
    ) else 0.0
    return {"rss_gb": round(rss_gb, 2), "driver_gb": round(drv_gb, 2)}


# ── Model Definitions ──────────────────────────────────────────────────────

@dataclass
class ModelRun:
    label: str
    model_path: str
    run_name: str  # Short name for output dirs
    backend_type: str = "auto"  # "auto" | "diffusion"
    diffusion_steps: int = 128
    guidance_scale: float = 2.0
    noise_schedule: str = "linear"
    max_input_tokens: int = 256
    max_new_tokens: int = 128

    def is_diffusion(self) -> bool:
        return self.backend_type == "diffusion" or "diffusiongemma" in self.model_path.lower()


# ── All 5 new models ───────────────────────────────────────────────────────

NEW_MODELS = [
    ModelRun(
        label="DiffusionGemma 26B-A4B",
        model_path="google/diffusiongemma-26B-A4B-it",
        run_name="diffusiongemma_26b",
        backend_type="diffusion",
        diffusion_steps=128,
        guidance_scale=2.0,
        noise_schedule="linear",
    ),
    ModelRun(
        label="Gemma 4 E2B QAT (BF16)",
        model_path="google/gemma-4-E2B-it-qat-mobile-ct",
        run_name="gemma4_e2b_qat_ct",
        backend_type="auto",
    ),
    ModelRun(
        label="Gemma 4 E4B QAT (BF16)",
        model_path="google/gemma-4-E4B-it-qat-mobile-ct",
        run_name="gemma4_e4b_qat_ct",
        backend_type="auto",
    ),
    ModelRun(
        label="Gemma 4 E2B Q4_0 (4-bit)",
        model_path="google/gemma-4-E2B-it-qat-mobile-transformers",
        run_name="gemma4_e2b_q4_0",
        backend_type="auto",
    ),
    ModelRun(
        label="Gemma 4 E4B Q4_0 (4-bit)",
        model_path="google/gemma-4-E4B-it-qat-mobile-transformers",
        run_name="gemma4_e4b_q4_0",
        backend_type="auto",
    ),
]


def run_benchmark(model: ModelRun, output_dir: Path, duration: int, dry_run: bool = False) -> dict:
    """Run a single model benchmark. Returns result dict."""
    from benchmark.config.schema import (
        BenchmarkConfig,
        ModelConfig,
        RuntimeConfig,
        DataConfig,
        ExtrapolationConfig,
    )
    from benchmark.hardware.backend import detect_backend
    from benchmark.inference.engine import InferenceEngine
    from benchmark.inference.sampling import DecodingParams
    from benchmark.data.loader import JSONLLoader
    from benchmark.data.chunker import TextChunker
    from benchmark.data.filters import ChunkFilter
    from benchmark.data.pipeline import AsyncPipeline
    from benchmark.metrics.collector import MetricsCollector
    from benchmark.orchestration.checkpoint import CheckpointManager
    from benchmark.reporting.aggregator import MetricsAggregator
    from benchmark.reporting.extrapolation import ExtrapolationModel
    from benchmark.reporting.json_report import JSONReportWriter
    from benchmark.reporting.markdown_report import MarkdownReportWriter
    from benchmark.utils.logging_setup import setup_logging
    from benchmark.utils.timer import PrecisionTimer

    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    import logging
    logger = logging.getLogger(f"bench.{model.run_name}")

    mem_before = _mps_mem_info()
    logger.info(
        "Starting %s (diffusion=%s) — RAM: %.1f GB RSS, %.1f GB driver",
        model.label, model.is_diffusion(), mem_before["rss_gb"], mem_before["driver_gb"],
    )

    config = BenchmarkConfig(
        backend="mps",
        model=ModelConfig(
            model_path=model.model_path,
            max_input_tokens=model.max_input_tokens if not dry_run else 64,
            max_new_tokens=model.max_new_tokens if not dry_run else 32,
            dtype="bfloat16",
            tensor_parallel_size=1,
            use_flash_attention=False,
            backend_type=model.backend_type,
            diffusion_steps=model.diffusion_steps if not dry_run else 16,
            guidance_scale=model.guidance_scale,
            noise_schedule=model.noise_schedule,
        ),
        runtime=RuntimeConfig(
            target_duration_seconds=duration if not dry_run else 30,
            checkpoint_interval_seconds=120,
            heartbeat_interval_seconds=15,
            seed=42,
        ),
        data=DataConfig(
            input_paths=["./data/input/fineweb_en_sample.jsonl.gz"],
            output_dir=str(output_dir),
            reference_set_path="./data/references/golden_en_tr.jsonl",
            prefetch_workers=2,
            shuffle=False,
        ),
        extrapolation=ExtrapolationConfig(
            total_clearnet_non_tr_tokens=6_230_000_000_000,
        ),
    )

    try:
        device_info = detect_backend("mps")
        logger.info("Backend: %s (%s)", device_info.backend, device_info.name)

        # Create engine with proper extra config
        extra = {
            "do_sample": config.model.do_sample,
            "num_beams": config.model.num_beams,
            "backend_type": config.model.backend_type,
            "diffusion": {
                "num_diffusion_steps": config.model.diffusion_steps,
                "noise_schedule": config.model.noise_schedule,
                "guidance_scale": config.model.guidance_scale,
            },
            "safe_mode": False,
        }

        engine = InferenceEngine(
            model_path=config.model.model_path,
            tokenizer_path=config.model.tokenizer_path,
            device_info=device_info,
            decoding_params=DecodingParams(
                max_new_tokens=config.model.max_new_tokens,
                temperature=config.model.temperature,
                do_sample=config.model.do_sample,
                num_beams=config.model.num_beams,
            ),
            use_flash_attention=False,
            use_torch_compile=True,
            max_input_tokens=config.model.max_input_tokens,
            backend_type=config.model.backend_type,
            extra=extra,
        )
        engine.load()

        # Batch size = 1 for MPS safety
        batch_size = 1
        engine._configured_batch_size = batch_size

        engine.warmup(batches=5 if not dry_run else 2)

        # Data pipeline
        loader = JSONLLoader(
            config.data.input_paths, shuffle=False, seed=42,
            max_shuffle_memory_gb=0.5,
        )
        chunker = TextChunker(
            engine.tokenizer, config.model.max_input_tokens, config.data.chunk_overlap_tokens,
        )
        filt = ChunkFilter(
            min_tokens=config.data.min_chunk_tokens,
            max_garbage_ratio=config.data.max_garbage_ratio,
        )
        pipeline = AsyncPipeline(
            loader, chunker, engine.tokenizer, filt,
            batch_size=batch_size, prefetch_workers=2, backend="mps",
        )

        metrics = MetricsCollector(output_dir / "metrics", device_info, 1)
        ckpt_mgr = CheckpointManager(output_dir, 120)

        pipeline.start_prefetch()
        timer = PrecisionTimer()
        timer.start()
        metrics.start(timer.start_time())

        target_duration = config.runtime.target_duration_seconds
        batches_completed = 0
        total_tokens = 0
        last_heartbeat = 0.0

        try:
            while timer.elapsed() < target_duration:
                batch = pipeline.next_batch()
                if batch is None:
                    if pipeline.draining():
                        break
                    continue

                result = engine.translate(batch)
                pipeline.release_batch(batch)
                metrics.log_batch(result)
                batches_completed += 1
                total_tokens += result.output_tokens_total

                now = timer.elapsed()
                if now - last_heartbeat >= config.runtime.heartbeat_interval_seconds:
                    tps = metrics.get_rolling_throughput()
                    logger.info(
                        "[%4.0fs] batches=%d tokens=%s tps=%.0f",
                        now, batches_completed, format(total_tokens, ","), tps,
                    )
                    last_heartbeat = now
        finally:
            metrics.stop()
            pipeline.stop_prefetch()
            run_duration = timer.elapsed()

        # ── Aggregate report ──
        aggregator = MetricsAggregator(output_dir / "metrics")
        summary = aggregator.aggregate()
        batch_stats = summary.get("batch", {})
        mean_tps = batch_stats.get("mean_tps", 0.0)

        ext = ExtrapolationModel(total_tokens=config.extrapolation.total_clearnet_non_tr_tokens)
        ext_result = ext.compute(mean_tps, batch_stats.get("std_tps", 0), 1)

        report = {
            "model": model.label,
            "model_path": model.model_path,
            "config": config.model_dump() if hasattr(config, "model_dump") else {},
            "runtime": {
                "actual_duration_seconds": round(run_duration, 1),
                "batches_completed": batches_completed,
                "total_tokens_translated": total_tokens,
            },
            "metrics": summary,
            "extrapolation": ext_result,
            "filter_stats": filt.stats.to_dict(),
        }

        JSONReportWriter().write(output_dir, report)
        MarkdownReportWriter().write(output_dir, report)

        mem_after = _mps_mem_info()
        logger.info(
            "Complete: %d batches, %d tokens, %.1f tok/s — RAM: %.1f GB RSS",
            batches_completed, total_tokens, mean_tps, mem_after["rss_gb"],
        )

        return {
            "model": model.label,
            "model_path": model.model_path,
            "success": True,
            "duration_s": round(run_duration, 1),
            "batches": batches_completed,
            "tokens": total_tokens,
            "mean_tps": round(mean_tps, 1),
            "median_tps": batch_stats.get("median_tps", 0),
            "p95_tps": batch_stats.get("p95_tps", 0),
            "mem_before_gb": mem_before["rss_gb"],
            "mem_after_gb": mem_after["rss_gb"],
        }

    except Exception as e:
        logger.error("Benchmark FAILED for %s: %s", model.label, e, exc_info=True)
        return {
            "model": model.label,
            "model_path": model.model_path,
            "success": False,
            "error": str(e),
        }
    finally:
        # ── Proper GPU memory cleanup for MPS (not just gc.collect()) ──
        # MPS allocator pool retains memory even after Python objects are freed;
        # we must expire the allocator pool + synchronize the MPS stream to
        # actually release the backing IOSurfaces back to the system.
        engine = None
        gc.collect()
        # Expire the MPS allocator so backing IOSurfaces are freed.
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        time.sleep(2)


def print_summary_table(results: list[dict]) -> None:
    """Print a formatted comparison table."""
    print()
    print("╔═════════════════════════════════════════════════════════════════════╗")
    print("║           SEQUENTIAL 5-MINUTE BENCHMARK — MPS RESULTS              ║")
    print("╠═════════════════════════════════════════════════════════════════════╣")
    header = f"║ {'Model':<30s} {'Status':>7s} {'Time':>8s} {'TPS':>10s} │"
    print(header)
    print("╠═════════════════════════════════════════════════════════════════════╣")

    for r in results:
        status = "OK" if r.get("success") else "FAIL"
        tps = r.get("mean_tps", "N/A")
        if isinstance(tps, (int, float)) and tps > 0:
            tps_str = f"{tps:>8.0f} tok/s"
        else:
            tps_str = f"{'N/A':>8s}"
        duration = r.get("duration_s", 0)
        print(f"║ {r['model']:<30s} {status:>7s} {str(duration)+'s':>8s} {tps_str} │")

    print("╚═════════════════════════════════════════════════════════════════════╝")
    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run sequential benchmarks for new models")
    parser.add_argument("--duration", type=int, default=300, help="Run duration in seconds (default: 300)")
    parser.add_argument("--dry-run", action="store_true", help="30-second smoke test per model")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Model labels to run (e.g. e2b_qat e4b_qat). Default: all 5")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    # Select models
    if args.models:
        selected = [m for m in NEW_MODELS if any(kw in m.run_name for kw in args.models)]
    else:
        selected = NEW_MODELS

    if not selected:
        print("No models matched. Available:", [m.run_name for m in NEW_MODELS])
        sys.exit(1)

    output_root = Path(args.output) if args.output else Path(
        f"data/output/benchmark_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    duration_label = "DRY-RUN (30s)" if args.dry_run else f"{args.duration}s"
    print(f"\n{'='*70}")
    print(f"  Sequential Benchmark — {len(selected)} models, {duration_label} each")
    print(f"  Output: {output_root}")
    print(f"  Platform: MPS — {torch.backends.mps.is_available()}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  RAM: {psutil.virtual_memory().total/(1024**3):.0f} GB")
    print(f"{'='*70}\n")

    results = []
    for i, model in enumerate(selected):
        print(f"\n── [{i+1}/{len(selected)}] {model.label} ──")
        run_dir = output_root / model.run_name
        result = run_benchmark(model, run_dir, args.duration, dry_run=args.dry_run)
        results.append(result)

        # Print quick status
        if result["success"]:
            print(f"  ✓ {result['mean_tps']:.0f} tok/s | {result['tokens']:,} tokens | {result['duration_s']:.0f}s")
        else:
            print(f"  ✗ FAILED: {result.get('error', 'unknown')}")

        # Summary JSONL
        with open(output_root / "results.jsonl", "a") as f:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")

        time.sleep(5)  # Cooldown

    # Final summary
    print_summary_table(results)

    # Write final report
    report_path = output_root / "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump({"timestamp": datetime.now(timezone.utc).isoformat(), "results": results}, f, indent=2)
    print(f"  Full report: {report_path}")
    print(f"  Run dirs:    {output_root}/\n")


if __name__ == "__main__":
    main()
