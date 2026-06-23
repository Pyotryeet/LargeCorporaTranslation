#!/usr/bin/env python3
"""Comparative benchmark: speed + accuracy across all models on MPS.

Runs each model sequentially in --dry-run mode (120 s translation + quality)
and collects throughput, latency, and quality scores into a single JSON file.

Usage:
  source .venv/bin/activate
  python benchmark_all_models.py

Models tested (all fit in 36-48 GB unified memory):
  - TranslateGemma 4B       (google/translategemma-4b-it)
  - Gemma 4 E2B QAT         (google/gemma-4-E2B-it-qat-mobile-ct)
  - Gemma 4 E4B QAT         (google/gemma-4-E4B-it-qat-mobile-ct)
  - Ministral 3B            (mistralai/Ministral-3B-Instruct)
  - NLLB 600M distilled     (facebook/nllb-200-distilled-600M)  ← encoder-decoder

Each model gets the same input data, same reference set, same conditions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
import gc
from datetime import datetime, timezone
from pathlib import Path

import torch
import warnings
warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*")
warnings.filterwarnings("ignore", message=".*pkg_resources.*deprecated.*")
warnings.filterwarnings("ignore", message=".*`torch_dtype`.*deprecated.*")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ["PYTHONUNBUFFERED"] = "1"  # flush stdout on every write

# ── Suppress httpx / urllib INFO spam from HF downloads ──
import logging as _logging
for _noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub.file_download"):
    _logging.getLogger(_noisy).setLevel(_logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.hardware.backend import detect_backend
from benchmark.inference.engine import InferenceEngine
from benchmark.inference.sampling import DecodingParams
from benchmark.inference.batch_tuner import BatchSizeTuner
from benchmark.data.loader import JSONLLoader
from benchmark.data.chunker import TextChunker
from benchmark.data.filters import ChunkFilter
from benchmark.data.pipeline import AsyncPipeline
from benchmark.metrics.collector import MetricsCollector
from benchmark.quality.benchmark import QualityBenchmark
from benchmark.reporting.aggregator import MetricsAggregator
from benchmark.reporting.extrapolation import ExtrapolationModel
from benchmark.orchestration.checkpoint import CheckpointManager
from benchmark.orchestration.signals import SignalHandler, register_cleanup
from benchmark.utils.logging_setup import setup_logging
from benchmark.utils.env_check import run_preflight_checks
from benchmark.utils.version import get_environment_snapshot
from benchmark.utils.timer import PrecisionTimer

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("benchmark_all")

# ═════════════════════════════════════════════════════════════════════════════
# Model definitions — all tested on MPS
# ═════════════════════════════════════════════════════════════════════════════

MODELS = [
    # ── NLLB 600M — encoder-decoder, greedy (beam=1 for MPS speed) ──
    {
        "name": "NLLB-200-distilled-600M",
        "model_path": "facebook/nllb-200-distilled-600M",
        "type": "encoder_decoder",
        "extra": {
            "nllb_source_lang": "eng_Latn",
            "nllb_target_lang": "tur_Latn",
            "num_beams": 1,
        },
        "tags": ["nllb", "600M", "encoder-decoder", "multilingual"],
    },
    # ── SmolLM2 1.7B — smallest autoregressive, fast ──
    {
        "name": "SmolLM2-1.7B-Instruct",
        "model_path": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "type": "autoregressive",
        "tags": ["smollm", "1.7B", "instruction", "bf16"],
    },
    # ── TranslateGemma 4B — autoregressive, most proven ──
    {
        "name": "TranslateGemma-4B",
        "model_path": "google/translategemma-4b-it",
        "type": "autoregressive",
        "tags": ["gemma", "4B", "translator", "bf16"],
    },
]

# Input data and references (project defaults).
INPUT_GLOB = "data/input/fineweb_en_sample.jsonl.gz"
REFERENCE_SET = "data/references/golden_en_tr.jsonl"
OUTPUT_DIR = "data/output"
RUN_DURATION = 120          # seconds per model
QUALITY_MAX_REFS = 32       # limit quality benchmark refs for speed
MPS_MAX_BATCH = 4           # MPS can't handle large batches efficiently
SEED = 42


def run_one(
    model_def: dict,
    device_info,
    env_snapshot: dict,
    run_index: int,
) -> dict | None:
    """Run one model through translation + quality benchmark.

    Returns a dict with speed, quality, and metadata, or None on failure.
    """
    name = model_def["name"]
    path = model_def["model_path"]
    backend_type = model_def.get("type", "auto")
    extra = dict(model_def.get("extra", {}))

    # ── Version compatibility check ──
    if "requires_transformers" in model_def:
        import transformers as _tf
        _required = tuple(map(int, model_def["requires_transformers"].split(".")))
        _current = tuple(map(int, _tf.__version__.split(".")[:len(_required)]))
        if _current < _required:
            msg = (
                f"Skipped (requires transformers>={model_def['requires_transformers']}, "
                f"have {_tf.__version__})"
            )
            print(f"    ⏭ {msg}")
            return {
                "model": name, "model_path": path, "type": backend_type,
                "tags": model_def.get("tags", []),
                "error": msg, "status": "skipped",
            }

    print(f"\n{'='*70}")
    print(f"[{run_index}] {name}")
    print(f"    Model: {path}")
    print(f"    Type:  {backend_type}")
    print(f"{'='*70}")

    # ── Create run directory ──
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(OUTPUT_DIR) / f"bm_{name.replace('-','_').replace(' ','_')}_{run_index:02d}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir)

    try:
        # ── Load model ──
        load_start = time.monotonic()
        engine = InferenceEngine(
            model_path=path,
            tokenizer_path="",
            device_info=device_info,
            decoding_params=DecodingParams(
                max_new_tokens=128,  # shorter = faster on MPS
                temperature=0.0,
            ),
            use_flash_attention=False,
            use_torch_compile=False,
            max_input_tokens=128,
            backend_type=backend_type,
            extra=extra,
        )
        engine.load()
        load_secs = time.monotonic() - load_start
        print(f"    Loaded in {load_secs:.1f}s")

        # ── Batch tune ──
        tuner = BatchSizeTuner()
        batch_size = tuner.tune(
            engine.model, engine.tokenizer,
            device_info.device, device_info.backend,
            128,
        )
        batch_size = min(batch_size, MPS_MAX_BATCH)
        print(f"    Batch size: {batch_size} (clamped to {MPS_MAX_BATCH} for MPS)")

        # ── Warmup ──
        engine.warmup(batches=10)

        # ── Data pipeline ──
        loader = JSONLLoader([INPUT_GLOB], shuffle=True, seed=SEED)
        chunker = TextChunker(engine.tokenizer, 128, 50)
        filt = ChunkFilter(min_tokens=10, max_garbage_ratio=0.95)
        pipeline = AsyncPipeline(
            loader, chunker, engine.tokenizer, filt,
            batch_size=batch_size,
            prefetch_workers=4,
            backend=device_info.backend,
        )

        # ── Metrics ──
        metrics = MetricsCollector(
            run_dir / "metrics", device_info, 1,
        )
        pipeline.start_prefetch()
        timer = PrecisionTimer()
        timer.start()
        metrics.start(timer.start_time())

        # ── Translation loop ──
        batches_completed = 0
        total_tokens = 0
        signals = SignalHandler()

        try:
            while timer.elapsed() < RUN_DURATION:
                if signals.killed.is_set():
                    break
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

                # Heartbeat every batch.
                tps = metrics.get_rolling_throughput()
                now = timer.elapsed()
                print(f"    [{now:5.0f}s] batch={batches_completed} "
                      f"tokens={total_tokens:,} tps={tps:.0f}", flush=True)

                if timer.elapsed() > RUN_DURATION:
                    break
        finally:
            metrics.stop()
            pipeline.stop_prefetch()
            run_duration = timer.elapsed()

        print(f"    Translation: {batches_completed} batches, "
              f"{total_tokens:,} tokens in {run_duration:.0f}s")

        # ── Aggregate metrics ──
        aggregator = MetricsAggregator(run_dir / "metrics")
        metrics_summary = aggregator.aggregate()
        batch_stats = metrics_summary.get("batch", {})
        device_stats = metrics_summary.get("device", {})

        mean_tps = batch_stats.get("mean_tps", 0)
        median_tps = batch_stats.get("median_tps", 0)
        p95_tps = batch_stats.get("p95_tps", 0)
        std_tps = batch_stats.get("std_tps", 0)
        mean_latency_ms = batch_stats.get("mean_latency_ms", 0)

        print(f"    Throughput: {mean_tps:.1f} tok/s (median={median_tps:.1f}, P95={p95_tps:.1f})")

        # ── Quality benchmark (BERTScore only on MPS — COMET hangs) ──
        quality = {}
        if Path(REFERENCE_SET).exists():
            print(f"    Running quality benchmark (max {QUALITY_MAX_REFS} refs, BERTScore only on MPS)...")
            try:
                from benchmark.quality.references import ReferenceLoader
                from benchmark.quality.metrics_bertscore import compute_bertscore
                from benchmark.quality.benchmark import _build_batch
                import time as _time

                q_start = _time.monotonic()
                loader = ReferenceLoader(REFERENCE_SET)
                sources, references = loader.load()

                # Cap references for speed.
                if QUALITY_MAX_REFS and QUALITY_MAX_REFS < len(sources):
                    sources = sources[:QUALITY_MAX_REFS]
                    references = references[:QUALITY_MAX_REFS]

                # Translate reference sentences in one batch.
                _MiniBatch = type("_MiniBatch", (), {})
                input_ids, attention_mask, _ = _build_batch(sources, engine.tokenizer, engine.devices[0])
                mb = _MiniBatch()
                mb.input_ids = input_ids
                mb.attention_mask = attention_mask
                mb.raw_texts = sources
                mb.batch_id = 0
                t_result = engine.translate(mb)
                hypotheses = [g.translated_text for g in t_result.generations]

                # BERTScore only — COMET hangs on MPS.
                bs_result = compute_bertscore(sources, hypotheses)
                quality = {
                    "bertscore": {"system_score": bs_result.get("system_score")},
                    "comet": {"system_score": None, "error": "skipped on MPS"},
                    "comet_kiwi": {"system_score": None, "error": "skipped on MPS"},
                    "num_references": len(references),
                    "num_translated": len(hypotheses),
                    "duration_seconds": round(_time.monotonic() - q_start, 1),
                }
                print(f"    BERTScore: {quality['bertscore']['system_score']}  "
                      f"(COMET skipped — hangs on MPS)")
            except Exception as e:
                print(f"    Quality benchmark error (non-fatal): {e}")
                quality = {"error": str(e)}

        # ── Assemble result ──
        result = {
            "model": name,
            "model_path": path,
            "type": backend_type,
            "tags": model_def.get("tags", []),
            "environment": {
                "backend": device_info.backend,
                "device": device_info.name,
                "pytorch": env_snapshot.get("pytorch_version", "?"),
                "python": env_snapshot.get("python_version", "?"),
            },
            "speed": {
                "mean_tps": round(mean_tps, 1),
                "median_tps": round(median_tps, 1),
                "p95_tps": round(p95_tps, 1),
                "std_tps": round(std_tps, 1),
                "mean_latency_ms": round(mean_latency_ms, 1),
                "batches_completed": batches_completed,
                "total_tokens_translated": total_tokens,
                "run_duration_seconds": round(run_duration, 1),
                "load_seconds": round(load_secs, 1),
                "batch_size": batch_size,
            },
            "quality": {
                "bertscore": quality.get("bertscore", {}).get("system_score"),
                "comet": quality.get("comet", {}).get("system_score"),
                "comet_kiwi": quality.get("comet_kiwi", {}).get("system_score"),
                "num_references": quality.get("num_references", 0),
                "num_translated": quality.get("num_translated", 0),
            },
            "device": {
                "mean_gpu_util_pct": device_stats.get("mean_util_pct"),
                "data_starvation_pct": device_stats.get("data_starvation_pct"),
            },
            "run_dir": str(run_dir),
        }

        return result

    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {
            "model": name,
            "model_path": path,
            "type": backend_type,
            "tags": model_def.get("tags", []),
            "error": str(e),
            "status": "failed",
        }

    finally:
        # ── Free model + clean GPU memory ──
        try:
            del engine
        except Exception:
            pass
        gc.collect()
        if torch.backends.mps.is_available():
            try:
                torch.mps.synchronize()
                torch.mps.empty_cache()
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(1)


def print_summary_table(results: list[dict]) -> None:
    """Print a formatted comparison table."""
    print(f"\n{'='*100}")
    header = (
        f"{'MODEL':<30} {'TPS':>8} {'Lat(ms)':>9} {'Batch':>7} "
        f"{'BERTScore':>10} {'COMET':>8} {'Load(s)':>8} {'Status'}"
    )
    print(header)
    print(f"{'─'*100}")

    for r in results:
        name = r["model"][:29]
        if r.get("error"):
            err_line = (f"{name:<30} {'—':>8} {'—':>9} {'—':>7} "
                        f"{'—':>10} {'—':>8} {'—':>8} ✗ {r['error'][:40]}")
            print(err_line)
            continue

        s = r["speed"]
        q = r["quality"]
        line = (f"{name:<30} {s['mean_tps']:>8.1f} {s['mean_latency_ms']:>9.0f} "
                f"{s['batch_size']:>7} "
                f"{q.get('bertscore') or 0:>10.4f} "
                f"{q.get('comet') or 0:>8.4f} "
                f"{s['load_seconds']:>8.1f}   ✓")
        print(line)

    print(f"{'─'*100}")

    # ── Summary stats ──
    ok = [r for r in results if not r.get("error")]
    if ok:
        tps_vals = [r["speed"]["mean_tps"] for r in ok]
        bs_vals = [r["quality"].get("bertscore") for r in ok if r["quality"].get("bertscore")]
        cm_vals = [r["quality"].get("comet") for r in ok if r["quality"].get("comet")]
        print(f"\n  Best TPS:        {max(tps_vals):.0f} tok/s  ({ok[tps_vals.index(max(tps_vals))]['model']})")
        if bs_vals:
            print(f"  Best BERTScore:  {max(bs_vals):.4f}     ({ok[bs_vals.index(max(bs_vals))]['model']})")
        if cm_vals:
            print(f"  Best COMET-22:   {max(cm_vals):.4f}     ({ok[cm_vals.index(max(cm_vals))]['model']})")


def main():
    print("=" * 100)
    print("  Turkish Corpus Translation — Comparative Model Benchmark")
    print(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Platform: MPS (Apple Silicon)")
    print(f"  Duration per model: {RUN_DURATION}s")
    print(f"  Quality refs: {QUALITY_MAX_REFS}")
    print("=" * 100)

    # ── Detect backend ──
    device_info = detect_backend("mps")
    print(f"\nBackend: {device_info.backend} ({device_info.name})")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"PyTorch: {torch.__version__}")

    env_snapshot = get_environment_snapshot()

    # ── Verify input data ──
    if not Path(INPUT_GLOB).exists() and not any(Path("data/input").glob("*.jsonl*")):
        print("\n✗ No input data found. Place a .jsonl or .jsonl.gz file in data/input/")
        sys.exit(1)

    if not Path(REFERENCE_SET).exists():
        print("\n⚠ Reference file not found — quality metrics will be skipped.")

    # ── Run each model ──
    results: list[dict] = []
    total_start = time.monotonic()

    for i, model_def in enumerate(MODELS, 1):
        r = run_one(model_def, device_info, env_snapshot, i)
        if r:
            results.append(r)

    total_elapsed = time.monotonic() - total_start
    print(f"\nTotal benchmark time: {total_elapsed/60:.1f} min ({total_elapsed:.0f}s)")

    # ── Print summary ──
    print_summary_table(results)

    # ── Write results file ──
    output_path = Path(OUTPUT_DIR) / "model_comparison.json"
    report = {
        "title": "Turkish Corpus Translation — Model Comparison (MPS)",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "platform": {
            "backend": "mps",
            "device": device_info.name,
            "pytorch": env_snapshot.get("pytorch_version", "?"),
        },
        "config": {
            "run_duration_per_model_s": RUN_DURATION,
            "quality_max_references": QUALITY_MAX_REFS,
            "seed": SEED,
        },
        "models_tested": len(results),
        "total_wallclock_s": round(total_elapsed, 1),
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results written to: {output_path}")
    print(f"  ({len([r for r in results if not r.get('error')])}/{len(results)} models succeeded)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
