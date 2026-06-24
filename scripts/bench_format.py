#!/usr/bin/env python3
"""Benchmark current JSONL ingestion against pretokenized Parquet.

This script compares the existing runtime path:

  JSONL/JSONL.gz -> chunk -> prompt -> tokenize -> pad -> translate

against the pretokenized fast path:

  pretokenized Parquet -> pad -> translate

The pretokenized cache is created once per model and then reused on later runs.
Shuffle is disabled for the dynamic JSONL path so both modes process the same
chunks in the same order.

Examples
--------
  python scripts/bench_format.py --config config.yaml
  python scripts/bench_format.py --config config.yaml --model translategemma-4b-bf16
  python scripts/bench_format.py --config config.yaml --model facebook/nllb-200-distilled-600M --backend-type encoder_decoder
  python scripts/bench_format.py --config config.yaml --prepare-only
"""

from __future__ import annotations

import argparse
import json
import logging as _logging
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoTokenizer

from benchmark.config.model_presets import get_preset_by_name
from benchmark.config.schema import load_config
from benchmark.data.chunker import TextChunker
from benchmark.data.filters import ChunkFilter
from benchmark.data.loader import JSONLLoader
from benchmark.data.pipeline import AsyncPipeline
from benchmark.data.pretokenizer import PreTokenizedLoader, ensure_pretokenized
from benchmark.hardware.backend import detect_backend
from benchmark.inference.engine import InferenceEngine
from benchmark.inference.sampling import DecodingParams

for _noisy in (
    "httpx",
    "httpcore",
    "urllib3",
    "huggingface_hub.file_download",
    "huggingface_hub.utils._http",
):
    _logging.getLogger(_noisy).setLevel(_logging.WARNING)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark JSONL vs pretokenized Parquet throughput",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Benchmark config to reuse for input paths and chunk settings",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model preset name or HF model ID. Repeat to benchmark multiple models.",
    )
    parser.add_argument(
        "--backend-type",
        choices=["auto", "autoregressive", "encoder_decoder", "diffusion"],
        default=None,
        help="Override backend type for all requested models",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--measure-batches",
        type=int,
        default=50,
        help="Measured batches per mode after warmup skips",
    )
    parser.add_argument(
        "--skip-batches",
        type=int,
        default=5,
        help="Initial batches to ignore per mode",
    )
    parser.add_argument(
        "--engine-warmup-batches",
        type=int,
        default=5,
        help="Warmup batches to run once after model load",
    )
    parser.add_argument(
        "--prefetch-workers",
        type=int,
        default=None,
        help="Override config.data.prefetch_workers for the dynamic JSONL path",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create or refresh the pretokenized cache, then exit",
    )
    parser.add_argument(
        "--force-pretokenize",
        action="store_true",
        help="Rebuild the pretokenized cache even if it already exists",
    )
    parser.add_argument(
        "--pretokenized-cache-dir",
        default=None,
        help="Override the pretokenized cache directory",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write the comparison results as JSON",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile for the comparison run",
    )
    return parser.parse_args()


def _resolve_backend_type(explicit_backend: str | None, config_backend: str, model_path: str) -> str:
    if explicit_backend:
        return explicit_backend
    if config_backend != "auto":
        return config_backend

    model_lower = model_path.lower()
    if "nllb" in model_lower:
        return "encoder_decoder"
    if "diffusiongemma" in model_lower or "llada" in model_lower:
        return "diffusion"
    return "auto"


def _resolve_models(args: argparse.Namespace, config) -> list[dict]:
    requested = args.model or [config.model.model_path]
    resolved: list[dict] = []

    for model_name in requested:
        preset = get_preset_by_name(model_name)
        model_path = preset.hf_model_id if preset else model_name
        quantization = preset.quantization if preset else config.model.quantization
        label = preset.display_name if preset else model_name
        tokenizer_path = (
            config.model.tokenizer_path
            if model_path == config.model.model_path and config.model.tokenizer_path
            else model_path
        )
        resolved.append({
            "label": label,
            "model_name": model_name,
            "model_path": model_path,
            "tokenizer_path": tokenizer_path,
            "quantization": quantization,
            "backend_type": _resolve_backend_type(
                args.backend_type,
                config.model.backend_type,
                model_path,
            ),
        })

    return resolved


def _cache_dir_from_args(args: argparse.Namespace) -> Path | None:
    if args.pretokenized_cache_dir:
        return Path(args.pretokenized_cache_dir)
    return None


def _ensure_cache(config, model_info: dict, tokenizer, cache_dir: Path | None, force: bool):
    return ensure_pretokenized(
        model_path=model_info["model_path"],
        tokenizer=tokenizer,
        max_input_tokens=config.model.max_input_tokens,
        overlap_tokens=config.data.chunk_overlap_tokens,
        min_chunk_tokens=config.data.min_chunk_tokens,
        max_garbage_ratio=config.data.max_garbage_ratio,
        input_paths=config.data.input_paths,
        cache_dir=cache_dir,
        force=force,
    )


def _build_dynamic_pipeline(config, engine, device_info, batch_size: int, prefetch_workers: int) -> AsyncPipeline:
    loader = JSONLLoader(
        config.data.input_paths,
        shuffle=False,
        seed=config.runtime.seed,
        max_shuffle_memory_gb=config.data.shuffle_max_memory_gb,
        shuffle_temp_dir=config.data.shuffle_temp_dir,
    )
    chunker = TextChunker(
        engine.tokenizer,
        config.model.max_input_tokens,
        config.data.chunk_overlap_tokens,
    )
    text_filter = ChunkFilter(
        min_tokens=config.data.min_chunk_tokens,
        max_garbage_ratio=config.data.max_garbage_ratio,
    )
    return AsyncPipeline(
        loader,
        chunker,
        engine.tokenizer,
        text_filter,
        batch_size=batch_size,
        prefetch_workers=prefetch_workers,
        backend=device_info.backend,
    )


def _build_pretokenized_pipeline(
    config,
    engine,
    device_info,
    batch_size: int,
    prefetch_workers: int,
    parquet_path: Path,
) -> AsyncPipeline:
    loader = JSONLLoader(
        config.data.input_paths,
        shuffle=False,
        seed=config.runtime.seed,
        max_shuffle_memory_gb=config.data.shuffle_max_memory_gb,
        shuffle_temp_dir=config.data.shuffle_temp_dir,
    )
    chunker = TextChunker(
        engine.tokenizer,
        config.model.max_input_tokens,
        config.data.chunk_overlap_tokens,
    )
    text_filter = ChunkFilter(
        min_tokens=config.data.min_chunk_tokens,
        max_garbage_ratio=config.data.max_garbage_ratio,
    )
    return AsyncPipeline(
        loader,
        chunker,
        engine.tokenizer,
        text_filter,
        batch_size=batch_size,
        prefetch_workers=prefetch_workers,
        backend=device_info.backend,
        pretokenized_loader=PreTokenizedLoader(parquet_path),
    )


def _run_benchmark(label: str, pipeline: AsyncPipeline, engine, skip_batches: int, measure_batches: int) -> dict:
    measured_batches = 0
    skipped_batches = 0
    output_tokens = 0
    input_tokens = 0
    elapsed_seconds = 0.0

    pipeline.start_prefetch()
    try:
        while measured_batches < measure_batches:
            start = time.monotonic()
            batch = pipeline.next_batch()
            if batch is None:
                if pipeline.draining():
                    break
                continue

            result = engine.translate(batch)
            pipeline.release_batch(batch)
            duration = time.monotonic() - start

            if skipped_batches < skip_batches:
                skipped_batches += 1
                continue

            measured_batches += 1
            elapsed_seconds += duration
            output_tokens += int(result.output_tokens_total)
            input_tokens += int(sum(batch.token_counts))
    finally:
        pipeline.stop_prefetch()

    mean_tps = output_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0
    mean_batch_ms = (elapsed_seconds / measured_batches * 1000.0) if measured_batches > 0 else 0.0

    return {
        "label": label,
        "measured_batches": measured_batches,
        "skipped_batches": skipped_batches,
        "output_tokens": output_tokens,
        "input_tokens": input_tokens,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "mean_tps": round(mean_tps, 1),
        "mean_batch_ms": round(mean_batch_ms, 1),
    }


def _build_engine(config, model_info: dict, device_info, args: argparse.Namespace):
    is_cuda = device_info.backend == "cuda"
    extra = {
        "do_sample": config.model.do_sample,
        "num_beams": config.model.num_beams,
        "backend_type": model_info["backend_type"],
        "diffusion": {
            "num_diffusion_steps": config.model.diffusion_steps,
            "noise_schedule": config.model.noise_schedule,
            "guidance_scale": config.model.guidance_scale,
            "target_length_multiplier": config.model.target_length_multiplier,
        },
        "nllb_source_lang": config.model.nllb_source_lang,
        "nllb_target_lang": config.model.nllb_target_lang,
        "quantization": model_info["quantization"],
    }
    engine = InferenceEngine(
        model_path=model_info["model_path"],
        tokenizer_path=model_info["tokenizer_path"],
        device_info=device_info,
        decoding_params=DecodingParams(
            max_new_tokens=config.model.max_new_tokens,
            temperature=config.model.temperature,
            do_sample=config.model.do_sample,
            num_beams=config.model.num_beams,
        ),
        use_flash_attention=is_cuda and config.model.use_flash_attention,
        use_torch_compile=is_cuda and not args.no_compile,
        max_input_tokens=config.model.max_input_tokens,
        backend_type=model_info["backend_type"],
        extra=extra,
    )
    engine.load()
    engine._configured_batch_size = args.batch_size
    engine.warmup(batches=args.engine_warmup_batches)
    return engine


def _load_tokenizer(model_info: dict):
    return AutoTokenizer.from_pretrained(
        model_info["tokenizer_path"],
        trust_remote_code=False,
    )


def _compare_one_model(config, model_info: dict, args: argparse.Namespace, device_info) -> dict:
    cache_dir = _cache_dir_from_args(args)
    engine = None
    try:
        engine = _build_engine(config, model_info, device_info, args)
        pretok_loader = _ensure_cache(
            config,
            model_info,
            engine.tokenizer,
            cache_dir,
            force=args.force_pretokenize,
        )
        parquet_path = pretok_loader.parquet_path
        prefetch_workers = args.prefetch_workers or config.data.prefetch_workers

        jsonl_result = _run_benchmark(
            "JSONL dynamic",
            _build_dynamic_pipeline(config, engine, device_info, args.batch_size, prefetch_workers),
            engine,
            skip_batches=args.skip_batches,
            measure_batches=args.measure_batches,
        )
        pretok_result = _run_benchmark(
            "Pretokenized Parquet",
            _build_pretokenized_pipeline(
                config,
                engine,
                device_info,
                args.batch_size,
                prefetch_workers,
                parquet_path,
            ),
            engine,
            skip_batches=args.skip_batches,
            measure_batches=args.measure_batches,
        )

        base_tps = jsonl_result["mean_tps"]
        pretok_tps = pretok_result["mean_tps"]
        delta_pct = ((pretok_tps - base_tps) / base_tps * 100.0) if base_tps else 0.0
        return {
            "model": model_info["label"],
            "model_name": model_info["model_name"],
            "model_path": model_info["model_path"],
            "backend_type": model_info["backend_type"],
            "quantization": model_info["quantization"],
            "cache_path": str(parquet_path),
            "cache_chunks": pretok_loader.total_chunks,
            "jsonl": jsonl_result,
            "pretokenized_parquet": pretok_result,
            "delta_tps_pct": round(delta_pct, 1),
        }
    finally:
        if engine is not None and hasattr(engine, "close"):
            try:
                engine.close()
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass


def _prepare_only(config, model_info: dict, args: argparse.Namespace) -> dict:
    tokenizer = _load_tokenizer(model_info)
    pretok_loader = _ensure_cache(
        config,
        model_info,
        tokenizer,
        _cache_dir_from_args(args),
        force=args.force_pretokenize,
    )
    return {
        "model": model_info["label"],
        "model_name": model_info["model_name"],
        "model_path": model_info["model_path"],
        "cache_path": str(pretok_loader.parquet_path),
        "cache_chunks": pretok_loader.total_chunks,
    }


def _print_prepare_result(result: dict) -> None:
    print("=" * 72)
    print(result["model"])
    print("=" * 72)
    print(f"Cache:  {result['cache_path']}")
    print(f"Chunks: {result['cache_chunks']}")


def _print_compare_result(result: dict) -> None:
    print("=" * 72)
    print(result["model"])
    print("=" * 72)
    print(f"Model path: {result['model_path']}")
    print(f"Cache:      {result['cache_path']} ({result['cache_chunks']} chunks)")
    print()
    print(f"  {'Mode':<24} {'Batches':>8} {'Out tok':>12} {'TPS':>10} {'ms/batch':>12}")
    print(f"  {'-' * 24} {'-' * 8} {'-' * 12} {'-' * 10} {'-' * 12}")
    for row in (result["jsonl"], result["pretokenized_parquet"]):
        print(
            f"  {row['label']:<24} {row['measured_batches']:>8} {row['output_tokens']:>12} "
            f"{row['mean_tps']:>10.1f} {row['mean_batch_ms']:>12.1f}"
        )
    print(f"\n  Pretokenized vs JSONL: {result['delta_tps_pct']:+.1f}% TPS")


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    device_info = detect_backend(config.backend)
    models = _resolve_models(args, config)

    print(f"Device: {device_info.backend} ({device_info.name}), GPUs: {device_info.num_devices}")
    print(f"Config: {args.config}")
    print()

    results: list[dict] = []
    for model_info in models:
        try:
            if args.prepare_only:
                result = _prepare_only(config, model_info, args)
                _print_prepare_result(result)
            else:
                result = _compare_one_model(config, model_info, args, device_info)
                _print_compare_result(result)
            results.append(result)
            print()
        except Exception as exc:
            error_result = {
                "model": model_info["label"],
                "model_name": model_info["model_name"],
                "model_path": model_info["model_path"],
                "error": str(exc),
            }
            results.append(error_result)
            print("=" * 72)
            print(model_info["label"])
            print("=" * 72)
            print(f"ERROR: {exc}")
            print()

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump({"results": results}, fh, indent=2)
        print(f"Results written to {output_path}")

    return 0 if all("error" not in result for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
