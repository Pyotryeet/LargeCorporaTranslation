#!/usr/bin/env python3
"""Benchmark model throughput on JSONL vs Parquet input.

Creates a temp Parquet file from the existing JSONL, then runs the model
against both formats and reports TPS for comparison.

Usage:
    python scripts/bench_format.py --model translategemma-4b-bf16 --docs 1000 --batch-size 16
    python scripts/bench_format.py --model nllb-600m --docs 500 --batch-size 32
"""
import argparse, json, os, sys, tempfile, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Parse args ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Benchmark JSONL vs Parquet throughput")
parser.add_argument("--model", default="translategemma-4b-bf16",
                    choices=["translategemma-4b-bf16", "ministral-3b-bf16",
                             "nllb-600m", "nllb-1.3b", "smollm2-1.7b"])
parser.add_argument("--docs", type=int, default=1000, help="Documents to translate")
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--warmup-batches", type=int, default=5)
parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
args = parser.parse_args()

# ── Import project modules (after arg parse so we can fail fast) ──────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from benchmark.hardware.backend import detect_backend
from benchmark.inference.engine import InferenceEngine
from benchmark.inference.sampling import DecodingParams
from benchmark.data.pipeline import AsyncPipeline
from benchmark.data.loader import JSONLLoader
from benchmark.data.chunker import TextChunker
from benchmark.data.filters import ChunkFilter

# ── Detect device ─────────────────────────────────────────────────────────
device_info = detect_backend("auto")
is_cuda = device_info.backend == "cuda"
print(f"Device: {device_info.backend} ({device_info.name}), GPUs: {device_info.num_devices}")

# ── Model mapping ─────────────────────────────────────────────────────────
MODELS = {
    "translategemma-4b-bf16": ("google/translategemma-4b-it", "auto"),
    "ministral-3b-bf16":     ("mistralai/Ministral-3-3B-Instruct-2512", "auto"),
    "nllb-600m":             ("facebook/nllb-200-distilled-600M", "encoder_decoder"),
    "nllb-1.3b":             ("facebook/nllb-200-distilled-1.3B", "encoder_decoder"),
    "smollm2-1.7b":          ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "auto"),
}
model_path, backend_type = MODELS[args.model]
print(f"Model: {model_path}  ({backend_type})")
print(f"Docs: {args.docs}  Batch: {args.batch_size}  Compile: {not args.no_compile}")
print()

# ── Create Parquet temp file from JSONL ────────────────────────────────────
JSONL_INPUT = str(PROJECT_ROOT / "data" / "input" / "fineweb_en_sample.jsonl.gz")
if not Path(JSONL_INPUT).exists():
    sys.exit(f"Input not found: {JSONL_INPUT}")

print("Creating Parquet test file ...")
t0 = time.monotonic()
texts = []
import gzip as _gz
with _gz.open(JSONL_INPUT, "rt", encoding="utf-8", errors="replace") as fh:
    for i, line in enumerate(fh):
        if i >= args.docs:
            break
        try:
            obj = json.loads(line)
            t = obj.get("text", "").strip()
            if t:
                texts.append(t)
        except json.JSONDecodeError:
            pass

# Match the loader's column name
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    sys.exit("pyarrow not installed — pip install pyarrow")

parquet_fd, parquet_path = tempfile.mkstemp(suffix=".parquet", prefix="bench_")
os.close(parquet_fd)
table = pa.table({"text": texts})
pq.write_table(table, parquet_path, compression="snappy")
print(f"  Parquet: {parquet_path}  ({len(texts)} docs, {os.path.getsize(parquet_path)/1024:.0f} KB)")
print(f"  Created in {time.monotonic() - t0:.1f}s")
print()

# ── Benchmark helper ───────────────────────────────────────────────────────
def run_bench(label, input_path, engine, n_docs, batch_size):
    loader = JSONLLoader([input_path], shuffle=False)
    tokenizer = engine.tokenizer
    chunker = TextChunker(tokenizer, max_input_tokens=512, overlap_tokens=50)
    filt = ChunkFilter(min_tokens=10, max_garbage_ratio=0.95)
    pipeline = AsyncPipeline(
        loader, chunker, tokenizer, filt,
        batch_size=batch_size, prefetch_workers=min(4, os.cpu_count() or 4),
        backend="cuda" if is_cuda else "cpu",
    )

    pipeline.start_prefetch()
    batches = 0
    total_tokens = 0
    batch_times = []
    drained = False

    try:
        while batches < (n_docs // batch_size) + 20:
            t_start = time.monotonic()
            batch = pipeline.next_batch()
            if batch is None:
                if pipeline.draining():
                    drained = True
                    break
                continue
            result = engine.translate(batch)
            pipeline.release_batch(batch)
            t_end = time.monotonic()
            batch_times.append(t_end - t_start)
            total_tokens += result.output_tokens_total
            batches += 1
    finally:
        pipeline.stop_prefetch()

    if not batch_times:
        return {"label": label, "batches": 0, "tokens": 0, "mean_tps": 0, "error": "no batches"}

    # Skip warmup batches for TPS calculation
    warmup = min(args.warmup_batches, len(batch_times) - 1)
    prod_times = batch_times[warmup:]
    prod_tokens = sum(
        result.output_tokens_total
        for t in batch_times[warmup:]
    ) if hasattr(batch_times[0], '__len__') else total_tokens * len(prod_times) / len(batch_times)

    mean_tps = total_tokens / sum(batch_times) if sum(batch_times) > 0 else 0
    mean_latency = sum(prod_times) / len(prod_times) * 1000 if prod_times else 0

    return {
        "label": label, "batches": batches, "drained": drained,
        "tokens": total_tokens, "mean_tps": round(mean_tps, 1),
        "mean_latency_ms": round(mean_latency, 1),
    }

# ── Load model once ────────────────────────────────────────────────────────
engine = InferenceEngine(
    model_path=model_path, tokenizer_path="",
    device_info=device_info,
    decoding_params=DecodingParams(max_new_tokens=128, temperature=0.0),
    use_flash_attention=is_cuda,
    use_torch_compile=is_cuda and not args.no_compile,
    max_input_tokens=512,
    backend_type=backend_type,
)
engine.load()
engine.warmup(batches=args.warmup_batches)
engine._configured_batch_size = args.batch_size
print(f"Model loaded. Starting benchmarks ...\n")

# ── Run both formats ──────────────────────────────────────────────────────
print("═" * 60)
results_jsonl = run_bench("JSONL (.jsonl.gz)", JSONL_INPUT, engine, args.docs, args.batch_size)
print(f"JSONL: {results_jsonl['batches']} batches, {results_jsonl['tokens']} tok, "
      f"{results_jsonl['mean_tps']} tps, {results_jsonl['mean_latency_ms']}ms/batch")

results_parquet = run_bench("Parquet", parquet_path, engine, args.docs, args.batch_size)
print(f"Parquet: {results_parquet['batches']} batches, {results_parquet['tokens']} tok, "
      f"{results_parquet['mean_tps']} tps, {results_parquet['mean_latency_ms']}ms/batch")

# ── Report ─────────────────────────────────────────────────────────────────
print()
print("═" * 60)
print("RESULTS")
print("═" * 60)
print(f"  {'Format':<20} {'Batches':>8} {'Tokens':>10} {'TPS':>10} {'Latency':>10}")
print(f"  {'─'*20} {'─'*8} {'─'*10} {'─'*10} {'─'*10}")
for r in [results_jsonl, results_parquet]:
    print(f"  {r['label']:<20} {r['batches']:>8} {r['tokens']:>10} {r['mean_tps']:>9.1f} {r['mean_latency_ms']:>9.1f}ms")

if results_jsonl["mean_tps"] and results_parquet["mean_tps"]:
    delta = (results_parquet["mean_tps"] - results_jsonl["mean_tps"]) / results_jsonl["mean_tps"] * 100
    print(f"\n  Parquet vs JSONL: {delta:+.1f}% TPS difference")
    if abs(delta) < 5:
        print("  → Within noise — formats are equivalent throughput (expected: bottleneck is GPU, not I/O)")
    elif delta > 0:
        print("  → Parquet faster (row-group streaming vs line-at-a-time gzip)")
    else:
        print("  → JSONL faster (unexpected — check compression settings)")

print(f"\n  Drained: JSONL={results_jsonl.get('drained')} Parquet={results_parquet.get('drained')}")

# Cleanup
engine.close() if hasattr(engine, 'close') else None
os.unlink(parquet_path)
print(f"\nCleaned up {parquet_path}")
