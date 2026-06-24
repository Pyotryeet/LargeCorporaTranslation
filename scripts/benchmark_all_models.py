#!/usr/bin/env python3
"""Comprehensive EN→TR translation benchmark — all backends, all models, MPS.

Runs every available model through the same 120s translation window + quality
benchmark and writes results to data/output/model_comparison.json.

Backends covered:
  - encoder_decoder (NLLB family via NLLBBackend)
  - autoregressive (TranslateGemma, SmolLM2 via AutoregressiveBackend)
  - diffusion (DiffusionGemma 26B-A4B via llama.cpp subprocess)
  - gguf (QAT E2B/E4B via llama.cpp subprocess)

Usage:
  source .venv/bin/activate
  python scripts/benchmark_all_models.py
"""

from __future__ import annotations

import json, logging, os, subprocess, sys, time, gc
from datetime import datetime, timezone
from pathlib import Path

import torch, warnings
warnings.filterwarnings("ignore", message=".*deprecated.*")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ["PYTHONUNBUFFERED"] = "1"

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
from benchmark.quality.references import ReferenceLoader
from benchmark.quality.metrics_bertscore import compute_bertscore
from benchmark.reporting.aggregator import MetricsAggregator
from benchmark.orchestration.signals import SignalHandler
from benchmark.utils.logging_setup import setup_logging
from benchmark.utils.version import get_environment_snapshot
from benchmark.utils.timer import PrecisionTimer

# ═════════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════════

INPUT_GLOB = "data/input/fineweb_en_sample.jsonl.gz"
REFERENCE_SET = "data/references/golden_en_tr.jsonl"
OUTPUT_DIR = "data/output"
RUN_DURATION = 120
QUALITY_MAX_REFS = 32
MPS_MAX_BATCH = 4
SEED = 42
DEVICE = "mps"

# llama.cpp paths
LLAMA_BUILD = Path.home() / "Documents/ComputerScience/Projects/llama/llama.cpp/build"
LLAMA_CLI = LLAMA_BUILD / "bin/llama-cli"
LLAMA_DIFFUSION_CLI = LLAMA_BUILD / "bin/llama-diffusion-cli"
LLAMA_MODELS = Path.home() / "Documents/ComputerScience/Projects/llama/models"

# ═════════════════════════════════════════════════════════════════════════════
# Model registry — every model that works on this machine
# ═════════════════════════════════════════════════════════════════════════════

MODELS = [
    # ── NLLB family (encoder-decoder) — already cached ────────────────────
    {
        "name": "NLLB-200-distilled-600M",
        "path": "facebook/nllb-200-distilled-600M",
        "backend": "nllb_python",
        "extra": {"nllb_source_lang": "eng_Latn", "nllb_target_lang": "tur_Latn", "num_beams": 1},
        "tags": ["nllb", "600M", "encoder-decoder"],
    },
    {
        "name": "NLLB-200-distilled-1.3B",
        "path": "facebook/nllb-200-distilled-1.3B",
        "backend": "nllb_python",
        "extra": {"nllb_source_lang": "eng_Latn", "nllb_target_lang": "tur_Latn", "num_beams": 1},
        "tags": ["nllb", "1.3B", "encoder-decoder"],
    },
    # NLLB-3.3B: needs 13GB download — skip until cached
    # ── Autoregressive (Transformers) — already cached ────────────────────
    {
        "name": "SmolLM2-1.7B-Instruct",
        "path": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "backend": "ar_python",
        "tags": ["smollm", "1.7B"],
    },
    {
        "name": "TranslateGemma-4B",
        "path": "google/translategemma-4b-it",
        "backend": "ar_python",
        "tags": ["gemma", "4B", "translator"],
    },
    # ── DiffusionGemma (llama.cpp) — 25GB GGUF already on disk ───────────
    {
        "name": "DiffusionGemma-26B-A4B-Q8_0",
        "path": str(LLAMA_MODELS / "diffusiongemma-26B-A4B-it-Q8_0.gguf"),
        "backend": "llama_diffusion",
        "llama_args": {
            "ngl": "all",
            "diffusion_steps": 64,
            "diffusion_algorithm": 4,
            "ctx_size": 4096,
            "n_predict": 256,
            "temp": 0.8,
        },
        "tags": ["diffusion", "26B-MoE", "Q8_0"],
    },
    # ── QAT models (llama.cpp GGUF) — download on demand ──────────────────
    {
        "name": "Gemma-4-E2B-QAT-UD-Q4_K_XL",
        "path": str(LLAMA_MODELS / "gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf"),
        "backend": "llama_text",
        "llama_args": {"ngl": "all", "ctx_size": 4096, "n_predict": 256, "temp": 0.0},
        "hf_dl": "unsloth/gemma-4-E2B-it-qat-GGUF:gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf",
        "tags": ["gemma", "E2B", "QAT", "GGUF"],
    },
    {
        "name": "Gemma-4-E4B-QAT-UD-Q4_K_XL",
        "path": str(LLAMA_MODELS / "gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"),
        "backend": "llama_text",
        "llama_args": {"ngl": "all", "ctx_size": 4096, "n_predict": 256, "temp": 0.0},
        "hf_dl": "unsloth/gemma-4-E4B-it-qat-GGUF:gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf",
        "tags": ["gemma", "E4B", "QAT", "GGUF"],
    },
]


# ═════════════════════════════════════════════════════════════════════════════
# Backend runners
# ═════════════════════════════════════════════════════════════════════════════

def run_via_nllb_backend(model_def, device_info, env_snapshot, run_idx, run_dir):
    """Run NLLB encoder-decoder model via existing Python backend."""
    extra = dict(model_def.get("extra", {}))
    is_cuda = device_info.backend == "cuda"
    engine = InferenceEngine(
        model_path=model_def["path"], tokenizer_path="",
        device_info=device_info,
        decoding_params=DecodingParams(max_new_tokens=128, temperature=0.0),
        use_flash_attention=is_cuda, use_torch_compile=is_cuda,
        max_input_tokens=128, backend_type="encoder_decoder", extra=extra,
    )
    engine.load()
    return run_python_translate_loop(engine, model_def, device_info, run_idx, run_dir)


def run_via_ar_backend(model_def, device_info, env_snapshot, run_idx, run_dir):
    """Run autoregressive model via existing Python backend."""
    is_cuda = device_info.backend == "cuda"
    engine = InferenceEngine(
        model_path=model_def["path"], tokenizer_path="",
        device_info=device_info,
        decoding_params=DecodingParams(max_new_tokens=128, temperature=0.0),
        use_flash_attention=is_cuda, use_torch_compile=is_cuda,
        max_input_tokens=128, backend_type="auto",
    )
    engine.load()
    return run_python_translate_loop(engine, model_def, device_info, run_idx, run_dir)


def run_via_llama(model_def, device_info, env_snapshot, run_idx, run_dir):
    """Run a GGUF model via llama.cpp CLI subprocess."""
    model_path = Path(model_def["path"])
    binary = LLAMA_DIFFUSION_CLI if model_def["backend"] == "llama_diffusion" else LLAMA_CLI

    if not binary.exists():
        return {"error": f"Binary not found: {binary}", "status": "missing_binary"}
    if not model_path.exists():
        # Try downloading
        hf_dl = model_def.get("hf_dl", "")
        if hf_dl:
            print(f"    Downloading GGUF: {hf_dl}...")
            try:
                from huggingface_hub import hf_hub_download
                repo, filename = hf_dl.split(":", 1)
                downloaded = hf_hub_download(repo_id=repo, filename=filename,
                                              local_dir=str(LLAMA_MODELS))
                model_path = Path(downloaded)
                print(f"    Downloaded to: {model_path}")
            except Exception as e:
                return {"error": f"Download failed: {e}", "status": "download_failed"}
        else:
            return {"error": f"Model not found: {model_path}", "status": "missing_model"}

    args = model_def.get("llama_args", {})
    ngl = args.get("ngl", "all")
    diff_steps = args.get("diffusion_steps", 64)
    diff_algo = args.get("diffusion_algorithm", 4)
    ctx = args.get("ctx_size", 4096)
    n_predict = args.get("n_predict", 256)
    temp = args.get("temp", 0.8)

    print(f"    Model: {model_path.name}")
    print(f"    GPU layers: {ngl}, steps: {diff_steps}, ctx: {ctx}")

    # Load reference texts for translation benchmark
    loader = ReferenceLoader(REFERENCE_SET)
    sources, references = loader.load()
    if QUALITY_MAX_REFS and QUALITY_MAX_REFS < len(sources):
        sources = sources[:QUALITY_MAX_REFS]
        references = references[:QUALITY_MAX_REFS]

    # Build translation prompts
    prompt_prefix = "Translate English to Turkish. Output only the Turkish translation, nothing else.\n\nEnglish: "
    prompts = [f"{prompt_prefix}{s}\nTurkish:" for s in sources]
    # Only benchmark a subset to keep runtime reasonable
    MAX_LLAMA_PROMPTS = 16
    prompts = prompts[:MAX_LLAMA_PROMPTS]
    sources = sources[:MAX_LLAMA_PROMPTS]
    references = references[:MAX_LLAMA_PROMPTS]

    total_tokens = 0
    batches = 0
    hypotheses = []
    wall_start = time.monotonic()

    for i, prompt in enumerate(prompts):
        if time.monotonic() - wall_start > RUN_DURATION:
            break

        # Sanitize: enforce max prompt length to prevent argument overflow.
        MAX_PROMPT_BYTES = 8192
        if len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
            prompt = prompt.encode("utf-8")[:MAX_PROMPT_BYTES].decode("utf-8", errors="replace")
            print(f"    ⚠ Prompt {i} truncated to {MAX_PROMPT_BYTES} bytes")
        # Reject null bytes — they cannot travel through argv safely.
        if "\0" in prompt:
            print(f"    ⚠ Prompt {i} contains null byte — skipping")
            hypotheses.append("")
            continue

        t0 = time.monotonic()
        try:
            is_diffusion = model_def["backend"] == "llama_diffusion"

            cmd = [
                str(binary),
                "-m", str(model_path),
                "-ngl", str(ngl),
            ]
            if is_diffusion:
                cmd += [
                    f"--diffusion-steps", str(diff_steps),
                    f"--diffusion-algorithm", str(diff_algo),
                ]
            cmd += [
                "-c", str(ctx),
                "-n", str(n_predict),
                "--temp", str(temp),
                "-t", "8",
                "-b", "2048",
                "-s", str(SEED + i),
                "--no-conversation",
                "--log-verbosity", "1",
                "-p", prompt,
            ]

            # CRITICAL: pass explicit env dict — never leak os.environ to subprocess.
            # Only whitelist vars needed by llama.cpp (Metal, MPS, DYLD, PATH).
            _env = {}
            for _k in ("PATH", "HOME", "USER", "SHELL", "TMPDIR", "TMP",
                       "GGML_METAL_PATH_RESOURCES",
                       "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH",
                       "METAL_DEVICE_WRAPPER_TYPE", "MTL_DEBUG_LAYER",
                       "MTL_SHADER_VALIDATION", "MPS_DEBUG", "MPS_ENABLE_POOL",
                       "PYTHONUNBUFFERED", "TRANSFORMERS_VERBOSITY",
                       "LOGNAME", "LANG", "LC_ALL"):
                if _k in os.environ:
                    _env[_k] = os.environ[_k]
            _env["GGML_METAL_PATH_RESOURCES"] = str(LLAMA_BUILD / "bin")
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=120, cwd=str(LLAMA_BUILD.parent),  # 2-min per-prompt cap
                env=_env,
            )
            output = result.stdout.strip()
            # Extract Turkish translation — strip prompt echo if present
            if "Turkish:" in output:
                output = output.split("Turkish:")[-1].strip()
            if output.startswith(prompt):
                output = output[len(prompt):].strip()

            hypotheses.append(output)
            out_toks = len(output.split()) * 1.3  # rough token estimate
            total_tokens += int(out_toks)
            batches += 1

            elapsed = time.monotonic() - t0
            tps = out_toks / elapsed if elapsed > 0 else 0
            print(f"    [{time.monotonic() - wall_start:5.0f}s] batch={batches} "
                  f"tokens≈{int(out_toks)} tps≈{tps:.0f}", flush=True)

        except subprocess.TimeoutExpired:
            print(f"    ⚠ Prompt {i} timed out — skipping")
            hypotheses.append("")
        except Exception as e:
            print(f"    ⚠ Prompt {i} error: {e}")
            hypotheses.append("")

    wall_end = time.monotonic()
    run_duration = wall_end - wall_start

    # ── BERTScore ──
    quality = {}
    try:
        if hypotheses and any(h for h in hypotheses):
            bs = compute_bertscore(references[:len(hypotheses)], hypotheses)
            quality = {
                "bertscore": bs.get("system_score"),
                "comet": None, "comet_kiwi": None,
                "num_references": len(references[:len(hypotheses)]),
                "num_translated": len([h for h in hypotheses if h]),
            }
            print(f"    BERTScore: {quality['bertscore']}")
    except Exception as e:
        print(f"    Quality error: {e}")

    return {
        "speed": {
            "mean_tps": round(total_tokens / max(run_duration, 0.1), 1),
            "batches_completed": batches,
            "total_tokens_translated": total_tokens,
            "run_duration_seconds": round(run_duration, 1),
        },
        "quality": quality,
    }


def run_python_translate_loop(engine, model_def, device_info, run_idx, run_dir):
    """Shared translation loop for NLLB and AR backends."""
    # Batch tune
    tuner = BatchSizeTuner()
    batch_size = tuner.tune(engine.model, engine.tokenizer,
                            device_info.device, device_info.backend, 128)
    batch_size = min(batch_size, MPS_MAX_BATCH)
    print(f"    Batch size: {batch_size} (clamped to {MPS_MAX_BATCH} for MPS)")

    engine.warmup(batches=10)

    loader = JSONLLoader([INPUT_GLOB], shuffle=True, seed=SEED)
    chunker = TextChunker(engine.tokenizer, 128, 50)
    filt = ChunkFilter(min_tokens=10, max_garbage_ratio=0.95)
    pipeline = AsyncPipeline(loader, chunker, engine.tokenizer, filt,
                             batch_size=batch_size, prefetch_workers=4,
                             backend=device_info.backend)
    metrics = MetricsCollector(run_dir / "metrics", device_info, 1)
    pipeline.start_prefetch()
    timer = PrecisionTimer(); timer.start()
    metrics.start(timer.start_time())

    batches_completed = 0; total_tokens = 0
    signals = SignalHandler()

    try:
        while timer.elapsed() < RUN_DURATION:
            if signals.killed.is_set(): break
            batch = pipeline.next_batch()
            if batch is None:
                if pipeline.draining(): break
                continue
            result = engine.translate(batch)
            pipeline.release_batch(batch)
            metrics.log_batch(result)
            batches_completed += 1
            total_tokens += result.output_tokens_total
            tps = metrics.get_rolling_throughput()
            print(f"    [{timer.elapsed():5.0f}s] batch={batches_completed} "
                  f"tokens={total_tokens:,} tps={tps:.0f}", flush=True)
            if timer.elapsed() > RUN_DURATION: break
    finally:
        metrics.stop(); pipeline.stop_prefetch()
        run_duration = timer.elapsed()

    aggregator = MetricsAggregator(run_dir / "metrics")
    ms = aggregator.aggregate()
    bs = ms.get("batch", {})

    # Quality
    quality = {}
    if Path(REFERENCE_SET).exists():
        print(f"    Running quality benchmark (BERTScore only, max {QUALITY_MAX_REFS} refs)...")
        try:
            loader2 = ReferenceLoader(REFERENCE_SET)
            sources, references = loader2.load()
            if QUALITY_MAX_REFS and QUALITY_MAX_REFS < len(sources):
                sources = sources[:QUALITY_MAX_REFS]
                references = references[:QUALITY_MAX_REFS]
            from benchmark.quality.benchmark import _build_batch
            _MiniBatch = type("_MiniBatch", (), {})
            iids, amask, _ = _build_batch(sources, engine.tokenizer, engine.devices[0])
            mb = _MiniBatch(); mb.input_ids = iids; mb.attention_mask = amask
            mb.raw_texts = sources; mb.batch_id = 0
            t_result = engine.translate(mb)
            hyps = [g.translated_text for g in t_result.generations]
            bs_r = compute_bertscore(references, hyps)
            quality = {"bertscore": bs_r.get("system_score"), "comet": None,
                       "comet_kiwi": None, "num_references": len(references),
                       "num_translated": len(hyps)}
            print(f"    BERTScore: {quality['bertscore']}")
        except Exception as e:
            print(f"    Quality error (non-fatal): {e}")

    return {
        "speed": {
            "mean_tps": bs.get("mean_tps", 0),
            "median_tps": bs.get("median_tps", 0),
            "p95_tps": bs.get("p95_tps", 0),
            "mean_latency_ms": bs.get("mean_latency_ms", 0),
            "batches_completed": batches_completed,
            "total_tokens_translated": total_tokens,
            "run_duration_seconds": round(run_duration, 1),
            "batch_size": batch_size,
        },
        "quality": quality,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 100)
    print("  Turkish Corpus Translation — Comprehensive Model Benchmark")
    print(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Platform: MPS (Apple Silicon)")
    print(f"  Duration per model: {RUN_DURATION}s")
    print("=" * 100)

    device_info = detect_backend("mps")
    print(f"\nBackend: {device_info.backend} ({device_info.name})")
    print(f"MPS: {torch.backends.mps.is_available()} | PyTorch: {torch.__version__}")
    env_snapshot = get_environment_snapshot()

    if not Path(INPUT_GLOB).exists():
        print("✗ No input data found.")
        sys.exit(1)

    results = []
    total_start = time.monotonic()

    for i, model_def in enumerate(MODELS, 1):
        name = model_def["name"]
        be = model_def["backend"]
        print(f"\n{'='*70}")
        print(f"[{i}/{len(MODELS)}] {name} ({be})")
        print(f"{'='*70}")

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = Path(OUTPUT_DIR) / f"bm_{name.replace('-','_').replace(' ','_')}_{i:02d}_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(run_dir)

        load_start = time.monotonic()
        try:
            if be == "nllb_python":
                r = run_via_nllb_backend(model_def, device_info, env_snapshot, i, run_dir)
            elif be == "ar_python":
                r = run_via_ar_backend(model_def, device_info, env_snapshot, i, run_dir)
            elif be in ("llama_text", "llama_diffusion"):
                r = run_via_llama(model_def, device_info, env_snapshot, i, run_dir)
            else:
                r = {"error": f"Unknown backend: {be}", "status": "unknown_backend"}

            r["model"] = name
            r["model_path"] = model_def["path"]
            r["type"] = be
            r["tags"] = model_def.get("tags", [])
            r["load_seconds"] = round(time.monotonic() - load_start, 1)
            r["environment"] = {"backend": device_info.backend, "device": device_info.name,
                                "pytorch": env_snapshot.get("pytorch_version", "?"),
                                "python": env_snapshot.get("python_version", "?")}
            r["run_dir"] = str(run_dir)
            results.append(r)

        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({"model": name, "model_path": model_def["path"],
                            "type": be, "tags": model_def.get("tags", []),
                            "error": str(e), "status": "failed"})

        # ── Cleanup ──
        gc.collect()
        if torch.backends.mps.is_available():
            try: torch.mps.empty_cache()
            except: pass
        time.sleep(1)

    total_elapsed = time.monotonic() - total_start
    print(f"\nTotal benchmark time: {total_elapsed/60:.1f} min ({total_elapsed:.0f}s)")

    # ── Summary table ──
    print(f"\n{'='*100}")
    header = (f"{'MODEL':<35} {'TPS':>8} {'Lat(ms)':>9} {'BATCH':>6} "
              f"{'BERTScore':>10} {'Load(s)':>8} {'Status'}")
    print(header)
    print(f"{'─'*100}")
    for r in results:
        n = r["model"][:34]
        if r.get("error"):
            line = (f"{n:<35} {'—':>8} {'—':>9} {'—':>6} {'—':>10} "
                    f"{r.get('load_seconds',0):>8.1f} ✗ {r['error'][:35]}")
            print(line)
            continue
        s = r.get("speed", {})
        q = r.get("quality", {})
        line = (f"{n:<35} {s.get('mean_tps',0):>8.1f} "
                f"{s.get('mean_latency_ms',0):>9.0f} {s.get('batch_size','?'):>6} "
                f"{q.get('bertscore') or 0:>10.4f} "
                f"{r.get('load_seconds',0):>8.1f}   ✓")
        print(line)
    print(f"{'─'*100}")

    # Best
    ok = [r for r in results if not r.get("error")]
    if ok:
        tps = [(r["speed"]["mean_tps"], r["model"]) for r in ok]
        bs = [(r["quality"].get("bertscore") or 0, r["model"]) for r in ok
              if r["quality"].get("bertscore")]
        if tps:
            best_tps = max(tps, key=lambda x: x[0])
            print(f"  Best TPS:       {best_tps[0]:.0f} tok/s  ({best_tps[1]})")
        if bs:
            best_bs = max(bs, key=lambda x: x[0])
            print(f"  Best BERTScore: {best_bs[0]:.4f}     ({best_bs[1]})")

    # ── Write JSON ──
    output_path = Path(OUTPUT_DIR) / "model_comparison.json"
    report = {
        "title": "Turkish Corpus Translation — Model Comparison (MPS)",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "platform": {"backend": "mps", "device": device_info.name,
                     "pytorch": env_snapshot.get("pytorch_version", "?")},
        "config": {"run_duration_per_model_s": RUN_DURATION,
                   "quality_max_references": QUALITY_MAX_REFS, "seed": SEED},
        "models_tested": len(results),
        "total_wallclock_s": round(total_elapsed, 1),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    ok_count = len([r for r in results if not r.get("error")])
    print(f"\n✓ Results written to: {output_path}  ({ok_count}/{len(results)} succeeded)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
