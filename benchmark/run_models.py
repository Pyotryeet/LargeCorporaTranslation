#!/usr/bin/env python3
"""Comprehensive EN→TR translation benchmark — CUDA + MPS optimizations active.

Runs every available model through the same 120s translation window + quality
benchmark and writes results to data/output/model_comparison.json.

Backends covered:
  - encoder_decoder (NLLB family via NLLBBackend v3.7 optimized)
  - autoregressive (TranslateGemma, SmolLM2 via AutoregressiveBackend)
  - diffusion (DiffusionGemma 26B-A4B via llama.cpp subprocess)
  - gguf_text (QAT E2B/E4B via llama.cpp subprocess)

Platform-specific optimizations:
  CUDA: torch.compile, async H2D, CUDA streams, pinned memory, fused kernels
  MPS:  skip warmup (MPSGraph shader waste), direct-to-Metal loading
  CPU:  low_cpu_mem_usage, single-threaded llama.cpp

Usage:
  source .venv/bin/activate
  python -m benchmark.run_models  [--cuda | --mps | --auto]
"""

from __future__ import annotations

import json, logging, os, subprocess, sys, time, gc, platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Environment ──────────────────────────────────────────────────────────
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ["PYTHONUNBUFFERED"] = "1"
import torch, warnings
warnings.filterwarnings("ignore", message=".*deprecated.*")

import logging as _logging
for _noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub.file_download",
               "huggingface_hub.utils._http"):
    _logging.getLogger(_noisy).setLevel(_logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.hardware.backend import detect_backend, DeviceInfo
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
SEED = 42

LLAMA_BUILD = Path.home() / "Documents/ComputerScience/Projects/llama/llama.cpp/build"
LLAMA_CLI = LLAMA_BUILD / "bin/llama-cli"
LLAMA_DIFFUSION_CLI = LLAMA_BUILD / "bin/llama-diffusion-cli"
LLAMA_MODELS = Path.home() / "Documents/ComputerScience/Projects/llama/models"


# ═════════════════════════════════════════════════════════════════════════════
# Platform detection
# ═════════════════════════════════════════════════════════════════════════════

def get_platform() -> dict:
    """Detect platform and return optimal settings."""
    info = {"name": "cpu"}

    if torch.cuda.is_available():
        info["name"] = "cuda"
        info["num_gpus"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["use_compile"] = True
        info["use_flash_attention"] = True
        info["use_fused_kernels"] = True
        info["use_cuda_streams"] = True
        info["skip_warmup"] = False
        info["llama_ngl"] = "999"
        info["llama_threads"] = 4
        info["llama_batch"] = 4096
    elif torch.backends.mps.is_available():
        info["name"] = "mps"
        info["num_gpus"] = 1
        info["gpu_name"] = "Apple Silicon (MPS)"
        info["use_compile"] = False  # deadlocks on MPS
        info["use_flash_attention"] = False
        info["use_fused_kernels"] = False  # CUDA-only
        info["use_cuda_streams"] = False
        info["skip_warmup"] = True   # MPS Graph warmup = shader waste
        info["llama_ngl"] = "all"
        info["llama_threads"] = 8
        info["llama_batch"] = 2048
    else:
        info["name"] = "cpu"
        info["num_gpus"] = 0
        info["gpu_name"] = "CPU"
        info["use_compile"] = False
        info["use_flash_attention"] = False
        info["use_fused_kernels"] = False
        info["use_cuda_streams"] = False
        info["skip_warmup"] = True
        info["llama_ngl"] = "0"
        info["llama_threads"] = f"{os.cpu_count() or 4}"
        info["llama_batch"] = 512

    return info


# ═════════════════════════════════════════════════════════════════════════════
# Models
# ═════════════════════════════════════════════════════════════════════════════

MODELS = [
    # ── NLLB family (encoder-decoder, greedy decode, fastest path) ───────
    {"name": "NLLB-200-distilled-600M", "path": "facebook/nllb-200-distilled-600M",
     "backend": "nllb_python",
     "extra": {"nllb_source_lang": "eng_Latn", "nllb_target_lang": "tur_Latn", "num_beams": 1},
     "tags": ["nllb", "600M"]},
    {"name": "NLLB-200-distilled-1.3B", "path": "facebook/nllb-200-distilled-1.3B",
     "backend": "nllb_python",
     "extra": {"nllb_source_lang": "eng_Latn", "nllb_target_lang": "tur_Latn", "num_beams": 1},
     "tags": ["nllb", "1.3B"]},
    # ── Autoregressive ───────────────────────────────────────────────────
    {"name": "SmolLM2-1.7B-Instruct", "path": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
     "backend": "ar_python", "tags": ["smollm", "1.7B"]},
    {"name": "TranslateGemma-4B", "path": "google/translategemma-4b-it",
     "backend": "ar_python", "tags": ["gemma", "4B"]},
    # ── DiffusionGemma (llama.cpp, 25GB GGUF on disk) ────────────────────
    {"name": "DiffusionGemma-26B-A4B-Q8_0", "backend": "llama_diffusion",
     "path": str(LLAMA_MODELS / "diffusiongemma-26B-A4B-it-Q8_0.gguf"),
     "llama_args": {"diffusion_steps": 64, "diffusion_algorithm": 4,
                    "ctx_size": 4096, "n_predict": 256, "temp": 0.8},
     "tags": ["diffusion", "26B-MoE", "Q8_0"]},
    # ── QAT GGUF (llama.cpp) ─────────────────────────────────────────────
    {"name": "Gemma-4-E2B-QAT-UD-Q4_K_XL", "backend": "llama_text",
     "path": str(LLAMA_MODELS / "gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf"),
     "llama_args": {"ctx_size": 4096, "n_predict": 256, "temp": 0.0},
     "hf_dl": "unsloth/gemma-4-E2B-it-qat-GGUF:gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf",
     "tags": ["gemma", "E2B", "QAT", "GGUF"]},
    {"name": "Gemma-4-E4B-QAT-UD-Q4_K_XL", "backend": "llama_text",
     "path": str(LLAMA_MODELS / "gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"),
     "llama_args": {"ctx_size": 4096, "n_predict": 256, "temp": 0.0},
     "hf_dl": "unsloth/gemma-4-E4B-it-qat-GGUF:gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf",
     "tags": ["gemma", "E4B", "QAT", "GGUF"]},
]


# ═════════════════════════════════════════════════════════════════════════════
# Runners
# ═════════════════════════════════════════════════════════════════════════════

def run_python_engine(model_def, device_info, plat, run_idx, run_dir):
    """Run NLLB or AR model via Python backend with platform optimizations."""
    name = model_def["name"]
    path = model_def["model_path"] if "model_path" in model_def else model_def["path"]
    be_type = model_def.get("backend_type",
        "encoder_decoder" if model_def["backend"] == "nllb_python" else "auto")
    extra = dict(model_def.get("extra", {}))
    extra["skip_warmup"] = plat["skip_warmup"]

    # Platform-specific env setup
    if plat["skip_warmup"]:
        os.environ["TR_SKIP_WARMUP"] = "1"
    elif "TR_SKIP_WARMUP" in os.environ:
        del os.environ["TR_SKIP_WARMUP"]

    engine = InferenceEngine(
        model_path=path, tokenizer_path="",
        device_info=device_info,
        decoding_params=DecodingParams(max_new_tokens=128, temperature=0.0),
        use_flash_attention=plat["use_flash_attention"],
        use_torch_compile=plat["use_compile"],
        max_input_tokens=128,
        backend_type=be_type,
        extra=extra,
    )
    engine.load()
    load_s = time.monotonic()
    print(f"    Loaded in {load_s - _global_load_start:.1f}s")

    # Batch tune (no hard clamp — let tuner decide)
    tuner = BatchSizeTuner()
    batch_size = tuner.tune(engine.model, engine.tokenizer,
                            device_info.device, device_info.backend, 128)
    # Safety clamp for very tight memory
    max_bs = 32 if plat["name"] == "cuda" else 16
    batch_size = min(batch_size, max_bs)
    print(f"    Batch: {batch_size} (tuned, max={max_bs})")

    engine.warmup(batches=5 if plat["skip_warmup"] else 10)

    # ── Translation loop ──
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

    batches = 0; total_tokens = 0
    try:
        while timer.elapsed() < RUN_DURATION:
            batch = pipeline.next_batch()
            if batch is None:
                if pipeline.draining(): break
                continue
            result = engine.translate(batch)
            pipeline.release_batch(batch)
            metrics.log_batch(result)
            batches += 1; total_tokens += result.output_tokens_total
            tps = metrics.get_rolling_throughput()
            print(f"    [{timer.elapsed():5.0f}s] b={batches} "
                  f"tok={total_tokens:,} tps={tps:.0f}", flush=True)
            if timer.elapsed() > RUN_DURATION: break
    finally:
        metrics.stop(); pipeline.stop_prefetch()
        dur = timer.elapsed()

    agg = MetricsAggregator(run_dir / "metrics"); ms = agg.aggregate()
    bs = ms.get("batch", {})

    # ── BERTScore ──
    quality = {}
    if Path(REFERENCE_SET).exists():
        print(f"    Quality (BERTScore, max {QUALITY_MAX_REFS} refs)...")
        try:
            rl = ReferenceLoader(REFERENCE_SET)
            srcs, refs = rl.load()
            if QUALITY_MAX_REFS and QUALITY_MAX_REFS < len(srcs):
                srcs, refs = srcs[:QUALITY_MAX_REFS], refs[:QUALITY_MAX_REFS]
            from benchmark.quality.benchmark import _build_batch
            MB = type("_MiniBatch", (), {})
            iids, amask, _ = _build_batch(srcs, engine.tokenizer, engine.devices[0])
            mb = MB(); mb.input_ids = iids; mb.attention_mask = amask
            mb.raw_texts = srcs; mb.batch_id = 0
            tres = engine.translate(mb)
            hyps = [g.translated_text for g in tres.generations]
            bs_r = compute_bertscore(srcs, hyps)
            quality = {"bertscore": bs_r.get("system_score"),
                       "num_references": len(refs), "num_translated": len(hyps)}
            print(f"    BERTScore: {quality['bertscore']}")
        except Exception as e:
            print(f"    quality err: {e}")

    return {
        "speed": {"mean_tps": bs.get("mean_tps", 0), "median_tps": bs.get("median_tps", 0),
                  "p95_tps": bs.get("p95_tps", 0),
                  "mean_latency_ms": bs.get("mean_latency_ms", 0),
                  "batches_completed": batches, "total_tokens_translated": total_tokens,
                  "run_duration_seconds": round(dur, 1), "batch_size": batch_size},
        "quality": quality,
    }


def run_llama_subprocess(model_def, device_info, plat, run_idx, run_dir):
    """Run a GGUF model via llama.cpp CLI subprocess (platform-optimized)."""
    path = Path(model_def["path"])
    binary = LLAMA_DIFFUSION_CLI if model_def["backend"] == "llama_diffusion" else LLAMA_CLI
    is_diffusion = model_def["backend"] == "llama_diffusion"

    if not binary.exists():
        return {"error": f"Binary missing: {binary}"}
    if not path.exists():
        hf = model_def.get("hf_dl", "")
        if hf:
            print(f"    Downloading {hf}...")
            try:
                from huggingface_hub import hf_hub_download
                repo, fname = hf.split(":", 1)
                path = Path(hf_hub_download(repo_id=repo, filename=fname,
                                            local_dir=str(LLAMA_MODELS)))
            except Exception as e:
                return {"error": f"DL failed: {e}"}
        else:
            return {"error": f"Model missing: {path}"}

    la = model_def.get("llama_args", {})
    ngl = plat["llama_ngl"]
    diff_steps = la.get("diffusion_steps", 64)
    diff_algo = la.get("diffusion_algorithm", 4)
    ctx = la.get("ctx_size", 4096)
    n_pred = la.get("n_predict", 256)
    temp = la.get("temp", 0.8)
    threads = plat["llama_threads"]
    batch = plat["llama_batch"]

    print(f"    llama.cpp: ngl={ngl} steps={diff_steps} ctx={ctx} threads={threads}")

    # Build test prompts
    rl = ReferenceLoader(REFERENCE_SET)
    srcs, refs = rl.load()
    if QUALITY_MAX_REFS and QUALITY_MAX_REFS < len(srcs):
        srcs, refs = srcs[:QUALITY_MAX_REFS], refs[:QUALITY_MAX_REFS]

    prefix = "Translate English to Turkish. Output only the Turkish translation.\n\nEnglish: "
    prompts = [f"{prefix}{s}\nTurkish:" for s in srcs]
    MAX_PROMPTS = 12  # limit for llama.cpp speed
    prompts = prompts[:MAX_PROMPTS]; srcs = srcs[:MAX_PROMPTS]; refs = refs[:MAX_PROMPTS]

    total_tokens = 0; bn = 0; hyps = []
    wall_start = time.monotonic()

    for pi, prompt in enumerate(prompts):
        if time.monotonic() - wall_start > RUN_DURATION:
            break
        t0 = time.monotonic()
        try:
            cmd = [str(binary), "-m", str(path), "-ngl", str(ngl)]
            if is_diffusion:
                cmd += [f"--diffusion-steps", str(diff_steps),
                        f"--diffusion-algorithm", str(diff_algo)]
            cmd += ["-c", str(ctx), "-n", str(n_pred), "--temp", str(temp),
                    "-t", str(threads), "-b", str(batch),
                    "-s", str(SEED + pi), "--no-conversation",
                    "--log-verbosity", "1", "-p", prompt]
            if plat["name"] == "cuda":
                cmd.insert(cmd.index("-t") if "-t" in cmd else len(cmd),
                            "--mlock")  # pin model in GPU memory

            env = os.environ.copy()
            if plat["name"] == "mps":
                env["GGML_METAL_PATH_RESOURCES"] = str(LLAMA_BUILD / "bin")

            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=300, env=env)
            out = r.stdout.strip()
            if "Turkish:" in out:
                out = out.split("Turkish:")[-1].strip()
            hyps.append(out)
            out_toks = max(len(out.split()) * 1.3, 1)
            total_tokens += int(out_toks); bn += 1
            elapsed = time.monotonic() - t0
            print(f"    [{time.monotonic()-wall_start:5.0f}s] b={bn} "
                  f"tok≈{int(out_toks)} tps≈{out_toks/max(elapsed,0.01):.0f}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"    ⚠ prompt {pi} timeout"); hyps.append("")
        except Exception as e:
            print(f"    ⚠ prompt {pi} err: {e}"); hyps.append("")

    dur = time.monotonic() - wall_start
    quality = {}
    try:
        if hyps and any(h for h in hyps):
            bs = compute_bertscore(srcs[:len(hyps)], hyps)
            quality = {"bertscore": bs.get("system_score"), "num_references": len(refs[:len(hyps)]),
                       "num_translated": len([h for h in hyps if h])}
            print(f"    BERTScore: {quality['bertscore']}")
    except Exception as e:
        print(f"    quality err: {e}")

    return {
        "speed": {"mean_tps": round(total_tokens / max(dur, 0.1), 1),
                  "batches_completed": bn, "total_tokens_translated": total_tokens,
                  "run_duration_seconds": round(dur, 1)},
        "quality": quality,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
_global_load_start = 0.0

def main(cli_args=None):
    """Run all models through the benchmark.

    cli_args: optional list of CLI argument strings (used for testing).
    When None, sys.argv[1:] is parsed.
    """
    import argparse as _argparse
    _parser = _argparse.ArgumentParser(
        description="Comprehensive EN->TR translation benchmark — runs all models.",
    )
    _parser.add_argument("--cuda", action="store_true",
                        help="Force CUDA backend (auto-detected by default)")
    _parser.add_argument("--mps", action="store_true",
                        help="Force MPS/Apple Silicon backend")
    _parser.add_argument("--auto", action="store_true",
                        help="Auto-detect backend (default behaviour)")
    _args = _parser.parse_args(cli_args)

    import logging as _logging
    for _noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub.file_download",
                   "huggingface_hub.utils._http"):
        _logging.getLogger(_noisy).setLevel(_logging.WARNING)

    global _global_load_start
    plat = get_platform()

    # CLI platform overrides
    if _args.cuda:
        plat.update({"name": "cuda", "use_compile": torch.cuda.is_available(),
                     "use_flash_attention": torch.cuda.is_available(),
                     "use_fused_kernels": torch.cuda.is_available(),
                     "use_cuda_streams": torch.cuda.is_available(),
                     "skip_warmup": False, "llama_ngl": "999",
                     "llama_threads": 4, "llama_batch": 4096})
    elif _args.mps:
        plat.update({"name": "mps", "use_compile": False,
                     "use_flash_attention": False, "use_fused_kernels": False,
                     "use_cuda_streams": False, "skip_warmup": True,
                     "llama_ngl": "all", "llama_threads": 8, "llama_batch": 2048})
    device_info = detect_backend(plat["name"])

    print("=" * 100)
    print(f"  TR Corpus Translation Benchmark — {plat['gpu_name']}")
    print(f"  Platform: {plat['name']} | PyTorch {torch.__version__}")
    print(f"  Warmup: {'skip' if plat['skip_warmup'] else 'full'} | "
          f"Compile: {plat['use_compile']} | FlashAttn: {plat['use_flash_attention']}")
    print(f"  Models: {len(MODELS)} | Duration/model: {RUN_DURATION}s")
    print("=" * 100)

    env_snapshot = get_environment_snapshot()
    results = []
    total_start = time.monotonic()

    for i, md in enumerate(MODELS, 1):
        name = md["name"]; be = md["backend"]
        print(f"\n{'='*70}")
        print(f"[{i}/{len(MODELS)}] {name}  ({be})")
        print(f"{'='*70}")

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        safe = name.replace('-','_').replace(' ','_')
        run_dir = Path(OUTPUT_DIR) / f"bm_{safe}_{i:02d}_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(run_dir)
        _global_load_start = time.monotonic()

        try:
            if be in ("nllb_python", "ar_python"):
                r = run_python_engine(md, device_info, plat, i, run_dir)
            elif be in ("llama_text", "llama_diffusion"):
                r = run_llama_subprocess(md, device_info, plat, i, run_dir)
            else:
                r = {"error": f"Unknown backend: {be}"}
        except Exception as e:
            import traceback; traceback.print_exc()
            r = {"error": str(e)}

        r["model"] = name; r["model_path"] = md["path"]; r["type"] = be
        r["tags"] = md.get("tags", [])
        r["load_seconds"] = round(time.monotonic() - _global_load_start, 1)
        r["environment"] = {"backend": plat["name"], "device": plat["gpu_name"],
                            "pytorch": env_snapshot.get("pytorch_version","?")}
        r["run_dir"] = str(run_dir)
        results.append(r)

        gc.collect()
        if plat["name"] == "mps":
            try: torch.mps.empty_cache()
            except: pass
        elif plat["name"] == "cuda":
            torch.cuda.empty_cache()
        time.sleep(1)

    total_elapsed = time.monotonic() - total_start
    print(f"\nTotal: {total_elapsed/60:.1f} min")

    # ── Table ──
    hdr = (f"\n{'MODEL':<35} {'TPS':>7} {'Lat(ms)':>8} {'B':>4}  "
           f"{'BERTScore':>9} {'Load':>6}  Status")
    print("=" * len(hdr)); print(hdr); print("=" * len(hdr))
    for r in results:
        n = r["model"][:34]
        if r.get("error"):
            print(f"{n:<35} {'—':>7} {'—':>8} {'—':>4}  {'—':>9} "
                  f"{r.get('load_seconds',0):>6.1f}  ✗ {r['error'][:30]}")
        else:
            s = r.get("speed", {}); q = r.get("quality", {})
            print(f"{n:<35} {s.get('mean_tps',0):>7.1f} "
                  f"{s.get('mean_latency_ms',0):>8.0f} {s.get('batch_size','?'):>4}  "
                  f"{q.get('bertscore') or 0:>9.4f} "
                  f"{r.get('load_seconds',0):>6.1f}  ✓")
    print("=" * len(hdr))

    ok = [r for r in results if not r.get("error")]
    if ok:
        tps = [(r["speed"]["mean_tps"], r["model"]) for r in ok]
        bs = [(r["quality"].get("bertscore") or 0, r["model"]) for r in ok
              if r["quality"].get("bertscore")]
        if tps: print(f"  Best TPS:       {max(tps,key=lambda x:x[0])[0]:.0f} tok/s  ({max(tps,key=lambda x:x[0])[1]})")
        if bs: print(f"  Best BERTScore: {max(bs,key=lambda x:x[0])[0]:.4f}     ({max(bs,key=lambda x:x[0])[1]})")

    out_path = Path(OUTPUT_DIR) / "model_comparison.json"
    report = {"title": "TR Corpus Translation — Model Comparison",
              "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "platform": {"backend": plat["name"], "device": plat["gpu_name"],
                           "pytorch": env_snapshot.get("pytorch_version","?"),
                           "optimizations": {k: v for k, v in plat.items() if isinstance(v, bool)}},
              "config": {"run_duration_per_model_s": RUN_DURATION, "quality_max_references": QUALITY_MAX_REFS},
              "models_tested": len(results), "total_wallclock_s": round(total_elapsed, 1),
              "results": results}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    ok_ct = len([r for r in results if not r.get("error")])
    print(f"\n✓ {out_path}  ({ok_ct}/{len(results)} succeeded)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
