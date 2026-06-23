#!/usr/bin/env python3
"""Single-model benchmark runner — invoked as subprocess per model for MPS memory isolation."""
import json, os, sys, time, gc, warnings
from datetime import datetime, timezone
from pathlib import Path

# Suppress all third-party noise BEFORE any imports.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
warnings.filterwarnings("ignore")
# torch._dynamo sprays "recompile_limit" warnings during warmup when
# accelerate hooks change tensor shapes — normal, suppress.
os.environ.setdefault("TORCH_LOGS", "-dynamo")

import torch
import logging as _logging
for _noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub.file_download",
               "huggingface_hub.utils._http"):
    _logging.getLogger(_noisy).setLevel(_logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
from benchmark.utils.timer import PrecisionTimer

INPUT_GLOB = "data/input/*.jsonl.gz"
REFERENCE_SET = "data/references/golden_en_tr.jsonl"
RUN_DURATION = 120
QUALITY_MAX_REFS = 32
SEED = 42

# Auto-detect data paths — check local first, then /data/ (H200 mount)
def _find_file(relative: str) -> str:
    if os.path.exists(relative):
        return relative
    alt = "/" + relative
    if os.path.exists(alt):
        return alt
    return relative  # let the caller fail with a clear FileNotFoundError

REFERENCE_SET = _find_file(REFERENCE_SET)
# INPUT_GLOB handled by JSONLLoader which supports multiple patterns


def run_one_model(model_def: dict) -> dict:
    """Run a single model and return TPS + quality results."""
    name = model_def["name"]
    path = model_def["path"]
    be_type = model_def.get("backend_type", "auto")
    extra = dict(model_def.get("extra", {}))

    plat = detect_backend("auto")
    is_cuda = plat.backend == "cuda"
    is_mps = plat.backend == "mps"

    # Platform-specific config — single source of truth for warmup behavior.
    # Model definitions do NOT set skip_warmup; the platform decides.
    if is_cuda:
        extra["skip_warmup"] = False  # full warmup: cuBLAS autotuning + graph capture
    else:
        extra["skip_warmup"] = True   # MPS/CPU: skip (IOAccelerator waste on MPS)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Path: {path}  |  Backend: {be_type}")
    if is_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU: {gpu_name} ({gpu_mem:.0f} GB)  |  Torch: {torch.__version__}")
    else:
        print(f"  Platform: {plat.backend}  |  Torch: {torch.__version__}")
    print(f"{'='*60}")

    t0 = time.monotonic()

    try:
        is_enc_dec = be_type == "encoder_decoder"
        engine = InferenceEngine(
            model_path=path, tokenizer_path="",
            device_info=plat,
            decoding_params=DecodingParams(max_new_tokens=128, temperature=0.0),
            use_flash_attention=is_cuda,
            # torch.compile pointless for encoder-decoder: model.generate()
            # handles its own graph optimisation.  compile introduces
            # shape-dependent recompilation stalls (minutes) at large batch.
            use_torch_compile=is_cuda and not is_enc_dec,
            max_input_tokens=128,
            backend_type=be_type,
            extra=extra,
        )
        engine.load()
    except Exception as e:
        load_s = time.monotonic() - t0
        err_msg = str(e)[:300]
        print(f"  ✗ Load failed after {load_s:.1f}s: {err_msg}")
        return {
            "model": name, "model_path": path, "backend_type": be_type,
            "error": f"Load: {err_msg}", "load_seconds": round(load_s, 1),
            "mean_tps": 0, "batches_completed": 0, "total_tokens_translated": 0,
            "platform": plat.backend,
        }
    load_s = time.monotonic() - t0
    print(f"  Loaded in {load_s:.1f}s")

    # Batch tuning — tuner does a binary search OOM test and applies
    # a 15% safety margin.  The returned batch_size is authoritative.
    try:
        tuner = BatchSizeTuner()
        batch_size = tuner.tune(engine.model, engine.tokenizer,
                               plat.device, plat.backend, 128)
        if is_cuda:
            # Autoregressive models with large vocab (Gemma: 262k) OOM on
            # lm_head projection at high batch × seq_len.  1740 × 128 × 262k
            # × 2 bytes ≈ 117 GB for logits alone — exceeds 140 GB GPU.
            # Encoder-decoder models need cap for tokenization throughput
            # (256k-vocab SentencePiece ≤ ~15 chunks/sec).
            cap = 256 if is_enc_dec else 512
            batch_size = min(batch_size, cap)
        elif not is_cuda:
            batch_size = min(batch_size, 4)
    except Exception:
        batch_size = 128 if is_cuda else 1
    print(f"  Batch size: {batch_size}")

    # Pass configured batch size to the backend so warmup matches production.
    # Without this, torch.compile needs to recompile from warmup shape (bs=1)
    # to production shape (bs=1740) — a multi-minute stall on large models.
    engine._configured_batch_size = batch_size

    # Warmup
    warmup_batches = 20 if is_cuda else 3
    try:
        engine.warmup(batches=warmup_batches)
    except Exception as e:
        print(f"  Warmup warning: {e}")

    # Translation loop
    loader = JSONLLoader([INPUT_GLOB], shuffle=True, seed=SEED)
    chunker = TextChunker(engine.tokenizer, 128, 50)
    filt = ChunkFilter(min_tokens=10, max_garbage_ratio=0.95)
    # Encoder-decoder tokenizers (NLLB, MADLAD) need more workers for their
    # 256k-vocab SentencePiece — ~10× slower than 32k-vocab LLaMA tokenizers.
    n_workers = 16 if is_enc_dec else (8 if is_cuda else 2)
    pipeline = AsyncPipeline(loader, chunker, engine.tokenizer, filt,
                            batch_size=batch_size, prefetch_workers=n_workers,
                            backend=plat.backend)

    # Pre-buffer: start tokenization 10s before the timer so the first
    # batch is already assembled when the clock starts.
    pipeline.start_prefetch()
    time.sleep(10 if is_enc_dec else 0)

    run_dir = Path("data/output") / f"bm_{name.replace(' ','_').replace('/','_')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = MetricsCollector(run_dir / "metrics", plat, 1)
    timer = PrecisionTimer(); timer.start()
    metrics.start(timer.start_time())

    batches = 0; total_tokens = 0
    last_heartbeat = 0
    try:
        while timer.elapsed() < RUN_DURATION:
            batch = pipeline.next_batch()
            if batch is None:
                if pipeline.draining():
                    break
                continue
            result = engine.translate(batch)
            pipeline.release_batch(batch)
            try:
                metrics.log_batch(result)
            except Exception:
                pass  # bypass internal MetricsCollector bugs
            batches += 1; total_tokens += result.output_tokens_total

            now = timer.elapsed()
            if now - last_heartbeat >= 15:
                try:
                    tps = metrics.get_rolling_throughput()
                except Exception:
                    tps = total_tokens / max(now, 0.01)
                print(f"  [{now:5.0f}s] b={batches} tok={total_tokens:,} tps={tps:.0f}")
                last_heartbeat = now
    finally:
        metrics.stop(); pipeline.stop_prefetch()
        dur = timer.elapsed()

    # Aggregate metrics (may fail if collector had issues)
    try:
        agg = MetricsAggregator(run_dir / "metrics")
        ms = agg.aggregate()
        bs = ms.get("batch", {})
    except Exception:
        bs = {}
    mean_tps = bs.get("mean_tps", 0) or (total_tokens / max(dur, 0.01))

    # Quality: BERTScore
    quality = {}
    if Path(REFERENCE_SET).exists():
        try:
            rl = ReferenceLoader(REFERENCE_SET)
            srcs, refs = rl.load()
            if QUALITY_MAX_REFS and QUALITY_MAX_REFS < len(srcs):
                srcs, refs = srcs[:QUALITY_MAX_REFS], refs[:QUALITY_MAX_REFS]

            # Translate reference source texts
            from benchmark.quality.benchmark import _build_batch
            MB = type("_MiniBatch", (), {})
            iids, amask, _ = _build_batch(
                srcs, engine.tokenizer, engine.devices[0], engine=engine,
            )
            mb = MB(); mb.input_ids = iids; mb.attention_mask = amask
            mb.raw_texts = srcs; mb.batch_id = 0
            tres = engine.translate(mb)
            hyps = [g.translated_text for g in tres.generations]
            bs_r = compute_bertscore(srcs, hyps)
            quality = {
                "bertscore": bs_r.get("system_score"),
                "num_references": len(refs),
                "num_translated": len(hyps),
            }
            print(f"  BERTScore: {quality['bertscore']:.4f}")
        except Exception as e:
            print(f"  Quality error: {e}")

    # Clean up
    try: engine.close()
    except: pass
    del engine, pipeline, metrics
    gc.collect()
    if is_mps:
        try: torch.mps.empty_cache()
        except: pass
    elif is_cuda:
        torch.cuda.empty_cache()

    result = {
        "model": name,
        "model_path": path,
        "backend_type": be_type,
        "mean_tps": mean_tps,
        "median_tps": bs.get("median_tps", 0),
        "p95_tps": bs.get("p95_tps", 0),
        "mean_latency_ms": bs.get("mean_latency_ms", 0),
        "batches_completed": batches,
        "total_tokens_translated": total_tokens,
        "run_duration_seconds": round(dur, 1),
        "batch_size": batch_size,
        "load_seconds": round(load_s, 1),
        "bertscore": quality.get("bertscore"),
        "quality_num_refs": quality.get("num_references", 0),
        "platform": plat.backend,
    }
    return result


def run_llama_model(model_def: dict) -> dict:
    """Run a GGUF model via llama.cpp subprocess."""
    import subprocess
    name = model_def["name"]
    gguf_path = Path(model_def["path"])
    llama_bin = Path(model_def["llama_binary"])
    is_diffusion = model_def.get("is_diffusion", False)
    hf_dl = model_def.get("hf_dl", "")

    plat = detect_backend("auto")

    if not llama_bin.exists():
        return {"model": name, "error": f"Binary missing: {llama_bin}"}
    if not gguf_path.exists():
        if hf_dl:
            print(f"  Downloading {hf_dl}...")
            try:
                from huggingface_hub import hf_hub_download
                repo, fname = hf_dl.split(":", 1)
                gguf_path = Path(hf_hub_download(
                    repo_id=repo, filename=fname,
                    local_dir=str(gguf_path.parent),
                    resume_download=True,
                ))
            except Exception as e:
                return {"model": name, "error": f"GGUF download failed: {e}"}
        else:
            return {"model": name, "error": f"GGUF missing: {gguf_path}"}

    print(f"\n{'='*60}")
    print(f"  {name}  (llama.cpp)")
    print(f"  GGUF: {gguf_path}")
    print(f"{'='*60}")

    rl = ReferenceLoader(REFERENCE_SET)
    srcs, refs = rl.load()
    if QUALITY_MAX_REFS and QUALITY_MAX_REFS < len(srcs):
        srcs, refs = srcs[:QUALITY_MAX_REFS], refs[:QUALITY_MAX_REFS]

    prefix = "Translate English to Turkish. Output only the Turkish translation.\n\nEnglish: "
    prompts = [f"{prefix}{s}\nTurkish:" for s in srcs]
    MAX_PROMPTS = 12
    prompts = prompts[:MAX_PROMPTS]; srcs = srcs[:MAX_PROMPTS]

    la = model_def.get("llama_args", {})
    ngl = model_def.get("ngl", "all")
    ctx = la.get("ctx_size", 4096)
    n_pred = la.get("n_predict", 256)
    temp = la.get("temp", 0.0)
    threads = 8

    total_tokens = 0; bn = 0; hyps = []
    wall_start = time.monotonic()

    for pi, prompt in enumerate(prompts):
        if time.monotonic() - wall_start > RUN_DURATION:
            break
        t0 = time.monotonic()
        try:
            cmd = [str(llama_bin), "-m", str(gguf_path), "-ngl", str(ngl)]
            if is_diffusion:
                cmd += [f"--diffusion-steps", str(la.get("diffusion_steps", 64)),
                       f"--diffusion-algorithm", str(la.get("diffusion_algorithm", 4))]
            cmd += ["-c", str(ctx), "-n", str(n_pred), "--temp", str(temp),
                    "-t", str(threads), "-b", "2048",
                    "-s", str(SEED + pi), "--no-conversation",
                    "--log-verbosity", "1", "-p", prompt]
            env = os.environ.copy()
            env["GGML_METAL_PATH_RESOURCES"] = str(llama_bin.parent)
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
            out = r.stdout.strip()
            if "Turkish:" in out:
                out = out.split("Turkish:")[-1].strip()
            hyps.append(out)
            out_toks = max(len(out.split()) * 1.3, 1)
            total_tokens += int(out_toks); bn += 1
            elapsed = time.monotonic() - t0
            print(f"  [{time.monotonic()-wall_start:5.0f}s] b={bn} "
                  f"tok≈{int(out_toks)} tps≈{out_toks/max(elapsed,0.01):.0f}")
        except subprocess.TimeoutExpired:
            print(f"  ⚠ prompt {pi} timeout"); hyps.append("")
        except Exception as e:
            print(f"  ⚠ prompt {pi} err: {e}"); hyps.append("")

    dur = time.monotonic() - wall_start
    quality = {}
    if hyps and any(h for h in hyps):
        try:
            bs = compute_bertscore(srcs[:len(hyps)], hyps)
            quality = {"bertscore": bs.get("system_score")}
            print(f"  BERTScore: {quality['bertscore']:.4f}")
        except Exception as e:
            print(f"  Quality error: {e}")

    return {
        "model": name,
        "model_path": str(gguf_path),
        "backend_type": "llama.cpp" + ("_diffusion" if is_diffusion else "_text"),
        "mean_tps": round(total_tokens / max(dur, 0.1), 1),
        "batches_completed": bn,
        "total_tokens_translated": total_tokens,
        "run_duration_seconds": round(dur, 1),
        "bertscore": quality.get("bertscore"),
        "platform": plat.backend if plat else "unknown",
    }


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else None
    if model_name is None:
        print("Usage: python run_one_model.py <model_name>")
        sys.exit(1)

    # ── Model definitions ──
    # Python backend models (NLLB, autoregressive, QAT CT/mobile)
    MODELS = {
        # NLLB family (proven EN→TR translators)
        "nllb_600m": {
            "name": "NLLB-200-distilled-600M",
            "path": "facebook/nllb-200-distilled-600M",
            "backend_type": "encoder_decoder",
            "extra": {"nllb_source_lang": "eng_Latn", "nllb_target_lang": "tur_Latn", "num_beams": 1},
        },
        "nllb_1.3b": {
            "name": "NLLB-200-distilled-1.3B",
            "path": "facebook/nllb-200-distilled-1.3B",
            "backend_type": "encoder_decoder",
            "extra": {"nllb_source_lang": "eng_Latn", "nllb_target_lang": "tur_Latn", "num_beams": 1},
        },
        "nllb_3.3b": {
            "name": "NLLB-200-3.3B",
            "path": "facebook/nllb-200-3.3B",
            "backend_type": "encoder_decoder",
            "extra": {"nllb_source_lang": "eng_Latn", "nllb_target_lang": "tur_Latn", "num_beams": 1},
        },
        # MADLAD-400 family (T5-based encoder-decoder, 450-language MT)
        # Uses <2tr> prefix token for Turkish — added by pipeline, not forced_bos.
        "madlad_3b": {
            "name": "MADLAD-400-3B-MT",
            "path": "google/madlad400-3b-mt",
            "backend_type": "encoder_decoder",
            "extra": {"num_beams": 1},
        },
        "madlad_10b": {
            "name": "MADLAD-400-10B-MT",
            "path": "google/madlad400-10b-mt",
            "backend_type": "encoder_decoder",
            "extra": {"num_beams": 1},
        },
        # Autoregressive (proven translators)
        "smollm2": {
            "name": "SmolLM2-1.7B-Instruct",
            "path": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "backend_type": "auto",
        },
        "translategemma": {
            "name": "TranslateGemma-4B",
            "path": "google/translategemma-4b-it",
            "backend_type": "auto",
        },
    }

    LLAMA_MODELS_DIR = Path.home() / "Documents/ComputerScience/Projects/llama/models"
    LLAMA_BUILD = Path.home() / "Documents/ComputerScience/Projects/llama/llama.cpp/build"
    GOOGLE_QAT_GGUF_DIR = LLAMA_MODELS_DIR / "google_qat"

    LLAMA_MODELS = {
        # ── Google QAT GGUF (llama.cpp) — official Google repos ──
        "gemma_e2b_qat_q4_0_gguf": {
            "name": "Gemma-4-E2B-QAT-Q4_0-GGUF (Google)",
            "path": str(GOOGLE_QAT_GGUF_DIR / "gemma-4-E2B_q4_0-it.gguf"),
            "llama_binary": str(LLAMA_BUILD / "bin/llama-cli"),
            "is_diffusion": False,
            "ngl": "all",
            "hf_dl": "google/gemma-4-E2B-it-qat-q4_0-gguf:gemma-4-E2B_q4_0-it.gguf",
            "llama_args": {"ctx_size": 4096, "n_predict": 256, "temp": 0.0},
        },
        "gemma_e4b_qat_q4_0_gguf": {
            "name": "Gemma-4-E4B-QAT-Q4_0-GGUF (Google)",
            "path": str(GOOGLE_QAT_GGUF_DIR / "gemma-4-E4B_q4_0-it.gguf"),
            "llama_binary": str(LLAMA_BUILD / "bin/llama-cli"),
            "is_diffusion": False,
            "ngl": "all",
            "hf_dl": "google/gemma-4-E4B-it-qat-q4_0-gguf:gemma-4-E4B_q4_0-it.gguf",
            "llama_args": {"ctx_size": 4096, "n_predict": 256, "temp": 0.0},
        },
        "gemma_26b_a4b_qat_q4_0_gguf": {
            "name": "Gemma-4-26B-A4B-QAT-Q4_0-GGUF (Google)",
            "path": str(GOOGLE_QAT_GGUF_DIR / "gemma-4-26B_q4_0-it.gguf"),
            "llama_binary": str(LLAMA_BUILD / "bin/llama-cli"),
            "is_diffusion": False,
            "ngl": "all",
            "hf_dl": "google/gemma-4-26B-A4B-it-qat-q4_0-gguf:gemma-4-26B_q4_0-it.gguf",
            "llama_args": {"ctx_size": 4096, "n_predict": 256, "temp": 0.0},
        },
        # ── DiffusionGemma (existing) ──
        "diffusiongemma": {
            "name": "DiffusionGemma-26B-A4B-Q8_0",
            "path": str(LLAMA_MODELS_DIR / "diffusiongemma-26B-A4B-it-Q8_0.gguf"),
            "llama_binary": str(LLAMA_BUILD / "bin/llama-diffusion-cli"),
            "is_diffusion": True,
            "ngl": "all",
            "llama_args": {"diffusion_steps": 64, "diffusion_algorithm": 4,
                          "ctx_size": 4096, "n_predict": 256, "temp": 0.8},
        },
    }

    if model_name in MODELS:
        result = run_one_model(MODELS[model_name])
    elif model_name in LLAMA_MODELS:
        result = run_llama_model(LLAMA_MODELS[model_name])
    else:
        print(f"Unknown model: {model_name}")
        print(f"Available: {list(MODELS.keys()) + list(LLAMA_MODELS.keys())}")
        sys.exit(1)

    # Write result to individual JSON file
    out_path = Path("data/output") / f"result_{model_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n✓ Result written to {out_path}")
    print(json.dumps(result, indent=2))
