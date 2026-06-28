#!/usr/bin/env python3
"""
TR Corpus Translation Benchmark — Complete Model Comparison

Models tested:
  Python backend (NLLB family):
    - facebook/nllb-200-distilled-600M
    - facebook/nllb-200-distilled-1.3B
    - facebook/nllb-200-3.3B (if downloaded)

  Python backend (autoregressive):
    - HuggingFaceTB/SmolLM2-1.7B-Instruct
    - google/translategemma-4b-it

  llama.cpp backend (GGUF models):
    - google/gemma-4-E2B-it-qat-q4_0-gguf  (→ gemma-4-E2B_q4_0-it.gguf)
    - google/gemma-4-E4B-it-qat-q4_0-gguf  (→ gemma-4-E4B_q4_0-it.gguf)
    - google/gemma-4-26B-A4B-it-qat-q4_0-gguf  (→ gemma-4-26B_q4_0-it.gguf)
    - ~/Documents/.../diffusiongemma-26B-A4B-it-Q8_0.gguf

  Python backend (QAT CT/mobile — expected to fail on MPS):
    - google/gemma-4-E2B-it-qat-mobile-ct
    - google/gemma-4-E4B-it-qat-mobile-ct
    - google/gemma-4-E2B-it-qat-mobile-transformers
    - google/gemma-4-E4B-it-qat-mobile-transformers
    - google/gemma-4-E2B-it-qat-w4a16-ct
    - google/gemma-4-E4B-it-qat-w4a16-ct
"""
import json, os, subprocess, sys, time, gc, warnings
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ["PYTHONUNBUFFERED"] = "1"
warnings.filterwarnings("ignore")

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

LLAMA_MODELS = Path.home() / "Documents/ComputerScience/Projects/llama/models"
LLAMA_BUILD = Path.home() / "Documents/ComputerScience/Projects/llama/llama.cpp/build"
LLAMA_CLI = LLAMA_BUILD / "bin/llama-cli"
LLAMA_DIFFUSION_CLI = LLAMA_BUILD / "bin/llama-diffusion-cli"
GOOGLE_QAT_DIR = LLAMA_MODELS / "google_qat"


def run_python_model(model_def: dict) -> dict:
    """Run a model via Python backend (NLLB, autoregressive, or QAT CT/mobile)."""
    name = model_def["name"]
    path = model_def["path"]
    be_type = model_def.get("backend_type", "auto")
    extra = dict(model_def.get("extra", {}))
    extra["skip_warmup"] = True
    expect_failure = model_def.get("expect_failure", False)

    plat = detect_backend("auto")
    is_cuda = plat.backend == "cuda"
    is_mps = plat.backend == "mps"

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Path: {path}  |  Backend: {be_type}")
    print(f"  Platform: {plat.backend}  |  {'⚠ EXPECT FAILURE' if expect_failure else ''}")
    print(f"{'='*60}")

    t0 = time.monotonic()

    try:
        engine = InferenceEngine(
            model_path=path, tokenizer_path="",
            device_info=plat,
            decoding_params=DecodingParams(max_new_tokens=128, temperature=0.0),
            use_flash_attention=is_cuda,
            use_torch_compile=is_cuda,
            max_input_tokens=128,
            backend_type=be_type,
            extra=extra,
        )
        engine.load()
    except Exception as e:
        elapsed = time.monotonic() - t0
        err_msg = str(e)[:200]
        print(f"  ✗ Load failed after {elapsed:.1f}s: {err_msg}")
        return {"model": name, "model_path": path, "backend_type": be_type,
                "error": f"Load: {err_msg}", "load_seconds": round(elapsed, 1)}

    load_s = time.monotonic() - t0
    print(f"  Loaded in {load_s:.1f}s")

    # Batch tuning
    try:
        tuner = BatchSizeTuner()
        batch_size = tuner.tune(engine.model, engine.tokenizer,
                               plat.device, plat.backend, 128)
        batch_size = min(batch_size, 16 if is_cuda else 4)
    except Exception:
        batch_size = 1
    print(f"  Batch size: {batch_size}")

    # Warmup
    try:
        engine.warmup(batches=3)
    except Exception as e:
        print(f"  Warmup warning: {e}")

    # Translation loop
    loader = JSONLLoader([INPUT_GLOB], shuffle=True, seed=SEED)
    chunker = TextChunker(engine.tokenizer, 128, 50)
    filt = ChunkFilter(min_tokens=10, max_garbage_ratio=0.95)
    pipeline = AsyncPipeline(loader, chunker, engine.tokenizer, filt,
                            batch_size=batch_size, prefetch_workers=2,
                            backend=plat.backend)

    run_dir = Path("data/output") / f"bm_{name.replace(' ','_').replace('/','_').replace(':','-')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = MetricsCollector(run_dir / "metrics", plat, 1)
    pipeline.start_prefetch()
    timer = PrecisionTimer(); timer.start()
    metrics.start(timer.start_time())

    batches = 0; total_tokens = 0; last_heartbeat = 0
    try:
        while timer.elapsed() < RUN_DURATION:
            batch = pipeline.next_batch()
            if batch is None:
                if pipeline.draining():
                    break
                continue
            result = engine.translate(batch)
            pipeline.release_batch(batch)
            metrics.log_batch(result)
            batches += 1; total_tokens += result.output_tokens_total
            now = timer.elapsed()
            if now - last_heartbeat >= 15:
                tps = metrics.get_rolling_throughput()
                print(f"  [{now:5.0f}s] b={batches} tok={total_tokens:,} tps={tps:.0f}")
                last_heartbeat = now
    finally:
        metrics.stop(); pipeline.stop_prefetch()
        dur = timer.elapsed()

    agg = MetricsAggregator(run_dir / "metrics")
    ms = agg.aggregate()
    bs = ms.get("batch", {})

    # Quality: BERTScore
    quality = {}
    if Path(REFERENCE_SET).exists():
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
            bs_r = compute_bertscore(refs, hyps)
            quality = {"bertscore": bs_r.get("system_score"),
                       "num_references": len(refs), "num_translated": len(hyps)}
            print(f"  BERTScore: {quality['bertscore']:.4f}")
        except Exception as e:
            print(f"  Quality error: {e}")

    try: engine.close()
    except (RuntimeError, AttributeError): pass  # MPS empty_cache is best-effort
    del engine, pipeline, metrics
    gc.collect()
    if is_mps:
        try: torch.mps.empty_cache()
        except (RuntimeError, AttributeError): pass  # MPS empty_cache is best-effort

    return {
        "model": name, "model_path": path, "backend_type": be_type,
        "mean_tps": bs.get("mean_tps", 0),
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


def run_llama_model(model_def: dict) -> dict:
    """Run a GGUF model via llama.cpp CLI subprocess."""
    name = model_def["name"]
    gguf_path = Path(model_def["path"])
    llama_bin = Path(model_def.get("llama_binary", str(LLAMA_CLI)))
    is_diffusion = model_def.get("is_diffusion", False)
    hf_repo = model_def.get("hf_repo", "")
    hf_file = model_def.get("hf_file", "")

    # Download if missing
    if not gguf_path.exists() and hf_repo:
        print(f"  Downloading {hf_repo}:{hf_file}...")
        try:
            from huggingface_hub import hf_hub_download
            gguf_path = Path(hf_hub_download(
                repo_id=hf_repo, filename=hf_file,
                local_dir=str(gguf_path.parent), resume_download=True,
            ))
        except Exception as e:
            return {"model": name, "error": f"GGUF download failed: {e}"}

    if not llama_bin.exists():
        return {"model": name, "error": f"Binary missing: {llama_bin}"}
    if not gguf_path.exists():
        return {"model": name, "error": f"GGUF missing: {gguf_path}"}

    size_gb = os.path.getsize(gguf_path) / (1024**3)
    print(f"\n{'='*60}")
    print(f"  {name}  (llama.cpp)")
    print(f"  GGUF: {gguf_path}  ({size_gb:.1f} GB)")
    print(f"{'='*60}")

    # Load references — guard against missing file (llama.cpp runs in-process,
    # so a missing file would crash the entire benchmark).
    refs = srcs = []
    if Path(REFERENCE_SET).exists():
        try:
            rl = ReferenceLoader(REFERENCE_SET)
            srcs, refs = rl.load()
        except Exception as e:
            print(f"  ⚠ Failed to load references: {e}")
    else:
        print(f"  ⚠ Reference set not found: {REFERENCE_SET} — quality scores will be N/A")
    if QUALITY_MAX_REFS and QUALITY_MAX_REFS < len(srcs):
        srcs, refs = srcs[:QUALITY_MAX_REFS], refs[:QUALITY_MAX_REFS]

    prefix = "Translate English to Turkish. Output only the Turkish translation.\n\nEnglish: "
    prompts = [f"{prefix}{s}\nTurkish:" for s in srcs]
    MAX_PROMPTS = 12
    prompts = prompts[:MAX_PROMPTS]; srcs = srcs[:MAX_PROMPTS]; refs = refs[:MAX_PROMPTS]

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
            if r.stderr and ("error" in r.stderr.lower() or "failed" in r.stderr.lower()):
                print(f"  ⚠ stderr: {r.stderr[:200]}")
            if "Turkish:" in out:
                out = out.split("Turkish:")[-1].strip()
            hyps.append(out)
            out_toks = max(len(out.split()) * 1.3, 1)
            total_tokens += int(out_toks); bn += 1
            elapsed = time.monotonic() - t0
            print(f"  [{elapsed:3.0f}s] prompt={pi+1} tok≈{int(out_toks)} tps≈{out_toks/max(elapsed,0.01):.0f}")
        except subprocess.TimeoutExpired:
            print(f"  ⚠ prompt {pi} timeout"); hyps.append("")
        except Exception as e:
            print(f"  ⚠ prompt {pi} err: {e}"); hyps.append("")

    dur = time.monotonic() - wall_start
    quality = {}
    if hyps and any(h for h in hyps):
        try:
            bs = compute_bertscore(refs[:len(hyps)], hyps)
            quality = {"bertscore": bs.get("system_score"),
                       "num_references": len(refs[:len(hyps)])}
            print(f"  BERTScore: {quality['bertscore']:.4f}")
        except Exception as e:
            print(f"  Quality error: {e}")

    return {
        "model": name, "model_path": str(gguf_path),
        "backend_type": "llama.cpp" + ("_diffusion" if is_diffusion else "_text"),
        "mean_tps": round(total_tokens / max(dur, 0.1), 1),
        "batches_completed": bn,
        "total_tokens_translated": total_tokens,
        "run_duration_seconds": round(dur, 1),
        "bertscore": quality.get("bertscore"),
        "quality_num_refs": quality.get("num_references", 0),
        "platform": "mps",
    }


# ═════════════════════════════════════════════════════════════════════════════
# Complete model list
# ═════════════════════════════════════════════════════════════════════════════

PYTHON_MODELS = [
    # ── NLLB family — proven translators ──
    {"name": "NLLB-200-distilled-600M", "path": "facebook/nllb-200-distilled-600M",
     "backend_type": "encoder_decoder",
     "extra": {"nllb_source_lang": "eng_Latn", "nllb_target_lang": "tur_Latn", "num_beams": 1}},
    {"name": "NLLB-200-distilled-1.3B", "path": "facebook/nllb-200-distilled-1.3B",
     "backend_type": "encoder_decoder",
     "extra": {"nllb_source_lang": "eng_Latn", "nllb_target_lang": "tur_Latn", "num_beams": 1}},
    {"name": "NLLB-200-3.3B", "path": "facebook/nllb-200-3.3B",
     "backend_type": "encoder_decoder",
     "extra": {"nllb_source_lang": "eng_Latn", "nllb_target_lang": "tur_Latn", "num_beams": 1}},
    # ── Autoregressive — proven translators ──
    {"name": "SmolLM2-1.7B-Instruct", "path": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
     "backend_type": "auto"},
    {"name": "TranslateGemma-4B", "path": "google/translategemma-4b-it",
     "backend_type": "auto"},
    # ── QAT CT models (compressed-tensors) — may fail on MPS ──
    {"name": "Gemma-4-E2B-QAT-mobile-ct", "path": "google/gemma-4-E2B-it-qat-mobile-ct",
     "backend_type": "auto", "expect_failure": True,
     "extra": {"quantization": "bf16"}},
    {"name": "Gemma-4-E4B-QAT-mobile-ct", "path": "google/gemma-4-E4B-it-qat-mobile-ct",
     "backend_type": "auto", "expect_failure": True,
     "extra": {"quantization": "bf16"}},
    # ── QAT mobile-transformers (Q4_0 weights) — may fail on MPS ──
    {"name": "Gemma-4-E2B-QAT-mobile-transformers", "path": "google/gemma-4-E2B-it-qat-mobile-transformers",
     "backend_type": "auto", "expect_failure": True,
     "extra": {"quantization": "bf16"}},
    {"name": "Gemma-4-E4B-QAT-mobile-transformers", "path": "google/gemma-4-E4B-it-qat-mobile-transformers",
     "backend_type": "auto", "expect_failure": True,
     "extra": {"quantization": "bf16"}},
    # ── QAT W4A16 CT models — may fail on MPS ──
    {"name": "Gemma-4-E2B-QAT-w4a16-ct", "path": "google/gemma-4-E2B-it-qat-w4a16-ct",
     "backend_type": "auto", "expect_failure": True,
     "extra": {"quantization": "int4"}},
    {"name": "Gemma-4-E4B-QAT-w4a16-ct", "path": "google/gemma-4-E4B-it-qat-w4a16-ct",
     "backend_type": "auto", "expect_failure": True,
     "extra": {"quantization": "int4"}},
]

LLAMA_MODELS_DEF = [
    # ── Google QAT GGUF models ──
    {"name": "Gemma-4-E2B-QAT-Q4_0-GGUF", "hf_repo": "google/gemma-4-E2B-it-qat-q4_0-gguf",
     "hf_file": "gemma-4-E2B_q4_0-it.gguf",
     "path": str(GOOGLE_QAT_DIR / "gemma-4-E2B_q4_0-it.gguf"),
     "llama_binary": str(LLAMA_CLI), "ngl": "all",
     "llama_args": {"ctx_size": 4096, "n_predict": 256, "temp": 0.0}},
    {"name": "Gemma-4-E4B-QAT-Q4_0-GGUF", "hf_repo": "google/gemma-4-E4B-it-qat-q4_0-gguf",
     "hf_file": "gemma-4-E4B_q4_0-it.gguf",
     "path": str(GOOGLE_QAT_DIR / "gemma-4-E4B_q4_0-it.gguf"),
     "llama_binary": str(LLAMA_CLI), "ngl": "all",
     "llama_args": {"ctx_size": 4096, "n_predict": 256, "temp": 0.0}},
    {"name": "Gemma-4-26B-A4B-QAT-Q4_0-GGUF", "hf_repo": "google/gemma-4-26B-A4B-it-qat-q4_0-gguf",
     "hf_file": "gemma-4-26B_q4_0-it.gguf",
     "path": str(GOOGLE_QAT_DIR / "gemma-4-26B_q4_0-it.gguf"),
     "llama_binary": str(LLAMA_CLI), "ngl": "all",
     "llama_args": {"ctx_size": 4096, "n_predict": 256, "temp": 0.0}},
    # ── DiffusionGemma (existing GGUF) ──
    {"name": "DiffusionGemma-26B-A4B-Q8_0", "path": str(LLAMA_MODELS / "diffusiongemma-26B-A4B-it-Q8_0.gguf"),
     "llama_binary": str(LLAMA_DIFFUSION_CLI), "is_diffusion": True, "ngl": "all",
     "llama_args": {"diffusion_steps": 64, "diffusion_algorithm": 4,
                    "ctx_size": 4096, "n_predict": 256, "temp": 0.8}},
]


def main():
    plat = detect_backend("auto")
    print("=" * 80)
    print(f"  TR Corpus Translation Benchmark — Complete Model Comparison")
    print(f"  Platform: {plat.backend} | Torch: {torch.__version__}")
    print(f"  Python models: {len(PYTHON_MODELS)} | llama.cpp models: {len(LLAMA_MODELS_DEF)}")
    print(f"  Total: {len(PYTHON_MODELS) + len(LLAMA_MODELS_DEF)} models")
    print(f"  Duration/model: {RUN_DURATION}s translation + quality eval")
    print(f"  Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 80)

    results = []
    total_start = time.monotonic()

    # ── Phase 1: Python models (each in subprocess for MPS memory isolation) ──
    for i, md in enumerate(PYTHON_MODELS, 1):
        name = md["name"]
        print(f"\n{'─'*60}")
        print(f"  [{i}/{len(PYTHON_MODELS)} Python] {name}")
        print(f"{'─'*60}")

        model_spec = json.dumps(md)
        t0 = time.monotonic()
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["TRANSFORMERS_VERBOSITY"] = "error"

        try:
            proc = subprocess.run(
                [sys.executable, "-c", f"""
import json, sys
sys.path.insert(0, '{PROJECT_ROOT}')
from scripts.bench_full import run_python_model
md = json.loads('''{model_spec}''')
result = run_python_model(md)
print('__RESULT_JSON_START__')
print(json.dumps(result, indent=2))
print('__RESULT_JSON_END__')
"""],
                capture_output=True, text=True, timeout=600,
                env=env, cwd=str(PROJECT_ROOT),
            )
            output = proc.stdout
            # Extract JSON between markers
            if "__RESULT_JSON_START__" in output:
                json_str = output.split("__RESULT_JSON_START__")[1].split("__RESULT_JSON_END__")[0]
                result = json.loads(json_str)
            else:
                result = {"model": name, "error": f"No output. stderr: {proc.stderr[:300]}"}
        except subprocess.TimeoutExpired:
            result = {"model": name, "error": "Timeout (600s)"}
        except Exception as e:
            result = {"model": name, "error": str(e)[:200]}

        result.setdefault("model", name)
        result.setdefault("platform", plat.backend)
        results.append(result)

        if result.get("error"):
            print(f"  ✗ {result['error'][:100]}")
        else:
            tps = result.get("mean_tps", 0)
            bert = result.get("bertscore") or 0
            print(f"  ✓ TPS={tps:.1f}  BERTScore={bert:.4f}")

        # Memory cleanup
        gc.collect()
        try: torch.mps.empty_cache()
        except (RuntimeError, AttributeError): pass  # MPS empty_cache is best-effort
        time.sleep(2)

    # ── Phase 2: llama.cpp models ──
    for i, md in enumerate(LLAMA_MODELS_DEF, 1):
        name = md["name"]
        print(f"\n{'─'*60}")
        print(f"  [{i}/{len(LLAMA_MODELS_DEF)} llama.cpp] {name}")
        print(f"{'─'*60}")

        # Check if GGUF file exists or can be downloaded
        gguf_path = Path(md["path"])
        if not gguf_path.exists() and md.get("hf_repo"):
            print(f"  Downloading {md['hf_repo']}:{md['hf_file']}...")
            try:
                from huggingface_hub import hf_hub_download
                gguf_path.parent.mkdir(parents=True, exist_ok=True)
                hf_hub_download(
                    repo_id=md["hf_repo"], filename=md["hf_file"],
                    local_dir=str(gguf_path.parent), resume_download=True,
                )
            except Exception as e:
                results.append({"model": name, "error": f"Download failed: {e}"})
                print(f"  ✗ Download failed: {e}")
                continue

        if not gguf_path.exists():
            results.append({"model": name, "error": f"GGUF not found: {gguf_path}"})
            print(f"  ✗ GGUF not found: {gguf_path}")
            continue

        result = run_llama_model(md)
        result.setdefault("model", name)
        result.setdefault("platform", plat.backend)
        results.append(result)

        if result.get("error"):
            print(f"  ✗ {result['error'][:100]}")
        else:
            tps = result.get("mean_tps", 0)
            bert = result.get("bertscore") or 0
            print(f"  ✓ TPS={tps:.1f}  BERTScore={bert:.4f}")

        gc.collect()
        try: torch.mps.empty_cache()
        except (RuntimeError, AttributeError): pass  # MPS empty_cache is best-effort
        time.sleep(2)

    # ──────────────────────────────────────────────────────────────────────
    # Final report
    # ──────────────────────────────────────────────────────────────────────
    total_elapsed = time.monotonic() - total_start
    print(f"\n{'='*80}")
    print(f"  BENCHMARK COMPLETE — {total_elapsed/60:.1f} minutes")
    print(f"{'='*80}")

    # Results table
    hdr = f"\n{'MODEL':<40} {'TPS':>8} {'Lat(ms)':>9} {'B':>4}  {'BERTScore':>10} {'Load(s)':>8}  Status"
    print("=" * len(hdr)); print(hdr); print("=" * len(hdr))

    ok = 0; fail = 0
    for r in results:
        n = r.get("model", "?")[:39]
        if r.get("error"):
            err = r['error'][:40]
            print(f"{n:<40} {'—':>8} {'—':>9} {'—':>4}  {'—':>10} {'—':>8}  ✗ {err}")
            fail += 1
        else:
            tps = r.get("mean_tps", 0)
            lat = r.get("mean_latency_ms", 0)
            bs_b = r.get("batch_size", "?")
            bert = r.get("bertscore") or 0
            load = r.get("load_seconds", 0)
            print(f"{n:<40} {tps:>8.1f} {lat:>9.0f} {str(bs_b):>4}  {bert:>10.4f} {load:>8.1f}  ✓")
            ok += 1
    print("=" * len(hdr))
    print(f"  ✓ {ok} succeeded  ✗ {fail} failed  Total: {ok+fail}")

    # Rankings
    ok_results = [r for r in results if not r.get("error")]
    if ok_results:
        by_tps = sorted(ok_results, key=lambda r: r.get("mean_tps", 0), reverse=True)
        by_bert = sorted([r for r in ok_results if r.get("bertscore")],
                        key=lambda r: r.get("bertscore", 0), reverse=True)

        print(f"\n══ RANKINGS ══")
        print(f"\n  By TPS (tokens/sec):")
        for j, r in enumerate(by_tps[:5], 1):
            print(f"    {j}. {r['model']:<40} {r.get('mean_tps',0):.0f} tok/s")
        if by_bert:
            print(f"\n  By BERTScore (quality):")
            for j, r in enumerate(by_bert[:5], 1):
                print(f"    {j}. {r['model']:<40} {r.get('bertscore',0):.4f}")

    # Write comprehensive JSON report
    out_path = Path("data/output") / "model_comparison.json"
    report = {
        "title": "TR Corpus Translation — Complete Model Comparison",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "platform": plat.backend,
        "pytorch_version": torch.__version__,
        "config": {"run_duration_per_model_s": RUN_DURATION, "quality_max_references": QUALITY_MAX_REFS},
        "models_tested": len(results), "models_succeeded": ok, "models_failed": fail,
        "total_wallclock_s": round(total_elapsed, 1),
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Full report: {out_path}")

    return 1 if fail > 0 else 0


if __name__ == "__main__":
    # If called with a specific model name, run just that one
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        model_name = sys.argv[1]
        for md in PYTHON_MODELS:
            if md["name"] == model_name:
                result = run_python_model(md)
                print(json.dumps(result, indent=2))
                break
        else:
            for md in LLAMA_MODELS_DEF:
                if md["name"] == model_name:
                    result = run_llama_model(md)
                    print(json.dumps(result, indent=2))
                    break
            else:
                print(f"Unknown model: {model_name}")
                sys.exit(1)
    else:
        sys.exit(main())
