# Architecture — TR Corpus Translation Benchmark

> **Purpose:** The reality-grounded description of how this codebase actually runs.
> **Status:** Engineering reality — current as of v3.8 (June 2026).
> **Audience:** Engineers and LLM coding agents working on this repo.
>
> ⚠️ **Read this before editing `benchmark/inference/` or `benchmark/hardware/`.**
> This is the single source of truth for what is wired vs. gated. The older
> `PRD%26SDD%26SRS/PRD.md` / `PRD%26SDD%26SRS/SRS.md` describe design *intent*; this document describes *current
> reality*. Where they disagree, **this document is correct.**

---

## Table of Contents

1. [Mental Model](#1-mental-model)
2. [Runtime Hot Path](#2-runtime-hot-path)
3. [The Backend Protocol](#3-the-backend-protocol)
4. [Backend Dispatch (ModelRegistry)](#4-backend-dispatch-modelregistry)
5. [Per-Backend Reality](#5-per-backend-reality)
6. [Module Map](#6-module-map)
7. [The Two Optimization Stacks](#7-the-two-optimization-stacks)
8. [Feature Status (the truth table)](#8-feature-status-the-truth-table)
9. [Known Correctness Risks](#9-known-correctness-risks)

---

## 1. Mental Model

This benchmark answers one question: *how many days to translate ~200B English
tokens into Turkish on 2× NVIDIA H200, at academic quality?*

It is **backend-dispatched**: one pipeline drives autoregressive (AR), encoder-decoder
(NLLB), diffusion, and custom-plugin backends through a single
`InferenceBackend` protocol, then measures throughput, extrapolates to
corpus-completion time, and scores quality.

**The single most important thing to understand about this codebase:** the
documentation historically advertised "39 optimizations (37 wired)," but the
production AR hot path is, in reality, **plain eager `model(...)` in a Python
loop with HuggingFace `past_key_values`**, accelerated only by:

- **Static FP8 weight quantization** (weights in float8_e4m3fn, dequantized on-chip
  at forward time — 2× memory bandwidth, zero per-token overhead),
- **Pre-tokenized Parquet cache** (skips CPU tokenization, +60% TPS),
- **TF32 + Flash SDPA** (always on for CUDA),
- **BF16 activations** (native H200 dtype), and
- **`torch.compile(mode="default")`** on PyTorch 2.12–2.13, or
  **`torch.compile(mode="reduce-overhead")`** on PyTorch 2.14+
  (cudagraph_trees KV-cache bug persists through 2.12.1).

Additionally, two advanced optimizations have been integrated:
- **FlashAttention-3** (Method 5) — Hopper-optimized attention kernels (SM90 WGMMA/TMA), auto-dispatched by PyTorch SDPA on CUDA, active when `flash-attn-3` is installed and `--use-flash-attention` is set.


Almost every other "optimization" is **built but gated off** — hardcoded `False`,
commented out, env-gated, safety-gated, or captured-but-never-replayed. The
[Feature Status table](#8-feature-status-the-truth-table) is the authoritative
list. **Do not reason about performance from the README's optimization count.**

---

## 2. Runtime Hot Path

```
python -m benchmark --config config.yaml
  └─ benchmark/__main__.py            CLI parsing; injects flag overrides into a temp YAML
     └─ BenchmarkHarness.run()        benchmark/orchestration/harness.py:143
        ├─ detect_backend(config.backend)  → DeviceInfo           hardware/backend.py:104
        ├─ run_preflight_checks(...)        utils/env_check.py
        ├─ InferenceEngine(...)             inference/engine.py:66
        │   └─ ModelRegistry().create_backend(config)             inference/backends/registry.py:171
        │       └─ <backend>.load()                              (see §5)
        ├─ backend.warmup(batches=10..20)
        ├─ BatchSizeTuner.tune(...) → batch_size                 inference/batch_tuner.py:71
        ├─ dispatch by run_mode:
        │   ├─ benchmark-only → _run_quality_only
        │   ├─ warmup-only    → warmup then exit
        │   ├─ resume         → _run_resume → _run_translation_core
        │   ├─ CB path*       → _run_continuous_batching_loop     harness.py:680
        │   └─ default        → _run_translation_loop → _run_translation_core  harness.py:328/347
        │
        │   * CB path only when: CUDA + use_continuous_batching + use_paged_attention
        │
        ├─ TRANSLATION LOOP (_run_translation_core, harness.py:347):
        │   while timer < target_duration:
        │     batch = AsyncPipeline.next_batch()        data/pipeline.py:313
        │     result = engine.translate(batch)          → backend.translate_batch()
        │     MetricsCollector.log_batch(result)        metrics/collector.py
        │     checkpoint every checkpoint_interval_seconds (default 300s)
        │   finally: stop metrics, flush, save final checkpoint
        ├─ QualityBenchmark.run(engine)  (skipped for translate-only / signal-killed / OOM-aborted)
        │     quality/benchmark.py:328  → BERTScore ‖ COMET-22 ‖ COMET-Kiwi ‖ MetricX-24 ‖ BLEU ‖ chrF++
        └─ Reports: MetricsAggregator → ExtrapolationModel → JSON + Markdown
```

**Run modes:** `full` (default, config duration), `quick` (300s), `dry-run` (60s),
`warmup-only`, `benchmark-only` (quality only), `translate-only` (skip quality),
`resume` (from a checkpoint dir). Duration resolution: `harness.py:316`.

**OOM recovery:** CUDA `OutOfMemoryError`, CPU `MemoryError`, and MPS-OOm
`RuntimeError` are all routed through `_handle_oom` (`harness.py:1060`), which
flushes a checkpoint, halves the batch size (floor 1), re-warms, and retries;
aborts only when already at batch size 1.

**Resume:** `CheckpointManager.load_latest()` (`orchestration/checkpoint.py:93`)
restores `batches_completed`, `total_tokens`, `current_file_name`,
`current_doc_id`, `elapsed_seconds`; the loader seeks to the doc position and the
remaining duration budget is `full_duration - previously_elapsed` (floor 60s).

---

## 3. The Backend Protocol

Every backend implements `InferenceBackend` (`inference/backends/protocol.py:174`):

```python
class InferenceBackend(ABC):
    model_type: ModelType          # AUTOREGRESSIVE | ENCODER_DECODER | DIFFUSION | CUSTOM
    capabilities: ModelCapability  # IntFlag bitmask
    display_name: str

    def __init__(self, config: BackendConfig): ...   # sets _configured_batch_size, devices, tokenizer, model
    @abstractmethod def load(self) -> None: ...      # load weights/tokenizer → device; set _loaded=True
    @abstractmethod def warmup(self, batches=20) -> None: ...
    @abstractmethod def translate_batch(self, batch) -> BatchGenerationOutput: ...
    def is_loaded(self) -> bool: ...
    # optional: encode_source, score_candidates, get_token_log_probs, kv_cache_config
```

**The harness and quality benchmark interact ONLY through this protocol** — they
never call `model.generate()` or backend-specific methods directly.

`BatchGenerationOutput` (`protocol.py:111`): `batch_id, generations[],
batch_size, input_tokens_total, output_tokens_total, total_latency_ms,
phase_timings`. Backends diverge in `phase_timings` (AR: `prefill_ms/decode_ms`;
NLLB: empty; diffusion: `encode_ms/denoise_ms`).

**Contract for new backends** (see also `docs/DEVELOPMENT.md`):

- Set `self._configured_batch_size` in `__init__` (the engine reads it via a
  property with **no default** — a missing attribute is a real bug, not a
  fallback case; `engine.py:148`).
- Return `input_tokens_total` computed from `attention_mask.sum()`, **not**
  padded `input_ids` length (see [§9](#9-known-correctness-risks)).
- `load()` must set `self._loaded = True` on success.

---

## 4. Backend Dispatch (ModelRegistry)

`ModelRegistry.create_backend` (`inference/backends/registry.py:171`) dispatches
in this fixed priority order:

1. **Explicit override** (`registry.py:214`) — `extra["backend_type"]` (except
   `"auto"`), validated against `ModelType`
2. **Auto-detect** (`registry.py:226`) via `_detect_model_type(model_path)`:
   - name contains `nllb`/`madlad` → `ENCODER_DECODER`
   - name matches a `DIFFUSION_KEYWORDS` (`llada, dream, mdlm, e2d2, bd3lm, diffusiongemma, …`, `constants.py:125`)
   - local `config.json` `model_type`/`architectures`/diffusion config keys
   - HF Hub `config.json` (same signals; `trust_remote_code=False`, 3 retries)
   - else `AUTOREGRESSIVE`
3. **Custom plugin** (`registry.py:233`) — `PluginRegistry.lookup()`; first plugin
   whose `detect(model_path)` is True. (Discovery is gated behind
   `TR_ALLOW_UNTRUSTED_PLUGINS=1`; see `custom_plugin.py`.)
4. **Fallback** → `AutoregressiveBackend` (including NLLB auto-detection).

Auto-detection results are cached in a 100-entry FIFO cache (`registry.py:54`).

---

## 5. Per-Backend Reality

### 5.1 AutoregressiveBackend (`inference/backends/autoregressive.py`)

A system-agnostic dispatcher wrapper. At runtime, it detects the compute hardware and delegates all execution to either `AutoregressiveCUDABackend` (`autoregressive_cuda.py`) or `AutoregressiveMPSBackend` (`autoregressive_mps.py`) via transparent delegation (`__getattr__`/`__setattr__`).

The underlying implementation subclasses have `model_type = AUTOREGRESSIVE`; capabilities `TRANSLATE | FORWARD_ENCODE | QUANTIZABLE_KV | SPECULATIVE | ENSEMBLE_READY`.

**`load()` actually does:**

1. Devices (`cuda:{i}` / `mps` / `cpu`), NCCL P2P enable on CUDA. **cudaMallocAsync enabled when safe** (active when `torch.compile` mode is NOT `reduce-overhead`, at `autoregressive_cuda.py:806`).
2. Tokenizer via `AutoTokenizer`, left padding, Gemma-4-QAT list-tokenizer fix.
3. Flash + mem-efficient SDPA on CUDA. Under Hopper architecture, checks for `flash-attn` package version 3 to enable Hopper-optimized FlashAttention-3.
4. Model load: QAT/Gemma-4/Q4_0 → `_try_load_qat_model`; else `_load_standard_model`
   (single-GPU fast path when model < 10% of one GPU's memory; else
   `device_map="auto"`).
5. Static FP8 weight quantization — **always on for CUDA** (not safe_mode).
   `StaticFP8Linear` replaces nn.Linear with weights stored in FP8 E4M3.
   Dequantized on-chip at forward time — zero per-token overhead, 2× memory
   bandwidth.  TE fused kernel attempted first; static is the fallback.
6. `torch.compile` (CUDA only; skipped on MPS/CPU/safe-mode). Version-gated:
   **< 2.12** → skipped (eager). **2.12–2.13** → `mode="default"` (inductor fusion,
   no cudagraph_trees — warmup takes 30s but decode is stable).
   **≥ 2.14** → `mode="reduce-overhead"` (frame-level CUDA graphs).
   `_apply_extreme_compile` in `autoregressive_cuda.py:1080`.
   **Measured (2026-06): 37,503 tok/s** (NLLB-600M, bs=1024, 1×H200, compile).
7. JIT kernel precompile — 🗑 REMOVED v3.7.
8. PagedAttention init — **opt-in** via `--paged-attention` (reads `extra.get("use_paged_attention", False)` at `autoregressive_cuda.py:539`).
9. FP8 KV-cache — **removed** (empirically showed 0% speedup for NLLB-600M / TranslateGemma 4B; cast/dequant overhead exactly cancelled bandwidth savings at these model sizes).
10. Speculative decoder — **opt-in** via `use_speculative=True` config flag or `--speculative` CLI. No env var required.

**`translate_batch` (`:1369`):** if speculative decoder is active, delegates
entirely to it (`:1385`); else async H2D on a transfer stream (CUDA), then
`_extreme_decode` (CUDA) or `_standard_decode` (MPS/CPU).

**`_extreme_decode` (`:1482`) — the real hot path:**
- **Prefill** (`:1521`): one `model(input_ids, attention_mask, position_ids,
  use_cache=True)` → `past_kv`. `position_ids = cumsum(attention_mask)-1` (handles
  left padding).
- **Decode loop** (`:1548`): a plain Python loop calling
  `model(next_input, past_key_values=past_kv, use_cache=True)` per token;
  `argmax` greedy; per-item EOS/EOT detection; breaks when all done.
  The comment at `:1548` is explicit: *"DECODE LOOP (standard forward, no graph
  replay)"*. **The captured CUDA graph is never replayed here.**
- Timings via CUDA events; returns `BatchGenerationOutput`.

`_standard_decode` (MPS/CPU, `:1602`) uses HF `model.generate(...)`.

### 5.2 NLLBBackend (`inference/backends/nllb.py`)

A system-agnostic dispatcher wrapper. At runtime, it detects the compute hardware and delegates all execution to either `NLLBCUDABackend` (`nllb_cuda.py`) or `NLLBMPSBackend` (`nllb_mps.py`) via transparent delegation (`__getattr__`/`__setattr__`).

The underlying implementation subclasses have `model_type = ENCODER_DECODER`; capabilities `TRANSLATE | FORWARD_ENCODE | ENSEMBLE_READY`.

Encoder-decoder (NLLB-200 / M2M100 / MADLAD). **v3.9:** The CUDA backend uses a
custom `_fast_decode_batch` loop — encoder runs once, then a tight per-token
greedy decoder loop with pre-allocated buffers and vectorized EOS detection.
This eliminates the ~26.8ms of HF `model.generate()` Python overhead per batch.
The MPS backend continues to use `model.generate()`.

`input_tokens_total` correctly uses `attention_mask.sum()` (the
padding bug was fixed here in commit `ffa707b`). `torch.compile(reduce-overhead)`
on CUDA. `forced_bos_token_id`
from `tgt_lang` (default `tur_Latn`); falls back to `None` if unresolved (may produce
wrong-language output).

### 5.3 DiffusionBackend — 🗑 REMOVED v3.9

The diffusion backend (`inference/backends/diffusion.py`, 1,030 lines) was
permanently removed in v3.9.  It was experimental and never on the production
benchmark path.  ``ModelType.DIFFUSION`` is retained in the protocol for
compatibility.

### 5.4 TensorRTBackend — 🗑 REMOVED v3.7

The TensorRT backend and builder were permanently deleted in v3.7
(`tensorrt_backend.py` 459L + `trt_builder.py` 727L).  TRT decode had no
KV-cache passthrough (broken by design), was safety-gated to raise
``RuntimeError``, and broke on TRT 10+.  ``ModelType.AUTOREGRESSIVE``
fallback was used instead.

### 5.5 Custom Plugin — 🔬 Gated behind `TR_ALLOW_UNTRUSTED_PLUGINS=1`

Drop a `.py` defining a `CustomModelPlugin` subclass in
`~/.tr_benchmark/plugins/`, `TR_BENCHMARK_PLUGIN_PATH`, entry-points, or
`./plugins/`.  Discovery is gated — plugins run with full process privileges,
no sandbox.  Explicit `register_plugin()` bypasses the gate.
See ``inference/backends/custom_plugin.py`` for the plugin API.

### 5.6 vLLMBackend — 🗑 REMOVED v3.9

The vLLM backend (`inference/backends/vllm.py`, 174 lines) was permanently
removed in v3.9.  It was disabled due to CUDA version conflicts (vLLM wheels
compile for CUDA 13.x; the H200 host uses CUDA 12.x).  ``ModelType.VLLM`` is
retained in the protocol for compatibility.

---

## 6. Module Map

| Subsystem | Key files |
|---|---|
| **Entry / orchestration** | `benchmark/__main__.py` · `orchestration/harness.py` · `orchestration/checkpoint.py` · `orchestration/signals.py` |
| **Inference engine** | `inference/engine.py` · `inference/backends/{protocol,registry,autoregressive,nllb,custom_plugin}.py` · `inference/sampling.py` |
| **Inference optimizations** | `inference/speculative.py` · `inference/paged_attention.py` · `inference/continuous_batcher.py` · `inference/batch_tuner.py` |
| **Hardware** | `hardware/backend.py` · `hardware/precision.py` · `hardware/parallelism.py` · `hardware/architecture.py` |
| **Quantization** | `quantization/smoothquant.py` (default on CUDA) · `quantization/qat.py` |
| **Data pipeline** | `data/loader.py` · `data/chunker.py` · `data/filters.py` · `data/pipeline.py` · `data/pretokenizer.py` · `data/parallel_gz.py` |
| **Quality** | `quality/benchmark.py` · `quality/{metrics_bertscore,metrics_comet,metrics_bleu,metrics_chrf,references}.py` |
| **Metrics** | `metrics/collector.py` · `metrics/throughput.py` · `metrics/gpu_sampler.py` · `metrics/system_sampler.py` · `metrics/batch_logger.py` |
| **Observability** | `observability/prometheus_metrics.py` |
| **Reporting** | `reporting/aggregator.py` · `reporting/extrapolation.py` · `reporting/degradation.py` · `reporting/json_report.py` · `reporting/markdown_report.py` |
| **Config** | `config/schema.py` · `config/capability.py` · `config/constants.py` · `config/model_presets.py` |

---

## 7. The Two Optimization Stacks

A subtle point that explains much of the doc-vs-reality confusion: there are
**two parallel, decoupled optimization stacks** that do not share state.

- **Stack A — the AR backend** (`autoregressive.py`): owns `_paged_kv` (opt-in
  via config), `_spec_decoder` (active when `use_speculative=True`), FP8
  (static weight-only, dequant-on-read), `torch.compile` (mode=default on
  PT≥2.12), `cudaMallocAsync` (active when compile≠reduce-overhead), and
  Flash SDPA. In v3.7, five previously-dead Stack A features were permanently
  deleted: CUDA graph capture, fused Triton kernels, JIT CUDA/Metal kernels,
  INT8 KV-cache quantization, and TensorRT.

- **Stack B — the harness-owned ContinuousBatcher path** (`harness.py` +
  `continuous_batcher.py` + `paged_attention.py`): the batcher builds its **own**
  `PagedKVCache` and a `PagedCache` (`DynamicCache`-compatibility shim) and passes
  it as `past_key_values` to `engine.model(...)` directly. **This is the only place
  where paged KV is fed to a model forward.** Gated behind `--continuous-batching
  --paged-attention` (CUDA). CB skips `torch.compile` because PagedCache.update()
  is not Dynamo-traceable.

---

## 8. Feature Status (the truth table)

**Legend:** ✅ Wired & on the hot path · 🟡 Built but gated off · 🔬 Experimental ·
⚠️ Broken/Disabled · 💀 Dead code (never called). Cite `file:line` to verify.

### Compute / decode

| # | Feature | Status | How to activate | Evidence | Notes |
|---|---|---|---|---|---|
| 1 | `torch.compile` | ⚠️ | on by default (CUDA); version-gated | `autoregressive_cuda.py:1080` | **< 2.12** → skipped (eager). **2.12–2.13** → `mode="default"` (stable, warmup 30s). **≥ 2.14** → `mode="reduce-overhead"`. No-compile baseline 37,456 tok/s (NLLB-600M, bs=1024, 1×H200). compile benefit negligible for encoder-decoder at this scale. |
| 2 | Static FP8 weight quantization | ✅ | on by default (CUDA); `TR_SKIP_FP8=1` to disable | `autoregressive_cuda.py:1767` | `StaticFP8Linear` — weights in FP8 E4M3, dequantized on-chip at forward time. Zero per-token overhead. 2× memory bandwidth vs BF16. lm_head excluded. TE fused kernel attempted first (best perf); static as fallback. |
| 2b | SmoothQuant FP8 calibration | ✅ | auto on CUDA (`TR_SKIP_SMOOTHQUANT=1` to skip) | `autoregressive_cuda.py:710` | Calibrates before static FP8 to migrate activation outliers into weights. 238 Linear layers smoothed at alpha=0.50. |
| 2c | Pre-tokenized Parquet cache | ✅ ENFORCED | Enforced (`~/.cache/tr_benchmark/pretokenized/`) | `data/pretokenizer.py` | Strictly required in v3.8. Bypasses dynamic tokenization. Compiled dynamically at startup if cache is missing. |
| 3 | Flash + mem-efficient SDPA | ✅ | CUDA default | `autoregressive_cuda.py:637` | 1.17-1.23× throughput. Attention-only speedup higher but attention ~20-30% of compute for 4B. |
| 3b | FlashAttention-3 | 🔬 | `--use-flash-attention` (with `flash-attn-3` package installed) | `autoregressive_cuda.py:643` | Hopper-optimized FlashAttention-3 kernels (WGMMA/TMA) wired on SM90 GPUs (H100/H200). Auto-detects FA3 package and GPU arch; falls back to standard Flash SDPA on non-Hopper. |
| 4 | CUDA Graph manual capture | 🗑 REMOVED v3.7 | — | `cuda_graphs.py` (389L) deleted in commit `19d979f` | Past-key-values can't be a static input. `torch.compile` handles internal graph capture correctly. |
| 5 | Fused Triton kernels | 🗑 REMOVED v3.7 | — | `fused_ops.py` (303L) + `triton_kernels_fused.py` (238L) deleted in `19d979f` | Crashed outside inductor graphs ("cpu tensor?" pointer error). `torch.compile` fuses kernels internally. |
| 6 | JIT CUDA C++ kernels | 🗑 REMOVED v3.7 | — | `jit_compiler.py` (678L) deleted in `926855e` | Both kernel sources were `None`. Metal wrapper returned inputs unchanged. |
| 7 | JIT Metal RMSNorm | 🗑 REMOVED v3.7 | — | Deleted with `jit_compiler.py` | Was a no-op (`return inputs[0]`). |
| 8 | cudaMallocAsync | ✅ | automatic when compile≠reduce-overhead | `autoregressive_cuda.py:750` | Enabled when `_compile_uses_graphs=False`. Stream-ordered allocation, zero-fragmentation. PT 2.12: compile uses `mode=default` (safe). |
| 9 | INT8 KV-cache quantization | 🗑 REMOVED v3.7 | — | `kv_cache_quant.py` (289L) deleted in `7b1ef87` | Unnecessary on H200 (141 GB, 4B model uses ~8 GB). |
| 10 | Speculative decoding | 🔬 | `--speculative` (no env var needed) | `speculative.py`; `autoregressive_cuda.py:766` | **v3.7:** Env gate removed. Verify runs **layers[D:L]** (not full model) — ~25% compute savings. Greedy-only, per-sequence. 8 draft / 26 verify layers for Gemma 34-layer. K=3. |
| 14 | TensorRT backend | 🗑 REMOVED v3.7 | — | `tensorrt_backend.py` (459L) + `trt_builder.py` (727L) deleted | No KV-cache passthrough → corrupted decode. Safety-gated to raise RuntimeError. Broken on TRT 10+. |
| 15 | Capability Registry | ✅ | Automatic in `load()` | `autoregressive_cuda.py:847`; `config/capability.py` | 14 features tracked with verified `ActivationState` (ACTIVE/INERT/BROKEN/UNKNOWN). Calls `reg.report_text()` at startup. Single source of truth for what's on the hot path — supersedes old manual log lines. |

### Memory / KV

| # | Feature | Status | How to activate | Evidence | Notes |
|---|---|---|---|---|---|
| 16 | PagedAttention (AR path) | 🟡 GATED | `--paged-attention` (CUDA) | `autoregressive_cuda.py:539`: `extra.get("use_paged_attention", False)` | Opt-in via config flag. Writes prefill KV into paged blocks, passes PagedCache as past_key_values. Opt-in because block pool is sized for CB (8192 blocks × 16 tokens), not large static batches. |
| 17 | PagedAttention (ContinuousBatcher path) | ✅ (gated) | `--continuous-batching --paged-attention` (CUDA) | `continuous_batcher.py:945`; `paged_attention.py` `PagedCache` shim | Production-quality chunked-prefill scheduler. 8192-block pool on H200. CB skips torch.compile (PagedCache non-Dynamo-traceable). |
| 18 | Continuous batching | ✅ (gated) | `--continuous-batching --paged-attention` (CUDA) + batch ≥ **2** | `continuous_batcher.py:100` (`MIN_BATCH_SIZE_FOR_CONTINUOUS=2`); `harness.py:311` | Production-quality chunked-prefill scheduler. **Note: the real threshold is 2, not 8 as old docs claim.** |
| 19 | Pinned-memory pipeline | ✅ | CUDA | `data/pipeline.py:183` (`_use_pinned = (backend == "cuda")`) | Disabled on MPS (unified memory). — measured 2026-06-24: 2.1× H2D speedup, ~6.6 GB/s effective bandwidth. See M2.5. |
| 20 | INT4/INT8 weight quantization | ✅ (gated) | `--quantization int8|int4` / QAT presets | `autoregressive_cuda.py:540` (`extra.get("use_quantized_weights", False)`), bitsandbytes | Separate axis from FP8 compute. — measured 2026-06-24: 41% memory savings (matches ~2× smaller). BUT throughput is 3.7× slower (213 vs 792 tok/s) — counterproductive on H200 unless VRAM-constrained. See M2.7. |
| 20b | FP8 KV-Cache Quantization | 🗑 **REMOVED** | — | `kv_cache_quant_fp8.py` deleted | Empirically showed **0% speedup** for NLLB-600M and TranslateGemma 4B. The 24 KB/token KV-cache fits comfortably in HBM3e even at bs=2048; cast+dequant overhead exactly cancels bandwidth savings. Only viable for 7B+ models with deep sequences. |

### Parallelism

| # | Feature | Status | How to activate | Evidence | Notes |
|---|---|---|---|---|---|
| 21 | `device_map="auto"` multi-GPU | ✅ | 2 GPUs, model exceeds single-GPU fast-path threshold | `autoregressive_cuda.py:990` | HF/accelerate pipeline parallelism. |
| 22 | Single-GPU fast path | ✅ | model < 10% of one GPU's memory | `autoregressive_cuda.py:1004` | **Bypasses multi-GPU for every model this benchmark actually runs** (4B/12B on 141GB H200). |
| 23 | Tensor parallelism (`apply_tensor_parallelism`) | 💀 | — | defined `hardware/parallelism.py:330`, **never called**; re-exported only | Hardcoded to Gemma-3-12B constants; layer-mismatch path crashes on read-only `@property` assignment. The "2×H200 TP=2" story is effectively inactive. `ensure_dist_initialized()` (public API at `parallelism.py:217`) auto-detects `torchrun` env vars but is not yet wired into any hot path. |
| 24 | NCCL P2P enable | ✅ | CUDA | `autoregressive_cuda.py:127` | Runs even for 1 GPU (early-returns). |

### Data / tokenization

| # | Feature | Status | Evidence | Notes |
|---|---|---|---|---|
| 25 | orjson parsing | ✅ | `data/loader.py:59` | stdlib fallback. — measured 4.2× vs stdlib json. See M2.8. |
| 26 | Parallel gzip (pigz) | ✅ | `data/parallel_gz.py`; wired via `loader._open:766` | — measured 1.62× on 118MB data (NOT 3-8× as old docs claimed). See M2.8. |
| 27 | Token-level chunking | 🟡 | `data/chunker.py:71` `chunk_with_tokens` | The fast token-level path is **not used by the pipeline** — it uses `chunk()` (`pipeline.py:493`) for thread-safety, so the CPU saving is unrealized. |
| 28 | Numpy garbage filter | ✅ | `data/filters.py:92` | Byte-ratio check; pure CJK/Arabic could false-positive at the 0.95 threshold. `rejected_language` counter is dead. — measured 10× for 200-char, 30-50× for 3000-char vs pure Python. See M2.8. |
| 29 | Async prefetch (lock-free tokenizers) | ✅ | `data/pipeline.py:161` | Thread-local tokenizers, sentinel-based shutdown. |
| 30 | In-memory + external-sort shuffle | ✅ | `data/loader.py:295,603` | External sort kicks in above the 2 GiB memory budget; deterministic given seed + file order. |
| 31 | MPS memory-safety | ✅ | `harness.py:248` (`TR_MPS_MEMORY_SAFE`) | Single-probe batch tuning, sequential iteration, per-batch `empty_cache`. |

### Quality / metrics / observability

| # | Feature | Status | Evidence | Notes |
|---|---|---|---|---|
| 32 | BERTScore | ✅ | `quality/metrics_bertscore.py:32` | Model is **`bert-base-multilingual-cased`** (not DeBERTa, despite old SRS FR-23). |
| 33 | COMET-22 | ✅ | `quality/metrics_comet.py:181` | `Unbabel/wmt22-comet-da`; instance-level tokenizer patch (was class-level — fixed). |
| 34 | COMET-Kiwi | ✅ | `quality/metrics_comet.py:232` | Reference-free. |
| 35 | BLEU + chrF++ | ✅ | `quality/benchmark.py:283-284` | **Wired** (old README's "stubs" claim is false). chrF++ uses `char_order=4` for Turkish morphology. |
| 36 | Parallel metric computation | ✅ | `quality/benchmark.py:284` | 3-worker pool for 6 metrics (COMET-22, COMET-Kiwi, BERTScore, MetricX-24, BLEU, chrF++). |
| 37 | Prometheus exporter | ✅ | `observability/prometheus_metrics.py` | 21 metrics, all quality gauges populated. `harness.py:655-662` passes BLEU, chrF, COMET, BERTScore, COMET-Kiwi, and MetricX scores. |
| 38 | Rolling TPS gauge | ✅ | Prometheus exporter | `observability/prometheus_metrics.py` | `throughput_rolling` gauge with 60s deque window. `snapshot()` uses rolling TPS instead of histogram mean. Safe for 15-60s scrape intervals. |
| 39 | "6 alerts" | 💀 | — | **No alerting code exists** — alerting is external (Grafana) only. |
| 40 | Performance regression (Welch t-test) | 🗑 REMOVED v3.7 | — | `perf_regression.py` (471L) deleted in `19d979f` | Implemented but never wired. Removed with 4 other dead modules. |
| 41 | Nsight profiler | 🗑 REMOVED | — | `observability/nsight_profiler.py` deleted 2026-06-24 (405 lines). Module was never wired into generation paths. |
| 42 | Ensemble translation | 🗑 REMOVED | — | `quality/ensemble.py` deleted 2026-06-24 (210 lines). Never imported outside its own module. |
| 43 | Confidence estimation | 🗑 REMOVED | — | `quality/confidence.py` deleted 2026-06-24 (218 lines). Never imported outside its own module. |
| 44 | Extrapolation (SEM + bootstrap) | ✅ | `reporting/extrapolation.py:50,144` | Median-based point estimate; **constant-throughput assumption validated** — 2.2h degradation test showed zero throughput change (+0.1%/hr, R²=0.000046) on 4B/H200. |
| 45 | Checkpoint / resume | ✅ | `orchestration/checkpoint.py` | Atomic rename; file+doc_id position tracking. |
| 46 | O(1) rolling throughput | ✅ | `metrics/throughput.py` | p50/p99 latency percentiles now populated — `MetricsCollector.log_batch` passes `latency_ms=batch_result.total_latency_ms`. |

**Honest summary:** of the ~35 active features, the ones materially affecting
production throughput are #1, #2, #2b, #3, #8, #21/22, #19, and (opt-in) #10, #16, #17, #18.
Eight previously-dead modules (#4, #5, #6, #7, #14, #40 — ~3,600 lines) were permanently
deleted in v3.7.

---

## 9. Known Correctness Risks

Carry these forward when editing. (Cross-listed from `docs/AI_CODING_ANTIPATTERNS.md`.)

1. ~~AR/TRT `input_tokens` counts padding~~ → **FIXED v3.6.** `_assemble_output` uses `attention_mask.sum().item()`. TRT backend removed v3.7.
2. **`precision_config.uses_fp8` lies under `--safe-mode`** — config says FP8,
   runtime runs BF16.
3. **BERTScore target-check is harsher than siblings** — a `None`/error result →
   `0.0` → fails the 0.55 target, where COMET/Kiwi/BLEU/chrF are skipped on `None`
   (`quality/benchmark.py` `scores_meet_targets`). Inconsistent partial-run handling.
4. ~~Prometheus `quality_bleu`/`quality_chrf` never populated~~ → **FIXED 2026-06-24.** `harness.py:625-626` now passes `bleu=quality_results.bleu.get('score')` and `chrf=quality_results.chrf.get('score')`.
5. **Markdown report omits BERTScore & COMET-Kiwi** from the quality table (only
   in JSON).
6. **Extrapolation returns a zeroed `error` dict** on `effective_tps ≤ 0` rather
   than raising → silent "0 days."
- **Extrapolation constant-throughput assumption — validated June 2026.** 2.2h degradation test (122K batches, 110M tokens, 4B on H200) showed zero detectable throughput change (slope +0.1%/hr, R²=0.000046). This risk is MITIGATED for 4B-class models on H200. Larger models or longer runs may still exhibit degradation.
7. **`parallelism.py` hardcoded to Gemma-3-12B constants**; the layer-mismatch
   branch assigns to read-only `@property` fields (would crash) — latent because
   `apply_tensor_parallelism` is never called.
8. ~~Draft-model speculative silently tolerates tokenizer mismatch~~ → **FIXED v3.7.** Now raises `ValueError`.
9. ~~Chunker drops the final <10-token tail~~ → **FIXED v3.6.**
10. ~~`perf_regression._save_raw` non-atomic~~ → **REMOVED v3.7.** Module deleted with 4 other dead modules.

---

*Cross-references: `docs/README.md` (navigation) · `docs/DEVELOPMENT.md`
(extending the codebase) · `docs/AI_CODING_ANTIPATTERNS.md` (past mistakes &
prevention) · `docs/COMPILATION_GUIDE.md` (running it) · `docs/H200_SETUP.md`
(deployment log).*
