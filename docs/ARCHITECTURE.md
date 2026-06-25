# Architecture — TR Corpus Translation Benchmark

> **Purpose:** The reality-grounded description of how this codebase actually runs.
> **Status:** Engineering reality — current as of v3.6 (June 2026).
> **Audience:** Engineers and LLM coding agents working on this repo.
>
> ⚠️ **Read this before editing `benchmark/inference/` or `benchmark/hardware/`.**
> This is the single source of truth for what is wired vs. gated. The older
> `PRD.md`/`SRS.md` describe design *intent*; this document describes *current
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

This benchmark answers one question: *how many days to translate ~6.23T English
tokens into Turkish on 2× NVIDIA H200, at academic quality?*

It is **model-agnostic**: one pipeline drives autoregressive (AR), encoder-decoder
(NLLB), diffusion, TensorRT, and custom-plugin backends through a single
`InferenceBackend` protocol, then measures throughput, extrapolates to
corpus-completion time, and scores quality.

**The single most important thing to understand about this codebase:** the
documentation historically advertised "39 optimizations (37 wired)," but the
production AR hot path is, in reality, **plain eager `model(...)` in a Python
loop with HuggingFace `past_key_values`**, accelerated only by:

- **Pre-tokenized Parquet cache** (skips CPU tokenization, +60% TPS),
- **TF32 + Flash SDPA** (always on for CUDA),
- **BF16** (native H200 dtype; FP8 TE blocked by driver — see
  [`FP8_TE_CUDA_ISSUES.md`](FP8_TE_CUDA_ISSUES.md)), and
- **`torch.compile(mode="default")`** on PyTorch 2.12–2.13, or
  **`torch.compile(mode="reduce-overhead")`** on PyTorch 2.14+
  (cudagraph_trees KV-cache bug persists through 2.12.1).

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
        │     quality/benchmark.py:328  → BERTScore ‖ COMET-22 ‖ COMET-Kiwi ‖ BLEU ‖ chrF++
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
NLLB: empty; diffusion: `encode_ms/denoise_ms`; TRT: `{"engine":"tensorrt"}`).

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

0. **TensorRT upgrade** (`registry.py:190`) — only on CUDA. `TensorRTBackend.create()`
   returns `None` (→ AR fallback) unless an engine can be built/loaded. In practice
   this almost always falls back (see TRT status in §8).
1. **Explicit override** (`registry.py:215`) — `extra["backend_type"]` (except
   `"auto"`), validated against `ModelType`.
2. **Auto-detect** (`registry.py:227`) via `_detect_model_type(model_path)`:
   - name contains `nllb`/`madlad` → `ENCODER_DECODER`
   - name matches a `DIFFUSION_KEYWORDS` (`llada, dream, mdlm, e2d2, bd3lm, diffusiongemma, …`, `constants.py:125`)
   - local `config.json` `model_type`/`architectures`/diffusion config keys
   - HF Hub `config.json` (same signals; `trust_remote_code=False`, 3 retries)
   - else `AUTOREGRESSIVE`
3. **Custom plugin** (`registry.py:234`) — `PluginRegistry.lookup()`; first plugin
   whose `detect(model_path)` is True. (Discovery is gated behind
   `TR_ALLOW_UNTRUSTED_PLUGINS=1`; see `custom_plugin.py`.)
4. **Fallback** → `AutoregressiveBackend`.

Auto-detection results are cached in a 100-entry FIFO cache (`registry.py:54`).

---

## 5. Per-Backend Reality

### 5.1 AutoregressiveBackend (`inference/backends/autoregressive.py`)

The primary backend. `model_type = AUTOREGRESSIVE`; capabilities
`TRANSLATE | FORWARD_ENCODE | CONFIDENCE | QUANTIZABLE_KV | SPECULATIVE | ENSEMBLE_READY`.

**`load()` (`:644`) actually does:**

1. Devices (`cuda:{i}` / `mps` / `cpu`), NCCL P2P enable on CUDA. **cudaMallocAsync
   is disabled** (commented out, `:654` — incompatible with `torch.compile`).
2. Tokenizer via `AutoTokenizer`, left padding, Gemma-4-QAT list-tokenizer fix.
3. Flash + mem-efficient SDPA on CUDA.
4. Model load: QAT/Gemma-4/Q4_0 → `_try_load_qat_model`; else `_load_standard_model`
   (single-GPU fast path when model < 10% of one GPU's memory; else
   `device_map="auto"`).
5. TE FP8 (`te.Linear` replacement, `lm_head` skipped) — **gate: TE must be installed
   and functional.**  On pip venvs TE source-build succeeds but runtime cuBLASLt
   crashes on all drivers tested (580, 565).  See
   [`FP8_TE_CUDA_ISSUES.md`](FP8_TE_CUDA_ISSUES.md).
   **Dynamic quantization (torch._scaled_mm) was removed in June 2026** — measured
   −40% TPS vs BF16 on 4B models.  Static quantization (SmoothQuant / QAT) is the
   path forward; pre-quantized weights are cached via `save_fp8_weights()` /
   `load_fp8_weights()`.  Default on pip: `TR_SKIP_FP8=1` ↔ pure BF16.
6. **Fused-kernel injection is hardcoded `if False:`** (`:761`) — dead.
7. `torch.compile` (CUDA only; skipped on MPS/CPU/safe-mode). Version-gated:
   **< 2.12** → skipped (eager). **2.12–2.13** → `mode="default"` (inductor fusion,
   no cudagraph_trees — warmup takes 30s but decode is stable).
   **≥ 2.14** → `mode="reduce-overhead"` (frame-level CUDA graphs).
   `_apply_extreme_compile:1167`.
   **Measured (no-compile, PyTorch 2.12.1): 1,650 tok/s** (4B, bs=32, 1×H200).
8. JIT kernel precompile (only the non-functional Metal RMSNorm is live; see §8).
9. PagedAttention init — **never runs** (`_use_paged_attention` hardcoded `False`, `:596`).
10. INT8 KV-cache object — **constructed but never read/written** (`:1184`).
11. Speculative decoder — **only if `TR_ENABLE_EXPERIMENTAL_SPECULATIVE=1`**.

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

`model_type = ENCODER_DECODER`; capabilities `TRANSLATE | FORWARD_ENCODE | ENSEMBLE_READY`.

Encoder-decoder (NLLB-200 / M2M100 / MADLAD). `translate_batch` (`:381`) runs
`model.generate(input_ids, attention_mask, forced_bos_token_id=tgt_lang,
num_beams=…)`; `input_tokens_total` correctly uses `attention_mask.sum()` (the
padding bug was fixed here in commit `ffa707b`). `torch.compile(reduce-overhead)`
on CUDA. `forced_bos_token_id` from `tgt_lang` (default `tur_Latn`); falls back to
`None` if unresolved (may produce wrong-language output).

### 5.3 DiffusionBackend (`inference/backends/diffusion.py`)

`model_type = DIFFUSION`; capabilities include `CLASSIFIER_FREE`.

Denoising loop (`:492`): encode source once (cached), initialize `[MASK]×target_len`,
iterate `T` reverse steps with timestep embeddings, optional batched CFG
(`cond+uncond` in one forward), reverse-diffusion step → argmax → re-embed.
DiffusionGemma auto-detected (steps=128, schedule=linear, guidance=2.0).

- **CUDA-graph denoising** exists (`:632`) but only if `use_cuda_graph_for_step=True`
  (default `False`).
- **Fast-dLLM caching** (`:709`) is **stats-only** — it counts cache hits but
  always falls through to the full forward.
- `_forward_step` (`:772`) is a compatibility ladder over encoder-decoder /
  decoder-only / projection signatures.

### 5.4 TensorRTBackend (`inference/backends/tensorrt_backend.py`)

`model_type = AUTOREGRESSIVE`. **Effectively non-functional for correct
translation.** The TRT decode loop has no `past_key_values` passthrough, so each
decode step is an isolated forward with zero accumulated context → output after
the first token is near-random. There is a hard safety gate
(`_allow_trtrt_decode_without_kv_cache`, `:182`, default `False`):
`translate_batch` **raises `RuntimeError`** if TRT is active and this flag is
False (`:334`). When True, it logs "output WILL be corrupted" and proceeds only
for latency benchmarking. Falls back to the HF `_hf_model.generate()` path
otherwise. Also broken on TRT 11.x (`trt_builder.py` uses removed
`EXPLICIT_BATCH`/`num_layers`). **Net: TRT almost always falls back to AR.**

### 5.5 Custom Plugin (`inference/backends/custom_plugin.py`)

Drop a `.py` defining a `CustomModelPlugin` subclass in
`~/.tr_benchmark/plugins/`, `TR_BENCHMARK_PLUGIN_PATH`, entry-points, or
`./plugins/`. Discovery is gated behind `TR_ALLOW_UNTRUSTED_PLUGINS=1` (plugins
run with full process privileges — no sandbox). Explicit `register_plugin()` bypasses
the gate.

---

## 6. Module Map

| Subsystem | Key files |
|---|---|
| **Entry / orchestration** | `benchmark/__main__.py` · `orchestration/harness.py` · `orchestration/checkpoint.py` · `orchestration/signals.py` |
| **Inference engine** | `inference/engine.py` · `inference/backends/{protocol,registry,autoregressive,nllb,diffusion,tensorrt_backend,custom_plugin}.py` · `inference/sampling.py` |
| **Inference optimizations** | `inference/speculative.py` · `inference/paged_attention.py` · `inference/continuous_batcher.py` · `inference/batch_assembly.py` · `inference/batch_tuner.py` |
| **Hardware** | `hardware/backend.py` · `hardware/precision.py` · `hardware/jit_compiler.py` · `hardware/fused_ops.py` · `hardware/triton_kernels_fused.py` · `hardware/kv_cache_quant.py` · `hardware/parallelism.py` · `hardware/trt_builder.py` · `hardware/cuda_graphs.py` |
| **Data pipeline** | `data/loader.py` · `data/chunker.py` · `data/filters.py` · `data/pipeline.py` · `data/parallel_gz.py` |
| **Quality** | `quality/benchmark.py` · `quality/{metrics_bertscore,metrics_comet,metrics_bleu,metrics_chrf,references}.py` |
| **Metrics** | `metrics/collector.py` · `metrics/throughput.py` · `metrics/gpu_sampler.py` · `metrics/system_sampler.py` · `metrics/batch_logger.py` |
| **Observability** | `observability/prometheus_metrics.py` · `observability/perf_regression.py` (unwired) |
| **Reporting** | `reporting/aggregator.py` · `reporting/extrapolation.py` · `reporting/degradation.py` · `reporting/json_report.py` · `reporting/markdown_report.py` |
| **Config** | `config/schema.py` · `config/capability.py` · `config/constants.py` · `config/model_presets.py` |

---

## 7. The Two Optimization Stacks

A subtle point that explains much of the doc-vs-reality confusion: there are
**two parallel, decoupled optimization stacks** that do not share state.

- **Stack A — the AR backend's own features** (`autoregressive.py`): owns
  `_paged_kv`, `_graph_pool`, `_spec_decoder`, `_kv_quant_cache`. Of these, only
  `_spec_decoder` can ever be live (and only under an env gate). `_paged_kv` is
  hardcoded off; `_graph_pool` is captured but never replayed; `_kv_quant_cache`
  is constructed but never read/written. The README's "paged KV is allocated with
  a no-op converter; not fed to the model forward" is **true for Stack A**.

- **Stack B — the harness-owned ContinuousBatcher path** (`harness.py:680` +
  `continuous_batcher.py` + `paged_attention.py`): the batcher builds its **own**
  `PagedKVCache` and a `PagedCache` (`DynamicCache`-compatibility shim) and passes
  it as `past_key_values` to `engine.model(...)` directly (`continuous_batcher.py:651`).
  **This is the only place where paged KV is genuinely fed to a model forward.**
  Gated behind `--continuous-batching --paged-attention` (CUDA) + batch size ≥ 2.

So "PagedAttention" is simultaneously dead (Stack A) and real (Stack B) depending
on which path you mean.

---

## 8. Feature Status (the truth table)

**Legend:** ✅ Wired & on the hot path · 🟡 Built but gated off · 🔬 Experimental ·
⚠️ Broken/Disabled · 💀 Dead code (never called). Cite `file:line` to verify.

### Compute / decode

| # | Feature | Status | How to activate | Evidence | Notes |
|---|---|---|---|---|---|
| 1 | `torch.compile` | ⚠️ | on by default (CUDA); version-gated | `autoregressive.py:1167` | **< 2.12** → skipped (eager). **2.12–2.13** → `mode="default"` (stable, warmup 30s). **≥ 2.14** → `mode="reduce-overhead"`. No-compile baseline 1,650 tok/s (2.12.1, 4B, bs=32). |
| 2 | Transformer-Engine FP8 (`te.Linear`) | ❌ | CUDA + TE installed and working | `autoregressive.py:762` | **TE source-build succeeds but runtime cuBLASLt crashes on all tested drivers (580, 565).** See [`FP8_TE_CUDA_ISSUES.md`](FP8_TE_CUDA_ISSUES.md). Dynamic quantization (torch._scaled_mm) removed — 2× slower than BF16. Static quantization (SmoothQuant/QAT) via `save_fp8_weights`/`load_fp8_weights` is the path forward. `TR_SKIP_FP8=1` is the practical default. |
| 2b | Pre-tokenized Parquet cache | ✅ | automatic (checks `~/.cache/tr_benchmark/pretokenized/`) | `benchmark/data/pretokenizer.py` | +60% TPS on AR models. 198K chunks cached. `--pretokenize` to create, auto-detected thereafter. |
| 3 | Flash + mem-efficient SDPA | ✅ | CUDA default | `autoregressive.py:695` | — measured 2026-06-24: 1.17-1.23× overall throughput for 4B model. Attention-only speedup is higher (likely 2-4×) but attention is ~20-30% of total compute for 4B. See M2.4. |
| 4 | CUDA Graph decode | ⚠️ | — (deprecated) | `hardware/cuda_graphs.py:3` (FutureWarning on import); capture at `autoregressive.py:1339`; **never replayed** in `_extreme_decode:1548` | Captured graph excludes `past_key_values` as a static input → replay would feed zero-context garbage. Capture cost is paid, benefit never collected. |
| 5 | Fused Triton kernels (RMSNorm, SwiGLU) | 💀 | — | injection hardcoded `if False:` (`autoregressive.py:761`) | Triton fused ops only work inside `torch.compile`; crash in eager (commit `804c0a6`). |
| 6 | JIT CUDA C++ kernels (QKV+RoPE, SwiGLU) | ⚠️ | — | sources set to `None` ("architecturally broken"), `jit_compiler.py:126,131` | Disabled 2026-06-23. |
| 7 | JIT Metal RMSNorm | ⚠️ | MPS | `jit_compiler.py:139` source exists, but `_make_metal_wrapper:529` returns inputs unchanged (non-functional) | Falls back to eager PyTorch. |
| 8 | cudaMallocAsync | ⚠️ | — (disabled) | commented out `autoregressive.py:654`; `_malloc_async_active=False:607` | Incompatible with `torch.compile` `cudagraph_trees` in PyTorch 2.6. |
| 9 | INT8 KV-cache quantization | 💀 | — | `_kv_quant_cache` constructed `autoregressive.py:1184`, never `.update()`/`.get()`-ed | Module docstring now honestly states "pure eager PyTorch — no Triton or Metal kernels exist" and "never wired into the decode loop." |
| 10 | Speculative decoding | 🔬 | `--speculative` **and** `TR_ENABLE_EXPERIMENTAL_SPECULATIVE=1` | `speculative.py:1129` (env gate); `autoregressive.py:1385` (delegation) | Greedy-only, serial per-sequence. — measured 2026-06-24 on 4B: 25 tok/s at bs=1 (SLOWER than non-spec 62 tok/s), 38% acceptance. 8-layer draft + 34-layer full verify overhead exceeds gain at 4B depth. May benefit 48+ layer models (12B/27B). See M2.3. **2026-06-24 fix:** model introspection updated for Gemma3 multimodal (`language_model.*` nesting). Dual-RoPE detection via `inspect.signature` on first decoder layer. Verify runs full L layers (not L−D). Draft-model mode silently tolerates tokenizer mismatch. |
| 11 | Batched CFG (diffusion) | ✅ | `guidance_scale > 1.0` | `diffusion.py:536` | cond+uncond in one forward. |
| 12 | Fast-dLLM caching (diffusion) | 🟡 | — (stats-only) | `diffusion.py:709` | Counts hits but always falls through to full forward. |
| 13 | CUDA-graph denoising (diffusion) | 🟡 | `use_cuda_graph_for_step=True` (default False) | `diffusion.py:632` | |
| 14 | TensorRT backend | ⚠️ | — (safety-gated to refuse decode) | `tensorrt_backend.py:334` raises unless `allow_trt_decode_without_kv_cache`; `trt_builder.py:394,633` broken on TRT 11.x | No KV passthrough → corrupted output. Falls back to AR. |
| 15 | Capability Registry | ✅ | Automatic in `load()` | `autoregressive.py:639-642`; `config/capability.py` | 14 features tracked with verified `ActivationState` (ACTIVE/INERT/BROKEN/UNKNOWN). Logs `active_vs_total()` and `reg.report_text()` at startup. Single source of truth for what's on the hot path — supersedes old manual log lines. |

### Memory / KV

| # | Feature | Status | How to activate | Evidence | Notes |
|---|---|---|---|---|---|
| 16 | PagedAttention (AR path) | 💀 | — | `_use_paged_attention` hardcoded `False`, `autoregressive.py:596`; `_convert_to_paged` doesn't exist (comments only) | AR attention reads HF `past_key_values`, so paged blocks would be dead memory. |
| 17 | PagedAttention (ContinuousBatcher path) | ✅ (gated) | `--continuous-batching --paged-attention` (CUDA) + batch ≥ 2 | `harness.py:680`; `paged_attention.py` `PagedCache` shim fed at `continuous_batcher.py:651` | The only real paged-KV→forward integration. — measured 2026-06-24: 60-87.5% KV memory savings for variable-length workloads, 0% for fixed-length. Validates and exceeds the 40-70% claim. Savings come from not pre-allocating max_seq_len. See M2.6. |
| 18 | Continuous batching | ✅ (gated) | `--continuous-batching --paged-attention` (CUDA) + batch ≥ **2** | `continuous_batcher.py:100` (`MIN_BATCH_SIZE_FOR_CONTINUOUS=2`); `harness.py:299` | Production-quality chunked-prefill scheduler. **Note: the real threshold is 2, not 8 as old docs claim.** |
| 19 | Pinned-memory pipeline | ✅ | CUDA | `data/pipeline.py:68` (`_should_pin = torch.cuda.is_available()`) | Disabled on MPS (unified memory). — measured 2026-06-24: 2.1× H2D speedup, ~6.6 GB/s effective bandwidth. See M2.5. |
| 20 | INT4/INT8 weight quantization | ✅ | `--quantization int8|int4` / QAT presets | `autoregressive.py:412` (QAT), bitsandbytes | Separate axis from FP8 compute. — measured 2026-06-24: 41% memory savings (matches ~2× smaller). BUT throughput is 3.7× slower (213 vs 792 tok/s) — counterproductive on H200 unless VRAM-constrained. See M2.7. |

### Parallelism

| # | Feature | Status | How to activate | Evidence | Notes |
|---|---|---|---|---|---|
| 21 | `device_map="auto"` multi-GPU | ✅ | 2 GPUs, model exceeds single-GPU fast-path threshold | `autoregressive.py:907` | HF/accelerate pipeline parallelism. |
| 22 | Single-GPU fast path | ✅ | model < 10% of one GPU's memory | `autoregressive.py:919` | **Bypasses multi-GPU for every model this benchmark actually runs** (4B/12B on 141GB H200). |
| 23 | Tensor parallelism (`apply_tensor_parallelism`) | 💀 | — | defined `hardware/parallelism.py:238`, **never called**; re-exported only | Hardcoded to Gemma-3-12B constants; layer-mismatch path crashes on read-only `@property` assignment. The "2×H200 TP=2" story is effectively inactive. `ensure_dist_initialized()` (public API at `parallelism.py:217`) auto-detects `torchrun` env vars but is not yet wired into any hot path. |
| 24 | NCCL P2P enable | ✅ | CUDA | `autoregressive.py:168` | Runs even for 1 GPU (early-returns). |

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
| 36 | Parallel metric computation | ✅ | `quality/benchmark.py:266` | 3-worker pool for 5 metrics — "wall=max" is approximate. |
| 37 | Prometheus exporter | ✅ | `observability/prometheus_metrics.py` | 20+ metrics, all quality gauges now populated. `harness.py:625-626` passes `bleu=quality_results.bleu.get('score')` and `chrf=quality_results.chrf.get('score')`. |
| 38 | Rolling TPS gauge | ✅ | Prometheus exporter | `observability/prometheus_metrics.py` | `throughput_rolling` gauge with 60s deque window. `snapshot()` uses rolling TPS instead of histogram mean. Safe for 15-60s scrape intervals. |
| 39 | "6 alerts" | 💀 | — | **No alerting code exists** — alerting is external (Grafana) only. |
| 40 | Performance regression (Welch t-test) | 💀 | `observability/perf_regression.py` | Implemented but **zero callers** — not wired into any run/CI path. Dashboard (`dashboard.py`), server (`server.py`), and nsight profiler (`nsight_profiler.py`) removed 2026-06-24; only the stat test remains. |
| 41 | Nsight profiler | 🗑 REMOVED | — | `observability/nsight_profiler.py` deleted 2026-06-24 (405 lines). Module was never wired into generation paths. |
| 42 | Ensemble translation | 🗑 REMOVED | — | `quality/ensemble.py` deleted 2026-06-24 (210 lines). Never imported outside its own module. |
| 43 | Confidence estimation | 🗑 REMOVED | — | `quality/confidence.py` deleted 2026-06-24 (218 lines). Never imported outside its own module. |
| 44 | Extrapolation (SEM + bootstrap) | ✅ | `reporting/extrapolation.py:50,144` | Median-based point estimate; **constant-throughput assumption validated** — 2.2h degradation test showed zero throughput change (+0.1%/hr, R²=0.000046) on 4B/H200. |
| 45 | Checkpoint / resume | ✅ | `orchestration/checkpoint.py` | Atomic rename; file+doc_id position tracking. |
| 46 | O(1) rolling throughput | ✅ | `metrics/throughput.py` | p50/p99 latency percentiles now populated — `MetricsCollector.log_batch` passes `latency_ms=batch_result.total_latency_ms`. |

**Honest summary:** of the ~46 features above, the ones materially affecting
production throughput are #1, #2, #3, #16, #21/22, #19, and (opt-in) #10, #17, #18.
The rest are wired-but-not-helping, gated off, broken, or dead.

---

## 9. Known Correctness Risks

Carry these forward when editing. (Cross-listed from `docs/AI_CODING_ANTIPATTERNS.md`.)

1. ~~AR/TRT `input_tokens` counts padding~~ → **FIXED 2026-06-24.** `autoregressive.py` `_assemble_output` and `tensorrt_backend.py` `_assemble_output` now use `attention_mask.sum().item()` to count real tokens. NLLB was fixed in `ffa707b`; AR/TRT followed.
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
8. **Draft-model speculative decoding silently tolerates tokenizer mismatch**
   (warns only).
9. ~~Chunker drops the final <10-token tail~~ → **FIXED 2026-06-24.** Tail-chunk minimum lowered from 10 tokens to 1 token in `chunker.py`. Only truly empty (0-token) chunks are now skipped.
10. **`perf_regression._save_raw` is non-atomic** — a crash mid-write silently
    destroys the baseline (load returns `[]` → auto-re-establish). Latent (unwired).

---

*Cross-references: `docs/README.md` (navigation) · `docs/DEVELOPMENT.md`
(extending the codebase) · `docs/AI_CODING_ANTIPATTERNS.md` (past mistakes &
prevention) · `docs/COMPILATION_GUIDE.md` (running it) · `docs/H200_SETUP.md`
(deployment log).*
