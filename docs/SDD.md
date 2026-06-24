# Software Design Document — v3.6

> ⚠️ **Historical design spec.** This SDD describes the *design intent* at v3.6.
> The authoritative description of what the code *actually does* today is
> [`ARCHITECTURE.md`](ARCHITECTURE.md). Where they disagree,
> **ARCHITECTURE is correct.** See especially
> [ARCHITECTURE §8 Feature Status](ARCHITECTURE.md#8-feature-status-the-truth-table)
> for the wired-vs-gated reality of every optimization listed herein.

**Turkish Corpus Translation Benchmark Harness**  
**Version**: 3.6 · **Status**: Implemented · **Last Updated**: 2026-06-23  
**References**: PRD v3.6, SRS v1.3

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2026-06-19 | Initial design — TranslateGemma AR-only |
| v1.1 | 2026-06-20 | Architecture corrected to Gemma 3 12B |
| v3.0 | 2026-06-21 | Model-agnostic backend protocol, diffusion support, plugin system |
| v3.1 | 2026-06-21 | All 35 optimizations wired into hot paths |
| v3.2 | 2026-06-21 | TensorRT backend, JIT kernel compilation, unified setup.sh + run.sh, pre-compiled Docker images, Docker Compose observability |
| v3.3 | 2026-06-22 | Speculative decoding (self-spec + draft-model), resume/checkpoint with position tracking, extrapolation CI fix (SEM + bootstrap), external sort shuffle, H200 production deployment, MPS IOAccelerator memory fixes |
| v3.6 | 2026-06-23 | NLLB-200 encoder-decoder backend (ModelType.ENCODER_DECODER), model presets registry (11 presets), quantization levels (bf16/fp16/int8/int4), Ministral 3B, Gemma4 QAT (ct/int4/q4_0), DiffusionGemma 26B, dead code cleanup |

---

## 1. Introduction

This document describes the complete v3.6 software design covering the model-agnostic backend architecture (AR, encoder-decoder/NLLB, diffusion, TensorRT, custom plugin), all 37+ extreme low-level optimizations, the JIT kernel compilation system, speculative decoding, model presets, quantization levels, checkpoint/resume with position tracking, and every subsystem's internal design.

### 1.1 Design Goals

1. **Model-agnostic** — One pipeline, any architecture (AR, diffusion, custom).
2. **Extreme performance** — Every optimization wired into the actual hot path.
3. **Portable** — CUDA (H200), MPS (Apple Silicon), CPU — identical APIs.
4. **Observable** — Every metric, every GPU stat, every token logged.
5. **Extensible** — Plugin system for custom models.

---

## 2. Architectural Overview

```
__main__.py → BenchmarkHarness.run()
  ├─ detect_backend("auto") → DeviceInfo
  ├─ InferenceEngine(model_path, backend_type, ...)
  │   └─ ModelRegistry.create_backend(config)
  │       ├─ Auto-detect: name → config keys → architecture heuristics
  │       └─ BackendClass(config) → InferenceBackend
  ├─ backend.load()
  │   ├─ Tokenizer (HF AutoTokenizer)
  │   ├─ Model weights (quantized or standard — bf16/fp16/int8/int4)
  │   ├─ torch.compile(mode="reduce-overhead")
  │   ├─ Fused kernels injection (⚠️ hardcoded `if False:` — disabled)
  │   ├─ cudaMallocAsync (⚠️ commented out — disabled)
  │   ├─ PagedAttention (⚠️ `_use_paged_attention` hardcoded `False` on AR path)
  │   ├─ INT8 KV-cache quant (⚠️ object constructed but never read/written — no-op)
  │   ├─ CUDA graph capture (⚠️ captured but never replayed — deprecated)
  │   └─ SpeculativeDecoder initialized (🔬 only if TR_ENABLE_EXPERIMENTAL_SPECULATIVE=1)
  ├─ backend.warmup(batches) + CUDA Graph capture
  └─ Translation loop
      ├─ AsyncPipeline.next_batch() → PipelineBatch
      ├─ backend.translate_batch(batch)
      │   ├─ [AR] Prefill → CUDA Graph decode loop (or standard)
      │   ├─ [AR + Spec] Self-speculative: early-layer draft → verify
      │   ├─ [Enc-Dec] encoder(src) → decoder.generate(beam search)
      │   └─ [Diffusion] Encode source → T-step denoising
      ├─ CheckpointManager.save() (every 5 min, with file+doc_id)
      └─ MetricsCollector.log_batch()
```

### 2.1 Inference Backend Protocol

```python
class InferenceBackend(ABC):
    model_type: ModelType          # AUTOREGRESSIVE | ENCODER_DECODER | DIFFUSION | TENSORRT | CUSTOM
    capabilities: ModelCapability  # Bitmask of supported features

    def load(self) -> None: ...
    def warmup(self, batches) -> None: ...
    def translate_batch(self, batch) -> BatchGenerationOutput: ...
    def is_loaded(self) -> bool: ...

    # Optional: encode_source, score_candidates, get_token_log_probs
```

### 2.2 Model Auto-Detection

1. Explicit `backend_type` in config → direct dispatch (auto | autoregressive | encoder_decoder | diffusion | custom).
2. Model name keywords (`nllb`, `m2m100`) → ENCODER_DECODER.
3. Config.json `architectures` containing `M2M100ForConditionalGeneration`, `NllbMoeForConditionalGeneration`, etc. → ENCODER_DECODER.
4. Model name keywords (llada, dream, mdlm, e2d2, bd3lm, sedd, etc.) → DIFFUSION.
5. Local config.json keys (diffusion_steps, noise_schedule) → DIFFUSION.
6. HF Hub config check → DIFFUSION.
7. Custom plugin match → CUSTOM.
8. Fallback → AUTOREGRESSIVE.

---

## 3. Component Design

### 3.1 JIT Compiler (`benchmark/hardware/jit_compiler.py`)

Compiles CUDA C++ and Metal MSL kernels at runtime, caches to `~/.cache/tr_benchmark/kernels/`.

**CUDA flow**: hash(source + arch) → check cache → MISS: `torch.utils.cpp_extension.load_inline()` via nvcc → .so → cache → load.

**Metal flow**: write .metal → `xcrun metal` → .air → `xcrun metallib` → .metallib → cache → load via Metal API.

**Cache**: key = SHA256(source + arch + flags), max 100 entries (LRU eviction).

### 3.2 Autoregressive Backend (Extreme-Optimized)

**Decode loop** (CUDA Graph path):
```
PREFILL (1 standard forward):
  model(input_ids, use_cache=True) → populate PagedAttention blocks
  CUDA event: prefill_start → prefill_end

DECODE (graph.replay() × N tokens):
  for token in 1..max_new_tokens:
    copy next_token → static buffer
    graph.replay()  ← 1 call instead of 200+ launches
    logits[:, -1, :] → argmax → next_token
    check EOS → break
  CUDA event: decode_start → decode_end
```

**Active optimizations** on `load()`: cudaMallocAsync (attempted, may be disabled for torch.compile compat), NCCL P2P, `torch.compile(mode="reduce-overhead")`, Flash SDPA, Triton fused kernels, JIT CUDA C++ kernels (QKV+RoPE, SwiGLU), PagedAttention, INT8 KV-cache quantization, INT4/INT8 weight quantization, speculative decoding (self-spec + draft-model).

### 3.2b Encoder-Decoder Backend (NLLB-200)

**Model family**: Facebook NLLB-200 (600M, 1.3B, 3.3B, 54B MoE distilled variants). Uses `AutoModelForSeq2SeqLM` + `AutoTokenizer` with language-code prefixes.

**Translation flow**:
```
1. Tokenize: tokenizer(src_text, src_lang="eng_Latn") → input_ids
2. Encode: model.encoder(input_ids) → encoder_hidden_states (once)
3. Decode: model.generate(
     encoder_outputs=encoder_hidden_states,
     forced_bos_token_id=tur_Latn,  ← constrain output language
     num_beams=4,                    ← beam search default
     early_stopping=True,
   )
4. Decode output: tokenizer.batch_decode(generated_ids)
```

**Auto-detection**: `nllb` in model path, `M2M100ForConditionalGeneration`/`NllbMoeForConditionalGeneration` in config.json, or `backend_type: "encoder_decoder"`. Registeres as `ModelType.ENCODER_DECODER`.

**Platform support**: CUDA (device_map="auto", memory budget), MPS, CPU. Beam search is deterministic (no sampling).

### 3.3 Diffusion Backend (Extreme-Optimized)

**Denoising algorithm**:
```
1. Encode source → source_hidden (once, cached)
2. Initialize target: [MASK] × target_len
3. for t = T down to 1:
     Embed target → x_t + timestep_embed
     [CUDA Graph] graph.replay()  or  [Fast-dLLM] skip if unchanged
     [Batched CFG] cond + uncond in one forward
     Forward: f_θ(x_t, source, t) → logits
     logits = uncond + w × (cond - uncond)
     x_t ← reverse_diffusion_step(logits, x_t, ᾱ_t)
4. Final forward at t=0 → argmax → decode
```

**Noise schedules**: cosine ᾱ_t = cos((t/T+0.008)/1.008 × π/2)², linear, sqrt.

### 3.4 Data Pipeline

- **JSONLLoader**: orjson + pigz + mmap + Fisher-Yates shuffle.
- **TextChunker**: Token-level (tokenize once, slice IDs).
- **ChunkFilter**: Numpy-vectorized non-ASCII detection (50× faster).
- **AsyncPipeline**: Lock-free thread-local tokenizers, pinned memory tensors, event-driven queue.
- **RustTokenizer**: `tokenizers` crate SIMD BPE (10-100× faster), HF fallback.

### 3.5 PagedAttention KV-Cache

Block-level virtual memory: 16-token physical blocks, per-sequence block table, free list with LIFO recycling, reference counting for prefix sharing. **40-70% memory savings** vs contiguous allocation.

⚠️ **Two paths, one real, one dead:** On the default AR hot path,
`_use_paged_attention` is hardcoded `False` and `_convert_to_paged` does not exist
(comments only). Paged KV that actually feeds the model forward exists only via
the `--continuous-batching --paged-attention` path (`ContinuousBatcher +
PagedCache` shim). See [`ARCHITECTURE.md` §7](ARCHITECTURE.md#7-the-two-optimization-stacks).

### 3.6 CUDA Graphs

`CUDAGraphPool`: pre-captures graphs for batch sizes [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]. Selects smallest ≥ current, pads.

`CUDAGraphDecoder`: static buffers → warmup 3× → capture with `torch.cuda.graph()` → replay.

⚠️ **Deprecated:** The graph pool is instantiated and graphs are captured during
warmup, but `_extreme_decode` never calls `graph.replay()` — it runs standard
`model(...)` forwards. See `benchmark/inference/backends/autoregressive.py:1548`.
The module emits a `FutureWarning` on import. The only active graph path is
`torch.compile(mode="reduce-overhead")`.

### 3.7 Observability

**PrometheusExporter**: 20+ metrics (counters, gauges, histograms), `/metrics` endpoint.
**PerfRegressionManager** (not yet wired): Welch t-test regression detection with severity classification.

*(DashboardServer, NsightProfiler, and LiveDashboard were removed in v3.6 cleanup — 
they had zero callers. The PrometheusExporter provides in-process HTTP; the full 
Grafana stack is external via `make dashboard`.)*

---

## 4. Data Flow

1. `JSONLLoader.iter_documents()` → (doc_id, file_name, text)
2. `TextChunker.chunk(text)` → chunk_text
3. `ChunkFilter.should_keep()` → pass/reject
4. `AsyncPipeline` tokenizer threads → tokenised_queue
5. `AsyncPipeline.next_batch()` → pinned-memory `PipelineBatch`
6. `InferenceBackend.translate_batch(batch)` → `BatchGenerationOutput`
7. `tokenizer.decode(output_ids)` → translated_text
8. `MetricsCollector.log_batch()` → batch/device/system logs
9. `CheckpointManager.save()` → atomic JSON write (with file_name + doc_id for position tracking)
10. Post-run: `QualityBenchmark.run()` → BERTScore/COMET-22/COMET-Kiwi in parallel

---

## 5. Error Handling

| Error | Response |
|-------|----------|
| CUDA OOM | Reduce batch size 50%, retry; save checkpoint with file+doc_id |
| JSON parse error | Skip line, continue |
| Tokenizer error | Skip chunk, continue |
| SIGTERM/SIGINT | Drain current batch, save checkpoint (with position), partial report |
| Disk full | Graceful stop, save checkpoint |
| NCCL hang | Timeout + worker restart |

## 6. Technology Stack & Model Presets

Python 3.11+, PyTorch 2.6+, HuggingFace transformers 4.57+, CUDA 12.4+, Triton 3.2+, Rust tokenizers crate 0.22+, orjson 3.11+, BERTScore 0.3+, COMET-22 + COMET-Kiwi 2.2+, Pydantic v2, pytest 9.1+, bitsandbytes (INT8/INT4 quantization), Docker, Grafana, Prometheus.

### 6.1 Model Presets Registry (`benchmark/config/model_presets.py`)

Central registry of 11 supported model configurations:

| Preset | Model ID | Layers | KV Heads | Quantization |
|--------|----------|--------|----------|-------------|
| translategemma-4b-bf16 | google/translategemma-4b-it | 36 | 4 | BF16 |
| translategemma-4b-int8 | google/translategemma-4b-it | 36 | 4 | INT8 (bnb) |
| translategemma-4b-int4 | google/translategemma-4b-it | 36 | 4 | INT4 (bnb NF4) |
| ministral-3b-bf16 | mistralai/Ministral-3-3B-Instruct-2512 | 24 | 8 | BF16 |
| gemma4-e2b-qat-ct | google/gemma-4-E2B-it-qat-mobile-ct | 26 | 4 | BF16 |
| gemma4-e2b-qat-int4 | google/gemma-4-E2B-it-qat-mobile-ct | 26 | 4 | INT4 |
| gemma4-e2b-q4_0 | google/gemma-4-E2B-it-qat-mobile-transformers | 26 | 4 | INT4 |
| gemma4-e4b-qat-ct | google/gemma-4-E4B-it-qat-mobile-ct | 34 | 8 | BF16 |
| gemma4-e4b-qat-int4 | google/gemma-4-E4B-it-qat-mobile-ct | 34 | 8 | INT4 |
| gemma4-e4b-q4_0 | google/gemma-4-E4B-it-qat-mobile-transformers | 34 | 8 | INT4 |
| diffusiongemma-26b-a4b | google/diffusiongemma-26B-A4B-it | 48 | 8 | BF16 |

Presets are resolved via `get_preset_by_name()` or `resolve_architecture_defaults()`. They provide architecture constants (num_layers, num_kv_heads, head_dim, hidden_size) to other components, eliminating hardcoded constants.

---

## 7. Current Reality vs. This Design

This SDD v3.6 was written at design time. Several optimizations described here
were implemented but subsequently **gated off** due to compatibility issues
(PyTorch 2.6/2.11 `cudagraph_trees`, `torch.compile` conflicts, TRT 11.x API
removals). The production AR hot path is:

- `torch.compile(mode="reduce-overhead")`
- Transformer-Engine FP8 (`te.Linear`)
- Flash + mem-efficient SDPA

Everything else (fused kernels, CUDA graphs, cudaMallocAsync, PagedAttention on
the AR path, INT8 KV-cache, Fast-dLLM caching, TensorRT decode, tensor
parallelism) is either gated off, broken, or dead code.

The authoritative status of every feature is
[`ARCHITECTURE.md` §8 Feature Status](ARCHITECTURE.md#8-feature-status-the-truth-table).

See also: [`AI_CODING_ANTIPATTERNS.md`](AI_CODING_ANTIPATTERNS.md) —
especially A1 (dead code masquerading as a feature), A2 (docstring/comment lies),
A3 (copy-paste divergence).
