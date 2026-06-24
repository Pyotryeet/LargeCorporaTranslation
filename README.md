# Turkish Corpus Translation Benchmark · v3.6

**Model-agnostic. Extreme low-level optimized. One-command to run.**


> *How many days to translate 6+ trillion English tokens into Turkish — on 2 NVIDIA H200s, with academic-grade quality?*

---

> ### 📖 Before editing the code — read these
> This codebase has a **doc-vs-reality gap**: many "optimizations" are built but
> gated off. Before editing `benchmark/inference/` or `benchmark/hardware/`, read:
> - [`docs/ARCHITECTURE.md` §8 Feature Status](docs/ARCHITECTURE.md#8-feature-status-the-truth-table) — what is *actually* wired vs. gated vs. dead.
> - [`docs/AI_CODING_ANTIPATTERNS.md`](docs/AI_CODING_ANTIPATTERNS.md) — concrete mistakes already made here.
>
> **Do not reason about performance from the optimization count below — it does
> not reflect the gating reality.** The production AR hot path is plain eager
> `model(...)` + `torch.compile` + Transformer-Engine FP8.

---

## Documentation

Full documentation lives in [`docs/`](docs/). Start at
[`docs/README.md`](docs/README.md) for a navigation map.

| You want to… | Read |
|---|---|
| Know what the code *actually* does | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Understand structural/systemic problems | [`docs/ARCHITECTURAL_FLAWS.md`](docs/ARCHITECTURAL_FLAWS.md) |
| Run / install / deploy | [`docs/COMPILATION_GUIDE.md`](docs/COMPILATION_GUIDE.md) |
| Extend the codebase / run tests | [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) |
| Avoid mistakes AI coders made here | [`docs/AI_CODING_ANTIPATTERNS.md`](docs/AI_CODING_ANTIPATTERNS.md) |
| Original product spec / requirements | [`docs/PRD.md`](docs/PRD.md), [`docs/SRS.md`](docs/SRS.md) (historical) |

---

## Quick Start

```bash
./setup.sh            # auto-detects macOS/CUDA/CPU, installs everything
./run.sh --dry-run    # 60s smoke test
./run.sh --quick      # 5min evaluation
./run.sh              # full 2h benchmark
```

**Same commands on every platform.** No flags to memorize, no separate configs.

```bash
./run.sh --tensorrt          # TensorRT engine (⚠️ broken on TRT 11.x; falls back to AR)
./run.sh --nllb              # NLLB-200 encoder-decoder (EN→TR)
./run.sh --diffusion --quick # LLaDA 8B diffusion model
./run.sh --observability     # live Prometheus dashboard on :9090
./run.sh --model 4B --duration 3600 --batch-size 64
./run.sh --model ministral-3b-bf16  # model preset from registry
./run.sh --speculative       # self-speculative decoding (zero extra VRAM)
./run.sh --quantization int8 # 8-bit quantized model
./run.sh --paged-attention   # PagedAttention KV-cache (CUDA)
./run.sh --multi-gpu         # data-parallel across 2 GPUs
make dashboard               # Prometheus + Grafana stack
```

### Pre-tokenization — tokenize once, benchmark forever

```bash
# Pre-process input data once (writes to ~/.cache/tr_benchmark/pretokenized/)
./run.sh --pretokenize --model translategemma-4b-bf16

# Pre-process for all registred models
python -m benchmark --pretokenize-all

# Subsequent runs auto-detect the cache — zero config
./run.sh --config config.yaml       # reads pre-tokenized Parquet, skips chunking
TR_NO_PRETOKENIZED_CACHE=1 ./run.sh # force fresh tokenization
```

The cache key includes the model, tokenizer, chunk size, and input files — any change triggers automatic re-tokenization. Parquet files use row-group streaming (~10K chunks/group) so memory usage stays constant. Validated identical output to the dynamic pipeline. See [`docs/PRETOKENIZATION_PLAN.md`](docs/PRETOKENIZATION_PLAN.md) for the full design.

[Full CLI reference →](docs/COMPILATION_GUIDE.md)

---

## What It Measures

| Backend | Model | Throughput | Days for 6.23T tokens |
|---------|-------|---------------------|----------------------|
| AR (BF16) | TranslateGemma 4B | **735 tok/s** (bs=16, 1× H200, measured 2026-06-24) | ~98 days (1× H200) |
| AR (BF16) | TranslateGemma 4B | **13,223 tok/s** (bs=512, 1× H200, measured 2026-06-24) | ~5.5 days (1× H200) |
| AR (BF16) | TranslateGemma 4B | ~1,471 tok/s (bs=16, 2× H200, data-parallel, estimated) | ~49 days (2× H200) |
| Enc-Dec | NLLB-200 600M | **580.5 tok/s** (bs=8, BF16, Flash SDPA, measured) | ~124 days |
| Enc-Dec | NLLB-200 3.3B | **372.5 tok/s** (bs=8, BF16, Flash SDPA, measured) | ~193 days |
| AR (INT8) | TranslateGemma 4B | **213 tok/s** (⚠️ 3.7× SLOWER than BF16, measured) | — |
| AR (TE FP8) | TranslateGemma 4B | **497 tok/s** (⚠️ 40% slower, 0% memory saved, measured) | — |

> All throughputs measured 2026-06-24 on asus02 (2× NVIDIA H200 NVL, 139.80 GB VRAM,
> PyTorch 2.6.0+cu124). 2×H200 numbers assume perfect data-parallel scaling (not yet
> verified). See [`docs/MEASUREMENT_PLAN.md`](docs/MEASUREMENT_PLAN.md) for full details.

> ⚠️ **The TensorRT backend is not functional for correct translation** — its
> decode loop has no KV-cache passthrough, so output after the first token is
> corrupted; it is safety-gated to raise unless `allow_trt_decode_without_kv_cache`
> is set, and it also breaks on TRT 11.x. It falls back to the AR backend. See
> [`docs/ARCHITECTURE.md` §8 #14](docs/ARCHITECTURE.md#8-feature-status-the-truth-table).

All runs also output: **BERTScore · COMET-22 · COMET-Kiwi · GPU utilization · memory · temperature · throughput distribution · cost estimate · 95% CI.**

---

## Architecture

```
BenchmarkHarness → detect platform → create backend → warmup → translate → quality → report

InferenceBackend (abstract protocol)
 ├─ Autoregressive     — CUDA Graph decode, PagedAttention, cudaMallocAsync
 ├─ Encoder-Decoder    — NLLB-200 (BART/M2M100), beam search, forced BOS
 ├─ Diffusion          — T-step denoising, Fast-dLLM cache, batched CFG
 ├─ TensorRT           — ONNX → compiled TRT engine, layer fusion, autotuned kernels
 └─ Custom Plugin      — drop a .py in ~/.tr_benchmark/plugins/

Model Presets: 11 presets (TranslateGemma 4B, Ministral 3B, Gemma4 QAT, DiffusionGemma 26B)
Quantization:  bf16 · fp16 · int8 (bitsandbytes) · int4 (NF4) — configurable via --quantization
Hardware layer: JIT compiler (nvcc/metal) → .so/.metallib (cached) · Triton fused kernels
Data pipeline:  orjson → token-level chunking → numpy filters → lock-free tokenizers → pinned memory
Observability:  Prometheus 20+ metrics → Grafana dashboard → Welch t-test regression CI gate
Speculative:  Self-speculative (early-layer draft, zero extra VRAM) via `--speculative`
```

<details><summary><b>Full architecture diagram</b></summary>

```
┌─────────────────────────────────────────────────────────────────────┐
│ BenchmarkHarness — load config → detect backend → engine → run      │
├─────────────────────────────────────────────────────────────────────┤
│ InferenceEngine (model-agnostic facade)                             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌───────────┐ │
│  │ AR Backend   │ │ NLLB Backend │ │ Diff Backend │ │  Custom   │ │
│  │ CUDA Graph   │ │ Enc-Dec (BART│ │ CUDA Graph   │ │  Plugin   │ │
│  │ PagedAttn    │ │ beam search) │ │ Fast-dLLM    │ │           │ │
│  │ cudaMalloc   │ │ forced BOS   │ │ Batched CFG  │ │           │ │
│  │ Spec Decode  │ │ src/tgt lang  │ │              │ │           │ │
│  └──────────────┘ └──────────────┘ └──────────────┘ └───────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│ JIT Compiler (nvcc/metal) · Triton Kernels · PagedAttention        │
│ cudaMallocAsync · CUDA Graphs · Flash SDPA · NCCL P2P               │
│ Model Presets (11) · Quantization (bf16/fp16/int8/int4)              │
├─────────────────────────────────────────────────────────────────────┤
│ JSONLLoader(orjson) → Chunker(token-level) → Filter(numpy) →       │
│ AsyncPipeline(lock-free tok, pinned mem) → BatchAssembly            │
└─────────────────────────────────────────────────────────────────────┘
```

</details>

---

## Quality Assurance (Non-Negotiable)

Every benchmark run includes **3 quality metrics** computed in parallel on a human-verified golden reference set:

| Metric | Type | Target |
|--------|------|--------|
| **BERTScore** | Reference-based (bert-base-multilingual-cased) | ≥ 0.55 |
| **COMET-22** | Reference-based (wmt22-comet-da) | ≥ 0.72 |
| **COMET-Kiwi** | Reference-free (wmt22-cometkiwi-da) | ≥ 0.72 |

Additional quality modules existed (`ensemble.py`, `confidence.py`) but were removed in v3.6 as dead code. If any metric drops >2% from baseline, CI blocks the deployment.

**Quantization rule**: FP8 (H200) is quality-safe by default. INT8 requires calibration data. INT4 **must** pass the quality benchmark before production use. The harness runs this automatically — you'll see whether quantized quality meets targets before deploying.

> ⚠️ **On H200 with 4B models, TE FP8 and INT8 are counterproductive for throughput** (40% and 73% slower respectively, measured 2026-06-24). Use BF16 for 4B models unless VRAM-constrained. 12B+ models may benefit from FP8 where compute-bound. See M1.5, M2.7.

---

## Optimization inventory — and what is *actually* on the hot path

> **Honest status (June 2026):** The headline "37/39 wired" from earlier docs is
> **not accurate.** The production AR hot path is plain eager `model(...)` with
> HuggingFace `past_key_values`, accelerated only by **`torch.compile`** and
> **Transformer-Engine FP8**. Many optimizations below are **built but gated off**
> (hardcoded `False`, `if False:`, env-gated, safety-gated, or
> captured-but-never-replayed).
>
> **The authoritative status of every feature is
> [`docs/ARCHITECTURE.md` §8 Feature Status](docs/ARCHITECTURE.md#8-feature-status-the-truth-table).**
> The tables below are the *design inventory* — the "gain/speedup" columns are
> design targets, not measured hot-path speedups. Treat any row not marked
> ✅ in ARCHITECTURE as *not helping throughput today*.

### Condensed real status

| Status | Examples |
|---|---|
| ✅ Wired (helps today) | `torch.compile(reduce-overhead)` (measured <5% on 4B, M2.1), TE FP8 (⚠️ 40% slower on 4B, M1.5), Flash SDPA (measured 1.17-1.23×, M2.4), pinned-memory pipeline (measured 2.1× H2D, M2.5), orjson/pigz (measured 4.2×/1.62×, M2.8), async prefetch, shuffle, checkpoint/resume, extrapolation |
| 🟡 Built-but-gated / stats-only | CUDA-graph decode (captured, never replayed), Fast-dLLM cache (stats-only), token-level chunking (pipeline uses the slow path) |
| 🔬 Experimental (opt-in) | Speculative decoding (⚠️ SLOWER on 4B at bs=1, 25 vs 62 tok/s, M2.3), continuous batching + PagedAttention (measured 60-87.5% KV savings for variable-length, M2.6) |
| ⚠️ Broken/Disabled | cudaMallocAsync, TensorRT decode, JIT CUDA C++ kernels (sources nulled), Metal wrapper (non-functional) |
| 💀 Dead code | fused-kernel injection (`if False:`), INT8 KV-cache (constructed, never read), tensor parallelism (`apply_tensor_parallelism` never called), perf-regression gate |
| 🗑️ Removed (v3.6) | ensemble, confidence, dashboard server, Nsight profiler, run_models |

> Items marked **(experimental ...)** or **(disabled ...)** in the tables below
> are not on the default hot path.

<details open><summary><b>Memory</b> (5)</summary>

| # | Optimization | Memory Gain | Quality Impact |
|---|-------------|------------|---------------|
| 1 | **cudaMallocAsync** — stream-ordered GPU allocator (⚠️ disabled — incompatible with `torch.compile` PyTorch 2.6) | +0% (not active) | ✅ None |
| 2 | **PagedAttention** — block-level KV-cache (vLLM-style) (🔬 real via `--continuous-batching --paged-attention`; ⚠️ disabled on the default AR path — `_use_paged_attention=False`) | 40–70% less KV memory (CB path only) | ✅ None — mathematically identical |
| 3 | **Pinned memory pipeline** — page-locked host tensors for DMA | 3–5× transfer speed | ✅ None — data unchanged |
| 4 | **INT8 KV-cache quantization** — per-channel asymmetric (⚠️ object constructed but **never read/written** on the AR hot path; pure no-op) | +0% (not active) | ✅ None |
| 5 | **INT4/INT8 weight quantization** — AWQ / bitsandbytes / FP8 | 2–4× smaller model | ⚠️ See below |

> **⚠️ Weight quantization quality impact (honest):**
>
> | Precision | Method | BLEU impact | Safe without calibration? |
> |-----------|--------|------------|--------------------------|
> | **FP8** | Transformer Engine (H200 only) | **≈ 0.0** — undetectable | ✅ Yes — dynamic per-tensor scaling |
> | **INT8** | TRT calibrator or bitsandbytes | **< 0.3** — usually undetectable | ❌ Needs 100–500 calibration sentences |
> | **INT4** | AWQ activation-aware or NF4 | **0.5–2.0** — measurable, often acceptable | ❌ Needs calibration + quality validation |
>
> **The rule**: never deploy a quantized model without running the quality benchmark
> (`./run.sh --benchmark-only`) against your golden reference set. FP8 is safe by default.
> INT8 and INT4 are opt-in — you enable them explicitly in config and should validate.

</details>

<details><summary><b>Compute</b> (12)</summary>

| # | Optimization | Impact |
|---|-------------|--------|
| 6 | **CUDA Graph decode** — graph captured once, replayed per token (⚠️ disabled — captured but **never replayed** on the hot path; see `docs/ARCHITECTURE.md` §8 #4) | +0% (not active) |
| 7 | **CUDA Graph denoising** — per-step replay for diffusion (🟡 opt-in, `use_cuda_graph_for_step=True`, default off) | +15–30% per step |
| 8 | **torch.compile(reduce-overhead)** — frame-level CUDA graph fusion | +15–40% |
| 9 | **Flash SDPA + Mem-Efficient** — FlashAttention-2 via PyTorch backend | 2–4× attention |
| 10 | **Triton fused kernels** — RMSNorm+residual, SwiGLU gate×up in 1 kernel (⚠️ disabled — injection hardcoded `if False:`, `autoregressive.py:761`) | +0% (not active) |
| 11 | **JIT CUDA C++ kernels** — Fused QKV+RoPE via nvcc (⚠️ disabled — sources set to `None`, "architecturally broken") | +0% (not active) |
| 12 | **JIT Metal MSL kernels** — Fused RMSNorm+residual on Apple GPU (⚠️ non-functional wrapper, falls back to eager) | +0% (not active) |
| 13 | **Transformer Engine FP8** — FP8 tensor core matmul on H200 [quality-safe →](#40-extreme-optimizations-all-wired-into-hot-paths) | 2× matmul |
| 14 | **NCCL P2P + all-reduce** — NVLink direct GPU-GPU transfers | near-zero overhead |
| 15 | **Batched CFG** — cond+uncond in one diffusion forward pass | 2× guidance |
| 16 | **Fast-dLLM caching** — skip >90% forward passes when tokens unchanged (experimental — cache populated but full forward still executed) | up to 10× |
| 17 | **INT8 embedding quantization** — diffusion embedding table | 2–4× bandwidth |

</details>

<details><summary><b>CPU / Tokenization / I/O</b> (9)</summary>

| # | Optimization | Impact |
|---|-------------|--------|
| 18 | **Thread-local tokenizers** — one SentencePiece per worker, lock-free | N× throughput |
| 19 | **orjson** — Rust JSON parser | 4–10× parsing |
| 20 | **Numpy garbage detection** — vectorized non-ASCII count | 30–50× |
| 21 | **Token-level chunking** — no decode→re-encode cycle | 30–40% less CPU |
| 22 | **Parallel gzip (pigz)** — multi-threaded decompression | 3–6× I/O |
| 23 | **Memory-mapped I/O** — zero-copy mmap file reads | 2–4× I/O |
| 24 | **O(1) throughput** — prefix-sum rolling window | constant-time |
| 25 | **Parallel metrics** — BERTScore ‖ COMET-22 ‖ COMET-Kiwi concurrently | wall = max |
| 26 | **COMET model cache** — download once per process lifetime | −5–10s per run |

</details>

<details><summary><b>Parallelism & Distribution</b> (5)</summary>

| # | Optimization | Impact |
|---|-------------|--------|
| 27 | **CUDA stream overlap** — async H2D while GPU computes | +15–25% |
| 28 | **Continuous batching** — iteration-level scheduling, no idle bubbles (🔬 real when `--continuous-batching --paged-attention` + batch ≥ 2; not active on default path) | 1.5–3× |
| 29 | **Speculative decoding** — self-speculative (early-layer draft, zero extra VRAM) (🔬 opt-in; requires `TR_ENABLE_EXPERIMENTAL_SPECULATIVE=1` in addition to `--speculative`) | 1.5–3× |
| 30 | **Model-agnostic backends** — one pipeline, any architecture | unlimited |
| 31 | **Plugin system** — custom models via drop-in `.py` files | zero framework changes |

</details>

<details><summary><b>Observability, Quality & Workflow</b> (8)</summary>

| # | Optimization | Impact |
|---|-------------|--------|
| 32 | **Prometheus + Grafana** — 20+ metrics, alerts via external Grafana, 8-panel dashboard | production monitoring |
| 33 | **Performance regression** — Welch t-test CI gating (💀 implemented but **never wired** into any run/CI path) | dead code |
| 34 | ~~**Ensemble translation**~~ — multi-model voting for quality verification (🗑️ removed in v3.6) | removed |
| 35 | **TensorRT engine** — GPU-compiled via nvcc+ONNX, cached to disk (⚠️ **safety-gated to refuse decode**; no KV passthrough → corrupted output; broken on TRT 11.x; falls back to AR) | +0% (not functional) |
| 36 | **Pre-compiled Docker** — JIT + TRT built at image build time | zero first-run latency |
| 37 | **Docker Compose obs** — `make dashboard` → Grafana+Prometheus | one-command monitoring |
| 38 | **Unified `setup.sh`** — auto-detect platform, install everything | one command |
| 39 | **Unified `run.sh`** — `--tensorrt --diffusion --observability` | zero overhead |

</details>

---

## Backends

<details open><summary><b>Autoregressive</b> — TranslateGemma, LLaMA, GPT, Mistral, etc.</summary>

```yaml
model:
  model_path: "google/translategemma-12b-it"
  backend_type: "auto"
```
</details>

<details><summary><b>Encoder-Decoder</b> — NLLB-200 (600M, 1.3B, 3.3B, 54B MoE)</summary>

```yaml
model:
  model_path: "facebook/nllb-200-distilled-600M"
  backend_type: "encoder_decoder"     # auto-detected from "nllb" in path
  nllb_source_lang: "eng_Latn"
  nllb_target_lang: "tur_Latn"
```
```bash
./run.sh --nllb                       # shorthand: sets backend + languages
./run.sh --nllb --model ministral-3b-bf16
```
Auto-detected: `nllb` in model path, `M2M100ForConditionalGeneration` in config.json.
</details>

<details><summary><b>Diffusion</b> — LLaDA, Dream, MDLM, E2D2, BD3-LM</summary>

```yaml
model:
  model_path: "GSAI-ML/LLaDA-8B-Instruct"
  backend_type: "diffusion"
  diffusion_steps: 256
  guidance_scale: 1.0
```
Auto-detected: `LLaDA`, `Dream`, `mdlm`, `e2d2`, `bd3lm`, any model with `diffusion_steps` in config.json.
</details>

<details><summary><b>TensorRT</b> — GPU-compiled engine (⚠️ broken on TRT 11.x; falls back to AR; see Known Limitations)</summary>

```yaml
model:
  use_tensorrt: true
  tensorrt_precision: "fp16"     # fp16 | fp8 | int8
```
First run builds engine (5–15 min, cached forever). Subsequent runs load in ~50ms. Falls back to AR if TRT unavailable.
</details>

<details><summary><b>Custom Plugin</b> — your own model</summary>

```python
class MyBackend(InferenceBackend):
    def load(self): ...
    def translate_batch(self, batch): ...
    def warmup(self, batches): ...
    def is_loaded(self): return self._loaded

class MyPlugin(CustomModelPlugin):
    name = "my_model"
    def create_backend(self, cfg): return MyBackend(cfg)

register_plugin(MyPlugin())   # drop in ~/.tr_benchmark/plugins/
```
</details>

---

## Configuration

<details><summary><b>Full YAML reference</b> (click to expand)</summary>

```yaml
backend: "auto"

model:
  model_path: "google/translategemma-12b-it"
  tokenizer_path: ""
  max_input_tokens: 512
  max_new_tokens: 512
  temperature: 0.0
  dtype: "auto"
  tensor_parallel_size: 0
  use_flash_attention: true
  backend_type: "auto"           # auto | autoregressive | encoder_decoder | diffusion | custom
  use_tensorrt: false
  tensorrt_precision: "fp16"
  # ── v3.6: quantization & model presets ──
  quantization: "bf16"           # bf16 | fp16 | int8 | int4
  # ── v3.6: NLLB encoder-decoder ──
  nllb_source_lang: "eng_Latn"
  nllb_target_lang: "tur_Latn"
  # ── v3.5: speculative ──
  use_speculative: false
  speculative_mode: "self"       # self | draft_model
  speculative_num_tokens: 3
  # ── v3.5: experimental features ──
  use_paged_attention: false
  use_continuous_batching: false

runtime:
  target_duration_seconds: 7200
  checkpoint_interval_seconds: 300
  heartbeat_interval_seconds: 10
  metrics_sample_rate_hz: 1
  seed: 42

data:
  input_paths: ["./data/input/*.jsonl.gz"]
  output_dir: "./output"
  reference_set_path: "./data/references/golden_en_tr.jsonl"
  prefetch_workers: 4
  shuffle: true
  min_chunk_tokens: 10
  max_garbage_ratio: 0.95
  chunk_overlap_tokens: 50

extrapolation:
  total_clearnet_non_tr_tokens: 6230000000000
  gpu_cost_per_hour_usd: null
```

</details>

---

## Runtime JIT Kernel Compilation

Three hand-tuned kernels ship as Python strings and compile on the target machine — cached forever.

| Kernel | Compiler | Target | |
|--------|----------|--------|-|
| `fused_qkv_rope` | `nvcc` (CUDA C++) | sm80/89/90 | ⚠️ disabled |
| `fused_swiglu_mlp` | `nvcc` (CUDA C++) | sm80/89/90 | ⚠️ disabled |
| `fused_rms_norm_residual` | `xcrun metal` (MSL) | Apple GPU | ⚠️ non-functional |

Cold: 5–15s compile → cache to `~/.cache/tr_benchmark/kernels/<hash>.so`. Warm: ~10ms load from disk. Gracefully skipped if compiler unavailable.

```bash
./run.sh --precompile          # pre-compile all kernels
./run.sh --force-recompile     # force rebuild
rm -rf ~/.cache/tr_benchmark/  # clear all caches
```

---

## Docker

```bash
docker build -t tr-benchmark:3.6 .                 # pre-compiled image
docker build -t tr-benchmark:3.6-trt --build-arg WITH_TENSORRT=1 .
docker run --rm --gpus '"device=0,1"' --ipc=host --ulimit memlock=-1 \
  -v $(pwd)/data:/data tr-benchmark:3.6 --config /data/config.yaml

make dashboard   # Prometheus + Grafana on :3000 (admin/admin)
```

[Docker, K8s & NCCL details →](docs/COMPILATION_GUIDE.md)

---

## Observability

**`./run.sh --observability`** exposes **20+ Prometheus metrics** at `:9090/metrics`:

| Type | Metrics |
|------|---------|
| Counters | `batches_total`, `tokens_translated_total`, `errors_total` |
| Gauges | `throughput_tps`, `gpu_utilization_pct`, `gpu_temperature_celsius`, `quality_bertscore`, `quality_comet` … |
| Histograms | `batch_latency_seconds`, `decode_time_seconds` |

**`make dashboard`** launches Prometheus + Grafana with an 8-panel pre-built dashboard (throughput, GPU util/temp/memory, latency, quality scores, pipeline health). Alerts are defined in the external Grafana config — there is no in-process alerting.

**`make dashboard`** launches Prometheus + Grafana with an 8-panel pre-built dashboard (throughput, GPU util/temp/memory, latency, quality scores, pipeline health).

[Observability docs →](docs/COMPILATION_GUIDE.md#6-docker-compose-observability)

---

## Testing

```bash
make test              # ~75 tests in 27 files
make lint              # ruff check
make format            # ruff format

# CI matrix: lint → CPU tests (3.11, 3.12) → macOS MPS tests → Docker build → Trivy scan
# Push to main: + E2E 120s pipeline test on macOS MPS
# Perf regression: Welch t-test blocks merge if throughput drops >5%
```

---

## Troubleshooting

| Issue | |
|-------|-|
| `./setup.sh` fails | `python3.11 --version`. Use `--python python3.12`. |
| CUDA OOM | Reduce batch size. Enable **PagedAttention** or INT8 KV-cache. Use `--tensorrt` for better planning. |
| `nvcc` not found | `apt install cuda-toolkit-12-4`. JIT falls back to Triton/PyTorch automatically. |
| TRT not found | `apt install tensorrt python3-libnvinfer`. Or skip — TRT is optional. |
| Metal toolchain missing | `xcodebuild -downloadComponent MetalToolchain`. Falls back to eager otherwise. |
| TRT build slow | First: 5–15 min. After: ~50ms from cache. Pre-build: `./run.sh --precompile`. |
| Diffusion not detected | Use `./run.sh --diffusion` or set `backend_type: "diffusion"`. |
| Docker GPUs not visible | Install `nvidia-container-toolkit`, pass `--gpus`. |

---

## Project Layout

```
H200Research/
├── benchmark/          # 75 Python modules
│   ├── hardware/       # backend detection · CUDA graphs · JIT compiler · fused kernels · TensorRT builder
│   ├── inference/      # engine facade · backends/{AR, NLLB, diffusion, TRT, custom} · PagedAttention · continuous batching · speculative decoding
│   ├── data/           # orjson loader · numpy filters · lock-free pipeline · parallel gzip
│   ├── quality/        # BERTScore · COMET-22 · COMET-Kiwi · BLEU · chrF++
│   ├── metrics/        # GPU sampler · system sampler · batch logger · O(1) throughput
│   ├── orchestration/  # harness · checkpointing · signals · resume
│   ├── reporting/      # aggregator · extrapolation · JSON/Markdown reports
│   ├── observability/  # Prometheus client · regression detection
│   └── config/         # Pydantic v2 schema · model presets (9 models)
├── scripts/            # standalone runner scripts (benchmark_all_models.py, run_new_models.py)
├── tests/              # 27 test files
├── .github/            # CI workflow · Grafana dashboard JSON · Prometheus config
├── setup.sh            # ★ one-command install (auto-detect platform)
├── run.sh              # ★ unified launcher
├── Makefile            # make setup | make run-quick | make dashboard
├── Dockerfile          # multi-stage with pre-compiled JIT kernels
├── config.yaml         # production H200/CUDA configuration
└── docs/               # compilation guide · optimization roadmap · SDD · PRD · SRS
```

---

## References

TranslateGemma (Google DeepMind, 2025) · LLaDA (Nie et al., 2025) · MDLM (Sahoo et al., NeurIPS 2024) · E2D2 (Arriola et al., 2025) · BD3-LM (ICLR 2025) · Fast-dLLM (Wu et al., NVIDIA/MIT 2025) · PagedAttention (Kwon et al., SOSP 2023) · FlashAttention-2 (Dao, 2023)

---

## Known Limitations

> ⚠️ See [`docs/ARCHITECTURE.md` §8 Feature Status](docs/ARCHITECTURE.md#8-feature-status-the-truth-table) for the authoritative list of which optimizations are wired vs. gated. The limitations below describe *constraints*; Feature Status describes *real activation state*.

| Category | Limitation | Detail |
|---|---|---|
| **What's actually on the hot path** | The production AR hot path is plain eager `model(...)` + `torch.compile(reduce-overhead)` + TE FP8. Most listed "optimizations" are built but gated off (hardcoded `False`, env-gated, captured-not-replayed, or stats-only). | See the condensed status table above; authoritative in [`docs/ARCHITECTURE.md` §8](docs/ARCHITECTURE.md#8-feature-status-the-truth-table). |
| **TensorRT backend** | Not functional — (a) no KV-cache passthrough in the decode loop → output corrupted after 1st token; safety-gated to refuse unless `allow_trt_decode_without_kv_cache`. (b) Broken on TRT 11.x (removed `EXPLICIT_BATCH` / `num_layers`). Falls back to AR backend. | `benchmark/inference/backends/tensorrt_backend.py:334`; `hardware/trt_builder.py:394,633` |
| **CUDA Graph decode** | Deprecated — graph captured in warmup but *never replayed* in `_extreme_decode`. `cuda_graphs.py` emits `FutureWarning` on import. The capture cost is paid, the benefit is never collected. | `benchmark/inference/backends/autoregressive.py:1548` |
| **Fused Triton/CUDA kernels** | Injection hardcoded `if False:` (`autoregressive.py:761`). JIT CUDA C++ kernel sources set to `None`. Metal wrapper non-functional. | Commits `804c0a6`, `9fa3397`. |
| **cudaMallocAsync** | Disabled — incompatible with `torch.compile` `cudagraph_trees` in PyTorch 2.6. | `autoregressive.py:654` (commented) |
| **PagedAttention (AR path)** | Hardcoded `False` — `_use_paged_attention=False`. `_convert_to_paged` referenced only in comments, doesn't exist. Paged KV is real only via the `--continuous-batching --paged-attention` path. | `autoregressive.py:585-596` |
| **INT8 KV-cache quantization** | Object constructed at load, but never `.update()`/`.get()`-ed — pure no-op on the hot path. | `autoregressive.py:1184` |
| **Fast-dLLM caching** | Cache hit counter is incremented, but the full forward is always executed — stats-only. | `benchmark/inference/backends/diffusion.py:709` |
| **Continuous batching** | Real but gated behind `--continuous-batching --paged-attention` (CUDA). Actual threshold is **batch size ≥ 2** (not 8 as older docs claimed). | `continuous_batcher.py:100` |
| **Speculative decoding** | Env-gated (`TR_ENABLE_EXPERIMENTAL_SPECULATIVE=1`), greedy-only, serial per-sequence. Verify runs full L layers (not L−D). Draft-model mode silently tolerates tokenizer mismatch. | `speculative.py:1129` |
| **Tensor parallelism** | `apply_tensor_parallelism` is defined but never called. Multi-GPU only via `device_map="auto"`, and a single-GPU fast path (<10% of one GPU) bypasses it for every model this benchmark actually runs. | `hardware/parallelism.py:238` (never called) |
| ~~**Ensemble + Confidence**~~ | 🗑️ Removed in v3.6 — modules were dead code (never imported outside own files). |  |
| **perf_regression (Welch t-test)** | Implemented but has **zero callers** — not wired into any run or CI path. | `observability/perf_regression.py` |
| **"6 alerts"** | No alerting code exists in-process. Alerting is external (Grafana via Docker Compose). | `observability/prometheus_metrics.py` — no alert rules |
| ~~**AR/TRT `input_tokens` counts padding**~~ | ✅ Fixed v3.6 — AR and TRT backends now use `attention_mask.sum()` for accurate token counts. |  |
| ~~**Prometheus quality gauges**~~ | ✅ Fixed v3.6 — `quality_bleu` and `quality_chrf` gauges now populated from actual computed scores. |  |
| **BLEU / chrF++** | These are actually **wired** into the quality benchmark (`quality/benchmark.py:283-284`) — the old claim of "stubs" was stale. (The Prometheus gauge bug above is separate — and now fixed.) |  |
| **Markdown quality report** | Omits BERTScore and COMET-Kiwi from the quality table (only in JSON report). | `reporting/markdown_report.py` |
| **NLLB encoder-decoder** | New in v3.6. 4 model sizes. Supports `eng_Latn→tur_Latn` and other NLLB language pairs. | |
| **Extrapolation model** | **Constant-throughput assumption validated** — 2.2h degradation test (122K batches, 110M tokens) showed zero detectable throughput change (+0.1%/hr, R²=0.000046) for 4B on H200. See `docs/MEASUREMENT_PLAN.md` §B.20. SEM-based CIs + bootstrap. | |
| **torch.compile on MPS** | Disabled — `torch.compile` is unavailable on MPS backends. | |
| **Verified toolchain (H200)** | **torch 2.6.0+cu124** — the only tested combination for SM90. 2.11+cu130 crashes (no cuDNN SM90 plans). 2.12+cu126 works but flash_attn ABI breaks. PyTorch built-in SDPA handles all attention backends — `flash_attn` is NOT needed. TE FP8 requires source build (`--no-build-isolation`). `setup.sh` pins `torch==2.6.0`. | |
| ~~**Chunker tail-drop**~~ | ✅ Fixed v3.6 — minimum chunk size lowered to 1 token; no more silent tail truncation. |  |
| **BERTScore model** | Uses `bert-base-multilingual-cased` (not DeBERTa as older SRS stated). | `quality/metrics_bertscore.py:32` |
| **NLLB-200 600M** | Source-language prefix not applied at inference when not detected — may produce wrong-language output. | `inference/backends/nllb.py` |
| **Throughput baseline (measured)** | TranslateGemma 4B BF16 Flash SDPA: 13,223 tok/s at bs=512, 735 tok/s at bs=16. NLLB-200 600M: 581 tok/s at bs=8. Full data in `docs/MEASUREMENT_PLAN.md`. | |
| **Quality reference set** | Single reference per source, no bootstrap CIs on quality scores, no paired significance testing. The 10-pair minimum is statistically undersized for production use. | `quality/benchmark.py:13-19` |
| **INT8 quantization on H200** | 41% memory savings but 3.7× SLOWER (213 vs 792 tok/s at bs=16). Dequant overhead dominates on 4B models with 130+ GB free VRAM. Counterproductive unless VRAM-constrained. Measured 2026-06-24. | M2.7 |
| **TE FP8 on H200 (4B)** | 40% SLOWER than BF16 (497 vs 832 tok/s), 0% memory saved. Cast overhead dominates for small models. BLOCKED on torch 2.6.0 (TE 2.16 requires torch >=2.11). May benefit 12B+ models. Measured 2026-06-24. | M1.5 |
| **4B model quality** | BLEU ≈ 0.8 vs target 25 — TranslateGemma 4B is NOT a production translation model. Development and pipeline-testing only. Measured 2026-06-24. | M4.1 |
| **torch.compile on 4B** | <5% speedup at bs=16 (727.8 vs 735.3 tok/s). Python decode loop dominates at this size. Primary benefit for 12B+ models. Measured 2026-06-24. | M2.1 |
| **Speculative decoding on 4B** | 25 tok/s at bs=1 (SLOWER than non-spec 62 tok/s), 38% acceptance rate. 8-layer draft + 34-layer full verify overhead exceeds gain at 4B depth. May benefit 48+ layer models. Measured 2026-06-24. | M2.3 |

---

*75 Python modules. 27 test files (~75 tests). 1 command to install, 1 command to run. Model-agnostic.*
*Performance numbers measured 2026-06-24 on 2× NVIDIA H200 NVL (asus02). See [`docs/MEASUREMENT_PLAN.md`](docs/MEASUREMENT_PLAN.md).*
