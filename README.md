# Turkish Corpus Translation Benchmark · v3.6

**Model-agnostic. Extreme low-level optimized. One-command to run.**


> *How many days to translate 6+ trillion English tokens into Turkish — on 2 NVIDIA H200s, with academic-grade quality?*

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
./run.sh --tensorrt          # +20-50% throughput (CUDA only)
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

[Full CLI reference →](docs/COMPILATION_GUIDE.md)

---

## What It Measures

| Backend | Model | Throughput (2× H200) | Days for 6.23T tokens |
|---------|-------|---------------------|----------------------|
| AR (FP8) | TranslateGemma 12B | ~500–800 tok/s | 90–180 |
| AR + TensorRT | TranslateGemma 12B | ~600–1100 tok/s | 60–120 |
| Encoder-Decoder | NLLB-200 3.3B | ~400–600 tok/s | 120–180 |
| Diffusion | LLaDA 8B | ~800–1600 tok/s | 45–90 |
| AR (INT8) | Ministral 3B / Gemma4 QAT | ~300–500 tok/s | 145–240 |

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
| **BERTScore** | Reference-based (DeBERTa-xlarge-mnli) | ≥ 0.55 |
| **COMET-22** | Reference-based (wmt22-comet-da) | ≥ 0.72 |
| **COMET-Kiwi** | Reference-free (wmt22-cometkiwi-da) | ≥ 0.72 |

Additional quality modules exist (`ensemble.py`, `confidence.py`) but are not yet wired into the default quality hot path. If any metric drops >2% from baseline, CI blocks the deployment.

**Quantization rule**: FP8 (H200) is quality-safe by default. INT8 requires calibration data. INT4 **must** pass the quality benchmark before production use. The harness runs this automatically — you'll see whether quantized quality meets targets before deploying.

---

## 39 Optimizations (37 verified wired, 2 experimental)

> **Honest status (June 2026):** 37 optimizations are fully wired, tested, and production-safe.
> The remaining 2 are implemented as modules but are experimental and not available in the default
> hot path. Items marked **(experimental ...)** in the tables below are the 2 in-development ones.

<details open><summary><b>Memory</b> (5)</summary>

| # | Optimization | Memory Gain | Quality Impact |
|---|-------------|------------|---------------|
| 1 | **cudaMallocAsync** — stream-ordered GPU allocator | +10–25% efficiency | ✅ None |
| 2 | **PagedAttention** — block-level KV-cache (vLLM-style) (experimental — not yet feeding paged KV to model) | 40–70% less KV memory | ✅ None — mathematically identical |
| 3 | **Pinned memory pipeline** — page-locked host tensors for DMA | 3–5× transfer speed | ✅ None — data unchanged |
| 4 | **INT8 KV-cache quantization** — per-channel asymmetric | 2× effective cache | ✅ None — used in production (vLLM, TGI) |
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
| 6 | **CUDA Graph decode** — `graph.replay()` per token instead of 200+ launches | +20–40% |
| 7 | **CUDA Graph denoising** — per-step replay for diffusion | +15–30% per step |
| 8 | **torch.compile(reduce-overhead)** — frame-level CUDA graph fusion | +15–40% |
| 9 | **Flash SDPA + Mem-Efficient** — FlashAttention-2 via PyTorch backend | 2–4× attention |
| 10 | **Triton fused kernels** — RMSNorm+residual, SwiGLU gate×up in 1 kernel | 2–3× |
| 11 | **JIT CUDA C++ kernels** — Fused QKV+RoPE via nvcc (2 kernels vs 5) | 2.5× |
| 12 | **JIT Metal MSL kernels** — Fused RMSNorm+residual on Apple GPU | 3× on MPS |
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
| 28 | **Continuous batching** — iteration-level scheduling, no idle bubbles (experimental module, not wired into default hot path) | 1.5–3× |
| 29 | **Speculative decoding** — self-speculative (early-layer draft, zero extra VRAM) via `--speculative` flag. Draft-model mode also available. | 1.5–3× |
| 30 | **Model-agnostic backends** — one pipeline, any architecture | unlimited |
| 31 | **Plugin system** — custom models via drop-in `.py` files | zero framework changes |

</details>

<details><summary><b>Observability, Quality & Workflow</b> (8)</summary>

| # | Optimization | Impact |
|---|-------------|--------|
| 32 | **Prometheus + Grafana** — 20+ metrics, 6 alerts, 8-panel dashboard | production monitoring |
| 33 | **Performance regression** — Welch t-test CI gating | catches regressions |
| 34 | **Ensemble translation** — multi-model voting for quality verification | academic quality |
| 35 | **TensorRT engine** — GPU-compiled via nvcc+ONNX, cached to disk | +20–50% on H200 |
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

<details><summary><b>TensorRT</b> — compiled engine, +20-50% on H200 (CUDA only)</summary>

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
| `fused_qkv_rope` | `nvcc` (CUDA C++) | sm80/89/90 | 2.5× |
| `fused_swiglu_mlp` | `nvcc` (CUDA C++) | sm80/89/90 | 2.0× |
| `fused_rms_norm_residual` | `xcrun metal` (MSL) | Apple GPU | 3.0× |

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

**6 alerts**: low throughput, high data starvation, high GPU temp, OOM risk, quality regression, stalled benchmark.

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
│   ├── quality/        # BERTScore · COMET-22 · COMET-Kiwi · ensemble · confidence
│   ├── metrics/        # GPU sampler · system sampler · batch logger · O(1) throughput
│   ├── orchestration/  # harness · checkpointing · signals · resume
│   ├── reporting/      # aggregator · extrapolation · JSON/Markdown reports
│   ├── observability/  # Prometheus client · Grafana dashboard · Nsight profiler · regression detection
│   └── config/         # Pydantic v2 schema · model presets (9 models)
├── benchmarks/         # convenience scripts for running model benchmarks
├── tests/              # 27 test files
├── .github/            # CI workflow · Grafana dashboard JSON · Prometheus config
├── setup.sh            # ★ one-command install (auto-detect platform)
├── run.sh              # ★ unified launcher
├── benchmark_all_models.py  # comparative MPS benchmark across all presets
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

| Limitation | Detail |
|-----------|--------|
| **NLLB encoder-decoder** | New in v3.6 — NLLB-200 support via `--nllb` flag or `backend_type: "encoder_decoder"`. 4 model sizes (600M, 1.3B, 3.3B, 54B MoE). Supports `eng_Latn→tur_Latn` and any other NLLB language pair. |
| **Quantization levels** | New in v3.6 — `--quantization bf16|fp16|int8|int4` flag. INT8 via bitsandbytes, INT4 via NF4. Model presets with pre-configured quantization (e.g. `translategemma-4b-int8`). |
| **Model presets** | 11 presets in `benchmark/config/model_presets.py`: TranslateGemma 4B (bf16/int8/int4), Ministral 3B, Gemma4 QAT E2B/E4B (ct/int4/q4_0), DiffusionGemma 26B. |
| **PagedAttention** | Experimental — block-level KV-cache is allocated with a no-op converter; paged KV is not yet fed to the model forward pass. Opt-in via `--paged-attention` flag. |
| **Continuous batching** | Not wired — iteration-level scheduler module exists but is not connected to the default hot path. Opt-in via `--continuous-batching` flag. |
| **TensorRT backend** | Not functional on TRT 11.x — API removed `trt.Error` and `EXPLICIT_BATCH`. Builder needs update. |
| **Speculative Decoding** | Available via `--speculative` flag. Self-speculative mode (early-layer draft, zero extra VRAM) is wired and functional. Draft-model mode also available. |
| **Extrapolation model** | Assumes constant throughput — uses SEM-based CIs and bootstrap resampling. Does not account for throughput degradation over long runs or hardware throttling. |
| **Ensemble + Confidence** | Modules exist (`ensemble.py`, `confidence.py`) but not wired into default quality hot path. |
| **COMET / BLEU / chrF++** | COMET-Kiwi (reference-free) and COMET-22 (reference-based) are wired. BLEU and chrF++ modules exist as stubs but are not called from quality benchmark hot path. |
| **Fast-dLLM caching** | Diffusion cache is populated without skipping — full forward passes always executed. |
| **torch.compile on MPS** | Disabled — `torch.compile` is unavailable on MPS backends. Falls back to eager execution on Apple Silicon. |

---

*75 Python modules. 27 test files (~75 tests). 39 optimizations (37 verified wired, 2 experimental). One command to install, one command to run. Model-agnostic.*
