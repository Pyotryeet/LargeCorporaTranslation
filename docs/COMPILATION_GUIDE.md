# Compilation & Deployment Guide — v3.7

**One command to install, one command to run. Every platform. Every backend.**

> ⚠️ **Some feature claims in this guide are aspirational.** The JIT CUDA kernels
> are all permanently deleted in v3.7. The remaining optimizations are
> non-functional. See [`ARCHITECTURE.md` §8 Feature Status`](ARCHITECTURE.md#8-feature-status-the-truth-table)
> for the authoritative wired-vs-gated reality.

---

## Table of Contents

1. [The 30-Second Quickstart](#1-the-30-second-quickstart)
2. [Setup (setup.sh)](#2-setup-setupsh)
3. [Run (run.sh)](#3-run-runsh)
4. [Make Targets](#4-make-targets)
5. [Docker (Pre-Compiled Images)](#5-docker-pre-compiled-images)
6. [Docker Compose Observability](#6-docker-compose-observability)
9. [Model Selection Guide](#9-model-selection-guide)
10. [Performance Tuning](#10-performance-tuning)
11. [Multi-Node Cluster](#11-multi-node-cluster)

---

## 1. The 30-Second Quickstart

```bash
git clone <repo> && cd H200Research
./setup.sh            # Auto-detects platform, installs everything
./run.sh --dry-run    # Smoke test (60s)
./run.sh --quick      # Evaluation (5 min)
./run.sh              # Full production benchmark (2 hours)
```

**That's the entire interface.** No platform-specific flags. No manual pip installs. No CUDA version checks. The scripts auto-detect macOS Apple Silicon vs Linux CUDA vs CPU and do the right thing.

---

## 2. Setup (`setup.sh`)

### Usage

```bash
./setup.sh                    # Full install, auto-detect platform
./setup.sh --quick            # Dev setup: skip model download
./setup.sh --minimal          # Bare minimum: Python + PyTorch + shared deps only
./setup.sh --cuda             # Force CUDA path (Linux only)
./setup.sh --cpu              # Force CPU-only
./setup.sh --no-model-dl      # Skip HuggingFace model download
./setup.sh --python python3.12  # Use different Python binary
./setup.sh --venv .myenv      # Custom venv directory
./setup.sh --help             # Full flag reference
```

### What It Does

```
Step 1: Platform detection
  macOS arm64 → MPS mode
  Linux + nvidia-smi → CUDA mode
  Otherwise → CPU mode

Step 2: Prerequisites check
  Python 3.11+, pip, nvidia-smi (CUDA), macOS version (MPS)

Step 3: Virtual environment
  python -m venv .venv → source .venv/bin/activate

Step 4: Install dependencies
  Core (always): requirements.txt + the package
  Platform PyTorch: CUDA wheels (cu124) / MPS / CPU
  Dev: pytest, ruff, black
  CUDA extras: transformer-engine, pynvml, flash-attn, triton
  MPS extras (full profile, macOS): mlx, mlx-lm

Step 5: Post-install compatibility check (CUDA only)
  Auto-detects known-broken PyTorch versions (2.11+cu130 leads to SDPA crash on H200).
  Removes torchao/compressed-tensors if they conflict with the installed torch.
  Runs an SDPA smoke test on GPU.  See docs/H200_SETUP.md for details.

Step 6: Verify installation
  Runs `import torch`, `import triton` checks.
  Runs 75 unit tests.

Step 7: HuggingFace login
  Auto-detects HF_TOKEN env var or cached token.

> 💡 **H200 / SM90 — verified toolchain (June 2026)**
>
> **Use PyTorch 2.12.1+cu126.** This is the RECOMMENDED version after extensive testing.
> - **2.12.1+cu126** — ✅ SDPA works, +27% TPS over 2.6.0 (1,650 tok/s vs 1,300 tok/s on 4B).
>   `torch.compile(mode="default")` active (stable, no cudagraph_trees crash).
> - **2.6.0+cu124** — ⚠️ Works but slower. `torch.compile` auto-skipped (cudagraph_trees bug).
> - **2.11.0+cu130** — ❌ SDPA crashes.
>
> **FP8 / Transformer Engine:** TE source-build compiles but **runtime cuBLASLt crashes on all
> tested driver versions (580, 565)**. See [`FP8_TE_CUDA_ISSUES.md`](FP8_TE_CUDA_ISSUES.md).
> The practical default is `TR_SKIP_FP8=1` (pure BF16). Only known-working FP8 path is
> the NGC container (`nvcr.io/nvidia/pytorch:24.12-py3`), which has its own transformers
> version compatibility issues.
>
> **`setup.sh` installs the latest stable PyTorch.** To pin a specific version:
> ```bash
> pip install 'torch==2.12.1' --index-url https://download.pytorch.org/whl/cu126
> ```

### Profiles

| Profile | Flag | What Gets Installed |
|---------|------|-------------------|
| **full** (default) | `./setup.sh` | Everything: PyTorch, CUDA/MPS deps, Triton, dev tools |
| **quick** | `./setup.sh --quick` | PyTorch, core deps, dev tools. Skip model DL. |
| **minimal** | `./setup.sh --minimal` | Python + PyTorch + shared deps only. No tests, no extras. |

### Transformer Engine FP8 (CUDA only) — CURRENTLY BROKEN ON PIP VENVS

**TLDR:** TE source-build compiles successfully but **runtime cuBLASLt crashes on all
tested driver versions (580, 565)**. See [`FP8_TE_CUDA_ISSUES.md`](FP8_TE_CUDA_ISSUES.md)
for the full investigation (18 errors, 5 driver combos, 4 TE versions).

The only known-working FP8 path is the **NGC container**:
```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 \
  -v ~/LargeCorporaTranslation:/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -w /workspace -e PYTHONPATH=/workspace \
  nvcr.io/nvidia/pytorch:24.12-py3 \
  python3 -m benchmark --model translategemma-4b-bf16 --dry-run --batch-size 32
```

For pip venvs, the practical default is `TR_SKIP_FP8=1` (pure BF16 at 1,650 tok/s on 2.12.1).

---

### Pre-tokenization — tokenize once, skip CPU work forever

Pre-tokenization runs the chunk→filter→prompt→tokenize pipeline once and caches
the result as model-specific Parquet files. On subsequent runs, the pipeline reads
token IDs directly from cache, eliminating the CPU bottleneck (+60% TPS).

```bash
# Pre-process for a model (run once)
python -m benchmark --pretokenize --model translategemma-4b-bf16

# Pre-process all registered models
python -m benchmark --pretokenize-all

# Subsequent runs auto-detect the cache — zero config
./run.sh --config config.yaml

# Force fresh tokenization
TR_NO_PRETOKENIZED_CACHE=1 ./run.sh
```

Cache location: `~/.cache/tr_benchmark/pretokenized/<key>.parquet`
Cache key: `SHA256(model_path + tokenizer_hash + max_input_tokens + input_files)[:16]`
Auto-invalidation on model update, tokenizer change, or input data change.

**Measured throughput impact:** TranslateGemma 4B: 816 → 1,300 tok/s (+59% on 2.6.0).

---

## 3. Benchmark Reference Results (GPU 1, June 2026)

| Model | TPS (bs=max) | Pre-tok | Compile | Notes |
|---|---|---|---|---|
| **NLLB-200-600M** | **~11,000** (bs=64) | ✅ | ✅ (generate path) | Encoder-decoder, highest throughput |
| **TranslateGemma 4B** | **~1,650** (bs=32) | ✅ | ❌ (eager, PT 2.12.1) | Decoder-only, stable baseline |
| NLLB-200-1.3B | ~5,000–8,000 (est.) | ✅ | ✅ | Cache ready, not yet measured |
| NLLB-200-3.3B | ~2,000–4,000 (est.) | — | — | Not yet measured |

---

## 3. Run (`run.sh`)

### Commands

```bash
# ── Modes ──
./run.sh                         # Full benchmark (auto-detect platform)
./run.sh --quick                 # 5-minute evaluation
./run.sh --dry-run               # 60-second smoke test
./run.sh --pretokenize          # Pre-tokenize data, then exit
./run.sh --warmup-only           # Load + warmup, then exit
./run.sh --benchmark-only        # Quality evaluation only
./run.sh --translate-only        # Translation only (skip quality)

# ── Backends ──
./run.sh --nllb                  # NLLB-200 encoder-decoder (EN→TR, 600M–54B)
./run.sh --nllb --model ministral-3b-bf16  # NLLB with custom model
./run.sh --diffusion             # Diffusion model (LLaDA 8B)
./run.sh --speculative           # Self-speculative decoding (zero extra VRAM)

# ── Model ──
./run.sh --model 4B              # TranslateGemma 4B (preset)
./run.sh --model 12B             # TranslateGemma 12B (default)
./run.sh --model 27B             # TranslateGemma 27B
./run.sh --model ministral-3b-bf16         # Ministral 3B BF16 preset
./run.sh --model translategemma-4b-int8    # TranslateGemma 4B INT8 preset
./run.sh --model gemma4-e2b-qat-ct         # Gemma4 2B QAT preset
./run.sh --model /path/to/model   # Custom local path

# ── New in v3.6 ──
./run.sh --quantization int8     # 8-bit quantization (bf16|fp16|int8|int4)
./run.sh --paged-attention       # PagedAttention KV-cache (CUDA only)
./run.sh --continuous-batching   # Continuous batching (experimental, CUDA only)
./run.sh --nllb-src-lang eng_Latn  # NLLB source language code
./run.sh --nllb-tgt-lang tur_Latn  # NLLB target language code

# ── New in v3.8 (Advanced Optimizations) ──
./run.sh --use-flash-attention   # Enable Hopper-optimized FlashAttention-3 (Method 5)
./run.sh --vllm                  # Run using vLLM engine backend (⚠️ Disabled on CUDA 12.x hosts due to CUDA 13 wheels)

# ── Options ──
./run.sh --duration 3600         # Run for 1 hour
./run.sh --batch-size 128        # Force batch size
./run.sh --observability         # Enable Prometheus dashboard on :9090
./run.sh --resume output/dir/    # Resume from checkpoint
./run.sh --data "*.jsonl.gz"     # Custom input glob
./run.sh --output /path/to/out   # Output directory
./run.sh --refs golden.jsonl     # Custom reference set
./run.sh --seed 123              # Random seed
./run.sh --cost 2.50             # GPU cost per hour
./run.sh --tokens 15000000000000 # Token count for extrapolation
./run.sh --no-compile            # Disable torch.compile
./run.sh --safe-mode             # Disable experimental optimizations
./run.sh --mps-safe              # Skip batch tuning on Apple Silicon

# ── Combined ──
./run.sh --nllb --quick          # Fast NLLB evaluation
./run.sh --diffusion --observability  # Diffusion with live dashboard
./run.sh --speculative --quick   # Test speculative decoding
./run.sh --model gemma4-e4b-qat-ct --quantization int4  # QAT 4B, 4-bit
./run.sh --model 4B --duration 600    # 10 min, 4B model
```

### Platform Auto-Detection

The script detects your platform and sets sensible defaults:

| Platform | Default Model | Duration | Backend |
|----------|--------------|----------|---------|
| macOS MPS | TranslateGemma 4B (BF16) | 3600s | AR |
| Linux CUDA | TranslateGemma 12B (FP8) | 7200s | AR |
| CPU | TranslateGemma 4B (FP32) | 300s | AR |

All defaults can be overridden via flags.

### What It Does

```
1. Detect platform → set defaults
2. Apply user overrides (--model, --duration, --speculative, etc.)
3. Write runtime config YAML to output/ dir
4. Print pre-flight summary (GPU, model, backend)
5. Activate venv (if exists)
6. Launch observability server (if --observability)
7. Run: python -m benchmark --config <auto-generated.yaml>
8. Print summary: throughput, days estimate, quality scores
```

---

## 4. Make Targets

```bash
make setup            # Full environment (same as ./setup.sh)
make setup-quick      # Dev setup

make run              # Full benchmark
make run-quick        # 5-minute eval
make run-dry          # Smoke test
make run-diffusion    # Diffusion model

make test             # ~75 unit tests in 24 files
make lint             # Ruff linter
make format           # Ruff + Black formatter

make pretokenize      # Pre-tokenize input data
make dashboard        # Launch Prometheus + Grafana stack

make docker-build     # Build Docker image
make docker-run       # Run in Docker

make clean            # Remove build artifacts
make clean-all        # Remove build artifacts + all caches + venv
```

---

## 5. Docker (Pre-Compiled Images)

### Build

```bash
# Standard — optimized image with all dependencies.
docker build -t tr-benchmark:3.9 .

docker build -t tr-benchmark:3.6-trt --build-arg WITH_TENSORRT=1 .
```

The multi-stage Dockerfile compiles everything at **build time** in the `builder` stage. The `runtime` stage copies only the compiled artifacts — resulting in a lean, zero-first-run-latency image.

### Run

```bash
# Single node.
docker run --rm \
  --gpus '"device=0,1"' \
  --ipc=host --ulimit memlock=-1 \
  -v $(pwd)/data:/data \
  tr-benchmark:3.6 --config /data/config.yaml

# Quick eval in Docker.
docker run --rm --gpus '"device=0,1"' --ipc=host --ulimit memlock=-1 \
  -v $(pwd)/data:/data tr-benchmark:3.6 --config /data/config.yaml --quick
```

### What's Pre-Compiled in the Image

| Artifact | Built At | Cached Where |
|----------|----------|-------------|
| JIT CUDA kernels | Docker build (⚠️ disabled; kernel sources nulled) | `/root/.cache/tr_benchmark/kernels/` (empty) |
| Python venv with all deps | Docker build | `/opt/venv/` |
| Package installed (editable) | Docker build | `/app/` |

This means **zero first-run compilation latency** when the container starts. The model forward pass is ready immediately.

---

## 6. Docker Compose Observability

```bash
# Launch the full monitoring stack.
make dashboard

# Or directly:
docker-compose -f .github/docker-compose.obs.yaml up -d
```

This starts three services:

| Service | Port | URL |
|---------|------|-----|
| Benchmark (with Prometheus metrics) | 9091 | `http://localhost:9091/metrics` |
| Prometheus (scrapes benchmark) | 9090 | `http://localhost:9090` |
| Grafana (8-panel dashboard) | 3000 | `http://localhost:3000` (admin/admin) |

The Grafana dashboard auto-loads with throughput, GPU utilization, temperature, latency, quality scores, and pipeline health panels — all refreshing every 2 seconds.

---

## 7. JIT Kernel Compilation — 🗑 REMOVED v3.7

The entire JIT compilation subsystem was permanently deleted in commit `926855e`:
`benchmark/hardware/jit_compiler.py` (678 lines). Both CUDA kernel sources were
set to `None` (architecturally broken), the Metal RMSNorm wrapper returned inputs
unchanged, and the sm90a detection was buggy. `torch.compile` with inductor handles
kernel fusion internally — the manual JIT path was redundant.
> PyTorch 2.6.0+cu124 (2026-06-24), the Triton fused kernels (RMSNorm+residual,
> SwiGLU gate×up) can potentially be re-enabled. The `if False:` guard at
> `autoregressive.py:761` that disabled them (after commit `804c0a6`) may no
> longer be necessary. These kernels only work inside `torch.compile` and crash
> in eager mode — with `torch.compile` verified operational, the crash risk is
> eliminated. See M2.1 for the `torch.compile` status.

---

## 8. TensorRT — 🗑 REMOVED v3.7

The TensorRT backend and builder were permanently deleted:
`benchmark/inference/backends/tensorrt_backend.py` (459 lines) and
`benchmark/hardware/trt_builder.py` (727 lines). The TRT decode loop had
no KV-cache passthrough (each step was isolated → corrupted output after
token 1), was safety-gated to raise `RuntimeError` by default, and broke
on TRT 10+ (removed `EXPLICIT_BATCH` and `ICudaEngine.num_layers`).
The `--tensorrt` flag references were removed from all docs and configs.


## 9. Model Selection Guide

### Autoregressive Models

| Model | Memory (BF16) | Throughput (measured 2026-06-24) | Best For |
|-------|--------------|----------------------------------|----------|
| TranslateGemma 4B | ~8 GB | **735 tok/s** (bs=16, BF16, 1× H200). 13,223 tok/s at bs=512 | MPS dev, fast iteration |
| TranslateGemma 12B | ~48 GB | not yet measured | Production quality |
| TranslateGemma 27B | ~100 GB (2× H200) | not yet measured | Maximum quality |
| Ministral 3B | ~12 GB | not yet measured | Low VRAM, fast benchmarks |
| Gemma4 E2B QAT | ~8 GB | not yet measured | Mobile-optimized, QAT-tuned |
| Gemma4 E4B QAT | ~16 GB | not yet measured | Mid-tier QAT-tuned |
| DiffusionGemma 26B A4B | ~52 GB | not yet measured | Diffusion AR hybrid |

```bash
./run.sh --model 4B                    # TranslateGemma 4B (preset)
./run.sh --model 12B                   # TranslateGemma 12B
./run.sh --model 27B                   # TranslateGemma 27B
./run.sh --model ministral-3b-bf16     # Ministral 3B
./run.sh --model gemma4-e2b-qat-ct     # Gemma4 2B QAT BF16
./run.sh --model gemma4-e2b-q4_0       # Gemma4 2B QAT INT4 (q4_0)
./run.sh --model gemma4-e4b-qat-int4   # Gemma4 4B QAT INT4
./run.sh --model gemma4-e4b-q4_0       # Gemma4 4B QAT INT4 (q4_0)
```

### Encoder-Decoder Models (NLLB-200)

| Model | Memory | Size | Throughput (measured 2026-06-24, bs=8, BF16, Flash SDPA) | Best For |
|-------|--------|------|---------------------------------------------------------|----------|
| nllb-200-distilled-600M | ~1.2 GB | 2.4 GB | **580.5 tok/s** | Fastest, lowest quality |
| nllb-200-distilled-1.3B | ~2.5 GB | 5 GB | **215.4 tok/s** | Good speed/quality balance |
| nllb-200-3.3B | ~6.3 GB | 13 GB | **372.5 tok/s** | Production quality |
| nllb-200-54B (MoE) | ~100 GB | 200 GB | not yet measured | Maximum quality, high VRAM |

```bash
./run.sh --nllb                          # 600M distilled (default)
./run.sh --nllb --model ministral-3b-bf16  # custom model with NLLB backend
```

### Diffusion Models

| Model | Steps | Throughput vs AR | Quality |
|-------|-------|-----------------|---------|
| LLaDA 8B | 64-256 | 1-3× | Competitive |
| E2D2 | 64-128 | 2× | SOTA for MT |
| BD3-LM | 32-128 | 1.3× | Near-AR |

```bash
./run.sh --diffusion --quick
```



### Quality vs Speed Trade-off

```
Quality ↑   More steps / larger model / beam search / more bits
Speed   ↑   Fewer steps / smaller model / greedy decode / BF16 (not quantization!)

Speed factors below are MEASURED (2026-06-24) on TranslateGemma 4B, single H200, unless noted:

torch.compile:       <1.05× on 4B (negligible at <12B). Expected 1.1-1.4× on 12B+ — M2.1
Flash SDPA:          1.17-1.23× overall throughput — M2.4
INT8 quantization:   0.27× throughput on H200 (3.7× SLOWER, despite 41% memory savings) — M2.7
TE FP8:              0.60× throughput on H200 (40% SLOWER, 0% memory saved for 4B) — M1.5
PagedAttention:      60-87.5% KV memory savings for variable-length workloads — M2.6

⚠️ INT8 and TE FP8 are COUNTERPRODUCTIVE for 4B models on H200 (130+ GB VRAM free).
Only use when VRAM-constrained or for 12B+ models where compute-bound.

Diffusion T=256:    ~1× AR speed, competitive quality
Diffusion T=64:     ~4× AR speed, ~2-3 BLEU drop from full quality
Diffusion T=32:     ~8× AR speed, ~4-5 BLEU drop from full quality

⚠️ The rule: after any precision change, run ./run.sh --benchmark-only
   to verify quality against your golden reference set before deploying.
   FP8 on H200 with 4B is 40% slower — not a free upgrade at this model size.
```

---

## 10. Performance Tuning

### Batch Size

```bash
./run.sh --batch-size 128     # Force specific size
./run.sh                       # Auto-tune (default)
```

### Precision

| dtype | When to use |
|-------|------------|
| `float8_e4m3fn` (FP8) | H200 production — fastest, minimal quality loss |
| `bfloat16` (BF16) | MPS dev, standard CUDA — baseline |
| `float16` (FP16) | Universal, good on all platforms |
| `float32` (FP32) | Reference only, CPU, debugging |

Set in `config.yaml`: `model.dtype`.

### Profiling

```bash
# Built-in lightweight profiler.
./run.sh --no-compile  # avoid graph overhead during profiling

# External Nsight (full detail).
nsys profile --trace=cuda,nvtx,osrt,cublas,cudnn \
  --cuda-memory-usage=true --output=profile \
  python -m benchmark --config config.yaml
```

### Measured Throughput Baselines (H200, June 2026)

All measurements on 2× NVIDIA H200 NVL (139.80 GB each), torch 2.6.0+cu124,
PyTorch built-in SDPA, BF16. Full data available in the benchmark reference results section above.

| Model | Backend | Batch | tok/s | Notes |
|---|---|---|---|---|
| TranslateGemma 4B | AR (Flash SDPA) | 16 | **735** | 8.0 GB VRAM, 42°C |
| TranslateGemma 4B | AR (Flash SDPA) | 512 | **13,223** | Optimal batch, no OOM |
| TranslateGemma 4B | AR (eager) | 16 | 630 | 1.17× slower than SDPA |
| TranslateGemma 4B | AR (torch.compile) | 16 | 728 | <1% gain over SDPA alone |
| TranslateGemma 4B | AR (TE FP8) | 16 | 497 | −40% vs BF16 (cast overhead) |
| TranslateGemma 4B | AR (INT8) | 16 | 213 | −73% vs BF16 (dequant overhead) |
| NLLB-200 600M | Enc-Dec | 8 | **581** | 1.1 GB VRAM |
| NLLB-200 1.3B | Enc-Dec | 8 | 215 | 2.6 GB VRAM |
| NLLB-200 3.3B | Enc-Dec | 8 | 373 | 6.3 GB VRAM |

**Throughput degradation:** Zero detectable change after 2.2h sustained inference
(122K batches, 110M tokens, mean 899 tok/s, slope +0.1%/hr, R²=0.000046).
The constant-throughput assumption is validated for 4B on H200.

### Verified Toolchain (H200 SM90)

After extensive testing (see [`H200_SETUP.md` §Known Issues](H200_SETUP.md)):

| Component | Verified | Notes |
|---|---|---|
| **torch** | **2.6.0+cu124** | Pinned in `setup.sh`. 2.11+cu130 crashes on SM90. 2.12+cu126 has flash_attn ABI break. |
| **SDPA** | PyTorch built-in | `F.scaled_dot_product_attention`. All backends (flash/mem_efficient/math) work. |
| **flash_attn** | NOT needed | PyTorch's built-in SDPA covers everything. flash_attn 2.8.3 only works with torch 2.11. |
| **TE FP8** | Source-build | `pip install 'transformer-engine[pytorch]' --no-build-isolation` with `CPATH` set. −40% TPS for 4B. |
| **torch.compile** | ✅ Works | `mode="reduce-overhead"`. Minimal gain for 4B; more benefit at 12B+. |
| **bitsandbytes** | ⚠️ Functional | INT8 works but −73% TPS vs BF16 (dequant overhead). Only useful when VRAM-constrained. |



## 11. Multi-Node Cluster

### Prerequisites
- K8s cluster with NVIDIA GPU operator
- Nodes with ≥2 H200, NVLink intra-node, InfiniBand/RoCE inter-node

### NCCL Environment

```bash
export NCCL_NET_GDR_LEVEL=5      # GPU Direct RDMA
export NCCL_IB_DISABLE=0         # InfiniBand
export NCCL_SOCKET_IFNAME=eth0
export NCCL_P2P_LEVEL=NVL        # NVLink intra-node
export NCCL_ALGO=Ring            # Optimal for TP
```

### Docker Launch

```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 \
  --network=host \
  -e NCCL_SOCKET_IFNAME=eth0 \
  -e NCCL_NET_GDR_LEVEL=5 \
  -e MASTER_ADDR=$MASTER_IP \
  -e MASTER_PORT=29500 \
  tr-benchmark:3.6 --config /data/config.yaml
```

---

*This guide is part of the documentation set. See [`docs/README.md`](README.md)
for navigation, [`ARCHITECTURE.md`](ARCHITECTURE.md) for the reality-grounded
architecture and Feature Status, and
[`AI_CODING_ANTIPATTERNS.md`](AI_CODING_ANTIPATTERNS.md) for mistakes to avoid.*
