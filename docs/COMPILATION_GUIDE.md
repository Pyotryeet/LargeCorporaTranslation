# Compilation & Deployment Guide — v3.6

**One command to install, one command to run. Every platform. Every backend.**

---

## Table of Contents

1. [The 30-Second Quickstart](#1-the-30-second-quickstart)
2. [Setup (setup.sh)](#2-setup-setupsh)
3. [Run (run.sh)](#3-run-runsh)
4. [Make Targets](#4-make-targets)
5. [Docker (Pre-Compiled Images)](#5-docker-pre-compiled-images)
6. [Docker Compose Observability](#6-docker-compose-observability)
7. [JIT Kernel Compilation](#7-jit-kernel-compilation)
8. [TensorRT Engine Build](#8-tensorrt-engine-build)
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
./setup.sh --quick            # Dev setup: skip TRT, skip model download
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
  TensorRT (full profile, CUDA only): tensorrt, onnx, onnxruntime
  MPS extras (full profile, macOS): mlx, mlx-lm

Step 5: Verify installation
  Runs `import torch`, `import triton`, `import tensorrt` checks.
  Runs 75 unit tests.

Step 6: HuggingFace login
  Auto-detects HF_TOKEN env var or cached token.

Step 7: Pre-compile JIT kernels (CUDA only)
  Compiles fused_qkv_rope + fused_swiglu_mlp to ~/.cache/tr_benchmark/kernels/
```

### Profiles

| Profile | Flag | What Gets Installed |
|---------|------|-------------------|
| **full** (default) | `./setup.sh` | Everything: PyTorch, CUDA/MPS deps, TRT, Triton, dev tools |
| **quick** | `./setup.sh --quick` | PyTorch, core deps, dev tools. Skip TRT, skip model DL. |
| **minimal** | `./setup.sh --minimal` | Python + PyTorch + shared deps only. No tests, no extras. |

---

## 3. Run (`run.sh`)

### Commands

```bash
# ── Modes ──
./run.sh                         # Full benchmark (auto-detect platform)
./run.sh --quick                 # 5-minute evaluation
./run.sh --dry-run               # 60-second smoke test
./run.sh --precompile            # Pre-compile JIT + TRT, then exit
./run.sh --warmup-only           # Load + warmup, then exit
./run.sh --benchmark-only        # Quality evaluation only
./run.sh --translate-only        # Translation only (skip quality)

# ── Backends ──
./run.sh --nllb                  # NLLB-200 encoder-decoder (EN→TR, 600M–54B)
./run.sh --nllb --model ministral-3b-bf16  # NLLB with custom model
./run.sh --diffusion             # Diffusion model (LLaDA 8B)
./run.sh --tensorrt              # TensorRT-accelerated (CUDA only, 20-50% more)
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

# ── Options ──
./run.sh --duration 3600         # Run for 1 hour
./run.sh --batch-size 128        # Force batch size
./run.sh --observability         # Enable Prometheus dashboard on :9090
./run.sh --force-recompile       # Force JIT + TRT recompilation
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
./run.sh --tensorrt --quick      # Fast TRT evaluation
./run.sh --diffusion --observability  # Diffusion with live dashboard
./run.sh --speculative --quick   # Test speculative decoding
./run.sh --model gemma4-e4b-qat-ct --quantization int4  # QAT 4B, 4-bit
./run.sh --model 4B --duration 600    # 10 min, 4B model
```

### Platform Auto-Detection

The script detects your platform and sets sensible defaults:

| Platform | Default Model | Duration | Backend |
|----------|--------------|----------|---------|
| macOS MPS | TranslateGemma 4B (BF16) | 3600s | AR (TRT not available) |
| Linux CUDA | TranslateGemma 12B (FP8) | 7200s | AR (TRT optional) |
| CPU | TranslateGemma 4B (FP32) | 300s | AR |

All defaults can be overridden via flags.

### What It Does

```
1. Detect platform → set defaults
2. Apply user overrides (--model, --duration, --tensorrt, etc.)
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
make run-tensorrt     # TensorRT
make run-diffusion    # Diffusion model

make test             # ~75 unit tests in 24 files
make lint             # Ruff linter
make format           # Ruff + Black formatter

make precompile       # Pre-compile JIT + TRT
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
# Standard — JIT kernels pre-compiled inside the image.
docker build -t tr-benchmark:3.6 .

# With TensorRT — TRT engines also pre-built.
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
| JIT CUDA kernels (QKV+RoPE, SwiGLU) | Docker build (builder stage) | `/root/.cache/tr_benchmark/kernels/` |
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

## 7. JIT Kernel Compilation

### Automated Path

```bash
# Pre-compile all kernels (runs once, cached forever).
./run.sh --precompile

# Force recompilation.
./run.sh --force-recompile --precompile
```

### Manual Path

```bash
# Check what's compiled.
python -c "from benchmark.hardware.jit_compiler import get_jit_compiler; print(get_jit_compiler().cache_stats())"

# Clear all compiled kernels.
rm -rf ~/.cache/tr_benchmark/kernels/

# Rebuild.
TR_BENCHMARK_FORCE_RECOMPILE=1 python -m benchmark --config config.yaml --warmup-only
```

### What Gets Compiled

| Kernel | Compiler | Speedup vs Eager |
|--------|----------|-----------------|
| Fused QKV+RoPE | `nvcc` (CUDA C++) | 2.5× |
| Fused SwiGLU MLP | `nvcc` (CUDA C++) | 2.0× |
| Fused RMSNorm+Residual | `xcrun metal` (MSL) | 3.0× on MPS |

If `nvcc` or `xcrun metal` are not installed, compilation is silently skipped and the kernels fall back to Triton (CUDA) or PyTorch eager (MPS/CPU).

---

## 8. TensorRT Engine Build

### Quick Start

```bash
# Enable TensorRT — engine auto-built on first run, cached forever.
./run.sh --tensorrt --quick

# First run output:
#   [TRT] cache MISS — building engine (precision=fp16) ...
#   [TRT] ONNX exported: 2478 nodes
#   [TRT] engine built in 485.2s — abc123.engine (384.5 MB)
#
# Second run output:
#   [TRT] cache HIT — abc123.engine (384.5 MB)
```

### Precision Modes

```bash
# FP16 — fast, universal, no calibration needed. Quality identical to BF16.
./run.sh --tensorrt  # defaults to fp16

# FP8 — Hopper only (H200). Same quality as FP16 (dynamic scaling).
TRT_PRECISION=fp8 ./run.sh --tensorrt

# INT8 — 2× smaller model, faster, but MUST validate quality after building.
#       Requires 100-500 representative EN sentences for calibration.
#       Expect < 0.3 BLEU impact with good calibration data.
./run.sh --tensorrt --data calibration_en.jsonl
# After building: ALWAYS run quality check before deploying.
./run.sh --benchmark-only
# If BLEU/chrF++/COMET meet targets → safe to use. If not → fall back to FP16.```
```

### Manual Build

```bash
python -c "
from benchmark.hardware.trt_builder import build_engine_if_needed
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    'google/translategemma-12b-it',
    torch_dtype=torch.bfloat16, device_map='auto',
)
tok = AutoTokenizer.from_pretrained('google/translategemma-12b-it')

engine_path = build_engine_if_needed(
    'google/translategemma-12b-it', model, tok,
    max_batch=32, max_input=512, max_output=512,
    precision='fp16',
)
print(f'Engine: {engine_path}')
"
```

### Cache

```bash
# View TRT engine cache.
ls -lh ~/.cache/tr_benchmark/engines/

# Clear.
rm -rf ~/.cache/tr_benchmark/engines/

# Force rebuild.
TR_BENCHMARK_FORCE_RECOMPILE=1 ./run.sh --tensorrt
```

---

## 9. Model Selection Guide

### Autoregressive Models

| Model | Memory (BF16) | Best For |
|-------|--------------|----------|
| TranslateGemma 4B | ~20 GB | MPS dev, fast iteration |
| TranslateGemma 12B | ~50 GB | Production quality |
| TranslateGemma 27B | ~100 GB (2× H200) | Maximum quality |
| Ministral 3B | ~12 GB | Low VRAM, fast benchmarks |
| Gemma4 E2B QAT | ~8 GB | Mobile-optimized, QAT-tuned |
| Gemma4 E4B QAT | ~16 GB | Mid-tier QAT-tuned |
| DiffusionGemma 26B A4B | ~52 GB | Diffusion AR hybrid |

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

| Model | Memory | Size | Best For |
|-------|--------|------|----------|
| nllb-200-distilled-600M | ~1.2 GB | 2.4 GB | Fastest, lowest quality |
| nllb-200-distilled-1.3B | ~2.5 GB | 5 GB | Good speed/quality balance |
| nllb-200-3.3B | ~6.3 GB | 13 GB | Production quality |
| nllb-200-54B (MoE) | ~100 GB | 200 GB | Maximum quality, high VRAM |

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

### TensorRT (CUDA Only)

Adds 20-50% on top of AR throughput. Works with any AR model.

```bash
./run.sh --tensorrt
```

### Quality vs Speed Trade-off

```
Quality ↑   More steps / larger model / beam search / more bits
Speed   ↑   Fewer steps / smaller model / greedy decode / TensorRT / quantization

AR + TensorRT FP16: ~1.4× AR speed, same quality (no calibration needed)
AR + TensorRT FP8:  ~1.6× AR speed, same quality (H200 only, dynamic scaling)
AR + TensorRT INT8: ~2.0× AR speed, < 0.3 BLEU drop with good calibration
AR + INT4 weights:  ~2.5× AR speed, 0.5-2.0 BLEU drop — validate before deploying

Diffusion T=256:    ~1× AR speed, competitive quality
Diffusion T=64:     ~4× AR speed, ~2-3 BLEU drop from full quality
Diffusion T=32:     ~8× AR speed, ~4-5 BLEU drop from full quality

⚠️ The rule: after any precision change, run ./run.sh --benchmark-only
   to verify quality against your golden reference set before deploying.
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

### Regression Monitoring

```python
from benchmark.observability.perf_regression import PerformanceBaselineManager

mgr = PerformanceBaselineManager("./baselines")
mgr.save_baseline("h200_fp8_12b", report["metrics"])
result = mgr.check("h200_fp8_12b", current_metrics)
if result.is_regression:
    print(f"REGRESSION: {result.reason}")
```

---

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
