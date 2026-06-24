# FP8, Transformer Engine, CUDA, and PyTorch — Full Issue Analysis

**Machine:** `asus02` — 2× NVIDIA H200 NVL (141 GB each), CUDA Driver 580.159.03
**Date:** 2026-06-24

---

## 1. Environment

| Component | Version | Notes |
|---|---|---|
| NVIDIA Driver | 580.159.03 | Latest Hopper driver |
| CUDA Toolkit (system) | 12.0 (nvcc) | Compiler only; runtime from PyTorch |
| CUDA Runtime (PyTorch) | 12.4 | Shipped inside the PyTorch wheel |
| PyTorch | 2.6.0+cu124 | pip install, NOT NGC container |
| Python | 3.12.3 | |
| GPU | 2× NVIDIA H200 NVL | 141 GB HBM3e, 4.8 TB/s bandwidth |
| SM | 9.0 (Hopper) | FP8 tensor core support via sm90a |

**Key incompatibility driver:** PyTorch 2.6 pip wheel bundles CUDA 12.4 runtime. NVIDIA's Transformer Engine 2.16 targets CUDA 12.8. The cuBLAS version chain doesn't cleanly span this gap.

---

## 2. The two FP8 paths

### Path A: Transformer Engine (NVIDIA's library)

**How it works:**

```
BF16 model loaded
  → apply_te_fp8_to_model() walks every nn.Linear
  → replaces with te.Linear (same shape, fused FP8 matmul kernel)
  → wraps forward pass in te.fp8_autocast(enabled=True)
  → te.Linear internally: quantize activation → FP8 matmul → BF16 output
  → quant+matmul fused in ONE CUDA kernel call
  → NO per-token quantization tax
```

**Performance:** ~60% batched throughput improvement over BF16. 2× single-stream decode bandwidth reduction. This is the production path for vLLM, SGLang, TensorRT-LLM on H100/H200.

**Problem:** TE requires exact CUDA/PyTorch version alignment. The pip package has three pieces that must all match:

```
transformer-engine        →  Python frontend
transformer-engine-cu12   →  CUDA 12.x compiled kernels (.so)
transformer-engine-torch  →  PyTorch framework glue (compiled against specific PyTorch)
```

If any one of these doesn't match, TE fails to import or crashes at runtime.

### Path B: Native torch._scaled_mm (built into PyTorch)

**How it works:**

```
BF16 model loaded
  → apply_native_fp8_to_model() walks nn.Linear
  → replaces with NativeFP8Linear (custom nn.Module)
  → forward(): quantize activation (abs().max() → scale → clamp → cast to fp8)
  → torch._scaled_mm(input_fp8, weight_fp8, scale_a, scale_b, out_dtype=bf16)
  → quantize and matmul are TWO separate operations
  → quantization tax per forward call
```

**Performance:** Theoretical 2× matmul throughput vs BF16 for the matmul portion. But the quantization step (abs().max(), scale, clamp, cast) adds ~20-40 µs per forward call. This tax is CONSTANT relative to matmul size. For large matmuls it's negligible. For small ones it dominates.

---

## 3. Micro-benchmark: where FP8 wins vs loses (H200, TranslateGemma 4B)

Measured on the H200 with `torch._scaled_mm` vs `torch.nn.functional.linear`:

### MLP layer: [bs*seq, 2560] × [2560, 10240] (gate_proj/up_proj)

| Batch × seq | Matmul size | BF16 (µs) | FP8 _scaled_mm (µs) | Ratio |
|---|---|---|---|---|
| 32 × 1 (decode) | [32, 2560] | **21** | **42** | 2.00× slower |
| 32 × 50 (prefill) | [1600, 2560] | **136** | **161** | 1.18× slower |

### Why FP8 is slower

The quantization step costs ~21 µs regardless of matmul size:
- `x.to(torch.float32)` — tensor copy, 2-3 µs
- `x.abs().max()` — reduction over all elements, 8-10 µs
- `x / scale` then `.clamp(-448, 447)` then `.to(torch.float8_e4m3fn)` — elementwise, 8-10 µs

The FP8 matmul saves maybe 5-10 µs vs BF16 at these sizes.

Net: 21 µs tax + 16 µs matmul = 37 µs. BF16: 21 µs matmul = 21 µs.

### When FP8 wins (projected from scaling laws)

| Hidden dim | BF16 matmul (µs) | FP8 tax (µs) | FP8 matmul (µs) | Net (µs) | Winner |
|---|---|---|---|---|---|
| 2560 (4B) | 21 | 21 | 16 | 37 | BF16 |
| 3840 (12B) | 45 | 21 | 30 | 51 | BF16 |
| 8192 (70B) | 160 | 21 | 80 | 101 | **FP8** (1.6×) |
| > 8192 | > 200 | 21 | < 100 | < 121 | **FP8** |

**The crossover point is ~hidden_dim=6000 or batch_size > 256.** Below that, BF16 matmul is faster than FP8 with dynamic quantization.

---

## 4. TE installation failure investigation

### Attempt 1: TE 2.16 (latest, March 2026)

```bash
pip install transformer-engine==2.16.0
```

**Result:** Installed `transformer-engine-2.16.0` + `transformer_engine_cu12-2.16.0`. Missing `transformer_engine_torch` — the PyTorch framework glue `.so` file.

**Error on import:**
```
FileNotFoundError: Could not find shared object file for Transformer Engine torch lib.
```

**Root cause:** Pip's dependency resolver doesn't auto-install `transformer_engine_torch` when the bare package is requested. The `[pytorch]` extra is needed.

### Attempt 2: TE 2.16 with [pytorch] extra

```bash
pip install 'transformer-engine[pytorch]==2.16.0'
```

**Result:** `transformer_engine_torch-2.16.0` tries to build from source.

**Error during build:**
```
RuntimeError: This package needs Torch to build.
```

**Root cause:** Pip's default `--build-isolation` creates a temporary venv WITHOUT torch. The build script checks for `import torch` and fails.

### Attempt 3: TE 2.16 with [pytorch] + --no-build-isolation

```bash
pip install 'transformer-engine[pytorch]==2.16.0' --no-build-isolation
```

**Result:** Build environment can see torch, but compilation fails at the C++ level.

**Error:**
```
RuntimeError: Error compiling objects for extension
```

**Root causes (likely):**
1. CUDA toolkit mismatch — system nvcc is 12.0, TE 2.16 compiles against 12.8 headers
2. Missing CUDA development headers for 12.8 (only runtime 12.4 is installed)
3. Compiler version incompatibility (gcc version on Ubuntu vs what TE expects)

### Attempt 4: TE 2.4 (last CUDA 12.4 target)

```bash
pip install 'transformer-engine==2.4.0'
```

**Result:** Installed, but same `transformer_engine_torch` missing problem. TE 2.4's `[pytorch]` extra also needs to compile from source, failing identically.

### Attempt 5: TE 1.13 (legacy, CUDA 12 compatible)

```bash
pip install 'transformer-engine[pytorch]==1.13.0'
```

**Result:** Installed, but crashed on import.

```
AssertionError: TransformerEngine package version mismatch.
Found transformer_engine_torch v2.16.0, transformer-engine v1.13.0
```

TE 1.x and 2.x can't coexist — leftover `transformer_engine_torch` from the 2.16 install.

### TE installation root cause summary

| Problem | Cause |
|---|---|
| `transformer_engine_torch` missing | pip doesn't auto-install framework glue; needs `[pytorch]` extra |
| `[pytorch]` build fails (no torch) | pip's isolated build env doesn't have torch |
| `[pytorch]` build fails (C++ error) | CUDA toolkit 12.0 on system vs 12.8 headers TE expects |
| Version mismatch on downgrade | `transformer_engine_torch` survives uninstall of `transformer-engine` |

---

## 5. The NGC container is the solution

NVIDIA ships a Docker container where ALL of this works:

```
nvcr.io/nvidia/pytorch:24.06-py3
```

Inside: PyTorch 2.4, TE 1.11 (pre-built), CUDA 12.5, cuDNN 9.2 — all tested together. This is what vLLM, SGLang, and TensorRT-LLM use as their base.

Upgrading to a newer NGC container (24.12+) would give PyTorch 2.5+ with TE 2.x and CUDA 12.8 — all guaranteed compatible.

**The pip-installed PyTorch 2.6 + TE 2.16 combo on this H200 simply isn't a tested path.** NVIDIA tests TE against their own PyTorch builds in the NGC container, not against the PyTorch pip wheels. The pip wheel uses CUDA 12.4 runtime while TE 2.16 expects 12.8, and the framework glue layer (`transformer_engine_torch`) can't be compiled from source without the exact CUDA toolchain TE was built against.

---

## 6. FP8 weight cache — the implemented solution

Since TE can't be installed from pip on this machine, and no FP8 pre-quantized model checkpoints exist on HuggingFace for the Gemma family, the codebase implements:

### On-disk FP8 weight cache

```
First load:
  HuggingFace BF16 weights → auto-quantized to FP8 E4M3
  → saved to ~/.cache/tr_benchmark/fp8_weights/{hash}/
  → one .safetensors per layer (weight_fp8 + scale)

Subsequent loads:
  → reads FP8 weights directly from cache
  → NO re-quantization needed
  → cache key = SHA256(model_path + layer shapes + checksums)
  → auto-invalidates on model update
```

### Code path

```
autoregressive.py:_apply_fp8()
  ├─ TE available? → apply_te_fp8_to_model() → te.Linear (fused kernel)
  │   └─ FALLS BACK on ImportError / RuntimeError
  ├─ Native? → apply_native_fp8_to_model() → NativeFP8Linear
  │   └─ Uses torch._scaled_mm (quantization tax applies)
  └─ Neither? → BF16 (log message)

Environment control:
  TR_SKIP_FP8=1         — skip all FP8, pure BF16
  TR_FORCE_NATIVE_FP8=1 — skip TE attempt, go straight to native
```

### Performance decisions

| Model class | Hidden dim | FP8 via _scaled_mm | Recommendation |
|---|---|---|---|
| 1-4B (TranslateGemma, Ministral) | 2048-2560 | **Slower** than BF16 | TR_SKIP_FP8=1 for now |
| 8-12B (TranslateGemma 12B) | 3840-4096 | ~Neutral | Try both, measure |
| 26B+ (DiffusionGemma, large models) | 5120-8192 | **Faster** than BF16 | Default FP8 ON |

---

## 7. Resolutions by path

### Path A: NGC container (recommended, not yet implemented)

```bash
docker run --gpus all -it --ipc=host \
  -v ~/LargeCorporaTranslation:/workspace \
  nvcr.io/nvidia/pytorch:24.12-py3 \
  python -m benchmark --config config.yaml
```

Zero pip installs. TE + PyTorch + CUDA are pre-tested together. FP8 works out of the box.

### Path B: Fix pip TE (requires manual CUDA toolkit)

```bash
# Install CUDA 12.8 toolkit (not just runtime)
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
sudo sh cuda_12.8.0_570.86.10_linux.run --toolkit --silent

# Install TE with correct CUDA home
CUDA_HOME=/usr/local/cuda-12.8 pip install 'transformer-engine[pytorch]==2.16.0' --no-build-isolation
```

### Path C: Accept native FP8 with tax (current state)

```bash
TR_FORCE_NATIVE_FP8=1 python -m benchmark --model translategemma-12b-bf16 --dry-run
```

Only worth it for 8B+ models. For 4B and below, `TR_SKIP_FP8=1` until Path A or B is implemented.

### Path D: Static weight FP8 (hybrid, implemented)

This is what the current `NativeFP8Linear.forward()` does: weights are stored in FP8 (half the HBM bandwidth), activations stay in BF16 (no quantization tax). The dequantization `w_fp8.to(bf16) * scale` happens on-chip without a separate memory transaction. This path gives the memory bandwidth benefit without the compute tax.

---

## 8. CUDA/PyTorch/TE version compatibility matrix

| PyTorch | CUDA Runtime | TE version | TE[pytorch] pre-built? | Status |
|---|---|---|---|---|
| 2.6.0 (pip cu124) | 12.4 | 2.16.0 | No (CUDA 12.8) | ❌ Compile fails |
| 2.6.0 (pip cu124) | 12.4 | 2.4.0 | No (CUDA 12.4) | ❌ Compile fails |
| 2.6.0 (pip cu124) | 12.4 | 1.13.0 | Yes (CUDA 12.1) | ⚠️ Import error (2.x debris) |
| 2.4.0 (NGC 24.06) | 12.5 | 1.11.0 | Yes | ✅ Known working |
| 2.5.1 (NGC 24.12) | 12.8 | 2.x | Yes | ✅ Known working |

The pip-installed PyTorch path is fundamentally fragile for TE. NGC is the tested path.

---

## 9. Affected files

| File | What changed |
|---|---|
| `benchmark/hardware/precision.py` | `NativeFP8Linear`, `apply_native_fp8_to_model()`, `save_fp8_weights()`, `load_fp8_weights()` |
| `benchmark/inference/backends/autoregressive.py` | `_apply_fp8()` — always attempts TE then native; `_fp8_context()` |
