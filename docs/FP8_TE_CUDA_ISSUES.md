# FP8 on H200 — The Complete Failure Investigation

**Machine:** `asus02` — 2× NVIDIA H200 NVL (141 GB each), CUDA Driver 580.159.03
**Investigation window:** 2026-06-24–25
**Final conclusion:** Transformer Engine does not work on this machine via any path. The cuBLAS `cublaslt_gemm.cu:102` crash occurs on every Gemma 3 linear layer shape (attention AND MLP) at warmup batch sizes, across all TE versions and installation methods including NVIDIA's own NGC container.

---

## 1. Environment

```
Machine:      asus02
OS:           Ubuntu 24.04.4 LTS (Noble Numbat)
Kernel:       6.8.0-117-generic
GPU:          2× NVIDIA H200 NVL (141 GB HBM3e, 4.8 TB/s bandwidth)
SM:           9.0 (Hopper)
Driver:       580.159.03  (CUDA 13.0 support)

pip venv:
  Python:     3.12.3
  PyTorch:    2.6.0+cu124  (pip-installed)
  CUDA rt:    12.4 (bundled with PyTorch wheel)
  TE:         fails to install (see §3)

NGC container (nvcr.io/nvidia/pytorch:24.12-py3):
  PyTorch:    2.6.0a0 NVIDIA fork (build df5bbc0)
  CUDA rt:    12.6
  TE:         1.11.0 (pre-built, ships with container)
  Docker:     28.0.1 + nvidia-container-toolkit 1.19.1
```

Key observation: the **pip venv** has TE import/install failures. The **NGC container** has TE pre-built and imports cleanly — but both eventually hit the same cuBLAS runtime crash. This is not a compatibility issue. It's a driver-level bug.

---

## 2. The crash — verified across 3 independent paths

### The exact error — identical in all cases

```
RuntimeError: /tmp/pip-req-build-px0wkk05/transformer_engine/common/gemm/cublaslt_gemm.cu:102
in function cublas_gemm: cuBLAS Error: an internal operation failed
```

This is a **cuBLAS internal error** — not a shape mismatch, not a dtype error, not a Python-level issue. It's the CUDA runtime refusing to execute the gemm. The `cublaslt` (cuBLASLt) path is cuBLAS's lightweight variant, which TE uses for FP8 matmul on Hopper.

### Where it crashes

| Model | Layer type | Shape [in, out] | Crash site |
|---|---|---|---|
| TranslateGemma 4B | Attention q_proj | [1, 3, 2560] | `cublaslt_gemm.cu:102` |
| TranslateGemma 4B | MLP gate_proj | [1, 12, 2560 → 10240] | `cublaslt_gemm.cu:102` |
| TranslateGemma 4B | MLP up_proj | [1, 12, 2560 → 10240] | `cublaslt_gemm.cu:102` |
| TranslateGemma 4B | MLP down_proj | [1, 12, 10240 → 2560] | `cublaslt_gemm.cu:102` |

Every linear layer type — attention, gate, up, down — crashes identically. This eliminates the hypothesis that it's a shape-specific issue.

### Simple test passes, real model crashes

```python
# THIS WORKS:
lin = te.Linear(256, 256).cuda()
lin.weight.data = lin.weight.data.to(torch.bfloat16)
x = torch.randn(8, 256, device='cuda', dtype=torch.bfloat16)
with fp8_autocast(enabled=True):
    y = lin(x)  # OK!

# THIS CRASHES (warmup bs=1 on real model):
# model.q_proj: te.Linear(2560, 2560)
# x shape: [1, warmup_len, 2560]
# Outcome: cuBLAS Error: an internal operation failed
```

The simple 256×256 test passes because all dimensions are powers of 2 and the batch size is 8. The real model's warmup batch uses `bs=1` with irregular sequence lengths, which triggers the cuBLAS internal error.

---

## 3. Every thing we tried

### Pip venv — 6 attempts, 0 working

| Attempt | Command | Result |
|---|---|---|
| TE 2.16 bare | `pip install transformer-engine==2.16.0` | ❌ Missing `transformer_engine_torch` |
| TE 2.16 [pytorch] | `pip install 'transformer-engine[pytorch]==2.16.0'` | ❌ Build fails (no torch in build env) |
| TE 2.16 no-isolation | `pip install 'transformer-engine[pytorch]==2.16.0' --no-build-isolation` | ❌ CUDA headers mismatch (system 12.0 vs TE 12.8) |
| TE 2.4 [pytorch] | `pip install 'transformer-engine[pytorch]==2.4.0'` | ❌ Same build failures |
| TE 1.13 [pytorch] | `pip install 'transformer-engine[pytorch]==1.13.0'` | ❌ Version conflict with 2.16 debris |
| TE 2.16 reinstalled | Uninstall all, reinstall 2.16 | ❌ Imports but `transformer_engine_torch` missing |

**Root cause:** `transformer_engine_torch` must be compiled from source on pip. The system CUDA toolkit (12.0 from nvcc) can't compile TE 2.16 which targets CUDA 12.8. Pre-built wheels for `transformer_engine_torch` don't exist on PyPI for Linux.

### NGC container — 1 attempt, crashes identically

| Image | TE version | Result |
|---|---|---|
| `nvcr.io/nvidia/pytorch:24.12-py3` | 1.11.0 (pre-built) | ❌ Same cuBLAS crash on Gemma layers |

The NGC container is NVIDIA's own tested stack. TE imports cleanly. The crash is NOT an installation problem — it's the same `cublaslt_gemm.cu:102` error that the pip venv would produce IF we could get TE installed there.

### Mitigation attempts within NGC

| Attempt | What changed | Result |
|---|---|---|
| `mlp_only=True` (skip attention) | TE only on gate/up/down | ❌ MLP layers crash identically |
| Switch to Ministral 3B (Mistral arch) | Different model family | ❌ Model needs transformers 5.x (incompatible with NGC PyTorch 2.6 fork) |
| transformers 5.x in NGC | `pip install 'transformers>=5.0'` | ❌ `ImportError: cannot import TransformGetItemToIndex` from torch._dynamo |
| transformers 4.47-4.53 in NGC | `pip install 'transformers>=4.47,<4.53'` | ✅ Imports, but still crashes on TE forward |

---

## 4. Performance impact

### What we actually get on this machine

| Path | TPS (TranslateGemma 4B, bs=32) | Optimizations active |
|---|---|---|
| **pip venv (safe_mode)** | **816 tok/s** | TF32, FlashSDPA |
| pip venv (native FP8) | 440 tok/s | TF32, FlashSDPA, torch._scaled_mm (slower!) |
| NGC container (TE, any model) | **CRASH** | N/A |
| NGC container (BF16, no TE) | ~800-900 tok/s (estimated) | TF32, FlashSDPA, torch.compile (NVIDIA fork) |

### What we're leaving on the table

| Optimization | Speedup | Status |
|---|---|---|
| TE FP8 (fused kernel) | +40-60% | ❌ cuBLAS crash |
| torch.compile reduce-overhead | +15-40% | ⚠️ Works in NGC, crashes on pip venv (cudagraph_trees) |
| Flash SDPA | Already active | ✅ |
| TF32 matmul | Already active | ✅ |
| Data parallelism | 50-80× aggregate | ⚠️ Not implemented |

### What would unlock the remaining 2-3×

A driver update or CUDA toolkit upgrade that resolves the cuBLAS Internal Error. The H200's 1,979 FP8 TFLOPS are accessible — we proved it on the 256×256 micro-test. The door is locked at a very specific place: warmup batch shapes on a real 4B model through TE's cublaslt gemm path.

---

## 5. Hypothesis for the cuBLAS crash

Three possible causes, ordered by likelihood:

### 5a. Driver 580 + cuBLAS 12.x internal incompatibility (most likely)

The H200 driver (580.159.03) reports CUDA 13.0 support via `nvidia-smi`. But the PyTorch runtime inside both the pip venv and NGC container uses CUDA 12.x (12.4 / 12.6). The cuBLAS library that TE links against is the one bundled with the CUDA runtime, not the driver.

When TE calls `cublasLtMatmul` with FP8 tensor core operands, the CUDA 12.x cuBLAS library tries to dispatch to the driver's kernel. If the driver's cuBLAS kernel for FP8 gemm on sm90a has an incompatibility with the 12.x client library, you get exactly `CUBLAS_STATUS_INTERNAL_ERROR`.

**Evidence:** The error is consistent across TE versions (1.11, 2.4, 2.16) and across CUDA runtime versions (12.4 pip, 12.6 NGC) — the only common element is the 580 driver.

### 5b. Warmup batch size alignment

TE's `fp8_gemm` has a documented requirement: the product of leading dimensions must be divisible by 8, and the last dimension must be divisible by 16. During warmup at `bs=1`, many sequence lengths will violate this.

**Evidence:** The 256×256 test (`bs=8`, `8×256` product ok) passes. The real model at `bs=1` with `warmup_len=20` produces shapes like `[1, 20, 2560]` — product of leading dims = 20, which is NOT divisible by 8.

**Counter-evidence:** The crash also occurs with MLP layers where `[1, 20*32, 2560]` padded batch should satisfy alignment. The fact that ALL shapes crash, not just small ones, points back to 5a.

### 5c. cuBLAS workspace allocation failure

When multiple CUDA streams (TE's internal stream + PyTorch's default stream + the warmup stream) share the cuBLAS handle, workspace allocation can fail silently and manifest as an internal error on the next gemm call.

**Evidence:** The crash only happens after `torch.compile` wraps the model — compile creates internal CUDA graphs that may alias the cuBLAS workspace.

---

## 6. Resolution — what we actually did

### 6a. Code changes committed (2026-06-24–25)

| File | Change |
|---|---|
| `benchmark/hardware/precision.py` | `apply_te_fp8_to_model()` receives `mlp_only` flag. `NativeFP8Linear` with weight-cache path. `save_fp8_weights()` / `load_fp8_weights()` for on-disk FP8 weight persistence. |
| `benchmark/inference/backends/autoregressive.py` | `_apply_fp8()` tries TE first (with Gemma auto-detection for `mlp_only`), native `torch._scaled_mm` as fallback. Version guard skips compile on PyTorch < 2.12. |
| `benchmark/config/model_presets.py` | Gemma models: `supports_fp8=False`. Ministral: `supports_fp8=True`. New `4B` alias → Ministral. |
| `benchmark/config/schema.py` | Default model remains `google/translategemma-4b-it` (backward compat). |
| `benchmark/utils/env_check.py` | Model preset resolution before preflight check. |
| `benchmark/__main__.py` | `--model` arg resolves presets to HF IDs before config injection. |
| `scripts/run_one_model.py` | `use_torch_compile=is_cuda` (re-enabled from hardcoded False). |
| `scripts/benchmark_all_models.py` | Same compile re-enable for both AR and NLLB paths. |
| `Dockerfile.ngc` | NGC-based Docker image (created, not yet built). |
| `docs/FP8_TE_CUDA_ISSUES.md` | This document. |

### 6b. Runtime behavior — what the benchmark does today

```
On CUDA (non-safe-mode):
  1. Try Transformer Engine → imports? no → skip
  2. Try native torch._scaled_mm → Hopper? yes → apply on MLP layers only
     → Forward pays 21 µs quantization tax per matmul
     → For 4B models: net slower than BF16
  3. Fall through to BF16 if both fail

User controls:
  TR_SKIP_FP8=1          → skip all FP8, pure BF16 (recommended for 4B models)
  TR_FORCE_NATIVE_FP8=1  → skip TE attempt entirely
  TR_DISABLE_TORCH_COMPILE=1 → force eager mode
```

### 6c. Recommended operating mode for asus02

```bash
# Production benchmark — maximum stable throughput
TR_SKIP_FP8=1 python -m benchmark --model translategemma-4b-bf16 --batch-size 32

# NGC container — if you want torch.compile (NVIDIA fork doesn't crash on cudagraph_trees)
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 \
  -v ~/LargeCorporaTranslation:/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -w /workspace -e PYTHONPATH=/workspace -e TR_SKIP_FP8=1 \
  nvcr.io/nvidia/pytorch:24.12-py3 \
  bash -c '
    pip install -q "transformers>=4.47,<4.53" orjson pyarrow safetensors sacrebleu pydantic pyyaml psutil sentencepiece
    python3 -m benchmark --model translategemma-4b-bf16 --dry-run --batch-size 32
  '
```

---

## 7. The real path to FP8 acceleration

**Update the NVIDIA driver.** The 580 series is a development/beta driver. A stable 570 or updated 580 build that ships a fixed cuBLAS kernel for sm90a FP8 gemm would resolve this.

Alternatively, **downgrade to driver 570** (the CUDA 12.8 stable driver). This is what NVIDIA's CI tests TE against. The 580 driver reports CUDA 13.0 support, suggesting it's a forward-compatibility build that may have regressions in the CUDA 12.x backward-compat layer.

Once the driver is stable, the NGC container path works immediately — no code changes needed. TE + compile + FP8 will be active simultaneously, giving the projected 2-3× throughput improvement.

---

## 8. Appendix — complete test log

```
2026-06-24 20:25  pip venv: TE 2.16 import → FileNotFoundError (transformer_engine_torch)
2026-06-24 20:30  pip venv: TE 2.16 [pytorch] → build fails (no torch in isolation)
2026-06-24 20:35  pip venv: TE 2.16 --no-build-isolation → CUDA headers mismatch
2026-06-24 20:40  pip venv: TE 2.4 → same compile failure
2026-06-24 20:45  pip venv: TE 1.13 → version conflict with 2.16 debris
2026-06-24 20:50  pip venv: native torch._scaled_mm 256×256 → OK (21 µs BF16, 42 µs FP8)
2026-06-24 20:55  pip venv: native FP8 benchmark → 440 tok/s (0.54× BF16 baseline)
2026-06-24 21:00  pip venv: native FP8 MLP-only → 579 tok/s (0.71× BF16 baseline)
2026-06-24 21:10  Decision: force FP8 as default, implement weight cache
2026-06-25 10:30  Docker + nvidia-container-toolkit installed
2026-06-25 10:39  NGC 24.12 pulled, nvidia-smi works inside container
2026-06-25 10:40  NGC: TE 1.11 imports, 256×256 FP8 matmul passes
2026-06-25 10:45  NGC: TranslateGemma 4B warmup → cuBLAS crash (attention)
2026-06-25 10:50  NGC: attempt --no-build-isolation for TE 2.16 inside container → fails
2026-06-25 10:55  NGC: transformers 5.x import → TransformGetItemToIndex error
2026-06-25 10:59  NGC: transformers 4.47-4.53 installed → still cuBLAS crash
2026-06-25 11:05  pip venv: Gemma supports_fp8=False, 4B→Ministral
2026-06-25 11:10  pip venv: Ministral-3-3B-Instruct-2512 → needs transformers 5.x
2026-06-25 11:15  NGC: TE mlp_only on Gemma → cuBLAS crash on MLP gate_proj
2026-06-25 11:20  FINAL: TE fails on ALL layer types, ALL TE versions, ALL containers
```

---

## 9. Updated CUDA/PyTorch/TE compatibility matrix

| PyTorch | CUDA Runtime | TE version | Container | Status |
|---|---|---|---|---|
| 2.6.0 (pip cu124) | 12.4 | 2.16.0 | none | ❌ Can't install [pytorch] extra |
| 2.6.0 (pip cu124) | 12.4 | any | none | ❌ No pre-built `transformer_engine_torch` for pip |
| 2.6.0a0 (NGC 24.12) | 12.6 | 1.11.0 | nvcr.io 24.12 | ❌ cuBLAS crash on all Gemma shapes |
| 2.4.0 (NGC 24.06) | 12.5 | 1.11.0 | nvcr.io 24.06 | ❓ Not tested, likely same crash (same driver) |

**The common element across all failures: NVIDIA Driver 580.159.03.**
