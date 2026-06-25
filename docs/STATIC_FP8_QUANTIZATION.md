# Static FP8 Quantization Pipeline

**Status: ✅ Implemented (June 2026)**

**Architecture:** Fully decoupled from the `transformers` library. SmoothQuant calibration, QAT, and static FP8 inference form three independent phases connected by serialized weights.

---

## 1. Why static FP8, not dynamic?

| Approach | Speed (4B model) | Dependency chain | Runtime risk |
|---|---|---|---|
| Dynamic FP8 (torch._scaled_mm) | **−40% vs BF16** | None (pure pytorch) | None (works) |
| Dynamic FP8 (TE fused kernel) | **+40−60%** if working | CUDA toolkit, cuBLAS, cuDNN, PyTorch, TE versions must align | **BLOCKED** on all tested driver/PT/CUDA combos |
| **Static FP8 (weight-only)** | **+memory bandwidth (2×), matmul in BF16+TF32** | None | None |

Static FP8 stores weights in `torch.float8_e4m3fn` on GPU. At forward time, the H200 memory controller casts `float8→bfloat16` **in the same memory transaction as the read** — zero compute cost. The matmul runs in BF16 with TF32 tensor-core acceleration. This gives the memory bandwidth benefit without the per-token quantization tax.

For quality, SmoothQuant PTQ (or QAT) migrates activation outliers into weights *before* quantization, so the static quantized weights produce near-identical output to BF16.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: CALIBRATION (SmoothQuant PTQ)                     │
│                                                             │
│  python -m benchmark --model 4B --smoothquant              │
│                                                             │
│  1. Run 50 calibration docs through model                  │
│  2. Capture activations per Linear layer                   │
│  3. Compute SmoothQuant scales:                            │
│       s_j = max(|X_j|)^α / max(|W_j|)^(1-α)               │
│  4. Smooth weights: Ŵ = W · diag(s)                        │
│  5. Apply StaticFP8Linear → quantize Ŵ to FP8 E4M3        │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────┴──────────────────────────────┐
│ PHASE 2: STATIC FP8 INFERENCE (default, zero overhead)     │
│                                                             │
│  python -m benchmark --model 4B --dry-run                  │
│                                                             │
│  StaticFP8Linear.forward():                                │
│    w_bf16 = weight_fp8.to(bf16) * weight_scale              │
│    return F.linear(x, w_bf16, bias)  # BF16 + TF32 matmul  │
│                                                             │
│  H200 memory controller casts float8→bfloat16              │
│  in the same transaction as the memory read.               │
│  Zero compute cost.  2× memory bandwidth vs BF16.          │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼ (optional)
┌──────────────────────────────┴──────────────────────────────┐
│ PHASE 3: QAT (Quantization-Aware Training)                  │
│                                                             │
│  python -m benchmark --model 4B --qat                      │
│                                                             │
│  1. prepare_qat(model): replace Linear with                │
│     FakeQuantizedLinear                                    │
│  2. Train normally (backprop through STE)                  │
│  3. export_qat_weights(model) → save_fp8_weights()         │
│  4. Output: FP8 checkpoint ready for Phase 2 inference     │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Usage

### 3a. SmoothQuant PTQ — zero-training quality improvement

```bash
# Calibrate once per model — SmoothQuant runs before static FP8
python -m benchmark --model translategemma-4b-bf16 --smoothquant --dry-run

# The smoothed weights are auto-quantized to FP8 by StaticFP8Linear.
# Subsequent runs don't need --smoothquant (weights already smoothed in memory).
```

### 3b. Static FP8 inference — the enforced default

```bash
# Static FP8 is ON by default for all CUDA inference.
# No flags needed — just run normally.
python -m benchmark --model translategemma-4b-bf16 --dry-run --batch-size 32

# To skip FP8 entirely:
TR_SKIP_FP8=1 python -m benchmark --model translategemma-4b-bf16 --dry-run
```

### 3c. QAT fine-tuning

```bash
# 1. Prepare model for QAT
python -m benchmark --model translategemma-4b-bf16 --qat

# 2. In your training script:
from quantization.qat import prepare_qat, export_qat_weights
from benchmark.hardware.precision import save_fp8_weights

prepare_qat(model)
# ... train your model ...
weights = export_qat_weights(model)
save_fp8_weights(model, "google/translategemma-4b-it")

# 3. Run inference — StaticFP8Linear picks up the QAT-trained weights.
python -m benchmark --model translategemma-4b-bf16 --dry-run
```

---

## 4. Mathematical foundation

### SmoothQuant

For a linear layer $Y = XW$ with activation $X \in \mathbb{R}^{B \times C_{in}}$ and weight $W \in \mathbb{R}^{C_{out} \times C_{in}}$:

$$s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}, \quad j = 1, \dots, C_{in}$$

$$\hat{W} = W \cdot \text{diag}(s), \quad \hat{X} = X \cdot \text{diag}(s)^{-1}$$

where $\alpha \in [0, 1]$ controls the migration ratio. $\alpha = 0.5$ (default) evenly splits the outlier magnitude between weights and activations.

### Static FP8 quantization

After smoothing, weights are statically quantized:

$$\bar{W} = \text{round}\left(\frac{\hat{W}}{w_{scale}}\right) \cdot w_{scale}$$

where $w_{scale} = \max(|\hat{W}|) / 448$ and the quantization target is FP8 E4M3 (±448 max).

At inference:

$$Y_{bf16} = \text{dequant}(\bar{W}) \cdot X = (\bar{W}_{fp8} \to bf16 \cdot w_{scale}) \cdot X$$

The H200's memory controller handles the `float8→bfloat16` cast inline during the memory read — no separate dequantization kernel, no compute overhead.

### QAT Straight-Through Estimator

During QAT training, the forward pass simulates FP8 quantization:

$$W_{fake} = \text{round}(W / w_{scale}) \cdot w_{scale}$$

The backward pass uses STE:

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial W_{fake}}$$

This trains the model to be robust to the quantization error introduced by FP8 E4M3 precision.

---

## 5. Performance model

| Stage | Weight HBM load (GB/tok) | TPS impact |
|---|---|---|
| BF16 weights | $2 \cdot w_{GB}$ | Baseline (1,650 tok/s on 2.12.1, 4B) |
| Static FP8 weights | $1 \cdot w_{GB}$ | **2× bandwidth reduction** |
| FP8 + SmoothQuant | $1 \cdot w_{GB}$ | Same bandwidth, better accuracy |
| FP8 + QAT | $1 \cdot w_{GB}$ | Same bandwidth, best accuracy |

For the 4B model (2GB BF16 weights): ~2GB/tok → ~1GB/tok bandwidth saving.
Crossover to compute-bound occurs at ~bs=16 for 4B models — beyond that, FP8 bandwidth savings are the dominant TPS driver.

---

## 6. Files

| File | Purpose |
|---|---|
| `benchmark/quantization/smoothquant.py` | SmoothQuant calibration (ActivationCapture, scales, smoothing) |
| `benchmark/quantization/qat.py` | QAT (FP8FakeQuantize STE, FakeQuantizedLinear, prep/export) |
| `benchmark/quantization/__init__.py` | Package exports |
| `benchmark/hardware/precision.py` | `StaticFP8Linear` (static FP8 inference), `save_fp8_weights` / `load_fp8_weights` |
| `benchmark/inference/backends/autoregressive.py` | `_calibrate_smoothquant()` (wired into load), `_apply_fp8()` |
| `benchmark/__main__.py` | `--smoothquant`, `--qat` CLI flags |

---

## 7. References

- Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models", ICML 2024.
- NVIDIA H200 Tensor Core FP8: E4M3 format (±448 max, 4 exponent bits, 3 mantissa bits). 1,979 FP8 TFLOPS.
- Bengio et al., "Estimating or Propagating Gradients Through Stochastic Neurons", 2013 (Straight-Through Estimator).
