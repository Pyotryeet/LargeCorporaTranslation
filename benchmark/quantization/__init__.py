"""Static quantization pipeline — SmoothQuant calibration, QAT, FP8 export.

This package is DECOUPLED from the inference backend and the transformers
library.  It operates on raw tensors and nn.Module graphs.

Quick start
-----------
    # 1. SmoothQuant PTQ (Post-Training Quantization)
    from benchmark.quantization.smoothquant import SmoothQuantCalibrator
    calibrator = SmoothQuantCalibrator(model, tokenizer)
    calibrator.calibrate(calibration_texts)

    # 2. Apply static FP8 quantization
    from benchmark.hardware.precision import apply_static_fp8_to_model
    apply_static_fp8_to_model(model)

    # 3. Run inference — StaticFP8Linear handles on-chip dequant

    # --- OR ---

    # 1. QAT (Quantization-Aware Training)
    from benchmark.quantization.qat import prepare_qat, export_qat_weights
    prepare_qat(model)
    # ... train ...
    weights = export_qat_weights(model)

    # 2. Save weights for static FP8 inference
    from benchmark.hardware.precision import save_fp8_weights, load_fp8_weights
    save_fp8_weights(model, model_path)
"""

from benchmark.quantization.smoothquant import (
    SmoothQuantCalibrator,
    ActivationCapture,
    compute_smooth_scales,
    apply_smooth_scales,
    compute_activation_scales,
)
from benchmark.quantization.qat import (
    FP8FakeQuantize,
    FakeQuantizedLinear,
    prepare_qat,
    export_qat_weights,
)

__all__ = [
    # SmoothQuant
    "SmoothQuantCalibrator",
    "ActivationCapture",
    "compute_smooth_scales",
    "apply_smooth_scales",
    "compute_activation_scales",
    # QAT
    "FP8FakeQuantize",
    "FakeQuantizedLinear",
    "prepare_qat",
    "export_qat_weights",
]
