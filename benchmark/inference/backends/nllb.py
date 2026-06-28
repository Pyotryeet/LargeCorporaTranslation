"""NLLB encoder-decoder dispatcher — routes to CUDA or MPS backend based on hardware.

Uses ``HardwareDispatcherBackend`` from protocol.py to transparently forward all
``InferenceBackend`` protocol calls to the appropriate platform implementation.
"""

from benchmark.inference.backends.protocol import (
    HardwareDispatcherBackend,
    ModelCapability,
    ModelType,
)


class NLLBBackend(HardwareDispatcherBackend):
    """NLLB encoder-decoder translation dispatcher."""

    model_type = ModelType.ENCODER_DECODER
    capabilities = (
        ModelCapability.TRANSLATE | ModelCapability.FORWARD_ENCODE
        | ModelCapability.ENSEMBLE_READY
    )
    display_name = "NLLB Encoder-Decoder (Dispatcher)"

    _cuda_module = "benchmark.inference.backends.nllb_cuda"
    _cuda_class = "NLLBCUDABackend"
    _mps_module = "benchmark.inference.backends.nllb_mps"
    _mps_class = "NLLBMPSBackend"
