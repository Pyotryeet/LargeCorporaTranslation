"""Autoregressive dispatcher — routes to CUDA or MPS backend based on hardware.

Uses ``HardwareDispatcherBackend`` from protocol.py to transparently forward all
``InferenceBackend`` protocol calls to the appropriate platform implementation.
"""

from benchmark.inference.backends.protocol import (
    HardwareDispatcherBackend,
    ModelCapability,
    ModelType,
)


class AutoregressiveBackend(HardwareDispatcherBackend):
    """Autoregressive translation dispatcher."""

    model_type = ModelType.AUTOREGRESSIVE
    capabilities = (
        ModelCapability.TRANSLATE | ModelCapability.FORWARD_ENCODE
        | ModelCapability.QUANTIZABLE_KV
        | ModelCapability.SPECULATIVE | ModelCapability.ENSEMBLE_READY
    )
    display_name = "Autoregressive (Dispatcher)"

    _cuda_module = "benchmark.inference.backends.autoregressive_cuda"
    _cuda_class = "AutoregressiveCUDABackend"
    _mps_module = "benchmark.inference.backends.autoregressive_mps"
    _mps_class = "AutoregressiveMPSBackend"
