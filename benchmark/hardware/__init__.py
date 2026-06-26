"""Hardware abstraction layer — backend detection, precision dispatch, parallelism."""

from benchmark.hardware.backend import BackendDetector, DeviceInfo, detect_backend
from benchmark.hardware.precision import get_dtype, PrecisionMode
from benchmark.hardware.parallelism import (
    TensorParallelConfig,
    apply_tensor_parallelism,
    ensure_dist_initialized,
    get_tensor_parallel_config,
)

__all__ = [
    "BackendDetector",
    "DeviceInfo",
    "detect_backend",
    "get_dtype",
    "PrecisionMode",
    "TensorParallelConfig",
    "apply_tensor_parallelism",
    "ensure_dist_initialized",
    "get_tensor_parallel_config",
]
