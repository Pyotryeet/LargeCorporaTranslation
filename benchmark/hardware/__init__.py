"""Hardware abstraction layer — backend detection, precision dispatch, parallelism.

v3.3 additions: Runtime JIT kernel compilation (CUDA C++ / PTX / Metal MSL),
                Triton fused kernels, CUDA graphs, KV-cache quantization.
"""

from benchmark.hardware.backend import BackendDetector, DeviceInfo, detect_backend
from benchmark.hardware.precision import get_dtype, PrecisionMode
from benchmark.hardware.parallelism import (
    TensorParallelConfig,
    apply_tensor_parallelism,
    ensure_dist_initialized,
    get_tensor_parallel_config,
)
from benchmark.hardware.jit_compiler import (
    JITCompiler,
    get_jit_compiler,
    precompile_all_kernels,
    get_kernel,
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
    # v3.3: JIT kernel compilation.
    "JITCompiler",
    "get_jit_compiler",
    "precompile_all_kernels",
    "get_kernel",
]
