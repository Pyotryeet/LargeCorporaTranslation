"""Autoregressive MPS backend — thin subclass of the CUDA backend.

All platform-specific logic (MPS vs CUDA) is handled internally via
``self.backend_name`` checks in ``AutoregressiveCUDABackend``.
This subclass exists for dispatcher compatibility — the dispatcher in
``autoregressive.py`` imports it by name when ``detect_backend()``
returns a non-CUDA device.

.. note::

    torch.compile is **disabled on MPS** (inductor deadlocks).
    Static FP8 is **skipped on MPS** (no H200 tensor cores).
    Both are enforced via ``self.backend_name != "cuda"`` checks in
    ``_apply_extreme_compile()`` and ``_apply_fp8()``.
"""

from benchmark.inference.backends.autoregressive_cuda import (
    AutoregressiveCUDABackend,
)


class AutoregressiveMPSBackend(AutoregressiveCUDABackend):
    """MPS-accelerated autoregressive inference backend.

    Inherits all behaviour from ``AutoregressiveCUDABackend``.
    Platform differences are handled by the base class via
    ``self.backend_name`` guards — no overrides needed.
    """
