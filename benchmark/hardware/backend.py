"""Backend detection for CUDA, MPS, and CPU with device-info reporting.

Supports 3B–4B model architectures (TranslateGemma 4B, Ministral 3B).

v2.0: Module-level Transformer Engine availability cache (P0).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional

import torch

from benchmark.hardware.precision import is_transformer_engine_available

logger = logging.getLogger(__name__)

BackendType = Literal["cuda", "mps", "cpu"]


@dataclass
class DeviceInfo:
    """Normalised device information — backend-agnostic."""

    backend: BackendType
    device: torch.device
    num_devices: int
    name: str = ""
    precision: str = "bfloat16"
    supports_tensor_parallelism: bool = False
    recommended_dtype: torch.dtype = torch.bfloat16
    total_memory_gb: float = 0.0
    te_available: bool = False

    def __post_init__(self):
        if self.backend == "cuda":
            self.supports_tensor_parallelism = self.num_devices >= 2
        if not self.name:
            self.name = self._resolve_name()
        if self.total_memory_gb <= 0:
            self.total_memory_gb = self._resolve_memory()

    @property
    def type(self) -> str:
        """Backward-compatible alias for ``backend``."""
        return self.backend

    def _resolve_name(self) -> str:
        if self.backend == "cuda":
            names = [torch.cuda.get_device_name(i) for i in range(self.num_devices)]
            return ", ".join(names)
        elif self.backend == "mps":
            return "Apple Silicon (MPS)"
        return "CPU"

    def _resolve_memory(self) -> float:
        import psutil
        if self.backend == "cuda":
            total = 0.0
            for i in range(self.num_devices):
                props = torch.cuda.get_device_properties(i)
                total += props.total_memory / (1024**3)
            return total
        if self.backend == "mps":
            return BackendDetector._resolve_mps_memory_gb()
        # CPU: system RAM (callers should budget at most ~60-70% for model weights).
        return psutil.virtual_memory().total / (1024**3)

    def per_device_memory_gb(self) -> float:
        if self.backend == "cuda" and self.num_devices > 0:
            return self.total_memory_gb / self.num_devices
        return self.total_memory_gb


class BackendDetector:
    """Detect the best available compute backend at startup.

    Priority: CUDA (>=2 GPUs) > MPS (Apple Silicon) > CPU (fallback).
    """

    @staticmethod
    def _resolve_mps_memory_gb() -> float:
        """Estimate the Metal carve-out on Apple Silicon.

        PyTorch 2.1+ provides ``torch.mps.recommended_max_memory()`` which
        returns the Metal-recommended memory limit (in bytes) — a fraction of
        total unified memory that the GPU driver can safely use.

        Fallback: ``psutil.virtual_memory().total * 0.75`` as a conservative
        estimate when ``recommended_max_memory`` is unavailable (older PyTorch).
        Unlike CUDA where ``get_device_properties`` reports dedicated VRAM,
        MPS shares unified memory with the system, so reporting total system
        RAM is misleading for GPU budget calculations.
        """
        try:
            # bytes -> GB
            return torch.mps.recommended_max_memory() / (1024**3)
        except (AttributeError, RuntimeError):
            import psutil
            return psutil.virtual_memory().total * 0.75 / (1024**3)

    @staticmethod
    def detect(preferred: str = "auto") -> DeviceInfo:
        preferred_lower = preferred.strip().lower()
        if preferred_lower == "auto":
            return BackendDetector._auto_detect()
        elif preferred_lower == "cuda":
            return BackendDetector._cuda_detect()
        elif preferred_lower == "mps":
            return BackendDetector._mps_detect()
        elif preferred_lower == "cpu":
            return BackendDetector._cpu_detect()
        else:
            raise ValueError(
                f"Unknown backend: {preferred!r}. Expected one of: auto, cuda, mps, cpu."
            )

    @staticmethod
    def _auto_detect() -> DeviceInfo:
        cuda_ok = torch.cuda.is_available()

        # torch.backends.mps.is_available() can segfault on some Linux
        # configurations (PyTorch < 2.3 with ancient GPU drivers).  Wrap it
        # so a crash is caught and treated as "MPS not available".
        mps_ok = False
        try:
            mps_ok = torch.backends.mps.is_available()
        except (RuntimeError, AttributeError, SystemError):
            mps_ok = False

        if cuda_ok:
            logger.info("Auto-detected CUDA with %d GPUs", torch.cuda.device_count())
            return BackendDetector._cuda_detect()
        elif mps_ok:
            logger.info("Auto-detected MPS (Apple Silicon)")
            return BackendDetector._mps_detect()
        else:
            logger.warning("No GPU backend detected — falling back to CPU")
            return BackendDetector._cpu_detect()

    @staticmethod
    def _cuda_detect() -> DeviceInfo:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA backend requested but not available.")
        n_devices = torch.cuda.device_count()
        if n_devices < 2:
            logger.warning(
                f"CUDA mode: only {n_devices} GPU(s). "
                "Tensor parallelism (TP=2) requires 2 GPUs."
            )
        device = torch.device("cuda:0")
        info = DeviceInfo(
            backend="cuda",
            device=device,
            num_devices=n_devices,
            precision="float8_e4m3fn",
            supports_tensor_parallelism=n_devices >= 2,
            recommended_dtype=torch.bfloat16,
            te_available=is_transformer_engine_available(),
        )
        logger.info("CUDA backend: %s, %d device(s)", info.name, info.num_devices)
        return info

    @staticmethod
    def _mps_detect() -> DeviceInfo:
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS backend requested but not available. "
                "This machine may be Intel-based. Use --backend cpu."
            )
        device = torch.device("mps")
        total_mem = BackendDetector._resolve_mps_memory_gb()
        logger.info(
            "MPS backend: Apple Silicon, %.1f GB Metal carve-out (of unified memory)",
            total_mem,
        )
        return DeviceInfo(
            backend="mps",
            device=device,
            num_devices=1,
            name="Apple Silicon (MPS)",
            precision="bfloat16",
            supports_tensor_parallelism=False,
            recommended_dtype=torch.bfloat16,
            total_memory_gb=total_mem,
            te_available=False,
        )

    @staticmethod
    def _cpu_detect() -> DeviceInfo:
        device = torch.device("cpu")
        import psutil
        # NOTE: total_memory_gb is set from psutil (system RAM), but usable headroom
        # for model loading is far lower. The OS, other processes, and page cache
        # consume significant memory. A large model (e.g. Gemma 3 12B at ~24 GB
        # in float32) will OOM well before total_memory_gb is reached. Callers
        # should budget at most ~60-70% of this value for model weights.
        total_mem = psutil.virtual_memory().total / (1024**3)
        return DeviceInfo(
            backend="cpu",
            device=device,
            num_devices=1,
            name="CPU",
            precision="float32",
            supports_tensor_parallelism=False,
            recommended_dtype=torch.float32,
            total_memory_gb=total_mem,
            te_available=False,
        )


def detect_backend(preferred: str = "auto") -> DeviceInfo:
    """Convenience — detect and return the best available backend."""
    return BackendDetector.detect(preferred)
