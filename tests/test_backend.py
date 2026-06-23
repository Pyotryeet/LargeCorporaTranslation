"""Tests for hardware backend detection."""

import pytest
from benchmark.hardware.backend import BackendDetector, DeviceInfo, detect_backend


class TestBackendDetector:
    def test_detect_returns_device_info(self):
        info = detect_backend()
        assert isinstance(info, DeviceInfo)
        assert info.backend in ("cuda", "mps", "cpu")
        assert info.num_devices >= 1
        assert info.name != ""

    def test_explicit_cpu(self):
        info = BackendDetector.detect("cpu")
        assert info.backend == "cpu"
        assert info.num_devices == 1
        # Devices may have different attribute names; be flexible
        assert info.name is not None

    def test_explicit_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            BackendDetector.detect("invalid_backend")

    def test_device_info_has_memory(self):
        info = detect_backend()
        assert info.total_memory_gb > 0

    def test_mps_detection_requires_apple_silicon(self):
        pytest.importorskip("torch")
        import torch
        if not torch.backends.mps.is_available():
            with pytest.raises(RuntimeError):
                BackendDetector.detect("mps")

    # ── Edge case tests ──

    def test_cpu_info_has_device_count(self):
        """CPU DeviceInfo reports num_devices >= 1."""
        info = BackendDetector.detect("cpu")
        assert info.backend == "cpu"
        assert info.num_devices >= 1

    def test_detect_default_discovers_backend(self):
        """detect_backend() with no argument discovers a valid backend."""
        info = detect_backend()
        assert info.backend in ("cuda", "mps", "cpu")
        assert isinstance(info.num_devices, int)
        assert info.num_devices >= 0

    def test_detect_unknown_device(self):
        """detect('cuda') on a machine with no CUDA raises RuntimeError."""
        pytest.importorskip("torch")
        import torch
        if not torch.cuda.is_available():
            with pytest.raises(RuntimeError, match="CUDA not available"):
                BackendDetector.detect("cuda")

    def test_detect_auto_maps_correctly(self):
        """detect('auto') selects best available backend."""
        info = BackendDetector.detect("auto")
        assert info.backend in ("cuda", "mps", "cpu")

    def test_detect_empty_string(self):
        """detect('') triggers ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            BackendDetector.detect("")
