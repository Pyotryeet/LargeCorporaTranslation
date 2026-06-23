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
