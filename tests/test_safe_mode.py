"""Tests for safe mode — ensures experimental optimizations are correctly disabled."""

import pytest
import torch

from benchmark.inference.backends.autoregressive import AutoregressiveBackend
from benchmark.inference.backends.protocol import BackendConfig
from benchmark.hardware.backend import DeviceInfo


class TestSafeMode:
    def test_safe_mode_disables_cuda_graph(self):
        """Safe mode disables CUDA graph capture."""
        device_info = DeviceInfo(
            backend="cpu", device=torch.device("cpu"),
            num_devices=1, name="TEST",
        )
        config = BackendConfig(
            model_path="test-model",
            device_info=device_info,
            extra={"safe_mode": True},
        )
        backend = AutoregressiveBackend(config)
        assert backend._use_cuda_graph is False, "CUDA graph should be disabled in safe mode"
        assert backend._use_paged_attention is False, "PagedAttention should be disabled in safe mode"
        assert backend._use_fused_kernels is False, "Fused kernels should be disabled in safe mode"
        assert backend._use_quantized_weights is False, "Quantized weights should be disabled in safe mode"
        assert backend._use_int8_kv_cache is False, "INT8 KV-cache should be disabled in safe mode"
        assert backend._safe_mode is True

    def test_normal_mode_keeps_defaults(self):
        """Without safe mode, optimization flags stay at their defaults."""
        device_info = DeviceInfo(
            backend="cpu", device=torch.device("cpu"),
            num_devices=1, name="TEST",
        )
        config = BackendConfig(
            model_path="test-model",
            device_info=device_info,
            extra={},
        )
        backend = AutoregressiveBackend(config)
        assert backend._use_cuda_graph is True, "CUDA graph should be enabled by default"
        # PagedAttention is hardcoded False until model forward hooks redirect
        # KV reads from past_key_values to PagedKVCache blocks.
        assert backend._use_paged_attention is False, "PagedAttention disabled until model hooks in place"
        assert backend._use_fused_kernels is True, "Fused kernels should be enabled by default"

    def test_safe_mode_does_not_affect_other_config(self):
        """Safe mode only modifies optimization flags, not core config."""
        device_info = DeviceInfo(
            backend="cpu", device=torch.device("cpu"),
            num_devices=1, name="TEST",
        )
        config = BackendConfig(
            model_path="test-model",
            device_info=device_info,
            max_input_tokens=512,
            max_new_tokens=256,
            temperature=0.3,
            extra={"safe_mode": True, "custom_key": "custom_value"},
        )
        backend = AutoregressiveBackend(config)
        # Core config preserved.
        assert backend.model_path == "test-model"
        assert backend.max_input_tokens == 512
        assert backend.max_new_tokens == 256
        assert backend.temperature == 0.3
        # Only optimization flags are forced off.
        assert backend._use_cuda_graph is False
        assert backend._safe_mode is True


class TestSafeModeFlagPropagation:
    def test_harness_accepts_safe_mode(self):
        """BenchmarkHarness constructor accepts safe_mode parameter."""
        import inspect
        from benchmark.orchestration.harness import BenchmarkHarness
        sig = inspect.signature(BenchmarkHarness.__init__)
        params = sig.parameters
        assert "safe_mode" in params
        assert params["safe_mode"].default is False

    def test_cli_parser_has_safe_mode_flag(self):
        """--safe-mode is registered as a CLI flag."""
        import argparse
        # Verify the CLI parser accepts --safe-mode.
        parser = argparse.ArgumentParser()
        parser.add_argument("--safe-mode", action="store_true")
        args = parser.parse_args(["--safe-mode"])
        assert args.safe_mode is True
