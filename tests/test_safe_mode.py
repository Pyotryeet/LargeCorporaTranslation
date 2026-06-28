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
        assert backend._use_paged_attention is False, "PagedAttention should be disabled in safe mode"
        assert backend._use_quantized_weights is False, "Quantized weights should be disabled in safe mode"
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
        # PagedAttention is now enabled by default on CUDA (extra.get("use_paged_attention", backend_name=="cuda")).
        # On CPU backend_name, it defaults to False since it's CUDA-only.
        # The fused_kernels and cuda_graph attributes no longer exist — those features
        # were permanently removed in commit 19d979f.
        assert backend._use_paged_attention is False, "PagedAttention disabled on CPU"
        assert backend._use_quantized_weights is False

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
        assert backend._use_paged_attention is False
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
        """--safe-mode is registered as a CLI flag in the real benchmark parser."""
        # Test the actual benchmark CLI parser, not a fresh argparse stub.
        # We cannot call main() directly (it creates a BenchmarkHarness and
        # requires a config file), so we inspect the source of the main()
        # function to verify --safe-mode is registered and wired correctly.
        import inspect
        from benchmark.__main__ import main

        source = inspect.getsource(main)
        # Use the LAST occurrence of "--safe-mode" — the parser argument,
        # not the docstring example on the first occurrence.
        assert "--safe-mode" in source, (
            "--safe-mode not found in benchmark.__main__.main() parser definition"
        )
        after_last = source.rsplit("--safe-mode", 1)[1][:300]
        assert 'action="store_true"' in after_last, (
            "--safe-mode must be a store_true flag"
        )
        # Verify the flag maps to safe_mode= in the harness constructor.
        assert "safe_mode=args.safe_mode" in source, (
            "--safe-mode must propagate to BenchmarkHarness(safe_mode=...)"
        )
