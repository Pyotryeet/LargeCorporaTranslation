"""Tests for precision dispatch."""

import sys
import pytest
import torch
from benchmark.hardware.precision import get_precision_config, get_dtype, has_fp8_support


class TestFP8Support:
    """FP8 (float8) precision tests — Transformer Engine dependent."""

    def test_fp8_not_supported_on_cpu(self):
        """FP8 is never available on CPU."""
        assert has_fp8_support("cpu") is False

    def test_fp8_not_supported_on_mps(self):
        """FP8 is never available on Apple Silicon MPS."""
        assert has_fp8_support("mps") is False

    def test_fp8_requires_transformer_engine_import(self):
        """has_fp8_support('cuda') checks for transformer_engine availability."""
        # On systems without CUDA or without transformer_engine, this returns False.
        # The key invariant: it returns a bool, never raises.
        result = has_fp8_support("cuda")
        assert isinstance(result, bool)

    def test_precision_config_fp8_flag(self):
        """When fp8 is not available, uses_transformer_engine is False."""
        cfg = get_precision_config("cuda")
        assert "uses_transformer_engine" in cfg.to_dict()
        assert isinstance(cfg.uses_transformer_engine, bool)

    def test_fp8_not_in_config_for_cpu(self):
        """CPU precision config never enables transformer engine."""
        cfg = get_precision_config("cpu")
        d = cfg.to_dict()
        assert d["uses_transformer_engine"] is False

    def test_fp8_backend_attribute_present(self):
        """All backend configs expose uses_transformer_engine."""
        for backend in ("cuda", "mps", "cpu"):
            cfg = get_precision_config(backend)
            assert hasattr(cfg, "uses_transformer_engine"), (
                f"{backend} config missing uses_transformer_engine"
            )


class TestPrecisionConfig:
    def test_cuda_auto_returns_bf16(self):
        cfg = get_precision_config("cuda", "auto")
        assert cfg.master_dtype == torch.bfloat16

    def test_mps_auto_returns_bf16(self):
        cfg = get_precision_config("mps", "auto")
        assert cfg.master_dtype == torch.bfloat16

    def test_cpu_auto_returns_fp32(self):
        cfg = get_precision_config("cpu", "auto")
        assert cfg.master_dtype == torch.float32

    def test_explicit_fp16(self):
        cfg = get_precision_config("cuda", "float16")
        assert cfg.master_dtype == torch.float16

    def test_explicit_bf16(self):
        cfg = get_precision_config("mps", "bfloat16")
        assert cfg.master_dtype == torch.bfloat16

    def test_get_dtype_convenience(self):
        assert get_dtype("cuda") == torch.bfloat16
        assert get_dtype("mps") == torch.bfloat16
        assert get_dtype("cpu") == torch.float32

    def test_fp8_only_on_cuda(self):
        assert has_fp8_support("mps") is False
        assert has_fp8_support("cpu") is False

    def test_config_to_dict(self):
        cfg = get_precision_config("cuda")
        d = cfg.to_dict()
        assert d["backend"] == "cuda"
        assert "master_dtype" in d
        assert "compute_dtype" in d
        assert "uses_transformer_engine" in d
