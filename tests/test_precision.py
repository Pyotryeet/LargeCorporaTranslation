"""Tests for precision dispatch."""

import torch
from benchmark.hardware.precision import get_precision_config, get_dtype, has_fp8_support


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
