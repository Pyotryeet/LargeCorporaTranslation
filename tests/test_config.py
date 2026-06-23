"""Tests for configuration schema validation."""

import pytest
import yaml
from pathlib import Path
from benchmark.config.schema import BenchmarkConfig, ModelConfig, RuntimeConfig, DataConfig, load_config


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.model_path == "google/translategemma-4b-it"
        assert cfg.max_input_tokens == 512
        assert cfg.max_new_tokens == 512
        assert cfg.temperature == 0.0
        assert cfg.do_sample is False
        assert cfg.dtype == "auto"

    def test_empty_tokenizer_falls_back_to_model_path(self):
        cfg = ModelConfig(model_path="/custom/model", tokenizer_path="")
        assert cfg.tokenizer_path == "/custom/model"

    def test_explicit_tokenizer(self):
        cfg = ModelConfig(model_path="/custom/model", tokenizer_path="/custom/tokenizer.model")
        assert cfg.tokenizer_path == "/custom/tokenizer.model"

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError):
            ModelConfig(dtype="int4")


class TestRuntimeConfig:
    def test_defaults(self):
        cfg = RuntimeConfig()
        assert cfg.target_duration_seconds == 7200
        assert cfg.checkpoint_interval_seconds == 300
        assert cfg.seed == 42

    def test_duration_bounds(self):
        with pytest.raises(ValueError):
            RuntimeConfig(target_duration_seconds=30)  # below min of 60


class TestDataConfig:
    def test_defaults(self):
        cfg = DataConfig()
        assert cfg.prefetch_workers == 4
        assert cfg.shuffle is True
        assert cfg.min_chunk_tokens == 10
        assert cfg.max_garbage_ratio == 0.95


class TestBenchmarkConfig:
    def test_valid_minimal_config(self):
        cfg = BenchmarkConfig()
        assert cfg.backend == "auto"
        assert cfg.model.model_path == "google/translategemma-4b-it"
        assert cfg.runtime.target_duration_seconds == 7200

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValueError):
            BenchmarkConfig(unknown_field="should_fail")

    def test_load_from_yaml(self, tmp_path, fixture_dir):
        config_path = fixture_dir / "config_test.yaml"
        if config_path.exists():
            cfg = load_config(str(config_path))
            assert cfg is not None
            assert cfg.backend == "cpu"
            assert cfg.runtime.seed == 42
            assert cfg.runtime.target_duration_seconds == 60
