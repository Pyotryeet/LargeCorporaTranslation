"""End-to-end integration tests for the benchmark harness.

The full pipeline E2E script has been moved to scripts/run_e2e_benchmark.py
because it downloads multi-GB models and runs for 120+ seconds, making it
unsuitable as an import-time pytest test.

This module contains lightweight pytest tests that validate the config
wiring, import paths, and module structure without loading real models.
"""

import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestE2EConfigWiring:
    """Verify all import paths used by the full E2E pipeline are valid."""

    def test_benchmark_modules_importable(self):
        """All major benchmark subpackages are importable."""
        import benchmark

        # Core subpackages must exist (import check only, no model loading).
        subpackages = [
            "benchmark.config",
            "benchmark.hardware",
            "benchmark.inference",
            "benchmark.data",
            "benchmark.metrics",
            "benchmark.orchestration",
            "benchmark.reporting",
            "benchmark.utils",
        ]
        for pkg_name in subpackages:
            __import__(pkg_name)

    def test_config_schema_constructs_minimal(self):
        """BenchmarkConfig can be constructed with minimal valid fields."""
        from benchmark.config.schema import (
            BenchmarkConfig, ModelConfig, RuntimeConfig,
            DataConfig, ExtrapolationConfig,
        )

        config = BenchmarkConfig(
            backend="cpu",
            model=ModelConfig(
                max_input_tokens=64,
                max_new_tokens=32,
            ),
            runtime=RuntimeConfig(
                target_duration_seconds=10,
            ),
            data=DataConfig(
                input_paths=["tests/fixtures/sample_input.jsonl"],
                output_dir="/tmp/test_e2e_output",
            ),
            extrapolation=ExtrapolationConfig(
                total_clearnet_non_tr_tokens=6_230_000_000_000,
            ),
        )
        assert config.backend == "cpu"
        assert config.model.max_input_tokens == 64
        assert config.runtime.target_duration_seconds == 10

    def test_device_info_detect_cpu(self):
        """DeviceInfo can be created for CPU backend."""
        import torch
        from benchmark.hardware.backend import DeviceInfo

        di = DeviceInfo(backend="cpu", device=torch.device("cpu"), num_devices=1)
        assert di.backend == "cpu"
        assert di.num_devices == 1

    def test_backend_detection_cpu(self):
        """detect_backend returns valid DeviceInfo for CPU."""
        from benchmark.hardware.backend import detect_backend

        device_info = detect_backend("cpu")
        assert device_info.backend == "cpu"
        assert device_info.num_devices >= 1

    def test_inference_engine_imports(self):
        """Inference engine module imports without model loads."""
        from benchmark.inference.engine import (
            InferenceEngine, TranslationResult, BatchResult,
        )
        # Classes exist — no assertion needed beyond import success.

    def test_e2e_script_exists_in_scripts(self):
        """The full E2E benchmark script lives in scripts/, not tests/."""
        script_path = PROJECT_ROOT / "scripts" / "run_e2e_benchmark.py"
        # The script should exist; if not, it needs to be moved from tests/.
        if not script_path.exists():
            pytest.fail(
                "E2E benchmark script not found at scripts/run_e2e_benchmark.py. "
                "The old tests/test_e2e.py was a 368-line script (not a test) "
                "that downloaded multi-GB models at import time. "
                "Move it to scripts/ and add proper test functions here."
            )
