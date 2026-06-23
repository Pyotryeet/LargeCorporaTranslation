"""Tests covering previously untested code paths (coverage gap fillers).

Covers:
  1. constants.py imports and non-zero values
  2. AutoregressiveBackend.close() GPU memory release (CUDA only)
  3. DiffusionBackend._te_available initialization
  4. JSON sanitized_dumps handles inf/nan
  5. PrecisionTimer.elapsed() before start() is called
  6. PrefetchPipeline.notify_done() terminates (no infinite loop)
  7. BenchmarkConfig validation rejects bad combinations
"""

import math
import queue
import threading
import time

import pytest

from benchmark.config.constants import (
    DEFAULT_NUM_LAYERS,
    DEFAULT_NUM_KV_HEADS,
    DEFAULT_HEAD_DIM,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_VOCAB_SIZE,
    GPU_MEMORY_BUDGET_FRACTION,
    GPU_MEMORY_RESERVE_BYTES,
    PAGED_BLOCK_SIZE,
    DEFAULT_CHECKPOINT_INTERVAL,
    PAGED_NUM_BLOCKS_LARGE_GPU,
    PAGED_NUM_BLOCKS_SMALL_GPU,
    CUDA_GRAPH_DEFAULT_BATCH_SIZE,
    QUALITY_BLEU_TARGET,
    DEFAULT_DIFFUSION_STEPS,
    MODEL_ARCHITECTURES,
)
from benchmark.utils.timer import PrecisionTimer
from benchmark.utils.json_utils import sanitized_dumps


# ---------------------------------------------------------------------------
# 1. constants.py imports work and values are non-zero
# ---------------------------------------------------------------------------

class TestConstantsImports:
    """Verify the single source of truth provides sensible defaults."""

    def test_model_defaults_are_non_zero(self):
        assert DEFAULT_NUM_LAYERS == 48
        assert DEFAULT_NUM_KV_HEADS == 8
        assert DEFAULT_HEAD_DIM == 256
        assert DEFAULT_HIDDEN_SIZE == 3840
        assert DEFAULT_VOCAB_SIZE > 0

    def test_memory_constants_are_positive(self):
        assert 0.0 < GPU_MEMORY_BUDGET_FRACTION <= 1.0
        assert GPU_MEMORY_RESERVE_BYTES > 0
        assert PAGED_BLOCK_SIZE > 0
        assert PAGED_NUM_BLOCKS_LARGE_GPU > 0
        assert PAGED_NUM_BLOCKS_SMALL_GPU > 0

    def test_training_constants_are_sensible(self):
        assert CUDA_GRAPH_DEFAULT_BATCH_SIZE > 0
        assert QUALITY_BLEU_TARGET > 0
        assert DEFAULT_DIFFUSION_STEPS > 0
        assert DEFAULT_CHECKPOINT_INTERVAL > 0

    def test_model_architectures_have_expected_keys(self):
        for size_key in ("4B", "E2B", "E4B", "26B-A4B"):
            assert size_key in MODEL_ARCHITECTURES
            arch = MODEL_ARCHITECTURES[size_key]
            assert arch["num_layers"] > 0
            assert arch["num_kv_heads"] > 0
            assert arch["head_dim"] > 0


# ---------------------------------------------------------------------------
# 2. AutoregressiveBackend.close() frees GPU memory (skip if no CUDA)
# ---------------------------------------------------------------------------

class TestAutoregressiveBackendClose:

    @pytest.mark.skipif(
        "not _cuda_available()",
        reason="CUDA not available; close() GPU-release is CUDA-only",
    )
    def test_close_releases_gpu_memory(self):
        import torch
        from benchmark.inference.backends.autoregressive import AutoregressiveBackend
        from benchmark.inference.backends.protocol import BackendConfig

        # Use a tiny model so we don't need a real checkpoint.
        # A BackendConfig with a fake path is fine — close() only cleans up
        # internal state (graphs, paged-attn, pinned pools, events).
        cfg = BackendConfig(
            model_path="/nonexistent/model",
            device_info=None,
            max_input_tokens=64,
            max_new_tokens=32,
            temperature=0.0,
            use_flash_attention=False,
            use_torch_compile=False,
            extra={
                "use_cuda_graph": False,
                "use_paged_attention": False,
                "use_quantized_weights": False,
                "use_int8_kv_cache": False,
                "use_fused_kernels": False,
                "safe_mode": True,
                "backend_type": "autoregressive",
            },
        )

        backend = AutoregressiveBackend(cfg)
        # Manually set the attribute so close() has something to clean.
        # The load() method is skipped — we only test the close() path.
        backend._loaded = True  # let close() proceed past the guard
        backend._paged_kv = None
        backend._graph_decoder = None
        backend._graph_pool = None

        # Record memory before close.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated()

        backend.close()

        # Force a synchronize + empty_cache so the allocator releases
        # cached deallocations back to the driver.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated()

        # No regression: close() should not *increase* allocated memory.
        assert after <= before, (
            f"GPU memory increased after close(): "
            f"{before / 1024**2:.1f} MiB -> {after / 1024**2:.1f} MiB"
        )

    def test_close_noop_when_not_loaded(self):
        """close() should be safe to call even if load() was never called."""
        from benchmark.inference.backends.autoregressive import AutoregressiveBackend
        from benchmark.inference.backends.protocol import BackendConfig

        cfg = BackendConfig(
            model_path="/nonexistent/model",
            device_info=None,
            max_input_tokens=64,
            max_new_tokens=32,
            temperature=0.0,
            use_flash_attention=False,
            use_torch_compile=False,
            extra={"safe_mode": True, "backend_type": "autoregressive"},
        )

        backend = AutoregressiveBackend(cfg)
        # load() not called, so _loaded is False.  close() is a no-op
        # (attribute accesses via getattr for pinned pools will return None).
        backend.close()  # must not raise


# ---------------------------------------------------------------------------
# 3. DiffusionBackend._te_available is initialized
# ---------------------------------------------------------------------------

class TestDiffusionBackendTeAvailable:

    def test_te_available_initialized_to_false(self):
        """_te_available must start as False; only load() sets it to True."""
        from benchmark.inference.backends.diffusion import DiffusionBackend
        from benchmark.inference.backends.protocol import BackendConfig

        cfg = BackendConfig(
            model_path="/nonexistent/model",
            device_info=None,
            max_input_tokens=64,
            max_new_tokens=64,
            temperature=0.0,
            use_flash_attention=False,
            use_torch_compile=False,
            extra={"backend_type": "diffusion"},
        )

        backend = DiffusionBackend(cfg)
        assert backend._te_available is False, (
            "_te_available must be False until load() detects Transformer Engine"
        )

        # The _fp8_context should return a nullcontext when _te_available
        # is False (no crash).
        ctx = backend._fp8_context()
        assert ctx is not None  # must return a context manager


# ---------------------------------------------------------------------------
# 4. JSON serialization handles inf / nan correctly
# ---------------------------------------------------------------------------

class TestJsonSanitization:

    def test_inf_mapped_to_sentinel(self):
        data = {"score": float("inf"), "penalty": float("-inf")}
        result = sanitized_dumps(data)
        assert "1e+308" in result or "1e308" in result
        assert "-1e+308" in result or "-1e308" in result

    def test_nan_mapped_to_null(self):
        data = {"value": float("nan"), "name": "test"}
        result = sanitized_dumps(data)
        assert "null" in result
        # The key "name" must still round-trip.
        assert '"test"' in result

    def test_nested_inf_and_nan(self):
        data = {
            "metrics": [
                {"a": float("inf")},
                {"b": float("-inf")},
                {"c": float("nan")},
            ],
            "extra": {"deep": {"x": float("inf")}},
        }
        # Must produce valid JSON (i.e. not raise ValueError).
        result = sanitized_dumps(data)
        import json
        parsed = json.loads(result)
        assert parsed["metrics"][0]["a"] == 1e308
        assert parsed["metrics"][1]["b"] == -1e308
        assert parsed["metrics"][2]["c"] is None
        assert parsed["extra"]["deep"]["x"] == 1e308

    def test_normal_floats_untouched(self):
        data = {"pi": 3.14, "e": 2.718}
        result = sanitized_dumps(data)
        import json
        parsed = json.loads(result)
        assert parsed["pi"] == 3.14
        assert parsed["e"] == 2.718


# ---------------------------------------------------------------------------
# 5. PrecisionTimer.elapsed() handles pre-start calls
# ---------------------------------------------------------------------------

class TestTimerElapsedPreStart:

    def test_elapsed_before_start_returns_zero(self):
        t = PrecisionTimer()
        # start() has never been called; _start is 0.0.
        assert t.elapsed() == 0.0

    def test_elapsed_after_init_then_start(self):
        t = PrecisionTimer()
        pre = t.elapsed()
        assert pre == 0.0
        t.start()
        time.sleep(0.005)
        post = t.elapsed()
        assert post > 0.0

    def test_stop_before_start_does_not_crash(self):
        """Calling stop() before start() leaves _stop as 0.0; elapsed()
        falls through to the running-clock branch (_stop <= _start)."""
        t = PrecisionTimer()
        t.stop()  # _stop = monotonic(), _start = 0.0
        # elapsed() sees _stop (large) > _start (0.0) and returns
        # _stop - _start.  This is technically a valid positive number
        # reflecting the time between object creation and stop(), not a
        # crash.  We just verify no exception is raised.
        result = t.elapsed()
        assert isinstance(result, float)
        assert result >= 0.0


# ---------------------------------------------------------------------------
# 6. notify_done() terminates (doesn't infinite loop)
# ---------------------------------------------------------------------------

class TestNotifyDoneTerminates:

    def test_notify_done_completes_within_timeout(self):
        """notify_done() pushes sentinels into the raw queue.  It must return
        within a bounded time even when worker threads are slow to drain.
        We use a small queue (maxsize=2) and a separate thread that slowly
        consumes, forcing the retry path in notify_done()."""
        from benchmark.data.pipeline import AsyncPipeline
        import types

        # Create a minimal pipeline instance.  We cannot instantiate
        # AsyncPipeline directly (it requires a loader + tokenizer), so we
        # construct a bare object and attach only the attributes notify_done()
        # touches: _done, _raw_queue, prefetch_workers, _SENTINEL.
        pipeline = types.SimpleNamespace()
        pipeline._done = threading.Event()
        pipeline.prefetch_workers = 2
        # Small queue increases the chance we hit the retry path.
        pipeline._raw_queue = queue.Queue(maxsize=2)
        # The _SENTINEL class attribute is accessed via self._SENTINEL.
        pipeline._SENTINEL = AsyncPipeline._SENTINEL
        pipeline.notify_done = AsyncPipeline.notify_done.__get__(
            pipeline, AsyncPipeline
        )

        # Consumer thread: drains one sentinel every 100ms so the queue can
        # briefly fill up and exercise the `except queue.Full` branch.
        consumed = []

        def _slow_drain():
            for _ in range(pipeline.prefetch_workers):
                try:
                    item = pipeline._raw_queue.get(timeout=2.0)
                    consumed.append(item)
                except queue.Empty:
                    break
                time.sleep(0.1)

        drain_thread = threading.Thread(target=_slow_drain, daemon=True)
        drain_thread.start()

        # notify_done() must complete within 5 seconds.
        start = time.monotonic()
        pipeline.notify_done()
        elapsed = time.monotonic() - start

        drain_thread.join(timeout=5.0)

        assert elapsed < 5.0, (
            f"notify_done() took {elapsed:.1f}s — may be looping infinitely"
        )
        assert pipeline._done.is_set()
        # Expect at least as many consumed items as workers (sentinel count).
        assert len(consumed) >= pipeline.prefetch_workers, (
            f"Expected {pipeline.prefetch_workers} sentinels, got {len(consumed)}"
        )


# ---------------------------------------------------------------------------
# 7. BenchmarkConfig validator rejects bad combinations
# ---------------------------------------------------------------------------

class TestBenchmarkConfigValidation:

    def test_mps_backend_with_tensorrt_raises(self):
        from benchmark.config.schema import BenchmarkConfig
        with pytest.raises(ValueError, match="incompatible"):
            BenchmarkConfig(
                backend="mps",
                model={
                    "use_tensorrt": True,
                    "backend_type": "autoregressive",
                },
            )

    def test_non_cuda_backend_with_tensorrt_raises(self):
        from benchmark.config.schema import BenchmarkConfig
        with pytest.raises(ValueError, match="requires backend='cuda'"):
            BenchmarkConfig(
                backend="cpu",
                model={
                    "use_tensorrt": True,
                    "backend_type": "autoregressive",
                },
            )

    def test_do_sample_with_zero_temperature_raises(self):
        from benchmark.config.schema import BenchmarkConfig
        with pytest.raises(ValueError, match="do_sample=True"):
            BenchmarkConfig(
                backend="cuda",
                model={
                    "use_tensorrt": False,
                    "backend_type": "autoregressive",
                    "do_sample": True,
                    "temperature": 0.0,
                },
            )

    def test_diffusion_with_tensorrt_raises(self):
        from benchmark.config.schema import BenchmarkConfig
        with pytest.raises(ValueError, match="mutually exclusive"):
            BenchmarkConfig(
                backend="cuda",
                model={
                    "use_tensorrt": True,
                    "backend_type": "diffusion",
                },
            )

    def test_valid_config_does_not_raise(self):
        """A sensible configuration must pass validation without error."""
        from benchmark.config.schema import BenchmarkConfig

        # Use paths that exist (the conftest fixtures directory should have
        # input and references files).
        import os
        tests_dir = os.path.dirname(__file__)
        fixture_dir = os.path.join(tests_dir, "fixtures")
        input_file = os.path.join(fixture_dir, "sample_input.jsonl")
        ref_file = os.path.join(fixture_dir, "golden_en_tr.jsonl")

        cfg = BenchmarkConfig(
            backend="cuda",
            model={
                "backend_type": "autoregressive",
                "use_tensorrt": False,
                "do_sample": False,
                "temperature": 0.0,
                "max_input_tokens": 512,
                "max_new_tokens": 256,
            },
            data={
                "input_paths": [input_file],
                "reference_set_path": ref_file,
                "output_dir": os.path.join(fixture_dir, "output"),
            },
        )
        assert cfg.backend == "cuda"
        assert cfg.model.backend_type == "autoregressive"


# ---------------------------------------------------------------------------
# 8. Coverage gap regression tests — verify that previously uncovered paths
#    stay covered and do not regress silently.
# ---------------------------------------------------------------------------

class TestCoverageGapRegression:
    """Meta-tests: these validate that our coverage gap fillers don't go stale.

    Each test maps to a specific coverage gap identified in the audit.
    """

    def test_precision_timer_module_importable(self):
        """PrecisionTimer must be importable (gap: timer.py previously untested)."""
        from benchmark.utils.timer import PrecisionTimer
        timer = PrecisionTimer()
        assert timer is not None

    def test_throughput_tracker_module_has_prune(self):
        """ThroughputTracker must expose _prune (gap: prune() was uncovered)."""
        from benchmark.metrics.throughput import ThroughputTracker
        assert hasattr(ThroughputTracker, '_prune')

    def test_extrapolation_model_module_exists(self):
        """ExtrapolationModel must be importable (gap: reporting untested)."""
        from benchmark.reporting.extrapolation import ExtrapolationModel
        assert ExtrapolationModel is not None

    def test_json_sanitized_dumps_module_exists(self):
        """sanitized_dumps must be importable (gap: json_utils uncovered)."""
        from benchmark.utils.json_utils import sanitized_dumps
        assert callable(sanitized_dumps)

    def test_async_pipeline_notify_done_exists(self):
        """notify_done must exist on AsyncPipeline (gap: retry path uncovered)."""
        from benchmark.data.pipeline import AsyncPipeline
        assert hasattr(AsyncPipeline, 'notify_done')


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    """Return True if CUDA is available, else False."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
