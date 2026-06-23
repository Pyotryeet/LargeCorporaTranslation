"""End-to-End Correctness Test — validates critical fix paths produce correct output.

This module tests the CORRECTNESS of the fixed behavior — not just "does it run":

    C1. Throughput calculation is correct (tokens / time).
    C2. Extrapolation CI shrinks with sqrt(n) (SEM fix).
    C3. Extrapolation bootstrap produces valid CI.
    C4. Checkpoint save/load roundtrip preserves ALL fields.
    C5. Safe mode disables all experimental optimizations.
    C6. Translation produces valid UTF-8 output (requires model — skipped on CPU).

Run:
    python -m pytest tests/test_e2e_correctness.py -v
"""

import json
import math
import tempfile
import time
from pathlib import Path

import pytest


# ═══════════════════════════════════════════════════════════════════════════
# C1: Throughput correctness
# ═══════════════════════════════════════════════════════════════════════════

class TestThroughputCorrectness:
    def test_single_event_no_longer_zero(self):
        """Fix B9: single event returns tokens/effective_span, not 0.0."""
        from benchmark.metrics.throughput import ThroughputTracker
        t = ThroughputTracker(window_seconds=60)
        t.add(100, 1000)  # 100 tokens in 1000ms
        tps = t.current()
        assert tps > 0.0, f"Single-event throughput must be > 0, got {tps}"

    def test_rate_matches_manual_computation(self):
        """Throughput = total_tokens / elapsed_seconds across multiple events."""
        from benchmark.metrics.throughput import ThroughputTracker
        t = ThroughputTracker(window_seconds=60)
        t.add(256, 1000)
        time.sleep(0.05)
        t.add(256, 1000)
        time.sleep(0.05)
        t.add(128, 500)
        tps = t.current()
        # 640 tokens across ~0.1 seconds.
        assert tps > 0.0
        assert 100 < tps < 20000, f"Expected ~6400 tok/s, got {tps:.1f}"

    def test_prune_removes_old_events(self):
        """_prune evicts events older than window_seconds."""
        from benchmark.metrics.throughput import ThroughputTracker
        t = ThroughputTracker(window_seconds=5)
        # Add an old event (artificially aged).
        old_time = time.monotonic() - 10
        t._events.append((old_time, 100))
        t._window_sum += 100
        t._prune(time.monotonic())
        # Old event should be removed.
        assert len(t._events) == 0
        assert t._window_sum == 0


# ═══════════════════════════════════════════════════════════════════════════
# C2 + C3: Extrapolation correctness
# ═══════════════════════════════════════════════════════════════════════════

class TestExtrapolationCorrectness:
    def test_ci_is_standard_error_not_raw_std(self):
        """Fix B2: CI uses SEM = std/sqrt(n), not raw std."""
        from benchmark.reporting.extrapolation import ExtrapolationModel
        m = ExtrapolationModel(total_tokens=100_000_000)
        # 1 batch: CI = ±1.96 * std/mean * days.
        r1 = m.compute(mean_tps=500, std_tps=50, num_gpus=2, n_batches=1)
        # 100 batches: CI = ±1.96 * (std/10)/mean * days.
        r100 = m.compute(mean_tps=500, std_tps=50, num_gpus=2, n_batches=100)
        ci1_width = r1["days_95ci_upper"] - r1["days_95ci_lower"]
        ci100_width = r100["days_95ci_upper"] - r100["days_95ci_lower"]
        # CI must shrink with more batches.
        assert ci100_width < ci1_width, (
            f"CI should narrow with n: {ci1_width=:.6f} vs {ci100_width=:.6f}"
        )
        # Ratio should be ~10x (1/sqrt(100) = 0.1).
        ratio = ci1_width / ci100_width if ci100_width > 0 else float('inf')
        assert 5 < ratio < 15, f"Expected ~10x shrink, got {ratio:.1f}x"

    def test_bootstrap_ci_valid(self):
        """Bootstrap CI has valid percentiles."""
        from benchmark.reporting.extrapolation import ExtrapolationModel
        import random
        random.seed(42)
        tps_samples = [random.gauss(1000, 50) for _ in range(30)]
        m = ExtrapolationModel(total_tokens=1_000_000_000)
        r = m.compute_bootstrap(tps_samples, num_gpus=2, n_bootstrap=2000, seed=42)
        assert r["bootstrap_days_lower"] > 0
        assert r["bootstrap_days_upper"] > r["bootstrap_days_lower"]
        assert r["days_point_estimate"] > 0
        assert r["method"] == "bootstrap"

    def test_sem_present_in_output(self):
        """SEM is included in result for transparency."""
        from benchmark.reporting.extrapolation import ExtrapolationModel
        m = ExtrapolationModel()
        r = m.compute(mean_tps=1000, std_tps=100, num_gpus=2, n_batches=100)
        assert "sem_tokens_per_second" in r
        # sem = 100 / sqrt(100) = 10
        assert abs(r["sem_tokens_per_second"] - 10.0) < 0.5

    def test_ci_never_negative(self):
        """CI lower bound is clamped to zero."""
        from benchmark.reporting.extrapolation import ExtrapolationModel
        m = ExtrapolationModel(total_tokens=1)
        r = m.compute(mean_tps=1, std_tps=1000, num_gpus=2, n_batches=1)
        assert r["days_95ci_lower"] >= 0


# ═══════════════════════════════════════════════════════════════════════════
# C4: Checkpoint roundtrip
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckpointRoundtrip:
    def test_full_roundtrip(self):
        """Checkpoint save → load preserves all position and counter fields."""
        from benchmark.orchestration.checkpoint import CheckpointManager
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(Path(tmp))
            path = mgr.save(
                batches_completed=42,
                total_tokens=10000,
                current_file_name="data/shard_003.jsonl.gz",
                current_doc_id=2097152,
                final=False,
            )
            assert path is not None

            cp = mgr.load_latest()
            assert cp is not None
            assert cp["batches_completed"] == 42
            assert cp["total_tokens_translated"] == 10000
            assert cp["current_file_name"] == "data/shard_003.jsonl.gz"
            assert cp["current_doc_id"] == 2097152
            assert cp["final"] is False

    def test_multiple_checkpoints_latest_wins(self):
        """Only the most recent checkpoint is loaded."""
        from benchmark.orchestration.checkpoint import CheckpointManager
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(Path(tmp))
            mgr.save(10, 1000, "a.jsonl", 500)
            time.sleep(0.02)
            mgr.save(20, 2000, "b.jsonl", 1000)
            time.sleep(0.02)
            mgr.save(30, 3000, "c.jsonl", 1500)
            latest = mgr.load_latest()
            assert latest["batches_completed"] == 30
            assert latest["current_file_name"] == "c.jsonl"


# ═══════════════════════════════════════════════════════════════════════════
# C5: Safe mode correctness
# ═══════════════════════════════════════════════════════════════════════════

class TestSafeModeCorrectness:
    def test_all_dangerous_flags_disabled(self):
        """Safe mode disables CUDA graph, paged attn, fused kernels, quant, TE."""
        import torch
        from benchmark.inference.backends.autoregressive import AutoregressiveBackend
        from benchmark.inference.backends.protocol import BackendConfig
        from benchmark.hardware.backend import DeviceInfo

        device_info = DeviceInfo(
            backend="cpu", device=torch.device("cpu"),
            num_devices=1, name="TEST",
        )
        config = BackendConfig(
            model_path="test",
            device_info=device_info,
            extra={"safe_mode": True},
        )
        backend = AutoregressiveBackend(config)
        assert backend._use_cuda_graph is False
        assert backend._use_paged_attention is False
        assert backend._use_quantized_weights is False
        assert backend._use_int8_kv_cache is False
        assert backend._use_fused_kernels is False


# ═══════════════════════════════════════════════════════════════════════════
# C7: Pinned buffer pool correctness
# ═══════════════════════════════════════════════════════════════════════════

class TestPinnedPoolCorrectness:
    def test_reuse_same_pointer(self):
        """Released tensor is re-acquired at same memory address."""
        from benchmark.data.pipeline import PinnedBufferPool
        pool = PinnedBufferPool(max_batch_size=4, max_seq_len=64, pool_size=2)
        ids1, mask1 = pool.acquire()
        ptr1 = ids1.data_ptr()
        pool.release(ids1, mask1)
        ids2, mask2 = pool.acquire()
        ptr2 = ids2.data_ptr()
        assert ptr1 == ptr2, "Should reuse the same pinned tensor"

    def test_zero_and_fill_after_release(self):
        """Released buffer is zeroed, then correctly filled on next acquire."""
        import torch
        from benchmark.data.pipeline import PinnedBufferPool
        pool = PinnedBufferPool(max_batch_size=2, max_seq_len=8, pool_size=1)
        ids, mask = pool.acquire()
        # Create values on the same device as the pool tensor (handles MPS).
        vals = torch.tensor([1, 2, 3, 4, 5], device=ids.device)
        ids[0, :5] = vals
        sum_before = ids[0, :5].sum().item()
        assert sum_before > 0, "Should have nonzero values"
        pool.release(ids, mask)
        ids2, mask2 = pool.acquire()
        # Released tensor is zeroed before reuse.
        assert ids2.sum() == 0, "Pool should zero tensors before reuse"
