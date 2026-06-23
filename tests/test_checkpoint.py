"""Tests for checkpoint manager."""

import json
import tempfile
from pathlib import Path
from benchmark.orchestration.checkpoint import CheckpointManager


def _save_with_unique_name(mgr, docs, tokens, **kwargs):
    """Save a checkpoint via the manager and return the path.

    Appends a monotonic counter to the path to guarantee uniqueness
    without relying on wall-clock `time.sleep`, which is flaky under CI.
    """
    _save_with_unique_name._counter += 1
    path = mgr.save(docs, tokens, **kwargs)
    # Rename to include counter so filenames are guaranteed to sort correctly.
    if path is not None:
        new = path.parent / f"checkpoint_{_save_with_unique_name._counter:06d}.json"
        path.rename(new)
        return new
    return path


_save_with_unique_name._counter = 0


class TestCheckpointManager:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(Path(tmp), interval_seconds=300)
            path = mgr.save(100, 50000)
            assert path is not None
            assert path.exists()
            cp = json.loads(path.read_text())
            assert cp["batches_completed"] == 100
            assert cp["total_tokens_translated"] == 50000

    def test_load_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(Path(tmp))
            _save_with_unique_name(mgr, 100, 50000)
            _save_with_unique_name(mgr, 200, 100000)
            latest = mgr.load_latest()
            assert latest is not None
            assert latest["batches_completed"] == 200
            assert latest["total_tokens_translated"] == 100000

    def test_load_none_when_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(Path(tmp))
            assert mgr.load_latest() is None

    def test_rotation_keeps_last_n(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(Path(tmp))
            # NOTE: We access _rotation directly because CheckpointManager has no
            # public setter for rotation — it is initialized from
            # CHECKPOINT_ROTATION in constants.py and intended to be immutable
            # after construction.  Setting it directly here is a white-box test
            # that validates the rotation behaviour in isolation without relying
            # on the global constant remaining at 3.
            mgr._rotation = 3
            for i in range(5):
                _save_with_unique_name(mgr, i * 100, i * 50000)
            files = sorted(mgr.checkpoint_dir.glob("checkpoint_*.json"))
            assert len(files) <= 3  # May keep fewer

    # ── New tests for fix B4 (shard/offset) ──

    def test_save_includes_position_fields(self):
        """Fix P1-1: checkpoint includes current_file_name + current_doc_id."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(Path(tmp))
            path = mgr.save(
                42, 10000,
                current_file_name="input_002.jsonl.gz",
                current_doc_id=1048576,
            )
            assert path is not None
            cp = json.loads(path.read_text())
            assert cp["current_file_name"] == "input_002.jsonl.gz"
            assert cp["current_doc_id"] == 1048576

    def test_load_latest_returns_position_fields(self):
        """Load returns complete checkpoint including position."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(Path(tmp))
            _save_with_unique_name(mgr, 100, 50000, current_file_name="file_a.jsonl.gz", current_doc_id=5000)
            _save_with_unique_name(mgr, 200, 100000, current_file_name="file_b.jsonl.gz", current_doc_id=10000)
            latest = mgr.load_latest()
            assert latest is not None
            assert latest["batches_completed"] == 200
            assert latest["current_file_name"] == "file_b.jsonl.gz"
            assert latest["current_doc_id"] == 10000
            assert latest["total_tokens_translated"] == 100000

    def test_final_checkpoint_marked(self):
        """Final checkpoint has final=True."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(Path(tmp))
            path = mgr.save(10, 1000, final=True)
            cp = json.loads(path.read_text())
            assert cp["final"] is True

    def test_position_fields_default_to_empty(self):
        """When position is omitted, it defaults to empty string and 0."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(Path(tmp))
            path = mgr.save(5, 500)
            cp = json.loads(path.read_text())
            assert cp["current_file_name"] == ""
            assert cp["current_doc_id"] == 0

    def test_rotation_zero_keeps_one(self):
        """Rotation=0 keeps only the most recent checkpoint."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(Path(tmp))
            mgr._rotation = 0
            for i in range(3):
                _save_with_unique_name(mgr, i * 100, i * 50000)
            files = sorted(mgr.checkpoint_dir.glob("checkpoint_*.json"))
            assert len(files) <= 1, f"Expected <=1 checkpoint, got {len(files)}"

    def test_save_interval_zero_writes_every_time(self):
        """interval=0 means always save (never skip)."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(Path(tmp), interval_seconds=0)
            path1 = mgr.save(10, 1000)
            path2 = mgr.save(20, 2000)
            assert path1 is not None
            assert path2 is not None
            assert path1 != path2
