"""Checkpoint manager — persists progress for crash recovery."""

import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from benchmark.config.constants import CHECKPOINT_ROTATION

logger = logging.getLogger(__name__)


class CheckpointManager:
    def __init__(self, run_dir: Path, interval_seconds: int = 300):
        self.run_dir = run_dir
        self.checkpoint_dir = run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.interval_seconds = interval_seconds
        self._last_elapsed = 0.0
        self._rotation = CHECKPOINT_ROTATION
        self._lock = threading.Lock()

    def save(self, batches_completed: int, total_tokens: int,
             current_file_name: str = "", current_doc_id: int = 0,
             elapsed_seconds: float = 0.0, final: bool = False) -> Path | None:
        """Persist current progress as a checkpoint file.

        Notes on correctness:
        - os.rename is atomic on local ext4/xfs filesystems, so a reader
          will never observe a partially-written checkpoint. This guarantee
          does NOT hold on NFS or S3 mounts — on those backends the rename
          is not atomic and the file may appear truncated or empty.
        - The checkpoint does NOT include a checksum. Corruption introduced
          by bit-rot, a faulty drive, or a bad intermediate copy will not be
          detected automatically. External validation (e.g. a separate
          manifest with SHA-256 hashes) is required if integrity matters.
        - json.dumps can raise MemoryError or be interrupted partway through
          on very large state objects, leaving the .tmp file incomplete.
        - On SIGKILL during write, the .tmp file may be partially written.
          The rename guards against this for SIGTERM (since the signal
          handler can defer the signal until write+rename complete), but
          a SIGKILL cannot be caught and will expose the stale .tmp on the
          next run.
        """
        now = datetime.now(timezone.utc)
        checkpoint = {"version": 1,
                      "checkpoint_time": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                      "batches_completed": batches_completed,
                      "total_tokens_translated": total_tokens,
                      "current_file_name": current_file_name,
                      "current_doc_id": current_doc_id,
                      "elapsed_seconds": elapsed_seconds,
                      "final": final}
        tmp_path = self.checkpoint_dir / "checkpoint.tmp"
        try:
            with self._lock:
                with open(tmp_path, "w") as f:
                    json.dump(checkpoint, f)
                    # json.dump writes to kernel buffer; fsync ensures data
                    # reaches disk so we don't lose it on crash.
                    f.flush()
                    os.fsync(f.fileno())
                # os.rename is atomic on ext4/xfs but NOT on NFS — data may
                # be lost on network filesystems.
                ts = now.strftime("%Y%m%d_%H%M%S")
                final_path = self.checkpoint_dir / f"checkpoint_{ts}.json"
                os.rename(str(tmp_path), str(final_path))
                # For final checkpoints fsync the directory fd so the rename
                # is durable even on filesystems with lazy directory updates.
                if final:
                    dir_fd = os.open(self.checkpoint_dir, os.O_RDONLY)
                    try:
                        os.fsync(dir_fd)
                    finally:
                        os.close(dir_fd)
                logger.debug(f"Checkpoint saved: {final_path.name}")
                self._rotate()
            return final_path
        except OSError as e:
            logger.warning(f"Checkpoint save failed: {e}")
            return None

    def load_latest(self) -> dict | None:
        files = sorted(self.checkpoint_dir.glob("checkpoint_*.json"))
        if not files:
            logger.info("No checkpoint files found in %s", self.checkpoint_dir)
            return None

        # Try checkpoints newest-first until one loads successfully.
        last_error = None
        for f in reversed(files):
            try:
                with open(f) as fh:
                    cp = json.load(fh)
            except (json.JSONDecodeError, OSError) as e:
                last_error = e
                logger.warning("Checkpoint %s corrupt, trying older: %s", f.name, e)
                continue

            # Version check — warn on missing or mismatched version.
            version = cp.get("version")
            if version is None:
                logger.warning(
                    "Checkpoint %s has no version field — "
                    "may be from a different version of the benchmark",
                    f.name,
                )
            elif version != 1:
                logger.warning(
                    "Checkpoint %s has version=%s but expected version=1 — "
                    "fields may have changed",
                    f.name, version,
                )

            logger.info("Loaded checkpoint: %s", f.name)
            return cp

        logger.warning(
            "All %d checkpoint files in %s are corrupt — "
            "starting from scratch",
            len(files), self.checkpoint_dir,
        )
        return None

    def _rotate(self) -> None:
        files = sorted(self.checkpoint_dir.glob("checkpoint_*.json"))
        while len(files) > self._rotation:
            try:
                files[0].unlink()
                files.pop(0)
            except OSError:
                logger.warning(
                    "Failed to rotate checkpoint %s — skipping rotation for this save",
                    files[0],
                )
                break
