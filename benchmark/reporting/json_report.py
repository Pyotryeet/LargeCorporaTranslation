"""JSON report writer.

Produces ``report/benchmark_report.json`` under the given output directory.
Writes are atomic (tempfile + rename) to guard against partial writes.
"""

import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from benchmark.utils.version import VERSION
from benchmark.utils.json_utils import sanitized_dump

logger = logging.getLogger(__name__)


class JSONReportWriter:
    """Atomic JSON report writer.

    Serialises a benchmark result dictionary to a JSON file under
    ``<output_dir>/report/benchmark_report.json``.  Metadata (generation
    timestamp and benchmark version) is injected automatically.  Writes are
    performed atomically via a tempfile + shutil.move, with a copy+delete
    fallback for cross-filesystem moves.
    """

    def write(self, output_dir: Path, report: dict) -> Path:
        """Write the benchmark report to disk atomically.

        Parameters
        ----------
        output_dir : pathlib.Path
            Parent directory under which ``report/benchmark_report.json`` will
            be created.  Intermediate directories are created if needed.
        report : dict
            The benchmark result payload.  A ``_metadata`` key is injected in-
            place with ``generated_at`` (ISO-8601 UTC) and ``benchmark_version``.

        Returns
        -------
        pathlib.Path
            Absolute path to the written JSON file.

        Side effects
        ------------
        - Creates ``<output_dir>/report/`` if it does not exist.
        - Mutates *report* in-place by adding the ``_metadata`` key.
        - Writes to a tempfile first, then atomically renames it to the target
          path.  On cross-filesystem rename failure, falls back to copy+unlink.

        Raises
        ------
        OSError
            If directory creation or file I/O fails.  The tempfile is cleaned up
            on error.
        """
        report_dir = output_dir / "report"
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / "benchmark_report.json"
        report["_metadata"] = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                               "benchmark_version": VERSION}
        fd, tmp_path = tempfile.mkstemp(dir=report_dir, suffix=".json", prefix=".tmp_report_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                sanitized_dump(report, f, indent=2, ensure_ascii=False)
            # Use shutil.move for atomic rename; falls back to copy+delete
            # when tmp and target reside on different filesystems (e.g. /tmp
            # is a ramdisk while output_dir is on a persistent volume).
            try:
                shutil.move(str(tmp_path), str(path))
            except OSError:
                shutil.copy2(str(tmp_path), str(path))
                os.unlink(tmp_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass  # cleanup failure must not mask the original exception
            raise
        logger.info(f"JSON report -> {path}")
        return path
