"""JSON report writer — produces benchmark_report.json."""

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
    def write(self, output_dir: Path, report: dict) -> Path:
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
