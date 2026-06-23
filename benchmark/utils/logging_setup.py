"""Structured JSON Lines logging setup."""

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path


class JSONFormatter(logging.Formatter):
    def format(self, record):
        now = datetime.now(timezone.utc)
        obj = {"timestamp": now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond//1000:03d}Z",
               "level": record.levelname, "logger": record.name, "thread": record.threadName,
               "module": record.module, "message": record.getMessage()}
        if record.exc_info and record.exc_info[0]:
            obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(obj, ensure_ascii=False)


def setup_logging(run_dir: Path, level: int = logging.INFO) -> None:
    """Configure structured JSON logging to *run_dir/benchmark.log*.

    Creates *run_dir* and any missing parent directories automatically.
    A rotating file handler (10 MB, 3 backups) is paired with a
    human-readable console handler on stdout.

    Parameters
    ----------
    run_dir : Path
        Directory for the log file.  Must be a directory path, not a file.
    level : int
        Log level (default: ``logging.INFO``).
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    # File handler — structured JSON, rotated at 10 MB with 3 backups.
    log_file = run_dir / "benchmark.log"
    try:
        fh = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10_000_000, backupCount=3,
        )
        fh.setLevel(level)
        fh.setFormatter(JSONFormatter())
        root.addHandler(fh)
    except OSError as exc:
        # Fall back to console-only logging if the log file cannot be opened
        # (e.g. read-only filesystem, permission denied, disk full).
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S",
        ))
        root.addHandler(ch)
        root.error("Cannot open log file %s (%s) — logging to stderr only", log_file, exc)
        return
    # Console handler — human-readable
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"))
    root.addHandler(ch)
