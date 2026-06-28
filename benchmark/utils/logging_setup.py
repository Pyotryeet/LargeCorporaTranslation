"""Structured JSON Lines logging setup."""

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """Logging formatter that serializes each :class:`logging.LogRecord` as a single-line JSON object.

    Produces output with the following keys: ``timestamp`` (ISO 8601 with
    millisecond precision in UTC), ``level``, ``logger``, ``thread``,
    ``module``, and ``message``.  If the record carries exception info
    (``exc_info``), an ``exception`` key holding the formatted traceback
    is also included.

    This formatter is intended for machine consumption (e.g. log
    aggregators) and emits one JSON object per line (JSON Lines).
    """

    def format(self, record):
        """Format *record* as a JSON string.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to format.

        Returns
        -------
        str
            A single-line JSON string (no trailing newline).

        Notes
        -----
        - Timestamps are UTC with millisecond precision.
        - Exception tracebacks are included under the ``exception`` key when
          ``record.exc_info`` is truthy.
        - The output uses ``ensure_ascii=False`` so non-ASCII characters are
          preserved as-is rather than escaped.
        """
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
