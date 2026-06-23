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
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    # File handler — structured JSON, rotated at 10 MB with 3 backups.
    log_file = run_dir / "benchmark.log"
    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10_000_000, backupCount=3,
    )
    fh.setLevel(level)
    fh.setFormatter(JSONFormatter())
    root.addHandler(fh)
    # Console handler — human-readable
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"))
    root.addHandler(ch)
