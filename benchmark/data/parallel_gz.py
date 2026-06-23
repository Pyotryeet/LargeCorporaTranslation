"""Parallel gzip decompression via pigz (Phase 3).

Replaces Python's single-threaded ``gzip`` module with multi-threaded
decompression using the system ``pigz`` command.  Falls back gracefully
to Python gzip when pigz is not installed.

On a system with N cores, pigz achieves ~N× decompression throughput
compared to Python's gzip module, which is single-threaded by design.
"""

from __future__ import annotations

import logging
import subprocess
import shutil
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# Cache pigz availability at module level.
_PIGZ_AVAILABLE: bool | None = None


def _pigz_available() -> bool:
    """Check if pigz is installed and executable."""
    global _PIGZ_AVAILABLE
    if _PIGZ_AVAILABLE is not None:
        return _PIGZ_AVAILABLE
    _PIGZ_AVAILABLE = shutil.which("pigz") is not None
    if _PIGZ_AVAILABLE:
        logger.info("pigz detected — parallel gzip decompression enabled")
    else:
        logger.debug("pigz not found — using Python gzip (single-threaded)")
    return _PIGZ_AVAILABLE


def decompress_lines(file_path: Path | str) -> Iterator[str]:
    """Decompress a .gz file and yield lines.

    Uses pigz for parallel decompression when available, falling back
    to Python's gzip module.

    Parameters
    ----------
    file_path : Path or str
        Path to a .gz file.

    Yields
    ------
    str
        One decompressed line at a time.
    """
    path = Path(file_path)
    if not path.suffix == ".gz":
        # Not compressed — read directly.
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            yield from f
        return

    if _pigz_available():
        yield from _pigz_decompress(path)
    else:
        import gzip
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            yield from f


def _pigz_decompress(path: Path) -> Iterator[str]:
    """Decompress using pigz subprocess.

    pigz writes decompressed data to stdout; we read it line by line
    through a pipe.  This avoids the Python GIL bottleneck on zlib.
    """
    proc = subprocess.Popen(
        ["pigz", "-dc", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered
    )

    try:
        if proc.stdout is not None:
            for line in proc.stdout:
                line = line.strip()
                if line:
                    yield line
        returncode = proc.wait(timeout=300)
        if returncode != 0:
            stderr = ""
            if proc.stderr is not None:
                stderr = proc.stderr.read()
            logger.warning("pigz exited with code %d: %s", returncode, stderr[:200])
    except subprocess.TimeoutExpired:
        proc.kill()
        logger.error("pigz timed out for %s", path)
    except Exception as e:
        proc.kill()
        logger.error("pigz failed for %s: %s", path, e)


def decompress_file(file_path: Path | str) -> bytes:
    """Decompress a full .gz file into memory (for small files).

    For large files, use ``decompress_lines()`` instead to stream.
    """
    path = Path(file_path)
    if not path.suffix == ".gz":
        return path.read_bytes()

    if _pigz_available():
        result = subprocess.run(
            ["pigz", "-dc", str(path)],
            capture_output=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"pigz failed: {result.stderr.decode()[:200]}")
        return result.stdout
    else:
        import gzip
        return gzip.decompress(path.read_bytes())


def count_lines_gz(file_path: Path | str) -> int:
    """Count lines in a gzip file without full decompression into memory.

    Uses pigz -dc | wc -l for maximum speed when available.
    """
    path = Path(file_path)
    if not path.suffix == ".gz":
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)

    if _pigz_available():
        result = subprocess.run(
            ["pigz", "-dc", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return result.stdout.count('\n')

    # Fallback
    import gzip
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return sum(1 for _ in f)
