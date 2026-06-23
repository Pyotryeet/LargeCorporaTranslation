"""Streaming JSONL reader with glob support and deterministic shuffle.

For datasets that fit within the configured memory budget (~2 GiB by default):
an in-memory Fisher-Yates shuffle that reads each file once, stores texts,
then yields in shuffled order.

For larger datasets: a disk-backed external sort (Phase A: key generation +
sorted-run creation; Phase B: k-way merge; Phase C: temp-file cleanup).
The external sort is deterministic — same seed + same input = same output
order.

Uses ``orjson`` for 4–10× faster JSON parsing compared to stdlib ``json``.
Falls back gracefully to stdlib if orjson is not installed.
"""

from contextlib import contextmanager
from glob import glob
import gzip
import heapq
import logging
import os
import random
import struct
import tempfile
import time
from pathlib import Path
from typing import Iterator, Optional

try:
    import orjson
# Module-level state: safe for single-process use. Not thread-safe for multi-harness scenarios.
    HAS_ORJSON = True
except ImportError:
    import json as _json
    HAS_ORJSON = False

logger = logging.getLogger(__name__)

# ── External shuffle constants (imported from central constants) ──────────
from benchmark.config.constants import (  # noqa: E402
    SHUFFLE_MEMORY_BUDGET_BYTES,
    SHUFFLE_MAX_OPEN_RUNS,
    SHUFFLE_BYTES_PER_CHAR_OVERHEAD,
    MAX_IN_MEMORY_DOCS,
)

#: Poll every N records during shuffle load to check for timeout.
_SHUFFLE_LOAD_POLL_INTERVAL = 500

#: Maximum wall-clock time (seconds) for shuffle load before falling back
#: to sequential iteration.
_SHUFFLE_LOAD_TIMEOUT = 15


def _parse_json_line(line: str) -> dict:
    """Parse a single JSON line — uses orjson when available, stdlib fallback."""
    if HAS_ORJSON:
        try:
            return orjson.loads(line)
        except orjson.JSONDecodeError:
            raise
    else:
        return _json.loads(line)


class JSONLLoader:
    """Streams English text from gzip-compressed JSONL files.

    Each line must contain a JSON object with at minimum a ``"text"`` field.
    Deterministic shuffle uses a Fisher-Yates permutation seeded from
    ``runtime.seed`` so two runs with identical config produce identical order.

    For datasets exceeding the memory budget (2 GiB by default), the shuffle
    switches to a disk-backed external sort that never holds more than the
    budget in RAM at once.
    """

    def __init__(
        self,
        input_patterns: list[str],
        shuffle: bool = True,
        seed: int = 42,
        max_shuffle_memory_gb: Optional[float] = None,
        shuffle_temp_dir: str = "",
    ):
        self.input_patterns = input_patterns
        self.shuffle = shuffle
        self.seed = seed
        self._files: list[Path] = []
        self._total_bytes: int = 0
        self._doc_count: Optional[int] = None
        # Cached total text bytes from _count_and_estimate_bytes pre-pass.
        self._total_text_bytes: Optional[int] = None

        # External shuffle configuration.
        self._max_shuffle_memory_bytes = (
            int(max_shuffle_memory_gb * 1024**3)
            if max_shuffle_memory_gb is not None
            else SHUFFLE_MEMORY_BUDGET_BYTES
        )
        self._shuffle_temp_dir = shuffle_temp_dir

        # Resume support: track current position for checkpointing.
        self._current_file: str = ""
        self._current_doc_id: int = 0
        self._seek_skip_docs: int = 0

        self._resolve_files()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _resolve_files(self) -> None:
        for pattern in self.input_patterns:
            for m in sorted(glob(pattern, recursive=True)):
                p = Path(m)
                if p.is_file():
                    self._files.append(p)
                    self._total_bytes += p.stat().st_size
        if not self._files:
            logger.warning(
                "No files matched patterns: %s — pipeline will drain immediately",
                self.input_patterns,
            )
            # Don't crash — let the pipeline drain gracefully.
            # This allows dry-run / smoke tests on machines without data.
        logger.info(
            "Found %d input file(s), %.2f GB total (orjson=%s)",
            len(self._files),
            self._total_bytes / (1024**3),
            HAS_ORJSON,
        )

    def seek_to(self, docs_processed: int) -> None:
        """Set a resume position — skip the first *docs_processed* documents.

        The next call to ``iter_documents()`` will skip the first
        *docs_processed* documents, resuming from doc_id = docs_processed.
        """
        self._seek_skip_docs = docs_processed
        self._current_doc_id = docs_processed
        logger.info(
            "Seek set: skipping %d documents (resume from doc_id=%d)",
            docs_processed, docs_processed,
        )

    @property
    def current_position(self) -> tuple[str, int]:
        """Return (current_file_name, current_doc_id) for checkpointing."""
        return self._current_file, self._current_doc_id

    # ------------------------------------------------------------------
    # Public iterator
    # ------------------------------------------------------------------

    def iter_documents(self) -> Iterator[tuple[int, str, str]]:
        """Yield ``(doc_id, file_name, text)`` tuples.

        When *shuffle* is enabled the order is a deterministic permutation
        of all documents (seeded from ``self.seed``).

        WARNING: This is a generator. Calling it twice on the same loader
        instance returns zero documents on the second call. Re-create the
        loader to re-iterate.
        """
        if self.shuffle:
            yield from self._shuffled_iter()
        else:
            yield from self._sequential_iter()

    def _sequential_iter(self) -> Iterator[tuple[int, str, str]]:
        doc_id = 0
        skip_remaining = self._seek_skip_docs

        for file_path in self._files:
            self._current_file = str(file_path)
            fname = file_path.name

            for text in self._read_file(file_path):
                if skip_remaining > 0:
                    doc_id += 1
                    skip_remaining -= 1
                    continue
                self._current_doc_id = doc_id
                yield doc_id, fname, text
                doc_id += 1

        # Reset seek so subsequent calls don't skip again.
        self._seek_skip_docs = 0

    # ------------------------------------------------------------------
    # Shuffled iterator — delegates to in-memory or external sort
    # ------------------------------------------------------------------

    def _count_documents(self) -> int:
        """Fast count pass — lines only, no JSON parsing."""
        if self._doc_count is not None:
            return self._doc_count
        total = 0
        for fp in self._files:
            with self._open(fp) as fh:
                total += sum(1 for _ in fh)
        self._doc_count = total
        return total

    def _count_and_estimate_bytes(self) -> tuple[int, int]:
        """Estimate document count and text size without parsing JSON.

        Uses line-counting (no allocation) plus compressed-size heuristic
        to avoid the O(n) JSON-parse pre-pass that fragments CPython's
        memory arenas on multi-million-document datasets.
        """
        if self._doc_count is not None and self._total_text_bytes is not None:
            return self._doc_count, self._total_text_bytes

        doc_count = 0
        for fp in self._files:
            with self._open(fp) as fh:
                doc_count += sum(1 for _ in fh)

        # Estimate decompressed text size from compressed size.
        # JSONL with English text typically compresses 3–5:1 with gzip.
        # Conservative: assume 4:1 ratio → total_text_bytes ≈ compressed_bytes × 4.
        # Also cap per-document average to avoid overestimating from metadata.
        compressed_bytes = sum(fp.stat().st_size for fp in self._files if fp.suffix == '.gz')
        uncompressed_bytes = sum(fp.stat().st_size for fp in self._files if fp.suffix != '.gz')
        estimated_text_bytes = compressed_bytes * 4 + uncompressed_bytes

        self._doc_count = doc_count
        self._total_text_bytes = estimated_text_bytes
        return doc_count, estimated_text_bytes

    def _should_use_external_sort(self, doc_count: int, total_text_bytes: int) -> bool:
        """Return True if the dataset exceeds the in-memory budget."""
        if doc_count == 0:
            return False
        if doc_count > MAX_IN_MEMORY_DOCS:
            return True
        estimated_memory = int(total_text_bytes * SHUFFLE_BYTES_PER_CHAR_OVERHEAD)
        return estimated_memory > self._max_shuffle_memory_bytes

    def _shuffled_iter(self) -> Iterator[tuple[int, str, str]]:
        """Dispatch to in-memory shuffle or external sort based on dataset size."""
        doc_count, total_text_bytes = self._count_and_estimate_bytes()

        if self._should_use_external_sort(doc_count, total_text_bytes):
            logger.info(
                "Dataset: %d docs, ~%.2f GB text — using external sort "
                "(budget: %.2f GB)",
                doc_count,
                total_text_bytes / (1024**3),
                self._max_shuffle_memory_bytes / (1024**3),
            )
            yield from self._external_shuffle_iter()
            return

        # Fast path: dataset fits in memory.
        try:
            yield from self._in_memory_shuffle()
        except MemoryError:
            logger.warning(
                "In-memory shuffle hit MemoryError — falling back to external sort"
            )
            # Reset the exhausted iterator state so external sort can re-read.
            self._doc_count = None
            self._total_text_bytes = None
            yield from self._external_shuffle_iter()

    # ── In-memory Fisher-Yates shuffle (fast path) ───────────────────────

    def _in_memory_shuffle(self) -> Iterator[tuple[int, str, str]]:
        """In-memory Fisher-Yates shuffle — the original fast path."""
        total = self._doc_count if self._doc_count is not None else self._count_documents()
        if total > MAX_IN_MEMORY_DOCS:
            logger.error(
                "Document count (%d) exceeds in-memory shuffle limit (%d). "
                "Falling back to sequential order. Set shuffle=false in config "
                "to suppress this error.",
                total,
                MAX_IN_MEMORY_DOCS,
            )
            yield from self._sequential_iter()
            return

        logger.info("Loading %d documents for shuffle...", total)

        # Single pass: read every file once, store (file_name, text).
        # Uses _stream_file (line-by-line, no full-file buffer) instead of
        # _read_file (which does lines=list(f)) to avoid materialising the
        # raw JSON strings alongside the extracted texts.  For 100K docs
        # this saves ~300 MB; for multi-million-document datasets it saves
        # tens of GB of transient memory.
        rng = random.Random(self.seed)
        records: list[tuple[str, str]] = []

        # Time-bounded load: if loading takes >_SHUFFLE_LOAD_TIMEOUT s, switch to sequential.
        load_start = time.monotonic()
        records_loaded = 0
        total_text_len = 0
        for file_path in self._files:
            for text in self._stream_file(file_path):
                records.append((file_path.name, text))
                records_loaded += 1
                total_text_len += len(text)
                # Check timeout every _SHUFFLE_LOAD_POLL_INTERVAL records (gzip is slow, we need tighter polling).
                if records_loaded % _SHUFFLE_LOAD_POLL_INTERVAL == 0 and time.monotonic() - load_start > _SHUFFLE_LOAD_TIMEOUT:
                    logger.warning(
                        "Shuffle load >%ds (%d of %d docs loaded) "
                        "— falling back to sequential.",
                        _SHUFFLE_LOAD_TIMEOUT, records_loaded, total,
                    )
                    yield from self._sequential_iter()
                    return

        # Estimate memory: text bytes + file name strings + list/str overhead.
        avg_text_len = total_text_len / records_loaded if records_loaded else 0
        estimated_memory_bytes = int(total_text_len * SHUFFLE_BYTES_PER_CHAR_OVERHEAD)
        logger.info(
            "Shuffle loaded %d docs (avg text len=%.0f chars) — "
            "estimated memory ~%.2f MB",
            records_loaded,
            avg_text_len,
            estimated_memory_bytes / (1024**2),
        )
        if estimated_memory_bytes > 1024**3:
            logger.warning(
                "Shuffle memory estimate %.2f GB exceeds 1 GB; "
                "consider reducing dataset size or enabling external sort",
                estimated_memory_bytes / (1024**3),
            )

        # Fisher-Yates shuffle in-place (deterministic via seed)
        for i in range(len(records) - 1, 0, -1):
            j = rng.randint(0, i)
            records[i], records[j] = records[j], records[i]

        skip = self._seek_skip_docs
        for doc_id, (file_name, text) in enumerate(records):
            if skip > 0:
                skip -= 1
                continue
            self._current_doc_id = doc_id
            yield doc_id, file_name, text

        self._seek_skip_docs = 0
        logger.info("Shuffled %d documents", len(records))

    # ── External sort (disk-backed) ──────────────────────────────────────

    # Binary record format (big-endian):
    #   Offset  Size  Type     Field
    #   ------  ----  ----     -----
    #   0       8     uint64   key
    #   8       8     uint64   doc_id
    #   16      2     uint16   file_name length (bytes)
    #   18      4     uint32   text length (bytes)
    #   22      N     bytes    file_name (UTF-8)
    #   22+N    M     bytes    text (UTF-8)
    _RUN_RECORD_HEADER = struct.Struct(">QQHI")  # key, doc_id, fname_len, text_len

    @staticmethod
    def _write_run_record(fh, key: int, doc_id: int, file_name: str, text: str) -> None:
        """Write one record to a binary run file."""
        fname_bytes = file_name.encode("utf-8")
        text_bytes = text.encode("utf-8")
        fh.write(JSONLLoader._RUN_RECORD_HEADER.pack(
            key, doc_id, len(fname_bytes), len(text_bytes),
        ))
        fh.write(fname_bytes)
        fh.write(text_bytes)

    @staticmethod
    def _read_run_record(fh) -> Optional[tuple[int, int, str, str]]:
        """Read one record from a binary run file.  Returns None at EOF."""
        header = fh.read(22)  # Q(8) + Q(8) + H(2) + I(4)
        if not header:
            return None
        if len(header) < 22:
            logger.warning("Truncated run file header at position %d", fh.tell())
            return None

        key, doc_id, fname_len, text_len = JSONLLoader._RUN_RECORD_HEADER.unpack(header)
        fname_bytes = fh.read(fname_len)
        text_bytes = fh.read(text_len)

        if len(fname_bytes) < fname_len or len(text_bytes) < text_len:
            logger.warning("Truncated run file record at position %d", fh.tell())
            return None

        file_name = fname_bytes.decode("utf-8", errors="replace")
        text = text_bytes.decode("utf-8", errors="replace")
        return key, doc_id, file_name, text

    def _flush_run(
        self,
        buffer: list[tuple[int, int, str, str]],
        tmp_root: Path,
        run_files: list[Path],
    ) -> None:
        """Sort *buffer* by key, serialise to a temp run file, clear buffer."""
        buffer.sort(key=lambda x: x[0])

        fd, tmp_path_str = tempfile.mkstemp(
            dir=str(tmp_root), prefix="shuffle_run_", suffix=".bin",
        )
        tmp_path = Path(tmp_path_str)
        try:
            with os.fdopen(fd, "wb") as f:
                for key, doc_id, file_name, text in buffer:
                    self._write_run_record(f, key, doc_id, file_name, text)
        except Exception:
            # The with-statement's __exit__ already closed fd via the file
            # object — do NOT os.close(fd) here (that would be a double-
            # close on a potentially-reused fd).  Just unlink the output.
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

        run_files.append(tmp_path)

    def _kway_merge(
        self,
        run_files: list[Path],
    ) -> Iterator[tuple[int, str, str]]:
        """K-way merge of sorted run files via min-heap.

        If *run_files* exceeds ``SHUFFLE_MAX_OPEN_RUNS``, delegates to
        ``_multi_pass_merge`` to stay within file-descriptor limits.
        """
        if len(run_files) > SHUFFLE_MAX_OPEN_RUNS:
            yield from self._multi_pass_merge(run_files)
            return

        file_handles: list = []
        heap: list[tuple[int, int, tuple[int, str, str]]] = []

        try:
            for i, path in enumerate(run_files):
                fh = open(path, "rb")
                file_handles.append(fh)
                record = self._read_run_record(fh)
                if record is not None:
                    key, doc_id, file_name, text = record
                    heapq.heappush(heap, (key, i, (doc_id, file_name, text)))

            while heap:
                key, file_idx, (doc_id, file_name, text) = heapq.heappop(heap)
                yield doc_id, file_name, text

                fh = file_handles[file_idx]
                record = self._read_run_record(fh)
                if record is not None:
                    next_key, next_doc_id, next_fname, next_text = record
                    heapq.heappush(
                        heap,
                        (next_key, file_idx, (next_doc_id, next_fname, next_text)),
                    )
        finally:
            for fh in file_handles:
                try:
                    fh.close()
                except OSError:
                    pass

    def _merge_chunk_to_temp(self, chunk: list[Path]) -> Path:
        """Merge a single chunk of run files into one new temp file."""
        tmp_root = chunk[0].parent
        fd, tmp_path_str = tempfile.mkstemp(
            dir=str(tmp_root), prefix="shuffle_merge_", suffix=".bin",
        )
        tmp_path = Path(tmp_path_str)

        try:
            with os.fdopen(fd, "wb") as outf:
                file_handles = [open(p, "rb") for p in chunk]
                heap: list[tuple[int, int, tuple[int, str, str]]] = []
                try:
                    for i, fh in enumerate(file_handles):
                        record = self._read_run_record(fh)
                        if record is not None:
                            key, doc_id, file_name, text = record
                            heapq.heappush(heap, (key, i, (doc_id, file_name, text)))

                    while heap:
                        key, file_idx, (doc_id, file_name, text) = heapq.heappop(heap)
                        self._write_run_record(outf, key, doc_id, file_name, text)

                        fh = file_handles[file_idx]
                        record = self._read_run_record(fh)
                        if record is not None:
                            nk, nd, nf, nt = record
                            heapq.heappush(heap, (nk, file_idx, (nd, nf, nt)))
                finally:
                    for fh in file_handles:
                        fh.close()
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

        return tmp_path

    def _multi_pass_merge(
        self,
        run_files: list[Path],
    ) -> Iterator[tuple[int, str, str]]:
        """Merge run files in batches to stay within file-descriptor limits."""
        current_files = list(run_files)

        while len(current_files) > SHUFFLE_MAX_OPEN_RUNS:
            logger.info(
                "Multi-pass merge: %d files → batches of %d",
                len(current_files), SHUFFLE_MAX_OPEN_RUNS,
            )
            new_files: list[Path] = []
            for chunk_start in range(0, len(current_files), SHUFFLE_MAX_OPEN_RUNS):
                chunk = current_files[chunk_start:chunk_start + SHUFFLE_MAX_OPEN_RUNS]
                merged = self._merge_chunk_to_temp(chunk)
                new_files.append(merged)

            # Delete intermediate files from this round.
            for f in current_files:
                try:
                    f.unlink(missing_ok=True)
                except OSError:
                    pass
            current_files = new_files

        # Final merge of the (now ≤ SHUFFLE_MAX_OPEN_RUNS) files.
        yield from self._kway_merge(current_files)

        # Clean up remaining temp files from the final round.
        for f in current_files:
            try:
                f.unlink(missing_ok=True)
            except OSError:
                pass

    def _external_shuffle_iter(self) -> Iterator[tuple[int, str, str]]:
        """Disk-backed external sort shuffle.

        Phase A: Stream documents, assign random keys, buffer by memory
                 budget, flush sorted runs to temp files.
        Phase B: K-way merge sorted runs via min-heap, yield documents.
        Phase C: Clean up temp files (in ``finally`` block).
        """
        doc_count, total_text_bytes = self._count_and_estimate_bytes()
        if doc_count == 0:
            return

        logger.info(
            "External shuffle: %d documents, ~%.2f GB text, budget %.2f GB",
            doc_count,
            total_text_bytes / (1024**3),
            self._max_shuffle_memory_bytes / (1024**3),
        )

        # Determine temp directory.
        if self._shuffle_temp_dir:
            tmp_root = Path(self._shuffle_temp_dir)
        else:
            tmp_root = Path(tempfile.gettempdir()) / "tr_benchmark_shuffle"
        tmp_root.mkdir(parents=True, exist_ok=True)

        run_files: list[Path] = []

        try:
            # ── Phase A: Key generation + sorted-run creation ──
            # All documents get keys and go into runs — seek/skip is
            # handled during Phase B (the k-way merge output) so that
            # the key sequence stays deterministic regardless of seek.
            rng = random.Random(self.seed)
            buffer: list[tuple[int, int, str, str]] = []  # (key, doc_id, file_name, text)
            buffer_bytes = 0
            doc_id = 0

            for file_path in self._files:
                self._current_file = str(file_path)
                fname = file_path.name
                for text in self._stream_file(file_path):
                    # Deterministic 64-bit shuffle key — consumed for every
                    # document so the key sequence is identical with or
                    # without seek.
                    key = rng.getrandbits(64)

                    # Estimate memory for this entry.
                    text_bytes = len(text) + len(fname) + 8 + 8  # key + doc_id
                    buffer.append((key, doc_id, fname, text))
                    buffer_bytes += int(text_bytes * SHUFFLE_BYTES_PER_CHAR_OVERHEAD)
                    doc_id += 1

                    if buffer_bytes >= self._max_shuffle_memory_bytes:
                        self._flush_run(buffer, tmp_root, run_files)
                        buffer.clear()
                        buffer_bytes = 0

            # Flush final partial buffer.
            if buffer:
                self._flush_run(buffer, tmp_root, run_files)
                buffer.clear()

            logger.info(
                "External shuffle Phase A: %d run files created",
                len(run_files),
            )

            # ── Phase B: K-way merge (with optional seek skip) ──
            skip_remaining = self._seek_skip_docs
            for doc_id, file_name, text in self._kway_merge(run_files):
                if skip_remaining > 0:
                    skip_remaining -= 1
                    continue
                self._current_doc_id = doc_id
                yield doc_id, file_name, text

            self._seek_skip_docs = 0

        finally:
            # ── Phase C: Cleanup ──
            for rf in run_files:
                try:
                    rf.unlink(missing_ok=True)
                except OSError:
                    pass
            try:
                tmp_root.rmdir()  # only removes if empty
            except OSError:
                pass

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _read_file(self, file_path: Path) -> Iterator[str]:
        """Read and parse one file, yielding text fields.

        Materialises all lines into a list first — used by the sequential
        and in-memory shuffle paths where per-file buffering is acceptable.
        For the external sort path, use ``_stream_file`` instead.
        """
        with self._open(file_path) as f:
            lines = list(f)
            total_lines = len(lines)
            is_last_file = (file_path == self._files[-1])
            for idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = _parse_json_line(line)
                except Exception:
                    if is_last_file and idx == total_lines - 1:
                        logger.warning(
                            "Failed to parse LAST line of last file %s — "
                            "this may indicate a truncated download.",
                            file_path.name,
                        )
                    else:
                        logger.warning("Skipping malformed JSON in %s", file_path.name)
                    continue
                text = obj.get("text", "")
                if text:
                    yield text

    def _stream_file(self, file_path: Path) -> Iterator[str]:
        """Yield text from one file line-by-line — no full-file buffer.

        Used by the external sort path and the byte-estimation pre-pass
        to avoid materialising entire files in memory.  Parse errors are
        counted and summarised at the end rather than logged per-line.
        """
        skipped = 0
        with self._open(file_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = _parse_json_line(line)
                except Exception:
                    skipped += 1
                    continue
                text = obj.get("text", "")
                if text:
                    yield text
        if skipped:
            logger.warning(
                "Skipped %d malformed JSON lines in %s",
                skipped, file_path.name,
            )

    @staticmethod
    @contextmanager
    def _open(file_path: Path):
        path_str = str(file_path)
        if path_str.endswith(".gz"):
            with gzip.open(path_str, "rt", encoding="utf-8", errors="replace") as fh:
                yield fh
        else:
            with open(path_str, "r", encoding="utf-8", errors="replace") as fh:
                yield fh

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def file_count(self) -> int:
        return len(self._files)

    @property
    def total_size_bytes(self) -> int:
        return self._total_bytes
