"""Streaming document reader with glob support, deterministic shuffle, and
multi-format input (JSONL, JSONL.gz, Parquet).

For datasets that fit within the configured memory budget (~2 GiB by default):
an in-memory Fisher-Yates shuffle.  Larger datasets use a disk-backed external
sort with binary-record k-way merge.

Uses ``orjson`` for 4–10× faster JSON parsing and ``pyarrow`` for
row-group-streamed Parquet reads.  Falls back gracefully to stdlib json and
plain-text iteration when optional dependencies are absent.
"""

from contextlib import contextmanager
from glob import glob
import gzip
import heapq
import json as _stdlib_json  # for JSONDecodeError in catch clauses
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
    HAS_ORJSON = True
except ImportError:
    import json as _json
    HAS_ORJSON = False

try:
    import pyarrow.parquet as _pq
    HAS_PARQUET = True
except ImportError:
    _pq = None  # type: ignore[assignment]
    HAS_PARQUET = False

logger = logging.getLogger(__name__)

# Module-level sentinel for object-identity checks (avoids magic-string bugs).
_SENTINEL_OBJ = object()

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
    """Parse a single JSON line — uses orjson when available, stdlib fallback.

    orjson is 4-10x faster but strict: it rejects NaN, Infinity, trailing
    commas, and some Unicode edge cases.  When orjson fails we fall back to
    stdlib json.loads, which is more lenient.  Both orjson.JSONDecodeError
    (a ValueError subclass) and generic ValueError are caught to handle
    orjson versions that raise TypeError or ValueError for malformed input.

    Raises
    ------
    json.JSONDecodeError
        If both orjson and stdlib json fail to parse the line.
    """
    if HAS_ORJSON:
        try:
            return orjson.loads(line)
        except (orjson.JSONDecodeError, ValueError):
            # orjson is strict — fall back to stdlib json for lines that
            # orjson rejects (e.g., NaN, Infinity, trailing commas, or
            # malformed UTF-8 that raises TypeError in older orjson).
            return _stdlib_json.loads(line)
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

        .. note::

           The ``doc_id`` yielded by ``iter_documents()`` after a seek is
           the iteration-sequence position (0-based after skip), NOT the
           original corpus document index.  When shuffle is enabled, this
           means doc_id values are permutation-relative and will differ
           from the unshuffled doc_id sequence.  Callers that need
           reproducible doc_id values should record the checkpoint
           ``current_position`` tuple before pausing.
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

    def _ensure_counts(self) -> tuple[int, int]:
        """Single-pass document count and text byte estimate, with caching.

        Reads each file once (line counting), then estimates decompressed
        text size from the compressed byte count.  Both fields are cached
        so subsequent calls are free.  All other count-related methods
        delegate here to prevent redundant file reads.

        Notes
        -----
        The text byte estimate uses a 4:1 compression-ratio heuristic for
        .gz files.  This is intentionally conservative: typical English JSONL
        with short text fields compresses at 3–5:1, while highly redundant
        JSONL (long repeated keys) can reach 10:1.  ``_should_use_external_sort``
        applies a safety floor to prevent OOM from underestimated sizes.
        """
        if self._doc_count is not None and self._total_text_bytes is not None:
            return self._doc_count, self._total_text_bytes

        doc_count = 0
        for fp in self._files:
            if self._is_parquet(fp):
                doc_count += self._count_parquet_rows(fp)
            else:
                with self._open(fp) as fh:
                    doc_count += sum(1 for _ in fh)

        # Parquet uses snappy/zstd compression (typically 2–5× for text).
        # JSONL.gz uses 3–5× for English text.  We use a uniform 3× multiplier
        # for compressed formats — conservative enough to not underestimate.
        _compressed = (
            sum(fp.stat().st_size for fp in self._files if fp.suffix in ('.gz', '.parquet'))
        )
        uncompressed_bytes = sum(
            fp.stat().st_size for fp in self._files if fp.suffix not in ('.gz', '.parquet')
        )
        estimated_text_bytes = _compressed * 3 + uncompressed_bytes

        self._doc_count = doc_count
        self._total_text_bytes = estimated_text_bytes
        return doc_count, estimated_text_bytes

    def _count_documents(self) -> int:
        """Return cached document count (delegates to _ensure_counts for single-pass)."""
        return self._ensure_counts()[0]

    def _count_and_estimate_bytes(self) -> tuple[int, int]:
        """Return (doc_count, estimated_text_bytes) — delegates to _ensure_counts."""
        return self._ensure_counts()

    def _should_use_external_sort(self, doc_count: int, total_text_bytes: int) -> bool:
        """Return True if the dataset exceeds the in-memory budget.

        Applies a safety cap to the heuristic text byte estimate — extremely
        compressible files (10:1+) produce artificially low estimates that
        can cause OOM when the actual decompressed text is much larger than
        predicted.  The cap prevents the 4:1 heuristic from underestimating
        by more than a configurable factor.
        """
        if doc_count == 0:
            return False
        if doc_count > MAX_IN_MEMORY_DOCS:
            return True
        estimated_memory = int(total_text_bytes * SHUFFLE_BYTES_PER_CHAR_OVERHEAD)
        # Safety cap: if the heuristic estimate is suspiciously low relative
        # to the number of documents, bump it.  A typical English JSONL doc
        # is ~500-2000 chars, so < 50 chars/doc suggests the heuristic is
        # badly underestimating (e.g., highly compressed metadata-heavy JSONL).
        if doc_count > 0 and estimated_memory < doc_count * 50:
            logger.warning(
                "Heuristic memory estimate (%.2f MB) is < 50 chars/doc — "
                "likely underestimating. Using doc-count-based fallback.",
                estimated_memory / (1024**2),
            )
            estimated_memory = doc_count * SHUFFLE_BYTES_PER_CHAR_OVERHEAD * 500
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
            # NOTE: Falling back to _sequential_iter() yields doc_id values
            # that start from the seek-skip offset rather than the shuffled
            # permutation indices.  doc_id is a monotonic sequence identifier,
            # not a canonical document index, so this is safe — but callers
            # that compare doc_id across shuffle paths may see inconsistencies.
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
            "Shuffle loaded %d docs (avg text len=%.0f chars, "
            "total chars=%d) — "
            "estimated memory ~%.2f MB",
            records_loaded,
            avg_text_len,
            total_text_len,
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

    #: Default column name for text data in Parquet files.
    PARQUET_TEXT_COLUMN = "text"

    @staticmethod
    def _is_parquet(file_path: Path) -> bool:
        """Check suffix AND magic bytes so we never misidentify a file."""
        if file_path.suffix not in (".parquet", ".arrow"):
            return False
        if not HAS_PARQUET:
            return False
        # Verify magic bytes: Parquet files start with "PAR1".
        try:
            with open(file_path, "rb") as probe:
                magic = probe.read(4)
            return magic == b"PAR1"
        except OSError:
            return False

    def _read_parquet(self, file_path: Path) -> Iterator[str]:
        """Read a Parquet file, yielding text from the configured column.

        Uses pyarrow's row-group streaming so memory usage is proportional to
        one row group, not the full file.
        """
        pf = _pq.ParquetFile(file_path)
        col_name = self.PARQUET_TEXT_COLUMN
        for rg_idx in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg_idx, columns=[col_name])
            col = table.column(col_name)
            for i in range(len(col)):
                val = col[i].as_py()
                if isinstance(val, str) and val.strip():
                    yield val

    def _stream_parquet(self, file_path: Path) -> Iterator[str]:
        """Same as ``_read_parquet`` — Parquet streaming is already row-grouped."""
        yield from self._read_parquet(file_path)

    def _count_parquet_rows(self, file_path: Path) -> int:
        """Fast row count from Parquet metadata — no data scan needed."""
        pf = _pq.ParquetFile(file_path)
        return pf.metadata.num_rows

    def _read_file(self, file_path: Path) -> Iterator[str]:
        """Read and parse one file, yielding text fields.
        Dispatches to Parquet reader when file is in Parquet format.

        Materialises all lines into a list first — used by the sequential
        and in-memory shuffle paths where per-file buffering is acceptable.
        For the external sort path, use ``_stream_file`` instead.
        """
        if self._is_parquet(file_path):
            yield from self._read_parquet(file_path)
            return

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
                except (_stdlib_json.JSONDecodeError, ValueError):
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
        """Yield text from one file — no full-file buffer.

        Dispatches to Parquet reader when the file is in Parquet format.
        Used by the external sort path and the byte-estimation pre-pass.
        """
        if self._is_parquet(file_path):
            yield from self._stream_parquet(file_path)
            return

        skipped = 0
        skipped_first_lines: list[str] = []  # capture first few malformed lines for diagnosis
        with self._open(file_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = _parse_json_line(line)
                except (_stdlib_json.JSONDecodeError, ValueError) as e:
                    # orjson raises orjson.JSONDecodeError (a subclass of
                    # ValueError), and _parse_json_line converts orjson failures
                    # to stdlib JSONDecodeError.  Catch both so we skip truly
                    # malformed lines but let fatal errors (MemoryError,
                    # KeyboardInterrupt, etc.) propagate.
                    skipped += 1
                    if len(skipped_first_lines) < 3:
                        skipped_first_lines.append(line[:120])
                    continue
                text = obj.get("text", "")
                if text:
                    yield text
        if skipped:
            sample = "; ".join(repr(s) for s in skipped_first_lines)
            logger.warning(
                "Skipped %d malformed JSON lines in %s (first %d: [%s])",
                skipped, file_path.name, len(skipped_first_lines), sample,
            )

    @staticmethod
    @contextmanager
    def _open(file_path: Path):
        """Open a file for line reading — uses pigz for .gz when available."""
        path_str = str(file_path)
        if path_str.endswith(".gz"):
            # Try parallel_gz (pigz) first — 3-8x faster than Python gzip on
            # multi-core machines.  Falls back to stdlib gzip when pigz is not
            # installed or the module is unavailable.
            try:
                from benchmark.data.parallel_gz import decompress_lines as _pgz_lines
                # Wrap the iterator so it behaves like a file object for
                # `for line in fh` consumers (context manager protocol).
                class _PgzWrapper:
                    def __init__(self, lines):
                        self._lines = lines
                    def __iter__(self):
                        return self
                    def __next__(self):
                        return next(self._lines)
                    def __enter__(self):
                        return self
                    def __exit__(self, *args):
                        pass
                yield _PgzWrapper(_pgz_lines(file_path))
                return
            except ImportError:
                pass
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
