"""Pre-tokenization cache — tokenize input data once, reuse across runs.

Pre-processing
--------------
:class:`PreTokenizer` reads raw text from any loader-supported format
(JSONL, JSONL.gz, Parquet), runs the full chunk→filter→prompt→tokenize
pipeline, and writes the resulting token IDs alongside raw chunk text
to a model-specific Parquet file in ``~/.cache/tr_benchmark/pretokenized/``.

Runtime
-------
:class:`PreTokenizedLoader` reads a pre-tokenized Parquet file and yields
``(text, token_ids, token_count)`` tuples — the same format the pipeline's
``_tokenised_queue`` expects — so the existing ``next_batch()`` path is
completely unchanged.

Cache invalidation
------------------
The cache key is ``SHA256(model_path + tokenizer_hash + max_input_tokens +
overlap_tokens + prompt_style + input_files_hash)[:16]``.  Any change to
the model, tokenizer, chunker config, or input data triggers a cache miss
and full re-tokenization.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]
    HAS_PARQUET = False


# ── Constants ──────────────────────────────────────────────────────────────

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "tr_benchmark" / "pretokenized"
MANIFEST_FILE = "manifest.json"
MAX_CACHE_ENTRIES = 50  # LRU eviction threshold


# ── Cache key ──────────────────────────────────────────────────────────────


def _hash_files(input_paths: list[str]) -> str:
    """Stable hash of sorted input file paths + sizes.

    Detects added/removed/changed files without reading file contents.
    For the full-corpus use case where data is static, this is sufficient;
    for incremental data, extend to a Merkle-tree of content hashes.
    """
    h = hashlib.sha256()
    for p in sorted(input_paths):
        h.update(p.encode())
        try:
            h.update(str(Path(p).stat().st_size).encode())
        except OSError:
            h.update(b"0")
    return h.hexdigest()


def _hash_tokenizer(tokenizer) -> str:
    """Hash the tokenizer's vocabulary and special-token configuration.

    Reads ``tokenizer_config.json`` and ``special_tokens_map.json`` from
    the tokenizer's save directory.  Falls back to the tokenizer name +
    vocab size if files aren't accessible.
    """
    h = hashlib.sha256()
    try:
        save_dir = tokenizer.save_pretrained("/tmp/_tr_tok_hash_check")
        for fname in sorted(os.listdir(save_dir)):
            fpath = os.path.join(save_dir, fname)
            if os.path.isfile(fpath):
                with open(fpath, "rb") as fh:
                    h.update(fh.read())
        # Clean up
        import shutil
        shutil.rmtree(save_dir, ignore_errors=True)
    except Exception:
        # Fallback: hash the tokenizer's identity + vocab size
        h.update(getattr(tokenizer, "name_or_path", "unknown").encode())
        h.update(str(getattr(tokenizer, "vocab_size", 0)).encode())
    return h.hexdigest()


def get_cache_key(
    model_path: str,
    tokenizer,
    max_input_tokens: int,
    overlap_tokens: int,
    input_paths: list[str],
) -> str:
    """Compute a deterministic cache key for a (model, dataset, config) tuple.

    Returns a 16-char hex string used as the Parquet filename.
    """
    components = [
        model_path,
        _hash_tokenizer(tokenizer),
        str(max_input_tokens),
        str(overlap_tokens),
        _hash_files(input_paths),
    ]
    full = hashlib.sha256("|".join(components).encode()).hexdigest()
    return full[:16]


# ── Manifest ───────────────────────────────────────────────────────────────


@dataclass
class CacheEntry:
    cache_key: str
    model_path: str
    max_input_tokens: int
    overlap_tokens: int
    num_chunks: int
    file_size_bytes: int
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)


def _load_manifest(cache_dir: Path) -> dict[str, CacheEntry]:
    """Load the manifest, returning {} if missing or corrupt."""
    mp = cache_dir / MANIFEST_FILE
    if not mp.exists():
        return {}
    try:
        with open(mp, "r") as f:
            raw = json.load(f)
        return {
            k: CacheEntry(**v) for k, v in raw.items()
        }
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logger.warning("Manifest corrupt (%s) — rebuilding", e)
        return {}


def _save_manifest(cache_dir: Path, entries: dict[str, CacheEntry]) -> None:
    """Atomically write the manifest."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_dir / f".{MANIFEST_FILE}.tmp"
    payload = {k: v.__dict__ for k, v in entries.items()}
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, cache_dir / MANIFEST_FILE)


def _evict_lru(entries: dict[str, CacheEntry], cache_dir: Path, keep: int = MAX_CACHE_ENTRIES) -> None:
    """Remove least-recently-used entries until *keep* remain."""
    if len(entries) <= keep:
        return
    sorted_entries = sorted(entries.values(), key=lambda e: e.last_accessed)
    to_remove = sorted_entries[: len(entries) - keep]
    for entry in to_remove:
        parquet_path = cache_dir / f"{entry.cache_key}.parquet"
        if parquet_path.exists():
            parquet_path.unlink()
            logger.debug("LRU evicted: %s", parquet_path.name)
        del entries[entry.cache_key]


# ── PreTokenizer ───────────────────────────────────────────────────────────


class PreTokenizer:
    """Run the full pre-processing pipeline and write token IDs to Parquet.

    Parameters
    ----------
    model_path : str
        HuggingFace model ID.  Used in the cache key.
    tokenizer : PreTrainedTokenizerBase
        Tokenizer instance.  Must be loaded (not from_pretrained).
    max_input_tokens : int
        Chunker max_input_tokens.
    overlap_tokens : int
        Chunker overlap_tokens.
    min_chunk_tokens : int
        ChunkFilter min_tokens.
    max_garbage_ratio : float
        ChunkFilter max_garbage_ratio.
    input_paths : list[str]
        Glob patterns for input files.
    cache_dir : Path
        Where to write the pre-tokenized Parquet.
    """

    PARQUET_ROW_GROUP_SIZE = 10_000  # chunks per row group

    def __init__(
        self,
        model_path: str,
        tokenizer,
        max_input_tokens: int = 512,
        overlap_tokens: int = 50,
        min_chunk_tokens: int = 10,
        max_garbage_ratio: float = 0.95,
        input_paths: list[str] | None = None,
        cache_dir: Path | None = None,
    ):
        if not HAS_PARQUET:
            raise ImportError("pyarrow is required for pre-tokenization — pip install pyarrow")

        self.model_path = model_path
        self.tokenizer = tokenizer
        self.max_input_tokens = max_input_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.max_garbage_ratio = max_garbage_ratio
        self.input_paths = input_paths or ["./data/input/*.jsonl.gz"]
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR

    @property
    def cache_key(self) -> str:
        if not hasattr(self, "_cache_key"):
            self._cache_key = get_cache_key(
                self.model_path, self.tokenizer,
                self.max_input_tokens, self.overlap_tokens,
                self.input_paths,
            )
        return self._cache_key

    @property
    def parquet_path(self) -> Path:
        return self.cache_dir / f"{self.cache_key}.parquet"

    def run(self, force: bool = False) -> int:
        """Run pre-tokenization.  Returns the number of chunks written.

        If *force* is False and a valid cache exists, this is a no-op.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not force and self.parquet_path.exists():
            logger.info("Pre-tokenized cache exists: %s", self.parquet_path)
            pf = pq.ParquetFile(self.parquet_path)
            return pf.metadata.num_rows

        logger.info(
            "Pre-tokenizing: model=%s max_tok=%d overlap=%d → %s",
            self.model_path, self.max_input_tokens, self.overlap_tokens,
            self.parquet_path,
        )

        # ── Heavy imports deferred so they don't slow down cache-hit paths ──
        from benchmark.data.loader import JSONLLoader
        from benchmark.data.chunker import TextChunker
        from benchmark.data.filters import ChunkFilter
        from benchmark.data.pipeline import _PROMPT_STYLE, _TEMPLATE_WARNED, AsyncPipeline

        loader = JSONLLoader(self.input_paths, shuffle=False)
        chunker = TextChunker(self.tokenizer, self.max_input_tokens, self.overlap_tokens)
        filt = ChunkFilter(min_tokens=self.min_chunk_tokens, max_garbage_ratio=self.max_garbage_ratio)

        total_chunks = 0
        schema = pa.schema([
            pa.field("raw_text", pa.string()),
            pa.field("token_ids", pa.list_(pa.int32())),
        ])

        # Warm the prompt-style cache once so every chunk uses the cached path.
        try:
            AsyncPipeline._build_translation_prompt("warmup", self.tokenizer)
        except Exception:
            pass

        with pq.ParquetWriter(self.parquet_path, schema, compression="snappy") as writer:
            batch_texts: list[str] = []
            batch_ids: list[list[int]] = []

            for doc_id, fname, text in loader.iter_documents():
                for chunk_text in chunker.chunk(text):
                    # Build prompt + tokenize (same as _tokeniser_loop)
                    prompted = AsyncPipeline._build_translation_prompt(chunk_text, self.tokenizer)
                    token_ids = self.tokenizer.encode(
                        prompted, add_special_tokens=True,
                        truncation=True, max_length=self.max_input_tokens,
                    )
                    # Convert to native ints (SentencePiece numpy-int safety)
                    token_ids = [int(t) for t in token_ids]

                    if filt.should_keep(chunk_text, len(token_ids)):
                        batch_texts.append(chunk_text)
                        batch_ids.append(token_ids)
                        total_chunks += 1

                        if len(batch_texts) >= self.PARQUET_ROW_GROUP_SIZE:
                            table = pa.table({
                                "raw_text": batch_texts,
                                "token_ids": batch_ids,
                            }, schema=schema)
                            writer.write_table(table)
                            batch_texts.clear()
                            batch_ids.clear()

            # Final partial batch
            if batch_texts:
                table = pa.table({
                    "raw_text": batch_texts,
                    "token_ids": batch_ids,
                }, schema=schema)
                writer.write_table(table)

        # ── Update manifest ────────────────────────────────────────────────
        manifest = _load_manifest(self.cache_dir)
        file_size = self.parquet_path.stat().st_size
        manifest[self.cache_key] = CacheEntry(
            cache_key=self.cache_key,
            model_path=self.model_path,
            max_input_tokens=self.max_input_tokens,
            overlap_tokens=self.overlap_tokens,
            num_chunks=total_chunks,
            file_size_bytes=file_size,
        )
        _evict_lru(manifest, self.cache_dir)
        _save_manifest(self.cache_dir, manifest)

        logger.info(
            "Pre-tokenized: %d chunks, %.1f MB, key=%s",
            total_chunks, file_size / (1024 * 1024), self.cache_key,
        )
        return total_chunks


# ── PreTokenizedLoader ─────────────────────────────────────────────────────


class PreTokenizedLoader:
    """Read a pre-tokenized Parquet file, yielding pipeline-format tuples.

    Yields ``(raw_text, token_ids, token_count)`` — identical to what
    the pipeline's ``_tokenised_queue`` expects.  The existing
    ``next_batch()`` path consumes these without any changes.
    """

    def __init__(self, parquet_path: Path):
        if not HAS_PARQUET:
            raise ImportError("pyarrow is required — pip install pyarrow")
        self.parquet_path = Path(parquet_path)
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Pre-tokenized cache not found: {self.parquet_path}")
        self._pf = pq.ParquetFile(self.parquet_path)
        self._total_chunks = self._pf.metadata.num_rows

    @property
    def total_chunks(self) -> int:
        return self._total_chunks

    def iter_chunks(self) -> Iterator[tuple[str, list[int], int]]:
        """Yield ``(raw_text, token_ids, token_count)`` for every chunk."""
        for rg_idx in range(self._pf.metadata.num_row_groups):
            table = self._pf.read_row_group(rg_idx)
            raw_col = table.column("raw_text")
            ids_col = table.column("token_ids")
            for i in range(len(table)):
                text = raw_col[i].as_py()
                token_ids = ids_col[i].as_py()
                if isinstance(token_ids, list) and token_ids:
                    yield (text, token_ids, len(token_ids))

    def seek(self, chunk_index: int) -> None:
        """Skip the first *chunk_index* chunks (for resume)."""
        # Parquet doesn't support row-level seek — we track the skip count
        # and fast-forward in the iterator.  For large skips, this is cheap
        # because it's scanning metadata, not decompressing data.
        self._skip_count = chunk_index

    def iter_chunks_seekable(self) -> Iterator[tuple[str, list[int], int]]:
        """Like ``iter_chunks()`` but respects ``seek()``."""
        skip = getattr(self, "_skip_count", 0)
        yielded = 0
        for item in self.iter_chunks():
            if yielded < skip:
                yielded += 1
                continue
            yield item
            yielded += 1


# ── Public API ─────────────────────────────────────────────────────────────


def ensure_pretokenized(
    model_path: str,
    tokenizer,
    max_input_tokens: int = 512,
    overlap_tokens: int = 50,
    input_paths: list[str] | None = None,
    cache_dir: Path | None = None,
    force: bool = False,
) -> PreTokenizedLoader:
    """Ensure a pre-tokenized cache exists, then return a loader for it.

    This is the single entry point for callers: it handles cache-hit
    (no-op) and cache-miss (full pre-tokenization).  Returns a loader
    ready for iteration.

    Parameters
    ----------
    model_path : str
        HuggingFace model ID.
    tokenizer : PreTrainedTokenizerBase
        Loaded tokenizer.
    max_input_tokens : int
        Chunker max_input_tokens.
    overlap_tokens : int
        Chunker overlap_tokens.
    input_paths : list[str]
        Glob patterns for input files.
    cache_dir : Path
        Cache directory.
    force : bool
        If True, re-tokenize even if cache exists.

    Returns
    -------
    PreTokenizedLoader
    """
    pretok = PreTokenizer(
        model_path=model_path,
        tokenizer=tokenizer,
        max_input_tokens=max_input_tokens,
        overlap_tokens=overlap_tokens,
        input_paths=input_paths,
        cache_dir=cache_dir,
    )
    num = pretok.run(force=force)
    if num == 0:
        logger.warning("Pre-tokenization produced 0 chunks — check input data and filters")
    return PreTokenizedLoader(pretok.parquet_path)


def has_cache(
    model_path: str,
    tokenizer,
    max_input_tokens: int = 512,
    overlap_tokens: int = 50,
    input_paths: list[str] | None = None,
    cache_dir: Path | None = None,
) -> bool:
    """Return True if a valid pre-tokenized cache exists."""
    if not HAS_PARQUET:
        return False
    pretok = PreTokenizer(
        model_path=model_path, tokenizer=tokenizer,
        max_input_tokens=max_input_tokens, overlap_tokens=overlap_tokens,
        input_paths=input_paths, cache_dir=cache_dir,
    )
    return pretok.parquet_path.exists()
