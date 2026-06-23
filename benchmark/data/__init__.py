"""Data pipeline — streaming JSONL loader, chunking, filtering, async prefetch.

v2.0 additions: parallel gzip decompression, memory-mapped I/O, orjson parsing.
"""

from benchmark.data.loader import JSONLLoader
from benchmark.data.chunker import TextChunker, NullChunker
from benchmark.data.filters import ChunkFilter, FilterStats
from benchmark.data.pipeline import AsyncPipeline, PipelineBatch
from benchmark.data import parallel_gz  # public API: decompress_lines, count_lines_gz, etc.

__all__ = [
    "JSONLLoader", "TextChunker", "NullChunker",
    "ChunkFilter", "FilterStats", "AsyncPipeline", "PipelineBatch",
    "parallel_gz",
]
