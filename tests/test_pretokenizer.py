"""Focused tests for pretokenization cache invalidation."""

from __future__ import annotations

import os
from pathlib import Path

from benchmark.data.pretokenizer import _hash_files, get_cache_key


class _DummyTokenizer:
    name_or_path = "dummy-tokenizer"
    vocab_size = 32

    def save_pretrained(self, save_dir: str):
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "tokenizer_config.json").write_text('{"dummy": true}', encoding="utf-8")
        (out / "special_tokens_map.json").write_text('{"eos_token": "</s>"}', encoding="utf-8")
        return str(out)


def test_hash_files_tracks_glob_membership(tmp_path):
    inp = tmp_path / "input"
    inp.mkdir()
    (inp / "a.jsonl").write_text('{"text": "one"}\n', encoding="utf-8")

    pattern = str(inp / "*.jsonl")
    before = _hash_files([pattern])

    (inp / "b.jsonl").write_text('{"text": "two"}\n', encoding="utf-8")
    after = _hash_files([pattern])

    assert before != after


def test_hash_files_tracks_file_metadata_changes(tmp_path):
    file_path = tmp_path / "sample.jsonl"
    file_path.write_text('{"text": "one"}\n', encoding="utf-8")

    before = _hash_files([str(file_path)])

    file_path.write_text('{"text": "two"}\n', encoding="utf-8")
    stat = file_path.stat()
    os.utime(file_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    after = _hash_files([str(file_path)])

    assert before != after


def test_cache_key_includes_filter_settings(tmp_path):
    inp = tmp_path / "sample.jsonl"
    inp.write_text('{"text": "hello"}\n', encoding="utf-8")

    tokenizer = _DummyTokenizer()
    base = get_cache_key(
        model_path="google/translategemma-4b-it",
        tokenizer=tokenizer,
        max_input_tokens=512,
        overlap_tokens=50,
        min_chunk_tokens=10,
        max_garbage_ratio=0.95,
        input_paths=[str(inp)],
    )
    changed_filter = get_cache_key(
        model_path="google/translategemma-4b-it",
        tokenizer=tokenizer,
        max_input_tokens=512,
        overlap_tokens=50,
        min_chunk_tokens=20,
        max_garbage_ratio=0.95,
        input_paths=[str(inp)],
    )
    changed_ratio = get_cache_key(
        model_path="google/translategemma-4b-it",
        tokenizer=tokenizer,
        max_input_tokens=512,
        overlap_tokens=50,
        min_chunk_tokens=10,
        max_garbage_ratio=0.80,
        input_paths=[str(inp)],
    )

    assert base != changed_filter
    assert base != changed_ratio
