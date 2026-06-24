# Pre-Tokenization Implementation Plan

**Status: ✅ IMPLEMENTED (2026-06-24)**

**Goal:** Pre-process input text into token IDs once, then skip chunking + tokenization on every subsequent benchmark run.

**Why:** The pipeline currently does `text → tokenize full doc → slice → decode chunks → re-tokenize chunks → prompt wrap → pad → GPU` on every run. Steps 1-5 are pure CPU waste — identical output every time. Pre-tokenization stores the post-chunk/post-filter token IDs, reducing runtime to `read token IDs → pad → GPU`.

---

## 1. Cost analysis (measured on 200 FineWeb docs, TranslateGemma 4B tokenizer)

```
  chunk (tokenize→slice→decode):   0.4 ms/chunk
  re-tokenize (encode chunk):       0.3 ms/chunk
  prompt wrapping:                  free (cached after first call)
  ─────────────────────────────────────────
  Total CPU per chunk:            ~0.7 ms

  Projected for 100K docs (~150K chunks):  ~105 seconds CPU saved per run
  Projected for 1M docs:                   ~17 minutes CPU saved per run
  Projected for 6.23T tokens (full corpus): ~hours of CPU saved per run
```

At current scale (100K docs) the saving is modest. At production scale or when running 12 different models, it becomes substantial.

---

## 2. File format

### Schema (Parquet)

```
pretokenized.parquet:
  ├── chunk_token_ids: list<int32>[]   — token IDs for each chunk (post-filter, post-prompt)
  ├── chunk_lengths:    int32[]        — number of valid tokens per chunk
  ├── raw_text:         string[]       — original chunk text (for quality eval / reference)
  ├── doc_id:           int32[]        — source document identifier (for resume)
  ├── model_hash:       string         — SHA256(model_path + tokenizer config) for cache invalidation
  └── metadata:
       ├── model_path:      string     — "google/translategemma-4b-it"
       ├── tokenizer_hash:  string     — hash of tokenizer_config.json + special_tokens_map.json
       ├── max_input_tokens: int32     — 512
       ├── prompt_style:    string     — "chat" | "nllb" | "madlad" | "plain"
       ├── source_files:    string[]   — which input files were processed
       └── created_at:      timestamp
```

### Cache key

```
cache_key = SHA256(
    model_path +
    tokenizer_hash +       # catches vocab changes
    str(max_input_tokens) + # catches chunk size changes
    prompt_style +         # catches template changes
    file_hash              # catches input data changes (hash of sorted input file paths + sizes)
)[:16]
```

### Cache location

```
~/.cache/tr_benchmark/pretokenized/
  ├── a3f8b2c1_4b_512.parquet          # TranslateGemma 4B, 512 tok chunks
  ├── d7e12f9a_4b_1024.parquet         # TranslateGemma 4B, 1024 tok chunks
  ├── f1c9a83b_nllb600m_512.parquet    # NLLB 600M
  └── manifest.json                    # cache metadata + LRU tracking
```

---

## 3. Architecture

### New files

```
benchmark/data/pretokenizer.py     (~200 lines)
  PreTokenizer        — reads input → chunks → tokenizes → writes Parquet
  PreTokenizedLoader  — reads pre-tokenized Parquet → yields PipelineBatch-ready tensors
  get_cache_key()     — computes the cache key for a model + dataset combination
  is_cache_valid()    — checks manifest for existing, up-to-date cache
```

### Modified files

```
benchmark/data/pipeline.py        (~20 lines changed)
  AsyncPipeline.__init__()        — accept optional pretokenized_path
  AsyncPipeline._loader_loop()   — fast path: skip chunker+tokenizer when pre-tokenized
  
benchmark/orchestration/harness.py  (~10 lines changed)
  BenchmarkHarness._setup()       — check cache, log if miss
  
benchmark/__main__.py             (~5 lines changed)
  --pretokenize flag              — run pre-processing only, then exit
  --pretokenized-cache DIR        — override cache directory
```

### No changes needed

- `loader.py` — Parquet reader already exists
- `chunker.py` — used during pre-processing, unchanged
- `filters.py` — used during pre-processing, unchanged
- `backends/*.py` — consume the same PipelineBatch format
- `engine.py` — unchanged

---

## 4. Data flow

### Phase A — Pre-processing (run once per model)

```
                     ┌──────────┐
  JSONLLoader ──────→│  text    │
  (Parquet or JSONL) └────┬─────┘
                          │
                     ┌────▼─────┐
  TextChunker ──────→│ chunks   │  (tokenize → slice → decode)
                     └────┬─────┘
                          │
                     ┌────▼─────┐
  ChunkFilter ──────→│ filtered │  (min_tokens, max_garbage)
                     └────┬─────┘
                          │
                     ┌────▼─────┐
  _build_prompt ────→│ prompted │  (chat template / NLLB prefix)
                     └────┬─────┘
                          │
                     ┌────▼─────┐
  Tokenizer.encode ─→│ token IDs│  (add_special_tokens=True, truncation=True)
                     └────┬─────┘
                          │
                     ┌────▼─────┐
  Parquet writer ───→│ .parquet │  → ~/.cache/tr_benchmark/pretokenized/<key>.parquet
                     └──────────┘
```

### Phase B — Runtime (every benchmark run)

```
                     ┌──────────────┐
  PreTokenizedLoader │ token_ids[]  │  (direct read, zero CPU processing)
  ──────────────────→│ lengths[]    │
                     │ raw_text[]   │
                     └──────┬───────┘
                            │
                     ┌──────▼───────┐
  AsyncPipeline       │ pad + stack  │  (same as current, but skipping
  (fast path)         │ → Pipeline   │   _loader_loop + _tokeniser_loop)
                      │    Batch     │
                     └──────┬───────┘
                            │
                     ┌──────▼───────┐
  Engine.translate()  │ GPU forward  │  (completely unchanged)
                     └──────────────┘
```

---

## 5. Pipeline fast-path design

Current `AsyncPipeline` flow:
```
_loader_loop:         loader.iter_documents() → chunker.chunk() → put raw_queue
_tokeniser_loop (×4): raw_queue.get() → prompt → tokenize → filter → put tokenised_queue
next_batch():         tokenised_queue.get() × batch_size → pad → PipelineBatch
```

Pre-tokenized fast path:
```
_pre_tokenized_loop:  pretokenized_loader.iter_chunks() → put tokenised_queue  (skip chunk, prompt, tokenize, filter)
next_batch():         unchanged
```

The `_pre_tokenized_loop` is a single thread (no tokenizer contention) that reads token IDs directly from Parquet and pushes `(raw_text, token_ids, token_count)` tuples onto the tokenised queue. The existing `next_batch()` pads and stacks them identically.

**Key design choice:** We keep the tokenised_queue + next_batch padding path. This minimizes changes — only the producer side changes, the consumer side is untouched.

---

## 6. Cache invalidation

Regenerate when any of these change:
- Model version (new fine-tune)
- Tokenizer config (vocab changes, special tokens)
- `max_input_tokens` setting
- Chat template (prompt wrapping changes)
- Input files (new/deleted files, different data)
- Chunker overlap setting

Detected via SHA256 hash of all the above, compared against the manifest at startup.

---

## 7. CLI integration

```bash
# Pre-tokenize once (writes to ~/.cache/tr_benchmark/pretokenized/)
python -m benchmark --pretokenize --model translategemma-4b-bf16

# Pre-tokenize for all models in the preset registry
python -m benchmark --pretokenize --all-models

# Benchmark with pre-tokenized cache (automatic detection)
python -m benchmark --config config.yaml
# → "Using pre-tokenized cache: a3f8b2c1 (150,234 chunks, 72 MB)"

# Force re-tokenization (ignore cache)
python -m benchmark --config config.yaml --no-pretokenized-cache

# Warm the cache for a specific set of models
python -m benchmark --pretokenize --model translategemma-4b-bf16 --model nllb-600m
```

---

## 8. Implementation order

### Step 1 — `PreTokenizer` (~2 hours)
- Takes a model path + input paths + chunker config
- Runs the full chunk→filter→prompt→tokenize pipeline
- Writes token IDs to a Parquet file with metadata
- Test: run against FineWeb sample, verify token IDs match what the pipeline would produce

### Step 2 — `PreTokenizedLoader` (~1 hour)
- Reads a pre-tokenized Parquet file
- `iter_chunks()` yields `(token_ids, length, raw_text)` tuples
- Handles resume (seeks to chunk index)
- Test: round-trip — write with PreTokenizer, read with PreTokenizedLoader, verify identical

### Step 3 — Pipeline fast path (~1 hour)
- `AsyncPipeline.__init__` accepts `pretokenized_loader` parameter
- When present, spawns `_pre_tokenized_loop` instead of `_loader_loop` + `_tokeniser_loop`
- Same `next_batch()` path → zero downstream changes
- Test: run benchmark with and without pre-tokenized path, verify identical BatchGenerationOutput

### Step 4 — Cache management (~1 hour)
- `manifest.json` in cache directory tracks what's cached
- `get_cache_key()` computes the key from model + dataset + config
- `is_cache_valid()` checks manifest
- Auto-detection at harness startup
- Test: change max_input_tokens, verify cache miss triggers re-tokenization

### Step 5 — CLI + harness integration (~30 min)
- `--pretokenize` flag in `__main__.py`
- Harness checks cache at startup, logs hit/miss
- `--no-pretokenized-cache` to force fresh run
- Test: end-to-end `--pretokenize` → `--config config.yaml` with auto-detection

---

## 9. Risk / edge cases

| Risk | Mitigation |
|---|---|
| Tokenizer changes silently (HF update) | `tokenizer_hash` in cache key includes vocab + special tokens JSON |
| Chat template changes (model card update) | Cache key includes SHA256 of the prompt style detection result |
| Different chunker config per run | `max_input_tokens` and `overlap_tokens` in cache key |
| Input data changes | Cache key includes hash of sorted input file paths + sizes |
| Pipeline run with different batch sizes | No issue — token IDs are per-chunk, padding happens at batch-time |
| Resume from checkpoint | PreTokenizedLoader supports seek (skip N chunks) |
| Disk space for cache | LRU eviction in manifest; `--pretokenize --max-cache-size-gb 50` flag |
| Multi-GPU / multi-process reads | Parquet is read-only after pre-processing; safe for concurrent readers |
