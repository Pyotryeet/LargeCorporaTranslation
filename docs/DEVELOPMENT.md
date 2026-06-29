# Development Guide

> **Purpose:** How to develop on, extend, and contribute to this codebase.
> **Status:** Current as of v3.7.
> **Audience:** Engineers and LLM coding agents.

---

## Table of Contents

1. [Before You Touch Code](#1-before-you-touch-code)
2. [Repository Layout](#2-repository-layout)
3. [Testing](#3-testing)
4. [Linting & Formatting](#4-linting--formatting)
5. [Extending the Codebase](#5-extending-the-codebase)
6. [Coding Conventions](#6-coding-conventions)
7. [LLM Coding Agent Orientation](#7-llm-coding-agent-orientation)

---

## 1. Before You Touch Code

If you are about to edit `benchmark/inference/` or `benchmark/hardware/`, read
these two docs first — they exist precisely because this codebase has a long
history of "optimizations" that look wired but aren't:

1. [`docs/ARCHITECTURE.md` §8 Feature Status](ARCHITECTURE.md#8-feature-status-the-truth-table)
   — the authoritative table of what is wired vs. gated vs. dead.
2. [`docs/AI_CODING_ANTIPATTERNS.md`](AI_CODING_ANTIPATTERNS.md) — concrete
   mistakes already made here (dead-code-as-feature, cascading disables,
   exception swallowing, copy-paste divergence) and how to avoid repeating them.

The single most common error in this repo's history is **treating a built-but-gated
module as if it were active** (or "wiring it up" without its integration contract).
Don't.

---

## 2. Repository Layout

```
H200Research/
├── benchmark/                 # the package (entry: benchmark/__main__.py)
│   ├── config/                # schema.py (Pydantic v2) · constants.py · model_presets.py
│   ├── data/                  # loader · chunker · filters · pipeline · parallel_gz
│   ├── hardware/              # backend detect · precision · parallelism · architecture
│   ├── inference/             # engine · backends/ · speculative · paged_attention · continuous_batcher · batch_* · sampling
│   ├── metrics/               # collector · throughput · gpu_sampler · system_sampler · batch_logger
│   ├── observability/         # prometheus_metrics
│   ├── orchestration/         # harness · checkpoint · signals
│   ├── quality/               # benchmark · metrics_{bertscore,comet,bleu,chrf} · references
│   ├── reporting/             # aggregator · extrapolation · json_report · markdown_report
│   └── utils/                 # env_check · logging_setup · timer · version · json_utils · print_summary
├── quantization/              # static FP8 pipeline (smoothquant · qat · static_fp8)
├── scripts/                   # benchmarks (benchmark_single · benchmark_models · run_nllb · bench_format)
│   └── evaluation/            # human eval (build_data · select_sentences · translate · evaluate · tune_weights)
├── tests/                     # pytest suite (~27 files)
├── docs/                      # this documentation set (see docs/README.md)
├── setup.sh / run.sh / Makefile / Dockerfile / Dockerfile.ngc   # operational entry points
└── config.yaml                # production H200/CUDA config
```

For the *runtime* layout and data flow, see
[`ARCHITECTURE.md` §2-§6](ARCHITECTURE.md#2-runtime-hot-path).

---

## 3. Testing

```bash
make test      # tests across 23 files (pytest)
make lint      # ruff check
make format    # ruff format (+ black)
```

### Test-integrity caveats (read before trusting a green CI)

These are documented weaknesses in the test suite — be aware when adding tests:

- **Real-vs-synthetic data:** some fixtures fall back to auto-generated
  "meaningless" synthetic data when real data is missing, and tests that hit this
  path **still pass green**. A green run is not proof the test exercised real data.
  Prefer fixtures that skip (not silently pass) when their precondition is missing.
- **Tests that pass when preconditions are absent:** e.g. a config test that does
  `if config_path.exists(): <assertions>` and silently passes (no assertions run)
  when the file is absent. Always `assert` the precondition, or `pytest.skip()`.
- **Tests that build their own fake parser** instead of exercising the real CLI —
  these test the fake, not the code. Import and exercise the real entry point.
- **Tests with no assertion** (e.g. "test cleanup happens" with no `assert` /
  `weakref` check) — these are lies. Convert to `xfail` or add a real assertion.

When you add a test, ask: *if the production code broke, would this test actually
fail?* If not, it's not a test.

### Real-data / integration tests

Heavy tests (real model downloads, GPU) are marked and skipped when hardware or
gated model access is unavailable. `tests/conftest.py` provides shared fixtures
(`real_tokenizer`, `sample_jsonl_path`, …). Don't define session-scoped duplicates
of conftest fixtures — it wastes memory and diverges.

---

## 4. Linting & Formatting

- **Ruff** is the linter/formatter of record (`make lint`, `make format`).
- **Black** is also configured (`make format` runs both).
- CI runs `ruff check`; a lint failure blocks merge.
- Follow the surrounding code's style: match comment density, naming, and idiom.

---

## 5. Extending the Codebase

### 5.1 Add an inference backend

1. Subclass `InferenceBackend` (`benchmark/inference/backends/protocol.py:174`).
2. Set class attributes: `model_type` (`ModelType`), `capabilities`
   (`ModelCapability` bitmask), `display_name`.
3. Implement `load()`, `warmup()`, `translate_batch()`. `load()` must set
   `self._loaded = True`.
4. **Initialize `self._configured_batch_size` in `__init__`** — the engine reads
   it via a property with *no default* (`engine.py:148`); a missing attribute is a
   bug, not a fallback.
5. **Separate CUDA and MPS implementations**: Following the v3.8 design pattern, split backend code into separate files (e.g. `yourbackend_cuda.py` and `yourbackend_mps.py`) and keep a dispatcher class in the main file (e.g. `yourbackend.py`) that uses `__getattr__` and `__setattr__` delegation proxies to route to the correct class at load time.
6. Register the dispatcher in `ModelRegistry._register_builtin` (or via the plugin system below).
7. Return `BatchGenerationOutput`; compute `input_tokens_total` from
   `attention_mask.sum()` (not padded length — see [§6](#6-coding-conventions)).

### 5.2 Add a custom plugin

Drop a `.py` in `~/.tr_benchmark/plugins/`, `TR_BENCHMARK_PLUGIN_PATH`, entry
points, or `./plugins/`, defining a `CustomModelPlugin` subclass with `name` and
`create_backend(config)`. Discovery is gated behind `TR_ALLOW_UNTRUSTED_PLUGINS=1`
— plugins run with **full process privileges, no sandbox**. Explicit
`register_plugin()` bypasses the gate. See `benchmark/inference/backends/custom_plugin.py`.

### 5.3 Add a model preset

Add a `ModelPreset` to `MODEL_PRESETS` in `benchmark/config/model_presets.py`
(there are currently 5). Presets provide architecture constants
(`num_layers`, `num_kv_heads`, `head_dim`, `hidden_size`) so other components
don't hardcode them. Resolve via `get_preset_by_name()` /
`resolve_architecture_defaults()`.

### 5.4 Re-enabling a gated feature

If you intend to turn a 🟡/💀 feature back on:

1. Read the gating comment **and** `git blame` it — the comment usually says *why*
   it was disabled (e.g. "incompatible with `torch.compile` `cudagraph_trees`").
2. Check [Feature Status](ARCHITECTURE.md#8-feature-status-the-truth-table) for
   the integration contract. Example: PagedKVCache on the AR path isn't just
   "allocate blocks" — the model's attention layers must read them, which requires
   the `PagedCache` `DynamicCache` shim (only the ContinuousBatcher path has this).
3. Verify the reason for disabling is actually resolved before re-enabling.

---

## 6. Coding Conventions

These conventions exist because violating them has caused real bugs here (see
[`AI_CODING_ANTIPATTERNS.md`](AI_CODING_ANTIPATTERNS.md)).

| Convention | Why | Evidence |
|---|---|---|
| **Don't hardcode constants** — import from `config/constants.py` or read `model.config`. | Hardcoded architecture constants crash on other models; magic token IDs drift. | `parallelism.py` (hardcoded Gemma-3-12B); `END_OF_TURN_TOKEN_ID` now correctly imported from `constants.py:82`. |
| **Compute `input_tokens_total` via `attention_mask.sum()`**, never padded `input_ids` length. | Padded length inflates TPS metrics. | NLLB fixed (commit `ffa707b`); AR/TRT still affected (`autoregressive.py` `_assemble_output`). |
| **Don't swallow exceptions** in quality / metrics / data-integrity paths. | Silent fallbacks produce plausible-but-wrong output with no indication. | 40+ bare `except`s flagged in audit; chat-template failure → garbage translations. |
| **Atomic writes for state files** (temp + `os.rename`). | Non-atomic writes lose state on crash. | `checkpoint.py` does this correctly; `perf_regression._save_raw` does not. |
| **Respect the `ModelCapability` bitmask** — don't call an optional method without checking the cap. | `encode_source`/`score_candidates` raise `NotImplementedError` by default. | `protocol.py:265`. |
| **Check Feature Status before touching optimizations.** | Most are gated; "wiring up" a dead module wastes effort and can crash. | `ARCHITECTURE.md` §8. |
| **When disabling a feature, grep for dependents.** | Disabling `torch.compile` once flipped a guard and crashed fused-kernel injection. | commits `9fa3397` → `804c0a6`. |
| **Don't monkey-patch at class/global scope** when instance scope suffices. | Global patches corrupt unrelated components. | COMET tokenizer patch (was class-level, now instance-level). |
| **Don't duplicate logic** across paths — extract a shared helper. | Duplicated copies diverge; bugs fixed in one survive in the other. | The harness translation loop (now de-duplicated into `_run_translation_core`). |
| **Tests must fail when the code breaks** — assert preconditions, skip (don't silently pass) when data is missing. | Green-but-meaningless tests are worse than no tests. | synthetic-data fixtures; `test_load_from_yaml`. |

### Measured constants (replace guesses)

These constants have been replaced with H200 measurements (June 2026).
See COMPILATION_GUIDE.md for measured baselines.

| Constant | Before (guess) | After (measured) | Source |
|---|---|---|---|
| `GPU_MEMORY_BUDGET_FRACTION` | 0.95 | **0.939** | M1.1 |
| `GPU_MEMORY_RESERVE_BYTES` | 4 GiB | **1 GiB** | M1.1 |
| `BYTES_PER_INPUT_TOKEN` | 4.0 | **4.45** | M0.2 |
| `SHUFFLE_BYTES_PER_CHAR_OVERHEAD` | 2.0 | **1.75** | M3.1 |
| `prefetch_workers` | 4 | **8** (64-core H200) | M3.2 |

---

## 7. LLM Coding Agent Orientation

> ⚠️ **Read before editing.** This codebase has been heavily audited and has a
> documented history of AI-introduced bugs. The risks are specific and known.

**Mandatory pre-reads:**
- [`ARCHITECTURE.md` §8 Feature Status](ARCHITECTURE.md#8-feature-status-the-truth-table)
- [`AI_CODING_ANTIPATTERNS.md`](AI_CODING_ANTIPATTERNS.md)

**Decode implementation status (v3.9):**
- **Autoregressive (TranslateGemma 4B)**: Custom ``_extreme_decode`` loop — per-token ``model()`` with vectorized EOS. Zero ``model.generate()`` overhead.
- **Encoder-decoder (NLLB/MADLAD CUDA)**: Custom ``_fast_decode_batch`` loop — encoder runs once, tight per-token greedy decoder loop with pre-allocated buffers and vectorized EOS detection. Replaces HF ``model.generate()``, eliminating ~26.8ms Python overhead per batch. Implemented in ``nllb_cuda.py``.
- **Encoder-decoder (NLLB/MADLAD MPS)**: Still uses HF ``model.generate()`` (Apple Silicon doesn't benefit from the same optimization).

**Verified toolchain (June 2026):**
- **torch 2.12.1+cu126** — recommended. +27% TPS over 2.6.0 (1,650 vs 1,300 tok/s on 4B).
  `torch.compile(mode="default")` active for 2.12-2.13; `mode="reduce-overhead"` for 2.14+.
- **torch 2.6.0+cu124** — still works but compile auto-skipped (cudagraph_trees bug).
- **PyTorch built-in SDPA** — `F.scaled_dot_product_attention` handles all attention.
- **FP8**: not working on pip venvs (TE cuBLASLt crash); TR_SKIP_FP8=1 is the default.

- **Pre-tokenization**: `--pretokenize` once per model; cache auto-detected thereafter (+60% TPS).
- **TE FP8** — source-build required: `pip install 'transformer-engine[pytorch]' --no-build-isolation` with CPATH set. −40% TPS for 4B models.
- **torch.compile** — works but minimal gain (<1%) for 4B. More benefit at 12B+.
- See `docs/COMPILATION_GUIDE.md` §Verified Toolchain and `docs/H200_SETUP.md` §Known Issues for full details.

**The four traps to avoid:**

1. **Don't trust the docstring/comment over the code.** Several docstrings
   contradict their own implementation (e.g. `_fused_swiglu_gate_up_triton`
   doesn't use Triton; `kv_cache_quant.py` claims nonexistent kernels). Verify
   claims against the actual code with `file:line`.
2. **Don't "wire up" a dead module without its integration contract.** A module
   existing ≠ it being usable. PagedKVCache needs the `PagedCache` shim; INT8
   KV-cache needs `.update()`/`.get()` calls in the decode loop; fused kernels
   need `torch.compile`. Read *why* it's gated before re-enabling.
3. **Don't edit one copy of duplicated logic without grepping for siblings.**
   "model" prefix stripping, config-hash computation, and (historically) the
   translation loop exist in multiple places. Fix all copies or extract a helper.
4. **Don't disable a feature without checking its dependents.** Gating one thing
   off can flip a `not X` guard elsewhere and crash an unrelated path.

**Verification before declaring done:**
- Cite `file:line` for every non-obvious claim you make or rely on.
- If you touch a metric path, sanity-check that `input_tokens_total` is computed
  from `attention_mask.sum()`, not padded length.
- If you touch an `except`, confirm it isn't silently swallowing a failure that
  should surface.
- Run `make lint` and `make test`.

---

*See also: [`ARCHITECTURE.md`](ARCHITECTURE.md) · [`AI_CODING_ANTIPATTERNS.md`](AI_CODING_ANTIPATTERNS.md) · [`COMPILATION_GUIDE.md`](COMPILATION_GUIDE.md) · [`docs/README.md`](README.md).*
