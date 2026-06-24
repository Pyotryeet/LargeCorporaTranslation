# AI Coding Antipatterns — TR Corpus Translation Benchmark

> **Purpose:** Document concrete mistakes AI coding agents have made in this
> project, and preventable pitfalls they could make — to create awareness for
> both human reviewers and LLM coders.
> **Status:** Living document. Current as of v3.6.
> **Audience:** Anyone (human or agent) about to edit this codebase.
>
> This is a sibling to [`ARCHITECTURE.md`](ARCHITECTURE.md). ARCHITECTURE tells
> you *what the code does*; this tells you *how to stop breaking it*.

---

## Legend

| Tag | Meaning |
|---|---|
| 🔴 | **Occurred** — this mistake actually happened in this project (with evidence: commit / audit / `file:line`). |
| 🟡 | **Risk** — a preventable pitfall an AI coder could fall into here, even if not every instance has been logged. |
| ℹ️ | **Guidance** — a positive rule derived from the pattern. |

---

## How to use this document

- **Before editing `benchmark/inference/` or `benchmark/hardware/`:** read Part C
  (the pre-flight checklist) and skim Part A.
- **When a change "feels easy"** (flip a flag, wire up a module, silence an
  exception): check whether it matches an antipattern below.
- **When reviewing an AI-generated diff:** scan for the red-flag patterns in Part A.

---

## Part A — Concrete antipatterns that occurred (🔴)

Each entry: **What** · **Evidence** · **Impact** · **Prevention**.

---

### A1. Dead code masquerading as a feature 🔴

**What:** An "optimization" is gated off (`if False:`, commented out, hardcoded
`False`, captured-but-never-replayed, stats-only) while docs and log lines still
advertise it as active. The code *looks* like it does something; it doesn't.

**Evidence:**
- Fused-kernel injection: `benchmark/inference/backends/autoregressive.py:761` — `if False:  # was: self._use_fused_kernels and ...`
- cudaMallocAsync: commented out at `autoregressive.py:654`; `_malloc_async_active = False` at `:607`.
- PagedAttention (AR path): `_use_paged_attention` hardcoded `False` at `autoregressive.py:596`; `_convert_to_paged` referenced only in comments and doesn't exist.
- CUDA Graph decode: graph captured at `autoregressive.py:1339` but **never replayed** — `_extreme_decode:1548` comment: *"DECODE LOOP (standard forward, no graph replay)"*. The whole `cuda_graphs.py` module emits a `FutureWarning` on import saying so.
- INT8 KV-cache: `_kv_quant_cache` constructed at `autoregressive.py:1184`, never `.update()`/`.get()`-ed.
- Fast-dLLM caching: `diffusion.py:709` counts cache hits but always falls through to the full forward.

**Impact:** Engineers and agents reason about performance/capability from false
premises. Someone "improves" a dead path; someone publishes a benchmark believing
PagedAttention is saving memory when it isn't. The README's old "37/39 wired"
claim was the aggregate symptom.

**Prevention:**
- The authoritative status of every feature is [`ARCHITECTURE.md` §8 Feature Status](ARCHITECTURE.md#8-feature-status-the-truth-table). Check it before assuming a feature is active.
- When you disable a feature, update *every* place that advertises it (log lines,
  docstrings, docs) the same day. A log line that prints `PagedAttn=True` while
  the feature is hardcoded off is itself a bug.
- ℹ️ If you must keep dead code, mark it `# DEAD:` / `# DISABLED:` with the
  reason and date, and make the log line reflect reality.

**Postscript June 2026 — some dead code actually deleted.** The following were
deleted from the codebase entirely:
- `benchmark/quality/confidence.py` (218 lines)
- `benchmark/quality/ensemble.py` (210 lines)
- `benchmark/observability/dashboard.py` (275 lines)
- `benchmark/observability/nsight_profiler.py` (405 lines)
- `benchmark/observability/server.py` (291 lines)
- `benchmark/run_models.py` (525 lines)

This is the *correct* resolution for genuinely dead code — delete it rather than
leaving it to accumulate stale docstrings and confuse readers. The lesson (don't
leave dead code around) was applied.

---

### A2. Docstring/comment lies 🔴

> ✅ **Progress June 2026:** Several of the docstring lies below have been fixed:
> `kv_cache_quant.py`'s module docstring no longer claims nonexistent Triton/Metal
> dequant kernels — it now honestly states "pure eager PyTorch" and "never wired
> into the decode loop." The `fused_ops.py` and `jit_compiler.py` disabled-kernel
> status is now clearly documented. This is the correct direction.

**What:** Code contradicts its own docstring or comments. The documentation
nearest the code is wrong, so the person reading it most carefully is misled most.

**Evidence:**
- `_fused_swiglu_gate_up_triton` (`hardware/fused_ops.py`) implements the op in pure eager PyTorch; its own docstring admits *"This function does NOT use Triton."* The name is a lie.
- `kv_cache_quant.py` module docstring claims *"Works on both CUDA (via Triton dequant kernels) and MPS (via Metal dequant shaders)"* — no such kernels exist; dequant is pure eager.
- `PinnedBufferPool.release()` (`data/pipeline.py`) docstring claimed it was "unused" while it is called from `release_batch`.
- Comments referencing `_convert_to_paged` as if it existed (it doesn't — only mentioned in comments).
- `fused_ops.py` docstring advertised `fused_rotary_qkv_projection` support — never implemented.

**Impact:** Anyone debugging performance or extending a module wastes time
chasing capabilities that don't exist. "The docstring said it used Triton, so I
assumed the slowness was a kernel bug."

**Prevention:**
- Treat docstrings as testable claims. If a function is named `_..._triton`, it
  must use Triton — or rename it.
- When you remove an implementation, remove its docstring claims in the same change.
- ℹ️ Code reviews should reject "docstring says X, code does Y" on sight.

---

### A3. Copy-paste divergence 🔴

**What:** Two near-identical code paths drift apart; a bug fixed in one copy
survives in the other.

**Evidence:**
- The harness previously had a ~250-line duplicated translation loop (new-run vs.
  resume). 9 bugs survived 6 audit rounds because fixes to one copy were never
  applied to the other. Now de-duplicated into `_run_translation_core`
  (`orchestration/harness.py:347`) — but the lesson stands.
- Config-hash computation appeared verbatim 4× at different line numbers.
- "model" prefix stripping implemented independently in 3 places.
- `BYTES_PER_MB`-style naming where the variable actually meant KB.

**Impact:** The project was unable to converge on correctness despite ~87
agent-hours of fixing, because each fix had to be independently rediscovered for
every copy.

**Prevention:**
- Before duplicating, extract a shared helper. Before fixing, `grep` for siblings.
- ℹ️ If two paths must differ, factor the *shared* core into one function and
  pass the differing bits as parameters (exactly the `_run_translation_core`
  pattern).

---

### A4. Exception swallowing 🔴

**What:** Bare `except:` / `except Exception: pass` / silent fallbacks in quality,
metrics, or data-integrity paths. Failures become silent defaults.

**Evidence:** 40+ instances flagged across the codebase in the Round-7 audit.
- Chat-template failure in quality benchmark → silently produces garbage
  translations with zero indication.
- Speculative decoder tests `skip` on `AssertionError` instead of failing.
- GPU metrics black out for 5 seconds on any transient `powermetrics` failure.
- Checkpoint loader silently drops corrupted JSONL lines with no count of loss.

**Impact:** The benchmark completes with plausible-looking, completely wrong
output. No alert fires because the alerts (where they exist) fire on the
already-corrupted data. This is the #1 systemic failure mode in the codebase.

**Prevention:**
- Distinguish *expected* fallbacks (optional dependency missing → degrade) from
  *unexpected* failures (a metric didn't compute → surface it). Only the former
  may be silent.
- ℹ️ In integrity paths, log the failure with enough context to diagnose, and
  record that a fallback occurred (e.g. set `error` on the result, not just a
  `None` score that downstream code treats as 0.0). See A15.

---

### A5. Silently wrong metrics 🔴

**What:** A metric is computed from the wrong quantity, producing numbers that
look fine but are inflated/deflated.

**Evidence:**
- NLLB `input_tokens` counted the **padded** input length instead of real tokens
  → inflated TPS. Fixed for NLLB in commit `ffa707b`. The AR and TRT backends
  had the same bug — also now **fixed**. `autoregressive.py` and `tensorrt_backend.py`
  now use `attention_mask.sum().item()` to count real tokens (not padded length),
  matching the NLLB and diffusion backends.

**Update June 2026:** This is now **fixed** on all backends. `autoregressive.py` and
`tensorrt_backend.py` now use `attention_mask.sum().item()` to count real tokens
(not padded length), matching the NLLB and diffusion backends. This was one of the
longest-surviving metric bugs.

**Impact:** Throughput numbers reported were inflated, corrupting the extrapolation
(days-to-completion) that is the entire point of the benchmark.

**Prevention:**
- Token counts must come from `attention_mask.sum()`, never from tensor shape.
- ℹ️ When fixing a metric bug in one backend, grep the other backends for the
  same pattern and fix all of them (this is the A3 lesson applied to A5).

---

### A5b. The bug that survived 6 audit rounds 🔴 (case study)

This is a concrete intersection of **A3** (copy-paste divergence) and **A5** (silently
wrong metrics):

- The AR input_tokens padding bug (`len(batch.input_ids[i])` instead of
  `attention_mask.sum().item()`) survived **6 audit rounds** because:
  1. NLLB was fixed in commit `ffa707b` but the fix was never applied to AR/TRT
     (classic copy-paste divergence — A3).
  2. Nobody grepped for the pattern `len(batch.input_ids[i])` across backends
     when the NLLB fix was made.
  3. The AR and TRT backends are near-identical copies — a `grep` would have
     found the sibling instantly.

**Lesson:** When you fix a bug in one backend, your first reflex must be to grep
the other backends for the same pattern. A3 and A5 reinforce each other: copy-paste
creates the duplicate, and silently-wrong-metrics keeps it hidden because the
numbers "look reasonable."

---

### A6. Cascading feature disable 🔴

**What:** Disabling one feature flips a `not X` guard elsewhere and crashes an
unrelated path.

**Evidence:**
- `torch.compile` was disabled to dodge a PyTorch 2.11 `cudagraph_trees` crash
  (commit `9fa3397`). The fused-kernel injection guard had been
  `if self._use_fused_kernels and backend=="cuda" and not self.use_torch_compile:`.
  With compile now off, `not use_torch_compile` flipped True and injected Triton
  fused kernels into **eager** forward — where they crash with a CPU-tensor
  pointer error. Fixed by hardcoding injection off (commit `804c0a6`, the `if False:` at `autoregressive.py:761`).

**Impact:** A targeted disable became a runtime crash in a different subsystem.

**Prevention:**
- When disabling feature X, grep for `not X` / `X` guards and check each
  dependent's new behavior under the disabled state.
- ℹ️ Prefer explicit opt-in flags over `not <feature>` guards, so disabling one
  feature can't implicitly enable another.

---

### A7. Regressions introduced by a fix-pass 🔴

**What:** An automated "fix all the audit findings" pass introduces new bugs
while fixing old ones.

**Evidence:** The Round-7 fix pass (94 fixes applied) introduced 3 regressions,
caught only by a separate verification agent:
1. `pipeline.py` sentinel unpack-order: moved `item is self._SENTINEL` check
   *before* the tuple unpack. Previously a sentinel arriving caused `TypeError`.
2. `autoregressive.py` speculative decoder: missing `if self._spec_decoder is not None`
   guard before `.load()`.
3. Missing `scripts/run_e2e_benchmark.py` referenced by tests.

**Impact:** Without an independent verification pass, these would have shipped.

**Prevention:**
- Treat a fix-pass as a diff requiring review, not a batch-apply. Run the test
  suite and a targeted verification after every batch.
- ℹ️ Order-of-operations matters: identity/sentinel checks before unpacking;
  None-guards before method calls. Reviewers should watch for these specifically.

---

### A8. Fork bomb via recursive auto-sharding 🔴

**What:** A launcher auto-detects N GPUs and spawns one child per GPU; each child
re-detects N GPUs and spawns N more, ad infinitum.

**Evidence:** `run.sh --multi-gpu` (commit `5ec301c` fix). Children inherited the
full CLI args including the trigger flag, re-detected 2 GPUs, and launched 2
children each → exponential fork bomb → system out of memory/processes, SSH
impossible. Recovered with `pkill -9 -f benchmark`.

**Impact:** The production machine became unresponsive.

**Prevention:**
- Recursive launchers must strip the trigger flag from child args and/or carry a
  depth guard (`TR_DEPTH` env var, hard-fail at depth ≥ 2). Both were added.
- ℹ️ Any "detect and spawn" logic needs a re-entry guard. Test it by running the
  child command verbatim and confirming it doesn't re-spawn.

---

### A9. Library version silently changed return type 🔴

**What:** A dependency upgrade changed an object's type so that idiomatic
unpacking silently iterated the wrong thing.

**Evidence:** COMET 2.2.7's `model.predict()` returns a `Prediction(dict)`
subclass. Code did `seg_scores, system_score = result`, which unpacks the dict
**keys** (the strings `"scores"`, `"system_score"`), not the values →
`ValueError: could not convert string to float: 's'`. Fixed by attribute access:
`result.scores`, `result.system_score` (`quality/metrics_comet.py`).

**Impact:** Quality metrics silently broken on the upgraded version.

**Prevention:**
- After a dependency bump, smoke-test the exact call sites that unpack returns.
- ℹ️ Prefer attribute access over positional unpacking for library objects whose
  type may be a `dict` subclass.

---

### A10. Class-level / global monkey-patching 🔴

**What:** Patching a base class or global to fix a library issue, corrupting
every unrelated consumer in the process.

**Evidence:** COMET's tokenizer fix originally monkey-patched
`PreTrainedTokenizerBase.build_inputs_with_special_tokens` at the **class** level
— a global modification affecting the data pipeline's tokenizer, the inference
backend's tokenizer, and any other library using transformers, applied
unconditionally at import. Later fixed to **instance** scope (only the COMET
tokenizer).

**Impact:** If quality evaluation ran in the same process as the throughput
benchmark, tokenization was silently corrupted for the benchmark.

**Prevention:**
- Patch the specific instance, never the base class. Use a wrapper or context
  manager.
- ℹ️ Better: contribute the fix upstream so the patch isn't needed.

---

### A11. Hardcoded architecture constants 🔴

**What:** Magic numbers / model-specific constants baked into general-purpose code.

**Evidence:**
- `hardware/parallelism.py` was hardcoded to Gemma-3-12B constants (48 layers,
  etc.). The layer-mismatch branch assigns to read-only `@property` fields →
  `AttributeError` for any other model. (Latent: `apply_tensor_parallelism` is
  never called — see ARCHITECTURE §8 #22.)
- `END_OF_TURN_TOKEN_ID = 106` was hardcoded in 3+ places. **Now fixed:** it's
  defined once in `config/constants.py:82` and imported everywhere (a successful
  application of the "don't hardcode" rule).
- `jit_compiler.py` sm90a misdetection: `hasattr(props, 'multi_processor_count')`
  is always True, mislabeling every Hopper GPU.

**Impact:** Code that works for one model crashes or misbehaves for another.

**Prevention:**
- Import constants from `config/constants.py`; read architecture dims from
  `model.config`. ℹ️ The `END_OF_TURN_TOKEN_ID` fix is the template.

---

### A12. Non-atomic writes / silent data loss 🔴

**What:** Writing state files non-atomically; on crash the file is truncated/
corrupt and the loader silently treats it as "no data."

**Evidence:**
- `observability/perf_regression._save_raw` does a plain `json.dump` — a crash
  mid-write corrupts the baseline JSON; `load_baseline` then catches
  `JSONDecodeError` and returns `[]`, silently auto-re-establishing and losing
  history.
- The checkpoint loader silently drops corrupted lines with no count.

**Impact:** Silent loss of monitoring/baseline history; resume robustness eroded.

**Prevention:**
- Write to a temp file, `flush`+`fsync`, then `os.rename` atomically
  (the `checkpoint.py` pattern). On load, count and log what was dropped.
- ℹ️ "Silent recovery" is worse than a loud error when the data is load-bearing.

---

### A13. Tests that pass when preconditions are missing 🔴

**What:** A test that silently does nothing (runs no assertions) when its
preconditions aren't met, so it shows green in CI while testing nothing.

**Evidence:**
- `test_load_from_yaml`: `if config_path.exists(): <assertions>` — if the file is
  absent, the test passes with zero assertions run.
- Fixtures that fall back to auto-generated "meaningless" synthetic data, with
  tests that still pass green against it.
- Tests that build their own `argparse.ArgumentParser` and assert on *that*,
  never touching the real CLI.
- "Test cleanup happens" with no `assert` / `weakref` check.

**Impact:** CI green while testing nothing of value. When real data goes missing
in CI, all such tests silently become no-ops.

**Prevention:**
- Assert the precondition, or `pytest.skip("reason")` when it's missing — never
  silently pass.
- Exercise the real entry point, not a hand-built fake.
- ℹ️ Every test must answer: *if the production code broke, would this fail?*

---

### A14. CUDA graph capture nested inside torch.compile 🔴

**What:** Nesting custom CUDA-graph capture inside `torch.compile`'s own graph
capture.

**Evidence:** `captures_underway.empty() INTERNAL ASSERT FAILED`. Custom graph
capture in warmup Phase 3 conflicted with `torch.compile`'s inductor graph
capture. Fixed by guarding capture with `and not self.use_torch_compile`
(`autoregressive.py:1343`) and wrapping `empty_cache()` in try/except.

**Impact:** Warmup crash when both were enabled.

**Prevention:**
- Two graph-capture mechanisms can't coexist without explicit coordination.
  ℹ️ When layering graph/compile optimizations, pick one owner of capture and
  gate the other.

---

### A15. Inconsistent partial-run handling 🔴

**What:** When a component fails to produce a value, different consumers treat the
`None`/error differently, so the same failure passes or fails the gate depending
on which metric it was.

**Evidence:** `QualityResults.scores_meet_targets` checks BERTScore with
`bs = self.bertscore.get("system_score", 0) or 0.0` — a `None`/error becomes `0.0`
and **fails** the 0.55 target. COMET/Kiwi/BLEU/chrF are *skipped* when
`system_score is None`. So a BERTScore load failure flips the overall pass/fail,
while a COMET failure is ignored.

**Impact:** The quality gate's verdict depends on which metric happened to fail,
not on a consistent policy.

**Prevention:**
- Pick one policy for "metric unavailable" (skip, or fail-loud) and apply it
  uniformly. Don't coerce `None → 0.0` in one place and skip in another.
- ℹ️ Partial results should be reported as partial, with an explicit
  `metrics_computed` set, not folded into a pass/fail by accident.

---

## Part B — Preventable pitfalls (🟡) an AI coder could fall into here

These are risks specific to this codebase. Each is preventable by a cheap check.

| 🟡 | Pitfall | Cheap prevention |
|---|---|---|
| 🟡 | **Re-enabling a gated feature without reading the gating comment + `git blame`.** The comment usually says *why* (e.g. "incompatible with `torch.compile` `cudagraph_trees`"). | Read the comment; verify the reason is resolved; check [Feature Status](ARCHITECTURE.md#8-feature-status-the-truth-table). |
| 🟡 | **"Wiring up" a built-but-unused module without its integration contract.** PagedKVCache needs the `PagedCache` HF-Cache shim, not just allocation; INT8 KV-cache needs decode-loop `.update()`/`.get()` calls; fused kernels need `torch.compile`. | Find the one path that already uses it correctly (the ContinuousBatcher for paged KV) and mirror its contract. |
| 🟡 | **Editing one copy of duplicated logic without grepping for siblings.** | `grep` the changed symbol across `benchmark/` before finalizing. |
| 🟡 | **Adding a backend that breaks the protocol contract.** | Subclass `InferenceBackend`; init `_configured_batch_size`; return `BatchGenerationOutput`; compute `input_tokens_total` from `attention_mask.sum()`. See [`DEVELOPMENT.md` §5.1](DEVELOPMENT.md#51-add-an-inference-backend). |
| 🟡 | **Reasoning about performance from the README's optimization count.** | Use [Feature Status](ARCHITECTURE.md#8-feature-status-the-truth-table); the hot path is eager + `torch.compile` + TE FP8. |
| 🟡 | **Disabling `torch.compile` without re-checking dependents.** | grep `not use_torch_compile` / `use_torch_compile` guards (see A6). |
| 🟡 | **Assuming `device_map="auto"` delivers the "2×H200" story.** | The single-GPU fast path (`autoregressive.py:919`) bypasses multi-GPU for every model actually run; `apply_tensor_parallelism` is never called. |
| 🟡 | **Passing user input through shell interpolation / `subprocess` without sanitizing.** (RCE history in `run.sh`, `benchmark_all_models.py`.) | Use `argv` lists, never shell interpolation; sanitize/whitelist env. |
| 🟡 | **Letting exceptions fall through to defaults in quality/metrics paths.** | Surface the failure (set `error`, log with context); see A4/A15. |
| 🟡 | **Hardcoding constants instead of importing from `config/constants.py` / reading `model.config`.** | See A11. |
| 🟡 | **Coercing `None → 0` in a target check.** | See A15 — skip or fail-loud, consistently. |
| 🟡 | **Writing state files non-atomically.** | temp + fsync + `os.rename`; see A12. |
| 🟡 | **Writing a test that passes when data is missing.** | `assert` preconditions or `pytest.skip`; see A13. |
| 🟡 | **Trusting a log line that prints a feature flag as `True`** (e.g. the H200 dry-run log `PagedAttn=True, CUDA_Graph=True, fused_kernels=True`). | The flag reflects *config*, not *hot-path activity*. Cross-check against Feature Status. |
| 🟡 | **The Capability Registry is now the single source of truth for what's active.** The old `print("PagedAttn=True")` log line pattern (prints config flag, not hot-path reality) is deprecated. Use `backend._capability_registry.report_text()` or check `ActivationState` directly. See `benchmark/config/capability.py`. |
| 🟡 | **Monkey-patching at class/global scope.** | Patch the instance; see A10. |
| 🟡 | **Positional-unpacking a library return value.** | Use attribute access; see A9. |

---

## Part C — Pre-flight checklist for AI coders in this repo

Before you start editing, confirm:

- [ ] I have read [`ARCHITECTURE.md` §8 Feature Status](ARCHITECTURE.md#8-feature-status-the-truth-table) and know which features near my change are wired vs. gated.
- [ ] I am not treating a 🟡/💀 feature as active, and not "wiring it up" without its integration contract.
- [ ] I have `git blame`d any gating comment / `if False:` near my change and understand *why* it's gated.
- [ ] I have grepped for siblings of any duplicated logic I'm touching.

While editing, confirm:

- [ ] Constants are imported from `config/constants.py` or read from `model.config` — not hardcoded.
- [ ] Token counts use `attention_mask.sum()`, not padded tensor length.
- [ ] Any `except` I add or touch is not silently swallowing a failure in a quality/metrics/data-integrity path.
- [ ] Any state file I write uses temp + fsync + atomic rename.
- [ ] I am not monkey-patching at class/global scope.
- [ ] If I disabled a feature, I grepped for `not <feature>` dependents.
- [ ] If a feature status changed, I updated the Capability Registry (not just a log line or docstring).

Before declaring done, confirm:

- [ ] Every non-obvious claim in my change message/PR cites a `file:line`.
- [ ] Docstrings/comments I touched still match the code (no A2 lies).
- [ ] Log lines I touched reflect reality (no A1 "prints True, does nothing").
- [ ] `make lint` passes; `make test` passes; any test I added *fails* if the code breaks (no A13).

---

*Cross-references: [`ARCHITECTURE.md`](ARCHITECTURE.md) (what the code does) · [`DEVELOPMENT.md`](DEVELOPMENT.md) (how to extend it) · [`docs/README.md`](README.md) (navigation).*
