# Architectural Flaws Analysis — H200 TR Corpus Translation Benchmark

> **Purpose:** A systems-architecture-level analysis of the structural flaws in
> this codebase — what they are, why they matter, how they interact, and how to
> fix them. **Not a bug list.** A diagnosis of *why* bugs keep occurring and
> *why* fixes keep regressing.
>
> **Methodology:** Full codebase read (every `.py` in `benchmark/`), verified
> against live-code `file:line` citations, cross-checked with 7 prior audit
> rounds, and enriched with research on production inference-framework patterns.
> **Status:** June 2026, v3.6 codebase. Companion to [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Flaw #1: The False Flag Architecture](#flaw-1-the-false-flag-architecture)
3. [Flaw #2: Two Parallel Optimization Stacks](#flaw-2-two-parallel-optimization-stacks)
4. [Flaw #3: Ad-Hoc Feature Gating System](#flaw-3-ad-hoc-feature-gating-system)
5. [Flaw #4: Backend Protocol Contract Erosion](#flaw-4-backend-protocol-contract-erosion)
6. [Flaw #5: The God Harness](#flaw-5-the-god-harness)
7. [Flaw #6: Config/Source-of-Truth Confusion](#flaw-6-configsource-of-truth-confusion)
8. [Flaw #7: Extrapolation Reliability Gap](#flaw-7-extrapolation-reliability-gap)
9. [Flaw #8: Fragmented Memory Architecture](#flaw-8-fragmented-memory-architecture)
10. [Flaw #9: Hardcoded Architectural Assumptions](#flaw-9-hardcoded-architectural-assumptions)
11. [Flaw #10: Observability of the Wrong Things](#flaw-10-observability-of-the-wrong-things)
12. [Flaw Interaction Graph](#12-flaw-interaction-graph)
13. [Prioritized Fix Roadmap](#13-prioritized-fix-roadmap)

---

## Executive Summary

The H200 TR Corpus Translation Benchmark is a ~23,500-line Python codebase
designed to answer one question: *how many days to translate 6.23T English
tokens into Turkish on 2× H200 GPUs?*

After a full-codebase audit (this analysis + 7 prior rounds + the
documentation overhaul), **10 architectural flaws** emerge. They are not
independent — several amplify each other, and their interaction explains why
this project has been unable to converge on correctness despite ~87 agent-hours
of fixing:

1. **The False Flag Architecture** — features exist in code but are gated off,
   yet the system prints "enabled" log lines and advertises speedups it doesn't
   deliver. There is no programmatic way to query "what is actually active?"

2. **Two Parallel Optimization Stacks** — the AR backend and the ContinuousBatcher
   own separate, non-communicating optimization objects. PagedAttention is
   simultaneously "dead" (AR path) and "real" (CB path) depending on which stack
   you examine.

3. **Ad-Hoc Feature Gating** — at least 7 different gating mechanisms
   (`if False:`, hardcoded booleans, env vars, commented-out code, config fields
   that print True but don't activate, captured-but-never-replayed,
   constructed-but-never-accessed). No registry, no lifecycle, no query API.

4. **Backend Protocol Contract Erosion** — the `InferenceBackend` ABC is
   under-specified and unenforced. `input_tokens_total` is computed two different
   ways (padded vs. real). `phase_timings` diverges across backends. Capability
   bitmasks are declarative only. No return-type validation.

5. **The God Harness** — `BenchmarkHarness` is a 1,178-line monolith that
   directly constructs and owns every subsystem. Observability is bolted on
   (11 direct `self._prometheus.record_*()` call sites). The harness has no
   event bus, no observer pattern, no dependency injection.

6. **Config/Source-of-Truth Confusion** — `precision_config.uses_fp8` stays
   `True` under `--safe-mode` even though FP8 is not applied. Config fields and
   runtime booleans diverge. Multiple systems disagree about architecture
   defaults. Log lines print config values, not runtime state.

7. **Extrapolation Reliability Gap** — the entire benchmark exists to produce
   a days-to-completion estimate, but it assumes constant throughput (no
   degradation model), counts padded tokens in half the backends (inflating TPS),
   and has no trend detection for thermal throttling.

8. **Fragmented Memory Architecture** — no unified memory planner. Model weights,
   KV cache, and pinned buffers are managed independently. OOM recovery halves
   batch size without accounting for KV-cache growth. MPS memory issues are
   worked around ad-hoc.

9. **Hardcoded Architectural Assumptions** — `parallelism.py` hardcoded to
   Gemma-3-12B constants. `constants.py` defaults to 12B-class dimensions while
   the default model is 4B. Capability detection is wrong (sm90a always detected).

10. **Observability of the Wrong Things** — `quality_bleu`/`quality_chrf`
    Prometheus gauges are never populated. The throughput gauge captures
    instantaneous per-batch TPS (near-useless for Prometheus scraping). Log
    lines print config-state, not runtime-state. Throughput tracker discards
    latency percentiles.

**The dependency chain:** Flaw #1 (False Flags) and Flaw #3 (Ad-Hoc Gating) are
the root causes. They enable Flaw #2 (Two Stacks), Flaw #4 (Protocol Erosion),
Flaw #5 (God Harness coupling), and Flaw #6 (Config Confusion). These in turn
cause Flaw #7 (Extrapolation Gap — the whole point of the benchmark is
compromised), Flaw #8 (Memory Fragmentation), and Flaw #10 (Wrong Observability).
Flaw #9 (Hardcoded Assumptions) is independent but amplifies Flaw #4.

The fixes are organized in three tiers:
- **Tier 1 (Do now — <2 weeks):** Fix the extrapolation pipeline (padded tokens,
  throughput trend detection). These directly affect the benchmark's output
  correctness and are quick wins.
- **Tier 2 (Do next — 2-6 weeks):** Centralize feature gating (#1, #3),
  enforce the backend protocol (#4). These are the root causes.
- **Tier 3 (Do eventually — 2-3 months):** Refactor the harness (#5), unify
  memory management (#8), fix config truth (#6), fix hardcoded assumptions (#9),
  fix observability (#10).

---

## Flaw #1: The False Flag Architecture

### Manifestation

The codebase does not distinguish between "a module exists" and "a feature is
active on the hot path." Features are implemented as modules, gated off with
ad-hoc mechanisms, but the system's log output and documentation continue to
advertise them as if they were helping throughput.

**The single most damaging example:** the AR backend's load log line
(`autoregressive.py:831-838`) prints:

```
cudaMallocAsync=False, PagedAttn=True, CUDA_Graph=True,
fused_kernels=True, compile=True, JIT_CUDA=False
```

Of these 6 flags, the hot-path reality is:

| Flag printed | What the log says | What the hot path actually does |
|---|---|---|
| `cudaMallocAsync=False` | ✅ Correct | Disabled (commented out, `:654`) |
| `PagedAttn=True` | ❌ False flag | `_use_paged_attention` hardcoded `False` at `:596` |
| `CUDA_Graph=True` | ❌ False flag | Graph captured at `:1339` but **never replayed** in `_extreme_decode:1548` |
| `fused_kernels=True` | ❌ False flag | Injection hardcoded `if False:` at `:761` |
| `compile=True` | ✅ Correct | `torch.compile(reduce-overhead)` is active |
| `JIT_CUDA=False` | ✅ Correct | Kernel sources set to `None` |

**Half the log line lies.** The flags are read from `config.extra` fields and
printed as-is, with no verification against the runtime boolean that actually
controls the hot path.

### Root Cause

There is no concept of a "feature activation" that is separately tracked. The
system conflates three distinct states:

1. **Is the module implemented?** (yes, the code exists)
2. **Is it configured?** (yes, the config flag is set)
3. **Is it active on the hot path?** (no — hardcoded off, env-gated, or never called)

Log lines and documentation report state #2. Callers reason from state #1.
The hot path runs state #3. These three states can all be different, and nothing
checks.

### Concrete Evidence

| Feature | Implemented? | Config says? | Hot path? | Gating mechanism | `file:line` |
|---|---|---|---|---|---|
| PagedAttention (AR) | Yes | `True` | **No** | `_use_paged_attention = False` | `autoregressive.py:596` |
| Fused kernels (Triton) | Yes | `True` | **No** | `if False:` block | `autoregressive.py:761` |
| CUDA Graph decode | Yes | `True` | **No** | Captured, never replayed | `autoregressive.py:1548` |
| cudaMallocAsync | Yes | N/A | **No** | Commented out | `autoregressive.py:654` |
| INT8 KV-cache quant | Yes | `True` | **No** | Constructed, never read/written | `autoregressive.py:1184` |
| Fast-dLLM caching | Yes | `True` | **No** | Stats-only, always falls through | `diffusion.py:709` |
| TensorRT decode | Yes | `True` | **No** | Safety-gate raises at runtime | `tensorrt_backend.py:334` |
| Tensor parallelism | Yes | Exported | **No** | Never called — zero callers | `parallelism.py:238` |
| Perf regression gate | Yes | Exported | **No** | Zero callers | `perf_regression.py` |

**9 major features are implemented but not on the hot path.** The system
advertises them as accomplishments (log lines, doc claims of "37/39 wired"),
but they contribute zero to benchmark throughput.

### Impact

- **Engineers and agents reason from false premises.** Someone sees
  `PagedAttn=True` in the log and concludes the benchmark is saving 40-70% KV
  memory. It isn't.
- **"Wiring up" a dead module wastes effort.** Someone sees "the module exists,
  I'll just connect it" — but the integration contract (e.g. PagedKVCache
  needs the `PagedCache` `DynamicCache` shim) is not satisfied.
- **Published benchmark numbers are misattributed.** "We achieved X tokens/sec
  with CUDA graphs + fused kernels + PagedAttention" — no, you achieved it with
  `torch.compile` and TE FP8. The rest were bystanders.
- **Fix passes miss the root cause.** 7 audit rounds fixed individual bugs
  without addressing *why* the bugs kept appearing — because the system lies
  about its own state.

### Merged Fix Design

The fix needs **three layers**: (a) a registry that tracks what every feature's
*actual runtime state* is, (b) a reconciliation step that detects gaps between
config-intended and runtime-actual, and (c) auto-generated documentation from
the same truth source so the README never drifts again.

#### Layer 1: CapabilityRegistry (from research + my reconciliation layer)

The research contributed a `CapabilityRegistry` with `ActivationState` enum
(ACTIVE/INERT/UNIMPLEMENTED/DEPRECATED). I augment it with a `divergence`
detection that catches the False Flag specifically — when a config field says
True but the runtime state is INERT:

```python
class ActivationState(Enum):
    ACTIVE = auto()       # wired into hot path, affects output
    INERT = auto()        # code present, not wired (no-op or unreachable)
    UNIMPLEMENTED = auto() # documented but no code exists
    DEPRECATED = auto()   # present but scheduled for removal

@dataclass(frozen=True)
class CapabilityEntry:
    feature_id: str
    display_name: str
    state: ActivationState
    reason: str              # WHY it's in this state
    intended: bool = True    # from config — did the user WANT this?
    validate_fn: Optional[Callable[[], bool]] = None  # runtime assertion

    @property
    def is_false_flag(self) -> bool:
        """User asked for this, config says yes, but runtime says no."""
        return self.intended and self.state == ActivationState.INERT
# ... full implementation in docs/FIX_DESIGNS.md
```

Every backend's `load()` calls `_report_capabilities()` which populates the
registry from the *actual* runtime state (not config fields). After `load()`,
`freeze()` prevents mutation. The registry can answer `active_ids()`,
`false_flags()`, and `report_table()` — the latter is rendered into the
Markdown report automatically.

**My key insight (the reconciliation layer):** The CapabilityRegistry tracks
state. But the *divergence* between config-intent and runtime-reality is a
separate signal. A `false_flags()` method returns every entry where
`intended=True` but `state=INERT` — these are the features that MUST be logged
as warnings at startup. Every False Flag becomes a logged warning, not a silent
lie.

#### Layer 2: The load log line reads from the registry

Replace the current misleading log line (`autoregressive.py:831-838`) with:

```python
def _log_capability_summary(self) -> None:
    active = self._caps.active_ids()
    false_flags = self._caps.false_flags()
    logger.info("Active features: %s", sorted(active))
    if false_flags:
        logger.warning("CONFIG/RUNTIME DIVERGENCE — %d features requested but not active:",
                       len(false_flags))
        for ff in false_flags:
            logger.warning("  %s: %s", ff.feature_id, ff.reason)
```

Every line printed here is verified against runtime truth, not config. No more
lying log lines.

#### Layer 3: Auto-generated documentation

The Markdown report renders `capability_registry.report_table()` as a "Real
Optimization State" section. A CI script can run a dry-run smoke test and
auto-update the Feature Status table from the registry — eliminating the
stale-docs problem at the root.

**Why not delete the dead code?** The dead code represents half-finished
features that the roadmap wants to activate (PagedAttention on AR path, CUDA
graphs via torch.compile, INT8 KV-cache). Removing it makes future activation
harder. The correct pattern is registration + state transparency, not deletion
— keep the code, but stop lying about it.

**Migration:** Phase 1 (2-3h): `CapabilityRegistry` + populate in `load()` +
honest log line. Phase 2 (1-2d): validate_fn assertions, auto-generated docs.
Phase 3 (3-5d): refactor config flags into the registry — one source of truth,
no duplication.

Full code sketches in [`docs/FIX_DESIGNS.md` §1](FIX_DESIGNS.md#1-false-flag-architecture).

---

## Flaw #2: Two Parallel Optimization Stacks

### Manifestation

There are **two separate, non-communicating optimization stacks** in the
codebase:

- **Stack A — the AR backend's own features** (`autoregressive.py`): owns
  `_paged_kv`, `_graph_pool`, `_spec_decoder`, `_kv_quant_cache`. Of these,
  only `_spec_decoder` can ever be live (and only under an env gate). The rest
  are hardcoded off, constructed-but-unused, or captured-but-never-replayed.

- **Stack B — the harness-owned ContinuousBatcher path** (`harness.py:680` +
  `continuous_batcher.py` + `paged_attention.py`): the harness builds its **own
  independent** `PagedKVCache` (`harness.py:724`) and a `PagedCache`
  (`DynamicCache`-compatibility shim) and passes it as `past_key_values` to
  `engine.model(...)` directly (`continuous_batcher.py:651`). **This is the only
  place where paged KV is genuinely fed to a model forward.**

The two stacks do not share:
- State (separate PagedKVCache instances)
- Code (`autoregressive.py`'s `_init_paged_attention` vs. `harness.py`'s direct
  construction)
- Configuration (AR's `_use_paged_attention` hardcoded False is never consulted
  by the CB path)
- Lifecycle (the harness allocates and frees CB's paged KV independently of the
  AR backend's lifecycle)

### Root Cause

The Optimization Stack was designed as a *property of the backend* (the AR
backend "has" PagedAttention, CUDA graphs, etc.), but the ContinuousBatcher
needed PagedAttention at the *harness level* (because it manages its own
iteration-level scheduling outside the backend's `translate_batch`). Rather
than refactoring the ownership model, the CB path was built as a parallel
stack that bypasses the backend's optimization objects entirely.

### Concrete Evidence

```python
# Stack A — AR backend (deactivated):
# autoregressive.py:596
self._use_paged_attention: bool = False  # hardcoded

# autoregressive.py:1163
self._paged_kv = PagedKVCache(...)  # only if _use_paged_attention True — never reaches this

# Stack B — harness CB path (active):
# harness.py:724
paged_kv = PagedKVCache(  # <-- separate construction
    num_layers=kv_cfg.get("num_layers", 24),
    ...
)
# harness.py:736
batcher = ContinuousBatcher(self.engine, paged_kv, ...)  # <-- harness-level ownership

# continuous_batcher.py:651
self.engine.model(input_ids=..., past_key_values=self._paged_cache, ...)
# ^^ THIS is where PagedKVCache actually feeds the model
```

### Impact

- **PagedAttention is simultaneously dead (AR) and real (CB).** The README
  couldn't describe this accurately because the architecture makes it impossible
  to describe simply.
- **Fixing one stack doesn't fix the other.** If someone fixes the AR backend's
  PagedAttention, the CB path still uses its own copy.
- **Code duplication.** The CB path reimplements prefill KV population, decode
  step management, and sequence lifecycle — all things the backend theoretically
  owns.
- **Anyone trying to add another optimization** (e.g., prefix caching) must
  decide which stack to put it in, and will likely end up creating a third.

### Fix Approaches

**Approach A: Merge the stacks — backend owns all optimizations**

Move the `PagedKVCache` lifecycle into the `InferenceBackend` protocol as an
optional capability. The `ContinuousBatcher` asks the backend for its paged
cache instead of building its own. The AR backend's `_use_paged_attention`
gate is replaced by a single `self._paged_kv is not None` check.

**Approach B: Extracted optimization layer**

Create an `InferenceOptimizations` object owned by the engine (not the backend
or the harness). Both the backend and the batcher reference the same instance.
Lifecycle is centralized — one allocate, one free.

#### Production Reference: vLLM's BlockPool + KVCacheManager

vLLM has converged on a pattern where ALL GPU memory for inference is managed
through a **single unified block pool**:

- **`BlockPool`** — physical GPU memory as fixed-size `KVCacheBlock` objects.
  A doubly-linked free block queue provides O(1) allocation and LRU eviction.
  Every block has a unique `block_id` for tracking.
- **`KVCacheManager`** — the sole owner of KV-cache allocation. Model runners
  request blocks through the manager; they never allocate directly.
- **No separate stacks** — attention backends, scheduler, and model runner all
  share the same block pool. Nothing is duplicated.

TGI (HuggingFace Text Generation Inference) uses a similar pattern: a
`MemoryTracker` that maintains a budget for weights + KV cache + activations,
with watermarks that trigger batching policy changes before OOM.

**Why this matters for the Two Stacks problem**: Both vLLM and TGI demonstrate
that KV-cache management should be a *shared service*, not a *per-path
property*. When the CB path built its own PagedKVCache, it was re-creating a
problem vLLM already solved — the block pool should be owned once and shared.

---

## Flaw #3: Ad-Hoc Feature Gating System

### Manifestation

Features are gated using **at least 7 different mechanisms** with no
centralization, no lifecycle tracking, and no programmatic query interface:

| # | Mechanism | Example | `file:line` |
|---|---|---|---|
| 1 | Hardcoded boolean `= False` | `_use_paged_attention = False` | `autoregressive.py:596` |
| 2 | `if False:` block with comment | `if False:  # was: self._use_fused_kernels and ...` | `autoregressive.py:761` |
| 3 | Commented-out code | `# _enable_cuda_malloc_async()` | `autoregressive.py:655` |
| 4 | Env var gate | `TR_ENABLE_EXPERIMENTAL_SPECULATIVE != "1"` | `speculative.py:1129` |
| 5 | Config field that doesn't match runtime | `extra["use_paged_attention"]` = True, `_use_paged_attention` = False | `autoregressive.py:786-798` |
| 6 | Captured-but-never-replayed | `_graph_decoder` stored, never called | `autoregressive.py:1227,1548` |
| 7 | Constructed-but-never-accessed | `_kv_quant_cache` created, no `.update()`/`.get()` | `autoregressive.py:1184` |
| 8 | Runtime safety gate | `raise RuntimeError("WILL be corrupted")` | `tensorrt_backend.py:335` |
| 9 | Safe-mode cascade (5 features disabled at once) | `if self._safe_mode: self._use_cuda_graph = False` etc. | `autoregressive.py:610-617` |

### Root Cause

Each feature was gated at the time of disablement with whatever mechanism was
convenient. There was no architectural decision about *how* to gate features,
no template to follow, and no review to ensure consistency. The `--safe-mode`
flag was a later addition that tried to batch-disable features but only covered
5 of them and doesn't interact with env-var gates or safety gates.

The deeper root cause: **there is no "feature" as a first-class concept in the
codebase.** Features aren't objects. They're boolean flags, import guards, and
code blocks scattered across modules. You can't iterate over them, query them,
or test them as a group.

### Impact

- **Disabling one feature can silently enable another.** The classic case:
  disabling `torch.compile` (`9fa3397`) flipped the `not use_torch_compile`
  guard and crashed fused-kernel injection (`804c0a6`). The guard was
  `not self.use_torch_compile`, so turning compile off turned injection on.
- **No way to answer "what's active?"** — the Feature Status table in
  ARCHITECTURE.md had to be produced by a human reading every line. It goes
  stale the moment any code changes.
- **New contributors can't discover gating.** An env var in a different file
  from the feature it gates. A `FutureWarning` on import. A comment in a
  1,859-line file. These are invisible to anyone who doesn't already know.
- **Testing is impossible.** You can't write `test_all_features_gated_correctly()`
  because there's no registry of features to iterate over.

### Fix Approaches

**Approach A: Centralized FeatureRegistry (see Flaw #1, Approach B)**

Every gating site registers with a single registry. The registry can answer
`active_features()`, `gated_features()`, and `why_gated(name)`. A test can
assert that no feature is simultaneously "configured=True" and "active=False"
without a logged reason.

**Approach B: Single gating mechanism — `FeatureToggle`**

Replace all 7+ mechanisms with one `FeatureToggle` enum:

```python
class Feature(Enum):
    PAGED_ATTENTION = "paged_attention"
    FUSED_KERNELS = "fused_kernels"
    CUDA_GRAPH = "cuda_graph"
    CUDA_MALLOC_ASYNC = "cuda_malloc_async"
    # ... all features

class FeatureToggle:
    def __init__(self, feature: Feature, reason_disabled: str = ""):
        self.feature = feature
        self._enabled: bool | None = None  # None = not yet resolved
        self._reason: str = reason_disabled

    def enable(self): ...
    def disable(self, reason: str): ...
    def is_active(self) -> bool: ...

# Usage:
if toggle(Feature.FUSED_KERNELS).is_active():
    self._inject_fused_kernels()
```

The resolution happens once at `load()` time, and the toggle logs its decision.
No more `if False:`, no more commented-out calls, no more env-var-in-different-file.

---

## Flaw #4: Backend Protocol Contract Erosion

### Manifestation

`InferenceBackend` (`protocol.py:174`) is an ABC with three abstract methods:
`load()`, `warmup()`, `translate_batch()`. But the *actual* contract that the
harness, engine, and quality benchmark depend on is much larger — and
**completely unenforced**.

#### 4.1 No `_configured_batch_size` enforcement

The engine reads `self._backend._configured_batch_size` with **no default**
(`engine.py:148-156`). A new backend that forgets to initialize this attribute
crashes at runtime. Two backends (NLLB, Diffusion) *default to 1 only by
accident* (via `getattr` in their own `warmup` methods, not at init).

#### 4.2 Divergent `input_tokens_total` semantics

| Backend | Computation | Correct? | `file:line` |
|---|---|---|---|
| Autoregressive | `len(batch.input_ids[i])` — **padded length** | ❌ Inflates TPS | `autoregressive.py:1688` |
| TensorRT | `len(batch.input_ids[i])` — **padded length** | ❌ Inflates TPS | `tensorrt_backend.py:413` |
| NLLB | `attention_mask[i].sum()` — **real tokens** | ✅ | `nllb.py:436` |
| Diffusion | `int(src_lens[i])` — **real tokens** | ✅ | `diffusion.py:476` |

Half the backends silently violate the contract. The protocol says nothing about
what `input_tokens_total` means. The ABC doesn't even declare it — it's only
implied by the `BatchGenerationOutput` return type.

#### 4.3 Divergent `phase_timings` format

| Backend | `phase_timings` dict keys | `file:line` |
|---|---|---|
| AR | `{"prefill_ms", "decode_ms", "total_gpu_ms", "method"}` | `autoregressive.py:1695` |
| NLLB | `{}` (empty dict) | `nllb.py:456` (no phase_timings passed) |
| Diffusion | `{"encode_ms", "denoise_ms", "denoise_steps", "ms_per_step"}` | `diffusion.py:448-485` |
| TRT | `{"engine": "tensorrt"}` | `tensorrt_backend.py:420` |

No consumer can safely read `phase_timings["prefill_ms"]` without knowing which
backend produced it. The protocol declares `phase_timings: dict[str, float]`
with no schema.

#### 4.4 Declarative-only capability bitmask

`ModelCapability` (`protocol.py:66`) is an `IntFlag` bitmask. Each backend sets
it. **Nothing reads it programmatically.** If a backend claims `CONFIDENCE` but
doesn't implement `get_token_log_probs()`, the first caller will get
`NotImplementedError` at runtime.

#### 4.5 No return-type validation

`translate_batch()` is declared to return `BatchGenerationOutput`, but **no
code in the engine, harness, or quality benchmark validates this**. A backend
that returns `None` or a plain `dict` would silently propagate until something
tries to access `.output_tokens_total` and crashes with `AttributeError`.

#### 4.6 Abstract methods are too few

Only `load`, `warmup`, and `translate_batch` are `@abstractmethod`. But the
harness also depends on `is_loaded()`, `kv_cache_config`, `devices`,
`encode_source()`, etc. — all of which are optional or have default
implementations that raise `NotImplementedError`.

### Root Cause

The protocol was designed incrementally. Each backend was added, and when a new
requirement emerged (e.g., `phase_timings`), it was added to `BatchGenerationOutput`
without retroactively enforcing a schema on existing backends. The capability
bitmask was a noble attempt to solve this, but it's never actually checked.

### Impact

- **Silently wrong benchmark output.** AR/TRT inflate TPS, which flows into the
  extrapolation → understates days-to-completion. The entire point of the
  benchmark is compromised for half the backends.
- **New backends are dangerous.** Nothing prevents a new backend from making
  any of these mistakes because nothing checks.
- **No interop between backends and consumers.** The quality benchmark,
  Prometheus exporter, and aggregator all have to defensive-code around
  `phase_timings` being differently shaped.

### Fix Approaches

**Approach A: Backend compliance test suite (recommended — quick win)**

A `pytest` module that imports every registered backend, instantiates it with
a mock config, and verifies:

```python
def test_backend_compliance(backend_class):
    backend = backend_class(mock_config())
    assert hasattr(backend, '_configured_batch_size'), "must init in __init__"
    # Run a mock batch, verify:
    result = backend.translate_batch(mock_batch())
    assert isinstance(result, BatchGenerationOutput)
    assert result.input_tokens_total > 0
    assert isinstance(result.phase_timings, dict)
    # Capability check:
    if ModelCapability.CONFIDENCE in backend.capabilities:
        assert not isinstance(backend.get_token_log_probs, type(NotImplementedError))
```

Cost: ~200 lines. Catches 80% of contract violations at test time.

**Approach B: Runtime validation in the engine**

After `load()`, the engine calls `validate_backend(backend)` which checks all
invariants (configured_batch_size, capability-implies-method, return types).

**Approach C: Typed protocol with Beartype/Pydantic enforcement**

Use a `typing.Protocol` or runtime type-checker to validate at the boundary.

#### Production Reference: vLLM's `validate_configuration()` + ONNX Runtime Schema Enforcement

##### vLLM Attention Backend ABC

vLLM enforces its attention-backend contract with a three-layer architecture:

1. **ABC inheritance gate** — every backend MUST subclass `AttentionBackend(ABC)`.
   The registry stores fully-qualified class paths; `get_class()` imports and
   verifies the type at registration time.
2. **`validate_configuration()` classmethod** — called during model startup for
   EVERY registered backend. Returns a list of validation errors. If ANY backend
   fails, the server refuses to start. This catches capability mismatches
   (backend claims SDPA support but CUDA arch is too old) before the first token.
3. **Runtime type enforcement** — `AttentionMetadata` is a per-backend typed
   dataclass. The scheduler constructs the correct type for the active backend.
   Mypy enforces this at static-analysis time.

##### ONNX Runtime Execution Provider Contract

ONNX Runtime enforces backend (Execution Provider) contracts through:

- **Op schema registration** — every kernel declares its supported ops, input
  shapes, and data types in a JSON schema. The registry validates at load time.
- **`KernelRegistry`** — maps `(op_type, provider, dtype, device)` → kernel
  function. If no kernel matches, the graph is partitioned across providers
  with a clear error for unsupported ops.
- **`GetCapability()` interface** — providers declare which graph nodes they
  can execute. The graph partitioner checks this, never assumes.

**Key takeaway for this codebase**: The `validate_configuration()` pattern
(vLLM) and `GetCapability()` pattern (ONNX RT) both say: *validate at
registration time, not at first call time.* The `ModelCapability` bitmask in
this codebase should be checked at backend registration, and the backend should
be rejected if its implementation doesn't match its claimed capabilities.

---

## Flaw #5: The God Harness

### Manifestation

`BenchmarkHarness` (`orchestration/harness.py`) is **1,178 lines, 16 methods**.
It directly constructs and owns:

| Subsystem | Construction site | `file:line` |
|---|---|---|
| `InferenceEngine` | `self.engine = InferenceEngine(...)` | `:217` |
| `BatchSizeTuner` | `tuner = BatchSizeTuner()` | `:266` |
| `JSONLLoader` | 2× (standard + CB paths) | `:378`, `:696` |
| `TextChunker` | 2× | `:388`, `:703` |
| `ChunkFilter` | 2× | `:393`, `:708` |
| `AsyncPipeline` | 2× | `:397`, `:712` |
| `MetricsCollector` | `self.metrics = MetricsCollector(...)` | `:1036` |
| `CheckpointManager` | `self.checkpoint_mgr = CheckpointManager(...)` | `:1042` |
| `SignalHandler` | `self.signal_handler = SignalHandler()` | `:1046` |
| `QualityBenchmark` | `quality_bench = QualityBenchmark(...)` | `:611` |
| `MetricsAggregator` | `aggregator = MetricsAggregator(...)` | `:626`, `:921` |
| `ExtrapolationModel` | 2× (standard + CB paths) | `:629`, `:924` |
| `JSONReportWriter` | 2× | `:674`, `:965` |
| `MarkdownReportWriter` | 2× | `:675`, `:966` |
| `PrometheusExporter` | `PrometheusExporter(...)` | `:1021` |
| `PagedKVCache` (CB path) | `paged_kv = PagedKVCache(...)` | `:724` |
| `ContinuousBatcher` (CB path) | `batcher = ContinuousBatcher(...)` | `:736` |
| `PrecisionTimer` | 2× | `:425`, `:748` |

**Every subsystem is a local variable or attribute constructed inline.**

Observability is bolted on with **11 direct `self._prometheus.record_*()` call
sites** scattered through the translation loop (`:482`), heartbeat logic
(`:561`), OOM handler (`:1073`), quality path (`:615`), and CB path (`:857`,
`:876`, `:896`, `:905`).

### Root Cause

The harness grew organically. Each new capability (Prometheus, continuous
batching, checkpointing, OOM recovery) was added as a method or inline code in
the same class because that was the path of least resistance. There was never a
refactoring pass to extract concerns.

### Impact

- **The harness cannot be unit-tested.** Its methods construct their dependencies
  inline — there's no injection point for mocks.
- **Adding a feature requires touching the harness.** Want a new metric? Wire it
  into the translation loop body. Want a new observability exporter? Another
  `record_*()` call site.
- **The CB and standard paths are structurally duplicated** (see `_run_translation_core`
  vs `_run_continuous_batching_loop` — they share no code for metrics, checkpoint,
  heartbeat, or report generation, despite doing the same things).
- **Observability is not an observer** — it's hardwired into the subject.

### Fix Approaches

**Approach A: Extract an event bus (recommended medium-term)**

Replace the 11 direct `self._prometheus.record_*()` calls with an event bus:

```python
class BenchmarkEvent(Enum):
    BATCH_COMPLETED = auto()
    PIPELINE_HEARTBEAT = auto()
    ERROR_OCCURRED = auto()
    QUALITY_COMPUTED = auto()

class EventBus:
    def emit(self, event: BenchmarkEvent, **data): ...
    def subscribe(self, event: BenchmarkEvent, handler: Callable): ...

# Harness — one line:
self.events.emit(BenchmarkEvent.BATCH_COMPLETED, tokens=result.output_tokens_total, ...)

# Prometheus — separate module, subscribes:
events.subscribe(BenchmarkEvent.BATCH_COMPLETED, self.record_batch)
```

Cost: ~300 lines. Decouples observability. Makes the harness testable.

**Approach B: Dependency injection**

Pass subsystems into the harness constructor instead of constructing them inline.
The harness becomes orchestrator-only (no construction). Tests inject mocks.

**Approach C: Subsystem coordinator pattern**

Each concern (metrics, checkpoint, pipeline, quality) becomes a separate
coordinator object owned by the harness. The harness delegates, not implements.

---

## Flaw #6: Config/Source-of-Truth Confusion

### Manifestation

Multiple systems disagree about the state of the system. The most damaging
instance:

**`precision_config.uses_fp8` lies under `--safe-mode`:**

```python
# precision.py computes uses_fp8=True based on config dtype
# autoregressive.py:751 — the guard:
if self.precision_config.uses_transformer_engine and not self._safe_mode:
    self._apply_te_fp8()  # FP8 te.Linear replacement
```

Under `--safe-mode`, `_apply_te_fp8()` is skipped, but
`precision_config.uses_fp8` remains `True`. Every consumer that reads
`uses_fp8` to decide behavior (matmul precision, memory estimates, log
output) gets the wrong answer.

**Config field vs. runtime boolean divergence:**

```python
# autoregressive.py:596 — the runtime truth:
self._use_paged_attention: bool = False  # hardcoded

# autoregressive.py:831-838 — the log line reads the CONFIG, not the runtime:
extra.get("use_paged_attention", False)  # can be True!
```

The `extra` dict from config can have `use_paged_attention=True`, while the
runtime boolean that actually gates the feature is hardcoded `False`. The log
line prints the config value, creating the False Flag.

**Architecture defaults disagree:**

| Source | Default layers | Default KV heads | Model size assumed |
|---|---|---|---|
| `constants.py:12-15` | 36 | 4 | 12B-class |
| `model_presets.py` | Varies by preset | Varies | Varies |
| Default model (`__main__.py`) | — | — | 4B (`translategemma-4b-it`) |

The constants module defaults to 12B-class dimensions. The default model is 4B.
Downstream code that reads `DEFAULT_NUM_LAYERS` gets wrong dimensions for the
model it's actually running.

### Root Cause

The system has at least 4 independent sources of truth for any given value:

1. `config.yaml` / `ModelConfig` (the user's intent)
2. `config.extra` dict (free-form backend-specific overrides)
3. Runtime booleans (`self._use_paged_attention`, etc.)
4. The actual code path (does `_extreme_decode` call `graph.replay()`?)

These can all disagree, and nothing reconciles them. The `PrecisionConfig` was
a step toward centralizing one dimension (precision), but it's not a general
pattern — every other config dimension has its own bespoke resolution.

### Impact

- Consumers make decisions on wrong information (memory estimates assuming FP8
  when BF16 is active).
- Logs and dashboards display config-intent, not runtime-reality.
- Developers debugging "why is this slow?" check the config, see
  `use_paged_attention: true`, and assume memory savings that don't exist.

### Fix Approaches

**Approach A: Runtime reconciliation snapshot (see Flaw #1, Approach A)**

After `load()`, compute a `RuntimeSnapshot` that IS the truth. Every consumer
(including log lines and the Prometheus exporter) reads from this snapshot.
Config fields are input; the snapshot is output. Config-vs-snapshot divergence
is logged as a warning at load time.

**Approach B: Immutable config → derived state pattern**

Freeze `BenchmarkConfig` after parsing. All derived state (effective precision,
effective batch size, active features) is computed once and stored in
`ResolvedConfig`. No code path reads from the raw config after load.

**Approach C: Model config as the primary source**

For architecture dimensions, always read `model.config` (HF `PretrainedConfig`)
at runtime, never assume constants. The constants file provides *fallbacks only*
when `model.config` is unavailable.

---

## Flaw #7: Extrapolation Reliability Gap

### Manifestation

The entire benchmark exists to answer "how many days to translate 6.23T tokens?"
The extrapolation pipeline has **four independent correctness problems**:

#### 7.1 Padded token inflation (see Flaw #4.2)

AR and TRT backends compute `input_tokens_total` from the padded tensor length,
not the real token count. This inflates the denominator in `tokens/second`,
producing **optimistic TPS numbers and understated days-to-completion** for the
two most commonly used backends.

#### 7.2 Constant-throughput assumption

`ExtrapolationModel.compute()` (`reporting/extrapolation.py:78`) uses the
point-estimate mean TPS and assumes it holds constant for the entire multi-week
run. It does not model:

- Thermal throttling (GPU clocks drop after sustained load)
- KV-cache growth (memory pressure increases with sequence length over time)
- Queue dynamics (data starvation varies with input distribution)
- The fact that the 2-hour sample may not reach steady-state

The Markdown report honestly renders: *"Extrapolation assumes constant throughput
and 24/7 operation"* — but this is a fundamental limitation, not a caveat.

#### 7.3 Latency distribution is discarded

`ThroughputTracker` (`metrics/throughput.py`) has `p50_latency_ms` and
`p99_latency_ms` fields. They are **always `None`** because
`MetricsCollector.log_batch()` (`collector.py:84-86`) never passes `latency_ms`.

#### 7.4 No throughput trend detection

There is no mechanism to detect whether throughput is stable, improving, or
degrading over the 2-hour window. The bootstrap CI captures *variance*, not
*drift*. A run whose throughput is steadily declining (thermal throttling) could
have the same mean as a stable run and produce the same extrapolation.

### Root Cause

The extrapolation was designed as a simple `total_tokens / tps / 86400` division
with CI propagation. Adding trend detection, degradation modeling, and per-batch
TPS time-series analysis was never scoped.

### Impact

The single number the benchmark exists to produce — the days estimate — is
**reliability-compromised** on at least two axes (padded tokens for AR/TRT,
constant-throughput for all backends). A user who reads "90 days" might actually
need 120 days once padding is fixed and degradation is modeled.

### Fix Approaches

**Approach A: Trend-aware extrapolation (recommended immediate fix)**

1. Fix `input_tokens_total` in AR/TRT (use `attention_mask.sum()` — trivial, ~2 lines each).
2. Pass `latency_ms` from `log_batch()` to `ThroughputTracker` (trivial, 1 line).
3. Add linear regression over per-batch TPS values; detect degradation slope.
   Report "if current trend continues" extrapolation alongside the constant
   estimate.
4. Report a conservative lower-bound extrapolation (e.g., use the lower 95%
   bootstrap CI, not the point estimate).

Cost: ~100 lines of changes. Fixes the four specific problems above.

**Approach B: Multi-phase benchmark with steady-state verification**

Run a short warmup phase (discard), a 2-hour measurement phase, and an optional
extended tail phase. If throughput in the tail phase differs from the measurement
phase by >5%, flag the extrapolation as unreliable.

**Approach C: Throughput curve modeling**

Model throughput as a function of time: `tps(t) = tps_0 * e^(-λt)` or similar
(for thermal degradation). Fit the model to per-batch TPS time-series data.
Extrapolate via integration of the fitted curve.

#### Production Reference: MLPerf Inference + TPC Benchmark Methodology

##### MLPerf Inference (MLCommons)

MLPerf Inference rules explicitly **prohibit linear extrapolation**. Key rules:

- **Duration**: minimum 600 seconds (10 minutes) for all scenarios. Shorter
  runs are invalid.
- **Sample requirements**: at least 24,576 samples for Offline, ~270,336 for
  Server at 99th percentile. Under-sampling invalidates results.
- **Anti-extrapolation**: *"Results from shorter runs must not be scaled to
  estimate longer-run performance."* The measured throughput IS the result.
- **Steady-state verification**: the load generator must reach steady state
  before measurement begins. Warmup samples are discarded.
- **Latency constraints**: Offline has no latency bound but Server has a
  99th-percentile latency target that must be met simultaneously with the
  throughput target.

**Relevance**: The H200 benchmark's 2-hour window and extrapolation to
multi-week runs is the exact opposite of MLPerf's methodology. MLPerf says
"measure what you actually run." The benchmark says "run 2 hours, predict 90
days." While the use case is different (prediction, not competition), MLPerf's
steady-state verification and anti-extrapolation rigor are applicable.

##### TPC (Transaction Processing Performance Council)

TPC benchmarks require: (a) **Minimum scale** — the system must process a
minimum data volume proportional to its rated performance, (b) **Steady-state
measurement** — a defined measurement interval after warmup, (c) **Price/Performance
disclosure** — both cost and performance must be reported together, (d) **Full
disclosure report** — all configuration, tuning, and measurement methodology
must be published.

**Relevance**: TPC's minimum-scale and steady-state requirements apply
directly. The H200 benchmark should verify that its 2-hour run reached
steady-state before computing the extrapolation.

##### Recommended Fix Strategy for This Codebase

The agent research converged on a three-phase approach:

**Phase 1 (Immediate): Conservative lower-bound + fix padded tokens.**
Replace `input_tokens` inflation with real token counts. Report a conservative
extrapolation (lower 95% bootstrap CI, not the point estimate) as the primary
number. This is the safe, honest answer while the framework improves.

**Phase 2 (Short-term): Trend detection.** Fit linear regression over per-batch
TPS. Detect degradation slope. Flag extrapolation as unreliable if the slope is
negative beyond a threshold. Report "if trend continues" alongside the constant
estimate.

**Phase 3 (Medium-term): Multi-phase measurement.** Run a discard warmup, a
fixed measurement interval, and an extended tail. Compare tail throughput to
measurement throughput. Flag degradation.

---

## Flaw #8: Fragmented Memory Architecture

### Manifestation

GPU memory management is spread across **4 independent subsystems** with no
unified planning or budget:

| Subsystem | What it manages | Where | How |
|---|---|---|---|
| Model loader | Weight tensors | `autoregressive.py:907` | `device_map="auto"` or single-GPU fast path |
| KV cache (standard) | HF `DynamicCache` per batch | Inside `model.generate()` or the decode loop | Implicit — HF manages it |
| PagedKVCache (CB) | Block-level KV storage | `paged_attention.py`, `harness.py:724` | Explicit block allocation, LIFO free list |
| PinnedBufferPool | Host→device DMA buffers | `data/pipeline.py:68` | CUDA pinned memory pool |

These subsystems do not share a memory budget. The OOM handler
(`harness.py:1060`) halves the batch size but does not account for:

- KV-cache memory growth at the new batch size
- PagedKVCache block consumption at the new batch size
- Whether the OOM was a weight-memory OOM or a KV-cache OOM (different fixes)

The single-GPU fast path (`autoregressive.py:919`) is a sharp architectural
edge: if the estimated model size is <10% of one GPU's memory, it loads on a
single GPU with `device_map=None`. For a 4B model on a 141GB H200, this
essentially always triggers — meaning the "2×H200" story is bypassed for every
model this benchmark actually runs.

MPS memory management is entirely ad-hoc: single-probe batch tuning, sequential
iteration, per-batch `empty_cache()`, and the `TR_MPS_MEMORY_SAFE` env var are
workarounds, not architecture.

### Root Cause

Memory was managed wherever memory happened to be allocated — in the model
loader, in the decode loop, in the CB path. There was never a unifying
`MemoryPlanner` that owns all allocation decisions.

### Impact

- OOM recovery is crude (halve-and-retry) and may not fix KV-cache OOMs.
- Memory estimation for multi-GPU is wrong (the 10% threshold bypasses
  multi-GPU for all practical models).
- Adding a new memory consumer (e.g., prefix cache) risks OOM with no budget
  checking.
- MPS development hits memory issues that the architecture should prevent.

### Fix Approaches

**Approach A: Unified MemoryPlanner (recommended medium-term)**

```python
class MemoryPlanner:
    def __init__(self, total_memory_gb: float, fraction: float = 0.95):
        self._budget = total_memory_gb * fraction
        self._allocated = {"weights": 0, "kv_cache": 0, "pinned": 0, "other": 0}

    def reserve(self, category: str, bytes_needed: int) -> bool: ...
    def available(self) -> float: ...
    def snapshot(self) -> dict: ...
```

Every allocation path queries the planner. The OOM handler asks "which category
exceeded budget?" and adjusts accordingly (reduce batch size for KV-cache OOM,
reduce model precision for weight OOM).

**Approach B: Fixed-budget allocator with watermarks**

Pre-allocate a fixed fraction of GPU memory at startup. Subsystems draw from
this pool with soft and hard watermarks. When a watermark is hit, the subsystem
degrades (reduce batch, evict cache blocks) rather than OOMing.

#### Production Reference: vLLM BlockPool + TensorRT-LLM Memory Planning

##### vLLM's Unified BlockPool

vLLM manages ALL GPU memory through a single `BlockPool` (`vllm/v1/core/block_pool.py`):

- Physical GPU memory is a pool of fixed-size `KVCacheBlock` objects
- A doubly-linked free block queue provides O(1) allocation and LRU eviction
- Every block has a unique `block_id` for tracking
- The `KVCacheManager` is the sole owner of KV-cache allocation — model
  runners request blocks, never allocate directly
- Prefix caching reuses blocks across sequences via content-hash matching,
  all within the same pool

##### TensorRT-LLM's Fixed-Budget Planning

TensorRT-LLM pre-plans all GPU memory at engine-build time:

1. **Weights** — fixed size, known at build time
2. **KV cache** — `max_batch_size × max_seq_len × bytes_per_token × num_layers × 2`
3. **Activations/workspace** — bounded by the engine's memory footprint

The engine refuses to initialize if the total exceeds GPU memory. No dynamic
allocation. No OOM at runtime.

##### TGI's Static Memory Budgeting

HuggingFace TGI uses `MemoryTracker` with watermarks:
- **Hard watermark**: absolute maximum. Above this → reject new requests.
- **Soft watermark**: triggers batching policy changes (reduce max batch size,
  evict cached blocks) BEFORE hitting the hard limit.

**Key takeaway for this codebase**: All three production systems share a
pattern — **allocate a fixed pool, plan budgets statically, degrade gracefully
before OOM.** The H200 codebase does the opposite: allocate reactively, halve
batch size on OOM, no pre-planning. The MemoryPlanner fix (Approach A) adopts
these patterns.

---

## Flaw #9: Hardcoded Architectural Assumptions

### Manifestation

Several modules are hardcoded to specific model architectures:

#### 9.1 `parallelism.py` hardcoded to Gemma-3-12B

```python
# parallelism.py:37-50 — hardcoded constants:
_NUM_LAYERS = 48
_HIDDEN_SIZE = 3840
_NUM_ATTENTION_HEADS = 16
_NUM_KV_HEADS = 8
_INTERMEDIATE_SIZE = 15360
_VOCAB_SIZE = 262144
```

The layer-mismatch branch (`parallelism.py:364-367`) assigns to read-only
`@property` fields → `AttributeError` for any non-48-layer model. This is
latent only because `apply_tensor_parallelism` is never called (see Flaw #1).

#### 9.2 Architecture defaults mismatch

`constants.py` defaults to 12B-class dimensions (`DEFAULT_NUM_LAYERS=36`,
`DEFAULT_NUM_KV_HEADS=4`, `DEFAULT_HEAD_DIM=256`, `DEFAULT_HIDDEN_SIZE=2560`).
The default model is `google/translategemma-4b-it` (4B-class). Any code path
that falls back to `DEFAULT_NUM_LAYERS` gets wrong dimensions.

#### 9.3 GPU capability misdetection

```python
# jit_compiler.py:102
if hasattr(props, 'multi_processor_count'):
    arch_str = arch_str + 'a'  # wgmma
```

`multi_processor_count` is **always present** on any CUDA device. This
misdetects every Hopper GPU as `sm90a` regardless of whether the kernel
actually uses wgmma instructions.

#### 9.4 `END_OF_TURN_TOKEN_ID = 106`

This was hardcoded in 3+ places (now partially fixed — imported from
`constants.py:82`). The value 106 is specific to Gemma-family tokenizers; a
non-Gemma model would silently get wrong EOS behavior.

### Root Cause

The codebase started as a TranslateGemma-12B-specific benchmark. When it became
model-agnostic, architecture constants were not systematically replaced with
`model.config` reads. Some were centralized to `constants.py` (good), but the
constants file was populated with 12B-class defaults (wrong for the default
model).

### Impact

- Running on a non-Gemma model hits latent bugs (wrong KV-cache config, wrong
  EOS token, wrong architecture constants).
- The "model-agnostic" claim is aspirational — the code still carries
  TranslateGemma-12B DNA.
- GPU capability detection is wrong in a way that only manifests in edge cases
  (non-wgmma kernel on Hopper).

### Fix Approaches

**Approach A: Always read `model.config` (recommended quick fix)**

Replace every `DEFAULT_NUM_LAYERS` / `DEFAULT_HEAD_DIM` / etc. fallback with
`model.config.num_hidden_layers` / `model.config.head_dim` (or compute from
`hidden_size / num_attention_heads`). The constants file provides fallbacks
only when `model.config` is genuinely unavailable (e.g., before model load).

**Approach B: Preset-driven architecture resolution**

Extend `get_preset_by_model_id()` to be the single source of architecture
defaults. Every module that needs `num_layers` calls through the preset
resolver. No module reads `DEFAULT_*` from constants directly.

---

## Flaw #10: Observability of the Wrong Things

### Manifestation

The observability system collects the wrong data at the wrong resolution:

#### 10.1 Dead quality gauges

`harness.py:615` hardcodes `bleu=None, chrf=None` when calling
`record_quality()`. The `quality_bleu` and `quality_chrf` Prometheus gauges
are **never populated** — they sit at 0.0 forever. Meanwhile,
`quality/benchmark.py:283-284` *does* compute BLEU and chrF++ and stores them
in `QualityResults`. A Prometheus dashboard showing BLEU=0.0 is a lie.

#### 10.2 Instantaneous TPS gauge

The Prometheus `throughput_tps` gauge captures a **single batch's instantaneous
TPS** (`tokens / latency_ms * 1000`). Prometheus scrapes at 15-60s intervals.
The gauge reflects whatever batch happened to complete in the scrape window —
essentially random noise.

#### 10.3 Log lines are config-state, not runtime-state

As documented in Flaw #1: the load log line prints config values, not hot-path
activation. The observability system inherits this falsehood.

#### 10.4 Throughput tracker discards latency

`ThroughputTracker` has p50/p99 latency infrastructure but it's never fed data.
The latency distribution of the benchmark is invisible.

#### 10.5 Markdown report omits metrics

The Markdown quality table only renders BLEU, chrF++, and COMET-22
(`markdown_report.py:113-119`). BERTScore and COMET-Kiwi are computed and
stored in the JSON report but omitted from the human-readable Markdown.

### Root Cause

Observability was added incrementally. Each `record_*()` call was wired at the
point of need, often with hardcoded `None` for values that weren't easily
available at that call site. There was no "what does the dashboard user actually
need to see?" design pass.

### Impact

- Operators monitoring the Prometheus dashboard see zero BLEU/chrF scores and
  believe quality evaluation failed.
- The throughput gauge is near-useless for capacity planning.
- The Markdown report — the human-facing output — omits two of five quality
  scores.
- No one can answer "what was the p99 latency?" from a run.

### Fix Approaches

**Approach A: Fix the wiring (recommended immediate fix)**

1. `harness.py:615`: pass the actual BLEU/chrF scores from `quality_results`.
2. Prometheus throughput: use a 60s rolling average (from `ThroughputTracker.current()`)
   instead of instantaneous per-batch TPS.
3. `log_batch()`: pass `latency_ms` to the tracker.
4. `markdown_report.py`: add BERTScore and COMET-Kiwi rows.

Cost: ~30 lines. All trivial fixes.

**Approach B: Observability review pass**

Systematic audit of every gauge/counter/histogram: does it reflect runtime
reality? Is it at a useful aggregation level? Is it actually populated?

---

## 12. Flaw Interaction Graph

```
                    ┌──────────────────────────┐
                    │  #1: False Flag Arch     │◄── root cause
                    │  #3: Ad-Hoc Gating       │◄── root cause
                    └──────────┬───────────────┘
                               │ enables
            ┌──────────────────┼──────────────────┐
            │                  │                  │
            ▼                  ▼                  ▼
   ┌────────────────┐  ┌──────────────┐  ┌──────────────────┐
   │#2: Two Stacks  │  │#4: Protocol  │  │#6: Config Truth  │
   │                │  │    Erosion   │  │    Confusion     │
   └───────┬────────┘  └──────┬───────┘  └────────┬─────────┘
           │                  │                    │
           └──────────────────┼────────────────────┘
                              │ causes
            ┌─────────────────┼─────────────────┐
            │                 │                 │
            ▼                 ▼                 ▼
   ┌────────────────┐ ┌──────────────┐ ┌──────────────────┐
   │#7: Extrapolation│ │#8: Memory   │ │#10: Wrong        │
   │   Gap           │ │   Fragmented│ │   Observability  │
   │(THE WHOLE POINT)│ │             │ │                  │
   └────────────────┘ └──────────────┘ └──────────────────┘

   ┌────────────────┐
   │#5: God Harness │◄── makes everything harder to fix
   └────────────────┘

   ┌────────────────┐
   │#9: Hardcoded   │◄── independent, amplifies #4 and #8
   │   Assumptions  │
   └────────────────┘
```

**Key causal chain:**

- **#1 + #3 → #4**: Because features are gated ad-hoc with no contract
  enforcement, backends diverge (some compute padded tokens, some real).
  There's no check to catch this.
- **#1 + #3 → #2**: Each feature was gated off individually, not removed.
  When the CB path needed paged KV, it couldn't re-use the AR backend's
  (dead) implementation, so it built a separate one.
- **#4 → #7**: Half the backends inflate TPS → extrapolation is wrong.
- **#3 → #6**: Ad-hoc gating means config fields and runtime booleans
  diverge with no reconciliation.
- **#1 + #2 + #6 → #10**: Log lines print config-state (from #6) for
  features that are gated off (#1) or implemented in a different stack (#2),
  so observability reports false state.
- **#5 amplifies everything**: Every fix needs to touch the 1,178-line
  harness. The duplicated CB path means every fix must be applied twice.

---

## 13. Prioritized Fix Roadmap

### Tier 1: Do Now (fixes the benchmark's output) — <2 weeks

| Priority | Fix | Flaw addressed | Effort |
|---|---|---|---|
| **P0** | Fix `input_tokens_total` in AR/TRT (use `attention_mask.sum()`) | #4, #7 | 4 lines |
| **P0** | Pass `latency_ms` to ThroughputTracker from log_batch | #7, #10 | 1 line |
| **P1** | Wire BLEU/chrF scores into Prometheus record_quality | #10 | 2 lines |
| **P1** | Add BERTScore + COMET-Kiwi rows to Markdown report | #10 | 10 lines |
| **P1** | Add linear throughput-trend detection + conservative lower-bound extrapolation | #7 | ~80 lines |
| **P2** | Use 60s rolling average for Prometheus throughput gauge, not instantaneous | #10 | 3 lines |

### Tier 2: Do Next (fixes the root causes) — 2-6 weeks

| Priority | Fix | Flaw addressed | Effort |
|---|---|---|---|
| **P0** | RuntimeSnapshot: compute feature activation truth after load(); log from it | #1, #3, #6 | ~200 lines |
| **P1** | Backend compliance test suite (validate all invariants for every backend) | #4 | ~200 lines |
| **P1** | FeatureRegistry: single registry for all feature gates; query API | #1, #3 | ~400 lines |
| **P2** | Merge the two optimization stacks (CB path delegates to backend for paged KV) | #2, #8 | ~300 lines |
| **P2** | Read architecture dimensions from model.config, not constants.DEFAULT_* | #9 | ~50 lines |

### Tier 3: Do Eventually (structural refactors) — 2-3 months

| Priority | Fix | Flaw addressed | Effort |
|---|---|---|---|
| **P1** | Event bus to decouple observability from harness | #5 | ~300 lines |
| **P1** | Unified MemoryPlanner across weight loading, KV cache, and pinned buffers | #8 | ~500 lines |
| **P2** | Extract harness concerns into coordinator objects (PipelineCoordinator, MetricsCoordinator, etc.) | #5 | ~800 lines |
| **P2** | Immutable config → ResolvedConfig pattern (freeze after load, compute derivations once) | #6 | ~400 lines |
| **P3** | Single FeatureToggle mechanism (replace all 7+ gating mechanisms) | #3 | ~600 lines |
| **P3** | Observability review pass (every metric verified against runtime reality) | #10 | ~200 lines |

---

## References

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — reality-grounded architecture and Feature Status table
- [`AI_CODING_ANTIPATTERNS.md`](AI_CODING_ANTIPATTERNS.md) — concrete mistakes (especially A1, A3, A6, A11)
- [`docs/README.md`](README.md) — documentation navigation
- Audit memory: 7 rounds of findings, systemic patterns documented in `h200-ruthless-audit-round7.md`
- Git log: commits `804c0a6`, `9fa3397`, `ffa707b`, `5ec301c`, `ecbade9`, `92d9476`

---

---

## 14. Merged Fix Designs — Research × Architecture Synthesis

> This section interweaves findings from 11 deep-research agents (606 tool
> calls, 707K tokens) studying vLLM, TGI, MLPerf, ONNX Runtime, TensorRT-LLM,
> and OpenFeature with my own systems-architecture analysis. The per-flaw
> "Fix Approaches" above are the quick/medium-term triage. This section is the
> **deep synthesis** — the merged design for the long-term architecture. Full
> code sketches live in [`docs/FIX_DESIGNS.md`](FIX_DESIGNS.md).

---

### 14.1 Flaws #1 + #3 (False Flag + Ad-Hoc Gating): Unified Feature Truth System

**My architectural insight**: The False Flag problem isn't about a missing
registry — it's about a missing **feedback loop** between config-intent and
runtime-state. A registry tells you what's active. A *reconciliation layer*
tells you what's *diverged* — what the user asked for vs. what they actually got.

**Research contribution**: The CapabilityRegistry (FIX_DESIGNS §1) with
`ActivationState` enum and `validate_fn` callbacks. Each backend's `load()`
populates it with VERIFIED states, not aspirational ones. After `freeze()`, the
registry is immutable — downstream consumers read from it, never from raw config.

**Merged design — three layers, one truth:**

1. **CapabilityRegistry** (state tracking): `CapabilityEntry` with `feature_id`,
   `state` (ACTIVE/INERT/DEPRECATED), `reason`, `intended` (from config),
   `validate_fn`. Every feature ~40 lines of registration. `is_false_flag()`
   returns entries where `intended=True, state=INERT` — the divergence signal.

2. **Reconciliation at load()**: After freeze, compute `diverged =
   [e for e in entries if e.is_false_flag]`. Log each as WARNING. The load log
   line reads from `active_ids()`, not from config booleans. The H200_SETUP
   dry-run log's misleading `PagedAttn=True` line becomes impossible — every
   flag printed is verified.

3. **Auto-generated documentation**: `cap.report_table()` renders as Markdown.
   A CI script runs a dry-run smoke test, extracts the registry, and updates
   the Feature Status table automatically. The stale-docs problem is eliminated
   at the root because truth is machine-readable, not manually maintained.

**Migration**: Phase 1 (2-3h): add `CapabilityRegistry` + populate in `load()`
+ honest log line. Phase 2 (1-2d): `validate_fn` assertions, auto-generated docs,
CI gate. Phase 3 (3-5d): migrate all config flags into the registry — the
registry becomes the single source of truth for both config AND runtime state.

---

### 14.2 Flaw #2 (Two Parallel Stacks): PagedCache Unification

**My architectural insight**: The two-stacks problem is a missing abstraction.
Define a `KVManager` protocol — the AR backend and CB path both implement it.
The harness composes them; it doesn't duplicate them.

**Research contribution**: The unification path — make PagedAttention the default
KV-cache for ALL CUDA paths (FIX_DESIGNS §2). Wrap the AR `_extreme_decode`
prefill output into `PagedCache` (which implements HF's `Cache` protocol) and
pass it as `past_key_values`. The model's attention layers call
`cache.update(key, value)`, which routes through `PagedLayer` → `PagedKVCache`.
The proof of concept already works in the CB path (`continuous_batcher.py:651`).

**Merged design:**

```python
# In _extreme_decode — replace past_kv passthrough:
# Step 1: Write prefill KV into paged blocks (PagedKVCache.write)
# Step 2: Use PagedCache as past_key_values
past_kv = PagedCache(paged_kv, seq_ids=seq_ids) if self._use_pa else prefill_out.past_key_values
# Step 3: Decode loop unchanged — model doesn't care which Cache impl
for step in range(max_new):
    out = self.model(input_ids=next_input, past_key_values=past_kv, use_cache=True)
```

This is ~50 lines of changed code. The CB path becomes a scheduling strategy
on top of the same KV-cache, not a separate stack. The `KVManager` protocol
prevents future stack proliferation: any new KV optimization implements
`KVManager` and drops in.

---

### 14.3 Flaw #3 (Ad-Hoc Gating): FeatureGate with Level Hierarchy

**My architectural insight**: The 7+ gating mechanisms differ in *who* controls
them: developer (hardcoded), operator (env var), or user (config). Each level
needs different visibility and override behavior.

**Research contribution**: `FeatureGate` with `GateLevel` enum (FIX_DESIGNS §3):
HARD (hardcoded in source — developer decides), CONFIG (operator sets via config
file or env var), ENV (runtime environment decides). Each gate has an
`evaluate()` method that returns `(allowed: bool, reason: str)`.

**Key design decision**: When a HARD gate says "disabled because incompatible
with torch.compile," overriding it via config should produce a WARNING, not
silently take effect. The gate hierarchy preserves the developer's intent while
showing the operator what they're overriding.

---

### 14.4 Flaw #4 (Protocol Erosion): Three-Point Enforcement

**My architectural insight + research concurrence**: Both vLLM and ONNX Runtime
enforce backend contracts at registration time, not at first-call time. The
research confirmed this pattern. Add a third point: lightweight per-batch
validation that catches silent divergence (like padded `input_tokens`).

**Three enforcement points:**

1. **Registration time** (vLLM `validate_configuration()` pattern): when a
   backend is registered in `ModelRegistry`, call `validate_backend(BackendClass)`.
   Checks: all required attrs exist; capability bitmask matches implemented methods;
   `_configured_batch_size` initialised. Fails fast.

2. **Post-load runtime** (custom): after `load()`, call `validate_loaded(backend)`.
   Checks: `is_loaded() == True`; `kv_cache_config` returns plausible values;
   model accepts `past_key_values`. One-time, ~100ms.

3. **Per-batch** (custom): a `@validated` decorator on `translate_batch` with
   `__debug__` guard. Asserts: return type is `BatchGenerationOutput`;
   `input_tokens_total > 0`; `output_tokens_total >= 0`. O(1) per batch.
   Compiled out in production.

---

### 14.5 Flaw #5 (God Harness): Incremental Composition Root

**My architectural insight**: A full DI framework (python-dependency-injector)
is overkill for 25K LOC. The right path is **incremental extraction**: Phase 1
decouples observability with an event bus (40 lines, no library). Phase 2
extracts coordinator objects. Phase 3 arrives at a Composition Root pattern.

**Research concurrence**: The "Composition Root" pattern (FIX_DESIGNS §5) — a
single function that wires everything, no library needed — fits a 3-dev research
project better than a DI container.

**The 40-line event bus (Phase 1):**

```python
class EventBus:
    def __init__(self):
        self._handlers: dict[str, list[Callable]] = defaultdict(list)
    def on(self, event: str, handler: Callable): ...
    def emit(self, event: str, **data):
        for h in self._handlers.get(event, []):
            h(**data)

# Registration — one block, not 11 scattered call sites:
events.on('batch_completed', metrics.log_batch)
events.on('batch_completed', prometheus.record_batch)
events.on('heartbeat', prometheus.record_pipeline)
events.on('error', prometheus.record_error)

# Translation loop — one line per logical event:
events.emit('batch_completed', tokens=total, latency_ms=ms, batch_size=bs)
```

This doesn't restructure the harness at all. It replaces 11 direct
`self._prometheus.record_*()` calls with 4 registrations + 1 emit per event.
Observability is now decoupled — you can add a new metric without touching the
harness. You can test the harness without Prometheus.

Phases 2 (coordinators) and 3 (Composition Root) follow incrementally.

---

### 14.6 Flaw #6 (Config Confusion): Immutable Config → ResolvedConfig

**My architectural insight**: The fix is to freeze config after parsing, compute
ALL derived state once, and store it in a `ResolvedConfig` that is the ONLY
object downstream code reads from. The raw `BenchmarkConfig` becomes an
intermediate artifact.

**Research contribution**: The Kubernetes reconciliation loop pattern applied
to config (FIX_DESIGNS §6): `desired_state` (user config) → `current_state`
(detected hardware) → `resolved_state` (what will actually run). If
`desired` and `resolved` diverge, the divergence is logged as a reconciliation
event.

**Merged design**:

```python
@dataclass(frozen=True)
class ResolvedConfig:
    effective_precision: PrecisionConfig   # reconciled with hardware
    effective_batch_size: int              # after tuning
    active_features: frozenset[str]        # from CapabilityRegistry
    divergences: tuple[Divergence, ...]    # config-said-X-runtime-did-Y

    @classmethod
    def resolve(cls, config: BenchmarkConfig, device: DeviceInfo,
                caps: CapabilityRegistry) -> 'ResolvedConfig':
        ...

# After load(), the ONLY config object passed around is ResolvedConfig.
# No code path reads from raw BenchmarkConfig after resolution.
```

The `--safe-mode` FP8 inconsistency becomes impossible: `ResolvedConfig` checks
whether TE was actually applied, not whether config asked for it.

---

### 14.7 Flaw #7 (Extrapolation Gap): Honest Prediction with Reliability Grading

**My architectural insight**: The benchmark's entire purpose is a prediction.
The prediction must carry an explicit **reliability grade** — "this prediction
is reliable / qualified / unreliable" — so the consumer knows whether to act on it.

**Research contribution**: An exponential decay model from the research
(FIX_DESIGNS §7): `TPS(t) = TPS_0 * exp(-λ * t) + TPS_a * (1 - exp(-λ * t))`
where `TPS_0` is initial throughput, `TPS_a` is asymptotic throughput, and λ
is the degradation rate. Fit this to the per-batch TPS time-series. If λ ≈ 0
(no degradation), the constant-throughput assumption is validated. If λ > 0,
the prediction accounts for it.

MLPerf's anti-extrapolation rules and TPC's steady-state verification
requirements are incorporated as: (a) discard warmup from measurement,
(b) verify that throughput is stable before extrapolating, and (c) flag
predictions from degraded runs as "unreliable."

**Reliability grading**:

- **Reliable**: throughput stable (λ ≈ 0), ≥100 batches, real (not padded) tokens
- **Qualified**: some degradation detected, or <100 batches
- **Unreliable**: significant degradation, padded tokens (AR/TRT), or <30 batches

The days estimate always includes the conservative lower-bound (lower 95%
bootstrap CI) as the primary number. The point estimate is secondary.

---

### 14.8 Flaw #8 (Fragmented Memory): Fixed-Budget Planner with Category-Aware OOM

**My architectural insight**: The OOM handler halves batch size blindly because
it doesn't know WHAT exhausted memory. A category-aware budget tells you: is
this a KV-cache OOM (→ reduce batch) or a weight OOM (→ different precision)?

**Research contribution**: Two-tier watermarks from vLLM/TGI (FIX_DESIGNS §8):
soft watermark triggers pre-emptive batching changes. Hard watermark rejects new
allocations. Both happen BEFORE OOM, not in response to it.

The `MemoryBudget.reserve()` returns `True`/`False` and the caller handles
degradation, rather than the caller allocating blindly and the OOM handler
reacting post-hoc.

**Category-aware OOM handler**:

```python
def _handle_allocation_failure(self, budget: MemoryBudget, category: str):
    status = budget.watermark_status()
    if category == "kv_cache":
        # Reduce batch size — frees KV-cache memory
        self._reduce_batch()
    elif category == "weights":
        # Consider lower precision or single-GPU fallback
        logger.error("Weight memory exhausted — cannot recover without precision change")
    elif status == "yellow":
        # Pre-emptive: reduce batch before OOM
        self._reduce_batch()
```

---

### 14.9 Flaw #9 (Hardcoded Assumptions): `model.config` as Single Source

**My architectural insight + research concurrence**: Every module that needs
architecture dimensions (num_layers, head_dim, etc.) should read them from
`model.config` at runtime. The constants file provides FALLBACKS only when
`model.config` is genuinely unavailable (e.g., before model load). The presets
in `model_presets.py` are used for *selection* (which model to load), not for
*dimension queries at runtime*.

The research added a `ModelArchitecture` frozen dataclass with `from_model()`
introspection (FIX_DESIGNS §9), which extracts `num_layers, num_kv_heads,
head_dim, hidden_size, vocab_size, rotary_emb_base` from `model.config` with
sensible fallbacks. Every module queries this object — no more `DEFAULT_NUM_LAYERS`.

---

### 14.10 Flaw #10 (Wrong Observability): Reflective Metrics + Runtime-Truth Data

**My architectural insight**: The observability system observes the wrong object
at the wrong resolution. It observes *config state* (what was requested), not
*runtime state* (what happened). The fix is two-fold: (1) plumb the
CapabilityRegistry into the `/status` health endpoint so external monitoring can
detect False Flags, and (2) fix the specific dead metrics.

**Research contribution**: Streaming t-digest for latency percentiles
(FIX_DESIGNS §10) — replaces the current `sorted()`-based percentile computation
(slow, memory-hungry) with a t-digest sketch (constant memory, streaming).

**Specific fixes** (30 lines total, Tier 1):
1. `harness.py:615`: pass actual BLEU/chrF scores, not `bleu=None, chrf=None`
2. Prometheus throughput: use 60s rolling average from `ThroughputTracker.current()`, not instantaneous per-batch TPS
3. `collector.log_batch()`: pass `latency_ms` to `ThroughputTracker`
4. `markdown_report.py`: add BERTScore and COMET-Kiwi rows
5. `/status` endpoint: serves `CapabilityRegistry.active_ids()` and `false_flags()`

---

### Cross-Flaw Integration

The recommended fix order (dependency-respecting):

```
Phase 1 ── #1 False Flag + #3 Ad-Hoc Gating ──→ CapabilityRegistry
               │                                        │
               └── enables ──→ #6 Config Confusion      │
               └── enables ──→ #10 Wrong Observability   │
               └── feeds ────→ #4 Protocol Enforcement   │
                                      │
Phase 2 ──────────────────────────────┤
               ┌── enables ──→ #2 Two Stacks merge ──→ #8 Memory Planner
               └── enables ──→ #5 God Harness refactor
               └── enables ──→ #9 Hardcoded Assumptions
                                      │
Phase 3 ──────────────────────────────┤
               └── converges on ──→ #7 Honest Extrapolation
```

The CapabilityRegistry is the keystone. Everything else builds on knowing the truth.

---

## References

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — reality-grounded architecture and Feature Status table
- [`AI_CODING_ANTIPATTERNS.md`](AI_CODING_ANTIPATTERNS.md) — concrete mistakes (especially A1, A3, A6, A11)
- [`FIX_DESIGNS.md`](FIX_DESIGNS.md) — full implementation sketches with code for all 10 flaws
- [`docs/README.md`](README.md) — documentation navigation
- Audit memory: 7 rounds of findings, systemic patterns documented in `h200-ruthless-audit-round7.md`
- Git log: commits `804c0a6`, `9fa3397`, `ffa707b`, `5ec301c`, `ecbade9`, `92d9476`

---

*This document is part of the engineering reality docs. See [`docs/README.md`](README.md)  
for navigation. Companions: [`ARCHITECTURE.md`](ARCHITECTURE.md), [`FIX_DESIGNS.md`](FIX_DESIGNS.md).*
