# Documentation Index

> **Purpose:** Navigation map for the TR Corpus Translation Benchmark docs.
> **Status:** Current as of v3.8.
>
> Start here. Every doc is listed below with a one-line purpose and a status badge.

---

## How to use this index

**If you want to…**

| …do this | …read this first |
|---|---|
| Understand the project at a glance | [`README.md`](../README.md) (root) |
| Know what the code *actually does* (not what the specs say) | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Find out which "optimizations" are real vs. gated off | [`ARCHITECTURE.md` §8 Feature Status](ARCHITECTURE.md#8-feature-status-the-truth-table) |
| Run / install / deploy the benchmark | [`COMPILATION_GUIDE.md`](COMPILATION_GUIDE.md) |
| See the production H200 deployment log | [`H200_SETUP.md`](H200_SETUP.md) |
| Add a backend / plugin / preset, run tests, follow conventions | [`DEVELOPMENT.md`](DEVELOPMENT.md) |
| Avoid mistakes AI coders have already made here | [`AI_CODING_ANTIPATTERNS.md`](AI_CODING_ANTIPATTERNS.md) |
| Understand the original product intent / requirements | [`PRD.md`](PRD%26SDD%26SRS/PRD.md), [`SRS.md`](PRD%26SDD%26SRS/SRS.md) (historical) |
| Understand the design and optimizations | [`SDD.md`](PRD%26SDD%26SRS/SDD.md) |
| Read the quality-metric research background | [`QUALITY_METRICS_RESEARCH.md`](Research/QUALITY_METRICS_RESEARCH.md) |

**For LLM coding agents:** before editing `benchmark/inference/` or
`benchmark/hardware/`, read [`ARCHITECTURE.md`](ARCHITECTURE.md) (especially
§8 Feature Status) and [`AI_CODING_ANTIPATTERNS.md`](AI_CODING_ANTIPATTERNS.md).
Do not reason about performance or capabilities from the README's optimization
count — it does not reflect the gating reality.

---

## Status legend

| Badge | Meaning |
|---|---|
| ✅ Wired | On the production hot path. |
| 🟡 Built-but-gated | Implemented, present, but not active unless explicitly activated (and even then may be stats-only). |
| 🔬 Experimental | Functional but opt-in / env-gated; correctness-preserving but not validated at scale. |
| ⚠️ Broken/Disabled | Disabled on purpose, or broken against current dependencies. |
| 💀 Dead code | Defined but never called on any path. |
| 🗑 REMOVED | Permanently deleted in v3.7. Does not exist in the codebase. |

*(Used in [`ARCHITECTURE.md` §8](ARCHITECTURE.md#8-feature-status-the-truth-table) and throughout.)*

---

## Document catalogue

### 1. Engineering reality (current truth — read these for what the code does)

- **[`ARCHITECTURE.md`](ARCHITECTURE.md)** — Reality-grounded architecture: the
  runtime hot path, the `InferenceBackend` protocol, backend dispatch (transparent CUDA/MPS dispatcher wrappers), per-backend
  reality, the module map, the two optimization stacks, and the authoritative
  **Feature Status** table. *Single source of truth; supersedes the specs where
  they disagree.*
- **[`DEVELOPMENT.md`](DEVELOPMENT.md)** — How to develop on this codebase: repo
  layout, testing, lint/format, adding backends/plugins/presets, coding
  conventions, and an orientation box for LLM coders.
- **[`AI_CODING_ANTIPATTERNS.md`](AI_CODING_ANTIPATTERNS.md)** — Concrete AI
  mistakes made in this project (with evidence) plus preventable pitfalls, tagged
  🔴 occurred / 🟡 risk / ℹ️ guidance, with a pre-flight checklist.

### 2. Operations (running it)

- **[`COMPILATION_GUIDE.md`](COMPILATION_GUIDE.md)** — Setup, run, Make targets,
  Docker, model selection, performance tuning. *(JIT/TRT sections removed v3.7.)*
- **[`LANGUAGE_EVALUATION_STUDY.md`](Research/LANGUAGE_EVALUATION_STUDY.md)** — EN→TR quality evaluation study design.
- **[`H200_SETUP.md`](H200_SETUP.md)** — Production H200 deployment log: installed packages, the 20 errors encountered and fixed, the fork-bomb incident, the MPS memory investigation, the external-sort shuffle. *(Engineering log; stale claims annotated.)*

### 3. Specifications (historical — design intent, not current reality)

- **[`PRD.md`](PRD%26SDD%26SRS/PRD.md)** — Product Requirements: the problem (Turkish
  underrepresentation), goals, KPIs, scope, risks. *Historical; see ARCHITECTURE
  for current reality.*
- **[`SRS.md`](PRD%26SDD%26SRS/SRS.md)** — Software Requirements: functional/non-functional
  requirements, data/interface requirements, traceability. *Historical; some
  factual errors corrected (BERTScore model, BLEU/chrF).*
- **[`SDD.md`](PRD%26SDD%26SRS/SDD.md)** — Software Design: component design, data flow, error
  handling, model presets. *Normalized to mark each optimization's real status.*

### 4. Research (background, not a description of the implementation)

- **[`QUALITY_METRICS_RESEARCH.md`](Research/QUALITY_METRICS_RESEARCH.md)** — Deep research
  on EN→TR translation-quality metrics (COMET/xCOMET/MetricX/BERTScore/…). *Background
  research; the actually-implemented metric stack is documented in ARCHITECTURE
  §8 (#31–#35) and DEVELOPMENT.md.*

### 5. Acceleration Plans and Benchmarks

- **[`NLLB_MADLAD_BENCHMARKS.md`](NLLB_MADLAD_BENCHMARKS.md)** — Empirical performance metrics, baseline comparisons, and scientific takeaways for Seq2Seq models.

---

## A note on doc-vs-reality (why this index exists)

**v3.7 cleanup:** 5 dead modules (~3,600 lines) were permanently deleted:
`jit_compiler.py`, `fused_ops.py`, `triton_kernels_fused.py`, `cuda_graphs.py`,
`kv_cache_quant.py`, `perf_regression.py`, `tensorrt_backend.py`, `trt_builder.py`.
The remaining ~41 features are either wired-to-hot-path, gated (opt-in), or marked
REMOVED. See [`ARCHITECTURE.md` §8](ARCHITECTURE.md#8-feature-status-the-truth-table)
for the authoritative truth table.

**When a spec and ARCHITECTURE disagree, ARCHITECTURE is correct.**

---

*Back to [`README.md`](../README.md).*
