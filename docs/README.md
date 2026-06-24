# Documentation Index

> **Purpose:** Navigation map for the TR Corpus Translation Benchmark docs.
> **Status:** Current as of v3.6.
>
> Start here. Every doc is listed below with a one-line purpose and a status badge.

---

## How to use this index

**If you want to…**

| …do this | …read this first |
|---|---|
| Understand the project at a glance | [`README.md`](../README.md) (root) |
| Know what the code *actually does* (not what the specs say) | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Understand why the codebase has structural problems | [`ARCHITECTURAL_FLAWS.md`](ARCHITECTURAL_FLAWS.md) |
| Find out which "optimizations" are real vs. gated off | [`ARCHITECTURE.md` §8 Feature Status](ARCHITECTURE.md#8-feature-status-the-truth-table) |
| Run / install / deploy the benchmark | [`COMPILATION_GUIDE.md`](COMPILATION_GUIDE.md) |
| See the production H200 deployment log | [`H200_SETUP.md`](H200_SETUP.md) |
| Add a backend / plugin / preset, run tests, follow conventions | [`DEVELOPMENT.md`](DEVELOPMENT.md) |
| Avoid mistakes AI coders have already made here | [`AI_CODING_ANTIPATTERNS.md`](AI_CODING_ANTIPATTERNS.md) |
| Understand the original product intent / requirements | [`PRD.md`](PRD.md), [`SRS.md`](SRS.md) (historical) |
| Understand the original design | [`SDD.md`](SDD.md) |
| Read the quality-metric research background | [`QUALITY_METRICS_RESEARCH.md`](QUALITY_METRICS_RESEARCH.md) |
| Measure real system performance (replace guesses) | [`MEASUREMENT_PLAN.md`](MEASUREMENT_PLAN.md) |

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

*(Used in [`ARCHITECTURE.md` §8](ARCHITECTURE.md#8-feature-status-the-truth-table) and throughout.)*

---

## Document catalogue

### 1. Engineering reality (current truth — read these for what the code does)

- **[`ARCHITECTURE.md`](ARCHITECTURE.md)** — Reality-grounded architecture: the
  runtime hot path, the `InferenceBackend` protocol, backend dispatch, per-backend
  reality, the module map, the two optimization stacks, and the authoritative
  **Feature Status** table. *Single source of truth; supersedes the specs where
  they disagree.*
- **[`ARCHITECTURAL_FLAWS.md`](ARCHITECTURAL_FLAWS.md)** — Systems-architecture-level
  analysis of 10 structural flaws (False Flag architecture, two optimization stacks,
  ad-hoc gating, protocol erosion, God harness, config confusion, extrapolation gap,
  memory fragmentation, hardcoded assumptions, wrong observability). Includes
  dependency graph and prioritized 3-tier fix roadmap.
- **[`DEVELOPMENT.md`](DEVELOPMENT.md)** — How to develop on this codebase: repo
  layout, testing, lint/format, adding backends/plugins/presets, coding
  conventions, and an orientation box for LLM coders.
- **[`AI_CODING_ANTIPATTERNS.md`](AI_CODING_ANTIPATTERNS.md)** — Concrete AI
  mistakes made in this project (with evidence) plus preventable pitfalls, tagged
  🔴 occurred / 🟡 risk / ℹ️ guidance, with a pre-flight checklist.

### 2. Operations (running it)

- **[`COMPILATION_GUIDE.md`](COMPILATION_GUIDE.md)** — Setup, run, Make targets,
  Docker, JIT/TRT, model selection, performance tuning, multi-node. *(Honest
  feature-gating status; the old "+20–50% TRT" / "37 wired" claims are corrected.)*
- **[`H200_SETUP.md`](H200_SETUP.md)** — Production H200 deployment log: installed
  packages, the 20 errors encountered and fixed, the fork-bomb incident, the MPS
  memory investigation, the external-sort shuffle. *(Engineering log; stale claims
  annotated.)*
- **[`MEASUREMENT_PLAN.md`](MEASUREMENT_PLAN.md)** — 24 measurements (P0–P4)
  needed to replace 158 hardcoded constants and aspirational speedup claims with
  real H200 system data. Includes measurement harness design script, per-measurement
  method/output/priority, and a top-5 quick-start. *Execution status: ⬜ none yet
  measured.*

### 3. Specifications (historical — design intent, not current reality)

- **[`PRD.md`](PRD.md)** — Product Requirements: the problem (Turkish
  underrepresentation), goals, KPIs, scope, risks. *Historical; see ARCHITECTURE
  for current reality.*
- **[`SRS.md`](SRS.md)** — Software Requirements: functional/non-functional
  requirements, data/interface requirements, traceability. *Historical; some
  factual errors corrected (BERTScore model, BLEU/chrF).*
- **[`SDD.md`](SDD.md)** — Software Design: component design, data flow, error
  handling, model presets. *Normalized to mark each optimization's real status.*

### 4. Research (background, not a description of the implementation)

- **[`QUALITY_METRICS_RESEARCH.md`](QUALITY_METRICS_RESEARCH.md)** — Deep research
  on EN→TR translation-quality metrics (COMET/xCOMET/MetricX/BERTScore/…). *Background
  research; the actually-implemented metric stack is documented in ARCHITECTURE
  §8 (#31–#35) and DEVELOPMENT.md.*

---

## A note on doc-vs-reality (why this index exists)

The earlier documentation set (README, SDD, COMPILATION_GUIDE) advertised "39
optimizations (37 wired, 2 experimental)." A full code audit showed that the
production autoregressive hot path is actually **plain eager `model(...)` with
HF `past_key_values`**, accelerated only by `torch.compile` and Transformer-Engine
FP8 — most other "optimizations" are built but gated off, broken, or dead. The
specs (PRD/SRS/SDD) were also written at design time and contain some claims that
no longer match the code.

This index, `ARCHITECTURE.md`, and the corrected docs fix that. **When a spec and
ARCHITECTURE disagree, ARCHITECTURE is correct.**

---

*Back to [`README.md`](../README.md).*
