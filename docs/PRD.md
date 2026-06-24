# Product Requirements Document (PRD)
## Turkish Corpus Translation Benchmark — Feasibility Study
### (Apple Silicon MPS Development → NVIDIA H200 Production)

---

| **Document** | Product Requirements Document |
|---|---|
| **Project** | Turkish ClearNet Corpus Translation Benchmark |
| **Version** | 3.6 |
| **Status** | Implemented |
| **Author** | — |
| **Date** | 2026-06-19 |
| **Revised** | 2026-06-23 |

---

> ⚠️ **Historical spec.** This PRD describes the *product intent* at design time.
> The authoritative description of what the code *actually does* today is
> [`ARCHITECTURE.md`](ARCHITECTURE.md). Where they disagree,
> **ARCHITECTURE is correct.** Several requirements here (tensor parallelism,
> CUDA graphs, fused kernels, TensorRT) were implemented but subsequently gated
> off or broken — see
> [ARCHITECTURE §8 Feature Status](ARCHITECTURE.md#8-feature-status-the-truth-table).
> This document is included for understanding the original goals and
> requirements, not for navigating the current code.

---

## Revision History

| Version | Date | Changes |
|---|---|---|
| 1.0 | 2026-06-19 | Initial draft |
| 1.1 | 2026-06-21 | Reflects implementation reality: TranslateGemma 4B added as dev model, macOS MPS baselines established, E2E test suite completed, real data (FineWeb + OPUS-100) integrated |
| 3.2 | 2026-06-21 | Model-agnostic backend protocol, diffusion support, plugin system, TensorRT, 40 extreme optimizations |
| 3.3 | 2026-06-22 | Speculative decoding (self-spec + draft-model), resume/checkpoint with position tracking, extrapolation CI fix (SEM-based + bootstrap), external sort shuffle, H200 production deployment fixes, MPS IOAccelerator memory investigation |
| 3.6 | 2026-06-23 | NLLB-200 encoder-decoder backend (600M–54B), model presets registry (11 presets), quantization levels (bf16/fp16/int8/int4), Ministral 3B, Gemma4 QAT (2B/4B, ct/int4/q4_0), DiffusionGemma 26B, --nllb/--model/--quantization/--paged-attention/--continuous-batching CLI flags, dead code cleanup (removed 4 modules: back_translate, domain_classifier, rust_tokenizer, mmap_reader; 4 stale config files; 2 stale shell scripts) |

---

## 1. Executive Summary

The Turkish language is underrepresented in public training corpora relative to English and other high-resource languages. The **ClearNet dataset** — a large, openly available web-crawl text corpus — contains predominantly non-Turkish text. Translating even a fraction of this dataset would substantially increase the volume of Turkish text available for language model pre-training and fine-tuning.

However, running a full translation is a multi-week, GPU-intensive undertaking. Before committing to that cost, we need a **rigorous feasibility benchmark** that answers a single predictive question:

> *Using 2× NVIDIA H200 GPUs running **TranslateGemma 12B** at FP8 precision, how long would it take to translate the entire non-Turkish ClearNet dataset into Turkish?*

This project builds the instrumentation harness that runs for a **2-hour burn-in window** (production) or a shorter **evaluation window** (development), collects hardware utilisation metrics, counts translated tokens, and runs a translation-quality benchmark — producing the dataset from which that prediction can be derived with confidence.

**Development workflow**: All code is developed and smoke-tested on **Apple Silicon (MPS backend)** using **TranslateGemma 4B** (BF16) before being promoted to the H200 cluster with **TranslateGemma 12B** (FP8). The harness is backend-agnostic — automatically detecting the available device (CUDA or MPS) and adapting tensor-parallelism strategy, precision, and metrics collection accordingly. The same Python code runs unmodified on both platforms; only the model ID and hardware differ.

**Implementation status (v3.6, note from June 2026):** 75 Python modules,
27 test files (~75 unit tests). ~44 optimization features implemented — many are
**built but gated off** (hardcoded `False`, env-gated, or broken against current
dependencies). The production AR hot path is plain eager `model(...)` + 
`torch.compile(reduce-overhead)` + Transformer-Engine FP8. See
[`ARCHITECTURE.md` §8](ARCHITECTURE.md#8-feature-status-the-truth-table) for the
authoritative Feature Status table. 4 backends defined (AR, encoder-decoder/NLLB,
diffusion, TensorRT; TensorRT is safety-gated and falls back to AR).

---

## 2. Problem Statement

### 2.1 Current State

The table below presents the language distribution in major public web-crawl corpora. These figures are drawn from published, peer-reviewed datasets and represent the best available estimates for the size of the non-Turkish web corpus that could be translated.

#### 2.1.1 Language Distribution in CulturaX (167 languages, 6.3 Trillion tokens total) [¹]

CulturaX is a cleaned and deduplicated merge of mC4 and all OSCAR releases through 2023. It is the largest public multilingual corpus with a published per-language token breakdown.

| Language | Token Count | Share of Total | Status |
|---|---|---|---|
| English (`en`) | 2,846.97 B (2.85 T) | 45.13 % | Dominant; the primary source for translation |
| Russian (`ru`) | 737.20 B | 11.69 % | Second-largest language |
| Spanish (`es`) | 373.85 B | 5.93 % | |
| German (`de`) | 357.03 B | 5.66 % | |
| French (`fr`) | 319.33 B | 5.06 % | |
| Chinese (`zh`) | 227.06 B | 3.60 % | |
| Other (161 languages) | 1,453.49 B | 22.93 % | Long-tail distribution |
| **Turkish (`tr`)** | **64.29 B** | **1.02 %** | **Severely underrepresented** |

**Key takeaway**: Non-Turkish content in CulturaX alone totals **≈ 6.23 trillion tokens**. Turkish represents only 1.02 % of the corpus.

#### 2.1.2 Broader Web-Crawl Corpus Estimates

| Corpus | Total Size | English | Turkish | Source |
|---|---|---|---|---|
| **FineWeb v1** (2024) | ~15 T tokens | ~15 T (overwhelmingly English) | negligible at this tier | HuggingFace, May 2024 [²] |
| **FineWeb-2** (2024) | 15 T+ tokens | ~60%+ (dominant) | long-tail tier (est. 1–100 B) | HuggingFace, late 2024 [²] |
| **HPLT v2** (2025) | ~8 T tokens (193 langs) | majority share | included, exact count TBD | Burchell et al., ACL 2025 [³] |
| **Nemotron-CC** (2024) | 6.3 T tokens (4.4 T real + 1.9 T synthetic) | 4.4 T (deduplicated real tokens) | N/A (English-only) | NVIDIA, 2024 [⁴] |
| **CulturaX** (2024) | 6.3 T tokens (167 langs) | 2.85 T | 64.3 B | Nguyen et al., LREC-COLING 2024 [¹] |
| **TURNA Corpus** (2025) | N/A (Turkish-only) | N/A | **84.88 B** | Türker et al. (TabiBERT), Dec 2025 [⁵] |

#### 2.1.3 Turkish Corpus Landscape

Even the largest assembled Turkish corpora are dwarfed by English data:

| Turkish Corpus | Size | Composition |
|---|---|---|
| **TURNA** (largest aggregate) | ~84.88 B tokens | FineWeb-2 Turkish (56 B) + Dergipark + Yoktez + books + parliamentary records + code |
| **CulturaX Turkish** | ~64.29 B tokens | Deduplicated web crawl (mC4 + OSCAR) |
| **FineWeb-2 Turkish** | ~56.04 B tokens | Cleaned web crawl, 88.8 M documents |
| **Bella Turca** | ~50–80 B tokens (est.) | 25 diverse subsets; news, web, literary, scientific |
| **VNGRS Web Corpus** | ~25.33 B tokens | Cleaned OSCAR + mC4 Turkish, 50.3 M pages |

**The gap**: The largest Turkish corpus (TURNA, ~84.88 B tokens) is **~33× smaller** than CulturaX English alone (2.85 T tokens), and **~70× smaller** than FineWeb v1 English (~15 T tokens). This is a two-order-of-magnitude deficit.

> **Sources**:
> [¹] Nguyen et al., "CulturaX: A Cleaned, Enormous, and Multilingual Dataset for LLMs in 167 Languages," *LREC-COLING 2024*. Dataset card: `huggingface.co/datasets/uonlp/CulturaX`.
> [²] HuggingFace FineWeb blog posts, May–Dec 2024. `huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1`.
> [³] Burchell et al., "An Expanded Massive Multilingual Dataset for High-Performance Language Technologies," *ACL 2025*. (HPLT v2). de Gibert et al., "A New Massive Multilingual Dataset for High-Performance Language Technologies," *LREC-COLING 2024* (HPLT v1).
> [⁴] NVIDIA, "Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset," 2024. 6.3 T total = 4.4 T real (deduplicated) + 1.9 T synthetic.
> [⁵] Türker et al., "TabiBERT: A Large-Scale ModernBERT Foundation Model and A Unified Benchmark for Turkish," Dec 2025. The expanded TURNA corpus (84.88 B tokens) was introduced in this paper. The original TURNA model (Uludoğan et al., Findings of ACL 2024) was trained on ~43 B tokens. `huggingface.co/datasets/boun-tabi-LMG/TURNA`.

### 2.2 Desired State

A **sufficiently large Turkish corpus** (target: ≥1T tokens) suitable for pre-training competitive Turkish LLMs, derived via machine translation of the existing non-Turkish ClearNet subset.

### 2.3 Gap

We do not know:

- The **sustained throughput** (tokens/second) achievable on our target hardware.
- The **quality ceiling** of TranslateGemma-12B-FP8 on web-crawl text.
- The **hardware utilisation efficiency** (are we GPU-bound, CPU-bound, or I/O-bound?).
- The **cost and wall-clock time** of a full-dataset run.

Without these numbers, committing to a full translation is a blind bet.

---

## 3. Project Vision

Build a **self-contained benchmarking harness** that:

1. Loads TranslateGemma 12B quantised to **FP8** and auto-detects the available backend (**MPS on Apple Silicon** for development; **CUDA with 2× NVIDIA H200 GPUs** for production benchmarking).
2. Streams a representative sample of the ClearNet English corpus through the model.
3. Translates continuously for a fixed **2-hour window**.
4. Simultaneously logs per-second hardware and throughput metrics.
5. At the 2-hour mark, runs a **translation-quality benchmark** (BERTScore,
   COMET-22, COMET-Kiwi, BLEU, chrF++) against a held-out golden reference set.
6. Aggregates all data into a single **benchmark report** suitable for extrapolation to full-dataset scale.

The harness itself is **not** the production translator — it is the measurement tool that informs whether and how to build the production translator.

---

## 4. Goals & Objectives

| # | Goal | Type | Success Metric |
|---|---|---|---|
| G1 | Measure sustained translation throughput | Quantitative | Mean tokens/second over the 2 h window, ±95% CI |
| G2 | Characterise GPU utilisation | Quantitative | Mean & p99 GPU SM utilisation, memory bandwidth utilisation, idle-time breakdown |
| G3 | Identify the bottleneck | Diagnostic | Definitively classify bound: GPU-compute, GPU-memory, CPU-prep, or I/O |
| G4 | Measure translation quality | Quantitative | BERTScore ≥ 0.55, COMET-22 ≥ 0.72, COMET-Kiwi ≥ 0.72, BLEU ≥ 25, chrF++ ≥ 54 on the reference set |
| G5 | Produce a full-dataset time/cost estimate | Predictive | Extrapolated days-to-completion with upper/lower bounds |
| G6 | Reproducibility | Process | Single-command launch; all config captured in the output; run-to-run variance < 5 % |

---

## 5. Stakeholders

| Role | Interest |
|---|---|
| **ML Engineering Lead** | Decides whether to green-light the full translation run |
| **Infrastructure / MLOps** | Needs GPU utilisation data to right-size the cluster |
| **Data Team** | Needs quality estimates to decide if MT output is usable for pre-training |
| **Research Leadership** | Needs cost & timeline estimates for grant / budget planning |
| **Open-Source Community** | Benchmark results may be published to inform other low-resource-language efforts |

---

## 6. Success Criteria (KPIs)

| KPI | Threshold | Target | macOS Baseline (4B/BF16) |
|---|---|---|---|
| Sustained throughput | ≥ 800 tok/s | ≥ 1 200 tok/s | 103 tok/s (dev only) |
| GPU SM utilisation (mean) | ≥ 70 % | ≥ 85 % | N/A on MPS (powermetrics needs sudo) |
| GPU memory bandwidth util. | ≥ 60 % | ≥ 80 % | N/A on MPS |
| Translation BERTScore (system) | ≥ 0.55 | ≥ 0.70 | Pending H200 run |
| Translation COMET-22 score | ≥ 0.72 | ≥ 0.80 | 0.8104 (H200 dry-run) |
| Translation COMET-Kiwi score | ≥ 0.72 | ≥ 0.80 | 0.8370 (H200 dry-run) |
| Run-to-run variance (throughput) | ≤ 8 % | ≤ 5 % | 2.1% on macOS |
| End-to-end runtime | 120 min ± 2 min | 120 min ± 1 min | 120s (E2E test) |
| Extrapolation confidence interval | ± 30 % | ± 15 % | ± 2.1% (tight sample) |

---

## 7. Scope

### 7.1 In Scope

- **Hardware abstraction layer** that auto-detects the backend (CUDA/MPS) at startup, adapting model loading, precision (FP8 on CUDA, BF16 on MPS), and tensor-parallelism strategy (TP=2 on CUDA, single-device on MPS).
- Loading TranslateGemma 12B with FP8 weights from a pre-quantised checkpoint (CUDA) or BF16 weights (MPS fallback).
- Tensor-parallel sharding across 2× H200 GPUs (production); single-device inference on Apple Silicon (development).
- Streaming English text from a local ClearNet sample (≥ 5 GB compressed JSONL).
- A continuous translation loop with batched inference for the 2 h window.
- Per-second logging of: GPU utilisation, GPU memory, GPU temperature, CPU utilisation, RAM, tokens generated, batch latency, queue depth.
- A quality benchmark run at T+2 h on a 1 000-sentence golden reference set (English → Turkish, human-verified).
- Aggregation script that produces a single JSON + Markdown report.
- Dockerised, single-command launch on both macOS (`pip install -e . && python -m benchmark`) and Linux/H200 (`make docker-run`).

### 7.2 Out of Scope

- Actual translation of the full ClearNet dataset.
- Training or fine-tuning any model.
- Multi-node (≥2 machine) orchestration.
- A web UI or dashboard (terminal + file output only).
- Turkish-specific tokeniser training (we use the Gemma tokeniser as-is, with its multilingual vocabulary).
- Streaming from cloud object storage (all data is local on the NVMe volume).

---

## 8. High-Level Requirements

| ID | Requirement | Priority |
|---|---|---|
| HL-1 | The system shall load TranslateGemma 12B in FP8 precision across 2 GPUs via tensor parallelism | P0 |
| HL-2 | The system shall translate English text to Turkish at the highest sustainable throughput achievable on the hardware | P0 |
| HL-3 | The system shall log all hardware metrics at 1 Hz granularity for the full 2 h run | P0 |
| HL-4 | The system shall compute and log per-batch translation latency and throughput | P0 |
| HL-5 | The system shall run a translation-quality benchmark at the conclusion of the 2 h window | P0 |
| HL-6 | The system shall produce a single aggregated report containing all metrics and the extrapolated full-dataset estimate | P0 |
| HL-7 | The system shall checkpoint progress at least every 5 minutes so a crash does not lose all data | P1 |
| HL-8 | The system shall validate the model output is valid UTF-8 Turkish text (sanity gate) | P1 |
| HL-9 | The system shall be fully containerised and reproducible | P1 |
| HL-10 | The system shall auto-detect the available compute backend (CUDA, MPS, or CPU) at startup and select the appropriate model-loading path, precision, and metrics collection strategy | P0 |
| HL-11 | The system shall run unmodified on Apple Silicon (MPS) for development/testing and on NVIDIA H200 (CUDA) for production benchmarks, differing only in configuration and performance | P0 |

---

## 9. Assumptions & Constraints

| # | Assumption / Constraint | Impact if invalid |
|---|---|---|
| A1 | TranslateGemma 12B fits on 2× H200 (141 GB × 2 = 282 GB HBM) at FP8 (~12 GB weights + KV cache + overhead) | If OOM, fallback to INT4 or reduce max batch size |
| A2 | FP8 quantisation maintains < 2 % quality degradation vs FP16 for this model on this language pair | If worse, re-evaluate quantisation strategy |
| A3 | A representative ≥ 5 GB ClearNet sample is available locally on an NVMe drive | If not, I/O becomes the bottleneck and invalidates the prediction |
| A4 | The golden reference set (1 000 sentences) is available and human-verified | Without it, quality metrics are unreliable |
| A5 | Both H200 GPUs are in the same NUMA node / PCIe switch for high-speed inter-GPU communication | If not, tensor-parallel all-reduce overhead increases |
| A6 | The 2 h window is long enough to reach steady-state throughput (excluding warm-up) | If there is significant thermal throttling after 2 h, the extrapolation will be optimistic |
| A7 | Apple Silicon Macs (M1–M4 series) provide sufficient unified memory (≥ 32 GB) to load TranslateGemma 12B at BF16 for development and smoke-testing (not for the 2 h benchmark itself) | If < 32 GB RAM, model must be quantised further (INT8/INT4) on macOS, or only a subset of the pipeline is testable |

---

## 10. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| OOM at FP8 with large batch size | Medium | High | Auto-tune batch size at startup; fallback to INT4 |
| Thermal throttling reduces throughput after 1 h+ | Medium | Medium | Log GPU temperature; note any clock-throttling events in the report; extend run to 4 h if throttling observed |
| I/O becomes the bottleneck (data loading slower than inference) | High | Medium | Use async prefetch with multiple worker threads; measure data-starvation time explicitly |
| Gemma tokeniser produces poor segmentations for Turkish morphology | Medium | Low | Document this as a known limitation; the benchmark is about throughput, not tokeniser quality |
| Run-to-run variance > 10 % | Low | Medium | Run 3× and report mean ± std; if variance persists, investigate non-determinism sources |
| NVMe fills up with translated output | Low | Low | Estimate output size upfront; ensure ≥ 500 GB free before starting |
| MPS-incompatible CUDA code slips into the codebase and is only discovered on H200 | Medium | High | CI pipeline includes a macOS MPS smoke test; all code paths are guarded by `if device.type == "cuda":` checks |

---

## 11. Timeline & Milestones

| Milestone | Description | Target |
|---|---|---|
| M0 | Documents signed off (PRD, SRS, SDD) | Day 0 |
| M1a | Apple Silicon dev environment set up (Python venv, PyTorch MPS, model download) | Day 1 |
| M1b | Model loading & single-batch inference working on MPS (smoke test) | Day 2 |
| M2 | Continuous translation loop + metrics logging working on MPS | Day 3 |
| M3 | Quality benchmark pipeline integrated; full dry-run passes on MPS | Day 4 |
| M4 | Code promoted to H200 node; environment provisioning (NVIDIA drivers, Docker, NVMe) | Day 5 |
| M5 | Model loading with FP8 + TP=2 working on H200; auto-tune batch size | Day 6 |
| M6 | First 2 h dry-run on H200; identify and fix bottlenecks | Day 7 |
| M7 | Three full 2 h benchmark runs completed on H200 | Day 8 |
| M8 | Final report with extrapolation delivered | Day 9 |

---

## 12. Glossary

| Term | Definition |
|---|---|
| **ClearNet** | The project's working name for a large, openly licensed web-crawl text corpus (analogous to CommonCrawl-derived datasets such as CulturaX [6.3 T tokens, 167 languages], FineWeb [15 T+ tokens], or HPLT [8 T tokens, 193 languages]). These corpora are overwhelmingly non-Turkish (Turkish ≈ 1 % of tokens). |
| **TranslateGemma 12B** | A 12-billion-parameter Gemma-family model fine-tuned for multilingual translation |
| **FP8** | 8-bit floating-point quantisation (E4M3 or E5M2 format); halves memory footprint vs FP16 |
| **H200** | NVIDIA H200 Tensor Core GPU with 141 GB HBM3e, ~4.8 TB/s memory bandwidth — the production benchmark target |
| **MPS** | Metal Performance Shaders — Apple's GPU acceleration backend for PyTorch on Apple Silicon (M1/M2/M3/M4 series). Used for development and smoke-testing; does not support FP8 or tensor parallelism in the same way as CUDA. |
| **Tensor Parallelism** | Splitting model weight matrices across GPUs so each GPU holds a shard; GPUs communicate via NVLink / NVSwitch during forward passes |
| **COMET** | Neural metric for MT quality evaluation; correlates better with human judgment than BLEU |
| **chrF** | Character n-gram F-score; robust for morphologically rich languages like Turkish |
| **Golden Reference Set** | A set of 1 000 English sentences with professionally verified Turkish translations, held out from the translation run and used only for quality evaluation |
| **SM Utilisation** | The fraction of Streaming Multiprocessor cycles spent on active warps (reported by `nvidia-smi`) |
| **Tokens Translated** | Output (Turkish) token count produced by the model, counted by the Gemma tokeniser decode step |


---

*This document is part of the historical spec set. See [`docs/README.md`](README.md)
for navigation, [`ARCHITECTURE.md`](ARCHITECTURE.md) for current reality, and
[`AI_CODING_ANTIPATTERNS.md`](AI_CODING_ANTIPATTERNS.md) for mistakes to avoid.*
