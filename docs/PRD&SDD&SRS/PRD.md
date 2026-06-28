# Product Requirements Document (PRD)
## Turkish Corpus Translation Benchmark — Feasibility Study

---

| **Document** | Product Requirements Document |
|---|---|
| **Project** | Turkish ClearNet Corpus Translation Benchmark |
| **Version** | 3.6 |
| **Status** | Implementation Stage |
| **Author** | — |
| **Date** | 2026-06-19 |
| **Revised** | 2026-06-23 |

---

> ℹ️ **Aligned spec.** This PRD has been updated to reflect the *actual, workable optimizations* identified during benchmarking and execution.
> The authoritative description of what the code *actually does* today is
> [`ARCHITECTURE.md`](ARCHITECTURE.md). Failed optimizations (TensorRT, vLLM, manual CUDA graphs replaying, FP8 KV-Cache) have been deactivated or marked as such, and the focus is shifted to verified paths: Data Parallelism (DP=2), Pinned-Memory pipeline, Decode Loop Vectorization, and Vocabulary-Pruned / Bilingual Custom Decoder architectures.

---

## Revision History

| Version | Date | Changes |
|---|---|---|
| 1.0 | 2026-06-19 | Initial draft |
| 1.1 | 2026-06-21 | Reflects implementation reality: TranslateGemma 4B added as dev model, macOS MPS baselines established, E2E test suite completed, real data integrated |
| 3.2 | 2026-06-21 | Model-agnostic backend protocol, diffusion support, plugin system, TensorRT, 40 extreme optimizations |
| 3.3 | 2026-06-22 | Speculative decoding (self-spec + draft-model), resume/checkpoint with position tracking, extrapolation CI fix (SEM-based + bootstrap), external sort shuffle, H200 production deployment fixes |
| 3.6 | 2026-06-23 | NLLB-200 encoder-decoder backend (600M–3B), model presets registry (11 presets), quantization levels (bf16/fp16/int8/int4), Ministral 3B, TranslateGemma 4B, --nllb/--model/--quantization/--paged-attention/--continuous-batching CLI flags, dead code cleanup (removed 4 modules: back_translate, domain_classifier, rust_tokenizer, mmap_reader; 4 stale config files; 2 stale shell scripts) |
| 3.7 | 2026-06-28 | Aligned PRD requirements with actual optimizations that work: switched target production parallelism to Data Parallelism (DP=2), added Vocabulary Pruning + Custom Greedy Decoder architecture requirements to hit the 200K+ TPS target, removed FP8 KV cache / TensorRT / vLLM from production path, and specified xCOMET-lite + paired bootstrap statistical significance gates for evaluation. |
| 3.8 | 2026-06-28 | Split execution backends into separate CUDA (`*_cuda.py`) and MPS (`*_mps.py`) files with system-agnostic dispatcher wrappers on the main directory to isolate production hot-path optimizations from dev/test validation. |

---

## 1. Executive Summary

The Turkish language is underrepresented in public training corpora relative to English and other high-resource languages. The **ClearNet dataset** — a large, openly available web-crawl text corpus — contains predominantly non-Turkish text. Translating even a fraction of this dataset would substantially increase the volume of Turkish text available for language model pre-training and fine-tuning.

However, running a full translation is a multi-month, multi GPU-intensive undertaking. Before committing to that cost, we need a **rigorous feasibility benchmark** that answers a single predictive question:

> *Using 2× NVIDIA H200 GPUs running **TranslateGemma 4B** at FP8 precision, how long would it take to translate the entire non-Turkish ClearNet dataset into Turkish?*

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

**Key takeaway**: Non-Turkish content in CulturaX alone totals **≈ 6.23 trillion tokens**. Turkish represents only 1.02 % of the corpora.

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

A **sufficiently large Turkish corpus** (target: ≥200B tokens) suitable for pre-training competitive Turkish LLMs.

### 2.3 Gap

We know:
- The **sustained throughput** of 73,770 TPS for NLLB-200 600M model baseline with extreme optimizations.
- The **feasibility of 200K+ TPS** using a vocabulary-pruned (50K active tokens) or bilingual model (e.g. Bilingual-240M) and custom decode loops (e.g. `fast_decode_batch` with vectorized EOS detection and no CPU-GPU syncs).
- The **hardware utilization efficiency** where AR inference is memory-bandwidth-bound during weight loads but heavily throttled by Python/HuggingFace overhead (27ms vs 0.5ms weight read time), which is bypassed by custom generators.

It will take around 10–12 days of uninterrupted 2x Hopper 200 GPUs to translate 700GB (~170B tokens) at 180K–220K TPS.

We do not know:

- The exact **quality ceiling** of pruned vs fresh bilingual models on diverse web-crawl segments.
- The **precise wall-clock time** and memory boundary for maximum-batch operations.

---

## 3. Project Vision

Build a **self-contained accuracy benchmarking harness** that:

1. Loads all supported candidate models (quantized to **FP8** where supported).
2. Streams a representative sample of the ClearNet English corpus through the model using PyArrow Parquet pre-tokenization.
3. Translates a selected 50-sentence corpus across all candidate models on the macOS (MPS) development path.
4. Logs and rates translations using all automated quality metrics (chrF++, spBLEU, COMET-22, COMET-Kiwi, and MetricX-24).
5. Deploys a blind human rating webpage where native Turkish speakers evaluate candidate translations.
6. Computes a single composite Quality Score (TTQS) by applying softmax to weights derived from human correlation feedback.
7. Selects the single best-performing model, discards the rest, and builds highly optimized CUDA code (Triton kernels, Inductor compile, NCCL parallelism) around the chosen model to run extremely fast on 2× H200 GPUs.
8. Performs the large-scale translation run on the full 200B Tokens dataset.

---

## 4. Goals & Objectives

| # | Goal | Type | Success Metric |
|---|---|---|---|
| G1 | Measure translation quality | Quantitative | xCOMET-lite (primary) ≥ 0.72, COMET-Kiwi (QE) ≥ 0.70, spBLEU ≥ 30 on FLORES-200 reference sets via paired bootstrap significance gates (95% CI) |
| G2 | Produce a full-dataset time/cost estimate | Predictive | Extrapolated days-to-completion with upper/lower bounds |
| G3 | Reproducibility | Process | Single-command launch; all config captured in the output; run-to-run variance < 5 % |

---

## 5. Scope

### 5.1 In Scope

- **Hardware abstraction layer** that auto-detects the backend (CUDA/MPS) at startup, adapting model loading, precision.
- Loading models in our model list with FP8 weights from a pre-quantised checkpoint.
- Data parallelism with extreme bathcing on multiple GPU devices.
- Streaming English text from a local ClearNet sample.
- A quality benchmark run to collect translated sentences for human evaluation.
- Aggregation script that produces a single JSON + Markdown report.

### 5.2 Out of Scope

- Actual translation of the full 200B token data.
- Training or fine-tuning any model.
- Multi-node (≥2 machine) orchestration.
- A web UI or dashboard (terminal + file output only).
- Turkish-specific tokeniser training (we use the Gemma tokeniser as-is, with its multilingual vocabulary).
- Streaming from cloud object storage (all data is local on the NVMe volume).

---

## 8. High-Level Requirements

| ID | Requirement | Priority |
|---|---|---|
| HL-1 | The system shall load model 4B at maximum in FP8 precision on 2 GPUs without tensor parallelism| P0 |
| HL-2 | The system shall translate English text to Turkish at the highest sustainable throughput achievable on the hardware | P0 |
| HL-3 | The system shall log all hardware metrics at 1 Hz granularity for the full 2 h run | P0 |
| HL-4 | The system shall compute and log per-batch translation latency and throughput | P0 |
| HL-5 | The system shall run a translation-quality benchmark at the conclusion of the 2 h window | P0 |
| HL-6 | The system shall produce a single aggregated report containing all metrics and the extrapolated full-dataset estimate | P0 |
| HL-7 | The system shall checkpoint progress at least every 5 minutes so a crash does not lose all data | P1 |
| HL-8 | The system shall validate the model output is valid UTF-8 Turkish text (sanity gate) | P1 |
| HL-9 | The system shall be fully containerised and reproducible | P1 |
| HL-10 | The system shall auto-detect the available compute backend (CUDA, MPS, or CPU) at startup and select the appropriate model-loading path, precision, and metrics collection strategy | P0 |

---

## 9. Assumptions & Constraints

| # | Assumption / Constraint | Impact if invalid |
|---|---|---|
| A1 | 4B model fits on 2× H200 (141 GB × 2 = 282 GB HBM) at FP8 (~12 GB weights + KV cache + overhead) | The project isnt feasable.
| A2 | FP8 quantisation maintains < 2 % quality degradation vs FP16 for this model on this language pair | If worse, re-evaluate quantisation strategy |
| A3 | A representative ≥ 5 GB ClearNet sample is available locally on an NVMe drive | If not, I/O becomes the bottleneck and invalidates the prediction |
| A4 | The accuracy metrics are human verified | Without it, quality metrics are unreliable |

---

## 10. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| OOM at FP8 with large batch size | Medium | High | Auto-tune batch size at startup; fallback to INT4 |
| I/O becomes the bottleneck (data loading slower than inference) | High | Medium | Use async prefetch with multiple worker threads; measure data-starvation time explicitly |
| Gemma tokeniser produces poor segmentations for Turkish morphology | Medium | Low | Document this as a known limitation; the benchmark is about throughput and quality, not tokeniser quality |
| Run-to-run variance > 10 % | Low | Medium | Run 3× and report mean ± std; if variance persists, investigate non-determinism sources |
| NVMe fills up with translated output | Low | Low | Estimate output size upfront; ensure ≥ 500 GB free before starting |

---

## 12. Glossary

| Term | Definition |
|---|---|
| **ClearNet** | The project's working name for a large, openly licensed web-crawl text corpus (analogous to CommonCrawl-derived datasets such as CulturaX [6.3 T tokens, 167 languages], FineWeb [15 T+ tokens], or HPLT [8 T tokens, 193 languages]). These corpora are overwhelmingly non-Turkish (Turkish ≈ 1 % of tokens). |
| **TranslateGemma 12B** | A 12-billion-parameter Gemma-family model fine-tuned for multilingual translation |
| **FP8** | 8-bit floating-point quantisation (E4M3 or E5M2 format); halves memory footprint vs FP16 |
| **H200** | NVIDIA H200 Tensor Core GPU with 141 GB HBM3e, ~4.8 TB/s memory bandwidth — the production benchmark target |
| **MPS** | Metal Performance Shaders — Apple's GPU acceleration backend for PyTorch on Apple Silicon (M1/M2/M3/M4 series). Used for development and smoke-testing; does not support FP8 or tensor parallelism in the same way as CUDA. |
| **Data Parallelism** | Running independent copies of the model on different GPUs, splitting the batch of data rather than the model weights, achieving near-perfect scaling without inter-GPU communication overhead |
| **COMET** | Neural metric for MT quality evaluation; correlates better with human judgment than BLEU |
| **chrF** | Character n-gram F-score; robust for morphologically rich languages like Turkish |
| **Golden Reference Set** | A set of 1 000 English sentences with professionally verified Turkish translations, held out from the translation run and used only for quality evaluation |
| **SM Utilisation** | The fraction of Streaming Multiprocessor cycles spent on active warps (reported by `nvidia-smi`) |
| **Tokens Translated** | Output (Turkish) token count produced by the model, counted by the Gemma tokeniser decode step |


---

*This document is part of the historical spec set. See [`docs/README.md`](README.md)
for navigation, [`ARCHITECTURE.md`](ARCHITECTURE.md) for current reality, and
[`AI_CODING_ANTIPATTERNS.md`](AI_CODING_ANTIPATTERNS.md) for mistakes to avoid.*
