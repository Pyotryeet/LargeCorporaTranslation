# Software Requirements Specification (SRS)
## Turkish Corpus Translation Benchmark — Feasibility Study

---

| **Document** | Software Requirements Specification |
|---|---|
| **Project** | Turkish ClearNet Corpus Translation Benchmark |
| **Version** | 1.4 |
| **Status** | Implemented |
| **Author** | — |
| **Date** | 2026-06-19 |
| **Revised** | 2026-06-28 |
| **References** | PRD v3.7 |

---

> ℹ️ **Aligned spec.** This SRS has been updated to reflect the *actual, workable optimizations* identified during benchmarking and execution.
> The authoritative description of what the code *actually does* today is
> [`ARCHITECTURE.md`](ARCHITECTURE.md). Failed optimizations (TensorRT, vLLM, manual CUDA graphs replaying, FP8 KV-Cache) have been deactivated or marked as such, and the focus is shifted to verified paths: Data Parallelism (DP=2), Pinned-Memory pipeline, Decode Loop Vectorization, and Vocabulary-Pruned / Bilingual Custom Decoder architectures.

## Revision History

| Version | Date | Changes |
|---|---|---|
| 1.0 | 2026-06-19 | Initial draft — TranslateGemma 12B, CulturaX, Gemma 2 assumed |
| 1.1 | 2026-06-21 | Model corrected to Gemma 3 architecture. TranslateGemma 4B added for macOS dev. Tokenizer thread-safety, single-pass shuffle, powermetrics caching, frozen config, single-GPU CUDA support, E2E test suite (20 assertions). Real data integration (FineWeb sample-10BT + OPUS-100). |
| 1.2 | 2026-06-22 | Dependency versions updated to match v3.3 deployment. Quality metrics updated: BERTScore + COMET-22 + COMET-Kiwi replace BLEU + chrF++ as primary. Resume/checkpoint with position tracking added. Speculative decoding requirement added. orjson added to data pipeline. External sort shuffle added. |
| 1.3 | 2026-06-23 | v3.6 support: NLLB-200 encoder-decoder backend, model presets registry (11 presets), quantization levels (bf16/fp16/int8/int4), Ministral 3B, Gemma4 QAT (ct/int4/q4_0), DiffusionGemma 26B. --nllb/--model/--quantization/--paged-attention/--continuous-batching CLI flags. Dead code cleanup. |
| 1.4 | 2026-06-28 | Aligned requirements with actual optimizations: switched parallel execution from tensor-parallel to replicated process Data Parallelism (DP=2) to achieve near-linear 1.97x scaling, added Vocabulary Pruning + Custom decode loop requirements to meet the 200K+ TPS goal, removed FP8 KV cache / TensorRT / vLLM from production specifications, and updated quality metrics to use xCOMET-lite and paired bootstrap significance gates. |
| 1.5 | 2026-06-28 | Split execution backends into hardware-isolated files (`*_cuda.py` and `*_mps.py`) with transparent delegation proxies in the main directory. Codified distinct roles for MPS (verification, local QA, metric weight calibration) vs CUDA (extreme H200 hot-path performance). |

---

## 1. Introduction

### 1.1 Purpose

This SRS defines the complete set of functional and non-functional requirements for the **Turkish Corpus Translation Benchmark Harness** ("the System"). It serves as the binding specification against which implementation correctness is verified.

### 1.2 Scope

The System is a **single-node inference benchmark** that supports **two hardware backends**:

- **Development**: Apple Silicon (M1–M4 series) via PyTorch MPS — single-device inference at BF16/FP16.
- **Production**: 2× NVIDIA H200 GPUs via CUDA — FP8 precision with data parallelism (DP=2).

To isolate production optimizations from development constraints, the execution backends are split into separate files (`*_cuda.py` and `*_mps.py` variants) under `benchmark/inference/backends/`. The core modules (`autoregressive.py` and `nllb.py`) serve as system-agnostic dispatchers that dynamically delegate execution to the appropriate backend.

The System:
- Auto-detects the available backend at startup.
- Loads and serves a translation model (e.g. NLLB-600M, custom Bilingual-240M, or TranslateGemma 4B) for English → Turkish translation via the backend-dispatched protocol (autoregressive, encoder-decoder, custom plugin).
- Streams input text, translates, collects metrics, and runs a quality benchmark.
- Produces a report suitable for extrapolating full-ClearNet translation cost and duration.

### 1.3 Document Conventions

| Convention | Meaning |
|---|---|
| **FR-X** | Functional Requirement X |
| **NFR-X** | Non-Functional Requirement X |
| **DR-X** | Data Requirement X |
| **IR-X** | Interface Requirement X |
| **P0 / P1 / P2** | Priority: Must-have / Should-have / Nice-to-have |

---

## 2. System Overview

### 2.1 Context Diagram (Textual)

```
┌──────────────┐    English JSONL     ┌──────────────────────────┐    Metrics JSONL     ┌──────────────┐
│  ClearNet    │ ──────────────────►  │   Translation Benchmark  │ ──────────────────►  │  Report      │
│  Sample      │                      │   Harness                │                      │  Generator   │
│  (NVMe/SSD)  │                      │                          │                      │              │
└──────────────┘                      │  ┌──────────────────┐    │                      └──────┬───────┘
                                      │  │ NLLB-600M /      │    │                             │
┌──────────────┐    Device metrics    │  │ Bilingual-240M   │    │   Translated Text          │
│  nvidia-ml-py│ ◄──────────────────  │  │ CUDA (DP=2) or   │    │ ──────────────────────────►│
│  MPS monitor │                      │  │ MPS (single dev) │    │                      ┌──────▼───────┐
└──────────────┘                      │  └──────────────────┘    │                      │  Final       │
                                      │                          │                      │  Report      │
┌──────────────┐    Reference set     │  ┌──────────────────┐    │   Quality Scores      │  (JSON + MD)  │
│  Golden      │ ◄──────────────────  │  │ Quality          │    │ ──────────────────────►│              │
│  References  │                      │  │ Benchmark        │    │                      └──────────────┘
└──────────────┘                      │  │ (xCOMET/COMET-QE/│    │
                                      │  │  BLEU/chrF++)    │    │
                                      └──────────────────────────┘
```

### 2.2 Operational Modes

| Mode | Description | Duration | Hardware |
|---|---|---|---|
| **Backend Detection** | Auto-detect CUDA, MPS, or CPU; select precision and parallelism strategy | ~1 s | Any |
| **Warm-up** | 10–20 small batches to prime device kernels and clock speeds | ~30 s | Any |
| **Translation Run** | Continuous batched inference with full metric logging | |H200 (primary); Apple Silicon (scaled-down dev-test) |
| **Quality Benchmark** | Evaluate translation quality | ~5 min | Any |
| **Report Generation** | Aggregate all logs, compute statistics, produce report | ~1 min | Any |

---

## 3. Functional Requirements

### 3.1 Model Loading & Serving

| ID | Requirement | Priority | Verification |
|---|---|---|---|
| FR-01 | The System shall auto-detect the available compute backend at startup (`torch.cuda.is_available()`, `torch.backends.mps.is_available()`, CPU fallback) and load supported model presets (NLLB models, TranslateGemma 4B, or MADLAD 3B) onto the detected device. On CUDA: FP8 weights. On MPS: BF16 via native PyTorch. The backend shall be logged and included in the report. | P0 | Model loads without error on both backends; `torch.cuda.memory_allocated()` reports valid memory usage per GPU. |
| FR-02 | The System shall adapt parallelism to the backend: **CUDA** — data-parallel replication (DP=2) across exactly 2 GPUs (independent processing on separate GPUs, zero-overhead scaling); **MPS** — single-device inference (all layers on the one MPS device); **CPU** — single-device (all layers on CPU, for debugging only). | P0 | CUDA: identical model copies run on each GPU, translating independent batch slices. Forward pass runs without cross-device copy errors. |
| FR-03 | The System shall load the model tokenizer (SentencePiece BPE `.model` file) and verify joint EN-TR vocabulary coverage. | P0 | Tokenizer encodes/decodes round-trip without corruption. |
| FR-04 | The System shall use the best available attention kernel for each backend: FlashAttention-2 on CUDA, PyTorch native `scaled_dot_product_attention` on MPS. | P1 | Attention kernel is verified via `torch.backends.cuda.flash_sdp_enabled()` (CUDA) or `torch.backends.mps.is_available()` path check (MPS). |
| FR-05 | The System shall support an automatic batch-size tuner that increases batch size until it approaches OOM, then backs off by 15 % to establish the maximum sustainable batch size. | P1 | Batch-size tuner runs at startup; final batch size is logged. |

### 3.2 Data Ingestion

| ID | Requirement | Priority | Verification |
|---|---|---|---|
| FR-06 | The System shall read input text from one or more JSONL files where each line contains a JSON object with at minimum a `"text"` field. | P0 | File is opened and first line is parsed successfully. |
| FR-06b | The System shall strictly enforce PyArrow and Parquet pre-tokenization. If the pre-tokenized cache is missing at startup, it shall dynamically process the raw input dataset into a model-specific pre-tokenized Parquet file before starting the benchmark, completely bypassing dynamic CPU-bound tokenization during the translation loop. | P0 | Check that no dynamic tokenizer encoding occurs during active benchmark; Parquet file existence checked in ~/.cache/tr_benchmark/pretokenized/. |
| FR-07 | The System shall chunk input text into segments of at most `max_input_tokens` (configurable, default 512) tokens to avoid exceeding the model's context window. | P0 | No chunk exceeds `max_input_tokens` when tokenised. |
| FR-08 | The System shall filter out input segments that contain < 10 tokens (too short to be meaningful) or > 95 % non-ASCII garbage characters. | P1 | Filtered segments are counted and logged; throughput calculation excludes them. |
| FR-09 | The System shall use an async prefetch pipeline with at least 4 worker threads so that tokenisation and data loading do not stall the GPU. | P0 | GPU idle time due to data starvation is < 5 % of total runtime. |
| FR-10 | The System shall shuffle input documents at startup (deterministic seed) to avoid locality biases. | P1 | Seed is logged; two runs with the same seed produce the same input order. |

### 3.3 Translation Loop

| ID | Requirement | Priority | Verification |
|---|---|---|---|
| FR-11 | The System shall run a continuous batched inference loop that: (a) dequeues pre-tokenised input batches, (b) runs `model.generate()`, (c) decodes output token IDs to Turkish text, (d) writes output to disk in JSONL format. | P0 | Output file is non-empty after 60 s; all lines are valid JSON. |
| FR-12 | The System shall produce output JSONL lines containing: `input_text`, `translated_text`, `input_tokens`, `output_tokens`, `latency_ms`, `timestamp_utc`. | P0 | Spot-check 10 lines after run; all fields present and plausible. |
| FR-13 | The System shall run for exactly `target_duration_seconds` (10 mins), measured from the completion of warm-up to the last batch completion. Batches in-flight at the 10 minutes mark shall be allowed to complete ("graceful stop") but no new batches shall be started. | P0 | Actual runtime is within ±60 s of the target. |
| FR-14 | The System shall log a heartbeat every 10 s with: elapsed time, batches completed, tokens translated so far, current tokens/second. | P0 | Heartbeat lines appear in stdout and the log file. |
| FR-15 | Generation parameters shall be configurable via a YAML config file and shall default to: `max_new_tokens=512`, `temperature=0.0` (greedy decoding), `do_sample=False`, `num_beams=1`. | P0 | Config is read at startup; parameters are logged in the output. |

### 3.4 Metrics Collection

| ID | Requirement | Priority | Verification |
|---|---|---|---|
| FR-16 | **CUDA backend**: The System shall sample GPU metrics at 1 Hz using `pynvml` (NVIDIA Management Library) or DCGM, recording for each GPU: `gpu_utilization_pct`, `memory_used_gb`, `memory_total_gb`, `temperature_c`, `power_draw_w`, `sm_clock_mhz`, `memory_clock_mhz`. **MPS backend**: The System shall sample MPS metrics at 1 Hz using a combination of `psutil` for unified-memory utilisation and the macOS `powermetrics` command-line tool or `psutil.sensors_temperatures()` for thermal data. The metrics schema shall be backend-normalised so report-generation code is backend-agnostic. | P0 | CUDA: metrics log file contains ~7 200 rows per GPU for a 2 h run. MPS: metrics log file contains ~7 200 rows for the single device. |
| FR-17 | The System shall record per-batch metrics: `batch_id`, `batch_size`, `input_tokens_total`, `output_tokens_total`, `prefill_time_ms`, `decode_time_ms`, `total_latency_ms`. | P0 | Batch metrics file contains one row per batch processed. |
| FR-18 | The System shall record system-level metrics at 1 Hz: `cpu_utilization_pct`, `ram_used_gb`, `ram_total_gb`, `disk_read_mbps`, `disk_write_mbps`. | P1 | System metrics log file is non-empty. |
| FR-19 | The System shall calculate and log a rolling 60 s average throughput (tokens/second) at each heartbeat. | P0 | Rolling average is computed correctly; spot-check 5 values. |
| FR-20 | All metric timestamps shall use UTC ISO-8601 format with millisecond precision. | P0 | `grep` the log file; all timestamps match `\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z`. |

### 3.5 Quality Benchmarking

| ID | Requirement | Priority | Verification |
|---|---|---|---|
| FR-21 | The System shall load a golden reference set from a JSONL file containing at minimum `source_text` (English) and `reference_translation` (Turkish) fields. | P0 | File is loaded; `len(references) == 1000`. |
| FR-22 | The System shall translate all 1 000 source sentences using the same model, config, and decoding parameters used in the main run. | P0 | Translation completes for all 1 000 sentences. |
| FR-23 | The System shall compute **xCOMET-lite** (Unbabel/xcomet-lite) as the primary Tier 1 quality metric (reference-free or reference-based). | P0 | xCOMET-lite system score is formatted to 4 decimals. |
| FR-24 | The System shall compute **COMET-Kiwi** (reference-free) using the `Unbabel/wmt22-cometkiwi-da` model for quality estimation without reference dependency. | P0 | COMET-Kiwi score is in [0, 1]; formatted to 4 decimals. |
| FR-25 | The System shall compute **COMET-22** (reference-based) using the `Unbabel/wmt22-comet-da` model, reporting the system score (mean over all segments). | P0 | COMET score is in [0, 1]; formatted to 4 decimals. |
| FR-25b | The System shall compute **MetricX-24** (reference-based quality estimation) score, reporting the system-level score to assess translation quality. | P0 | MetricX-24 score is formatted to 4 decimals. |
| FR-26 | The System shall run a **paired bootstrap significance test** (with 95% CI) comparing candidate model segments against the baseline model to statistically validate quality differences. | P0 | Output logs show observed difference, CI bounds, and significance indicator. |
| FR-26b | The System shall execute translation of a designated 50-sentence representative English corpus on the Apple Silicon (MPS) development backend across all candidate models to generate comparative outputs. The candidate translations shall be rated by automated metrics (chrF++, spBLEU, COMET models, and MetricX-24). | P0 | Output logs confirm 50 sentences translated and scored across all presets. |
| FR-26c | The System shall support human-in-the-loop validation using a blind evaluation webpage. Evaluators shall blindly rate model outputs. Ratings shall be used to softmax-normalize metric weights to generate a single composite quality score (TTQS). Production H200 execution code is subsequently locked and hardened around the highest-scoring model. | P0 | Verified that overall TTQS is calculated using softmax-derived weights from evaluator feedback records. |

### 3.6 Report Generation

| ID | Requirement | Priority | Verification |
|---|---|---|---|
| FR-27 | The System shall produce a JSON report (`benchmark_report.json`) containing all aggregate metrics, statistics, and the extrapolation estimate. | P0 | File is valid JSON; `jq .` parses it without error. |
| FR-28 | The System shall produce a human-readable Markdown report (`benchmark_report.md`) with tables, summary statistics, and the extrapolation. | P1 | File is valid Markdown; renders correctly in a viewer. |
| FR-29 | The JSON report shall include: (a) configuration snapshot, (b) summary statistics (mean, median, p5, p95, std for all metrics), (c) time-series of throughput, (d) GPU utilisation distribution, (e) quality scores, (f) extrapolation parameters. | P0 | All six sections are present and non-empty. |
| FR-30 | The extrapolation shall compute: `estimated_days = (total_clearnet_non_tr_tokens) / (mean_tok_per_s * 3600 * 24)` with propagated uncertainty bounds. | P0 | Estimate is a positive float; bounds are computed and reported. |

### 3.7 Resilience & Checkpointing

| ID | Requirement | Priority | Verification |
|---|---|---|---|
| FR-31 | The System shall write a checkpoint every 5 minutes containing: number of documents processed, total tokens translated, current file name, current document ID, and elapsed seconds. | P1 | Checkpoint file is updated at least once every 300 s; timestamp is monotonically increasing. |
| FR-32 | The System shall handle SIGTERM / SIGINT by writing a final checkpoint (with position tracking) and flushing all log buffers before exiting. | P1 | Send `kill -TERM <pid>`; verify checkpoint and logs are complete. |
| FR-33 | On restart with `--resume <dir>`, the System shall load the latest checkpoint, seek the data loader to the saved position, restore batch/token counters, and continue translation from where it left off. | P1 | Run 30 min, kill, resume; verify no duplicate translations and no gaps. |

---

## 4. Non-Functional Requirements

### 4.1 Performance

| ID | Requirement | Priority | Target | Measurement |
|---|---|---|---|---|
| NFR-01 | Sustained translation throughput | P0 | ≥ 800 tok/s mean over 2 h | Tokens decoded / elapsed wall-clock |
| NFR-02 | GPU idle time (data starvation) | P0 | < 5 % of total runtime | Fraction of 1 Hz samples where GPU util < 20 % while queue is non-empty |
| NFR-03 | Batch prefill latency (p95) | P1 | < 2 000 ms for batch_size=max | Per-batch timing from the metrics log |
| NFR-04 | Batch decode latency per token (p95) | P1 | < 50 ms | Total decode time / output tokens per batch |
| NFR-05 | Quality benchmark duration | P1 | < 10 min for 1 000 sentences | Wall-clock time of the benchmark phase |

### 4.2 Reliability

| ID | Requirement | Priority | Target |
|---|---|---|---|
| NFR-06 | The System shall complete the full 2 h run without crashing under normal conditions (no hardware faults, sufficient disk space). | P0 | 3/3 runs complete without intervention. |
| NFR-07 | On CUDA: The System shall handle CUDA OOM gracefully by reducing batch size and retrying the failed batch. On MPS: The System shall handle MPS OOM (macOS unified memory exhaustion) by reducing batch size similarly. | P0 | Inject a batch-size spike; verify automatic back-off and log message on both backends. |
| NFR-08 | The System shall validate the available hardware at startup: on CUDA, verify that exactly 2 GPUs are present and NVLink is operational; on MPS, verify the MPS backend is available and log the Unified Memory size. Abort with a clear error message if pre-requisites are not met. | P1 | Set `CUDA_VISIBLE_DEVICES` to a single GPU (CUDA) or run on Intel Mac (MPS unavailable); verify clear error. |

### 4.3 Reproducibility

| ID | Requirement | Priority | Target |
|---|---|---|---|
| NFR-09 | All random seeds shall be set deterministically (Python `random`, NumPy, PyTorch, CUDA). | P0 | Seed values are logged; two runs with identical config produce identical output order. |
| NFR-10 | The System shall log its exact dependency versions (`pip freeze` or equivalent) at the start of each run. | P0 | `pip freeze` output is written to the run directory. |
| NFR-11 | Run-to-run throughput variance (coefficient of variation across 3 identical runs) shall be ≤ 5 %. | P1 | CV = std(throughputs) / mean(throughputs) ≤ 0.05. |

### 4.4 Observability

| ID | Requirement | Priority | Target |
|---|---|---|---|
| NFR-12 | All log output shall be structured (JSON Lines) to enable automated parsing. | P0 | Every log line parses as valid JSON. |
| NFR-13 | The System shall emit structured logs to both stdout and a log file. | P0 | Log file exists and is non-empty post-run. |
| NFR-14 | The System shall print a human-readable status line every 10 s with: elapsed, throughput, GPU%, memory, tokens so far. | P1 | Status line is visible in `docker logs` output. |

### 4.5 Portability

| ID | Requirement | Priority | Target |
|---|---|---|---|
| NFR-15 | On H200/Linux: The System shall run inside a Docker container built from a provided `Dockerfile`. | P0 | `docker build && docker run` succeeds on the target H200 node. |
| NFR-16 | The Docker image shall pin exact versions of all Python packages. | P0 | `Dockerfile` uses `==` constraints, not `>=`. |
| NFR-17 | The System shall read all configuration from a single YAML file. | P0 | Changing the YAML changes behaviour without rebuilding. |
| NFR-18 | On Apple Silicon/macOS: The System shall run natively via `pip install -e . && python -m benchmark --config config.yaml` without Docker (macOS Docker does not support GPU passthrough). Apple Silicon (MPS) and NVIDIA (CUDA) execution logic shall be structurally isolated into dedicated, platform-specific files (`*_mps.py` vs `*_cuda.py`) with transparent system-agnostic dispatcher facades in the main directory, avoiding environment dependency pollution and allowing target-specific optimization. | P0 | `pip install -e . && python -m benchmark --config config.yaml --dry-run` succeeds on an M-series Mac with ≥ 32 GB Unified Memory. |

---

## 5. Data Requirements

### 5.1 Input Data

| ID | Requirement | Priority |
|---|---|---|
| DR-01 | The ClearNet sample shall be provided as one or more gzip-compressed JSONL files (`.jsonl.gz`) stored on the local NVMe volume. | P0 |
| DR-02 | Each JSONL line shall contain at minimum a `"text"` field with UTF-8 encoded English text. | P0 |
| DR-03 | The total uncompressed size of the input sample shall be ≥ 20 GB to ensure the 2 h run does not exhaust the input. | P0 |
| DR-04 | The input data shall not contain any Turkish text (to avoid translating already-Turkish content). A language-ID pre-filter may be applied offline before the run. | P1 |

### 5.2 Golden Reference Data

| ID | Requirement | Priority |
|---|---|---|
| DR-05 | The golden reference set shall contain exactly 1 000 English–Turkish sentence pairs. | P0 |
| DR-06 | All Turkish references shall be human-verified or professionally translated (not machine-translated). | P0 |
| DR-07 | The reference set shall cover a diverse range of domains (news, web, technical, conversational) to match the ClearNet distribution. | P1 |
| DR-08 | The reference set shall be stored in a separate file from the main input corpus and shall be excluded from the translation run. | P0 |

### 5.3 Output Data

| ID | Requirement | Priority |
|---|---|---|
| DR-09 | Translated output shall be written to `output/translations/` as gzip-compressed JSONL, sharded into files of ≤ 100 MB each. | P0 |
| DR-10 | Metrics shall be written to `output/metrics/` with subdirectories for `gpu/`, `batch/`, and `system/`. | P0 |
| DR-11 | The final report shall be written to `output/report/`. | P0 |
| DR-12 | All output shall be written to a timestamped run directory: `output/<YYYY-MM-DD_HH-MM-SS>/`. | P0 |

---

## 6. External Interface Requirements

### 6.1 Command-Line Interface

| ID | Requirement | Priority |
|---|---|---|
| IR-01 | The System shall provide a single entrypoint script: `python -m benchmark --config config.yaml`. | P0 |
| IR-02 | The `--config` flag shall accept a path to a YAML configuration file. All parameters in §3.3.15 shall be overridable via this file. | P0 |
| IR-03 | The System shall support `--resume <checkpoint_path>` to restart from a checkpoint. | P1 |
| IR-04 | The System shall support `--benchmark-only` to skip the 2 h translation run and only run the quality benchmark on previously translated output. | P1 |
| IR-05 | The System shall support `--dry-run` to validate configuration, load the model, run 10 batches, and exit. | P1 |

### 6.2 Configuration File Format

| ID | Requirement | Priority |
|---|---|---|
| IR-06 | The configuration file shall be valid YAML 1.2. | P0 |
| IR-07 | The configuration schema shall be documented with comments in a reference config file shipped with the repository. | P1 |
| IR-08 | Unknown or misspelled configuration keys shall cause a startup error with a descriptive message, not be silently ignored. | P1 |

### 6.3 File System Interface

| ID | Requirement | Priority |
|---|---|---|
| IR-09 | All input and output paths shall be configurable via the YAML config file. | P0 |
| IR-10 | The System shall verify that all input paths exist and are readable at startup. | P0 |
| IR-11 | The System shall verify that the output directory has sufficient free space (≥ 200 GB) before starting the translation run. | P0 |

---

## 7. System Constraints

| ID | Constraint |
|---|---|
| SC-01 | The System shall operate on: (a) **Apple Silicon Mac** (M1–M4 series, ≥ 32 GB Unified Memory) for development/smoke-testing with MPS backend; (b) **2× NVIDIA H200 GPUs** with NVLink / NVSwitch connectivity for the production 2 h benchmark run. Platform execution logic is isolated into target-specific files (`*_mps.py` vs `*_cuda.py`) with system-agnostic facades, sharing identical model presets and data loader schema. |
| SC-02 | The System shall not require more than 256 GB of system RAM (H200 node) or more than the available Unified Memory minus 4 GB (Apple Silicon). |
| SC-03 | The System shall not produce more than 500 GB of output data over a single 2 h run. |
| SC-04 | The System shall be implemented in Python 3.11+ with PyTorch 2.4+ as the primary ML framework. |
| SC-05 | **CUDA**: FP8 inference shall use the `transformer-engine` library (NVIDIA) or PyTorch's native `torch.ao.float8` with `te.Linear` replacements and `fp8_autocast` context (implemented in `*_cuda.py`). **MPS**: BF16 inference via native PyTorch (implemented in `*_mps.py`); FP8 is not supported on MPS. Platform execution files prevent dynamic runtime import errors. GPU telemetry uses `nvidia-ml-py` (replaces deprecated `pynvml`). |
| SC-06 | The System shall not depend on any paid or proprietary software beyond NVIDIA's freely available CUDA stack (CUDA runtime only; no paid licences). On macOS, the CUDA dependencies are not required. |
| SC-07 | Network access is not required during the translation run (model weights and data are pre-staged). |

---

## 8. Assumptions & Dependencies

### 8.1 Shared Dependencies (Both Platforms)

| ID | Dependency | Version / Details |
|---|---|---|
| D-01 | Python | ≥ 3.11 (3.12.3 deployed) |
| D-03 | HuggingFace Transformers | ≥ 4.47.0 (4.57.6 deployed) |
| D-07 | SacreBLEU | ≥ 2.4.0 (2.6.0 deployed) |
| D-08 | COMET (Unbabel) | ≥ 2.2.0 (2.2.7 deployed) |
| D-08b | PyArrow | ≥ 15.0.0 (15.0.2 deployed) — for enforced pre-tokenized cache loading |
| D-12 | NLLB, TranslateGemma 4B, MADLAD 3B checkpoints | BF16 (MPS) and FP8/BF16 (CUDA) variants pre-quantised and verified |
| D-13 | ClearNet English sample | ≥ 20 GB uncompressed on local SSD/NVMe |
| D-14 | Golden reference set | 1 000 EN→TR pairs (1,960 deployed) |

### 8.2 CUDA/H200 Dependencies (Production)

| ID | Dependency | Version / Details | Conditional Import |
|---|---|---|---|
| D-02a | PyTorch (CUDA) | ≥ 2.4.0 with CUDA 12.4+ (2.6.0+cu124 deployed) | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| D-04 | NVIDIA Transformer Engine | ≥ 2.0 (2.16.0 deployed) | `try: import transformer_engine` |
| D-05 | FlashAttention-2 | ≥ 2.6.0 (2.8.3 deployed) | `try: import flash_attn` |
| D-06 | nvidia-ml-py | ≥ 13.0 (13.610.43 deployed) — replaces deprecated pynvml | `try: import nvidia_ml_py` |
| D-09 | CUDA Toolkit | 12.4 or 12.5 (nvcc 12.0 system, CUDA 12.4 PyTorch, CUDA 13.0 TRT) | System-level |
| D-10 | NVIDIA Driver | ≥ 550 (580.159.03 deployed) | System-level |
| D-11 | Docker | ≥ 24.0 with `--gpus` support | System-level |

### 8.3 Apple Silicon / MPS Dependencies (Development)

| ID | Dependency | Version / Details | Conditional Import |
|---|---|---|---|
| D-02b | PyTorch (MPS) | ≥ 2.4.0 (nightly recommended for latest MPS ops) | `pip install torch torchvision` (no `--index-url`) |
| D-15 | macOS | ≥ 14.0 (Sonoma) for full MPS coverage of all ops used |
| D-16 | Apple Silicon Mac | M1 Max / M2 / M3 / M4 series with ≥ 32 GB Unified Memory | Hardware |

**Conditional requirements files**:
- `requirements.txt` — shared dependencies.
- `requirements-cuda.txt` — CUDA-specific (`transformer-engine`, `pynvml`, `flash-attn`).
- `requirements-mps.txt` — MPS-specific (effectively empty; PyTorch MPS is built-in).

---

## 9. Traceability Matrix

| PRD Goal | SRS Requirements |
|---|---|
| G1 — Measure throughput | FR-11, FR-13, FR-19, NFR-01 |
| G2 — Characterise GPU utilisation | FR-16, FR-17, NFR-02 |
| G3 — Identify bottleneck | FR-09, FR-16, FR-17, FR-18, NFR-02 |
| G4 — Measure translation quality | FR-21 through FR-26, NFR-05 |
| G5 — Produce full-dataset estimate | FR-27 through FR-30 |
| G6 — Reproducibility | FR-10, NFR-09, NFR-10, NFR-11, NFR-15, NFR-16 |

---

## 10. Performance Verification & Measurement Plan (Merged Plan)

To validate non-functional constraints and obtain empirical constants for production calculations, the system uses a 24-point measurement plan categorized by priority.

### 10.1 Priority Classification
*   **P0 (Extrapolation & Cost)**: Directly impacts final dataset extrapolation estimates.
*   **P1 (Memory & Compute Calibration)**: Calibration constants for batch-size tuners, preventing OOM crashes.
*   **P2 (Throughput & Latency Baselines)**: Individual feature-level speedup/memory gains.
*   **P3 (Pipeline & Data Constants)**: Queue sizes, I/O rates, and telemetry collection overheads.
*   **P4 (Quality Targets & Degradation)**: Long-run quality decay and significance thresholds.

### 10.2 The 24 Core Measurements

#### P0 — Extrapolation & Cost
*   **M0.1 — Sustained Throughput Baseline**: Measures steady-state tokens/sec for each model/backend/precision over a minimum of 1 hour. Target variance (CV) across 3 runs ≤ 5%.
*   **M0.2 — Tokenization Overhead**: Measures character-per-token and bytes-per-token ratios on ClearNet data. Replaces theoretical constant `BYTES_PER_INPUT_TOKEN = 4.0`.
*   **M0.3 — Corpus Token Count Validation**: Validates the non-TR fraction of the 6.23T target tokens using language detection sampling.
*   **M0.4 — Real GPU Cost per Hour**: Cloud provider or amortized hardware costs for H200 instances.
*   **M0.5 — Throughput Degradation over Time**: Run 4-hour benchmarks to measure slope of TPS over time and calculate max stable extrapolation horizons.

#### P1 — Memory & Compute Calibration
*   **M1.1 — Actual GPU Memory Budget**: Measures CUDA context, workspace, and model memory overheads to establish safe headrooms.
*   **M1.2 — KV-Cache Memory per Token**: Computes and allocates dummy caches to verify the actual bytes per token (head_dim × kv_heads × layers × precision).
*   **M1.3 — Batch-Size Ceiling**: Binary searches optimal batch size for highest throughput rather than just OOM limits.
*   **M1.4 — torch.compile Memory Overhead**: Records VRAM compilation spikes and steady-state allocations under `mode="reduce-overhead"`.
*   **M1.5 — TE FP8 Memory & Throughput**: Compares FP8 vs BF16 memory usages and sustained execution speeds.

#### P2 — Throughput & Latency Baselines (Isolation Runs)
*   **M2.1 — torch.compile Speedup**: Measures Welch t-test significance of compiler vs eager mode.
*   **M2.2 — Continuous Batching Throughput**: Compares dynamic iteration scheduling vs static batching.
*   **M2.3 — Speculative Decoding Throughput**: Measures serial speedup factor at `batch_size=1`.
*   **M2.4 — Flash SDPA Speedup**: Quantifies FlashAttention speed improvement.
*   **M2.5 — Pinned Memory H2D Speedup**: Times transfers of pinned vs pageable tensors.
*   **M2.6 — PagedAttention Memory Savings**: Compares block-based allocation vs contiguous pre-allocation memory.
*   **M2.7 — INT4/INT8 Weight Quantization Speedup**: Measures memory savings and execution speed comparison.
*   **M2.8 — Utility Speedups**: Measures speed improvements of `orjson`, `pigz`, and vectorized filters.

#### P3 — Pipeline & Data Constants
*   **M3.1 — Queue Wait Latency**: Times thread block durations in `AsyncPipeline`.
*   **M3.2 — Parquet Read/Write Speed**: Records disk-bound throughput of PyArrow Parquet files.
*   **M3.3 — Prometheus Collection Latency**: Measures overhead of telemetry polling.
*   **M3.4 — Host Memory Leakage**: Monitors RAM usage growth over 2-hour runs.

#### P4 — Quality Targets & Degradation
*   **M4.1 — FP8 Quality Ceiling**: Calculates quality degradation of FP8 vs BF16 across Golden References.
*   **M4.2 — Speculative Acceptance Rate**: Tracks percentage of tokens accepted from draft models.
*   **M4.3 — Morphological Pre-processing Impact**: Measures BLEU gains from CSE segmentations.
*   **M4.4 — Bootstrap Gate Calibration**: Establishes significance thresholds for paired bootstrap tests.

---

## 11. Implementation & Verification Matrix (Merged Plan)

### 11.1 Priority & Effort Matrix

| Priority | Tier | Optimization | Expected Impact | Implementation Effort | Dependencies |
|---|---|---|---|---|---|
| 🔴 P0 | 1 | Data Parallelism (DP=2) | ~2.0× speedup | Low (1-2 days) | None |
| 🔴 P0 | 2 | Decode Loop Vectorization | ~1.2× speedup | Low (1 day) | None |
| 🟡 P1 | 3 | PagedAttention (AR Path) | ~1.8× speedup | Medium (2-3 days) | None |
| 🟡 P1 | 4 | torch.compile Upgrade | ~1.3× speedup | Medium (1-2 days) | PyTorch 2.14+ |
| 🟢 P2 | 5 | Pipeline Overlap | ~1.1× speedup | Low (1 day) | None |

### 11.2 Verification Commands
To verify the performance steps and regression boundaries, run the following benchmark commands:
```bash
# Baseline Run (1 GPU, bs=512)
./run.sh --model translategemma-4b-bf16 --batch-size 512 --duration 300

# Tier 1 (Data Parallelism - 2 GPUs)
./run.sh --model translategemma-4b-bf16 --batch-size 512 --data-parallel 2 --duration 300

# Tier 3 (PagedAttention)
./run.sh --model translategemma-4b-bf16 --batch-size 2048 --data-parallel 2 --paged-attention --duration 300

# Quality Verification Gate (Mandatory)
./run.sh --model translategemma-4b-bf16 --benchmark-only --batch-size 64
```

---

*This document is part of the historical spec set. See [`docs/README.md`](README.md)
for navigation, [`ARCHITECTURE.md`](ARCHITECTURE.md) for current reality, and
[`AI_CODING_ANTIPATTERNS.md`](AI_CODING_ANTIPATTERNS.md) for mistakes to avoid.*
