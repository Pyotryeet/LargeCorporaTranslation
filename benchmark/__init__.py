"""Turkish Corpus Translation Benchmark Harness — v3.8.

Model-agnostic EN→TR translation benchmarking on NVIDIA H200 GPUs.
One command to install, one command to run.

Key packages
------------
config          — Pydantic v2 schema, model presets, capability registry
data            — JSONL/Parquet loader, tokenizer pipeline, pre-tokenization
hardware        — backend detection, precision/FP8, parallelism
inference       — backend-dispatched engine, AR/NLLB encoder-decoder backends
metrics         — GPU/system samplers, O(1) throughput, batch logger
observability   — Prometheus exporter
orchestration   — harness, checkpoint, signal handler
quality         — BERTScore, COMET-22, COMET-Kiwi, MetricX-24, BLEU, chrF++
reporting       — aggregation, extrapolation, JSON/Markdown writers
utils           — timer, preflight checks, logging, JSON sanitization
"""

from benchmark.utils.version import VERSION as __version__
