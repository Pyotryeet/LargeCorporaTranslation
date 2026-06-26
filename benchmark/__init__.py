"""Turkish Corpus Translation Benchmark Harness — v3.6.

Model-agnostic EN→TR translation benchmarking on NVIDIA H200 GPUs.
One command to install, one command to run.

Key packages
------------
config          — Pydantic v2 schema, model presets, capability registry
data            — JSONL/Parquet loader, tokenizer pipeline, pre-tokenization
hardware        — backend detection, precision/FP8, parallelism
inference       — model-agnostic engine, AR/NLLB/Diffusion/TRT backends
metrics         — GPU/system samplers, O(1) throughput, batch logger
observability   — Prometheus exporter, Nsight profiler
orchestration   — harness, checkpoint, signal handler
quality         — BERTScore, COMET-22, COMET-Kiwi, BLEU, chrF++
reporting       — aggregation, extrapolation, JSON/Markdown writers
utils           — timer, preflight checks, logging, JSON sanitization
"""

from benchmark.utils.version import VERSION as __version__
