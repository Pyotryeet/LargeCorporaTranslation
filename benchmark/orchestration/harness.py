"""Top-level benchmark harness — owns the lifecycle of all subsystems.

Run modes
---------
  full           2-hour production benchmark (default)
  quick          5-minute evaluation run
  dry-run        10-batch smoke test, then exit
  warmup-only    Load model + warm-up, then exit
  benchmark-only Skip translation entirely, run quality benchmark only
  translate-only Skip quality benchmark after translation
  resume         Continue from a checkpoint directory

Override parameters
-------------------
  batch_size_override   Force a specific batch size (None = auto-tune)
  duration_override     Override target_duration_seconds from config
  seed_override         Override random seed
  resume_dir            Checkpoint directory for resume
  no_torch_compile      Disable torch.compile (useful for debugging)

v2.0: Passes backend info to AsyncPipeline for pinned memory decisions.
"""

import logging
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, TYPE_CHECKING

import torch

from benchmark.config.schema import BenchmarkConfig
from benchmark.hardware.backend import detect_backend, DeviceInfo
from benchmark.data.loader import JSONLLoader
from benchmark.data.chunker import TextChunker
from benchmark.data.filters import ChunkFilter
from benchmark.data.pipeline import AsyncPipeline
from benchmark.inference.engine import InferenceEngine
from benchmark.inference.batch_tuner import BatchSizeTuner
from benchmark.inference.sampling import DecodingParams
from benchmark.metrics.collector import MetricsCollector
from benchmark.quality.benchmark import QualityBenchmark
from benchmark.reporting.aggregator import MetricsAggregator
from benchmark.reporting.extrapolation import ExtrapolationModel
from benchmark.reporting.json_report import JSONReportWriter
from benchmark.reporting.markdown_report import MarkdownReportWriter
from benchmark.orchestration.checkpoint import CheckpointManager
from benchmark.orchestration.signals import SignalHandler, register_cleanup
from benchmark.utils.logging_setup import setup_logging
from benchmark.utils.env_check import run_preflight_checks
from benchmark.utils.version import get_environment_snapshot
from benchmark.utils.timer import PrecisionTimer

if TYPE_CHECKING:
    from benchmark.observability.prometheus_metrics import PrometheusExporter

logger = logging.getLogger(__name__)


class BenchmarkHarness:
    """Top-level orchestrator — one instance per run."""

    def __init__(
        self,
        config_path: str,
        *,
        run_mode: Literal[
            "full", "quick", "dry-run", "warmup-only",
            "benchmark-only", "translate-only", "resume",
        ] = "full",
        batch_size_override: Optional[int] = None,
        duration_override: Optional[int] = None,
        seed_override: Optional[int] = None,
        resume_dir: Optional[str] = None,
        no_torch_compile: bool = False,
        safe_mode: bool = False,
        observability_enabled: bool = False,
    ):
        from benchmark.config.schema import load_config

        self.config = load_config(config_path)
        self.run_mode = run_mode
        self.batch_size_override = batch_size_override
        self.duration_override = duration_override
        self.seed_override = seed_override
        self.resume_dir = resume_dir
        self.no_torch_compile = no_torch_compile
        self.safe_mode = safe_mode
        self.observability_enabled = observability_enabled
        if self.observability_enabled:
            self.config = self.config.model_copy(update={
                "runtime": self.config.runtime.model_copy(
                    update={"observability_enabled": True},
                ),
            })

        self.run_dir: Path | None = None
        self.device_info: DeviceInfo | None = None
        self.engine: InferenceEngine | None = None
        self.pipeline: AsyncPipeline | None = None
        self.metrics: MetricsCollector | None = None
        self.checkpoint_mgr: CheckpointManager | None = None
        self.signal_handler: SignalHandler | None = None
        self._prometheus: Optional['PrometheusExporter'] = None  # type: ignore[valid-type]
        self._setup()

    # ── Setup ────────────────────────────────────────────────────────────
    def _setup(self) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = Path(self.config.data.output_dir) / ts
        self.run_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(self.run_dir)
        logger.info("=" * 60)
        logger.info("Turkish Corpus Translation Benchmark Harness  v3.6")
        logger.info("  Run mode:   %s", self.run_mode)
        logger.info("  Run dir:    %s", self.run_dir)
        logger.info("  torch.compile: %s", not self.no_torch_compile)
        logger.info("=" * 60)

    # ── Public entrypoint ────────────────────────────────────────────────
    def run(self) -> dict:
        """Execute the benchmark according to the configured run mode."""
        # Apply overrides before detection
        if self.seed_override is not None:
            self.config = self.config.model_copy(update={
                "runtime": self.config.runtime.model_copy(
                    update={"seed": self.seed_override},
                ),
            })

        if self.config.runtime.seed is not None:
            torch.manual_seed(self.config.runtime.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.runtime.seed)
            # Warn about nondeterminism
            logger.info(
                "Seed set to %d. Full determinism requires --deterministic mode "
                "(not yet implemented).",
                self.config.runtime.seed,
            )

        device_info = detect_backend(self.config.backend)
        logger.info(
            "Backend: %s (%s), %d device(s)",
            device_info.backend, device_info.name, device_info.num_devices,
        )
        self.device_info = device_info
        run_preflight_checks(self.config, device_info,
                             dry_run=(self.run_mode == "dry-run"))
        env_snapshot = get_environment_snapshot()
        logger.info(
            "PyTorch %s, Python %s",
            env_snapshot["pytorch_version"], env_snapshot["python_version"],
        )

        # ── Load model (v3.0: model-agnostic backend dispatch) ──
        model_cfg = self.config.model
        extra = {
            "do_sample": model_cfg.do_sample,
            "num_beams": model_cfg.num_beams,
            "backend_type": model_cfg.backend_type,
            "diffusion": {
                "num_diffusion_steps": model_cfg.diffusion_steps,
                "noise_schedule": model_cfg.noise_schedule,
                "guidance_scale": model_cfg.guidance_scale,
                "target_length_multiplier": model_cfg.target_length_multiplier,
            },
            "plugin_name": model_cfg.plugin_name,
            "plugin_config": model_cfg.plugin_config,
            # v3.3: TensorRT engine optimization.
            "use_tensorrt": model_cfg.use_tensorrt,
            "tensorrt": {
                "precision": model_cfg.tensorrt_precision,
                "max_batch": model_cfg.tensorrt_max_batch,
                "cache_dir": model_cfg.tensorrt_cache_dir or "",
                "force_rebuild": False,
            },
            # v3.3: safe mode disables experimental/risky optimizations.
            "safe_mode": self.safe_mode,
            # v3.4: Speculative decoding.
            "use_speculative": model_cfg.use_speculative,
            "speculative_mode": model_cfg.speculative_mode,
            "speculative_num_tokens": model_cfg.speculative_num_tokens,
            "speculative_draft_model": model_cfg.speculative_draft_model,
            "speculative_num_draft_layers": model_cfg.speculative_num_draft_layers,
            # v3.6: NLLB encoder-decoder parameters.
            "nllb_source_lang": model_cfg.nllb_source_lang,
            "nllb_target_lang": model_cfg.nllb_target_lang,
            # v3.4: QAT model configuration.
            "use_qat_model": getattr(model_cfg, "use_qat_model", False),
            "qat_precision": getattr(model_cfg, "qat_precision", "auto"),
            # v3.6: Quantization level (bf16, int8, int4).
            "quantization": getattr(model_cfg, "quantization", "bf16"),
        }
        self.engine = InferenceEngine(
            model_path=model_cfg.model_path,
            tokenizer_path=model_cfg.tokenizer_path,
            device_info=device_info,
            decoding_params=DecodingParams(
                max_new_tokens=model_cfg.max_new_tokens,
                temperature=model_cfg.temperature,
                do_sample=model_cfg.do_sample,
                num_beams=model_cfg.num_beams,
            ),
            use_flash_attention=model_cfg.use_flash_attention,
            use_torch_compile=not self.no_torch_compile,
            max_input_tokens=self.config.model.max_input_tokens,
            backend_type=model_cfg.backend_type,
            extra=extra,
        )
        self.engine.load()

        # ── Help the allocator consolidate memory ────────────────────────
        # After model load + fused-kernel injection, a significant amount of
        # CPU-side memory may be held by freed-but-not-reclaimed arenas
        # (CPython's pymalloc, PyTorch's caching allocator).  An explicit
        # GC pass lets the various allocators release whatever they can.
        import gc
        gc.collect()
        if device_info.backend == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

        # ── Batch size ──
        if self.batch_size_override:
            batch_size = self.batch_size_override
            logger.info("Batch size forced: %d", batch_size)
        elif (device_info.backend == "mps" and
              os.environ.get("TR_MPS_MEMORY_SAFE", "1") != "0"):
            # MPS: skip batch tuning.  Each unique tensor shape probed by
            # the tuner compiles 5-12 GB of MPSGraph Metal shaders into
            # IOAccelerator memory that CANNOT be freed until the process
            # exits.  A single tuner probe at bs=32 followed by warmup at
            # the real batch size creates TWO distinct shader caches
            # (10-25 GB), pushing total RSS past 48 GB on a unified-memory
            # Mac, into swap.  Use a safe default; pass --batch-size to
            # override for production throughput.
            batch_size = 1
            logger.info(
                "MPS memory-safe mode: batch_size=%d (skip tuner — "
                "pass --batch-size N to override or set "
                "TR_MPS_MEMORY_SAFE=0 to re-enable tuning)",
                batch_size,
            )
        else:
            tuner = BatchSizeTuner()
            batch_size = tuner.tune(
                self.engine.model, self.engine.tokenizer,
                device_info.device, device_info.backend,
                self.config.model.max_input_tokens,
            )
            # MPS: batch tuner tests multiple batch sizes, each creating
            # distinct MPSGraph shader caches.  Trim before proceeding.
            if device_info.backend == "mps" and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

        # batch_size passthrough: the harness sets the configured batch
        # size after tuning.  The InferenceEngine._configured_batch_size
        # property delegates to the backend, which uses it for warmup
        # sizing, CUDA graph capture, and decode-loop memory planning.
        self.engine._configured_batch_size = batch_size

        # ── Dispatch by mode ──
        if self.run_mode == "benchmark-only":
            return self._run_quality_only(env_snapshot)

        if self.run_mode == "warmup-only":
            self.engine.warmup(batches=20)
            logger.info("Warm-up complete — exiting (warmup-only mode)")
            return {
                "environment": env_snapshot,
                "runtime": {"mode": "warmup-only"},
            }

        if self.run_mode == "resume":
            return self._run_resume(batch_size, env_snapshot, device_info)

        # ── Continuous batching (CUDA only, batch_size >= 8) ──
        if (
            device_info.backend == "cuda"
            and self.config.model.use_continuous_batching
            and self.config.model.use_paged_attention
        ):
            from benchmark.inference.continuous_batcher import should_use_continuous_batching
            if should_use_continuous_batching(
                device_info.backend, batch_size,
                use_paged_attention=True,
            ):
                return self._run_continuous_batching_loop(
                    batch_size, env_snapshot, device_info,
                )

        return self._run_translation_loop(batch_size, env_snapshot, device_info)

    # ── Translation loop ─────────────────────────────────────────────────
    def _run_translation_loop(
        self, batch_size: int, env_snapshot: dict, device_info: DeviceInfo,
    ) -> dict:
        """Core translation loop — shared by full, quick, dry-run, translate-only."""

        # Determine duration
        if self.duration_override is not None and self.duration_override > 0:
            target_duration = self.duration_override
        elif self.duration_override is not None:
            # duration_override is 0 — use config default with warning
            logger.warning("duration_override=0 is invalid; using config default")
            target_duration = self.config.runtime.target_duration_seconds
        elif self.run_mode == "dry-run":
            target_duration = 60
        elif self.run_mode == "quick":
            target_duration = 300
        else:
            target_duration = self.config.runtime.target_duration_seconds

        # Warm-up
        self.engine.warmup(batches=10 if self.run_mode == "dry-run" else 20)

        # Data pipeline — passes backend for pinned memory decisions (P0-04)
        # On MPS, in-memory shuffle loads 100K docs → 320 MB Python strings
        # that fragment the heap and aren't reclaimed to the OS.  Sequential
        # iteration is memory-safe.
        _shuffle = (
            self.config.data.shuffle and
            not (device_info.backend == "mps" and
                 os.environ.get("TR_MPS_MEMORY_SAFE", "1") != "0")
        )
        loader = JSONLLoader(
            self.config.data.input_paths,
            shuffle=_shuffle,
            seed=self.config.runtime.seed,
            max_shuffle_memory_gb=self.config.data.shuffle_max_memory_gb,
            shuffle_temp_dir=self.config.data.shuffle_temp_dir,
        )
        chunker = TextChunker(
            self.engine.tokenizer,
            self.config.model.max_input_tokens,
            self.config.data.chunk_overlap_tokens,
        )
        filt = ChunkFilter(
            min_tokens=self.config.data.min_chunk_tokens,
            max_garbage_ratio=self.config.data.max_garbage_ratio,
        )
        self.pipeline = AsyncPipeline(
            loader, chunker, self.engine.tokenizer, filt,
            batch_size=batch_size,
            prefetch_workers=self.config.data.prefetch_workers,
            backend=device_info.backend,  # ← v2.0: pinned memory on CUDA
        )

        # Metrics
        self.metrics = MetricsCollector(
            self.run_dir / "metrics",
            device_info,
            self.config.runtime.metrics_sample_rate_hz,
        )

        # Start Prometheus metrics server if observability is enabled.
        if self.observability_enabled or self.config.runtime.observability_enabled:
            from benchmark.observability.prometheus_metrics import PrometheusExporter
            self._prometheus = PrometheusExporter(port=9090)
            self._prometheus.start()
            self.metrics.set_prometheus_exporter(self._prometheus)

        self.checkpoint_mgr = CheckpointManager(
            self.run_dir,
            self.config.runtime.checkpoint_interval_seconds,
        )
        self.signal_handler = SignalHandler()

        # Register cleanup: pipeline + metrics get flushed on signal.
        register_cleanup("pipeline_stop", self.pipeline.stop_prefetch)
        register_cleanup("metrics_stop", self.metrics.stop)
        register_cleanup("metrics_flush", lambda: (
            self.metrics.device_sampler.flush(),
            self.metrics.system_sampler.flush(),
            self.metrics.batch_logger.flush(),
        ))

        logger.info(
            "Starting translation run: %ds, batch_size=%d, mode=%s",
            target_duration, batch_size, self.run_mode,
        )

        # ── MPS: trim Metal driver pools before data pipeline starts ───
        if device_info.backend == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
            import psutil as _ps, gc as _gc
            _gc.collect()
            _proc = _ps.Process()
            _rss = _proc.memory_info().rss / (1024**3)
            _drv = torch.mps.driver_allocated_memory() / (1024**3)
            logger.info(
                "Pre-shuffle memory: RSS %.1f GB  driver %.1f GB",
                _rss, _drv,
            )

        self.pipeline.start_prefetch()
        timer = PrecisionTimer()
        timer.start()
        self.metrics.start(timer.start_time())

        last_checkpoint = timer.elapsed()
        last_heartbeat = timer.elapsed()
        batches_completed = 0
        total_tokens = 0
        killed_by_signal = False
        oom_aborted = False

        register_cleanup(
            "checkpoint_save",
            lambda: self.checkpoint_mgr.save(
                batches_completed, total_tokens,
                elapsed_seconds=timer.elapsed(),
                final=True,
            ),
        )

        try:
            while timer.elapsed() < target_duration:
                if self.signal_handler.killed.is_set():
                    logger.error(
                        "Killed by signal %d — stopping immediately. "
                        "batches=%d tokens=%s",
                        self.signal_handler.signal_number or 0,
                        batches_completed,
                        format(total_tokens, ','),
                    )
                    self.signal_handler.cleanup()
                    killed_by_signal = True
                    break

                batch = self.pipeline.next_batch()
                if batch is None:
                    if self.pipeline.draining():
                        break
                    continue

                # P0-06: Re-check signal before expensive model.generate() call.
                # The outer while-loop check is only evaluated between batches,
                # so a Ctrl+C that arrives right after next_batch() returns would
                # not be seen until translate() finishes (up to ~30 s).  This
                # second check shortens worst-case shutdown latency to near-zero.
                if self.signal_handler.killed.is_set():
                    logger.error(
                        "Killed by signal %d — stopping immediately (pre-translate check). "
                        "batches=%d tokens=%s",
                        self.signal_handler.signal_number or 0,
                        batches_completed,
                        format(total_tokens, ','),
                    )
                    self.pipeline.release_batch(batch)
                    self.signal_handler.cleanup()
                    killed_by_signal = True
                    break

                try:
                    result = self.engine.translate(batch)
                    self.pipeline.release_batch(batch)
                    self.metrics.log_batch(result)
                    if self._prometheus is not None:
                        self._prometheus.record_batch(
                            tokens=result.output_tokens_total,
                            latency_ms=result.total_latency_ms,
                            prefill_ms=(
                                result.prefill_time_ms
                                if hasattr(result, 'prefill_time_ms') and result.prefill_time_ms is not None
                                else (result.phase_timings.get('prefill_ms', 0)
                                      if hasattr(result, 'phase_timings') and result.phase_timings is not None
                                      else 0)
                            ),
                            decode_ms=(
                                result.decode_time_ms
                                if hasattr(result, 'decode_time_ms') and result.decode_time_ms is not None
                                else (result.phase_timings.get('decode_ms', 0)
                                      if hasattr(result, 'phase_timings') and result.phase_timings is not None
                                      else 0)
                            ),
                            batch_size=result.batch_size,
                        )
                    batches_completed += 1
                    total_tokens += result.output_tokens_total
                except torch.cuda.OutOfMemoryError as exc:
                    logger.error(
                        "CUDA OOM at batch %d (batch_size=%d): %s",
                        batches_completed, batch_size, exc,
                    )
                    oom_aborted = self._handle_oom(
                        batch, loader, timer, batches_completed, total_tokens,
                        batch_size, oom_aborted,
                    )
                    if oom_aborted:
                        break
                    batch_size = self.engine._configured_batch_size
                except (RuntimeError, MemoryError) as exc:
                    # MPS reports OOM as RuntimeError with specific messages;
                    # CPU OOM surfaces as MemoryError.  Handle identically to
                    # CUDA OOM — checkpoint, halve batch size, re-warm.
                    msg = str(exc)
                    is_mps_oom = any(
                        kw in msg for kw in (
                            "MPS", "mps", "Placeholder storage",
                            "not enough memory", "out of memory",
                        )
                    )
                    if is_mps_oom or isinstance(exc, MemoryError):
                        logger.error(
                            "MPS/CPU OOM at batch %d (batch_size=%d): %s",
                            batches_completed, batch_size, msg[:200],
                        )
                        oom_aborted = self._handle_oom(
                            batch, loader, timer, batches_completed,
                            total_tokens, batch_size, oom_aborted,
                        )
                        if oom_aborted:
                            break
                        batch_size = self.engine._configured_batch_size
                    else:
                        raise

                now = timer.elapsed()
                if now - last_heartbeat >= self.config.runtime.heartbeat_interval_seconds:
                    tps = self.metrics.get_rolling_throughput()
                    logger.info(
                        "[%4.0fs] batches=%d tokens=%s tps=%.0f node=%s",
                        now, batches_completed, format(total_tokens, ','), tps,
                        device_info.name,
                    )
                    last_heartbeat = now
                    # Push pipeline-level metrics to Prometheus.
                    if self._prometheus is not None:
                        qd = self.pipeline.queue_depth()
                        self._prometheus.record_pipeline(queue_depth=qd)
                if now - last_checkpoint >= self.config.runtime.checkpoint_interval_seconds:
                    fname, doc_id = self._current_data_position(loader)
                    self.checkpoint_mgr.save(
                        batches_completed, total_tokens,
                        current_file_name=fname, current_doc_id=doc_id,
                        elapsed_seconds=now,
                    )
                    last_checkpoint = now

        finally:
            self.metrics.stop()
            if self._prometheus is not None:
                self._prometheus.stop()
            self.pipeline.stop_prefetch()
            fname, doc_id = self._current_data_position(loader)
            self.checkpoint_mgr.save(
                batches_completed, total_tokens,
                current_file_name=fname, current_doc_id=doc_id,
                elapsed_seconds=timer.elapsed(),
                final=True,
            )
            run_duration = timer.elapsed()
            if killed_by_signal:
                logger.error(
                    "Run aborted by signal %d: %d batches, %s tokens in %.1fs",
                    self.signal_handler.signal_number or 0,
                    batches_completed, format(total_tokens, ','), run_duration,
                )
            elif oom_aborted:
                logger.info(
                    "Run complete: %d batches, %s tokens in %.1fs (aborted: OOM recovery exhausted)",
                    batches_completed, format(total_tokens, ','), run_duration,
                )
            else:
                logger.info(
                    "Run complete: %d batches, %s tokens in %.1fs",
                    batches_completed, format(total_tokens, ','), run_duration,
                )

        # ── Quality benchmark (skip for translate-only, skip if killed, skip if OOM-aborted) ──
        quality_results = None
        if self.run_mode != "translate-only" and not killed_by_signal and not oom_aborted:
            ref_path = Path(self.config.data.reference_set_path)
            if ref_path.exists():
                logger.info("Running quality benchmark...")
                quality_bench = QualityBenchmark(self.config.data.reference_set_path)
                # Dry-run is a 60s smoke test — limit quality to 32 references
                # to avoid spending 2+ hours on 1960 reference sentences.
                max_q_refs = 32 if self.run_mode == "dry-run" else None
                quality_results = quality_bench.run(self.engine, max_references=max_q_refs)
                if quality_results is not None and self._prometheus is not None:
                    self._prometheus.record_quality(
                        bleu=None,  # BLEU not computed (omitted from parallel metric computation)
                        chrf=None,  # chrF++ not computed (omitted)
                        comet=quality_results.comet.get('system_score'),
                        bertscore=quality_results.bertscore.get('system_score'),
                        comet_kiwi=quality_results.comet_kiwi.get('system_score'),
                    )
            else:
                logger.info("Skipping quality benchmark — reference file not found: %s", ref_path)

        # ── Reports ──
        logger.info("Generating reports...")
        aggregator = MetricsAggregator(self.run_dir / "metrics")
        metrics_summary = aggregator.aggregate()

        extrapolator = ExtrapolationModel(
            total_tokens=self.config.extrapolation.total_clearnet_non_tr_tokens,
            gpu_cost_per_hour=self.config.extrapolation.gpu_cost_per_hour_usd,
        )
        mean_tps = metrics_summary.get("batch", {}).get("mean_tps", 0)
        std_tps = metrics_summary.get("batch", {}).get("std_tps", 0)
        n_batches = metrics_summary.get("batch", {}).get("total_batches", 1)
        # Parametric CI (SEM-based) — always computed.
        extrapolation = extrapolator.compute(mean_tps, std_tps, device_info.num_devices,
                                             n_batches=n_batches)
        # Bootstrap CI (resampling-based) — computed when per-batch TPS
        # data is available.  More robust than parametric for skewed
        # distributions and small samples.  The bootstrap fields are
        # merged into the extrapolation dict alongside the parametric CI.
        tps_samples = metrics_summary.get("batch", {}).get("tps_values")
        if tps_samples and len(tps_samples) >= 5:
            try:
                bootstrap = extrapolator.compute_bootstrap(
                    tps_samples, num_gpus=device_info.num_devices,
                )
                # Merge bootstrap fields — keep parametric CI as primary
                # for backward compatibility, bootstrap as supplementary.
                extrapolation["bootstrap_days_lower"] = bootstrap.get("bootstrap_days_lower")
                extrapolation["bootstrap_days_upper"] = bootstrap.get("bootstrap_days_upper")
                extrapolation["bootstrap_method"] = bootstrap.get("method", "bootstrap")
            except Exception as e:
                logger.debug("Bootstrap extrapolation failed: %s — using parametric CI only", e)

        config_dict = self.config.model_dump() if hasattr(self.config, "model_dump") else {}
        config_hash = hashlib.sha256(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()[:16]

        report = {
            "config": config_dict,
            "environment": env_snapshot,
            "runtime": {
                "actual_duration_seconds": round(run_duration, 1),
                "batches_completed": batches_completed,
                "total_tokens_translated": total_tokens,
                "mode": self.run_mode,
                "config_hash": config_hash,
            },
            "metrics": metrics_summary,
            "quality": quality_results.to_dict() if quality_results else {},
            "extrapolation": extrapolation,
            "filter_stats": filt.stats.to_dict(),
        }

        JSONReportWriter().write(self.run_dir, report)
        MarkdownReportWriter().write(self.run_dir, report)
        logger.info("Report written to %s/report/", self.run_dir)
        return report

    # ── Continuous batching (CUDA, PagedAttention, batch_size >= 8) ──────────
    def _run_continuous_batching_loop(
        self, batch_size: int, env_snapshot: dict, device_info: DeviceInfo,
    ) -> dict:
        """CUDA continuous batching: dynamically schedules sequences into
        the GPU batch, replacing completed sequences with waiting ones at
        every decode step.  Requires --paged-attention --continuous-batching.
        """
        from benchmark.inference.continuous_batcher import ContinuousBatcher
        from benchmark.inference.paged_attention import PagedKVCache

        # Determine duration.
        if self.duration_override is not None and self.duration_override > 0:
            target_duration = self.duration_override
        elif self.run_mode == "dry-run":
            target_duration = 60
        elif self.run_mode == "quick":
            target_duration = 300
        else:
            target_duration = self.config.runtime.target_duration_seconds

        # Warm-up.
        self.engine.warmup(batches=10 if self.run_mode == "dry-run" else 20)

        # ── Data pipeline ──
        loader = JSONLLoader(
            self.config.data.input_paths,
            shuffle=self.config.data.shuffle,
            seed=self.config.runtime.seed,
            max_shuffle_memory_gb=self.config.data.shuffle_max_memory_gb,
            shuffle_temp_dir=self.config.data.shuffle_temp_dir,
        )
        chunker = TextChunker(
            self.engine.tokenizer,
            self.config.model.max_input_tokens,
            self.config.data.chunk_overlap_tokens,
        )
        filt = ChunkFilter(
            min_tokens=self.config.data.min_chunk_tokens,
            max_garbage_ratio=self.config.data.max_garbage_ratio,
        )
        self.pipeline = AsyncPipeline(
            loader, chunker, self.engine.tokenizer, filt,
            batch_size=1,  # single-sequence batches for dynamic scheduling
            prefetch_workers=self.config.data.prefetch_workers,
            backend=device_info.backend,
        )

        # ── Metrics ──
        self.metrics = MetricsCollector(
            self.run_dir / "metrics",
            device_info,
            self.config.runtime.metrics_sample_rate_hz,
        )
        if self.observability_enabled or self.config.runtime.observability_enabled:
            from benchmark.observability.prometheus_metrics import PrometheusExporter
            self._prometheus = PrometheusExporter(port=9090)
            self._prometheus.start()
            self.metrics.set_prometheus_exporter(self._prometheus)

        self.checkpoint_mgr = CheckpointManager(
            self.run_dir,
            self.config.runtime.checkpoint_interval_seconds,
        )
        self.signal_handler = SignalHandler()

        register_cleanup("pipeline_stop", self.pipeline.stop_prefetch)
        register_cleanup("metrics_stop", self.metrics.stop)
        register_cleanup("metrics_flush", lambda: (
            self.metrics.device_sampler.flush(),
            self.metrics.system_sampler.flush(),
            self.metrics.batch_logger.flush(),
        ))

        # ── PagedAttention pool ──
        backend = self.engine.backend
        kv_cfg = backend.kv_cache_config
        paged_kv = PagedKVCache(
            num_layers=kv_cfg.get("num_layers", 24),
            num_kv_heads=kv_cfg.get("num_kv_heads", 4),
            head_dim=kv_cfg.get("head_dim", 256),
            block_size=16,
            num_blocks=1024,
            dtype=backend.precision_config.master_dtype
            if backend.precision_config else torch.bfloat16,
            device=self.engine.devices[0],
        )

        # ── Continuous batcher ──
        batcher = ContinuousBatcher(
            self.engine, paged_kv,
            max_batch_size=batch_size,
            pad_token_id=self.engine.tokenizer.pad_token_id or 0,
        )

        logger.info(
            "Continuous batching active: max_batch=%d, paged_blocks=%d",
            batch_size, paged_kv.num_blocks,
        )

        self.pipeline.start_prefetch()
        timer = PrecisionTimer()
        timer.start()
        self.metrics.start(timer.start_time())

        last_checkpoint = timer.elapsed()
        last_heartbeat = timer.elapsed()
        batches_completed = 0
        total_tokens = 0
        killed_by_signal = False

        register_cleanup(
            "checkpoint_save",
            lambda: self.checkpoint_mgr.save(
                batches_completed, total_tokens,
                elapsed_seconds=timer.elapsed(),
                final=True,
            ),
        )

        try:
            while timer.elapsed() < target_duration:
                if self.signal_handler.killed.is_set():
                    killed_by_signal = True
                    self.signal_handler.cleanup()
                    break

                # Feed the batcher with individual sequences from the pipeline.
                batch = self.pipeline.next_batch()
                if batch is None:
                    if self.pipeline.draining():
                        break
                    continue

                if self.signal_handler.killed.is_set():
                    self.pipeline.release_batch(batch)
                    self.signal_handler.cleanup()
                    killed_by_signal = True
                    break

                # Submit individual sequences.
                for i in range(batch.input_ids.shape[0]):
                    raw = (
                        batch.raw_texts[i]
                        if hasattr(batch, 'raw_texts') and i < len(batch.raw_texts)
                        else ""
                    )
                    batcher.submit(batch.input_ids[i:i + 1], raw)
                self.pipeline.release_batch(batch)

                # Run decode steps while batch is full enough.
                while batcher.running_count() > 0:
                    try:
                        completed = batcher.step()
                    except torch.cuda.OutOfMemoryError as exc:
                        logger.error("CB OOM: %s — draining", exc)
                        if self._prometheus is not None:
                            self._prometheus.record_error()
                        # Drain running sequences to free paged blocks.
                        for sid in list(batcher._running.keys()):
                            try:
                                batcher._paged_kv.free(sid)
                            except Exception:
                                pass
                        batcher._running.clear()
                        batcher._active_order.clear()
                        batcher._paged_cache = None
                        oom_aborted = True
                        break
                    except (RuntimeError, MemoryError) as exc:
                        msg = str(exc)
                        if any(kw in msg for kw in ("MPS", "mps", "out of memory", "memory")):
                            logger.error("CB MPS/CPU OOM: %s", exc)
                            oom_aborted = True
                            break
                        raise
                    for seq in completed:
                        total_tokens += len(seq.generated_ids)
                        batches_completed += 1
                        # Log metrics per completed sequence.
                        from benchmark.inference.backends.protocol import (
                            BatchGenerationOutput, GenerationOutput,
                        )
                        from datetime import datetime, timezone
                        ts = datetime.now(timezone.utc).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        )
                        gen_out = GenerationOutput(
                            input_text=seq.raw_text,
                            translated_text=self.engine.tokenizer.decode(
                                seq.generated_ids, skip_special_tokens=True,
                            ).strip(),
                            input_tokens=len(seq.input_ids),
                            output_tokens=len(seq.generated_ids),
                            total_latency_ms=0.0,  # batcher doesn't track per-seq
                            timestamp_utc=ts,
                        )
                        batch_out = BatchGenerationOutput(
                            batch_id=batches_completed,
                            generations=[gen_out],
                            batch_size=1,
                            input_tokens_total=len(seq.input_ids),
                            output_tokens_total=len(seq.generated_ids),
                            total_latency_ms=0.0,
                        )
                        self.metrics.log_batch(batch_out)
                        if self._prometheus is not None:
                            self._prometheus.record_batch(
                                tokens=len(seq.generated_ids),
                                latency_ms=0.0, batch_size=1,
                            )

                    # Check heartbeat.
                    now = timer.elapsed()
                    if now - last_heartbeat >= self.config.runtime.heartbeat_interval_seconds:
                        tps = self.metrics.get_rolling_throughput()
                        logger.info(
                            "[%4.0fs] cb: completed=%d tokens=%s tps=%.0f "
                            "running=%d waiting=%d active=%d",
                            now, batches_completed, format(total_tokens, ','), tps,
                            batcher.running_count(), batcher.waiting_count(),
                            batcher.active_batch_size(),
                        )
                        last_heartbeat = now
                        # Push pipeline and batcher metrics to Prometheus.
                        if self._prometheus is not None:
                            self._prometheus.record_pipeline(
                                queue_depth=batcher.waiting_count(),
                                starvation_pct=(
                                    100.0 if batcher.running_count() == 0 and batcher.waiting_count() > 0
                                    else 0.0
                                ),
                            )

                    if now - last_checkpoint >= self.config.runtime.checkpoint_interval_seconds:
                        fname, doc_id = self._current_data_position(loader)
                        self.checkpoint_mgr.save(
                            batches_completed, total_tokens,
                            current_file_name=fname, current_doc_id=doc_id,
                            elapsed_seconds=now,
                        )
                        last_checkpoint = now

        finally:
            self.metrics.stop()
            if self._prometheus is not None:
                self._prometheus.stop()
            self.pipeline.stop_prefetch()

            # Flush remaining sequences.
            flushed = batcher.flush_completed()
            for seq in flushed:
                total_tokens += len(seq.generated_ids)
                batches_completed += 1
                if self._prometheus is not None:
                    self._prometheus.record_batch(
                        tokens=len(seq.generated_ids),
                        latency_ms=0.0, batch_size=1,
                    )

            fname, doc_id = self._current_data_position(loader)
            self.checkpoint_mgr.save(
                batches_completed, total_tokens,
                current_file_name=fname, current_doc_id=doc_id,
                elapsed_seconds=timer.elapsed(),
                final=True,
            )
            run_duration = timer.elapsed()

        # ── Reports (same as _run_translation_loop) ──
        logger.info("Generating reports...")
        aggregator = MetricsAggregator(self.run_dir / "metrics")
        metrics_summary = aggregator.aggregate()

        extrapolator = ExtrapolationModel(
            total_tokens=self.config.extrapolation.total_clearnet_non_tr_tokens,
            gpu_cost_per_hour=self.config.extrapolation.gpu_cost_per_hour_usd,
        )
        mean_tps = metrics_summary.get("batch", {}).get("mean_tps", 0)
        std_tps = metrics_summary.get("batch", {}).get("std_tps", 0)
        n_batches = metrics_summary.get("batch", {}).get("total_batches", 1)
        extrapolation = extrapolator.compute(
            mean_tps, std_tps, device_info.num_devices, n_batches=n_batches,
        )

        config_dict = self.config.model_dump() if hasattr(self.config, "model_dump") else {}
        config_hash = hashlib.sha256(
            json.dumps(config_dict, sort_keys=True).encode()
        ).hexdigest()[:16]

        report = {
            "config": config_dict,
            "environment": env_snapshot,
            "runtime": {
                "actual_duration_seconds": round(run_duration, 1),
                "batches_completed": batches_completed,
                "total_tokens_translated": total_tokens,
                "mode": f"continuous_batching_{self.run_mode}",
                "config_hash": config_hash,
            },
            "metrics": metrics_summary,
            "extrapolation": extrapolation,
            "filter_stats": filt.stats.to_dict(),
        }

        JSONReportWriter().write(self.run_dir, report)
        MarkdownReportWriter().write(self.run_dir, report)
        logger.info(
            "Continuous batching run complete: %d batches, %s tokens in %.1fs",
            batches_completed, format(total_tokens, ','), run_duration,
        )
        return report

    # ── Benchmark-only ───────────────────────────────────────────────────
    def _run_quality_only(self, env_snapshot: dict) -> dict:
        logger.info(
            "Benchmark-only mode: running quality evaluation (no translation)"
        )
        quality_bench = QualityBenchmark(self.config.data.reference_set_path)
        quality_results = quality_bench.run(self.engine)

        config_dict = self.config.model_dump() if hasattr(self.config, "model_dump") else {}
        config_hash = hashlib.sha256(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()[:16]

        report = {
            "config": config_dict,
            "environment": env_snapshot,
            "quality": quality_results.to_dict(),
            "runtime": {
                "mode": "benchmark-only",
                "config_hash": config_hash,
            },
        }
        JSONReportWriter().write(self.run_dir, report)
        return report

    # ── Shared OOM handler — called from both translation and resume loops ───

    _OOM_BATCH_SIZE_MIN = 1
    _OOM_WARMUP_BATCHES = 3

    def _handle_oom(
        self, batch, loader, timer,
        batches_completed: int, total_tokens: int,
        batch_size: int, oom_aborted: bool,
    ) -> bool:
        """Handle OOM across all backends (CUDA, MPS, CPU).

        Flushes checkpoint with position tracking, cleans up the failed
        batch from GPU/system memory, halves the batch size, and re-warms.
        Returns True if the run should abort (batch size cannot be reduced
        further).
        """
        if self._prometheus is not None:
            self._prometheus.record_error()
        # Flush checkpoint before reducing batch size.
        fname, doc_id = self._current_data_position(loader)
        self.checkpoint_mgr.save(
            batches_completed, total_tokens,
            current_file_name=fname, current_doc_id=doc_id,
            elapsed_seconds=timer.elapsed(),
        )
        # Clean up the failed batch.
        # DO NOT release back to the pinned buffer pool — it will be
        # deallocated when update_batch_size creates a new pool below.
        # Just drop the reference and let the GC collect.
        del batch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

        # Halve the batch size, clamped to minimum.
        new_batch_size = max(batch_size // 2, self._OOM_BATCH_SIZE_MIN)
        if new_batch_size == batch_size:
            logger.critical(
                "OOM persists at minimum batch size %d — aborting.",
                self._OOM_BATCH_SIZE_MIN,
            )
            return True

        logger.warning(
            "Reducing batch size from %d to %d and re-warming.",
            batch_size, new_batch_size,
        )
        self.engine._configured_batch_size = new_batch_size
        self.pipeline.update_batch_size(new_batch_size)
        self.engine.warmup(batches=self._OOM_WARMUP_BATCHES)
        return False

    # ── Resume support ────────────────────────────────────────────────────

    def _current_data_position(self, loader=None) -> tuple[str, int]:
        """Return (current_file_name, current_doc_id) for checkpointing."""
        if loader is not None and hasattr(loader, 'current_position'):
            return loader.current_position
        # Fallback: pipeline doesn't expose this directly
        return "", 0

    def _run_resume(
        self, batch_size: int, env_snapshot: dict, device_info: DeviceInfo,
    ) -> dict:
        """Resume translation from a checkpoint directory.

        Loads the latest checkpoint from *resume_dir*, seeks the data loader
        to the saved file/offset, restores the batch and token counters,
        and continues the translation loop from where it left off.
        """
        resume_path = Path(self.resume_dir)
        if not resume_path.exists():
            raise FileNotFoundError(
                f"Resume directory not found: {self.resume_dir}"
            )

        mgr = CheckpointManager(
            resume_path, self.config.runtime.checkpoint_interval_seconds,
        )
        cp = mgr.load_latest()
        if cp is None:
            raise ValueError(
                f"No checkpoint found in: {self.resume_dir}"
            )

        batches_completed = cp["batches_completed"]
        total_tokens = cp["total_tokens_translated"]
        current_file = cp.get("current_file_name", "")
        current_doc_id = cp.get("current_doc_id", 0)
        previously_elapsed = cp.get("elapsed_seconds", 0)

        logger.info(
            "Resuming from checkpoint: %d batches, %d tokens, file=%s doc_id=%d, elapsed=%.1fs",
            batches_completed, total_tokens, current_file, current_doc_id,
            previously_elapsed,
        )

        # ── Warm-up with reduced batches (model should still be in memory) ──
        self.engine.warmup(batches=5)

        # ── Data pipeline with seek to checkpoint position ──
        loader = JSONLLoader(
            self.config.data.input_paths,
            shuffle=self.config.data.shuffle,
            seed=self.config.runtime.seed,
            max_shuffle_memory_gb=self.config.data.shuffle_max_memory_gb,
            shuffle_temp_dir=self.config.data.shuffle_temp_dir,
        )
        if current_doc_id > 0:
            loader.seek_to(current_doc_id)

        chunker = TextChunker(
            self.engine.tokenizer,
            self.config.model.max_input_tokens,
            self.config.data.chunk_overlap_tokens,
        )
        filt = ChunkFilter(
            min_tokens=self.config.data.min_chunk_tokens,
            max_garbage_ratio=self.config.data.max_garbage_ratio,
        )
        self.pipeline = AsyncPipeline(
            loader, chunker, self.engine.tokenizer, filt,
            batch_size=batch_size,
            prefetch_workers=self.config.data.prefetch_workers,
            backend=device_info.backend,
        )

        # ── Metrics + checkpointing ──
        self.metrics = MetricsCollector(
            self.run_dir / "metrics",
            device_info,
            self.config.runtime.metrics_sample_rate_hz,
        )
        self.checkpoint_mgr = CheckpointManager(
            self.run_dir,
            self.config.runtime.checkpoint_interval_seconds,
        )

        # Start Prometheus metrics server if observability is enabled.
        if self.observability_enabled or self.config.runtime.observability_enabled:
            from benchmark.observability.prometheus_metrics import PrometheusExporter
            self._prometheus = PrometheusExporter(port=9090)
            self._prometheus.start()
            self.metrics.set_prometheus_exporter(self._prometheus)

        self.signal_handler = SignalHandler()

        register_cleanup("pipeline_stop", self.pipeline.stop_prefetch)
        register_cleanup("metrics_stop", self.metrics.stop)
        register_cleanup("metrics_flush", lambda: (
            self.metrics.device_sampler.flush(),
            self.metrics.system_sampler.flush(),
            self.metrics.batch_logger.flush(),
        ))

        # Subtract already-elapsed time from the duration budget so a resumed
        # run does not silently exceed the intended total runtime.
        if self.duration_override is not None and self.duration_override > 0:
            full_duration = self.duration_override
        elif self.duration_override is not None:
            logger.warning("duration_override=0 is invalid; using config default")
            full_duration = self.config.runtime.target_duration_seconds
        else:
            full_duration = self.config.runtime.target_duration_seconds
        remaining = full_duration - previously_elapsed
        target_duration = max(remaining, 60)  # floor of 60 s so the loop still runs

        logger.info(
            "Resuming translation loop: target=%ds, batch_size=%d",
            target_duration, batch_size,
        )

        self.pipeline.start_prefetch()
        timer = PrecisionTimer()
        timer.start()
        self.metrics.start(timer.start_time())

        # Register checkpoint cleanup AFTER timer is created so the
        # lambda can safely reference timer.elapsed().
        register_cleanup(
            "checkpoint_save",
            lambda: self.checkpoint_mgr.save(
                batches_completed, total_tokens,
                current_file_name=loader.current_position[0],
                current_doc_id=loader.current_position[1],
                elapsed_seconds=timer.elapsed(),
                final=True,
            ),
        )

        last_checkpoint = timer.elapsed()
        last_heartbeat = timer.elapsed()
        killed_by_signal = False
        oom_aborted = False

        MIN_BATCH_SIZE = 1

        try:
            while timer.elapsed() < target_duration:
                if self.signal_handler.killed.is_set():
                    logger.error(
                        "Killed by signal %d — stopping. "
                        "batches=%d tokens=%s",
                        self.signal_handler.signal_number or 0,
                        batches_completed,
                        format(total_tokens, ','),
                    )
                    self.signal_handler.cleanup()
                    killed_by_signal = True
                    break

                batch = self.pipeline.next_batch()
                if batch is None:
                    if self.pipeline.draining():
                        break
                    continue

                # P0-06: Re-check signal before expensive model.generate() call.
                # The outer while-loop check is only evaluated between batches,
                # so a Ctrl+C that arrives right after next_batch() returns would
                # not be seen until translate() finishes (up to ~30 s).  This
                # second check shortens worst-case shutdown latency to near-zero.
                if self.signal_handler.killed.is_set():
                    logger.error(
                        "Killed by signal %d — stopping immediately (pre-translate check). "
                        "batches=%d tokens=%s",
                        self.signal_handler.signal_number or 0,
                        batches_completed,
                        format(total_tokens, ','),
                    )
                    self.pipeline.release_batch(batch)
                    self.signal_handler.cleanup()
                    killed_by_signal = True
                    break

                try:
                    result = self.engine.translate(batch)
                    self.pipeline.release_batch(batch)
                    self.metrics.log_batch(result)
                    if self._prometheus is not None:
                        self._prometheus.record_batch(
                            tokens=result.output_tokens_total,
                            latency_ms=result.total_latency_ms,
                            prefill_ms=(
                                result.prefill_time_ms
                                if hasattr(result, 'prefill_time_ms') and result.prefill_time_ms is not None
                                else (result.phase_timings.get('prefill_ms', 0)
                                      if hasattr(result, 'phase_timings') and result.phase_timings is not None
                                      else 0)
                            ),
                            decode_ms=(
                                result.decode_time_ms
                                if hasattr(result, 'decode_time_ms') and result.decode_time_ms is not None
                                else (result.phase_timings.get('decode_ms', 0)
                                      if hasattr(result, 'phase_timings') and result.phase_timings is not None
                                      else 0)
                            ),
                            batch_size=result.batch_size,
                        )
                    batches_completed += 1
                    total_tokens += result.output_tokens_total
                except torch.cuda.OutOfMemoryError as exc:
                    logger.error(
                        "CUDA OOM at batch %d (batch_size=%d): %s",
                        batches_completed, batch_size, exc,
                    )
                    oom_aborted = self._handle_oom(
                        batch, loader, timer, batches_completed, total_tokens,
                        batch_size, oom_aborted,
                    )
                    if oom_aborted:
                        break
                    batch_size = self.engine._configured_batch_size
                except (RuntimeError, MemoryError) as exc:
                    # MPS reports OOM as RuntimeError with specific messages;
                    # CPU OOM surfaces as MemoryError.  Handle identically to
                    # CUDA OOM — checkpoint, halve batch size, re-warm.
                    msg = str(exc)
                    is_mps_oom = any(
                        kw in msg for kw in (
                            "MPS", "mps", "Placeholder storage",
                            "not enough memory", "out of memory",
                        )
                    )
                    if is_mps_oom or isinstance(exc, MemoryError):
                        logger.error(
                            "MPS/CPU OOM at batch %d (batch_size=%d): %s",
                            batches_completed, batch_size, msg[:200],
                        )
                        oom_aborted = self._handle_oom(
                            batch, loader, timer, batches_completed,
                            total_tokens, batch_size, oom_aborted,
                        )
                        if oom_aborted:
                            break
                        batch_size = self.engine._configured_batch_size
                    else:
                        raise

                now = timer.elapsed()
                if now - last_heartbeat >= self.config.runtime.heartbeat_interval_seconds:
                    tps = self.metrics.get_rolling_throughput()
                    logger.info(
                        "[%4.0fs] batches=%d tokens=%s tps=%.0f (resumed) node=%s",
                        now, batches_completed, format(total_tokens, ','), tps,
                        device_info.name,
                    )
                    last_heartbeat = now
                    # Push pipeline-level metrics to Prometheus.
                    if self._prometheus is not None:
                        qd = self.pipeline.queue_depth()
                        self._prometheus.record_pipeline(queue_depth=qd)
                if now - last_checkpoint >= self.config.runtime.checkpoint_interval_seconds:
                    fname, doc_id = self._current_data_position(loader)
                    self.checkpoint_mgr.save(
                        batches_completed, total_tokens,
                        current_file_name=fname, current_doc_id=doc_id,
                        elapsed_seconds=now,
                    )
                    last_checkpoint = now

        finally:
            self.metrics.stop()
            if self._prometheus is not None:
                self._prometheus.stop()
            self.pipeline.stop_prefetch()
            fname, doc_id = self._current_data_position(loader)
            self.checkpoint_mgr.save(
                batches_completed, total_tokens,
                current_file_name=fname, current_doc_id=doc_id,
                elapsed_seconds=timer.elapsed(),
                final=True,
            )
            run_duration = timer.elapsed()
            if killed_by_signal:
                logger.error(
                    "Resume run aborted by signal %d: %d batches (%d new), %s tokens in %.1fs",
                    self.signal_handler.signal_number or 0,
                    batches_completed, batches_completed - cp["batches_completed"],
                    format(total_tokens, ','), run_duration,
                )
            elif oom_aborted:
                logger.info(
                    "Resume run complete: %d batches (%d new), %s tokens in %.1fs (aborted: OOM recovery exhausted)",
                    batches_completed, batches_completed - cp["batches_completed"],
                    format(total_tokens, ','), run_duration,
                )
            else:
                logger.info(
                    "Resume run complete: %d batches (%d new), %s tokens in %.1fs",
                    batches_completed, batches_completed - cp["batches_completed"],
                    format(total_tokens, ','), run_duration,
                )

        # ── Quality benchmark (skip if killed by signal or OOM-aborted) ──
        quality_results = None
        if not killed_by_signal and not oom_aborted:
            logger.info("Running quality benchmark...")
            quality_bench = QualityBenchmark(self.config.data.reference_set_path)
            # Dry-run is a 60s smoke test — limit quality to 32 references
            # to avoid spending 2+ hours on 1960 reference sentences.
            max_q_refs = 32 if self.run_mode == "dry-run" else None
            quality_results = quality_bench.run(self.engine, max_references=max_q_refs)
            if quality_results is not None and self._prometheus is not None:
                self._prometheus.record_quality(
                    bleu=None,  # BLEU not computed (omitted from parallel metric computation)
                    chrf=None,  # chrF++ not computed (omitted)
                    comet=quality_results.comet.get('system_score'),
                    bertscore=quality_results.bertscore.get('system_score'),
                    comet_kiwi=quality_results.comet_kiwi.get('system_score'),
                )

        # ── Reports ──
        logger.info("Generating reports...")
        aggregator = MetricsAggregator(self.run_dir / "metrics")
        metrics_summary = aggregator.aggregate()

        extrapolator = ExtrapolationModel(
            total_tokens=self.config.extrapolation.total_clearnet_non_tr_tokens,
            gpu_cost_per_hour=self.config.extrapolation.gpu_cost_per_hour_usd,
        )
        mean_tps = metrics_summary.get("batch", {}).get("mean_tps", 0)
        std_tps = metrics_summary.get("batch", {}).get("std_tps", 0)
        n_batches = metrics_summary.get("batch", {}).get("total_batches", 1)
        # Parametric CI (SEM-based) — always computed.
        extrapolation = extrapolator.compute(mean_tps, std_tps, device_info.num_devices,
                                             n_batches=n_batches)
        # Bootstrap CI (resampling-based) — computed when per-batch TPS
        # data is available.  More robust than parametric for skewed
        # distributions and small samples.  The bootstrap fields are
        # merged into the extrapolation dict alongside the parametric CI.
        tps_samples = metrics_summary.get("batch", {}).get("tps_values")
        if tps_samples and len(tps_samples) >= 5:
            try:
                bootstrap = extrapolator.compute_bootstrap(
                    tps_samples, num_gpus=device_info.num_devices,
                )
                # Merge bootstrap fields — keep parametric CI as primary
                # for backward compatibility, bootstrap as supplementary.
                extrapolation["bootstrap_days_lower"] = bootstrap.get("bootstrap_days_lower")
                extrapolation["bootstrap_days_upper"] = bootstrap.get("bootstrap_days_upper")
                extrapolation["bootstrap_method"] = bootstrap.get("method", "bootstrap")
            except Exception as e:
                logger.debug("Bootstrap extrapolation failed: %s — using parametric CI only", e)

        config_dict = self.config.model_dump() if hasattr(self.config, "model_dump") else {}
        config_hash = hashlib.sha256(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()[:16]

        report = {
            "config": config_dict,
            "environment": env_snapshot,
            "runtime": {
                "actual_duration_seconds": round(run_duration, 1),
                "batches_completed": batches_completed,
                "total_tokens_translated": total_tokens,
                "mode": "resume",
                "resumed_from": str(resume_path),
                "config_hash": config_hash,
            },
            "metrics": metrics_summary,
            "quality": quality_results.to_dict() if quality_results else {},
            "extrapolation": extrapolation,
            "filter_stats": filt.stats.to_dict(),
        }

        JSONReportWriter().write(self.run_dir, report)
        MarkdownReportWriter().write(self.run_dir, report)
        logger.info("Report written to %s/report/", self.run_dir)
        return report
