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
import gc
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, TYPE_CHECKING

import torch

try:
    import psutil as _psutil
    HAS_PSUTIL = True
except ImportError:
    _psutil = None  # type: ignore[assignment]
    HAS_PSUTIL = False

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

    # Prometheus metrics exporter port.  Configurable via the
    # ``PROMETHEUS_PORT`` environment variable or the
    # ``benchmark.runtime.prometheus_port`` config field.
    # Falls back to 9090 when neither is set.
    @staticmethod
    def _resolve_prometheus_port(config) -> int:
        port = os.environ.get("PROMETHEUS_PORT")
        if port is not None:
            return int(port)
        return getattr(config.runtime, "prometheus_port", 9090) or 9090

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
        self._cached_config_hash: str = ""
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
            # v3.4: QAT model configuration. QAT is auto-detected via
            # _is_qat_model (keywords in model_path); there is no
            # `use_qat_model` field on ModelConfig.
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
        gc.collect()
        if device_info.backend == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

        # ── Capability registry (verified feature states) ──
        if (hasattr(self.engine.backend, '_capability_registry')
                and self.engine.backend._capability_registry is not None):
            reg = self.engine.backend._capability_registry
            active, total = reg.active_vs_total()
            logger.info("Capability summary: %d active / %d total features", active, total)
            logger.info(reg.report_text())

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
    def _resolve_duration(self) -> int:
        """Determine target duration in seconds from run mode and overrides."""
        if self.duration_override is not None and self.duration_override > 0:
            return self.duration_override
        if self.duration_override is not None:
            logger.warning("duration_override=0 is invalid; using config default")
        if self.run_mode == "dry-run":
            return 60
        if self.run_mode == "quick":
            return 300
        return self.config.runtime.target_duration_seconds

    def _run_translation_loop(
        self, batch_size: int, env_snapshot: dict, device_info: DeviceInfo,
    ) -> dict:
        """Translation loop — full, quick, dry-run, translate-only."""
        target_duration = self._resolve_duration()
        self.engine.warmup(batches=10 if self.run_mode == "dry-run" else 20)
        return self._run_translation_core(
            batch_size=batch_size,
            target_duration=target_duration,
            env_snapshot=env_snapshot,
            device_info=device_info,
            batches_completed=0,
            total_tokens=0,
            resume_path=None,
            resume_base_batches=0,
        )

    # ── Continuous batching (CUDA, PagedAttention, batch_size >= 8) ──────────

    def _run_translation_core(
        self,
        batch_size: int,
        target_duration: int,
        env_snapshot: dict,
        device_info: DeviceInfo,
        batches_completed: int,
        total_tokens: int,
        *,
        resume_path: Path | None = None,
        resume_base_batches: int = 0,
        loader_seek_doc_id: int = 0,
        extra_runtime_fields: dict | None = None,
    ) -> dict:
        """Shared translation core — used by both new runs and resume.

        All the heavy translation-loop logic lives here.  ``_run_translation_loop``
        and ``_run_resume`` are thin wrappers that compute the initial state
        and then delegate to this method.
        """
        resume_tag = " (resumed)" if resume_path else ""

        # Data pipeline — passes backend for pinned memory decisions (P0-04).
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
        if loader_seek_doc_id > 0:
            loader.seek_to(loader_seek_doc_id)

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
            pretokenized_loader=self._resolve_pretokenized_loader(),
        )

        self._init_translation_infra(device_info)

        logger.info(
            "Starting translation run: %ds, batch_size=%d, mode=%s%s",
            target_duration, batch_size, self.run_mode, resume_tag,
        )

        # ── MPS: trim Metal driver pools before data pipeline starts ───
        if device_info.backend == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
            gc.collect()
            if HAS_PSUTIL:
                _proc = _psutil.Process()
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
                        batch_size,
                    )
                    if oom_aborted:
                        break
                    batch_size = self.engine._configured_batch_size
                except MemoryError as exc:
                    logger.error(
                        "CPU OOM at batch %d (batch_size=%d): %s",
                        batches_completed, batch_size, exc,
                    )
                    oom_aborted = self._handle_oom(
                        batch, loader, timer, batches_completed,
                        total_tokens, batch_size,
                    )
                    if oom_aborted:
                        break
                    batch_size = self.engine._configured_batch_size
                except RuntimeError as exc:
                    msg = str(exc).casefold()
                    is_mps_oom = any(
                        kw.casefold() in msg for kw in (
                            "MPS", "Placeholder storage",
                            "not enough memory", "out of memory",
                        )
                    )
                    if is_mps_oom:
                        logger.error(
                            "MPS OOM at batch %d (batch_size=%d): %s",
                            batches_completed, batch_size, str(exc)[:200],
                        )
                        oom_aborted = self._handle_oom(
                            batch, loader, timer, batches_completed,
                            total_tokens, batch_size,
                        )
                        if oom_aborted:
                            break
                        batch_size = self.engine._configured_batch_size
                    else:
                        raise

                now = timer.elapsed()
                if now - last_heartbeat >= self.config.runtime.heartbeat_interval_seconds:
                    # First few batches: use cumulative TPS (rolling window not yet populated).
                    if batches_completed < 3 and now > 0:
                        tps = total_tokens / now
                    else:
                        tps = self.metrics.get_rolling_throughput()
                    logger.info(
                        "[%4.0fs] batches=%d tokens=%s tps=%.0f%s node=%s",
                        now, batches_completed, format(total_tokens, ','), tps,
                        resume_tag, device_info.name,
                    )
                    last_heartbeat = now
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
            new_batches = batches_completed - resume_base_batches
            if killed_by_signal:
                logger.error(
                    "Run aborted by signal %d: %d batches (%d new), %s tokens in %.1fs",
                    self.signal_handler.signal_number or 0,
                    batches_completed, new_batches,
                    format(total_tokens, ','), run_duration,
                )
            elif oom_aborted:
                logger.info(
                    "Run complete: %d batches (%d new), %s tokens in %.1fs (OOM recovery exhausted)",
                    batches_completed, new_batches,
                    format(total_tokens, ','), run_duration,
                )
            else:
                logger.info(
                    "Run complete: %d batches (%d new), %s tokens in %.1fs",
                    batches_completed, new_batches,
                    format(total_tokens, ','), run_duration,
                )

        # ── Quality benchmark (skip for translate-only, skip if killed, skip if OOM-aborted) ──
        quality_results = None
        if self.run_mode != "translate-only" and not killed_by_signal and not oom_aborted:
            ref_path = Path(self.config.data.reference_set_path)
            if ref_path.exists():
                logger.info("Running quality benchmark...")
                quality_bench = QualityBenchmark(self.config.data.reference_set_path)
                max_q_refs = 32 if self.run_mode == "dry-run" else None
                quality_results = quality_bench.run(self.engine, max_references=max_q_refs)
                if quality_results is not None and self._prometheus is not None:
                    self._prometheus.record_quality(
                        bleu=quality_results.bleu.get('score'),
                        chrf=quality_results.chrf.get('score'),
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
        median_tps = metrics_summary.get("batch", {}).get("median_tps")
        extrapolation = extrapolator.compute(
            mean_tps, std_tps, device_info.num_devices,
            n_batches=n_batches, median_tps=median_tps,
        )
        tps_samples = metrics_summary.get("batch", {}).get("tps_values")
        if tps_samples and len(tps_samples) >= 5:
            try:
                bootstrap = extrapolator.compute_bootstrap(
                    tps_samples, num_gpus=device_info.num_devices,
                )
                extrapolation["bootstrap_days_lower"] = bootstrap.get("bootstrap_days_lower")
                extrapolation["bootstrap_days_upper"] = bootstrap.get("bootstrap_days_upper")
                extrapolation["bootstrap_method"] = bootstrap.get("method", "bootstrap")
            except Exception as e:
                logger.debug("Bootstrap extrapolation failed: %s — using parametric CI only", e)

        config_hash = self._compute_config_hash()
        runtime_fields = {
            "actual_duration_seconds": round(run_duration, 1),
            "batches_completed": batches_completed,
            "total_tokens_translated": total_tokens,
            "mode": self.run_mode,
            "config_hash": config_hash,
        }
        if extra_runtime_fields:
            runtime_fields.update(extra_runtime_fields)

        report = {
            "config": self.config.model_dump() if hasattr(self.config, "model_dump") else {},
            "environment": env_snapshot,
            "runtime": runtime_fields,
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

        target_duration = self._resolve_duration()

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

        self._init_translation_infra(device_info)

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
                        batcher.drain_running()
                        oom_aborted = True
                        break
                    except MemoryError as exc:
                        # CPU OOM in continuous batcher — drain and abort.
                        logger.error("CB CPU OOM: %s", exc)
                        batcher.drain_running()
                        oom_aborted = True
                        break
                    except RuntimeError as exc:
                        # RuntimeError may be MPS OOM.  Use casefold() for
                        # locale-independent matching.
                        msg = str(exc).casefold()
                        if any(kw.casefold() in msg for kw in
                               ("MPS", "out of memory", "memory")):
                            logger.error("CB MPS OOM: %s", exc)
                            batcher.drain_running()
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
        median_tps = metrics_summary.get("batch", {}).get("median_tps")
        extrapolation = extrapolator.compute(
            mean_tps, std_tps, device_info.num_devices,
            n_batches=n_batches, median_tps=median_tps,
        )
        tps_samples = metrics_summary.get("batch", {}).get("tps_values")
        if tps_samples and len(tps_samples) >= 5:
            try:
                bootstrap = extrapolator.compute_bootstrap(
                    tps_samples, num_gpus=device_info.num_devices,
                )
                extrapolation["bootstrap_days_lower"] = bootstrap.get("bootstrap_days_lower")
                extrapolation["bootstrap_days_upper"] = bootstrap.get("bootstrap_days_upper")
                extrapolation["bootstrap_method"] = bootstrap.get("method", "bootstrap")
            except Exception as e:
                logger.debug("Bootstrap extrapolation failed: %s — using parametric CI only", e)

        config_hash = self._compute_config_hash()

        report = {
            "config": self.config.model_dump() if hasattr(self.config, "model_dump") else {},
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

        config_hash = self._compute_config_hash()

        report = {
            "config": self.config.model_dump() if hasattr(self.config, "model_dump") else {},
            "environment": env_snapshot,
            "quality": quality_results.to_dict(),
            "runtime": {
                "mode": "benchmark-only",
                "config_hash": config_hash,
            },
        }
        JSONReportWriter().write(self.run_dir, report)
        return report

    # ── Helpers ───────────────────────────────────────────────────────────

    def _compute_config_hash(self) -> str:
        """Return a deterministic 16-char hex hash of the current config.

        Computed once and cached — subsequent calls return the cached value.
        """
        if self._cached_config_hash:
            return self._cached_config_hash
        config_dict = self.config.model_dump() if hasattr(self.config, "model_dump") else {}
        self._cached_config_hash = hashlib.sha256(
            json.dumps(config_dict, sort_keys=True).encode()
        ).hexdigest()[:16]
        return self._cached_config_hash

    def _start_prometheus_if_enabled(self) -> None:
        """Start the Prometheus metrics exporter if observability is enabled.

        Creates exactly one :class:`PrometheusExporter` instance, starts its
        HTTP server, and wires it into the active metrics collector.  Safe to
        call multiple times — subsequent calls are no-ops.
        """
        if self._prometheus is not None:
            return
        if not (self.observability_enabled or self.config.runtime.observability_enabled):
            return
        from benchmark.observability.prometheus_metrics import PrometheusExporter
        self._prometheus = PrometheusExporter(
            port=self._resolve_prometheus_port(self.config),
        )
        self._prometheus.start()
        if self.metrics is not None:
            self.metrics.set_prometheus_exporter(self._prometheus)

    # ── Pre-tokenized cache resolution ─────────────────────────────────────

    def _resolve_pretokenized_loader(self):
        """Return a :class:`PreTokenizedLoader` if a valid cache exists.

        Checks the cache directory for a pre-tokenized Parquet file matching
        the current model + tokenizer + data configuration.  Returns ``None``
        on cache miss (the pipeline falls back to normal chunk→tokenize path).
        Set ``TR_NO_PRETOKENIZED_CACHE=1`` to force-disable.
        """
        if os.environ.get("TR_NO_PRETOKENIZED_CACHE") == "1":
            return None
        try:
            from benchmark.data.pretokenizer import has_cache, ensure_pretokenized
        except ImportError:
            return None  # pyarrow not installed

        model_cfg = self.config.model
        cache_dir = None
        if os.environ.get("TR_PRETOKENIZED_CACHE_DIR"):
            cache_dir = Path(os.environ["TR_PRETOKENIZED_CACHE_DIR"])
        if has_cache(
            model_cfg.model_path, self.engine.tokenizer,
            max_input_tokens=model_cfg.max_input_tokens,
            overlap_tokens=self.config.data.chunk_overlap_tokens,
            min_chunk_tokens=self.config.data.min_chunk_tokens,
            max_garbage_ratio=self.config.data.max_garbage_ratio,
            input_paths=self.config.data.input_paths,
            cache_dir=cache_dir,
        ):
            logger.info("Using pre-tokenized cache for %s", model_cfg.model_path)
            return ensure_pretokenized(
                model_cfg.model_path, self.engine.tokenizer,
                max_input_tokens=model_cfg.max_input_tokens,
                overlap_tokens=self.config.data.chunk_overlap_tokens,
                min_chunk_tokens=self.config.data.min_chunk_tokens,
                max_garbage_ratio=self.config.data.max_garbage_ratio,
                input_paths=self.config.data.input_paths,
                cache_dir=cache_dir,
            )
        return None

    def _init_translation_infra(self, device_info: DeviceInfo) -> None:
        """Shared setup for metrics, checkpoint, signal handler, and cleanup.

        Called once per translation run (both standard and continuous-batching
        paths).  Registers cleanup callbacks in FIFO order and starts the
        Prometheus exporter when observability is enabled.
        """
        self.metrics = MetricsCollector(
            self.run_dir / "metrics",
            device_info,
            self.config.runtime.metrics_sample_rate_hz,
        )
        self._start_prometheus_if_enabled()
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

    # ── Shared OOM handler — called from both translation and resume loops ───

    _OOM_BATCH_SIZE_MIN = 1
    _OOM_WARMUP_BATCHES = 3

    def _handle_oom(
        self, batch, loader, timer,
        batches_completed: int, total_tokens: int,
        batch_size: int,
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

        Loads the latest checkpoint, restores counters, seeks the data
        loader, subtracts elapsed time from the duration budget, and
        delegates to the shared ``_run_translation_core``.
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
        current_doc_id = cp.get("current_doc_id", 0)
        previously_elapsed = cp.get("elapsed_seconds", 0)

        logger.info(
            "Resuming from checkpoint: %d batches, %d tokens, "
            "file=%s doc_id=%d, elapsed=%.1fs",
            batches_completed, total_tokens,
            cp.get("current_file_name", ""), current_doc_id,
            previously_elapsed,
        )

        # Warm-up with reduced batches (model should still be in memory).
        self.engine.warmup(batches=5)

        # Subtract already-elapsed time from the duration budget.
        full_duration = self._resolve_duration()
        remaining = full_duration - previously_elapsed
        target_duration = max(remaining, 60)  # floor of 60 s

        return self._run_translation_core(
            batch_size=batch_size,
            target_duration=target_duration,
            env_snapshot=env_snapshot,
            device_info=device_info,
            batches_completed=batches_completed,
            total_tokens=total_tokens,
            resume_path=resume_path,
            resume_base_batches=batches_completed,
            loader_seek_doc_id=current_doc_id,
            extra_runtime_fields={
                "mode": "resume",
                "resumed_from": str(resume_path),
            },
        )
