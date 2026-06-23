"""Per-batch structured logger."""

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from benchmark.utils.json_utils import sanitized_dumps
from benchmark.config.constants import MAX_METRICS_BUFFER_SIZE, BATCH_FLUSH_INTERVAL

logger = logging.getLogger(__name__)


class BatchLogger:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.batch_count = 0
        self._log_file: Optional[Path] = None
        self._buffer: list[str] = []
        self._flush_interval = BATCH_FLUSH_INTERVAL
        self._buffer_lock = threading.Lock()

    def start(self) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._log_file = self.output_dir / f"batch_metrics_{ts}.jsonl"
        logger.info(f"Batch logger -> {self._log_file}")

    def log(self, batch_result) -> None:
        if not self._log_file:
            self.start()
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond//1000:03d}Z"

        # Handle both v2.0 BatchResult (has prefill/decode attrs) and
        # v3.0 BatchGenerationOutput (has phase_timings dict).
        prefill_ms = getattr(batch_result, 'prefill_time_ms', None)
        if prefill_ms is None and hasattr(batch_result, 'phase_timings') and batch_result.phase_timings is not None:
            prefill_ms = batch_result.phase_timings.get('prefill_ms', 0)
        prefill_ms = prefill_ms or 0
        decode_ms = (
            batch_result.decode_time_ms
            if hasattr(batch_result, 'decode_time_ms')
            else (batch_result.phase_timings.get('decode_ms', 0)
                  if hasattr(batch_result, 'phase_timings') and batch_result.phase_timings is not None
                  else 0)
        )
        total_ms = (
            batch_result.total_latency_ms
            if hasattr(batch_result, 'total_latency_ms')
            else 0
        )
        tps_val = getattr(batch_result, 'tokens_per_second', None)
        if tps_val is not None:
            if callable(tps_val):
                tps = tps_val()
            else:
                tps = tps_val
        else:
            tps = (batch_result.output_tokens_total / total_ms * 1000 if total_ms > 0 else 0)

        entry = {"batch_id": batch_result.batch_id, "timestamp": ts,
                 "batch_size": batch_result.batch_size,
                 "input_tokens_total": batch_result.input_tokens_total,
                 "output_tokens_total": batch_result.output_tokens_total,
                 "prefill_time_ms": round(prefill_ms, 1),
                 "decode_time_ms": round(decode_ms, 1),
                 "total_latency_ms": round(total_ms, 1),
                 "tokens_per_second": round(tps, 1)}
        with self._buffer_lock:
            self._buffer.append(sanitized_dumps(entry, ensure_ascii=False))
            self.batch_count += 1
        if len(self._buffer) >= self._flush_interval:
            self.flush()

    def flush(self) -> None:
        if not self._buffer or not self._log_file:
            return
        try:
            with self._buffer_lock:
                if not self._buffer:
                    return
                with open(self._log_file, "a") as f:
                    for line in self._buffer:
                        f.write(line + "\n")
                self._buffer.clear()
        except OSError as e:
            logger.error(f"Flush failed — keeping buffer for next retry ({len(self._buffer)} entries): {e}")
            # Prevent unbounded buffer growth on persistent flush failures.
            excess = len(self._buffer) - MAX_METRICS_BUFFER_SIZE
            if excess > 0:
                with self._buffer_lock:
                    dropped = self._buffer[:excess]
                    self._buffer = self._buffer[excess:]
                logger.warning(
                    f"Buffer exceeded MAX_METRICS_BUFFER_SIZE ({MAX_METRICS_BUFFER_SIZE}); "
                    f"dropped oldest {len(dropped)} batch log entries to prevent OOM"
                )
