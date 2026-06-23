"""Example custom model plugin — demonstrates the plugin system (v3.0).

This file shows how to create a custom inference backend and register it
as a plugin.  It implements a simple "echo" backend for testing purposes.

Place custom plugins in any of:
  1. ~/.tr_benchmark/plugins/*.py    (auto-discovered)
  2. ./plugins/*.py                    (project-local)
  3. $TR_BENCHMARK_PLUGIN_PATH/*.py   (environment variable)
  4. Or register at runtime via ``register_plugin()``.

To use this example:
  cp examples/example_echo_plugin.py ~/.tr_benchmark/plugins/
  python -m benchmark --config config.yaml  # auto-discovered
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

try:
    import torch
except ImportError as e:
    raise ImportError(
        "EchoPlugin requires PyTorch. Install via: pip install torch"
    ) from e

try:
    from benchmark.inference.backends.protocol import (
        BackendConfig,
        BatchGenerationOutput,
        GenerationOutput,
        InferenceBackend,
        ModelCapability,
        ModelType,
    )
    from benchmark.inference.backends.custom_plugin import CustomModelPlugin, register_plugin
except ImportError as e:
    raise ImportError(
        "EchoPlugin requires the benchmark package. Install via: pip install -e ."
    ) from e

logger = logging.getLogger(__name__)


# ── Example: EchoBackend (for testing) ─────────────────────────────────────


class EchoBackend(InferenceBackend):
    """Echo backend — returns input as output (for pipeline testing).

    Useful as a no-model test bed: validates the data pipeline, metrics,
    checkpointing, and reporting without requiring a real model.
    """

    model_type = ModelType.CUSTOM
    capabilities = ModelCapability.TRANSLATE | ModelCapability.ENSEMBLE_READY
    display_name = "Echo (Test Backend)"

    def __init__(self, config: BackendConfig):
        super().__init__(config)
        self._dummy_tokenizer = None

    def load(self) -> None:
        """Load a dummy tokenizer (anything with encode/decode)."""
        from transformers import AutoTokenizer

        # Try to load any small tokenizer for encode/decode.
        # Use local_files_only=True to avoid network hangs in offline environments.
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "HuggingFaceTB/SmolLM2-135M-Instruct",
                local_files_only=True,
            )
        except (OSError, EnvironmentError):
            # Fallback: try loading from HuggingFace Hub with a short timeout.
            try:
                import signal
                def _timeout_handler(signum, frame):
                    raise TimeoutError("Tokenizer download timed out")
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(30)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "HuggingFaceTB/SmolLM2-135M-Instruct",
                )
                signal.alarm(0)
            except Exception:
                # Ultra-fallback: use a simple word-split tokenizer.
                self.tokenizer = _SimpleTokenizer()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.devices = [torch.device("cpu")]
        self._loaded = True
        logger.info("EchoBackend loaded (no real model — test only)")

    def warmup(self, batches: int = 5) -> None:
        logger.info("EchoBackend warmup (no-op)")

    def translate_batch(self, batch: Any) -> BatchGenerationOutput:
        n_items = len(batch.raw_texts) if hasattr(batch, 'raw_texts') else 1
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"

        generations = []
        for i in range(n_items):
            src = batch.raw_texts[i] if hasattr(batch, 'raw_texts') and i < len(batch.raw_texts) else ""
            # Echo: "translate" by prepending "[TR]" marker.
            echo_text = f"[TR] {src}" if src else ""

            generations.append(GenerationOutput(
                input_text=src,
                translated_text=echo_text,
                input_tokens=len(src.split()) if src else 0,
                output_tokens=len(echo_text.split()) if echo_text else 0,
                total_latency_ms=1.0,
                timestamp_utc=ts,
            ))

        return BatchGenerationOutput(
            batch_id=batch.batch_id if hasattr(batch, 'batch_id') else 0,
            generations=generations,
            batch_size=n_items,
            input_tokens_total=sum(g.input_tokens for g in generations),
            output_tokens_total=sum(g.output_tokens for g in generations),
            total_latency_ms=1.0,
        )

    def is_loaded(self) -> bool:
        return self._loaded


class _SimpleTokenizer:
    """Minimal tokenizer for testing without HF dependency.

    Uses a deterministic hashing scheme (zlib.adler32) to guarantee
    the same word maps to the same token ID across all runs, unlike
    Python's built-in hash() which is salted per process via PYTHONHASHSEED.
    """

    vocab_size = 1000
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "[PAD]"
    eos_token = "[EOS]"
    padding_side = "left"

    @staticmethod
    def _word_to_id(word: str) -> int:
        import zlib
        return zlib.adler32(word.encode("utf-8")) % 1000

    def encode(self, text, **kwargs):
        return [self._word_to_id(w) for w in text.split()]

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(i) for i in ids if i not in (0, 1))


# ── Plugin registration ────────────────────────────────────────────────────


class EchoPlugin(CustomModelPlugin):
    """Plugin that provides the EchoBackend for testing."""

    name = "echo_test_backend"
    version = "1.0.0"
    description = "Echo backend for pipeline testing (no real model)"

    def create_backend(self, config: BackendConfig) -> InferenceBackend:
        return EchoBackend(config)

    def detect(self, model_path: str) -> bool:
        return "echo" in model_path.lower()


# ── Auto-register when imported ────────────────────────────────────────────

register_plugin(EchoPlugin())
