"""Tests for NLLB encoder-decoder backend — loading, translation, pipeline."""

import gc
import sys
import types
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ═════════════════════════════════════════════════════════════════════════════
# Unit tests — backend detection and config
# ═════════════════════════════════════════════════════════════════════════════


class TestNLLBDetection:
    """NLLB auto-detection in ModelRegistry."""

    def test_nllb_detected_by_name(self):
        """Model path with 'nllb' is auto-detected as ENCODER_DECODER."""
        from benchmark.inference.backends.registry import ModelRegistry
        from benchmark.inference.backends.protocol import ModelType, BackendConfig
        from benchmark.hardware.backend import DeviceInfo

        registry = ModelRegistry()
        di = DeviceInfo(backend="cpu", device=torch.device("cpu"), num_devices=1)
        config = BackendConfig(
            model_path="facebook/nllb-200-distilled-600M",
            device_info=di,
        )
        # create_backend picks NLLBBackend via auto-detect
        backend = registry.create_backend(config)
        assert backend.model_type == ModelType.ENCODER_DECODER
        assert "NLLB" in backend.display_name

    def test_nllb_detected_by_explicit_type(self):
        """Explicit backend_type='encoder_decoder' works."""
        from benchmark.inference.backends.registry import ModelRegistry
        from benchmark.inference.backends.protocol import BackendConfig
        from benchmark.hardware.backend import DeviceInfo

        registry = ModelRegistry()
        di = DeviceInfo(backend="cpu", device=torch.device("cpu"), num_devices=1)
        config = BackendConfig(
            model_path="facebook/nllb-200-distilled-600M",
            device_info=di,
            extra={"backend_type": "encoder_decoder"},
        )
        backend = registry.create_backend(config)
        assert backend.model_type.value == "encoder_decoder"


class TestNLLBConfig:
    """NLLB-specific config fields."""

    def test_default_language_codes(self):
        """Source and target language codes default to eng_Latn/tur_Latn."""
        from benchmark.config.schema import ModelConfig
        cfg = ModelConfig()
        assert cfg.nllb_source_lang == "eng_Latn"
        assert cfg.nllb_target_lang == "tur_Latn"

    def test_custom_language_codes(self):
        """Custom language codes are accepted."""
        from benchmark.config.schema import ModelConfig
        cfg = ModelConfig(
            nllb_source_lang="spa_Latn",
            nllb_target_lang="eng_Latn",
        )
        assert cfg.nllb_source_lang == "spa_Latn"
        assert cfg.nllb_target_lang == "eng_Latn"


# ═════════════════════════════════════════════════════════════════════════════
# Integration tests — backend lifecycle
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestNLLBBackendE2E:
    """End-to-end tests with a real NLLB model (CPU, requires download)."""

    @pytest.fixture(scope="class")
    def backend(self):
        """Load NLLB 600M distilled on CPU.

        Teardown: explicit del + gc.collect() to release the 600M model memory.
        Without this, the model stays resident in memory until the test process
        exits, consuming ~2.4 GB of RAM for the duration of the test suite.
        """
        from benchmark.hardware.backend import DeviceInfo
        from benchmark.inference.backends.protocol import BackendConfig
        from benchmark.inference.backends.nllb import NLLBBackend

        try:
            from transformers import AutoModelForSeq2SeqLM
        except ImportError:
            pytest.skip("Transformers not installed")

        di = DeviceInfo(backend="cpu", device=torch.device("cpu"), num_devices=1, name="TEST")
        config = BackendConfig(
            model_path="facebook/nllb-200-distilled-600M",
            device_info=di,
            max_input_tokens=128,
            max_new_tokens=64,
            extra={
                "nllb_source_lang": "eng_Latn",
                "nllb_target_lang": "tur_Latn",
                "num_beams": 4,
            },
        )
        be = NLLBBackend(config)
        be.load()
        yield be
        # ── Teardown: force-unload the 600M model ──
        if hasattr(be, 'model') and be.model is not None:
            be.model.cpu()
            del be.model
        if hasattr(be, 'tokenizer') and be.tokenizer is not None:
            del be.tokenizer
        del be
        gc.collect()
        # On CUDA, also clear the cache.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_load_sets_attributes(self, backend):
        """After load(), key attributes are populated."""
        assert backend.is_loaded()
        assert backend.model is not None
        assert backend.tokenizer is not None
        assert len(backend.devices) >= 1

    def test_model_type(self, backend):
        """Backend reports correct model type."""
        from benchmark.inference.backends.protocol import ModelType
        assert backend.model_type == ModelType.ENCODER_DECODER

    def test_translate_single(self, backend):
        """Single-sentence translation produces Turkish output."""
        tok = backend.tokenizer
        # NLLB works without chat templates — just text
        enc = tok("The weather is nice today.", return_tensors="pt")

        batch = types.SimpleNamespace(
            batch_id=1,
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            raw_texts=["The weather is nice today."],
        )

        result = backend.translate_batch(batch)

        assert result.batch_size == 1
        assert len(result.generations) == 1
        gen = result.generations[0]
        assert len(gen.translated_text) > 0
        assert gen.output_tokens > 0
        assert gen.input_text == "The weather is nice today."

    def test_translate_batch(self, backend):
        """Batch translation works for multiple sentences."""
        tok = backend.tokenizer
        texts = [
            "The weather is nice today.",
            "Machine learning is transforming the world.",
            "I love programming computers.",
        ]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)

        batch = types.SimpleNamespace(
            batch_id=2,
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            raw_texts=texts,
        )

        result = backend.translate_batch(batch)

        assert result.batch_size == 3
        assert len(result.generations) == 3
        for i, gen in enumerate(result.generations):
            assert gen.input_text == texts[i]
            assert len(gen.translated_text) > 0, f"Empty translation for: {texts[i]}"
            assert gen.output_tokens > 0

    def test_output_is_utf8(self, backend):
        """Translations are valid UTF-8."""
        tok = backend.tokenizer
        enc = tok("Hello world.", return_tensors="pt")

        batch = types.SimpleNamespace(
            batch_id=3,
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            raw_texts=["Hello world."],
        )

        result = backend.translate_batch(batch)
        gen = result.generations[0]
        gen.translated_text.encode("utf-8")  # Must not raise

    def test_forced_bos_token_resolved(self, backend):
        """Target language token ID is resolved correctly."""
        assert backend._forced_bos_id is not None
        assert isinstance(backend._forced_bos_id, int)

    def test_capabilities(self, backend):
        """Backend declares correct capabilities."""
        from benchmark.inference.backends.protocol import ModelCapability
        caps = backend.capabilities
        assert ModelCapability.TRANSLATE in caps
        assert ModelCapability.ENSEMBLE_READY in caps


# ═════════════════════════════════════════════════════════════════════════════
# Pipeline test — _build_translation_prompt handles NLLB correctly
# ═════════════════════════════════════════════════════════════════════════════


class TestNLLBPipelinePrompt:
    """_build_translation_prompt returns raw text for NLLB tokenizers."""

    def test_nllb_tokenizer_skips_chat_template(self, real_tokenizer):
        """NLLB tokenizer with src_lang skips chat template."""
        from benchmark.data.pipeline import AsyncPipeline

        # Create a mock NLLB-like tokenizer object
        class MockNLLBTokenizer:
            src_lang = "eng_Latn"

        tok = MockNLLBTokenizer()
        result = AsyncPipeline._build_translation_prompt("hello world", tok)
        assert result == "hello world"  # raw text, no chat template

    def test_gemma_tokenizer_uses_chat_template(self, real_tokenizer):
        """Gemma tokenizer continues to use chat template."""
        from benchmark.data.pipeline import AsyncPipeline

        # real_tokenizer is Gemma 4B — has no src_lang attr
        result = AsyncPipeline._build_translation_prompt("hello world", real_tokenizer)
        assert "<start_of_turn>" in result or "source_lang_code" in result or len(result) > len("hello world") * 2
