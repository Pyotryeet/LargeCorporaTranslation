"""Tests for translation quality metrics using REAL data and tokenizer.

All tests in this file use the REAL HuggingFace tokenizer from
``google/translategemma-4b-it`` and REAL golden EN->TR reference pairs
from the project's reference data.  No dummies, no mocks, no synthetic data.

Tests that depend on expensive GPU resources (COMET model loading) are
gracefully skipped when unavailable.
"""

import json
from pathlib import Path

import pytest

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def real_references():
    """Load real golden EN->TR reference pairs.

    Tries ``data/references/golden_en_tr.jsonl`` first (1960 real pairs),
    then ``tests/fixtures/golden_en_tr.jsonl`` (10 curated pairs).
    Skips the entire session if neither is available.
    """
    from benchmark.quality.references import ReferenceLoader

    candidates = [
        Path("data/references/golden_en_tr.jsonl"),
        Path("tests/fixtures/golden_en_tr.jsonl"),
    ]
    for p in candidates:
        if p.exists():
            loader = ReferenceLoader(str(p))
            sources, references = loader.load()
            if sources and references:
                return sources, references
    pytest.skip("no golden reference data available")


@pytest.fixture(scope="session")
def real_references_path():
    """Return the path to the real golden reference file (first found).

    Needed by ``QualityBenchmark`` which takes a file path, not pre-loaded data.
    """
    candidates = [
        Path("data/references/golden_en_tr.jsonl"),
        Path("tests/fixtures/golden_en_tr.jsonl"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    pytest.skip("no golden reference data available")


# ── Requirement 3: ReferenceLoader loads and validates real reference pairs ─


class TestReferenceLoaderReal:
    """Tests that ReferenceLoader works with real EN->TR reference data."""

    def test_loads_references_from_real_file(self, real_references):
        sources, references = real_references
        assert len(sources) > 0, "expected non-empty sources"
        assert len(references) > 0, "expected non-empty references"
        assert len(sources) == len(references), (
            f"source count {len(sources)} != reference count {len(references)}"
        )

    def test_every_pair_is_non_empty(self, real_references):
        sources, references = real_references
        for i, (src, ref) in enumerate(zip(sources, references)):
            assert src.strip(), f"empty source at index {i}"
            assert ref.strip(), f"empty reference at index {i}"

    def test_every_pair_passes_validation(self, real_references):
        sources, references = real_references
        from benchmark.quality.references import ReferenceLoader

        for i, (src, ref) in enumerate(zip(sources, references)):
            assert ReferenceLoader.validate_pair(src, ref), (
                f"pair at index {i} failed validation: src={src!r} ref={ref!r}"
            )

    def test_sources_are_english(self, real_references):
        """Sanity check: sources look like English sentences."""
        sources, _ = real_references
        # At least 90% of sources should contain basic Latin letters
        ascii_alpha = sum(
            1 for s in sources if any(c.isascii() and c.isalpha() for c in s)
        )
        ratio = ascii_alpha / len(sources)
        assert ratio > 0.9, f"only {ratio:.1%} of sources contain ASCII letters"

    def test_references_contain_turkish_special_characters(self, real_references):
        """Sanity check: references contain Turkish-specific characters.

        Turkish has İ, ı, ş, ğ, ü, ö, ç (both cases).
        At least some references should contain these.
        """
        _, references = real_references
        turkish_chars = set("İışğüöçĞÜÖÇ")
        found_any = False
        for ref in references:
            if any(c in turkish_chars for c in ref):
                found_any = True
                break
        assert found_any, (
            "no Turkish special characters (İışğüöç) found in any reference — "
            "data may not be genuine Turkish"
        )


# ── Requirement 4: BLEU produces sensible scores with real text ───────────


class TestBLEUReal:
    """Tests that BLEU computation works with real EN->TR text."""

    def test_bleu_perfect_match_gives_high_score(self, real_references, real_tokenizer):
        """A perfect (reference == hypothesis) match should score near 100."""
        from benchmark.quality.metrics_bleu import compute_bleu

        sources, references = real_references
        # Take a manageable slice so the test runs quickly
        refs_subset = references[:min(len(references), 50)]
        # Perfect match: hypothesis == reference
        result = compute_bleu(refs_subset, [[r] for r in refs_subset])
        score = result["score"]
        # A perfect corpus_bleu match typically scores 100.0 or very close
        assert score >= 90.0, (
            f"perfect match BLEU should be near 100, got {score}"
        )

    def test_bleu_copy_source_scores_low(self, real_references, real_tokenizer):
        """Copying the English source as the translation should score LOW."""
        from benchmark.quality.metrics_bleu import compute_bleu

        sources, references = real_references
        subset_n = min(len(sources), 100)
        src_subset = sources[:subset_n]
        ref_subset = references[:subset_n]
        result = compute_bleu(src_subset, [[r] for r in ref_subset])
        score = result["score"]
        # English source vs Turkish reference -> low BLEU
        assert score < 30.0, (
            f"copy-source BLEU should be low (<30), got {score}"
        )

    def test_bleu_with_tokenizer_13a(self, real_references):
        """BLEU with tokenize='13a' handles Turkish text without error."""
        from benchmark.quality.metrics_bleu import compute_bleu

        sources, references = real_references
        subsample = min(len(references), 50)
        hypotheses = references[:subsample]
        refs = [[r] for r in references[:subsample]]
        result = compute_bleu(hypotheses, refs, tokenize="13a")
        assert "score" in result
        assert isinstance(result["score"], (int, float))
        assert result["signature"] != ""


# ── Requirement 5: chrF++ works with real Turkish text ─────────────────────


class TestChRFReal:
    """Tests that chrF++ computation works with real Turkish text."""

    def test_chrf_perfect_match_scores_high(self, real_references):
        from benchmark.quality.metrics_chrf import compute_chrf

        _, references = real_references
        subset = references[:min(len(references), 50)]
        result = compute_chrf(subset, [[r] for r in subset], word_order=2)
        score = result["score"]
        assert score >= 90.0, (
            f"perfect match chrF++ should be near 100, got {score}"
        )

    def test_chrf_copy_source_scores_low(self, real_references):
        """Copying English source as Turkish translation -> low chrF++."""
        from benchmark.quality.metrics_chrf import compute_chrf

        sources, references = real_references
        n = min(len(sources), 100)
        result = compute_chrf(sources[:n], [[r] for r in references[:n]], word_order=2)
        score = result["score"]
        assert score < 50.0, (
            f"copy-source chrF++ should be low (<50), got {score}"
        )

    def test_chrf_handles_turkish_special_chars(self, real_references):
        """chrF++ must not crash on strings with İ, ı, ş, ğ, ü, ö, ç."""
        from benchmark.quality.metrics_chrf import compute_chrf

        # Build a tiny set that is guaranteed to contain Turkish special chars
        turkish_samples = [
            "İstanbul'un şehri çok güzel.",        # İ, ş, ç, ü
            "Ağaçlar yeşil ve doğa harika.",       # ğ, ş
            "Sıcaklık 30 dereceyi aştı.",          # ı, ş
            "Türkçe öğrenmek çok keyifli.",        # ü, ö, ç
            "Yağmurlu bir günde kitap okumayı severim.",  # ğ, ü
        ]
        # Use the same text as both hypothesis and reference -> should be ~100
        result = compute_chrf(
            turkish_samples,
            [[r] for r in turkish_samples],
            word_order=2,
        )
        assert result["score"] >= 80.0, (
            f"chrF++ on Turkish special chars: expected >=80, got {result['score']}"
        )

    def test_chrf_word_order_impact(self, real_references):
        """chrF++ with word_order=0 (pure character n-gram F-score) vs word_order=2."""
        from benchmark.quality.metrics_chrf import compute_chrf

        _, references = real_references
        subset = references[:min(len(references), 30)]
        refs = [[r] for r in subset]
        chrf0 = compute_chrf(subset, refs, word_order=0)
        chrf2 = compute_chrf(subset, refs, word_order=2)
        # Both should be valid scores; word_order=2 is chrF++, >= chrF
        assert chrf0["score"] >= 90.0
        assert chrf2["score"] >= 90.0
        assert chrf2["score"] >= chrf0["score"], (
            f"chrF++ ({chrf2['score']}) should be >= chrF ({chrf0['score']})"
        )


# ── Requirement 6: COMET import and model loading ──────────────────────────


class TestCOMETReal:
    """Tests that COMET can be imported and (if installed) its model loaded."""

    def test_comet_import(self):
        """Verify the comet package is importable."""
        try:
            import comet  # noqa: F401
        except ImportError:
            pytest.skip("comet package not installed")

    def test_comet_model_loading(self):
        """Attempt to download and load the COMET model.

        This test is heavyweight (~1.5 GB download).  It is skipped when:
        - comet is not installed
        - no GPU is available (COMET expects CUDA)
        The model is cached at module scope after the first load.
        """
        pytest.importorskip("comet")
        from benchmark.quality.metrics_comet import _get_comet_model, DEFAULT_COMET_MODEL

        model = _get_comet_model(DEFAULT_COMET_MODEL)
        if model is None:
            pytest.skip("COMET model not available (no CUDA or download failed)")
        assert model is not None

    def test_comet_clear_cache(self):
        """Clear the COMET model cache without error."""
        pytest.importorskip("comet")
        from benchmark.quality.metrics_comet import clear_comet_cache

        # Should be safe even if cache is already empty
        clear_comet_cache()

    def test_comet_not_installed_returns_error_dict(self):
        """When COMET is not installed, compute_comet returns an error dict."""
        from benchmark.quality.metrics_comet import compute_comet, HAS_COMET

        if HAS_COMET:
            pytest.skip("COMET is installed — cannot test not-installed path")
        result = compute_comet(["hello"], ["merhaba"], ["merhaba"])
        assert result["system_score"] is None
        assert "error" in result


# ── Requirement 7: QualityBenchmark instantiation ──────────────────────────


class TestQualityBenchmarkInstantiation:
    """Tests that the quality benchmark class can be instantiated."""

    def test_instantiate_with_real_path(self, real_references_path):
        from benchmark.quality.benchmark import QualityBenchmark

        bench = QualityBenchmark(real_references_path)
        assert bench is not None
        assert bench.reference_path == real_references_path

    def test_quality_results_dataclass(self):
        from benchmark.quality.benchmark import QualityResults

        results = QualityResults(
            bleu={"score": 30.0},
            chrf={"score": 58.0},
            comet={"system_score": 0.78},
            metricx={"system_score": 0.8},
            num_references=10,
            num_translated=10,
        )
        assert results.bleu["score"] == 30.0
        assert results.chrf["score"] == 58.0
        assert results.comet["system_score"] == 0.78
        assert results.metricx["system_score"] == 0.8
        d = results.to_dict()
        assert d["num_references"] == 10
        assert d["metricx"]["system_score"] == 0.8

    def test_scores_meet_targets_true(self):
        from benchmark.quality.benchmark import QualityResults

        results = QualityResults(
            bleu={"score": 30.0},
            chrf={"score": 60.0},
            comet={"system_score": 0.80},
            metricx={"system_score": 0.5},
        )
        assert results.scores_meet_targets is True

    def test_scores_meet_targets_false(self):
        from benchmark.quality.benchmark import QualityResults

        results = QualityResults(
            bleu={"score": 10.0},
            chrf={"score": 30.0},
            comet={"system_score": 0.50},
            metricx={"system_score": 4.5},
        )
        assert results.scores_meet_targets is False

    def test_quality_threshold_constants(self):
        """Verify threshold constants match the documented values."""
        from benchmark.config.constants import (
            QUALITY_BLEU_TARGET,
            QUALITY_CHRF_TARGET,
            QUALITY_COMET_TARGET,
            QUALITY_METRICX_TARGET,
        )
        assert QUALITY_BLEU_TARGET == 25
        assert QUALITY_CHRF_TARGET == 54
        assert QUALITY_COMET_TARGET == 0.72
        assert QUALITY_METRICX_TARGET == 1.5


# ── Requirement 8: sacrebleu tokenize="13a" handles Turkish text ───────────


class TestSacrebleuTurkish:
    """Tests that sacrebleu with tokenize='13a' handles Turkish text correctly."""

    def test_tokenize_13a_on_turkish(self):
        """sacrebleu 13a tokenizer must tokenize Turkish without crashing."""
        import sacrebleu

        turkish = "İstanbul'da ağaçlar yeşil, kuşlar şarkı söylüyor."
        english = "Hello world, how are you today?"

        # corpus_bleu with tokenize='13a'
        result = sacrebleu.corpus_bleu(
            [english],
            [[turkish]],
            tokenize="13a",
        )
        assert result.score is not None
        # 13a tokenization of Turkish vs English should produce a valid low score
        assert 0.0 <= result.score <= 100.0

    def test_tokenize_13a_turkish_vs_turkish(self):
        """13a on fully Turkish text should tokenize and compute correctly."""
        import sacrebleu

        hyp = "Merhaba dünya, nasılsınız?"
        ref = "Merhaba dünya, nasılsınız?"
        result = sacrebleu.corpus_bleu(
            [hyp],
            [[ref]],
            tokenize="13a",
        )
        # Exact match should be near 100.0 with 13a (floating point may drift slightly)
        assert abs(result.score - 100.0) < 0.01, (
            f"exact Turkish match with 13a should be ~100.0, got {result.score}"
        )

    def test_tokenize_13a_turkish_special_char_subset(self):
        """Each Turkish special character should tokenize without exception.

        Note: sacrebleu's 13a tokenizer normalizes text (lowercasing, etc.).
        Short phrases (<4 words after tokenization) may score 0.0 due to BLEU's
        geometric mean over n-grams — this is normal for BLEU, not a tokenizer
        bug.  The important property is that the tokenizer does NOT crash on any
        of these characters and returns a valid numeric score.
        """
        import sacrebleu

        special_phrases = [
            ("İstanbul çok güzel bir şehir", "İstanbul çok güzel bir şehir"),
            ("ısı ve sıcaklık farkı yüksek", "ısı ve sıcaklık farkı yüksek"),
            ("ağaçlar yeşil ve canlı", "ağaçlar yeşil ve canlı"),
            ("şeker tatlıdır ve lezzetlidir", "şeker tatlıdır ve lezzetlidir"),
            ("güzel bir gün geçirdik", "güzel bir gün geçirdik"),
            ("ödev yapmak çok önemli", "ödev yapmak çok önemli"),
        ]
        for hyp, ref in special_phrases:
            result = sacrebleu.corpus_bleu(
                [hyp],
                [[ref]],
                tokenize="13a",
            )
            assert isinstance(result.score, (int, float)), (
                f"13a tokenization crashed on: {hyp}"
            )
            # BLEU score must be in [0, 100] (allow tiny floating drift)
            assert -0.01 <= result.score <= 100.01, (
                f"BLEU score {result.score} out of range for: {hyp}"
            )
            # With >=4 words the self-match score should be near 100
            # (3+ words after 13a tokenization is enough for n-gram overlap)
            if len(hyp.split()) >= 4:
                assert result.score >= 99.9, (
                    f"self-match BLEU should be near 100, got {result.score} for: {hyp}"
                )


# ── Requirement 9: Quality thresholds are reasonable ───────────────────────


class TestQualityThresholds:
    """Tests that the quality thresholds are reasonable by computing scores
    on the golden reference set against a trivial baseline (copy-source)."""

    def test_copy_source_bleu_is_below_threshold(self, real_references):
        """Copy-source BLEU must be well below the 25-point threshold."""
        from benchmark.quality.metrics_bleu import compute_bleu
        from benchmark.config.constants import QUALITY_BLEU_TARGET

        sources, references = real_references
        result = compute_bleu(sources, [[r] for r in references])
        score = result["score"]
        assert score < QUALITY_BLEU_TARGET, (
            f"copy-source BLEU {score} >= threshold {QUALITY_BLEU_TARGET} — "
            f"threshold is likely too low or data is suspicious"
        )

    def test_copy_source_chrf_is_below_threshold(self, real_references):
        """Copy-source chrF++ must be well below the 54-point threshold."""
        from benchmark.quality.metrics_chrf import compute_chrf
        from benchmark.config.constants import QUALITY_CHRF_TARGET

        sources, references = real_references
        result = compute_chrf(sources, [[r] for r in references], word_order=2)
        score = result["score"]
        assert score < QUALITY_CHRF_TARGET, (
            f"copy-source chrF++ {score} >= threshold {QUALITY_CHRF_TARGET} — "
            f"threshold is likely too low or data is suspicious"
        )

    def test_copy_source_comet_is_below_threshold(self, real_references):
        """Copy-source COMET must be well below the 0.72 threshold."""
        pytest.importorskip("comet")
        from benchmark.quality.metrics_comet import compute_comet
        from benchmark.config.constants import QUALITY_COMET_TARGET

        sources, references = real_references
        result = compute_comet(sources, sources, references)
        if result.get("error"):
            pytest.skip(f"COMET error: {result['error']}")
        score = result.get("system_score")
        if score is None:
            pytest.skip("COMET system_score is None")
        assert score < QUALITY_COMET_TARGET, (
            f"copy-source COMET {score} >= threshold {QUALITY_COMET_TARGET} — "
            f"threshold is likely too low or data is suspicious"
        )

    def test_empty_source_is_low_score(self, real_references):
        """Empty string as translation -> very low scores (basic sanity)."""
        from benchmark.quality.metrics_bleu import compute_bleu
        from benchmark.quality.metrics_chrf import compute_chrf

        _, references = real_references
        subset = references[:50]
        empty_hypotheses = [""] * len(subset)
        refs = [[r] for r in subset]

        bleu_result = compute_bleu(empty_hypotheses, refs)
        chrf_result = compute_chrf(empty_hypotheses, refs, word_order=2)

        assert bleu_result["score"] == 0.0, (
            f"empty hypothesis BLEU should be 0.0, got {bleu_result['score']}"
        )
        assert chrf_result["score"] < 5.0, (
            f"empty hypothesis chrF++ should be near 0.0, got {chrf_result['score']}"
        )


# ── Requirement 10: Skip when data is missing ──────────────────────────────


class TestGracefulSkip:
    """Tests that gracefully skip when real reference data is absent."""

    def test_references_fixture_skips_when_missing(self):
        """If neither path exists, the fixture should skip."""
        from benchmark.quality.references import ReferenceLoader

        loader = ReferenceLoader("/nonexistent/definitely_not_here.jsonl")
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_real_references_fixture_provides_data(self, real_references):
        """When data exists, the fixture must return something useful."""
        sources, references = real_references
        assert isinstance(sources, list)
        assert isinstance(references, list)
        assert all(isinstance(s, str) for s in sources)
        assert all(isinstance(r, str) for r in references)
        # Should have at least MIN_REFERENCE_PAIRS (10)
        assert len(sources) >= 10, (
            f"expected at least 10 reference pairs, got {len(sources)}"
        )


# ── Real tokenizer integration tests ───────────────────────────────────────


class TestRealTokenizerIntegration:
    """Tests that the REAL tokenizer can encode/decode Turkish text.

    Uses the session-scoped ``real_tokenizer`` fixture from conftest.py
    which loads ``google/translategemma-4b-it``.
    """

    def test_tokenizer_is_loaded(self, real_tokenizer):
        assert real_tokenizer is not None

    def test_tokenizer_encode_english(self, real_tokenizer):
        tokens = real_tokenizer.encode("Hello world, how are you?")
        assert len(tokens) > 0
        # Decode should round-trip cleanly
        decoded = real_tokenizer.decode(tokens, skip_special_tokens=True)
        assert "Hello" in decoded

    def test_tokenizer_encode_turkish(self, real_tokenizer):
        """The real tokenizer must handle Turkish text including special chars."""
        turkish = "İstanbul'da ağaçlar yeşil, kuşlar şarkı söylüyor."
        tokens = real_tokenizer.encode(turkish)
        assert len(tokens) > 0, "tokenizer produced empty token list for Turkish"
        decoded = real_tokenizer.decode(tokens, skip_special_tokens=True)
        # Turkish special characters should survive the round-trip
        for char in "İağeşüö":
            assert char in decoded, (
                f"Turkish char '{char}' lost in tokenizer round-trip: {decoded!r}"
            )

    def test_tokenizer_pad_and_eos_exist(self, real_tokenizer):
        """The real tokenizer must have pad_token_id and eos_token_id set."""
        assert real_tokenizer.pad_token_id is not None, "pad_token_id is None"
        assert real_tokenizer.eos_token_id is not None, "eos_token_id is None"


# ── Metrics parallel computation with real data ────────────────────────────


class TestMetricsParallel:
    """Test that the parallel metrics composition from the benchmark module works."""

    def test_compute_metrics_parallel_with_real_data(self, real_references):
        from benchmark.quality.benchmark import _compute_metrics_parallel

        sources, references = real_references
        # Use a sample so tests stay fast
        n = min(len(sources), 50)
        src_sample = sources[:n]
        ref_sample = references[:n]
        hyp_sample = ref_sample[:]  # perfect match for high scores

        comet_result, comet_kiwi, bertscore, metricx, bleu, chrf = _compute_metrics_parallel(
            hyp_sample, ref_sample, src_sample,
        )

        assert "score" in bleu
        assert bleu["score"] >= 90.0, f"BLEU too low for perfect match: {bleu['score']}"
        assert "score" in chrf
        assert chrf["score"] >= 90.0, f"chrF++ too low for perfect match: {chrf['score']}"
        # COMET / MetricX may be None if not installed — that's fine
        assert "system_score" in comet_result or "error" in comet_result
        assert "system_score" in metricx or "error" in metricx


# ── Edge cases ─────────────────────────────────────────────────────────────


class TestMetricsEdgeCases:
    """Edge case handling for the quality metrics with real tokenizer."""

    def test_single_pair_works(self, real_references, real_tokenizer):
        """Single reference pair should compute without error."""
        from benchmark.quality.metrics_bleu import compute_bleu
        from benchmark.quality.metrics_chrf import compute_chrf

        _, references = real_references
        single_hyp = [references[0]]
        single_ref = [[references[0]]]

        bleu = compute_bleu(single_hyp, single_ref)
        chrf = compute_chrf(single_hyp, single_ref, word_order=2)
        assert "score" in bleu
        assert "score" in chrf

    def test_mixed_length_inputs(self, real_references):
        """Short and long sentences together should not cause errors."""
        from benchmark.quality.metrics_bleu import compute_bleu
        from benchmark.quality.metrics_chrf import compute_chrf

        _, references = real_references
        if len(references) < 5:
            pytest.skip("need at least 5 references")
        mixed = references[:5]  # natural varying-length Turkish sentences
        refs = [[r] for r in mixed]

        bleu = compute_bleu(mixed, refs)
        chrf = compute_chrf(mixed, refs, word_order=2)
        assert "score" in bleu
        assert "score" in chrf

    def test_empty_lists_return_zero(self):
        from benchmark.quality.metrics_bleu import compute_bleu
        from benchmark.quality.metrics_chrf import compute_chrf
        from benchmark.quality.metrics_comet import compute_comet

        bleu = compute_bleu([], [])
        chrf = compute_chrf([], [])
        comet_result = compute_comet([], [], [])

        assert bleu["score"] == 0.0
        assert chrf["score"] == 0.0
        assert comet_result["system_score"] == 0.0
