"""Tests for decoding parameters."""

import pytest
from benchmark.inference.sampling import DecodingParams


class TestDecodingParams:
    def test_defaults(self):
        p = DecodingParams()
        assert p.max_new_tokens == 512
        assert p.temperature == 0.0
        assert p.do_sample is False
        assert p.num_beams == 1

    def test_to_dict(self):
        p = DecodingParams(max_new_tokens=256, temperature=0.7, do_sample=True)
        d = p.to_dict()
        assert d["max_new_tokens"] == 256
        assert d["temperature"] == 0.7
        assert d["do_sample"] is True

    # ── Temperature edge case tests ──

    def test_temperature_zero_with_sample_fails(self):
        """do_sample=True + temperature=0.0 should be rejected."""
        with pytest.raises(ValueError, match="do_sample=True"):
            DecodingParams(do_sample=True, temperature=0.0)

    def test_temperature_zero_with_greedy_ok(self):
        """do_sample=False + temperature=0.0 is valid (greedy)."""
        p = DecodingParams(do_sample=False, temperature=0.0)
        assert p.temperature == 0.0

    def test_temperature_high(self):
        """Very high temperature should be accepted."""
        p = DecodingParams(temperature=100.0, do_sample=True)
        assert p.temperature == 100.0

    def test_temperature_negative_raises(self):
        """Negative temperature should be rejected."""
        with pytest.raises(ValueError):
            DecodingParams(temperature=-0.1)

    def test_temperature_is_zero_property(self):
        """temperature_is_zero property reflects exact zero."""
        p = DecodingParams(temperature=0.0)
        assert p.temperature_is_zero is True
        p2 = DecodingParams(temperature=0.001, do_sample=True)
        assert p2.temperature_is_zero is False

    def test_num_beams_bounds(self):
        """num_beams must be >= 1."""
        p = DecodingParams(num_beams=1)
        assert p.num_beams == 1
        with pytest.raises(ValueError):
            DecodingParams(num_beams=0)

    def test_max_new_tokens_bounds(self):
        """max_new_tokens must be >= 1."""
        with pytest.raises(ValueError):
            DecodingParams(max_new_tokens=0)
