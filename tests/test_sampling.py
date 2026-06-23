"""Tests for decoding parameters."""

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
