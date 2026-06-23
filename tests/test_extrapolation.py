"""Tests for the extrapolation model (v3.3 — SEM + bootstrap)."""

import math
from benchmark.reporting.extrapolation import ExtrapolationModel


class TestExtrapolationModel:
    def test_basic_computation(self):
        m = ExtrapolationModel(total_tokens=1_000_000_000)
        result = m.compute(mean_tps=1000, std_tps=50, num_gpus=2, n_batches=100)
        assert result["mean_tokens_per_second"] == 1000
        assert result["days_point_estimate"] > 0
        assert result["days_95ci_lower"] >= 0
        assert result["days_95ci_upper"] > result["days_point_estimate"]
        assert result["gpu_hours"] > 0
        assert result["num_gpus"] == 2
        assert result["n_batches"] == 100

    def test_ci_uses_sem_not_std(self):
        """Fix P0-2: CI shrinks with sqrt(n) because SEM replaces raw std."""
        m = ExtrapolationModel(total_tokens=1_000_000_000)
        # Same mean/std, different batch counts.
        r1 = m.compute(mean_tps=1000, std_tps=50, num_gpus=2, n_batches=1)
        r100 = m.compute(mean_tps=1000, std_tps=50, num_gpus=2, n_batches=100)
        ci1 = r1["days_95ci_upper"] - r1["days_95ci_lower"]
        ci100 = r100["days_95ci_upper"] - r100["days_95ci_lower"]
        assert ci100 < ci1, (
            f"CI should shrink with more batches: {ci1=} vs {ci100=}"
        )
        # With 100 batches, raw CI should be ~10x narrower (1/sqrt(100)=0.1).
        ratio = ci1 / ci100 if ci100 > 0 else float('inf')
        assert 5 < ratio < 15, f"Expected ~10x shrink ratio, got {ratio:.1f}x"

    def test_n_batches_1_uses_raw_std(self):
        """With n_batches=1, SEM=std (sqrt(1)=1) — degenerate case."""
        m = ExtrapolationModel(total_tokens=1_000_000_000)
        r = m.compute(mean_tps=1000, std_tps=50, num_gpus=2, n_batches=1)
        # rel_uncertainty ≈ 5% (50/1000).
        assert abs(r["relative_uncertainty_pct"] - 5.0) < 2.0

    def test_confidence_interval_wider_with_more_uncertainty(self):
        m = ExtrapolationModel(total_tokens=1_000_000_000)
        low_var = m.compute(mean_tps=1000, std_tps=10, num_gpus=2, n_batches=25)
        high_var = m.compute(mean_tps=1000, std_tps=200, num_gpus=2, n_batches=25)
        low_range = low_var["days_95ci_upper"] - low_var["days_95ci_lower"]
        high_range = high_var["days_95ci_upper"] - high_var["days_95ci_lower"]
        assert high_range > low_range

    def test_zero_tps_returns_error(self):
        m = ExtrapolationModel()
        result = m.compute(mean_tps=0)
        assert "error" in result

    def test_cost_estimation(self):
        m = ExtrapolationModel(total_tokens=1_000_000_000, gpu_cost_per_hour=2.50)
        result = m.compute(mean_tps=1000, std_tps=50, num_gpus=2, n_batches=10)
        assert result["estimated_cost_usd"] is not None
        assert result["estimated_cost_usd"] > 0

    def test_negative_lower_bound_clamped_to_zero(self):
        m = ExtrapolationModel(total_tokens=1_000)
        result = m.compute(mean_tps=10, std_tps=100, num_gpus=2, n_batches=1)
        assert result["days_95ci_lower"] >= 0

    def test_sem_field_present(self):
        """SEM is included in the result for transparency."""
        m = ExtrapolationModel(total_tokens=1_000_000_000)
        r = m.compute(mean_tps=1000, std_tps=100, num_gpus=2, n_batches=25)
        assert "sem_tokens_per_second" in r
        # sem = 100 / sqrt(25) = 20
        assert abs(r["sem_tokens_per_second"] - 20.0) < 0.5


class TestBootstrapExtrapolation:
    def test_bootstrap_ci_produces_valid_percentiles(self):
        """Bootstrap CI produces symmetric-ish percentiles around point estimate."""
        import random
        random.seed(42)
        tps_samples = [random.gauss(1000, 50) for _ in range(30)]
        m = ExtrapolationModel(total_tokens=1_000_000_000)
        r = m.compute_bootstrap(tps_samples, num_gpus=2, n_bootstrap=2000, seed=42)
        assert "bootstrap_days_lower" in r
        assert "bootstrap_days_upper" in r
        assert r["bootstrap_days_lower"] > 0
        assert r["bootstrap_days_upper"] > r["bootstrap_days_lower"]
        assert r["days_point_estimate"] > 0
        assert r["method"] == "bootstrap"
        assert r["n_batches"] == 30

    def test_bootstrap_single_sample(self):
        """Bootstrap with 1 sample returns non-error (degenerate CI)."""
        m = ExtrapolationModel(total_tokens=1_000_000_000)
        r = m.compute_bootstrap([1000.0], num_gpus=2, n_bootstrap=100, seed=42)
        assert "error" not in r
        assert r["days_point_estimate"] > 0

    def test_bootstrap_empty_sample_errors(self):
        m = ExtrapolationModel()
        r = m.compute_bootstrap([], num_gpus=2)
        assert "error" in r

    def test_bootstrap_ci_deterministic(self):
        """Same seed = same CI."""
        tps_samples = [987.0, 1012.0, 998.0, 1005.0, 995.0]
        m = ExtrapolationModel(total_tokens=1_000_000_000)
        r1 = m.compute_bootstrap(tps_samples, n_bootstrap=500, seed=42)
        r2 = m.compute_bootstrap(tps_samples, n_bootstrap=500, seed=42)
        assert r1["bootstrap_days_lower"] == r2["bootstrap_days_lower"]
        assert r1["bootstrap_days_upper"] == r2["bootstrap_days_upper"]
