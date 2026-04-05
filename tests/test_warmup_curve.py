"""Tests for warmup curve analysis (M90)."""

from __future__ import annotations

import math

import numpy as np

from xpyd_bench.bench.warmup_curve import (
    WarmupCurveResult,
    build_warmup_curve,
    detect_convergence,
    fit_exponential_decay,
    render_ascii_curve,
)


class TestFitExponentialDecay:
    """Tests for exponential decay curve fitting."""

    def test_perfect_decay(self) -> None:
        """Fit perfectly generated exponential decay data."""
        a, b, c = 100.0, 0.2, 50.0
        x = np.arange(30, dtype=np.float64)
        latencies = [a * math.exp(-b * xi) + c for xi in x]
        fa, fb, fc, r_sq = fit_exponential_decay(latencies)
        # R² should be very high for perfect data.
        assert r_sq > 0.95
        # Steady state should be close to c.
        assert abs(fc - c) < 10.0

    def test_noisy_decay(self) -> None:
        """Fit noisy exponential decay data."""
        rng = np.random.default_rng(42)
        a, b, c = 80.0, 0.1, 30.0
        x = np.arange(50, dtype=np.float64)
        latencies = [a * math.exp(-b * xi) + c + rng.normal(0, 5) for xi in x]
        fa, fb, fc, r_sq = fit_exponential_decay(latencies)
        assert r_sq > 0.5  # Should still be decent fit.
        assert fc > 0  # Steady state positive.

    def test_flat_data(self) -> None:
        """Flat data should yield a=0, high R² not meaningful."""
        latencies = [50.0] * 20
        fa, fb, fc, r_sq = fit_exponential_decay(latencies)
        # a should be ~0 (no decay).
        assert abs(fa) < 1.0 or fb == 0.0

    def test_too_few_points(self) -> None:
        """Less than 3 points returns mean as c."""
        latencies = [100.0, 50.0]
        fa, fb, fc, r_sq = fit_exponential_decay(latencies)
        assert fa == 0.0
        assert fb == 0.0
        assert fc == 75.0
        assert r_sq == 0.0

    def test_empty_data(self) -> None:
        """Empty list returns zeros."""
        fa, fb, fc, r_sq = fit_exponential_decay([])
        assert fa == 0.0 and fb == 0.0 and fc == 0.0


class TestDetectConvergence:
    """Tests for convergence detection."""

    def test_decay_convergence(self) -> None:
        """Should detect convergence for exponential decay."""
        latencies = [100.0 * math.exp(-0.2 * i) + 20.0 for i in range(30)]
        conv = detect_convergence(latencies, a=100.0, b=0.2, c=20.0)
        assert conv is not None
        assert 10 <= conv <= 20  # Should converge around index 15.

    def test_no_decay(self) -> None:
        """No decay (b=0, a=0) should return 0 (already converged)."""
        latencies = [50.0] * 10
        conv = detect_convergence(latencies, a=0.0, b=0.0, c=50.0)
        assert conv == 0

    def test_slow_convergence(self) -> None:
        """Very slow decay may not converge within sample."""
        latencies = [100.0 * math.exp(-0.01 * i) + 20.0 for i in range(20)]
        conv = detect_convergence(latencies, a=100.0, b=0.01, c=20.0)
        # With b=0.01 and threshold=0.05, need ~300 requests.
        assert conv is None


class TestBuildWarmupCurve:
    """Tests for the build_warmup_curve function."""

    def test_basic_curve(self) -> None:
        """Build curve from decay data."""
        latencies = [100.0 * math.exp(-0.2 * i) + 30.0 for i in range(40)]
        result = build_warmup_curve(latencies)
        assert isinstance(result, WarmupCurveResult)
        assert result.fit_r_squared > 0.8
        assert result.steady_state_ms > 0
        assert result.cold_start_penalty_ms >= 0
        assert len(result.latencies_ms) == 40

    def test_to_dict(self) -> None:
        """Serialization to dict."""
        latencies = [100.0 * math.exp(-0.2 * i) + 30.0 for i in range(20)]
        result = build_warmup_curve(latencies)
        d = result.to_dict()
        assert "fit_params" in d
        assert "a" in d["fit_params"]
        assert "b" in d["fit_params"]
        assert "c" in d["fit_params"]
        assert "fit_r_squared" in d
        assert "convergence_index" in d
        assert "cold_start_penalty_ms" in d
        assert "steady_state_ms" in d
        assert "latencies_ms" in d

    def test_few_points(self) -> None:
        """Handles edge case with minimal points."""
        result = build_warmup_curve([100.0, 60.0, 40.0])
        assert isinstance(result, WarmupCurveResult)
        d = result.to_dict()
        assert len(d["latencies_ms"]) == 3

    def test_empty(self) -> None:
        """Empty latencies."""
        result = build_warmup_curve([])
        assert result.cold_start_penalty_ms == 0.0


class TestRenderAsciiCurve:
    """Tests for ASCII curve rendering."""

    def test_basic_render(self) -> None:
        """Should produce a multi-line ASCII plot."""
        latencies = [100.0 * math.exp(-0.2 * i) + 30.0 for i in range(20)]
        output = render_ascii_curve(latencies, 100.0, 0.2, 30.0, 15)
        assert isinstance(output, str)
        assert "*" in output  # Observed data points.
        assert "-" in output  # Fitted curve.
        assert "|" in output  # Convergence marker.
        assert "Convergence" in output

    def test_empty_data(self) -> None:
        """Empty data returns placeholder."""
        output = render_ascii_curve([], 0, 0, 0, None)
        assert output == "(no data)"

    def test_no_convergence(self) -> None:
        """No convergence marker when None."""
        latencies = [50.0] * 5
        output = render_ascii_curve(latencies, 0, 0, 50.0, None)
        assert "Convergence" not in output


class TestWarmupCurveIntegration:
    """Integration-style tests."""

    def test_realistic_warmup_pattern(self) -> None:
        """Simulate realistic server warmup pattern."""
        rng = np.random.default_rng(123)
        # First request very slow, rapid improvement, then steady.
        latencies = []
        for i in range(50):
            base = 200.0 * math.exp(-0.3 * i) + 45.0
            noise = rng.normal(0, 3)
            latencies.append(max(base + noise, 1.0))

        result = build_warmup_curve(latencies)
        assert result.fit_r_squared > 0.7
        assert result.cold_start_penalty_ms > 100  # Significant cold start.
        assert result.convergence_index is not None
        assert result.convergence_index < 30  # Should converge within 30 requests.

    def test_no_warmup_needed(self) -> None:
        """Flat latencies = no warmup needed."""
        rng = np.random.default_rng(99)
        latencies = [50.0 + rng.normal(0, 2) for _ in range(30)]
        result = build_warmup_curve(latencies)
        # Cold start penalty should be minimal.
        assert result.cold_start_penalty_ms < 20
