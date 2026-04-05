"""Tests for M84: Confidence Interval Reporting."""

from __future__ import annotations

from xpyd_bench.bench.confidence_intervals import (
    ConfidenceInterval,
    bootstrap_ci,
    compute_confidence_intervals,
)
from xpyd_bench.bench.models import RequestResult

# ---------------------------------------------------------------------------
# bootstrap_ci unit tests
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_empty_values(self):
        pt, lo, hi = bootstrap_ci([])
        assert pt == 0.0
        assert lo == 0.0
        assert hi == 0.0

    def test_single_value(self):
        pt, lo, hi = bootstrap_ci([42.0])
        assert pt == 42.0
        assert lo == 42.0
        assert hi == 42.0

    def test_deterministic_with_seed(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        r1 = bootstrap_ci(values, seed=123)
        r2 = bootstrap_ci(values, seed=123)
        assert r1 == r2

    def test_lower_le_upper(self):
        values = [float(v) for v in range(1, 101)]
        pt, lo, hi = bootstrap_ci(values)
        assert lo <= pt <= hi

    def test_confidence_level_narrows(self):
        values = [float(v) for v in range(1, 101)]
        _, lo90, hi90 = bootstrap_ci(values, confidence_level=0.90)
        _, lo99, hi99 = bootstrap_ci(values, confidence_level=0.99)
        # 99% CI should be wider than 90% CI
        assert (hi99 - lo99) >= (hi90 - lo90)

    def test_tight_ci_for_constant_values(self):
        values = [5.0] * 50
        pt, lo, hi = bootstrap_ci(values)
        assert pt == 5.0
        assert lo == 5.0
        assert hi == 5.0


# ---------------------------------------------------------------------------
# ConfidenceInterval dataclass
# ---------------------------------------------------------------------------


class TestConfidenceIntervalDataclass:
    def test_to_dict(self):
        ci = ConfidenceInterval(
            metric="mean_ttft_ms",
            point_estimate=10.1234,
            lower=9.5678,
            upper=10.6789,
            confidence_level=0.95,
        )
        d = ci.to_dict()
        assert d["metric"] == "mean_ttft_ms"
        assert d["confidence_level"] == 0.95
        assert isinstance(d["point_estimate"], float)
        assert isinstance(d["lower"], float)
        assert isinstance(d["upper"], float)


# ---------------------------------------------------------------------------
# compute_confidence_intervals integration
# ---------------------------------------------------------------------------


def _make_request(
    success: bool = True,
    ttft_ms: float | None = None,
    tpot_ms: float | None = None,
    latency_ms: float = 100.0,
) -> RequestResult:
    r = RequestResult()
    r.success = success
    r.ttft_ms = ttft_ms
    r.tpot_ms = tpot_ms
    r.latency_ms = latency_ms
    return r


class TestComputeConfidenceIntervals:
    def test_empty_results(self):
        assert compute_confidence_intervals([]) == {}

    def test_basic_computation(self):
        reqs = [
            _make_request(
                ttft_ms=float(i),
                tpot_ms=float(i * 2),
                latency_ms=float(i * 3),
            )
            for i in range(1, 51)
        ]
        ci = compute_confidence_intervals(reqs, confidence_level=0.95)
        assert ci["confidence_level"] == 0.95
        assert "mean_ttft_ms" in ci["metrics"]
        assert "mean_tpot_ms" in ci["metrics"]
        assert "mean_latency_ms" in ci["metrics"]
        assert "throughput_rps" in ci["metrics"]

        # Check structure of each metric
        for name in ["mean_ttft_ms", "mean_tpot_ms", "mean_latency_ms"]:
            m = ci["metrics"][name]
            assert m["lower"] <= m["point_estimate"] <= m["upper"]
            assert m["confidence_level"] == 0.95

    def test_failed_requests_excluded(self):
        reqs = [
            _make_request(success=False, ttft_ms=1000.0, tpot_ms=1000.0, latency_ms=1000.0),
            _make_request(success=True, ttft_ms=10.0, tpot_ms=20.0, latency_ms=30.0),
            _make_request(success=True, ttft_ms=12.0, tpot_ms=22.0, latency_ms=32.0),
        ]
        ci = compute_confidence_intervals(reqs)
        assert "mean_ttft_ms" in ci["metrics"]

    def test_small_sample_warning(self):
        reqs = [
            _make_request(ttft_ms=10.0, tpot_ms=20.0, latency_ms=30.0)
            for _ in range(3)
        ]
        ci = compute_confidence_intervals(reqs)
        assert "warning" in ci

    def test_no_warning_for_large_sample(self):
        reqs = [
            _make_request(ttft_ms=float(i), tpot_ms=float(i), latency_ms=float(i))
            for i in range(1, 51)
        ]
        ci = compute_confidence_intervals(reqs)
        assert "warning" not in ci

    def test_custom_confidence_level(self):
        reqs = [
            _make_request(ttft_ms=float(i), tpot_ms=float(i), latency_ms=float(i))
            for i in range(1, 101)
        ]
        ci = compute_confidence_intervals(reqs, confidence_level=0.99)
        assert ci["confidence_level"] == 0.99

    def test_only_latency_no_ttft(self):
        reqs = [
            _make_request(ttft_ms=None, tpot_ms=None, latency_ms=float(i))
            for i in range(1, 51)
        ]
        ci = compute_confidence_intervals(reqs)
        assert "mean_ttft_ms" not in ci["metrics"]
        assert "mean_tpot_ms" not in ci["metrics"]
        assert "mean_latency_ms" in ci["metrics"]
