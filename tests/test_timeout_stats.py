"""Tests for M70: Request Timeout Classification & Reporting."""

from __future__ import annotations

from xpyd_bench.bench.models import BenchmarkResult, RequestResult
from xpyd_bench.bench.timeout_stats import compute_timeout_summary


class TestComputeTimeoutSummary:
    """Tests for compute_timeout_summary()."""

    def test_no_requests(self) -> None:
        assert compute_timeout_summary([]) is None

    def test_no_timeouts(self) -> None:
        requests = [
            RequestResult(success=True, latency_ms=100.0),
            RequestResult(success=True, latency_ms=150.0),
            RequestResult(success=False, error="server error", latency_ms=200.0),
        ]
        assert compute_timeout_summary(requests) is None

    def test_some_timeouts(self) -> None:
        requests = [
            RequestResult(success=True, latency_ms=100.0),
            RequestResult(
                success=False, error="timeout", latency_ms=300000.0,
                timeout_detected=True,
            ),
            RequestResult(success=True, latency_ms=120.0),
            RequestResult(
                success=False, error="timeout", latency_ms=310000.0,
                timeout_detected=True,
            ),
        ]
        summary = compute_timeout_summary(requests)
        assert summary is not None
        assert summary["timeout_count"] == 2
        assert summary["total_requests"] == 4
        assert summary["timeout_percentage"] == 50.0
        assert summary["mean_latency_at_timeout_ms"] == 305000.0
        assert summary["min_latency_at_timeout_ms"] == 300000.0
        assert summary["max_latency_at_timeout_ms"] == 310000.0

    def test_all_timeouts(self) -> None:
        requests = [
            RequestResult(
                success=False, error="timeout", latency_ms=300000.0,
                timeout_detected=True,
            ),
            RequestResult(
                success=False, error="timeout", latency_ms=300000.0,
                timeout_detected=True,
            ),
        ]
        summary = compute_timeout_summary(requests)
        assert summary is not None
        assert summary["timeout_count"] == 2
        assert summary["timeout_percentage"] == 100.0

    def test_single_timeout(self) -> None:
        requests = [
            RequestResult(
                success=False, error="ReadTimeout", latency_ms=5000.0,
                timeout_detected=True,
            ),
        ]
        summary = compute_timeout_summary(requests)
        assert summary is not None
        assert summary["timeout_count"] == 1
        assert summary["total_requests"] == 1
        assert summary["timeout_percentage"] == 100.0
        assert summary["mean_latency_at_timeout_ms"] == 5000.0
        assert summary["min_latency_at_timeout_ms"] == 5000.0
        assert summary["max_latency_at_timeout_ms"] == 5000.0


class TestRequestResultTimeoutField:
    """Tests for timeout_detected field on RequestResult."""

    def test_default_false(self) -> None:
        r = RequestResult()
        assert r.timeout_detected is False

    def test_set_true(self) -> None:
        r = RequestResult(timeout_detected=True)
        assert r.timeout_detected is True


class TestBenchmarkResultTimeoutSummary:
    """Tests for timeout_summary field on BenchmarkResult."""

    def test_default_none(self) -> None:
        br = BenchmarkResult()
        assert br.timeout_summary is None

    def test_set_summary(self) -> None:
        summary = {"timeout_count": 3, "total_requests": 10, "timeout_percentage": 30.0}
        br = BenchmarkResult(timeout_summary=summary)
        assert br.timeout_summary == summary
