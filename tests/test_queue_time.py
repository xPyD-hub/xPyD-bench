"""Tests for M71: Request Queuing Time Measurement."""

from __future__ import annotations

import unittest

from xpyd_bench.bench.models import BenchmarkResult, RequestResult
from xpyd_bench.bench.queue_time import compute_queue_time_summary


class TestComputeQueueTimeSummary(unittest.TestCase):
    """Test queue time summary computation."""

    def test_basic_summary(self):
        """Queue time summary returns expected stats."""
        requests = [
            RequestResult(success=True, queue_time_ms=10.0),
            RequestResult(success=True, queue_time_ms=20.0),
            RequestResult(success=True, queue_time_ms=30.0),
            RequestResult(success=True, queue_time_ms=40.0),
            RequestResult(success=True, queue_time_ms=50.0),
        ]
        summary = compute_queue_time_summary(requests)
        assert summary is not None
        assert summary["count"] == 5
        assert summary["mean_ms"] == 30.0
        assert summary["min_ms"] == 10.0
        assert summary["max_ms"] == 50.0
        assert "p50_ms" in summary
        assert "p90_ms" in summary
        assert "p95_ms" in summary
        assert "p99_ms" in summary

    def test_no_queue_times(self):
        """Returns None when no queue times recorded."""
        requests = [
            RequestResult(success=True, queue_time_ms=None),
            RequestResult(success=True, queue_time_ms=None),
        ]
        assert compute_queue_time_summary(requests) is None

    def test_empty_requests(self):
        """Returns None for empty request list."""
        assert compute_queue_time_summary([]) is None

    def test_only_failed_requests_excluded(self):
        """Failed requests are excluded from summary."""
        requests = [
            RequestResult(success=False, queue_time_ms=100.0),
            RequestResult(success=False, queue_time_ms=200.0),
        ]
        assert compute_queue_time_summary(requests) is None

    def test_mixed_success_failure(self):
        """Only successful requests contribute to summary."""
        requests = [
            RequestResult(success=True, queue_time_ms=10.0),
            RequestResult(success=False, queue_time_ms=999.0),
            RequestResult(success=True, queue_time_ms=20.0),
        ]
        summary = compute_queue_time_summary(requests)
        assert summary is not None
        assert summary["count"] == 2
        assert summary["mean_ms"] == 15.0

    def test_single_request(self):
        """Summary works with a single request."""
        requests = [RequestResult(success=True, queue_time_ms=5.0)]
        summary = compute_queue_time_summary(requests)
        assert summary is not None
        assert summary["count"] == 1
        assert summary["mean_ms"] == 5.0
        assert summary["min_ms"] == 5.0
        assert summary["max_ms"] == 5.0

    def test_high_concurrency_scenario(self):
        """Queue times increase under high concurrency."""
        # Simulate increasing queue times (as would happen with concurrency limits)
        requests = [
            RequestResult(success=True, queue_time_ms=float(i * 10))
            for i in range(100)
        ]
        summary = compute_queue_time_summary(requests)
        assert summary is not None
        assert summary["count"] == 100
        assert summary["p99_ms"] > summary["p50_ms"]


class TestRequestResultQueueTimeField(unittest.TestCase):
    """Test queue_time_ms field on RequestResult."""

    def test_default_none(self):
        """queue_time_ms defaults to None."""
        r = RequestResult()
        assert r.queue_time_ms is None

    def test_set_value(self):
        """queue_time_ms can be set."""
        r = RequestResult(queue_time_ms=42.5)
        assert r.queue_time_ms == 42.5


class TestBenchmarkResultQueueTimeSummary(unittest.TestCase):
    """Test queue_time_summary field on BenchmarkResult."""

    def test_default_none(self):
        """queue_time_summary defaults to None."""
        r = BenchmarkResult()
        assert r.queue_time_summary is None

    def test_set_summary(self):
        """queue_time_summary can be set."""
        r = BenchmarkResult()
        r.queue_time_summary = {"count": 5, "mean_ms": 10.0}
        assert r.queue_time_summary["count"] == 5


if __name__ == "__main__":
    unittest.main()
