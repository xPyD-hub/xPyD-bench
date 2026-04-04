"""Tests for time-series bucketing using actual request timestamps (issue #96)."""

from __future__ import annotations

from xpyd_bench.bench.models import BenchmarkResult, RequestResult
from xpyd_bench.reporting.metrics import compute_time_series


class TestTimeSeriesWithTimestamps:
    """Verify compute_time_series uses start_time when available."""

    def test_bursty_requests_bucketed_correctly(self):
        """All requests in first second should land in bucket 0."""
        bench_start = 1000.0
        result = BenchmarkResult(
            total_duration_s=10.0,
            bench_start_time=bench_start,
        )
        # 50 requests all starting within the first second
        result.requests = [
            RequestResult(
                completion_tokens=10,
                success=True,
                start_time=bench_start + i * 0.02,  # 0.00 to 0.98s
            )
            for i in range(50)
        ]
        result.completed = 50

        ts = compute_time_series(result, window_s=1.0)
        assert len(ts) > 0
        # All 50 requests should be in the first bucket
        assert ts[0]["requests"] == 50
        # Remaining buckets should be empty
        for bucket in ts[1:]:
            assert bucket["requests"] == 0

    def test_evenly_spread_requests(self):
        """Requests spread across time should be distributed across buckets."""
        bench_start = 1000.0
        result = BenchmarkResult(
            total_duration_s=5.0,
            bench_start_time=bench_start,
        )
        # 5 requests, one per second
        result.requests = [
            RequestResult(
                completion_tokens=10,
                success=True,
                start_time=bench_start + i,
            )
            for i in range(5)
        ]
        result.completed = 5

        ts = compute_time_series(result, window_s=1.0)
        # First 5 buckets should each have 1 request
        for i in range(5):
            assert ts[i]["requests"] == 1

    def test_fallback_without_timestamps(self):
        """Without start_time, falls back to index-based distribution."""
        result = BenchmarkResult(
            total_duration_s=10.0,
            bench_start_time=0.0,  # No bench start time
        )
        result.requests = [
            RequestResult(completion_tokens=10, success=True)
            for _ in range(10)
        ]
        result.completed = 10

        ts = compute_time_series(result, window_s=1.0)
        assert len(ts) > 0
        # Should still work (legacy uniform distribution)
        total_reqs = sum(b["requests"] for b in ts)
        assert total_reqs == 10

    def test_start_time_set_on_request_result(self):
        """RequestResult has start_time field."""
        r = RequestResult(start_time=42.0)
        assert r.start_time == 42.0

    def test_bench_start_time_on_result(self):
        """BenchmarkResult has bench_start_time field."""
        r = BenchmarkResult(bench_start_time=100.0)
        assert r.bench_start_time == 100.0
