"""Tests for issue #78: unmeasured latency percentiles should be None, not 0.0."""

from __future__ import annotations

from xpyd_bench.bench.models import BenchmarkResult, RequestResult
from xpyd_bench.bench.runner import _compute_metrics, _to_dict


class TestUnmeasuredPercentiles:
    """Percentiles for unmeasured metrics must be None."""

    def test_non_streaming_ttft_is_none(self):
        """Non-streaming requests have no TTFT; percentiles should be None."""
        result = BenchmarkResult()
        result.requests = [
            RequestResult(prompt_tokens=10, completion_tokens=5, latency_ms=100.0),
            RequestResult(prompt_tokens=10, completion_tokens=5, latency_ms=120.0),
        ]
        result.total_duration_s = 1.0
        _compute_metrics(result)
        assert result.mean_ttft_ms is None
        assert result.p50_ttft_ms is None
        assert result.p99_ttft_ms is None
        # E2E latency should be measured
        assert result.mean_e2el_ms is not None

    def test_to_dict_none_serialization(self):
        """_to_dict should include None values (serialized as null in JSON)."""
        result = BenchmarkResult()
        result.requests = [
            RequestResult(prompt_tokens=10, completion_tokens=5, latency_ms=100.0),
        ]
        result.total_duration_s = 1.0
        _compute_metrics(result)
        d = _to_dict(result)
        assert d["mean_ttft_ms"] is None
        assert d["mean_e2el_ms"] is not None

    def test_default_benchmark_result_is_none(self):
        """Fresh BenchmarkResult should have None percentiles, not 0.0."""
        r = BenchmarkResult()
        assert r.mean_ttft_ms is None
        assert r.p99_e2el_ms is None
