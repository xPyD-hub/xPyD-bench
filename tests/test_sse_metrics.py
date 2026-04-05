"""Tests for M53: Server-Sent Events (SSE) Metrics."""

from __future__ import annotations

import argparse

from xpyd_bench.bench.models import BenchmarkResult, RequestResult
from xpyd_bench.bench.sse_metrics import (
    ChunkTiming,
    RequestSSEMetrics,
    StallEvent,
    analyze_chunk_timings,
    compute_sse_aggregate,
)

# ---------------------------------------------------------------------------
# ChunkTiming / analyze_chunk_timings tests
# ---------------------------------------------------------------------------


class TestAnalyzeChunkTimings:
    """Test per-request SSE chunk analysis."""

    def test_basic_analysis(self):
        """Compute ITL stats from chunk timings."""
        timings = [
            ChunkTiming(timestamp=0.0, tokens=1, inter_token_ms=None),  # first chunk
            ChunkTiming(timestamp=0.05, tokens=1, inter_token_ms=50.0),
            ChunkTiming(timestamp=0.10, tokens=1, inter_token_ms=50.0),
            ChunkTiming(timestamp=0.16, tokens=1, inter_token_ms=60.0),
        ]
        result = analyze_chunk_timings(timings, stall_threshold_ms=1000.0)
        assert result.chunk_count == 4
        assert result.mean_itl_ms is not None
        assert abs(result.mean_itl_ms - 53.33) < 1.0
        assert result.jitter_ms is not None
        assert result.jitter_ms > 0
        assert len(result.stalls) == 0

    def test_stall_detection(self):
        """Detect stalls when gap exceeds threshold."""
        timings = [
            ChunkTiming(timestamp=0.0, tokens=1, inter_token_ms=None),
            ChunkTiming(timestamp=0.05, tokens=1, inter_token_ms=50.0),
            ChunkTiming(timestamp=2.10, tokens=1, inter_token_ms=2050.0),  # stall
            ChunkTiming(timestamp=2.15, tokens=1, inter_token_ms=50.0),
        ]
        result = analyze_chunk_timings(timings, stall_threshold_ms=1000.0)
        assert len(result.stalls) == 1
        assert result.stalls[0].duration_ms == 2050.0

    def test_custom_threshold(self):
        """Custom stall threshold."""
        timings = [
            ChunkTiming(timestamp=0.0, tokens=1, inter_token_ms=None),
            ChunkTiming(timestamp=0.2, tokens=1, inter_token_ms=200.0),
        ]
        result = analyze_chunk_timings(timings, stall_threshold_ms=100.0)
        assert len(result.stalls) == 1

        result2 = analyze_chunk_timings(timings, stall_threshold_ms=500.0)
        assert len(result2.stalls) == 0

    def test_empty_timings(self):
        """Empty chunk timings."""
        result = analyze_chunk_timings([], stall_threshold_ms=1000.0)
        assert result.chunk_count == 0
        assert result.mean_itl_ms is None
        assert result.jitter_ms is None

    def test_single_chunk(self):
        """Single chunk: no ITL data."""
        timings = [ChunkTiming(timestamp=0.0, tokens=1, inter_token_ms=None)]
        result = analyze_chunk_timings(timings, stall_threshold_ms=1000.0)
        assert result.chunk_count == 1
        assert result.mean_itl_ms is None

    def test_to_dict(self):
        """RequestSSEMetrics.to_dict produces expected structure."""
        metrics = RequestSSEMetrics(
            chunk_count=3,
            stalls=[StallEvent(start_s=1.0, duration_ms=1500.0)],
            mean_itl_ms=50.0,
            p50_itl_ms=48.0,
            p90_itl_ms=55.0,
            p99_itl_ms=60.0,
            jitter_ms=5.0,
        )
        d = metrics.to_dict()
        assert d["chunk_count"] == 3
        assert d["stall_count"] == 1
        assert d["mean_itl_ms"] == 50.0
        assert d["jitter_ms"] == 5.0


# ---------------------------------------------------------------------------
# compute_sse_aggregate tests
# ---------------------------------------------------------------------------


class TestComputeSSEAggregate:
    """Test cross-request SSE metric aggregation."""

    def test_aggregate_basic(self):
        """Aggregate across multiple requests."""
        rm1 = RequestSSEMetrics(
            chunk_count=3,
            chunk_timings=[
                ChunkTiming(timestamp=0.0, tokens=1, inter_token_ms=None),
                ChunkTiming(timestamp=0.05, tokens=1, inter_token_ms=50.0),
                ChunkTiming(timestamp=0.10, tokens=1, inter_token_ms=50.0),
            ],
        )
        rm2 = RequestSSEMetrics(
            chunk_count=2,
            chunk_timings=[
                ChunkTiming(timestamp=0.0, tokens=1, inter_token_ms=None),
                ChunkTiming(timestamp=0.1, tokens=1, inter_token_ms=100.0),
            ],
            stalls=[StallEvent(start_s=5.0, duration_ms=1500.0)],
        )
        result = compute_sse_aggregate([rm1, rm2])
        assert result["total_chunks"] == 5
        assert result["total_stalls"] == 1
        assert result["requests_with_stalls"] == 1
        assert result["mean_itl_ms"] is not None
        # (50 + 50 + 100) / 3 ≈ 66.67
        assert abs(result["mean_itl_ms"] - 66.67) < 1.0

    def test_aggregate_empty(self):
        """Empty list returns empty dict."""
        result = compute_sse_aggregate([])
        assert result == {}

    def test_aggregate_no_itl(self):
        """All single-chunk requests have no ITL data."""
        rm = RequestSSEMetrics(
            chunk_count=1,
            chunk_timings=[ChunkTiming(timestamp=0.0, tokens=1, inter_token_ms=None)],
        )
        result = compute_sse_aggregate([rm])
        assert result["mean_itl_ms"] is None
        assert result["jitter_ms"] is None


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


class TestSSEMetricsCLI:
    """Test --sse-metrics and --sse-stall-threshold-ms CLI arguments."""

    def test_default_off(self):
        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args([])
        assert args.sse_metrics is False
        assert args.sse_stall_threshold_ms == 1000.0

    def test_enable_flag(self):
        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--sse-metrics"])
        assert args.sse_metrics is True

    def test_custom_threshold(self):
        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--sse-stall-threshold-ms", "500"])
        assert args.sse_stall_threshold_ms == 500.0


# ---------------------------------------------------------------------------
# YAML config known keys
# ---------------------------------------------------------------------------


class TestSSEMetricsYAMLConfig:
    def test_known_keys(self):
        from xpyd_bench.config_cmd import _KNOWN_KEYS

        assert "sse_metrics" in _KNOWN_KEYS
        assert "sse_stall_threshold_ms" in _KNOWN_KEYS


# ---------------------------------------------------------------------------
# Model field tests
# ---------------------------------------------------------------------------


class TestModelFields:
    def test_request_result_chunk_timings_default(self):
        r = RequestResult()
        assert r.chunk_timings is None

    def test_request_result_chunk_timings_set(self):
        r = RequestResult(chunk_timings=[{"ts": 0}])
        assert r.chunk_timings is not None

    def test_benchmark_result_sse_metrics_default(self):
        r = BenchmarkResult()
        assert r.sse_metrics is None

    def test_benchmark_result_sse_metrics_set(self):
        r = BenchmarkResult(sse_metrics={"total_chunks": 10})
        assert r.sse_metrics["total_chunks"] == 10


# ---------------------------------------------------------------------------
# HTML report SSE section
# ---------------------------------------------------------------------------


class TestSSEHTMLReport:
    def test_sse_section_present(self):
        from xpyd_bench.reporting.html_report import _sse_section

        result = BenchmarkResult(sse_metrics={
            "total_chunks": 100,
            "total_stalls": 2,
            "requests_with_stalls": 1,
            "total_stall_duration_ms": 3000.0,
            "mean_itl_ms": 45.0,
            "p50_itl_ms": 42.0,
            "p90_itl_ms": 55.0,
            "p95_itl_ms": 58.0,
            "p99_itl_ms": 62.0,
            "jitter_ms": 8.0,
        })
        html = _sse_section(result)
        assert "SSE Streaming Metrics" in html
        assert "100" in html  # total_chunks
        assert "45.0" in html  # mean_itl_ms
        assert "Total stalls: 2" in html

    def test_sse_section_empty_when_no_metrics(self):
        from xpyd_bench.reporting.html_report import _sse_section

        result = BenchmarkResult()
        assert _sse_section(result) == ""
