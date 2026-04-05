"""Tests for streaming vs non-streaming overhead analysis (M76)."""

from __future__ import annotations

import asyncio
import json
from argparse import Namespace
from unittest.mock import patch

from xpyd_bench.bench.models import BenchmarkResult, RequestResult
from xpyd_bench.stream_compare import (
    StreamCompareResult,
    StreamOverhead,
    _compute_overhead,
    export_stream_compare_json,
    export_stream_compare_markdown,
    format_stream_compare_markdown,
    format_stream_compare_summary,
    run_stream_compare,
)


def _make_result(
    latencies: list[float],
    ttfts: list[float] | None = None,
    throughput: float = 10.0,
) -> tuple[dict, BenchmarkResult]:
    """Create a fake benchmark result."""
    if ttfts is None:
        ttfts = [lat * 0.3 for lat in latencies]
    requests = []
    for lat, ttft in zip(latencies, ttfts):
        requests.append(RequestResult(
            latency_ms=lat,
            ttft_ms=ttft,
            tpot_ms=lat * 0.05,
            prompt_tokens=10,
            completion_tokens=20,
            success=True,
        ))
    mean_lat = sum(latencies) / len(latencies)
    sorted_lat = sorted(latencies)
    mean_ttft = sum(ttfts) / len(ttfts)
    sorted_ttft = sorted(ttfts)
    br = BenchmarkResult(
        model="test-model",
        base_url="http://localhost:8000",
        completed=len(latencies),
        failed=0,
        total_duration_s=5.0,
        request_throughput=throughput,
        output_throughput=throughput * 20.0,
        total_token_throughput=throughput * 30.0,
        mean_ttft_ms=mean_ttft,
        p50_ttft_ms=sorted_ttft[len(ttfts) // 2],
        p90_ttft_ms=sorted_ttft[int(len(ttfts) * 0.9)],
        p99_ttft_ms=sorted_ttft[-1],
        mean_tpot_ms=mean_lat * 0.05,
        mean_e2el_ms=mean_lat,
        p50_e2el_ms=sorted_lat[len(latencies) // 2],
        p90_e2el_ms=sorted_lat[int(len(latencies) * 0.9)],
        p99_e2el_ms=sorted_lat[-1],
        requests=requests,
    )
    rd = {
        "model": "test-model",
        "completed": br.completed,
        "failed": br.failed,
        "request_throughput": br.request_throughput,
        "output_throughput": br.output_throughput,
        "total_token_throughput": br.total_token_throughput,
        "mean_ttft_ms": br.mean_ttft_ms,
        "p50_ttft_ms": br.p50_ttft_ms,
        "p90_ttft_ms": br.p90_ttft_ms,
        "p99_ttft_ms": br.p99_ttft_ms,
        "mean_tpot_ms": br.mean_tpot_ms,
        "mean_e2el_ms": br.mean_e2el_ms,
        "p50_e2el_ms": br.p50_e2el_ms,
        "p90_e2el_ms": br.p90_e2el_ms,
        "p99_e2el_ms": br.p99_e2el_ms,
    }
    return rd, br


class TestComputeOverhead:
    """Tests for overhead calculation."""

    def test_streaming_slower(self):
        _, ns = _make_result([100.0, 110.0, 105.0], throughput=10.0)
        _, s = _make_result([150.0, 160.0, 155.0], throughput=8.0)
        o = _compute_overhead(s, ns)
        assert o.mean_latency_delta_ms > 0  # streaming slower
        assert o.throughput_delta_pct < 0  # streaming lower throughput

    def test_streaming_beneficial_ttft(self):
        # Streaming has lower TTFT
        _, ns = _make_result([100.0, 110.0], ttfts=[50.0, 55.0], throughput=10.0)
        _, s = _make_result([120.0, 130.0], ttfts=[20.0, 25.0], throughput=9.0)
        o = _compute_overhead(s, ns)
        assert o.ttft_delta_ms is not None
        assert o.ttft_delta_ms < 0  # streaming has lower TTFT
        assert o.streaming_beneficial is True

    def test_identical_results(self):
        _, ns = _make_result([100.0, 100.0], throughput=10.0)
        _, s = _make_result([100.0, 100.0], throughput=10.0)
        o = _compute_overhead(s, ns)
        assert o.mean_latency_delta_ms == 0.0
        assert o.throughput_delta_pct == 0.0

    def test_zero_throughput_baseline(self):
        _, ns = _make_result([100.0], throughput=0.0)
        _, s = _make_result([100.0], throughput=5.0)
        o = _compute_overhead(s, ns)
        assert o.throughput_delta_pct == 0.0  # avoid division by zero


class TestRunStreamCompare:
    """Tests for run_stream_compare."""

    def test_basic_comparison(self):
        ns_rd, ns_br = _make_result([100.0, 110.0, 105.0], throughput=10.0)
        s_rd, s_br = _make_result([150.0, 160.0, 155.0], throughput=8.0)

        call_count = 0

        async def fake_run(args, base_url):
            nonlocal call_count
            if call_count == 0:
                # First call is non-streaming
                assert args.stream is False
                call_count += 1
                return ns_rd, ns_br
            assert args.stream is True
            call_count += 1
            return s_rd, s_br

        args = Namespace(
            base_url="http://localhost:8000",
            model="test-model",
            stream=None,
        )

        with patch("xpyd_bench.stream_compare.run_benchmark", side_effect=fake_run):
            result = asyncio.run(run_stream_compare(args, threshold_pct=5.0))

        assert result.streaming_result is not None
        assert result.non_streaming_result is not None
        assert result.overhead.mean_latency_delta_ms > 0
        assert result.comparison is not None

    def test_args_not_mutated(self):
        """Original args should not be modified."""
        rd, br = _make_result([100.0], throughput=10.0)

        async def fake_run(args, base_url):
            return rd, br

        args = Namespace(
            base_url="http://localhost:8000",
            model="test-model",
            stream=None,
        )

        with patch("xpyd_bench.stream_compare.run_benchmark", side_effect=fake_run):
            asyncio.run(run_stream_compare(args, threshold_pct=5.0))

        assert args.stream is None  # not mutated


class TestStreamCompareResult:
    """Tests for StreamCompareResult dataclass."""

    def test_to_dict_empty(self):
        r = StreamCompareResult()
        d = r.to_dict()
        assert d["base_url"] == ""
        assert d["overhead"]["ttft_delta_ms"] is None

    def test_to_dict_with_data(self):
        rd, br = _make_result([100.0], throughput=10.0)
        r = StreamCompareResult(
            base_url="http://localhost",
            model="m",
            streaming_result=br,
            non_streaming_result=br,
            streaming_dict=rd,
            non_streaming_dict=rd,
            overhead=StreamOverhead(
                ttft_delta_ms=-5.0,
                streaming_beneficial=True,
            ),
        )
        d = r.to_dict()
        assert d["model"] == "m"
        assert d["overhead"]["streaming_beneficial"] is True


class TestFormatting:
    """Tests for summary formatting."""

    def test_format_summary_no_results(self):
        r = StreamCompareResult()
        s = format_stream_compare_summary(r)
        assert "No results" in s

    def test_format_summary_with_results(self):
        _, ns = _make_result([100.0, 110.0], throughput=10.0)
        _, s = _make_result([150.0, 160.0], throughput=8.0)
        r = StreamCompareResult(
            base_url="http://localhost",
            model="test",
            streaming_result=s,
            non_streaming_result=ns,
            overhead=_compute_overhead(s, ns),
        )
        text = format_stream_compare_summary(r)
        assert "Streaming vs Non-Streaming" in text
        assert "TTFT Delta" in text

    def test_format_markdown_no_results(self):
        r = StreamCompareResult()
        md = format_stream_compare_markdown(r)
        assert "No results" in md

    def test_format_markdown_with_results(self):
        _, ns = _make_result([100.0, 110.0], throughput=10.0)
        _, s = _make_result([120.0, 130.0], ttfts=[10.0, 12.0], throughput=9.0)
        r = StreamCompareResult(
            base_url="http://localhost",
            model="test",
            streaming_result=s,
            non_streaming_result=ns,
            overhead=_compute_overhead(s, ns),
        )
        md = format_stream_compare_markdown(r)
        assert "Non-Streaming" in md
        assert "Streaming" in md
        assert "Overhead Summary" in md


class TestExport:
    """Tests for JSON and Markdown export."""

    def test_export_json(self, tmp_path):
        r = StreamCompareResult(
            base_url="http://localhost",
            model="test",
            overhead=StreamOverhead(ttft_delta_ms=-5.0),
        )
        p = export_stream_compare_json(r, tmp_path / "out.json")
        assert p.exists()
        data = json.loads(p.read_text())
        assert data["overhead"]["ttft_delta_ms"] == -5.0

    def test_export_markdown(self, tmp_path):
        _, ns = _make_result([100.0], throughput=10.0)
        _, s = _make_result([120.0], throughput=9.0)
        r = StreamCompareResult(
            base_url="http://localhost",
            model="test",
            streaming_result=s,
            non_streaming_result=ns,
            overhead=_compute_overhead(s, ns),
        )
        p = export_stream_compare_markdown(r, tmp_path / "out.md")
        assert p.exists()
        assert "Streaming" in p.read_text()


class TestCLIIntegration:
    """Tests for CLI integration."""

    def test_stream_compare_in_subcommands(self):
        from xpyd_bench.main import _SUBCOMMANDS
        assert "stream-compare" in _SUBCOMMANDS

    def test_cli_entry_exists(self):
        from xpyd_bench.cli import stream_compare_main
        assert callable(stream_compare_main)
