"""Tests for M35: Request Latency Heatmap."""

from __future__ import annotations

import argparse

from xpyd_bench.bench.models import BenchmarkResult, RequestResult
from xpyd_bench.reporting.heatmap import (
    HeatmapData,
    compute_heatmap,
    heatmap_html_snippet,
    render_terminal_heatmap,
)


def _make_result(n: int = 50) -> BenchmarkResult:
    """Build a synthetic BenchmarkResult with request data."""
    bench_start = 1000.0
    requests = []
    for i in range(n):
        requests.append(
            RequestResult(
                success=True,
                latency_ms=50 + (i % 10) * 20,
                start_time=bench_start + i * 0.5,
                prompt_tokens=10,
                completion_tokens=10,
            )
        )
    return BenchmarkResult(
        requests=requests,
        bench_start_time=bench_start,
        completed=n,
        total_duration_s=n * 0.5,
        num_prompts=n,
    )


class TestComputeHeatmap:
    """Tests for heatmap data computation."""

    def test_basic_grid_shape(self) -> None:
        result = _make_result(50)
        data = compute_heatmap(result, time_bins=10, latency_bins=8)
        assert len(data.grid) == 10
        assert all(len(row) == 8 for row in data.grid)
        assert len(data.time_edges) == 11
        assert len(data.latency_edges) == 9

    def test_max_count_positive(self) -> None:
        result = _make_result(50)
        data = compute_heatmap(result)
        assert data.max_count > 0

    def test_total_requests(self) -> None:
        result = _make_result(30)
        data = compute_heatmap(result)
        assert data.total_requests == 30

    def test_grid_sums_to_total(self) -> None:
        result = _make_result(40)
        data = compute_heatmap(result, time_bins=5, latency_bins=5)
        total = sum(sum(row) for row in data.grid)
        assert total == 40

    def test_empty_result(self) -> None:
        result = BenchmarkResult()
        data = compute_heatmap(result)
        assert data.grid == []
        assert data.max_count == 0
        assert data.total_requests == 0

    def test_single_request(self) -> None:
        result = BenchmarkResult(
            requests=[
                RequestResult(success=True, latency_ms=100.0, start_time=5.0)
            ],
            bench_start_time=5.0,
            completed=1,
        )
        data = compute_heatmap(result, time_bins=3, latency_bins=3)
        total = sum(sum(row) for row in data.grid)
        assert total == 1

    def test_failed_requests_excluded(self) -> None:
        result = BenchmarkResult(
            requests=[
                RequestResult(success=True, latency_ms=100.0, start_time=1.0),
                RequestResult(success=False, latency_ms=200.0, start_time=2.0),
                RequestResult(success=True, latency_ms=150.0, start_time=3.0),
            ],
            bench_start_time=1.0,
            completed=2,
            failed=1,
        )
        data = compute_heatmap(result)
        assert data.total_requests == 2

    def test_no_start_time_excluded(self) -> None:
        result = BenchmarkResult(
            requests=[
                RequestResult(success=True, latency_ms=100.0, start_time=None),
                RequestResult(success=True, latency_ms=150.0, start_time=2.0),
            ],
            bench_start_time=1.0,
            completed=2,
        )
        data = compute_heatmap(result)
        assert data.total_requests == 1


class TestTerminalHeatmap:
    """Tests for terminal heatmap rendering."""

    def test_renders_string(self) -> None:
        result = _make_result(50)
        data = compute_heatmap(result)
        output = render_terminal_heatmap(data)
        assert isinstance(output, str)
        assert "Latency Heatmap" in output
        assert "Legend" in output

    def test_empty_data(self) -> None:
        data = HeatmapData()
        output = render_terminal_heatmap(data)
        assert "no data" in output

    def test_contains_time_labels(self) -> None:
        result = _make_result(50)
        data = compute_heatmap(result)
        output = render_terminal_heatmap(data)
        assert "s" in output

    def test_contains_latency_labels(self) -> None:
        result = _make_result(50)
        data = compute_heatmap(result)
        output = render_terminal_heatmap(data)
        assert "ms" in output


class TestHtmlHeatmap:
    """Tests for HTML heatmap snippet."""

    def test_renders_html(self) -> None:
        result = _make_result(50)
        data = compute_heatmap(result)
        snippet = heatmap_html_snippet(data)
        assert "<canvas" in snippet
        assert "heatmapCanvas" in snippet
        assert "<script>" in snippet

    def test_empty_data(self) -> None:
        data = HeatmapData()
        snippet = heatmap_html_snippet(data)
        assert "No data" in snippet

    def test_contains_grid_data(self) -> None:
        result = _make_result(20)
        data = compute_heatmap(result, time_bins=5, latency_bins=5)
        snippet = heatmap_html_snippet(data)
        assert "var grid=" in snippet
        assert "var maxCount=" in snippet


class TestHeatmapInHtmlReport:
    """Test that heatmap is included in the HTML report."""

    def test_html_report_includes_heatmap(self, tmp_path) -> None:
        from xpyd_bench.reporting.html_report import export_html_report

        result = _make_result(30)
        out = tmp_path / "report.html"
        export_html_report(result, str(out))
        content = out.read_text()
        assert "heatmapCanvas" in content
        assert "Latency Heatmap" in content


class TestHeatmapCliFlag:
    """Test --heatmap CLI flag is accepted."""

    def test_heatmap_flag_parsed(self) -> None:
        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--heatmap"])
        assert args.heatmap is True

    def test_heatmap_default_false(self) -> None:
        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args([])
        assert args.heatmap is False
