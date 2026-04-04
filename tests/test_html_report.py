"""Tests for HTML report generation (M17)."""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

from xpyd_bench.bench.models import BenchmarkResult, RequestResult
from xpyd_bench.reporting.html_report import export_html_report


def _make_result(n: int = 20, partial: bool = False) -> BenchmarkResult:
    """Create a minimal BenchmarkResult with *n* successful requests."""
    requests = []
    for i in range(n):
        requests.append(
            RequestResult(
                prompt_tokens=10,
                completion_tokens=20,
                ttft_ms=10.0 + i,
                tpot_ms=5.0 + i * 0.5,
                itl_ms=[5.0, 6.0],
                latency_ms=100.0 + i * 10,
                success=True,
            )
        )
    return BenchmarkResult(
        backend="openai",
        base_url="http://localhost:8000",
        endpoint="/v1/completions",
        model="test-model",
        num_prompts=n,
        completed=n,
        failed=0,
        total_duration_s=10.0,
        request_throughput=n / 10.0,
        output_throughput=20.0 * n / 10.0,
        total_token_throughput=30.0 * n / 10.0,
        total_input_tokens=10 * n,
        total_output_tokens=20 * n,
        requests=requests,
        mean_ttft_ms=19.5,
        median_ttft_ms=19.5,
        p50_ttft_ms=19.5,
        p90_ttft_ms=28.0,
        p95_ttft_ms=28.5,
        p99_ttft_ms=29.0,
        mean_tpot_ms=9.75,
        median_tpot_ms=9.75,
        p50_tpot_ms=9.75,
        p90_tpot_ms=14.0,
        p95_tpot_ms=14.25,
        p99_tpot_ms=14.5,
        mean_itl_ms=5.5,
        median_itl_ms=5.5,
        p50_itl_ms=5.5,
        p90_itl_ms=5.5,
        p95_itl_ms=5.5,
        p99_itl_ms=5.5,
        mean_e2el_ms=195.0,
        median_e2el_ms=195.0,
        p50_e2el_ms=195.0,
        p90_e2el_ms=280.0,
        p95_e2el_ms=285.0,
        p99_e2el_ms=290.0,
        partial=partial,
    )


class TestExportHtmlReport:
    """Tests for export_html_report function."""

    def test_creates_file(self, tmp_path: Path) -> None:
        result = _make_result()
        out = tmp_path / "report.html"
        ret = export_html_report(result, out)
        assert ret == out
        assert out.exists()

    def test_html_is_self_contained(self, tmp_path: Path) -> None:
        """No external resource references (http:// or https://)."""
        result = _make_result()
        out = tmp_path / "report.html"
        export_html_report(result, out)
        content = out.read_text()
        # Should not reference external scripts/stylesheets
        assert "src=\"http" not in content
        assert "href=\"http" not in content
        # Internal data URL or base URL in the report body is fine

    def test_contains_key_elements(self, tmp_path: Path) -> None:
        result = _make_result()
        out = tmp_path / "report.html"
        export_html_report(result, out)
        content = out.read_text()
        assert "<!DOCTYPE html>" in content
        assert "xPyD-bench Report" in content
        assert "test-model" in content
        assert "canvas" in content
        assert "__BENCH_DATA__" in content

    def test_contains_summary_table(self, tmp_path: Path) -> None:
        result = _make_result()
        out = tmp_path / "report.html"
        export_html_report(result, out)
        content = out.read_text()
        assert "Request Throughput" in content
        assert "Output Throughput" in content

    def test_contains_percentile_table(self, tmp_path: Path) -> None:
        result = _make_result()
        out = tmp_path / "report.html"
        export_html_report(result, out)
        content = out.read_text()
        assert "TTFT" in content
        assert "TPOT" in content
        assert "E2E Latency" in content
        # Check actual values appear
        assert "19.50" in content  # mean_ttft_ms

    def test_chart_data_embedded(self, tmp_path: Path) -> None:
        result = _make_result()
        out = tmp_path / "report.html"
        export_html_report(result, out)
        content = out.read_text()
        assert '"latencies"' in content
        assert '"ttfts"' in content
        assert '"timeSeries"' in content

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        result = _make_result()
        out = tmp_path / "sub" / "dir" / "report.html"
        export_html_report(result, out)
        assert out.exists()

    def test_partial_flag_shown(self, tmp_path: Path) -> None:
        result = _make_result(partial=True)
        out = tmp_path / "report.html"
        export_html_report(result, out)
        content = out.read_text()
        assert "PARTIAL" in content

    def test_empty_requests(self, tmp_path: Path) -> None:
        result = _make_result(n=0)
        result.completed = 0
        out = tmp_path / "report.html"
        export_html_report(result, out)
        content = out.read_text()
        assert "<!DOCTYPE html>" in content
        # Should still be valid HTML even with no data

    def test_single_request(self, tmp_path: Path) -> None:
        result = _make_result(n=1)
        out = tmp_path / "report.html"
        export_html_report(result, out)
        assert out.exists()
        content = out.read_text()
        assert "<!DOCTYPE html>" in content

    def test_chart_functions_present(self, tmp_path: Path) -> None:
        result = _make_result()
        out = tmp_path / "report.html"
        export_html_report(result, out)
        content = out.read_text()
        assert "function histogram" in content
        assert "function timeLine" in content
        assert "function cdf" in content
        assert "function scatter" in content


class TestHtmlReportCLI:
    """Test --html-report CLI integration."""

    def test_cli_flag_recognized(self) -> None:
        """CLI parser accepts --html-report without error."""
        from xpyd_bench.cli import bench_main

        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                bench_main(["--help"])
        except SystemExit:
            pass
        assert "--html-report" in buf.getvalue()

    def test_yaml_config_key(self, tmp_path: Path) -> None:
        """Verify html_report YAML key maps to the argument."""

        from xpyd_bench.cli import _load_yaml_config

        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text("html_report: /tmp/out.html\n")
        args = argparse.Namespace(html_report=None)
        _load_yaml_config(str(yaml_file), args)
        assert args.html_report == "/tmp/out.html"
