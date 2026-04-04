"""Tests for M14: CSV & Markdown Export Formats."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

from xpyd_bench.bench.models import BenchmarkResult, RequestResult
from xpyd_bench.reporting.formats import (
    _PER_REQUEST_COLUMNS,
    _SUMMARY_COLUMNS,
    export_csv_report,
    export_markdown_report,
    export_per_request_csv,
)


@pytest.fixture()
def sample_result() -> BenchmarkResult:
    """Build a minimal BenchmarkResult for testing."""
    reqs = [
        RequestResult(
            prompt_tokens=10,
            completion_tokens=20,
            ttft_ms=5.0,
            tpot_ms=2.0,
            itl_ms=[2.0, 2.0],
            latency_ms=50.0,
            retries=0,
            success=True,
            error=None,
        ),
        RequestResult(
            prompt_tokens=15,
            completion_tokens=25,
            ttft_ms=7.0,
            tpot_ms=3.0,
            itl_ms=[3.0, 3.0],
            latency_ms=70.0,
            retries=1,
            success=True,
            error=None,
        ),
        RequestResult(
            prompt_tokens=10,
            completion_tokens=0,
            ttft_ms=None,
            tpot_ms=None,
            itl_ms=[],
            latency_ms=100.0,
            retries=2,
            success=False,
            error="timeout",
        ),
    ]
    return BenchmarkResult(
        backend="openai",
        model="test-model",
        num_prompts=3,
        completed=2,
        failed=1,
        total_duration_s=1.5,
        request_throughput=2.0,
        output_throughput=30.0,
        total_token_throughput=50.0,
        mean_ttft_ms=6.0,
        p50_ttft_ms=5.0,
        p90_ttft_ms=7.0,
        p95_ttft_ms=7.0,
        p99_ttft_ms=7.0,
        mean_tpot_ms=2.5,
        p50_tpot_ms=2.0,
        p90_tpot_ms=3.0,
        p95_tpot_ms=3.0,
        p99_tpot_ms=3.0,
        mean_itl_ms=2.5,
        p50_itl_ms=2.0,
        p90_itl_ms=3.0,
        p95_itl_ms=3.0,
        p99_itl_ms=3.0,
        mean_e2el_ms=60.0,
        p50_e2el_ms=50.0,
        p90_e2el_ms=70.0,
        p95_e2el_ms=70.0,
        p99_e2el_ms=70.0,
        requests=reqs,
    )


# ---------------------------------------------------------------------------
# CSV summary report
# ---------------------------------------------------------------------------


class TestCsvReport:
    def test_creates_file(self, tmp_path: Path, sample_result: BenchmarkResult) -> None:
        out = tmp_path / "report.csv"
        result_path = export_csv_report(sample_result, out)
        assert result_path == out
        assert out.exists()

    def test_header_row(self, tmp_path: Path, sample_result: BenchmarkResult) -> None:
        out = tmp_path / "report.csv"
        export_csv_report(sample_result, out)
        reader = csv.reader(io.StringIO(out.read_text()))
        header = next(reader)
        expected = [col[0] for col in _SUMMARY_COLUMNS]
        assert header == expected

    def test_data_row_count(self, tmp_path: Path, sample_result: BenchmarkResult) -> None:
        out = tmp_path / "report.csv"
        export_csv_report(sample_result, out)
        rows = list(csv.reader(io.StringIO(out.read_text())))
        assert len(rows) == 2  # header + 1 data row

    def test_model_value(self, tmp_path: Path, sample_result: BenchmarkResult) -> None:
        out = tmp_path / "report.csv"
        export_csv_report(sample_result, out)
        reader = csv.DictReader(io.StringIO(out.read_text()))
        row = next(reader)
        assert row["model"] == "test-model"
        assert row["completed"] == "2"

    def test_creates_parent_dirs(self, tmp_path: Path, sample_result: BenchmarkResult) -> None:
        out = tmp_path / "sub" / "dir" / "report.csv"
        export_csv_report(sample_result, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# Markdown summary report
# ---------------------------------------------------------------------------


class TestMarkdownReport:
    def test_creates_file(self, tmp_path: Path, sample_result: BenchmarkResult) -> None:
        out = tmp_path / "report.md"
        result_path = export_markdown_report(sample_result, out)
        assert result_path == out
        assert out.exists()

    def test_has_three_lines(self, tmp_path: Path, sample_result: BenchmarkResult) -> None:
        out = tmp_path / "report.md"
        export_markdown_report(sample_result, out)
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 3  # header, separator, data

    def test_separator_format(self, tmp_path: Path, sample_result: BenchmarkResult) -> None:
        out = tmp_path / "report.md"
        export_markdown_report(sample_result, out)
        lines = out.read_text().strip().split("\n")
        # Separator line should contain only |, -, and spaces
        sep = lines[1]
        assert sep.startswith("|")
        assert all(c in "|- " for c in sep.replace("|", ""))

    def test_column_count_consistent(
        self, tmp_path: Path, sample_result: BenchmarkResult
    ) -> None:
        out = tmp_path / "report.md"
        export_markdown_report(sample_result, out)
        lines = out.read_text().strip().split("\n")
        for line in lines:
            # Count pipe-delimited columns (strip outer pipes)
            cols = [c.strip() for c in line.strip("|").split("|")]
            assert len(cols) == len(_SUMMARY_COLUMNS)

    def test_contains_model(self, tmp_path: Path, sample_result: BenchmarkResult) -> None:
        out = tmp_path / "report.md"
        export_markdown_report(sample_result, out)
        content = out.read_text()
        assert "test-model" in content


# ---------------------------------------------------------------------------
# Per-request CSV
# ---------------------------------------------------------------------------


class TestPerRequestCsv:
    def test_creates_file(self, tmp_path: Path, sample_result: BenchmarkResult) -> None:
        out = tmp_path / "requests.csv"
        result_path = export_per_request_csv(sample_result, out)
        assert result_path == out
        assert out.exists()

    def test_header_columns(self, tmp_path: Path, sample_result: BenchmarkResult) -> None:
        out = tmp_path / "requests.csv"
        export_per_request_csv(sample_result, out)
        reader = csv.reader(io.StringIO(out.read_text()))
        header = next(reader)
        assert header == list(_PER_REQUEST_COLUMNS)

    def test_row_count(self, tmp_path: Path, sample_result: BenchmarkResult) -> None:
        out = tmp_path / "requests.csv"
        export_per_request_csv(sample_result, out)
        rows = list(csv.reader(io.StringIO(out.read_text())))
        # header + 3 data rows
        assert len(rows) == 4

    def test_failed_request_error(
        self, tmp_path: Path, sample_result: BenchmarkResult
    ) -> None:
        out = tmp_path / "requests.csv"
        export_per_request_csv(sample_result, out)
        reader = csv.DictReader(io.StringIO(out.read_text()))
        rows = list(reader)
        failed = [r for r in rows if r["success"] == "False"]
        assert len(failed) == 1
        assert failed[0]["error"] == "timeout"
        assert failed[0]["retries"] == "2"

    def test_none_ttft_becomes_empty(
        self, tmp_path: Path, sample_result: BenchmarkResult
    ) -> None:
        out = tmp_path / "requests.csv"
        export_per_request_csv(sample_result, out)
        reader = csv.DictReader(io.StringIO(out.read_text()))
        rows = list(reader)
        failed = [r for r in rows if r["success"] == "False"]
        assert failed[0]["ttft_ms"] == ""
        assert failed[0]["tpot_ms"] == ""

    def test_successful_request_values(
        self, tmp_path: Path, sample_result: BenchmarkResult
    ) -> None:
        out = tmp_path / "requests.csv"
        export_per_request_csv(sample_result, out)
        reader = csv.DictReader(io.StringIO(out.read_text()))
        row = next(reader)
        assert row["prompt_tokens"] == "10"
        assert row["completion_tokens"] == "20"
        assert row["ttft_ms"] == "5.0"
        assert row["latency_ms"] == "50.0"
        assert row["retries"] == "0"


# ---------------------------------------------------------------------------
# CLI integration (argument parsing)
# ---------------------------------------------------------------------------


class TestCliArgs:
    def _make_parser(self):
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        return parser

    def test_csv_report_arg(self) -> None:
        parser = self._make_parser()
        args = parser.parse_args(["--model", "m", "--csv-report", "/tmp/r.csv"])
        assert args.csv_report == "/tmp/r.csv"

    def test_markdown_report_arg(self) -> None:
        parser = self._make_parser()
        args = parser.parse_args(["--model", "m", "--markdown-report", "/tmp/r.md"])
        assert args.markdown_report == "/tmp/r.md"

    def test_export_requests_csv_arg(self) -> None:
        parser = self._make_parser()
        args = parser.parse_args(["--model", "m", "--export-requests-csv", "/tmp/r.csv"])
        assert args.export_requests_csv == "/tmp/r.csv"

    def test_defaults_none(self) -> None:
        parser = self._make_parser()
        args = parser.parse_args(["--model", "m"])
        assert args.csv_report is None
        assert args.markdown_report is None
        assert args.export_requests_csv is None
