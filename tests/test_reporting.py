"""Tests for M6 — Extended Metrics & Reporting."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from xpyd_bench.bench.models import BenchmarkResult, RequestResult
from xpyd_bench.reporting.formats import (
    export_json_report,
    export_per_request,
    format_text_report,
)
from xpyd_bench.reporting.metrics import compute_time_series
from xpyd_bench.reporting.rich_output import RichProgressReporter


def _make_result(n: int = 20) -> BenchmarkResult:
    """Create a BenchmarkResult with *n* fake successful requests."""

    requests = []
    for i in range(n):
        r = RequestResult(
            prompt_tokens=10,
            completion_tokens=5 + i,
            ttft_ms=10.0 + i * 0.5,
            tpot_ms=2.0 + i * 0.1,
            itl_ms=[1.0 + i * 0.05, 1.5 + i * 0.05],
            latency_ms=50.0 + i * 2.0,
            success=True,
        )
        requests.append(r)
    # Add one failure
    requests.append(
        RequestResult(success=False, error="timeout", latency_ms=999.0)
    )

    result = BenchmarkResult(
        backend="openai",
        base_url="http://localhost:8000",
        endpoint="/v1/completions",
        model="test-model",
        num_prompts=n + 1,
        request_rate=10.0,
        total_duration_s=5.0,
        requests=requests,
    )
    # Compute aggregated metrics
    from xpyd_bench.bench.runner import _compute_metrics

    _compute_metrics(result)
    return result


class TestExtendedPercentiles:
    """Verify that P50/P90/P95/P99 are populated."""

    def test_all_percentiles_set(self) -> None:
        result = _make_result()
        for prefix in ("ttft", "tpot", "itl", "e2el"):
            for stat in ("mean", "median", "p50", "p90", "p95", "p99"):
                val = getattr(result, f"{stat}_{prefix}_ms")
                assert val >= 0.0, f"{stat}_{prefix}_ms should be >= 0"

    def test_p50_equals_median(self) -> None:
        result = _make_result()
        for prefix in ("ttft", "tpot", "itl", "e2el"):
            assert abs(
                getattr(result, f"p50_{prefix}_ms")
                - getattr(result, f"median_{prefix}_ms")
            ) < 1e-6

    def test_ordering(self) -> None:
        result = _make_result(100)
        for prefix in ("ttft", "tpot", "itl", "e2el"):
            p50 = getattr(result, f"p50_{prefix}_ms")
            p90 = getattr(result, f"p90_{prefix}_ms")
            p95 = getattr(result, f"p95_{prefix}_ms")
            p99 = getattr(result, f"p99_{prefix}_ms")
            assert p50 <= p90 <= p95 <= p99


class TestTimeSeries:
    """Test time-series bucketing."""

    def test_basic(self) -> None:
        result = _make_result(20)
        ts = compute_time_series(result, window_s=1.0)
        assert len(ts) > 0
        for bucket in ts:
            assert "window_start_s" in bucket
            assert "request_throughput" in bucket

    def test_empty_result(self) -> None:
        result = BenchmarkResult()
        ts = compute_time_series(result)
        assert ts == []


class TestExportPerRequest:
    """Test per-request JSON export."""

    def test_export(self) -> None:
        result = _make_result()
        with tempfile.TemporaryDirectory() as td:
            p = export_per_request(result, Path(td) / "requests.json")
            assert p.exists()
            data = json.loads(p.read_text())
            assert isinstance(data, list)
            assert len(data) == len(result.requests)
            assert "ttft_ms" in data[0]
            assert "latency_ms" in data[0]


class TestJsonReport:
    """Test full JSON report export."""

    def test_export(self) -> None:
        result = _make_result()
        summary = {"completed": result.completed}
        with tempfile.TemporaryDirectory() as td:
            p = export_json_report(result, summary, Path(td) / "report.json")
            assert p.exists()
            report = json.loads(p.read_text())
            assert "summary" in report
            assert "time_series" in report
            assert isinstance(report["time_series"], list)


class TestTextReport:
    """Test human-readable text report."""

    def test_format(self) -> None:
        result = _make_result()
        text = format_text_report(result)
        assert "Benchmark Report" in text
        assert "TTFT" in text
        assert "P50" in text
        assert "P99" in text


class TestRichProgressReporter:
    """Basic smoke test for RichProgressReporter."""

    def test_lifecycle(self) -> None:
        reporter = RichProgressReporter(total=5)
        reporter.start()
        for _ in range(4):
            reporter.advance(success=True)
        reporter.advance(success=False)
        reporter.stop()
        assert reporter._completed == 4
        assert reporter._failed == 1

    def test_summary_table(self) -> None:
        result = _make_result()
        reporter = RichProgressReporter(total=1)
        # Should not raise
        reporter.print_summary_table(result)
