"""Tests for benchmark comparison and regression detection (M8)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from xpyd_bench.compare import (
    compare,
    format_comparison_table,
    load_result,
)


def _make_result(**overrides: float) -> dict:
    """Create a minimal benchmark result dict with default metrics."""
    base = {
        "mean_ttft_ms": 50.0,
        "p50_ttft_ms": 45.0,
        "p90_ttft_ms": 80.0,
        "p95_ttft_ms": 95.0,
        "p99_ttft_ms": 120.0,
        "mean_tpot_ms": 10.0,
        "p50_tpot_ms": 9.0,
        "p90_tpot_ms": 15.0,
        "p95_tpot_ms": 18.0,
        "p99_tpot_ms": 25.0,
        "mean_itl_ms": 8.0,
        "p50_itl_ms": 7.0,
        "p90_itl_ms": 12.0,
        "p95_itl_ms": 14.0,
        "p99_itl_ms": 20.0,
        "mean_e2el_ms": 500.0,
        "p50_e2el_ms": 480.0,
        "p90_e2el_ms": 700.0,
        "p95_e2el_ms": 800.0,
        "p99_e2el_ms": 1000.0,
        "request_throughput": 100.0,
        "output_throughput": 5000.0,
        "total_token_throughput": 8000.0,
    }
    base.update(overrides)
    return base


class TestCompareIdentical:
    """Compare identical results — all unchanged, exit 0."""

    def test_all_unchanged(self) -> None:
        data = _make_result()
        result = compare(data, data)
        assert not result.has_regression
        for m in result.metrics:
            assert m.direction == "unchanged"
            assert m.delta == 0.0
            assert not m.regressed


class TestCompareRegression:
    """Compare with regressions — correct detection."""

    def test_latency_regression(self) -> None:
        baseline = _make_result()
        # 20% worse TTFT
        candidate = _make_result(mean_ttft_ms=60.0)
        result = compare(baseline, candidate, threshold_pct=5.0)
        assert result.has_regression

        ttft = next(m for m in result.metrics if m.name == "mean_ttft_ms")
        assert ttft.direction == "regressed"
        assert ttft.regressed
        assert ttft.pct_change == 20.0

    def test_throughput_regression(self) -> None:
        baseline = _make_result()
        # 10% worse throughput
        candidate = _make_result(request_throughput=90.0)
        result = compare(baseline, candidate, threshold_pct=5.0)
        assert result.has_regression

        rtp = next(m for m in result.metrics if m.name == "request_throughput")
        assert rtp.direction == "regressed"
        assert rtp.regressed

    def test_regression_exit_code(self) -> None:
        """CLI exits with code 1 on regression."""
        from xpyd_bench.cli import compare_main

        baseline = _make_result()
        candidate = _make_result(mean_ttft_ms=100.0)

        with tempfile.TemporaryDirectory() as td:
            bp = Path(td) / "base.json"
            cp = Path(td) / "cand.json"
            bp.write_text(json.dumps(baseline))
            cp.write_text(json.dumps(candidate))

            with pytest.raises(SystemExit) as exc_info:
                compare_main([str(bp), str(cp)])
            assert exc_info.value.code == 1


class TestCompareImprovement:
    """Compare with improvements — correct detection, exit 0."""

    def test_latency_improvement(self) -> None:
        baseline = _make_result()
        # 20% better TTFT (lower)
        candidate = _make_result(mean_ttft_ms=40.0)
        result = compare(baseline, candidate, threshold_pct=5.0)
        assert not result.has_regression

        ttft = next(m for m in result.metrics if m.name == "mean_ttft_ms")
        assert ttft.direction == "improved"
        assert not ttft.regressed

    def test_throughput_improvement(self) -> None:
        baseline = _make_result()
        candidate = _make_result(request_throughput=120.0)
        result = compare(baseline, candidate, threshold_pct=5.0)
        assert not result.has_regression

        rtp = next(m for m in result.metrics if m.name == "request_throughput")
        assert rtp.direction == "improved"


class TestCustomThreshold:
    """Custom threshold works."""

    def test_below_threshold_no_regression(self) -> None:
        baseline = _make_result()
        # 3% regression — below 5% threshold
        candidate = _make_result(mean_ttft_ms=51.5)
        result = compare(baseline, candidate, threshold_pct=5.0)
        assert not result.has_regression

    def test_above_threshold_triggers_regression(self) -> None:
        baseline = _make_result()
        # 3% regression — above 2% threshold
        candidate = _make_result(mean_ttft_ms=51.5)
        result = compare(baseline, candidate, threshold_pct=2.0)
        assert result.has_regression

    def test_zero_threshold(self) -> None:
        baseline = _make_result()
        candidate = _make_result(mean_ttft_ms=50.01)
        result = compare(baseline, candidate, threshold_pct=0.0)
        assert result.has_regression


class TestJsonDiffExport:
    """JSON diff export matches expected structure."""

    def test_export_structure(self) -> None:
        baseline = _make_result()
        candidate = _make_result(mean_ttft_ms=60.0)
        result = compare(baseline, candidate)
        d = result.to_dict()

        assert "threshold_pct" in d
        assert "has_regression" in d
        assert "metrics" in d
        assert isinstance(d["metrics"], list)
        assert len(d["metrics"]) > 0

        m = d["metrics"][0]
        assert set(m.keys()) == {
            "name",
            "baseline",
            "candidate",
            "delta",
            "pct_change",
            "direction",
            "regressed",
        }

    def test_json_serializable(self) -> None:
        baseline = _make_result()
        candidate = _make_result(output_throughput=4000.0)
        result = compare(baseline, candidate)
        # Must not raise
        serialized = json.dumps(result.to_dict())
        parsed = json.loads(serialized)
        assert parsed["has_regression"] is True


class TestCliIntegration:
    """CLI integration test with real files."""

    def test_no_regression_exit_0(self) -> None:
        from xpyd_bench.cli import compare_main

        data = _make_result()
        with tempfile.TemporaryDirectory() as td:
            bp = Path(td) / "base.json"
            cp = Path(td) / "cand.json"
            bp.write_text(json.dumps(data))
            cp.write_text(json.dumps(data))

            # Should NOT raise SystemExit
            compare_main([str(bp), str(cp)])

    def test_json_output_flag(self) -> None:
        from xpyd_bench.cli import compare_main

        baseline = _make_result()
        candidate = _make_result(mean_ttft_ms=100.0)

        with tempfile.TemporaryDirectory() as td:
            bp = Path(td) / "base.json"
            cp = Path(td) / "cand.json"
            out = Path(td) / "diff.json"
            bp.write_text(json.dumps(baseline))
            cp.write_text(json.dumps(candidate))

            with pytest.raises(SystemExit):
                compare_main([str(bp), str(cp), "--output", str(out)])

            assert out.exists()
            diff = json.loads(out.read_text())
            assert diff["has_regression"] is True

    def test_custom_threshold_flag(self) -> None:
        from xpyd_bench.cli import compare_main

        baseline = _make_result()
        # 3% regression
        candidate = _make_result(mean_ttft_ms=51.5)

        with tempfile.TemporaryDirectory() as td:
            bp = Path(td) / "base.json"
            cp = Path(td) / "cand.json"
            bp.write_text(json.dumps(baseline))
            cp.write_text(json.dumps(candidate))

            # Default 5% threshold — no regression
            compare_main([str(bp), str(cp)])

            # 2% threshold — regression detected
            with pytest.raises(SystemExit) as exc_info:
                compare_main([str(bp), str(cp), "--threshold", "2.0"])
            assert exc_info.value.code == 1


class TestSummaryWrapper:
    """Results wrapped in a 'summary' key should still work."""

    def test_summary_key(self) -> None:
        baseline = {"summary": _make_result()}
        candidate = {"summary": _make_result(mean_ttft_ms=60.0)}
        result = compare(baseline, candidate)
        assert result.has_regression


class TestFormatTable:
    """format_comparison_table produces readable output."""

    def test_table_contains_metrics(self) -> None:
        baseline = _make_result()
        candidate = _make_result(mean_ttft_ms=60.0)
        result = compare(baseline, candidate)
        table = format_comparison_table(result)
        assert "mean_ttft_ms" in table
        assert "REGRESS" in table

    def test_table_no_regression(self) -> None:
        data = _make_result()
        result = compare(data, data)
        table = format_comparison_table(result)
        assert "No regressions" in table


class TestLoadResult:
    """load_result reads JSON files correctly."""

    def test_load(self) -> None:
        data = _make_result()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            loaded = load_result(f.name)
        assert loaded["mean_ttft_ms"] == 50.0


class TestEdgeCases:
    """Edge cases for comparison logic."""

    def test_zero_baseline(self) -> None:
        baseline = _make_result(mean_ttft_ms=0.0)
        candidate = _make_result(mean_ttft_ms=10.0)
        result = compare(baseline, candidate)
        ttft = next(m for m in result.metrics if m.name == "mean_ttft_ms")
        assert ttft.pct_change == float("inf")

    def test_missing_metrics_skipped(self) -> None:
        baseline = {"mean_ttft_ms": 50.0}
        candidate = {"mean_ttft_ms": 60.0, "request_throughput": 100.0}
        result = compare(baseline, candidate)
        # Only mean_ttft_ms should be compared (common key)
        assert len(result.metrics) == 1
        assert result.metrics[0].name == "mean_ttft_ms"
