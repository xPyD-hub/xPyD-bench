"""Tests for M25: Result Aggregation Across Multiple Runs."""

from __future__ import annotations

import json

import pytest

from xpyd_bench.aggregate import (
    _compute_stats,
    _detect_outliers,
    aggregate_main,
    aggregate_results,
)


def _make_result(
    throughput: float = 10.0,
    mean_ttft: float = 50.0,
    p99_ttft: float = 100.0,
    mean_tpot: float = 20.0,
) -> dict:
    """Create a minimal BenchmarkResult-like dict."""
    return {
        "request_throughput": throughput,
        "output_throughput": throughput * 10,
        "total_token_throughput": throughput * 15,
        "mean_ttft_ms": mean_ttft,
        "p50_ttft_ms": mean_ttft * 0.9,
        "p90_ttft_ms": mean_ttft * 1.5,
        "p95_ttft_ms": mean_ttft * 1.8,
        "p99_ttft_ms": p99_ttft,
        "mean_tpot_ms": mean_tpot,
        "p50_tpot_ms": mean_tpot * 0.9,
        "p90_tpot_ms": mean_tpot * 1.5,
        "p95_tpot_ms": mean_tpot * 1.8,
        "p99_tpot_ms": mean_tpot * 2.0,
        "mean_itl_ms": mean_tpot * 0.8,
        "p50_itl_ms": mean_tpot * 0.7,
        "p90_itl_ms": mean_tpot * 1.2,
        "p95_itl_ms": mean_tpot * 1.5,
        "p99_itl_ms": mean_tpot * 1.8,
        "mean_e2el_ms": 200.0,
        "p50_e2el_ms": 180.0,
        "p90_e2el_ms": 250.0,
        "p95_e2el_ms": 280.0,
        "p99_e2el_ms": 300.0,
    }


class TestComputeStats:
    """Test statistical computation."""

    def test_basic_stats(self):
        stats = _compute_stats([10.0, 20.0, 30.0])
        assert abs(stats.mean - 20.0) < 0.01
        assert stats.min == 10.0
        assert stats.max == 30.0
        assert stats.stddev > 0

    def test_single_value(self):
        stats = _compute_stats([5.0])
        assert stats.mean == 5.0
        assert stats.stddev == 0.0
        assert stats.cv == 0.0

    def test_identical_values(self):
        stats = _compute_stats([7.0, 7.0, 7.0])
        assert stats.cv == 0.0

    def test_cv_calculation(self):
        stats = _compute_stats([100.0, 100.0])
        assert stats.cv == 0.0


class TestDetectOutliers:
    """Test outlier detection."""

    def test_no_outliers(self):
        assert _detect_outliers([10.0, 10.1, 9.9, 10.0]) == []

    def test_outlier_detected(self):
        values = [10.0, 10.0, 10.0, 10.0, 10.0, 50.0]
        outliers = _detect_outliers(values)
        assert 5 in outliers

    def test_too_few_values(self):
        assert _detect_outliers([1.0, 2.0]) == []

    def test_all_same(self):
        assert _detect_outliers([5.0, 5.0, 5.0, 5.0]) == []


class TestAggregateResults:
    """Test the main aggregation logic."""

    def test_two_runs(self):
        r1 = _make_result(throughput=10.0, mean_ttft=50.0)
        r2 = _make_result(throughput=12.0, mean_ttft=60.0)
        agg = aggregate_results([r1, r2])
        assert agg.num_runs == 2
        assert "request_throughput" in agg.metrics
        assert abs(agg.metrics["request_throughput"].mean - 11.0) < 0.01

    def test_itl_metrics_included(self):
        """ITL percentile metrics must be aggregated (issue #81)."""
        r1 = _make_result(mean_tpot=20.0)
        r2 = _make_result(mean_tpot=25.0)
        agg = aggregate_results([r1, r2])
        for key in ("mean_itl_ms", "p50_itl_ms", "p90_itl_ms", "p95_itl_ms", "p99_itl_ms"):
            assert key in agg.metrics, f"{key} missing from aggregated metrics"

    def test_to_dict(self):
        r1 = _make_result()
        r2 = _make_result()
        agg = aggregate_results([r1, r2])
        d = agg.to_dict()
        assert d["num_runs"] == 2
        assert isinstance(d["metrics"], dict)
        for v in d["metrics"].values():
            assert "mean" in v
            assert "stddev" in v

    def test_outlier_in_results(self):
        runs = [_make_result(throughput=10.0) for _ in range(5)]
        runs.append(_make_result(throughput=100.0))  # outlier
        agg = aggregate_results(runs)
        assert 5 in agg.outlier_runs


class TestAggregateCLI:
    """Test CLI integration."""

    def test_aggregate_two_files(self, tmp_path):
        f1 = tmp_path / "r1.json"
        f2 = tmp_path / "r2.json"
        f1.write_text(json.dumps(_make_result(throughput=10.0)))
        f2.write_text(json.dumps(_make_result(throughput=12.0)))
        aggregate_main([str(f1), str(f2)])

    def test_aggregate_with_output(self, tmp_path):
        f1 = tmp_path / "r1.json"
        f2 = tmp_path / "r2.json"
        out = tmp_path / "agg.json"
        f1.write_text(json.dumps(_make_result()))
        f2.write_text(json.dumps(_make_result()))
        aggregate_main([str(f1), str(f2), "--output", str(out)])
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["num_runs"] == 2

    def test_min_runs_error(self, tmp_path):
        f1 = tmp_path / "r1.json"
        f1.write_text(json.dumps(_make_result()))
        with pytest.raises(SystemExit) as exc_info:
            aggregate_main([str(f1)])
        assert exc_info.value.code == 1

    def test_min_runs_override(self, tmp_path):
        f1 = tmp_path / "r1.json"
        f1.write_text(json.dumps(_make_result()))
        aggregate_main([str(f1), "--min-runs", "1"])

    def test_file_not_found(self, tmp_path):
        with pytest.raises(SystemExit) as exc_info:
            aggregate_main([str(tmp_path / "nope.json"), str(tmp_path / "nope2.json")])
        assert exc_info.value.code == 1

    def test_help(self):
        with pytest.raises(SystemExit) as exc_info:
            aggregate_main(["--help"])
        assert exc_info.value.code == 0


class TestUnifiedCLIAggregate:
    """Test aggregate routing through unified CLI."""

    def test_aggregate_subcommand_help(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["xpyd-bench", "aggregate", "--help"])
        from xpyd_bench.main import main

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
