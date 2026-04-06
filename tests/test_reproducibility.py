"""Tests for Benchmark Reproducibility Score (M99)."""

from __future__ import annotations

import json

from xpyd_bench.reproducibility import (
    MetricReproducibility,
    classify_cv,
    compute_reproducibility,
    print_reproducibility_report,
)

# ---------------------------------------------------------------------------
# classify_cv
# ---------------------------------------------------------------------------


class TestClassifyCV:
    """Tests for CV classification thresholds."""

    def test_excellent_zero_cv(self) -> None:
        label, score = classify_cv(0.0)
        assert label == "excellent"
        assert score == 100.0

    def test_excellent_low_cv(self) -> None:
        label, score = classify_cv(2.5)
        assert label == "excellent"
        assert 90.0 <= score <= 100.0

    def test_excellent_boundary(self) -> None:
        label, score = classify_cv(4.99)
        assert label == "excellent"
        assert score >= 90.0

    def test_good_boundary(self) -> None:
        label, score = classify_cv(5.0)
        assert label == "good"
        assert 70.0 <= score <= 89.0

    def test_good_mid(self) -> None:
        label, score = classify_cv(7.5)
        assert label == "good"
        assert 70.0 <= score <= 89.0

    def test_fair_boundary(self) -> None:
        label, score = classify_cv(10.0)
        assert label == "fair"
        assert 50.0 <= score <= 69.0

    def test_fair_mid(self) -> None:
        label, score = classify_cv(12.5)
        assert label == "fair"
        assert 50.0 <= score <= 69.0

    def test_poor_boundary(self) -> None:
        label, score = classify_cv(15.0)
        assert label == "poor"
        assert 0.0 <= score <= 49.0

    def test_poor_high_cv(self) -> None:
        label, score = classify_cv(30.0)
        assert label == "poor"
        assert score == 0.0

    def test_poor_extreme_cv(self) -> None:
        label, score = classify_cv(100.0)
        assert label == "poor"
        assert score == 0.0


# ---------------------------------------------------------------------------
# compute_reproducibility
# ---------------------------------------------------------------------------


def _make_run(
    throughput: float = 10.0,
    ttft: float = 50.0,
    tpot: float = 5.0,
    latency: float = 200.0,
    output_tp: float = 100.0,
) -> dict:
    return {
        "request_throughput": throughput,
        "mean_ttft_ms": ttft,
        "mean_tpot_ms": tpot,
        "mean_e2el_ms": latency,
        "output_throughput": output_tp,
    }


class TestComputeReproducibility:
    """Tests for the core reproducibility computation."""

    def test_insufficient_data_single_run(self) -> None:
        result = compute_reproducibility([_make_run()])
        assert result.overall_classification == "insufficient_data"
        assert result.overall_score == 0.0
        assert len(result.recommendations) > 0

    def test_insufficient_data_empty(self) -> None:
        result = compute_reproducibility([])
        assert result.overall_classification == "insufficient_data"

    def test_perfect_reproducibility(self) -> None:
        """Identical runs should score ~100."""
        runs = [_make_run(10.0, 50.0, 5.0, 200.0, 100.0) for _ in range(5)]
        result = compute_reproducibility(runs)
        assert result.overall_score == 100.0
        assert result.overall_classification == "excellent"
        assert len(result.unstable_metrics) == 0

    def test_low_variance_excellent(self) -> None:
        """Small variance -> excellent score."""
        runs = [
            _make_run(10.0, 50.0, 5.0, 200.0, 100.0),
            _make_run(10.1, 50.5, 5.05, 201.0, 100.5),
            _make_run(9.9, 49.5, 4.95, 199.0, 99.5),
        ]
        result = compute_reproducibility(runs)
        assert result.overall_classification == "excellent"
        assert result.overall_score >= 90.0

    def test_high_variance_poor(self) -> None:
        """Large variance -> poor score."""
        runs = [
            _make_run(10.0, 50.0, 5.0, 200.0, 100.0),
            _make_run(20.0, 100.0, 10.0, 400.0, 200.0),
            _make_run(5.0, 25.0, 2.5, 100.0, 50.0),
        ]
        result = compute_reproducibility(runs)
        assert result.overall_classification == "poor"
        assert result.overall_score < 50.0
        assert len(result.unstable_metrics) > 0

    def test_mixed_stability(self) -> None:
        """Some metrics stable, some not."""
        runs = [
            _make_run(10.0, 50.0, 5.0, 200.0, 100.0),
            _make_run(10.05, 80.0, 5.02, 201.0, 100.2),
            _make_run(9.95, 30.0, 4.98, 199.0, 99.8),
        ]
        result = compute_reproducibility(runs)
        # TTFT is very unstable, others are stable
        assert any(m.name == "mean_ttft_ms" and m.cv_pct > 10 for m in result.metrics)
        assert "Mean TTFT (ms)" in result.unstable_metrics

    def test_num_runs_tracking(self) -> None:
        runs = [_make_run() for _ in range(3)]
        result = compute_reproducibility(runs, num_runs=5)
        assert result.num_runs == 5
        assert result.completed_runs == 3

    def test_missing_metrics_skipped(self) -> None:
        """Runs missing some metrics should still work."""
        runs = [
            {"request_throughput": 10.0},
            {"request_throughput": 10.5},
        ]
        result = compute_reproducibility(runs)
        # Should compute at least throughput
        assert len(result.metrics) >= 1
        assert any(m.name == "request_throughput" for m in result.metrics)


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


class TestRecommendations:
    """Tests for recommendation generation."""

    def test_few_runs_recommendation(self) -> None:
        runs = [_make_run(), _make_run()]
        result = compute_reproducibility(runs)
        assert any("more iterations" in r for r in result.recommendations)

    def test_poor_score_recommendations(self) -> None:
        runs = [
            _make_run(10.0, 50.0, 5.0, 200.0, 100.0),
            _make_run(20.0, 100.0, 10.0, 400.0, 200.0),
            _make_run(5.0, 25.0, 2.5, 100.0, 50.0),
            _make_run(15.0, 75.0, 7.5, 300.0, 150.0),
            _make_run(8.0, 40.0, 4.0, 160.0, 80.0),
            _make_run(25.0, 120.0, 12.0, 500.0, 250.0),
        ]
        result = compute_reproducibility(runs)
        recs_text = " ".join(result.recommendations)
        assert "warmup" in recs_text.lower() or "duration" in recs_text.lower()

    def test_excellent_no_action_needed(self) -> None:
        runs = [_make_run(10.0, 50.0, 5.0, 200.0, 100.0) for _ in range(10)]
        result = compute_reproducibility(runs)
        assert any("no action" in r.lower() for r in result.recommendations)

    def test_unstable_ttft_recommendation(self) -> None:
        """Highly variable TTFT triggers specific recommendation."""
        runs = [
            _make_run(10.0, 50.0, 5.0, 200.0, 100.0),
            _make_run(10.0, 150.0, 5.0, 200.0, 100.0),
            _make_run(10.0, 30.0, 5.0, 200.0, 100.0),
            _make_run(10.0, 200.0, 5.0, 200.0, 100.0),
            _make_run(10.0, 20.0, 5.0, 200.0, 100.0),
            _make_run(10.0, 180.0, 5.0, 200.0, 100.0),
        ]
        result = compute_reproducibility(runs)
        assert any("TTFT" in r for r in result.recommendations)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    """Tests for JSON serialization roundtrip."""

    def test_to_dict_roundtrip(self) -> None:
        runs = [_make_run(10.0), _make_run(10.5), _make_run(9.5)]
        result = compute_reproducibility(runs)
        d = result.to_dict()
        # Should be JSON-serializable
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["overall_score"] == d["overall_score"]
        assert parsed["overall_classification"] == d["overall_classification"]
        assert len(parsed["metrics"]) == len(d["metrics"])

    def test_metric_repro_to_dict(self) -> None:
        mr = MetricReproducibility(
            name="test",
            display_name="Test Metric",
            values=[1.0, 2.0, 3.0],
            mean=2.0,
            stddev=1.0,
            cv=0.5,
            cv_pct=50.0,
            classification="poor",
        )
        d = mr.to_dict()
        assert d["name"] == "test"
        assert d["cv_pct"] == 50.0
        assert d["classification"] == "poor"


# ---------------------------------------------------------------------------
# Print report (smoke test)
# ---------------------------------------------------------------------------


class TestPrintReport:
    """Smoke tests for print output."""

    def test_print_report_doesnt_crash(self, capsys) -> None:
        runs = [_make_run(10.0), _make_run(10.5), _make_run(9.5)]
        result = compute_reproducibility(runs)
        print_reproducibility_report(result)
        captured = capsys.readouterr()
        assert "Reproducibility Report" in captured.out
        assert "Overall Score" in captured.out

    def test_print_insufficient_data(self, capsys) -> None:
        result = compute_reproducibility([_make_run()])
        print_reproducibility_report(result)
        captured = capsys.readouterr()
        assert "Reproducibility Report" in captured.out


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCLIIntegration:
    """Tests for CLI argument parsing."""

    def test_reproducibility_check_arg_parsed(self) -> None:
        """Verify the --reproducibility-check argument is accepted."""
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--reproducibility-check", "5"])
        assert args.reproducibility_check == 5

    def test_reproducibility_check_default_none(self) -> None:
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args([])
        assert args.reproducibility_check is None


# ---------------------------------------------------------------------------
# Config key validation
# ---------------------------------------------------------------------------


class TestConfigKey:
    """Tests for YAML config key recognition."""

    def test_reproducibility_check_known(self) -> None:
        from xpyd_bench.config_cmd import _KNOWN_KEYS

        assert "reproducibility_check" in _KNOWN_KEYS
