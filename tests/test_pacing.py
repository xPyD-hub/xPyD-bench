"""Tests for request pacing accuracy report (M93)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xpyd_bench.bench.pacing import (
    _detect_bursts,
    _detect_drift,
    compute_pacing_report,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeResult:
    start_time: float | None = None
    success: bool = True
    queue_time_ms: float | None = None


def _make_requests(
    n: int, interval_s: float = 0.1, start: float = 1000.0, jitter: float = 0.0
) -> list[_FakeResult]:
    """Create fake request results with controlled timing."""
    results = []
    t = start
    for _ in range(n):
        results.append(_FakeResult(start_time=t))
        t += interval_s + jitter
    return results


# ---------------------------------------------------------------------------
# compute_pacing_report
# ---------------------------------------------------------------------------


class TestComputePacingReport:
    """Test main pacing report computation."""

    def test_returns_none_for_empty(self):
        assert compute_pacing_report([]) is None

    def test_returns_none_for_single_request(self):
        reqs = [_FakeResult(start_time=1.0)]
        assert compute_pacing_report(reqs) is None

    def test_returns_none_when_no_start_times(self):
        reqs = [_FakeResult(start_time=None), _FakeResult(start_time=None)]
        assert compute_pacing_report(reqs) is None

    def test_basic_report_structure(self):
        reqs = _make_requests(10, interval_s=0.1)
        report = compute_pacing_report(reqs, target_interval_s=0.1)
        assert report is not None
        assert report["num_requests"] == 10
        assert report["num_intervals"] == 9
        assert "actual_interval_ms" in report
        assert "pacing_error_ms" in report
        assert "pacing_accuracy_pct" in report
        assert "drift" in report
        assert "bursts" in report

    def test_perfect_pacing(self):
        """Perfectly spaced requests should have ~0 error."""
        reqs = _make_requests(20, interval_s=0.1)
        report = compute_pacing_report(reqs, target_interval_s=0.1)
        assert report is not None
        ai = report["actual_interval_ms"]
        assert abs(ai["mean"] - 100.0) < 0.01
        pe = report["pacing_error_ms"]
        assert pe["mean"] < 0.01
        assert report["pacing_accuracy_pct"] > 99.9

    def test_without_target_interval(self):
        """Report without target should omit error metrics."""
        reqs = _make_requests(10, interval_s=0.05)
        report = compute_pacing_report(reqs, target_interval_s=None)
        assert report is not None
        assert "target_interval_ms" not in report
        assert "pacing_error_ms" not in report
        assert "pacing_accuracy_pct" not in report
        # But still has interval stats, drift, bursts
        assert "actual_interval_ms" in report
        assert "drift" in report
        assert "bursts" in report

    def test_high_jitter_lowers_accuracy(self):
        """Requests with timing jitter should show lower accuracy."""
        np.random.seed(42)
        reqs = []
        t = 0.0
        for _ in range(50):
            reqs.append(_FakeResult(start_time=t))
            t += 0.1 + np.random.uniform(-0.03, 0.03)
        report = compute_pacing_report(reqs, target_interval_s=0.1)
        assert report is not None
        assert report["pacing_accuracy_pct"] < 100.0
        assert report["pacing_error_ms"]["mean"] > 0.0

    def test_unsorted_requests(self):
        """Requests should be sorted by start_time internally."""
        reqs = [
            _FakeResult(start_time=1.3),
            _FakeResult(start_time=1.0),
            _FakeResult(start_time=1.2),
            _FakeResult(start_time=1.1),
        ]
        report = compute_pacing_report(reqs, target_interval_s=0.1)
        assert report is not None
        assert report["num_requests"] == 4
        assert abs(report["actual_interval_ms"]["mean"] - 100.0) < 0.1

    def test_mixed_none_start_times(self):
        """Only requests with start_time should be counted."""
        reqs = [
            _FakeResult(start_time=1.0),
            _FakeResult(start_time=None),
            _FakeResult(start_time=1.1),
            _FakeResult(start_time=None),
            _FakeResult(start_time=1.2),
        ]
        report = compute_pacing_report(reqs)
        assert report is not None
        assert report["num_requests"] == 3


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------


class TestDriftDetection:
    def test_no_drift_uniform(self):
        intervals = np.array([100.0] * 20)
        result = _detect_drift(intervals, 0.1)
        assert result["detected"] is False

    def test_drift_detected_increasing(self):
        """Linearly increasing intervals should be detected as drift."""
        intervals = np.array([100.0 + i * 5 for i in range(20)])
        result = _detect_drift(intervals, 0.1)
        assert result["detected"] is True
        assert result["slope_ms_per_request"] > 0

    def test_drift_detected_decreasing(self):
        intervals = np.array([200.0 - i * 5 for i in range(20)])
        result = _detect_drift(intervals, 0.1)
        assert result["detected"] is True
        assert result["slope_ms_per_request"] < 0

    def test_too_few_intervals(self):
        intervals = np.array([100.0, 100.0, 100.0])
        result = _detect_drift(intervals, 0.1)
        assert result["detected"] is False

    def test_drift_without_target(self):
        intervals = np.array([100.0 + i * 5 for i in range(20)])
        result = _detect_drift(intervals, None)
        assert result["detected"] is True


# ---------------------------------------------------------------------------
# Burst detection
# ---------------------------------------------------------------------------


class TestBurstDetection:
    def test_no_bursts(self):
        intervals = np.array([100.0] * 10)
        result = _detect_bursts(intervals, 0.1)
        assert result["count"] == 0
        assert result["burst_intervals"] == 0

    def test_bursts_detected(self):
        """Intervals much shorter than target count as bursts."""
        intervals = np.array([100.0, 100.0, 5.0, 3.0, 100.0, 100.0, 2.0, 100.0])
        result = _detect_bursts(intervals, 0.1)
        assert result["count"] >= 2  # two burst groups
        assert result["burst_intervals"] == 3

    def test_single_burst_group(self):
        intervals = np.array([100.0, 1.0, 1.0, 1.0, 100.0])
        result = _detect_bursts(intervals, 0.1)
        assert result["count"] == 1
        assert result["burst_intervals"] == 3

    def test_burst_without_target(self):
        """Should use median as reference when no target."""
        intervals = np.array([100.0] * 8 + [1.0, 1.0])
        result = _detect_bursts(intervals, None)
        assert result["burst_intervals"] >= 2

    def test_empty_intervals(self):
        intervals = np.array([])
        result = _detect_bursts(intervals, 0.1)
        assert result["count"] == 0


# ---------------------------------------------------------------------------
# Integration: CLI flag and YAML config key
# ---------------------------------------------------------------------------


class TestCLIIntegration:
    def test_cli_flag_parsed(self):
        """--pacing-report should set pacing_report=True."""
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--pacing-report"])
        assert args.pacing_report is True

    def test_cli_flag_default(self):
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args([])
        assert args.pacing_report is False


class TestConfigKey:
    def test_pacing_report_in_known_keys(self):
        from xpyd_bench.config_cmd import _KNOWN_KEYS

        assert "pacing_report" in _KNOWN_KEYS


class TestBenchmarkResultField:
    def test_pacing_report_field_exists(self):
        from xpyd_bench.bench.models import BenchmarkResult

        r = BenchmarkResult()
        assert r.pacing_report is None
        r.pacing_report = {"test": True}
        assert r.pacing_report == {"test": True}
