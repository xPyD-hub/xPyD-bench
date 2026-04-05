"""Tests for rolling window metrics (M81)."""

from __future__ import annotations

from xpyd_bench.bench.rolling_metrics import compute_rolling_metrics


class TestComputeRollingMetrics:
    """Tests for compute_rolling_metrics."""

    def test_empty_data(self) -> None:
        assert compute_rolling_metrics([], [], 0.0) == {}

    def test_single_data_point(self) -> None:
        assert compute_rolling_metrics([100.0], [1.0], 0.0) == {}

    def test_basic_windows(self) -> None:
        # 10 requests spread over 20 seconds
        bench_start = 0.0
        start_times = [float(i * 2) for i in range(10)]  # 0, 2, 4, ..., 18
        latencies = [100.0 + i * 10 for i in range(10)]  # 100, 110, ..., 190

        result = compute_rolling_metrics(
            latencies, start_times, bench_start,
            window_seconds=10.0, step_seconds=5.0,
        )

        assert "windows" in result
        assert "config" in result
        assert len(result["windows"]) > 0

        # Check first window
        first = result["windows"][0]
        assert first["time_offset_s"] == 0.0
        assert first["count"] > 0
        assert "p50_latency_ms" in first
        assert "p90_latency_ms" in first
        assert "p99_latency_ms" in first

    def test_degradation_detection(self) -> None:
        # Simulate degradation: low latency first, high latency later
        bench_start = 0.0
        start_times = list(range(20))
        latencies = [50.0] * 10 + [200.0] * 10

        result = compute_rolling_metrics(
            latencies, start_times, bench_start,
            window_seconds=10.0, step_seconds=10.0,
        )

        assert result.get("degradation") is not None
        assert result["degradation"]["degraded"] is True
        assert result["degradation"]["change_pct"] > 10.0

    def test_no_degradation(self) -> None:
        # Stable latency
        bench_start = 0.0
        start_times = list(range(20))
        latencies = [100.0] * 20

        result = compute_rolling_metrics(
            latencies, start_times, bench_start,
            window_seconds=10.0, step_seconds=10.0,
        )

        assert result.get("degradation") is not None
        assert result["degradation"]["degraded"] is False
        assert result["degradation"]["change_pct"] == 0.0

    def test_custom_percentiles(self) -> None:
        bench_start = 0.0
        start_times = [float(i) for i in range(10)]
        latencies = [float(i * 10) for i in range(10)]

        result = compute_rolling_metrics(
            latencies, start_times, bench_start,
            window_seconds=20.0, step_seconds=20.0,
            percentiles=(50.0, 75.0, 95.0),
        )

        assert len(result["windows"]) >= 1
        w = result["windows"][0]
        assert "p50_latency_ms" in w
        assert "p75_latency_ms" in w
        assert "p95_latency_ms" in w

    def test_config_in_result(self) -> None:
        result = compute_rolling_metrics(
            [100.0, 200.0], [0.0, 1.0], 0.0,
            window_seconds=5.0, step_seconds=2.0,
        )
        cfg = result["config"]
        assert cfg["window_seconds"] == 5.0
        assert cfg["step_seconds"] == 2.0
        assert cfg["percentiles"] == [50.0, 90.0, 99.0]

    def test_none_start_times_filtered(self) -> None:
        # None start times should be filtered out
        result = compute_rolling_metrics(
            [100.0, 200.0, 300.0],
            [0.0, None, 2.0],  # type: ignore[list-item]
            0.0,
            window_seconds=10.0,
            step_seconds=10.0,
        )
        # Should still produce results from the 2 valid points
        assert result != {}
        assert result["windows"][0]["count"] == 2


class TestBenchmarkResultField:
    """Test that rolling_metrics field exists on BenchmarkResult."""

    def test_field_default(self) -> None:
        from xpyd_bench.bench.models import BenchmarkResult

        r = BenchmarkResult()
        assert r.rolling_metrics is None

    def test_field_set(self) -> None:
        from xpyd_bench.bench.models import BenchmarkResult

        r = BenchmarkResult(rolling_metrics={"windows": []})
        assert r.rolling_metrics == {"windows": []}


class TestConfigKey:
    """Test that rolling_metrics config keys are recognized."""

    def test_known_keys(self) -> None:
        from xpyd_bench.config_cmd import _KNOWN_KEYS

        assert "rolling_metrics" in _KNOWN_KEYS
        assert "rolling_window" in _KNOWN_KEYS
        assert "rolling_step" in _KNOWN_KEYS
