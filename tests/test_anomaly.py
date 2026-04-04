"""Tests for request latency anomaly detection (M43)."""

from __future__ import annotations

from xpyd_bench.bench.anomaly import detect_anomalies


class TestDetectAnomalies:
    """Tests for the detect_anomalies function."""

    def test_detects_obvious_outliers(self) -> None:
        """Requests with much higher latency are flagged."""
        latencies = [10.0] * 50 + [100.0, 200.0]
        result = detect_anomalies(latencies, multiplier=1.5)
        assert result is not None
        assert result.count == 2
        assert result.anomalies[0].latency_ms == 200.0
        assert result.anomalies[1].latency_ms == 100.0

    def test_no_anomalies_in_uniform_data(self) -> None:
        """Uniform latencies should have zero anomalies."""
        latencies = [10.0] * 100
        result = detect_anomalies(latencies, multiplier=1.5)
        assert result is not None
        assert result.count == 0

    def test_disabled_when_multiplier_zero(self) -> None:
        """Multiplier of 0 disables detection."""
        latencies = [10.0] * 50 + [1000.0]
        result = detect_anomalies(latencies, multiplier=0)
        assert result is None

    def test_disabled_when_multiplier_negative(self) -> None:
        """Negative multiplier disables detection."""
        result = detect_anomalies([10.0, 20.0, 30.0, 40.0, 500.0], multiplier=-1.0)
        assert result is None

    def test_insufficient_data(self) -> None:
        """Less than 4 data points returns None."""
        result = detect_anomalies([10.0, 20.0, 30.0], multiplier=1.5)
        assert result is None

    def test_custom_threshold(self) -> None:
        """Higher multiplier means fewer anomalies."""
        latencies = [10.0] * 50 + [50.0]
        # With low multiplier, 50 may be an anomaly
        r_low = detect_anomalies(latencies, multiplier=0.5)
        # With high multiplier, 50 may not be
        r_high = detect_anomalies(latencies, multiplier=10.0)
        assert r_low is not None
        assert r_high is not None
        assert r_low.count >= r_high.count

    def test_to_dict_structure(self) -> None:
        """to_dict returns expected keys."""
        latencies = [10.0] * 50 + [200.0]
        result = detect_anomalies(latencies, multiplier=1.5)
        assert result is not None
        d = result.to_dict()
        assert "count" in d
        assert "q1_ms" in d
        assert "q3_ms" in d
        assert "iqr_ms" in d
        assert "threshold_ms" in d
        assert "multiplier" in d
        assert "flagged_requests" in d
        assert isinstance(d["flagged_requests"], list)

    def test_anomalies_sorted_by_latency_desc(self) -> None:
        """Anomalies should be sorted worst-first."""
        latencies = [10.0] * 50 + [150.0, 300.0, 200.0]
        result = detect_anomalies(latencies, multiplier=1.5)
        assert result is not None
        assert result.count >= 2
        for i in range(len(result.anomalies) - 1):
            assert result.anomalies[i].latency_ms >= result.anomalies[i + 1].latency_ms

    def test_deviation_factor_positive(self) -> None:
        """Deviation factors should be positive for anomalies."""
        latencies = [10.0] * 50 + [500.0]
        result = detect_anomalies(latencies, multiplier=1.5)
        assert result is not None
        for a in result.anomalies:
            assert a.deviation_factor > 0
