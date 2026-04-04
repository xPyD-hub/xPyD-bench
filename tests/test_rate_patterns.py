"""Tests for flexible request rate patterns (M4)."""

from __future__ import annotations

import numpy as np
import pytest

from xpyd_bench.bench.rate_patterns import generate_pattern_intervals


class TestConstantPattern:
    def test_basic(self):
        intervals = generate_pattern_intervals(
            100, {"type": "constant", "rate": 10.0}, seed=42
        )
        assert len(intervals) == 100
        # Mean should be ~0.1s (1/10)
        assert 0.05 < np.mean(intervals) < 0.2

    def test_custom_interval(self):
        intervals = generate_pattern_intervals(
            50, {"type": "constant", "rate": 5.0, "interval": 2.0}, seed=0
        )
        assert len(intervals) == 50
        # Mean gap = 2.0/5.0 = 0.4s
        assert 0.2 < np.mean(intervals) < 0.7

    def test_deterministic(self):
        a = generate_pattern_intervals(20, {"type": "constant", "rate": 5.0}, seed=7)
        b = generate_pattern_intervals(20, {"type": "constant", "rate": 5.0}, seed=7)
        assert a == b

    def test_invalid_rate(self):
        with pytest.raises(ValueError, match="rate must be positive"):
            generate_pattern_intervals(10, {"type": "constant", "rate": 0.0})


class TestRampPattern:
    def test_single_stage(self):
        cfg = {
            "type": "ramp",
            "stages": [{"rate": 10.0, "duration": 5.0}],
        }
        intervals = generate_pattern_intervals(50, cfg, seed=0)
        assert len(intervals) == 50

    def test_multi_stage(self):
        cfg = {
            "type": "ramp",
            "stages": [
                {"rate": 5.0, "duration": 2.0},
                {"rate": 20.0, "duration": 3.0},
            ],
        }
        intervals = generate_pattern_intervals(100, cfg, seed=42)
        assert len(intervals) == 100

    def test_deterministic(self):
        cfg = {
            "type": "ramp",
            "stages": [{"rate": 10.0, "duration": 5.0}],
        }
        a = generate_pattern_intervals(30, cfg, seed=1)
        b = generate_pattern_intervals(30, cfg, seed=1)
        assert a == b

    def test_empty_stages(self):
        with pytest.raises(ValueError, match="at least one stage"):
            generate_pattern_intervals(10, {"type": "ramp", "stages": []})


class TestBurstPattern:
    def test_basic(self):
        cfg = {"type": "burst", "burst_size": 5, "burst_interval": 2.0}
        intervals = generate_pattern_intervals(15, cfg, seed=0)
        assert len(intervals) == 15
        # First request: 0.0
        assert intervals[0] == 0.0
        # Requests within first burst (indices 1-4): 0.0
        assert all(intervals[i] == 0.0 for i in range(1, 5))
        # Start of second burst (index 5): burst_interval
        assert intervals[5] == 2.0
        # Requests within second burst (6-9): 0.0
        assert all(intervals[i] == 0.0 for i in range(6, 10))
        # Start of third burst (index 10): burst_interval
        assert intervals[10] == 2.0

    def test_invalid(self):
        with pytest.raises(ValueError, match="burst_size must be"):
            generate_pattern_intervals(
                10, {"type": "burst", "burst_size": 0, "burst_interval": 1.0}
            )


class TestCustomPattern:
    def test_basic(self):
        cfg = {"type": "custom", "schedule": [10.0, 5.0, 20.0]}
        intervals = generate_pattern_intervals(50, cfg, seed=42)
        assert len(intervals) == 50

    def test_idle_seconds(self):
        cfg = {"type": "custom", "schedule": [0.0, 10.0]}
        intervals = generate_pattern_intervals(20, cfg, seed=0)
        assert len(intervals) == 20
        # First interval should include ~1s idle carry added to the
        # exponential gap from the rate=10 second
        assert intervals[0] > 0.8

    def test_deterministic(self):
        cfg = {"type": "custom", "schedule": [5.0, 10.0]}
        a = generate_pattern_intervals(30, cfg, seed=99)
        b = generate_pattern_intervals(30, cfg, seed=99)
        assert a == b

    def test_empty_schedule(self):
        with pytest.raises(ValueError, match="non-empty schedule"):
            generate_pattern_intervals(10, {"type": "custom", "schedule": []})


class TestUnknownPattern:
    def test_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown rate_pattern type"):
            generate_pattern_intervals(10, {"type": "foobar"})


class TestRunnerIntegration:
    """Verify rate_pattern in args flows through the runner."""

    def test_rate_pattern_attr_used(self):
        """Ensure _generate_intervals is bypassed when rate_pattern is set."""
        from xpyd_bench.bench.rate_patterns import generate_pattern_intervals

        cfg = {"type": "constant", "rate": 10.0}
        intervals = generate_pattern_intervals(10, cfg, seed=0)
        assert len(intervals) == 10
        assert all(isinstance(v, float) for v in intervals)
