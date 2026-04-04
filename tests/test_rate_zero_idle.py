"""Tests for rate=0 idle behavior in custom rate patterns (issue #95)."""

from __future__ import annotations

from xpyd_bench.bench.rate_patterns import generate_pattern_intervals


class TestCustomRateZeroIdle:
    """Verify rate=0 entries produce no requests."""

    def test_zero_rate_does_not_emit_request(self):
        """A zero-rate slot followed by active slot works correctly."""
        # Schedule: [0, 5] — first second idle, second second active
        cfg = {"type": "custom", "schedule": [0, 5]}
        intervals = generate_pattern_intervals(5, cfg, seed=42)
        assert len(intervals) == 5
        # First request should have ~1s idle carry added
        assert intervals[0] > 0.8

    def test_zero_rate_no_requests_in_idle_slots(self):
        """rate=0 slots should not contribute any requests."""
        # Schedule: [10, 0, 0, 10] — only seconds 0 and 3 produce requests
        cfg = {"type": "custom", "schedule": [10, 0, 0, 10]}
        intervals = generate_pattern_intervals(10, cfg, seed=42)
        assert len(intervals) == 10
        # All intervals should be positive (no zero-length gaps from idle)
        assert all(i > 0 for i in intervals)

    def test_idle_time_accumulated_to_next_request(self):
        """Idle seconds should add delay to the first request after idle."""
        # Schedule: [10, 0, 0, 10] — 2 idle seconds between bursts
        cfg = {"type": "custom", "schedule": [10, 0, 0, 10]}
        intervals = generate_pattern_intervals(20, cfg, seed=42)
        assert len(intervals) == 20
        # The first 10 requests come from rate=10 in second 0
        # Then 2 idle seconds, then next requests from rate=10 in second 3
        # Request 11 (index 10) should have ~2s extra delay from idle
        assert intervals[10] > 1.5  # Should include ~2s idle carry

    def test_single_zero_between_active(self):
        """Single idle second adds ~1s to next request delay."""
        cfg = {"type": "custom", "schedule": [5, 0, 5]}
        intervals = generate_pattern_intervals(10, cfg, seed=0)
        assert len(intervals) == 10
        # First 5 from rate=5, then 1 idle second, then 5 from rate=5
        assert intervals[5] > 0.8  # Should include ~1s idle carry

    def test_no_idle_carry_when_no_zeros(self):
        """Without zeros, intervals behave normally."""
        cfg = {"type": "custom", "schedule": [10]}
        intervals = generate_pattern_intervals(10, cfg, seed=0)
        assert len(intervals) == 10
        # Mean should be ~0.1s (1/10)
        mean_gap = sum(intervals) / len(intervals)
        assert 0.01 < mean_gap < 0.5
