"""Tests for M86: Adaptive Timeout (Auto-Tuning)."""

from __future__ import annotations

import pytest

from xpyd_bench.bench.adaptive_timeout import (
    MIN_SAMPLES_FOR_ADAPTATION,
    AdaptiveTimeout,
)
from xpyd_bench.bench.models import RequestResult


class TestAdaptiveTimeout:
    """Unit tests for AdaptiveTimeout class."""

    def test_initial_timeout_returned_before_enough_samples(self):
        at = AdaptiveTimeout(initial_timeout=60.0, multiplier=3.0)
        assert at.get_timeout() == 60.0
        # Record fewer than MIN_SAMPLES_FOR_ADAPTATION
        for i in range(MIN_SAMPLES_FOR_ADAPTATION - 1):
            at.record(1.0)
        assert at.get_timeout() == 60.0

    def test_adaptive_after_enough_samples(self):
        at = AdaptiveTimeout(initial_timeout=300.0, multiplier=3.0)
        # Record consistent 1s latencies
        for _ in range(20):
            at.record(1.0)
        timeout = at.get_timeout()
        # P99 of all-1.0 values = 1.0, so adaptive = 3.0
        assert timeout == pytest.approx(3.0, abs=0.1)

    def test_multiplier_effect(self):
        at_low = AdaptiveTimeout(initial_timeout=300.0, multiplier=2.0)
        at_high = AdaptiveTimeout(initial_timeout=300.0, multiplier=5.0)
        for _ in range(20):
            at_low.record(1.0)
            at_high.record(1.0)
        assert at_low.get_timeout() < at_high.get_timeout()

    def test_clamped_to_initial(self):
        """Adaptive timeout should never exceed initial_timeout."""
        at = AdaptiveTimeout(initial_timeout=10.0, multiplier=3.0)
        # Record very high latencies
        for _ in range(20):
            at.record(100.0)
        assert at.get_timeout() == 10.0

    def test_minimum_timeout(self):
        """Adaptive timeout should not go below 1.0s."""
        at = AdaptiveTimeout(initial_timeout=300.0, multiplier=3.0)
        # Record very small latencies
        for _ in range(20):
            at.record(0.001)
        timeout = at.get_timeout()
        assert timeout >= 1.0

    def test_sample_count(self):
        at = AdaptiveTimeout(initial_timeout=60.0)
        assert at.sample_count() == 0
        at.record(1.0)
        at.record(2.0)
        assert at.sample_count() == 2

    def test_rolling_window(self):
        """Older samples should be dropped when window is full."""
        at = AdaptiveTimeout(initial_timeout=300.0, multiplier=3.0, window_size=10)
        # Fill with high latencies
        for _ in range(10):
            at.record(50.0)
        high_timeout = at.get_timeout()
        # Replace all with low latencies
        for _ in range(10):
            at.record(0.5)
        low_timeout = at.get_timeout()
        assert low_timeout < high_timeout

    def test_invalid_initial_timeout(self):
        with pytest.raises(ValueError, match="initial_timeout must be positive"):
            AdaptiveTimeout(initial_timeout=0)

    def test_invalid_multiplier(self):
        with pytest.raises(ValueError, match="multiplier must be positive"):
            AdaptiveTimeout(initial_timeout=60.0, multiplier=0)

    def test_properties(self):
        at = AdaptiveTimeout(initial_timeout=42.0, multiplier=2.5)
        assert at.initial_timeout == 42.0
        assert at.multiplier == 2.5


class TestRequestResultEffectiveTimeout:
    """Test effective_timeout field on RequestResult."""

    def test_default_none(self):
        r = RequestResult()
        assert r.effective_timeout is None

    def test_set_value(self):
        r = RequestResult()
        r.effective_timeout = 15.5
        assert r.effective_timeout == 15.5


class TestConfigKeys:
    """Test that adaptive timeout config keys are registered."""

    def test_known_keys(self):
        from xpyd_bench.config_cmd import _KNOWN_KEYS

        assert "adaptive_timeout" in _KNOWN_KEYS
        assert "adaptive_timeout_multiplier" in _KNOWN_KEYS


class TestCLIArgs:
    """Test CLI argument parsing for adaptive timeout."""

    def _make_parser(self):
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        parser.add_argument("base_url", nargs="?", default="http://localhost:8000")
        _add_vllm_compat_args(parser)
        return parser

    def test_adaptive_timeout_flag(self):
        parser = self._make_parser()
        args = parser.parse_args(["--adaptive-timeout"])
        assert args.adaptive_timeout is True

    def test_adaptive_timeout_multiplier(self):
        parser = self._make_parser()
        args = parser.parse_args([
            "--adaptive-timeout",
            "--adaptive-timeout-multiplier", "5.0",
        ])
        assert args.adaptive_timeout_multiplier == 5.0

    def test_default_multiplier(self):
        parser = self._make_parser()
        args = parser.parse_args([])
        assert args.adaptive_timeout is False
        assert args.adaptive_timeout_multiplier == 3.0
