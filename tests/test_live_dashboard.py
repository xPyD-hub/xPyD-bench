"""Tests for M29: Live Progress Dashboard (Terminal UI)."""

from __future__ import annotations

import io
import sys
from unittest.mock import patch

from xpyd_bench.reporting.rich_output import LiveDashboard, _sparkline

# ---------------------------------------------------------------------------
# Unit: sparkline helper
# ---------------------------------------------------------------------------


class TestSparkline:
    def test_empty(self) -> None:
        assert _sparkline([]) == ""

    def test_constant(self) -> None:
        result = _sparkline([5.0, 5.0, 5.0])
        assert len(result) == 3
        # All same value → all same char
        assert len(set(result)) == 1

    def test_ascending(self) -> None:
        result = _sparkline([1.0, 2.0, 3.0, 4.0, 5.0])
        assert len(result) == 5
        # Should be monotonically non-decreasing in char
        assert result[0] <= result[-1]

    def test_width_truncation(self) -> None:
        result = _sparkline(list(range(100)), width=10)
        assert len(result) == 10


# ---------------------------------------------------------------------------
# Unit: LiveDashboard data feed
# ---------------------------------------------------------------------------


class TestLiveDashboard:
    def test_init_disabled(self) -> None:
        """When disabled=True, dashboard should be disabled."""
        dash = LiveDashboard(total=10, disabled=True)
        assert dash.disabled is True

    def test_non_tty_auto_disable(self) -> None:
        """Non-TTY stderr → auto-disabled."""
        with patch.object(sys, "stderr", new_callable=io.StringIO):
            dash = LiveDashboard(total=10)
            assert dash.disabled is True

    def test_advance_tracks_counts(self) -> None:
        """Advance properly counts completed/failed."""
        dash = LiveDashboard(total=5, disabled=True)
        dash.start()
        dash.advance(success=True, latency_ms=10.0)
        dash.advance(success=True, latency_ms=20.0)
        dash.advance(success=False, latency_ms=50.0)
        assert dash._completed == 2
        assert dash._failed == 1
        assert len(dash._latencies) == 3
        dash.stop()

    def test_advance_no_latency(self) -> None:
        """Advance works without latency_ms."""
        dash = LiveDashboard(total=2, disabled=True)
        dash.start()
        dash.advance(success=True)
        assert dash._completed == 1
        assert len(dash._latencies) == 0
        dash.stop()

    def test_start_stop_disabled(self) -> None:
        """Start/stop on disabled dashboard should not raise."""
        dash = LiveDashboard(total=5, disabled=True)
        dash.start()
        dash.stop()

    def test_build_layout(self) -> None:
        """_build_layout should not raise."""
        dash = LiveDashboard(total=5, disabled=True)
        dash.start()
        dash.advance(success=True, latency_ms=15.0)
        layout = dash._build_layout()
        assert layout is not None
        dash.stop()


# ---------------------------------------------------------------------------
# CLI: --no-live flag
# ---------------------------------------------------------------------------


class TestNoLiveFlag:
    def test_no_live_parsed(self) -> None:
        """--no-live flag should set no_live=True."""
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--no-live", "--model", "m", "--num-prompts", "1"])
        assert args.no_live is True

    def test_default_no_live_false(self) -> None:
        """Default no_live should be False."""
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--model", "m", "--num-prompts", "1"])
        assert args.no_live is False


# ---------------------------------------------------------------------------
# Config: no_live is a known key
# ---------------------------------------------------------------------------


class TestConfigKnownKey:
    def test_no_live_in_known_keys(self) -> None:
        from xpyd_bench.config_cmd import _KNOWN_KEYS

        assert "no_live" in _KNOWN_KEYS
