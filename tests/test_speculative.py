"""Tests for M88: Speculative Decoding Metrics."""

from __future__ import annotations

import argparse

from xpyd_bench.bench.models import BenchmarkResult, RequestResult
from xpyd_bench.bench.speculative import (
    SpecTokenEvent,
    analyze_spec_events,
    compute_speculative_aggregate,
    parse_spec_data_from_chunk,
)

# ---------------------------------------------------------------------------
# parse_spec_data_from_chunk tests
# ---------------------------------------------------------------------------


class TestParseSpecDataFromChunk:
    """Test SSE chunk parsing for speculative decoding data."""

    def test_top_level_x_spec(self):
        """Parse x_spec from top-level chunk field."""
        chunk = {
            "id": "cmpl-1",
            "choices": [{"index": 0, "text": "token"}],
            "x_spec": {"draft_tokens": 5, "accepted_tokens": 4},
        }
        ev = parse_spec_data_from_chunk(chunk)
        assert ev is not None
        assert ev.draft_tokens == 5
        assert ev.accepted_tokens == 4

    def test_top_level_speculative_key(self):
        """Parse alternative 'speculative' key."""
        chunk = {
            "id": "cmpl-1",
            "choices": [{"index": 0, "text": "token"}],
            "speculative": {"draft_tokens": 3, "accepted_tokens": 2},
        }
        ev = parse_spec_data_from_chunk(chunk)
        assert ev is not None
        assert ev.draft_tokens == 3
        assert ev.accepted_tokens == 2

    def test_choice_level_x_spec(self):
        """Parse x_spec from choice-level field."""
        chunk = {
            "id": "cmpl-1",
            "choices": [
                {
                    "index": 0,
                    "text": "token",
                    "x_spec": {"draft_tokens": 4, "accepted_tokens": 3},
                }
            ],
        }
        ev = parse_spec_data_from_chunk(chunk)
        assert ev is not None
        assert ev.draft_tokens == 4
        assert ev.accepted_tokens == 3

    def test_no_spec_data(self):
        """Return None when no speculative data present."""
        chunk = {
            "id": "cmpl-1",
            "choices": [{"index": 0, "text": "token"}],
        }
        ev = parse_spec_data_from_chunk(chunk)
        assert ev is None

    def test_zero_draft_tokens_ignored(self):
        """Ignore x_spec with draft_tokens=0."""
        chunk = {
            "id": "cmpl-1",
            "x_spec": {"draft_tokens": 0, "accepted_tokens": 0},
        }
        ev = parse_spec_data_from_chunk(chunk)
        assert ev is None

    def test_invalid_types_ignored(self):
        """Ignore x_spec with non-integer values."""
        chunk = {
            "id": "cmpl-1",
            "x_spec": {"draft_tokens": "five", "accepted_tokens": 4},
        }
        ev = parse_spec_data_from_chunk(chunk)
        assert ev is None


# ---------------------------------------------------------------------------
# analyze_spec_events tests
# ---------------------------------------------------------------------------


class TestAnalyzeSpecEvents:
    """Test per-request speculative decoding analysis."""

    def test_basic_analysis(self):
        """Compute acceptance rate and batch size from events."""
        events = [
            SpecTokenEvent(draft_tokens=5, accepted_tokens=4, timestamp=0.1),
            SpecTokenEvent(draft_tokens=5, accepted_tokens=3, timestamp=0.2),
            SpecTokenEvent(draft_tokens=5, accepted_tokens=5, timestamp=0.3),
        ]
        result = analyze_spec_events(events)
        assert result.total_draft_tokens == 15
        assert result.total_accepted_tokens == 12
        assert result.total_rejected_tokens == 3
        assert result.tokens_saved == 12
        assert result.mean_acceptance_rate is not None
        assert 0.7 < result.mean_acceptance_rate < 0.9
        assert result.mean_draft_batch_size == 5.0

    def test_empty_events(self):
        """Handle empty event list."""
        result = analyze_spec_events([])
        assert result.total_draft_tokens == 0
        assert result.mean_acceptance_rate is None
        assert result.tokens_saved == 0

    def test_single_event(self):
        """Handle single event."""
        events = [SpecTokenEvent(draft_tokens=4, accepted_tokens=2, timestamp=0.1)]
        result = analyze_spec_events(events)
        assert result.total_draft_tokens == 4
        assert result.total_accepted_tokens == 2
        assert result.total_rejected_tokens == 2
        assert result.mean_acceptance_rate == 0.5

    def test_to_dict(self):
        """Verify dict serialization."""
        events = [SpecTokenEvent(draft_tokens=5, accepted_tokens=4, timestamp=0.1)]
        result = analyze_spec_events(events)
        d = result.to_dict()
        assert "event_count" in d
        assert d["event_count"] == 1
        assert d["total_draft_tokens"] == 5
        assert d["tokens_saved"] == 4


# ---------------------------------------------------------------------------
# compute_speculative_aggregate tests
# ---------------------------------------------------------------------------


class TestComputeSpeculativeAggregate:
    """Test cross-request speculative decoding aggregation."""

    def test_aggregate_multiple_requests(self):
        """Aggregate across multiple requests."""
        r1 = analyze_spec_events([
            SpecTokenEvent(draft_tokens=5, accepted_tokens=4, timestamp=0.1),
            SpecTokenEvent(draft_tokens=5, accepted_tokens=5, timestamp=0.2),
        ])
        r2 = analyze_spec_events([
            SpecTokenEvent(draft_tokens=3, accepted_tokens=2, timestamp=0.1),
        ])
        agg = compute_speculative_aggregate([r1, r2])
        assert agg["requests_with_spec_data"] == 2
        assert agg["total_draft_tokens"] == 13
        assert agg["total_accepted_tokens"] == 11
        assert agg["total_rejected_tokens"] == 2
        assert agg["overall_acceptance_rate"] is not None
        assert agg["mean_acceptance_rate"] is not None

    def test_empty_list(self):
        """Handle empty per-request list."""
        agg = compute_speculative_aggregate([])
        assert agg == {}

    def test_no_spec_data_requests(self):
        """Handle requests with no speculative events."""
        r1 = analyze_spec_events([])
        agg = compute_speculative_aggregate([r1])
        assert agg["requests_with_spec_data"] == 0

    def test_mixed_with_and_without(self):
        """Aggregate correctly when some requests lack spec data."""
        r1 = analyze_spec_events([
            SpecTokenEvent(draft_tokens=5, accepted_tokens=4, timestamp=0.1),
        ])
        r2 = analyze_spec_events([])
        agg = compute_speculative_aggregate([r1, r2])
        assert agg["requests_with_spec_data"] == 1
        assert agg["total_requests"] == 2


# ---------------------------------------------------------------------------
# Model integration tests
# ---------------------------------------------------------------------------


class TestModelIntegration:
    """Test that model fields work correctly."""

    def test_request_result_spec_events(self):
        """RequestResult has spec_events field."""
        r = RequestResult()
        assert r.spec_events is None
        r.spec_events = [SpecTokenEvent(draft_tokens=5, accepted_tokens=4, timestamp=0.1)]
        assert len(r.spec_events) == 1

    def test_benchmark_result_speculative_summary(self):
        """BenchmarkResult has speculative_summary field."""
        r = BenchmarkResult()
        assert r.speculative_summary is None
        r.speculative_summary = {"overall_acceptance_rate": 0.8}
        assert r.speculative_summary["overall_acceptance_rate"] == 0.8


# ---------------------------------------------------------------------------
# CLI integration test
# ---------------------------------------------------------------------------


class TestCLIFlag:
    """Test CLI argument parsing for speculative metrics."""

    def test_speculative_metrics_flag(self):
        """--speculative-metrics flag is parsed correctly."""
        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--speculative-metrics"])
        assert args.speculative_metrics is True

    def test_speculative_metrics_default_false(self):
        """--speculative-metrics defaults to False."""
        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args([])
        assert args.speculative_metrics is False


# ---------------------------------------------------------------------------
# Config key test
# ---------------------------------------------------------------------------


class TestConfigKey:
    """Test that speculative_metrics is a known config key."""

    def test_known_key(self):
        """speculative_metrics is in _KNOWN_KEYS."""
        from xpyd_bench.config_cmd import _KNOWN_KEYS

        assert "speculative_metrics" in _KNOWN_KEYS


# ---------------------------------------------------------------------------
# Dummy server speculative config test
# ---------------------------------------------------------------------------


class TestDummyServerConfig:
    """Test dummy server speculative decoding configuration."""

    def test_server_config_fields(self):
        """ServerConfig has speculative fields."""
        from xpyd_bench.dummy.server import ServerConfig

        cfg = ServerConfig(speculative_draft_size=5, speculative_acceptance_rate=0.7)
        assert cfg.speculative_draft_size == 5
        assert cfg.speculative_acceptance_rate == 0.7

    def test_default_disabled(self):
        """Speculative decoding is disabled by default."""
        from xpyd_bench.dummy.server import ServerConfig

        cfg = ServerConfig()
        assert cfg.speculative_draft_size == 0
