"""Tests for M66: Rate-Limit Header Tracking & Backpressure Reporting."""

from __future__ import annotations

import httpx

from xpyd_bench.bench.ratelimit import (
    RatelimitSummary,
    aggregate_ratelimit,
    parse_ratelimit_headers,
)

# ---------------------------------------------------------------------------
# Unit tests — header parsing
# ---------------------------------------------------------------------------


class TestParseRatelimitHeaders:
    def test_standard_headers(self):
        headers = {
            "X-RateLimit-Limit": "100",
            "X-RateLimit-Remaining": "42",
            "X-RateLimit-Reset": "1700000000",
        }
        parsed = parse_ratelimit_headers(headers)
        assert parsed["limit"] == "100"
        assert parsed["remaining"] == "42"
        assert parsed["reset"] == "1700000000"

    def test_openai_style_headers(self):
        headers = {
            "x-ratelimit-limit-requests": "60",
            "x-ratelimit-remaining-requests": "59",
            "x-ratelimit-limit-tokens": "100000",
            "x-ratelimit-remaining-tokens": "99500",
            "x-ratelimit-reset-requests": "1s",
            "x-ratelimit-reset-tokens": "200ms",
        }
        parsed = parse_ratelimit_headers(headers)
        assert parsed["limit_requests"] == "60"
        assert parsed["remaining_requests"] == "59"
        assert parsed["limit_tokens"] == "100000"
        assert parsed["remaining_tokens"] == "99500"

    def test_retry_after(self):
        headers = {"Retry-After": "30"}
        parsed = parse_ratelimit_headers(headers)
        assert parsed["retry_after"] == "30"

    def test_no_ratelimit_headers(self):
        headers = {"Content-Type": "application/json", "X-Custom": "foo"}
        parsed = parse_ratelimit_headers(headers)
        assert parsed == {}

    def test_case_insensitive(self):
        # httpx headers are case-insensitive by default
        headers = httpx.Headers({"x-ratelimit-remaining": "5"})
        parsed = parse_ratelimit_headers(headers)
        assert parsed["remaining"] == "5"


# ---------------------------------------------------------------------------
# Unit tests — aggregation
# ---------------------------------------------------------------------------


class TestAggregateRatelimit:
    def test_basic_aggregation(self):
        per_request = [
            {"limit": "100", "remaining": "99"},
            {"limit": "100", "remaining": "98"},
            {"limit": "100", "remaining": "97"},
        ]
        summary = aggregate_ratelimit(per_request)
        assert summary.min_remaining == 97
        assert summary.max_limit == 100
        assert summary.tracked_responses == 3
        assert summary.total_responses == 3
        assert summary.throttle_count == 0

    def test_with_none_entries(self):
        per_request = [
            {"remaining": "50"},
            None,
            {"remaining": "48"},
        ]
        summary = aggregate_ratelimit(per_request)
        assert summary.min_remaining == 48
        assert summary.tracked_responses == 2
        assert summary.total_responses == 3

    def test_throttle_detection(self):
        per_request = [None, None, None]
        errors = [None, "Client error '429 Too Many Requests'", None]
        summary = aggregate_ratelimit(per_request, errors)
        assert summary.throttle_count == 1

    def test_token_remaining(self):
        per_request = [
            {"remaining_tokens": "10000"},
            {"remaining_tokens": "9500"},
        ]
        summary = aggregate_ratelimit(per_request)
        assert summary.min_remaining_tokens == 9500

    def test_empty_input(self):
        summary = aggregate_ratelimit([])
        assert summary.total_responses == 0
        assert summary.min_remaining is None
        assert summary.throttle_count == 0

    def test_to_dict(self):
        summary = RatelimitSummary(
            min_remaining=5, max_limit=100, throttle_count=2,
            tracked_responses=10, total_responses=12,
        )
        d = summary.to_dict()
        assert d["min_remaining"] == 5
        assert d["max_limit"] == 100
        assert d["throttle_count"] == 2


# ---------------------------------------------------------------------------
# Integration test — dummy server with ratelimit headers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CLI integration test
# ---------------------------------------------------------------------------


def test_cli_track_ratelimits_flag():
    """--track-ratelimits flag is parsed correctly."""
    import argparse

    from xpyd_bench.cli import _add_vllm_compat_args

    parser = argparse.ArgumentParser()
    _add_vllm_compat_args(parser)
    ns = parser.parse_args(["--track-ratelimits", "--base-url", "http://localhost:8000"])
    assert ns.track_ratelimits is True


def test_cli_track_ratelimits_default():
    """--track-ratelimits defaults to False."""
    import argparse

    from xpyd_bench.cli import _add_vllm_compat_args

    parser = argparse.ArgumentParser()
    _add_vllm_compat_args(parser)
    ns = parser.parse_args(["--base-url", "http://localhost:8000"])
    assert ns.track_ratelimits is False


def test_config_known_keys_includes_track_ratelimits():
    """track_ratelimits is in the known config keys set."""
    from xpyd_bench.config_cmd import _KNOWN_KEYS

    assert "track_ratelimits" in _KNOWN_KEYS
