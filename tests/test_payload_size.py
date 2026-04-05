"""Tests for M67: Request/Response Payload Size Tracking."""

from __future__ import annotations

import argparse
import json

from xpyd_bench.bench.payload_size import (
    PayloadSummary,
    aggregate_payload_sizes,
    compute_payload_bytes,
)

# ---------------------------------------------------------------------------
# Unit tests — compute_payload_bytes
# ---------------------------------------------------------------------------


class TestComputePayloadBytes:
    def test_dict_payload(self):
        payload = {"model": "test", "prompt": "hello"}
        size = compute_payload_bytes(payload)
        expected = len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        assert size == expected

    def test_string_payload(self):
        payload = "hello world"
        size = compute_payload_bytes(payload)
        assert size == len(payload.encode("utf-8"))

    def test_bytes_payload(self):
        payload = b"raw bytes"
        assert compute_payload_bytes(payload) == len(payload)

    def test_none_payload(self):
        assert compute_payload_bytes(None) == 0

    def test_unicode_string(self):
        payload = "你好世界"
        size = compute_payload_bytes(payload)
        assert size == len(payload.encode("utf-8"))
        assert size > len(payload)  # UTF-8 multi-byte


# ---------------------------------------------------------------------------
# Unit tests — aggregation
# ---------------------------------------------------------------------------


class TestAggregatePayloadSizes:
    def test_basic_aggregation(self):
        req = [100, 200, 300]
        resp = [500, 600, 700]
        summary = aggregate_payload_sizes(req, resp)
        assert summary.total_request_bytes == 600
        assert summary.total_response_bytes == 1800
        assert summary.mean_request_bytes == 200.0
        assert summary.mean_response_bytes == 600.0
        assert summary.min_request_bytes == 100
        assert summary.max_request_bytes == 300
        assert summary.tracked_requests == 3

    def test_with_none_entries(self):
        req = [100, None, 300]
        resp = [500, None, 700]
        summary = aggregate_payload_sizes(req, resp)
        assert summary.total_request_bytes == 400
        assert summary.total_response_bytes == 1200
        assert summary.tracked_requests == 2

    def test_single_entry(self):
        summary = aggregate_payload_sizes([42], [99])
        assert summary.total_request_bytes == 42
        assert summary.p50_request_bytes == 42.0
        assert summary.p99_request_bytes == 42.0

    def test_empty_input(self):
        summary = aggregate_payload_sizes([], [])
        assert summary.tracked_requests == 0
        assert summary.total_request_bytes == 0
        assert summary.total_response_bytes == 0

    def test_to_dict(self):
        summary = PayloadSummary(
            total_request_bytes=1000,
            total_response_bytes=5000,
            mean_request_bytes=250.0,
            mean_response_bytes=1250.0,
            tracked_requests=4,
        )
        d = summary.to_dict()
        assert d["total_request_bytes"] == 1000
        assert d["total_response_bytes"] == 5000
        assert d["tracked_requests"] == 4


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


def test_cli_track_payload_size_flag():
    """--track-payload-size flag is parsed correctly."""
    from xpyd_bench.cli import _add_vllm_compat_args

    parser = argparse.ArgumentParser()
    _add_vllm_compat_args(parser)
    ns = parser.parse_args(["--track-payload-size", "--base-url", "http://localhost:8000"])
    assert ns.track_payload_size is True


def test_cli_track_payload_size_default():
    """--track-payload-size defaults to False."""
    from xpyd_bench.cli import _add_vllm_compat_args

    parser = argparse.ArgumentParser()
    _add_vllm_compat_args(parser)
    ns = parser.parse_args(["--base-url", "http://localhost:8000"])
    assert ns.track_payload_size is False


def test_config_known_keys_includes_track_payload_size():
    """track_payload_size is in the known config keys set."""
    from xpyd_bench.config_cmd import _KNOWN_KEYS

    assert "track_payload_size" in _KNOWN_KEYS


# ---------------------------------------------------------------------------
# Model integration tests
# ---------------------------------------------------------------------------


def test_request_result_payload_fields():
    """RequestResult has request_bytes and response_bytes fields."""
    from xpyd_bench.bench.models import RequestResult

    r = RequestResult()
    assert r.request_bytes is None
    assert r.response_bytes is None
    r.request_bytes = 256
    r.response_bytes = 1024
    assert r.request_bytes == 256
    assert r.response_bytes == 1024


def test_benchmark_result_payload_summary():
    """BenchmarkResult has payload_summary field."""
    from xpyd_bench.bench.models import BenchmarkResult

    br = BenchmarkResult()
    assert br.payload_summary is None
    br.payload_summary = {"total_request_bytes": 100}
    assert br.payload_summary["total_request_bytes"] == 100
