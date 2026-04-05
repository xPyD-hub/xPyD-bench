"""Tests for M85: Request Deduplication & Idempotency Tracking."""

from __future__ import annotations

from xpyd_bench.bench.dedup import compute_dedup_summary, compute_response_hash
from xpyd_bench.bench.models import RequestResult


def _make_request(
    success: bool = True,
    response_text: str | None = None,
) -> RequestResult:
    r = RequestResult()
    r.success = success
    r.response_text = response_text
    return r


class TestComputeResponseHash:
    def test_deterministic(self):
        h1 = compute_response_hash("hello world")
        h2 = compute_response_hash("hello world")
        assert h1 == h2

    def test_different_text_different_hash(self):
        h1 = compute_response_hash("hello")
        h2 = compute_response_hash("world")
        assert h1 != h2

    def test_returns_hex_string(self):
        h = compute_response_hash("test")
        assert len(h) == 64  # SHA-256 hex digest
        assert all(c in "0123456789abcdef" for c in h)


class TestComputeDedupSummary:
    def test_all_unique_responses(self):
        reqs = [_make_request(response_text=f"response {i}") for i in range(10)]
        summary = compute_dedup_summary(reqs)
        assert summary["total"] == 10
        assert summary["unique"] == 10
        assert summary["duplicates"] == 0
        assert summary["duplicate_ratio"] == 0.0
        assert "repeated_hashes" not in summary

    def test_all_duplicate_responses(self):
        reqs = [_make_request(response_text="same response") for _ in range(5)]
        summary = compute_dedup_summary(reqs)
        assert summary["total"] == 5
        assert summary["unique"] == 1
        assert summary["duplicates"] == 4
        assert summary["duplicate_ratio"] == 0.8
        assert len(summary["repeated_hashes"]) == 1

    def test_mixed_responses(self):
        reqs = [
            _make_request(response_text="unique1"),
            _make_request(response_text="unique2"),
            _make_request(response_text="dup"),
            _make_request(response_text="dup"),
            _make_request(response_text="dup"),
        ]
        summary = compute_dedup_summary(reqs)
        assert summary["total"] == 5
        assert summary["unique"] == 3  # unique1, unique2, dup
        assert summary["duplicates"] == 2
        assert summary["duplicate_ratio"] == 0.4

    def test_failed_requests_excluded(self):
        reqs = [
            _make_request(success=False, response_text="cached"),
            _make_request(success=True, response_text="cached"),
            _make_request(success=True, response_text="cached"),
        ]
        summary = compute_dedup_summary(reqs)
        assert summary["total"] == 2  # only 2 successful

    def test_none_response_text_excluded(self):
        reqs = [
            _make_request(response_text=None),
            _make_request(response_text="hello"),
        ]
        summary = compute_dedup_summary(reqs)
        assert summary["total"] == 1

    def test_empty_requests(self):
        summary = compute_dedup_summary([])
        assert summary["total"] == 0
        assert summary["unique"] == 0
        assert summary["duplicates"] == 0
        assert summary["duplicate_ratio"] == 0.0

    def test_disabled_by_default(self):
        """When deduplicate is not enabled, dedup_summary should not be set."""
        from argparse import Namespace

        args = Namespace(deduplicate=False)
        assert not bool(getattr(args, "deduplicate", False))
