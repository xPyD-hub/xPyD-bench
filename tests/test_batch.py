"""Tests for M41: Batch Inference API Benchmarking."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from xpyd_bench.batch import (
    BatchRequestResult,
    _compute_batch_metrics,
    poll_batch,
)

# ---------------------------------------------------------------------------
# Unit tests for BatchRequestResult
# ---------------------------------------------------------------------------


class TestBatchRequestResult:
    def test_defaults(self):
        r = BatchRequestResult()
        assert r.batch_id == ""
        assert r.queue_time_ms == 0.0
        assert r.processing_time_ms == 0.0
        assert r.success is True
        assert r.results == []


# ---------------------------------------------------------------------------
# Unit tests for _compute_batch_metrics
# ---------------------------------------------------------------------------


class TestComputeBatchMetrics:
    def test_completed_batch(self):
        batch_obj = {
            "id": "batch_123",
            "status": "completed",
            "created_at": 1000.0,
            "in_progress_at": 1002.0,
            "completed_at": 1010.0,
            "request_counts": {"total": 5, "completed": 5, "failed": 0},
            "results": [{"custom_id": "r-0"}],
        }
        m = _compute_batch_metrics(batch_obj, 999.0)
        assert m.batch_id == "batch_123"
        assert m.status == "completed"
        assert m.success is True
        assert m.queue_time_ms == pytest.approx(2000.0)
        assert m.processing_time_ms == pytest.approx(8000.0)
        assert m.total_time_ms == pytest.approx(10000.0)
        assert m.total_requests == 5
        assert m.completed_requests == 5
        assert m.failed_requests == 0

    def test_failed_batch(self):
        batch_obj = {
            "id": "batch_fail",
            "status": "failed",
            "created_at": 100.0,
            "in_progress_at": 101.0,
            "completed_at": None,
            "request_counts": {"total": 3, "completed": 0, "failed": 3},
            "errors": {"object": "list", "data": [{"message": "server error"}]},
        }
        m = _compute_batch_metrics(batch_obj, 99.0)
        assert m.success is False
        assert m.error == "server error"
        assert m.failed_requests == 3


class TestPollTimeout:
    def test_poll_timeout_raises(self):
        """Poll should raise TimeoutError if batch never completes."""

        async def _run():
            transport = httpx.MockTransport(
                lambda req: httpx.Response(
                    200,
                    json={
                        "id": "batch_slow",
                        "status": "in_progress",
                        "request_counts": {"total": 1, "completed": 0, "failed": 0},
                    },
                )
            )
            async with httpx.AsyncClient(transport=transport) as client:
                with pytest.raises(TimeoutError, match="did not complete"):
                    await poll_batch(
                        client,
                        "http://fake/v1/batches",
                        "batch_slow",
                        poll_interval=0.05,
                        timeout=0.2,
                    )

        asyncio.run(_run())


class TestBatchCLI:
    def test_batch_subcommand_routing(self):
        """Verify 'batch' is in the subcommands set."""
        from xpyd_bench.main import _SUBCOMMANDS
        assert "batch" in _SUBCOMMANDS
