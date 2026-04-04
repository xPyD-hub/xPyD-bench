"""Tests for M41: Batch Inference API Benchmarking."""

from __future__ import annotations

import asyncio
import time
from argparse import Namespace

import httpx
import pytest

from xpyd_bench.batch import (
    BatchRequestResult,
    _compute_batch_metrics,
    poll_batch,
    run_batch_benchmark,
    submit_batch,
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


# ---------------------------------------------------------------------------
# Integration test with dummy server
# ---------------------------------------------------------------------------


@pytest.fixture()
def dummy_batch_server():
    """Start the dummy server with batch endpoints in a background thread."""
    import threading

    import uvicorn

    from xpyd_bench.dummy.server import ServerConfig, create_app

    cfg = ServerConfig(prefill_ms=10, decode_ms=2, model_name="test-model")
    app = create_app(cfg)

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"))

    # Find a free port
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    server_cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(server_cfg)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to be ready
    import time as _time
    base_url = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            httpx.get(f"{base_url}/health", timeout=1.0)
            break
        except Exception:
            _time.sleep(0.1)
    else:
        raise RuntimeError("Dummy server failed to start")

    yield base_url

    server.should_exit = True
    thread.join(timeout=5)


class TestDummyServerBatch:
    def test_create_and_retrieve_batch(self, dummy_batch_server):
        base_url = dummy_batch_server

        async def _run():
            async with httpx.AsyncClient() as client:
                # Submit batch
                batch = await submit_batch(
                    client,
                    f"{base_url}/v1/batches",
                    [
                        {"model": "test-model", "prompt": "hello", "max_tokens": 10},
                        {"model": "test-model", "prompt": "world", "max_tokens": 10},
                    ],
                    model="test-model",
                )
                assert "id" in batch
                assert batch["status"] == "validating"

                # Poll until complete
                final = await poll_batch(
                    client,
                    f"{base_url}/v1/batches",
                    batch["id"],
                    poll_interval=0.1,
                    timeout=30.0,
                )
                assert final["status"] == "completed"
                assert final["request_counts"]["completed"] == 2
                assert len(final["results"]) == 2
                assert final["in_progress_at"] is not None
                assert final["completed_at"] is not None

        asyncio.run(_run())

    def test_batch_metrics(self, dummy_batch_server):
        base_url = dummy_batch_server

        async def _run():
            async with httpx.AsyncClient() as client:
                batch = await submit_batch(
                    client,
                    f"{base_url}/v1/batches",
                    [{"model": "m", "prompt": "p", "max_tokens": 5}],
                )
                final = await poll_batch(
                    client,
                    f"{base_url}/v1/batches",
                    batch["id"],
                    poll_interval=0.1,
                    timeout=30.0,
                )
                metrics = _compute_batch_metrics(final, time.time())
                assert metrics.success is True
                assert metrics.queue_time_ms > 0
                assert metrics.processing_time_ms > 0

        asyncio.run(_run())

    def test_batch_not_found(self, dummy_batch_server):
        base_url = dummy_batch_server

        async def _run():
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{base_url}/v1/batches/nonexistent")
                assert resp.status_code == 404

        asyncio.run(_run())


class TestRunBatchBenchmark:
    def test_full_batch_benchmark(self, dummy_batch_server):
        base_url = dummy_batch_server
        args = Namespace(
            num_prompts=3,
            input_len=32,
            output_len=16,
            model="test-model",
            seed=42,
            poll_interval=0.1,
            batch_timeout=30.0,
            batch_endpoint="/v1/completions",
            api_key=None,
            timeout=300.0,
            dataset_path=None,
            disable_tqdm=True,
            custom_headers=None,
        )

        result_dict, bench_result = asyncio.run(
            run_batch_benchmark(args, base_url)
        )

        assert result_dict["endpoint"] == "/v1/batch"
        assert result_dict["completed"] == 3
        assert result_dict["failed"] == 0
        assert result_dict["batch_status"] == "completed"
        assert result_dict["queue_time_ms"] > 0
        assert result_dict["processing_time_ms"] > 0
        assert "batch_id" in result_dict
        assert bench_result.total_duration_s > 0


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
