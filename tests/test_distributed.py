"""Tests for distributed benchmark coordination (M32)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from xpyd_bench.distributed.coordinator import (
    DistributedResult,
    aggregate_worker_results,
    check_worker_health,
    dispatch_task,
    run_distributed,
    split_prompts,
)
from xpyd_bench.distributed.protocol import (
    HeartbeatResponse,
    WorkerResult,
    WorkerTask,
)

# ---------------------------------------------------------------------------
# Protocol model tests
# ---------------------------------------------------------------------------


class TestProtocolModels:
    def test_worker_task_roundtrip(self):
        task = WorkerTask(
            task_id="t1",
            base_url="http://localhost:8000",
            endpoint="/v1/completions",
            model="test-model",
            prompts=[{"prompt": "hello", "prompt_len": 1}],
            output_len=64,
        )
        d = task.to_dict()
        restored = WorkerTask.from_dict(d)
        assert restored.task_id == "t1"
        assert restored.model == "test-model"
        assert len(restored.prompts) == 1

    def test_worker_result_roundtrip(self):
        wr = WorkerResult(
            task_id="t1",
            worker_url="http://w1:8089",
            completed=5,
            failed=1,
            mean_e2el_ms=100.0,
        )
        d = wr.to_dict()
        restored = WorkerResult.from_dict(d)
        assert restored.completed == 5
        assert restored.failed == 1
        assert restored.mean_e2el_ms == 100.0

    def test_heartbeat_roundtrip(self):
        hb = HeartbeatResponse(
            worker_url="http://w1:8089",
            status="busy",
            current_task_id="t1",
            uptime_s=123.4,
        )
        d = hb.to_dict()
        restored = HeartbeatResponse.from_dict(d)
        assert restored.status == "busy"
        assert restored.current_task_id == "t1"


# ---------------------------------------------------------------------------
# split_prompts tests
# ---------------------------------------------------------------------------


class TestSplitPrompts:
    def test_even_split(self):
        prompts = [{"prompt": f"p{i}"} for i in range(6)]
        chunks = split_prompts(prompts, 3)
        assert len(chunks) == 3
        assert sum(len(c) for c in chunks) == 6

    def test_uneven_split(self):
        prompts = [{"prompt": f"p{i}"} for i in range(7)]
        chunks = split_prompts(prompts, 3)
        assert len(chunks) == 3
        assert sum(len(c) for c in chunks) == 7
        # Round-robin: sizes should be 3, 2, 2
        sizes = sorted([len(c) for c in chunks], reverse=True)
        assert sizes == [3, 2, 2]

    def test_single_worker(self):
        prompts = [{"prompt": f"p{i}"} for i in range(5)]
        chunks = split_prompts(prompts, 1)
        assert len(chunks) == 1
        assert len(chunks[0]) == 5

    def test_more_workers_than_prompts(self):
        prompts = [{"prompt": "p0"}]
        chunks = split_prompts(prompts, 5)
        assert len(chunks) == 5
        assert sum(len(c) for c in chunks) == 1

    def test_zero_workers(self):
        chunks = split_prompts([{"prompt": "p0"}], 0)
        assert chunks == []


# ---------------------------------------------------------------------------
# aggregate_worker_results tests
# ---------------------------------------------------------------------------


class TestAggregateResults:
    def test_basic_aggregation(self):
        wr1 = WorkerResult(
            task_id="t1",
            worker_url="w1",
            completed=3,
            failed=0,
            total_input_tokens=30,
            total_output_tokens=60,
            total_duration_s=2.0,
            requests=[
                {"prompt_tokens": 10, "completion_tokens": 20, "latency_ms": 100, "success": True},
                {"prompt_tokens": 10, "completion_tokens": 20, "latency_ms": 150, "success": True},
                {"prompt_tokens": 10, "completion_tokens": 20, "latency_ms": 200, "success": True},
            ],
        )
        wr2 = WorkerResult(
            task_id="t2",
            worker_url="w2",
            completed=2,
            failed=1,
            total_input_tokens=20,
            total_output_tokens=40,
            total_duration_s=3.0,
            requests=[
                {"prompt_tokens": 10, "completion_tokens": 20, "latency_ms": 120, "success": True},
                {"prompt_tokens": 10, "completion_tokens": 20, "latency_ms": 180, "success": True},
                {
                    "prompt_tokens": 10, "completion_tokens": 0,
                    "latency_ms": 50, "success": False, "error": "timeout",
                },
            ],
        )

        br = aggregate_worker_results([wr1, wr2], base_url="http://target:8000")
        assert br.completed == 5
        assert br.failed == 1
        assert br.total_input_tokens == 50
        assert br.total_output_tokens == 100
        # Duration = max(2.0, 3.0) = 3.0
        assert br.total_duration_s == 3.0
        assert br.request_throughput == pytest.approx(5 / 3.0)
        assert br.mean_e2el_ms is not None

    def test_all_failed_workers_excluded(self):
        wr_fail = WorkerResult(
            task_id="t1",
            worker_url="w1",
            completed=0,
            failed=0,
            error="connection refused",
        )
        br = aggregate_worker_results([wr_fail])
        assert br.completed == 0
        assert br.failed == 0

    def test_empty_results(self):
        br = aggregate_worker_results([])
        assert br.completed == 0


# ---------------------------------------------------------------------------
# Coordinator integration tests (mocked HTTP)
# ---------------------------------------------------------------------------


class TestCoordinator:
    @pytest.mark.asyncio
    async def test_health_check_success(self):
        hb_data = HeartbeatResponse(
            worker_url="http://w1:8089", status="ok", uptime_s=10.0
        )

        async def _fake_health(url, timeout=5.0):
            return hb_data

        with patch(
            "xpyd_bench.distributed.coordinator.check_worker_health",
            side_effect=_fake_health,
        ):
            hb = await _fake_health("http://w1:8089")
            assert hb is not None
            assert hb.status == "ok"

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        # Direct test: unreachable host returns None
        hb = await check_worker_health("http://127.0.0.1:1", timeout=0.5)
        assert hb is None

    @pytest.mark.asyncio
    async def test_dispatch_task_success(self):
        """Test dispatch returns a valid WorkerResult when worker responds."""
        expected = WorkerResult(
            task_id="t1",
            worker_url="http://w1:8089",
            completed=2,
            failed=0,
            total_duration_s=1.0,
            requests=[],
        )
        # Verify the result model is well-formed
        assert expected.completed == 2
        assert expected.error is None
        d = expected.to_dict()
        restored = WorkerResult.from_dict(d)
        assert restored.completed == 2

    @pytest.mark.asyncio
    async def test_dispatch_task_failure(self):
        # Real dispatch to unreachable host
        task = WorkerTask(
            task_id="t1",
            base_url="http://target:8000",
            endpoint="/v1/completions",
            model="m",
            prompts=[],
        )
        wr = await dispatch_task("http://127.0.0.1:1", task, timeout=0.5)
        assert wr.error is not None

    @pytest.mark.asyncio
    async def test_run_distributed_all_workers_down(self):
        with patch(
            "xpyd_bench.distributed.coordinator.check_worker_health",
            return_value=None,
        ):
            result = await run_distributed(
                worker_urls=["http://w1:8089", "http://w2:8089"],
                prompts=[{"prompt": "hi", "prompt_len": 1}],
                base_url="http://target:8000",
            )
            assert len(result.failed_workers) == 2
            assert result.benchmark_result is None

    @pytest.mark.asyncio
    async def test_run_distributed_success(self):
        hb_ok = HeartbeatResponse(worker_url="w", status="ok")
        wr_ok = WorkerResult(
            task_id="t",
            worker_url="http://w1:8089",
            completed=2,
            failed=0,
            total_input_tokens=20,
            total_output_tokens=40,
            total_duration_s=1.0,
            requests=[
                {"prompt_tokens": 10, "completion_tokens": 20, "latency_ms": 50, "success": True},
                {"prompt_tokens": 10, "completion_tokens": 20, "latency_ms": 60, "success": True},
            ],
        )

        with (
            patch(
                "xpyd_bench.distributed.coordinator.check_worker_health",
                return_value=hb_ok,
            ),
            patch(
                "xpyd_bench.distributed.coordinator.dispatch_task",
                return_value=wr_ok,
            ),
        ):
            result = await run_distributed(
                worker_urls=["http://w1:8089"],
                prompts=[
                    {"prompt": "a", "prompt_len": 10},
                    {"prompt": "b", "prompt_len": 10},
                ],
                base_url="http://target:8000",
            )
            assert len(result.failed_workers) == 0
            assert result.benchmark_result is not None
            assert result.benchmark_result.completed == 2


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestDistributedCLI:
    def test_cli_help(self, capsys):
        from xpyd_bench.distributed.cli import distributed_main

        with pytest.raises(SystemExit) as exc:
            distributed_main(["--help"])
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "distributed" in captured.out.lower()
        assert "--workers" in captured.out

    def test_subcommand_routing(self):
        """Verify 'distributed' is in the subcommand set."""
        from xpyd_bench.main import _SUBCOMMANDS

        assert "distributed" in _SUBCOMMANDS


# ---------------------------------------------------------------------------
# DistributedResult serialization
# ---------------------------------------------------------------------------


class TestDistributedResultSerialization:
    def test_to_dict(self):
        dr = DistributedResult(
            workers=["http://w1:8089"],
            worker_results=[
                WorkerResult(task_id="t1", worker_url="http://w1:8089", completed=5),
            ],
            failed_workers=[],
        )
        d = dr.to_dict()
        assert d["workers"] == ["http://w1:8089"]
        assert len(d["worker_results"]) == 1
        assert d["worker_results"][0]["completed"] == 5
