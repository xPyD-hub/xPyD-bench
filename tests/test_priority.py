"""Tests for M52: Request Priority & Queuing."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from xpyd_bench.bench.models import BenchmarkResult, RequestResult
from xpyd_bench.bench.priority import (
    PriorityScheduler,
    compute_priority_metrics,
)

# ---------------------------------------------------------------------------
# PriorityScheduler tests
# ---------------------------------------------------------------------------


class TestPriorityScheduler:
    """Test the async priority queue."""

    def test_basic_ordering(self):
        """Items are dequeued in priority order (0 = highest)."""
        scheduler = PriorityScheduler(num_levels=5)
        scheduler.put("low", priority=4)
        scheduler.put("high", priority=0)
        scheduler.put("mid", priority=2)

        items = []
        while not scheduler.empty():
            entry = scheduler.get_nowait()
            assert entry is not None
            items.append(entry)

        assert [p for p, _ in items] == [0, 2, 4]
        assert [v for _, v in items] == ["high", "mid", "low"]

    def test_fifo_within_same_priority(self):
        """Items with the same priority are returned FIFO."""
        scheduler = PriorityScheduler(num_levels=3)
        scheduler.put("first", priority=1)
        scheduler.put("second", priority=1)
        scheduler.put("third", priority=1)

        items = []
        while not scheduler.empty():
            entry = scheduler.get_nowait()
            assert entry is not None
            items.append(entry[1])

        assert items == ["first", "second", "third"]

    def test_priority_clamped(self):
        """Priority is clamped to valid range."""
        scheduler = PriorityScheduler(num_levels=3)
        scheduler.put("over", priority=10)  # clamped to 2
        scheduler.put("under", priority=-5)  # clamped to 0

        items = []
        while not scheduler.empty():
            entry = scheduler.get_nowait()
            assert entry is not None
            items.append((entry[0], entry[1]))

        assert items == [(0, "under"), (2, "over")]

    def test_empty(self):
        scheduler = PriorityScheduler(num_levels=3)
        assert scheduler.empty()
        assert scheduler.qsize() == 0
        assert scheduler.get_nowait() is None

    @pytest.mark.asyncio
    async def test_async_get(self):
        """Async get waits for items."""
        scheduler = PriorityScheduler(num_levels=3)

        async def producer():
            await asyncio.sleep(0.01)
            scheduler.put("item", priority=1)

        asyncio.create_task(producer())
        priority, item = await asyncio.wait_for(
            scheduler.get(), timeout=1.0,
        )
        assert item == "item"
        assert priority == 1


# ---------------------------------------------------------------------------
# compute_priority_metrics tests
# ---------------------------------------------------------------------------


class TestComputePriorityMetrics:
    """Test per-priority metrics computation."""

    def test_basic_metrics(self):
        """Verify per-level metric breakdown."""
        requests = [
            RequestResult(
                latency_ms=100.0, success=True, priority=0,
                completion_tokens=10, prompt_tokens=5,
            ),
            RequestResult(
                latency_ms=120.0, success=True, priority=0,
                completion_tokens=12, prompt_tokens=5,
            ),
            RequestResult(
                latency_ms=200.0, success=True, priority=1,
                completion_tokens=8, prompt_tokens=5,
            ),
            RequestResult(
                latency_ms=250.0, success=False, priority=1,
                error="timeout",
            ),
        ]

        result = compute_priority_metrics(requests, num_levels=2)
        assert result["num_levels"] == 2
        assert "0" in result["levels"]
        assert "1" in result["levels"]

        lvl0 = result["levels"]["0"]
        assert lvl0["total"] == 2
        assert lvl0["completed"] == 2
        assert lvl0["failed"] == 0
        assert lvl0["error_rate"] == 0.0
        assert lvl0["mean_latency_ms"] == 110.0

        lvl1 = result["levels"]["1"]
        assert lvl1["total"] == 2
        assert lvl1["completed"] == 1
        assert lvl1["failed"] == 1
        assert lvl1["error_rate"] == 0.5

    def test_none_priority_defaults_to_zero(self):
        """Requests with no priority default to level 0."""
        requests = [
            RequestResult(
                latency_ms=50.0, success=True, priority=None,
                completion_tokens=5, prompt_tokens=5,
            ),
        ]
        result = compute_priority_metrics(requests, num_levels=3)
        assert "0" in result["levels"]
        assert result["levels"]["0"]["total"] == 1

    def test_empty_requests(self):
        """Empty requests produce empty levels."""
        result = compute_priority_metrics([], num_levels=3)
        assert result["num_levels"] == 3
        assert result["levels"] == {}

    def test_all_failed(self):
        """All-failed level has None latencies."""
        requests = [
            RequestResult(
                latency_ms=0.0, success=False, priority=0, error="fail",
            ),
        ]
        result = compute_priority_metrics(requests, num_levels=2)
        lvl0 = result["levels"]["0"]
        assert lvl0["completed"] == 0
        assert lvl0["failed"] == 1
        assert lvl0["mean_latency_ms"] is None


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


class TestPriorityCLI:
    """Test --priority-levels CLI argument parsing."""

    def test_cli_arg_default(self):
        """Default priority_levels is 0."""
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args([])
        assert args.priority_levels == 0

    def test_cli_arg_set(self):
        """--priority-levels sets the value."""
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--priority-levels", "5"])
        assert args.priority_levels == 5


# ---------------------------------------------------------------------------
# Dataset priority field tests
# ---------------------------------------------------------------------------


class TestDatasetPriority:
    """Test priority field in dataset loading."""

    def test_jsonl_with_priority(self, tmp_path: Path):
        """JSONL entries with priority field are loaded correctly."""
        from xpyd_bench.datasets.loader import load_jsonl

        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text(
            '{"prompt": "hello", "priority": 0}\n'
            '{"prompt": "world", "priority": 2}\n'
            '{"prompt": "test"}\n'
        )
        entries = load_jsonl(jsonl_file)
        assert len(entries) == 3
        assert entries[0].priority == 0
        assert entries[1].priority == 2
        assert entries[2].priority is None

    def test_json_with_priority(self, tmp_path: Path):
        """JSON array entries with priority field."""
        from xpyd_bench.datasets.loader import load_json

        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps([
            {"prompt": "a", "priority": 1},
            {"prompt": "b", "priority": 0},
        ]))
        entries = load_json(json_file)
        assert entries[0].priority == 1
        assert entries[1].priority == 0


# ---------------------------------------------------------------------------
# YAML config tests
# ---------------------------------------------------------------------------


class TestPriorityYAMLConfig:
    """Test priority_levels in YAML config."""

    def test_yaml_config_known_key(self):
        """priority_levels is a known YAML config key."""
        from xpyd_bench.config_cmd import _KNOWN_KEYS

        assert "priority_levels" in _KNOWN_KEYS


# ---------------------------------------------------------------------------
# RequestResult priority field tests
# ---------------------------------------------------------------------------


class TestRequestResultPriority:
    """Test priority field on RequestResult."""

    def test_default_none(self):
        r = RequestResult()
        assert r.priority is None

    def test_set_priority(self):
        r = RequestResult(priority=3)
        assert r.priority == 3


# ---------------------------------------------------------------------------
# BenchmarkResult priority_metrics field tests
# ---------------------------------------------------------------------------


class TestBenchmarkResultPriorityMetrics:
    """Test priority_metrics field on BenchmarkResult."""

    def test_default_none(self):
        r = BenchmarkResult()
        assert r.priority_metrics is None

    def test_set_priority_metrics(self):
        r = BenchmarkResult(
            priority_metrics={"num_levels": 3, "levels": {}},
        )
        assert r.priority_metrics["num_levels"] == 3
