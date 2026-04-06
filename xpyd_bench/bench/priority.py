"""Request priority scheduling and per-priority metrics (M52)."""

from __future__ import annotations

import asyncio
import heapq
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(order=True)
class PrioritizedItem:
    """Wrapper for priority queue ordering. Lower priority value = higher priority."""

    priority: int
    sequence: int  # tie-breaker for FIFO within same priority
    item: Any = field(compare=False)


class PriorityScheduler:
    """Async priority queue that dispatches higher-priority requests first.

    Priority 0 is highest. Items are dispatched in priority order,
    with FIFO ordering within the same priority level.
    """

    def __init__(self, num_levels: int = 10) -> None:
        self._num_levels = num_levels
        self._heap: list[PrioritizedItem] = []
        self._seq = 0
        self._event = asyncio.Event()

    @property
    def num_levels(self) -> int:
        return self._num_levels

    def put(self, item: Any, priority: int = 0) -> None:
        """Add an item with given priority (0 = highest)."""
        priority = max(0, min(priority, self._num_levels - 1))
        heapq.heappush(self._heap, PrioritizedItem(priority, self._seq, item))
        self._seq += 1
        self._event.set()

    async def get(self) -> tuple[int, Any]:
        """Get the highest-priority item. Returns (priority, item)."""
        while not self._heap:
            self._event.clear()
            await self._event.wait()
        entry = heapq.heappop(self._heap)
        if not self._heap:
            self._event.clear()
        return entry.priority, entry.item

    def get_nowait(self) -> tuple[int, Any] | None:
        """Get highest-priority item without waiting. Returns None if empty."""
        if not self._heap:
            return None
        entry = heapq.heappop(self._heap)
        if not self._heap:
            self._event.clear()
        return entry.priority, entry.item

    def empty(self) -> bool:
        return len(self._heap) == 0

    def qsize(self) -> int:
        return len(self._heap)


def compute_priority_metrics(
    requests: list[Any],
    num_levels: int,
) -> dict[str, Any]:
    """Compute per-priority-level metrics breakdown.

    Args:
        requests: List of RequestResult objects with ``priority`` field.
        num_levels: Number of priority levels configured.

    Returns:
        Dict with per-level metrics and summary.
    """
    from xpyd_bench.bench.models import RequestResult

    # Group by priority
    by_level: dict[int, list[RequestResult]] = {}
    for r in requests:
        p = r.priority if r.priority is not None else 0
        by_level.setdefault(p, []).append(r)

    levels: dict[str, Any] = {}
    for level in sorted(by_level.keys()):
        reqs = by_level[level]
        successes = [r for r in reqs if r.success]
        failures = [r for r in reqs if not r.success]
        latencies = [r.latency_ms for r in successes]

        level_metrics: dict[str, Any] = {
            "total": len(reqs),
            "completed": len(successes),
            "failed": len(failures),
            "error_rate": round(len(failures) / len(reqs), 4) if reqs else 0.0,
        }

        if latencies:
            arr = np.array(latencies)
            level_metrics["mean_latency_ms"] = round(float(np.mean(arr)), 2)
            level_metrics["p50_latency_ms"] = round(float(np.percentile(arr, 50)), 2)
            level_metrics["p90_latency_ms"] = round(float(np.percentile(arr, 90)), 2)
            level_metrics["p95_latency_ms"] = round(float(np.percentile(arr, 95)), 2)
            level_metrics["p99_latency_ms"] = round(float(np.percentile(arr, 99)), 2)

            ttfts = [r.ttft_ms for r in successes if r.ttft_ms is not None]
            if ttfts:
                tarr = np.array(ttfts)
                level_metrics["mean_ttft_ms"] = round(float(np.mean(tarr)), 2)
                level_metrics["p99_ttft_ms"] = round(float(np.percentile(tarr, 99)), 2)

            total_output = sum(r.completion_tokens for r in successes)
            total_dur = sum(r.latency_ms for r in successes) / 1000.0
            if total_dur > 0:
                level_metrics["throughput_tok_s"] = round(total_output / total_dur, 2)
        else:
            level_metrics["mean_latency_ms"] = None
            level_metrics["p50_latency_ms"] = None
            level_metrics["p90_latency_ms"] = None
            level_metrics["p95_latency_ms"] = None
            level_metrics["p99_latency_ms"] = None

        levels[str(level)] = level_metrics

    return {
        "num_levels": num_levels,
        "levels": levels,
    }
