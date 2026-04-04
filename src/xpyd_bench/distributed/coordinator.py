"""Distributed benchmark coordinator (M32).

Splits prompts across workers, dispatches tasks, collects results,
and aggregates into a unified BenchmarkResult.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np

from xpyd_bench.bench.models import BenchmarkResult, RequestResult
from xpyd_bench.distributed.protocol import (
    HeartbeatResponse,
    WorkerResult,
    WorkerTask,
)


@dataclass
class DistributedResult:
    """Aggregated result from a distributed benchmark run."""

    workers: list[str]
    worker_results: list[WorkerResult] = field(default_factory=list)
    failed_workers: list[str] = field(default_factory=list)
    benchmark_result: BenchmarkResult | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "workers": self.workers,
            "worker_results": [wr.to_dict() for wr in self.worker_results],
            "failed_workers": self.failed_workers,
        }
        if self.benchmark_result:
            from dataclasses import asdict

            d["benchmark_result"] = asdict(self.benchmark_result)
        return d


async def check_worker_health(
    worker_url: str, timeout: float = 5.0
) -> HeartbeatResponse | None:
    """Send a heartbeat request to a worker. Returns None if unreachable."""
    url = f"{worker_url.rstrip('/')}/heartbeat"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return HeartbeatResponse.from_dict(resp.json())
    except Exception:
        return None


def split_prompts(
    prompts: list[dict[str, Any]], num_workers: int
) -> list[list[dict[str, Any]]]:
    """Split prompts into roughly equal chunks for each worker."""
    if num_workers <= 0:
        return []
    chunks: list[list[dict[str, Any]]] = [[] for _ in range(num_workers)]
    for i, prompt in enumerate(prompts):
        chunks[i % num_workers].append(prompt)
    return chunks


async def dispatch_task(
    worker_url: str, task: WorkerTask, timeout: float = 600.0
) -> WorkerResult:
    """Send a task to a worker and wait for the result."""
    url = f"{worker_url.rstrip('/')}/run"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=task.to_dict())
            resp.raise_for_status()
            return WorkerResult.from_dict(resp.json())
    except Exception as exc:
        return WorkerResult(
            task_id=task.task_id,
            worker_url=worker_url,
            error=f"Worker dispatch failed: {exc}",
        )


def aggregate_worker_results(
    worker_results: list[WorkerResult],
    base_url: str = "",
    endpoint: str = "/v1/completions",
    model: str = "",
) -> BenchmarkResult:
    """Merge results from all workers into a single BenchmarkResult."""
    all_requests: list[RequestResult] = []
    total_completed = 0
    total_failed = 0
    total_input_tokens = 0
    total_output_tokens = 0
    max_duration = 0.0

    ttfts: list[float] = []
    tpots: list[float] = []
    e2els: list[float] = []

    for wr in worker_results:
        if wr.error and wr.completed == 0:
            continue
        total_completed += wr.completed
        total_failed += wr.failed
        total_input_tokens += wr.total_input_tokens
        total_output_tokens += wr.total_output_tokens
        if wr.total_duration_s > max_duration:
            max_duration = wr.total_duration_s

        for req in wr.requests:
            rr = RequestResult(
                prompt_tokens=req.get("prompt_tokens", 0),
                completion_tokens=req.get("completion_tokens", 0),
                ttft_ms=req.get("ttft_ms"),
                latency_ms=req.get("latency_ms", 0),
                success=req.get("success", True),
                error=req.get("error"),
            )
            all_requests.append(rr)
            if rr.success:
                e2els.append(rr.latency_ms)
                if rr.ttft_ms is not None:
                    ttfts.append(rr.ttft_ms)
                if rr.completion_tokens > 1 and rr.latency_ms > 0:
                    tpots.append(rr.latency_ms / rr.completion_tokens)

    def _p(vals: list[float], pct: float) -> float | None:
        return float(np.percentile(vals, pct)) if vals else None

    total_duration = max_duration if max_duration > 0 else 1.0
    num_prompts = total_completed + total_failed

    return BenchmarkResult(
        base_url=base_url,
        endpoint=endpoint,
        model=model,
        num_prompts=num_prompts,
        requests=all_requests,
        total_duration_s=total_duration,
        completed=total_completed,
        failed=total_failed,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        request_throughput=total_completed / total_duration,
        output_throughput=total_output_tokens / total_duration,
        total_token_throughput=(total_input_tokens + total_output_tokens) / total_duration,
        mean_ttft_ms=float(np.mean(ttfts)) if ttfts else None,
        median_ttft_ms=_p(ttfts, 50),
        p50_ttft_ms=_p(ttfts, 50),
        p90_ttft_ms=_p(ttfts, 90),
        p95_ttft_ms=_p(ttfts, 95),
        p99_ttft_ms=_p(ttfts, 99),
        mean_tpot_ms=float(np.mean(tpots)) if tpots else None,
        median_tpot_ms=_p(tpots, 50),
        p50_tpot_ms=_p(tpots, 50),
        p90_tpot_ms=_p(tpots, 90),
        p95_tpot_ms=_p(tpots, 95),
        p99_tpot_ms=_p(tpots, 99),
        mean_e2el_ms=float(np.mean(e2els)) if e2els else None,
        median_e2el_ms=_p(e2els, 50),
        p50_e2el_ms=_p(e2els, 50),
        p90_e2el_ms=_p(e2els, 90),
        p95_e2el_ms=_p(e2els, 95),
        p99_e2el_ms=_p(e2els, 99),
    )


async def run_distributed(
    worker_urls: list[str],
    prompts: list[dict[str, Any]],
    base_url: str,
    endpoint: str = "/v1/completions",
    model: str = "",
    output_len: int = 128,
    stream: bool | None = None,
    api_key: str | None = None,
    timeout: float = 300.0,
    custom_headers: dict[str, str] | None = None,
    sampling_params: dict[str, Any] | None = None,
    heartbeat_interval: float = 10.0,
    task_timeout: float = 600.0,
) -> DistributedResult:
    """Coordinate a distributed benchmark across multiple workers.

    1. Health-check all workers
    2. Split prompts evenly
    3. Dispatch tasks in parallel
    4. Monitor heartbeats during execution
    5. Aggregate results
    """
    result = DistributedResult(workers=list(worker_urls))

    # 1. Health check
    healthy_workers: list[str] = []
    for url in worker_urls:
        hb = await check_worker_health(url)
        if hb is not None and hb.status != "error":
            healthy_workers.append(url)
        else:
            result.failed_workers.append(url)

    if not healthy_workers:
        return result

    # 2. Split prompts
    chunks = split_prompts(prompts, len(healthy_workers))

    # 3. Create and dispatch tasks
    tasks: list[tuple[str, WorkerTask]] = []
    for i, worker_url in enumerate(healthy_workers):
        task = WorkerTask(
            task_id=str(uuid.uuid4()),
            base_url=base_url,
            endpoint=endpoint,
            model=model,
            prompts=chunks[i],
            output_len=output_len,
            stream=stream,
            api_key=api_key,
            timeout=timeout,
            custom_headers=custom_headers or {},
            sampling_params=sampling_params or {},
        )
        tasks.append((worker_url, task))

    # 4. Dispatch all tasks concurrently
    coros = [dispatch_task(url, task, timeout=task_timeout) for url, task in tasks]
    worker_results = await asyncio.gather(*coros)
    result.worker_results = list(worker_results)

    # Mark workers that returned errors
    for wr in worker_results:
        if wr.error and wr.completed == 0 and wr.worker_url not in result.failed_workers:
            result.failed_workers.append(wr.worker_url)

    # 5. Aggregate
    result.benchmark_result = aggregate_worker_results(
        [wr for wr in worker_results if not (wr.error and wr.completed == 0)],
        base_url=base_url,
        endpoint=endpoint,
        model=model,
    )

    return result
