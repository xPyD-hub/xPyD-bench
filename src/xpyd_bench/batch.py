"""Batch Inference API benchmarking (M41).

Supports the OpenAI Batch API workflow:
  1. Submit batch job (POST /v1/batches)
  2. Poll for completion (GET /v1/batches/{id})
  3. Retrieve results from completed batch

Metrics tracked: queue_time_ms, processing_time_ms per batch.
"""

from __future__ import annotations

import asyncio
import time
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any

import httpx

from xpyd_bench.bench.env import collect_env_info
from xpyd_bench.bench.models import BenchmarkResult


@dataclass
class BatchRequestResult:
    """Metrics for a single batch job."""

    batch_id: str = ""
    status: str = ""
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    queue_time_ms: float = 0.0  # Time from submission to processing start
    processing_time_ms: float = 0.0  # Time from processing start to completion
    total_time_ms: float = 0.0  # End-to-end time
    error: str | None = None
    success: bool = True
    results: list[dict[str, Any]] = field(default_factory=list)


async def submit_batch(
    client: httpx.AsyncClient,
    url: str,
    requests: list[dict[str, Any]],
    *,
    model: str = "",
    endpoint: str = "/v1/completions",
    metadata: dict[str, str] | None = None,
    request_timeout: float = 300.0,
) -> dict[str, Any]:
    """Submit a batch job. Returns the batch object."""
    # Build JSONL input_data inline (simplified from file-based approach)
    input_requests = []
    for i, req in enumerate(requests):
        input_requests.append({
            "custom_id": f"request-{i}",
            "method": "POST",
            "url": endpoint,
            "body": req,
        })

    payload: dict[str, Any] = {
        "input": input_requests,
        "endpoint": endpoint,
        "completion_window": "24h",
    }
    if model:
        payload["model"] = model
    if metadata:
        payload["metadata"] = metadata

    resp = await client.post(url, json=payload, timeout=request_timeout)
    resp.raise_for_status()
    return resp.json()


async def poll_batch(
    client: httpx.AsyncClient,
    url: str,
    batch_id: str,
    *,
    poll_interval: float = 2.0,
    timeout: float = 600.0,
    request_timeout: float = 300.0,
) -> dict[str, Any]:
    """Poll batch status until terminal state. Returns final batch object."""
    terminal_states = {"completed", "failed", "cancelled", "expired"}
    start = time.monotonic()

    while True:
        resp = await client.get(
            f"{url}/{batch_id}", timeout=request_timeout,
        )
        resp.raise_for_status()
        batch = resp.json()

        if batch.get("status") in terminal_states:
            return batch

        elapsed = time.monotonic() - start
        if elapsed >= timeout:
            raise TimeoutError(
                f"Batch {batch_id} did not complete within {timeout}s "
                f"(current status: {batch.get('status')})"
            )

        await asyncio.sleep(poll_interval)


def _compute_batch_metrics(
    batch_obj: dict[str, Any],
    submit_time: float,
) -> BatchRequestResult:
    """Extract metrics from a completed batch object."""
    result = BatchRequestResult()
    result.batch_id = batch_obj.get("id", "")
    result.status = batch_obj.get("status", "unknown")

    counts = batch_obj.get("request_counts", {})
    result.total_requests = counts.get("total", 0)
    result.completed_requests = counts.get("completed", 0)
    result.failed_requests = counts.get("failed", 0)

    created_at = batch_obj.get("created_at", 0)
    in_progress_at = batch_obj.get("in_progress_at", 0)
    completed_at = batch_obj.get("completed_at", 0)

    if in_progress_at and created_at:
        result.queue_time_ms = (in_progress_at - created_at) * 1000.0
    if completed_at and in_progress_at:
        result.processing_time_ms = (completed_at - in_progress_at) * 1000.0
    if completed_at and created_at:
        result.total_time_ms = (completed_at - created_at) * 1000.0

    result.success = result.status == "completed"
    if result.status == "failed":
        errors = batch_obj.get("errors", {}).get("data", [])
        if errors:
            result.error = errors[0].get("message", "batch failed")
        else:
            result.error = "batch failed"

    result.results = batch_obj.get("results", [])
    return result


async def run_batch_benchmark(
    args: Namespace, base_url: str,
) -> tuple[dict[str, Any], BenchmarkResult]:
    """Execute a batch inference benchmark.

    Submits prompts as a batch job, polls for completion, and collects metrics.
    """
    from xpyd_bench.bench.runner import _generate_random_prompts

    # Generate prompts
    num_prompts = args.num_prompts
    input_len = args.input_len
    output_len = args.output_len
    model = args.model or ""
    seed = args.seed

    dataset_path = getattr(args, "dataset_path", None)
    if dataset_path:
        from xpyd_bench.datasets.loader import load_dataset

        entries = load_dataset(
            path=dataset_path,
            num_prompts=num_prompts,
            input_len=input_len,
            output_len=output_len,
            seed=seed,
        )
        prompts = [e.prompt for e in entries]
        num_prompts = len(prompts)
    else:
        prompts = _generate_random_prompts(num_prompts, input_len, seed)

    # Build per-request payloads
    is_chat = "chat" in getattr(args, "batch_endpoint", "/v1/completions")
    request_bodies = []
    for prompt in prompts:
        if is_chat:
            body: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": output_len,
            }
        else:
            body = {
                "model": model,
                "prompt": prompt,
                "max_tokens": output_len,
            }
        request_bodies.append(body)

    # Headers
    headers: dict[str, str] = {}
    api_key = getattr(args, "api_key", None)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    custom_headers = getattr(args, "custom_headers", None) or {}
    headers.update(custom_headers)

    poll_interval = getattr(args, "poll_interval", 2.0) or 2.0
    batch_timeout = getattr(args, "batch_timeout", 600.0) or 600.0
    request_timeout = getattr(args, "timeout", 300.0) or 300.0
    batch_endpoint = getattr(args, "batch_endpoint", "/v1/completions")

    batches_url = f"{base_url}/v1/batches"

    result = BenchmarkResult(
        backend="openai",
        base_url=base_url,
        endpoint="/v1/batch",
        model=model,
        num_prompts=num_prompts,
        input_len=input_len,
        output_len=output_len,
        environment=collect_env_info(),
    )

    overall_start = time.perf_counter()

    async with httpx.AsyncClient(headers=headers) as client:
        # Submit batch
        submit_start = time.perf_counter()
        batch_obj = await submit_batch(
            client,
            batches_url,
            request_bodies,
            model=model,
            endpoint=batch_endpoint,
            request_timeout=request_timeout,
        )
        batch_id = batch_obj["id"]

        if not getattr(args, "disable_tqdm", False):
            print(f"Batch submitted: {batch_id} ({num_prompts} requests)")

        # Poll for completion
        final_batch = await poll_batch(
            client,
            batches_url,
            batch_id,
            poll_interval=poll_interval,
            timeout=batch_timeout,
            request_timeout=request_timeout,
        )

    overall_end = time.perf_counter()
    result.total_duration_s = overall_end - overall_start

    # Compute batch-specific metrics
    batch_metrics = _compute_batch_metrics(final_batch, submit_start)

    # Convert to RequestResults for compatibility
    result.completed = batch_metrics.completed_requests
    result.failed = batch_metrics.failed_requests

    # Build output dict
    result_dict: dict[str, Any] = {
        "backend": result.backend,
        "base_url": result.base_url,
        "endpoint": result.endpoint,
        "model": result.model,
        "num_prompts": result.num_prompts,
        "total_duration_s": result.total_duration_s,
        "completed": result.completed,
        "failed": result.failed,
        "batch_id": batch_metrics.batch_id,
        "batch_status": batch_metrics.status,
        "queue_time_ms": batch_metrics.queue_time_ms,
        "processing_time_ms": batch_metrics.processing_time_ms,
        "total_batch_time_ms": batch_metrics.total_time_ms,
    }
    if result.environment:
        result_dict["environment"] = result.environment

    if not getattr(args, "disable_tqdm", False):
        print("=" * 60)
        print("Batch Benchmark Results")
        print("=" * 60)
        print(f"  Batch ID:              {batch_metrics.batch_id}")
        print(f"  Status:                {batch_metrics.status}")
        print(f"  Completed:             {batch_metrics.completed_requests}")
        print(f"  Failed:                {batch_metrics.failed_requests}")
        print(f"  Queue time:            {batch_metrics.queue_time_ms:.2f} ms")
        print(f"  Processing time:       {batch_metrics.processing_time_ms:.2f} ms")
        print(f"  Total batch time:      {batch_metrics.total_time_ms:.2f} ms")
        print(f"  Total duration:        {result.total_duration_s:.2f} s")
        print("=" * 60)

    return result_dict, result
