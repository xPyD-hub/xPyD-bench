"""Distributed worker server (M32).

A lightweight HTTP server that accepts benchmark tasks from the coordinator,
executes them locally, and returns results.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import numpy as np
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from xpyd_bench.distributed.protocol import (
    HeartbeatResponse,
    WorkerResult,
    WorkerTask,
)


class WorkerApp:
    """Starlette-based worker that accepts and executes benchmark tasks."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8089) -> None:
        self.host = host
        self.port = port
        self._start_time = time.monotonic()
        self._current_task_id: str | None = None
        self._status = "ok"
        self.app = Starlette(
            routes=[
                Route("/healthz", self._healthz, methods=["GET"]),
                Route("/heartbeat", self._heartbeat, methods=["GET"]),
                Route("/run", self._run_task, methods=["POST"]),
            ],
        )

    async def _healthz(self, request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def _heartbeat(self, request: Request) -> JSONResponse:
        hb = HeartbeatResponse(
            worker_url=f"http://{self.host}:{self.port}",
            status=self._status,
            current_task_id=self._current_task_id,
            uptime_s=time.monotonic() - self._start_time,
        )
        return JSONResponse(hb.to_dict())

    async def _run_task(self, request: Request) -> JSONResponse:
        """Execute a benchmark task and return results."""
        body = await request.json()
        task = WorkerTask.from_dict(body)
        self._current_task_id = task.task_id
        self._status = "busy"
        try:
            result = await _execute_task(task)
            result.worker_url = f"http://{self.host}:{self.port}"
            return JSONResponse(result.to_dict())
        except Exception as exc:
            return JSONResponse(
                WorkerResult(
                    task_id=task.task_id,
                    worker_url=f"http://{self.host}:{self.port}",
                    error=str(exc),
                ).to_dict(),
                status_code=500,
            )
        finally:
            self._current_task_id = None
            self._status = "ok"

    def run(self) -> None:
        """Run the worker server (blocking)."""
        import uvicorn

        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")


async def _execute_task(task: WorkerTask) -> WorkerResult:
    """Execute a single benchmark task: send prompts to the target and collect metrics."""
    results: list[dict[str, Any]] = []
    completed = 0
    failed = 0
    total_input_tokens = 0
    total_output_tokens = 0
    ttfts: list[float] = []
    tpots: list[float] = []
    e2els: list[float] = []

    headers: dict[str, str] = dict(task.custom_headers)
    if task.api_key:
        headers["Authorization"] = f"Bearer {task.api_key}"

    is_chat = "chat" in task.endpoint
    stream = task.stream if task.stream is not None else is_chat

    bench_start = time.perf_counter()

    async with httpx.AsyncClient(timeout=task.timeout) as client:
        for prompt_entry in task.prompts:
            prompt_text = prompt_entry.get("prompt", "")
            prompt_tokens = prompt_entry.get("prompt_len", 0)
            req_start = time.perf_counter()
            ttft_ms = None
            completion_tokens = 0

            try:
                if is_chat:
                    payload: dict[str, Any] = {
                        "model": task.model,
                        "messages": [{"role": "user", "content": prompt_text}],
                        "max_tokens": task.output_len,
                        "stream": stream,
                    }
                else:
                    payload = {
                        "model": task.model,
                        "prompt": prompt_text,
                        "max_tokens": task.output_len,
                        "stream": stream,
                    }
                payload.update(task.sampling_params)

                url = f"{task.base_url}{task.endpoint}"

                if stream:
                    first_token_time = None
                    async with client.stream(
                        "POST", url, json=payload, headers=headers
                    ) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            data = line[6:].strip()
                            if data == "[DONE]":
                                break
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                                ttft_ms = (first_token_time - req_start) * 1000
                            completion_tokens += 1
                else:
                    resp = await client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    resp_data = resp.json()
                    if is_chat:
                        choices = resp_data.get("choices", [])
                        if choices:
                            content = choices[0].get("message", {}).get("content", "")
                            completion_tokens = len(content.split())
                    else:
                        choices = resp_data.get("choices", [])
                        if choices:
                            text = choices[0].get("text", "")
                            completion_tokens = len(text.split())

                req_end = time.perf_counter()
                latency_ms = (req_end - req_start) * 1000
                e2els.append(latency_ms)
                if ttft_ms is not None:
                    ttfts.append(ttft_ms)
                if completion_tokens > 1 and latency_ms > 0:
                    tpot = latency_ms / completion_tokens
                    tpots.append(tpot)

                total_input_tokens += prompt_tokens
                total_output_tokens += completion_tokens
                completed += 1

                results.append({
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "ttft_ms": ttft_ms,
                    "latency_ms": latency_ms,
                    "success": True,
                })

            except Exception as exc:
                failed += 1
                results.append({
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": 0,
                    "latency_ms": (time.perf_counter() - req_start) * 1000,
                    "success": False,
                    "error": str(exc),
                })

    total_duration = time.perf_counter() - bench_start

    def _percentile(vals: list[float], p: float) -> float | None:
        if not vals:
            return None
        return float(np.percentile(vals, p))

    return WorkerResult(
        task_id=task.task_id,
        worker_url="",
        completed=completed,
        failed=failed,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_duration_s=total_duration,
        request_throughput=completed / total_duration if total_duration > 0 else 0,
        output_throughput=(
            total_output_tokens / total_duration if total_duration > 0 else 0
        ),
        mean_ttft_ms=float(np.mean(ttfts)) if ttfts else None,
        mean_tpot_ms=float(np.mean(tpots)) if tpots else None,
        mean_e2el_ms=float(np.mean(e2els)) if e2els else None,
        p50_e2el_ms=_percentile(e2els, 50),
        p90_e2el_ms=_percentile(e2els, 90),
        p99_e2el_ms=_percentile(e2els, 99),
        p50_ttft_ms=_percentile(ttfts, 50),
        p90_ttft_ms=_percentile(ttfts, 90),
        p99_ttft_ms=_percentile(ttfts, 99),
        requests=results,
    )
