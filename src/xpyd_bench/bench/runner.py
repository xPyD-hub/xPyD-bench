"""Benchmark runner — async HTTP client with Poisson scheduling."""

from __future__ import annotations

import asyncio
import json
import random
import time
from argparse import Namespace
from typing import Any

import httpx
import numpy as np

from xpyd_bench.bench.models import BenchmarkResult, RequestResult

# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------


def _generate_random_prompts(num: int, input_len: int, seed: int) -> list[str]:
    """Generate random prompts of approximately *input_len* tokens."""
    rng = random.Random(seed)
    # ~4 chars per token is a rough estimate
    words = [
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dog",
        "a",
        "an",
        "is",
        "was",
        "hello",
        "world",
        "benchmark",
        "test",
        "data",
        "model",
        "request",
        "response",
        "token",
        "stream",
        "latency",
        "throughput",
    ]
    prompts: list[str] = []
    for _ in range(num):
        # Approx input_len tokens → input_len words
        prompt = " ".join(rng.choice(words) for _ in range(input_len))
        prompts.append(prompt)
    return prompts


# ---------------------------------------------------------------------------
# Request scheduling (Poisson / Gamma)
# ---------------------------------------------------------------------------


def _generate_intervals(
    num: int,
    rate: float,
    burstiness: float,
    seed: int,
) -> list[float]:
    """Return list of inter-arrival times in seconds.

    * rate=inf  → all zeros (fire at once)
    * burstiness=1.0 → Poisson process (exponential intervals)
    * burstiness!=1.0 → Gamma distribution
    """
    if rate == float("inf"):
        return [0.0] * num

    rng = np.random.RandomState(seed)
    mean_interval = 1.0 / rate

    if burstiness == 1.0:
        # Poisson process → exponential inter-arrivals
        intervals = rng.exponential(mean_interval, size=num).tolist()
    else:
        # Gamma distribution: shape=burstiness, scale=mean/burstiness
        shape = burstiness
        scale = mean_interval / burstiness
        intervals = rng.gamma(shape, scale, size=num).tolist()

    return intervals


# ---------------------------------------------------------------------------
# Single-request sender
# ---------------------------------------------------------------------------


async def _send_request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    is_streaming: bool,
) -> RequestResult:
    """Send one request and collect metrics."""
    result = RequestResult()
    start = time.perf_counter()

    try:
        if is_streaming:
            result = await _send_streaming(client, url, payload, start)
        else:
            resp = await client.post(url, json=payload, timeout=300.0)
            resp.raise_for_status()
            end = time.perf_counter()
            result.latency_ms = (end - start) * 1000.0

            body = resp.json()
            usage = body.get("usage", {})
            result.prompt_tokens = usage.get("prompt_tokens", 0)
            result.completion_tokens = usage.get("completion_tokens", 0)
            if result.completion_tokens > 0:
                result.tpot_ms = result.latency_ms / result.completion_tokens

    except Exception as exc:  # noqa: BLE001
        end = time.perf_counter()
        result.latency_ms = (end - start) * 1000.0
        result.success = False
        result.error = str(exc)

    return result


async def _send_streaming(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    start: float,
) -> RequestResult:
    """Send a streaming request, measure TTFT / ITL."""
    result = RequestResult()
    payload["stream"] = True
    first_token_time: float | None = None
    last_token_time: float = start
    token_count = 0

    try:
        async with client.stream("POST", url, json=payload, timeout=300.0) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str.strip() == "[DONE]":
                    break

                now = time.perf_counter()
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Check if this chunk has content
                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0]
                # completions endpoint uses "text", chat uses "delta.content"
                text = delta.get("text") or (delta.get("delta") or {}).get("content")
                if not text:
                    continue

                token_count += 1
                if first_token_time is None:
                    first_token_time = now
                    result.ttft_ms = (now - start) * 1000.0
                else:
                    itl = (now - last_token_time) * 1000.0
                    result.itl_ms.append(itl)
                last_token_time = now

    except Exception as exc:  # noqa: BLE001
        result.success = False
        result.error = str(exc)

    end = time.perf_counter()
    result.latency_ms = (end - start) * 1000.0
    result.completion_tokens = token_count
    if token_count > 1 and first_token_time is not None:
        generation_time_ms = (last_token_time - first_token_time) * 1000.0
        result.tpot_ms = generation_time_ms / (token_count - 1)

    return result


# ---------------------------------------------------------------------------
# Build request payload
# ---------------------------------------------------------------------------


def _build_payload(
    args: Namespace,
    prompt: str,
    is_chat: bool,
) -> dict[str, Any]:
    """Build the JSON body for a single request."""
    payload: dict[str, Any] = {"max_tokens": args.output_len}

    if args.model:
        payload["model"] = args.model

    if is_chat:
        payload["messages"] = [{"role": "user", "content": prompt}]
    else:
        payload["prompt"] = prompt

    # Sampling parameters
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    if args.top_p is not None:
        payload["top_p"] = args.top_p
    if args.top_k is not None:
        payload["top_k"] = args.top_k
    if args.frequency_penalty is not None:
        payload["frequency_penalty"] = args.frequency_penalty
    if args.presence_penalty is not None:
        payload["presence_penalty"] = args.presence_penalty
    if args.best_of is not None:
        payload["best_of"] = args.best_of
    if args.use_beam_search:
        payload["use_beam_search"] = True
    if args.logprobs is not None:
        payload["logprobs"] = args.logprobs
    if args.ignore_eos:
        payload["ignore_eos"] = True
    if getattr(args, "stop", None) is not None:
        payload["stop"] = args.stop
    if getattr(args, "n", None) is not None:
        payload["n"] = args.n
    if getattr(args, "api_seed", None) is not None:
        payload["seed"] = args.api_seed

    return payload


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def _compute_metrics(result: BenchmarkResult) -> None:
    """Compute aggregate metrics from per-request results."""
    successful = [r for r in result.requests if r.success]
    result.completed = len(successful)
    result.failed = len(result.requests) - result.completed

    if not successful:
        return

    result.total_input_tokens = sum(r.prompt_tokens for r in successful)
    result.total_output_tokens = sum(r.completion_tokens for r in successful)

    if result.total_duration_s > 0:
        result.request_throughput = result.completed / result.total_duration_s
        result.output_throughput = result.total_output_tokens / result.total_duration_s
        result.total_token_throughput = (
            result.total_input_tokens + result.total_output_tokens
        ) / result.total_duration_s

    def _set_percentiles(
        values: list[float], prefix: str, target: BenchmarkResult
    ) -> None:
        """Compute and set mean/median/p50/p90/p95/p99 on *target*."""
        if not values:
            return
        setattr(target, f"mean_{prefix}", float(np.mean(values)))
        setattr(target, f"median_{prefix}", float(np.median(values)))
        setattr(target, f"p50_{prefix}", float(np.percentile(values, 50)))
        setattr(target, f"p90_{prefix}", float(np.percentile(values, 90)))
        setattr(target, f"p95_{prefix}", float(np.percentile(values, 95)))
        setattr(target, f"p99_{prefix}", float(np.percentile(values, 99)))

    # E2E latency
    e2els = [r.latency_ms for r in successful]
    _set_percentiles(e2els, "e2el_ms", result)

    # TTFT
    ttfts = [r.ttft_ms for r in successful if r.ttft_ms is not None]
    _set_percentiles(ttfts, "ttft_ms", result)

    # TPOT
    tpots = [r.tpot_ms for r in successful if r.tpot_ms is not None]
    _set_percentiles(tpots, "tpot_ms", result)

    # ITL
    all_itls = [itl for r in successful for itl in r.itl_ms]
    _set_percentiles(all_itls, "itl_ms", result)


async def run_benchmark(args: Namespace, base_url: str) -> tuple[dict, BenchmarkResult]:
    """Execute the benchmark and return (result_dict, BenchmarkResult)."""
    is_chat = "chat" in args.endpoint
    is_streaming = is_chat  # streaming by default for chat endpoint
    url = f"{base_url}{args.endpoint}"

    # Generate or load prompts
    dataset_path = getattr(args, "dataset_path", None)
    if dataset_path:
        from xpyd_bench.datasets.loader import load_dataset, validate_and_report

        entries = load_dataset(
            path=dataset_path,
            num_prompts=args.num_prompts,
            input_len=args.input_len,
            output_len=args.output_len,
            input_len_dist=getattr(args, "synthetic_input_len_dist", "fixed"),
            output_len_dist=getattr(args, "synthetic_output_len_dist", "fixed"),
            seed=args.seed,
        )
        validate_and_report(entries, dataset_path)
        prompts = [e.prompt for e in entries]
        # Override num_prompts to match actual dataset size
        args.num_prompts = len(prompts)
    else:
        prompts = _generate_random_prompts(args.num_prompts, args.input_len, args.seed)

    # Generate inter-arrival intervals
    rate_pattern = getattr(args, "rate_pattern", None)
    if rate_pattern and isinstance(rate_pattern, dict):
        from xpyd_bench.bench.rate_patterns import generate_pattern_intervals

        intervals = generate_pattern_intervals(args.num_prompts, rate_pattern, args.seed)
    else:
        intervals = _generate_intervals(
            args.num_prompts, args.request_rate, args.burstiness, args.seed
        )

    # Concurrency limiter
    semaphore = asyncio.Semaphore(args.max_concurrency) if args.max_concurrency else None

    result = BenchmarkResult(
        backend=args.backend,
        base_url=base_url,
        endpoint=args.endpoint,
        model=args.model or "",
        num_prompts=args.num_prompts,
        request_rate=args.request_rate,
        max_concurrency=args.max_concurrency,
        input_len=args.input_len,
        output_len=args.output_len,
    )

    async def _task(client: httpx.AsyncClient, prompt: str) -> RequestResult:
        if semaphore:
            async with semaphore:
                return await _send_request(
                    client, url, _build_payload(args, prompt, is_chat), is_streaming
                )
        return await _send_request(
            client, url, _build_payload(args, prompt, is_chat), is_streaming
        )

    # Rich progress bar
    use_rich = getattr(args, "rich_progress", False) and not args.disable_tqdm
    reporter = None
    if use_rich:
        from xpyd_bench.reporting.rich_output import RichProgressReporter

        reporter = RichProgressReporter(total=args.num_prompts)
        reporter.start()

    tasks: list[asyncio.Task] = []
    overall_start = time.perf_counter()

    async def _tracked_task(client: httpx.AsyncClient, prompt: str) -> RequestResult:
        r = await _task(client, prompt)
        if reporter:
            reporter.advance(success=r.success)
        return r

    async with httpx.AsyncClient() as client:
        for i, prompt in enumerate(prompts):
            if i > 0:
                await asyncio.sleep(intervals[i])
            task = asyncio.create_task(_tracked_task(client, prompt))
            tasks.append(task)

            if not use_rich and not args.disable_tqdm and (i + 1) % 100 == 0:
                print(f"  Launched {i + 1}/{args.num_prompts} requests...")

        # Wait for all to complete
        results_list = await asyncio.gather(*tasks)

    overall_end = time.perf_counter()
    if reporter:
        reporter.stop()

    result.total_duration_s = overall_end - overall_start
    result.requests = list(results_list)

    _compute_metrics(result)

    if reporter:
        reporter.print_summary_table(result)
    else:
        _print_summary(result)

    return _to_dict(result), result


def _print_summary(r: BenchmarkResult) -> None:
    """Print human-readable benchmark summary (plain text fallback)."""
    print("=" * 60)
    print("Benchmark Results")
    print("=" * 60)
    print(f"  Completed:             {r.completed}")
    print(f"  Failed:                {r.failed}")
    print(f"  Total duration:        {r.total_duration_s:.2f} s")
    print(f"  Request throughput:    {r.request_throughput:.2f} req/s")
    print(f"  Output throughput:     {r.output_throughput:.2f} tok/s")
    print(f"  Total tok throughput:  {r.total_token_throughput:.2f} tok/s")
    print()
    for label, prefix in [("TTFT", "ttft"), ("TPOT", "tpot"), ("ITL", "itl"), ("E2EL", "e2el")]:
        mean = getattr(r, f"mean_{prefix}_ms")
        p50 = getattr(r, f"p50_{prefix}_ms")
        p90 = getattr(r, f"p90_{prefix}_ms")
        p95 = getattr(r, f"p95_{prefix}_ms")
        p99 = getattr(r, f"p99_{prefix}_ms")
        print(
            f"  {label:5s}  mean={mean:8.2f}  P50={p50:8.2f}  "
            f"P90={p90:8.2f}  P95={p95:8.2f}  P99={p99:8.2f} ms"
        )
    print("=" * 60)


def _to_dict(r: BenchmarkResult) -> dict:
    """Convert BenchmarkResult to a JSON-serializable dict."""
    d: dict[str, Any] = {
        "backend": r.backend,
        "base_url": r.base_url,
        "endpoint": r.endpoint,
        "model": r.model,
        "num_prompts": r.num_prompts,
        "request_rate": r.request_rate,
        "max_concurrency": r.max_concurrency,
        "input_len": r.input_len,
        "output_len": r.output_len,
        "total_duration_s": r.total_duration_s,
        "completed": r.completed,
        "failed": r.failed,
        "total_input_tokens": r.total_input_tokens,
        "total_output_tokens": r.total_output_tokens,
        "request_throughput": r.request_throughput,
        "output_throughput": r.output_throughput,
        "total_token_throughput": r.total_token_throughput,
    }
    for prefix in ("ttft", "tpot", "itl", "e2el"):
        for stat in ("mean", "median", "p50", "p90", "p95", "p99"):
            key = f"{stat}_{prefix}_ms"
            d[key] = getattr(r, key)
    return d
