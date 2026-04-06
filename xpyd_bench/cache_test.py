"""Prefix caching impact analysis (M87).

Measures the performance impact of prefix caching by comparing TTFT and
throughput between requests that share a common prefix vs requests with
unique prefixes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import string
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx


def _random_text(n_words: int) -> str:
    """Generate random text with *n_words* words."""
    return " ".join(
        "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 10)))
        for _ in range(n_words)
    )


def generate_shared_prefix_prompts(
    num_prompts: int,
    prefix_word_count: int = 200,
    suffix_word_count: int = 30,
    shared_prefix_ratio: float = 0.5,
) -> tuple[list[str], list[str]]:
    """Return (shared_prompts, unique_prompts).

    *shared_prompts* all start with the same prefix text followed by a unique
    suffix.  *unique_prompts* have fully unique text of comparable length.
    """
    if not 0.0 <= shared_prefix_ratio <= 1.0:
        raise ValueError(f"shared_prefix_ratio must be 0.0-1.0, got {shared_prefix_ratio}")

    shared_words = max(1, int(prefix_word_count * shared_prefix_ratio))
    unique_suffix_words = prefix_word_count - shared_words + suffix_word_count

    common_prefix = _random_text(shared_words)

    shared_prompts = [
        common_prefix + " " + _random_text(unique_suffix_words)
        for _ in range(num_prompts)
    ]

    total_words = prefix_word_count + suffix_word_count
    unique_prompts = [_random_text(total_words) for _ in range(num_prompts)]

    return shared_prompts, unique_prompts


@dataclass
class CacheTestMetrics:
    """Metrics for one group of requests."""

    label: str = ""
    num_requests: int = 0
    mean_ttft_ms: float = 0.0
    p50_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    mean_latency_ms: float = 0.0
    throughput_rps: float = 0.0
    errors: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "num_requests": self.num_requests,
            "mean_ttft_ms": round(self.mean_ttft_ms, 2),
            "p50_ttft_ms": round(self.p50_ttft_ms, 2),
            "p99_ttft_ms": round(self.p99_ttft_ms, 2),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "throughput_rps": round(self.throughput_rps, 4),
            "errors": self.errors,
        }


@dataclass
class CacheAnalysis:
    """Full cache-test result."""

    shared: CacheTestMetrics = field(default_factory=CacheTestMetrics)
    unique: CacheTestMetrics = field(default_factory=CacheTestMetrics)
    ttft_improvement_pct: float = 0.0
    throughput_improvement_pct: float = 0.0
    shared_prefix_ratio: float = 0.5
    base_url: str = ""
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_analysis": {
                "shared_prefix_ratio": self.shared_prefix_ratio,
                "base_url": self.base_url,
                "model": self.model,
                "shared": self.shared.to_dict(),
                "unique": self.unique.to_dict(),
                "ttft_improvement_pct": round(self.ttft_improvement_pct, 2),
                "throughput_improvement_pct": round(self.throughput_improvement_pct, 2),
            }
        }


def _percentile(values: list[float], pct: float) -> float:
    """Simple percentile without numpy dependency."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct / 100.0
    f = int(k)
    c = f + 1 if f + 1 < len(s) else f
    d = k - f
    return s[f] + d * (s[c] - s[f])


async def _send_one(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompt: str,
    api_key: str | None,
    timeout: float,
) -> tuple[float, float, bool]:
    """Send a single streaming chat request. Returns (ttft_ms, latency_ms, success)."""
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": 50,
    }

    t0 = time.perf_counter()
    ttft: float | None = None
    try:
        async with client.stream(
            "POST", url, json=payload, headers=headers, timeout=timeout
        ) as resp:
            resp.raise_for_status()
            async for _line in resp.aiter_lines():
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000.0
        latency = (time.perf_counter() - t0) * 1000.0
        return (ttft or latency, latency, True)
    except Exception:
        latency = (time.perf_counter() - t0) * 1000.0
        return (latency, latency, False)


async def _run_group(
    base_url: str,
    model: str,
    prompts: list[str],
    label: str,
    api_key: str | None = None,
    timeout: float = 120.0,
    concurrency: int = 4,
) -> CacheTestMetrics:
    """Run a group of requests and collect metrics."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    ttfts: list[float] = []
    latencies: list[float] = []
    errors = 0

    sem = asyncio.Semaphore(concurrency)

    async def _task(prompt: str) -> None:
        nonlocal errors
        async with sem:
            async with httpx.AsyncClient() as client:
                ttft, lat, ok = await _send_one(client, url, model, prompt, api_key, timeout)
                if ok:
                    ttfts.append(ttft)
                    latencies.append(lat)
                else:
                    errors += 1

    t0 = time.perf_counter()
    await asyncio.gather(*[_task(p) for p in prompts])
    wall_time = time.perf_counter() - t0

    n = len(ttfts)
    return CacheTestMetrics(
        label=label,
        num_requests=len(prompts),
        mean_ttft_ms=sum(ttfts) / n if n else 0.0,
        p50_ttft_ms=_percentile(ttfts, 50),
        p99_ttft_ms=_percentile(ttfts, 99),
        mean_latency_ms=sum(latencies) / n if n else 0.0,
        throughput_rps=n / wall_time if wall_time > 0 else 0.0,
        errors=errors,
    )


async def run_cache_test(
    base_url: str,
    model: str,
    num_prompts: int = 20,
    shared_prefix_ratio: float = 0.5,
    api_key: str | None = None,
    timeout: float = 120.0,
    concurrency: int = 4,
) -> CacheAnalysis:
    """Execute prefix-caching impact analysis."""
    shared_prompts, unique_prompts = generate_shared_prefix_prompts(
        num_prompts=num_prompts,
        shared_prefix_ratio=shared_prefix_ratio,
    )

    print(f"Running {num_prompts} shared-prefix requests (ratio={shared_prefix_ratio})...")
    shared_metrics = await _run_group(
        base_url, model, shared_prompts, "shared-prefix",
        api_key=api_key, timeout=timeout, concurrency=concurrency,
    )

    print(f"Running {num_prompts} unique-prefix requests...")
    unique_metrics = await _run_group(
        base_url, model, unique_prompts, "unique-prefix",
        api_key=api_key, timeout=timeout, concurrency=concurrency,
    )

    ttft_imp = 0.0
    if unique_metrics.mean_ttft_ms > 0:
        ttft_imp = (
            (unique_metrics.mean_ttft_ms - shared_metrics.mean_ttft_ms)
            / unique_metrics.mean_ttft_ms
            * 100.0
        )

    thr_imp = 0.0
    if unique_metrics.throughput_rps > 0:
        thr_imp = (
            (shared_metrics.throughput_rps - unique_metrics.throughput_rps)
            / unique_metrics.throughput_rps
            * 100.0
        )

    return CacheAnalysis(
        shared=shared_metrics,
        unique=unique_metrics,
        ttft_improvement_pct=ttft_imp,
        throughput_improvement_pct=thr_imp,
        shared_prefix_ratio=shared_prefix_ratio,
        base_url=base_url,
        model=model,
    )


def format_cache_test_summary(result: CacheAnalysis) -> str:
    """Format a human-readable summary."""
    lines = [
        "Prefix Caching Impact Analysis",
        "=" * 50,
        f"  Base URL:             {result.base_url}",
        f"  Model:                {result.model}",
        f"  Shared prefix ratio:  {result.shared_prefix_ratio}",
        "",
        f"  {'Metric':<25} {'Shared':>12} {'Unique':>12} {'Delta':>10}",
        f"  {'-' * 25} {'-' * 12} {'-' * 12} {'-' * 10}",
    ]

    def _row(label: str, shared: float, unique: float, unit: str = "ms") -> str:
        delta = shared - unique
        sign = "+" if delta >= 0 else ""
        return (
            f"  {label:<25} {shared:>10.2f}{unit:>2}"
            f" {unique:>10.2f}{unit:>2} {sign}{delta:>7.2f}{unit}"
        )

    lines.append(_row("Mean TTFT", result.shared.mean_ttft_ms, result.unique.mean_ttft_ms))
    lines.append(_row("P50 TTFT", result.shared.p50_ttft_ms, result.unique.p50_ttft_ms))
    lines.append(_row("P99 TTFT", result.shared.p99_ttft_ms, result.unique.p99_ttft_ms))
    lines.append(_row("Mean Latency", result.shared.mean_latency_ms, result.unique.mean_latency_ms))
    lines.append(
        _row("Throughput", result.shared.throughput_rps, result.unique.throughput_rps, "/s")
    )

    lines.append("")
    sign = "+" if result.ttft_improvement_pct >= 0 else ""
    lines.append(f"  TTFT improvement:       {sign}{result.ttft_improvement_pct:.1f}%")
    sign = "+" if result.throughput_improvement_pct >= 0 else ""
    lines.append(f"  Throughput improvement:  {sign}{result.throughput_improvement_pct:.1f}%")

    return "\n".join(lines)


def cache_test_main(argv: list[str] | None = None) -> None:
    """Entry point for ``xpyd-bench cache-test`` subcommand."""
    import os

    parser = argparse.ArgumentParser(
        prog="xpyd-bench cache-test",
        description="Analyze prefix caching impact on TTFT and throughput.",
    )
    parser.add_argument("--base-url", required=True, help="Server base URL")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument(
        "--shared-prefix-ratio",
        type=float,
        default=0.5,
        help="Ratio of shared prefix (0.0-1.0, default 0.5)",
    )
    parser.add_argument(
        "--cache-test-prompts",
        type=int,
        default=20,
        help="Number of prompts per group (default 20)",
    )
    parser.add_argument("--api-key", default=None, help="API key for auth")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout")
    parser.add_argument("--concurrency", type=int, default=4, help="Max concurrent requests")
    parser.add_argument("--json-output", default=None, help="Save results as JSON")

    args = parser.parse_args(argv)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")

    result = asyncio.run(
        run_cache_test(
            base_url=args.base_url,
            model=args.model,
            num_prompts=args.cache_test_prompts,
            shared_prefix_ratio=args.shared_prefix_ratio,
            api_key=api_key,
            timeout=args.timeout,
            concurrency=args.concurrency,
        )
    )

    print()
    print(format_cache_test_summary(result))

    if args.json_output:
        p = Path(args.json_output)
        p.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"\nResults saved to {p}")
