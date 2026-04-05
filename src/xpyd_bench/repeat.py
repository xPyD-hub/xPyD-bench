"""Benchmark Repeat Mode (M49).

Run the same benchmark N times with optional delay between runs,
then produce an aggregated summary.
"""

from __future__ import annotations

import asyncio
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any

from xpyd_bench.aggregate import AggregateResult, aggregate_results


@dataclass
class RepeatResult:
    """Result of a repeated benchmark run."""

    num_runs: int = 0
    completed_runs: int = 0
    repeat_delay: float = 0.0
    partial: bool = False
    per_run_results: list[dict] = field(default_factory=list)
    aggregate: AggregateResult | None = None

    def to_dict(self) -> dict:
        """Serialize to plain dict for JSON output."""
        d: dict[str, Any] = {
            "repeat_runs": self.num_runs,
            "completed_runs": self.completed_runs,
            "repeat_delay": self.repeat_delay,
        }
        if self.partial:
            d["partial"] = True
        d["repeat_results"] = self.per_run_results
        if self.aggregate is not None:
            d["repeat_summary"] = self.aggregate.to_dict()
        return d


async def run_repeated_benchmark(
    args: Namespace,
    base_url: str,
) -> RepeatResult:
    """Run the benchmark multiple times and return aggregated results."""
    from xpyd_bench.bench.runner import run_benchmark

    repeat_count: int = getattr(args, "repeat", 1)
    repeat_delay: float = getattr(args, "repeat_delay", 0.0)

    rr = RepeatResult(
        num_runs=repeat_count,
        repeat_delay=repeat_delay,
    )

    interrupted = False
    for i in range(repeat_count):
        if interrupted:
            break
        print(f"\n{'='*60}")
        print(f"  Repeat run {i + 1}/{repeat_count}")
        print(f"{'='*60}")
        try:
            result_dict, _bench_result = await run_benchmark(args, base_url)
            rr.per_run_results.append(result_dict)
            rr.completed_runs += 1
        except KeyboardInterrupt:
            interrupted = True
            rr.partial = True
            break

        if repeat_delay > 0 and i < repeat_count - 1:
            print(f"\n  Waiting {repeat_delay}s before next run...")
            try:
                await asyncio.sleep(repeat_delay)
            except (KeyboardInterrupt, asyncio.CancelledError):
                interrupted = True
                rr.partial = True
                break

    if interrupted:
        rr.partial = True

    if rr.completed_runs >= 2:
        rr.aggregate = aggregate_results(rr.per_run_results)

    return rr


def print_repeat_summary(rr: RepeatResult) -> None:
    """Print a human-readable summary of repeated benchmark runs."""
    print(f"\n{'='*60}")
    print(f"  Repeat Summary ({rr.completed_runs}/{rr.num_runs} runs)")
    if rr.partial:
        print("  ⚠ Interrupted — results are partial")
    print(f"{'='*60}")

    print(f"\n{'Run':>4} {'Throughput':>12} {'Mean TTFT':>12} {'Mean Latency':>14}")
    print("-" * 46)
    for i, r in enumerate(rr.per_run_results):
        tp = r.get("request_throughput", 0.0)
        ttft = r.get("mean_ttft_ms", 0.0) or 0.0
        lat = r.get("mean_e2el_ms", 0.0) or 0.0
        print(f"{i+1:>4} {tp:>11.2f}  {ttft:>11.2f}  {lat:>13.2f}")

    if rr.aggregate is not None:
        from xpyd_bench.aggregate import _print_table

        _print_table(rr.aggregate)
    elif rr.completed_runs < 2:
        print("\n  (Need at least 2 completed runs for aggregation)")
    print()
