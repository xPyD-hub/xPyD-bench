"""Concurrency sweep mode (M44).

Automatically benchmarks across a range of concurrency levels to find
optimal throughput configuration.

Usage:
    xpyd-bench sweep --base-url <url> --concurrency-range 1,2,4,8,16,32
    xpyd-bench sweep --base-url <url> --concurrency-range 1:32:2x
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xpyd_bench.bench.runner import run_benchmark


@dataclass
class SweepLevelResult:
    """Result for a single concurrency level."""

    concurrency: int
    throughput_rps: float
    throughput_tps: float
    mean_latency_ms: float
    p99_latency_ms: float
    error_rate: float
    total_requests: int
    raw_dict: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-friendly dict."""
        return {
            "concurrency": self.concurrency,
            "throughput_rps": round(self.throughput_rps, 2),
            "throughput_tps": round(self.throughput_tps, 2),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "error_rate": round(self.error_rate, 4),
            "total_requests": self.total_requests,
        }


@dataclass
class SweepResult:
    """Aggregated sweep results across concurrency levels."""

    levels: list[SweepLevelResult] = field(default_factory=list)
    optimal_concurrency: int | None = None
    optimal_throughput_rps: float = 0.0
    max_error_rate: float = 0.05  # 5% default

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-friendly dict."""
        return {
            "levels": [lv.to_dict() for lv in self.levels],
            "optimal": {
                "concurrency": self.optimal_concurrency,
                "throughput_rps": round(self.optimal_throughput_rps, 2),
            },
            "max_error_rate_threshold": self.max_error_rate,
        }


def parse_concurrency_range(spec: str) -> list[int]:
    """Parse concurrency range specification.

    Supports:
        - Comma-separated: "1,2,4,8,16,32"
        - Exponential range: "1:32:2x" (start:stop:factor_x)
        - Linear range: "1:32:4" (start:stop:step)

    Returns:
        Sorted list of unique positive concurrency values.
    """
    if "," in spec:
        values = [int(v.strip()) for v in spec.split(",")]
    elif ":" in spec:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Range spec must be start:stop:step or start:stop:Nx, got '{spec}'"
            )
        start = int(parts[0])
        stop = int(parts[1])
        step_str = parts[2]
        if step_str.endswith("x"):
            # Exponential: multiply by factor
            factor = int(step_str[:-1])
            if factor < 2:
                raise ValueError(f"Exponential factor must be >= 2, got {factor}")
            values = []
            v = start
            while v <= stop:
                values.append(v)
                v *= factor
        else:
            step = int(step_str)
            if step < 1:
                raise ValueError(f"Step must be >= 1, got {step}")
            values = list(range(start, stop + 1, step))
    else:
        values = [int(spec)]

    values = sorted(set(v for v in values if v > 0))
    if not values:
        raise ValueError(f"No valid concurrency values from spec '{spec}'")
    return values


def _find_optimal(
    levels: list[SweepLevelResult], max_error_rate: float
) -> tuple[int | None, float]:
    """Find concurrency with max throughput under error rate threshold."""
    best_conc: int | None = None
    best_tps: float = 0.0
    for lv in levels:
        if lv.error_rate <= max_error_rate and lv.throughput_rps > best_tps:
            best_tps = lv.throughput_rps
            best_conc = lv.concurrency
    return best_conc, best_tps


async def run_sweep(
    args: argparse.Namespace,
    concurrency_levels: list[int],
    sweep_prompts: int = 100,
    max_error_rate: float = 0.05,
) -> SweepResult:
    """Run benchmarks at each concurrency level.

    Args:
        args: Parsed CLI arguments (used as template).
        concurrency_levels: List of concurrency values to test.
        sweep_prompts: Number of prompts per level.
        max_error_rate: Max error rate for optimal selection.

    Returns:
        SweepResult with per-level metrics and optimal concurrency.
    """
    sweep = SweepResult(max_error_rate=max_error_rate)

    for conc in concurrency_levels:
        level_args = deepcopy(args)
        level_args.max_concurrency = conc
        level_args.num_prompts = sweep_prompts
        # Disable live dashboard for sweep (too noisy)
        level_args.no_live = True

        print(f"\n{'='*60}")
        print(f"  Sweep: concurrency={conc}, prompts={sweep_prompts}")
        print(f"{'='*60}")

        result_dict, bench_result = await run_benchmark(level_args, args.base_url)

        # Extract metrics
        metrics = result_dict.get("metrics", {})
        e2e = metrics.get("e2e_latency", {})
        throughput_rps = metrics.get("request_throughput", 0.0)
        throughput_tps = metrics.get("output_throughput", 0.0)
        mean_lat = e2e.get("mean", 0.0)
        p99_lat = e2e.get("P99", 0.0)

        total = bench_result.completed + bench_result.failed
        error_rate = bench_result.failed / total if total > 0 else 0.0

        level_result = SweepLevelResult(
            concurrency=conc,
            throughput_rps=throughput_rps,
            throughput_tps=throughput_tps,
            mean_latency_ms=mean_lat,
            p99_latency_ms=p99_lat,
            error_rate=error_rate,
            total_requests=total,
            raw_dict=result_dict,
        )
        sweep.levels.append(level_result)

    sweep.optimal_concurrency, sweep.optimal_throughput_rps = _find_optimal(
        sweep.levels, max_error_rate
    )

    return sweep


def _print_sweep_summary(sweep: SweepResult) -> None:
    """Print a summary table of sweep results."""
    print(f"\n{'='*80}")
    print("  Concurrency Sweep Summary")
    print(f"{'='*80}")
    print(
        f"  {'Concurrency':>12}  {'RPS':>10}  {'TPS':>10}  "
        f"{'Mean(ms)':>10}  {'P99(ms)':>10}  {'Errors':>8}"
    )
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")

    for lv in sweep.levels:
        marker = " ★" if lv.concurrency == sweep.optimal_concurrency else ""
        print(
            f"  {lv.concurrency:>12}  {lv.throughput_rps:>10.2f}  "
            f"{lv.throughput_tps:>10.2f}  {lv.mean_latency_ms:>10.2f}  "
            f"{lv.p99_latency_ms:>10.2f}  {lv.error_rate:>7.1%}{marker}"
        )

    if sweep.optimal_concurrency is not None:
        print(
            f"\n  ★ Optimal: concurrency={sweep.optimal_concurrency} "
            f"({sweep.optimal_throughput_rps:.2f} req/s, "
            f"<={sweep.max_error_rate:.0%} error rate)"
        )
    else:
        print(
            f"\n  ⚠ No concurrency level met the "
            f"<={sweep.max_error_rate:.0%} error rate threshold"
        )
    print(f"{'='*80}")


def sweep_main(argv: list[str] | None = None) -> None:
    """CLI entry point for sweep subcommand."""
    parser = argparse.ArgumentParser(
        prog="xpyd-bench sweep",
        description="Concurrency sweep: benchmark at multiple concurrency levels.",
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Target server base URL.",
    )
    parser.add_argument(
        "--concurrency-range",
        required=True,
        help=(
            "Concurrency levels to test. "
            "Comma-separated (1,2,4,8) or range (1:32:2x for exponential)."
        ),
    )
    parser.add_argument(
        "--sweep-prompts",
        type=int,
        default=100,
        help="Number of prompts per concurrency level (default: 100).",
    )
    parser.add_argument(
        "--max-error-rate",
        type=float,
        default=0.05,
        help="Max error rate for optimal selection (default: 0.05).",
    )
    parser.add_argument(
        "--sweep-output",
        type=str,
        default=None,
        help="Path to save sweep results as JSON.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mock-model",
        help="Model name for benchmark requests.",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint path (default: /v1/completions).",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Request rate (default: inf for max throughput).",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=256,
        help="Input prompt length in tokens (default: 256).",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        help="Expected output length in tokens (default: 128).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config file path.",
    )

    args = parser.parse_args(argv)

    # Parse concurrency range
    try:
        levels = parse_concurrency_range(args.concurrency_range)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Set defaults expected by runner
    args.num_prompts = args.sweep_prompts
    args.max_concurrency = None  # Will be overridden per level
    args.no_live = True
    args.streaming = False
    args.burstiness = 1.0
    args.dataset = None
    args.dataset_format = None
    args.scenario = None
    args.warmup = 0
    args.timeout = 300
    args.retries = 0
    args.retry_delay = 1.0
    args.header = None
    args.compress = False
    args.request_id_prefix = None
    args.anomaly_threshold = 1.5
    args.save_result = None
    args.csv_report = None
    args.markdown_report = None
    args.export_requests_csv = None
    args.html_report = None
    args.debug_log = None
    args.result_dir = None
    args.cost_model = None
    args.sla = None
    args.heatmap = False
    args.tag = None
    args.template_vars = None
    args.backend = None
    args.backend_plugin = None
    args.tokenizer = None
    args.http2 = False
    args.max_connections = 100
    args.max_keepalive = 20
    args.prometheus_export = None
    args.metrics_ws_port = None
    args.rate_algorithm = "default"
    args.adaptive_concurrency = False
    args.adaptive_target_latency = 500.0
    args.adaptive_min_concurrency = 1
    args.adaptive_max_concurrency = 256
    args.adaptive_initial_concurrency = 8
    args.dry_run = False

    sweep_result = asyncio.run(
        run_sweep(args, levels, args.sweep_prompts, args.max_error_rate)
    )

    _print_sweep_summary(sweep_result)

    if args.sweep_output:
        output_path = Path(args.sweep_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(sweep_result.to_dict(), indent=2))
        print(f"\nSweep results saved to {output_path}")
