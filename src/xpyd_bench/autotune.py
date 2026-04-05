"""Auto-tuning optimal configuration (M98).

Automatically find optimal concurrency and request rate for maximum throughput
using binary search.

Usage:
    xpyd-bench autotune --base-url <url> --model <model>
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

import yaml

from xpyd_bench.bench.runner import run_benchmark


@dataclass
class TuneProbe:
    """Result of a single probing run at a given concurrency level."""

    concurrency: int
    throughput_rps: float
    throughput_tps: float
    mean_latency_ms: float
    p99_latency_ms: float
    error_rate: float
    total_requests: int

    def to_dict(self) -> dict[str, Any]:
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
class AutotuneResult:
    """Aggregated autotune results."""

    trajectory: list[TuneProbe] = field(default_factory=list)
    optimal_concurrency: int | None = None
    optimal_throughput_rps: float = 0.0
    optimal_throughput_tps: float = 0.0
    target: str = "throughput"
    error_budget: float = 0.01
    saturation_concurrency: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory": [p.to_dict() for p in self.trajectory],
            "optimal": {
                "concurrency": self.optimal_concurrency,
                "throughput_rps": round(self.optimal_throughput_rps, 2),
                "throughput_tps": round(self.optimal_throughput_tps, 2),
            },
            "saturation_concurrency": self.saturation_concurrency,
            "target": self.target,
            "error_budget": self.error_budget,
        }


def _make_probe_args(
    base_args: argparse.Namespace,
    concurrency: int,
    num_prompts: int,
) -> argparse.Namespace:
    """Create a modified args namespace for a single probe run."""
    args = deepcopy(base_args)
    args.max_concurrency = concurrency
    args.num_prompts = num_prompts
    # Disable features that slow down probes
    args.html_report = None
    args.csv_report = None
    args.markdown_report = None
    args.junit_xml = None
    args.save_result = None
    args.debug_log = None
    args.webhook_url = None
    args.otlp_endpoint = None
    args.prometheus_export = None
    args.heatmap_export = None
    args.no_live = True
    args.warmup = 0
    return args


async def _run_probe(
    args: argparse.Namespace,
    base_url: str,
    concurrency: int,
    num_prompts: int,
) -> TuneProbe:
    """Run a single benchmark probe at a given concurrency level."""
    probe_args = _make_probe_args(args, concurrency, num_prompts)
    result_dict, bench_result = await run_benchmark(probe_args, base_url)

    total = bench_result.completed + bench_result.failed
    if total == 0:
        total = len(bench_result.requests)
    failed = bench_result.failed
    error_rate = failed / total if total > 0 else 0.0

    return TuneProbe(
        concurrency=concurrency,
        throughput_rps=bench_result.request_throughput,
        throughput_tps=bench_result.total_token_throughput,
        mean_latency_ms=bench_result.mean_e2el_ms or 0.0,
        p99_latency_ms=bench_result.p99_e2el_ms or 0.0,
        error_rate=error_rate,
        total_requests=total,
    )


def _generate_concurrency_levels(max_concurrency: int) -> list[int]:
    """Generate exponential concurrency levels: 1, 2, 4, 8, ..., max."""
    levels = []
    c = 1
    while c <= max_concurrency:
        levels.append(c)
        c *= 2
    if levels[-1] != max_concurrency:
        levels.append(max_concurrency)
    return levels


def find_optimal(
    trajectory: list[TuneProbe],
    target: str,
    error_budget: float,
) -> tuple[int | None, float, float, int | None]:
    """Find the optimal concurrency from trajectory.

    Returns (optimal_concurrency, optimal_rps, optimal_tps, saturation_concurrency).
    """
    valid = [p for p in trajectory if p.error_rate <= error_budget]
    if not valid:
        return None, 0.0, 0.0, None

    if target == "throughput":
        best = max(valid, key=lambda p: p.throughput_rps)
    elif target == "latency":
        best = min(valid, key=lambda p: p.p99_latency_ms)
    elif target == "cost-efficiency":
        # Best throughput-per-concurrency ratio
        best = max(valid, key=lambda p: p.throughput_rps / max(p.concurrency, 1))
    else:
        best = max(valid, key=lambda p: p.throughput_rps)

    # Saturation: first concurrency where error_rate > budget
    saturation = None
    for p in trajectory:
        if p.error_rate > error_budget:
            saturation = p.concurrency
            break

    return best.concurrency, best.throughput_rps, best.throughput_tps, saturation


async def run_autotune(
    args: argparse.Namespace,
    base_url: str,
    target: str = "throughput",
    max_concurrency: int = 128,
    error_budget: float = 0.01,
    num_prompts: int = 50,
) -> AutotuneResult:
    """Run autotune: probe at exponential concurrency levels, find optimal."""
    levels = _generate_concurrency_levels(max_concurrency)
    trajectory: list[TuneProbe] = []

    for conc in levels:
        probe = await _run_probe(args, base_url, conc, num_prompts)
        trajectory.append(probe)

        # Early stop: if error rate is way over budget, no point going higher
        if probe.error_rate > error_budget * 3 and conc > 1:
            break

    opt_conc, opt_rps, opt_tps, sat_conc = find_optimal(
        trajectory, target, error_budget
    )

    return AutotuneResult(
        trajectory=trajectory,
        optimal_concurrency=opt_conc,
        optimal_throughput_rps=opt_rps,
        optimal_throughput_tps=opt_tps,
        target=target,
        error_budget=error_budget,
        saturation_concurrency=sat_conc,
    )


def generate_config(result: AutotuneResult, base_url: str, model: str) -> dict[str, Any]:
    """Generate a recommended YAML config from autotune results."""
    config: dict[str, Any] = {
        "base_url": base_url,
        "model": model,
    }
    if result.optimal_concurrency is not None:
        config["max_concurrency"] = result.optimal_concurrency
        # Set request rate slightly below throughput to avoid saturation
        config["request_rate"] = round(result.optimal_throughput_rps * 0.9, 1)
    return config


def format_autotune_summary(result: AutotuneResult) -> str:
    """Format autotune results as a human-readable summary."""
    lines = ["", "=== Auto-Tune Results ===", ""]
    lines.append(f"Target: {result.target}")
    lines.append(f"Error budget: {result.error_budget:.1%}")
    lines.append("")

    # Trajectory table
    lines.append(
        f"{'Concurrency':>12} {'RPS':>10} {'TPS':>10} "
        f"{'Mean(ms)':>10} {'P99(ms)':>10} {'ErrRate':>10}"
    )
    lines.append("-" * 72)
    for p in result.trajectory:
        marker = " *" if p.concurrency == result.optimal_concurrency else ""
        lines.append(
            f"{p.concurrency:>12} {p.throughput_rps:>10.1f} "
            f"{p.throughput_tps:>10.1f} {p.mean_latency_ms:>10.1f} "
            f"{p.p99_latency_ms:>10.1f} {p.error_rate:>10.2%}{marker}"
        )
    lines.append("")

    if result.optimal_concurrency is not None:
        lines.append(
            f"Optimal concurrency: {result.optimal_concurrency} "
            f"(throughput: {result.optimal_throughput_rps:.1f} req/s)"
        )
    else:
        lines.append("No optimal concurrency found within error budget.")

    if result.saturation_concurrency is not None:
        lines.append(f"Saturation detected at concurrency: {result.saturation_concurrency}")

    return "\n".join(lines)


def autotune_main(argv: list[str] | None = None) -> None:
    """Entry point for ``xpyd-bench autotune`` subcommand."""
    parser = argparse.ArgumentParser(
        prog="xpyd-bench autotune",
        description="Auto-tune optimal concurrency and request rate (M98).",
    )
    parser.add_argument("--base-url", required=True, help="Target server base URL")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument(
        "--autotune-target",
        choices=["throughput", "latency", "cost-efficiency"],
        default="throughput",
        help="Optimization target (default: throughput)",
    )
    parser.add_argument(
        "--autotune-max-concurrency",
        type=int,
        default=128,
        help="Maximum concurrency to test (default: 128)",
    )
    parser.add_argument(
        "--autotune-error-budget",
        type=float,
        default=0.01,
        help="Acceptable error rate during tuning (default: 0.01)",
    )
    parser.add_argument(
        "--autotune-prompts",
        type=int,
        default=50,
        help="Number of prompts per probe (default: 50)",
    )
    parser.add_argument(
        "--generate-config",
        default=None,
        help="Write optimized YAML config to path",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Write autotune results to JSON file",
    )
    # Pass-through args needed for benchmark probes
    parser.add_argument("--endpoint", default="/v1/completions")
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--request-rate", type=float, default=0)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--stream", action="store_true", default=False)

    if argv is None:
        argv = sys.argv[1:]
    args = parser.parse_args(argv)

    # Set defaults that run_benchmark expects
    _set_probe_defaults(args)

    from xpyd_bench import __version__

    print(f"xpyd-bench autotune v{__version__}")
    print(f"Target: {args.base_url}, Model: {args.model}")
    print(f"Optimization: {args.autotune_target}, Max concurrency: {args.autotune_max_concurrency}")
    print()

    result = asyncio.run(
        run_autotune(
            args,
            args.base_url,
            target=args.autotune_target,
            max_concurrency=args.autotune_max_concurrency,
            error_budget=args.autotune_error_budget,
            num_prompts=args.autotune_prompts,
        )
    )

    print(format_autotune_summary(result))

    if args.json_output:
        p = Path(args.json_output)
        p.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"\nResults saved to {p}")

    if args.generate_config:
        config = generate_config(result, args.base_url, args.model)
        p = Path(args.generate_config)
        p.write_text(yaml.dump(config, default_flow_style=False))
        print(f"Optimized config saved to {p}")

    if result.optimal_concurrency is None:
        sys.exit(1)


def _set_probe_defaults(args: argparse.Namespace) -> None:
    """Set default attribute values that run_benchmark expects."""
    defaults = {
        "num_prompts": 50,
        "max_concurrency": 1,
        "dataset": None,
        "config": None,
        "backend": "openai",
        "backend_plugin": None,
        "http2": False,
        "max_connections": 100,
        "max_keepalive": 20,
        "header": None,
        "compress": False,
        "image_url": None,
        "image_dir": None,
        "synthetic_images": 0,
        "synthetic_image_size": "64x64",
        "image_detail": "auto",
        "tools": None,
        "response_format": None,
        "multi_turn": None,
        "max_turns": None,
        "duration": None,
        "scenario": None,
        "tokenizer": None,
        "rate_algorithm": "default",
        "adaptive_concurrency": False,
        "adaptive_concurrency_target": 500,
        "no_live": True,
        "warmup": 0,
        "warmup_profile": False,
        "warmup_curve": False,
        "retries": 0,
        "retry_delay": 1.0,
        "save_result": None,
        "result_dir": None,
        "debug_log": None,
        "html_report": None,
        "csv_report": None,
        "markdown_report": None,
        "export_requests_csv": None,
        "junit_xml": None,
        "webhook_url": None,
        "webhook_secret": None,
        "otlp_endpoint": None,
        "prometheus_export": None,
        "heatmap_export": None,
        "heatmap_bucket_width": 1.0,
        "heatmap_bins": None,
        "sla": None,
        "note": None,
        "tag": None,
        "template_vars": None,
        "preset": None,
        "checkpoint_dir": None,
        "checkpoint_interval": 50,
        "validate_response": None,
        "track_ratelimits": False,
        "track_payload_size": False,
        "measure_generation_speed": False,
        "sse_metrics": False,
        "speculative_metrics": False,
        "token_cdf": False,
        "analyze_cache_savings": False,
        "cache_pricing_ratio": 0.5,
        "pacing_report": False,
        "quality_check": None,
        "consistency_check": 0,
        "deduplicate": False,
        "rolling_metrics": False,
        "rolling_window": 10,
        "rolling_step": 5,
        "workload_stats": False,
        "latency_breakdown": False,
        "inject_delay": 0,
        "inject_error_rate": 0.0,
        "inject_payload_corruption": 0.0,
        "request_id_prefix": None,
        "anomaly_threshold": 1.5,
        "max_error_rate": None,
        "max_error_rate_window": 10,
        "adaptive_timeout": False,
        "adaptive_timeout_multiplier": 3.0,
        "confidence_intervals": False,
        "confidence_level": 0.95,
        "percentiles": None,
        "priority_levels": 0,
        "repeat": 1,
        "repeat_delay": 0,
        "dry_run": False,
        "compare_baseline": None,
        "baseline_dir": None,
        "heatmap": False,
        "quiet": False,
        "verbose": False,
        "cost_model": None,
        "load_shed_threshold": None,
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)
