"""Request Payload Size Impact Analysis (M100).

Sweep across prompt sizes and measure TTFT, TPOT, throughput at each level.
Detect scaling behavior and recommend optimal prompt sizes.

Usage:
    xpyd-bench size-impact --base-url <url> --model <model>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from xpyd_bench.bench.runner import run_benchmark

DEFAULT_SIZE_LEVELS = [10, 100, 500, 1000, 2000, 4000]
_PROBE_NUM_PROMPTS = 20


@dataclass
class SizeProbe:
    """Result of a single probing run at a given prompt size."""

    prompt_tokens: int
    mean_ttft_ms: float | None
    p99_ttft_ms: float | None
    mean_tpot_ms: float | None
    p99_tpot_ms: float | None
    throughput_rps: float
    throughput_tps: float
    mean_e2el_ms: float | None
    error_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "mean_ttft_ms": _r(self.mean_ttft_ms),
            "p99_ttft_ms": _r(self.p99_ttft_ms),
            "mean_tpot_ms": _r(self.mean_tpot_ms),
            "p99_tpot_ms": _r(self.p99_tpot_ms),
            "throughput_rps": round(self.throughput_rps, 2),
            "throughput_tps": round(self.throughput_tps, 2),
            "mean_e2el_ms": _r(self.mean_e2el_ms),
            "error_rate": round(self.error_rate, 4),
        }


def _r(v: float | None) -> float | None:
    return round(v, 2) if v is not None else None


@dataclass
class ScalingAnalysis:
    """Scaling behavior analysis."""

    behaviour: str  # "linear", "sublinear", "superlinear", "constant", "unknown"
    slope: float | None  # ms per token (TTFT regression slope)
    inflection_points: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "behaviour": self.behaviour,
            "slope_ms_per_token": _r(self.slope),
            "inflection_points": self.inflection_points,
        }


@dataclass
class SizeImpactResult:
    """Aggregated size impact results."""

    probes: list[SizeProbe] = field(default_factory=list)
    scaling: ScalingAnalysis | None = None
    recommended_max_prompt_tokens: int | None = None
    target_latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "probes": [p.to_dict() for p in self.probes],
        }
        if self.scaling:
            d["scaling"] = self.scaling.to_dict()
        if self.recommended_max_prompt_tokens is not None:
            d["recommended_max_prompt_tokens"] = self.recommended_max_prompt_tokens
        if self.target_latency_ms is not None:
            d["target_latency_ms"] = self.target_latency_ms
        return d


def _make_probe_args(
    base_args: argparse.Namespace,
    prompt_tokens: int,
) -> argparse.Namespace:
    """Create a modified args namespace for a single size probe."""
    args = deepcopy(base_args)
    args.input_len = prompt_tokens
    args.num_prompts = _PROBE_NUM_PROMPTS
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
    prompt_tokens: int,
) -> SizeProbe:
    """Run a benchmark probe at a given prompt size."""
    probe_args = _make_probe_args(args, prompt_tokens)
    _result_dict, bench = await run_benchmark(probe_args, base_url)

    total = bench.completed + bench.failed
    if total == 0:
        total = len(bench.requests)
    error_rate = bench.failed / total if total > 0 else 0.0

    return SizeProbe(
        prompt_tokens=prompt_tokens,
        mean_ttft_ms=bench.mean_ttft_ms,
        p99_ttft_ms=bench.p99_ttft_ms,
        mean_tpot_ms=bench.mean_tpot_ms,
        p99_tpot_ms=bench.p99_tpot_ms,
        throughput_rps=bench.request_throughput,
        throughput_tps=bench.total_token_throughput,
        mean_e2el_ms=bench.mean_e2el_ms,
        error_rate=error_rate,
    )


def parse_size_levels(spec: str) -> list[int]:
    """Parse size levels from a comma-separated string.

    Supports plain integers and range notation start:end:step.
    """
    levels: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if ":" in part:
            pieces = part.split(":")
            if len(pieces) == 3:
                start, end, step = int(pieces[0]), int(pieces[1]), int(pieces[2])
                levels.extend(range(start, end + 1, step))
            elif len(pieces) == 2:
                start, end = int(pieces[0]), int(pieces[1])
                levels.extend(range(start, end + 1, max(1, (end - start) // 5)))
            else:
                raise ValueError(f"Invalid range notation: {part}")
        else:
            levels.append(int(part))
    return sorted(set(levels))


def detect_scaling(probes: list[SizeProbe]) -> ScalingAnalysis:
    """Detect scaling behavior from TTFT vs prompt size.

    Uses log-log linear regression to determine the exponent:
      - exponent ≈ 1 → linear
      - exponent < 0.8 → sublinear
      - exponent > 1.2 → superlinear
    """
    # Filter probes that have valid TTFT
    valid = [
        (p.prompt_tokens, p.mean_ttft_ms)
        for p in probes
        if p.mean_ttft_ms and p.mean_ttft_ms > 0
    ]

    if len(valid) < 2:
        return ScalingAnalysis(behaviour="unknown", slope=None)

    # Simple linear regression on (size, ttft) for slope
    xs = [v[0] for v in valid]
    ys = [v[1] for v in valid]
    n = len(xs)
    sx = sum(xs)
    sy = sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sxx = sum(x * x for x in xs)

    denom = n * sxx - sx * sx
    if denom == 0:
        return ScalingAnalysis(behaviour="constant", slope=0.0)

    slope = (n * sxy - sx * sy) / denom

    # Log-log regression for exponent
    log_valid = [(v[0], v[1]) for v in valid if v[0] > 0 and v[1] > 0]
    if len(log_valid) >= 2:
        lxs = [math.log(v[0]) for v in log_valid]
        lys = [math.log(v[1]) for v in log_valid]
        ln = len(lxs)
        lsx = sum(lxs)
        lsy = sum(lys)
        lsxy = sum(x * y for x, y in zip(lxs, lys))
        lsxx = sum(x * x for x in lxs)
        ldenom = ln * lsxx - lsx * lsx
        if ldenom != 0:
            exponent = (ln * lsxy - lsx * lsy) / ldenom
        else:
            exponent = 1.0
    else:
        exponent = 1.0

    if exponent < 0.8:
        behaviour = "sublinear"
    elif exponent > 1.2:
        behaviour = "superlinear"
    else:
        behaviour = "linear"

    # Detect inflection points: where the rate of increase jumps significantly
    inflection_points: list[int] = []
    if len(valid) >= 3:
        rates = []
        for i in range(1, len(valid)):
            dx = valid[i][0] - valid[i - 1][0]
            dy = valid[i][1] - valid[i - 1][1]
            rates.append(dy / dx if dx != 0 else 0.0)
        for i in range(1, len(rates)):
            if rates[i - 1] != 0 and abs(rates[i] / rates[i - 1]) > 2.0:
                inflection_points.append(valid[i + 1][0])

    return ScalingAnalysis(
        behaviour=behaviour,
        slope=slope,
        inflection_points=inflection_points,
    )


def recommend_max_size(
    probes: list[SizeProbe],
    target_latency_ms: float,
) -> int | None:
    """Find the largest prompt size where mean TTFT is within target."""
    candidates = [
        p.prompt_tokens
        for p in probes
        if p.mean_ttft_ms is not None and p.mean_ttft_ms <= target_latency_ms
    ]
    return max(candidates) if candidates else None


async def run_size_impact(
    base_args: argparse.Namespace,
    base_url: str,
    size_levels: list[int],
    target_latency_ms: float | None = None,
) -> SizeImpactResult:
    """Run the size impact sweep."""
    probes: list[SizeProbe] = []
    for size in size_levels:
        probe = await _run_probe(base_args, base_url, size)
        probes.append(probe)

    scaling = detect_scaling(probes)
    rec = None
    if target_latency_ms is not None:
        rec = recommend_max_size(probes, target_latency_ms)

    return SizeImpactResult(
        probes=probes,
        scaling=scaling,
        recommended_max_prompt_tokens=rec,
        target_latency_ms=target_latency_ms,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xpyd-bench size-impact",
        description="Analyze request payload size impact on inference performance.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        required=True,
        help="Server base URL.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name.",
    )
    parser.add_argument(
        "--size-levels",
        type=str,
        default=None,
        help="Comma-separated prompt sizes or range notation start:end:step "
        "(default: 10,100,500,1000,2000,4000).",
    )
    parser.add_argument(
        "--target-latency-ms",
        type=float,
        default=None,
        help="Target TTFT latency in ms for max-size recommendation.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        help="Output length for probes (default: 128).",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint path.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=False,
        help="Use streaming responses for TTFT measurement.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        dest="json_output",
        help="Output raw JSON.",
    )
    return parser


def _make_base_namespace(args: argparse.Namespace) -> argparse.Namespace:
    """Build a minimal namespace compatible with run_benchmark."""
    ns = argparse.Namespace()
    # Required by bench runner
    ns.backend = "openai"
    ns.backend_plugin = None
    ns.list_backends = False
    ns.base_url = args.base_url
    ns.host = "127.0.0.1"
    ns.port = 8000
    ns.endpoint = args.endpoint
    ns.model = args.model
    ns.stream = args.stream
    ns.no_stream = not args.stream
    ns.request_rate = float("inf")
    ns.max_concurrency = None
    ns.output_len = args.output_len
    ns.input_len = 256  # will be overridden per probe
    ns.num_prompts = _PROBE_NUM_PROMPTS
    ns.dataset = None
    ns.dataset_name = None
    ns.dataset_path = None
    ns.random_input_len = None
    ns.random_output_len = None
    ns.random_prefix_len = None
    ns.config = None
    ns.scenario = None
    ns.list_scenarios = False
    ns.duration = None
    ns.seed = None
    ns.timeout = 300
    ns.retries = 0
    ns.retry_delay = 1.0
    ns.api_key = args.api_key
    ns.api_key_file = None
    # Disable outputs
    ns.save_result = None
    ns.html_report = None
    ns.csv_report = None
    ns.markdown_report = None
    ns.junit_xml = None
    ns.debug_log = None
    ns.webhook_url = None
    ns.otlp_endpoint = None
    ns.prometheus_export = None
    ns.heatmap_export = None
    ns.no_live = True
    ns.warmup = 0
    ns.verbose = False
    ns.percentile_metrics = "ttft,tpot,itl,e2el"
    ns.metric_percentiles = "p50,p90,p95,p99"
    ns.goodput_config = None
    ns.sonnet_prefix = None
    ns.sharegpt_output_len = None
    # Tags / notes
    ns.tag = None
    ns.note = None
    ns.tokenizer = None
    ns.tokenizer_mode = "auto"
    ns.trust_remote_code = False
    # Load shedding / advanced fields that runner might check
    ns.adaptive_rate = False
    ns.max_error_rate = None
    ns.abort_on_error_pct = None
    ns.reproducibility = None
    return ns


def size_impact_main(argv: list[str] | None = None) -> None:
    """CLI entry point for size-impact subcommand."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    size_levels = (
        parse_size_levels(args.size_levels)
        if args.size_levels
        else list(DEFAULT_SIZE_LEVELS)
    )

    base_ns = _make_base_namespace(args)

    result = asyncio.run(
        run_size_impact(
            base_ns,
            args.base_url,
            size_levels,
            target_latency_ms=args.target_latency_ms,
        )
    )

    if args.json_output:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        _print_report(result)


def _print_report(result: SizeImpactResult) -> None:
    """Print a human-readable report."""
    print("\n=== Request Payload Size Impact Analysis ===\n")
    header = (
        f"{'Tokens':>8} | {'TTFT(ms)':>10} | {'P99 TTFT':>10} | "
        f"{'TPOT(ms)':>10} | {'Tput(r/s)':>10} | {'Tput(t/s)':>10} | {'Errors':>8}"
    )
    print(header)
    print("-" * len(header))
    for p in result.probes:
        ttft = f"{p.mean_ttft_ms:.1f}" if p.mean_ttft_ms is not None else "N/A"
        p99 = f"{p.p99_ttft_ms:.1f}" if p.p99_ttft_ms is not None else "N/A"
        tpot = f"{p.mean_tpot_ms:.1f}" if p.mean_tpot_ms is not None else "N/A"
        print(
            f"{p.prompt_tokens:>8} | {ttft:>10} | {p99:>10} | "
            f"{tpot:>10} | {p.throughput_rps:>10.2f} | "
            f"{p.throughput_tps:>10.2f} | {p.error_rate:>7.2%}"
        )

    if result.scaling:
        s = result.scaling
        print(f"\nScaling behavior: {s.behaviour}")
        if s.slope is not None:
            print(f"TTFT slope: {s.slope:.4f} ms/token")
        if s.inflection_points:
            print(f"Inflection points: {s.inflection_points} tokens")

    if result.recommended_max_prompt_tokens is not None:
        print(
            f"\nRecommended max prompt size for {result.target_latency_ms}ms "
            f"TTFT target: {result.recommended_max_prompt_tokens} tokens"
        )
    print()
