"""Streaming vs Non-Streaming Overhead Analysis (M76).

Runs the same prompts in both streaming and non-streaming modes on the same
endpoint, then reports the overhead of streaming (TTFT delta, total latency
delta, throughput impact).
"""

from __future__ import annotations

import json
from argparse import Namespace
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xpyd_bench.bench.models import BenchmarkResult
from xpyd_bench.bench.runner import run_benchmark
from xpyd_bench.compare import ComparisonResult, compare


@dataclass
class StreamOverhead:
    """Overhead metrics comparing streaming vs non-streaming."""

    ttft_delta_ms: float | None = None  # streaming TTFT - non-streaming TTFT
    mean_latency_delta_ms: float = 0.0  # streaming mean - non-streaming mean
    p99_latency_delta_ms: float = 0.0
    throughput_delta_pct: float = 0.0  # (streaming - non_streaming) / non_streaming * 100
    streaming_beneficial: bool = False  # True if streaming has lower TTFT


@dataclass
class StreamCompareResult:
    """Results from streaming vs non-streaming comparison."""

    base_url: str = ""
    model: str = ""
    streaming_result: BenchmarkResult | None = None
    non_streaming_result: BenchmarkResult | None = None
    streaming_dict: dict[str, Any] = field(default_factory=dict)
    non_streaming_dict: dict[str, Any] = field(default_factory=dict)
    overhead: StreamOverhead = field(default_factory=StreamOverhead)
    comparison: ComparisonResult | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-friendly dict."""
        d: dict[str, Any] = {
            "base_url": self.base_url,
            "model": self.model,
            "streaming": self.streaming_dict,
            "non_streaming": self.non_streaming_dict,
            "overhead": {
                "ttft_delta_ms": self.overhead.ttft_delta_ms,
                "mean_latency_delta_ms": self.overhead.mean_latency_delta_ms,
                "p99_latency_delta_ms": self.overhead.p99_latency_delta_ms,
                "throughput_delta_pct": self.overhead.throughput_delta_pct,
                "streaming_beneficial": self.overhead.streaming_beneficial,
            },
        }
        if self.comparison:
            d["comparison"] = self.comparison.to_dict()
        return d


def _compute_overhead(
    streaming: BenchmarkResult,
    non_streaming: BenchmarkResult,
) -> StreamOverhead:
    """Compute streaming overhead from two benchmark results."""
    overhead = StreamOverhead()

    # TTFT delta
    s_ttft = getattr(streaming, "mean_ttft_ms", None)
    ns_ttft = getattr(non_streaming, "mean_ttft_ms", None)
    if s_ttft is not None and ns_ttft is not None:
        overhead.ttft_delta_ms = s_ttft - ns_ttft
        overhead.streaming_beneficial = s_ttft < ns_ttft

    # Mean latency delta
    s_lat = getattr(streaming, "mean_e2el_ms", None) or 0.0
    ns_lat = getattr(non_streaming, "mean_e2el_ms", None) or 0.0
    overhead.mean_latency_delta_ms = s_lat - ns_lat

    # P99 latency delta
    s_p99 = getattr(streaming, "p99_e2el_ms", None) or 0.0
    ns_p99 = getattr(non_streaming, "p99_e2el_ms", None) or 0.0
    overhead.p99_latency_delta_ms = s_p99 - ns_p99

    # Throughput delta
    s_tp = getattr(streaming, "request_throughput", None) or 0.0
    ns_tp = getattr(non_streaming, "request_throughput", None) or 0.0
    if ns_tp > 0:
        overhead.throughput_delta_pct = (s_tp - ns_tp) / ns_tp * 100.0

    return overhead


async def run_stream_compare(
    args: Namespace,
    threshold_pct: float = 5.0,
) -> StreamCompareResult:
    """Run benchmarks in streaming and non-streaming modes.

    Args:
        args: Parsed CLI arguments.
        threshold_pct: Regression detection threshold.

    Returns:
        StreamCompareResult with both results and overhead analysis.
    """
    result = StreamCompareResult(
        base_url=getattr(args, "base_url", ""),
        model=getattr(args, "model", ""),
    )

    # Run non-streaming first
    ns_args = deepcopy(args)
    ns_args.stream = False
    ns_dict, ns_bench = await run_benchmark(ns_args, ns_args.base_url)
    result.non_streaming_result = ns_bench
    result.non_streaming_dict = ns_dict

    # Run streaming
    s_args = deepcopy(args)
    s_args.stream = True
    s_dict, s_bench = await run_benchmark(s_args, s_args.base_url)
    result.streaming_result = s_bench
    result.streaming_dict = s_dict

    # Compute overhead (non-streaming is baseline)
    result.overhead = _compute_overhead(s_bench, ns_bench)

    # Run comparison (non-streaming as baseline)
    result.comparison = compare(ns_dict, s_dict, threshold_pct=threshold_pct)

    return result


def format_stream_compare_summary(result: StreamCompareResult) -> str:
    """Format a human-readable summary of streaming vs non-streaming comparison."""
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("  xPyD-bench — Streaming vs Non-Streaming Overhead Analysis")
    lines.append(f"  Base URL: {result.base_url}")
    lines.append(f"  Model:    {result.model}")
    lines.append("=" * 80)

    if not result.streaming_result or not result.non_streaming_result:
        lines.append("  No results.")
        return "\n".join(lines)

    sr = result.streaming_result
    nr = result.non_streaming_result

    display_metrics = [
        ("completed", "Completed Requests"),
        ("failed", "Failed Requests"),
        ("total_duration_s", "Total Duration (s)"),
        ("request_throughput", "Request Throughput (req/s)"),
        ("output_throughput", "Output Throughput (tok/s)"),
        ("mean_ttft_ms", "Mean TTFT (ms)"),
        ("p50_ttft_ms", "P50 TTFT (ms)"),
        ("p99_ttft_ms", "P99 TTFT (ms)"),
        ("mean_tpot_ms", "Mean TPOT (ms)"),
        ("mean_e2el_ms", "Mean E2E Latency (ms)"),
        ("p50_e2el_ms", "P50 E2E Latency (ms)"),
        ("p99_e2el_ms", "P99 E2E Latency (ms)"),
    ]

    header = f"  {'Metric':<30s} {'Non-Streaming':>15s} {'Streaming':>15s} {'Delta':>15s}"
    lines.append(header)
    lines.append("  " + "-" * 78)

    for attr, label in display_metrics:
        ns_val = getattr(nr, attr, None)
        s_val = getattr(sr, attr, None)
        ns_str = f"{ns_val:.2f}" if isinstance(ns_val, float) else str(ns_val or "N/A")
        s_str = f"{s_val:.2f}" if isinstance(s_val, float) else str(s_val or "N/A")
        if isinstance(ns_val, (int, float)) and isinstance(s_val, (int, float)):
            delta = s_val - ns_val
            delta_str = f"{delta:+.2f}"
        else:
            delta_str = "N/A"
        lines.append(f"  {label:<30s} {ns_str:>15s} {s_str:>15s} {delta_str:>15s}")

    lines.append("  " + "-" * 78)
    lines.append("")
    lines.append("  Overhead Summary:")
    o = result.overhead
    if o.ttft_delta_ms is not None:
        lines.append(f"    TTFT Delta:       {o.ttft_delta_ms:+.2f} ms")
    lines.append(f"    Latency Delta:    {o.mean_latency_delta_ms:+.2f} ms (mean)")
    lines.append(f"    P99 Latency Delta: {o.p99_latency_delta_ms:+.2f} ms")
    lines.append(f"    Throughput Delta:  {o.throughput_delta_pct:+.1f}%")
    if o.streaming_beneficial:
        lines.append("    ✅ Streaming has lower TTFT — beneficial for interactive use")
    else:
        lines.append("    ℹ️  Non-streaming has lower or equal TTFT")

    if result.comparison and result.comparison.has_regression:
        lines.append("    ⚠️  Regression detected in streaming mode")

    lines.append("=" * 80)
    return "\n".join(lines)


def format_stream_compare_markdown(result: StreamCompareResult) -> str:
    """Format streaming comparison as Markdown."""
    if not result.streaming_result or not result.non_streaming_result:
        return "No results.\n"

    sr = result.streaming_result
    nr = result.non_streaming_result

    display_metrics = [
        ("completed", "Completed"),
        ("failed", "Failed"),
        ("request_throughput", "Request Throughput"),
        ("output_throughput", "Output Throughput"),
        ("mean_ttft_ms", "Mean TTFT (ms)"),
        ("p50_ttft_ms", "P50 TTFT (ms)"),
        ("p99_ttft_ms", "P99 TTFT (ms)"),
        ("mean_tpot_ms", "Mean TPOT (ms)"),
        ("mean_e2el_ms", "Mean E2E Latency (ms)"),
        ("p50_e2el_ms", "P50 E2E Latency (ms)"),
        ("p99_e2el_ms", "P99 E2E Latency (ms)"),
    ]

    rows = [
        "| Metric | Non-Streaming | Streaming | Delta |",
        "| --- | --- | --- | --- |",
    ]
    for attr, label in display_metrics:
        ns_val = getattr(nr, attr, None)
        s_val = getattr(sr, attr, None)
        ns_str = f"{ns_val:.2f}" if isinstance(ns_val, float) else str(ns_val or "N/A")
        s_str = f"{s_val:.2f}" if isinstance(s_val, float) else str(s_val or "N/A")
        if isinstance(ns_val, (int, float)) and isinstance(s_val, (int, float)):
            delta_str = f"{s_val - ns_val:+.2f}"
        else:
            delta_str = "N/A"
        rows.append(f"| {label} | {ns_str} | {s_str} | {delta_str} |")

    rows.append("")
    o = result.overhead
    rows.append("### Overhead Summary")
    rows.append("")
    if o.ttft_delta_ms is not None:
        rows.append(f"- **TTFT Delta:** {o.ttft_delta_ms:+.2f} ms")
    rows.append(f"- **Mean Latency Delta:** {o.mean_latency_delta_ms:+.2f} ms")
    rows.append(f"- **P99 Latency Delta:** {o.p99_latency_delta_ms:+.2f} ms")
    rows.append(f"- **Throughput Delta:** {o.throughput_delta_pct:+.1f}%")
    if o.streaming_beneficial:
        verdict = "Streaming beneficial (lower TTFT)"
    else:
        verdict = "Non-streaming has lower or equal TTFT"
    rows.append(f"- **Verdict:** {verdict}")
    rows.append("")

    return "\n".join(rows) + "\n"


def export_stream_compare_json(
    result: StreamCompareResult, path: str | Path,
) -> Path:
    """Export stream comparison results to JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    return p


def export_stream_compare_markdown(
    result: StreamCompareResult, path: str | Path,
) -> Path:
    """Export stream comparison Markdown to a file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(format_stream_compare_markdown(result))
    return p
