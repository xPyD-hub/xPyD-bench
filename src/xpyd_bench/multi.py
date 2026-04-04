"""Multi-endpoint benchmark comparison.

Runs the same workload against multiple endpoints and produces
side-by-side comparison results.
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
from xpyd_bench.compare import (
    ComparisonResult,
    compare,
)


@dataclass
class MultiEndpointResult:
    """Results from benchmarking multiple endpoints."""

    endpoints: list[str] = field(default_factory=list)
    results: list[BenchmarkResult] = field(default_factory=list)
    raw_dicts: list[dict[str, Any]] = field(default_factory=list)
    comparisons: list[ComparisonResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-friendly dict."""
        d: dict[str, Any] = {
            "endpoints": self.endpoints,
            "results": self.raw_dicts,
        }
        if self.comparisons:
            d["comparisons"] = [c.to_dict() for c in self.comparisons]
        return d


async def run_multi_benchmark(
    args: Namespace,
    endpoints: list[str],
    threshold_pct: float = 5.0,
) -> MultiEndpointResult:
    """Run benchmarks against multiple endpoints sequentially.

    The same prompts and parameters are used for each endpoint to ensure
    fair comparison. The first endpoint is treated as the baseline for
    pairwise regression detection.

    Args:
        args: Parsed CLI arguments (same as single-endpoint bench).
        endpoints: List of base URLs to benchmark.
        threshold_pct: Regression detection threshold.

    Returns:
        MultiEndpointResult with per-endpoint results and comparisons.
    """
    multi = MultiEndpointResult(endpoints=list(endpoints))

    for ep in endpoints:
        ep_args = deepcopy(args)
        # Override base_url-related args
        ep_args.base_url = ep
        result_dict, bench_result = await run_benchmark(ep_args, ep)
        multi.results.append(bench_result)
        multi.raw_dicts.append(result_dict)

    # Pairwise comparison: first endpoint is baseline
    if len(multi.raw_dicts) >= 2:
        baseline = multi.raw_dicts[0]
        for candidate in multi.raw_dicts[1:]:
            cmp = compare(baseline, candidate, threshold_pct=threshold_pct)
            multi.comparisons.append(cmp)

    return multi


def format_multi_summary(multi: MultiEndpointResult) -> str:
    """Format a side-by-side summary table for multiple endpoints."""
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("  xPyD-bench — Multi-Endpoint Comparison")
    lines.append("=" * 100)

    if not multi.results:
        lines.append("  No results.")
        return "\n".join(lines)

    # Header
    ep_labels = [f"EP{i}" for i in range(len(multi.endpoints))]
    header_parts = [f"{'Metric':<28s}"]
    for i, ep in enumerate(multi.endpoints):
        label = f"{ep_labels[i]} ({ep})"
        header_parts.append(f"{label:>20s}")
    lines.append("  " + " ".join(header_parts))
    lines.append("  " + "-" * (28 + 21 * len(multi.endpoints)))

    # Key metrics to display
    display_metrics = [
        ("completed", "completed"),
        ("failed", "failed"),
        ("total_duration_s", "total_duration_s"),
        ("request_throughput", "request_throughput"),
        ("output_throughput", "output_throughput"),
        ("total_token_throughput", "total_token_throughput"),
        ("mean_ttft_ms", "mean_ttft_ms"),
        ("p50_ttft_ms", "p50_ttft_ms"),
        ("p90_ttft_ms", "p90_ttft_ms"),
        ("p99_ttft_ms", "p99_ttft_ms"),
        ("mean_tpot_ms", "mean_tpot_ms"),
        ("p50_tpot_ms", "p50_tpot_ms"),
        ("p90_tpot_ms", "p90_tpot_ms"),
        ("p99_tpot_ms", "p99_tpot_ms"),
        ("mean_e2el_ms", "mean_e2el_ms"),
        ("p50_e2el_ms", "p50_e2el_ms"),
        ("p90_e2el_ms", "p90_e2el_ms"),
        ("p99_e2el_ms", "p99_e2el_ms"),
    ]

    for label, attr in display_metrics:
        row = [f"{label:<28s}"]
        for r in multi.results:
            val = getattr(r, attr, 0)
            if isinstance(val, float):
                row.append(f"{val:>20.2f}")
            else:
                row.append(f"{val!s:>20s}")
        lines.append("  " + " ".join(row))

    lines.append("  " + "-" * (28 + 21 * len(multi.endpoints)))

    # Regression summary
    for i, cmp in enumerate(multi.comparisons):
        ep_idx = i + 1
        if cmp.has_regression:
            lines.append(
                f"  ⚠️  EP{ep_idx} ({multi.endpoints[ep_idx]}) "
                f"has regressions vs EP0 ({multi.endpoints[0]})"
            )
        else:
            lines.append(
                f"  ✅  EP{ep_idx} ({multi.endpoints[ep_idx]}) "
                f"— no regressions vs EP0"
            )

    lines.append("=" * 100)
    return "\n".join(lines)


def export_multi_json(multi: MultiEndpointResult, path: str | Path) -> Path:
    """Export multi-endpoint results to JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(multi.to_dict(), f, indent=2, default=str)
    return p


def format_multi_markdown(multi: MultiEndpointResult) -> str:
    """Format multi-endpoint results as a Markdown table."""
    if not multi.results:
        return "No results.\n"

    display_metrics = [
        "completed",
        "failed",
        "request_throughput",
        "output_throughput",
        "mean_ttft_ms",
        "p50_ttft_ms",
        "p90_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "p50_tpot_ms",
        "p90_tpot_ms",
        "p99_tpot_ms",
        "mean_e2el_ms",
        "p50_e2el_ms",
        "p90_e2el_ms",
        "p99_e2el_ms",
    ]

    headers = ["Metric"] + [ep for ep in multi.endpoints]
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in headers) + " |"

    rows = [header_line, sep_line]
    for metric in display_metrics:
        cells = [metric]
        for r in multi.results:
            val = getattr(r, metric, 0)
            if isinstance(val, float):
                cells.append(f"{val:.2f}")
            else:
                cells.append(str(val))
        rows.append("| " + " | ".join(cells) + " |")

    return "\n".join(rows) + "\n"


def export_multi_markdown(multi: MultiEndpointResult, path: str | Path) -> Path:
    """Export multi-endpoint Markdown table to a file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(format_multi_markdown(multi))
    return p
