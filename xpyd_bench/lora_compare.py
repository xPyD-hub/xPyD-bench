"""Multi-LoRA Endpoint Benchmarking (M89).

Benchmark the same prompts against multiple LoRA adapters on the same base
model endpoint, measuring per-adapter performance and adapter-switching
overhead via interleaved request scheduling.
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
from xpyd_bench.diff import _mann_whitney_u


@dataclass
class AdapterOverhead:
    """Adapter-switching overhead metrics."""

    sequential_mean_ttft_ms: float = 0.0
    interleaved_mean_ttft_ms: float = 0.0
    switching_overhead_ms: float = 0.0  # interleaved - sequential
    switching_overhead_pct: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequential_mean_ttft_ms": self.sequential_mean_ttft_ms,
            "interleaved_mean_ttft_ms": self.interleaved_mean_ttft_ms,
            "switching_overhead_ms": self.switching_overhead_ms,
            "switching_overhead_pct": self.switching_overhead_pct,
        }


@dataclass
class AdapterResult:
    """Per-adapter benchmark result."""

    adapter: str = ""
    result: BenchmarkResult | None = None
    raw_dict: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "result": self.raw_dict,
        }


@dataclass
class AdapterComparison:
    """Pairwise comparison between two adapters."""

    baseline_adapter: str = ""
    candidate_adapter: str = ""
    comparison: ComparisonResult | None = None
    significance: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LoRACompareResult:
    """Results from multi-LoRA adapter benchmarking."""

    base_url: str = ""
    adapters: list[str] = field(default_factory=list)
    interleave: bool = False
    adapter_results: list[AdapterResult] = field(default_factory=list)
    interleaved_results: list[AdapterResult] = field(default_factory=list)
    overhead: AdapterOverhead | None = None
    comparisons: list[AdapterComparison] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "base_url": self.base_url,
            "adapters": self.adapters,
            "interleave": self.interleave,
            "sequential_results": [ar.to_dict() for ar in self.adapter_results],
        }
        if self.interleaved_results:
            d["interleaved_results"] = [ar.to_dict() for ar in self.interleaved_results]
        if self.overhead:
            d["overhead"] = self.overhead.to_dict()
        if self.comparisons:
            d["comparisons"] = []
            for ac in self.comparisons:
                ac_dict: dict[str, Any] = {
                    "baseline_adapter": ac.baseline_adapter,
                    "candidate_adapter": ac.candidate_adapter,
                }
                if ac.comparison:
                    ac_dict["comparison"] = ac.comparison.to_dict()
                if ac.significance:
                    ac_dict["significance"] = ac.significance
                d["comparisons"].append(ac_dict)
        return d


_LATENCY_METRICS = ["ttft_ms", "tpot_ms", "latency_ms"]


def _compute_adapter_significance(
    baseline: BenchmarkResult,
    candidate: BenchmarkResult,
) -> list[dict[str, Any]]:
    """Run Mann-Whitney U test on per-request latency metrics."""
    results: list[dict[str, Any]] = []
    for metric in _LATENCY_METRICS:
        baseline_vals = [
            getattr(r, metric) for r in baseline.requests
            if getattr(r, metric, None) is not None
        ]
        candidate_vals = [
            getattr(r, metric) for r in candidate.requests
            if getattr(r, metric, None) is not None
        ]
        if len(baseline_vals) < 2 or len(candidate_vals) < 2:
            continue
        u_stat, p_value = _mann_whitney_u(baseline_vals, candidate_vals)
        results.append({
            "metric": metric,
            "u_stat": u_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
        })
    return results


def _mean_ttft(result: BenchmarkResult) -> float:
    """Compute mean TTFT from per-request data."""
    vals = [
        r.ttft_ms for r in result.requests
        if getattr(r, "ttft_ms", None) is not None
    ]
    if not vals:
        return getattr(result, "mean_ttft_ms", 0.0) or 0.0
    return sum(vals) / len(vals)


async def _run_sequential(
    args: Namespace,
    adapters: list[str],
) -> list[AdapterResult]:
    """Run benchmarks for each adapter sequentially (no interleaving)."""
    results: list[AdapterResult] = []
    for adapter in adapters:
        adapter_args = deepcopy(args)
        adapter_args.model = adapter
        raw_dict, bench_result = await run_benchmark(adapter_args, adapter_args.base_url)
        results.append(AdapterResult(
            adapter=adapter,
            result=bench_result,
            raw_dict=raw_dict,
        ))
    return results


async def _run_interleaved(
    args: Namespace,
    adapters: list[str],
) -> list[AdapterResult]:
    """Run benchmarks with round-robin adapter switching.

    Each adapter gets ``num_prompts // len(adapters)`` requests, but they
    are dispatched in round-robin order to force adapter switching on the
    server side.
    """
    results: list[AdapterResult] = []
    # We run each adapter separately but with reduced num_prompts and
    # interleaved ordering simulated by running them in alternating small
    # batches.
    num_adapters = len(adapters)
    total_prompts = getattr(args, "num_prompts", 100) or 100
    per_adapter = max(1, total_prompts // num_adapters)
    batch_size = max(1, per_adapter // num_adapters)

    # Collect all per-adapter results across batches
    adapter_dicts: dict[str, list[dict[str, Any]]] = {a: [] for a in adapters}
    adapter_benches: dict[str, BenchmarkResult | None] = {a: None for a in adapters}

    remaining = {a: per_adapter for a in adapters}
    while any(r > 0 for r in remaining.values()):
        for adapter in adapters:
            if remaining[adapter] <= 0:
                continue
            this_batch = min(batch_size, remaining[adapter])
            adapter_args = deepcopy(args)
            adapter_args.model = adapter
            adapter_args.num_prompts = this_batch
            raw_dict, bench_result = await run_benchmark(
                adapter_args, adapter_args.base_url,
            )
            adapter_dicts[adapter].append(raw_dict)
            # Keep the last result (aggregated metrics from last batch)
            adapter_benches[adapter] = bench_result
            remaining[adapter] -= this_batch

    for adapter in adapters:
        # Use the last batch result as representative
        results.append(AdapterResult(
            adapter=adapter,
            result=adapter_benches[adapter],
            raw_dict=adapter_dicts[adapter][-1] if adapter_dicts[adapter] else {},
        ))
    return results


async def run_lora_compare(
    args: Namespace,
    adapters: list[str],
    interleave: bool = False,
    threshold_pct: float = 5.0,
) -> LoRACompareResult:
    """Run multi-LoRA adapter benchmarks.

    Args:
        args: Parsed CLI arguments.
        adapters: List of model/adapter names (e.g. ``["adapter1", "adapter2", "base"]``).
        interleave: If True, also run interleaved (round-robin) benchmark
            to measure adapter-switching overhead.
        threshold_pct: Regression detection threshold.

    Returns:
        LoRACompareResult with per-adapter results and comparisons.
    """
    lora_result = LoRACompareResult(
        base_url=getattr(args, "base_url", ""),
        adapters=list(adapters),
        interleave=interleave,
    )

    # Sequential runs
    lora_result.adapter_results = await _run_sequential(args, adapters)

    # Interleaved runs (if requested)
    if interleave and len(adapters) >= 2:
        lora_result.interleaved_results = await _run_interleaved(args, adapters)

        # Compute switching overhead (compare sequential vs interleaved mean TTFT)
        seq_ttfts = [
            _mean_ttft(ar.result) for ar in lora_result.adapter_results
            if ar.result is not None
        ]
        int_ttfts = [
            _mean_ttft(ar.result) for ar in lora_result.interleaved_results
            if ar.result is not None
        ]
        if seq_ttfts and int_ttfts:
            seq_mean = sum(seq_ttfts) / len(seq_ttfts)
            int_mean = sum(int_ttfts) / len(int_ttfts)
            overhead_ms = int_mean - seq_mean
            overhead_pct = (overhead_ms / seq_mean * 100) if seq_mean > 0 else 0.0
            lora_result.overhead = AdapterOverhead(
                sequential_mean_ttft_ms=seq_mean,
                interleaved_mean_ttft_ms=int_mean,
                switching_overhead_ms=overhead_ms,
                switching_overhead_pct=overhead_pct,
            )

    # Pairwise comparisons: first adapter is baseline
    if len(lora_result.adapter_results) >= 2:
        baseline = lora_result.adapter_results[0]
        for i in range(1, len(lora_result.adapter_results)):
            candidate = lora_result.adapter_results[i]
            cmp = compare(
                baseline.raw_dict, candidate.raw_dict, threshold_pct=threshold_pct,
            )
            sig: list[dict[str, Any]] = []
            if baseline.result and candidate.result:
                sig = _compute_adapter_significance(baseline.result, candidate.result)
            lora_result.comparisons.append(AdapterComparison(
                baseline_adapter=baseline.adapter,
                candidate_adapter=candidate.adapter,
                comparison=cmp,
                significance=sig,
            ))

    return lora_result


def format_lora_compare_summary(result: LoRACompareResult) -> str:
    """Format a terminal summary for LoRA comparison results."""
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("  xPyD-bench — Multi-LoRA Adapter Comparison (M89)")
    lines.append(f"  Base URL: {result.base_url}")
    lines.append(f"  Adapters: {', '.join(result.adapters)}")
    lines.append(f"  Interleave: {'Yes' if result.interleave else 'No'}")
    lines.append("=" * 100)

    if not result.adapter_results:
        lines.append("  No results.")
        return "\n".join(lines)

    # Per-adapter metrics table
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
    ]

    header_parts = [f"{'Metric':<28s}"]
    for ar in result.adapter_results:
        header_parts.append(f"{ar.adapter:>20s}")
    lines.append("")
    lines.append("  Sequential Results")
    lines.append("  " + " ".join(header_parts))
    lines.append("  " + "-" * (28 + 21 * len(result.adapter_results)))

    for metric in display_metrics:
        row = [f"{metric:<28s}"]
        for ar in result.adapter_results:
            val = getattr(ar.result, metric, None) if ar.result else None
            if val is None:
                row.append(f"{'N/A':>20s}")
            elif isinstance(val, float):
                row.append(f"{val:>20.2f}")
            else:
                row.append(f"{val!s:>20s}")
        lines.append("  " + " ".join(row))

    # Switching overhead
    if result.overhead:
        lines.append("")
        lines.append("  Adapter Switching Overhead")
        lines.append("  " + "-" * 60)
        oh = result.overhead
        lines.append(f"  Sequential mean TTFT:   {oh.sequential_mean_ttft_ms:>10.2f} ms")
        lines.append(f"  Interleaved mean TTFT:  {oh.interleaved_mean_ttft_ms:>10.2f} ms")
        lines.append(f"  Switching overhead:     {oh.switching_overhead_ms:>10.2f} ms "
                      f"({oh.switching_overhead_pct:+.1f}%)")

    # Comparison summary
    for ac in result.comparisons:
        cmp = ac.comparison
        if cmp and cmp.has_regression:
            lines.append(
                f"  ⚠️  {ac.candidate_adapter} has regressions vs {ac.baseline_adapter}"
            )
        else:
            lines.append(
                f"  ✅  {ac.candidate_adapter} — no regressions vs {ac.baseline_adapter}"
            )

    lines.append("=" * 100)
    return "\n".join(lines)


def format_lora_compare_markdown(result: LoRACompareResult) -> str:
    """Format LoRA comparison results as Markdown."""
    if not result.adapter_results:
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
    ]

    headers = ["Metric"] + [ar.adapter for ar in result.adapter_results]
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for metric in display_metrics:
        cells = [metric]
        for ar in result.adapter_results:
            val = getattr(ar.result, metric, None) if ar.result else None
            if val is None:
                cells.append("N/A")
            elif isinstance(val, float):
                cells.append(f"{val:.2f}")
            else:
                cells.append(str(val))
        rows.append("| " + " | ".join(cells) + " |")

    if result.overhead:
        oh = result.overhead
        rows.append("")
        rows.append("### Adapter Switching Overhead")
        rows.append("")
        rows.append("| Metric | Value |")
        rows.append("| --- | --- |")
        rows.append(f"| Sequential mean TTFT | {oh.sequential_mean_ttft_ms:.2f} ms |")
        rows.append(f"| Interleaved mean TTFT | {oh.interleaved_mean_ttft_ms:.2f} ms |")
        rows.append(f"| Switching overhead | {oh.switching_overhead_ms:.2f} ms "
                     f"({oh.switching_overhead_pct:+.1f}%) |")

    rows.append("")
    return "\n".join(rows) + "\n"


def export_lora_compare_json(
    result: LoRACompareResult, path: str | Path,
) -> Path:
    """Export LoRA comparison results to JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    return p


def export_lora_compare_markdown(
    result: LoRACompareResult, path: str | Path,
) -> Path:
    """Export LoRA comparison Markdown to a file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(format_lora_compare_markdown(result))
    return p
