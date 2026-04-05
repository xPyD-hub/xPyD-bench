"""Multi-model comparison mode (M75).

Benchmarks the same prompts against multiple models on the same endpoint,
producing side-by-side metrics with statistical significance testing.
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
from xpyd_bench.diff import _mann_whitney_u


@dataclass
class ModelSignificance:
    """Statistical significance result for a metric between two models."""

    metric: str = ""
    u_stat: float = 0.0
    p_value: float = 1.0
    significant: bool = False  # p < 0.05


@dataclass
class ModelComparisonResult:
    """Pairwise comparison between two models with significance testing."""

    baseline_model: str = ""
    candidate_model: str = ""
    comparison: ComparisonResult | None = None
    significance: list[ModelSignificance] = field(default_factory=list)


@dataclass
class MultiModelResult:
    """Results from benchmarking multiple models."""

    models: list[str] = field(default_factory=list)
    base_url: str = ""
    results: list[BenchmarkResult] = field(default_factory=list)
    raw_dicts: list[dict[str, Any]] = field(default_factory=list)
    comparisons: list[ModelComparisonResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-friendly dict."""
        d: dict[str, Any] = {
            "models": self.models,
            "base_url": self.base_url,
            "results": self.raw_dicts,
        }
        if self.comparisons:
            d["comparisons"] = []
            for mc in self.comparisons:
                mc_dict: dict[str, Any] = {
                    "baseline_model": mc.baseline_model,
                    "candidate_model": mc.candidate_model,
                }
                if mc.comparison:
                    mc_dict["comparison"] = mc.comparison.to_dict()
                if mc.significance:
                    mc_dict["significance"] = [
                        {
                            "metric": s.metric,
                            "u_stat": s.u_stat,
                            "p_value": s.p_value,
                            "significant": s.significant,
                        }
                        for s in mc.significance
                    ]
                d["comparisons"].append(mc_dict)
        return d


_LATENCY_METRICS = ["ttft_ms", "tpot_ms", "latency_ms"]


def _compute_significance(
    baseline: BenchmarkResult,
    candidate: BenchmarkResult,
) -> list[ModelSignificance]:
    """Run Mann-Whitney U test on per-request latency metrics."""
    results: list[ModelSignificance] = []
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
        results.append(ModelSignificance(
            metric=metric,
            u_stat=u_stat,
            p_value=p_value,
            significant=p_value < 0.05,
        ))
    return results


async def run_model_compare(
    args: Namespace,
    models: list[str],
    threshold_pct: float = 5.0,
) -> MultiModelResult:
    """Run benchmarks against multiple models on the same endpoint.

    The same prompts and parameters are used for each model to ensure
    fair comparison. The first model is treated as the baseline for
    pairwise regression detection and significance testing.

    Args:
        args: Parsed CLI arguments.
        models: List of model names to benchmark.
        threshold_pct: Regression detection threshold.

    Returns:
        MultiModelResult with per-model results and comparisons.
    """
    multi = MultiModelResult(
        models=list(models),
        base_url=getattr(args, "base_url", ""),
    )

    for model in models:
        model_args = deepcopy(args)
        model_args.model = model
        result_dict, bench_result = await run_benchmark(model_args, model_args.base_url)
        multi.results.append(bench_result)
        multi.raw_dicts.append(result_dict)

    # Pairwise comparison: first model is baseline
    if len(multi.results) >= 2:
        baseline_dict = multi.raw_dicts[0]
        baseline_result = multi.results[0]
        for i in range(1, len(multi.results)):
            cmp = compare(
                baseline_dict, multi.raw_dicts[i], threshold_pct=threshold_pct,
            )
            sig = _compute_significance(baseline_result, multi.results[i])
            multi.comparisons.append(ModelComparisonResult(
                baseline_model=models[0],
                candidate_model=models[i],
                comparison=cmp,
                significance=sig,
            ))

    return multi


def format_model_compare_summary(multi: MultiModelResult) -> str:
    """Format a side-by-side summary table for multiple models."""
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("  xPyD-bench — Multi-Model Comparison")
    lines.append(f"  Base URL: {multi.base_url}")
    lines.append("=" * 100)

    if not multi.results:
        lines.append("  No results.")
        return "\n".join(lines)

    # Header
    header_parts = [f"{'Metric':<28s}"]
    for model in multi.models:
        header_parts.append(f"{model:>20s}")
    lines.append("  " + " ".join(header_parts))
    lines.append("  " + "-" * (28 + 21 * len(multi.models)))

    display_metrics = [
        "completed",
        "failed",
        "total_duration_s",
        "request_throughput",
        "output_throughput",
        "total_token_throughput",
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

    for metric in display_metrics:
        row = [f"{metric:<28s}"]
        for r in multi.results:
            val = getattr(r, metric, None)
            if val is None:
                row.append(f"{'N/A':>20s}")
            elif isinstance(val, float):
                row.append(f"{val:>20.2f}")
            else:
                row.append(f"{val!s:>20s}")
        lines.append("  " + " ".join(row))

    lines.append("  " + "-" * (28 + 21 * len(multi.models)))

    # Regression and significance summary
    for mc in multi.comparisons:
        cmp = mc.comparison
        if cmp and cmp.has_regression:
            lines.append(
                f"  ⚠️  {mc.candidate_model} has regressions vs {mc.baseline_model}"
            )
        else:
            lines.append(
                f"  ✅  {mc.candidate_model} — no regressions vs {mc.baseline_model}"
            )
        for sig in mc.significance:
            marker = "***" if sig.significant else ""
            lines.append(
                f"      {sig.metric}: U={sig.u_stat:.1f}, "
                f"p={sig.p_value:.4f} {marker}"
            )

    lines.append("=" * 100)
    return "\n".join(lines)


def format_model_compare_markdown(multi: MultiModelResult) -> str:
    """Format multi-model results as a Markdown table."""
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

    headers = ["Metric"] + list(multi.models)
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in headers) + " |"

    rows = [header_line, sep_line]
    for metric in display_metrics:
        cells = [metric]
        for r in multi.results:
            val = getattr(r, metric, None)
            if val is None:
                cells.append("N/A")
            elif isinstance(val, float):
                cells.append(f"{val:.2f}")
            else:
                cells.append(str(val))
        rows.append("| " + " | ".join(cells) + " |")

    # Significance section
    rows.append("")
    rows.append("### Statistical Significance")
    rows.append("")
    for mc in multi.comparisons:
        rows.append(
            f"**{mc.candidate_model} vs {mc.baseline_model}**"
        )
        if mc.significance:
            rows.append("| Metric | U-stat | p-value | Significant |")
            rows.append("| --- | --- | --- | --- |")
            for sig in mc.significance:
                rows.append(
                    f"| {sig.metric} | {sig.u_stat:.1f} | "
                    f"{sig.p_value:.4f} | {'Yes' if sig.significant else 'No'} |"
                )
        else:
            rows.append("Not enough data for significance testing.")
        rows.append("")

    return "\n".join(rows) + "\n"


def export_model_compare_json(
    multi: MultiModelResult, path: str | Path,
) -> Path:
    """Export multi-model results to JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(multi.to_dict(), f, indent=2, default=str)
    return p


def export_model_compare_markdown(
    multi: MultiModelResult, path: str | Path,
) -> Path:
    """Export multi-model Markdown table to a file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(format_model_compare_markdown(multi))
    return p
