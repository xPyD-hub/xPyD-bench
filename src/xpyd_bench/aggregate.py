"""Result aggregation across multiple benchmark runs (M25).

Usage:
    xpyd-bench aggregate result1.json result2.json [--output agg.json] [--min-runs N]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Metrics to aggregate from BenchmarkResult JSON
_METRIC_KEYS: list[str] = [
    "mean_ttft_ms",
    "p50_ttft_ms",
    "p90_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p50_tpot_ms",
    "p90_tpot_ms",
    "p95_tpot_ms",
    "p99_tpot_ms",
    "mean_e2el_ms",
    "p50_e2el_ms",
    "p90_e2el_ms",
    "p95_e2el_ms",
    "p99_e2el_ms",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
]


@dataclass
class MetricStats:
    """Statistical summary for a single metric across runs."""

    mean: float = 0.0
    stddev: float = 0.0
    min: float = 0.0
    max: float = 0.0
    cv: float = 0.0  # coefficient of variation


@dataclass
class AggregateResult:
    """Aggregated statistics across multiple benchmark runs."""

    num_runs: int = 0
    metrics: dict[str, MetricStats] = field(default_factory=dict)
    outlier_runs: list[int] = field(default_factory=list)  # 0-based indices

    def to_dict(self) -> dict:
        """Serialize to plain dict for JSON output."""
        return {
            "num_runs": self.num_runs,
            "outlier_run_indices": self.outlier_runs,
            "metrics": {
                k: {
                    "mean": round(v.mean, 4),
                    "stddev": round(v.stddev, 4),
                    "min": round(v.min, 4),
                    "max": round(v.max, 4),
                    "cv": round(v.cv, 4),
                }
                for k, v in self.metrics.items()
            },
        }


def _compute_stats(values: list[float]) -> MetricStats:
    """Compute statistical summary for a list of values."""
    arr = np.array(values, dtype=np.float64)
    mean = float(np.mean(arr))
    stddev = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    cv = stddev / mean if mean != 0 else 0.0
    return MetricStats(
        mean=mean,
        stddev=stddev,
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        cv=cv,
    )


def _detect_outliers(values: list[float]) -> list[int]:
    """Return indices of values >2 stddev from mean."""
    if len(values) < 3:
        return []
    arr = np.array(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    if std == 0:
        return []
    return [i for i, v in enumerate(values) if abs(v - mean) > 2 * std]


def aggregate_results(results: list[dict]) -> AggregateResult:
    """Aggregate multiple BenchmarkResult dicts into statistical summary."""
    agg = AggregateResult(num_runs=len(results))

    for key in _METRIC_KEYS:
        values = [r.get(key, 0.0) for r in results]
        # Skip metrics that are all zero
        if all(v == 0.0 for v in values):
            continue
        agg.metrics[key] = _compute_stats(values)

    # Detect outlier runs based on request_throughput
    tp_values = [r.get("request_throughput", 0.0) for r in results]
    if not all(v == 0.0 for v in tp_values):
        agg.outlier_runs = _detect_outliers(tp_values)

    return agg


def _print_table(agg: AggregateResult) -> None:
    """Print human-readable summary table."""
    print(f"\n{'='*80}")
    print(f"Aggregate Summary ({agg.num_runs} runs)")
    print(f"{'='*80}")
    header = f"{'Metric':<30} {'Mean':>10} {'Stddev':>10} {'Min':>10} {'Max':>10} {'CV':>8}"
    print(header)
    print("-" * 80)
    for key, stats in agg.metrics.items():
        cv_flag = " ⚠" if stats.cv > 0.15 else ""
        print(
            f"{key:<30} {stats.mean:>10.2f} {stats.stddev:>10.2f} "
            f"{stats.min:>10.2f} {stats.max:>10.2f} {stats.cv:>7.2%}{cv_flag}"
        )
    if agg.outlier_runs:
        print(f"\n⚠ Outlier runs detected (indices): {agg.outlier_runs}")
    print()


def aggregate_main(argv: list[str] | None = None) -> None:
    """CLI entry point for aggregate subcommand."""
    parser = argparse.ArgumentParser(
        prog="xpyd-bench aggregate",
        description="Aggregate metrics across multiple benchmark result files.",
    )
    parser.add_argument(
        "results",
        nargs="+",
        help="Paths to benchmark result JSON files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Save aggregated summary to JSON file",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=2,
        help="Minimum number of runs required (default: 2)",
    )

    args = parser.parse_args(argv)

    if len(args.results) < args.min_runs:
        print(
            f"Error: need at least {args.min_runs} result files, got {len(args.results)}.",
            file=sys.stderr,
        )
        sys.exit(1)

    results: list[dict] = []
    for path_str in args.results:
        path = Path(path_str)
        if not path.exists():
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        with open(path) as f:
            results.append(json.load(f))

    agg = aggregate_results(results)
    _print_table(agg)

    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w") as f:
            json.dump(agg.to_dict(), f, indent=2)
        print(f"Saved aggregated summary to {out_path}")
