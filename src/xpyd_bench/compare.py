"""Benchmark comparison and regression detection.

Compares two JSON result files (baseline vs candidate) and reports metric
deltas with configurable regression thresholds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Metrics where lower is better (latencies)
_LOWER_IS_BETTER = {
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
    "mean_itl_ms",
    "p50_itl_ms",
    "p90_itl_ms",
    "p95_itl_ms",
    "p99_itl_ms",
    "mean_e2el_ms",
    "p50_e2el_ms",
    "p90_e2el_ms",
    "p95_e2el_ms",
    "p99_e2el_ms",
}

# Metrics where higher is better (throughput)
_HIGHER_IS_BETTER = {
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
}

# All compared metrics
COMPARED_METRICS = sorted(_LOWER_IS_BETTER | _HIGHER_IS_BETTER)


@dataclass
class MetricDelta:
    """Comparison result for a single metric."""

    name: str
    baseline: float
    candidate: float
    delta: float  # candidate - baseline
    pct_change: float  # percentage change
    direction: str  # "improved", "regressed", "unchanged"
    regressed: bool  # True if regression exceeds threshold


@dataclass
class ComparisonResult:
    """Full comparison result."""

    metrics: list[MetricDelta]
    has_regression: bool
    threshold_pct: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        return {
            "threshold_pct": self.threshold_pct,
            "has_regression": self.has_regression,
            "metrics": [
                {
                    "name": m.name,
                    "baseline": m.baseline,
                    "candidate": m.candidate,
                    "delta": m.delta,
                    "pct_change": m.pct_change,
                    "direction": m.direction,
                    "regressed": m.regressed,
                }
                for m in self.metrics
            ],
        }


def _extract_metrics(data: dict[str, Any]) -> dict[str, float]:
    """Extract comparable metrics from a result dict.

    Supports both flat dicts and dicts with a ``summary`` key.
    """
    src = data.get("summary", data)
    result: dict[str, float] = {}
    for key in COMPARED_METRICS:
        val = src.get(key)
        if val is not None:
            try:
                result[key] = float(val)
            except (TypeError, ValueError):
                pass
    return result


def load_result(path: str | Path) -> dict[str, Any]:
    """Load a JSON benchmark result file."""
    with open(path) as f:
        return json.load(f)


def compare(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    threshold_pct: float = 5.0,
) -> ComparisonResult:
    """Compare two benchmark results and detect regressions.

    Args:
        baseline: Parsed JSON result (baseline run).
        candidate: Parsed JSON result (candidate run).
        threshold_pct: Percentage threshold for regression detection.

    Returns:
        ComparisonResult with per-metric deltas and overall regression flag.
    """
    base_m = _extract_metrics(baseline)
    cand_m = _extract_metrics(candidate)

    # Compare only metrics present in both
    common_keys = sorted(set(base_m) & set(cand_m))

    deltas: list[MetricDelta] = []
    has_regression = False

    for key in common_keys:
        bval = base_m[key]
        cval = cand_m[key]
        delta = cval - bval

        if bval == 0:
            pct = 0.0 if cval == 0 else float("inf")
        else:
            pct = (delta / abs(bval)) * 100.0

        # Determine direction
        lower_better = key in _LOWER_IS_BETTER
        if abs(pct) < 0.01:
            direction = "unchanged"
            regressed = False
        elif lower_better:
            # Lower is better: positive delta = regression
            direction = "regressed" if delta > 0 else "improved"
            regressed = delta > 0 and pct > threshold_pct
        else:
            # Higher is better: negative delta = regression
            direction = "regressed" if delta < 0 else "improved"
            regressed = delta < 0 and abs(pct) > threshold_pct

        if regressed:
            has_regression = True

        deltas.append(
            MetricDelta(
                name=key,
                baseline=bval,
                candidate=cval,
                delta=round(delta, 4),
                pct_change=round(pct, 2),
                direction=direction,
                regressed=regressed,
            )
        )

    return ComparisonResult(
        metrics=deltas,
        has_regression=has_regression,
        threshold_pct=threshold_pct,
    )


def format_comparison_table(result: ComparisonResult) -> str:
    """Format comparison as a human-readable table."""
    lines: list[str] = []
    lines.append("=" * 90)
    lines.append("  xPyD-bench — Benchmark Comparison")
    lines.append(f"  Regression threshold: {result.threshold_pct}%")
    lines.append("=" * 90)
    lines.append("")

    header = (
        f"  {'Metric':<28s} {'Baseline':>10s} {'Candidate':>10s} "
        f"{'Delta':>10s} {'Change':>8s} {'Status':>10s}"
    )
    lines.append(header)
    lines.append("  " + "-" * 86)

    for m in result.metrics:
        indicator = "↓ REGRESS" if m.regressed else m.direction
        sign = "+" if m.pct_change > 0 else ""
        lines.append(
            f"  {m.name:<28s} {m.baseline:>10.2f} {m.candidate:>10.2f} "
            f"{m.delta:>+10.2f} {sign}{m.pct_change:>6.1f}% {indicator:>10s}"
        )

    lines.append("  " + "-" * 86)
    lines.append("")

    if result.has_regression:
        lines.append("  ⚠️  REGRESSION DETECTED — one or more metrics exceeded threshold")
    else:
        lines.append("  ✅  No regressions detected")

    lines.append("=" * 90)
    return "\n".join(lines)
