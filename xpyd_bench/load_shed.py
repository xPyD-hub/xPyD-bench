"""Load shedding simulation (M55).

Gradually increases request rate until the server starts rejecting requests
(429/503), then finds maximum sustainable throughput automatically.

Usage:
    xpyd-bench run --load-shed-threshold <starting_rps> ...
"""

from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from xpyd_bench.bench.runner import run_benchmark


@dataclass
class LoadShedLevel:
    """Result for a single RPS level during load shedding test."""

    rps: float
    throughput_rps: float
    mean_latency_ms: float
    p99_latency_ms: float
    error_rate: float
    rejected_count: int
    total_requests: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rps": round(self.rps, 2),
            "throughput_rps": round(self.throughput_rps, 2),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "error_rate": round(self.error_rate, 4),
            "rejected_count": self.rejected_count,
            "total_requests": self.total_requests,
        }


@dataclass
class SaturationAnalysis:
    """Full load shedding analysis result."""

    levels: list[LoadShedLevel] = field(default_factory=list)
    saturation_rps: float | None = None
    max_sustainable_rps: float | None = None
    degradation_curve: list[dict[str, float]] = field(default_factory=list)
    recovery_rps: float | None = None
    recovery_latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "levels": [lv.to_dict() for lv in self.levels],
            "saturation_rps": (
                round(self.saturation_rps, 2) if self.saturation_rps is not None else None
            ),
            "max_sustainable_rps": (
                round(self.max_sustainable_rps, 2)
                if self.max_sustainable_rps is not None
                else None
            ),
            "degradation_curve": [
                {k: round(v, 4) for k, v in pt.items()} for pt in self.degradation_curve
            ],
            "recovery_rps": (
                round(self.recovery_rps, 2) if self.recovery_rps is not None else None
            ),
            "recovery_latency_ms": (
                round(self.recovery_latency_ms, 2)
                if self.recovery_latency_ms is not None
                else None
            ),
        }


# Rejection HTTP status codes
_REJECTION_CODES = {429, 503}

# Default error rate threshold for "sustainable"
_MAX_SUSTAINABLE_ERROR_RATE = 0.05


def _count_rejections(result_dict: dict[str, Any]) -> int:
    """Count requests that got 429 or 503 responses."""
    requests = result_dict.get("requests", [])
    count = 0
    for r in requests:
        err = r.get("error", "") or ""
        # Check for status code in error string or status_code field
        status = r.get("status_code")
        if status in _REJECTION_CODES:
            count += 1
        elif any(str(code) in err for code in _REJECTION_CODES):
            count += 1
    return count


def _extract_metrics(result_dict: dict[str, Any]) -> dict[str, float]:
    """Extract key metrics from a benchmark result dict."""
    summary = result_dict.get("summary", result_dict)
    return {
        "throughput_rps": summary.get("request_throughput", 0.0),
        "mean_latency_ms": summary.get("mean_e2el_ms", 0.0),
        "p99_latency_ms": summary.get("p99_e2el_ms", 0.0),
    }


async def run_load_shed(
    args: Namespace,
    base_url: str,
    starting_rps: float,
    *,
    ramp_step: float = 0.0,
    ramp_multiplier: float = 1.5,
    prompts_per_level: int = 50,
    max_levels: int = 20,
    error_threshold: float = _MAX_SUSTAINABLE_ERROR_RATE,
    recovery_check: bool = True,
) -> SaturationAnalysis:
    """Run load shedding simulation.

    Ramps up RPS starting from *starting_rps*, multiplying by *ramp_multiplier*
    each level (or adding *ramp_step* if set).  Stops when error rate exceeds
    *error_threshold* for two consecutive levels or *max_levels* is reached.

    If *recovery_check* is True, after saturation is found, runs one more level
    at 50% of saturation RPS to measure recovery behaviour.
    """
    analysis = SaturationAnalysis()
    current_rps = starting_rps
    consecutive_over = 0
    saturation_found = False

    for level_idx in range(max_levels):
        # Build args for this level
        level_args = deepcopy(args)
        level_args.request_rate = current_rps
        level_args.num_prompts = prompts_per_level
        # Disable features that interfere with load shedding test
        level_args.warmup = 0
        level_args.repeat = 1
        level_args.duration = None
        # Disable retries so rejections are captured
        level_args.retries = 0

        result_dict, bench_result = await run_benchmark(level_args, base_url)

        metrics = _extract_metrics(result_dict)
        total = result_dict.get("summary", result_dict).get(
            "completed", prompts_per_level
        )
        if total == 0:
            total = prompts_per_level
        rejected = _count_rejections(result_dict)
        error_count = result_dict.get("summary", result_dict).get("errors", 0)
        error_rate = error_count / total if total > 0 else 0.0

        level_result = LoadShedLevel(
            rps=current_rps,
            throughput_rps=metrics["throughput_rps"],
            mean_latency_ms=metrics["mean_latency_ms"],
            p99_latency_ms=metrics["p99_latency_ms"],
            error_rate=error_rate,
            rejected_count=rejected,
            total_requests=total,
        )
        analysis.levels.append(level_result)
        analysis.degradation_curve.append(
            {
                "rps": current_rps,
                "throughput": metrics["throughput_rps"],
                "error_rate": error_rate,
                "mean_latency_ms": metrics["mean_latency_ms"],
            }
        )

        # Check saturation
        if error_rate > error_threshold:
            consecutive_over += 1
            if consecutive_over >= 2:
                analysis.saturation_rps = current_rps
                saturation_found = True
                break
        else:
            consecutive_over = 0

        # Ramp up
        if ramp_step > 0:
            current_rps += ramp_step
        else:
            current_rps *= ramp_multiplier

    # Determine max sustainable RPS
    sustainable_levels = [
        lv for lv in analysis.levels if lv.error_rate <= error_threshold
    ]
    if sustainable_levels:
        analysis.max_sustainable_rps = max(lv.rps for lv in sustainable_levels)

    # Recovery check: run at 50% of saturation RPS
    if recovery_check and saturation_found and analysis.saturation_rps:
        recovery_rps = analysis.saturation_rps * 0.5
        recovery_args = deepcopy(args)
        recovery_args.request_rate = recovery_rps
        recovery_args.num_prompts = prompts_per_level
        recovery_args.warmup = 0
        recovery_args.repeat = 1
        recovery_args.duration = None
        recovery_args.retries = 0

        rec_dict, _ = await run_benchmark(recovery_args, base_url)
        rec_metrics = _extract_metrics(rec_dict)
        rec_errors = rec_dict.get("summary", rec_dict).get("errors", 0)
        rec_total = rec_dict.get("summary", rec_dict).get(
            "completed", prompts_per_level
        )
        rec_error_rate = rec_errors / rec_total if rec_total > 0 else 0.0

        if rec_error_rate <= error_threshold:
            analysis.recovery_rps = recovery_rps
            analysis.recovery_latency_ms = rec_metrics["mean_latency_ms"]

    return analysis


def format_load_shed_table(analysis: SaturationAnalysis) -> str:
    """Format load shedding results as a terminal table."""
    lines: list[str] = []
    lines.append("=" * 90)
    lines.append("  xPyD-bench — Load Shedding Analysis")
    lines.append("=" * 90)
    lines.append("")
    hdr = (
        f"  {'RPS':>8s} {'Throughput':>12s} {'Mean Lat':>10s} "
        f"{'P99 Lat':>10s} {'Error%':>8s} {'Rejected':>10s}"
    )
    lines.append(hdr)
    lines.append("  " + "-" * 86)

    for lv in analysis.levels:
        lines.append(
            f"  {lv.rps:>8.1f} {lv.throughput_rps:>12.2f} "
            f"{lv.mean_latency_ms:>10.1f} {lv.p99_latency_ms:>10.1f} "
            f"{lv.error_rate * 100:>7.1f}% {lv.rejected_count:>10d}"
        )

    lines.append("  " + "-" * 86)
    lines.append("")

    if analysis.max_sustainable_rps is not None:
        lines.append(
            f"  Max sustainable RPS: {analysis.max_sustainable_rps:.1f}"
        )
    if analysis.saturation_rps is not None:
        lines.append(
            f"  Saturation point:    {analysis.saturation_rps:.1f} RPS"
        )
    if analysis.recovery_rps is not None:
        lines.append(
            f"  Recovery at:         {analysis.recovery_rps:.1f} RPS "
            f"(latency: {analysis.recovery_latency_ms:.1f}ms)"
        )
    if analysis.saturation_rps is None:
        lines.append("  ⚠️  Saturation not reached within test range")

    lines.append("=" * 90)
    return "\n".join(lines)
