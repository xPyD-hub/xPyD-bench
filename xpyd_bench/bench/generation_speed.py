"""Output token generation speed calculation and aggregation (M68)."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any


@dataclass
class GenerationSpeedSummary:
    """Aggregated generation speed statistics across a benchmark run."""

    mean_tps: float = 0.0
    p50_tps: float = 0.0
    p90_tps: float = 0.0
    p99_tps: float = 0.0
    min_tps: float = 0.0
    max_tps: float = 0.0
    tracked_requests: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_tps": round(self.mean_tps, 2),
            "p50_tps": round(self.p50_tps, 2),
            "p90_tps": round(self.p90_tps, 2),
            "p99_tps": round(self.p99_tps, 2),
            "min_tps": round(self.min_tps, 2),
            "max_tps": round(self.max_tps, 2),
            "tracked_requests": self.tracked_requests,
        }


def compute_generation_tps(
    completion_tokens: int,
    ttft_ms: float | None,
    latency_ms: float,
) -> float | None:
    """Compute output generation tokens per second for a single request.

    Generation time = total latency - TTFT (the decode phase).
    If TTFT is not available (non-streaming), use total latency as fallback.
    Returns None if tokens or time is zero/negative.
    """
    if completion_tokens <= 0:
        return None

    if ttft_ms is not None:
        generation_ms = latency_ms - ttft_ms
    else:
        generation_ms = latency_ms

    if generation_ms <= 0:
        return None

    return completion_tokens / (generation_ms / 1000.0)


def aggregate_generation_speeds(
    tps_values: list[float | None],
) -> GenerationSpeedSummary:
    """Aggregate per-request generation TPS into a summary."""
    vals = [v for v in tps_values if v is not None]
    summary = GenerationSpeedSummary()
    summary.tracked_requests = len(vals)

    if not vals:
        return summary

    summary.mean_tps = statistics.mean(vals)
    summary.min_tps = min(vals)
    summary.max_tps = max(vals)
    summary.p50_tps = float(statistics.median(vals))

    if len(vals) >= 2:
        quantiles = statistics.quantiles(vals, n=100)
        summary.p90_tps = float(quantiles[89])
        summary.p99_tps = float(quantiles[-1])
    else:
        summary.p90_tps = float(vals[0])
        summary.p99_tps = float(vals[0])

    return summary
