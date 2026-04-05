"""Workload distribution statistics for prompt and output token lengths (M78)."""

from __future__ import annotations

import statistics
from typing import Any


def compute_workload_stats(
    prompt_tokens: list[int],
    completion_tokens: list[int],
) -> dict[str, Any]:
    """Compute distribution statistics for prompt and output token lengths.

    Returns a dict with ``prompt`` and ``completion`` sub-dicts, each containing
    mean, std, min, max, p50, p90, p99, and count.  Returns an empty dict when
    *both* lists are empty.
    """
    if not prompt_tokens and not completion_tokens:
        return {}

    result: dict[str, Any] = {}
    for name, values in [("prompt", prompt_tokens), ("completion", completion_tokens)]:
        result[name] = _stats_for(values)
    return result


def _stats_for(values: list[int]) -> dict[str, Any]:
    """Return distribution stats for a list of integer values."""
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0,
            "max": 0,
            "p50": 0.0,
            "p90": 0.0,
            "p99": 0.0,
        }

    n = len(values)
    mean = statistics.mean(values)
    std = statistics.pstdev(values) if n >= 2 else 0.0
    p50 = float(statistics.median(values))

    if n >= 2:
        quantiles = statistics.quantiles(values, n=100)
        p90 = float(quantiles[89])
        p99 = float(quantiles[-1])
    else:
        p90 = float(values[0])
        p99 = float(values[0])

    return {
        "count": n,
        "mean": round(mean, 2),
        "std": round(std, 2),
        "min": min(values),
        "max": max(values),
        "p50": round(p50, 2),
        "p90": round(p90, 2),
        "p99": round(p99, 2),
    }
