"""Confidence interval reporting via bootstrap resampling (M84)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xpyd_bench.bench.models import RequestResult

DEFAULT_CONFIDENCE_LEVEL: float = 0.95
BOOTSTRAP_ITERATIONS: int = 1000
SMALL_SAMPLE_THRESHOLD: int = 5


@dataclass
class ConfidenceInterval:
    """CI result for a single metric."""

    metric: str
    point_estimate: float
    lower: float
    upper: float
    confidence_level: float

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "point_estimate": self.point_estimate,
            "lower": self.lower,
            "upper": self.upper,
            "confidence_level": self.confidence_level,
        }


def bootstrap_ci(
    values: list[float],
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    n_iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return ``(point_estimate, lower, upper)`` via bootstrap resampling."""
    if not values:
        return 0.0, 0.0, 0.0

    arr = np.array(values, dtype=np.float64)
    point = float(np.mean(arr))

    if len(values) == 1:
        return point, point, point

    rng = np.random.default_rng(seed)
    n = len(arr)
    boot: list[float] = []
    for _ in range(n_iterations):
        sample = rng.choice(arr, size=n, replace=True)
        boot.append(float(np.mean(sample)))

    boot_arr = np.array(boot)
    alpha = 1 - confidence_level
    lower = float(np.percentile(boot_arr, 100 * alpha / 2))
    upper = float(np.percentile(boot_arr, 100 * (1 - alpha / 2)))

    return point, lower, upper


def compute_confidence_intervals(
    requests: list[RequestResult],
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> dict:
    """Compute bootstrap confidence intervals for key metrics.

    Returns dict with ``confidence_level``, ``metrics`` mapping, and
    optionally ``warning`` for small samples.  Returns ``{}`` if no
    requests.
    """
    if not requests:
        return {}

    if not (0 < confidence_level < 1):
        raise ValueError(
            f"confidence_level must be between 0 and 1 exclusive, got {confidence_level}"
        )

    successful = [r for r in requests if r.success]
    if not successful:
        return {}

    result: dict = {"confidence_level": confidence_level, "metrics": {}}

    if len(successful) <= SMALL_SAMPLE_THRESHOLD:
        result["warning"] = (
            f"Only {len(successful)} successful requests; confidence intervals "
            f"may be unreliable (minimum recommended: {SMALL_SAMPLE_THRESHOLD})."
        )

    def _add(name: str, values: list[float]) -> None:
        if not values:
            return
        pt, lo, hi = bootstrap_ci(values, confidence_level=confidence_level)
        result["metrics"][name] = {
            "point_estimate": pt,
            "lower": lo,
            "upper": hi,
            "confidence_level": confidence_level,
        }

    _add("mean_latency_ms", [r.latency_ms for r in successful])

    ttfts = [r.ttft_ms for r in successful if r.ttft_ms is not None]
    _add("mean_ttft_ms", ttfts)

    tpots = [r.tpot_ms for r in successful if r.tpot_ms is not None]
    _add("mean_tpot_ms", tpots)

    # Throughput CI
    latencies = [r.latency_ms for r in successful]
    if latencies:
        arr = np.array(latencies, dtype=np.float64)
        rng = np.random.default_rng(42)
        n = len(arr)
        boot_tp: list[float] = []
        for _ in range(BOOTSTRAP_ITERATIONS):
            sample = rng.choice(arr, size=n, replace=True)
            mean_lat = float(np.mean(sample))
            if mean_lat > 0:
                boot_tp.append(1000.0 / mean_lat)
        if boot_tp:
            boot_arr = np.array(boot_tp)
            alpha = 1 - confidence_level
            mean_lat_all = float(np.mean(arr))
            result["metrics"]["throughput_rps"] = {
                "point_estimate": 1000.0 / mean_lat_all if mean_lat_all > 0 else 0,
                "lower": float(np.percentile(boot_arr, 100 * alpha / 2)),
                "upper": float(np.percentile(boot_arr, 100 * (1 - alpha / 2))),
                "confidence_level": confidence_level,
            }

    return result
