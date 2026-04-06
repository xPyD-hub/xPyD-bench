"""Request pacing accuracy report (M93).

Measures how accurately the benchmark client maintains the target request
rate by comparing actual inter-request intervals to the intended schedule.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from xpyd_bench.bench.models import RequestResult


def compute_pacing_report(
    requests: list[RequestResult],
    target_interval_s: float | None = None,
) -> dict | None:
    """Compute pacing accuracy statistics from request results.

    Parameters
    ----------
    requests:
        Completed request results with ``start_time`` populated.
    target_interval_s:
        Expected interval between requests in seconds (1/request_rate).
        If *None*, pacing error metrics are omitted but drift/burst are
        still reported.

    Returns *None* when fewer than 2 requests have valid ``start_time``.
    """
    # Collect actual send timestamps (start_time is perf_counter based)
    times = sorted(
        r.start_time for r in requests if r.start_time is not None
    )
    if len(times) < 2:
        return None

    arr = np.array(times)
    actual_intervals_s = np.diff(arr)
    actual_intervals_ms = actual_intervals_s * 1000.0

    report: dict = {
        "num_requests": len(times),
        "num_intervals": len(actual_intervals_ms),
        "actual_interval_ms": {
            "mean": round(float(np.mean(actual_intervals_ms)), 3),
            "p50": round(float(np.percentile(actual_intervals_ms, 50)), 3),
            "p90": round(float(np.percentile(actual_intervals_ms, 90)), 3),
            "p99": round(float(np.percentile(actual_intervals_ms, 99)), 3),
            "min": round(float(np.min(actual_intervals_ms)), 3),
            "max": round(float(np.max(actual_intervals_ms)), 3),
            "stddev": round(float(np.std(actual_intervals_ms)), 3),
        },
    }

    # Pacing error: difference between actual and target interval
    if target_interval_s is not None and target_interval_s > 0:
        target_ms = target_interval_s * 1000.0
        errors_ms = actual_intervals_ms - target_ms
        abs_errors_ms = np.abs(errors_ms)
        report["target_interval_ms"] = round(target_ms, 3)
        report["pacing_error_ms"] = {
            "mean": round(float(np.mean(abs_errors_ms)), 3),
            "p50": round(float(np.percentile(abs_errors_ms, 50)), 3),
            "p99": round(float(np.percentile(abs_errors_ms, 99)), 3),
            "min": round(float(np.min(abs_errors_ms)), 3),
            "max": round(float(np.max(abs_errors_ms)), 3),
        }
        # Relative accuracy (percentage of target interval)
        report["pacing_accuracy_pct"] = round(
            float(1.0 - np.mean(abs_errors_ms) / target_ms) * 100.0, 2
        )

    # Drift detection: is the pacing error growing over time?
    report["drift"] = _detect_drift(actual_intervals_ms, target_interval_s)

    # Burst detection: clusters of requests sent too close together
    report["bursts"] = _detect_bursts(actual_intervals_ms, target_interval_s)

    return report


def _detect_drift(
    intervals_ms: np.ndarray,
    target_interval_s: float | None,
) -> dict:
    """Detect whether pacing error drifts (grows) over time.

    Uses linear regression on signed error to find a trend.
    """
    n = len(intervals_ms)
    if n < 4:
        return {"detected": False, "slope_ms_per_request": 0.0}

    if target_interval_s is not None and target_interval_s > 0:
        target_ms = target_interval_s * 1000.0
        errors = intervals_ms - target_ms
    else:
        # Without target, measure drift relative to mean
        errors = intervals_ms - float(np.mean(intervals_ms))

    x = np.arange(n, dtype=float)
    # Simple linear regression: slope of error over index
    slope = float(np.polyfit(x, errors, 1)[0])

    # Drift is "detected" when absolute slope > 1% of mean interval
    mean_interval = float(np.mean(intervals_ms))
    threshold = mean_interval * 0.01 if mean_interval > 0 else 0.1
    detected = abs(slope) > threshold

    return {
        "detected": detected,
        "slope_ms_per_request": round(slope, 4),
    }


def _detect_bursts(
    intervals_ms: np.ndarray,
    target_interval_s: float | None,
) -> dict:
    """Detect bursts — clusters of requests sent much faster than target.

    A burst is any interval shorter than 20% of target (or 20% of median
    if no target).
    """
    n = len(intervals_ms)
    if n < 1:
        return {"count": 0, "burst_intervals": 0, "burst_ratio": 0.0}

    if target_interval_s is not None and target_interval_s > 0:
        threshold_ms = target_interval_s * 1000.0 * 0.2
    else:
        threshold_ms = float(np.median(intervals_ms)) * 0.2

    # Avoid zero threshold
    if threshold_ms <= 0:
        threshold_ms = 0.1

    burst_mask = intervals_ms < threshold_ms
    burst_intervals = int(np.sum(burst_mask))

    # Count contiguous burst groups
    burst_count = 0
    in_burst = False
    for b in burst_mask:
        if b and not in_burst:
            burst_count += 1
            in_burst = True
        elif not b:
            in_burst = False

    return {
        "count": burst_count,
        "burst_intervals": burst_intervals,
        "burst_ratio": round(burst_intervals / n, 4) if n > 0 else 0.0,
    }
