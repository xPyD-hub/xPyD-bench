"""Rolling window metrics — percentile-over-time tracking (M81).

Computes latency percentiles (P50/P90/P99) in sliding time windows to detect
mid-run performance degradation.
"""

from __future__ import annotations

from typing import Any


def compute_rolling_metrics(
    latencies_ms: list[float],
    start_times: list[float],
    bench_start_time: float,
    window_seconds: float = 10.0,
    step_seconds: float = 5.0,
    percentiles: tuple[float, ...] = (50.0, 90.0, 99.0),
) -> dict[str, Any]:
    """Compute rolling window percentile metrics over benchmark duration.

    Parameters
    ----------
    latencies_ms:
        Per-request end-to-end latencies in milliseconds.
    start_times:
        Per-request start timestamps (``perf_counter``).
    bench_start_time:
        Benchmark start timestamp (``perf_counter``).
    window_seconds:
        Sliding window size in seconds.
    step_seconds:
        Step size between windows in seconds.
    percentiles:
        Which percentiles to compute (0-100 scale).

    Returns
    -------
    dict with ``windows`` (list of window dicts) and ``config`` (window params).
    Each window dict has ``time_offset_s``, ``count``, and ``pN_latency_ms`` keys.
    Empty dict when fewer than 2 data points.
    """
    if len(latencies_ms) < 2 or len(start_times) < 2:
        return {}

    # Pair data and compute relative times
    pairs = [
        (st - bench_start_time, lat)
        for st, lat in zip(start_times, latencies_ms)
        if st is not None
    ]
    if len(pairs) < 2:
        return {}

    pairs.sort(key=lambda p: p[0])
    total_duration = pairs[-1][0]

    windows: list[dict[str, Any]] = []
    t = 0.0
    while t <= total_duration:
        window_end = t + window_seconds
        window_lats = [lat for rel_t, lat in pairs if t <= rel_t < window_end]

        if window_lats:
            entry: dict[str, Any] = {
                "time_offset_s": round(t, 2),
                "count": len(window_lats),
            }
            for p in percentiles:
                key = f"p{int(p)}_latency_ms" if p == int(p) else f"p{p}_latency_ms"
                entry[key] = round(_percentile(window_lats, p), 2)
            windows.append(entry)

        t += step_seconds

    if not windows:
        return {}

    # Detect degradation: compare first and last window P99
    p99_key = "p99_latency_ms"
    degradation: dict[str, Any] | None = None
    if len(windows) >= 2 and p99_key in windows[0] and p99_key in windows[-1]:
        first_p99 = windows[0][p99_key]
        last_p99 = windows[-1][p99_key]
        if first_p99 > 0:
            change_pct = round(((last_p99 - first_p99) / first_p99) * 100, 1)
            degradation = {
                "first_window_p99_ms": first_p99,
                "last_window_p99_ms": last_p99,
                "change_pct": change_pct,
                "degraded": change_pct > 10.0,
            }

    return {
        "config": {
            "window_seconds": window_seconds,
            "step_seconds": step_seconds,
            "percentiles": list(percentiles),
        },
        "windows": windows,
        "degradation": degradation,
    }


def _percentile(values: list[float], pct: float) -> float:
    """Compute a percentile value from a sorted-capable list."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_v = sorted(values)
    k = (pct / 100.0) * (len(sorted_v) - 1)
    f = int(k)
    c = f + 1
    if c >= len(sorted_v):
        return sorted_v[-1]
    return sorted_v[f] + (k - f) * (sorted_v[c] - sorted_v[f])
