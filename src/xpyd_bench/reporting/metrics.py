"""Extended metrics computation — time-series bucketing."""

from __future__ import annotations

from xpyd_bench.bench.models import BenchmarkResult, RequestResult


def compute_time_series(
    result: BenchmarkResult,
    window_s: float = 1.0,
) -> list[dict]:
    """Bucket completed requests into time windows and compute per-window throughput.

    Returns a list of dicts, one per window:
        {"window_start_s": float, "window_end_s": float,
         "requests": int, "output_tokens": int,
         "request_throughput": float, "output_throughput": float}
    """
    successful = [r for r in result.requests if r.success]
    if not successful or result.total_duration_s <= 0:
        return []

    num_windows = max(1, int(result.total_duration_s / window_s) + 1)

    # We don't have absolute timestamps per request, so approximate by order.
    # Distribute requests evenly across duration for bucketing.
    buckets: list[list[RequestResult]] = [[] for _ in range(num_windows)]
    for idx, req in enumerate(successful):
        # Estimate request finish time proportionally
        frac = idx / max(len(successful) - 1, 1)
        t = frac * result.total_duration_s
        bucket_idx = min(int(t / window_s), num_windows - 1)
        buckets[bucket_idx].append(req)

    series = []
    for i, bucket in enumerate(buckets):
        start = i * window_s
        end = min(start + window_s, result.total_duration_s)
        dur = end - start if end > start else window_s
        out_tokens = sum(r.completion_tokens for r in bucket)
        series.append(
            {
                "window_start_s": round(start, 3),
                "window_end_s": round(end, 3),
                "requests": len(bucket),
                "output_tokens": out_tokens,
                "request_throughput": round(len(bucket) / dur, 2),
                "output_throughput": round(out_tokens / dur, 2),
            }
        )
    return series
