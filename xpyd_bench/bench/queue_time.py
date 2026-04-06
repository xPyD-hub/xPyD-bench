"""Request queuing time measurement and reporting (M71)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from xpyd_bench.bench.models import RequestResult


def compute_queue_time_summary(requests: list[RequestResult]) -> dict | None:
    """Compute queue time summary from request results.

    Queue time is the client-side delay between when a request is scheduled
    (created/enqueued) and when it is actually sent over the network.

    Returns None if no queue times were recorded.
    """
    queue_times = [
        r.queue_time_ms for r in requests if r.queue_time_ms is not None and r.success
    ]

    if not queue_times:
        return None

    arr = np.array(queue_times)
    return {
        "count": len(queue_times),
        "mean_ms": round(float(np.mean(arr)), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p90_ms": round(float(np.percentile(arr, 90)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2),
    }
