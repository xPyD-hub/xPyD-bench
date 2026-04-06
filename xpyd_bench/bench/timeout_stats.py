"""Request timeout classification and reporting (M70)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xpyd_bench.bench.models import RequestResult


def compute_timeout_summary(requests: list[RequestResult]) -> dict | None:
    """Compute timeout classification summary from request results.

    Returns None if no timeouts occurred.
    """
    if not requests:
        return None

    timed_out = [r for r in requests if r.timeout_detected]
    if not timed_out:
        return None

    total = len(requests)
    timeout_count = len(timed_out)
    timeout_latencies = [r.latency_ms for r in timed_out if r.latency_ms > 0]

    summary: dict = {
        "timeout_count": timeout_count,
        "total_requests": total,
        "timeout_percentage": round(timeout_count / total * 100, 2),
    }

    if timeout_latencies:
        timeout_latencies.sort()
        summary["mean_latency_at_timeout_ms"] = round(
            sum(timeout_latencies) / len(timeout_latencies), 2,
        )
        summary["min_latency_at_timeout_ms"] = round(timeout_latencies[0], 2)
        summary["max_latency_at_timeout_ms"] = round(timeout_latencies[-1], 2)

    return summary
