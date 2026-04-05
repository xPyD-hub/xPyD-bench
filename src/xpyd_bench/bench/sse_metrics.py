"""Server-Sent Events (SSE) streaming metrics analysis (M53)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ChunkTiming:
    """Timing data for a single SSE chunk."""

    timestamp: float  # perf_counter relative to request start (seconds)
    tokens: int  # number of tokens in this chunk
    inter_token_ms: float | None = None  # ms since previous content chunk


@dataclass
class StallEvent:
    """A detected stall (gap) in token delivery."""

    start_s: float  # seconds since request start
    duration_ms: float  # stall duration in milliseconds


@dataclass
class RequestSSEMetrics:
    """Per-request SSE analysis results."""

    chunk_count: int = 0
    chunk_timings: list[ChunkTiming] = field(default_factory=list)
    stalls: list[StallEvent] = field(default_factory=list)
    mean_itl_ms: float | None = None
    p50_itl_ms: float | None = None
    p90_itl_ms: float | None = None
    p99_itl_ms: float | None = None
    jitter_ms: float | None = None  # stddev of inter-token latencies

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_count": self.chunk_count,
            "stalls": [
                {"start_s": round(s.start_s, 4), "duration_ms": round(s.duration_ms, 2)}
                for s in self.stalls
            ],
            "stall_count": len(self.stalls),
            "mean_itl_ms": self.mean_itl_ms,
            "p50_itl_ms": self.p50_itl_ms,
            "p90_itl_ms": self.p90_itl_ms,
            "p99_itl_ms": self.p99_itl_ms,
            "jitter_ms": self.jitter_ms,
        }


def analyze_chunk_timings(
    chunk_timings: list[ChunkTiming],
    stall_threshold_ms: float = 1000.0,
) -> RequestSSEMetrics:
    """Analyze per-chunk timing data for a single request.

    Args:
        chunk_timings: List of ChunkTiming from streaming response.
        stall_threshold_ms: Gap threshold (ms) to flag as a stall.

    Returns:
        RequestSSEMetrics with computed statistics.
    """
    result = RequestSSEMetrics(
        chunk_count=len(chunk_timings),
        chunk_timings=chunk_timings,
    )

    itls = [ct.inter_token_ms for ct in chunk_timings if ct.inter_token_ms is not None]
    if not itls:
        return result

    arr = np.array(itls)
    result.mean_itl_ms = round(float(np.mean(arr)), 2)
    result.p50_itl_ms = round(float(np.percentile(arr, 50)), 2)
    result.p90_itl_ms = round(float(np.percentile(arr, 90)), 2)
    result.p99_itl_ms = round(float(np.percentile(arr, 99)), 2)
    result.jitter_ms = round(float(np.std(arr)), 2)

    # Detect stalls
    for ct in chunk_timings:
        if ct.inter_token_ms is not None and ct.inter_token_ms >= stall_threshold_ms:
            result.stalls.append(
                StallEvent(
                    start_s=ct.timestamp - ct.inter_token_ms / 1000.0,
                    duration_ms=ct.inter_token_ms,
                )
            )

    return result


def compute_sse_aggregate(
    per_request: list[RequestSSEMetrics],
) -> dict[str, Any]:
    """Aggregate SSE metrics across all requests.

    Args:
        per_request: List of per-request SSE metrics.

    Returns:
        Dict with aggregate SSE statistics.
    """
    if not per_request:
        return {}

    all_itls: list[float] = []
    total_stalls = 0
    total_stall_duration_ms = 0.0
    total_chunks = 0

    for rm in per_request:
        itls = [ct.inter_token_ms for ct in rm.chunk_timings if ct.inter_token_ms is not None]
        all_itls.extend(itls)
        total_stalls += len(rm.stalls)
        total_stall_duration_ms += sum(s.duration_ms for s in rm.stalls)
        total_chunks += rm.chunk_count

    result: dict[str, Any] = {
        "total_chunks": total_chunks,
        "total_stalls": total_stalls,
        "total_stall_duration_ms": round(total_stall_duration_ms, 2),
        "requests_with_stalls": sum(1 for rm in per_request if rm.stalls),
    }

    if all_itls:
        arr = np.array(all_itls)
        result["mean_itl_ms"] = round(float(np.mean(arr)), 2)
        result["p50_itl_ms"] = round(float(np.percentile(arr, 50)), 2)
        result["p90_itl_ms"] = round(float(np.percentile(arr, 90)), 2)
        result["p95_itl_ms"] = round(float(np.percentile(arr, 95)), 2)
        result["p99_itl_ms"] = round(float(np.percentile(arr, 99)), 2)
        result["jitter_ms"] = round(float(np.std(arr)), 2)
    else:
        result["mean_itl_ms"] = None
        result["p50_itl_ms"] = None
        result["p90_itl_ms"] = None
        result["p95_itl_ms"] = None
        result["p99_itl_ms"] = None
        result["jitter_ms"] = None

    return result
