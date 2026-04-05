"""Request/response payload size tracking and aggregation (M67)."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any


@dataclass
class PayloadSummary:
    """Aggregated payload size statistics across a benchmark run."""

    total_request_bytes: int = 0
    total_response_bytes: int = 0
    mean_request_bytes: float = 0.0
    mean_response_bytes: float = 0.0
    min_request_bytes: int = 0
    min_response_bytes: int = 0
    max_request_bytes: int = 0
    max_response_bytes: int = 0
    p50_request_bytes: float = 0.0
    p50_response_bytes: float = 0.0
    p99_request_bytes: float = 0.0
    p99_response_bytes: float = 0.0
    tracked_requests: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_request_bytes": self.total_request_bytes,
            "total_response_bytes": self.total_response_bytes,
            "mean_request_bytes": round(self.mean_request_bytes, 1),
            "mean_response_bytes": round(self.mean_response_bytes, 1),
            "min_request_bytes": self.min_request_bytes,
            "min_response_bytes": self.min_response_bytes,
            "max_request_bytes": self.max_request_bytes,
            "max_response_bytes": self.max_response_bytes,
            "p50_request_bytes": round(self.p50_request_bytes, 1),
            "p50_response_bytes": round(self.p50_response_bytes, 1),
            "p99_request_bytes": round(self.p99_request_bytes, 1),
            "p99_response_bytes": round(self.p99_response_bytes, 1),
            "tracked_requests": self.tracked_requests,
        }


def compute_payload_bytes(payload: str | bytes | dict | None) -> int:
    """Compute the byte size of a request payload.

    For dicts (JSON bodies), encode to UTF-8 JSON. For strings, encode to UTF-8.
    For bytes, return len directly.
    """
    if payload is None:
        return 0
    if isinstance(payload, bytes):
        return len(payload)
    if isinstance(payload, str):
        return len(payload.encode("utf-8"))
    if isinstance(payload, dict):
        import json

        return len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    return 0


def aggregate_payload_sizes(
    request_bytes_list: list[int | None],
    response_bytes_list: list[int | None],
) -> PayloadSummary:
    """Aggregate per-request payload sizes into a summary.

    ``None`` entries are excluded from statistics.
    """
    req_vals = [v for v in request_bytes_list if v is not None]
    resp_vals = [v for v in response_bytes_list if v is not None]

    summary = PayloadSummary()
    tracked = max(len(req_vals), len(resp_vals))
    summary.tracked_requests = tracked

    if req_vals:
        summary.total_request_bytes = sum(req_vals)
        summary.mean_request_bytes = statistics.mean(req_vals)
        summary.min_request_bytes = min(req_vals)
        summary.max_request_bytes = max(req_vals)
        summary.p50_request_bytes = float(statistics.median(req_vals))
        if len(req_vals) >= 2:
            summary.p99_request_bytes = float(
                statistics.quantiles(req_vals, n=100)[-1]
            )
        else:
            summary.p99_request_bytes = float(req_vals[0])

    if resp_vals:
        summary.total_response_bytes = sum(resp_vals)
        summary.mean_response_bytes = statistics.mean(resp_vals)
        summary.min_response_bytes = min(resp_vals)
        summary.max_response_bytes = max(resp_vals)
        summary.p50_response_bytes = float(statistics.median(resp_vals))
        if len(resp_vals) >= 2:
            summary.p99_response_bytes = float(
                statistics.quantiles(resp_vals, n=100)[-1]
            )
        else:
            summary.p99_response_bytes = float(resp_vals[0])

    return summary
