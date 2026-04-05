"""Rate-limit header tracking and aggregation (M66)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Common rate-limit header names (case-insensitive lookup)
_HEADER_MAP = {
    "x-ratelimit-limit": "limit",
    "x-ratelimit-remaining": "remaining",
    "x-ratelimit-reset": "reset",
    "ratelimit-limit": "limit",
    "ratelimit-remaining": "remaining",
    "ratelimit-reset": "reset",
    "retry-after": "retry_after",
    "x-ratelimit-limit-requests": "limit_requests",
    "x-ratelimit-limit-tokens": "limit_tokens",
    "x-ratelimit-remaining-requests": "remaining_requests",
    "x-ratelimit-remaining-tokens": "remaining_tokens",
    "x-ratelimit-reset-requests": "reset_requests",
    "x-ratelimit-reset-tokens": "reset_tokens",
}


def parse_ratelimit_headers(headers: dict[str, str] | Any) -> dict[str, str]:
    """Extract rate-limit related headers from an HTTP response headers mapping.

    Returns a dict with normalised keys (e.g. ``remaining``, ``limit``).
    """
    result: dict[str, str] = {}
    for raw_name, value in headers.items():
        key = raw_name.lower()
        mapped = _HEADER_MAP.get(key)
        if mapped is not None:
            result[mapped] = value
    return result


@dataclass
class RatelimitSummary:
    """Aggregated rate-limit statistics across a benchmark run."""

    min_remaining: int | None = None
    min_remaining_tokens: int | None = None
    max_limit: int | None = None
    throttle_count: int = 0  # number of 429 responses
    tracked_responses: int = 0  # responses that had any ratelimit header
    total_responses: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_remaining": self.min_remaining,
            "min_remaining_tokens": self.min_remaining_tokens,
            "max_limit": self.max_limit,
            "throttle_count": self.throttle_count,
            "tracked_responses": self.tracked_responses,
            "total_responses": self.total_responses,
        }


def aggregate_ratelimit(
    per_request_headers: list[dict[str, str] | None],
    error_messages: list[str | None] | None = None,
) -> RatelimitSummary:
    """Aggregate per-request rate-limit headers into a summary.

    *per_request_headers* is a list aligned with request order.
    *error_messages* (optional) is checked for 429 status indicators.
    """
    summary = RatelimitSummary()
    summary.total_responses = len(per_request_headers)

    for i, hdrs in enumerate(per_request_headers):
        if not hdrs:
            continue
        summary.tracked_responses += 1

        # remaining (requests)
        raw_rem = hdrs.get("remaining") or hdrs.get("remaining_requests")
        if raw_rem is not None:
            try:
                val = int(raw_rem)
                if summary.min_remaining is None or val < summary.min_remaining:
                    summary.min_remaining = val
            except (ValueError, TypeError):
                pass

        # remaining tokens
        raw_rem_tok = hdrs.get("remaining_tokens")
        if raw_rem_tok is not None:
            try:
                val = int(raw_rem_tok)
                if summary.min_remaining_tokens is None or val < summary.min_remaining_tokens:
                    summary.min_remaining_tokens = val
            except (ValueError, TypeError):
                pass

        # limit
        raw_lim = hdrs.get("limit") or hdrs.get("limit_requests")
        if raw_lim is not None:
            try:
                val = int(raw_lim)
                if summary.max_limit is None or val > summary.max_limit:
                    summary.max_limit = val
            except (ValueError, TypeError):
                pass

    # Count 429 throttle events from error strings
    if error_messages:
        for err in error_messages:
            if err and "429" in err:
                summary.throttle_count += 1

    return summary
