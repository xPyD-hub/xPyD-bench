"""Speculative decoding metrics analysis (M88).

Parses speculative decoding indicators from SSE streaming responses
and computes acceptance rate, draft batch sizes, and speculation overhead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SpecTokenEvent:
    """A single speculative decoding event from an SSE chunk."""

    draft_tokens: int  # Number of draft tokens proposed
    accepted_tokens: int  # Number of draft tokens accepted by verifier
    timestamp: float  # perf_counter relative to request start (seconds)


@dataclass
class RequestSpecMetrics:
    """Per-request speculative decoding analysis results."""

    events: list[SpecTokenEvent] = field(default_factory=list)
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    total_rejected_tokens: int = 0
    mean_acceptance_rate: float | None = None
    mean_draft_batch_size: float | None = None
    tokens_saved: int = 0  # accepted tokens that skipped full verification

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_count": len(self.events),
            "total_draft_tokens": self.total_draft_tokens,
            "total_accepted_tokens": self.total_accepted_tokens,
            "total_rejected_tokens": self.total_rejected_tokens,
            "mean_acceptance_rate": (
                round(self.mean_acceptance_rate, 4)
                if self.mean_acceptance_rate is not None
                else None
            ),
            "mean_draft_batch_size": (
                round(self.mean_draft_batch_size, 2)
                if self.mean_draft_batch_size is not None
                else None
            ),
            "tokens_saved": self.tokens_saved,
        }


def analyze_spec_events(events: list[SpecTokenEvent]) -> RequestSpecMetrics:
    """Analyze speculative decoding events for a single request.

    Args:
        events: List of SpecTokenEvent parsed from SSE chunks.

    Returns:
        RequestSpecMetrics with computed statistics.
    """
    result = RequestSpecMetrics(events=events)
    if not events:
        return result

    draft_totals: list[int] = []
    accepted_totals: list[int] = []
    acceptance_rates: list[float] = []

    for ev in events:
        draft_totals.append(ev.draft_tokens)
        accepted_totals.append(ev.accepted_tokens)
        rejected = ev.draft_tokens - ev.accepted_tokens
        result.total_draft_tokens += ev.draft_tokens
        result.total_accepted_tokens += ev.accepted_tokens
        result.total_rejected_tokens += rejected
        if ev.draft_tokens > 0:
            acceptance_rates.append(ev.accepted_tokens / ev.draft_tokens)

    result.tokens_saved = result.total_accepted_tokens

    if acceptance_rates:
        result.mean_acceptance_rate = float(np.mean(acceptance_rates))
    if draft_totals:
        result.mean_draft_batch_size = float(np.mean(draft_totals))

    return result


def compute_speculative_aggregate(
    per_request: list[RequestSpecMetrics],
) -> dict[str, Any]:
    """Aggregate speculative decoding metrics across all requests.

    Args:
        per_request: List of per-request speculative metrics.

    Returns:
        Dict with aggregate speculative decoding statistics.
    """
    if not per_request:
        return {}

    # Filter out requests with no speculative data
    with_data = [r for r in per_request if r.events]
    if not with_data:
        return {"requests_with_spec_data": 0, "total_requests": len(per_request)}

    total_draft = sum(r.total_draft_tokens for r in with_data)
    total_accepted = sum(r.total_accepted_tokens for r in with_data)
    total_rejected = sum(r.total_rejected_tokens for r in with_data)
    total_saved = sum(r.tokens_saved for r in with_data)

    all_acceptance_rates = [
        r.mean_acceptance_rate for r in with_data if r.mean_acceptance_rate is not None
    ]
    all_batch_sizes = [
        r.mean_draft_batch_size for r in with_data if r.mean_draft_batch_size is not None
    ]

    result: dict[str, Any] = {
        "requests_with_spec_data": len(with_data),
        "total_requests": len(per_request),
        "total_draft_tokens": total_draft,
        "total_accepted_tokens": total_accepted,
        "total_rejected_tokens": total_rejected,
        "total_tokens_saved": total_saved,
        "overall_acceptance_rate": (
            round(total_accepted / total_draft, 4) if total_draft > 0 else None
        ),
    }

    if all_acceptance_rates:
        arr = np.array(all_acceptance_rates)
        result["mean_acceptance_rate"] = round(float(np.mean(arr)), 4)
        result["p50_acceptance_rate"] = round(float(np.percentile(arr, 50)), 4)
        result["p90_acceptance_rate"] = round(float(np.percentile(arr, 90)), 4)

    if all_batch_sizes:
        arr = np.array(all_batch_sizes)
        result["mean_draft_batch_size"] = round(float(np.mean(arr)), 2)

    return result


def parse_spec_data_from_chunk(chunk_data: dict[str, Any]) -> SpecTokenEvent | None:
    """Extract speculative decoding data from an SSE chunk if present.

    Servers that support speculative decoding may include extra fields in
    their SSE chunks. We look for:
      - ``x_spec``: dict with ``draft_tokens`` and ``accepted_tokens``
      - ``speculative``: dict with same fields (alternative key)
      - Choice-level ``x_spec`` field

    Args:
        chunk_data: Parsed JSON dict from an SSE ``data:`` line.

    Returns:
        SpecTokenEvent if speculative data found, None otherwise.
    """
    # Check top-level x_spec / speculative
    spec = chunk_data.get("x_spec") or chunk_data.get("speculative")
    if spec and isinstance(spec, dict):
        draft = spec.get("draft_tokens", 0)
        accepted = spec.get("accepted_tokens", 0)
        if isinstance(draft, int) and isinstance(accepted, int) and draft > 0:
            return SpecTokenEvent(
                draft_tokens=draft,
                accepted_tokens=accepted,
                timestamp=0.0,  # caller sets real timestamp
            )

    # Check choice-level x_spec
    choices = chunk_data.get("choices", [])
    if choices and isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            choice_spec = choice.get("x_spec") or choice.get("speculative")
            if choice_spec and isinstance(choice_spec, dict):
                draft = choice_spec.get("draft_tokens", 0)
                accepted = choice_spec.get("accepted_tokens", 0)
                if isinstance(draft, int) and isinstance(accepted, int) and draft > 0:
                    return SpecTokenEvent(
                        draft_tokens=draft,
                        accepted_tokens=accepted,
                        timestamp=0.0,
                    )

    return None
