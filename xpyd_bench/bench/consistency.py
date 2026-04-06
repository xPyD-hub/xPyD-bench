"""Endpoint response consistency check (M96).

Send the same prompt N times and measure response variance to detect
non-deterministic behavior even with temperature=0.

Metrics:
- **token_divergence_rate**: fraction of token positions that differ across
  responses (0.0 = fully deterministic, 1.0 = completely different).
- **length_variance**: variance and CV of response lengths in tokens.
- **latency_cv**: coefficient of variation of per-request latency.
- **unique_responses**: count of distinct response texts.
"""

from __future__ import annotations

import hashlib
import re
import statistics
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xpyd_bench.bench.models import RequestResult


def _tokenize(text: str) -> list[str]:
    """Cheap whitespace + punctuation tokenizer."""
    return re.findall(r"\w+", text.lower())


def _pairwise_divergence(tokens_a: list[str], tokens_b: list[str]) -> float:
    """Fraction of positions where two token lists differ.

    When lists have different lengths the extra tokens all count as
    divergent.
    """
    max_len = max(len(tokens_a), len(tokens_b))
    if max_len == 0:
        return 0.0
    mismatches = 0
    for i in range(max_len):
        a = tokens_a[i] if i < len(tokens_a) else None
        b = tokens_b[i] if i < len(tokens_b) else None
        if a != b:
            mismatches += 1
    return mismatches / max_len


def compute_consistency_summary(
    requests: list[RequestResult],
) -> dict:
    """Analyse a set of requests (same prompt repeated) for consistency.

    Returns a summary dict with divergence, length, and latency stats.
    """
    if not requests:
        return {}

    # Collect response texts and latencies
    texts: list[str] = []
    latencies: list[float] = []
    for rq in requests:
        texts.append(rq.response_text or "")
        latencies.append(rq.latency_ms)

    n = len(texts)

    # --- Unique responses ---
    hashes = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
    unique_count = len(set(hashes))

    # --- Token-level divergence ---
    token_lists = [_tokenize(t) for t in texts]
    divergences: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            divergences.append(_pairwise_divergence(token_lists[i], token_lists[j]))
    mean_divergence = statistics.mean(divergences) if divergences else 0.0

    # --- Response length variance ---
    lengths = [len(tl) for tl in token_lists]
    mean_length = statistics.mean(lengths) if lengths else 0.0
    length_stddev = statistics.stdev(lengths) if len(lengths) > 1 else 0.0
    length_cv = (length_stddev / mean_length) if mean_length > 0 else 0.0

    # --- Latency consistency ---
    mean_latency = statistics.mean(latencies) if latencies else 0.0
    latency_stddev = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    latency_cv = (latency_stddev / mean_latency) if mean_latency > 0 else 0.0

    # --- Determinism verdict ---
    is_deterministic = unique_count == 1

    return {
        "num_requests": n,
        "unique_responses": unique_count,
        "is_deterministic": is_deterministic,
        "token_divergence_rate": round(mean_divergence, 6),
        "length_mean": round(mean_length, 2),
        "length_stddev": round(length_stddev, 4),
        "length_cv": round(length_cv, 6),
        "latency_mean_ms": round(mean_latency, 3),
        "latency_stddev_ms": round(latency_stddev, 3),
        "latency_cv": round(latency_cv, 6),
    }
