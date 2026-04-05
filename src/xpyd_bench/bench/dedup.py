"""Request deduplication & idempotency tracking (M85)."""

from __future__ import annotations

import hashlib
from typing import Any

from xpyd_bench.bench.models import RequestResult


def compute_response_hash(text: str) -> str:
    """Return SHA-256 hex digest of *text*."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def compute_dedup_summary(
    requests: list[RequestResult],
) -> dict[str, Any]:
    """Compute deduplication summary from request results.

    Only considers successful requests with non-None ``response_text``.

    Returns dict with ``total``, ``unique``, ``duplicates``,
    ``duplicate_ratio``, and ``hash_counts`` (hash → count mapping for
    hashes that appear more than once).
    """
    hashes: list[str] = []
    for r in requests:
        if r.success and r.response_text is not None:
            hashes.append(compute_response_hash(r.response_text))

    if not hashes:
        return {
            "total": 0,
            "unique": 0,
            "duplicates": 0,
            "duplicate_ratio": 0.0,
        }

    from collections import Counter

    counts = Counter(hashes)
    total = len(hashes)
    unique = len(counts)
    duplicates = total - unique
    duplicate_ratio = round(duplicates / total, 4) if total > 0 else 0.0

    # Only include hashes that appear more than once
    repeated: dict[str, int] = {h: c for h, c in counts.items() if c > 1}

    result: dict[str, Any] = {
        "total": total,
        "unique": unique,
        "duplicates": duplicates,
        "duplicate_ratio": duplicate_ratio,
    }
    if repeated:
        result["repeated_hashes"] = repeated

    return result
