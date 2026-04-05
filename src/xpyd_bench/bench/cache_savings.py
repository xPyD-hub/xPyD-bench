"""Prompt Caching Cost Analysis (M92).

Analyze dataset prompts for shared prefix patterns and estimate
potential cost savings from prompt caching.
"""

from __future__ import annotations

from typing import Any


def _longest_common_prefix(a: str, b: str) -> int:
    """Return length of longest common prefix between two strings."""
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    return i


def _estimate_token_count(text: str) -> int:
    """Rough token estimate (word-split heuristic). ~4 chars per token."""
    return max(1, len(text) // 4)


def analyze_cache_savings(
    prompts: list[str],
    *,
    cache_pricing_ratio: float = 0.5,
    cost_per_1k_input: float | None = None,
) -> dict[str, Any]:
    """Analyze prompts for prefix caching savings potential.

    Parameters
    ----------
    prompts:
        List of prompt strings from the dataset.
    cache_pricing_ratio:
        Ratio of cached token cost to uncached (0.5 = cached tokens cost 50%).
    cost_per_1k_input:
        If provided, compute dollar-amount savings estimate.

    Returns
    -------
    dict with cache savings analysis.
    """
    n = len(prompts)
    if n < 2:
        return {
            "num_prompts": n,
            "cacheable_token_ratio": 0.0,
            "estimated_cache_hit_rate": 0.0,
            "total_prompt_tokens": _estimate_token_count(prompts[0]) if n == 1 else 0,
            "cacheable_tokens": 0,
            "unique_tokens": _estimate_token_count(prompts[0]) if n == 1 else 0,
            "savings_ratio": 0.0,
            "cost_savings": None,
        }

    # Sort prompts to find shared prefixes efficiently
    sorted_prompts = sorted(prompts)

    # Compute pairwise LCP between adjacent sorted prompts
    lcp_chars: list[int] = []
    for i in range(len(sorted_prompts) - 1):
        lcp = _longest_common_prefix(sorted_prompts[i], sorted_prompts[i + 1])
        lcp_chars.append(lcp)

    # Estimate total and cacheable tokens
    total_tokens = sum(_estimate_token_count(p) for p in prompts)

    # For each prompt (except first in sorted order), the prefix shared with
    # at least one neighbor represents cacheable tokens.
    # Use max LCP per prompt position as the cacheable portion.
    cacheable_chars = 0
    for lcp in lcp_chars:
        cacheable_chars += lcp

    cacheable_tokens = max(0, cacheable_chars // 4)
    cacheable_ratio = round(cacheable_tokens / total_tokens, 4) if total_tokens > 0 else 0.0

    # Cache hit rate: fraction of prompt pairs with non-trivial shared prefix (>10 chars)
    hits = sum(1 for lcp in lcp_chars if lcp > 10)
    hit_rate = round(hits / len(lcp_chars), 4) if lcp_chars else 0.0

    # Savings: cached tokens cost less
    # Without cache: total_tokens * full_price
    # With cache: uncached * full + cached * (full * ratio)
    # Savings = cacheable_tokens * full_price * (1 - ratio)
    savings_ratio = round(cacheable_ratio * (1 - cache_pricing_ratio), 4)

    cost_savings: dict[str, Any] | None = None
    if cost_per_1k_input is not None and cost_per_1k_input > 0:
        full_cost = total_tokens / 1000 * cost_per_1k_input
        saved = cacheable_tokens / 1000 * cost_per_1k_input * (1 - cache_pricing_ratio)
        cost_savings = {
            "currency": "USD",
            "full_cost": round(full_cost, 6),
            "cached_cost": round(full_cost - saved, 6),
            "saved": round(saved, 6),
        }

    unique_tokens = total_tokens - cacheable_tokens

    return {
        "num_prompts": n,
        "cacheable_token_ratio": cacheable_ratio,
        "estimated_cache_hit_rate": hit_rate,
        "total_prompt_tokens": total_tokens,
        "cacheable_tokens": cacheable_tokens,
        "unique_tokens": unique_tokens,
        "savings_ratio": savings_ratio,
        "cache_pricing_ratio": cache_pricing_ratio,
        "cost_savings": cost_savings,
    }
