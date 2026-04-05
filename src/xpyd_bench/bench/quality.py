"""Model output quality scoring (M94).

Lightweight output quality assessment to detect quality degradation under
high load.  Three scoring modes are supported:

- ``perplexity-proxy``: response length vs prompt complexity ratio.
- ``repetition``: n-gram repetition rate (bigram + trigram).
- ``coherence``: sentence-level token-overlap similarity.
"""

from __future__ import annotations

import re
import statistics
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xpyd_bench.bench.models import RequestResult


def _tokenize(text: str) -> list[str]:
    """Cheap whitespace + punctuation tokenizer."""
    return re.findall(r"\w+", text.lower())


def _sentences(text: str) -> list[str]:
    """Split text into sentences."""
    parts = re.split(r"[.!?]+", text.strip())
    return [s.strip() for s in parts if s.strip()]


# ------------------------------------------------------------------
# Individual scoring functions
# ------------------------------------------------------------------

def score_perplexity_proxy(prompt_tokens: int, completion_tokens: int) -> float:
    """Ratio of completion tokens to prompt tokens.

    A very low ratio may indicate the model produced an unusually short
    (possibly degenerate) response.  Returns 0.0 when *prompt_tokens* is
    zero to avoid division errors.
    """
    if prompt_tokens <= 0:
        return 0.0
    return completion_tokens / prompt_tokens


def score_repetition(text: str) -> float:
    """N-gram repetition rate (average of bigram and trigram rates).

    Returns a value in [0, 1] where 0 means no repetition and 1 means
    every n-gram is repeated at least once.
    """
    tokens = _tokenize(text)
    if len(tokens) < 3:
        return 0.0

    def _ngram_rep_rate(n: int) -> float:
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        if not ngrams:
            return 0.0
        unique = len(set(ngrams))
        total = len(ngrams)
        return 1.0 - unique / total

    bigram_rate = _ngram_rep_rate(2)
    trigram_rate = _ngram_rep_rate(3)
    return (bigram_rate + trigram_rate) / 2.0


def score_coherence(text: str) -> float:
    """Sentence-level token-overlap similarity (Jaccard).

    Computes mean pairwise Jaccard similarity between consecutive
    sentences.  Returns 0.0 for texts with fewer than two sentences.
    High values may indicate repetitive sentence structures.
    """
    sents = _sentences(text)
    if len(sents) < 2:
        return 0.0

    similarities: list[float] = []
    for i in range(len(sents) - 1):
        a = set(_tokenize(sents[i]))
        b = set(_tokenize(sents[i + 1]))
        if not a and not b:
            continue
        union = a | b
        if not union:
            continue
        similarities.append(len(a & b) / len(union))

    return statistics.mean(similarities) if similarities else 0.0


# ------------------------------------------------------------------
# Aggregate helpers
# ------------------------------------------------------------------

_MODE_FN = {
    "perplexity-proxy": None,  # handled specially (needs tokens, not text)
    "repetition": score_repetition,
    "coherence": score_coherence,
}

VALID_MODES = frozenset(_MODE_FN.keys())


def compute_request_quality(
    request: RequestResult,
    modes: list[str],
) -> dict[str, float]:
    """Compute quality scores for a single request.

    Returns a dict mapping mode name to score.
    """
    scores: dict[str, float] = {}
    for mode in modes:
        if mode == "perplexity-proxy":
            scores[mode] = score_perplexity_proxy(
                request.prompt_tokens, request.completion_tokens
            )
        elif mode in ("repetition", "coherence"):
            text = request.response_text or ""
            fn = _MODE_FN[mode]
            scores[mode] = fn(text) if fn else 0.0
        # unknown modes silently ignored
    return scores


def compute_quality_summary(
    all_scores: list[dict[str, float]],
    modes: list[str],
) -> dict:
    """Aggregate per-request quality scores into a summary.

    Returns a dict with per-mode statistics: mean, min, max, stddev.
    """
    if not all_scores:
        return {}

    summary: dict[str, dict] = {}
    for mode in modes:
        values = [s[mode] for s in all_scores if mode in s]
        if not values:
            continue
        summary[mode] = {
            "mean": round(statistics.mean(values), 6),
            "min": round(min(values), 6),
            "max": round(max(values), 6),
            "stddev": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
            "count": len(values),
        }
    return summary
