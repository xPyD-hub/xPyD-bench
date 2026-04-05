"""Tests for model output quality scoring (M94)."""

from __future__ import annotations

from xpyd_bench.bench.quality import (
    VALID_MODES,
    compute_quality_summary,
    compute_request_quality,
    score_coherence,
    score_perplexity_proxy,
    score_repetition,
)


def _make_request(
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    response_text: str | None = None,
    success: bool = True,
):
    """Create a minimal RequestResult-like object."""
    from dataclasses import dataclass

    @dataclass
    class FakeRequest:
        prompt_tokens: int = 0
        completion_tokens: int = 0
        response_text: str | None = None
        success: bool = True
        quality_scores: dict | None = None

    return FakeRequest(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        response_text=response_text,
        success=success,
    )


# --- score_perplexity_proxy ---


def test_perplexity_proxy_normal():
    assert score_perplexity_proxy(100, 50) == 0.5


def test_perplexity_proxy_zero_prompt():
    assert score_perplexity_proxy(0, 50) == 0.0


def test_perplexity_proxy_negative_prompt():
    assert score_perplexity_proxy(-1, 50) == 0.0


def test_perplexity_proxy_zero_completion():
    assert score_perplexity_proxy(100, 0) == 0.0


# --- score_repetition ---


def test_repetition_no_repeat():
    text = "the quick brown fox jumps over a lazy dog today"
    rate = score_repetition(text)
    assert rate >= 0.0


def test_repetition_high_repeat():
    text = "hello hello hello hello hello hello hello hello"
    rate = score_repetition(text)
    assert rate > 0.5


def test_repetition_short_text():
    assert score_repetition("hi") == 0.0


def test_repetition_empty():
    assert score_repetition("") == 0.0


# --- score_coherence ---


def test_coherence_similar_sentences():
    text = "The cat sat on the mat. The cat lay on the mat."
    score = score_coherence(text)
    assert score > 0.3


def test_coherence_different_sentences():
    text = "The sun is bright. Quantum mechanics is complex."
    score = score_coherence(text)
    assert score < 0.5


def test_coherence_single_sentence():
    assert score_coherence("Just one sentence here") == 0.0


def test_coherence_empty():
    assert score_coherence("") == 0.0


# --- compute_request_quality ---


def test_request_quality_all_modes():
    req = _make_request(
        prompt_tokens=100,
        completion_tokens=50,
        response_text="The quick brown fox. The lazy dog sleeps.",
    )
    scores = compute_request_quality(req, list(VALID_MODES))
    assert "perplexity-proxy" in scores
    assert "repetition" in scores
    assert "coherence" in scores
    assert scores["perplexity-proxy"] == 0.5


def test_request_quality_single_mode():
    req = _make_request(response_text="hello world test data")
    scores = compute_request_quality(req, ["repetition"])
    assert "repetition" in scores
    assert "perplexity-proxy" not in scores


def test_request_quality_no_response_text():
    req = _make_request(response_text=None)
    scores = compute_request_quality(req, ["repetition", "coherence"])
    assert scores["repetition"] == 0.0
    assert scores["coherence"] == 0.0


# --- compute_quality_summary ---


def test_quality_summary_basic():
    all_scores = [
        {"repetition": 0.1, "coherence": 0.3},
        {"repetition": 0.2, "coherence": 0.5},
        {"repetition": 0.3, "coherence": 0.4},
    ]
    summary = compute_quality_summary(all_scores, ["repetition", "coherence"])
    assert "repetition" in summary
    assert "coherence" in summary
    assert summary["repetition"]["count"] == 3
    assert summary["repetition"]["min"] == 0.1
    assert summary["repetition"]["max"] == 0.3


def test_quality_summary_empty():
    assert compute_quality_summary([], ["repetition"]) == {}


def test_quality_summary_single_request():
    all_scores = [{"repetition": 0.2}]
    summary = compute_quality_summary(all_scores, ["repetition"])
    assert summary["repetition"]["stddev"] == 0.0
    assert summary["repetition"]["count"] == 1


# --- VALID_MODES ---


def test_valid_modes():
    assert "perplexity-proxy" in VALID_MODES
    assert "repetition" in VALID_MODES
    assert "coherence" in VALID_MODES


# --- Edge cases ---


def test_repetition_all_same_word():
    text = "word " * 50
    rate = score_repetition(text)
    assert rate > 0.9


def test_coherence_identical_sentences():
    text = "Same sentence here. Same sentence here. Same sentence here."
    score = score_coherence(text)
    assert score == 1.0


def test_request_quality_unknown_mode_ignored():
    req = _make_request(response_text="test")
    scores = compute_request_quality(req, ["unknown_mode"])
    assert "unknown_mode" not in scores
