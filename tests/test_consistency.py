"""Tests for endpoint response consistency check (M96)."""

from __future__ import annotations

from xpyd_bench.bench.consistency import (
    _pairwise_divergence,
    _tokenize,
    compute_consistency_summary,
)
from xpyd_bench.bench.models import BenchmarkResult, RequestResult

# ------------------------------------------------------------------
# _tokenize
# ------------------------------------------------------------------


class TestTokenize:
    def test_basic(self):
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_empty(self):
        assert _tokenize("") == []

    def test_punctuation(self):
        assert _tokenize("one, two. three!") == ["one", "two", "three"]


# ------------------------------------------------------------------
# _pairwise_divergence
# ------------------------------------------------------------------


class TestPairwiseDivergence:
    def test_identical(self):
        assert _pairwise_divergence(["a", "b"], ["a", "b"]) == 0.0

    def test_completely_different(self):
        assert _pairwise_divergence(["a", "b"], ["c", "d"]) == 1.0

    def test_different_lengths(self):
        # ["a", "b"] vs ["a"] -> position 0 same, position 1 divergent = 0.5
        assert _pairwise_divergence(["a", "b"], ["a"]) == 0.5

    def test_both_empty(self):
        assert _pairwise_divergence([], []) == 0.0

    def test_one_empty(self):
        assert _pairwise_divergence(["a", "b"], []) == 1.0

    def test_partial_match(self):
        # 2 out of 3 match
        result = _pairwise_divergence(["a", "b", "c"], ["a", "x", "c"])
        assert abs(result - 1 / 3) < 1e-6


# ------------------------------------------------------------------
# compute_consistency_summary — deterministic
# ------------------------------------------------------------------


class TestConsistencySummaryDeterministic:
    def _make_requests(self, text: str, n: int, latency: float = 100.0):
        return [
            RequestResult(
                response_text=text,
                latency_ms=latency,
                success=True,
            )
            for _ in range(n)
        ]

    def test_deterministic_responses(self):
        requests = self._make_requests("hello world foo bar", 5)
        summary = compute_consistency_summary(requests)
        assert summary["num_requests"] == 5
        assert summary["unique_responses"] == 1
        assert summary["is_deterministic"] is True
        assert summary["token_divergence_rate"] == 0.0
        assert summary["length_cv"] == 0.0

    def test_single_request(self):
        requests = self._make_requests("single response", 1)
        summary = compute_consistency_summary(requests)
        assert summary["num_requests"] == 1
        assert summary["unique_responses"] == 1
        assert summary["is_deterministic"] is True

    def test_empty_requests(self):
        summary = compute_consistency_summary([])
        assert summary == {}


# ------------------------------------------------------------------
# compute_consistency_summary — non-deterministic
# ------------------------------------------------------------------


class TestConsistencySummaryNonDeterministic:
    def test_different_responses(self):
        requests = [
            RequestResult(
                response_text="the cat sat on the mat",
                latency_ms=100.0,
                success=True,
            ),
            RequestResult(
                response_text="a dog ran through the park",
                latency_ms=150.0,
                success=True,
            ),
            RequestResult(
                response_text="the cat sat on the mat",
                latency_ms=120.0,
                success=True,
            ),
        ]
        summary = compute_consistency_summary(requests)
        assert summary["num_requests"] == 3
        assert summary["unique_responses"] == 2
        assert summary["is_deterministic"] is False
        assert summary["token_divergence_rate"] > 0.0

    def test_all_different(self):
        requests = [
            RequestResult(response_text="alpha beta", latency_ms=100.0, success=True),
            RequestResult(response_text="gamma delta", latency_ms=200.0, success=True),
            RequestResult(
                response_text="epsilon zeta", latency_ms=300.0, success=True
            ),
        ]
        summary = compute_consistency_summary(requests)
        assert summary["unique_responses"] == 3
        assert summary["is_deterministic"] is False
        assert summary["token_divergence_rate"] == 1.0
        # Latency CV should be > 0
        assert summary["latency_cv"] > 0.0


# ------------------------------------------------------------------
# compute_consistency_summary — edge cases
# ------------------------------------------------------------------


class TestConsistencyEdgeCases:
    def test_empty_response_texts(self):
        requests = [
            RequestResult(response_text="", latency_ms=50.0, success=True),
            RequestResult(response_text="", latency_ms=60.0, success=True),
        ]
        summary = compute_consistency_summary(requests)
        assert summary["is_deterministic"] is True
        assert summary["token_divergence_rate"] == 0.0
        assert summary["length_mean"] == 0.0

    def test_none_response_texts(self):
        requests = [
            RequestResult(response_text=None, latency_ms=50.0, success=True),
            RequestResult(response_text=None, latency_ms=60.0, success=True),
        ]
        summary = compute_consistency_summary(requests)
        assert summary["is_deterministic"] is True

    def test_mixed_none_and_text(self):
        requests = [
            RequestResult(response_text="hello", latency_ms=50.0, success=True),
            RequestResult(response_text=None, latency_ms=60.0, success=True),
        ]
        summary = compute_consistency_summary(requests)
        assert summary["is_deterministic"] is False
        assert summary["unique_responses"] == 2

    def test_latency_consistency_identical(self):
        """All same latency → CV = 0."""
        requests = [
            RequestResult(response_text="same", latency_ms=100.0, success=True),
            RequestResult(response_text="same", latency_ms=100.0, success=True),
            RequestResult(response_text="same", latency_ms=100.0, success=True),
        ]
        summary = compute_consistency_summary(requests)
        assert summary["latency_cv"] == 0.0


# ------------------------------------------------------------------
# BenchmarkResult model field
# ------------------------------------------------------------------


class TestBenchmarkResultField:
    def test_consistency_summary_default_none(self):
        result = BenchmarkResult()
        assert result.consistency_summary is None

    def test_consistency_summary_assignable(self):
        result = BenchmarkResult()
        result.consistency_summary = {"is_deterministic": True}
        assert result.consistency_summary["is_deterministic"] is True


# ------------------------------------------------------------------
# CLI integration
# ------------------------------------------------------------------


class TestConsistencyCLI:
    def test_cli_flag_parsed(self):
        """--consistency-check N is parsed into args."""
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--consistency-check", "5"])
        assert args.consistency_check == 5

    def test_cli_flag_default_none(self):
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args([])
        assert args.consistency_check is None


# ------------------------------------------------------------------
# Config key
# ------------------------------------------------------------------


class TestConfigKey:
    def test_consistency_check_in_known_keys(self):
        from xpyd_bench.config_cmd import _KNOWN_KEYS

        assert "consistency_check" in _KNOWN_KEYS
