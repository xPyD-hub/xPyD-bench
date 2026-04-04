"""Tests for metrics calculation and request scheduling."""

from __future__ import annotations

import numpy as np
import pytest

from xpyd_bench.bench.models import BenchmarkResult, RequestResult
from xpyd_bench.bench.runner import (
    _build_payload,
    _compute_metrics,
    _generate_intervals,
    _generate_random_prompts,
)


class TestGenerateIntervals:
    """Tests for Poisson / Gamma request scheduling."""

    def test_inf_rate_returns_zeros(self):
        intervals = _generate_intervals(10, float("inf"), 1.0, 42)
        assert len(intervals) == 10
        assert all(i == 0.0 for i in intervals)

    def test_poisson_process(self):
        """burstiness=1.0 should produce exponential intervals."""
        intervals = _generate_intervals(1000, 10.0, 1.0, 42)
        assert len(intervals) == 1000
        # Mean should be ~0.1s (1/10 rps)
        mean = np.mean(intervals)
        assert 0.05 < mean < 0.15

    def test_gamma_distribution(self):
        """burstiness!=1.0 should produce gamma-distributed intervals."""
        intervals = _generate_intervals(1000, 10.0, 0.5, 42)
        assert len(intervals) == 1000
        mean = np.mean(intervals)
        # Mean should still be ~0.1s
        assert 0.05 < mean < 0.15

    def test_deterministic_with_seed(self):
        a = _generate_intervals(100, 5.0, 1.0, 123)
        b = _generate_intervals(100, 5.0, 1.0, 123)
        assert a == b


class TestGenerateRandomPrompts:
    def test_correct_count(self):
        prompts = _generate_random_prompts(5, 100, 42)
        assert len(prompts) == 5

    def test_deterministic(self):
        a = _generate_random_prompts(3, 50, 0)
        b = _generate_random_prompts(3, 50, 0)
        assert a == b

    def test_approximate_length(self):
        prompts = _generate_random_prompts(1, 200, 0)
        # Each word is ~1 token, so word count should be ~200
        word_count = len(prompts[0].split())
        assert word_count == 200


class TestBuildPayload:
    def _make_args(self, **kwargs):
        import argparse

        defaults = {
            "output_len": 128,
            "model": "test-model",
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "frequency_penalty": None,
            "presence_penalty": None,
            "best_of": None,
            "use_beam_search": False,
            "logprobs": None,
            "ignore_eos": False,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_completions_payload(self):
        args = self._make_args()
        payload = _build_payload(args, "Hello world", is_chat=False)
        assert payload["prompt"] == "Hello world"
        assert payload["max_tokens"] == 128
        assert payload["model"] == "test-model"
        assert "messages" not in payload

    def test_chat_payload(self):
        args = self._make_args()
        payload = _build_payload(args, "Hello world", is_chat=True)
        assert "messages" in payload
        assert payload["messages"][0]["content"] == "Hello world"
        assert "prompt" not in payload

    def test_sampling_params(self):
        args = self._make_args(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            frequency_penalty=0.5,
            presence_penalty=0.3,
            best_of=2,
            use_beam_search=True,
            logprobs=3,
            ignore_eos=True,
        )
        payload = _build_payload(args, "test", is_chat=False)
        assert payload["temperature"] == 0.7
        assert payload["top_p"] == 0.9
        assert payload["top_k"] == 40
        assert payload["frequency_penalty"] == 0.5
        assert payload["presence_penalty"] == 0.3
        assert payload["best_of"] == 2
        assert payload["use_beam_search"] is True
        assert payload["logprobs"] == 3
        assert payload["ignore_eos"] is True

    def test_none_sampling_excluded(self):
        args = self._make_args()
        payload = _build_payload(args, "test", is_chat=False)
        assert "temperature" not in payload
        assert "top_p" not in payload


class TestComputeMetrics:
    def test_all_successful(self):
        result = BenchmarkResult(total_duration_s=10.0)
        result.requests = [
            RequestResult(
                prompt_tokens=100,
                completion_tokens=50,
                latency_ms=200.0,
                ttft_ms=20.0,
                tpot_ms=3.6,
                itl_ms=[3.5, 3.7],
            ),
            RequestResult(
                prompt_tokens=100,
                completion_tokens=60,
                latency_ms=250.0,
                ttft_ms=25.0,
                tpot_ms=3.8,
                itl_ms=[3.6, 4.0],
            ),
        ]
        _compute_metrics(result)

        assert result.completed == 2
        assert result.failed == 0
        assert result.total_input_tokens == 200
        assert result.total_output_tokens == 110
        assert result.request_throughput == pytest.approx(0.2, rel=0.01)
        assert result.output_throughput == pytest.approx(11.0, rel=0.01)
        assert result.mean_e2el_ms == pytest.approx(225.0, rel=0.01)

    def test_with_failures(self):
        result = BenchmarkResult(total_duration_s=5.0)
        result.requests = [
            RequestResult(latency_ms=100.0, completion_tokens=10, prompt_tokens=50),
            RequestResult(latency_ms=50.0, success=False, error="timeout"),
        ]
        _compute_metrics(result)

        assert result.completed == 1
        assert result.failed == 1
        assert result.request_throughput == pytest.approx(0.2, rel=0.01)

    def test_empty_results(self):
        result = BenchmarkResult(total_duration_s=1.0)
        result.requests = []
        _compute_metrics(result)
        assert result.completed == 0
        assert result.failed == 0
