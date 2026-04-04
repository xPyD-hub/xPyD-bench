"""Tests for M16: Token Bucket & Adaptive Concurrency."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest

from xpyd_bench.bench.token_bucket import AdaptiveConcurrencyLimiter, TokenBucket

# ---------------------------------------------------------------------------
# TokenBucket
# ---------------------------------------------------------------------------


class TestTokenBucket:
    def test_invalid_rate(self) -> None:
        with pytest.raises(ValueError, match="rate must be positive"):
            TokenBucket(rate=0)

    def test_invalid_burst(self) -> None:
        with pytest.raises(ValueError, match="burst must be positive"):
            TokenBucket(rate=10, burst=0)

    def test_default_burst_equals_rate(self) -> None:
        tb = TokenBucket(rate=5.0)
        assert tb.burst == 5.0

    def test_custom_burst(self) -> None:
        tb = TokenBucket(rate=10.0, burst=20.0)
        assert tb.burst == 20.0

    @pytest.mark.asyncio
    async def test_acquire_immediate_when_tokens_available(self) -> None:
        tb = TokenBucket(rate=100.0, burst=10.0)
        start = time.monotonic()
        wait = await tb.acquire()
        elapsed = time.monotonic() - start
        assert wait == 0.0
        assert elapsed < 0.05

    @pytest.mark.asyncio
    async def test_acquire_waits_when_empty(self) -> None:
        tb = TokenBucket(rate=10.0, burst=1.0)
        # First acquire should be instant (bucket starts full)
        await tb.acquire()
        # Second should wait ~0.1s (1/rate)
        start = time.monotonic()
        await tb.acquire()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.05  # allow some slack

    @pytest.mark.asyncio
    async def test_burst_allows_multiple_immediate(self) -> None:
        tb = TokenBucket(rate=10.0, burst=5.0)
        start = time.monotonic()
        for _ in range(5):
            await tb.acquire()
        elapsed = time.monotonic() - start
        # All 5 should be near-instant (burst capacity)
        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_rate_accuracy(self) -> None:
        """Verify ~10 requests/sec over a short window."""
        rate = 10.0
        tb = TokenBucket(rate=rate, burst=1.0)
        # Drain the initial token
        await tb.acquire()
        count = 5
        start = time.monotonic()
        for _ in range(count):
            await tb.acquire()
        elapsed = time.monotonic() - start
        expected = count / rate
        # Allow 50% tolerance for CI variance
        assert elapsed >= expected * 0.5
        assert elapsed <= expected * 2.0


# ---------------------------------------------------------------------------
# AdaptiveConcurrencyLimiter
# ---------------------------------------------------------------------------


class TestAdaptiveConcurrencyLimiter:
    def test_initial_limit(self) -> None:
        acl = AdaptiveConcurrencyLimiter(initial=8)
        assert acl.limit == 8

    @pytest.mark.asyncio
    async def test_acquire_release(self) -> None:
        acl = AdaptiveConcurrencyLimiter(initial=2)
        await acl.acquire()
        await acl.acquire()
        acl.release()
        acl.release()

    @pytest.mark.asyncio
    async def test_increases_when_latency_low(self) -> None:
        acl = AdaptiveConcurrencyLimiter(
            initial=4,
            target_latency_ms=100.0,
            increase_step=2,
            max_concurrency=20,
            window_size=5,
        )
        # Record latencies well below target
        for _ in range(5):
            await acl.record_latency(20.0)
        assert acl.limit > 4

    @pytest.mark.asyncio
    async def test_decreases_when_latency_high(self) -> None:
        acl = AdaptiveConcurrencyLimiter(
            initial=16,
            target_latency_ms=100.0,
            decrease_factor=0.5,
            min_concurrency=1,
            window_size=5,
        )
        for _ in range(5):
            await acl.record_latency(500.0)
        assert acl.limit < 16

    @pytest.mark.asyncio
    async def test_respects_min_concurrency(self) -> None:
        acl = AdaptiveConcurrencyLimiter(
            initial=2,
            min_concurrency=2,
            target_latency_ms=10.0,
            decrease_factor=0.1,
            window_size=3,
        )
        for _ in range(10):
            await acl.record_latency(1000.0)
        assert acl.limit >= 2

    @pytest.mark.asyncio
    async def test_respects_max_concurrency(self) -> None:
        acl = AdaptiveConcurrencyLimiter(
            initial=10,
            max_concurrency=12,
            target_latency_ms=1000.0,
            increase_step=5,
            window_size=3,
        )
        for _ in range(10):
            await acl.record_latency(1.0)
        assert acl.limit <= 12

    @pytest.mark.asyncio
    async def test_no_adjust_with_few_samples(self) -> None:
        acl = AdaptiveConcurrencyLimiter(initial=8, window_size=10)
        await acl.record_latency(1000.0)
        await acl.record_latency(1000.0)
        # Only 2 samples, need >=3 to adjust
        assert acl.limit == 8


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


class TestCliArgs:
    def test_rate_algorithm_default(self) -> None:
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args([])
        assert args.rate_algorithm == "default"

    def test_rate_algorithm_token_bucket(self) -> None:
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--rate-algorithm", "token-bucket"])
        assert args.rate_algorithm == "token-bucket"

    def test_adaptive_concurrency_flag(self) -> None:
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--adaptive-concurrency"])
        assert args.adaptive_concurrency is True

    def test_adaptive_target_latency(self) -> None:
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--adaptive-target-latency", "200"])
        assert args.adaptive_target_latency == 200.0

    def test_token_bucket_burst(self) -> None:
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--token-bucket-burst", "50"])
        assert args.token_bucket_burst == 50.0


# ---------------------------------------------------------------------------
# Integration: runner uses token bucket
# ---------------------------------------------------------------------------


class TestRunnerTokenBucketIntegration:
    @pytest.mark.asyncio
    async def test_token_bucket_used_in_runner(self) -> None:
        """Verify that token-bucket algorithm is activated in runner."""
        from argparse import Namespace

        from xpyd_bench.bench.runner import run_benchmark

        args = Namespace(
            backend="openai",
            base_url=None,
            host="127.0.0.1",
            port=8000,
            endpoint="/v1/completions",
            model="test",
            num_prompts=3,
            request_rate=100.0,
            burstiness=1.0,
            max_concurrency=None,
            input_len=10,
            output_len=5,
            seed=0,
            disable_tqdm=True,
            save_result=False,
            warmup=0,
            api_key=None,
            custom_headers={},
            timeout=5.0,
            retries=0,
            retry_delay=1.0,
            config=None,
            dataset_path=None,
            dataset_name="random",
            rich_progress=False,
            scenario=None,
            rate_algorithm="token-bucket",
            token_bucket_burst=None,
            adaptive_concurrency=False,
            rate_pattern=None,
            temperature=None,
            top_p=None,
            top_k=None,
            frequency_penalty=None,
            presence_penalty=None,
            best_of=None,
            use_beam_search=False,
            logprobs=None,
            ignore_eos=False,
        )

        # Mock _send_request to avoid real HTTP
        mock_result = AsyncMock()
        mock_result.return_value.__class__ = type("MockResult", (), {
            "success": True, "latency_ms": 10.0, "prompt_tokens": 5,
            "completion_tokens": 5, "ttft_ms": None, "tpot_ms": 2.0,
            "itl_ms": [], "error": None, "retries": 0,
        })

        from xpyd_bench.bench.models import RequestResult

        def make_result(*a, **kw):
            r = RequestResult()
            r.latency_ms = 10.0
            r.prompt_tokens = 5
            r.completion_tokens = 5
            return r

        async def mock_send(*a, **kw):
            return make_result()

        with patch("xpyd_bench.bench.runner._send_request", side_effect=mock_send):
            result_dict, bench_result = await run_benchmark(args, "http://localhost:8000")

        assert bench_result.completed == 3

    @pytest.mark.asyncio
    async def test_adaptive_concurrency_in_runner(self) -> None:
        """Verify adaptive concurrency limiter is activated in runner."""
        from argparse import Namespace

        from xpyd_bench.bench.models import RequestResult
        from xpyd_bench.bench.runner import run_benchmark

        args = Namespace(
            backend="openai",
            base_url=None,
            host="127.0.0.1",
            port=8000,
            endpoint="/v1/completions",
            model="test",
            num_prompts=3,
            request_rate=float("inf"),
            burstiness=1.0,
            max_concurrency=None,
            input_len=10,
            output_len=5,
            seed=0,
            disable_tqdm=True,
            save_result=False,
            warmup=0,
            api_key=None,
            custom_headers={},
            timeout=5.0,
            retries=0,
            retry_delay=1.0,
            config=None,
            dataset_path=None,
            dataset_name="random",
            rich_progress=False,
            scenario=None,
            rate_algorithm="default",
            token_bucket_burst=None,
            adaptive_concurrency=True,
            adaptive_target_latency=500.0,
            adaptive_min_concurrency=1,
            adaptive_max_concurrency=64,
            adaptive_initial_concurrency=4,
            rate_pattern=None,
            temperature=None,
            top_p=None,
            top_k=None,
            frequency_penalty=None,
            presence_penalty=None,
            best_of=None,
            use_beam_search=False,
            logprobs=None,
            ignore_eos=False,
        )

        async def mock_send(*a, **kw):
            r = RequestResult()
            r.latency_ms = 10.0
            r.prompt_tokens = 5
            r.completion_tokens = 5
            return r

        with patch("xpyd_bench.bench.runner._send_request", side_effect=mock_send):
            result_dict, bench_result = await run_benchmark(args, "http://localhost:8000")

        assert bench_result.completed == 3
