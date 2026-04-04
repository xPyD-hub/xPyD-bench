"""Tests for configurable timeout and retry logic (M10)."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from xpyd_bench.bench.models import RequestResult
from xpyd_bench.bench.runner import _is_retryable, _send_request

# ---------------------------------------------------------------------------
# _is_retryable tests
# ---------------------------------------------------------------------------


class TestIsRetryable:
    def test_connect_error(self):
        exc = httpx.ConnectError("connection refused")
        assert _is_retryable(exc) is True

    def test_connect_timeout(self):
        exc = httpx.ConnectTimeout("timed out")
        assert _is_retryable(exc) is True

    def test_pool_timeout(self):
        exc = httpx.PoolTimeout("pool full")
        assert _is_retryable(exc) is True

    def test_http_429(self):
        resp = httpx.Response(429, request=httpx.Request("POST", "http://x"))
        exc = httpx.HTTPStatusError("rate limit", request=resp.request, response=resp)
        assert _is_retryable(exc) is True

    def test_http_503(self):
        resp = httpx.Response(503, request=httpx.Request("POST", "http://x"))
        exc = httpx.HTTPStatusError("unavailable", request=resp.request, response=resp)
        assert _is_retryable(exc) is True

    def test_http_400_not_retryable(self):
        resp = httpx.Response(400, request=httpx.Request("POST", "http://x"))
        exc = httpx.HTTPStatusError("bad request", request=resp.request, response=resp)
        assert _is_retryable(exc) is False

    def test_http_500_not_retryable(self):
        resp = httpx.Response(500, request=httpx.Request("POST", "http://x"))
        exc = httpx.HTTPStatusError("server error", request=resp.request, response=resp)
        assert _is_retryable(exc) is False

    def test_generic_exception_not_retryable(self):
        exc = ValueError("something")
        assert _is_retryable(exc) is False


# ---------------------------------------------------------------------------
# _send_request with timeout
# ---------------------------------------------------------------------------


class TestTimeout:
    @pytest.mark.asyncio
    async def test_custom_timeout_passed_to_client(self):
        """Verify custom timeout is used for non-streaming requests."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"usage": {"prompt_tokens": 5, "completion_tokens": 3}}

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_resp

        result = await _send_request(
            mock_client, "http://test/v1/completions", {"prompt": "hi"}, False,
            request_timeout=42.0,
        )

        assert result.success is True
        mock_client.post.assert_called_once()
        _, kwargs = mock_client.post.call_args
        assert kwargs["timeout"] == 42.0


# ---------------------------------------------------------------------------
# _send_request with retries
# ---------------------------------------------------------------------------


class TestRetry:
    @pytest.mark.asyncio
    async def test_retry_on_connect_error_then_succeed(self):
        """First call fails with ConnectError, second succeeds."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"usage": {"prompt_tokens": 5, "completion_tokens": 3}}

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.side_effect = [
            httpx.ConnectError("refused"),
            mock_resp,
        ]

        result = await _send_request(
            mock_client, "http://test/v1/completions", {"prompt": "hi"}, False,
            retries=2, retry_delay=0.01,
        )

        assert result.success is True
        assert result.retries == 1
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        """All retries fail → final result is failure."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.side_effect = httpx.ConnectError("refused")

        result = await _send_request(
            mock_client, "http://test/v1/completions", {"prompt": "hi"}, False,
            retries=2, retry_delay=0.01,
        )

        assert result.success is False
        assert result.retries == 2
        assert mock_client.post.call_count == 3  # 1 initial + 2 retries

    @pytest.mark.asyncio
    async def test_no_retry_on_non_retryable_error(self):
        """Non-retryable errors should not be retried."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.side_effect = ValueError("bad value")

        result = await _send_request(
            mock_client, "http://test/v1/completions", {"prompt": "hi"}, False,
            retries=3, retry_delay=0.01,
        )

        assert result.success is False
        assert result.retries == 0
        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_429(self):
        """HTTP 429 should trigger retry."""
        resp_429 = httpx.Response(429, request=httpx.Request("POST", "http://test"))
        resp_ok = MagicMock()
        resp_ok.raise_for_status = MagicMock()
        resp_ok.json.return_value = {"usage": {"prompt_tokens": 5, "completion_tokens": 3}}

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.side_effect = [
            httpx.HTTPStatusError("rate limit", request=resp_429.request, response=resp_429),
            resp_ok,
        ]

        result = await _send_request(
            mock_client, "http://test/v1/completions", {"prompt": "hi"}, False,
            retries=1, retry_delay=0.01,
        )

        assert result.success is True
        assert result.retries == 1

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self):
        """Verify exponential backoff increases delay between retries."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.side_effect = httpx.ConnectError("refused")

        start = time.monotonic()
        result = await _send_request(
            mock_client, "http://test/v1/completions", {"prompt": "hi"}, False,
            retries=2, retry_delay=0.05,
        )
        elapsed = time.monotonic() - start

        # retry_delay=0.05: attempt 0→1 sleeps 0.05, attempt 1→2 sleeps 0.10
        # total minimum sleep ≈ 0.15s
        assert result.success is False
        assert elapsed >= 0.12  # allow small timing margin

    @pytest.mark.asyncio
    async def test_zero_retries_no_retry(self):
        """retries=0 means no retry at all."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.side_effect = httpx.ConnectError("refused")

        result = await _send_request(
            mock_client, "http://test/v1/completions", {"prompt": "hi"}, False,
            retries=0, retry_delay=0.01,
        )

        assert result.success is False
        assert result.retries == 0
        assert mock_client.post.call_count == 1


# ---------------------------------------------------------------------------
# RequestResult.retries field
# ---------------------------------------------------------------------------


class TestRequestResultRetries:
    def test_default_retries(self):
        r = RequestResult()
        assert r.retries == 0

    def test_retries_field(self):
        r = RequestResult(retries=3)
        assert r.retries == 3


# ---------------------------------------------------------------------------
# CLI argument tests
# ---------------------------------------------------------------------------


class TestCLIArgs:
    def test_timeout_arg(self):

        import argparse

        # Just test the parser accepts the args
        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)

        args = parser.parse_args(["--timeout", "60"])
        assert args.timeout == 60.0

    def test_retries_arg(self):
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)

        args = parser.parse_args(["--retries", "3"])
        assert args.retries == 3

    def test_retry_delay_arg(self):
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)

        args = parser.parse_args(["--retry-delay", "2.5"])
        assert args.retry_delay == 2.5

    def test_default_values(self):
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)

        args = parser.parse_args([])
        assert args.timeout == 300.0
        assert args.retries == 0
        assert args.retry_delay == 1.0
