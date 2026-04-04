"""Tests for issue #77: streaming TPOT calc with stream_usage override.

Verifies that when stream_usage reports more completion_tokens than the
number of content chunks received, the TPOT calculation uses the
server-reported token count instead of the chunk count.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

from xpyd_bench.bench.runner import _send_streaming


def _make_mock_stream(lines: list[str]):
    """Build a mock httpx streaming response from SSE lines."""

    async def mock_aiter_lines():
        for line in lines:
            yield line

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.aiter_lines = mock_aiter_lines
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=mock_resp)
    return mock_client


class TestStreamUsageTPOT:
    """stream_usage completion_tokens should be used for TPOT calculation."""

    def test_tpot_uses_stream_usage_token_count(self):
        """When stream_usage reports more tokens than chunks, TPOT uses usage count."""
        # 2 content chunks but usage says 10 tokens
        lines = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            "data: "
            '{"choices":[],"usage":'
            '{"prompt_tokens":5,"completion_tokens":10,"total_tokens":15}}',
            "data: [DONE]",
        ]
        client = _make_mock_stream(lines)
        payload = {"model": "test", "max_tokens": 10}

        async def run():
            return await _send_streaming(
                client,
                "http://localhost/v1/chat/completions",
                payload,
                time.perf_counter(),
            )

        result = asyncio.run(run())
        assert result.completion_tokens == 10
        # With 10 tokens (from usage), TPOT should be computed using (10-1)=9
        # not (2-1)=1 which would give a much larger value
        assert result.tpot_ms is not None

    def test_single_chunk_with_stream_usage_gets_tpot(self):
        """Single content chunk but usage says 5 tokens should still compute TPOT."""
        lines = [
            'data: {"choices":[{"delta":{"content":"Hi"}}]}',
            "data: "
            '{"choices":[],"usage":'
            '{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8}}',
            "data: [DONE]",
        ]
        client = _make_mock_stream(lines)
        payload = {"model": "test", "max_tokens": 10}

        async def run():
            return await _send_streaming(
                client,
                "http://localhost/v1/chat/completions",
                payload,
                time.perf_counter(),
            )

        result = asyncio.run(run())
        assert result.completion_tokens == 5
        # Without fix: token_count=1, TPOT would be None
        # With fix: token_count=5, TPOT = generation_time / 4
        # However, with only 1 chunk, first_token_time == last_token_time
        # so generation_time_ms = 0, tpot_ms = 0.0
        # The key is that tpot_ms IS computed (not None)
        assert result.tpot_ms is not None

    def test_single_token_without_usage_has_no_tpot(self):
        """Single token without stream_usage should still have tpot_ms=None."""
        lines = [
            'data: {"choices":[{"delta":{"content":"Hi"}}]}',
            "data: [DONE]",
        ]
        client = _make_mock_stream(lines)
        payload = {"model": "test", "max_tokens": 1}

        async def run():
            return await _send_streaming(
                client,
                "http://localhost/v1/chat/completions",
                payload,
                time.perf_counter(),
            )

        result = asyncio.run(run())
        assert result.completion_tokens == 1
        assert result.tpot_ms is None
