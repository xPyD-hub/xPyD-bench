"""Tests for streaming with n>1 choices (issue #94)."""

import json
import time

import pytest

from xpyd_bench.bench.runner import _send_streaming

try:
    from unittest.mock import MagicMock
except ImportError:
    from unittest.mock import MagicMock


class FakeStreamResponse:
    """Simulate an httpx streaming response with controllable SSE lines."""

    def __init__(self, lines):
        self._lines = lines
        self.status_code = 200

    def raise_for_status(self):
        pass

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


@pytest.mark.asyncio
async def test_streaming_counts_all_n_choices():
    """When n=2, tokens from both choices should be counted."""
    chunks = []
    for i in range(3):
        chunk = {
            "choices": [
                {"index": 0, "delta": {"content": f"word{i}a"}},
                {"index": 1, "delta": {"content": f"word{i}b"}},
            ]
        }
        chunks.append(f"data: {json.dumps(chunk)}")
    chunks.append("data: [DONE]")

    fake_resp = FakeStreamResponse(chunks)
    client = MagicMock()
    client.stream = MagicMock(return_value=fake_resp)

    result = await _send_streaming(
        client=client,
        url="http://localhost:8000/v1/chat/completions",
        payload={"messages": [{"role": "user", "content": "hi"}]},
        start=time.perf_counter(),
        request_timeout=30.0,
    )

    assert result.success
    # 3 chunks × 2 choices = 6 tokens
    assert result.completion_tokens == 6


@pytest.mark.asyncio
async def test_streaming_single_choice_unchanged():
    """n=1 (default) still works correctly."""
    chunks = []
    for i in range(4):
        chunk = {"choices": [{"index": 0, "delta": {"content": f"tok{i}"}}]}
        chunks.append(f"data: {json.dumps(chunk)}")
    chunks.append("data: [DONE]")

    fake_resp = FakeStreamResponse(chunks)
    client = MagicMock()
    client.stream = MagicMock(return_value=fake_resp)

    result = await _send_streaming(
        client=client,
        url="http://localhost:8000/v1/completions",
        payload={"prompt": "hello"},
        start=time.perf_counter(),
        request_timeout=30.0,
    )

    assert result.success
    assert result.completion_tokens == 4
