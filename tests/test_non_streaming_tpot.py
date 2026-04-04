"""Test that non-streaming requests do not compute TPOT (issue #93).

Non-streaming latency includes prefill time, so dividing by token count
would inflate TPOT compared to streaming (which correctly excludes TTFT).
The fix sets tpot_ms = None for non-streaming requests.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from xpyd_bench.bench.runner import _send_request


class TestNonStreamingTpotIsNone:
    """Non-streaming requests should NOT compute tpot_ms."""

    @pytest.mark.asyncio
    async def test_completions_non_streaming_tpot_is_none(self):
        """Non-streaming /v1/completions should leave tpot_ms as None."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"text": "hello world"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp

        result = await _send_request(
            client=mock_client,
            url="http://localhost:8000/v1/completions",
            payload={"prompt": "test", "max_tokens": 10},
            is_streaming=False,
            request_timeout=30.0,
            retries=0,
            retry_delay=0.0,
        )

        assert result.success is True
        assert result.completion_tokens == 5
        assert result.tpot_ms is None, (
            "Non-streaming TPOT should be None because latency includes prefill"
        )
        assert result.latency_ms is not None
        assert result.latency_ms > 0
