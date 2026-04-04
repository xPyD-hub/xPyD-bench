"""Tests for issue #16: stream_options parameter."""

from __future__ import annotations

import argparse
from argparse import Namespace

from xpyd_bench.bench.runner import _build_payload


class TestStreamOptionsInPayload:
    def _base_args(self, **kwargs):
        defaults = dict(
            output_len=10, model="m", temperature=None, top_p=None, top_k=None,
            frequency_penalty=None, presence_penalty=None, best_of=None,
            use_beam_search=False, logprobs=None, ignore_eos=False,
            stop=None, n=None, api_seed=None, echo=False, suffix=None,
            logit_bias=None, user=None, stream_options_include_usage=False,
        )
        defaults.update(kwargs)
        return Namespace(**defaults)

    def test_stream_options_included_when_true(self):
        args = self._base_args(stream_options_include_usage=True)
        payload = _build_payload(args, "hello", is_chat=False)
        assert payload["stream_options"] == {"include_usage": True}

    def test_stream_options_omitted_when_false(self):
        args = self._base_args(stream_options_include_usage=False)
        payload = _build_payload(args, "hello", is_chat=False)
        assert "stream_options" not in payload

    def test_stream_options_omitted_when_missing(self):
        args = self._base_args()
        delattr(args, "stream_options_include_usage")
        payload = _build_payload(args, "hello", is_chat=False)
        assert "stream_options" not in payload

    def test_stream_options_with_chat(self):
        args = self._base_args(stream_options_include_usage=True)
        payload = _build_payload(args, "hello", is_chat=True)
        assert payload["stream_options"] == {"include_usage": True}
        assert "messages" in payload


class TestStreamOptionsUsageCapture:
    """Test that streaming usage is captured from final chunk."""

    def test_usage_from_stream_chunk(self):
        """Integration-style test: verify _send_streaming captures usage."""
        import asyncio
        import time
        from unittest.mock import AsyncMock, MagicMock

        from xpyd_bench.bench.runner import _send_streaming

        # Build mock streaming response lines
        usage_data = (
            '{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}'
        )
        lines = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            f'data: {{"choices":[],"usage":{usage_data}}}',
            "data: [DONE]",
        ]

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

        payload = {"model": "test", "max_tokens": 10}

        async def run():
            return await _send_streaming(
                mock_client,
                "http://localhost/v1/chat/completions",
                payload,
                time.perf_counter(),
            )

        result = asyncio.run(run())
        assert result.prompt_tokens == 5
        assert result.completion_tokens == 2


class TestCLIParsing:
    def test_stream_options_flag(self):
        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--stream-options-include-usage"])
        assert args.stream_options_include_usage is True

    def test_stream_options_default_false(self):
        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args([])
        assert args.stream_options_include_usage is False
