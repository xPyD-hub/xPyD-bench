"""Tests for M9: Warmup Requests feature."""

from __future__ import annotations

import json
from argparse import Namespace
from unittest.mock import patch

import pytest

from xpyd_bench.bench.runner import run_benchmark


def _default_args(**overrides) -> Namespace:
    """Create a default args Namespace for benchmarking."""
    defaults = {
        "backend": "openai",
        "base_url": None,
        "host": "127.0.0.1",
        "port": 8000,
        "endpoint": "/v1/completions",
        "model": "test-model",
        "num_prompts": 3,
        "request_rate": float("inf"),
        "burstiness": 1.0,
        "max_concurrency": None,
        "input_len": 16,
        "output_len": 8,
        "dataset_name": "random",
        "dataset_path": None,
        "seed": 42,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "frequency_penalty": None,
        "presence_penalty": None,
        "logprobs": None,
        "stop": None,
        "n": None,
        "api_seed": None,
        "echo": False,
        "suffix": None,
        "logit_bias": None,
        "user": None,
        "stream_options_include_usage": False,
        "best_of": None,
        "use_beam_search": False,
        "ignore_eos": False,
        "save_result": False,
        "result_dir": None,
        "result_filename": None,
        "metadata": None,
        "disable_tqdm": True,
        "config": None,
        "scenario": None,
        "list_scenarios": False,
        "rich_progress": False,
        "export_requests": None,
        "json_report": None,
        "synthetic_input_len_dist": "fixed",
        "synthetic_output_len_dist": "fixed",
        "warmup": None,
        "response_format": None,
        "tools": None,
        "tool_choice": None,
        "parallel_tool_calls": None,
        "top_logprobs": None,
        "max_completion_tokens": None,
        "service_tier": None,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def _fake_response(status_code: int = 200) -> dict:
    """Build a fake completions response."""
    return {
        "id": "cmpl-test",
        "object": "text_completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "text": "hello world response tokens",
                "index": 0,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 16, "completion_tokens": 8, "total_tokens": 24},
    }


class FakeHTTPResponse:
    """Minimal fake httpx.Response for non-streaming requests."""

    def __init__(self, data: dict, status_code: int = 200):
        self.status_code = status_code
        self._data = data
        self.text = json.dumps(data)

    def json(self) -> dict:
        return self._data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


class FakeAsyncClient:
    """Fake httpx.AsyncClient tracking request counts."""

    def __init__(self):
        self.request_count = 0

    async def post(self, url: str, json: dict | None = None, **kwargs) -> FakeHTTPResponse:
        self.request_count += 1
        return FakeHTTPResponse(_fake_response())

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


@pytest.mark.asyncio
async def test_warmup_zero_default():
    """With no warmup (default None), no warmup requests are sent."""
    args = _default_args(warmup=None, num_prompts=2)
    client = FakeAsyncClient()

    with patch("httpx.AsyncClient", return_value=client):
        _, result = await run_benchmark(args, "http://localhost:8000")

    # Only benchmark requests, no warmup
    assert result.completed + result.failed == 2
    # Total requests = 2 (no warmup)
    assert client.request_count == 2


@pytest.mark.asyncio
async def test_warmup_sends_extra_requests():
    """With --warmup 3, 3 warmup + N benchmark requests are sent."""
    args = _default_args(warmup=3, num_prompts=2)

    # We need two separate clients: one for warmup, one for benchmark
    clients = [FakeAsyncClient(), FakeAsyncClient()]
    call_idx = {"i": 0}

    class MultiClient:
        def __init__(self):
            self.idx = call_idx["i"]
            call_idx["i"] += 1
            self._client = clients[self.idx] if self.idx < len(clients) else FakeAsyncClient()

        async def post(self, url, json=None, **kwargs):
            return await self._client.post(url, json=json, **kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    with patch("httpx.AsyncClient", MultiClient):
        _, result = await run_benchmark(args, "http://localhost:8000")

    # Warmup client got 3 requests
    assert clients[0].request_count == 3
    # Benchmark client got 2 requests
    assert clients[1].request_count == 2
    # Result only contains benchmark requests
    assert result.completed + result.failed == 2


@pytest.mark.asyncio
async def test_warmup_excluded_from_metrics():
    """Warmup results must NOT appear in BenchmarkResult.requests."""
    args = _default_args(warmup=2, num_prompts=3)

    clients = [FakeAsyncClient(), FakeAsyncClient()]
    call_idx = {"i": 0}

    class MultiClient:
        def __init__(self):
            self.idx = call_idx["i"]
            call_idx["i"] += 1
            self._client = clients[self.idx] if self.idx < len(clients) else FakeAsyncClient()

        async def post(self, url, json=None, **kwargs):
            return await self._client.post(url, json=json, **kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    with patch("httpx.AsyncClient", MultiClient):
        _, result = await run_benchmark(args, "http://localhost:8000")

    # Only 3 benchmark requests in results
    assert len(result.requests) == 3
    assert result.num_prompts == 3


@pytest.mark.asyncio
async def test_warmup_with_chat_endpoint():
    """Warmup works with chat completions endpoint."""
    args = _default_args(
        warmup=1,
        num_prompts=1,
        endpoint="/v1/chat/completions",
    )

    # For chat streaming, we need a stream-capable fake
    class FakeStreamResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        async def aiter_lines(self):
            data = {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "choices": [{"delta": {"content": "hi"}, "index": 0, "finish_reason": None}],
            }
            yield f"data: {json.dumps(data)}"
            done = {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 16, "completion_tokens": 1, "total_tokens": 17},
            }
            yield f"data: {json.dumps(done)}"
            yield "data: [DONE]"

        async def aclose(self):
            pass

    class FakeStreamClient:
        def __init__(self):
            self.count = 0

        def stream(self, method, url, **kwargs):
            self.count += 1
            return _AsyncCtx(FakeStreamResponse())

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class _AsyncCtx:
        def __init__(self, resp):
            self.resp = resp

        async def __aenter__(self):
            return self.resp

        async def __aexit__(self, *args):
            if hasattr(self.resp, "aclose"):
                await self.resp.aclose()

    stream_clients = [FakeStreamClient(), FakeStreamClient()]
    call_idx = {"i": 0}

    class MultiClient:
        def __init__(self):
            self.idx = call_idx["i"]
            call_idx["i"] += 1
            self._client = (
                stream_clients[self.idx]
                if self.idx < len(stream_clients)
                else FakeStreamClient()
            )

        def stream(self, method, url, **kwargs):
            return self._client.stream(method, url, **kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    with patch("httpx.AsyncClient", MultiClient):
        _, result = await run_benchmark(args, "http://localhost:8000")

    # 1 warmup + 1 benchmark
    assert stream_clients[0].count == 1  # warmup
    assert stream_clients[1].count == 1  # benchmark


def test_cli_warmup_arg():
    """CLI parses --warmup correctly."""
    import argparse

    from xpyd_bench.cli import _add_vllm_compat_args

    parser = argparse.ArgumentParser()
    _add_vllm_compat_args(parser)

    args = parser.parse_args(["--warmup", "5"])
    assert args.warmup == 5


def test_cli_warmup_default_none():
    """CLI --warmup defaults to None (so YAML can override)."""
    import argparse

    from xpyd_bench.cli import _add_vllm_compat_args

    parser = argparse.ArgumentParser()
    _add_vllm_compat_args(parser)

    args = parser.parse_args([])
    assert args.warmup is None


def test_yaml_config_warmup(tmp_path):
    """YAML config can set warmup."""
    import argparse

    from xpyd_bench.cli import _load_yaml_config

    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("warmup: 10\n")

    args = argparse.Namespace(warmup=None)
    args = _load_yaml_config(str(yaml_file), args)
    assert args.warmup == 10


def test_yaml_config_warmup_cli_precedence(tmp_path):
    """CLI --warmup takes precedence over YAML config."""
    import argparse

    from xpyd_bench.cli import _load_yaml_config

    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("warmup: 10\n")

    args = argparse.Namespace(warmup=5)
    args = _load_yaml_config(str(yaml_file), args)
    # CLI value 5 is not None, so YAML doesn't override
    assert args.warmup == 5
