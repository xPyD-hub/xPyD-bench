"""Example vLLM native protocol backend plugin (M31).

Demonstrates how to implement a custom backend for xpyd-bench.
This plugin targets the vLLM ``/generate`` endpoint which uses a
different payload format than the OpenAI-compatible API.

Usage::

    xpyd-bench run --backend vllm-native --backend-plugin xpyd_bench.plugins.examples.vllm_native \\
        --base-url http://localhost:8000 --model my-model --num-prompts 10

Or register via entry point in your ``pyproject.toml``::

    [project.entry-points."xpyd.backends"]
    vllm-native = "xpyd_bench.plugins.examples.vllm_native:Plugin"
"""

from __future__ import annotations

import json
import time
from argparse import Namespace
from typing import Any

from xpyd_bench.bench.models import RequestResult
from xpyd_bench.plugins import BackendPlugin


class Plugin(BackendPlugin):
    """Backend plugin for vLLM native ``/generate`` endpoint."""

    @property
    def name(self) -> str:
        return "vllm-native"

    def build_url(self, base_url: str, args: Namespace) -> str:
        """vLLM native endpoint is ``/generate``."""
        return f"{base_url.rstrip('/')}/generate"

    def build_payload(
        self,
        args: Namespace,
        prompt: str,
        *,
        is_chat: bool = False,
        is_embeddings: bool = False,
    ) -> dict[str, Any]:
        """Build a vLLM native ``/generate`` payload."""
        payload: dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": getattr(args, "output_len", 128),
        }
        if args.model:
            payload["model"] = args.model
        if getattr(args, "temperature", None) is not None:
            payload["temperature"] = args.temperature
        if getattr(args, "top_p", None) is not None:
            payload["top_p"] = args.top_p
        if getattr(args, "top_k", None) is not None:
            payload["top_k"] = args.top_k
        if getattr(args, "stream", False):
            payload["stream"] = True
        return payload

    async def send_request(
        self,
        client: Any,
        url: str,
        payload: dict[str, Any],
        *,
        is_streaming: bool = False,
        request_timeout: float = 300.0,
        retries: int = 0,
        retry_delay: float = 1.0,
    ) -> RequestResult:
        """Send a request to vLLM native endpoint."""
        result = RequestResult()
        start = time.perf_counter()
        result.start_time = start

        try:
            if is_streaming:
                result = await self._stream(client, url, payload, start, request_timeout)
            else:
                resp = await client.post(url, json=payload, timeout=request_timeout)
                resp.raise_for_status()
                end = time.perf_counter()
                result.latency_ms = (end - start) * 1000.0

                body = resp.json()
                # vLLM native returns text in "text" field (list)
                texts = body.get("text", [])
                if texts:
                    # Rough token estimate: split by whitespace
                    result.completion_tokens = sum(
                        len(t.split()) for t in texts
                    )
        except Exception as exc:  # noqa: BLE001
            end = time.perf_counter()
            result.latency_ms = (end - start) * 1000.0
            result.success = False
            result.error = str(exc)

        return result

    async def _stream(
        self,
        client: Any,
        url: str,
        payload: dict[str, Any],
        start: float,
        request_timeout: float,
    ) -> RequestResult:
        """Handle streaming from vLLM native endpoint."""
        result = RequestResult()
        result.start_time = start
        first_token_time: float | None = None
        last_token_time = start
        token_count = 0

        try:
            async with client.stream("POST", url, json=payload, timeout=request_timeout) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):]
                    if data_str.strip() == "[DONE]":
                        break
                    now = time.perf_counter()
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    texts = chunk.get("text", [])
                    if not texts or not any(texts):
                        continue
                    token_count += 1
                    if first_token_time is None:
                        first_token_time = now
                        result.ttft_ms = (now - start) * 1000.0
                    else:
                        result.itl_ms.append((now - last_token_time) * 1000.0)
                    last_token_time = now
        except Exception as exc:  # noqa: BLE001
            result.success = False
            result.error = str(exc)

        end = time.perf_counter()
        result.latency_ms = (end - start) * 1000.0
        result.completion_tokens = token_count
        if token_count > 1 and first_token_time is not None:
            generation_ms = (last_token_time - first_token_time) * 1000.0
            result.tpot_ms = generation_ms / (token_count - 1)
        return result


# Module-level instance for ``load_module_plugin`` convenience
plugin = Plugin()
