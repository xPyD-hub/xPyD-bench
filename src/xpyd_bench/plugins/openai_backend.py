"""Built-in OpenAI-compatible backend plugin (M31).

Wraps the existing ``runner._send_request`` / ``runner._build_payload`` so
the default behaviour is unchanged while satisfying the plugin interface.
"""

from __future__ import annotations

from argparse import Namespace
from typing import Any

from xpyd_bench.bench.models import RequestResult
from xpyd_bench.plugins import BackendPlugin


class OpenAIBackendPlugin(BackendPlugin):
    """Default backend — speaks the OpenAI HTTP API."""

    @property
    def name(self) -> str:
        return "openai"

    def build_payload(
        self,
        args: Namespace,
        prompt: str,
        *,
        is_chat: bool = False,
        is_embeddings: bool = False,
    ) -> dict[str, Any]:
        from xpyd_bench.bench.runner import _build_payload

        return _build_payload(args, prompt, is_chat, is_embeddings)

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
        from xpyd_bench.bench.runner import _send_request

        return await _send_request(
            client,
            url,
            payload,
            is_streaming,
            request_timeout=request_timeout,
            retries=retries,
            retry_delay=retry_delay,
        )
