"""Dummy OpenAI-compatible server simulating vLLM prefill/decode behavior."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route


@dataclass
class ServerConfig:
    """Configuration for the dummy server."""

    prefill_ms: float = 50.0  # Simulated prefill latency
    decode_ms: float = 10.0  # Simulated per-token decode latency
    model_name: str = "dummy-model"
    max_tokens_default: int = 128


# Global config — set before starting the server
_config = ServerConfig()


def set_config(config: ServerConfig) -> None:
    """Set server configuration."""
    global _config  # noqa: PLW0603
    _config = config


def _estimate_prompt_tokens(prompt: str | list | None, messages: list | None) -> int:
    """Rough token count estimation (~4 chars per token)."""
    if prompt is not None:
        if isinstance(prompt, str):
            return max(1, len(prompt) // 4)
        if isinstance(prompt, list):
            total = sum(len(str(p)) for p in prompt)
            return max(1, total // 4)
    if messages is not None:
        total = sum(len(m.get("content", "")) for m in messages)
        return max(1, total // 4)
    return 1


def _make_completion_id() -> str:
    return f"cmpl-{uuid.uuid4().hex[:24]}"


def _make_chat_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


# ---------------------------------------------------------------------------
# /v1/completions
# ---------------------------------------------------------------------------


async def _handle_completions(request: Request) -> JSONResponse | StreamingResponse:
    body = await request.json()
    prompt = body.get("prompt", "")
    max_tokens = body.get("max_tokens", _config.max_tokens_default)
    stream = body.get("stream", False)
    model = body.get("model", _config.model_name)
    prompt_tokens = _estimate_prompt_tokens(prompt, None)

    if stream:
        return StreamingResponse(
            _stream_completions(model, prompt_tokens, max_tokens),
            media_type="text/event-stream",
        )

    # Non-streaming: simulate full latency
    total_ms = _config.prefill_ms + _config.decode_ms * max_tokens
    await asyncio.sleep(total_ms / 1000.0)

    generated_text = " ".join(["token"] * max_tokens)
    return JSONResponse({
        "id": _make_completion_id(),
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": generated_text,
                "finish_reason": "length",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": max_tokens,
            "total_tokens": prompt_tokens + max_tokens,
        },
    })


async def _stream_completions(model: str, prompt_tokens: int, max_tokens: int):
    """Generate SSE stream for completions endpoint."""
    comp_id = _make_completion_id()
    created = int(time.time())

    # Prefill delay
    await asyncio.sleep(_config.prefill_ms / 1000.0)

    for i in range(max_tokens):
        chunk = {
            "id": comp_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "text": "token " if i < max_tokens - 1 else "token",
                    "finish_reason": None if i < max_tokens - 1 else "length",
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(_config.decode_ms / 1000.0)

    # Include usage in final data before [DONE]
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# /v1/chat/completions
# ---------------------------------------------------------------------------


async def _handle_chat_completions(request: Request) -> JSONResponse | StreamingResponse:
    body = await request.json()
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", _config.max_tokens_default)
    stream = body.get("stream", False)
    model = body.get("model", _config.model_name)
    prompt_tokens = _estimate_prompt_tokens(None, messages)

    if stream:
        return StreamingResponse(
            _stream_chat_completions(model, prompt_tokens, max_tokens),
            media_type="text/event-stream",
        )

    # Non-streaming
    total_ms = _config.prefill_ms + _config.decode_ms * max_tokens
    await asyncio.sleep(total_ms / 1000.0)

    generated_text = " ".join(["token"] * max_tokens)
    return JSONResponse({
        "id": _make_chat_completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": generated_text},
                "finish_reason": "length",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": max_tokens,
            "total_tokens": prompt_tokens + max_tokens,
        },
    })


async def _stream_chat_completions(model: str, prompt_tokens: int, max_tokens: int):
    """Generate SSE stream for chat completions endpoint."""
    comp_id = _make_chat_completion_id()
    created = int(time.time())

    # Prefill delay
    await asyncio.sleep(_config.prefill_ms / 1000.0)

    for i in range(max_tokens):
        chunk = {
            "id": comp_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": "token " if i < max_tokens - 1 else "token",
                    },
                    "finish_reason": None if i < max_tokens - 1 else "length",
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(_config.decode_ms / 1000.0)

    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------


async def _handle_models(request: Request) -> JSONResponse:
    return JSONResponse({
        "object": "list",
        "data": [
            {
                "id": _config.model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "xpyd-dummy",
            }
        ],
    })


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


async def _handle_health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(config: ServerConfig | None = None) -> Starlette:
    """Create the dummy server ASGI application."""
    if config is not None:
        set_config(config)

    routes = [
        Route("/v1/completions", _handle_completions, methods=["POST"]),
        Route("/v1/chat/completions", _handle_chat_completions, methods=["POST"]),
        Route("/v1/models", _handle_models, methods=["GET"]),
        Route("/health", _handle_health, methods=["GET"]),
    ]
    return Starlette(routes=routes)
