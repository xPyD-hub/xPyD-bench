"""Dummy OpenAI-compatible server simulating vLLM prefill/decode behavior."""

from __future__ import annotations

import asyncio
import json
import random
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


def _normalize_prompt(prompt: str | list | None) -> str:
    """Normalize all 4 OpenAI prompt formats to a single string.

    Supported formats:
      1. string
      2. array of strings
      3. array of token integers
      4. array of mixed strings/token-arrays
    """
    if prompt is None:
        return ""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts: list[str] = []
        for item in prompt:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, int):
                # Token id — represent as placeholder text
                parts.append(f"<tok:{item}>")
            elif isinstance(item, list):
                # Sub-array of token ids
                parts.append("".join(f"<tok:{t}>" for t in item))
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(prompt)


def _estimate_prompt_tokens(prompt: str | list | None, messages: list | None) -> int:
    """Rough token count estimation (~4 chars per token)."""
    if prompt is not None:
        text = _normalize_prompt(prompt)
        return max(1, len(text) // 4)
    if messages is not None:
        total = sum(len(m.get("content", "")) for m in messages)
        return max(1, total // 4)
    return 1


def _make_completion_id() -> str:
    return f"cmpl-{uuid.uuid4().hex[:24]}"


def _make_chat_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

_PARAM_RANGES = {
    "temperature": (0.0, 2.0),
    "top_p": (0.0, 1.0),
    "frequency_penalty": (-2.0, 2.0),
    "presence_penalty": (-2.0, 2.0),
}


def _validate_params(body: dict) -> str | None:
    """Validate request parameters. Returns error message or None."""
    for param, (lo, hi) in _PARAM_RANGES.items():
        if param in body and body[param] is not None:
            val = body[param]
            if not isinstance(val, (int, float)):
                return f"'{param}' must be a number"
            if val < lo or val > hi:
                return f"'{param}' must be between {lo} and {hi}, got {val}"

    n = body.get("n")
    if n is not None:
        if not isinstance(n, int) or n < 1:
            return "'n' must be a positive integer"

    max_tokens = body.get("max_tokens")
    if max_tokens is not None:
        if not isinstance(max_tokens, int) or max_tokens < 1:
            return "'max_tokens' must be a positive integer"

    best_of = body.get("best_of")
    if best_of is not None:
        if not isinstance(best_of, int) or best_of < 1:
            return "'best_of' must be a positive integer"
        n_val = body.get("n", 1)
        if best_of < n_val:
            return f"'best_of' ({best_of}) must be >= 'n' ({n_val})"

    logprobs_val = body.get("logprobs")
    if logprobs_val is not None:
        # Completions: integer 0-5; Chat: boolean (handled separately)
        if isinstance(logprobs_val, bool):
            pass  # valid for chat
        elif isinstance(logprobs_val, int):
            if logprobs_val < 0 or logprobs_val > 5:
                return "'logprobs' must be between 0 and 5"
        else:
            return "'logprobs' must be an integer (0-5) or boolean"

    top_logprobs = body.get("top_logprobs")
    if top_logprobs is not None:
        if not isinstance(top_logprobs, int) or top_logprobs < 0 or top_logprobs > 20:
            return "'top_logprobs' must be an integer between 0 and 20"

    return None


def _make_logprobs_data(token: str, num_logprobs: int = 1) -> dict:
    """Generate dummy logprobs data for a token."""
    log_prob = -random.uniform(0.01, 5.0)
    top = [
        {
            "token": f"tok_{i}",
            "logprob": -random.uniform(0.01, 10.0),
            "bytes": list(f"tok_{i}".encode()),
        }
        for i in range(max(num_logprobs, 1))
    ]
    # Insert the actual token at position 0
    top[0] = {
        "token": token,
        "logprob": log_prob,
        "bytes": list(token.encode()),
    }
    return {
        "token": token,
        "logprob": log_prob,
        "bytes": list(token.encode()),
        "top_logprobs": top,
    }


def _check_stop(text: str, stop: list[str] | str | None) -> tuple[bool, str]:
    """Check if any stop sequence is found in text.

    Returns (should_stop, truncated_text).
    """
    if stop is None:
        return False, text
    if isinstance(stop, str):
        stop = [stop]
    for seq in stop:
        idx = text.find(seq)
        if idx >= 0:
            return True, text[:idx]
    return False, text


# ---------------------------------------------------------------------------
# /v1/completions
# ---------------------------------------------------------------------------


async def _handle_completions(request: Request) -> JSONResponse | StreamingResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {
                "error": {
                    "message": "Invalid JSON body",
                    "type": "invalid_request_error",
                    "code": "invalid_json",
                }
            },
            status_code=400,
        )

    # Validate parameters
    err = _validate_params(body)
    if err:
        return JSONResponse(
            {"error": {"message": err, "type": "invalid_request_error"}},
            status_code=400,
        )
    prompt = body.get("prompt", "")
    max_tokens = body.get("max_tokens", _config.max_tokens_default)
    stream = body.get("stream", False)
    model = body.get("model", _config.model_name)
    n = body.get("n", 1)
    seed = body.get("seed", None)
    echo = body.get("echo", False)
    stop = body.get("stop")
    logprobs_count = body.get("logprobs")  # int or None
    stream_options = body.get("stream_options") or {}
    include_usage = stream_options.get("include_usage", False)
    prompt_tokens = _estimate_prompt_tokens(prompt, None)
    prompt_text = _normalize_prompt(prompt)

    if stream:
        return StreamingResponse(
            _stream_completions(
                model,
                prompt_tokens,
                max_tokens,
                n=n,
                echo=echo,
                prompt_text=prompt_text,
                stop=stop,
                logprobs_count=logprobs_count,
                include_usage=include_usage,
                seed=seed,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming: simulate full latency
    total_ms = _config.prefill_ms + _config.decode_ms * max_tokens
    await asyncio.sleep(total_ms / 1000.0)

    choices = []
    total_completion_tokens = 0
    for i in range(n):
        generated_text = " ".join(["token"] * max_tokens)
        finish_reason = "length"
        stopped, generated_text = _check_stop(generated_text, stop)
        if stopped:
            finish_reason = "stop"

        if echo:
            generated_text = prompt_text + generated_text

        choice: dict = {
            "index": i,
            "text": generated_text,
            "finish_reason": finish_reason,
        }

        if logprobs_count is not None:
            tokens = generated_text.split(" ") if generated_text else []
            choice["logprobs"] = {
                "tokens": tokens,
                "token_logprobs": [-random.uniform(0.01, 5.0) for _ in tokens],
                "top_logprobs": [
                    {f"tok_{j}": -random.uniform(0.01, 10.0) for j in range(logprobs_count)}
                    for _ in tokens
                ],
                "text_offset": list(range(0, len(generated_text), max(1, len(generated_text) // max(len(tokens), 1)))),  # noqa: E501
            }

        # Count completion tokens based on generated token count (not echo)
        gen_token_count = len(generated_text.split()) if generated_text else 0
        if echo and prompt_text:
            # Subtract prompt tokens from word count
            prompt_word_count = len(prompt_text.split()) if prompt_text else 0
            gen_token_count = max(0, gen_token_count - prompt_word_count)
        total_completion_tokens += gen_token_count
        choices.append(choice)

    resp_body: dict = {
        "id": _make_completion_id(),
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": prompt_tokens + total_completion_tokens,
        },
    }
    if seed is not None:
        resp_body["system_fingerprint"] = f"fp_seed_{seed}"
    return JSONResponse(resp_body)


async def _stream_completions(
    model: str,
    prompt_tokens: int,
    max_tokens: int,
    *,
    n: int = 1,
    echo: bool = False,
    prompt_text: str = "",
    stop: list[str] | str | None = None,
    logprobs_count: int | None = None,
    include_usage: bool = False,
    seed: int | None = None,
):
    """Generate SSE stream for completions endpoint."""
    comp_id = _make_completion_id()
    created = int(time.time())

    # Prefill delay
    await asyncio.sleep(_config.prefill_ms / 1000.0)

    total_completion_tokens = 0

    for choice_idx in range(n):
        # Echo prefix as first chunk if requested
        if echo and prompt_text:
            chunk: dict = {
                "id": comp_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": choice_idx,
                        "text": prompt_text,
                        "finish_reason": None,
                    }
                ],
            }
            if seed is not None:
                chunk["system_fingerprint"] = f"fp_seed_{seed}"
            yield f"data: {json.dumps(chunk)}\n\n"

        accumulated = ""
        stopped = False
        for i in range(max_tokens):
            token_text = "token " if i < max_tokens - 1 else "token"
            accumulated += token_text

            # Check stop sequences
            if stop:
                stopped, _ = _check_stop(accumulated, stop)
                if stopped:
                    # Emit final chunk with stop reason
                    choice_data: dict = {
                        "index": choice_idx,
                        "text": token_text,
                        "finish_reason": "stop",
                    }
                    if logprobs_count is not None:
                        choice_data["logprobs"] = _make_logprobs_data(
                            token_text, logprobs_count
                        )
                    else:
                        choice_data["logprobs"] = None
                    chunk = {
                        "id": comp_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model,
                        "choices": [choice_data],
                    }
                    if seed is not None:
                        chunk["system_fingerprint"] = f"fp_seed_{seed}"
                    total_completion_tokens += i + 1
                    yield f"data: {json.dumps(chunk)}\n\n"
                    break

            is_last = i == max_tokens - 1
            choice_data = {
                "index": choice_idx,
                "text": token_text,
                "finish_reason": "length" if is_last else None,
            }
            if logprobs_count is not None:
                choice_data["logprobs"] = _make_logprobs_data(token_text, logprobs_count)
            else:
                choice_data["logprobs"] = None

            chunk = {
                "id": comp_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [choice_data],
            }
            if seed is not None:
                chunk["system_fingerprint"] = f"fp_seed_{seed}"
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(_config.decode_ms / 1000.0)

        if not stopped:
            total_completion_tokens += max_tokens

    # Include usage chunk if requested
    if include_usage:
        usage_chunk = {
            "id": comp_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": prompt_tokens + total_completion_tokens,
            },
        }
        yield f"data: {json.dumps(usage_chunk)}\n\n"

    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# /v1/chat/completions
# ---------------------------------------------------------------------------


async def _handle_chat_completions(request: Request) -> JSONResponse | StreamingResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {
                "error": {
                    "message": "Invalid JSON body",
                    "type": "invalid_request_error",
                    "code": "invalid_json",
                }
            },
            status_code=400,
        )

    # Validate parameters
    err = _validate_params(body)
    if err:
        return JSONResponse(
            {"error": {"message": err, "type": "invalid_request_error"}},
            status_code=400,
        )
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", _config.max_tokens_default)
    stream = body.get("stream", False)
    model = body.get("model", _config.model_name)
    n = body.get("n", 1)
    seed = body.get("seed", None)
    stop = body.get("stop")
    logprobs = body.get("logprobs", False)
    top_logprobs = body.get("top_logprobs")
    stream_options = body.get("stream_options") or {}
    include_usage = stream_options.get("include_usage", False)
    prompt_tokens = _estimate_prompt_tokens(None, messages)

    if stream:
        return StreamingResponse(
            _stream_chat_completions(
                model,
                prompt_tokens,
                max_tokens,
                n=n,
                stop=stop,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                include_usage=include_usage,
                seed=seed,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming
    total_ms = _config.prefill_ms + _config.decode_ms * max_tokens
    await asyncio.sleep(total_ms / 1000.0)

    choices = []
    total_completion_tokens = 0
    for i in range(n):
        generated_text = " ".join(["token"] * max_tokens)
        finish_reason = "length"
        stopped, generated_text = _check_stop(generated_text, stop)
        if stopped:
            finish_reason = "stop"

        choice: dict = {
            "index": i,
            "message": {"role": "assistant", "content": generated_text},
            "finish_reason": finish_reason,
        }

        if logprobs and top_logprobs is not None:
            tokens = generated_text.split(" ") if generated_text else []
            choice["logprobs"] = {
                "content": [
                    _make_logprobs_data(tok, top_logprobs) for tok in tokens
                ],
            }

        gen_token_count = len(generated_text.split()) if generated_text else 0
        total_completion_tokens += gen_token_count
        choices.append(choice)

    resp_body: dict = {
        "id": _make_chat_completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": prompt_tokens + total_completion_tokens,
        },
    }
    if seed is not None:
        resp_body["system_fingerprint"] = f"fp_seed_{seed}"
    return JSONResponse(resp_body)


async def _stream_chat_completions(
    model: str,
    prompt_tokens: int,
    max_tokens: int,
    *,
    n: int = 1,
    stop: list[str] | str | None = None,
    logprobs: bool = False,
    top_logprobs: int | None = None,
    include_usage: bool = False,
    seed: int | None = None,
):
    """Generate SSE stream for chat completions endpoint."""
    comp_id = _make_chat_completion_id()
    created = int(time.time())

    # Prefill delay
    await asyncio.sleep(_config.prefill_ms / 1000.0)

    total_completion_tokens = 0

    for choice_idx in range(n):
        # Emit initial chunk with role for each choice (OpenAI spec)
        role_chunk: dict = {
            "id": comp_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": choice_idx,
                    "delta": {"role": "assistant", "content": ""},
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
        }
        if seed is not None:
            role_chunk["system_fingerprint"] = f"fp_seed_{seed}"
        yield f"data: {json.dumps(role_chunk)}\n\n"

        accumulated = ""
        stopped = False
        for i in range(max_tokens):
            token_text = "token " if i < max_tokens - 1 else "token"
            accumulated += token_text

            # Check stop sequences
            if stop:
                stopped, _ = _check_stop(accumulated, stop)
                if stopped:
                    delta: dict = {"content": token_text}
                    choice_data: dict = {
                        "index": choice_idx,
                        "delta": delta,
                        "finish_reason": "stop",
                    }
                    if logprobs and top_logprobs is not None:
                        choice_data["logprobs"] = {
                            "content": [_make_logprobs_data(token_text, top_logprobs)]
                        }
                    else:
                        choice_data["logprobs"] = None
                    chunk: dict = {
                        "id": comp_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [choice_data],
                    }
                    if seed is not None:
                        chunk["system_fingerprint"] = f"fp_seed_{seed}"
                    total_completion_tokens += i + 1
                    yield f"data: {json.dumps(chunk)}\n\n"
                    break

            is_last = i == max_tokens - 1
            delta = {"content": token_text}
            choice_data = {
                "index": choice_idx,
                "delta": delta,
                "finish_reason": "length" if is_last else None,
            }
            if logprobs and top_logprobs is not None:
                choice_data["logprobs"] = {
                    "content": [_make_logprobs_data(token_text, top_logprobs)]
                }
            else:
                choice_data["logprobs"] = None

            chunk = {
                "id": comp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [choice_data],
            }
            if seed is not None:
                chunk["system_fingerprint"] = f"fp_seed_{seed}"
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(_config.decode_ms / 1000.0)

        if not stopped:
            total_completion_tokens += max_tokens

    # Include usage chunk if requested
    if include_usage:
        usage_chunk = {
            "id": comp_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": prompt_tokens + total_completion_tokens,
            },
        }
        yield f"data: {json.dumps(usage_chunk)}\n\n"

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
