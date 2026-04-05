"""Dummy OpenAI-compatible server simulating vLLM prefill/decode behavior."""

from __future__ import annotations

import asyncio
import gzip
import json
import random
import time
import uuid
from dataclasses import dataclass

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route


async def _parse_json_body(request: Request) -> dict:
    """Parse JSON body, handling gzip Content-Encoding transparently."""
    raw = await request.body()
    encoding = request.headers.get("content-encoding", "").lower()
    if encoding == "gzip":
        raw = gzip.decompress(raw)
    return json.loads(raw)


@dataclass
class ServerConfig:
    """Configuration for the dummy server."""

    prefill_ms: float = 50.0  # Simulated prefill latency
    decode_ms: float = 10.0  # Simulated per-token decode latency
    model_name: str = "dummy-model"
    max_tokens_default: int = 128
    eos_min_ratio: float = 0.5  # EOS won't fire before this fraction of max_tokens
    require_api_key: str | None = None  # If set, require this API key for auth
    embedding_dim: int = 1536  # Dimensionality for embedding vectors
    max_rps: float | None = None  # If set, reject requests above this rate (429)


# Global config — set before starting the server
_config = ServerConfig()


def set_config(config: ServerConfig) -> None:
    """Set server configuration."""
    global _config  # noqa: PLW0603
    _config = config


# Standard headers to exclude from echo
_STANDARD_HEADERS = frozenset({
    "host", "user-agent", "accept", "accept-encoding", "accept-language",
    "connection", "content-length", "content-type", "authorization",
    "transfer-encoding", "te",
})


def _extract_custom_headers(request: Request) -> dict[str, str]:
    """Extract non-standard HTTP headers from the request for echo."""
    custom: dict[str, str] = {}
    for key, value in request.headers.items():
        if key.lower() not in _STANDARD_HEADERS:
            custom[key] = value
    return custom


def _echo_headers_dict(request: Request) -> dict[str, str]:
    """Build response headers that echo custom request headers.

    Also explicitly echoes X-Request-ID for direct correlation (M42).
    """
    custom = _extract_custom_headers(request)
    result: dict[str, str] = {}
    req_id = request.headers.get("x-request-id")
    if req_id:
        result["X-Request-ID"] = req_id
    if custom:
        result["x-echo-headers"] = json.dumps(custom, separators=(",", ":"))
    return result


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


def _eos_index(max_tokens: int, ignore_eos: bool = False) -> int | None:
    """Return the 0-based token index at which to emit EOS, or None to skip.

    Uses ``_config.eos_min_ratio`` to set the lower bound.  When
    *ignore_eos* is ``True`` the model should produce all *max_tokens*
    so we return ``None``.
    """
    if ignore_eos or max_tokens <= 0:
        return None
    lo = max(1, int(max_tokens * _config.eos_min_ratio))
    return random.randint(lo, max_tokens)  # inclusive; max_tokens means "no early stop"


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
        body = await _parse_json_body(request)
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
    ignore_eos = body.get("ignore_eos", False)
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
                ignore_eos=ignore_eos,
            ),
            media_type="text/event-stream", headers=_echo_headers_dict(request),
        )

    # Non-streaming: simulate full latency
    total_ms = _config.prefill_ms + _config.decode_ms * max_tokens
    await asyncio.sleep(total_ms / 1000.0)

    choices = []
    total_completion_tokens = 0
    for i in range(n):
        eos_idx = _eos_index(max_tokens, ignore_eos)
        actual_tokens = eos_idx if eos_idx is not None and eos_idx < max_tokens else max_tokens
        generated_text = " ".join(["token"] * actual_tokens)
        finish_reason = "stop" if (eos_idx is not None and eos_idx < max_tokens) else "length"
        stopped, generated_text = _check_stop(generated_text, stop)
        if stopped:
            finish_reason = "stop"

        # Track generated token count before echo prepend
        gen_token_count = actual_tokens

        if echo:
            generated_text = prompt_text + " " + generated_text if generated_text else prompt_text

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
    return JSONResponse(resp_body, headers=_echo_headers_dict(request))


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
    ignore_eos: bool = False,
):
    """Generate SSE stream for completions endpoint."""
    comp_id = _make_completion_id()
    created = int(time.time())

    # Prefill delay
    await asyncio.sleep(_config.prefill_ms / 1000.0)

    total_completion_tokens = 0

    for choice_idx in range(n):
        eos_idx = _eos_index(max_tokens, ignore_eos)
        effective_max = eos_idx if eos_idx is not None and eos_idx < max_tokens else max_tokens

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
        for i in range(effective_max):
            token_text = "token " if i < effective_max - 1 else "token"
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

            is_last = i == effective_max - 1
            is_eos = effective_max < max_tokens and is_last
            finish_reason: str | None = None
            if is_last:
                finish_reason = "stop" if is_eos else "length"
            choice_data = {
                "index": choice_idx,
                "text": token_text,
                "finish_reason": finish_reason,
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
            total_completion_tokens += effective_max

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


def _build_tool_calls(
    tools: list[dict],
    tool_choice: str | dict | None = None,
    parallel: bool | None = None,
) -> list[dict]:
    """Build synthetic tool call response from tool definitions."""
    import uuid

    selected_tools: list[dict] = []
    if isinstance(tool_choice, dict):
        # Specific function requested
        fname = tool_choice.get("function", {}).get("name", "")
        for t in tools:
            if t.get("type") == "function" and t.get("function", {}).get("name") == fname:
                selected_tools.append(t)
                break
        if not selected_tools and tools:
            selected_tools.append(tools[0])
    elif tool_choice == "required" or tool_choice == "auto" or tool_choice is None:
        if parallel and len(tools) > 1:
            selected_tools = tools[:2]  # Return up to 2 tool calls for parallel
        else:
            selected_tools = tools[:1]
    # tool_choice == "none" should not reach here

    result = []
    for t in selected_tools:
        func = t.get("function", {})
        fname = func.get("name", "unknown")
        params = func.get("parameters", {})
        # Generate dummy arguments matching the schema
        args = _generate_dummy_args(params)
        result.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": fname,
                "arguments": json.dumps(args),
            },
        })
    return result


def _generate_dummy_args(schema: dict) -> dict:
    """Generate dummy arguments matching a JSON schema."""
    if not schema or schema.get("type") != "object":
        return {}
    props = schema.get("properties", {})
    result: dict = {}
    for key, prop in props.items():
        ptype = prop.get("type", "string")
        if ptype == "string":
            enum = prop.get("enum")
            result[key] = enum[0] if enum else "dummy_value"
        elif ptype == "integer":
            result[key] = 42
        elif ptype == "number":
            result[key] = 3.14
        elif ptype == "boolean":
            result[key] = True
        elif ptype == "array":
            result[key] = []
        elif ptype == "object":
            result[key] = _generate_dummy_args(prop)
    return result


def _build_json_response(response_format: dict, max_tokens: int) -> str:
    """Build a JSON response conforming to response_format spec."""
    fmt_type = response_format.get("type")
    if fmt_type == "json_schema":
        schema_def = response_format.get("json_schema", {})
        schema = schema_def.get("schema", {})
        if schema:
            data = _generate_dummy_args(schema) if schema.get("type") == "object" else {}
            return json.dumps(data)
    # json_object: return a simple JSON object
    return json.dumps({"result": " ".join(["token"] * min(max_tokens, 5))})


async def _handle_chat_completions(request: Request) -> JSONResponse | StreamingResponse:
    try:
        body = await _parse_json_body(request)
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
    ignore_eos = body.get("ignore_eos", False)
    stream_options = body.get("stream_options") or {}
    include_usage = stream_options.get("include_usage", False)
    # Chat-specific params
    response_format = body.get("response_format")
    tools = body.get("tools")
    tool_choice = body.get("tool_choice")
    parallel_tool_calls = body.get("parallel_tool_calls")
    max_completion_tokens = body.get("max_completion_tokens")
    body.get("service_tier")
    # Use max_completion_tokens as fallback for max_tokens if provided
    if max_completion_tokens is not None and body.get("max_tokens") is None:
        max_tokens = max_completion_tokens
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
                ignore_eos=ignore_eos,
            ),
            media_type="text/event-stream", headers=_echo_headers_dict(request),
        )

    # Non-streaming
    total_ms = _config.prefill_ms + _config.decode_ms * max_tokens
    await asyncio.sleep(total_ms / 1000.0)

    # Determine if we should generate tool calls
    _generate_tool_calls = tools and tool_choice != "none"

    choices = []
    total_completion_tokens = 0
    for i in range(n):
        eos_idx = _eos_index(max_tokens, ignore_eos)
        actual_tokens = eos_idx if eos_idx is not None and eos_idx < max_tokens else max_tokens

        if _generate_tool_calls:
            # Generate synthetic tool call response
            tool_calls_out = _build_tool_calls(tools, tool_choice, parallel_tool_calls)
            generated_text = None
            finish_reason = "tool_calls"
            # Estimate tokens from serialized tool calls
            tc_json = json.dumps(tool_calls_out)
            gen_token_count = max(1, len(tc_json.split()))
            choice: dict = {
                "index": i,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls_out,
                },
                "finish_reason": finish_reason,
            }
        elif response_format and response_format.get("type") in (
            "json_object",
            "json_schema",
        ):
            # Generate JSON-conforming response
            generated_text = _build_json_response(response_format, actual_tokens)
            finish_reason = "stop"
            gen_token_count = max(1, len(generated_text.split()))
            choice = {
                "index": i,
                "message": {"role": "assistant", "content": generated_text},
                "finish_reason": finish_reason,
            }
        else:
            generated_text = " ".join(["token"] * actual_tokens)
            finish_reason = "stop" if (eos_idx is not None and eos_idx < max_tokens) else "length"
            stopped, generated_text = _check_stop(generated_text, stop)
            if stopped:
                finish_reason = "stop"
            gen_token_count = len(generated_text.split()) if generated_text else 0
            choice = {
                "index": i,
                "message": {"role": "assistant", "content": generated_text},
                "finish_reason": finish_reason,
            }

        if logprobs and top_logprobs is not None and generated_text:
            tokens = generated_text.split(" ") if generated_text else []
            choice["logprobs"] = {
                "content": [
                    _make_logprobs_data(tok, top_logprobs) for tok in tokens
                ],
            }

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
    return JSONResponse(resp_body, headers=_echo_headers_dict(request))


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
    ignore_eos: bool = False,
):
    """Generate SSE stream for chat completions endpoint."""
    comp_id = _make_chat_completion_id()
    created = int(time.time())

    # Prefill delay
    await asyncio.sleep(_config.prefill_ms / 1000.0)

    total_completion_tokens = 0

    for choice_idx in range(n):
        eos_idx = _eos_index(max_tokens, ignore_eos)
        effective_max = eos_idx if eos_idx is not None and eos_idx < max_tokens else max_tokens

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
        for i in range(effective_max):
            token_text = "token " if i < effective_max - 1 else "token"
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

            is_last = i == effective_max - 1
            is_eos = effective_max < max_tokens and is_last
            finish_reason: str | None = None
            if is_last:
                finish_reason = "stop" if is_eos else "length"
            delta = {"content": token_text}
            choice_data = {
                "index": choice_idx,
                "delta": delta,
                "finish_reason": finish_reason,
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
            total_completion_tokens += effective_max

    # Include usage chunk if requested (chat)
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


async def _handle_embeddings(request: Request) -> JSONResponse:
    """Handle POST /v1/embeddings — return random embedding vectors."""
    cfg = _config
    body = await _parse_json_body(request)

    model = body.get("model", cfg.model_name)
    raw_input = body.get("input", "")
    encoding_format = body.get("encoding_format", "float")

    # Normalize input to list of strings
    if isinstance(raw_input, str):
        inputs = [raw_input]
    elif isinstance(raw_input, list):
        inputs = [str(item) for item in raw_input]
    else:
        inputs = [str(raw_input)]

    # Simulate prefill latency per input
    if cfg.prefill_ms > 0:
        await asyncio.sleep(cfg.prefill_ms * len(inputs) / 1000.0)

    # Count tokens (simple word-split approximation)
    total_tokens = sum(len(text.split()) for text in inputs)

    data = []
    for i, text in enumerate(inputs):
        # Deterministic pseudo-random vector based on input text
        rng = random.Random(hash(text) & 0xFFFFFFFF)
        vector = [rng.gauss(0, 1) for _ in range(cfg.embedding_dim)]

        if encoding_format == "base64":
            import base64
            import struct

            raw_bytes = struct.pack(f"{len(vector)}f", *vector)
            embedding_value = base64.b64encode(raw_bytes).decode("ascii")
        else:
            embedding_value = vector

        data.append({
            "object": "embedding",
            "index": i,
            "embedding": embedding_value,
        })

    custom = _extract_custom_headers(request)
    response_body: dict = {
        "object": "list",
        "data": data,
        "model": model,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        },
    }
    if custom:
        response_body["_echoed_headers"] = custom

    return JSONResponse(response_body)


async def _handle_health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


# ---------------------------------------------------------------------------
# Batch API (M41)
# ---------------------------------------------------------------------------

# In-memory batch store for the dummy server
_batches: dict[str, dict] = {}


async def _handle_create_batch(request: Request) -> JSONResponse:
    """Handle POST /v1/batches — create a batch job."""
    body = await _parse_json_body(request)
    cfg = _config

    input_requests = body.get("input", [])
    endpoint = body.get("endpoint", "/v1/completions")
    model = body.get("model", cfg.model_name)
    metadata = body.get("metadata", {})

    batch_id = f"batch_{uuid.uuid4().hex[:16]}"
    now = time.time()
    queue_delay = cfg.prefill_ms / 1000.0  # Simulate queue time using prefill_ms

    batch_obj: dict = {
        "id": batch_id,
        "object": "batch",
        "endpoint": endpoint,
        "model": model,
        "status": "validating",
        "created_at": now,
        "in_progress_at": None,
        "completed_at": None,
        "request_counts": {
            "total": len(input_requests),
            "completed": 0,
            "failed": 0,
        },
        "metadata": metadata,
        "input": input_requests,
        "results": [],
        "errors": {"object": "list", "data": []},
        "_queue_delay": queue_delay,
        "_process_delay": cfg.decode_ms * len(input_requests) / 1000.0,
    }
    _batches[batch_id] = batch_obj

    # Schedule async processing
    asyncio.create_task(_process_batch(batch_id))

    # Return initial state
    return JSONResponse({
        "id": batch_id,
        "object": "batch",
        "endpoint": endpoint,
        "status": "validating",
        "created_at": now,
        "request_counts": batch_obj["request_counts"],
        "metadata": metadata,
    })


async def _process_batch(batch_id: str) -> None:
    """Simulate batch processing in the background."""
    batch = _batches.get(batch_id)
    if not batch:
        return

    cfg = _config
    queue_delay = batch["_queue_delay"]
    process_delay = batch["_process_delay"]

    # Simulate queue time
    await asyncio.sleep(queue_delay)
    now = time.time()
    batch["status"] = "in_progress"
    batch["in_progress_at"] = now

    # Simulate processing time
    await asyncio.sleep(process_delay)

    # Generate dummy results
    results = []
    input_requests = batch.get("input", [])
    for req in input_requests:
        custom_id = req.get("custom_id", "")
        body = req.get("body", {})
        max_tokens = body.get("max_tokens", cfg.max_tokens_default)
        rng = random.Random(hash(custom_id) & 0xFFFFFFFF)
        n_tokens = rng.randint(
            max(1, int(max_tokens * cfg.eos_min_ratio)), max_tokens,
        )
        text = " ".join(
            rng.choice(["the", "a", "an", "is", "was", "will", "be", "to"])
            for _ in range(n_tokens)
        )
        results.append({
            "custom_id": custom_id,
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": len(
                            str(body.get("prompt", body.get("messages", ""))).split()
                        ),
                        "completion_tokens": n_tokens,
                        "total_tokens": len(
                            str(body.get("prompt", body.get("messages", ""))).split()
                        ) + n_tokens,
                    },
                },
            },
        })

    batch["results"] = results
    batch["request_counts"]["completed"] = len(results)
    batch["status"] = "completed"
    batch["completed_at"] = time.time()


async def _handle_retrieve_batch(request: Request) -> JSONResponse:
    """Handle GET /v1/batches/{batch_id} — retrieve batch status."""
    batch_id = request.path_params["batch_id"]
    batch = _batches.get(batch_id)
    if not batch:
        return JSONResponse(
            {"error": {"message": f"Batch {batch_id} not found", "type": "not_found"}},
            status_code=404,
        )

    return JSONResponse({
        "id": batch["id"],
        "object": "batch",
        "endpoint": batch["endpoint"],
        "status": batch["status"],
        "created_at": batch["created_at"],
        "in_progress_at": batch.get("in_progress_at"),
        "completed_at": batch.get("completed_at"),
        "request_counts": batch["request_counts"],
        "metadata": batch.get("metadata", {}),
        "results": batch.get("results", []),
        "errors": batch.get("errors", {"object": "list", "data": []}),
    })


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(config: ServerConfig | None = None) -> Starlette:
    """Create the dummy server ASGI application."""
    if config is not None:
        set_config(config)

    from starlette.middleware import Middleware
    from starlette.middleware.base import BaseHTTPMiddleware

    class AuthMiddleware(BaseHTTPMiddleware):
        """Optionally enforce Bearer token auth."""

        async def dispatch(self, request: Request, call_next):  # type: ignore[override]
            cfg = _config
            if cfg.require_api_key and request.url.path.startswith("/v1/"):
                auth = request.headers.get("authorization", "")
                expected = f"Bearer {cfg.require_api_key}"
                if auth != expected:
                    return JSONResponse(
                        {"error": {"message": "Invalid API key", "type": "auth_error"}},
                        status_code=401,
                    )
            return await call_next(request)

    import time as _time

    class RateLimitMiddleware(BaseHTTPMiddleware):
        """Reject requests above max_rps with 429 Too Many Requests."""

        def __init__(self, app, max_rps: float):  # type: ignore[no-untyped-def]
            super().__init__(app)
            self._max_rps = max_rps
            self._window: list[float] = []

        async def dispatch(self, request: Request, call_next):  # type: ignore[override]
            if not request.url.path.startswith("/v1/"):
                return await call_next(request)
            now = _time.monotonic()
            # Slide 1-second window
            self._window = [t for t in self._window if now - t < 1.0]
            if len(self._window) >= self._max_rps:
                return JSONResponse(
                    {
                        "error": {
                            "message": "Rate limit exceeded",
                            "type": "rate_limit_error",
                        }
                    },
                    status_code=429,
                )
            self._window.append(now)
            return await call_next(request)

    # Clear batch store on app creation
    _batches.clear()

    routes = [
        Route("/v1/completions", _handle_completions, methods=["POST"]),
        Route("/v1/chat/completions", _handle_chat_completions, methods=["POST"]),
        Route("/v1/embeddings", _handle_embeddings, methods=["POST"]),
        Route("/v1/batches", _handle_create_batch, methods=["POST"]),
        Route("/v1/batches/{batch_id}", _handle_retrieve_batch, methods=["GET"]),
        Route("/v1/models", _handle_models, methods=["GET"]),
        Route("/health", _handle_health, methods=["GET"]),
    ]

    middleware = [Middleware(AuthMiddleware)]
    if _config.max_rps is not None:
        middleware.append(Middleware(RateLimitMiddleware, max_rps=_config.max_rps))
    return Starlette(routes=routes, middleware=middleware)
