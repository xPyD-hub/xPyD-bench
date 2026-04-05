"""Benchmark runner — async HTTP client with Poisson scheduling."""

from __future__ import annotations

import asyncio
import gzip
import json
import random
import signal
import time
import uuid
from argparse import Namespace
from typing import Any

import httpx
import numpy as np

from xpyd_bench.bench.debug_log import DebugLogger
from xpyd_bench.bench.env import collect_env_info
from xpyd_bench.bench.models import BenchmarkResult, RequestResult

# ---------------------------------------------------------------------------
# HTTP client helpers
# ---------------------------------------------------------------------------


def _build_client_kwargs(
    args: Namespace, headers: dict[str, str] | None = None
) -> dict[str, Any]:
    """Build kwargs dict for ``httpx.AsyncClient`` from CLI/config args."""
    kwargs: dict[str, Any] = {}
    if headers:
        kwargs["headers"] = headers
    if getattr(args, "http2", False):
        try:
            import h2  # noqa: F401
        except ImportError:
            raise SystemExit(
                "ERROR: --http2 requires the 'h2' package. "
                "Install it with: pip install h2"
            )
        kwargs["http2"] = True
    max_connections = getattr(args, "max_connections", 100) or 100
    max_keepalive = getattr(args, "max_keepalive", 20) or 20
    kwargs["limits"] = httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive,
    )
    return kwargs

# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------


def _generate_random_prompts(num: int, input_len: int, seed: int) -> list[str]:
    """Generate random prompts of approximately *input_len* tokens."""
    rng = random.Random(seed)
    # ~4 chars per token is a rough estimate
    words = [
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dog",
        "a",
        "an",
        "is",
        "was",
        "hello",
        "world",
        "benchmark",
        "test",
        "data",
        "model",
        "request",
        "response",
        "token",
        "stream",
        "latency",
        "throughput",
    ]
    prompts: list[str] = []
    for _ in range(num):
        # Approx input_len tokens → input_len words
        prompt = " ".join(rng.choice(words) for _ in range(input_len))
        prompts.append(prompt)
    return prompts


# ---------------------------------------------------------------------------
# Request scheduling (Poisson / Gamma)
# ---------------------------------------------------------------------------


def _generate_intervals(
    num: int,
    rate: float,
    burstiness: float,
    seed: int,
) -> list[float]:
    """Return list of inter-arrival times in seconds.

    * rate=inf  → all zeros (fire at once)
    * burstiness=1.0 → Poisson process (exponential intervals)
    * burstiness!=1.0 → Gamma distribution
    """
    if rate == float("inf"):
        return [0.0] * num

    rng = np.random.RandomState(seed)
    mean_interval = 1.0 / rate

    if burstiness == 1.0:
        # Poisson process → exponential inter-arrivals
        intervals = rng.exponential(mean_interval, size=num).tolist()
    else:
        # Gamma distribution: shape=burstiness, scale=mean/burstiness
        shape = burstiness
        scale = mean_interval / burstiness
        intervals = rng.gamma(shape, scale, size=num).tolist()

    return intervals


# ---------------------------------------------------------------------------
# Single-request sender
# ---------------------------------------------------------------------------


# Status codes that are retryable
_RETRYABLE_STATUS_CODES = {429, 503}


def _is_retryable(exc: Exception) -> bool:
    """Check if an exception is retryable (connection error or retryable HTTP status)."""
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS_CODES
    return False


def _compressed_request_kwargs(
    payload: dict[str, Any],
    compress: bool,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build request kwargs, optionally gzip-compressing the body."""
    if not compress:
        kw: dict[str, Any] = {"json": payload}
        if extra_headers:
            kw["headers"] = extra_headers
        return kw
    raw = json.dumps(payload).encode()
    compressed = gzip.compress(raw)
    hdrs = {
        "Content-Encoding": "gzip",
        "Content-Type": "application/json",
    }
    if extra_headers:
        hdrs.update(extra_headers)
    return {
        "content": compressed,
        "headers": hdrs,
    }


def _generate_request_id(prefix: str | None = None) -> str:
    """Generate a unique request ID, optionally with a prefix."""
    uid = uuid.uuid4().hex
    if prefix:
        return f"{prefix}{uid}"
    return uid


async def _send_request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    is_streaming: bool,
    request_timeout: float = 300.0,
    retries: int = 0,
    retry_delay: float = 1.0,
    compress: bool = False,
    request_id: str | None = None,
    sse_metrics: bool = False,
    track_ratelimits: bool = False,
    track_payload_size: bool = False,
) -> RequestResult:
    """Send one request and collect metrics, with optional retry."""
    last_result = RequestResult()
    attempts = 0
    _extra_hdrs = {"X-Request-ID": request_id} if request_id else None

    # Pre-compute request payload size if tracking enabled (M67)
    _req_bytes: int | None = None
    if track_payload_size:
        from xpyd_bench.bench.payload_size import compute_payload_bytes
        _req_bytes = compute_payload_bytes(payload)

    for attempt in range(retries + 1):
        attempts = attempt
        result = RequestResult()
        result.request_id = request_id
        start = time.perf_counter()
        result.start_time = start

        try:
            if is_streaming:
                result = await _send_streaming(
                    client, url, payload, start, request_timeout=request_timeout,
                    compress=compress, extra_headers=_extra_hdrs,
                    sse_metrics=sse_metrics,
                    track_ratelimits=track_ratelimits,
                    track_payload_size=track_payload_size,
                )
                result.request_id = request_id
                # Payload size tracking for streaming (M67)
                if track_payload_size:
                    result.request_bytes = _req_bytes
            else:
                req_kw = _compressed_request_kwargs(payload, compress, extra_headers=_extra_hdrs)
                resp = await client.post(url, **req_kw, timeout=request_timeout)
                resp.raise_for_status()
                end = time.perf_counter()
                result.latency_ms = (end - start) * 1000.0

                # Rate-limit header tracking (M66)
                if track_ratelimits:
                    from xpyd_bench.bench.ratelimit import parse_ratelimit_headers
                    rl = parse_ratelimit_headers(resp.headers)
                    if rl:
                        result.ratelimit_headers = rl

                # Payload size tracking (M67)
                if track_payload_size:
                    result.request_bytes = _req_bytes
                    result.response_bytes = len(resp.content)

                body = resp.json()
                usage = body.get("usage", {})
                result.prompt_tokens = usage.get("prompt_tokens", 0)
                result.completion_tokens = usage.get("completion_tokens", 0)

                # Extract response text for validation (M47)
                choices = body.get("choices", [])
                if choices:
                    c = choices[0]
                    result.response_text = (
                        c.get("text")
                        or (c.get("message") or {}).get("content")
                        or ""
                    )
                    # Extract tool calls (M56)
                    msg = c.get("message") or {}
                    tc_list = msg.get("tool_calls", [])
                    result.tool_calls_found = len(tc_list)
                    result._response_body = body  # stash for structured output validation
                # Non-streaming: TPOT cannot be accurately measured because
                # latency includes prefill time. Leave as None for consistency
                # with streaming TPOT which correctly excludes TTFT/prefill.
                # result.tpot_ms remains None

        except Exception as exc:  # noqa: BLE001
            end = time.perf_counter()
            result.latency_ms = (end - start) * 1000.0
            result.success = False
            result.error = str(exc)

            # Retry if applicable
            if attempt < retries and _is_retryable(exc):
                delay = retry_delay * (2**attempt)
                await asyncio.sleep(delay)
                continue

        result.retries = attempt
        return result

    # Should not reach here, but just in case
    last_result.retries = attempts
    return last_result


async def _send_streaming(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    start: float,
    request_timeout: float = 300.0,
    compress: bool = False,
    extra_headers: dict[str, str] | None = None,
    sse_metrics: bool = False,
    track_ratelimits: bool = False,
    track_payload_size: bool = False,
) -> RequestResult:
    """Send a streaming request, measure TTFT / ITL."""
    result = RequestResult()
    result.start_time = start
    payload["stream"] = True
    first_token_time: float | None = None
    last_token_time: float = start
    token_count = 0
    collected_text_parts: list[str] = []  # M47: collect response text for validation
    stream_usage: dict[str, Any] | None = None
    chunk_timing_list: list[Any] = []  # M53: per-chunk timing data
    _stream_response_bytes = 0  # M67: track streamed response size

    try:
        req_kw = _compressed_request_kwargs(payload, compress, extra_headers=extra_headers)
        async with client.stream("POST", url, **req_kw, timeout=request_timeout) as resp:
            resp.raise_for_status()
            # Rate-limit header tracking (M66)
            if track_ratelimits:
                from xpyd_bench.bench.ratelimit import parse_ratelimit_headers
                rl = parse_ratelimit_headers(resp.headers)
                if rl:
                    result.ratelimit_headers = rl
            async for line in resp.aiter_lines():
                if track_payload_size:
                    _stream_response_bytes += len(line.encode("utf-8")) + 1  # +1 for newline
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

                # Capture usage from final streaming chunk (stream_options.include_usage)
                if "usage" in chunk and chunk["usage"] is not None:
                    stream_usage = chunk["usage"]

                # Check if this chunk has content
                choices = chunk.get("choices", [])
                if not choices:
                    continue

                # Iterate all choices to handle n>1 completions (M3)
                has_content = False
                for delta in choices:
                    # completions endpoint uses "text", chat uses "delta.content"
                    text = delta.get("text") or (delta.get("delta") or {}).get("content")
                    if text:
                        token_count += 1
                        has_content = True
                        collected_text_parts.append(text)
                if not has_content:
                    continue
                if first_token_time is None:
                    first_token_time = now
                    result.ttft_ms = (now - start) * 1000.0
                    if sse_metrics:
                        from xpyd_bench.bench.sse_metrics import ChunkTiming

                        chunk_timing_list.append(
                            ChunkTiming(timestamp=now - start, tokens=1, inter_token_ms=None)
                        )
                else:
                    itl = (now - last_token_time) * 1000.0
                    result.itl_ms.append(itl)
                    if sse_metrics:
                        from xpyd_bench.bench.sse_metrics import ChunkTiming

                        chunk_timing_list.append(
                            ChunkTiming(timestamp=now - start, tokens=1, inter_token_ms=itl)
                        )
                last_token_time = now

    except Exception as exc:  # noqa: BLE001
        result.success = False
        result.error = str(exc)

    end = time.perf_counter()
    result.latency_ms = (end - start) * 1000.0
    result.completion_tokens = token_count
    if stream_usage:
        result.prompt_tokens = stream_usage.get("prompt_tokens", 0)
        if stream_usage.get("completion_tokens"):
            result.completion_tokens = stream_usage["completion_tokens"]
            # Use server-reported token count for TPOT when available
            token_count = stream_usage["completion_tokens"]
    if token_count > 1 and first_token_time is not None:
        generation_time_ms = (last_token_time - first_token_time) * 1000.0
        result.tpot_ms = generation_time_ms / (token_count - 1)

    # M47: Store collected response text for validation
    result.response_text = "".join(collected_text_parts)

    # M53: Store chunk timings if SSE metrics enabled
    if sse_metrics and chunk_timing_list:
        result.chunk_timings = chunk_timing_list

    # M67: Store streamed response bytes
    if track_payload_size:
        result.response_bytes = _stream_response_bytes

    return result


# ---------------------------------------------------------------------------
# Build request payload
# ---------------------------------------------------------------------------


def _build_payload(
    args: Namespace,
    prompt: str,
    is_chat: bool,
    is_embeddings: bool = False,
) -> dict[str, Any]:
    """Build the JSON body for a single request."""
    if is_embeddings:
        payload: dict[str, Any] = {"input": prompt}
        if args.model:
            payload["model"] = args.model
        encoding_format = getattr(args, "encoding_format", None)
        if encoding_format is not None:
            payload["encoding_format"] = encoding_format
        return payload

    payload = {"max_tokens": args.output_len}

    if args.model:
        payload["model"] = args.model

    if is_chat:
        payload["messages"] = [{"role": "user", "content": prompt}]
    else:
        payload["prompt"] = prompt

    # Sampling parameters
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    if args.top_p is not None:
        payload["top_p"] = args.top_p
    if args.top_k is not None:
        payload["top_k"] = args.top_k
    if args.frequency_penalty is not None:
        payload["frequency_penalty"] = args.frequency_penalty
    if args.presence_penalty is not None:
        payload["presence_penalty"] = args.presence_penalty
    if args.best_of is not None:
        payload["best_of"] = args.best_of
    if args.use_beam_search:
        payload["use_beam_search"] = True
    if args.logprobs is not None:
        payload["logprobs"] = args.logprobs
    if args.ignore_eos:
        payload["ignore_eos"] = True
    if getattr(args, "stop", None) is not None:
        payload["stop"] = args.stop
    if getattr(args, "n", None) is not None:
        payload["n"] = args.n
    if getattr(args, "api_seed", None) is not None:
        payload["seed"] = args.api_seed
    if getattr(args, "echo", False):
        payload["echo"] = True
    if getattr(args, "suffix", None) is not None:
        payload["suffix"] = args.suffix
    if getattr(args, "logit_bias", None) is not None:
        import json as _json

        if isinstance(args.logit_bias, str):
            payload["logit_bias"] = _json.loads(args.logit_bias)
        else:
            payload["logit_bias"] = args.logit_bias
    if getattr(args, "user", None) is not None:
        payload["user"] = args.user
    if getattr(args, "stream_options_include_usage", False):
        payload["stream_options"] = {"include_usage": True}

    # Chat-specific parameters (only for chat endpoints)
    if is_chat:
        if getattr(args, "response_format", None) is not None:
            import json as _json

            if isinstance(args.response_format, str):
                payload["response_format"] = _json.loads(args.response_format)
            else:
                payload["response_format"] = args.response_format
        if getattr(args, "tools", None) is not None:
            import json as _json
            from pathlib import Path

            tools_path = Path(args.tools)
            if tools_path.is_file():
                with open(tools_path) as f:
                    payload["tools"] = _json.load(f)
            else:
                payload["tools"] = _json.loads(args.tools)
        if getattr(args, "tool_choice", None) is not None:
            import json as _json

            try:
                payload["tool_choice"] = _json.loads(args.tool_choice)
            except (ValueError, TypeError):
                payload["tool_choice"] = args.tool_choice
        if getattr(args, "parallel_tool_calls", None) is not None:
            payload["parallel_tool_calls"] = args.parallel_tool_calls
        if getattr(args, "top_logprobs", None) is not None:
            payload["top_logprobs"] = args.top_logprobs
        if getattr(args, "max_completion_tokens", None) is not None:
            payload["max_completion_tokens"] = args.max_completion_tokens
        if getattr(args, "service_tier", None) is not None:
            payload["service_tier"] = args.service_tier

    return payload


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def _compute_metrics(result: BenchmarkResult) -> None:
    """Compute aggregate metrics from per-request results."""
    successful = [r for r in result.requests if r.success]
    result.completed = len(successful)
    result.failed = len(result.requests) - result.completed

    if not successful:
        return

    result.total_input_tokens = sum(r.prompt_tokens for r in successful)
    result.total_output_tokens = sum(r.completion_tokens for r in successful)

    if result.total_duration_s > 0:
        result.request_throughput = result.completed / result.total_duration_s
        result.output_throughput = result.total_output_tokens / result.total_duration_s
        result.total_token_throughput = (
            result.total_input_tokens + result.total_output_tokens
        ) / result.total_duration_s

    def _set_percentiles(
        values: list[float], prefix: str, target: BenchmarkResult
    ) -> None:
        """Compute and set mean/median/p50/p90/p95/p99 on *target*."""
        if not values:
            return
        setattr(target, f"mean_{prefix}", float(np.mean(values)))
        setattr(target, f"median_{prefix}", float(np.median(values)))
        setattr(target, f"p50_{prefix}", float(np.percentile(values, 50)))
        setattr(target, f"p90_{prefix}", float(np.percentile(values, 90)))
        setattr(target, f"p95_{prefix}", float(np.percentile(values, 95)))
        setattr(target, f"p99_{prefix}", float(np.percentile(values, 99)))

    # E2E latency
    e2els = [r.latency_ms for r in successful]
    _set_percentiles(e2els, "e2el_ms", result)

    # TTFT
    ttfts = [r.ttft_ms for r in successful if r.ttft_ms is not None]
    _set_percentiles(ttfts, "ttft_ms", result)

    # TPOT
    tpots = [r.tpot_ms for r in successful if r.tpot_ms is not None]
    _set_percentiles(tpots, "tpot_ms", result)

    # ITL
    all_itls = [itl for r in successful for itl in r.itl_ms]
    _set_percentiles(all_itls, "itl_ms", result)


async def run_benchmark(args: Namespace, base_url: str) -> tuple[dict, BenchmarkResult]:
    """Execute the benchmark and return (result_dict, BenchmarkResult)."""
    # Resolve backend plugin (M31)
    backend_name = getattr(args, "backend", "openai")
    use_plugin = False
    plugin = None

    # "openai" and "openai-chat" use the legacy code path for full backward compat
    if backend_name not in ("openai", "openai-chat"):
        from xpyd_bench.plugins import registry

        plugin = registry.get(backend_name)
        use_plugin = True

    is_chat = "chat" in args.endpoint
    is_embeddings = "embeddings" in args.endpoint
    # Use explicit --stream/--no-stream if provided; default to endpoint type for
    # backward compatibility (chat=streaming, completions=non-streaming).
    # Embeddings are always non-streaming.
    stream_flag = getattr(args, "stream", None)
    if is_embeddings:
        is_streaming = False
    else:
        is_streaming = stream_flag if stream_flag is not None else is_chat
    url = plugin.build_url(base_url, args) if use_plugin else f"{base_url}{args.endpoint}"

    # Build default headers (authentication + custom)
    headers: dict[str, str] = {}
    # Custom headers first (lower priority than auth unless explicitly overridden)
    custom_headers = getattr(args, "custom_headers", None) or {}
    headers.update(custom_headers)
    # Auth header (can be overridden by custom headers if user explicitly sets Authorization)
    api_key = getattr(args, "api_key", None)
    if api_key and "Authorization" not in custom_headers:
        headers["Authorization"] = f"Bearer {api_key}"

    # Generate or load prompts
    dataset_path = getattr(args, "dataset_path", None)
    if dataset_path:
        from xpyd_bench.datasets.loader import load_dataset, validate_and_report

        entries = load_dataset(
            path=dataset_path,
            num_prompts=args.num_prompts,
            input_len=args.input_len,
            output_len=args.output_len,
            input_len_dist=getattr(args, "synthetic_input_len_dist", "fixed"),
            output_len_dist=getattr(args, "synthetic_output_len_dist", "fixed"),
            seed=args.seed,
            tokenizer=getattr(args, "tokenizer", None),
        )
        validate_and_report(entries, dataset_path, tokenizer=getattr(args, "tokenizer", None))
        prompts = [e.prompt for e in entries]
        prompt_priorities = [e.priority for e in entries]
        # Override num_prompts to match actual dataset size
        args.num_prompts = len(prompts)
    else:
        dataset_name = getattr(args, "dataset_name", "random")
        input_dist = getattr(args, "synthetic_input_len_dist", "fixed")
        output_dist = getattr(args, "synthetic_output_len_dist", "fixed")
        if (
            dataset_name == "synthetic"
            or input_dist != "fixed"
            or output_dist != "fixed"
        ):
            from xpyd_bench.datasets.loader import load_dataset, validate_and_report

            entries = load_dataset(
                path=None,
                name=dataset_name,
                num_prompts=args.num_prompts,
                input_len=args.input_len,
                output_len=args.output_len,
                input_len_dist=input_dist,
                output_len_dist=output_dist,
                seed=args.seed,
                tokenizer=getattr(args, "tokenizer", None),
            )
            tokenizer_arg = getattr(args, "tokenizer", None)
            validate_and_report(
                entries, f"synthetic ({dataset_name})", tokenizer=tokenizer_arg
            )
            prompts = [e.prompt for e in entries]
            prompt_priorities = [e.priority for e in entries]
            args.num_prompts = len(prompts)
        else:
            prompts = _generate_random_prompts(args.num_prompts, args.input_len, args.seed)
            prompt_priorities = [None] * args.num_prompts

    # Apply template variable substitution (M37)
    template_vars_path = getattr(args, "template_vars", None)
    if template_vars_path:
        from xpyd_bench.templating import apply_templates, load_template_vars

        tvars = load_template_vars(template_vars_path)
        prompts = apply_templates(prompts, tvars)

    # Generate inter-arrival intervals
    rate_algorithm = getattr(args, "rate_algorithm", "default")
    rate_pattern = getattr(args, "rate_pattern", None)
    token_bucket = None

    if rate_algorithm == "token-bucket" and args.request_rate != float("inf"):
        from xpyd_bench.bench.token_bucket import TokenBucket

        burst = getattr(args, "token_bucket_burst", None) or args.request_rate
        token_bucket = TokenBucket(rate=args.request_rate, burst=burst)
        intervals = [0.0] * args.num_prompts  # token bucket handles pacing
    elif rate_pattern and isinstance(rate_pattern, dict):
        from xpyd_bench.bench.rate_patterns import generate_pattern_intervals

        intervals = generate_pattern_intervals(args.num_prompts, rate_pattern, args.seed)
    else:
        intervals = _generate_intervals(
            args.num_prompts, args.request_rate, args.burstiness, args.seed
        )

    # Adaptive concurrency limiter (M16)
    adaptive_limiter = None
    use_adaptive = getattr(args, "adaptive_concurrency", False)
    if use_adaptive:
        from xpyd_bench.bench.token_bucket import AdaptiveConcurrencyLimiter

        adaptive_limiter = AdaptiveConcurrencyLimiter(
            initial=getattr(args, "adaptive_initial_concurrency", 16),
            min_concurrency=getattr(args, "adaptive_min_concurrency", 1),
            max_concurrency=getattr(args, "adaptive_max_concurrency", 256),
            target_latency_ms=getattr(args, "adaptive_target_latency", 500.0),
        )

    # Concurrency limiter
    semaphore = asyncio.Semaphore(args.max_concurrency) if args.max_concurrency else None

    # Duration mode (M46)
    duration_limit = getattr(args, "duration", None)

    result = BenchmarkResult(
        backend=args.backend,
        base_url=base_url,
        endpoint=args.endpoint,
        model=args.model or "",
        num_prompts=args.num_prompts,
        request_rate=args.request_rate,
        max_concurrency=args.max_concurrency,
        input_len=args.input_len,
        output_len=args.output_len,
        duration_limit=duration_limit,
        environment=collect_env_info(),
    )

    # Timeout & retry settings
    request_timeout = getattr(args, "timeout", 300.0) or 300.0
    req_retries = getattr(args, "retries", 0) or 0
    req_retry_delay = getattr(args, "retry_delay", 1.0) or 1.0
    req_compress = getattr(args, "compress", False) or False

    def _mk_payload(prompt: str) -> dict[str, Any]:
        if use_plugin:
            return plugin.build_payload(
                args, prompt, is_chat=is_chat, is_embeddings=is_embeddings
            )
        return _build_payload(args, prompt, is_chat, is_embeddings)

    # Request ID prefix (M42)
    req_id_prefix = getattr(args, "request_id_prefix", None) or ""

    def _make_request_id() -> str:
        uid = uuid.uuid4().hex[:12]
        return f"{req_id_prefix}{uid}" if req_id_prefix else uid

    # SSE metrics flag (M53)
    sse_metrics_enabled = bool(getattr(args, "sse_metrics", False))
    sse_stall_threshold = float(getattr(args, "sse_stall_threshold_ms", 1000.0) or 1000.0)
    track_ratelimits_enabled = bool(getattr(args, "track_ratelimits", False))
    track_payload_size_enabled = bool(getattr(args, "track_payload_size", False))
    measure_generation_speed = bool(getattr(args, "measure_generation_speed", False))

    async def _do_send(
        client: httpx.AsyncClient, payload: dict[str, Any]
    ) -> RequestResult:
        rid = _make_request_id()
        if use_plugin:
            r = await plugin.send_request(
                client, url, payload,
                is_streaming=is_streaming,
                request_timeout=request_timeout,
                retries=req_retries,
                retry_delay=req_retry_delay,
            )
            r.request_id = rid
            return r
        return await _send_request(
            client, url, payload, is_streaming,
            request_timeout=request_timeout,
            retries=req_retries,
            retry_delay=req_retry_delay,
            compress=req_compress,
            request_id=rid,
            sse_metrics=sse_metrics_enabled,
            track_ratelimits=track_ratelimits_enabled,
            track_payload_size=track_payload_size_enabled,
        )

    # Noise injection (M60)
    from xpyd_bench.noise import NoiseInjector, build_noise_config_from_args

    noise_config = build_noise_config_from_args(args)
    noise_injector = NoiseInjector(noise_config) if noise_config.enabled else None

    async def _task(client: httpx.AsyncClient, prompt: str) -> RequestResult:
        payload = _mk_payload(prompt)

        # Noise injection: delay
        if noise_injector:
            await noise_injector.maybe_delay()

        # Noise injection: client-side error abort
        if noise_injector and noise_injector.should_inject_error():
            noise_injector.stats.errors_injected += 1
            noise_injector.stats.total_requests += 1
            r = RequestResult()
            r.start_time = time.perf_counter()
            r.success = False
            r.error = "noise-injection: client-side abort"
            r.latency_ms = 0.0
            return r

        # Noise injection: payload corruption
        if noise_injector:
            payload, _corrupted = noise_injector.corrupt_payload(payload)
            noise_injector.stats.total_requests += 1

        if adaptive_limiter:
            await adaptive_limiter.acquire()
            try:
                r = await _do_send(client, payload)
            finally:
                adaptive_limiter.release()
            await adaptive_limiter.record_latency(r.latency_ms)
            return r
        if semaphore:
            async with semaphore:
                return await _do_send(client, payload)
        return await _do_send(client, payload)

    # Live dashboard (M29) — use LiveDashboard when not explicitly disabled
    no_live = getattr(args, "no_live", False)
    use_live = not args.disable_tqdm and not no_live
    reporter = None
    _LiveDashboardType: type = type(None)  # sentinel for isinstance check
    if use_live:
        from xpyd_bench.reporting.rich_output import LiveDashboard

        _LiveDashboardType = LiveDashboard
        reporter = LiveDashboard(total=args.num_prompts if not duration_limit else 0)
    elif not args.disable_tqdm and not no_live:
        use_rich = getattr(args, "rich_progress", False)
        if use_rich:
            from xpyd_bench.reporting.rich_output import RichProgressReporter

            reporter = RichProgressReporter(total=args.num_prompts if not duration_limit else 0)

    # --- Warmup phase ---
    warmup_count = getattr(args, "warmup", 0) or 0
    warmup_profile_enabled = getattr(args, "warmup_profile", False)
    warmup_profile_result = None
    if warmup_count > 0:
        if not args.disable_tqdm:
            print(f"Warmup: sending {warmup_count} request(s)...")
        warmup_prompts = prompts[:warmup_count] if len(prompts) >= warmup_count else (
            prompts * ((warmup_count // len(prompts)) + 1)
        )[:warmup_count]
        warmup_kw = _build_client_kwargs(args, headers=headers)
        warmup_latencies: list[float] = []
        warmup_start_time = time.perf_counter()
        async with httpx.AsyncClient(**warmup_kw) as warmup_client:
            for wi, wp in enumerate(warmup_prompts):
                payload = _build_payload(args, wp, is_chat, is_embeddings)
                wr = await _send_request(warmup_client, url, payload, is_streaming)
                warmup_latencies.append(wr.latency_ms)  # already in ms
                if not args.disable_tqdm:
                    status = "ok" if wr.success else f"FAIL: {wr.error}"
                    print(f"  Warmup {wi + 1}/{warmup_count}: {status}")
        warmup_total_s = time.perf_counter() - warmup_start_time
        if warmup_profile_enabled and warmup_latencies:
            from xpyd_bench.bench.warmup_profile import (
                build_warmup_profile,
                print_warmup_profile,
            )
            warmup_profile_result = build_warmup_profile(
                warmup_latencies, warmup_total_s,
            )
            if not args.disable_tqdm:
                print_warmup_profile(warmup_profile_result)
        if not args.disable_tqdm:
            print("Warmup complete. Starting benchmark...")

    # Debug logger (M22)
    debug_log_path = getattr(args, "debug_log", None)
    debug_logger: DebugLogger | None = None
    if debug_log_path:
        debug_logger = DebugLogger(debug_log_path)
        debug_logger.log_connection_config(
            http2=getattr(args, "http2", False),
            max_connections=getattr(args, "max_connections", 100),
            max_keepalive=getattr(args, "max_keepalive", 20),
        )

    if reporter:
        reporter.start()

    tasks: list[asyncio.Task] = []
    overall_start = time.perf_counter()
    result.bench_start_time = overall_start
    shutdown_requested = False

    # Metrics WebSocket server (M33)
    metrics_ws_server = None
    metrics_collector = None
    ws_port = getattr(args, "metrics_ws_port", None)
    if ws_port:
        from xpyd_bench.metrics_ws import MetricsCollector, MetricsWebSocketServer

        metrics_collector = MetricsCollector()
        metrics_collector.start()
        metrics_ws_server = MetricsWebSocketServer(metrics_collector, port=ws_port)
        await metrics_ws_server.start()
        if not args.disable_tqdm:
            print(f"Metrics WebSocket server started on ws://0.0.0.0:{ws_port}/metrics")

    async def _tracked_task(
        client: httpx.AsyncClient, prompt: str, priority: int | None = None,
    ) -> RequestResult:
        payload = _mk_payload(prompt)
        r = await _task(client, prompt)
        if debug_logger:
            p_bytes: int | None = None
            c_bytes: int | None = None
            if req_compress:
                import gzip as _gzip

                raw = json.dumps(payload).encode()
                p_bytes = len(raw)
                c_bytes = len(_gzip.compress(raw))
            debug_logger.log(url, payload, r, payload_bytes=p_bytes, compressed_bytes=c_bytes)
        if reporter:
            latency = r.latency * 1000 if r.latency is not None else None
            if isinstance(reporter, _LiveDashboardType):
                reporter.advance(success=r.success, latency_ms=latency)
            else:
                reporter.advance(success=r.success)
        if metrics_collector:
            metrics_collector.record(
                success=r.success,
                latency_ms=r.latency_ms,
                ttft_ms=r.ttft_ms,
                prompt_tokens=r.prompt_tokens,
                completion_tokens=r.completion_tokens,
            )
        r.priority = priority
        return r

    grace_period = getattr(args, "shutdown_grace_period", 5.0) or 5.0

    async with httpx.AsyncClient(**_build_client_kwargs(args, headers=headers)) as client:
        # Install signal handler for graceful shutdown
        loop = asyncio.get_running_loop()

        def _signal_handler() -> None:
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                print("\n⚠️  SIGINT received — graceful shutdown initiated. "
                      "Waiting for in-flight requests...")

        try:
            loop.add_signal_handler(signal.SIGINT, _signal_handler)
        except NotImplementedError:
            pass  # Windows doesn't support add_signal_handler

        # Duration-based or count-based request dispatch
        prompt_count = len(prompts)
        i = 0
        deadline = (overall_start + duration_limit) if duration_limit else None

        # Priority scheduling (M52) — sort prompts by priority before dispatch
        priority_levels = getattr(args, "priority_levels", 0) or 0
        if priority_levels > 0 and not duration_limit:
            # Build (priority, original_index) pairs and sort by priority (0=highest)
            indexed = list(range(len(prompts)))
            pri_list = [
                (
                    prompt_priorities[j]
                    if prompt_priorities[j] is not None
                    else priority_levels - 1,
                    j,
                )
                for j in indexed
            ]
            pri_list.sort(key=lambda x: x[0])
            sorted_prompts = [prompts[j] for _, j in pri_list]
            sorted_priorities = [p for p, _ in pri_list]
            prompts = sorted_prompts
            prompt_priorities = sorted_priorities

        while True:
            if shutdown_requested:
                break
            # Check duration limit
            if deadline and time.perf_counter() >= deadline:
                break
            # Check num_prompts limit (unless in pure duration mode)
            if not duration_limit and i >= len(prompts):
                break
            if duration_limit and args.num_prompts and i >= args.num_prompts:
                break

            prompt = prompts[i % prompt_count]
            prompt_pri = prompt_priorities[i % prompt_count] if prompt_priorities else None

            if i > 0:
                interval_idx = i if i < len(intervals) else i % len(intervals)
                if token_bucket:
                    await token_bucket.acquire()
                else:
                    await asyncio.sleep(intervals[interval_idx])
                if shutdown_requested:
                    break
                if deadline and time.perf_counter() >= deadline:
                    break
            task = asyncio.create_task(_tracked_task(client, prompt, priority=prompt_pri))
            tasks.append(task)

            if not reporter and not args.disable_tqdm and (i + 1) % 100 == 0:
                total_label = args.num_prompts if not duration_limit else "∞"
                print(f"  Launched {i + 1}/{total_label} requests...")

            i += 1

        # Wait for all launched tasks to complete (with grace period if shutting down)
        if shutdown_requested and tasks:
            done, pending = await asyncio.wait(
                tasks, timeout=grace_period, return_when=asyncio.ALL_COMPLETED
            )
            for t in pending:
                t.cancel()
            # Suppress CancelledError from cancelled tasks
            await asyncio.gather(*pending, return_exceptions=True)
        elif tasks:
            await asyncio.gather(*tasks)

        try:
            loop.remove_signal_handler(signal.SIGINT)
        except NotImplementedError:
            pass

    overall_end = time.perf_counter()
    if reporter:
        reporter.stop()

    # Collect results from completed (non-cancelled) tasks
    results_list = []
    for t in tasks:
        if t.done() and not t.cancelled():
            try:
                results_list.append(t.result())
            except Exception:  # noqa: BLE001
                pass

    result.total_duration_s = overall_end - overall_start
    result.requests = results_list
    result.partial = shutdown_requested

    # Update num_prompts to actual count in duration mode
    if duration_limit:
        result.num_prompts = len(results_list)

    if debug_logger:
        debug_logger.close()

    if metrics_ws_server:
        await metrics_ws_server.stop()

    _compute_metrics(result)

    # Network latency decomposition (M57)
    latency_breakdown_enabled = getattr(args, "latency_breakdown", False)
    if latency_breakdown_enabled and result.requests:
        from xpyd_bench.bench.latency_breakdown import (
            LatencyBreakdown,
            compute_breakdown_summary,
            estimate_server_processing,
            measure_connection_phases,
            parse_url,
        )

        lb_host, lb_port, lb_tls = parse_url(base_url)
        # Probe connection phases once (representative of first-connection cost)
        try:
            conn_probe = await measure_connection_phases(lb_host, lb_port, lb_tls)
        except Exception:  # noqa: BLE001
            conn_probe = LatencyBreakdown()

        breakdowns: list[LatencyBreakdown] = []
        first_done = False
        for req in result.requests:
            if not req.success:
                continue
            if not first_done:
                # First request: attribute connection setup phases
                bd = LatencyBreakdown(
                    dns_ms=conn_probe.dns_ms,
                    connect_ms=conn_probe.connect_ms,
                    tls_ms=conn_probe.tls_ms,
                    server_ms=estimate_server_processing(
                        req.latency_ms,
                        conn_probe.dns_ms,
                        conn_probe.connect_ms,
                        conn_probe.tls_ms,
                    ),
                )
                first_done = True
            else:
                # Keep-alive requests: no connection overhead
                bd = LatencyBreakdown(
                    dns_ms=0.0,
                    connect_ms=0.0,
                    tls_ms=0.0,
                    server_ms=req.latency_ms,
                )
            req.latency_breakdown = bd.to_dict()
            breakdowns.append(bd)

        if breakdowns:
            result.latency_breakdown = compute_breakdown_summary(breakdowns)

    # Anomaly detection (M43)
    anomaly_threshold = getattr(args, "anomaly_threshold", 1.5)
    if anomaly_threshold and anomaly_threshold > 0:
        from xpyd_bench.bench.anomaly import detect_anomalies

        latencies = [r.latency_ms for r in result.requests if r.success]
        anomaly_result = detect_anomalies(latencies, multiplier=anomaly_threshold)
        if anomaly_result is not None and anomaly_result.count > 0:
            result.anomalies = anomaly_result.to_dict()

    # Priority metrics breakdown (M52)
    if priority_levels > 0 and result.requests:
        from xpyd_bench.bench.priority import compute_priority_metrics

        result.priority_metrics = compute_priority_metrics(
            result.requests, priority_levels,
        )

    # SSE metrics aggregate (M53)
    if sse_metrics_enabled and result.requests:
        from xpyd_bench.bench.sse_metrics import analyze_chunk_timings, compute_sse_aggregate

        per_request_sse = []
        for req in result.requests:
            if req.chunk_timings:
                per_request_sse.append(
                    analyze_chunk_timings(req.chunk_timings, stall_threshold_ms=sse_stall_threshold)
                )
        if per_request_sse:
            result.sse_metrics = compute_sse_aggregate(per_request_sse)

    # Rate-limit header aggregation (M66)
    if track_ratelimits_enabled and result.requests:
        from xpyd_bench.bench.ratelimit import aggregate_ratelimit

        rl_headers = [r.ratelimit_headers for r in result.requests]
        error_msgs = [r.error for r in result.requests]
        rl_summary = aggregate_ratelimit(rl_headers, error_msgs)
        result.ratelimit_summary = rl_summary.to_dict()

    # Payload size aggregation (M67)
    if track_payload_size_enabled and result.requests:
        from xpyd_bench.bench.payload_size import aggregate_payload_sizes

        req_bytes = [r.request_bytes for r in result.requests]
        resp_bytes = [r.response_bytes for r in result.requests]
        ps_summary = aggregate_payload_sizes(req_bytes, resp_bytes)
        result.payload_summary = ps_summary.to_dict()

    # Generation speed calculation and aggregation (M68)
    if measure_generation_speed and result.requests:
        from xpyd_bench.bench.generation_speed import (
            aggregate_generation_speeds,
            compute_generation_tps,
        )

        for req in result.requests:
            if req.success:
                req.generation_tps = compute_generation_tps(
                    req.completion_tokens, req.ttft_ms, req.latency_ms
                )
        tps_vals = [r.generation_tps for r in result.requests]
        gs_summary = aggregate_generation_speeds(tps_vals)
        result.generation_speed_summary = gs_summary.to_dict()

    # Response validation (M47)
    validators_specs = getattr(args, "validate_response", None) or []
    if validators_specs:
        from xpyd_bench.bench.validation import ValidationResult as VResult
        from xpyd_bench.bench.validation import (
            aggregate_validations,
            parse_validators,
            validate_response,
        )

        validators = parse_validators(validators_specs)
        v_results: list[VResult] = []
        for req in result.requests:
            if req.success and req.response_text is not None:
                vr = validate_response(req.response_text, validators)
                req.validation_errors = vr.errors
                v_results.append(vr)
        if v_results:
            summary = aggregate_validations(v_results)
            result.validation_summary = {
                "total": summary.total,
                "passed": summary.passed,
                "failed": summary.failed,
                "pass_rate": round(summary.pass_rate, 2),
                "error_counts": summary.error_counts,
            }

    if shutdown_requested:
        print(
            f"\n⚠️  Partial results: {result.completed} completed, "
            f"{len(prompts) - len(tasks)} not launched, "
            f"{len(tasks) - len(results_list)} cancelled"
        )

    # Attach warmup profile to result (M51)
    if warmup_profile_result is not None:
        result.warmup_profile = warmup_profile_result.to_dict()

    # Attach noise injection stats (M60)
    if noise_injector:
        result.noise_injection = {
            "config": noise_injector.config.to_dict(),
            "stats": noise_injector.stats.to_dict(),
        }

    # Structured output validation (M56)
    _tools_path = getattr(args, "tools", None)
    _response_format = getattr(args, "response_format", None)
    if _tools_path or _response_format:
        from xpyd_bench.bench.structured_output import (
            StructuredOutputResult,
            aggregate_structured_output,
            validate_json_response,
            validate_tool_calls,
        )

        # Parse tools once
        _tools_list = None
        if _tools_path:
            import json as _json
            from pathlib import Path as _Path

            tp = _Path(_tools_path)
            if tp.is_file():
                with open(tp) as f:
                    _tools_list = _json.load(f)
            else:
                _tools_list = _json.loads(_tools_path)

        # Parse response_format once
        _rf_dict = None
        if _response_format:
            import json as _json

            if isinstance(_response_format, str):
                _rf_dict = _json.loads(_response_format)
            else:
                _rf_dict = _response_format

        so_results: list[StructuredOutputResult] = []
        for req in result.requests:
            if not req.success:
                continue
            resp_body = getattr(req, "_response_body", None)
            if _tools_list and resp_body:
                so_r = validate_tool_calls(resp_body, _tools_list)
                req.tool_call_success = so_r.success
                req.tool_calls_found = so_r.tool_calls_found
                so_results.append(so_r)
            elif _rf_dict:
                so_r = validate_json_response(req.response_text, _rf_dict)
                req.schema_valid = so_r.json_schema_valid
                so_results.append(so_r)
        if so_results:
            so_summary = aggregate_structured_output(so_results)
            result.structured_output_metrics = so_summary.to_dict()

    if reporter:
        reporter.print_summary_table(result)
    else:
        _print_summary(result)

    return _to_dict(result), result


def _print_summary(r: BenchmarkResult) -> None:
    """Print human-readable benchmark summary (plain text fallback)."""
    print("=" * 60)
    print("Benchmark Results")
    print("=" * 60)
    print(f"  Completed:             {r.completed}")
    print(f"  Failed:                {r.failed}")
    print(f"  Total duration:        {r.total_duration_s:.2f} s")
    print(f"  Request throughput:    {r.request_throughput:.2f} req/s")
    print(f"  Output throughput:     {r.output_throughput:.2f} tok/s")
    print(f"  Total tok throughput:  {r.total_token_throughput:.2f} tok/s")
    print()
    for label, prefix in [("TTFT", "ttft"), ("TPOT", "tpot"), ("ITL", "itl"), ("E2EL", "e2el")]:
        mean = getattr(r, f"mean_{prefix}_ms")
        p50 = getattr(r, f"p50_{prefix}_ms")
        p90 = getattr(r, f"p90_{prefix}_ms")
        p95 = getattr(r, f"p95_{prefix}_ms")
        p99 = getattr(r, f"p99_{prefix}_ms")
        if mean is None:
            print(f"  {label:5s}  N/A (not measured)")
        else:
            print(
                f"  {label:5s}  mean={mean:8.2f}  P50={p50:8.2f}  "
                f"P90={p90:8.2f}  P95={p95:8.2f}  P99={p99:8.2f} ms"
            )
    if r.anomalies:
        count = r.anomalies["count"]
        threshold = r.anomalies["threshold_ms"]
        print(f"\n  ⚠️  Anomalies: {count} request(s) exceeded {threshold:.1f} ms threshold")
        for a in r.anomalies["flagged_requests"][:5]:
            idx = a["index"]
            lat = a["latency_ms"]
            dev = a["deviation_factor"]
            print(f"      Request #{idx}: {lat:.2f} ms ({dev:.1f}x IQR above Q3)")
    if r.validation_summary:
        vs = r.validation_summary
        print(
            f"\n  🔍 Validation: {vs['passed']}/{vs['total']} passed "
            f"({vs['pass_rate']}%)"
        )
        if vs["failed"] > 0:
            for err_type, cnt in vs["error_counts"].items():
                print(f"      {err_type}: {cnt}")
    if r.structured_output_metrics:
        so = r.structured_output_metrics
        if so.get("tool_call_requests", 0) > 0:
            print(
                f"\n  🔧 Tool Calls: {so['tool_call_successes']}/{so['tool_call_requests']} "
                f"successful ({so['tool_call_success_rate']}%)"
            )
            print(f"      Total tool calls extracted: {so['total_tool_calls_extracted']}")
        if so.get("schema_validations", 0) > 0:
            print(
                f"\n  📋 Schema Conformance: {so['schema_passes']}/{so['schema_validations']} "
                f"passed ({so['schema_conformance_rate']}%)"
            )
    if r.latency_breakdown:
        from xpyd_bench.bench.latency_breakdown import print_breakdown_summary

        print_breakdown_summary(r.latency_breakdown)
    if r.ratelimit_summary:
        rl = r.ratelimit_summary
        parts: list[str] = []
        if rl.get("max_limit") is not None:
            parts.append(f"limit={rl['max_limit']}")
        if rl.get("min_remaining") is not None:
            parts.append(f"min_remaining={rl['min_remaining']}")
        if rl.get("min_remaining_tokens") is not None:
            parts.append(f"min_remaining_tokens={rl['min_remaining_tokens']}")
        if rl.get("throttle_count", 0) > 0:
            parts.append(f"throttled={rl['throttle_count']}")
        tracked = rl.get("tracked_responses", 0)
        total = rl.get("total_responses", 0)
        parts.append(f"tracked={tracked}/{total}")
        print(f"\n  🚦 Rate-Limits: {', '.join(parts)}")
    print("=" * 60)


async def replay_trace(
    trace: Any,
    delays: list[float],
    base_url: str,
    model: str = "",
    api_key: str | None = None,
    timeout: float = 300.0,
    http2: bool = False,
    max_connections: int = 100,
    max_keepalive: int = 20,
) -> BenchmarkResult:
    """Replay a recorded trace against a target server.

    Sends requests with the same inter-request timing as the original trace.

    Args:
        trace: TraceData object with recorded entries.
        delays: Pre-computed inter-request delays in seconds.
        base_url: Target server base URL.
        model: Model name override.
        api_key: Optional API key for authentication.
        timeout: Per-request timeout in seconds.

    Returns:
        BenchmarkResult with metrics from the replay run.
    """
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    result = BenchmarkResult(
        backend="openai",
        base_url=base_url,
        model=model,
        num_prompts=len(trace.entries),
        environment=collect_env_info(),
    )

    replay_kwargs: dict[str, Any] = {
        "headers": headers,
        "limits": httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive,
        ),
    }
    if http2:
        replay_kwargs["http2"] = True
    async with httpx.AsyncClient(**replay_kwargs) as client:
        start_time = time.perf_counter()

        for i, entry in enumerate(trace.entries):
            if i < len(delays) and delays[i] > 0:
                await asyncio.sleep(delays[i])

            url = base_url.rstrip("/") + entry.endpoint
            is_chat = "chat" in entry.endpoint

            if is_chat:
                payload: dict[str, Any] = {
                    "model": model or entry.model,
                    "messages": [{"role": "user", "content": entry.prompt or "hello"}],
                    "max_tokens": entry.max_tokens,
                    "temperature": entry.temperature,
                    "stream": entry.stream,
                }
            else:
                payload = {
                    "model": model or entry.model,
                    "prompt": entry.prompt or "hello",
                    "max_tokens": entry.max_tokens,
                    "temperature": entry.temperature,
                    "stream": entry.stream,
                }

            req_result = await _send_request(
                client,
                url,
                payload,
                is_streaming=entry.stream,
                request_timeout=timeout,
            )
            result.requests.append(req_result)

        end_time = time.perf_counter()
        result.total_duration_s = end_time - start_time

    _compute_metrics(result)
    return result


def _to_dict(r: BenchmarkResult) -> dict:
    """Convert BenchmarkResult to a JSON-serializable dict."""
    d: dict[str, Any] = {
        "backend": r.backend,
        "base_url": r.base_url,
        "endpoint": r.endpoint,
        "model": r.model,
        "num_prompts": r.num_prompts,
        "request_rate": r.request_rate,
        "max_concurrency": r.max_concurrency,
        "input_len": r.input_len,
        "output_len": r.output_len,
        "total_duration_s": r.total_duration_s,
        "completed": r.completed,
        "failed": r.failed,
        "total_input_tokens": r.total_input_tokens,
        "total_output_tokens": r.total_output_tokens,
        "request_throughput": r.request_throughput,
        "output_throughput": r.output_throughput,
        "total_token_throughput": r.total_token_throughput,
    }
    for prefix in ("ttft", "tpot", "itl", "e2el"):
        for stat in ("mean", "median", "p50", "p90", "p95", "p99"):
            key = f"{stat}_{prefix}_ms"
            val = getattr(r, key)
            d[key] = val  # None serializes as JSON null
    if r.environment:
        d["environment"] = r.environment
    if r.partial:
        d["partial"] = True
    if r.tags:
        d["tags"] = r.tags
    if r.anomalies:
        d["anomalies"] = r.anomalies
    if r.ratelimit_summary:
        d["ratelimit_summary"] = r.ratelimit_summary
    if r.warmup_profile:
        d["warmup_profile"] = r.warmup_profile
    if r.priority_metrics:
        d["priority_metrics"] = r.priority_metrics
    if r.sse_metrics:
        d["sse_metrics"] = r.sse_metrics
    if r.structured_output_metrics:
        d["structured_output_metrics"] = r.structured_output_metrics
    if r.latency_breakdown:
        d["latency_breakdown"] = r.latency_breakdown
    if r.noise_injection:
        d["noise_injection"] = r.noise_injection
    return d
