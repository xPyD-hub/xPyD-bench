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
) -> RequestResult:
    """Send one request and collect metrics, with optional retry."""
    last_result = RequestResult()
    attempts = 0
    _extra_hdrs = {"X-Request-ID": request_id} if request_id else None

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
                )
                result.request_id = request_id
            else:
                req_kw = _compressed_request_kwargs(payload, compress, extra_headers=_extra_hdrs)
                resp = await client.post(url, **req_kw, timeout=request_timeout)
                resp.raise_for_status()
                end = time.perf_counter()
                result.latency_ms = (end - start) * 1000.0

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

    try:
        req_kw = _compressed_request_kwargs(payload, compress, extra_headers=extra_headers)
        async with client.stream("POST", url, **req_kw, timeout=request_timeout) as resp:
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
                else:
                    itl = (now - last_token_time) * 1000.0
                    result.itl_ms.append(itl)
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
            args.num_prompts = len(prompts)
        else:
            prompts = _generate_random_prompts(args.num_prompts, args.input_len, args.seed)

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
        )

    async def _task(client: httpx.AsyncClient, prompt: str) -> RequestResult:
        payload = _mk_payload(prompt)
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
    if warmup_count > 0:
        if not args.disable_tqdm:
            print(f"Warmup: sending {warmup_count} request(s)...")
        warmup_prompts = prompts[:warmup_count] if len(prompts) >= warmup_count else (
            prompts * ((warmup_count // len(prompts)) + 1)
        )[:warmup_count]
        warmup_kw = _build_client_kwargs(args, headers=headers)
        async with httpx.AsyncClient(**warmup_kw) as warmup_client:
            for wi, wp in enumerate(warmup_prompts):
                payload = _build_payload(args, wp, is_chat, is_embeddings)
                wr = await _send_request(warmup_client, url, payload, is_streaming)
                if not args.disable_tqdm:
                    status = "ok" if wr.success else f"FAIL: {wr.error}"
                    print(f"  Warmup {wi + 1}/{warmup_count}: {status}")
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

    async def _tracked_task(client: httpx.AsyncClient, prompt: str) -> RequestResult:
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
            task = asyncio.create_task(_tracked_task(client, prompt))
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

    # Anomaly detection (M43)
    anomaly_threshold = getattr(args, "anomaly_threshold", 1.5)
    if anomaly_threshold and anomaly_threshold > 0:
        from xpyd_bench.bench.anomaly import detect_anomalies

        latencies = [r.latency_ms for r in result.requests if r.success]
        anomaly_result = detect_anomalies(latencies, multiplier=anomaly_threshold)
        if anomaly_result is not None and anomaly_result.count > 0:
            result.anomalies = anomaly_result.to_dict()

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
    return d
