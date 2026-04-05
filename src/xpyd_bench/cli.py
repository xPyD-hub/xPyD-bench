"""CLI entry point for xpyd-bench — vLLM bench compatible."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import yaml


def _add_vllm_compat_args(parser: argparse.ArgumentParser) -> None:
    """Add vLLM-bench compatible CLI arguments."""
    # Connection
    parser.add_argument(
        "--backend",
        type=str,
        default="openai",
        help="Backend type (default: openai). Use --list-backends to see available.",
    )
    parser.add_argument(
        "--backend-plugin",
        type=str,
        default=None,
        help="Dotted module path for a custom backend plugin (e.g. my_pkg.my_backend).",
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        default=False,
        help="List available backend plugins and exit.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server base URL (e.g. http://localhost:8000). "
        "Overrides --host/--port.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=8000, help="Server port.")
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint path (default: /v1/completions).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for requests. If omitted, fetched from server.",
    )

    # Streaming control
    stream_group = parser.add_mutually_exclusive_group()
    stream_group.add_argument(
        "--stream",
        action="store_true",
        default=None,
        dest="stream",
        help="Enable streaming responses (default: True for chat, False for completions).",
    )
    stream_group.add_argument(
        "--no-stream",
        action="store_false",
        dest="stream",
        help="Disable streaming responses.",
    )

    # Workload
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Total number of prompts/requests to send (default: 1000).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help=(
            "Run benchmark for a fixed duration in seconds. Prompts cycle "
            "round-robin until time expires. When combined with --num-prompts, "
            "benchmark stops at whichever limit is reached first."
        ),
    )
    # Repeat mode (M49)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat the benchmark (default: 1).",
    )
    parser.add_argument(
        "--repeat-delay",
        type=float,
        default=0.0,
        dest="repeat_delay",
        help="Delay in seconds between repeated runs (default: 0).",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Requests per second. inf = send all at once (default: inf).",
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor for request scheduling (default: 1.0 = Poisson).",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum concurrent in-flight requests.",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=256,
        help="Input prompt length in tokens (for random dataset, default: 256).",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        help="Max output tokens per request (default: 128).",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="random",
        choices=["random", "synthetic"],
        help="Dataset name (default: random). 'synthetic' enables distribution control.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to dataset file (.jsonl, .json, or .csv).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    parser.add_argument(
        "--synthetic-input-len-dist",
        type=str,
        default="fixed",
        choices=["fixed", "uniform", "normal", "zipf"],
        help="Input length distribution for synthetic datasets (default: fixed).",
    )
    parser.add_argument(
        "--synthetic-output-len-dist",
        type=str,
        default="fixed",
        choices=["fixed", "uniform", "normal", "zipf"],
        help="Output length distribution for synthetic datasets (default: fixed).",
    )

    # Sampling parameters (OpenAI API standard)
    sampling = parser.add_argument_group("sampling parameters (OpenAI API standard)")
    sampling.add_argument(
        "--temperature", type=float, default=None, help="Sampling temperature."
    )
    sampling.add_argument("--top-p", type=float, default=None, help="Nucleus sampling top-p.")
    sampling.add_argument("--frequency-penalty", type=float, default=None)
    sampling.add_argument("--presence-penalty", type=float, default=None)
    sampling.add_argument("--logprobs", type=int, default=None)
    sampling.add_argument(
        "--stop",
        type=str,
        nargs="*",
        default=None,
        help="Stop sequence(s) for generation.",
    )
    sampling.add_argument(
        "--n",
        type=int,
        default=None,
        help="Number of completions to generate per request.",
    )
    sampling.add_argument(
        "--api-seed",
        type=int,
        default=None,
        help="Seed sent in API requests (deterministic generation).",
    )
    sampling.add_argument(
        "--echo",
        action="store_true",
        default=False,
        help="Echo back the prompt in addition to the completion (completions only).",
    )
    sampling.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Suffix that comes after a completion of inserted text (completions only).",
    )
    sampling.add_argument(
        "--logit-bias",
        type=str,
        default=None,
        help="JSON string of token ID to bias value mappings, e.g. '{\"50256\": -100}'.",
    )
    sampling.add_argument(
        "--user",
        type=str,
        default=None,
        help="Unique identifier representing the end-user.",
    )
    sampling.add_argument(
        "--stream-options-include-usage",
        action="store_true",
        default=False,
        help="Request usage stats in the final streaming chunk (stream_options).",
    )

    # vLLM-specific extensions (not part of the OpenAI API spec)
    vllm_ext = parser.add_argument_group(
        "vLLM extensions",
        "Parameters specific to vLLM serving. These are NOT part of the "
        "OpenAI API spec and may cause errors with non-vLLM backends.",
    )
    vllm_ext.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="[vLLM only] Top-k sampling. Not in OpenAI API spec.",
    )
    vllm_ext.add_argument(
        "--best-of",
        type=int,
        default=None,
        help="[vLLM only] Generate best_of sequences and return the best. Not in OpenAI API spec.",
    )
    vllm_ext.add_argument(
        "--use-beam-search",
        action="store_true",
        help="[vLLM only] Use beam search instead of sampling. Not in OpenAI API spec.",
    )

    # Chat-specific parameters (OpenAI API)
    chat_params = parser.add_argument_group(
        "chat-specific parameters",
        "Parameters specific to /v1/chat/completions (OpenAI API standard).",
    )
    chat_params.add_argument(
        "--response-format",
        type=str,
        default=None,
        help='Force output format. JSON string, e.g. \'{"type": "json_object"}\'.',
    )
    chat_params.add_argument(
        "--tools",
        type=str,
        default=None,
        help="Path to JSON file defining tools/functions the model may call.",
    )
    chat_params.add_argument(
        "--tool-choice",
        type=str,
        default=None,
        help='Tool choice: "none", "auto", "required", or JSON string for specific function.',
    )
    chat_params.add_argument(
        "--parallel-tool-calls",
        action="store_true",
        default=None,
        help="Allow parallel tool calls (default: server decides).",
    )
    chat_params.add_argument(
        "--top-logprobs",
        type=int,
        default=None,
        help="Number of most likely tokens to return at each position (0-20).",
    )
    chat_params.add_argument(
        "--max-completion-tokens",
        type=int,
        default=None,
        help="Upper bound on completion tokens (newer alternative to max_tokens).",
    )
    chat_params.add_argument(
        "--service-tier",
        type=str,
        default=None,
        help='Latency tier for processing, e.g. "auto" or "default".',
    )

    # Output
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Save benchmark results to JSON file.",
    )
    parser.add_argument("--result-dir", type=str, default=None, help="Result output directory.")
    parser.add_argument("--result-filename", type=str, default=None, help="Result filename.")
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Metadata key-value pairs for result file.",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Disable progress bar.",
    )
    parser.add_argument(
        "--no-live",
        action="store_true",
        default=False,
        help="Disable live progress dashboard (auto-disabled for non-TTY).",
    )

    # Verbosity (M65)
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        "--quiet",
        "-q",
        action="store_const",
        const="quiet",
        dest="verbosity",
        default=None,
        help="Suppress non-essential output (errors and final JSON only).",
    )
    verbosity_group.add_argument(
        "--verbose",
        "-v",
        action="store_const",
        const="verbose",
        dest="verbosity",
        default=None,
        help="Print extra detail (config summary, per-request progress).",
    )

    vllm_ext.add_argument(
        "--ignore-eos",
        action="store_true",
        help="[vLLM only] Ignore EOS token in generation. Not in OpenAI API spec.",
    )

    # Warmup
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Number of warmup requests before benchmark "
        "(excluded from metrics). Default: 0.",
    )
    parser.add_argument(
        "--warmup-profile",
        action="store_true",
        default=False,
        help="Enable warmup profiling: track per-request warmup latencies "
        "and detect stabilization. Requires --warmup >= 2.",
    )

    # Authentication
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for Bearer token authentication. "
        "Falls back to OPENAI_API_KEY env var if not provided.",
    )

    # Timeout & retry
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-request HTTP timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Number of retries on transient errors (connection, 429, 503). Default: 0.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=1.0,
        help="Base delay between retries in seconds, with exponential backoff. Default: 1.0.",
    )

    # Request ID tracking (M42)
    parser.add_argument(
        "--request-id-prefix",
        type=str,
        default=None,
        dest="request_id_prefix",
        help="Prefix for auto-generated X-Request-ID headers (UUID4). "
             "E.g. --request-id-prefix bench- produces bench-<uuid>.",
    )

    # Request compression (M40)
    parser.add_argument(
        "--compress",
        action="store_true",
        default=False,
        help="Enable gzip compression of request bodies (Content-Encoding: gzip).",
    )

    # Anomaly detection (M43)
    parser.add_argument(
        "--anomaly-threshold",
        type=float,
        default=1.5,
        dest="anomaly_threshold",
        help="IQR multiplier for latency anomaly detection (default: 1.5, 0 disables).",
    )

    # Request priority (M52)
    parser.add_argument(
        "--priority-levels",
        type=int,
        default=0,
        dest="priority_levels",
        help="Number of priority levels for request scheduling (0 disables, default: 0). "
        "Levels range from 0 (highest) to N-1 (lowest).",
    )

    # SSE metrics (M53)
    parser.add_argument(
        "--sse-metrics",
        action="store_true",
        default=False,
        dest="sse_metrics",
        help="Enable detailed SSE (Server-Sent Events) streaming metrics analysis.",
    )
    parser.add_argument(
        "--sse-stall-threshold-ms",
        type=float,
        default=1000.0,
        dest="sse_stall_threshold_ms",
        help="Threshold in ms to flag a gap between chunks as a stall (default: 1000).",
    )

    # Network latency decomposition (M57)
    parser.add_argument(
        "--latency-breakdown",
        action="store_true",
        default=False,
        dest="latency_breakdown",
        help=(
            "Enable detailed network latency decomposition "
            "(DNS, TCP connect, TLS, server processing)."
        ),
    )

    # Rate-limit header tracking (M66)
    parser.add_argument(
        "--track-ratelimits",
        action="store_true",
        default=False,
        dest="track_ratelimits",
        help="Track rate-limit response headers and report backpressure summary.",
    )

    # Payload size tracking (M67)
    parser.add_argument(
        "--track-payload-size",
        action="store_true",
        default=False,
        dest="track_payload_size",
        help="Track request/response payload sizes and report bandwidth summary.",
    )

    # Response validation (M47)
    parser.add_argument(
        "--validate-response",
        type=str,
        action="append",
        default=None,
        dest="validate_response",
        metavar="MODE",
        help=(
            "Validate response content (repeatable). Modes: non-empty, json, "
            "regex:<pattern>, min-tokens:<N>."
        ),
    )

    # Custom headers
    parser.add_argument(
        "--header",
        type=str,
        action="append",
        default=None,
        dest="header",
        metavar="KEY:VALUE",
        help='Custom HTTP header (repeatable). Format: "Key: Value".',
    )

    # Rate algorithm (M16)
    parser.add_argument(
        "--rate-algorithm",
        type=str,
        default="default",
        choices=["default", "token-bucket"],
        help="Rate limiting algorithm. 'default' uses sleep-based Poisson scheduling, "
        "'token-bucket' uses a token bucket for smoother rate control (default: default).",
    )
    parser.add_argument(
        "--token-bucket-burst",
        type=float,
        default=None,
        help="Token bucket burst capacity. Defaults to --request-rate value.",
    )

    # Adaptive concurrency (M16)
    parser.add_argument(
        "--adaptive-concurrency",
        action="store_true",
        default=False,
        help="Enable adaptive concurrency adjustment based on server response latency.",
    )
    parser.add_argument(
        "--adaptive-target-latency",
        type=float,
        default=500.0,
        help="Target latency in ms for adaptive concurrency (default: 500).",
    )
    parser.add_argument(
        "--adaptive-min-concurrency",
        type=int,
        default=1,
        help="Minimum concurrency for adaptive mode (default: 1).",
    )
    parser.add_argument(
        "--adaptive-max-concurrency",
        type=int,
        default=256,
        help="Maximum concurrency for adaptive mode (default: 256).",
    )
    parser.add_argument(
        "--adaptive-initial-concurrency",
        type=int,
        default=16,
        help="Initial concurrency for adaptive mode (default: 16).",
    )

    # Dry run
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Validate configuration and dataset without sending requests. "
        "Prints execution plan and exits.",
    )

    # SLA validation (M20)
    parser.add_argument(
        "--sla",
        type=str,
        default=None,
        help="Path to YAML file with SLA targets for validation.",
    )

    # Tokenizer (M27)
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tiktoken model or encoding name for accurate token counting "
        "(e.g. cl100k_base, gpt-4). Falls back to word-split when tiktoken "
        "is not installed. (default: None = word-split)",
    )

    # Connection pool & HTTP/2 (M28)
    parser.add_argument(
        "--http2",
        action="store_true",
        default=False,
        help="Enable HTTP/2 multiplexing (requires h2 package).",
    )
    parser.add_argument(
        "--max-connections",
        type=int,
        default=100,
        help="Maximum number of connections in the pool (default: 100).",
    )
    parser.add_argument(
        "--max-keepalive",
        type=int,
        default=20,
        help="Maximum number of keepalive connections (default: 20).",
    )

    # Extended config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file for extended options.",
    )

    # Scenarios (M7)
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Use a built-in scenario preset (short, long_context, mixed, stress).",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List available built-in scenarios and exit.",
    )

    # Reporting (M6)
    reporting = parser.add_argument_group("reporting")
    reporting.add_argument(
        "--export-requests",
        type=str,
        default=None,
        metavar="PATH",
        help="Export per-request detailed metrics to a JSON file.",
    )
    reporting.add_argument(
        "--json-report",
        type=str,
        default=None,
        metavar="PATH",
        help="Export full JSON report (summary + time-series) to a file.",
    )
    reporting.add_argument(
        "--text-report",
        type=str,
        default=None,
        metavar="PATH",
        help="Export human-readable text report to a file.",
    )
    reporting.add_argument(
        "--time-series-window",
        type=float,
        default=1.0,
        help="Time-series bucketing window in seconds (default: 1.0).",
    )
    reporting.add_argument(
        "--csv-report",
        type=str,
        default=None,
        metavar="PATH",
        help="Export summary metrics as a single-row CSV file.",
    )
    reporting.add_argument(
        "--markdown-report",
        type=str,
        default=None,
        metavar="PATH",
        help="Export summary metrics as a Markdown table.",
    )
    reporting.add_argument(
        "--export-requests-csv",
        type=str,
        default=None,
        metavar="PATH",
        help="Export per-request detailed metrics as CSV.",
    )
    reporting.add_argument(
        "--html-report",
        type=str,
        default=None,
        metavar="PATH",
        help="Export an interactive HTML report dashboard.",
    )
    reporting.add_argument(
        "--prometheus-export",
        type=str,
        default=None,
        metavar="PATH",
        help="Export metrics in Prometheus text exposition format.",
    )
    reporting.add_argument(
        "--junit-xml",
        type=str,
        default=None,
        metavar="PATH",
        help="Export results as JUnit XML for CI integration.",
    )
    reporting.add_argument(
        "--debug-log",
        type=str,
        default=None,
        metavar="PATH",
        help="Write per-request debug logs (JSONL) for diagnosing failures.",
    )
    reporting.add_argument(
        "--metrics-ws-port",
        type=int,
        default=None,
        metavar="PORT",
        help="Expose live metrics via WebSocket on the given port during benchmark.",
    )
    reporting.add_argument(
        "--rich-progress",
        action="store_true",
        help="Use rich progress bar and summary table.",
    )
    reporting.add_argument(
        "--heatmap",
        action="store_true",
        default=False,
        help="Display a latency heatmap in the terminal after the benchmark.",
    )

    # Tags (M36)
    parser.add_argument(
        "--tag",
        action="append",
        dest="tags",
        default=None,
        metavar="KEY=VALUE",
        help="Attach a metadata tag (repeatable). E.g. --tag env=prod --tag gpu=A100",
    )

    # Template variables (M37)
    parser.add_argument(
        "--template-vars",
        type=str,
        default=None,
        dest="template_vars",
        metavar="PATH",
        help="Path to JSON/YAML file with template variables for prompt substitution.",
    )

    # Presets (M38)
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Use a benchmark preset (e.g. throughput-max, latency-optimal, soak-test, "
        "cold-start). CLI flags override preset values. Use 'xpyd-bench presets list' "
        "to see available presets.",
    )
    parser.add_argument(
        "--presets-dir",
        type=str,
        default=None,
        help="Path to user presets directory (default: ~/.xpyd-bench/presets/).",
    )

    # Cost estimation (M39)
    parser.add_argument(
        "--cost-model",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to YAML cost model file mapping model names to $/1K tokens.",
    )

    # Multi-turn conversation benchmarking (M45)
    parser.add_argument(
        "--multi-turn",
        type=str,
        default=None,
        dest="multi_turn",
        metavar="PATH",
        help=(
            "Path to JSONL file with multi-turn conversations, or 'synthetic' "
            "for auto-generated conversations."
        ),
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        dest="max_turns",
        help="Maximum number of turns per conversation (default: unlimited).",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=5,
        help="Number of turns for synthetic multi-turn generation (default: 5).",
    )

    # Load shedding simulation (M55)
    parser.add_argument(
        "--load-shed-threshold",
        type=float,
        default=None,
        dest="load_shed_threshold",
        help="Starting RPS for load shedding simulation. Ramps up until server rejects.",
    )
    parser.add_argument(
        "--load-shed-step",
        type=float,
        default=0.0,
        dest="load_shed_step",
        help="Additive RPS step for load shedding ramp (0 = use multiplier, default: 0).",
    )
    parser.add_argument(
        "--load-shed-multiplier",
        type=float,
        default=1.5,
        dest="load_shed_multiplier",
        help="Multiplicative RPS ramp factor for load shedding (default: 1.5).",
    )
    parser.add_argument(
        "--load-shed-prompts",
        type=int,
        default=50,
        dest="load_shed_prompts",
        help="Number of prompts per load shedding level (default: 50).",
    )

    # Noise injection / Chaos testing (M60)
    parser.add_argument(
        "--inject-delay",
        type=float,
        default=0.0,
        dest="inject_delay",
        help="Artificial client-side delay in ms before each request (default: 0).",
    )
    parser.add_argument(
        "--inject-error-rate",
        type=float,
        default=0.0,
        dest="inject_error_rate",
        help="Fraction of requests to abort client-side (0.0-1.0, default: 0).",
    )
    parser.add_argument(
        "--inject-payload-corruption",
        type=float,
        default=0.0,
        dest="inject_payload_corruption",
        help="Fraction of request payloads to corrupt before sending (0.0-1.0, default: 0).",
    )

    # Webhook notifications (M61)
    parser.add_argument(
        "--webhook-url",
        type=str,
        action="append",
        default=None,
        dest="webhook_url",
        help="URL to POST benchmark results to (repeatable for multiple endpoints).",
    )
    parser.add_argument(
        "--webhook-secret",
        type=str,
        default=None,
        dest="webhook_secret",
        help="Secret for HMAC-SHA256 webhook signature (X-Webhook-Signature header).",
    )

    # OTLP trace export (M62)
    parser.add_argument(
        "--otlp-endpoint",
        type=str,
        default=None,
        dest="otlp_endpoint",
        help="OTLP/HTTP endpoint URL to export request trace spans (e.g. http://localhost:4318).",
    )

    # On-complete command (M63)
    parser.add_argument(
        "--on-complete",
        type=str,
        default=None,
        dest="on_complete",
        help="Shell command to run after benchmark completes (e.g. 'notify-send done').",
    )


def _resolve_base_url(args: argparse.Namespace) -> str:
    """Resolve the base URL from --base-url or --host/--port."""
    if args.base_url:
        return args.base_url.rstrip("/")
    return f"http://{args.host}:{args.port}"


def _parse_tags(args: argparse.Namespace) -> dict[str, str]:
    """Parse --tag KEY=VALUE items and YAML ``tags`` config into a dict."""
    tags: dict[str, str] = {}
    # YAML config may set args.tags as a dict directly
    yaml_tags = getattr(args, "tags", None)
    if isinstance(yaml_tags, dict):
        tags.update({str(k): str(v) for k, v in yaml_tags.items()})
        return tags
    # CLI sets args.tags as a list of "KEY=VALUE" strings
    if yaml_tags:
        for item in yaml_tags:
            if "=" in item:
                k, v = item.split("=", 1)
                tags[k.strip()] = v.strip()
    return tags


def _dry_run(args: argparse.Namespace, base_url: str) -> None:
    """Validate configuration and dataset, print execution plan, then exit."""
    from xpyd_bench import __version__
    from xpyd_bench.datasets.loader import load_dataset, validate_and_report

    print(f"xpyd-bench v{__version__} — DRY RUN")
    print(f"{'='*60}")
    print()

    # Execution plan
    print("Execution Plan:")
    print(f"  Base URL:        {base_url}")
    print(f"  Endpoint:        {args.endpoint}")
    stream_val = args.stream if args.stream is not None else ("chat" in args.endpoint)
    print(f"  Streaming:       {stream_val}")
    print(f"  Backend:         {args.backend}")
    print(f"  Model:           {args.model or '(auto-detect)'}")
    print(f"  Num prompts:     {args.num_prompts}")
    duration_limit = getattr(args, "duration", None)
    if duration_limit:
        print(f"  Duration limit:  {duration_limit}s")
        print("  Mode:            duration (prompts cycle until time expires)")
    else:
        print(f"  Mode:            count-based ({args.num_prompts} prompts)")
    print(f"  Request rate:    {args.request_rate}")
    print(f"  Max concurrency: {args.max_concurrency or 'unlimited'}")
    print(f"  Input len:       {args.input_len}")
    print(f"  Output len:      {args.output_len}")
    print(f"  Seed:            {args.seed}")
    print(f"  Rate algorithm:  {args.rate_algorithm}")

    # Tokenizer
    tokenizer = getattr(args, "tokenizer", None)
    print(f"  Tokenizer:       {tokenizer or '(word-split)'}")

    # Repeat mode (M49)
    repeat_count = getattr(args, "repeat", 1)
    repeat_delay = getattr(args, "repeat_delay", 0.0)
    if repeat_count > 1:
        print(f"  Repeat:          {repeat_count}x (delay: {repeat_delay}s)")

    # Connection pool (M28)
    http2 = getattr(args, "http2", False)
    max_conn = getattr(args, "max_connections", 100)
    max_ka = getattr(args, "max_keepalive", 20)
    print(f"  HTTP/2:          {http2}")
    print(f"  Max connections: {max_conn}")
    print(f"  Max keepalive:   {max_ka}")

    # Auth
    api_key_source = "none"
    if args.api_key:
        api_key_source = "CLI/config (set)"
    print(f"  API key:         {api_key_source}")

    # Custom headers
    if args.custom_headers:
        print(f"  Custom headers:  {len(args.custom_headers)} header(s)")
        for k, v in args.custom_headers.items():
            print(f"    {k}: {v}")
    else:
        print("  Custom headers:  none")

    # Warmup
    warmup = getattr(args, "warmup", None) or 0
    print(f"  Warmup requests: {warmup}")
    if getattr(args, "warmup_profile", False):
        print("  Warmup profiling: enabled")

    # Timeout / retry
    print(f"  Timeout:         {args.timeout}s")
    print(f"  Retries:         {args.retries}")
    if args.retries > 0:
        print(f"  Retry delay:     {args.retry_delay}s (exponential backoff)")

    # Scenario
    scenario = getattr(args, "scenario", None)
    if scenario:
        print(f"  Scenario:        {scenario}")

    print()

    # Dataset validation
    print("Dataset Validation:")
    try:
        entries = load_dataset(
            path=args.dataset_path,
            name=args.dataset_name,
            num_prompts=args.num_prompts,
            input_len=args.input_len,
            output_len=args.output_len,
            input_len_dist=getattr(args, "synthetic_input_len_dist", "fixed"),
            output_len_dist=getattr(args, "synthetic_output_len_dist", "fixed"),
            seed=args.seed,
            tokenizer=getattr(args, "tokenizer", None),
        )
        source = args.dataset_path or f"synthetic ({args.dataset_name})"
        validate_and_report(entries, source, tokenizer=getattr(args, "tokenizer", None))
    except (ValueError, FileNotFoundError) as exc:
        print(f"  ERROR: {exc}")
        raise SystemExit(1) from exc

    print()

    # Estimated duration
    rate = args.request_rate
    if rate != float("inf") and rate > 0:
        est_seconds = args.num_prompts / rate
        if est_seconds >= 60:
            print(f"Estimated duration: {est_seconds / 60:.1f} min ({est_seconds:.0f}s)")
        else:
            print(f"Estimated duration: {est_seconds:.1f}s")
    else:
        print("Estimated duration: all requests sent at once (rate=inf)")

    print()
    print("Dry run complete. Configuration is valid.")

    # Cost estimation in dry-run (M39)
    cost_model_path = getattr(args, "cost_model", None)
    if cost_model_path:
        from xpyd_bench.cost import (
            estimate_cost_from_counts,
            format_cost_summary,
            load_cost_model,
        )

        cmodel = load_cost_model(cost_model_path)
        est_input = args.num_prompts * args.input_len
        est_output = args.num_prompts * args.output_len
        cost_est = estimate_cost_from_counts(
            est_input, est_output, cmodel, model_name=args.model or ""
        )
        print()
        print(format_cost_summary(cost_est))
        print("  (estimated from num_prompts × input_len / output_len)")

    # Environment info
    from xpyd_bench.bench.env import collect_env_info

    env = collect_env_info()
    print()
    print("Environment:")
    for k, v in env.items():
        print(f"  {k}: {v}")

    print(f"{'='*60}")


def _parse_header(raw: str) -> tuple[str, str]:
    """Parse a 'Key: Value' string into (key, value)."""
    if ":" not in raw:
        raise ValueError(f"Invalid header format (expected 'Key: Value'): {raw!r}")
    key, value = raw.split(":", 1)
    return key.strip(), value.strip()


def _resolve_custom_headers(args: argparse.Namespace) -> dict[str, str]:
    """Build custom headers dict from YAML config + CLI --header flags.

    CLI headers take precedence over YAML ``headers`` dict.
    """
    headers: dict[str, str] = {}
    # YAML headers (lower priority)
    yaml_headers = getattr(args, "headers", None)
    if isinstance(yaml_headers, dict):
        for k, v in yaml_headers.items():
            headers[str(k)] = str(v)
    # CLI --header flags (higher priority)
    cli_headers = getattr(args, "header", None)
    if cli_headers:
        for raw in cli_headers:
            k, v = _parse_header(raw)
            headers[k] = v
    return headers


def _get_explicit_keys(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> set[str]:
    """Return the set of attribute names explicitly provided on the CLI.

    Compares *args* against the parser defaults — any attribute whose value
    differs from the default is considered explicitly provided.
    """
    defaults = vars(parser.parse_args([]))
    return {k for k, v in vars(args).items() if v != defaults.get(k)}


def _load_yaml_config(
    path: str,
    args: argparse.Namespace,
    explicit_keys: set[str] | None = None,
) -> argparse.Namespace:
    """Merge YAML config into args (explicitly-provided CLI flags take precedence).

    Parameters
    ----------
    path:
        Path to the YAML configuration file.
    args:
        Parsed CLI namespace.
    explicit_keys:
        Set of attribute names that were explicitly provided on the command
        line.  When given, only those keys are protected from YAML overrides.
        When *None* (legacy call-sites), the function falls back to the
        previous behaviour of skipping keys whose current value is not
        ``None``.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    for key, value in cfg.items():
        attr = key.replace("-", "_")
        if explicit_keys is not None:
            # Only skip keys the user explicitly typed on the CLI
            if attr in explicit_keys:
                continue
            if hasattr(args, attr):
                setattr(args, attr, value)
        else:
            # Legacy fallback: only set if current value is None
            if getattr(args, attr, None) is None:
                setattr(args, attr, value)
    return args


def bench_main(argv: list[str] | None = None) -> None:
    """Entry point for ``xpyd-bench`` command."""
    parser = argparse.ArgumentParser(
        prog="xpyd-bench",
        description="Benchmark an OpenAI-compatible serving endpoint (vLLM bench compatible)",
    )
    _add_vllm_compat_args(parser)
    args = parser.parse_args(argv)

    # List backends and exit
    if getattr(args, "list_backends", False):
        from xpyd_bench.plugins import registry

        backends = registry.list_backends()
        print("Available backends:\n")
        for b in backends:
            print(f"  {b}")
        print(
            "\nUse --backend <name> to select a backend, or "
            "--backend-plugin <module> to load a custom one."
        )
        return

    # Load custom backend plugin if specified
    backend_plugin_module = getattr(args, "backend_plugin", None)
    if backend_plugin_module:
        from xpyd_bench.plugins import registry

        registry.load_module_plugin(backend_plugin_module)

    # List scenarios and exit
    if args.list_scenarios:
        from xpyd_bench.scenarios import list_scenarios

        scenarios = list_scenarios()
        print("Available scenarios:\n")
        for s in scenarios:
            print(f"  {s.name:15s} {s.description}")
            overrides = s.to_overrides()
            for k, v in overrides.items():
                print(f"    {k}: {v}")
            print()
        return

    # Apply scenario defaults (CLI flags take precedence)
    if args.scenario:
        from xpyd_bench.scenarios import get_scenario

        scenario = get_scenario(args.scenario)
        overrides = scenario.to_overrides()
        for key, value in overrides.items():
            # Only apply if the user did not explicitly set the flag
            current = getattr(args, key, None)
            if current is None or current == parser.get_default(key):
                setattr(args, key, value)

    # Apply preset defaults (CLI flags take precedence) (M38)
    preset_name = getattr(args, "preset", None)
    if preset_name:
        from pathlib import Path as _Path

        from xpyd_bench.presets import get_preset

        pdir = _Path(args.presets_dir) if getattr(args, "presets_dir", None) else None
        preset = get_preset(preset_name, pdir)
        for key, value in preset.to_overrides().items():
            current = getattr(args, key, None)
            if current is None or current == parser.get_default(key):
                setattr(args, key, value)

    # Merge YAML config if provided
    if args.config:
        explicit = _get_explicit_keys(parser, args)
        args = _load_yaml_config(args.config, args, explicit_keys=explicit)

    # Apply preset from YAML config if not already set via CLI (M38)
    yaml_preset = getattr(args, "preset", None)
    if yaml_preset and not preset_name:
        from pathlib import Path as _Path

        from xpyd_bench.presets import get_preset

        pdir = _Path(args.presets_dir) if getattr(args, "presets_dir", None) else None
        preset = get_preset(yaml_preset, pdir)
        for key, value in preset.to_overrides().items():
            current = getattr(args, key, None)
            if current is None or current == parser.get_default(key):
                setattr(args, key, value)

    base_url = _resolve_base_url(args)

    # Resolve API key: CLI > YAML config > env var
    import os

    if args.api_key is None:
        args.api_key = os.environ.get("OPENAI_API_KEY")

    # Resolve custom headers: YAML + CLI (CLI wins)
    args.custom_headers = _resolve_custom_headers(args)

    from xpyd_bench import __version__
    from xpyd_bench.bench.runner import run_benchmark

    # Dry run: validate config and dataset, print plan, exit
    if getattr(args, "dry_run", False):
        _dry_run(args, base_url)
        return

    # Load shedding simulation (M55)
    load_shed_rps = getattr(args, "load_shed_threshold", None)
    if load_shed_rps is not None:
        from xpyd_bench.load_shed import (
            format_load_shed_table,
            run_load_shed,
        )

        print(f"xpyd-bench v{__version__} (load shedding mode)")
        print(f"  Base URL:        {base_url}")
        print(f"  Starting RPS:    {load_shed_rps}")
        print()

        analysis = asyncio.run(
            run_load_shed(
                args,
                base_url,
                starting_rps=load_shed_rps,
                ramp_step=getattr(args, "load_shed_step", 0.0),
                ramp_multiplier=getattr(args, "load_shed_multiplier", 1.5),
                prompts_per_level=getattr(args, "load_shed_prompts", 50),
            )
        )
        print(format_load_shed_table(analysis))

        result = {"saturation_analysis": analysis.to_dict()}
        if getattr(args, "save_result", None):
            _save_result(args, result)
        return

    # Multi-turn conversation mode (M45)
    multi_turn_path = getattr(args, "multi_turn", None)
    if multi_turn_path:
        from xpyd_bench.multi_turn import (
            compute_multi_turn_stats,
            generate_synthetic_conversations,
            load_multi_turn_dataset,
            run_conversation,
        )

        print(f"xpyd-bench v{__version__} (multi-turn mode)")
        print(f"  Base URL:       {base_url}")
        print(f"  Model:          {args.model or '(auto-detect)'}")
        max_turns = getattr(args, "max_turns", None)
        print(f"  Max turns:      {max_turns or 'unlimited'}")

        if multi_turn_path == "synthetic":
            conversations = generate_synthetic_conversations(
                num_conversations=args.num_prompts,
                turns=getattr(args, "turns", 5),
                input_len=args.input_len,
                seed=getattr(args, "seed", 42),
            )
            print(f"  Conversations:  {len(conversations)} (synthetic)")
        else:
            conversations = load_multi_turn_dataset(multi_turn_path)
            print(f"  Conversations:  {len(conversations)} (from {multi_turn_path})")
        print()

        import httpx

        async def _run_multi_turn() -> list:
            async with httpx.AsyncClient() as client:
                results = []
                for i, msgs in enumerate(conversations):
                    conv_result = await run_conversation(
                        client=client,
                        base_url=base_url,
                        model=args.model or "mock-model",
                        messages=msgs,
                        conversation_id=i,
                        max_turns=max_turns,
                        endpoint="/v1/chat/completions",
                        api_key=getattr(args, "api_key", None),
                        timeout=getattr(args, "timeout", 300.0),
                    )
                    results.append(conv_result)
                    print(
                        f"  Conversation {i+1}/{len(conversations)}: "
                        f"{conv_result.total_turns} turns, "
                        f"{conv_result.total_latency_ms:.0f}ms"
                    )
                return results

        conv_results = asyncio.run(_run_multi_turn())
        mt_result = compute_multi_turn_stats(conv_results)

        # Print summary
        stats = mt_result.aggregate_stats
        print(f"\n{'='*60}")
        print("  Multi-Turn Benchmark Summary")
        print(f"{'='*60}")
        print(f"  Conversations:    {stats.get('total_conversations', 0)}")
        print(f"  Total turns:      {stats.get('total_turns', 0)}")
        print(f"  Errors:           {stats.get('total_errors', 0)}")
        print(f"  Mean TTFT:        {stats.get('mean_ttft_ms', 0):.2f} ms")
        print(f"  Mean latency:     {stats.get('mean_latency_ms', 0):.2f} ms")
        if "p50_latency_ms" in stats:
            print(f"  P50 latency:      {stats['p50_latency_ms']:.2f} ms")
            print(f"  P99 latency:      {stats['p99_latency_ms']:.2f} ms")

        # Per-turn stats
        if mt_result.per_turn_stats:
            print("\n  Per-Turn Statistics:")
            print(
                f"  {'Turn':>6}  {'Count':>6}  {'TTFT(ms)':>10}  "
                f"{'Latency(ms)':>12}  {'Context(tok)':>13}"
            )
            for idx in sorted(mt_result.per_turn_stats.keys()):
                ts = mt_result.per_turn_stats[idx]
                print(
                    f"  {idx:>6}  {ts['count']:>6}  "
                    f"{ts['mean_ttft_ms']:>10.2f}  "
                    f"{ts['mean_latency_ms']:>12.2f}  "
                    f"{ts['mean_context_tokens']:>13.0f}"
                )
        print(f"{'='*60}")

        # Save result if requested
        result = mt_result.to_dict()
        if getattr(args, "save_result", None):
            p = Path(args.save_result)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(result, indent=2))
            print(f"\nResults saved to {p}")

        return

    print(f"xpyd-bench v{__version__}")
    print(f"  Base URL:       {base_url}")
    print(f"  Endpoint:       {args.endpoint}")
    print(f"  Model:          {args.model or '(auto-detect)'}")
    print(f"  Num prompts:    {args.num_prompts}")
    print(f"  Request rate:   {args.request_rate}")
    print(f"  Max concurrency:{args.max_concurrency or 'unlimited'}")
    print(f"  Input len:      {args.input_len}")
    print(f"  Output len:     {args.output_len}")
    repeat_count = getattr(args, "repeat", 1)
    repeat_delay = getattr(args, "repeat_delay", 0.0)
    if repeat_count > 1:
        print(f"  Repeat:         {repeat_count}x (delay: {repeat_delay}s)")
    print()

    if repeat_count > 1:
        from xpyd_bench.repeat import (
            print_repeat_summary,
            run_repeated_benchmark,
        )

        repeat_result = asyncio.run(run_repeated_benchmark(args, base_url))
        print_repeat_summary(repeat_result)

        # Use last run as the "primary" result for downstream exports
        if repeat_result.per_run_results:
            result = repeat_result.per_run_results[-1]
            # Merge repeat summary into result
            result.update(repeat_result.to_dict())
        else:
            print("No runs completed.")
            return

        # Reconstruct a minimal bench_result from the last run dict
        from xpyd_bench.bench.models import BenchmarkResult

        bench_result = BenchmarkResult(
            base_url=base_url,
            model=args.model or "",
            num_prompts=args.num_prompts,
            request_rate=args.request_rate,
            completed=result.get("completed", 0),
            failed=result.get("failed", 0),
            request_throughput=result.get("request_throughput", 0.0),
            output_throughput=result.get("output_throughput", 0.0),
            total_token_throughput=result.get("total_token_throughput", 0.0),
            mean_ttft_ms=result.get("mean_ttft_ms"),
            mean_tpot_ms=result.get("mean_tpot_ms"),
            total_duration_s=result.get("total_duration_s", 0.0),
        )
    else:
        result, bench_result = asyncio.run(run_benchmark(args, base_url))

    # Inject tags (M36)
    parsed_tags = _parse_tags(args)
    if parsed_tags:
        bench_result.tags = parsed_tags
        result["tags"] = parsed_tags

    # Export reports (M6)
    if args.export_requests:
        from xpyd_bench.reporting.formats import export_per_request

        p = export_per_request(bench_result, args.export_requests)
        print(f"\nPer-request metrics saved to {p}")

    if args.json_report:
        from xpyd_bench.reporting.formats import export_json_report

        p = export_json_report(
            bench_result, result, args.json_report, args.time_series_window
        )
        print(f"\nJSON report saved to {p}")

    if args.text_report:
        from xpyd_bench.reporting.formats import format_text_report

        text = format_text_report(bench_result)
        rpath = Path(args.text_report)
        rpath.parent.mkdir(parents=True, exist_ok=True)
        rpath.write_text(text)
        print(f"\nText report saved to {rpath}")

    if args.csv_report:
        from xpyd_bench.reporting.formats import export_csv_report

        p = export_csv_report(bench_result, args.csv_report)
        print(f"\nCSV report saved to {p}")

    if args.markdown_report:
        from xpyd_bench.reporting.formats import export_markdown_report

        p = export_markdown_report(bench_result, args.markdown_report)
        print(f"\nMarkdown report saved to {p}")

    if args.export_requests_csv:
        from xpyd_bench.reporting.formats import export_per_request_csv

        p = export_per_request_csv(bench_result, args.export_requests_csv)
        print(f"\nPer-request CSV saved to {p}")

    if getattr(args, "html_report", None):
        from xpyd_bench.reporting.html_report import export_html_report

        p = export_html_report(bench_result, args.html_report)
        print(f"\nHTML report saved to {p}")

    # Prometheus export (M34)
    if getattr(args, "prometheus_export", None):
        from xpyd_bench.reporting.prometheus import export_prometheus

        scenario = getattr(args, "scenario", None)
        p = export_prometheus(bench_result, args.prometheus_export, scenario)
        print(f"\nPrometheus metrics saved to {p}")

    # JUnit XML export (M49)
    if getattr(args, "junit_xml", None):
        from xpyd_bench.junit_xml import write_junit_xml

        write_junit_xml(bench_result, args.junit_xml)
        print(f"\nJUnit XML report saved to {args.junit_xml}")

    # Debug log notification (M22)
    if getattr(args, "debug_log", None):
        print(f"\nDebug log saved to {args.debug_log}")

    # Terminal heatmap (M35)
    if getattr(args, "heatmap", False):
        from xpyd_bench.reporting.heatmap import compute_heatmap, render_terminal_heatmap

        heatmap_data = compute_heatmap(bench_result)
        print()
        print(render_terminal_heatmap(heatmap_data))

    # Auto-save to result-dir when set (M30)
    if args.result_dir and not args.save_result:
        from xpyd_bench.history import auto_save_result

        saved = auto_save_result(result, args.result_dir, args)
        print(f"\nResults auto-saved to {saved}")

    # Save results if requested
    if args.save_result:
        _save_result(args, result)

    # Cost estimation (M39)
    cost_model_path = getattr(args, "cost_model", None)
    if cost_model_path:
        from xpyd_bench.cost import (
            cost_to_dict,
            estimate_cost,
            format_cost_summary,
            load_cost_model,
        )

        cmodel = load_cost_model(cost_model_path)
        cost_est = estimate_cost(bench_result, cmodel)
        print()
        print(format_cost_summary(cost_est))
        result["cost"] = cost_to_dict(cost_est)

    # Webhook notifications (M61)
    webhook_urls = getattr(args, "webhook_url", None)
    if webhook_urls:
        from xpyd_bench.webhook import format_webhook_summary, send_webhooks

        webhook_secret = getattr(args, "webhook_secret", None)
        deliveries = send_webhooks(webhook_urls, result, secret=webhook_secret)
        print()
        print(format_webhook_summary(deliveries))

    # OTLP trace export (M62)
    otlp_endpoint = getattr(args, "otlp_endpoint", None)
    if otlp_endpoint:
        from xpyd_bench.otlp import export_traces, format_otlp_summary

        delivery = export_traces(otlp_endpoint, result)
        print()
        print(format_otlp_summary(delivery))

    # SLA validation (M20)
    sla_path = getattr(args, "sla", None)
    if sla_path:
        from xpyd_bench.sla import (
            format_sla_table,
            load_sla_targets,
            validate_sla,
        )

        targets = load_sla_targets(sla_path)
        sla_report = validate_sla(bench_result, targets)
        print(format_sla_table(sla_report))
        if not sla_report.all_passed:
            raise SystemExit(1)

    # On-complete command (M63)
    on_complete = getattr(args, "on_complete", None)
    if on_complete:
        from xpyd_bench.schedule import run_on_complete

        oc_result = run_on_complete(on_complete)
        if oc_result.returncode == 0:
            print(f"\nOn-complete command succeeded: {on_complete}")
        else:
            print(f"\nOn-complete command failed (exit {oc_result.returncode}): {on_complete}")
            if oc_result.stderr:
                print(f"  stderr: {oc_result.stderr.strip()}")


def compare_main(argv: list[str] | None = None) -> None:
    """Entry point for ``xpyd-bench-compare`` command."""
    from xpyd_bench.compare import (
        compare,
        format_comparison_table,
        load_result,
    )

    parser = argparse.ArgumentParser(
        prog="xpyd-bench-compare",
        description="Compare two benchmark results and detect regressions",
    )
    parser.add_argument("baseline", help="Path to baseline result JSON file")
    parser.add_argument("candidate", help="Path to candidate result JSON file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Regression threshold percentage (default: 5.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help="Export JSON diff to a file",
    )
    args = parser.parse_args(argv)

    baseline = load_result(args.baseline)
    candidate = load_result(args.candidate)
    result = compare(baseline, candidate, threshold_pct=args.threshold)

    print(format_comparison_table(result))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nJSON diff saved to {out_path}")

    if result.has_regression:
        raise SystemExit(1)


def multi_main(argv: list[str] | None = None) -> None:
    """Entry point for ``xpyd-bench-multi`` command."""
    parser = argparse.ArgumentParser(
        prog="xpyd-bench-multi",
        description="Benchmark multiple endpoints with the same workload and compare results",
    )
    parser.add_argument(
        "--endpoints",
        type=str,
        nargs="+",
        required=True,
        help="Base URLs of endpoints to benchmark (at least 2).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Regression threshold percentage (default: 5.0).",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        metavar="PATH",
        help="Export all results and comparisons to a JSON file.",
    )
    parser.add_argument(
        "--markdown-output",
        type=str,
        default=None,
        metavar="PATH",
        help="Export comparison as a Markdown table.",
    )
    _add_vllm_compat_args(parser)
    args = parser.parse_args(argv)

    if len(args.endpoints) < 2:
        parser.error("--endpoints requires at least 2 URLs")

    # Merge YAML config if provided
    if args.config:
        explicit = _get_explicit_keys(parser, args)
        args = _load_yaml_config(args.config, args, explicit_keys=explicit)

    # Resolve API key
    import os

    if args.api_key is None:
        args.api_key = os.environ.get("OPENAI_API_KEY")

    # Resolve custom headers
    args.custom_headers = _resolve_custom_headers(args)

    from xpyd_bench import __version__
    from xpyd_bench.multi import (
        export_multi_json,
        export_multi_markdown,
        format_multi_summary,
        run_multi_benchmark,
    )

    print(f"xpyd-bench-multi v{__version__}")
    print(f"  Endpoints: {', '.join(args.endpoints)}")
    print(f"  Threshold: {args.threshold}%")
    print()

    multi = asyncio.run(
        run_multi_benchmark(args, args.endpoints, threshold_pct=args.threshold)
    )

    print(format_multi_summary(multi))

    if args.json_output:
        p = export_multi_json(multi, args.json_output)
        print(f"\nJSON output saved to {p}")

    if args.markdown_output:
        p = export_multi_markdown(multi, args.markdown_output)
        print(f"\nMarkdown output saved to {p}")

    # Exit code 1 if any regression detected
    if any(c.has_regression for c in multi.comparisons):
        raise SystemExit(1)


def profile_main(argv: list[str] | None = None) -> None:
    """Entry point for ``xpyd-bench-profile`` command.

    Runs a benchmark and records a request timing trace to a JSON file.
    """
    from xpyd_bench.profile import TraceRecorder, save_trace

    parser = argparse.ArgumentParser(
        prog="xpyd-bench-profile",
        description="Run a benchmark and record request timing trace",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        metavar="PATH",
        help="Output path for the trace JSON file.",
    )
    _add_vllm_compat_args(parser)
    args = parser.parse_args(argv)

    # Merge YAML config if provided
    if args.config:
        explicit = _get_explicit_keys(parser, args)
        args = _load_yaml_config(args.config, args, explicit_keys=explicit)

    import os

    if args.api_key is None:
        args.api_key = os.environ.get("OPENAI_API_KEY")
    args.custom_headers = _resolve_custom_headers(args)

    base_url = _resolve_base_url(args)
    args.base_url = base_url

    from xpyd_bench import __version__
    from xpyd_bench.bench.runner import run_benchmark
    from xpyd_bench.datasets.loader import load_dataset

    print(f"xpyd-bench-profile v{__version__}")
    print(f"  Base URL: {base_url}")
    print(f"  Output:   {args.output}")
    print()

    # Create recorder
    recorder = TraceRecorder(base_url=base_url)

    # Load dataset
    prompts = load_dataset(args)

    # Record each prompt into the trace as we dispatch
    recorder.start()
    for p in prompts:
        prompt_len = p.get("prompt_len", len(str(p.get("prompt", ""))))
        recorder.record(
            prompt_len=prompt_len,
            output_len=getattr(args, "output_len", 128),
            endpoint=args.endpoint,
            model=args.model or "",
            prompt=str(p.get("prompt", ""))[:200],  # truncate for trace
            temperature=getattr(args, "temperature", 1.0),
            max_tokens=getattr(args, "max_tokens", 128),
            stream=getattr(args, "stream", True),
        )

    # Run the actual benchmark
    bench_result, result = asyncio.run(run_benchmark(args))

    # Finalize trace
    trace = recorder.finish()
    trace_path = save_trace(trace, args.output)
    print(f"\nTrace saved to {trace_path} ({trace.num_entries} entries)")


def replay_main(argv: list[str] | None = None) -> None:
    """Entry point for ``xpyd-bench-replay`` command.

    Replays a recorded trace against a target server.
    """
    import os

    from xpyd_bench.profile import compute_delays, load_trace

    parser = argparse.ArgumentParser(
        prog="xpyd-bench-replay",
        description="Replay a recorded request timing trace against a server",
    )
    parser.add_argument(
        "--trace",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to the trace JSON file.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        required=True,
        help="Target server base URL.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Replay speed multiplier (default: 1.0). "
        "2.0 = 2x faster, 0.5 = half speed.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name from trace.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-request timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--save-result",
        type=str,
        default=None,
        metavar="PATH",
        help="Save benchmark result JSON to this path.",
    )
    args = parser.parse_args(argv)

    if args.api_key is None:
        args.api_key = os.environ.get("OPENAI_API_KEY")

    from xpyd_bench import __version__

    print(f"xpyd-bench-replay v{__version__}")

    trace = load_trace(args.trace)
    print(f"  Trace:    {args.trace} ({trace.num_entries} entries)")
    print(f"  Original: {trace.base_url} ({trace.total_duration_s:.1f}s)")
    print(f"  Target:   {args.base_url}")
    print(f"  Speed:    {args.speed}x")
    print()

    delays = compute_delays(trace, speed=args.speed)
    model = args.model or (trace.entries[0].model if trace.entries else "")

    # Build requests and replay with timing
    from xpyd_bench.bench.runner import replay_trace

    bench_result = asyncio.run(
        replay_trace(
            trace=trace,
            delays=delays,
            base_url=args.base_url,
            model=model,
            api_key=args.api_key,
            timeout=args.timeout,
            http2=getattr(args, "http2", False),
            max_connections=getattr(args, "max_connections", 100),
            max_keepalive=getattr(args, "max_keepalive", 20),
        )
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"Replay complete: {bench_result.completed}/{bench_result.num_prompts} succeeded")
    print(f"Total duration:  {bench_result.total_duration_s:.2f}s")
    print(f"Throughput:      {bench_result.request_throughput:.2f} req/s")
    if bench_result.mean_ttft_ms is not None:
        print(f"Mean TTFT:       {bench_result.mean_ttft_ms:.2f}ms")
    if bench_result.mean_e2el_ms is not None:
        print(f"Mean E2E:        {bench_result.mean_e2el_ms:.2f}ms")

    if args.save_result:
        from dataclasses import asdict

        p = Path(args.save_result)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(asdict(bench_result), f, indent=2, default=str)
        print(f"\nResult saved to {p}")


def _save_result(args: argparse.Namespace, result: dict) -> None:
    """Save benchmark result to JSON."""
    from datetime import datetime

    result_dir = Path(args.result_dir) if args.result_dir else Path(".")
    result_dir.mkdir(parents=True, exist_ok=True)

    if args.result_filename:
        filename = args.result_filename
    else:
        dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        rate = args.request_rate if args.request_rate != float("inf") else "inf"
        model = args.model or "unknown"
        filename = f"{args.backend}-{rate}qps-{model}-{dt}.json"

    filepath = result_dir / filename

    # Add metadata
    if args.metadata:
        meta = {}
        for item in args.metadata:
            if "=" in item:
                k, v = item.split("=", 1)
                meta[k] = v
        result["metadata"] = meta

    with open(filepath, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults saved to {filepath}")


def batch_main(argv: list[str] | None = None) -> None:
    """CLI entry point for batch inference benchmarking (M41)."""
    parser = argparse.ArgumentParser(
        description="Batch inference API benchmarking",
    )
    parser.add_argument("--base-url", default="http://localhost:8000",
                        help="Server base URL")
    parser.add_argument("--model", default="", help="Model name")
    parser.add_argument("--num-prompts", type=int, default=10,
                        help="Number of prompts in the batch")
    parser.add_argument("--input-len", type=int, default=256,
                        help="Input length in tokens")
    parser.add_argument("--output-len", type=int, default=128,
                        help="Max output tokens")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--poll-interval", type=float, default=2.0,
                        help="Batch status poll interval in seconds")
    parser.add_argument("--batch-timeout", type=float, default=600.0,
                        help="Maximum time to wait for batch completion")
    parser.add_argument("--batch-endpoint", default="/v1/completions",
                        help="Target endpoint for batch requests")
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="Per-request HTTP timeout")
    parser.add_argument("--dataset-path", default=None,
                        help="Path to dataset file")
    parser.add_argument("--save-result", default=None,
                        help="Save result JSON to this path")
    parser.add_argument("--disable-tqdm", action="store_true",
                        help="Disable progress output")
    parser.add_argument("--config", default=None, help="YAML config file")

    if argv is not None:
        args = parser.parse_args(argv)
    else:
        args = parser.parse_args()

    # Load YAML config overrides
    if args.config:
        from xpyd_bench.config_cmd import _load_yaml_config
        yaml_cfg = _load_yaml_config(args.config)
        for key, value in yaml_cfg.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)

    # Env var fallback for API key
    if not args.api_key:
        import os
        args.api_key = os.environ.get("OPENAI_API_KEY")

    from xpyd_bench.batch import run_batch_benchmark

    result_dict, bench_result = asyncio.run(
        run_batch_benchmark(args, args.base_url)
    )

    if args.save_result:
        p = Path(args.save_result)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(result_dict, f, indent=2, default=str)
        print(f"\nResult saved to {p}")
