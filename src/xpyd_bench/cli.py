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
        choices=["openai", "openai-chat"],
        help="Backend type (default: openai).",
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

    # Workload
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Total number of prompts/requests to send (default: 1000).",
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
        "--rich-progress",
        action="store_true",
        help="Use rich progress bar and summary table.",
    )


def _resolve_base_url(args: argparse.Namespace) -> str:
    """Resolve the base URL from --base-url or --host/--port."""
    if args.base_url:
        return args.base_url.rstrip("/")
    return f"http://{args.host}:{args.port}"


def _load_yaml_config(path: str, args: argparse.Namespace) -> argparse.Namespace:
    """Merge YAML config into args (CLI takes precedence)."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    for key, value in cfg.items():
        attr = key.replace("-", "_")
        # Only set if not explicitly provided on CLI
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

    # Merge YAML config if provided
    if args.config:
        args = _load_yaml_config(args.config, args)

    base_url = _resolve_base_url(args)

    # Resolve API key: CLI > YAML config > env var
    import os

    if args.api_key is None:
        args.api_key = os.environ.get("OPENAI_API_KEY")

    from xpyd_bench import __version__
    from xpyd_bench.bench.runner import run_benchmark

    print(f"xpyd-bench v{__version__}")
    print(f"  Base URL:       {base_url}")
    print(f"  Endpoint:       {args.endpoint}")
    print(f"  Model:          {args.model or '(auto-detect)'}")
    print(f"  Num prompts:    {args.num_prompts}")
    print(f"  Request rate:   {args.request_rate}")
    print(f"  Max concurrency:{args.max_concurrency or 'unlimited'}")
    print(f"  Input len:      {args.input_len}")
    print(f"  Output len:     {args.output_len}")
    print()

    result, bench_result = asyncio.run(run_benchmark(args, base_url))

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

    # Save results if requested
    if args.save_result:
        _save_result(args, result)


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
