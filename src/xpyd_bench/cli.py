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
        "--debug-log",
        type=str,
        default=None,
        metavar="PATH",
        help="Write per-request debug logs (JSONL) for diagnosing failures.",
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
    print(f"  Backend:         {args.backend}")
    print(f"  Model:           {args.model or '(auto-detect)'}")
    print(f"  Num prompts:     {args.num_prompts}")
    print(f"  Request rate:    {args.request_rate}")
    print(f"  Max concurrency: {args.max_concurrency or 'unlimited'}")
    print(f"  Input len:       {args.input_len}")
    print(f"  Output len:      {args.output_len}")
    print(f"  Seed:            {args.seed}")
    print(f"  Rate algorithm:  {args.rate_algorithm}")

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
        )
        source = args.dataset_path or f"synthetic ({args.dataset_name})"
        validate_and_report(entries, source)
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

    # Resolve custom headers: YAML + CLI (CLI wins)
    args.custom_headers = _resolve_custom_headers(args)

    from xpyd_bench import __version__
    from xpyd_bench.bench.runner import run_benchmark

    # Dry run: validate config and dataset, print plan, exit
    if getattr(args, "dry_run", False):
        _dry_run(args, base_url)
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

    # Debug log notification (M22)
    if getattr(args, "debug_log", None):
        print(f"\nDebug log saved to {args.debug_log}")

    # Save results if requested
    if args.save_result:
        _save_result(args, result)

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
        args = _load_yaml_config(args.config, args)

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
        args = _load_yaml_config(args.config, args)

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
