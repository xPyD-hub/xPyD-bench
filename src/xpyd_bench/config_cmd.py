"""Configuration dump and validation subcommands (M23)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

# Known YAML config keys (matching CLI argument names with underscores)
_KNOWN_KEYS: set[str] = {
    "adaptive_concurrency",
    "adaptive_initial_concurrency",
    "adaptive_max_concurrency",
    "adaptive_min_concurrency",
    "adaptive_target_latency",
    "api_key",
    "api_seed",
    "backend",
    "base_url",
    "best_of",
    "burstiness",
    "csv_report",
    "dataset_name",
    "dataset_path",
    "debug_log",
    "disable_tqdm",
    "dry_run",
    "echo",
    "endpoint",
    "export_requests",
    "export_requests_csv",
    "frequency_penalty",
    "header",
    "headers",
    "host",
    "html_report",
    "ignore_eos",
    "input_len",
    "json_report",
    "logit_bias",
    "logprobs",
    "markdown_report",
    "max_completion_tokens",
    "max_concurrency",
    "metadata",
    "model",
    "n",
    "no_live",
    "num_prompts",
    "output_len",
    "parallel_tool_calls",
    "port",
    "presence_penalty",
    "rate_algorithm",
    "rate_pattern",
    "request_rate",
    "response_format",
    "result_dir",
    "result_filename",
    "retries",
    "retry_delay",
    "rich_progress",
    "save_result",
    "scenario",
    "shutdown_grace_period",
    "seed",
    "service_tier",
    "sla",
    "stop",
    "stream",
    "stream_options_include_usage",
    "suffix",
    "synthetic_input_len_dist",
    "synthetic_output_len_dist",
    "temperature",
    "text_report",
    "time_series_window",
    "timeout",
    "token_bucket_burst",
    "tool_choice",
    "tools",
    "top_k",
    "top_logprobs",
    "top_p",
    "use_beam_search",
    "user",
    "metrics_ws_port",
    "tags",
    "template_vars",
    "warmup",
    "compress",
    "request_id_prefix",
}

# Deprecated keys (currently none, placeholder for future use)
_DEPRECATED_KEYS: dict[str, str] = {}


def _load_and_check_yaml(path: str) -> tuple[dict[str, Any], list[str], list[str]]:
    """Load YAML config and return (config, warnings, errors)."""
    warnings: list[str] = []
    errors: list[str] = []

    config_path = Path(path)
    if not config_path.exists():
        errors.append(f"Config file not found: {path}")
        return {}, warnings, errors

    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as e:
        errors.append(f"Invalid YAML syntax: {e}")
        return {}, warnings, errors

    if cfg is None:
        warnings.append("Config file is empty")
        return {}, warnings, errors

    if not isinstance(cfg, dict):
        errors.append(f"Config must be a YAML mapping, got {type(cfg).__name__}")
        return {}, warnings, errors

    for key in cfg:
        normalized = str(key).replace("-", "_")
        if normalized in _DEPRECATED_KEYS:
            warnings.append(
                f"Deprecated key '{key}': {_DEPRECATED_KEYS[normalized]}"
            )
        elif normalized not in _KNOWN_KEYS:
            warnings.append(f"Unknown key '{key}' (will be ignored)")

    return cfg, warnings, errors


def config_dump_main(argv: list[str] | None = None) -> None:
    """Entry point for ``xpyd-bench-config dump``."""
    parser = argparse.ArgumentParser(
        prog="xpyd-bench-config dump",
        description="Print resolved configuration (CLI defaults + YAML merged).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to YAML config file to merge with defaults.",
    )
    args = parser.parse_args(argv)

    # Get CLI defaults from bench parser
    from xpyd_bench.cli import _add_vllm_compat_args

    bench_parser = argparse.ArgumentParser()
    _add_vllm_compat_args(bench_parser)
    defaults = vars(bench_parser.parse_args([]))

    # Remove internal-only keys
    for k in ("list_scenarios",):
        defaults.pop(k, None)

    # Merge YAML if provided
    if args.config:
        cfg, warnings, errors = _load_and_check_yaml(args.config)
        if errors:
            for e in errors:
                print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        for w in warnings:
            print(f"WARNING: {w}", file=sys.stderr)
        for key, value in cfg.items():
            attr = str(key).replace("-", "_")
            defaults[attr] = value

    # Clean up for display: convert inf to string
    for k, v in defaults.items():
        if isinstance(v, float) and v == float("inf"):
            defaults[k] = "inf"

    print(yaml.dump(defaults, default_flow_style=False, sort_keys=True), end="")


def config_validate_main(argv: list[str] | None = None) -> None:
    """Entry point for ``xpyd-bench-config validate``."""
    parser = argparse.ArgumentParser(
        prog="xpyd-bench-config validate",
        description="Validate a YAML config file without running a benchmark.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to YAML config file to validate.",
    )
    args = parser.parse_args(argv)

    cfg, warnings, errors = _load_and_check_yaml(args.config)

    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    for e in errors:
        print(f"ERROR: {e}", file=sys.stderr)

    if errors:
        print("Validation FAILED.")
        sys.exit(1)
    else:
        print("Validation OK.")
        sys.exit(0)
