"""Benchmark configuration fingerprinting (M72).

Generates a deterministic SHA-256 hash from normalized benchmark configuration
so that runs with identical settings can be grouped and compared over time.
"""

from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from typing import Any

# Keys that do NOT affect benchmark behavior and should be excluded from fingerprint
_EXCLUDED_KEYS: frozenset[str] = frozenset({
    # Output / reporting paths (don't affect benchmark execution)
    "save_result",
    "result_dir",
    "result_filename",
    "json_report",
    "csv_report",
    "markdown_report",
    "html_report",
    "export_requests",
    "export_requests_csv",
    "debug_log",
    "junit_xml",
    "prometheus_export",
    "text_report",
    # Display options
    "disable_tqdm",
    "no_live",
    "verbose",
    "quiet",
    "verbosity",
    "list_scenarios",
    "list_backends",
    "heatmap",
    # Metadata / annotations (user-provided labels, not config)
    "note",
    "tag",
    "tags",
    # Webhook / notification
    "webhook_url",
    "webhook_secret",
    # Telemetry export
    "otlp_endpoint",
    "metrics_ws_port",
    # Scheduling
    "on_complete",
    # Dry run flag
    "dry_run",
    # Archive
    "archive",
    # Sweep output
    "sweep_output",
    # Config file path itself
    "config",
})


def _normalize_value(v: Any) -> Any:
    """Normalize a config value for consistent hashing."""
    if v is None:
        return None
    if isinstance(v, float) and v == float("inf"):
        return "inf"
    if isinstance(v, (list, tuple)):
        return [_normalize_value(item) for item in v]
    if isinstance(v, dict):
        return {str(k): _normalize_value(val) for k, val in sorted(v.items())}
    return v


def compute_fingerprint(args: Namespace | dict) -> str:
    """Compute a deterministic SHA-256 fingerprint from benchmark configuration.

    Parameters
    ----------
    args:
        CLI Namespace or dict of configuration values.

    Returns
    -------
    str:
        Hex-encoded SHA-256 hash string.
    """
    if isinstance(args, Namespace):
        cfg = vars(args)
    else:
        cfg = dict(args)

    # Filter out excluded keys and None values (defaults)
    filtered: dict[str, Any] = {}
    for key, value in cfg.items():
        if key.startswith("_"):
            continue
        if key in _EXCLUDED_KEYS:
            continue
        normalized = _normalize_value(value)
        if normalized is not None:
            filtered[key] = normalized

    # Sort keys for deterministic JSON serialization
    canonical = json.dumps(filtered, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
