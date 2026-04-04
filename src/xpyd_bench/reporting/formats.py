"""Report format generators — JSON and human-readable text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from xpyd_bench.bench.models import BenchmarkResult
from xpyd_bench.reporting.metrics import compute_time_series


def _request_to_dict(r: Any) -> dict:
    """Convert a RequestResult to a plain dict."""
    return {
        "prompt_tokens": r.prompt_tokens,
        "completion_tokens": r.completion_tokens,
        "ttft_ms": r.ttft_ms,
        "tpot_ms": r.tpot_ms,
        "itl_ms": r.itl_ms,
        "latency_ms": r.latency_ms,
        "success": r.success,
        "error": r.error,
    }


def export_per_request(result: BenchmarkResult, path: str | Path) -> Path:
    """Export per-request detailed metrics to a JSON file.

    Returns the written path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = [_request_to_dict(r) for r in result.requests]
    with open(p, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return p


def export_json_report(
    result: BenchmarkResult,
    summary: dict,
    path: str | Path,
    time_series_window: float = 1.0,
) -> Path:
    """Export a comprehensive JSON report including summary + time series.

    Returns the written path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ts = compute_time_series(result, window_s=time_series_window)
    report = {
        "summary": summary,
        "time_series": ts,
    }
    with open(p, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return p


def format_text_report(result: BenchmarkResult) -> str:
    """Generate a human-readable text report string."""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("  xPyD-bench — Benchmark Report")
    lines.append("=" * 70)
    lines.append(f"  Backend:            {result.backend}")
    lines.append(f"  Base URL:           {result.base_url}")
    lines.append(f"  Endpoint:           {result.endpoint}")
    lines.append(f"  Model:              {result.model}")
    lines.append(f"  Num prompts:        {result.num_prompts}")
    lines.append(f"  Request rate:       {result.request_rate}")
    lines.append(f"  Max concurrency:    {result.max_concurrency or 'unlimited'}")
    lines.append("")
    lines.append(f"  Completed:          {result.completed}")
    lines.append(f"  Failed:             {result.failed}")
    lines.append(f"  Total duration:     {result.total_duration_s:.2f} s")
    lines.append(f"  Request throughput: {result.request_throughput:.2f} req/s")
    lines.append(f"  Output throughput:  {result.output_throughput:.2f} tok/s")
    lines.append(f"  Total tok thpt:     {result.total_token_throughput:.2f} tok/s")
    lines.append("")
    lines.append("  Latency Percentiles (ms)")
    lines.append("  " + "-" * 66)
    header = f"  {'Metric':6s} {'Mean':>8s} {'P50':>8s} {'P90':>8s} {'P95':>8s} {'P99':>8s}"
    lines.append(header)
    lines.append("  " + "-" * 66)
    for label, prefix in [("TTFT", "ttft"), ("TPOT", "tpot"), ("ITL", "itl"), ("E2EL", "e2el")]:
        mean = getattr(result, f"mean_{prefix}_ms")
        p50 = getattr(result, f"p50_{prefix}_ms")
        p90 = getattr(result, f"p90_{prefix}_ms")
        p95 = getattr(result, f"p95_{prefix}_ms")
        p99 = getattr(result, f"p99_{prefix}_ms")
        lines.append(
            f"  {label:6s} {mean:8.2f} {p50:8.2f} {p90:8.2f} {p95:8.2f} {p99:8.2f}"
        )
    lines.append("  " + "-" * 66)
    lines.append("=" * 70)
    return "\n".join(lines)
