"""Report format generators — JSON, text, CSV, and Markdown."""

from __future__ import annotations

import csv
import io
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


# ---------------------------------------------------------------------------
# Summary column definitions (shared by CSV and Markdown)
# ---------------------------------------------------------------------------

_SUMMARY_COLUMNS = [
    ("backend", lambda r: r.backend),
    ("model", lambda r: r.model),
    ("num_prompts", lambda r: r.num_prompts),
    ("completed", lambda r: r.completed),
    ("failed", lambda r: r.failed),
    ("total_duration_s", lambda r: f"{r.total_duration_s:.2f}"),
    ("request_throughput", lambda r: f"{r.request_throughput:.2f}"),
    ("output_throughput", lambda r: f"{r.output_throughput:.2f}"),
    ("total_token_throughput", lambda r: f"{r.total_token_throughput:.2f}"),
    ("mean_ttft_ms", lambda r: f"{r.mean_ttft_ms:.2f}"),
    ("p50_ttft_ms", lambda r: f"{r.p50_ttft_ms:.2f}"),
    ("p90_ttft_ms", lambda r: f"{r.p90_ttft_ms:.2f}"),
    ("p95_ttft_ms", lambda r: f"{r.p95_ttft_ms:.2f}"),
    ("p99_ttft_ms", lambda r: f"{r.p99_ttft_ms:.2f}"),
    ("mean_tpot_ms", lambda r: f"{r.mean_tpot_ms:.2f}"),
    ("p50_tpot_ms", lambda r: f"{r.p50_tpot_ms:.2f}"),
    ("p90_tpot_ms", lambda r: f"{r.p90_tpot_ms:.2f}"),
    ("p95_tpot_ms", lambda r: f"{r.p95_tpot_ms:.2f}"),
    ("p99_tpot_ms", lambda r: f"{r.p99_tpot_ms:.2f}"),
    ("mean_itl_ms", lambda r: f"{r.mean_itl_ms:.2f}"),
    ("p50_itl_ms", lambda r: f"{r.p50_itl_ms:.2f}"),
    ("p90_itl_ms", lambda r: f"{r.p90_itl_ms:.2f}"),
    ("p95_itl_ms", lambda r: f"{r.p95_itl_ms:.2f}"),
    ("p99_itl_ms", lambda r: f"{r.p99_itl_ms:.2f}"),
    ("mean_e2el_ms", lambda r: f"{r.mean_e2el_ms:.2f}"),
    ("p50_e2el_ms", lambda r: f"{r.p50_e2el_ms:.2f}"),
    ("p90_e2el_ms", lambda r: f"{r.p90_e2el_ms:.2f}"),
    ("p95_e2el_ms", lambda r: f"{r.p95_e2el_ms:.2f}"),
    ("p99_e2el_ms", lambda r: f"{r.p99_e2el_ms:.2f}"),
]

_PER_REQUEST_COLUMNS = [
    "prompt_tokens",
    "completion_tokens",
    "ttft_ms",
    "tpot_ms",
    "latency_ms",
    "success",
    "error",
    "retries",
]


def export_csv_report(result: BenchmarkResult, path: str | Path) -> Path:
    """Export summary metrics as a single-row CSV with header.

    Returns the written path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    headers = [col[0] for col in _SUMMARY_COLUMNS]
    values = [str(col[1](result)) for col in _SUMMARY_COLUMNS]
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(headers)
    writer.writerow(values)
    p.write_text(buf.getvalue())
    return p


def export_markdown_report(result: BenchmarkResult, path: str | Path) -> Path:
    """Export summary metrics as a Markdown table.

    Returns the written path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    headers = [col[0] for col in _SUMMARY_COLUMNS]
    values = [str(col[1](result)) for col in _SUMMARY_COLUMNS]
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    data_line = "| " + " | ".join(values) + " |"
    content = f"{header_line}\n{separator}\n{data_line}\n"
    p.write_text(content)
    return p


def export_per_request_csv(result: BenchmarkResult, path: str | Path) -> Path:
    """Export per-request detailed metrics as CSV.

    Returns the written path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(_PER_REQUEST_COLUMNS)
    for r in result.requests:
        writer.writerow([
            r.prompt_tokens,
            r.completion_tokens,
            r.ttft_ms if r.ttft_ms is not None else "",
            r.tpot_ms if r.tpot_ms is not None else "",
            r.latency_ms,
            r.success,
            r.error or "",
            r.retries,
        ])
    p.write_text(buf.getvalue())
    return p
