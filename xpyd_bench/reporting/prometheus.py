"""Prometheus / OpenMetrics text exposition format export."""

from __future__ import annotations

import math
from pathlib import Path

from xpyd_bench.bench.models import BenchmarkResult

# Default histogram buckets (seconds) — Prometheus convention
_DEFAULT_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    7.5,
    10.0,
    float("inf"),
)


def _histogram_lines(
    name: str,
    help_text: str,
    values: list[float],
    labels: dict[str, str],
    buckets: tuple[float, ...] = _DEFAULT_BUCKETS,
) -> list[str]:
    """Generate Prometheus histogram lines for a metric."""
    lines: list[str] = []
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} histogram")

    label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    if label_str:
        label_str = f"{{{label_str}}}"

    # Compute cumulative bucket counts
    total = len(values)
    total_sum = sum(values)

    for bound in buckets:
        if math.isinf(bound):
            le_label = "+Inf"
            count = total
        else:
            le_label = f"{bound}"
            count = sum(1 for v in values if v <= bound)

        if label_str:
            # Insert le into existing labels
            inner = label_str[1:-1]
            lines.append(f'{name}_bucket{{{inner},le="{le_label}"}} {count}')
        else:
            lines.append(f'{name}_bucket{{le="{le_label}"}} {count}')

    suffix = label_str
    lines.append(f"{name}_count{suffix} {total}")
    lines.append(f"{name}_sum{suffix} {total_sum:.6f}")
    return lines


def _gauge_lines(
    name: str, help_text: str, value: float, labels: dict[str, str]
) -> list[str]:
    """Generate Prometheus gauge lines."""
    lines: list[str] = []
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} gauge")
    label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    suffix = f"{{{label_str}}}" if label_str else ""
    lines.append(f"{name}{suffix} {value:.6f}")
    return lines


def _counter_lines(
    name: str, help_text: str, value: int, labels: dict[str, str]
) -> list[str]:
    """Generate Prometheus counter lines."""
    lines: list[str] = []
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} counter")
    label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    suffix = f"{{{label_str}}}" if label_str else ""
    lines.append(f"{name}{suffix} {value}")
    return lines


def export_prometheus(
    result: BenchmarkResult,
    path: str | Path,
    scenario: str | None = None,
) -> Path:
    """Export benchmark results in Prometheus text exposition format.

    Args:
        result: Completed benchmark result.
        path: Output file path.
        scenario: Optional scenario name label.

    Returns:
        Path to the written file.
    """
    path = Path(path)
    labels: dict[str, str] = {}
    if result.model:
        labels["model"] = result.model
    if result.endpoint:
        labels["endpoint"] = result.endpoint
    if scenario:
        labels["scenario"] = scenario

    all_lines: list[str] = []

    # TTFT histogram (convert ms -> seconds)
    ttft_values = [
        r.ttft_ms / 1000.0 for r in result.requests if r.ttft_ms is not None
    ]
    if ttft_values:
        all_lines.extend(
            _histogram_lines(
                "xpyd_bench_ttft_seconds",
                "Time to first token in seconds.",
                ttft_values,
                labels,
            )
        )
        all_lines.append("")

    # TPOT histogram (convert ms -> seconds)
    tpot_values = [
        r.tpot_ms / 1000.0 for r in result.requests if r.tpot_ms is not None
    ]
    if tpot_values:
        all_lines.extend(
            _histogram_lines(
                "xpyd_bench_tpot_seconds",
                "Time per output token in seconds.",
                tpot_values,
                labels,
            )
        )
        all_lines.append("")

    # Request latency histogram (convert ms -> seconds)
    latency_values = [
        r.latency_ms / 1000.0 for r in result.requests if r.success
    ]
    if latency_values:
        all_lines.extend(
            _histogram_lines(
                "xpyd_bench_request_latency_seconds",
                "End-to-end request latency in seconds.",
                latency_values,
                labels,
            )
        )
        all_lines.append("")

    # Throughput gauge
    all_lines.extend(
        _gauge_lines(
            "xpyd_bench_throughput_tokens_per_second",
            "Output token throughput.",
            result.output_throughput,
            labels,
        )
    )
    all_lines.append("")

    # Requests total counter
    all_lines.extend(
        _counter_lines(
            "xpyd_bench_requests_total",
            "Total benchmark requests.",
            result.completed + result.failed,
            labels,
        )
    )
    all_lines.append("")

    # Errors total counter
    all_lines.extend(
        _counter_lines(
            "xpyd_bench_errors_total",
            "Total failed requests.",
            result.failed,
            labels,
        )
    )
    all_lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(all_lines))
    return path
