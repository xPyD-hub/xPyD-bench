"""Request latency heatmap data export (M97).

Exports time-bucketed latency histogram data in a JSON format suitable for
heatmap visualization tools. Each time bucket contains a histogram of latency
counts across configurable bin edges.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Default latency bin edges in milliseconds
DEFAULT_BIN_EDGES_MS: list[float] = [0, 50, 100, 200, 500, 1000]

# Default time bucket width in seconds
DEFAULT_BUCKET_WIDTH_S: float = 1.0


@dataclass
class HeatmapBucket:
    """A single time bucket with latency histogram."""

    time_start: float  # seconds from benchmark start
    time_end: float
    counts: list[int] = field(default_factory=list)  # count per latency bin
    total: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_start": round(self.time_start, 3),
            "time_end": round(self.time_end, 3),
            "counts": self.counts,
            "total": self.total,
        }


@dataclass
class HeatmapExportData:
    """Complete heatmap export data."""

    bucket_width_s: float = DEFAULT_BUCKET_WIDTH_S
    bin_edges_ms: list[float] = field(default_factory=lambda: list(DEFAULT_BIN_EDGES_MS))
    buckets: list[HeatmapBucket] = field(default_factory=list)
    total_requests: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "bucket_width_s": self.bucket_width_s,
            "bin_edges_ms": self.bin_edges_ms,
            "buckets": [b.to_dict() for b in self.buckets],
            "total_requests": self.total_requests,
        }


def _assign_bin(latency_ms: float, bin_edges: list[float]) -> int:
    """Return the bin index for a latency value.

    Bins are defined by edges: [edge0, edge1), [edge1, edge2), ..., [edgeN, +inf).
    The last bin captures everything >= the last edge.

    Returns an index in range [0, len(bin_edges)].
    """
    for i in range(len(bin_edges) - 1, -1, -1):
        if latency_ms >= bin_edges[i]:
            return i
    return 0


def compute_heatmap_export(
    requests: list,
    bench_start_time: float,
    *,
    bucket_width_s: float = DEFAULT_BUCKET_WIDTH_S,
    bin_edges_ms: list[float] | None = None,
) -> HeatmapExportData:
    """Compute heatmap export data from benchmark request results.

    Parameters
    ----------
    requests:
        List of RequestResult objects.
    bench_start_time:
        The perf_counter timestamp when the benchmark started.
    bucket_width_s:
        Width of each time bucket in seconds.
    bin_edges_ms:
        Latency bin edges in milliseconds. If None, uses DEFAULT_BIN_EDGES_MS.

    Returns
    -------
    HeatmapExportData ready for JSON serialization.
    """
    if bin_edges_ms is None:
        bin_edges_ms = list(DEFAULT_BIN_EDGES_MS)

    if bucket_width_s <= 0:
        bucket_width_s = DEFAULT_BUCKET_WIDTH_S

    # Number of latency bins = len(bin_edges_ms) + 1
    # (one bin below first edge is impossible since first edge is 0,
    #  but we keep len+1 for the overflow bin above last edge)
    num_bins = len(bin_edges_ms) + 1

    successful = [
        r for r in requests
        if r.success and r.start_time is not None
    ]

    if not successful:
        return HeatmapExportData(
            bucket_width_s=bucket_width_s,
            bin_edges_ms=bin_edges_ms,
        )

    # Determine time range
    elapsed_times = [r.start_time - bench_start_time for r in successful]
    max_elapsed = max(elapsed_times)

    # Number of time buckets
    import math
    num_time_buckets = max(1, math.ceil(max_elapsed / bucket_width_s))

    # Initialize buckets
    buckets: list[HeatmapBucket] = []
    for i in range(num_time_buckets):
        t_start = i * bucket_width_s
        t_end = (i + 1) * bucket_width_s
        buckets.append(HeatmapBucket(
            time_start=t_start,
            time_end=t_end,
            counts=[0] * num_bins,
            total=0,
        ))

    # Populate buckets
    for req in successful:
        elapsed = req.start_time - bench_start_time
        time_idx = min(int(elapsed / bucket_width_s), num_time_buckets - 1)
        bin_idx = _assign_bin(req.latency_ms, bin_edges_ms)
        buckets[time_idx].counts[bin_idx] += 1
        buckets[time_idx].total += 1

    return HeatmapExportData(
        bucket_width_s=bucket_width_s,
        bin_edges_ms=bin_edges_ms,
        buckets=buckets,
        total_requests=len(successful),
    )


def export_heatmap_json(
    data: HeatmapExportData,
    path: str | Path,
) -> Path:
    """Write heatmap data to a JSON file.

    Parameters
    ----------
    data:
        HeatmapExportData to serialize.
    path:
        Output file path.

    Returns
    -------
    Resolved Path of the written file.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data.to_dict(), indent=2) + "\n", encoding="utf-8")
    return p.resolve()


def parse_bin_edges(raw: str) -> list[float]:
    """Parse a comma-separated string of bin edges into a sorted float list.

    Parameters
    ----------
    raw:
        Comma-separated string, e.g. "0,50,100,200,500,1000".

    Returns
    -------
    Sorted list of float bin edges.

    Raises
    ------
    ValueError:
        If any value is not a valid number or list is empty.
    """
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("bin edges string is empty")
    edges = sorted(float(p) for p in parts)
    return edges
