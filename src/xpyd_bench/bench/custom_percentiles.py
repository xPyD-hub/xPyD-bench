"""Custom percentile configuration (M73)."""

from __future__ import annotations

from argparse import Namespace

import numpy as np

from xpyd_bench.bench.models import BenchmarkResult

DEFAULT_PERCENTILES: list[float] = [50, 90, 95, 99]


def parse_percentiles(raw: str | list | None) -> list[float]:
    """Parse percentile spec from CLI string or YAML list.

    Accepts comma-separated string (``"50,90,99.9"``) or list of
    numbers.  Returns sorted, deduplicated list of floats.
    """
    if raw is None:
        return list(DEFAULT_PERCENTILES)
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
    elif isinstance(raw, (list, tuple)):
        parts = [str(p) for p in raw]
    else:
        return list(DEFAULT_PERCENTILES)

    result: list[float] = []
    for p in parts:
        val = float(p)
        if not (0 < val < 100):
            raise ValueError(f"Percentile must be between 0 and 100 exclusive, got {val}")
        result.append(val)

    return sorted(set(result))


def compute_custom_percentiles(
    result: BenchmarkResult, args: Namespace
) -> None:
    """Compute custom percentiles and store in ``result.custom_percentiles``."""
    raw = getattr(args, "percentiles", None)
    pcts = parse_percentiles(raw)

    successful = [r for r in result.requests if r.success]
    if not successful:
        return

    metrics: dict[str, list[float]] = {}

    e2els = [r.latency_ms for r in successful]
    if e2els:
        metrics["e2el_ms"] = e2els

    ttfts = [r.ttft_ms for r in successful if r.ttft_ms is not None]
    if ttfts:
        metrics["ttft_ms"] = ttfts

    tpots = [r.tpot_ms for r in successful if r.tpot_ms is not None]
    if tpots:
        metrics["tpot_ms"] = tpots

    all_itls = [itl for r in successful for itl in r.itl_ms]
    if all_itls:
        metrics["itl_ms"] = all_itls

    custom: dict[str, dict[str, float]] = {}
    for prefix, values in metrics.items():
        arr = np.array(values)
        entry: dict[str, float] = {}
        for p in pcts:
            key = f"p{p:g}"
            entry[key] = float(np.percentile(arr, p))
        custom[prefix] = entry

    result.custom_percentiles = custom
