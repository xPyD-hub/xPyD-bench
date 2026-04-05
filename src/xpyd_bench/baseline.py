"""Benchmark Baseline Registry (M82).

Save, list, show, delete named benchmark baselines and auto-compare
against them after a benchmark run.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DEFAULT_DIR = os.path.expanduser("~/.xpyd-bench/baselines")


def _registry_dir(baseline_dir: str | None = None) -> Path:
    """Return the baseline registry directory, creating if needed."""
    d = Path(baseline_dir) if baseline_dir else Path(_DEFAULT_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _index_path(baseline_dir: str | None = None) -> Path:
    return _registry_dir(baseline_dir) / "index.json"


def _load_index(baseline_dir: str | None = None) -> dict[str, Any]:
    p = _index_path(baseline_dir)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _save_index(index: dict[str, Any], baseline_dir: str | None = None) -> None:
    p = _index_path(baseline_dir)
    p.write_text(json.dumps(index, indent=2) + "\n")


def save_baseline(
    name: str,
    result_path: str,
    baseline_dir: str | None = None,
) -> dict[str, Any]:
    """Save a benchmark result as a named baseline.

    Returns the baseline entry dict.
    """
    src = Path(result_path)
    if not src.exists():
        raise FileNotFoundError(f"Result file not found: {result_path}")

    data = json.loads(src.read_text())
    reg = _registry_dir(baseline_dir)
    dest = reg / f"{name}.json"
    dest.write_text(json.dumps(data, indent=2) + "\n")

    index = _load_index(baseline_dir)
    entry: dict[str, Any] = {
        "name": name,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "model": data.get("model", ""),
        "file": str(dest),
    }
    # Extract summary metrics
    for key in ("mean_ttft_ms", "mean_tpot_ms", "output_throughput"):
        if key in data:
            entry[key] = data[key]

    index[name] = entry
    _save_index(index, baseline_dir)
    return entry


def list_baselines(baseline_dir: str | None = None) -> list[dict[str, Any]]:
    """List all saved baselines."""
    index = _load_index(baseline_dir)
    return list(index.values())


def show_baseline(name: str, baseline_dir: str | None = None) -> dict[str, Any]:
    """Load and return a named baseline's full data."""
    index = _load_index(baseline_dir)
    if name not in index:
        raise KeyError(f"Baseline not found: {name}")
    fpath = Path(index[name]["file"])
    if not fpath.exists():
        raise FileNotFoundError(f"Baseline file missing: {fpath}")
    return json.loads(fpath.read_text())


def delete_baseline(name: str, baseline_dir: str | None = None) -> None:
    """Delete a named baseline."""
    index = _load_index(baseline_dir)
    if name not in index:
        raise KeyError(f"Baseline not found: {name}")
    fpath = Path(index[name]["file"])
    if fpath.exists():
        fpath.unlink()
    del index[name]
    _save_index(index, baseline_dir)


def compare_against_baseline(
    name: str,
    result_data: dict[str, Any],
    baseline_dir: str | None = None,
    threshold_pct: float = 5.0,
) -> dict[str, Any]:
    """Compare benchmark result against a named baseline.

    Returns a comparison dict with metrics, deltas, and regression status.
    """
    baseline_data = show_baseline(name, baseline_dir)

    metrics = [
        "mean_ttft_ms",
        "mean_tpot_ms",
        "mean_e2e_latency_ms",
        "output_throughput",
        "p99_ttft_ms",
        "p99_tpot_ms",
        "p99_e2e_latency_ms",
    ]

    comparisons: list[dict[str, Any]] = []
    regression_detected = False

    for m in metrics:
        bv = baseline_data.get(m)
        cv = result_data.get(m)
        if bv is None or cv is None:
            continue

        if bv == 0:
            delta_pct = 0.0
        else:
            delta_pct = round(((cv - bv) / abs(bv)) * 100, 1)

        # For throughput, higher is better; for latency, lower is better
        higher_is_better = "throughput" in m
        if higher_is_better:
            regressed = delta_pct < -threshold_pct
        else:
            regressed = delta_pct > threshold_pct

        if regressed:
            regression_detected = True

        comparisons.append({
            "metric": m,
            "baseline": round(bv, 2),
            "current": round(cv, 2),
            "delta_pct": delta_pct,
            "regressed": regressed,
        })

    return {
        "baseline_name": name,
        "comparisons": comparisons,
        "regression_detected": regression_detected,
        "threshold_pct": threshold_pct,
    }
