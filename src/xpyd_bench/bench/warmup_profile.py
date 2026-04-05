"""Warmup Profiling (M51).

Track and analyze warmup request latencies to detect stabilization
and quantify cold-start penalty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class WarmupProfile:
    """Warmup profiling results."""

    latencies_ms: list[float] = field(default_factory=list)
    stabilization_index: int | None = None
    cold_start_penalty_ms: float = 0.0
    warmup_duration_s: float = 0.0
    steady_state_latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON output."""
        d: dict[str, Any] = {
            "latencies_ms": self.latencies_ms,
            "stabilization_index": self.stabilization_index,
            "cold_start_penalty_ms": round(self.cold_start_penalty_ms, 3),
            "warmup_duration_s": round(self.warmup_duration_s, 6),
        }
        if self.steady_state_latency_ms is not None:
            d["steady_state_latency_ms"] = round(self.steady_state_latency_ms, 3)
        return d


def detect_stabilization(
    latencies: list[float],
    window_size: int = 3,
    threshold: float = 0.10,
) -> int | None:
    """Find the first index where the rolling window stddev < threshold * mean.

    Returns the start index of the first stable window, or None if no
    stabilization detected.

    Parameters
    ----------
    latencies : list[float]
        Per-request latencies in ms.
    window_size : int
        Rolling window size (default 3).
    threshold : float
        Coefficient of variation threshold (default 0.10 = 10%).
    """
    if len(latencies) < window_size:
        return None

    arr = np.array(latencies, dtype=np.float64)
    for i in range(len(arr) - window_size + 1):
        window = arr[i : i + window_size]
        mean = window.mean()
        if mean <= 0:
            continue
        cv = window.std(ddof=0) / mean
        if cv < threshold:
            return i
    return None


def build_warmup_profile(
    latencies_ms: list[float],
    total_duration_s: float,
    window_size: int = 3,
    threshold: float = 0.10,
) -> WarmupProfile:
    """Build a WarmupProfile from measured warmup latencies.

    Parameters
    ----------
    latencies_ms : list[float]
        Per-warmup-request latency in milliseconds.
    total_duration_s : float
        Total wall-clock time for the warmup phase.
    window_size : int
        Rolling window size for stabilization detection.
    threshold : float
        CV threshold for stabilization.
    """
    profile = WarmupProfile(
        latencies_ms=list(latencies_ms),
        warmup_duration_s=total_duration_s,
    )

    if len(latencies_ms) < 2:
        return profile

    stab_idx = detect_stabilization(latencies_ms, window_size, threshold)
    profile.stabilization_index = stab_idx

    if stab_idx is not None and stab_idx < len(latencies_ms):
        steady = np.array(latencies_ms[stab_idx:], dtype=np.float64)
        profile.steady_state_latency_ms = float(steady.mean())
        profile.cold_start_penalty_ms = float(
            latencies_ms[0] - profile.steady_state_latency_ms
        )
    else:
        # No stabilization — use last value as rough steady state
        profile.steady_state_latency_ms = latencies_ms[-1]
        profile.cold_start_penalty_ms = latencies_ms[0] - latencies_ms[-1]

    return profile


def print_warmup_profile(profile: WarmupProfile) -> None:
    """Print warmup profiling summary to terminal."""
    n = len(profile.latencies_ms)
    print(f"\n  Warmup Profile ({n} requests):")
    print(f"    Duration:            {profile.warmup_duration_s:.3f}s")
    if profile.steady_state_latency_ms is not None:
        print(f"    Steady-state latency: {profile.steady_state_latency_ms:.1f}ms")
    print(f"    Cold-start penalty:  {profile.cold_start_penalty_ms:.1f}ms")
    if profile.stabilization_index is not None:
        print(f"    Stabilized at request: {profile.stabilization_index + 1}")
    else:
        print("    Stabilization: not detected")
    if n > 0:
        lats = ", ".join(f"{lat:.1f}" for lat in profile.latencies_ms[:10])
        suffix = "..." if n > 10 else ""
        print(f"    Latencies (ms): [{lats}{suffix}]")
