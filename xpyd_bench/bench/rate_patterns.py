"""Flexible request rate pattern generators.

Supports four patterns configured via YAML ``rate_pattern`` section:

* **constant** — fixed rate per configurable interval
* **ramp** — linear interpolation between rate stages
* **burst** — periodic bursts separated by idle time
* **custom** — explicit per-second rate schedule

Each generator returns a list of inter-arrival times (seconds) for *num*
requests, deterministic given *seed*.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def generate_pattern_intervals(
    num: int,
    pattern_cfg: dict[str, Any],
    seed: int = 0,
) -> list[float]:
    """Dispatch to the correct pattern generator.

    Parameters
    ----------
    num:
        Total number of requests.
    pattern_cfg:
        Parsed YAML ``rate_pattern`` dict.  Must contain a ``type`` key.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    list[float]
        Inter-arrival intervals in seconds, length == *num*.
    """
    ptype = pattern_cfg.get("type", "constant")
    generators = {
        "constant": _constant_intervals,
        "ramp": _ramp_intervals,
        "burst": _burst_intervals,
        "custom": _custom_intervals,
    }
    gen = generators.get(ptype)
    if gen is None:
        raise ValueError(
            f"Unknown rate_pattern type '{ptype}'. "
            f"Supported: {sorted(generators.keys())}"
        )
    return gen(num, pattern_cfg, seed)


# ---------------------------------------------------------------------------
# Pattern implementations
# ---------------------------------------------------------------------------


def _constant_intervals(
    num: int,
    cfg: dict[str, Any],
    seed: int,
) -> list[float]:
    """Constant rate: *rate* requests every *interval* seconds.

    Config keys:
        rate (float): requests per interval (required)
        interval (float): seconds per interval, default 1.0
    """
    rate = float(cfg["rate"])
    interval = float(cfg.get("interval", 1.0))
    if rate <= 0:
        raise ValueError("rate must be positive")
    mean_gap = interval / rate
    rng = np.random.RandomState(seed)
    # Exponential around the mean gap for realistic jitter
    return rng.exponential(mean_gap, size=num).tolist()


def _ramp_intervals(
    num: int,
    cfg: dict[str, Any],
    seed: int,
) -> list[float]:
    """Linear ramp through a sequence of (rate, duration) stages.

    Config keys:
        stages (list[dict]): each with ``rate`` (req/s) and ``duration`` (s).

    Requests are distributed proportionally across stages.  Within each
    stage the instantaneous rate is linearly interpolated between the
    start and end rate.
    """
    stages: list[dict[str, Any]] = cfg["stages"]
    if not stages:
        raise ValueError("ramp pattern requires at least one stage")

    rng = np.random.RandomState(seed)

    # Compute expected request count per stage to allocate proportionally.
    total_expected = 0.0
    stage_expected: list[float] = []
    for s in stages:
        rate = float(s["rate"])
        duration = float(s["duration"])
        total_expected += rate * duration
        stage_expected.append(rate * duration)

    if total_expected <= 0:
        return [0.0] * num

    intervals: list[float] = []
    remaining = num

    for idx, s in enumerate(stages):
        if remaining <= 0:
            break
        rate = float(s["rate"])
        duration = float(s["duration"])
        # Allocate proportional share of requests
        if idx == len(stages) - 1:
            n_stage = remaining
        else:
            n_stage = max(1, round(num * stage_expected[idx] / total_expected))
            n_stage = min(n_stage, remaining)

        if rate <= 0 or duration <= 0:
            intervals.extend([0.0] * n_stage)
        else:
            mean_gap = duration / n_stage
            gaps = rng.exponential(mean_gap, size=n_stage).tolist()
            intervals.extend(gaps)
        remaining -= n_stage

    # Pad if rounding left us short
    while len(intervals) < num:
        intervals.append(0.0)

    return intervals[:num]


def _burst_intervals(
    num: int,
    cfg: dict[str, Any],
    seed: int,
) -> list[float]:
    """Periodic bursts of requests.

    Config keys:
        burst_size (int): requests per burst (required)
        burst_interval (float): seconds between burst starts (required)

    Within a burst, requests fire with zero delay.  Between bursts, the
    first request of the next burst waits *burst_interval* seconds.
    """
    burst_size = int(cfg["burst_size"])
    burst_interval = float(cfg["burst_interval"])
    if burst_size <= 0 or burst_interval < 0:
        raise ValueError("burst_size must be >0 and burst_interval >=0")

    intervals: list[float] = []
    for i in range(num):
        pos_in_burst = i % burst_size
        if pos_in_burst == 0 and i > 0:
            intervals.append(burst_interval)
        else:
            intervals.append(0.0)
    return intervals


def _custom_intervals(
    num: int,
    cfg: dict[str, Any],
    seed: int,
) -> list[float]:
    """Explicit per-second rate schedule.

    Config keys:
        schedule (list[float]): rate (req/s) for each second.

    The schedule repeats cyclically if *num* exceeds the total requests
    implied by the schedule.
    """
    schedule: list[float] = [float(r) for r in cfg["schedule"]]
    if not schedule:
        raise ValueError("custom pattern requires non-empty schedule")

    rng = np.random.RandomState(seed)
    intervals: list[float] = []
    idx = 0
    idle_carry = 0.0  # accumulated idle time to prepend to next request

    while len(intervals) < num:
        rate = schedule[idx % len(schedule)]
        idx += 1
        if rate <= 0:
            # Idle second — skip without scheduling any request.
            # Accumulate the idle time onto the next real request's delay.
            idle_carry += 1.0
            continue
        # Generate ~rate requests within this 1-second window
        n_in_sec = max(1, round(rate))
        n_in_sec = min(n_in_sec, num - len(intervals))
        gap = 1.0 / rate
        gaps = rng.exponential(gap, size=n_in_sec).tolist()
        # Add accumulated idle time to the first request after idle period
        if idle_carry > 0 and gaps:
            gaps[0] += idle_carry
            idle_carry = 0.0
        intervals.extend(gaps)

    return intervals[:num]
