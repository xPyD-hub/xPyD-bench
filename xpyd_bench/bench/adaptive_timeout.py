"""Adaptive timeout auto-tuning based on observed latencies (M86)."""

from __future__ import annotations

import threading
from collections import deque

import numpy as np

DEFAULT_MULTIPLIER: float = 3.0
ROLLING_WINDOW_SIZE: int = 100
MIN_SAMPLES_FOR_ADAPTATION: int = 5


class AdaptiveTimeout:
    """Track request latencies and compute adaptive per-request timeouts.

    The effective timeout is ``rolling_p99 * multiplier``, clamped to
    ``[min_timeout, initial_timeout]``.  Until enough samples are collected
    the initial (static) timeout is returned.

    Parameters
    ----------
    initial_timeout:
        The starting timeout in seconds (from ``--timeout``).
    multiplier:
        Safety margin multiplier applied to rolling P99.
    window_size:
        Maximum number of recent latencies to consider.
    """

    def __init__(
        self,
        initial_timeout: float = 300.0,
        multiplier: float = DEFAULT_MULTIPLIER,
        window_size: int = ROLLING_WINDOW_SIZE,
    ) -> None:
        if initial_timeout <= 0:
            raise ValueError(f"initial_timeout must be positive, got {initial_timeout}")
        if multiplier <= 0:
            raise ValueError(f"multiplier must be positive, got {multiplier}")
        self._initial = initial_timeout
        self._multiplier = multiplier
        self._window_size = window_size
        self._latencies: deque[float] = deque(maxlen=window_size)
        self._lock = threading.Lock()
        # Minimum timeout to avoid overly aggressive shrinking
        self._min_timeout = 1.0

    @property
    def initial_timeout(self) -> float:
        return self._initial

    @property
    def multiplier(self) -> float:
        return self._multiplier

    def record(self, latency_s: float) -> None:
        """Record a completed request latency (seconds)."""
        with self._lock:
            self._latencies.append(latency_s)

    def get_timeout(self) -> float:
        """Return the current adaptive timeout in seconds."""
        with self._lock:
            if len(self._latencies) < MIN_SAMPLES_FOR_ADAPTATION:
                return self._initial
            arr = np.array(self._latencies, dtype=np.float64)
        p99 = float(np.percentile(arr, 99))
        adaptive = p99 * self._multiplier
        # Clamp between min and initial
        return max(self._min_timeout, min(adaptive, self._initial))

    def sample_count(self) -> int:
        """Number of latency samples recorded so far."""
        with self._lock:
            return len(self._latencies)
