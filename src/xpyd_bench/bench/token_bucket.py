"""Token bucket rate limiter for request scheduling.

Provides a smoother, more accurate rate limiting mechanism compared to
sleep-based Poisson scheduling. Tokens refill at a constant rate, and
each request consumes one token.
"""

from __future__ import annotations

import asyncio
import time


class TokenBucket:
    """Async token bucket rate limiter.

    Parameters
    ----------
    rate:
        Tokens added per second.
    burst:
        Maximum tokens that can accumulate (bucket capacity).
        Defaults to *rate* (allows 1 second of burst).
    """

    def __init__(self, rate: float, burst: float | None = None) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate = rate
        self.burst = burst if burst is not None else rate
        if self.burst <= 0:
            raise ValueError("burst must be positive")
        self._tokens = self.burst  # start full
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> float:
        """Wait until a token is available.

        Returns the time spent waiting (seconds).
        """
        async with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return 0.0
            # Calculate wait time for 1 token
            deficit = 1.0 - self._tokens
            wait = deficit / self.rate

        await asyncio.sleep(wait)

        async with self._lock:
            self._refill()
            self._tokens = max(0.0, self._tokens - 1.0)
            return wait

    def _refill(self) -> None:
        """Add tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate)


class AdaptiveConcurrencyLimiter:
    """Dynamically adjust concurrency based on observed latency.

    Increases concurrency when latency is below target, decreases when above.

    Parameters
    ----------
    initial:
        Starting concurrency limit.
    min_concurrency:
        Floor for concurrency.
    max_concurrency:
        Ceiling for concurrency.
    target_latency_ms:
        Target latency in milliseconds. Concurrency is reduced when
        observed latency exceeds this.
    increase_step:
        How many slots to add when latency is good.
    decrease_factor:
        Multiplicative factor when latency is bad (e.g. 0.8 = reduce by 20%).
    window_size:
        Number of recent latencies to consider for the moving average.
    """

    def __init__(
        self,
        initial: int = 16,
        min_concurrency: int = 1,
        max_concurrency: int = 256,
        target_latency_ms: float = 500.0,
        increase_step: int = 1,
        decrease_factor: float = 0.8,
        window_size: int = 20,
    ) -> None:
        self._limit = initial
        self._min = min_concurrency
        self._max = max_concurrency
        self._target_ms = target_latency_ms
        self._increase_step = increase_step
        self._decrease_factor = decrease_factor
        self._window_size = window_size
        self._latencies: list[float] = []
        self._lock = asyncio.Lock()
        self._in_flight = 0
        self._condition = asyncio.Condition(self._lock)

    @property
    def limit(self) -> int:
        """Current concurrency limit."""
        return self._limit

    async def acquire(self) -> None:
        """Acquire a concurrency slot."""
        async with self._condition:
            while self._in_flight >= self._limit:
                await self._condition.wait()
            self._in_flight += 1

    def release(self) -> None:
        """Release a concurrency slot.

        Schedules waiters to be notified so new requests can proceed.
        """
        # Use the running loop to schedule the async notification.
        loop = asyncio.get_running_loop()
        loop.create_task(self._async_release())

    async def _async_release(self) -> None:
        """Decrement in-flight counter and wake a waiter."""
        async with self._condition:
            self._in_flight -= 1
            self._condition.notify()

    async def record_latency(self, latency_ms: float) -> None:
        """Record a request latency and adjust concurrency."""
        async with self._lock:
            self._latencies.append(latency_ms)
            if len(self._latencies) > self._window_size:
                self._latencies = self._latencies[-self._window_size:]

            if len(self._latencies) < 3:
                return  # Need minimum samples

            avg = sum(self._latencies) / len(self._latencies)
            old_limit = self._limit

            if avg <= self._target_ms:
                # Latency is good → increase concurrency
                new_limit = min(self._max, self._limit + self._increase_step)
            else:
                # Latency is bad → decrease concurrency
                new_limit = max(
                    self._min, int(self._limit * self._decrease_factor)
                )

            if new_limit != old_limit:
                self._limit = new_limit
                # Notify waiters so they can re-check against the new limit
                self._condition.notify_all()
