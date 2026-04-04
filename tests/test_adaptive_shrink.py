"""Tests for issue #73: AdaptiveConcurrencyLimiter semaphore shrink.

Verifies that when latency exceeds the target, the limiter actually
reduces effective concurrency (not just the internal counter).
"""

from __future__ import annotations

import asyncio

from xpyd_bench.bench.token_bucket import AdaptiveConcurrencyLimiter


class TestAdaptiveShrink:
    """Concurrency limit must actually decrease when latency is high."""

    def test_decrease_reduces_effective_concurrency(self):
        """After recording high latencies, fewer concurrent acquires succeed."""

        async def _run():
            limiter = AdaptiveConcurrencyLimiter(
                initial=10,
                min_concurrency=1,
                max_concurrency=20,
                target_latency_ms=100.0,
                decrease_factor=0.5,
                window_size=5,
            )

            # Record high latencies to trigger decrease
            for _ in range(5):
                await limiter.record_latency(500.0)

            # Limit should have decreased from 10
            assert limiter.limit < 10

            # Now try to acquire slots — should only get limit slots
            acquired = 0
            for _ in range(10):
                try:
                    await asyncio.wait_for(limiter.acquire(), timeout=0.05)
                    acquired += 1
                except asyncio.TimeoutError:
                    break

            # Should only be able to acquire up to the new (reduced) limit
            assert acquired == limiter.limit

        asyncio.run(_run())

    def test_increase_then_decrease(self):
        """Concurrency can go up then come back down."""

        async def _run():
            limiter = AdaptiveConcurrencyLimiter(
                initial=5,
                min_concurrency=1,
                max_concurrency=20,
                target_latency_ms=100.0,
                increase_step=2,
                decrease_factor=0.5,
                window_size=3,
            )

            # Good latencies → increase
            for _ in range(5):
                await limiter.record_latency(50.0)
            high_limit = limiter.limit
            assert high_limit > 5

            # Bad latencies → decrease
            for _ in range(5):
                await limiter.record_latency(500.0)
            low_limit = limiter.limit
            assert low_limit < high_limit

            # Verify effective concurrency matches the low limit
            acquired = 0
            for _ in range(high_limit + 5):
                try:
                    await asyncio.wait_for(limiter.acquire(), timeout=0.05)
                    acquired += 1
                except asyncio.TimeoutError:
                    break
            assert acquired == low_limit

        asyncio.run(_run())

    def test_release_unblocks_after_shrink(self):
        """After shrinking, releasing a slot unblocks a waiting acquire."""

        async def _run():
            limiter = AdaptiveConcurrencyLimiter(
                initial=4,
                min_concurrency=2,
                max_concurrency=10,
                target_latency_ms=100.0,
                decrease_factor=0.5,
                window_size=3,
            )

            # Shrink to 2
            for _ in range(5):
                await limiter.record_latency(500.0)
            assert limiter.limit == 2

            # Acquire both slots
            await limiter.acquire()
            await limiter.acquire()

            # Third acquire should block
            released = asyncio.Event()

            async def _release_later():
                await asyncio.sleep(0.05)
                limiter.release()
                released.set()

            asyncio.create_task(_release_later())

            # Should succeed after release
            await asyncio.wait_for(limiter.acquire(), timeout=1.0)
            assert released.is_set()

        asyncio.run(_run())
