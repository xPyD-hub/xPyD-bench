"""Tests for M68: Output Token Speed Benchmarking & Comparison."""

from __future__ import annotations

from xpyd_bench.bench.generation_speed import (
    GenerationSpeedSummary,
    aggregate_generation_speeds,
    compute_generation_tps,
)


class TestComputeGenerationTps:
    """Tests for per-request generation TPS calculation."""

    def test_streaming_with_ttft(self):
        """Streaming: generation_time = latency - ttft."""
        # 100 tokens, latency=500ms, ttft=100ms => gen_time=400ms => 250 tok/s
        tps = compute_generation_tps(100, ttft_ms=100.0, latency_ms=500.0)
        assert tps is not None
        assert abs(tps - 250.0) < 0.01

    def test_non_streaming_no_ttft(self):
        """Non-streaming: use full latency as generation time."""
        # 50 tokens, latency=1000ms => 50 tok/s
        tps = compute_generation_tps(50, ttft_ms=None, latency_ms=1000.0)
        assert tps is not None
        assert abs(tps - 50.0) < 0.01

    def test_zero_tokens_returns_none(self):
        assert compute_generation_tps(0, ttft_ms=10.0, latency_ms=100.0) is None

    def test_negative_tokens_returns_none(self):
        assert compute_generation_tps(-5, ttft_ms=10.0, latency_ms=100.0) is None

    def test_zero_generation_time_returns_none(self):
        """ttft == latency => generation_time = 0 => None."""
        assert compute_generation_tps(10, ttft_ms=100.0, latency_ms=100.0) is None

    def test_negative_generation_time_returns_none(self):
        """ttft > latency => negative generation_time => None."""
        assert compute_generation_tps(10, ttft_ms=200.0, latency_ms=100.0) is None

    def test_zero_latency_returns_none(self):
        assert compute_generation_tps(10, ttft_ms=None, latency_ms=0.0) is None

    def test_large_token_count(self):
        # 10000 tokens in 2000ms => 5000 tok/s
        tps = compute_generation_tps(10000, ttft_ms=None, latency_ms=2000.0)
        assert tps is not None
        assert abs(tps - 5000.0) < 0.01

    def test_small_generation_time(self):
        # 1 token in 1ms generation time
        tps = compute_generation_tps(1, ttft_ms=99.0, latency_ms=100.0)
        assert tps is not None
        assert abs(tps - 1000.0) < 0.01


class TestAggregateGenerationSpeeds:
    """Tests for TPS aggregation."""

    def test_basic_aggregation(self):
        vals = [100.0, 200.0, 150.0, 250.0, 50.0]
        summary = aggregate_generation_speeds(vals)
        assert summary.tracked_requests == 5
        assert abs(summary.mean_tps - 150.0) < 0.01
        assert summary.min_tps == 50.0
        assert summary.max_tps == 250.0
        assert summary.p50_tps == 150.0

    def test_empty_list(self):
        summary = aggregate_generation_speeds([])
        assert summary.tracked_requests == 0
        assert summary.mean_tps == 0.0

    def test_all_none(self):
        summary = aggregate_generation_speeds([None, None, None])
        assert summary.tracked_requests == 0

    def test_mixed_none_and_values(self):
        vals = [100.0, None, 200.0, None]
        summary = aggregate_generation_speeds(vals)
        assert summary.tracked_requests == 2
        assert abs(summary.mean_tps - 150.0) < 0.01

    def test_single_value(self):
        summary = aggregate_generation_speeds([42.0])
        assert summary.tracked_requests == 1
        assert summary.mean_tps == 42.0
        assert summary.p50_tps == 42.0
        assert summary.p90_tps == 42.0
        assert summary.p99_tps == 42.0

    def test_to_dict(self):
        summary = aggregate_generation_speeds([100.0, 200.0])
        d = summary.to_dict()
        assert "mean_tps" in d
        assert "p50_tps" in d
        assert "p90_tps" in d
        assert "p99_tps" in d
        assert "min_tps" in d
        assert "max_tps" in d
        assert "tracked_requests" in d
        assert d["tracked_requests"] == 2


class TestGenerationSpeedSummaryToDict:
    """Tests for to_dict rounding."""

    def test_rounding(self):
        s = GenerationSpeedSummary(
            mean_tps=123.456789,
            p50_tps=100.111,
            p90_tps=200.999,
            p99_tps=300.005,
            min_tps=50.123,
            max_tps=400.789,
            tracked_requests=10,
        )
        d = s.to_dict()
        assert d["mean_tps"] == 123.46
        assert d["min_tps"] == 50.12
        assert d["max_tps"] == 400.79


class TestConfigKey:
    """Test that measure_generation_speed is a known config key."""

    def test_known_key(self):
        from xpyd_bench.config_cmd import _KNOWN_KEYS

        assert "measure_generation_speed" in _KNOWN_KEYS
