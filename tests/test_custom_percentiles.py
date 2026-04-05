"""Tests for M73: Custom Percentile Configuration."""

from __future__ import annotations

from argparse import Namespace

import numpy as np
import pytest

from xpyd_bench.bench.custom_percentiles import (
    DEFAULT_PERCENTILES,
    compute_custom_percentiles,
    parse_percentiles,
)
from xpyd_bench.bench.models import BenchmarkResult, RequestResult

# ---------------------------------------------------------------------------
# parse_percentiles
# ---------------------------------------------------------------------------

class TestParsePercentiles:
    def test_none_returns_defaults(self):
        assert parse_percentiles(None) == DEFAULT_PERCENTILES

    def test_string_input(self):
        assert parse_percentiles("50,90,99.9") == [50.0, 90.0, 99.9]

    def test_list_input(self):
        assert parse_percentiles([50, 95, 99]) == [50.0, 95.0, 99.0]

    def test_dedup_and_sort(self):
        assert parse_percentiles("99,50,50,90") == [50.0, 90.0, 99.0]

    def test_invalid_zero_raises(self):
        with pytest.raises(ValueError, match="between 0 and 100"):
            parse_percentiles("0,50")

    def test_invalid_100_raises(self):
        with pytest.raises(ValueError, match="between 0 and 100"):
            parse_percentiles("50,100")

    def test_non_string_non_list_returns_defaults(self):
        assert parse_percentiles(42) == DEFAULT_PERCENTILES


# ---------------------------------------------------------------------------
# compute_custom_percentiles
# ---------------------------------------------------------------------------

def _make_result(latencies: list[float], ttfts: list[float] | None = None) -> BenchmarkResult:
    """Build a minimal BenchmarkResult with given latencies."""
    result = BenchmarkResult()
    for i, lat in enumerate(latencies):
        rr = RequestResult(latency_ms=lat, success=True)
        if ttfts and i < len(ttfts):
            rr.ttft_ms = ttfts[i]
        result.requests.append(rr)
    return result


class TestComputeCustomPercentiles:
    def test_default_percentiles(self):
        result = _make_result([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        args = Namespace(percentiles=None)
        compute_custom_percentiles(result, args)

        assert result.custom_percentiles is not None
        assert "e2el_ms" in result.custom_percentiles
        pcts = result.custom_percentiles["e2el_ms"]
        assert "p50" in pcts
        assert "p90" in pcts
        assert "p95" in pcts
        assert "p99" in pcts

    def test_custom_percentiles_string(self):
        result = _make_result([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        args = Namespace(percentiles="50,99.9")
        compute_custom_percentiles(result, args)

        pcts = result.custom_percentiles["e2el_ms"]
        assert "p50" in pcts
        assert "p99.9" in pcts
        assert len(pcts) == 2

    def test_custom_percentiles_list(self):
        result = _make_result([10, 20, 30, 40, 50])
        args = Namespace(percentiles=[25, 75])
        compute_custom_percentiles(result, args)

        pcts = result.custom_percentiles["e2el_ms"]
        assert "p25" in pcts
        assert "p75" in pcts

    def test_includes_ttft_when_present(self):
        result = _make_result([10, 20, 30], ttfts=[1.0, 2.0, 3.0])
        args = Namespace(percentiles="50")
        compute_custom_percentiles(result, args)

        assert "ttft_ms" in result.custom_percentiles

    def test_no_successful_requests(self):
        result = BenchmarkResult()
        result.requests.append(RequestResult(success=False))
        args = Namespace(percentiles="50")
        compute_custom_percentiles(result, args)
        assert result.custom_percentiles is None

    def test_no_percentiles_attr(self):
        """When args has no 'percentiles' attr, use defaults."""
        result = _make_result([10, 20, 30])
        args = Namespace()
        compute_custom_percentiles(result, args)
        assert result.custom_percentiles is not None

    def test_values_correctness(self):
        """Verify computed values match numpy directly."""
        data = list(range(1, 101))  # 1..100
        result = _make_result(data)
        args = Namespace(percentiles="50,99")
        compute_custom_percentiles(result, args)

        pcts = result.custom_percentiles["e2el_ms"]
        assert pcts["p50"] == pytest.approx(np.percentile(data, 50))
        assert pcts["p99"] == pytest.approx(np.percentile(data, 99))


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

class TestCLIIntegration:
    def test_cli_flag_parsed(self):
        """Verify --percentiles flag is accepted by the CLI parser."""
        import argparse

        # We can't easily run bench_main, but we test the parser accepts the flag
        # by importing and parsing
        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--percentiles", "50,90,99.9"])
        assert args.percentiles == "50,90,99.9"


# ---------------------------------------------------------------------------
# Config known keys
# ---------------------------------------------------------------------------

class TestConfigKnownKeys:
    def test_percentiles_in_known_keys(self):
        from xpyd_bench.config_cmd import _KNOWN_KEYS

        assert "percentiles" in _KNOWN_KEYS
