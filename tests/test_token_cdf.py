"""Tests for token-level streaming latency CDF (M91)."""

from __future__ import annotations

import numpy as np

from xpyd_bench.bench.token_cdf import (
    TokenCDFResult,
    _detect_bimodal,
    collect_itl_from_requests,
    compute_token_cdf,
    generate_cdf_html_chart,
)


class TestComputeTokenCDF:
    """Tests for CDF computation."""

    def test_basic_cdf(self) -> None:
        """Basic CDF from uniform data."""
        itl = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = compute_token_cdf(itl)
        assert result.total_tokens == 5
        assert len(result.cdf_points) == 5
        # Last point should have cumulative fraction 1.0
        assert result.cdf_points[-1][1] == 1.0
        # Sorted order
        assert result.cdf_points[0][0] == 10.0
        assert result.cdf_points[-1][0] == 50.0

    def test_percentiles(self) -> None:
        """Percentiles computed correctly."""
        rng = np.random.default_rng(42)
        itl = rng.exponential(scale=10.0, size=1000).tolist()
        result = compute_token_cdf(itl)
        assert "p50" in result.percentiles
        assert "p99" in result.percentiles
        assert "p99.9" in result.percentiles
        assert "p1" in result.percentiles
        # p99 > p50
        assert result.percentiles["p99"] > result.percentiles["p50"]
        # p1 < p50
        assert result.percentiles["p1"] < result.percentiles["p50"]

    def test_downsampling(self) -> None:
        """CDF points are downsampled when data is large."""
        itl = list(range(1, 10001))  # 10000 values
        result = compute_token_cdf(itl, max_cdf_points=100)
        assert len(result.cdf_points) == 100
        assert result.total_tokens == 10000

    def test_empty_input(self) -> None:
        """Empty input returns empty result."""
        result = compute_token_cdf([])
        assert result.total_tokens == 0
        assert result.cdf_points == []
        assert result.percentiles == {}
        assert not result.is_bimodal

    def test_single_value(self) -> None:
        """Single value produces single CDF point."""
        result = compute_token_cdf([42.0])
        assert result.total_tokens == 1
        assert len(result.cdf_points) == 1
        assert result.cdf_points[0] == (42.0, 1.0)

    def test_to_dict(self) -> None:
        """Serialization to dict."""
        itl = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = compute_token_cdf(itl)
        d = result.to_dict()
        assert "cdf_points" in d
        assert "percentiles" in d
        assert "total_tokens" in d
        assert "is_bimodal" in d
        assert "bimodal_details" in d
        assert d["total_tokens"] == 5
        assert len(d["cdf_points"]) == 5
        # Check CDF point structure
        assert "latency_ms" in d["cdf_points"][0]
        assert "cumulative_fraction" in d["cdf_points"][0]


class TestBimodalDetection:
    """Tests for bimodal distribution detection."""

    def test_unimodal(self) -> None:
        """Normal distribution should not be bimodal."""
        rng = np.random.default_rng(42)
        data = rng.normal(50.0, 5.0, size=200)
        is_bm, details = _detect_bimodal(data)
        assert not is_bm

    def test_bimodal(self) -> None:
        """Clear bimodal distribution should be detected."""
        rng = np.random.default_rng(42)
        mode1 = rng.normal(20.0, 2.0, size=200)
        mode2 = rng.normal(80.0, 2.0, size=200)
        data = np.concatenate([mode1, mode2])
        is_bm, details = _detect_bimodal(data)
        assert is_bm
        assert "mode1_ms" in details
        assert "mode2_ms" in details
        assert "valley_drop_ratio" in details

    def test_insufficient_data(self) -> None:
        """Too few samples returns not bimodal."""
        data = np.array([1.0, 2.0, 3.0])
        is_bm, details = _detect_bimodal(data)
        assert not is_bm
        assert details.get("reason") == "insufficient_data"

    def test_bimodal_via_compute(self) -> None:
        """Bimodal detection through compute_token_cdf."""
        rng = np.random.default_rng(42)
        mode1 = rng.normal(10.0, 1.0, size=300).tolist()
        mode2 = rng.normal(50.0, 1.0, size=300).tolist()
        result = compute_token_cdf(mode1 + mode2)
        assert result.is_bimodal


class TestCollectITL:
    """Tests for collecting ITL from request results."""

    def test_collect(self) -> None:
        """Collect ITL from mock requests."""

        class MockReq:
            def __init__(self, itl: list[float]):
                self.itl_ms = itl

        reqs = [MockReq([10.0, 20.0]), MockReq([30.0]), MockReq([])]
        result = collect_itl_from_requests(reqs)
        assert result == [10.0, 20.0, 30.0]

    def test_empty_requests(self) -> None:
        """Empty request list returns empty ITL."""
        assert collect_itl_from_requests([]) == []

    def test_no_itl_attr(self) -> None:
        """Requests without itl_ms are skipped."""

        class MockReq:
            pass

        assert collect_itl_from_requests([MockReq()]) == []


class TestHTMLChart:
    """Tests for CDF HTML chart generation."""

    def test_generates_html(self) -> None:
        """Non-empty result produces HTML with canvas."""
        itl = list(range(1, 101))
        result = compute_token_cdf([float(x) for x in itl])
        html = generate_cdf_html_chart(result)
        assert "<h2>" in html
        assert "tokenCdfCanvas" in html
        assert "Percentile" in html

    def test_empty_result(self) -> None:
        """Empty result produces no HTML."""
        result = TokenCDFResult()
        html = generate_cdf_html_chart(result)
        assert html == ""

    def test_bimodal_warning(self) -> None:
        """Bimodal result includes warning in HTML."""
        rng = np.random.default_rng(42)
        mode1 = rng.normal(10.0, 1.0, size=300).tolist()
        mode2 = rng.normal(50.0, 1.0, size=300).tolist()
        result = compute_token_cdf(mode1 + mode2)
        html = generate_cdf_html_chart(result)
        assert "Bimodal" in html

    def test_percentile_table(self) -> None:
        """HTML includes percentile rows."""
        result = compute_token_cdf([10.0, 20.0, 30.0, 40.0, 50.0])
        html = generate_cdf_html_chart(result)
        assert "p50" in html
        assert "p99" in html


class TestYAMLConfig:
    """Tests for YAML config support."""

    def test_token_cdf_in_known_keys(self) -> None:
        """token_cdf is a known config key."""
        from xpyd_bench.config_cmd import _KNOWN_KEYS

        assert "token_cdf" in _KNOWN_KEYS


class TestCLIIntegration:
    """Tests for CLI flag parsing."""

    def test_token_cdf_flag(self) -> None:
        """--token-cdf flag is parsed correctly."""
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--token-cdf"])
        assert args.token_cdf is True

    def test_default_disabled(self) -> None:
        """token_cdf is disabled by default."""
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args([])
        assert args.token_cdf is False
