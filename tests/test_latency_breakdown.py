"""Tests for M57: Network Latency Decomposition."""

from __future__ import annotations

import pytest

from xpyd_bench.bench.latency_breakdown import (
    LatencyBreakdown,
    compute_breakdown_summary,
    estimate_server_processing,
    parse_url,
    print_breakdown_summary,
)

# ---------------------------------------------------------------------------
# Unit tests for LatencyBreakdown
# ---------------------------------------------------------------------------


class TestLatencyBreakdown:
    """Tests for LatencyBreakdown dataclass."""

    def test_to_dict(self) -> None:
        bd = LatencyBreakdown(dns_ms=1.234, connect_ms=2.567, tls_ms=3.891, server_ms=50.0)
        d = bd.to_dict()
        assert d == {
            "dns_ms": 1.234,
            "connect_ms": 2.567,
            "tls_ms": 3.891,
            "server_ms": 50.0,
        }

    def test_defaults(self) -> None:
        bd = LatencyBreakdown()
        assert bd.dns_ms == 0.0
        assert bd.connect_ms == 0.0
        assert bd.tls_ms == 0.0
        assert bd.server_ms == 0.0


class TestComputeBreakdownSummary:
    """Tests for compute_breakdown_summary."""

    def test_empty(self) -> None:
        assert compute_breakdown_summary([]) == {}

    def test_single_breakdown(self) -> None:
        bd = LatencyBreakdown(dns_ms=1.0, connect_ms=2.0, tls_ms=3.0, server_ms=50.0)
        summary = compute_breakdown_summary([bd])
        assert "dns_ms" in summary
        assert "connect_ms" in summary
        assert "tls_ms" in summary
        assert "server_ms" in summary
        # Single value: mean == p50 == p99
        assert summary["dns_ms"]["mean"] == 1.0
        assert summary["dns_ms"]["p50"] == 1.0
        assert summary["dns_ms"]["p99"] == 1.0

    def test_multiple_breakdowns(self) -> None:
        breakdowns = [
            LatencyBreakdown(dns_ms=1.0, connect_ms=2.0, tls_ms=0.0, server_ms=50.0),
            LatencyBreakdown(dns_ms=0.0, connect_ms=0.0, tls_ms=0.0, server_ms=45.0),
            LatencyBreakdown(dns_ms=0.0, connect_ms=0.0, tls_ms=0.0, server_ms=55.0),
        ]
        summary = compute_breakdown_summary(breakdowns)
        # DNS: [1.0, 0.0, 0.0] → mean ≈ 0.333
        assert 0.3 <= summary["dns_ms"]["mean"] <= 0.4
        # Server: [50, 45, 55] → mean = 50
        assert summary["server_ms"]["mean"] == 50.0


class TestEstimateServerProcessing:
    """Tests for estimate_server_processing."""

    def test_basic(self) -> None:
        server = estimate_server_processing(100.0, 1.0, 2.0, 3.0)
        assert server == 94.0

    def test_no_network_overhead(self) -> None:
        server = estimate_server_processing(50.0, 0.0, 0.0, 0.0)
        assert server == 50.0

    def test_negative_clamped_to_zero(self) -> None:
        # Network phases exceed total (edge case)
        server = estimate_server_processing(5.0, 3.0, 3.0, 3.0)
        assert server == 0.0


class TestParseUrl:
    """Tests for parse_url."""

    def test_http(self) -> None:
        host, port, tls = parse_url("http://localhost:8000")
        assert host == "localhost"
        assert port == 8000
        assert tls is False

    def test_https_default_port(self) -> None:
        host, port, tls = parse_url("https://api.example.com/v1")
        assert host == "api.example.com"
        assert port == 443
        assert tls is True

    def test_http_default_port(self) -> None:
        host, port, tls = parse_url("http://10.0.0.1")
        assert host == "10.0.0.1"
        assert port == 80
        assert tls is False


class TestPrintBreakdownSummary:
    """Tests for print_breakdown_summary."""

    def test_empty_no_output(self, capsys: pytest.CaptureFixture) -> None:
        print_breakdown_summary({})
        assert capsys.readouterr().out == ""

    def test_prints_table(self, capsys: pytest.CaptureFixture) -> None:
        summary = {
            "dns_ms": {"mean": 1.0, "p50": 0.8, "p99": 2.0},
            "connect_ms": {"mean": 2.0, "p50": 1.5, "p99": 3.0},
            "tls_ms": {"mean": 5.0, "p50": 4.5, "p99": 8.0},
            "server_ms": {"mean": 50.0, "p50": 45.0, "p99": 80.0},
        }
        print_breakdown_summary(summary)
        out = capsys.readouterr().out
        assert "Network Latency Breakdown" in out
        assert "DNS" in out
        assert "CONNECT" in out
        assert "TLS" in out
        assert "SERVER" in out


# ---------------------------------------------------------------------------
# Integration: RequestResult and BenchmarkResult fields
# ---------------------------------------------------------------------------


class TestModelFields:
    """Verify M57 fields exist on data models."""

    def test_request_result_has_breakdown(self) -> None:
        from xpyd_bench.bench.models import RequestResult

        r = RequestResult()
        assert r.latency_breakdown is None
        r.latency_breakdown = {"dns_ms": 1.0}
        assert r.latency_breakdown["dns_ms"] == 1.0

    def test_benchmark_result_has_breakdown(self) -> None:
        from xpyd_bench.bench.models import BenchmarkResult

        br = BenchmarkResult()
        assert br.latency_breakdown is None


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCLIFlag:
    """Test --latency-breakdown CLI flag is recognized."""

    def test_flag_parsed(self) -> None:
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--latency-breakdown"])
        assert args.latency_breakdown is True

    def test_flag_default_false(self) -> None:
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args([])
        assert args.latency_breakdown is False


# ---------------------------------------------------------------------------
# YAML config support
# ---------------------------------------------------------------------------


class TestYAMLConfig:
    """Test YAML config support for latency_breakdown."""

    def test_yaml_key_in_known_keys(self) -> None:
        from xpyd_bench.config_cmd import _KNOWN_KEYS

        assert "latency_breakdown" in _KNOWN_KEYS


# ---------------------------------------------------------------------------
# JSON output integration
# ---------------------------------------------------------------------------


class TestJSONOutput:
    """Test latency_breakdown appears in JSON result dict."""

    def test_to_dict_includes_breakdown(self) -> None:
        from xpyd_bench.bench.models import BenchmarkResult
        from xpyd_bench.bench.runner import _to_dict

        br = BenchmarkResult()
        br.latency_breakdown = {
            "dns_ms": {"mean": 1.0, "p50": 0.8, "p99": 2.0},
            "server_ms": {"mean": 50.0, "p50": 45.0, "p99": 80.0},
        }
        d = _to_dict(br)
        assert "latency_breakdown" in d
        assert d["latency_breakdown"]["dns_ms"]["mean"] == 1.0

    def test_to_dict_no_breakdown(self) -> None:
        from xpyd_bench.bench.models import BenchmarkResult
        from xpyd_bench.bench.runner import _to_dict

        br = BenchmarkResult()
        d = _to_dict(br)
        assert "latency_breakdown" not in d


# ---------------------------------------------------------------------------
# HTML report integration
# ---------------------------------------------------------------------------


class TestHTMLReport:
    """Test latency breakdown section in HTML report."""

    def test_no_breakdown_no_section(self) -> None:
        from xpyd_bench.bench.models import BenchmarkResult
        from xpyd_bench.reporting.html_report import _latency_breakdown_section

        br = BenchmarkResult()
        assert _latency_breakdown_section(br) == ""

    def test_breakdown_renders_section(self) -> None:
        from xpyd_bench.bench.models import BenchmarkResult
        from xpyd_bench.reporting.html_report import _latency_breakdown_section

        br = BenchmarkResult()
        br.latency_breakdown = {
            "dns_ms": {"mean": 1.0, "p50": 0.8, "p99": 2.0},
            "connect_ms": {"mean": 2.0, "p50": 1.5, "p99": 3.0},
            "tls_ms": {"mean": 0.0, "p50": 0.0, "p99": 0.0},
            "server_ms": {"mean": 50.0, "p50": 45.0, "p99": 80.0},
        }
        html = _latency_breakdown_section(br)
        assert "Network Latency Breakdown" in html
        assert "DNS" in html
        assert "TCP Connect" in html
        assert "Server Processing" in html
        assert "Waterfall" in html
