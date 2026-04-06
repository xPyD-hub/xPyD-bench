"""Tests for size_impact module (M100)."""

from __future__ import annotations

import argparse
import json
from unittest.mock import patch

import pytest

from xpyd_bench.size_impact import (
    ScalingAnalysis,
    SizeImpactResult,
    SizeProbe,
    detect_scaling,
    parse_size_levels,
    recommend_max_size,
    run_size_impact,
    size_impact_main,
)


def _probe(
    tokens: int,
    ttft: float | None = None,
    p99_ttft: float | None = None,
    tpot: float | None = None,
    p99_tpot: float | None = None,
    rps: float = 0.0,
    tps: float = 0.0,
    e2el: float | None = None,
    err: float = 0.0,
) -> SizeProbe:
    return SizeProbe(
        prompt_tokens=tokens,
        mean_ttft_ms=ttft,
        p99_ttft_ms=p99_ttft,
        mean_tpot_ms=tpot,
        p99_tpot_ms=p99_tpot,
        throughput_rps=rps,
        throughput_tps=tps,
        mean_e2el_ms=e2el,
        error_rate=err,
    )


# -------------------------------------------------------------------
# parse_size_levels
# -------------------------------------------------------------------


class TestParseSizeLevels:
    def test_comma_separated(self) -> None:
        assert parse_size_levels("10,100,500") == [10, 100, 500]

    def test_range_notation_three_parts(self) -> None:
        result = parse_size_levels("100:500:100")
        assert result == [100, 200, 300, 400, 500]

    def test_range_notation_two_parts(self) -> None:
        result = parse_size_levels("100:500")
        assert len(result) > 1
        assert result[0] == 100
        assert result[-1] <= 500

    def test_mixed(self) -> None:
        result = parse_size_levels("10,100:300:100,4000")
        assert 10 in result
        assert 100 in result
        assert 200 in result
        assert 300 in result
        assert 4000 in result

    def test_dedup_and_sort(self) -> None:
        result = parse_size_levels("500,100,500,10")
        assert result == [10, 100, 500]

    def test_invalid_range(self) -> None:
        with pytest.raises(ValueError, match="Invalid range notation"):
            parse_size_levels("1:2:3:4")


# -------------------------------------------------------------------
# detect_scaling
# -------------------------------------------------------------------


class TestDetectScaling:
    def test_linear(self) -> None:
        probes = [
            _probe(100, ttft=10.0, p99_ttft=12.0, tpot=1.0,
                   p99_tpot=2.0, rps=10.0, tps=100.0, e2el=50.0),
            _probe(200, ttft=20.0, p99_ttft=24.0, tpot=1.0,
                   p99_tpot=2.0, rps=10.0, tps=100.0, e2el=50.0),
            _probe(400, ttft=40.0, p99_ttft=48.0, tpot=1.0,
                   p99_tpot=2.0, rps=10.0, tps=100.0, e2el=50.0),
        ]
        s = detect_scaling(probes)
        assert s.behaviour == "linear"
        assert s.slope is not None and s.slope > 0

    def test_sublinear(self) -> None:
        probes = [
            _probe(100, ttft=10.0),
            _probe(400, ttft=14.0),
            _probe(1600, ttft=20.0),
        ]
        s = detect_scaling(probes)
        assert s.behaviour == "sublinear"

    def test_superlinear(self) -> None:
        probes = [
            _probe(10, ttft=1.0),
            _probe(100, ttft=100.0),
            _probe(1000, ttft=100000.0),
        ]
        s = detect_scaling(probes)
        assert s.behaviour == "superlinear"

    def test_insufficient_data(self) -> None:
        probes = [_probe(100, ttft=10.0)]
        s = detect_scaling(probes)
        assert s.behaviour == "unknown"

    def test_no_ttft(self) -> None:
        probes = [_probe(100), _probe(200)]
        s = detect_scaling(probes)
        assert s.behaviour == "unknown"

    def test_inflection_detection(self) -> None:
        probes = [
            _probe(100, ttft=10.0),
            _probe(200, ttft=12.0),
            _probe(300, ttft=14.0),
            _probe(400, ttft=50.0),
        ]
        s = detect_scaling(probes)
        assert len(s.inflection_points) > 0


# -------------------------------------------------------------------
# recommend_max_size
# -------------------------------------------------------------------


class TestRecommendMaxSize:
    def test_basic(self) -> None:
        probes = [
            _probe(100, ttft=10.0),
            _probe(500, ttft=50.0),
            _probe(1000, ttft=200.0),
        ]
        assert recommend_max_size(probes, 100.0) == 500
        assert recommend_max_size(probes, 5.0) is None
        assert recommend_max_size(probes, 200.0) == 1000

    def test_no_ttft(self) -> None:
        probes = [_probe(100)]
        assert recommend_max_size(probes, 100.0) is None


# -------------------------------------------------------------------
# Serialization
# -------------------------------------------------------------------


class TestSerialization:
    def test_size_probe_to_dict(self) -> None:
        p = _probe(
            100, ttft=10.123, p99_ttft=15.678,
            tpot=1.234, p99_tpot=2.345,
            rps=10.567, tps=100.891, e2el=50.5, err=0.01234,
        )
        d = p.to_dict()
        assert d["prompt_tokens"] == 100
        assert d["mean_ttft_ms"] == 10.12
        assert d["error_rate"] == 0.0123

    def test_scaling_to_dict(self) -> None:
        s = ScalingAnalysis(
            behaviour="linear", slope=0.05,
            inflection_points=[500],
        )
        d = s.to_dict()
        assert d["behaviour"] == "linear"
        assert d["inflection_points"] == [500]

    def test_result_to_dict(self) -> None:
        r = SizeImpactResult(
            probes=[],
            scaling=ScalingAnalysis(
                behaviour="sublinear", slope=0.01,
            ),
            recommended_max_prompt_tokens=2000,
            target_latency_ms=100.0,
        )
        d = r.to_dict()
        assert d["scaling"]["behaviour"] == "sublinear"
        assert d["recommended_max_prompt_tokens"] == 2000
        assert json.dumps(d)  # JSON serializable

    def test_result_to_dict_minimal(self) -> None:
        r = SizeImpactResult()
        d = r.to_dict()
        assert "probes" in d
        assert "recommended_max_prompt_tokens" not in d


# -------------------------------------------------------------------
# run_size_impact (integration with mocked runner)
# -------------------------------------------------------------------


def _make_mock_bench_result(
    ttft: float | None = 10.0,
    tpot: float | None = 1.0,
    throughput_rps: float = 10.0,
) -> object:
    class MockResult:
        completed = 20
        failed = 0
        requests: list = []
        mean_ttft_ms = ttft
        p99_ttft_ms = ttft
        mean_tpot_ms = tpot
        p99_tpot_ms = tpot
        request_throughput = throughput_rps
        total_token_throughput = throughput_rps * 10
        mean_e2el_ms = 50.0

    return MockResult()


def _base_ns() -> argparse.Namespace:
    """Build a minimal namespace for tests."""
    ns = argparse.Namespace()
    ns.backend = "openai"
    ns.backend_plugin = None
    ns.list_backends = False
    ns.base_url = "http://localhost:8000"
    ns.host = "127.0.0.1"
    ns.port = 8000
    ns.endpoint = "/v1/completions"
    ns.model = "test"
    ns.stream = False
    ns.no_stream = True
    ns.request_rate = float("inf")
    ns.max_concurrency = None
    ns.output_len = 128
    ns.input_len = 256
    ns.num_prompts = 20
    ns.dataset = None
    ns.dataset_name = None
    ns.dataset_path = None
    ns.random_input_len = None
    ns.random_output_len = None
    ns.random_prefix_len = None
    ns.config = None
    ns.scenario = None
    ns.list_scenarios = False
    ns.duration = None
    ns.seed = None
    ns.timeout = 300
    ns.retries = 0
    ns.retry_delay = 1.0
    ns.api_key = None
    ns.api_key_file = None
    ns.save_result = None
    ns.html_report = None
    ns.csv_report = None
    ns.markdown_report = None
    ns.junit_xml = None
    ns.debug_log = None
    ns.webhook_url = None
    ns.otlp_endpoint = None
    ns.prometheus_export = None
    ns.heatmap_export = None
    ns.no_live = True
    ns.warmup = 0
    ns.verbose = False
    ns.percentile_metrics = "ttft,tpot,itl,e2el"
    ns.metric_percentiles = "p50,p90,p95,p99"
    ns.goodput_config = None
    ns.sonnet_prefix = None
    ns.sharegpt_output_len = None
    ns.tag = None
    ns.note = None
    ns.tokenizer = None
    ns.tokenizer_mode = "auto"
    ns.trust_remote_code = False
    ns.adaptive_rate = False
    ns.max_error_rate = None
    ns.abort_on_error_pct = None
    ns.reproducibility = None
    return ns


def test_run_size_impact() -> None:
    import asyncio

    call_count = 0

    async def mock_run(args, base_url):
        nonlocal call_count
        call_count += 1
        ttft = args.input_len * 0.1
        return {}, _make_mock_bench_result(ttft=ttft)

    async def _run():
        return await run_size_impact(
            _base_ns(),
            "http://localhost:8000",
            [100, 500, 1000],
            target_latency_ms=80.0,
        )

    with patch(
        "xpyd_bench.size_impact.run_benchmark",
        side_effect=mock_run,
    ):
        result = asyncio.run(_run())

    assert call_count == 3
    assert len(result.probes) == 3
    assert result.probes[0].prompt_tokens == 100
    assert result.probes[1].prompt_tokens == 500
    assert result.probes[2].prompt_tokens == 1000
    assert result.scaling is not None
    assert result.scaling.behaviour in (
        "linear", "sublinear", "superlinear",
    )
    # 500*0.1=50 < 80
    assert result.recommended_max_prompt_tokens == 500


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------


class TestCLI:
    def test_cli_json_output(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        async def mock_run(args, base_url):
            return {}, _make_mock_bench_result()

        with patch(
            "xpyd_bench.size_impact.run_benchmark",
            side_effect=mock_run,
        ):
            size_impact_main([
                "--base-url", "http://localhost:8000",
                "--model", "test",
                "--size-levels", "100,200",
                "--json",
            ])

        out = capsys.readouterr().out
        data = json.loads(out)
        assert len(data["probes"]) == 2
        assert data["probes"][0]["prompt_tokens"] == 100

    def test_cli_text_output(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        async def mock_run(args, base_url):
            return {}, _make_mock_bench_result()

        with patch(
            "xpyd_bench.size_impact.run_benchmark",
            side_effect=mock_run,
        ):
            size_impact_main([
                "--base-url", "http://localhost:8000",
                "--size-levels", "100",
            ])

        out = capsys.readouterr().out
        assert "Size Impact" in out

    def test_cli_with_target_latency(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        async def mock_run(args, base_url):
            ttft = args.input_len * 0.05
            return {}, _make_mock_bench_result(ttft=ttft)

        with patch(
            "xpyd_bench.size_impact.run_benchmark",
            side_effect=mock_run,
        ):
            size_impact_main([
                "--base-url", "http://localhost:8000",
                "--size-levels", "100,1000",
                "--target-latency-ms", "20",
                "--json",
            ])

        data = json.loads(capsys.readouterr().out)
        assert data["recommended_max_prompt_tokens"] == 100
        assert data["target_latency_ms"] == 20.0

    def test_cli_custom_endpoint(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        async def mock_run(args, base_url):
            assert args.endpoint == "/v1/chat/completions"
            return {}, _make_mock_bench_result()

        with patch(
            "xpyd_bench.size_impact.run_benchmark",
            side_effect=mock_run,
        ):
            size_impact_main([
                "--base-url", "http://localhost:8000",
                "--size-levels", "100",
                "--endpoint", "/v1/chat/completions",
                "--json",
            ])
