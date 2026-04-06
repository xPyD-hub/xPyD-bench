"""Tests for load shedding simulation (M55)."""

from __future__ import annotations

import json
from argparse import Namespace
from unittest.mock import patch

import pytest

from xpyd_bench.load_shed import (
    LoadShedLevel,
    SaturationAnalysis,
    _count_rejections,
    format_load_shed_table,
    run_load_shed,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    throughput: float = 10.0,
    mean_latency: float = 100.0,
    p99_latency: float = 200.0,
    completed: int = 50,
    errors: int = 0,
    requests: list | None = None,
) -> tuple[dict, object]:
    """Build a fake (result_dict, bench_result) pair."""
    if requests is None:
        requests = [{"latency_ms": mean_latency, "status_code": 200} for _ in range(completed)]
    summary = {
        "request_throughput": throughput,
        "mean_e2el_ms": mean_latency,
        "p99_e2el_ms": p99_latency,
        "completed": completed,
        "errors": errors,
    }
    result_dict = {"summary": summary, "requests": requests}
    return result_dict, None


def _base_args() -> Namespace:
    return Namespace(
        base_url="http://localhost:8000",
        endpoint="/v1/completions",
        model="test-model",
        num_prompts=50,
        request_rate=1.0,
        max_concurrency=None,
        input_len=128,
        output_len=64,
        warmup=0,
        repeat=1,
        duration=None,
        retries=0,
        burstiness=1.0,
        stream=None,
        config=None,
        api_key=None,
        custom_headers={},
        http2=False,
        max_connections=100,
        max_keepalive=20,
        compress=False,
        debug_log=None,
        request_id_prefix=None,
        tokenizer=None,
        backend="openai",
        backend_plugin=None,
        no_live=True,
        priority_levels=0,
        sse_metrics=False,
        sse_stall_threshold_ms=1000.0,
        anomaly_threshold=0,
        seed=42,
        validate_response=[],
        timeout=300.0,
        metrics_ws_port=None,
        warmup_profile=False,
    )


# ---------------------------------------------------------------------------
# Unit tests: _count_rejections
# ---------------------------------------------------------------------------

class TestCountRejections:
    def test_no_rejections(self):
        result = {"requests": [{"status_code": 200}, {"status_code": 200}]}
        assert _count_rejections(result) == 0

    def test_429_rejections(self):
        result = {
            "requests": [
                {"status_code": 429},
                {"status_code": 200},
                {"status_code": 429},
            ]
        }
        assert _count_rejections(result) == 2

    def test_503_rejections(self):
        result = {"requests": [{"status_code": 503}, {"status_code": 200}]}
        assert _count_rejections(result) == 1

    def test_rejection_in_error_string(self):
        result = {
            "requests": [
                {"error": "HTTP 429 Too Many Requests", "status_code": None},
            ]
        }
        assert _count_rejections(result) == 1

    def test_empty_requests(self):
        assert _count_rejections({}) == 0
        assert _count_rejections({"requests": []}) == 0


# ---------------------------------------------------------------------------
# Unit tests: data structures
# ---------------------------------------------------------------------------

class TestLoadShedLevel:
    def test_to_dict(self):
        lv = LoadShedLevel(
            rps=10.0,
            throughput_rps=9.5,
            mean_latency_ms=100.0,
            p99_latency_ms=200.0,
            error_rate=0.05,
            rejected_count=3,
            total_requests=50,
        )
        d = lv.to_dict()
        assert d["rps"] == 10.0
        assert d["error_rate"] == 0.05
        assert d["rejected_count"] == 3


class TestSaturationAnalysis:
    def test_to_dict_empty(self):
        sa = SaturationAnalysis()
        d = sa.to_dict()
        assert d["levels"] == []
        assert d["saturation_rps"] is None
        assert d["max_sustainable_rps"] is None

    def test_to_dict_with_data(self):
        sa = SaturationAnalysis(
            saturation_rps=20.0,
            max_sustainable_rps=15.0,
            recovery_rps=10.0,
            recovery_latency_ms=80.0,
        )
        d = sa.to_dict()
        assert d["saturation_rps"] == 20.0
        assert d["max_sustainable_rps"] == 15.0
        assert d["recovery_rps"] == 10.0


# ---------------------------------------------------------------------------
# Integration: run_load_shed
# ---------------------------------------------------------------------------

class TestRunLoadShed:
    @pytest.mark.asyncio
    async def test_saturation_detected(self):
        """Simulate saturation at the 3rd level."""
        call_count = 0

        async def mock_benchmark(args, base_url):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return _make_result(errors=0, completed=50)
            else:
                # Simulate errors exceeding threshold
                reqs = [{"status_code": 429} for _ in range(30)] + [
                    {"status_code": 200} for _ in range(20)
                ]
                return _make_result(errors=30, completed=50, requests=reqs)

        with patch("xpyd_bench.load_shed.run_benchmark", side_effect=mock_benchmark):
            analysis = await run_load_shed(
                _base_args(),
                "http://localhost:8000",
                starting_rps=5.0,
                ramp_multiplier=2.0,
                prompts_per_level=50,
                max_levels=10,
                recovery_check=False,
            )

        assert analysis.saturation_rps is not None
        assert analysis.max_sustainable_rps is not None
        assert len(analysis.levels) >= 3

    @pytest.mark.asyncio
    async def test_no_saturation(self):
        """All levels pass — no saturation found."""
        async def mock_benchmark(args, base_url):
            return _make_result(errors=0, completed=50)

        with patch("xpyd_bench.load_shed.run_benchmark", side_effect=mock_benchmark):
            analysis = await run_load_shed(
                _base_args(),
                "http://localhost:8000",
                starting_rps=5.0,
                max_levels=3,
                recovery_check=False,
            )

        assert analysis.saturation_rps is None
        assert analysis.max_sustainable_rps is not None
        assert len(analysis.levels) == 3

    @pytest.mark.asyncio
    async def test_recovery_check(self):
        """Recovery check runs after saturation."""
        call_count = 0

        async def mock_benchmark(args, base_url):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return _make_result(errors=0, completed=50)
            elif call_count <= 4:
                # Two consecutive error levels to trigger saturation
                return _make_result(errors=30, completed=50)
            else:
                # Recovery level — no errors
                return _make_result(errors=0, completed=50, mean_latency=80.0)

        with patch("xpyd_bench.load_shed.run_benchmark", side_effect=mock_benchmark):
            analysis = await run_load_shed(
                _base_args(),
                "http://localhost:8000",
                starting_rps=5.0,
                ramp_multiplier=2.0,
                prompts_per_level=50,
                recovery_check=True,
            )

        assert analysis.saturation_rps is not None
        assert analysis.recovery_rps is not None

    @pytest.mark.asyncio
    async def test_additive_step(self):
        """Use additive ramp_step instead of multiplier."""
        rps_values = []

        async def mock_benchmark(args, base_url):
            rps_values.append(args.request_rate)
            return _make_result(errors=0, completed=50)

        with patch("xpyd_bench.load_shed.run_benchmark", side_effect=mock_benchmark):
            await run_load_shed(
                _base_args(),
                "http://localhost:8000",
                starting_rps=5.0,
                ramp_step=5.0,
                max_levels=3,
                recovery_check=False,
            )

        assert rps_values == [5.0, 10.0, 15.0]

    @pytest.mark.asyncio
    async def test_degradation_curve(self):
        """Degradation curve should have entries for each level."""
        async def mock_benchmark(args, base_url):
            return _make_result(errors=0, completed=50)

        with patch("xpyd_bench.load_shed.run_benchmark", side_effect=mock_benchmark):
            analysis = await run_load_shed(
                _base_args(),
                "http://localhost:8000",
                starting_rps=5.0,
                max_levels=3,
                recovery_check=False,
            )

        assert len(analysis.degradation_curve) == 3
        for pt in analysis.degradation_curve:
            assert "rps" in pt
            assert "throughput" in pt
            assert "error_rate" in pt


# ---------------------------------------------------------------------------
# Format table
# ---------------------------------------------------------------------------

class TestFormatTable:
    def test_format_with_saturation(self):
        analysis = SaturationAnalysis(
            levels=[
                LoadShedLevel(5.0, 4.8, 100.0, 200.0, 0.0, 0, 50),
                LoadShedLevel(10.0, 8.0, 150.0, 300.0, 0.4, 20, 50),
            ],
            saturation_rps=10.0,
            max_sustainable_rps=5.0,
            recovery_rps=5.0,
            recovery_latency_ms=90.0,
        )
        table = format_load_shed_table(analysis)
        assert "Load Shedding" in table
        assert "Max sustainable" in table
        assert "Saturation" in table
        assert "Recovery" in table

    def test_format_no_saturation(self):
        analysis = SaturationAnalysis(
            levels=[LoadShedLevel(5.0, 4.8, 100.0, 200.0, 0.0, 0, 50)],
            max_sustainable_rps=5.0,
        )
        table = format_load_shed_table(analysis)
        assert "not reached" in table


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

class TestCLIIntegration:
    def test_load_shed_cli_args(self):
        """Verify CLI accepts load-shed arguments."""
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args([
            "--base-url", "http://localhost:8000",
            "--model", "test",
            "--load-shed-threshold", "5.0",
            "--load-shed-step", "2.0",
            "--load-shed-multiplier", "2.0",
            "--load-shed-prompts", "100",
        ])
        assert args.load_shed_threshold == 5.0
        assert args.load_shed_step == 2.0
        assert args.load_shed_multiplier == 2.0
        assert args.load_shed_prompts == 100


# ---------------------------------------------------------------------------
# YAML config support
# ---------------------------------------------------------------------------

class TestYAMLConfig:
    def test_load_shed_yaml_keys(self):
        """Ensure load_shed_threshold is a recognized YAML config key."""
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        # Verify the dest names exist in default namespace
        defaults = parser.parse_args(["--base-url", "http://x", "--model", "m"])
        assert hasattr(defaults, "load_shed_threshold")
        assert hasattr(defaults, "load_shed_step")
        assert hasattr(defaults, "load_shed_multiplier")
        assert hasattr(defaults, "load_shed_prompts")


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

class TestJSONOutput:
    def test_saturation_analysis_json(self):
        analysis = SaturationAnalysis(
            levels=[
                LoadShedLevel(5.0, 4.8, 100.0, 200.0, 0.0, 0, 50),
            ],
            saturation_rps=10.0,
            max_sustainable_rps=5.0,
        )
        d = analysis.to_dict()
        # Should be JSON-serializable
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["saturation_rps"] == 10.0
        assert len(parsed["levels"]) == 1
