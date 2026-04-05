"""Tests for autotune module (M98)."""

from __future__ import annotations

import json
from argparse import Namespace
from unittest.mock import patch

import pytest

from xpyd_bench.autotune import (
    AutotuneResult,
    TuneProbe,
    _generate_concurrency_levels,
    _make_probe_args,
    autotune_main,
    find_optimal,
    format_autotune_summary,
    generate_config,
    run_autotune,
)

# ---------------------------------------------------------------------------
# TuneProbe
# ---------------------------------------------------------------------------


class TestTuneProbe:
    def test_to_dict(self):
        p = TuneProbe(
            concurrency=4,
            throughput_rps=100.123,
            throughput_tps=500.456,
            mean_latency_ms=12.345,
            p99_latency_ms=45.678,
            error_rate=0.01234,
            total_requests=50,
        )
        d = p.to_dict()
        assert d["concurrency"] == 4
        assert d["throughput_rps"] == 100.12
        assert d["throughput_tps"] == 500.46
        assert d["mean_latency_ms"] == 12.35
        assert d["p99_latency_ms"] == 45.68
        assert d["error_rate"] == 0.0123
        assert d["total_requests"] == 50

    def test_to_dict_rounding(self):
        p = TuneProbe(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
        d = p.to_dict()
        assert d["throughput_rps"] == 0.0
        assert d["error_rate"] == 0.0


# ---------------------------------------------------------------------------
# AutotuneResult
# ---------------------------------------------------------------------------


class TestAutotuneResult:
    def test_to_dict_empty(self):
        r = AutotuneResult()
        d = r.to_dict()
        assert d["trajectory"] == []
        assert d["optimal"]["concurrency"] is None
        assert d["target"] == "throughput"
        assert d["error_budget"] == 0.01

    def test_to_dict_with_data(self):
        probe = TuneProbe(4, 100.0, 500.0, 10.0, 30.0, 0.005, 50)
        r = AutotuneResult(
            trajectory=[probe],
            optimal_concurrency=4,
            optimal_throughput_rps=100.0,
            optimal_throughput_tps=500.0,
            target="latency",
            error_budget=0.05,
            saturation_concurrency=16,
        )
        d = r.to_dict()
        assert len(d["trajectory"]) == 1
        assert d["optimal"]["concurrency"] == 4
        assert d["saturation_concurrency"] == 16
        assert d["target"] == "latency"


# ---------------------------------------------------------------------------
# _generate_concurrency_levels
# ---------------------------------------------------------------------------


class TestGenerateConcurrencyLevels:
    def test_power_of_two(self):
        assert _generate_concurrency_levels(16) == [1, 2, 4, 8, 16]

    def test_non_power_of_two(self):
        levels = _generate_concurrency_levels(10)
        assert levels == [1, 2, 4, 8, 10]

    def test_one(self):
        assert _generate_concurrency_levels(1) == [1]

    def test_large(self):
        levels = _generate_concurrency_levels(128)
        assert levels[0] == 1
        assert levels[-1] == 128
        assert 64 in levels


# ---------------------------------------------------------------------------
# find_optimal
# ---------------------------------------------------------------------------


class TestFindOptimal:
    def _probes(self):
        return [
            TuneProbe(1, 10.0, 50.0, 5.0, 10.0, 0.0, 50),
            TuneProbe(2, 18.0, 90.0, 6.0, 12.0, 0.0, 50),
            TuneProbe(4, 30.0, 150.0, 8.0, 20.0, 0.005, 50),
            TuneProbe(8, 35.0, 175.0, 15.0, 50.0, 0.02, 50),
            TuneProbe(16, 32.0, 160.0, 25.0, 80.0, 0.1, 50),
        ]

    def test_throughput_target(self):
        opt_c, opt_rps, opt_tps, sat = find_optimal(
            self._probes(), "throughput", 0.01
        )
        assert opt_c == 4  # highest RPS within 1% error
        assert opt_rps == 30.0

    def test_latency_target(self):
        opt_c, opt_rps, opt_tps, sat = find_optimal(
            self._probes(), "latency", 0.01
        )
        assert opt_c == 1  # lowest P99 within budget

    def test_cost_efficiency_target(self):
        opt_c, opt_rps, opt_tps, sat = find_optimal(
            self._probes(), "cost-efficiency", 0.01
        )
        # Best RPS/concurrency: 10/1=10, 18/2=9, 30/4=7.5
        assert opt_c == 1

    def test_saturation_detection(self):
        opt_c, opt_rps, opt_tps, sat = find_optimal(
            self._probes(), "throughput", 0.01
        )
        assert sat == 8  # first concurrency with error_rate > 1%

    def test_all_over_budget(self):
        probes = [TuneProbe(1, 10.0, 50.0, 5.0, 10.0, 0.5, 50)]
        opt_c, opt_rps, opt_tps, sat = find_optimal(probes, "throughput", 0.01)
        assert opt_c is None
        assert opt_rps == 0.0

    def test_empty_trajectory(self):
        opt_c, opt_rps, opt_tps, sat = find_optimal([], "throughput", 0.01)
        assert opt_c is None

    def test_unknown_target_defaults_to_throughput(self):
        probes = self._probes()
        opt_c, _, _, _ = find_optimal(probes, "unknown", 0.01)
        assert opt_c == 4  # same as throughput


# ---------------------------------------------------------------------------
# generate_config
# ---------------------------------------------------------------------------


class TestGenerateConfig:
    def test_basic_config(self):
        result = AutotuneResult(
            optimal_concurrency=8,
            optimal_throughput_rps=50.0,
        )
        config = generate_config(result, "http://localhost:8000", "gpt-4")
        assert config["base_url"] == "http://localhost:8000"
        assert config["model"] == "gpt-4"
        assert config["max_concurrency"] == 8
        assert config["request_rate"] == 45.0  # 50 * 0.9

    def test_no_optimal(self):
        result = AutotuneResult()
        config = generate_config(result, "http://localhost:8000", "gpt-4")
        assert "max_concurrency" not in config


# ---------------------------------------------------------------------------
# format_autotune_summary
# ---------------------------------------------------------------------------


class TestFormatSummary:
    def test_with_optimal(self):
        probes = [
            TuneProbe(1, 10.0, 50.0, 5.0, 10.0, 0.0, 50),
            TuneProbe(2, 18.0, 90.0, 6.0, 12.0, 0.0, 50),
        ]
        result = AutotuneResult(
            trajectory=probes,
            optimal_concurrency=2,
            optimal_throughput_rps=18.0,
            target="throughput",
            error_budget=0.01,
            saturation_concurrency=None,
        )
        text = format_autotune_summary(result)
        assert "Auto-Tune Results" in text
        assert "Optimal concurrency: 2" in text
        assert "throughput" in text.lower()

    def test_no_optimal(self):
        result = AutotuneResult(trajectory=[], target="throughput", error_budget=0.01)
        text = format_autotune_summary(result)
        assert "No optimal concurrency found" in text

    def test_with_saturation(self):
        result = AutotuneResult(
            trajectory=[],
            optimal_concurrency=4,
            optimal_throughput_rps=30.0,
            saturation_concurrency=16,
        )
        text = format_autotune_summary(result)
        assert "Saturation detected at concurrency: 16" in text


# ---------------------------------------------------------------------------
# _make_probe_args
# ---------------------------------------------------------------------------


class TestMakeProbeArgs:
    def test_overrides(self):
        base = Namespace(
            max_concurrency=1,
            num_prompts=100,
            html_report="/tmp/test.html",
            no_live=False,
            warmup=5,
        )
        probe = _make_probe_args(base, concurrency=8, num_prompts=30)
        assert probe.max_concurrency == 8
        assert probe.num_prompts == 30
        assert probe.html_report is None
        assert probe.no_live is True
        assert probe.warmup == 0
        # Original unchanged
        assert base.max_concurrency == 1


# ---------------------------------------------------------------------------
# run_autotune (mocked)
# ---------------------------------------------------------------------------


class TestRunAutotune:
    @pytest.mark.asyncio
    async def test_basic_run(self):
        call_count = 0

        async def mock_run_benchmark(args, base_url):
            nonlocal call_count
            call_count += 1
            conc = args.max_concurrency

            from xpyd_bench.bench.models import BenchmarkResult, RequestResult

            rr = RequestResult(
                success=True,
                latency_ms=100.0,
            )
            br = BenchmarkResult(
                requests=[rr] * 50,
                completed=50 if conc <= 4 else 45,
                failed=0 if conc <= 4 else 5,
                request_throughput=float(conc * 10),
                total_token_throughput=float(conc * 50),
                mean_e2el_ms=10.0 + conc,
                p99_e2el_ms=20.0 + conc * 2,
            )
            return {}, br

        base_args = Namespace(seed=42, input_len=128, output_len=64, endpoint="/v1/completions")

        with patch("xpyd_bench.autotune.run_benchmark", side_effect=mock_run_benchmark):
            from xpyd_bench.autotune import _set_probe_defaults

            _set_probe_defaults(base_args)
            result = await run_autotune(
                base_args,
                "http://localhost:8000",
                target="throughput",
                max_concurrency=8,
                error_budget=0.05,
                num_prompts=50,
            )

        assert len(result.trajectory) > 0
        assert result.optimal_concurrency is not None
        assert result.target == "throughput"

    @pytest.mark.asyncio
    async def test_early_stop_on_high_errors(self):
        async def mock_run_benchmark(args, base_url):
            from xpyd_bench.bench.models import BenchmarkResult, RequestResult

            rr = RequestResult(success=False, latency_ms=100.0)
            br = BenchmarkResult(
                requests=[rr] * 50,
                completed=0,
                failed=50,
                request_throughput=0.0,
                total_token_throughput=0.0,
                mean_e2el_ms=100.0,
                p99_e2el_ms=200.0,
            )
            return {}, br

        base_args = Namespace(seed=42, input_len=128, output_len=64, endpoint="/v1/completions")

        with patch("xpyd_bench.autotune.run_benchmark", side_effect=mock_run_benchmark):
            from xpyd_bench.autotune import _set_probe_defaults

            _set_probe_defaults(base_args)
            result = await run_autotune(
                base_args,
                "http://localhost:8000",
                max_concurrency=128,
                error_budget=0.01,
            )

        # Should have stopped early (not all 8 levels)
        assert len(result.trajectory) < 8
        assert result.optimal_concurrency is None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestAutotuneCLI:
    def test_missing_required_args(self):
        with pytest.raises(SystemExit):
            autotune_main([])

    def test_help(self):
        with pytest.raises(SystemExit) as exc_info:
            autotune_main(["--help"])
        assert exc_info.value.code == 0

    def test_json_output(self, tmp_path):
        result = AutotuneResult(
            trajectory=[TuneProbe(1, 10.0, 50.0, 5.0, 10.0, 0.0, 50)],
            optimal_concurrency=1,
            optimal_throughput_rps=10.0,
        )
        out = tmp_path / "result.json"
        out.write_text(json.dumps(result.to_dict(), indent=2))
        data = json.loads(out.read_text())
        assert data["optimal"]["concurrency"] == 1

    def test_generate_config_output(self, tmp_path):
        import yaml

        result = AutotuneResult(
            optimal_concurrency=4,
            optimal_throughput_rps=30.0,
        )
        config = generate_config(result, "http://localhost:8000", "gpt-4")
        out = tmp_path / "config.yaml"
        out.write_text(yaml.dump(config))
        loaded = yaml.safe_load(out.read_text())
        assert loaded["max_concurrency"] == 4
        assert loaded["request_rate"] == 27.0


# ---------------------------------------------------------------------------
# Serialization roundtrip
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_roundtrip(self):
        probe = TuneProbe(4, 100.0, 500.0, 10.0, 30.0, 0.005, 50)
        result = AutotuneResult(
            trajectory=[probe],
            optimal_concurrency=4,
            optimal_throughput_rps=100.0,
            optimal_throughput_tps=500.0,
            target="throughput",
            error_budget=0.01,
            saturation_concurrency=16,
        )
        data = json.loads(json.dumps(result.to_dict()))
        assert data["optimal"]["concurrency"] == 4
        assert len(data["trajectory"]) == 1
        assert data["saturation_concurrency"] == 16
