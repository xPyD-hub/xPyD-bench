"""Tests for concurrency sweep mode (M44)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from xpyd_bench.sweep import (
    SweepLevelResult,
    SweepResult,
    _find_optimal,
    parse_concurrency_range,
    sweep_main,
)


class TestParseConcurrencyRange:
    """Tests for parse_concurrency_range."""

    def test_comma_separated(self) -> None:
        assert parse_concurrency_range("1,2,4,8,16") == [1, 2, 4, 8, 16]

    def test_comma_separated_dedup_sort(self) -> None:
        assert parse_concurrency_range("8,2,4,2,1") == [1, 2, 4, 8]

    def test_exponential_range(self) -> None:
        assert parse_concurrency_range("1:32:2x") == [1, 2, 4, 8, 16, 32]

    def test_linear_range(self) -> None:
        assert parse_concurrency_range("2:10:2") == [2, 4, 6, 8, 10]

    def test_single_value(self) -> None:
        assert parse_concurrency_range("4") == [4]

    def test_invalid_range_spec(self) -> None:
        with pytest.raises(ValueError, match="Range spec"):
            parse_concurrency_range("1:2")

    def test_exponential_factor_too_small(self) -> None:
        with pytest.raises(ValueError, match="factor must be >= 2"):
            parse_concurrency_range("1:32:1x")

    def test_negative_step(self) -> None:
        with pytest.raises(ValueError, match="Step must be >= 1"):
            parse_concurrency_range("1:10:0")

    def test_no_valid_values(self) -> None:
        with pytest.raises(ValueError, match="No valid"):
            parse_concurrency_range("-1,-2")


class TestFindOptimal:
    """Tests for _find_optimal."""

    def test_finds_highest_throughput_under_error_threshold(self) -> None:
        levels = [
            SweepLevelResult(1, 10.0, 100.0, 50.0, 100.0, 0.0, 100),
            SweepLevelResult(4, 30.0, 300.0, 60.0, 120.0, 0.02, 100),
            SweepLevelResult(8, 25.0, 250.0, 80.0, 200.0, 0.10, 100),
        ]
        conc, tps = _find_optimal(levels, max_error_rate=0.05)
        assert conc == 4
        assert tps == 30.0

    def test_no_level_meets_threshold(self) -> None:
        levels = [
            SweepLevelResult(1, 10.0, 100.0, 50.0, 100.0, 0.10, 100),
            SweepLevelResult(4, 30.0, 300.0, 60.0, 120.0, 0.20, 100),
        ]
        conc, tps = _find_optimal(levels, max_error_rate=0.05)
        assert conc is None
        assert tps == 0.0

    def test_all_levels_meet_threshold(self) -> None:
        levels = [
            SweepLevelResult(1, 10.0, 100.0, 50.0, 100.0, 0.0, 100),
            SweepLevelResult(4, 40.0, 400.0, 60.0, 120.0, 0.01, 100),
            SweepLevelResult(8, 35.0, 350.0, 70.0, 150.0, 0.03, 100),
        ]
        conc, tps = _find_optimal(levels, max_error_rate=0.05)
        assert conc == 4  # highest throughput


class TestSweepResult:
    """Tests for SweepResult serialization."""

    def test_to_dict_structure(self) -> None:
        lv = SweepLevelResult(
            concurrency=4,
            throughput_rps=30.0,
            throughput_tps=300.0,
            mean_latency_ms=60.0,
            p99_latency_ms=120.0,
            error_rate=0.02,
            total_requests=100,
        )
        sweep = SweepResult(
            levels=[lv],
            optimal_concurrency=4,
            optimal_throughput_rps=30.0,
            max_error_rate=0.05,
        )
        d = sweep.to_dict()
        assert d["optimal"]["concurrency"] == 4
        assert d["optimal"]["throughput_rps"] == 30.0
        assert len(d["levels"]) == 1
        assert d["levels"][0]["concurrency"] == 4
        assert d["max_error_rate_threshold"] == 0.05

    def test_level_to_dict(self) -> None:
        lv = SweepLevelResult(
            concurrency=8,
            throughput_rps=25.123,
            throughput_tps=251.456,
            mean_latency_ms=80.789,
            p99_latency_ms=200.321,
            error_rate=0.0312,
            total_requests=100,
        )
        d = lv.to_dict()
        assert d["concurrency"] == 8
        assert d["throughput_rps"] == 25.12
        assert d["error_rate"] == 0.0312


class TestSweepCLI:
    """Tests for sweep CLI integration."""

    def test_sweep_output_file(self, tmp_path: Path) -> None:
        """Sweep writes JSON output when --sweep-output is given."""
        output_file = tmp_path / "sweep.json"

        # Mock run_benchmark to return canned results
        async def mock_run_benchmark(args, base_url):
            from xpyd_bench.bench.models import BenchmarkResult

            result = BenchmarkResult(
                completed=100, failed=0
            )
            result_dict = {
                "metrics": {
                    "request_throughput": 10.0 * (args.max_concurrency or 1),
                    "output_throughput": 100.0 * (args.max_concurrency or 1),
                    "e2e_latency": {
                        "mean": 50.0 + (args.max_concurrency or 1),
                        "P99": 100.0 + (args.max_concurrency or 1) * 2,
                    },
                }
            }
            return result_dict, result

        with patch(
            "xpyd_bench.sweep.run_benchmark",
            side_effect=mock_run_benchmark,
        ):
            sweep_main([
                "--base-url", "http://localhost:8000",
                "--concurrency-range", "1,2,4",
                "--sweep-prompts", "10",
                "--sweep-output", str(output_file),
            ])

        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert "levels" in data
        assert len(data["levels"]) == 3
        assert data["optimal"]["concurrency"] is not None

    def test_sweep_exponential_range(self, tmp_path: Path) -> None:
        """Sweep with exponential range notation."""
        output_file = tmp_path / "sweep_exp.json"

        async def mock_run_benchmark(args, base_url):
            from xpyd_bench.bench.models import BenchmarkResult

            result = BenchmarkResult(
                completed=10, failed=0
            )
            result_dict = {
                "metrics": {
                    "request_throughput": 5.0,
                    "output_throughput": 50.0,
                    "e2e_latency": {"mean": 100.0, "P99": 200.0},
                }
            }
            return result_dict, result

        with patch(
            "xpyd_bench.sweep.run_benchmark",
            side_effect=mock_run_benchmark,
        ):
            sweep_main([
                "--base-url", "http://localhost:8000",
                "--concurrency-range", "1:8:2x",
                "--sweep-prompts", "5",
                "--sweep-output", str(output_file),
            ])

        data = json.loads(output_file.read_text())
        concurrencies = [lv["concurrency"] for lv in data["levels"]]
        assert concurrencies == [1, 2, 4, 8]
