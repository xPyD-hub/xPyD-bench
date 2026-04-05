"""Tests for Benchmark Repeat Mode (M49)."""

from __future__ import annotations

import asyncio
import time
from argparse import Namespace
from unittest.mock import patch

from xpyd_bench.repeat import RepeatResult, print_repeat_summary, run_repeated_benchmark


def _make_result_dict(throughput=10.0, ttft=50.0):
    """Create a minimal benchmark result dict."""
    return {
        "completed": 100,
        "failed": 0,
        "request_throughput": throughput,
        "output_throughput": throughput * 10,
        "total_token_throughput": throughput * 15,
        "mean_ttft_ms": ttft,
        "mean_tpot_ms": 5.0,
        "mean_e2el_ms": 200.0,
        "total_duration_s": 10.0,
    }


def _make_args(repeat=3, repeat_delay=0.0):
    return Namespace(
        repeat=repeat,
        repeat_delay=repeat_delay,
        model="test-model",
        num_prompts=10,
        request_rate=float("inf"),
    )


class TestRepeatResult:
    def test_to_dict_basic(self):
        rr = RepeatResult(num_runs=3, completed_runs=3, repeat_delay=1.0)
        rr.per_run_results = [_make_result_dict(i * 5 + 5) for i in range(3)]
        d = rr.to_dict()
        assert d["repeat_runs"] == 3
        assert d["completed_runs"] == 3
        assert d["repeat_delay"] == 1.0
        assert "partial" not in d
        assert len(d["repeat_results"]) == 3

    def test_to_dict_partial(self):
        rr = RepeatResult(num_runs=5, completed_runs=2, partial=True)
        d = rr.to_dict()
        assert d["partial"] is True
        assert d["completed_runs"] == 2


class TestRunRepeatedBenchmark:
    def test_repeat_three_times(self):
        args = _make_args(repeat=3, repeat_delay=0.0)
        mock_results = [
            (_make_result_dict(10.0), None),
            (_make_result_dict(12.0), None),
            (_make_result_dict(11.0), None),
        ]
        call_count = 0

        async def fake_run(a, b):
            nonlocal call_count
            r = mock_results[call_count]
            call_count += 1
            return r

        with patch("xpyd_bench.bench.runner.run_benchmark", side_effect=fake_run):
            rr = asyncio.run(run_repeated_benchmark(args, "http://localhost:8000"))

        assert rr.completed_runs == 3
        assert rr.num_runs == 3
        assert not rr.partial
        assert rr.aggregate is not None
        assert rr.aggregate.num_runs == 3

    def test_repeat_with_delay(self):
        args = _make_args(repeat=2, repeat_delay=0.1)
        mock_results = [
            (_make_result_dict(10.0), None),
            (_make_result_dict(12.0), None),
        ]
        call_count = 0

        async def fake_run(a, b):
            nonlocal call_count
            r = mock_results[call_count]
            call_count += 1
            return r

        with patch("xpyd_bench.bench.runner.run_benchmark", side_effect=fake_run):
            start = time.monotonic()
            rr = asyncio.run(run_repeated_benchmark(args, "http://localhost:8000"))
            elapsed = time.monotonic() - start

        assert rr.completed_runs == 2
        assert elapsed >= 0.08

    def test_single_run_no_aggregate(self):
        args = _make_args(repeat=1)

        async def fake_run(a, b):
            return (_make_result_dict(), None)

        with patch("xpyd_bench.bench.runner.run_benchmark", side_effect=fake_run):
            rr = asyncio.run(run_repeated_benchmark(args, "http://localhost:8000"))

        assert rr.completed_runs == 1
        assert rr.aggregate is None


class TestPrintRepeatSummary:
    def test_prints_without_error(self, capsys):
        rr = RepeatResult(num_runs=2, completed_runs=2)
        rr.per_run_results = [_make_result_dict(10.0), _make_result_dict(12.0)]
        from xpyd_bench.aggregate import aggregate_results

        rr.aggregate = aggregate_results(rr.per_run_results)
        print_repeat_summary(rr)
        out = capsys.readouterr().out
        assert "Repeat Summary" in out
        assert "2/2 runs" in out

    def test_prints_partial(self, capsys):
        rr = RepeatResult(num_runs=5, completed_runs=1, partial=True)
        rr.per_run_results = [_make_result_dict()]
        print_repeat_summary(rr)
        out = capsys.readouterr().out
        assert "Interrupted" in out


class TestCLIArgs:
    def test_repeat_arg_parsed(self):
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--repeat", "5", "--repeat-delay", "2.5"])
        assert args.repeat == 5
        assert args.repeat_delay == 2.5

    def test_repeat_default(self):
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args([])
        assert args.repeat == 1
        assert args.repeat_delay == 0.0


class TestYAMLConfig:
    def test_repeat_in_known_keys(self):
        from xpyd_bench.config_cmd import _KNOWN_KEYS

        assert "repeat" in _KNOWN_KEYS
        assert "repeat_delay" in _KNOWN_KEYS
