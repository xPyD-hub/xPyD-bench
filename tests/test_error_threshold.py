"""Tests for M83: Error Threshold Abort."""

from __future__ import annotations

import argparse

from xpyd_bench.cli import _add_vllm_compat_args


def test_cli_max_error_rate_flag():
    """CLI parses --max-error-rate flag."""
    parser = argparse.ArgumentParser()
    _add_vllm_compat_args(parser)
    args = parser.parse_args(["--max-error-rate", "0.3"])
    assert args.max_error_rate == 0.3


def test_cli_max_error_rate_default():
    """--max-error-rate defaults to None (disabled)."""
    parser = argparse.ArgumentParser()
    _add_vllm_compat_args(parser)
    args = parser.parse_args([])
    assert args.max_error_rate is None


def test_cli_max_error_rate_window_flag():
    """CLI parses --max-error-rate-window flag."""
    parser = argparse.ArgumentParser()
    _add_vllm_compat_args(parser)
    args = parser.parse_args(["--max-error-rate-window", "20"])
    assert args.max_error_rate_window == 20


def test_cli_max_error_rate_window_default():
    """--max-error-rate-window defaults to 10."""
    parser = argparse.ArgumentParser()
    _add_vllm_compat_args(parser)
    args = parser.parse_args([])
    assert args.max_error_rate_window == 10


def test_error_abort_sets_partial_and_reason():
    """BenchmarkResult with aborted_reason should have partial=True."""
    from xpyd_bench.bench.models import BenchmarkResult

    result = BenchmarkResult()
    result.partial = True
    result.aborted_reason = "Error rate 50.0% exceeded threshold 30.0% after 10 requests"
    assert result.partial is True
    assert "exceeded threshold" in result.aborted_reason


def test_aborted_reason_in_serialization():
    """aborted_reason should appear in dict serialization."""
    from dataclasses import asdict

    from xpyd_bench.bench.models import BenchmarkResult

    result = BenchmarkResult()
    result.aborted_reason = "Error rate too high"
    d = asdict(result)
    assert d["aborted_reason"] == "Error rate too high"


def test_aborted_reason_none_by_default():
    """aborted_reason is None by default."""
    from xpyd_bench.bench.models import BenchmarkResult

    result = BenchmarkResult()
    assert result.aborted_reason is None


def test_known_config_keys_include_error_rate():
    """max_error_rate and max_error_rate_window are in known config keys."""
    from xpyd_bench.config_cmd import _KNOWN_KEYS

    assert "max_error_rate" in _KNOWN_KEYS
    assert "max_error_rate_window" in _KNOWN_KEYS
