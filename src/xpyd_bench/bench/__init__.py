"""Benchmark runner — sends requests and collects metrics."""

from __future__ import annotations

from xpyd_bench.bench.models import BenchmarkResult, RequestResult
from xpyd_bench.bench.runner import run_benchmark

__all__ = ["BenchmarkResult", "RequestResult", "run_benchmark"]
