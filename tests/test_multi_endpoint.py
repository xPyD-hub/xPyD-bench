"""Tests for M15: Multi-Endpoint Comparison Mode."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from xpyd_bench.bench.models import BenchmarkResult
from xpyd_bench.multi import (
    MultiEndpointResult,
    export_multi_json,
    export_multi_markdown,
    format_multi_markdown,
    format_multi_summary,
    run_multi_benchmark,
)


def _make_bench_result(
    base_url: str = "http://localhost:8000",
    completed: int = 10,
    mean_ttft: float = 5.0,
    request_throughput: float = 100.0,
) -> BenchmarkResult:
    """Build a minimal BenchmarkResult for testing."""
    return BenchmarkResult(
        backend="openai",
        base_url=base_url,
        endpoint="/v1/completions",
        model="test-model",
        num_prompts=10,
        completed=completed,
        failed=0,
        total_duration_s=1.0,
        request_throughput=request_throughput,
        output_throughput=500.0,
        total_token_throughput=700.0,
        mean_ttft_ms=mean_ttft,
        p50_ttft_ms=mean_ttft,
        p90_ttft_ms=mean_ttft + 1,
        p95_ttft_ms=mean_ttft + 2,
        p99_ttft_ms=mean_ttft + 3,
        mean_tpot_ms=2.0,
        p50_tpot_ms=2.0,
        p90_tpot_ms=3.0,
        p95_tpot_ms=3.0,
        p99_tpot_ms=3.0,
        mean_itl_ms=2.0,
        p50_itl_ms=2.0,
        p90_itl_ms=3.0,
        p95_itl_ms=3.0,
        p99_itl_ms=3.0,
        mean_e2el_ms=50.0,
        p50_e2el_ms=50.0,
        p90_e2el_ms=60.0,
        p95_e2el_ms=65.0,
        p99_e2el_ms=70.0,
        requests=[],
    )


def _make_result_dict(
    base_url: str = "http://localhost:8000",
    request_throughput: float = 100.0,
    mean_ttft: float = 5.0,
) -> dict:
    """Build a minimal result dict matching runner._to_dict output."""
    return {
        "backend": "openai",
        "base_url": base_url,
        "endpoint": "/v1/completions",
        "model": "test-model",
        "num_prompts": 10,
        "request_rate": float("inf"),
        "max_concurrency": None,
        "input_len": 256,
        "output_len": 128,
        "total_duration_s": 1.0,
        "completed": 10,
        "failed": 0,
        "total_input_tokens": 2560,
        "total_output_tokens": 500,
        "request_throughput": request_throughput,
        "output_throughput": 500.0,
        "total_token_throughput": 700.0,
        "mean_ttft_ms": mean_ttft,
        "median_ttft_ms": mean_ttft,
        "p50_ttft_ms": mean_ttft,
        "p90_ttft_ms": mean_ttft + 1,
        "p95_ttft_ms": mean_ttft + 2,
        "p99_ttft_ms": mean_ttft + 3,
        "mean_tpot_ms": 2.0,
        "median_tpot_ms": 2.0,
        "p50_tpot_ms": 2.0,
        "p90_tpot_ms": 3.0,
        "p95_tpot_ms": 3.0,
        "p99_tpot_ms": 3.0,
        "mean_itl_ms": 2.0,
        "median_itl_ms": 2.0,
        "p50_itl_ms": 2.0,
        "p90_itl_ms": 3.0,
        "p95_itl_ms": 3.0,
        "p99_itl_ms": 3.0,
        "mean_e2el_ms": 50.0,
        "median_e2el_ms": 50.0,
        "p50_e2el_ms": 50.0,
        "p90_e2el_ms": 60.0,
        "p95_e2el_ms": 65.0,
        "p99_e2el_ms": 70.0,
    }


# ---------------------------------------------------------------------------
# MultiEndpointResult dataclass
# ---------------------------------------------------------------------------


class TestMultiEndpointResult:
    def test_to_dict_empty(self) -> None:
        m = MultiEndpointResult()
        d = m.to_dict()
        assert d["endpoints"] == []
        assert d["results"] == []

    def test_to_dict_with_results(self) -> None:
        m = MultiEndpointResult(
            endpoints=["http://a", "http://b"],
            raw_dicts=[{"a": 1}, {"b": 2}],
        )
        d = m.to_dict()
        assert len(d["results"]) == 2
        assert d["endpoints"] == ["http://a", "http://b"]


# ---------------------------------------------------------------------------
# run_multi_benchmark
# ---------------------------------------------------------------------------


class TestRunMultiBenchmark:
    @pytest.mark.asyncio
    async def test_runs_all_endpoints(self) -> None:
        urls = ["http://ep1:8000", "http://ep2:8000"]
        call_count = 0

        async def mock_run(args, base_url):
            nonlocal call_count
            call_count += 1
            return (
                _make_result_dict(base_url=base_url),
                _make_bench_result(base_url=base_url),
            )

        from argparse import Namespace

        args = Namespace(
            backend="openai",
            base_url=None,
            host="127.0.0.1",
            port=8000,
            endpoint="/v1/completions",
            model="m",
            num_prompts=10,
            request_rate=float("inf"),
            burstiness=1.0,
            max_concurrency=None,
            input_len=256,
            output_len=128,
            seed=0,
            disable_tqdm=True,
            save_result=False,
            warmup=0,
            api_key=None,
            custom_headers={},
            timeout=300.0,
            retries=0,
            retry_delay=1.0,
            config=None,
            dataset_path=None,
            dataset_name="random",
            rich_progress=False,
            scenario=None,
        )

        with patch("xpyd_bench.multi.run_benchmark", side_effect=mock_run):
            multi = await run_multi_benchmark(args, urls, threshold_pct=5.0)

        assert call_count == 2
        assert len(multi.results) == 2
        assert len(multi.raw_dicts) == 2
        assert multi.endpoints == urls

    @pytest.mark.asyncio
    async def test_comparisons_generated(self) -> None:
        urls = ["http://ep1:8000", "http://ep2:8000", "http://ep3:8000"]

        async def mock_run(args, base_url):
            return (
                _make_result_dict(base_url=base_url),
                _make_bench_result(base_url=base_url),
            )

        from argparse import Namespace

        args = Namespace(
            backend="openai",
            base_url=None,
            host="127.0.0.1",
            port=8000,
            endpoint="/v1/completions",
            model="m",
            num_prompts=10,
            request_rate=float("inf"),
            burstiness=1.0,
            max_concurrency=None,
            input_len=256,
            output_len=128,
            seed=0,
            disable_tqdm=True,
            save_result=False,
            warmup=0,
            api_key=None,
            custom_headers={},
            timeout=300.0,
            retries=0,
            retry_delay=1.0,
            config=None,
            dataset_path=None,
            dataset_name="random",
            rich_progress=False,
            scenario=None,
        )

        with patch("xpyd_bench.multi.run_benchmark", side_effect=mock_run):
            multi = await run_multi_benchmark(args, urls)

        # 3 endpoints → 2 comparisons (EP1 vs EP0, EP2 vs EP0)
        assert len(multi.comparisons) == 2


# ---------------------------------------------------------------------------
# format_multi_summary
# ---------------------------------------------------------------------------


class TestFormatMultiSummary:
    def test_empty(self) -> None:
        m = MultiEndpointResult()
        text = format_multi_summary(m)
        assert "No results" in text

    def test_two_endpoints(self) -> None:
        m = MultiEndpointResult(
            endpoints=["http://a", "http://b"],
            results=[
                _make_bench_result("http://a"),
                _make_bench_result("http://b"),
            ],
            raw_dicts=[
                _make_result_dict("http://a"),
                _make_result_dict("http://b"),
            ],
        )
        from xpyd_bench.compare import compare

        m.comparisons = [compare(m.raw_dicts[0], m.raw_dicts[1])]
        text = format_multi_summary(m)
        assert "EP0" in text
        assert "EP1" in text
        assert "http://a" in text
        assert "http://b" in text

    def test_regression_shown(self) -> None:
        m = MultiEndpointResult(
            endpoints=["http://a", "http://b"],
            results=[
                _make_bench_result("http://a", mean_ttft=5.0),
                _make_bench_result("http://b", mean_ttft=50.0),
            ],
            raw_dicts=[
                _make_result_dict("http://a", mean_ttft=5.0),
                _make_result_dict("http://b", mean_ttft=50.0),
            ],
        )
        from xpyd_bench.compare import compare

        m.comparisons = [compare(m.raw_dicts[0], m.raw_dicts[1], threshold_pct=5.0)]
        text = format_multi_summary(m)
        assert "regressions" in text.lower() or "⚠️" in text


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------


class TestExportMultiJson:
    def test_creates_file(self, tmp_path: Path) -> None:
        m = MultiEndpointResult(
            endpoints=["http://a"],
            raw_dicts=[{"x": 1}],
        )
        p = export_multi_json(m, tmp_path / "out.json")
        assert p.exists()
        data = json.loads(p.read_text())
        assert data["endpoints"] == ["http://a"]

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        m = MultiEndpointResult()
        p = export_multi_json(m, tmp_path / "sub" / "dir" / "out.json")
        assert p.exists()


class TestExportMultiMarkdown:
    def test_creates_file(self, tmp_path: Path) -> None:
        m = MultiEndpointResult(
            endpoints=["http://a", "http://b"],
            results=[
                _make_bench_result("http://a"),
                _make_bench_result("http://b"),
            ],
        )
        p = export_multi_markdown(m, tmp_path / "out.md")
        assert p.exists()
        content = p.read_text()
        assert "http://a" in content
        assert "http://b" in content

    def test_table_structure(self, tmp_path: Path) -> None:
        m = MultiEndpointResult(
            endpoints=["http://a", "http://b"],
            results=[
                _make_bench_result("http://a"),
                _make_bench_result("http://b"),
            ],
        )
        p = export_multi_markdown(m, tmp_path / "out.md")
        lines = p.read_text().strip().split("\n")
        assert len(lines) >= 3  # header + separator + at least 1 data row
        assert lines[0].startswith("|")
        assert "---" in lines[1]

    def test_empty_results(self, tmp_path: Path) -> None:
        m = MultiEndpointResult()
        content = format_multi_markdown(m)
        assert "No results" in content


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


class TestMultiCliArgs:
    def _make_parser(self):
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        parser.add_argument("--endpoints", type=str, nargs="+", required=True)
        parser.add_argument("--threshold", type=float, default=5.0)
        parser.add_argument("--json-output", type=str, default=None)
        parser.add_argument("--markdown-output", type=str, default=None)
        _add_vllm_compat_args(parser)
        return parser

    def test_endpoints_parsed(self) -> None:
        parser = self._make_parser()
        args = parser.parse_args(
            ["--endpoints", "http://a", "http://b", "--model", "m"]
        )
        assert args.endpoints == ["http://a", "http://b"]

    def test_threshold_default(self) -> None:
        parser = self._make_parser()
        args = parser.parse_args(["--endpoints", "http://a", "http://b"])
        assert args.threshold == 5.0

    def test_json_output(self) -> None:
        parser = self._make_parser()
        args = parser.parse_args(
            ["--endpoints", "http://a", "http://b", "--json-output", "/tmp/o.json"]
        )
        assert args.json_output == "/tmp/o.json"

    def test_markdown_output(self) -> None:
        parser = self._make_parser()
        args = parser.parse_args(
            ["--endpoints", "http://a", "http://b", "--markdown-output", "/tmp/o.md"]
        )
        assert args.markdown_output == "/tmp/o.md"
