"""Tests for multi-model comparison mode (M75)."""

from __future__ import annotations

import asyncio
import json
from argparse import Namespace
from unittest.mock import patch

import pytest

from xpyd_bench.bench.models import BenchmarkResult, RequestResult
from xpyd_bench.model_compare import (
    ModelComparisonResult,
    ModelSignificance,
    MultiModelResult,
    _compute_significance,
    export_model_compare_json,
    export_model_compare_markdown,
    format_model_compare_markdown,
    format_model_compare_summary,
    run_model_compare,
)


def _make_result(model: str, latencies: list[float]) -> tuple[dict, BenchmarkResult]:
    """Create a fake benchmark result for testing."""
    requests = []
    for lat in latencies:
        requests.append(RequestResult(
            latency_ms=lat,
            ttft_ms=lat * 0.3,
            tpot_ms=lat * 0.05,
            prompt_tokens=10,
            completion_tokens=20,
            success=True,
        ))
    br = BenchmarkResult(
        model=model,
        base_url="http://localhost:8000",
        completed=len(latencies),
        failed=0,
        total_duration_s=5.0,
        request_throughput=len(latencies) / 5.0,
        output_throughput=20.0 * len(latencies) / 5.0,
        total_token_throughput=30.0 * len(latencies) / 5.0,
        mean_ttft_ms=sum(lat * 0.3 for lat in latencies) / len(latencies),
        p50_ttft_ms=sorted(lat * 0.3 for lat in latencies)[len(latencies) // 2],
        p90_ttft_ms=sorted(lat * 0.3 for lat in latencies)[int(len(latencies) * 0.9)],
        p99_ttft_ms=sorted(lat * 0.3 for lat in latencies)[-1],
        mean_tpot_ms=sum(lat * 0.05 for lat in latencies) / len(latencies),
        mean_e2el_ms=sum(latencies) / len(latencies),
        p50_e2el_ms=sorted(latencies)[len(latencies) // 2],
        p90_e2el_ms=sorted(latencies)[int(len(latencies) * 0.9)],
        p99_e2el_ms=sorted(latencies)[-1],
        requests=requests,
    )
    rd = {
        "model": model,
        "completed": br.completed,
        "failed": br.failed,
        "request_throughput": br.request_throughput,
        "output_throughput": br.output_throughput,
        "total_token_throughput": br.total_token_throughput,
        "mean_ttft_ms": br.mean_ttft_ms,
        "p50_ttft_ms": br.p50_ttft_ms,
        "p90_ttft_ms": br.p90_ttft_ms,
        "p99_ttft_ms": br.p99_ttft_ms,
        "mean_tpot_ms": br.mean_tpot_ms,
        "mean_e2el_ms": br.mean_e2el_ms,
        "p50_e2el_ms": br.p50_e2el_ms,
        "p90_e2el_ms": br.p90_e2el_ms,
        "p99_e2el_ms": br.p99_e2el_ms,
    }
    return rd, br


class TestMultiModelResult:
    """Tests for MultiModelResult dataclass."""

    def test_to_dict_empty(self):
        r = MultiModelResult()
        d = r.to_dict()
        assert d["models"] == []
        assert d["results"] == []
        assert "comparisons" not in d

    def test_to_dict_with_comparisons(self):
        from xpyd_bench.compare import ComparisonResult
        cmp = ComparisonResult(metrics=[], has_regression=False, threshold_pct=5.0)
        mc = ModelComparisonResult(
            baseline_model="a",
            candidate_model="b",
            comparison=cmp,
            significance=[
                ModelSignificance(metric="ttft_ms", u_stat=10.0, p_value=0.03, significant=True),
            ],
        )
        r = MultiModelResult(
            models=["a", "b"],
            base_url="http://localhost",
            comparisons=[mc],
        )
        d = r.to_dict()
        assert len(d["comparisons"]) == 1
        assert d["comparisons"][0]["baseline_model"] == "a"
        assert d["comparisons"][0]["significance"][0]["significant"] is True


class TestRunModelCompare:
    """Tests for run_model_compare."""

    def test_basic_comparison(self):
        latencies_a = [100.0, 110.0, 105.0, 102.0, 108.0]
        latencies_b = [200.0, 210.0, 205.0, 202.0, 208.0]
        rd_a, br_a = _make_result("model-a", latencies_a)
        rd_b, br_b = _make_result("model-b", latencies_b)

        call_count = 0

        async def fake_run(args, base_url):
            nonlocal call_count
            if call_count == 0:
                call_count += 1
                return rd_a, br_a
            call_count += 1
            return rd_b, br_b

        args = Namespace(base_url="http://localhost:8000", model="x")

        with patch("xpyd_bench.model_compare.run_benchmark", side_effect=fake_run):
            result = asyncio.run(
                run_model_compare(args, ["model-a", "model-b"], threshold_pct=5.0)
            )

        assert len(result.models) == 2
        assert len(result.results) == 2
        assert len(result.comparisons) == 1
        assert result.comparisons[0].baseline_model == "model-a"
        assert result.comparisons[0].candidate_model == "model-b"

    def test_single_model_no_comparison(self):
        rd, br = _make_result("model-a", [100.0, 110.0])

        async def fake_run(args, base_url):
            return rd, br

        args = Namespace(base_url="http://localhost:8000", model="x")

        with patch("xpyd_bench.model_compare.run_benchmark", side_effect=fake_run):
            result = asyncio.run(
                run_model_compare(args, ["model-a"], threshold_pct=5.0)
            )

        assert len(result.comparisons) == 0

    def test_three_models(self):
        results_data = []
        for name in ["a", "b", "c"]:
            results_data.append(_make_result(name, [100.0, 110.0, 105.0]))

        idx = 0

        async def fake_run(args, base_url):
            nonlocal idx
            rd, br = results_data[idx]
            idx += 1
            return rd, br

        args = Namespace(base_url="http://localhost:8000", model="x")

        with patch("xpyd_bench.model_compare.run_benchmark", side_effect=fake_run):
            result = asyncio.run(
                run_model_compare(args, ["a", "b", "c"], threshold_pct=5.0)
            )

        assert len(result.comparisons) == 2
        assert result.comparisons[0].baseline_model == "a"
        assert result.comparisons[0].candidate_model == "b"
        assert result.comparisons[1].baseline_model == "a"
        assert result.comparisons[1].candidate_model == "c"


class TestSignificance:
    """Tests for statistical significance computation."""

    def test_identical_distributions(self):
        br1 = BenchmarkResult(requests=[
            RequestResult(latency_ms=100.0, ttft_ms=30.0, tpot_ms=5.0)
            for _ in range(10)
        ])
        br2 = BenchmarkResult(requests=[
            RequestResult(latency_ms=100.0, ttft_ms=30.0, tpot_ms=5.0)
            for _ in range(10)
        ])
        sigs = _compute_significance(br1, br2)
        assert len(sigs) == 3
        for s in sigs:
            assert not s.significant  # identical → not significant

    def test_very_different_distributions(self):
        br1 = BenchmarkResult(requests=[
            RequestResult(latency_ms=float(i), ttft_ms=float(i), tpot_ms=float(i))
            for i in range(1, 21)
        ])
        br2 = BenchmarkResult(requests=[
            RequestResult(latency_ms=float(i + 100), ttft_ms=float(i + 100), tpot_ms=float(i + 100))
            for i in range(1, 21)
        ])
        sigs = _compute_significance(br1, br2)
        for s in sigs:
            assert s.significant

    def test_insufficient_data(self):
        br1 = BenchmarkResult(requests=[RequestResult(latency_ms=100.0)])
        br2 = BenchmarkResult(requests=[RequestResult(latency_ms=200.0)])
        sigs = _compute_significance(br1, br2)
        assert len(sigs) == 0  # not enough data


class TestFormatting:
    """Tests for summary and markdown formatting."""

    def test_format_summary_empty(self):
        r = MultiModelResult()
        s = format_model_compare_summary(r)
        assert "No results" in s

    def test_format_summary_with_results(self):
        rd, br = _make_result("test-model", [100.0, 110.0, 105.0])
        r = MultiModelResult(
            models=["test-model"],
            base_url="http://localhost",
            results=[br],
            raw_dicts=[rd],
        )
        s = format_model_compare_summary(r)
        assert "test-model" in s
        assert "Multi-Model Comparison" in s

    def test_format_markdown_empty(self):
        r = MultiModelResult()
        md = format_model_compare_markdown(r)
        assert "No results" in md

    def test_format_markdown_with_results(self):
        rd_a, br_a = _make_result("model-a", [100.0, 110.0])
        rd_b, br_b = _make_result("model-b", [200.0, 210.0])
        from xpyd_bench.compare import ComparisonResult
        mc = ModelComparisonResult(
            baseline_model="model-a",
            candidate_model="model-b",
            comparison=ComparisonResult(metrics=[], has_regression=True, threshold_pct=5.0),
            significance=[
                ModelSignificance(metric="latency_ms", u_stat=0.0, p_value=0.001, significant=True),
            ],
        )
        r = MultiModelResult(
            models=["model-a", "model-b"],
            base_url="http://localhost",
            results=[br_a, br_b],
            raw_dicts=[rd_a, rd_b],
            comparisons=[mc],
        )
        md = format_model_compare_markdown(r)
        assert "model-a" in md
        assert "model-b" in md
        assert "Statistical Significance" in md


class TestExport:
    """Tests for JSON and Markdown export."""

    def test_export_json(self, tmp_path):
        rd, br = _make_result("test", [100.0])
        r = MultiModelResult(
            models=["test"],
            base_url="http://localhost",
            results=[br],
            raw_dicts=[rd],
        )
        p = export_model_compare_json(r, tmp_path / "out.json")
        assert p.exists()
        data = json.loads(p.read_text())
        assert data["models"] == ["test"]

    def test_export_markdown(self, tmp_path):
        rd, br = _make_result("test", [100.0])
        r = MultiModelResult(
            models=["test"],
            base_url="http://localhost",
            results=[br],
            raw_dicts=[rd],
        )
        p = export_model_compare_markdown(r, tmp_path / "out.md")
        assert p.exists()
        content = p.read_text()
        assert "test" in content


class TestCLIIntegration:
    """Tests for CLI integration."""

    def test_model_compare_in_subcommands(self):
        from xpyd_bench.main import _SUBCOMMANDS
        assert "model-compare" in _SUBCOMMANDS

    def test_cli_requires_models(self):
        from xpyd_bench.cli import model_compare_main
        with pytest.raises(SystemExit):
            model_compare_main(["--base-url", "http://localhost"])

    def test_cli_requires_at_least_two_models(self):
        from xpyd_bench.cli import model_compare_main
        with pytest.raises(SystemExit):
            model_compare_main(["--models", "a", "--base-url", "http://localhost"])
