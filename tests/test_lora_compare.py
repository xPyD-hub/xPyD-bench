"""Tests for multi-LoRA endpoint benchmarking (M89)."""

from __future__ import annotations

import json
from argparse import Namespace
from unittest.mock import patch

import pytest

from xpyd_bench.bench.models import BenchmarkResult, RequestResult
from xpyd_bench.lora_compare import (
    AdapterOverhead,
    AdapterResult,
    LoRACompareResult,
    _compute_adapter_significance,
    _mean_ttft,
    export_lora_compare_json,
    export_lora_compare_markdown,
    format_lora_compare_markdown,
    format_lora_compare_summary,
    run_lora_compare,
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


class TestLoRACompareResult:
    """Tests for LoRACompareResult dataclass."""

    def test_empty_result_to_dict(self):
        result = LoRACompareResult()
        d = result.to_dict()
        assert d["adapters"] == []
        assert d["sequential_results"] == []
        assert d["interleave"] is False

    def test_to_dict_with_adapters(self):
        rd1, br1 = _make_result("adapter1", [100.0, 120.0, 110.0])
        rd2, br2 = _make_result("adapter2", [90.0, 95.0, 88.0])
        result = LoRACompareResult(
            base_url="http://localhost:8000",
            adapters=["adapter1", "adapter2"],
            adapter_results=[
                AdapterResult(adapter="adapter1", result=br1, raw_dict=rd1),
                AdapterResult(adapter="adapter2", result=br2, raw_dict=rd2),
            ],
        )
        d = result.to_dict()
        assert len(d["sequential_results"]) == 2
        assert d["sequential_results"][0]["adapter"] == "adapter1"
        assert d["sequential_results"][1]["adapter"] == "adapter2"

    def test_to_dict_with_overhead(self):
        result = LoRACompareResult(
            overhead=AdapterOverhead(
                sequential_mean_ttft_ms=30.0,
                interleaved_mean_ttft_ms=35.0,
                switching_overhead_ms=5.0,
                switching_overhead_pct=16.7,
            ),
        )
        d = result.to_dict()
        assert d["overhead"]["switching_overhead_ms"] == 5.0
        assert d["overhead"]["switching_overhead_pct"] == 16.7


class TestAdapterOverhead:
    """Tests for AdapterOverhead dataclass."""

    def test_to_dict(self):
        oh = AdapterOverhead(
            sequential_mean_ttft_ms=25.0,
            interleaved_mean_ttft_ms=30.0,
            switching_overhead_ms=5.0,
            switching_overhead_pct=20.0,
        )
        d = oh.to_dict()
        assert d["sequential_mean_ttft_ms"] == 25.0
        assert d["switching_overhead_pct"] == 20.0


class TestComputeAdapterSignificance:
    """Tests for _compute_adapter_significance."""

    def test_significance_with_different_distributions(self):
        _, br1 = _make_result("adapter1", [100.0, 110.0, 105.0, 108.0, 102.0])
        _, br2 = _make_result("adapter2", [200.0, 210.0, 205.0, 208.0, 202.0])
        sigs = _compute_adapter_significance(br1, br2)
        assert len(sigs) > 0
        # With such different distributions, at least one should be significant
        any_sig = any(s["significant"] for s in sigs)
        assert any_sig

    def test_significance_with_similar_distributions(self):
        _, br1 = _make_result("adapter1", [100.0, 101.0, 100.5])
        _, br2 = _make_result("adapter2", [100.0, 101.0, 100.5])
        sigs = _compute_adapter_significance(br1, br2)
        # Same data → should not be significant
        for s in sigs:
            assert not s["significant"]


class TestMeanTtft:
    """Tests for _mean_ttft helper."""

    def test_from_requests(self):
        _, br = _make_result("adapter1", [100.0, 200.0])
        result = _mean_ttft(br)
        # ttft = lat * 0.3 → 30, 60 → mean = 45
        assert abs(result - 45.0) < 0.01

    def test_fallback_to_attribute(self):
        br = BenchmarkResult(mean_ttft_ms=42.0)
        result = _mean_ttft(br)
        assert result == 42.0


class TestRunLoRACompare:
    """Tests for run_lora_compare orchestration."""

    @pytest.mark.asyncio
    async def test_sequential_only(self):
        rd1, br1 = _make_result("adapter1", [100.0, 110.0, 105.0])
        rd2, br2 = _make_result("adapter2", [90.0, 95.0, 88.0])

        call_count = 0

        async def mock_run_benchmark(args, base_url):
            nonlocal call_count
            if args.model == "adapter1":
                call_count += 1
                return rd1, br1
            call_count += 1
            return rd2, br2

        args = Namespace(
            base_url="http://localhost:8000",
            num_prompts=100,
        )

        with patch("xpyd_bench.lora_compare.run_benchmark", side_effect=mock_run_benchmark):
            result = await run_lora_compare(args, ["adapter1", "adapter2"])

        assert len(result.adapter_results) == 2
        assert result.adapter_results[0].adapter == "adapter1"
        assert result.adapter_results[1].adapter == "adapter2"
        assert len(result.comparisons) == 1
        assert result.overhead is None
        assert not result.interleaved_results

    @pytest.mark.asyncio
    async def test_with_interleave(self):
        rd1, br1 = _make_result("adapter1", [100.0, 110.0, 105.0])
        rd2, br2 = _make_result("adapter2", [90.0, 95.0, 88.0])

        async def mock_run_benchmark(args, base_url):
            if args.model == "adapter1":
                return rd1, br1
            return rd2, br2

        args = Namespace(
            base_url="http://localhost:8000",
            num_prompts=100,
        )

        with patch("xpyd_bench.lora_compare.run_benchmark", side_effect=mock_run_benchmark):
            result = await run_lora_compare(
                args, ["adapter1", "adapter2"], interleave=True,
            )

        assert len(result.adapter_results) == 2
        assert len(result.interleaved_results) == 2
        assert result.overhead is not None
        assert isinstance(result.overhead.switching_overhead_ms, float)

    @pytest.mark.asyncio
    async def test_three_adapters(self):
        results_map = {}
        for name in ["base", "lora-a", "lora-b"]:
            rd, br = _make_result(name, [100.0, 110.0, 105.0])
            results_map[name] = (rd, br)

        async def mock_run_benchmark(args, base_url):
            return results_map[args.model]

        args = Namespace(
            base_url="http://localhost:8000",
            num_prompts=100,
        )

        with patch("xpyd_bench.lora_compare.run_benchmark", side_effect=mock_run_benchmark):
            result = await run_lora_compare(
                args, ["base", "lora-a", "lora-b"],
            )

        assert len(result.adapter_results) == 3
        # 2 comparisons: lora-a vs base, lora-b vs base
        assert len(result.comparisons) == 2
        assert result.comparisons[0].baseline_adapter == "base"
        assert result.comparisons[0].candidate_adapter == "lora-a"
        assert result.comparisons[1].candidate_adapter == "lora-b"


class TestFormatSummary:
    """Tests for format_lora_compare_summary."""

    def test_empty_results(self):
        result = LoRACompareResult()
        text = format_lora_compare_summary(result)
        assert "No results" in text

    def test_with_results(self):
        rd1, br1 = _make_result("adapter1", [100.0, 110.0])
        rd2, br2 = _make_result("adapter2", [90.0, 95.0])
        result = LoRACompareResult(
            base_url="http://localhost:8000",
            adapters=["adapter1", "adapter2"],
            adapter_results=[
                AdapterResult(adapter="adapter1", result=br1, raw_dict=rd1),
                AdapterResult(adapter="adapter2", result=br2, raw_dict=rd2),
            ],
        )
        text = format_lora_compare_summary(result)
        assert "Multi-LoRA" in text
        assert "adapter1" in text
        assert "adapter2" in text

    def test_with_overhead(self):
        rd1, br1 = _make_result("adapter1", [100.0])
        result = LoRACompareResult(
            base_url="http://localhost:8000",
            adapters=["adapter1"],
            adapter_results=[
                AdapterResult(adapter="adapter1", result=br1, raw_dict=rd1),
            ],
            overhead=AdapterOverhead(
                sequential_mean_ttft_ms=30.0,
                interleaved_mean_ttft_ms=35.0,
                switching_overhead_ms=5.0,
                switching_overhead_pct=16.7,
            ),
        )
        text = format_lora_compare_summary(result)
        assert "Switching" in text or "switching" in text


class TestFormatMarkdown:
    """Tests for format_lora_compare_markdown."""

    def test_empty(self):
        result = LoRACompareResult()
        md = format_lora_compare_markdown(result)
        assert "No results" in md

    def test_with_data(self):
        rd1, br1 = _make_result("adapter1", [100.0, 110.0])
        result = LoRACompareResult(
            adapters=["adapter1"],
            adapter_results=[
                AdapterResult(adapter="adapter1", result=br1, raw_dict=rd1),
            ],
        )
        md = format_lora_compare_markdown(result)
        assert "adapter1" in md
        assert "|" in md

    def test_with_overhead_section(self):
        rd1, br1 = _make_result("adapter1", [100.0])
        result = LoRACompareResult(
            adapters=["adapter1"],
            adapter_results=[
                AdapterResult(adapter="adapter1", result=br1, raw_dict=rd1),
            ],
            overhead=AdapterOverhead(
                sequential_mean_ttft_ms=30.0,
                interleaved_mean_ttft_ms=35.0,
                switching_overhead_ms=5.0,
                switching_overhead_pct=16.7,
            ),
        )
        md = format_lora_compare_markdown(result)
        assert "Adapter Switching Overhead" in md


class TestExport:
    """Tests for JSON and Markdown export."""

    def test_json_export(self, tmp_path):
        rd1, br1 = _make_result("adapter1", [100.0])
        result = LoRACompareResult(
            base_url="http://localhost:8000",
            adapters=["adapter1"],
            adapter_results=[
                AdapterResult(adapter="adapter1", result=br1, raw_dict=rd1),
            ],
        )
        p = export_lora_compare_json(result, tmp_path / "out.json")
        assert p.exists()
        data = json.loads(p.read_text())
        assert data["adapters"] == ["adapter1"]

    def test_markdown_export(self, tmp_path):
        rd1, br1 = _make_result("adapter1", [100.0])
        result = LoRACompareResult(
            adapters=["adapter1"],
            adapter_results=[
                AdapterResult(adapter="adapter1", result=br1, raw_dict=rd1),
            ],
        )
        p = export_lora_compare_markdown(result, tmp_path / "out.md")
        assert p.exists()
        assert "adapter1" in p.read_text()
