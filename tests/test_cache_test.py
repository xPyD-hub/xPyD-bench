"""Tests for M87: Prefix Caching Impact Analysis."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from xpyd_bench.cache_test import (
    CacheAnalysis,
    CacheTestMetrics,
    _percentile,
    _random_text,
    cache_test_main,
    format_cache_test_summary,
    generate_shared_prefix_prompts,
    run_cache_test,
)


class TestRandomText:
    def test_word_count(self):
        text = _random_text(10)
        assert len(text.split()) == 10

    def test_empty(self):
        text = _random_text(0)
        assert text == ""


class TestGeneratePrompts:
    def test_returns_two_lists(self):
        shared, unique = generate_shared_prefix_prompts(5)
        assert len(shared) == 5
        assert len(unique) == 5

    def test_shared_prefix_present(self):
        shared, _ = generate_shared_prefix_prompts(3, shared_prefix_ratio=0.8)
        # With high ratio, shared prompts should share a common prefix
        # Find common prefix
        prefix = shared[0]
        for p in shared[1:]:
            while not p.startswith(prefix):
                prefix = prefix[: prefix.rfind(" ")]
        assert len(prefix) > 10  # Non-trivial shared prefix

    def test_unique_prompts_differ(self):
        _, unique = generate_shared_prefix_prompts(5)
        assert len(set(unique)) == 5

    def test_invalid_ratio(self):
        with pytest.raises(ValueError, match="shared_prefix_ratio"):
            generate_shared_prefix_prompts(5, shared_prefix_ratio=1.5)

    def test_ratio_zero(self):
        shared, unique = generate_shared_prefix_prompts(3, shared_prefix_ratio=0.0)
        assert len(shared) == 3
        assert len(unique) == 3


class TestPercentile:
    def test_basic(self):
        values = list(range(100))
        assert _percentile(values, 50) == pytest.approx(49.5, abs=0.1)

    def test_empty(self):
        assert _percentile([], 50) == 0.0

    def test_single(self):
        assert _percentile([42.0], 99) == 42.0


class TestCacheTestMetrics:
    def test_to_dict(self):
        m = CacheTestMetrics(
            label="test",
            num_requests=10,
            mean_ttft_ms=50.0,
            p50_ttft_ms=45.0,
            p99_ttft_ms=100.0,
            mean_latency_ms=200.0,
            throughput_rps=5.0,
            errors=1,
        )
        d = m.to_dict()
        assert d["label"] == "test"
        assert d["num_requests"] == 10
        assert d["errors"] == 1


class TestCacheAnalysis:
    def test_to_dict(self):
        analysis = CacheAnalysis(
            shared=CacheTestMetrics(label="shared", mean_ttft_ms=40.0),
            unique=CacheTestMetrics(label="unique", mean_ttft_ms=60.0),
            ttft_improvement_pct=33.33,
            throughput_improvement_pct=10.0,
            shared_prefix_ratio=0.5,
            base_url="http://localhost:8000",
            model="test-model",
        )
        d = analysis.to_dict()
        assert "cache_analysis" in d
        ca = d["cache_analysis"]
        assert ca["shared_prefix_ratio"] == 0.5
        assert ca["ttft_improvement_pct"] == 33.33
        assert ca["shared"]["label"] == "shared"
        assert ca["unique"]["label"] == "unique"

    def test_json_serializable(self):
        analysis = CacheAnalysis()
        json.dumps(analysis.to_dict())  # Should not raise


class TestFormatSummary:
    def test_contains_key_fields(self):
        result = CacheAnalysis(
            shared=CacheTestMetrics(label="shared", mean_ttft_ms=40.0, throughput_rps=10.0),
            unique=CacheTestMetrics(label="unique", mean_ttft_ms=60.0, throughput_rps=8.0),
            ttft_improvement_pct=33.3,
            throughput_improvement_pct=25.0,
            base_url="http://localhost:8000",
            model="test-model",
        )
        text = format_cache_test_summary(result)
        assert "Prefix Caching" in text
        assert "TTFT improvement" in text
        assert "Throughput improvement" in text
        assert "test-model" in text


class TestRunCacheTest:
    @pytest.mark.asyncio
    async def test_basic_flow(self):
        """Test run_cache_test with mocked _run_group."""
        shared_m = CacheTestMetrics(
            label="shared-prefix",
            num_requests=5,
            mean_ttft_ms=30.0,
            p50_ttft_ms=28.0,
            p99_ttft_ms=50.0,
            mean_latency_ms=100.0,
            throughput_rps=10.0,
        )
        unique_m = CacheTestMetrics(
            label="unique-prefix",
            num_requests=5,
            mean_ttft_ms=60.0,
            p50_ttft_ms=55.0,
            p99_ttft_ms=100.0,
            mean_latency_ms=150.0,
            throughput_rps=8.0,
        )

        with patch("xpyd_bench.cache_test._run_group", new_callable=AsyncMock) as mock_rg:
            mock_rg.side_effect = [shared_m, unique_m]
            result = await run_cache_test(
                base_url="http://localhost:8000",
                model="test",
                num_prompts=5,
                shared_prefix_ratio=0.5,
            )

        assert result.ttft_improvement_pct == pytest.approx(50.0, abs=0.1)
        assert result.throughput_improvement_pct == pytest.approx(25.0, abs=0.1)
        assert result.model == "test"


class TestCLI:
    def test_cli_help(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            cache_test_main(["--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "prefix caching" in captured.out.lower()

    def test_cli_missing_required(self):
        with pytest.raises(SystemExit) as exc_info:
            cache_test_main([])
        assert exc_info.value.code != 0

    def test_cli_json_output(self, tmp_path):
        """Test JSON output flag with mocked run."""
        out = tmp_path / "result.json"
        with patch("xpyd_bench.cache_test.run_cache_test", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = CacheAnalysis(
                base_url="http://localhost:8000",
                model="test",
            )
            cache_test_main([
                "--base-url", "http://localhost:8000",
                "--model", "test",
                "--json-output", str(out),
            ])
        assert out.exists()
        data = json.loads(out.read_text())
        assert "cache_analysis" in data
