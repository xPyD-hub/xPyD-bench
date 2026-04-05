"""Tests for M92: Prompt Caching Cost Analysis."""

from __future__ import annotations

from xpyd_bench.bench.cache_savings import (
    _estimate_token_count,
    _longest_common_prefix,
    analyze_cache_savings,
)

# --- Unit tests for helpers ---


class TestLongestCommonPrefix:
    def test_identical(self):
        assert _longest_common_prefix("hello", "hello") == 5

    def test_no_common(self):
        assert _longest_common_prefix("abc", "xyz") == 0

    def test_partial(self):
        assert _longest_common_prefix("abcdef", "abcxyz") == 3

    def test_empty(self):
        assert _longest_common_prefix("", "abc") == 0
        assert _longest_common_prefix("abc", "") == 0

    def test_different_lengths(self):
        assert _longest_common_prefix("ab", "abcdef") == 2


class TestEstimateTokenCount:
    def test_basic(self):
        assert _estimate_token_count("a" * 20) == 5

    def test_minimum(self):
        assert _estimate_token_count("a") == 1

    def test_empty(self):
        assert _estimate_token_count("") == 1


# --- Core analysis tests ---


class TestAnalyzeCacheSavings:
    def test_single_prompt(self):
        result = analyze_cache_savings(["Hello world this is a test prompt"])
        assert result["num_prompts"] == 1
        assert result["cacheable_token_ratio"] == 0.0
        assert result["estimated_cache_hit_rate"] == 0.0
        assert result["savings_ratio"] == 0.0
        assert result["cost_savings"] is None

    def test_identical_prompts(self):
        prompts = ["The quick brown fox jumps over the lazy dog"] * 10
        result = analyze_cache_savings(prompts)
        assert result["num_prompts"] == 10
        assert result["cacheable_token_ratio"] > 0.5
        assert result["estimated_cache_hit_rate"] == 1.0
        assert result["savings_ratio"] > 0.0

    def test_no_shared_prefix(self):
        prompts = [
            "Alpha bravo charlie delta echo foxtrot golf hotel",
            "India juliet kilo lima mike november oscar papa",
            "Quebec romeo sierra tango uniform victor whiskey",
        ]
        result = analyze_cache_savings(prompts)
        assert result["num_prompts"] == 3
        assert result["cacheable_token_ratio"] < 0.1

    def test_shared_prefix(self):
        prefix = "You are a helpful AI assistant. Please answer: "
        prompts = [
            prefix + "What is Python?",
            prefix + "What is Java?",
            prefix + "What is Rust?",
            prefix + "What is Go?",
        ]
        result = analyze_cache_savings(prompts)
        assert result["cacheable_token_ratio"] > 0.3
        assert result["estimated_cache_hit_rate"] > 0.5

    def test_custom_pricing_ratio(self):
        prompts = [
            "Shared prefix content here: query A",
            "Shared prefix content here: query B",
        ]
        r1 = analyze_cache_savings(prompts, cache_pricing_ratio=0.5)
        r2 = analyze_cache_savings(prompts, cache_pricing_ratio=0.0)
        assert r2["savings_ratio"] >= r1["savings_ratio"]

    def test_cost_savings_with_cost_model(self):
        prompts = [
            "Reusable system prompt for benchmarking: question alpha",
            "Reusable system prompt for benchmarking: question beta",
        ]
        result = analyze_cache_savings(
            prompts,
            cache_pricing_ratio=0.5,
            cost_per_1k_input=0.03,
        )
        assert result["cost_savings"] is not None
        cs = result["cost_savings"]
        assert cs["currency"] == "USD"
        assert cs["full_cost"] > 0
        assert cs["saved"] >= 0
        assert cs["cached_cost"] <= cs["full_cost"]

    def test_cost_savings_none_without_cost_model(self):
        prompts = ["Hello world test prompt", "Hello world other prompt"]
        result = analyze_cache_savings(prompts)
        assert result["cost_savings"] is None

    def test_two_prompts(self):
        result = analyze_cache_savings(["abc def ghi", "abc def xyz"])
        assert result["num_prompts"] == 2
        assert result["total_prompt_tokens"] > 0
        assert result["unique_tokens"] >= 0

    def test_empty_prompts_list_single(self):
        result = analyze_cache_savings([""])
        assert result["num_prompts"] == 1


# --- BenchmarkResult integration ---


class TestBenchmarkResultIntegration:
    def test_cache_savings_field(self):
        from xpyd_bench.bench.models import BenchmarkResult

        r = BenchmarkResult()
        assert r.cache_savings is None
        r.cache_savings = {"cacheable_token_ratio": 0.5}
        assert r.cache_savings["cacheable_token_ratio"] == 0.5

    def test_json_serialization(self):
        from xpyd_bench.bench.models import BenchmarkResult
        from xpyd_bench.bench.runner import _to_dict as _result_to_dict

        r = BenchmarkResult()
        r.cache_savings = {
            "num_prompts": 5,
            "cacheable_token_ratio": 0.42,
            "savings_ratio": 0.21,
        }
        d = _result_to_dict(r)
        assert "cache_savings" in d
        assert d["cache_savings"]["cacheable_token_ratio"] == 0.42

    def test_json_serialization_absent(self):
        from xpyd_bench.bench.models import BenchmarkResult
        from xpyd_bench.bench.runner import _to_dict as _result_to_dict

        r = BenchmarkResult()
        d = _result_to_dict(r)
        assert "cache_savings" not in d


# --- Config key validation ---


class TestConfigKey:
    def test_known_keys(self):
        from xpyd_bench.config_cmd import _KNOWN_KEYS as KNOWN_KEYS

        assert "analyze_cache_savings" in KNOWN_KEYS
        assert "cache_pricing_ratio" in KNOWN_KEYS


# --- CLI integration ---


class TestCLIIntegration:
    def test_flag_parsing(self):
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--analyze-cache-savings"])
        assert args.analyze_cache_savings is True

    def test_default_disabled(self):
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args([])
        assert args.analyze_cache_savings is False

    def test_pricing_ratio_parsing(self):
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--cache-pricing-ratio", "0.3"])
        assert args.cache_pricing_ratio == 0.3

    def test_pricing_ratio_default(self):
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args([])
        assert args.cache_pricing_ratio == 0.5
