"""Tests for noise injection & chaos testing (M60)."""

from __future__ import annotations

import asyncio
import time
from argparse import Namespace

from xpyd_bench.noise import (
    NoiseConfig,
    NoiseInjector,
    NoiseStats,
    _corrupt_string,
    build_noise_config_from_args,
)

# ---------------------------------------------------------------------------
# Unit tests for NoiseConfig
# ---------------------------------------------------------------------------


class TestNoiseConfig:
    def test_disabled_by_default(self):
        cfg = NoiseConfig()
        assert not cfg.enabled

    def test_enabled_with_delay(self):
        cfg = NoiseConfig(inject_delay_ms=100)
        assert cfg.enabled

    def test_enabled_with_error_rate(self):
        cfg = NoiseConfig(inject_error_rate=0.1)
        assert cfg.enabled

    def test_enabled_with_corruption(self):
        cfg = NoiseConfig(inject_payload_corruption=0.05)
        assert cfg.enabled

    def test_to_dict(self):
        cfg = NoiseConfig(
            inject_delay_ms=50,
            inject_error_rate=0.2,
            inject_payload_corruption=0.1,
        )
        d = cfg.to_dict()
        assert d["inject_delay_ms"] == 50
        assert d["inject_error_rate"] == 0.2
        assert d["inject_payload_corruption"] == 0.1


# ---------------------------------------------------------------------------
# Unit tests for NoiseInjector
# ---------------------------------------------------------------------------


class TestNoiseInjector:
    def test_maybe_delay_zero(self):
        cfg = NoiseConfig(inject_delay_ms=0)
        inj = NoiseInjector(cfg)
        result = asyncio.get_event_loop().run_until_complete(inj.maybe_delay())
        assert result == 0.0
        assert inj.stats.delays_injected == 0

    def test_maybe_delay_positive(self):
        cfg = NoiseConfig(inject_delay_ms=10)
        inj = NoiseInjector(cfg)
        start = time.monotonic()
        result = asyncio.get_event_loop().run_until_complete(inj.maybe_delay())
        elapsed = (time.monotonic() - start) * 1000
        assert result == 10.0
        assert elapsed >= 8  # allow some tolerance
        assert inj.stats.delays_injected == 1

    def test_should_inject_error_zero_rate(self):
        cfg = NoiseConfig(inject_error_rate=0.0, seed=42)
        inj = NoiseInjector(cfg)
        assert not any(inj.should_inject_error() for _ in range(100))

    def test_should_inject_error_full_rate(self):
        cfg = NoiseConfig(inject_error_rate=1.0, seed=42)
        inj = NoiseInjector(cfg)
        assert all(inj.should_inject_error() for _ in range(100))

    def test_should_inject_error_partial_rate(self):
        cfg = NoiseConfig(inject_error_rate=0.5, seed=42)
        inj = NoiseInjector(cfg)
        results = [inj.should_inject_error() for _ in range(1000)]
        rate = sum(results) / len(results)
        assert 0.35 < rate < 0.65

    def test_corrupt_payload_zero_rate(self):
        cfg = NoiseConfig(inject_payload_corruption=0.0, seed=42)
        inj = NoiseInjector(cfg)
        payload = {"prompt": "hello world"}
        result, corrupted = inj.corrupt_payload(payload)
        assert not corrupted
        assert result["prompt"] == "hello world"

    def test_corrupt_payload_full_rate(self):
        cfg = NoiseConfig(inject_payload_corruption=1.0, seed=42)
        inj = NoiseInjector(cfg)
        payload = {"prompt": "hello world test data"}
        result, corrupted = inj.corrupt_payload(payload)
        assert corrupted
        assert result["prompt"] != "hello world test data"
        assert inj.stats.payloads_corrupted == 1

    def test_corrupt_payload_chat_messages(self):
        cfg = NoiseConfig(inject_payload_corruption=1.0, seed=42)
        inj = NoiseInjector(cfg)
        payload = {
            "messages": [
                {"role": "user", "content": "hello world test"},
            ]
        }
        result, corrupted = inj.corrupt_payload(payload)
        assert corrupted
        assert result["messages"][-1]["content"] != "hello world test"

    def test_corrupt_payload_embeddings_input(self):
        cfg = NoiseConfig(inject_payload_corruption=1.0, seed=42)
        inj = NoiseInjector(cfg)
        payload = {"input": "some text to embed"}
        result, corrupted = inj.corrupt_payload(payload)
        assert corrupted

    def test_corrupt_payload_preserves_original(self):
        """Corruption should not modify the original payload."""
        cfg = NoiseConfig(inject_payload_corruption=1.0, seed=42)
        inj = NoiseInjector(cfg)
        original = {"prompt": "hello world"}
        result, _ = inj.corrupt_payload(original)
        assert original["prompt"] == "hello world"


# ---------------------------------------------------------------------------
# Unit tests for _corrupt_string
# ---------------------------------------------------------------------------


class TestCorruptString:
    def test_empty_string(self):
        import random

        rng = random.Random(42)
        assert _corrupt_string("", rng) == ""

    def test_non_empty_produces_different(self):
        import random

        rng = random.Random(42)
        original = "hello world this is a test string"
        result = _corrupt_string(original, rng)
        assert result != original


# ---------------------------------------------------------------------------
# Unit tests for build_noise_config_from_args
# ---------------------------------------------------------------------------


class TestBuildNoiseConfigFromArgs:
    def test_defaults(self):
        args = Namespace(seed=42)
        cfg = build_noise_config_from_args(args)
        assert not cfg.enabled

    def test_with_values(self):
        args = Namespace(
            inject_delay=100,
            inject_error_rate=0.1,
            inject_payload_corruption=0.05,
            seed=42,
        )
        cfg = build_noise_config_from_args(args)
        assert cfg.enabled
        assert cfg.inject_delay_ms == 100
        assert cfg.inject_error_rate == 0.1
        assert cfg.inject_payload_corruption == 0.05
        assert cfg.seed == 42


# ---------------------------------------------------------------------------
# NoiseStats
# ---------------------------------------------------------------------------


class TestNoiseStats:
    def test_to_dict(self):
        stats = NoiseStats(
            delays_injected=5,
            total_delay_ms=500.123,
            errors_injected=3,
            payloads_corrupted=2,
            total_requests=10,
        )
        d = stats.to_dict()
        assert d["delays_injected"] == 5
        assert d["total_delay_ms"] == 500.12
        assert d["errors_injected"] == 3
        assert d["payloads_corrupted"] == 2
        assert d["total_requests"] == 10


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestNoiseInjectionCLI:
    def test_cli_args_parsed(self):
        """CLI flags for noise injection are parsed correctly."""
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args([
            "--inject-delay", "50",
            "--inject-error-rate", "0.1",
            "--inject-payload-corruption", "0.05",
        ])
        assert args.inject_delay == 50.0
        assert args.inject_error_rate == 0.1
        assert args.inject_payload_corruption == 0.05

    def test_cli_defaults(self):
        """Default values are 0 (disabled)."""
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args([])
        assert args.inject_delay == 0.0
        assert args.inject_error_rate == 0.0
        assert args.inject_payload_corruption == 0.0


# ---------------------------------------------------------------------------
# YAML config keys
# ---------------------------------------------------------------------------


class TestNoiseYAMLConfig:
    def test_known_keys(self):
        """Noise injection keys are in _KNOWN_KEYS."""
        from xpyd_bench.config_cmd import _KNOWN_KEYS

        assert "inject_delay" in _KNOWN_KEYS
        assert "inject_error_rate" in _KNOWN_KEYS
        assert "inject_payload_corruption" in _KNOWN_KEYS


# ---------------------------------------------------------------------------
# BenchmarkResult model
# ---------------------------------------------------------------------------


class TestNoiseInBenchmarkResult:
    def test_default_none(self):
        from xpyd_bench.bench.models import BenchmarkResult

        r = BenchmarkResult()
        assert r.noise_injection is None

    def test_set_noise_injection(self):
        from xpyd_bench.bench.models import BenchmarkResult

        r = BenchmarkResult()
        r.noise_injection = {
            "config": {
                "inject_delay_ms": 50,
                "inject_error_rate": 0.1,
                "inject_payload_corruption": 0.0,
            },
            "stats": {
                "delays_injected": 10,
                "total_delay_ms": 500,
                "errors_injected": 1,
                "payloads_corrupted": 0,
                "total_requests": 10,
            },
        }
        assert r.noise_injection["stats"]["errors_injected"] == 1


# ---------------------------------------------------------------------------
# Combined modes
# ---------------------------------------------------------------------------


class TestCombinedModes:
    def test_all_modes_enabled(self):
        """All three injection modes can work together."""
        cfg = NoiseConfig(
            inject_delay_ms=1,
            inject_error_rate=0.5,
            inject_payload_corruption=0.5,
            seed=42,
        )
        inj = NoiseInjector(cfg)

        # Run delay
        asyncio.get_event_loop().run_until_complete(inj.maybe_delay())
        assert inj.stats.delays_injected == 1

        # Error injection works
        errors = sum(1 for _ in range(100) if inj.should_inject_error())
        assert 20 < errors < 80

        # Corruption works
        corrupted_count = 0
        for _ in range(100):
            _, was_corrupted = inj.corrupt_payload(
                {"prompt": "test data here"}
            )
            if was_corrupted:
                corrupted_count += 1
        assert 20 < corrupted_count < 80
