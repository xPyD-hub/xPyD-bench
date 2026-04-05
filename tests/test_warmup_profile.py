"""Tests for Warmup Profiling (M51)."""

from __future__ import annotations

import argparse

from xpyd_bench.bench.warmup_profile import (
    WarmupProfile,
    build_warmup_profile,
    detect_stabilization,
    print_warmup_profile,
)


class TestDetectStabilization:
    def test_stable_from_start(self):
        # All values similar → stabilize at index 0
        latencies = [100.0, 101.0, 99.0, 100.5, 100.2]
        idx = detect_stabilization(latencies, window_size=3, threshold=0.10)
        assert idx == 0

    def test_stabilize_after_warmup(self):
        # First 2 are high, then stable
        latencies = [500.0, 300.0, 100.0, 101.0, 99.0, 100.5]
        idx = detect_stabilization(latencies, window_size=3, threshold=0.10)
        assert idx is not None
        assert idx >= 2  # Should stabilize at or after index 2

    def test_no_stabilization(self):
        # Wildly varying values
        latencies = [100.0, 500.0, 50.0, 800.0, 20.0]
        idx = detect_stabilization(latencies, window_size=3, threshold=0.05)
        assert idx is None

    def test_too_few_values(self):
        latencies = [100.0, 101.0]
        idx = detect_stabilization(latencies, window_size=3, threshold=0.10)
        assert idx is None

    def test_empty(self):
        assert detect_stabilization([], window_size=3) is None

    def test_exact_window_size(self):
        latencies = [100.0, 100.0, 100.0]
        idx = detect_stabilization(latencies, window_size=3, threshold=0.10)
        assert idx == 0


class TestBuildWarmupProfile:
    def test_basic_profile(self):
        latencies = [500.0, 200.0, 100.0, 101.0, 99.0]
        profile = build_warmup_profile(latencies, total_duration_s=2.5)
        assert profile.warmup_duration_s == 2.5
        assert len(profile.latencies_ms) == 5
        assert profile.steady_state_latency_ms is not None
        assert profile.cold_start_penalty_ms > 0

    def test_single_request(self):
        profile = build_warmup_profile([100.0], total_duration_s=0.1)
        assert len(profile.latencies_ms) == 1
        assert profile.stabilization_index is None

    def test_empty(self):
        profile = build_warmup_profile([], total_duration_s=0.0)
        assert len(profile.latencies_ms) == 0
        assert profile.stabilization_index is None

    def test_to_dict(self):
        latencies = [500.0, 200.0, 100.0, 101.0, 99.0]
        profile = build_warmup_profile(latencies, total_duration_s=2.5)
        d = profile.to_dict()
        assert "latencies_ms" in d
        assert "stabilization_index" in d
        assert "cold_start_penalty_ms" in d
        assert "warmup_duration_s" in d
        assert isinstance(d["latencies_ms"], list)

    def test_stabilization_detected(self):
        latencies = [800.0, 400.0, 100.0, 101.0, 99.0, 100.0]
        profile = build_warmup_profile(latencies, total_duration_s=3.0)
        assert profile.stabilization_index is not None
        assert profile.steady_state_latency_ms is not None
        # Cold start penalty should be roughly 800 - steady_state
        assert profile.cold_start_penalty_ms > 600


class TestWarmupProfileDataclass:
    def test_defaults(self):
        wp = WarmupProfile()
        assert wp.latencies_ms == []
        assert wp.stabilization_index is None
        assert wp.cold_start_penalty_ms == 0.0
        assert wp.warmup_duration_s == 0.0

    def test_to_dict_no_steady_state(self):
        wp = WarmupProfile(latencies_ms=[100.0])
        d = wp.to_dict()
        assert "steady_state_latency_ms" not in d


class TestPrintWarmupProfile:
    def test_prints_without_error(self, capsys):
        latencies = [500.0, 200.0, 100.0, 101.0, 99.0]
        profile = build_warmup_profile(latencies, total_duration_s=2.5)
        print_warmup_profile(profile)
        out = capsys.readouterr().out
        assert "Warmup Profile" in out
        assert "5 requests" in out
        assert "Duration" in out

    def test_prints_no_stabilization(self, capsys):
        latencies = [100.0, 500.0, 50.0, 800.0, 20.0]
        profile = build_warmup_profile(latencies, total_duration_s=1.0)
        profile.stabilization_index = None
        print_warmup_profile(profile)
        out = capsys.readouterr().out
        assert "not detected" in out


class TestCLIArgs:
    def test_warmup_profile_flag(self):
        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--warmup", "5", "--warmup-profile"])
        assert args.warmup == 5
        assert args.warmup_profile is True

    def test_warmup_profile_default_false(self):
        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args([])
        assert args.warmup_profile is False


class TestYAMLConfig:
    def test_warmup_profile_in_known_keys(self):
        from xpyd_bench.config_cmd import _KNOWN_KEYS

        assert "warmup_profile" in _KNOWN_KEYS
