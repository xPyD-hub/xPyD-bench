"""Tests for M21: Environment Info Capture."""

from __future__ import annotations

import json
import platform
import socket
import sys
from dataclasses import asdict

from xpyd_bench.bench.env import collect_env_info
from xpyd_bench.bench.models import BenchmarkResult
from xpyd_bench.bench.runner import _to_dict


class TestCollectEnvInfo:
    """Tests for collect_env_info()."""

    def test_returns_dict(self):
        info = collect_env_info()
        assert isinstance(info, dict)

    def test_required_keys(self):
        info = collect_env_info()
        expected_keys = {
            "python_version",
            "os",
            "platform",
            "architecture",
            "hostname",
            "xpyd_bench_version",
            "timestamp",
        }
        assert expected_keys <= set(info.keys())

    def test_all_values_are_strings(self):
        info = collect_env_info()
        for k, v in info.items():
            assert isinstance(v, str), f"{k} should be str, got {type(v)}"

    def test_python_version(self):
        info = collect_env_info()
        assert sys.version in info["python_version"]

    def test_os(self):
        info = collect_env_info()
        assert info["os"] == platform.system()

    def test_architecture(self):
        info = collect_env_info()
        assert info["architecture"] == platform.machine()

    def test_hostname(self):
        info = collect_env_info()
        assert info["hostname"] == socket.gethostname()

    def test_version(self):
        from xpyd_bench import __version__

        info = collect_env_info()
        assert info["xpyd_bench_version"] == __version__

    def test_timestamp_iso_format(self):
        from datetime import datetime

        info = collect_env_info()
        # Should parse without error
        datetime.fromisoformat(info["timestamp"])


class TestBenchmarkResultEnvironment:
    """Tests that BenchmarkResult includes environment field."""

    def test_default_empty(self):
        r = BenchmarkResult()
        assert r.environment == {}

    def test_with_env_info(self):
        env = collect_env_info()
        r = BenchmarkResult(environment=env)
        assert r.environment == env
        assert "python_version" in r.environment

    def test_serialization(self):
        env = collect_env_info()
        r = BenchmarkResult(environment=env)
        data = asdict(r)
        assert "environment" in data
        assert data["environment"]["os"] == platform.system()

    def test_json_roundtrip(self):
        env = collect_env_info()
        r = BenchmarkResult(environment=env)
        data = asdict(r)
        raw = json.dumps(data, default=str)
        loaded = json.loads(raw)
        assert loaded["environment"]["hostname"] == socket.gethostname()


class TestToDictEnvironment:
    """Tests that _to_dict() includes environment in output."""

    def test_to_dict_includes_environment(self):
        env = collect_env_info()
        r = BenchmarkResult(environment=env)
        d = _to_dict(r)
        assert "environment" in d
        assert d["environment"]["os"] == platform.system()

    def test_to_dict_omits_empty_environment(self):
        r = BenchmarkResult()
        d = _to_dict(r)
        assert "environment" not in d
