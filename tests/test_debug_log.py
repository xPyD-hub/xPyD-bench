"""Tests for M22: Request Logging & Debug Mode."""

from __future__ import annotations

import json

from xpyd_bench.bench.debug_log import DebugLogEntry, DebugLogger
from xpyd_bench.bench.models import RequestResult


class TestDebugLogEntry:
    """Test DebugLogEntry dataclass."""

    def test_to_dict_success(self):
        entry = DebugLogEntry(
            timestamp="2026-04-04T19:00:00+0800",
            url="http://localhost:8000/v1/completions",
            payload='{"prompt": "hello", "max_tokens": 128}',
            status="ok",
            latency_ms=42.123456,
            success=True,
        )
        d = entry.to_dict()
        assert d["timestamp"] == "2026-04-04T19:00:00+0800"
        assert d["url"] == "http://localhost:8000/v1/completions"
        assert d["success"] is True
        assert d["latency_ms"] == 42.123
        assert "error" not in d
        assert "retries" not in d

    def test_to_dict_error(self):
        entry = DebugLogEntry(
            timestamp="2026-04-04T19:00:00+0800",
            url="http://localhost:8000/v1/completions",
            payload="{}",
            status="error",
            latency_ms=100.0,
            success=False,
            error="Connection refused",
            retries=2,
        )
        d = entry.to_dict()
        assert d["success"] is False
        assert d["error"] == "Connection refused"
        assert d["retries"] == 2


class TestDebugLogger:
    """Test DebugLogger file output."""

    def test_log_creates_file(self, tmp_path):
        log_path = tmp_path / "debug.jsonl"
        logger = DebugLogger(log_path)
        result = RequestResult(latency_ms=50.0, success=True)
        logger.log(
            url="http://localhost:8000/v1/completions",
            payload={"prompt": "hello", "max_tokens": 128},
            result=result,
        )
        logger.close()

        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["url"] == "http://localhost:8000/v1/completions"
        assert entry["success"] is True
        assert entry["latency_ms"] == 50.0
        assert "timestamp" in entry
        assert "payload" in entry

    def test_log_multiple_entries(self, tmp_path):
        log_path = tmp_path / "debug.jsonl"
        logger = DebugLogger(log_path)
        for i in range(5):
            result = RequestResult(latency_ms=float(i * 10), success=True)
            logger.log(
                url="http://localhost:8000/v1/completions",
                payload={"prompt": f"prompt {i}"},
                result=result,
            )
        logger.close()

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 5
        for i, line in enumerate(lines):
            entry = json.loads(line)
            assert entry["latency_ms"] == float(i * 10)

    def test_log_error_entry(self, tmp_path):
        log_path = tmp_path / "debug.jsonl"
        logger = DebugLogger(log_path)
        result = RequestResult(
            latency_ms=200.0, success=False, error="timeout", retries=3
        )
        logger.log(
            url="http://localhost:8000/v1/completions",
            payload={"prompt": "test"},
            result=result,
        )
        logger.close()

        entry = json.loads(log_path.read_text().strip())
        assert entry["success"] is False
        assert entry["error"] == "timeout"
        assert entry["retries"] == 3
        assert entry["status"] == "error"

    def test_payload_truncation(self, tmp_path):
        log_path = tmp_path / "debug.jsonl"
        logger = DebugLogger(log_path)
        # Create a very large payload
        long_prompt = "x" * 2000
        result = RequestResult(latency_ms=10.0, success=True)
        logger.log(
            url="http://localhost:8000/v1/completions",
            payload={"prompt": long_prompt},
            result=result,
        )
        logger.close()

        entry = json.loads(log_path.read_text().strip())
        assert entry["payload"].endswith("...(truncated)")
        assert len(entry["payload"]) < 600  # 512 + truncation marker

    def test_creates_parent_dirs(self, tmp_path):
        log_path = tmp_path / "subdir" / "deep" / "debug.jsonl"
        logger = DebugLogger(log_path)
        result = RequestResult(latency_ms=1.0, success=True)
        logger.log(url="http://x", payload={}, result=result)
        logger.close()
        assert log_path.exists()


class TestDebugLogCLI:
    """Test --debug-log CLI argument parsing."""

    def test_cli_arg_parsed(self):
        """Verify --debug-log is recognized by the argument parser."""
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--debug-log", "/tmp/test.jsonl"])
        assert args.debug_log == "/tmp/test.jsonl"

    def test_cli_arg_default_none(self):
        """Verify --debug-log defaults to None."""
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args([])
        assert args.debug_log is None


class TestDebugLogYAMLConfig:
    """Test YAML config support for debug_log."""

    def test_yaml_config_sets_debug_log(self, tmp_path):
        import argparse

        import yaml

        from xpyd_bench.cli import _add_vllm_compat_args, _load_yaml_config

        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({"debug_log": "/tmp/debug.jsonl"}))

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args([])
        args = _load_yaml_config(str(config_path), args)
        assert args.debug_log == "/tmp/debug.jsonl"
