"""Tests for M28: Connection Pool & HTTP/2 Configuration."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from xpyd_bench.bench.debug_log import DebugLogger
from xpyd_bench.bench.runner import _build_client_kwargs

# ---------------------------------------------------------------------------
# _build_client_kwargs tests
# ---------------------------------------------------------------------------


class TestBuildClientKwargs:
    """Tests for the HTTP client factory helper."""

    def test_defaults(self):
        """Default args produce correct limits."""
        args = Namespace(http2=False, max_connections=100, max_keepalive=20)
        kwargs = _build_client_kwargs(args)
        assert "http2" not in kwargs
        limits = kwargs["limits"]
        assert limits.max_connections == 100
        assert limits.max_keepalive_connections == 20

    def test_http2_enabled(self):
        """--http2 sets http2=True when h2 is available."""
        args = Namespace(http2=True, max_connections=100, max_keepalive=20)
        kwargs = _build_client_kwargs(args)
        assert kwargs["http2"] is True

    def test_http2_missing_h2(self):
        """--http2 without h2 package raises SystemExit."""
        args = Namespace(http2=True, max_connections=100, max_keepalive=20)
        with patch.dict("sys.modules", {"h2": None}):
            with pytest.raises(SystemExit, match="h2"):
                _build_client_kwargs(args)

    def test_custom_pool_sizes(self):
        """Custom max-connections and max-keepalive are applied."""
        args = Namespace(http2=False, max_connections=50, max_keepalive=10)
        kwargs = _build_client_kwargs(args)
        limits = kwargs["limits"]
        assert limits.max_connections == 50
        assert limits.max_keepalive_connections == 10

    def test_headers_passed(self):
        """Headers are included in kwargs."""
        args = Namespace(http2=False, max_connections=100, max_keepalive=20)
        kwargs = _build_client_kwargs(args, headers={"X-Test": "1"})
        assert kwargs["headers"] == {"X-Test": "1"}

    def test_none_pool_values_fallback(self):
        """None pool values fall back to defaults."""
        args = Namespace(http2=False, max_connections=None, max_keepalive=None)
        kwargs = _build_client_kwargs(args)
        limits = kwargs["limits"]
        assert limits.max_connections == 100
        assert limits.max_keepalive_connections == 20


# ---------------------------------------------------------------------------
# CLI argument parsing tests
# ---------------------------------------------------------------------------


class TestCLIArgs:
    """Test that CLI args are parsed correctly."""

    def test_http2_flag(self):
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--http2"])
        assert args.http2 is True

    def test_max_connections_flag(self):
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--max-connections", "50"])
        assert args.max_connections == 50

    def test_max_keepalive_flag(self):
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--max-keepalive", "10"])
        assert args.max_keepalive == 10

    def test_defaults(self):
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args([])
        assert args.http2 is False
        assert args.max_connections == 100
        assert args.max_keepalive == 20


# ---------------------------------------------------------------------------
# YAML config tests
# ---------------------------------------------------------------------------


class TestYAMLConfig:
    """Test YAML config support for connection parameters."""

    def test_yaml_connection_config(self, tmp_path: Path):
        import argparse

        import yaml

        from xpyd_bench.cli import _add_vllm_compat_args, _load_yaml_config

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args([])

        config = {"http2": True, "max_connections": 200, "max_keepalive": 50}
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(config))

        args = _load_yaml_config(str(cfg_path), args, explicit_keys=set())
        assert args.http2 is True
        assert args.max_connections == 200
        assert args.max_keepalive == 50

    def test_cli_overrides_yaml(self, tmp_path: Path):
        import argparse

        import yaml

        from xpyd_bench.cli import _add_vllm_compat_args, _load_yaml_config

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--max-connections", "75"])

        config = {"max_connections": 200}
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(config))

        args = _load_yaml_config(
            str(cfg_path), args, explicit_keys={"max_connections"}
        )
        assert args.max_connections == 75


# ---------------------------------------------------------------------------
# Dry run output tests
# ---------------------------------------------------------------------------


class TestDryRunOutput:
    """Test that dry-run shows connection config."""

    def test_dry_run_shows_connection_info(self, capsys):
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args, _dry_run

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--http2", "--max-connections", "50"])
        # Ensure custom_headers is resolved
        args.custom_headers = {}

        _dry_run(args, "http://localhost:8000")

        captured = capsys.readouterr().out
        assert "HTTP/2:          True" in captured
        assert "Max connections: 50" in captured
        assert "Max keepalive:   20" in captured


# ---------------------------------------------------------------------------
# Debug log connection config tests
# ---------------------------------------------------------------------------


class TestDebugLogConnectionConfig:
    """Test that debug log includes connection pool info."""

    def test_log_connection_config(self, tmp_path: Path):
        log_path = tmp_path / "debug.jsonl"
        logger = DebugLogger(str(log_path))
        logger.log_connection_config(http2=True, max_connections=50, max_keepalive=10)
        logger.close()

        lines = log_path.read_text().strip().split("\n")
        entry = json.loads(lines[0])
        assert entry["type"] == "connection_config"
        assert entry["http2"] is True
        assert entry["max_connections"] == 50
        assert entry["max_keepalive"] == 10
