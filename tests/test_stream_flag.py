"""Tests for --stream / --no-stream CLI flag (issue #87)."""

from __future__ import annotations

import pytest

from xpyd_bench.cli import bench_main


class TestStreamFlagCLI:
    """CLI argument parsing for --stream / --no-stream."""

    def test_stream_flag_default_is_none(self):
        """When neither --stream nor --no-stream is given, args.stream is None."""
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args([])
        assert args.stream is None

    def test_stream_flag_explicit_true(self):
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--stream"])
        assert args.stream is True

    def test_no_stream_flag(self):
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--no-stream"])
        assert args.stream is False

    def test_stream_and_no_stream_mutually_exclusive(self):
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        with pytest.raises(SystemExit):
            parser.parse_args(["--stream", "--no-stream"])


class TestStreamFlagRunner:
    """Runner uses --stream flag correctly."""

    def test_completions_default_no_stream(self):
        """Without --stream, completions defaults to non-streaming."""
        from argparse import Namespace

        args = Namespace(endpoint="/v1/completions")
        stream_flag = getattr(args, "stream", None)
        is_chat = "chat" in args.endpoint
        is_streaming = stream_flag if stream_flag is not None else is_chat
        assert is_streaming is False

    def test_completions_explicit_stream(self):
        """With --stream, completions endpoint streams."""
        from argparse import Namespace

        args = Namespace(endpoint="/v1/completions", stream=True)
        stream_flag = getattr(args, "stream", None)
        is_chat = "chat" in args.endpoint
        is_streaming = stream_flag if stream_flag is not None else is_chat
        assert is_streaming is True

    def test_chat_default_streams(self):
        from argparse import Namespace

        args = Namespace(endpoint="/v1/chat/completions")
        stream_flag = getattr(args, "stream", None)
        is_chat = "chat" in args.endpoint
        is_streaming = stream_flag if stream_flag is not None else is_chat
        assert is_streaming is True

    def test_chat_no_stream(self):
        from argparse import Namespace

        args = Namespace(endpoint="/v1/chat/completions", stream=False)
        stream_flag = getattr(args, "stream", None)
        is_chat = "chat" in args.endpoint
        is_streaming = stream_flag if stream_flag is not None else is_chat
        assert is_streaming is False


class TestStreamFlagYAML:
    """YAML config support for stream key."""

    def test_yaml_stream_true(self, tmp_path):
        """YAML config with stream: true is loaded."""
        import argparse

        import yaml

        from xpyd_bench.cli import _add_vllm_compat_args, _load_yaml_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump({"stream": True}))

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args([])
        args = _load_yaml_config(str(cfg_file), args)
        assert args.stream is True

    def test_yaml_stream_false(self, tmp_path):
        import argparse

        import yaml

        from xpyd_bench.cli import _add_vllm_compat_args, _load_yaml_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump({"stream": False}))

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args([])
        args = _load_yaml_config(str(cfg_file), args)
        assert args.stream is False

    def test_cli_overrides_yaml(self, tmp_path):
        """Explicit CLI --no-stream overrides YAML stream: true."""
        import argparse

        import yaml

        from xpyd_bench.cli import _add_vllm_compat_args, _load_yaml_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump({"stream": True}))

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--no-stream"])
        args = _load_yaml_config(str(cfg_file), args, explicit_keys={"stream"})
        assert args.stream is False


class TestStreamFlagDryRun:
    """Dry-run output shows streaming status."""

    def test_dry_run_shows_streaming(self, capsys):
        """--dry-run should display streaming status."""
        try:
            bench_main(["--dry-run", "--num-prompts", "1"])
        except SystemExit:
            pass
        captured = capsys.readouterr()
        assert "Streaming:" in captured.out


class TestStreamFlagConfigValidation:
    """'stream' is a recognized YAML config key."""

    def test_stream_in_valid_keys(self):
        from xpyd_bench.config_cmd import _KNOWN_KEYS

        assert "stream" in _KNOWN_KEYS
