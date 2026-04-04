"""Tests for YAML config overriding CLI defaults (issue #82)."""

from __future__ import annotations

import argparse
import os
import tempfile

from xpyd_bench.cli import (
    _add_vllm_compat_args,
    _get_explicit_keys,
    _load_yaml_config,
)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    _add_vllm_compat_args(parser)
    return parser


def _write_yaml(content: str) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    f.write(content)
    f.close()
    return f.name


class TestGetExplicitKeys:
    def test_no_args_yields_empty(self):
        parser = _make_parser()
        args = parser.parse_args([])
        assert _get_explicit_keys(parser, args) == set()

    def test_explicit_flags_detected(self):
        parser = _make_parser()
        args = parser.parse_args(["--num-prompts", "42", "--base-url", "http://x"])
        explicit = _get_explicit_keys(parser, args)
        assert "num_prompts" in explicit
        assert "base_url" in explicit
        assert "request_rate" not in explicit


class TestYamlOverride:
    def test_yaml_overrides_defaults(self):
        parser = _make_parser()
        args = parser.parse_args([])
        explicit = _get_explicit_keys(parser, args)
        path = _write_yaml("num_prompts: 500\nrequest_rate: 10.0\n")
        try:
            args = _load_yaml_config(path, args, explicit_keys=explicit)
            assert args.num_prompts == 500
            assert args.request_rate == 10.0
        finally:
            os.unlink(path)

    def test_cli_explicit_wins_over_yaml(self):
        parser = _make_parser()
        args = parser.parse_args(["--num-prompts", "200"])
        explicit = _get_explicit_keys(parser, args)
        path = _write_yaml("num_prompts: 500\nrequest_rate: 10.0\n")
        try:
            args = _load_yaml_config(path, args, explicit_keys=explicit)
            assert args.num_prompts == 200  # CLI wins
            assert args.request_rate == 10.0  # YAML applied
        finally:
            os.unlink(path)

    def test_legacy_none_fallback(self):
        """Without explicit_keys, legacy behaviour: only None attrs are set."""
        parser = _make_parser()
        args = parser.parse_args([])
        path = _write_yaml("num_prompts: 500\n")
        try:
            args = _load_yaml_config(path, args, explicit_keys=None)
            # Legacy: num_prompts has default 1000 (not None), so YAML is ignored
            assert args.num_prompts == 1000
        finally:
            os.unlink(path)
