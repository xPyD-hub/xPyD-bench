"""Tests for output verbosity control (M65)."""

from __future__ import annotations

import io

import pytest

from xpyd_bench.bench.verbosity import Verbosity, VerbosityPrinter, parse_verbosity

# ---------------------------------------------------------------------------
# parse_verbosity
# ---------------------------------------------------------------------------


class TestParseVerbosity:
    """Tests for the ``parse_verbosity`` helper."""

    def test_none_returns_normal(self) -> None:
        assert parse_verbosity(None) is Verbosity.NORMAL

    def test_quiet(self) -> None:
        assert parse_verbosity("quiet") is Verbosity.QUIET

    def test_normal(self) -> None:
        assert parse_verbosity("normal") is Verbosity.NORMAL

    def test_verbose(self) -> None:
        assert parse_verbosity("verbose") is Verbosity.VERBOSE

    def test_case_insensitive(self) -> None:
        assert parse_verbosity("QUIET") is Verbosity.QUIET
        assert parse_verbosity("Verbose") is Verbosity.VERBOSE

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid verbosity level"):
            parse_verbosity("debug")


# ---------------------------------------------------------------------------
# VerbosityPrinter
# ---------------------------------------------------------------------------


class TestVerbosityPrinter:
    """Tests for the ``VerbosityPrinter`` helper class."""

    # -- quiet mode --------------------------------------------------------

    def test_quiet_mode_suppresses_normal(self) -> None:
        buf = io.StringIO()
        vp = VerbosityPrinter(Verbosity.QUIET, file=buf)
        vp.normal("should not appear")
        assert buf.getvalue() == ""

    def test_quiet_mode_suppresses_verbose(self) -> None:
        buf = io.StringIO()
        vp = VerbosityPrinter(Verbosity.QUIET, file=buf)
        vp.verbose("should not appear")
        assert buf.getvalue() == ""

    def test_quiet_mode_allows_error(self) -> None:
        buf = io.StringIO()
        vp = VerbosityPrinter(Verbosity.QUIET, file=buf)
        vp.error("fatal")
        assert "fatal" in buf.getvalue()

    # -- normal mode -------------------------------------------------------

    def test_normal_mode_prints_normal(self) -> None:
        buf = io.StringIO()
        vp = VerbosityPrinter(Verbosity.NORMAL, file=buf)
        vp.normal("hello")
        assert "hello" in buf.getvalue()

    def test_normal_mode_suppresses_verbose(self) -> None:
        buf = io.StringIO()
        vp = VerbosityPrinter(Verbosity.NORMAL, file=buf)
        vp.verbose("extra detail")
        assert buf.getvalue() == ""

    # -- verbose mode ------------------------------------------------------

    def test_verbose_mode_prints_all(self) -> None:
        buf = io.StringIO()
        vp = VerbosityPrinter(Verbosity.VERBOSE, file=buf)
        vp.quiet("a")
        vp.normal("b")
        vp.verbose("c")
        out = buf.getvalue()
        assert "a" in out
        assert "b" in out
        assert "c" in out

    # -- helpers -----------------------------------------------------------

    def test_is_quiet(self) -> None:
        assert VerbosityPrinter(Verbosity.QUIET).is_quiet() is True
        assert VerbosityPrinter(Verbosity.NORMAL).is_quiet() is False

    def test_is_verbose(self) -> None:
        assert VerbosityPrinter(Verbosity.VERBOSE).is_verbose() is True
        assert VerbosityPrinter(Verbosity.NORMAL).is_verbose() is False


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCLIVerbosityFlags:
    """Test that CLI parses --quiet and --verbose flags correctly."""

    def test_quiet_flag(self) -> None:

        # We test via bench_main's argparse indirectly
        import argparse

        parser = argparse.ArgumentParser()

        from xpyd_bench.cli import _add_vllm_compat_args

        _add_vllm_compat_args(parser)
        ns = parser.parse_args(["--quiet"])
        assert ns.verbosity == "quiet"

    def test_verbose_flag(self) -> None:
        import argparse

        parser = argparse.ArgumentParser()

        from xpyd_bench.cli import _add_vllm_compat_args

        _add_vllm_compat_args(parser)
        ns = parser.parse_args(["--verbose"])
        assert ns.verbosity == "verbose"

    def test_default_is_none(self) -> None:
        import argparse

        parser = argparse.ArgumentParser()

        from xpyd_bench.cli import _add_vllm_compat_args

        _add_vllm_compat_args(parser)
        ns = parser.parse_args([])
        assert ns.verbosity is None

    def test_short_flags(self) -> None:
        import argparse

        parser = argparse.ArgumentParser()

        from xpyd_bench.cli import _add_vllm_compat_args

        _add_vllm_compat_args(parser)

        ns_q = parser.parse_args(["-q"])
        assert ns_q.verbosity == "quiet"

        ns_v = parser.parse_args(["-v"])
        assert ns_v.verbosity == "verbose"


class TestYAMLVerbosityConfig:
    """Test that verbosity can be set via YAML config."""

    def test_yaml_verbosity_quiet(self, tmp_path) -> None:
        import argparse

        import yaml

        from xpyd_bench.cli import _add_vllm_compat_args, _load_yaml_config

        cfg_path = tmp_path / "bench.yaml"
        cfg_path.write_text(yaml.dump({"verbosity": "quiet"}))

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args([])

        args = _load_yaml_config(str(cfg_path), args)
        assert args.verbosity == "quiet"

    def test_cli_overrides_yaml(self, tmp_path) -> None:
        import argparse

        import yaml

        from xpyd_bench.cli import _add_vllm_compat_args, _load_yaml_config

        cfg_path = tmp_path / "bench.yaml"
        cfg_path.write_text(yaml.dump({"verbosity": "quiet"}))

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--verbose"])

        args = _load_yaml_config(str(cfg_path), args, explicit_keys={"verbosity"})
        assert args.verbosity == "verbose"
