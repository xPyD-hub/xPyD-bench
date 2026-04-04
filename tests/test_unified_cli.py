"""Tests for M24: Unified CLI Subcommand Interface."""

from __future__ import annotations

import warnings

import pytest
import yaml

from xpyd_bench.main import (
    _get_version,
    deprecated_compare_main,
    deprecated_config_dump_main,
    deprecated_config_validate_main,
    deprecated_multi_main,
    deprecated_profile_main,
    deprecated_replay_main,
    main,
)


class TestVersionFlag:
    """Test --version flag."""

    def test_version_prints_and_exits(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["xpyd-bench", "--version"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "xpyd-bench" in out

    def test_get_version_returns_string(self):
        v = _get_version()
        assert isinstance(v, str)
        assert len(v) > 0


class TestSubcommandRouting:
    """Test that subcommands route correctly."""

    def test_run_subcommand(self, monkeypatch):
        """'xpyd-bench run --help' should trigger bench_main."""
        monkeypatch.setattr("sys.argv", ["xpyd-bench", "run", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

    def test_compare_subcommand(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["xpyd-bench", "compare", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

    def test_multi_subcommand(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["xpyd-bench", "multi", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

    def test_profile_subcommand(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["xpyd-bench", "profile", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

    def test_replay_subcommand(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["xpyd-bench", "replay", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

    def test_config_dump_subcommand(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["xpyd-bench", "config", "dump"])
        main()
        out = capsys.readouterr().out
        parsed = yaml.safe_load(out)
        assert isinstance(parsed, dict)
        assert "model" in parsed

    def test_config_validate_subcommand(self, monkeypatch, tmp_path, capsys):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.dump({"model": "gpt-4"}))
        monkeypatch.setattr(
            "sys.argv", ["xpyd-bench", "config", "validate", "--config", str(cfg)]
        )
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        assert "Validation OK" in capsys.readouterr().out

    def test_config_no_sub(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["xpyd-bench", "config"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_config_unknown_sub(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["xpyd-bench", "config", "bogus"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


class TestBackwardCompat:
    """Test that no subcommand defaults to 'run'."""

    def test_no_subcommand_defaults_to_run(self, monkeypatch):
        """'xpyd-bench --help' should trigger bench_main (run)."""
        monkeypatch.setattr("sys.argv", ["xpyd-bench", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

    def test_unknown_arg_goes_to_run(self, monkeypatch):
        """Unknown first arg (not a subcommand) passes to run."""
        monkeypatch.setattr("sys.argv", ["xpyd-bench", "--model", "gpt-4", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0


class TestDeprecationWarnings:
    """Test legacy entry points emit deprecation warnings."""

    def test_deprecated_compare(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["xpyd-bench-compare", "--help"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with pytest.raises(SystemExit):
                deprecated_compare_main()
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()

    def test_deprecated_multi(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["xpyd-bench-multi", "--help"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with pytest.raises(SystemExit):
                deprecated_multi_main()
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_deprecated_profile(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["xpyd-bench-profile", "--help"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with pytest.raises(SystemExit):
                deprecated_profile_main()
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_deprecated_replay(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["xpyd-bench-replay", "--help"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with pytest.raises(SystemExit):
                deprecated_replay_main()
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_deprecated_config_dump(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["xpyd-bench-config-dump"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            deprecated_config_dump_main()
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_deprecated_config_validate(self, monkeypatch, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.dump({"model": "x"}))
        monkeypatch.setattr(
            "sys.argv", ["xpyd-bench-config-validate", "--config", str(cfg)]
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with pytest.raises(SystemExit):
                deprecated_config_validate_main()
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
