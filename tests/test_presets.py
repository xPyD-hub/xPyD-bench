"""Tests for M38: Benchmark Presets Library."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from xpyd_bench.presets import (
    COLD_START,
    LATENCY_OPTIMAL,
    SOAK_TEST,
    THROUGHPUT_MAX,
    PresetConfig,
    format_preset_detail,
    format_preset_list,
    get_preset,
    list_presets,
    load_user_presets,
)


class TestBuiltinPresets:
    """Verify built-in presets exist and have expected structure."""

    def test_all_builtins_present(self):
        presets = list_presets(presets_dir=Path("/nonexistent"))
        names = {p.name for p in presets}
        assert names == {"throughput-max", "latency-optimal", "soak-test", "cold-start"}

    @pytest.mark.parametrize(
        "preset",
        [THROUGHPUT_MAX, LATENCY_OPTIMAL, SOAK_TEST, COLD_START],
        ids=lambda p: p.name,
    )
    def test_preset_has_config(self, preset: PresetConfig):
        assert preset.name
        assert preset.description
        assert preset.source == "built-in"
        assert isinstance(preset.config, dict)
        assert len(preset.config) > 0

    def test_to_overrides(self):
        overrides = THROUGHPUT_MAX.to_overrides()
        assert overrides["num_prompts"] == 10000
        assert overrides["max_concurrency"] == 512

    def test_get_preset_builtin(self):
        p = get_preset("latency-optimal")
        assert p.name == "latency-optimal"
        assert p.config["max_concurrency"] == 1

    def test_get_preset_not_found(self):
        with pytest.raises(KeyError, match="Unknown preset"):
            get_preset("nonexistent-preset")


class TestUserPresets:
    """Test user-defined preset loading from YAML files."""

    def test_load_from_dir(self, tmp_path: Path):
        preset_file = tmp_path / "my-preset.yaml"
        preset_file.write_text(
            yaml.dump(
                {
                    "name": "my-preset",
                    "description": "Custom test preset",
                    "num_prompts": 42,
                    "input_len": 512,
                    "request_rate": 5.0,
                }
            )
        )
        presets = load_user_presets(tmp_path)
        assert "my-preset" in presets
        p = presets["my-preset"]
        assert p.source == "user"
        assert p.config["num_prompts"] == 42
        assert p.config["input_len"] == 512

    def test_user_overrides_builtin(self, tmp_path: Path):
        preset_file = tmp_path / "throughput-max.yaml"
        preset_file.write_text(
            yaml.dump(
                {
                    "name": "throughput-max",
                    "description": "My custom throughput preset",
                    "num_prompts": 999,
                }
            )
        )
        p = get_preset("throughput-max", presets_dir=tmp_path)
        assert p.source == "user"
        assert p.config["num_prompts"] == 999

    def test_name_defaults_to_stem(self, tmp_path: Path):
        preset_file = tmp_path / "auto-name.yaml"
        preset_file.write_text(yaml.dump({"description": "No name field", "num_prompts": 7}))
        presets = load_user_presets(tmp_path)
        assert "auto-name" in presets

    def test_empty_dir(self, tmp_path: Path):
        presets = load_user_presets(tmp_path)
        assert presets == {}

    def test_nonexistent_dir(self):
        presets = load_user_presets(Path("/nonexistent/path"))
        assert presets == {}

    def test_malformed_yaml_skipped(self, tmp_path: Path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(": : : invalid yaml {{{{")
        good = tmp_path / "good.yaml"
        good.write_text(yaml.dump({"name": "good", "description": "ok", "num_prompts": 1}))
        presets = load_user_presets(tmp_path)
        assert "good" in presets
        assert "bad" not in presets

    def test_list_presets_includes_user(self, tmp_path: Path):
        preset_file = tmp_path / "extra.yaml"
        preset_file.write_text(
            yaml.dump({"name": "extra", "description": "Extra preset", "num_prompts": 5})
        )
        all_presets = list_presets(presets_dir=tmp_path)
        names = {p.name for p in all_presets}
        assert "extra" in names
        assert "throughput-max" in names


class TestFormatting:
    """Test display formatting functions."""

    def test_format_preset_detail(self):
        output = format_preset_detail(LATENCY_OPTIMAL)
        assert "latency-optimal" in output
        assert "built-in" in output
        assert "max_concurrency: 1" in output

    def test_format_preset_list(self):
        presets = list_presets(presets_dir=Path("/nonexistent"))
        output = format_preset_list(presets)
        assert "throughput-max" in output
        assert "soak-test" in output
        assert "Available presets" in output

    def test_format_user_preset_shows_tag(self):
        user_preset = PresetConfig(
            name="custom", description="Test", source="user", config={"num_prompts": 1}
        )
        output = format_preset_list([user_preset])
        assert "[user]" in output


class TestCLIIntegration:
    """Test CLI --preset flag and presets subcommand."""

    def test_presets_list_subcommand(self, capsys):
        from xpyd_bench.main import main

        with patch("sys.argv", ["xpyd-bench", "presets", "list"]):
            main()
        out = capsys.readouterr().out
        assert "throughput-max" in out
        assert "latency-optimal" in out

    def test_presets_show_subcommand(self, capsys):
        from xpyd_bench.main import main

        with patch("sys.argv", ["xpyd-bench", "presets", "show", "cold-start"]):
            main()
        out = capsys.readouterr().out
        assert "cold-start" in out
        assert "num_prompts" in out

    def test_presets_show_not_found(self):
        from xpyd_bench.main import main

        with patch("sys.argv", ["xpyd-bench", "presets", "show", "nope"]):
            with pytest.raises(SystemExit, match="1"):
                main()

    def test_preset_flag_in_dry_run(self, capsys):
        from xpyd_bench.main import main

        with patch(
            "sys.argv",
            ["xpyd-bench", "run", "--preset", "throughput-max", "--dry-run", "--model", "test"],
        ):
            main()
        out = capsys.readouterr().out
        assert "10000" in out

    def test_cli_flag_overrides_preset(self, capsys):
        from xpyd_bench.main import main

        with patch(
            "sys.argv",
            [
                "xpyd-bench", "run", "--preset", "throughput-max",
                "--num-prompts", "42", "--dry-run", "--model", "test",
            ],
        ):
            main()
        out = capsys.readouterr().out
        assert "42" in out

    def test_presets_list_with_user_dir(self, tmp_path: Path, capsys):
        from xpyd_bench.main import main

        preset_file = tmp_path / "my-test.yaml"
        preset_file.write_text(
            yaml.dump({"name": "my-test", "description": "CLI test preset", "num_prompts": 77})
        )
        with patch(
            "sys.argv",
            ["xpyd-bench", "presets", "list", "--presets-dir", str(tmp_path)],
        ):
            main()
        out = capsys.readouterr().out
        assert "my-test" in out
        assert "[user]" in out


class TestYAMLConfigPreset:
    """Test preset application via YAML config."""

    def test_preset_from_yaml(self, tmp_path: Path, capsys):
        from xpyd_bench.main import main

        config = tmp_path / "config.yaml"
        config.write_text(yaml.dump({"preset": "latency-optimal"}))
        with patch(
            "sys.argv",
            ["xpyd-bench", "run", "--config", str(config), "--dry-run", "--model", "test"],
        ):
            main()
        out = capsys.readouterr().out
        assert "1.0" in out
