"""Tests for M23: Configuration Dump & Validation."""

from __future__ import annotations

import pytest
import yaml

from xpyd_bench.config_cmd import (
    _load_and_check_yaml,
    config_dump_main,
    config_validate_main,
)


class TestLoadAndCheckYaml:
    """Test the YAML loading and validation helper."""

    def test_valid_config(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({"model": "gpt-4", "num_prompts": 100}))
        cfg, warnings, errors = _load_and_check_yaml(str(cfg_path))
        assert cfg == {"model": "gpt-4", "num_prompts": 100}
        assert not warnings
        assert not errors

    def test_unknown_key_warning(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({"model": "gpt-4", "bogus_key": 42}))
        cfg, warnings, errors = _load_and_check_yaml(str(cfg_path))
        assert len(warnings) == 1
        assert "Unknown key 'bogus_key'" in warnings[0]
        assert not errors

    def test_file_not_found(self):
        cfg, warnings, errors = _load_and_check_yaml("/nonexistent/path.yaml")
        assert len(errors) == 1
        assert "not found" in errors[0]

    def test_invalid_yaml_syntax(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("{{invalid: yaml: [")
        cfg, warnings, errors = _load_and_check_yaml(str(cfg_path))
        assert len(errors) == 1
        assert "Invalid YAML syntax" in errors[0]

    def test_empty_config(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("")
        cfg, warnings, errors = _load_and_check_yaml(str(cfg_path))
        assert len(warnings) == 1
        assert "empty" in warnings[0].lower()
        assert not errors

    def test_non_dict_config(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("- item1\n- item2\n")
        cfg, warnings, errors = _load_and_check_yaml(str(cfg_path))
        assert len(errors) == 1
        assert "mapping" in errors[0].lower()

    def test_hyphenated_keys_normalized(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({"request-rate": 10, "input-len": 128}))
        cfg, warnings, errors = _load_and_check_yaml(str(cfg_path))
        assert not warnings
        assert not errors


class TestConfigDump:
    """Test config dump subcommand."""

    def test_dump_defaults(self, capsys):
        config_dump_main([])
        output = capsys.readouterr().out
        parsed = yaml.safe_load(output)
        assert isinstance(parsed, dict)
        assert "model" in parsed
        assert "num_prompts" in parsed
        assert parsed["num_prompts"] == 1000

    def test_dump_with_yaml(self, tmp_path, capsys):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({"model": "my-model", "num_prompts": 50}))
        config_dump_main(["--config", str(cfg_path)])
        output = capsys.readouterr().out
        parsed = yaml.safe_load(output)
        assert parsed["model"] == "my-model"
        assert parsed["num_prompts"] == 50

    def test_dump_with_invalid_config(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("{{bad yaml")
        with pytest.raises(SystemExit) as exc_info:
            config_dump_main(["--config", str(cfg_path)])
        assert exc_info.value.code == 1

    def test_dump_unknown_key_warning(self, tmp_path, capsys):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({"model": "x", "fake_option": True}))
        config_dump_main(["--config", str(cfg_path)])
        err = capsys.readouterr().err
        assert "WARNING" in err
        assert "fake_option" in err


class TestConfigValidate:
    """Test config validate subcommand."""

    def test_valid_config(self, tmp_path, capsys):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({"model": "gpt-4", "num_prompts": 100}))
        with pytest.raises(SystemExit) as exc_info:
            config_validate_main(["--config", str(cfg_path)])
        assert exc_info.value.code == 0
        assert "Validation OK" in capsys.readouterr().out

    def test_invalid_yaml(self, tmp_path, capsys):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("{{bad")
        with pytest.raises(SystemExit) as exc_info:
            config_validate_main(["--config", str(cfg_path)])
        assert exc_info.value.code == 1
        assert "FAILED" in capsys.readouterr().out

    def test_missing_file(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            config_validate_main(["--config", "/no/such/file.yaml"])
        assert exc_info.value.code == 1

    def test_unknown_keys_still_pass(self, tmp_path, capsys):
        """Unknown keys produce warnings but validation still passes."""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({"model": "x", "unknown_key": 1}))
        with pytest.raises(SystemExit) as exc_info:
            config_validate_main(["--config", str(cfg_path)])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "Validation OK" in captured.out

    def test_non_dict_fails(self, tmp_path, capsys):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("[1, 2, 3]")
        with pytest.raises(SystemExit) as exc_info:
            config_validate_main(["--config", str(cfg_path)])
        assert exc_info.value.code == 1
