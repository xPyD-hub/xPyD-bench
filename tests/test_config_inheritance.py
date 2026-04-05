"""Tests for configuration inheritance via ``extends`` (M80)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import yaml

from xpyd_bench.config_cmd import (
    ConfigInheritanceError,
    _load_and_check_yaml,
    _resolve_yaml_chain,
)


@pytest.fixture()
def tmp_configs(tmp_path: Path):
    """Helper to write YAML config files in tmp_path."""

    def _write(name: str, data: dict) -> str:
        p = tmp_path / name
        p.write_text(yaml.dump(data))
        return str(p)

    return _write


class TestResolveYamlChain:
    """Tests for _resolve_yaml_chain()."""

    def test_simple_no_extends(self, tmp_configs) -> None:
        path = tmp_configs("base.yaml", {"model": "gpt-4", "num_prompts": 10})
        cfg = _resolve_yaml_chain(path)
        assert cfg == {"model": "gpt-4", "num_prompts": 10}

    def test_single_level_inheritance(self, tmp_configs) -> None:
        tmp_configs("base.yaml", {"model": "gpt-4", "num_prompts": 100, "temperature": 0.5})
        child_path = tmp_configs("child.yaml", {"extends": "base.yaml", "num_prompts": 10})
        cfg = _resolve_yaml_chain(child_path)
        assert cfg["model"] == "gpt-4"
        assert cfg["num_prompts"] == 10  # child overrides
        assert cfg["temperature"] == 0.5  # inherited from parent

    def test_multi_level_inheritance(self, tmp_configs) -> None:
        tmp_configs("grandparent.yaml", {"model": "gpt-3.5", "temperature": 0.1, "stream": True})
        tmp_configs("parent.yaml", {"extends": "grandparent.yaml", "model": "gpt-4"})
        child_path = tmp_configs("child.yaml", {"extends": "parent.yaml", "temperature": 0.9})
        cfg = _resolve_yaml_chain(child_path)
        assert cfg["model"] == "gpt-4"  # from parent
        assert cfg["temperature"] == 0.9  # from child
        assert cfg["stream"] is True  # from grandparent

    def test_circular_inheritance_detected(self, tmp_configs) -> None:
        tmp_configs("a.yaml", {"extends": "b.yaml", "model": "x"})
        tmp_configs("b.yaml", {"extends": "a.yaml", "model": "y"})
        with pytest.raises(ConfigInheritanceError, match="Circular"):
            _resolve_yaml_chain(str(Path(tmp_configs("a.yaml", {"extends": "b.yaml"}))))

    def test_self_reference_detected(self, tmp_configs) -> None:
        path = tmp_configs("self.yaml", {"extends": "self.yaml"})
        with pytest.raises(ConfigInheritanceError, match="Circular"):
            _resolve_yaml_chain(path)

    def test_missing_parent(self, tmp_configs) -> None:
        path = tmp_configs("child.yaml", {"extends": "nonexistent.yaml"})
        with pytest.raises(ConfigInheritanceError, match="not found"):
            _resolve_yaml_chain(path)

    def test_extends_key_removed(self, tmp_configs) -> None:
        tmp_configs("base.yaml", {"model": "gpt-4"})
        child_path = tmp_configs("child.yaml", {"extends": "base.yaml", "num_prompts": 5})
        cfg = _resolve_yaml_chain(child_path)
        assert "extends" not in cfg

    def test_empty_parent(self, tmp_configs) -> None:
        tmp_configs("empty.yaml", {})
        child_path = tmp_configs("child.yaml", {"extends": "empty.yaml", "model": "gpt-4"})
        cfg = _resolve_yaml_chain(child_path)
        assert cfg == {"model": "gpt-4"}


class TestLoadAndCheckYamlWithExtends:
    """Integration tests for _load_and_check_yaml with extends."""

    def test_inheritance_resolved(self, tmp_configs) -> None:
        tmp_configs("base.yaml", {"model": "gpt-4", "num_prompts": 100})
        child_path = tmp_configs("child.yaml", {"extends": "base.yaml", "num_prompts": 5})
        cfg, warnings, errors = _load_and_check_yaml(child_path)
        assert not errors
        assert cfg["model"] == "gpt-4"
        assert cfg["num_prompts"] == 5

    def test_circular_reports_error(self, tmp_configs) -> None:
        tmp_configs("a.yaml", {"extends": "b.yaml"})
        tmp_configs("b.yaml", {"extends": "a.yaml"})
        _, _, errors = _load_and_check_yaml(
            tmp_configs("a.yaml", {"extends": "b.yaml"})
        )
        assert any("Circular" in e for e in errors)

    def test_missing_parent_reports_error(self, tmp_configs) -> None:
        path = tmp_configs("child.yaml", {"extends": "ghost.yaml"})
        _, _, errors = _load_and_check_yaml(path)
        assert any("not found" in e for e in errors)

    def test_unknown_keys_in_parent_warned(self, tmp_configs) -> None:
        tmp_configs("base.yaml", {"model": "gpt-4", "bogus_key": 123})
        child_path = tmp_configs("child.yaml", {"extends": "base.yaml"})
        cfg, warnings, errors = _load_and_check_yaml(child_path)
        assert not errors
        assert any("bogus_key" in w for w in warnings)


class TestCliLoadYamlConfig:
    """Test CLI _load_yaml_config with extends support."""

    def test_cli_precedence_over_inheritance(self, tmp_configs) -> None:
        from xpyd_bench.cli import _load_yaml_config

        tmp_configs("base.yaml", {"model": "gpt-3.5", "num_prompts": 100})
        child_path = tmp_configs("child.yaml", {"extends": "base.yaml", "num_prompts": 50})

        args = argparse.Namespace(model="cli-model", num_prompts=None, temperature=None)
        result = _load_yaml_config(child_path, args, explicit_keys={"model"})
        assert result.model == "cli-model"  # CLI wins
        assert result.num_prompts == 50  # from child yaml
