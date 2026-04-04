"""Tests for M37: Request Body Templating."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from xpyd_bench.templating import (
    TemplateError,
    apply_templates,
    load_template_vars,
    render_template,
)

# ---------------------------------------------------------------------------
# render_template
# ---------------------------------------------------------------------------


class TestRenderTemplate:
    def test_simple_substitution(self):
        result = render_template("Hello {{ name }}!", {"name": "World"})
        assert result == "Hello World!"

    def test_multiple_variables(self):
        result = render_template(
            "{{ greeting }} {{ name }}, welcome to {{ place }}.",
            {"greeting": "Hi", "name": "Alice", "place": "Wonderland"},
        )
        assert result == "Hi Alice, welcome to Wonderland."

    def test_no_placeholders(self):
        result = render_template("No placeholders here.", {"foo": "bar"})
        assert result == "No placeholders here."

    def test_missing_variable_raises(self):
        with pytest.raises(TemplateError, match="Undefined template variable: 'missing'"):
            render_template("Hello {{ missing }}!", {})

    def test_whitespace_variants(self):
        for template in ["{{name}}", "{{ name }}", "{{  name  }}"]:
            assert render_template(template, {"name": "OK"}) == "OK"

    def test_numeric_value(self):
        result = render_template("Count: {{ n }}", {"n": 42})
        assert result == "Count: 42"

    def test_repeated_variable(self):
        result = render_template("{{ x }} and {{ x }}", {"x": "same"})
        assert result == "same and same"

    def test_empty_string_value(self):
        result = render_template("prefix{{ v }}suffix", {"v": ""})
        assert result == "prefixsuffix"


# ---------------------------------------------------------------------------
# apply_templates
# ---------------------------------------------------------------------------


class TestApplyTemplates:
    def test_list_of_prompts(self):
        prompts = ["Hello {{ name }}", "Goodbye {{ name }}"]
        result = apply_templates(prompts, {"name": "Bob"})
        assert result == ["Hello Bob", "Goodbye Bob"]

    def test_empty_list(self):
        assert apply_templates([], {"a": "b"}) == []

    def test_error_propagates(self):
        with pytest.raises(TemplateError):
            apply_templates(["{{ undefined }}"], {})


# ---------------------------------------------------------------------------
# load_template_vars
# ---------------------------------------------------------------------------


class TestLoadTemplateVars:
    def test_load_json(self, tmp_path: Path):
        p = tmp_path / "vars.json"
        p.write_text(json.dumps({"env": "prod", "gpu": "A100"}))
        result = load_template_vars(str(p))
        assert result == {"env": "prod", "gpu": "A100"}

    def test_load_yaml(self, tmp_path: Path):
        p = tmp_path / "vars.yaml"
        p.write_text(yaml.dump({"model": "llama", "size": "70b"}))
        result = load_template_vars(str(p))
        assert result == {"model": "llama", "size": "70b"}

    def test_load_yml(self, tmp_path: Path):
        p = tmp_path / "vars.yml"
        p.write_text(yaml.dump({"key": "value"}))
        result = load_template_vars(str(p))
        assert result == {"key": "value"}

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_template_vars("/nonexistent/path.json")

    def test_unsupported_extension(self, tmp_path: Path):
        p = tmp_path / "vars.txt"
        p.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported template vars format"):
            load_template_vars(str(p))

    def test_non_dict_raises(self, tmp_path: Path):
        p = tmp_path / "vars.json"
        p.write_text(json.dumps([1, 2, 3]))
        with pytest.raises(ValueError, match="must contain a dict"):
            load_template_vars(str(p))


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCLIIntegration:
    """Test --template-vars flag is recognized by the argument parser."""

    def _make_parser(self):
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        return parser

    def test_parser_accepts_flag(self):
        parser = self._make_parser()
        args = parser.parse_args(["--template-vars", "/some/path.json"])
        assert args.template_vars == "/some/path.json"

    def test_parser_default_none(self):
        parser = self._make_parser()
        args = parser.parse_args([])
        assert args.template_vars is None


# ---------------------------------------------------------------------------
# YAML config integration
# ---------------------------------------------------------------------------


class TestYAMLConfig:
    def test_template_vars_is_known_key(self):
        from xpyd_bench.config_cmd import _KNOWN_KEYS

        assert "template_vars" in _KNOWN_KEYS


# ---------------------------------------------------------------------------
# End-to-end: load + render
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_load_and_apply(self, tmp_path: Path):
        p = tmp_path / "vars.json"
        p.write_text(json.dumps({"user": "Alice", "topic": "AI"}))
        variables = load_template_vars(str(p))
        prompts = [
            "{{ user }} asks about {{ topic }}",
            "Tell {{ user }} more about {{ topic }}",
        ]
        result = apply_templates(prompts, variables)
        assert result == [
            "Alice asks about AI",
            "Tell Alice more about AI",
        ]
