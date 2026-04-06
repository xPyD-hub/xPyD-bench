"""Tests for M13: Custom HTTP Headers."""

from __future__ import annotations

from argparse import Namespace

import pytest
import yaml

from xpyd_bench.cli import _parse_header, _resolve_custom_headers

# ---------------------------------------------------------------------------
# CLI parsing tests
# ---------------------------------------------------------------------------


class TestParseHeader:
    """Test _parse_header helper."""

    def test_simple(self):
        assert _parse_header("X-Custom: value") == ("X-Custom", "value")

    def test_strips_whitespace(self):
        assert _parse_header("  X-Key :  some value  ") == ("X-Key", "some value")

    def test_value_with_colons(self):
        k, v = _parse_header("X-Data: a:b:c")
        assert k == "X-Data"
        assert v == "a:b:c"

    def test_missing_colon_raises(self):
        with pytest.raises(ValueError, match="Invalid header format"):
            _parse_header("NoColonHere")


class TestResolveCustomHeaders:
    """Test _resolve_custom_headers merging logic."""

    def test_empty(self):
        args = Namespace(header=None, headers=None)
        assert _resolve_custom_headers(args) == {}

    def test_yaml_only(self):
        args = Namespace(header=None, headers={"X-From": "yaml", "X-Other": "v"})
        result = _resolve_custom_headers(args)
        assert result == {"X-From": "yaml", "X-Other": "v"}

    def test_cli_only(self):
        args = Namespace(header=["X-Cli: value1", "X-Cli2: value2"], headers=None)
        result = _resolve_custom_headers(args)
        assert result == {"X-Cli": "value1", "X-Cli2": "value2"}

    def test_cli_overrides_yaml(self):
        args = Namespace(
            header=["X-Key: from-cli"],
            headers={"X-Key": "from-yaml", "X-Only-Yaml": "y"},
        )
        result = _resolve_custom_headers(args)
        assert result["X-Key"] == "from-cli"
        assert result["X-Only-Yaml"] == "y"


# ---------------------------------------------------------------------------
# YAML config integration
# ---------------------------------------------------------------------------


class TestYamlHeaders:
    """Test that YAML config headers are loaded correctly."""

    def test_yaml_config_headers(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump({"headers": {"X-Yaml-Header": "yaml-value"}})
        )

        from xpyd_bench.cli import _load_yaml_config

        args = Namespace(
            header=None,
            headers=None,
            base_url="http://localhost:8000",
        )
        args = _load_yaml_config(str(config_file), args)
        assert args.headers == {"X-Yaml-Header": "yaml-value"}


class TestAuthPrecedence:
    """Test that explicit custom Authorization header overrides api_key."""

    def test_custom_auth_overrides_api_key(self):
        args = Namespace(
            header=["Authorization: Token custom-tok"],
            headers=None,
            api_key="bearer-key",
            custom_headers=None,
        )
        custom = _resolve_custom_headers(args)
        assert custom["Authorization"] == "Token custom-tok"

        # Simulate runner logic: custom_headers has Authorization,
        # so api_key's Bearer should NOT override
        headers: dict[str, str] = {}
        headers.update(custom)
        if args.api_key and "Authorization" not in custom:
            headers["Authorization"] = f"Bearer {args.api_key}"
        assert headers["Authorization"] == "Token custom-tok"
