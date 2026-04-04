"""Tests for _KNOWN_KEYS completeness in config_cmd."""

from xpyd_bench.config_cmd import _KNOWN_KEYS


def test_rate_pattern_in_known_keys():
    """rate_pattern is a valid YAML-only key used by runner (M4)."""
    assert "rate_pattern" in _KNOWN_KEYS


def test_shutdown_grace_period_in_known_keys():
    """shutdown_grace_period is a valid YAML-only key used by runner (M12)."""
    assert "shutdown_grace_period" in _KNOWN_KEYS


def test_config_validate_no_spurious_warnings(tmp_path):
    """Config with rate_pattern and shutdown_grace_period should not warn."""
    import yaml

    from xpyd_bench.config_cmd import _load_and_check_yaml

    cfg = {
        "rate_pattern": [{"duration": 10, "rate": 5.0}],
        "shutdown_grace_period": 10.0,
        "base_url": "http://localhost:8000",
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg))

    _, warnings, errors = _load_and_check_yaml(str(p))
    assert not errors
    # No warnings about unknown keys
    unknown_warnings = [w for w in warnings if "Unknown key" in w]
    assert not unknown_warnings, f"Spurious warnings: {unknown_warnings}"
