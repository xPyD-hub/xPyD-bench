"""Tests for M39: Cost Estimation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from xpyd_bench.bench.models import BenchmarkResult
from xpyd_bench.cost import (
    cost_to_dict,
    estimate_cost,
    estimate_cost_from_counts,
    format_cost_summary,
    load_cost_model,
)


@pytest.fixture()
def cost_model_file(tmp_path: Path) -> Path:
    """Create a sample cost model YAML file."""
    data = {
        "currency": "USD",
        "default": {"input": 0.01, "output": 0.03},
        "models": {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        },
    }
    p = tmp_path / "cost_model.yaml"
    p.write_text(yaml.dump(data))
    return p


@pytest.fixture()
def cost_model_no_default(tmp_path: Path) -> Path:
    """Cost model without default pricing."""
    data = {
        "models": {
            "gpt-4": {"input": 0.03, "output": 0.06},
        },
    }
    p = tmp_path / "cost_no_default.yaml"
    p.write_text(yaml.dump(data))
    return p


class TestLoadCostModel:
    def test_load_valid(self, cost_model_file: Path) -> None:
        cm = load_cost_model(cost_model_file)
        assert "gpt-4" in cm.models
        assert cm.models["gpt-4"]["input"] == 0.03
        assert cm.models["gpt-4"]["output"] == 0.06
        assert cm.default is not None
        assert cm.default["input"] == 0.01

    def test_load_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_cost_model(tmp_path / "nope.yaml")

    def test_load_invalid_format(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("- a list")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_cost_model(p)

    def test_load_no_default(self, cost_model_no_default: Path) -> None:
        cm = load_cost_model(cost_model_no_default)
        assert cm.default is None


class TestEstimateCost:
    def test_known_model(self, cost_model_file: Path) -> None:
        cm = load_cost_model(cost_model_file)
        result = BenchmarkResult(
            model="gpt-4",
            total_input_tokens=10000,
            total_output_tokens=5000,
        )
        est = estimate_cost(result, cm)
        assert est.matched is True
        assert est.model == "gpt-4"
        assert est.input_cost == pytest.approx(10000 / 1000 * 0.03)
        assert est.output_cost == pytest.approx(5000 / 1000 * 0.06)
        assert est.total_cost == pytest.approx(est.input_cost + est.output_cost)
        assert est.currency == "USD"

    def test_fallback_to_default(self, cost_model_file: Path) -> None:
        cm = load_cost_model(cost_model_file)
        result = BenchmarkResult(
            model="unknown-model",
            total_input_tokens=2000,
            total_output_tokens=1000,
        )
        est = estimate_cost(result, cm)
        assert est.matched is False
        assert est.input_cost == pytest.approx(2000 / 1000 * 0.01)
        assert est.output_cost == pytest.approx(1000 / 1000 * 0.03)

    def test_no_pricing_raises(self, cost_model_no_default: Path) -> None:
        cm = load_cost_model(cost_model_no_default)
        result = BenchmarkResult(
            model="unknown-model",
            total_input_tokens=1000,
            total_output_tokens=500,
        )
        with pytest.raises(ValueError, match="No pricing found"):
            estimate_cost(result, cm)

    def test_model_override(self, cost_model_file: Path) -> None:
        cm = load_cost_model(cost_model_file)
        result = BenchmarkResult(
            model="some-other",
            total_input_tokens=1000,
            total_output_tokens=500,
        )
        est = estimate_cost(result, cm, model_override="gpt-3.5-turbo")
        assert est.model == "gpt-3.5-turbo"
        assert est.matched is True
        assert est.input_cost_per_1k == 0.0005


class TestEstimateCostFromCounts:
    def test_dry_run_estimation(self, cost_model_file: Path) -> None:
        cm = load_cost_model(cost_model_file)
        est = estimate_cost_from_counts(256000, 128000, cm, model_name="gpt-4")
        assert est.input_tokens == 256000
        assert est.output_tokens == 128000
        assert est.total_cost == pytest.approx(
            256000 / 1000 * 0.03 + 128000 / 1000 * 0.06
        )

    def test_empty_model_fallback(self, cost_model_file: Path) -> None:
        cm = load_cost_model(cost_model_file)
        est = estimate_cost_from_counts(1000, 500, cm, model_name="")
        assert est.matched is False


class TestFormatCostSummary:
    def test_format_output(self, cost_model_file: Path) -> None:
        cm = load_cost_model(cost_model_file)
        result = BenchmarkResult(
            model="gpt-4",
            total_input_tokens=10000,
            total_output_tokens=5000,
        )
        est = estimate_cost(result, cm)
        text = format_cost_summary(est)
        assert "Cost Estimation:" in text
        assert "gpt-4" in text
        assert "Total cost:" in text
        assert "USD" in text

    def test_format_unmatched(self, cost_model_file: Path) -> None:
        cm = load_cost_model(cost_model_file)
        result = BenchmarkResult(
            model="unknown",
            total_input_tokens=1000,
            total_output_tokens=500,
        )
        est = estimate_cost(result, cm)
        text = format_cost_summary(est)
        assert "default" in text


class TestCostToDict:
    def test_serialization(self, cost_model_file: Path) -> None:
        cm = load_cost_model(cost_model_file)
        result = BenchmarkResult(
            model="gpt-4",
            total_input_tokens=10000,
            total_output_tokens=5000,
        )
        est = estimate_cost(result, cm)
        d = cost_to_dict(est)
        assert d["model"] == "gpt-4"
        assert d["total_cost"] == est.total_cost
        assert d["model_matched"] is True
        # Ensure JSON-serializable
        json.dumps(d)


class TestCLIIntegration:
    """Test cost estimation via CLI flags."""

    def test_dry_run_with_cost_model(self, cost_model_file: Path, capsys) -> None:
        """--dry-run with --cost-model should print cost estimate."""
        from xpyd_bench.cli import bench_main

        bench_main([
            "--base-url", "http://localhost:9999",
            "--model", "gpt-4",
            "--num-prompts", "10",
            "--input-len", "100",
            "--output-len", "50",
            "--dry-run",
            "--cost-model", str(cost_model_file),
        ])
        captured = capsys.readouterr()
        assert "Cost Estimation:" in captured.out
        assert "gpt-4" in captured.out
        assert "Total cost:" in captured.out

    def test_cost_model_cli_flag_parsed(self) -> None:
        """Verify --cost-model is a recognized CLI flag."""
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--cost-model", "/tmp/test.yaml"])
        assert args.cost_model == "/tmp/test.yaml"

    def test_cost_model_default_none(self) -> None:
        """--cost-model defaults to None."""
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args([])
        assert args.cost_model is None
