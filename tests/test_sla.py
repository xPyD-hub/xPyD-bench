"""Tests for SLA validation (M20)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from xpyd_bench.bench.models import BenchmarkResult
from xpyd_bench.sla import (
    SLATarget,
    format_sla_table,
    load_sla_targets,
    sla_report_to_dict,
    validate_sla,
)


def _make_result(**kwargs) -> BenchmarkResult:
    """Create a BenchmarkResult with sensible defaults."""
    defaults = dict(
        completed=100,
        failed=0,
        p99_ttft_ms=400.0,
        p95_e2el_ms=1500.0,
        output_throughput=200.0,
        mean_ttft_ms=200.0,
        request_throughput=50.0,
    )
    defaults.update(kwargs)
    return BenchmarkResult(**defaults)


class TestLoadSLATargets:
    def test_load_valid(self, tmp_path: Path) -> None:
        sla_file = tmp_path / "sla.yaml"
        sla_file.write_text(
            yaml.dump(
                {
                    "targets": {
                        "p99_ttft_ms": {"max": 500},
                        "output_throughput": {"min": 100},
                    }
                }
            )
        )
        targets = load_sla_targets(sla_file)
        assert len(targets) == 2
        metrics = {t.metric: t for t in targets}
        assert "p99_ttft_ms" in metrics
        assert metrics["p99_ttft_ms"].max == 500
        assert "output_throughput" in metrics
        assert metrics["output_throughput"].min == 100

    def test_load_missing_targets_key(self, tmp_path: Path) -> None:
        sla_file = tmp_path / "sla.yaml"
        sla_file.write_text(yaml.dump({"foo": "bar"}))
        with pytest.raises(ValueError, match="targets"):
            load_sla_targets(sla_file)

    def test_load_invalid_constraint(self, tmp_path: Path) -> None:
        sla_file = tmp_path / "sla.yaml"
        sla_file.write_text(yaml.dump({"targets": {"p99_ttft_ms": 500}}))
        with pytest.raises(ValueError, match="dict"):
            load_sla_targets(sla_file)


class TestValidateSLA:
    def test_all_pass(self) -> None:
        result = _make_result(p99_ttft_ms=400.0, output_throughput=200.0)
        targets = [
            SLATarget(metric="p99_ttft_ms", max=500),
            SLATarget(metric="output_throughput", min=100),
        ]
        report = validate_sla(result, targets)
        assert report.all_passed
        assert all(c.passed for c in report.checks)

    def test_max_violation(self) -> None:
        result = _make_result(p99_ttft_ms=600.0)
        targets = [SLATarget(metric="p99_ttft_ms", max=500)]
        report = validate_sla(result, targets)
        assert not report.all_passed
        assert not report.checks[0].passed
        assert "max" in report.checks[0].reason

    def test_min_violation(self) -> None:
        result = _make_result(output_throughput=50.0)
        targets = [SLATarget(metric="output_throughput", min=100)]
        report = validate_sla(result, targets)
        assert not report.all_passed
        assert "min" in report.checks[0].reason

    def test_missing_metric(self) -> None:
        result = _make_result()
        targets = [SLATarget(metric="nonexistent_metric", max=100)]
        report = validate_sla(result, targets)
        assert not report.all_passed
        assert "not found" in report.checks[0].reason

    def test_error_rate_virtual_metric(self) -> None:
        result = _make_result(completed=95, failed=5)
        targets = [SLATarget(metric="error_rate", max=0.01)]
        report = validate_sla(result, targets)
        assert not report.all_passed  # 5% > 1%

    def test_error_rate_passes(self) -> None:
        result = _make_result(completed=999, failed=1)
        targets = [SLATarget(metric="error_rate", max=0.01)]
        report = validate_sla(result, targets)
        assert report.all_passed

    def test_partial_sla(self) -> None:
        """Some pass, some fail."""
        result = _make_result(p99_ttft_ms=400.0, output_throughput=50.0)
        targets = [
            SLATarget(metric="p99_ttft_ms", max=500),
            SLATarget(metric="output_throughput", min=100),
        ]
        report = validate_sla(result, targets)
        assert not report.all_passed
        assert report.checks[0].passed
        assert not report.checks[1].passed


class TestFormatSLATable:
    def test_table_contains_metrics(self) -> None:
        result = _make_result(p99_ttft_ms=400.0)
        targets = [SLATarget(metric="p99_ttft_ms", max=500)]
        report = validate_sla(result, targets)
        table = format_sla_table(report)
        assert "p99_ttft_ms" in table
        assert "PASS" in table
        assert "ALL PASSED" in table

    def test_table_shows_fail(self) -> None:
        result = _make_result(p99_ttft_ms=600.0)
        targets = [SLATarget(metric="p99_ttft_ms", max=500)]
        report = validate_sla(result, targets)
        table = format_sla_table(report)
        assert "FAIL" in table


class TestSLAReportToDict:
    def test_serializable(self) -> None:
        result = _make_result(p99_ttft_ms=400.0)
        targets = [SLATarget(metric="p99_ttft_ms", max=500)]
        report = validate_sla(result, targets)
        d = sla_report_to_dict(report)
        # Must be JSON serializable
        json.dumps(d)
        assert d["all_passed"] is True
        assert len(d["checks"]) == 1
        assert d["checks"][0]["passed"] is True
