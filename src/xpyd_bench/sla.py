"""SLA validation for benchmark results.

Loads SLA targets from a YAML file and validates BenchmarkResult metrics
against user-defined thresholds (max/min).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from xpyd_bench.bench.models import BenchmarkResult


@dataclass
class SLATarget:
    """A single SLA target for a metric."""

    metric: str
    max: float | None = None
    min: float | None = None


@dataclass
class SLACheckResult:
    """Result of checking one SLA target."""

    metric: str
    actual: float | None
    target_max: float | None = None
    target_min: float | None = None
    passed: bool = True
    reason: str = ""


@dataclass
class SLAReport:
    """Full SLA validation report."""

    checks: list[SLACheckResult] = field(default_factory=list)
    all_passed: bool = True


def load_sla_targets(path: str | Path) -> list[SLATarget]:
    """Load SLA targets from a YAML file.

    Expected format:
        targets:
          p99_ttft_ms: {max: 500}
          output_throughput: {min: 100}
          error_rate: {max: 0.01}
    """
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "targets" not in data:
        raise ValueError(f"SLA file must contain a 'targets' key: {path}")

    targets: list[SLATarget] = []
    for metric, constraints in data["targets"].items():
        if not isinstance(constraints, dict):
            raise ValueError(f"SLA target for '{metric}' must be a dict with max/min keys")
        targets.append(
            SLATarget(
                metric=metric,
                max=constraints.get("max"),
                min=constraints.get("min"),
            )
        )
    return targets


def _get_metric_value(result: BenchmarkResult, metric: str) -> float | None:
    """Extract a metric value from BenchmarkResult.

    Supports 'error_rate' as a virtual metric (failed / (completed + failed)).
    """
    if metric == "error_rate":
        total = result.completed + result.failed
        if total == 0:
            return 0.0
        return result.failed / total

    result_dict = asdict(result)
    # Remove non-numeric fields
    if metric in result_dict:
        val = result_dict[metric]
        if isinstance(val, (int, float)):
            return float(val)
    return None


def validate_sla(result: BenchmarkResult, targets: list[SLATarget]) -> SLAReport:
    """Validate benchmark result against SLA targets."""
    report = SLAReport()
    for target in targets:
        actual = _get_metric_value(result, target.metric)
        check = SLACheckResult(
            metric=target.metric,
            actual=actual,
            target_max=target.max,
            target_min=target.min,
        )

        if actual is None:
            check.passed = False
            check.reason = "metric not found"
        else:
            if target.max is not None and actual > target.max:
                check.passed = False
                check.reason = f"actual {actual:.4g} > max {target.max:.4g}"
            if target.min is not None and actual < target.min:
                check.passed = False
                check.reason = f"actual {actual:.4g} < min {target.min:.4g}"
            if check.passed:
                check.reason = "ok"

        report.checks.append(check)
        if not check.passed:
            report.all_passed = False

    return report


def format_sla_table(report: SLAReport) -> str:
    """Format SLA report as a human-readable table."""
    lines = []
    lines.append("")
    lines.append("SLA Validation Results")
    lines.append("-" * 72)
    lines.append(f"{'Metric':<25} {'Actual':>12} {'Threshold':>18} {'Status':>8}")
    lines.append("-" * 72)

    for check in report.checks:
        actual_str = f"{check.actual:.4g}" if check.actual is not None else "N/A"

        parts = []
        if check.target_max is not None:
            parts.append(f"max={check.target_max:.4g}")
        if check.target_min is not None:
            parts.append(f"min={check.target_min:.4g}")
        threshold_str = ", ".join(parts) if parts else "N/A"

        status = "PASS" if check.passed else "FAIL"
        lines.append(f"{check.metric:<25} {actual_str:>12} {threshold_str:>18} {status:>8}")

    lines.append("-" * 72)
    overall = "ALL PASSED" if report.all_passed else "FAILED"
    lines.append(f"Overall: {overall}")
    lines.append("")
    return "\n".join(lines)


def sla_report_to_dict(report: SLAReport) -> dict[str, Any]:
    """Convert SLA report to JSON-serializable dict."""
    return {
        "all_passed": report.all_passed,
        "checks": [
            {
                "metric": c.metric,
                "actual": c.actual,
                "target_max": c.target_max,
                "target_min": c.target_min,
                "passed": c.passed,
                "reason": c.reason,
            }
            for c in report.checks
        ],
    }
