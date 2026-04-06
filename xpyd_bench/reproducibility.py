"""Benchmark Reproducibility Score (M99).

Run the benchmark N times and compute a reproducibility score based on
coefficient of variation (CV) across runs for key metrics.

Score thresholds:
    CV < 5%  → Excellent (90-100)
    CV < 10% → Good (70-89)
    CV < 15% → Fair (50-69)
    CV >= 15% → Poor (0-49)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# Key metrics to evaluate for reproducibility
_METRIC_KEYS = [
    ("mean_ttft_ms", "Mean TTFT (ms)"),
    ("mean_tpot_ms", "Mean TPOT (ms)"),
    ("mean_e2el_ms", "Mean Latency (ms)"),
    ("request_throughput", "Throughput (req/s)"),
    ("output_throughput", "Output Throughput (tok/s)"),
]


@dataclass
class MetricReproducibility:
    """Reproducibility stats for a single metric."""

    name: str = ""
    display_name: str = ""
    values: list[float] = field(default_factory=list)
    mean: float = 0.0
    stddev: float = 0.0
    cv: float = 0.0  # Coefficient of variation (0-1)
    cv_pct: float = 0.0  # CV as percentage
    classification: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "values": self.values,
            "mean": round(self.mean, 4),
            "stddev": round(self.stddev, 4),
            "cv_pct": round(self.cv_pct, 2),
            "classification": self.classification,
        }


@dataclass
class ReproducibilityResult:
    """Overall reproducibility assessment."""

    num_runs: int = 0
    completed_runs: int = 0
    metrics: list[MetricReproducibility] = field(default_factory=list)
    overall_score: float = 0.0
    overall_classification: str = "unknown"
    unstable_metrics: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "num_runs": self.num_runs,
            "completed_runs": self.completed_runs,
            "overall_score": round(self.overall_score, 1),
            "overall_classification": self.overall_classification,
            "metrics": [m.to_dict() for m in self.metrics],
            "unstable_metrics": self.unstable_metrics,
            "recommendations": self.recommendations,
        }


def classify_cv(cv_pct: float) -> tuple[str, float]:
    """Classify CV percentage into a label and score.

    Returns (classification, score) where score is 0-100.
    """
    if cv_pct < 5.0:
        # Excellent: linear map 0-5% → 100-90
        score = 100.0 - (cv_pct / 5.0) * 10.0
        return "excellent", score
    elif cv_pct < 10.0:
        # Good: linear map 5-10% → 89-70
        score = 89.0 - ((cv_pct - 5.0) / 5.0) * 19.0
        return "good", score
    elif cv_pct < 15.0:
        # Fair: linear map 10-15% → 69-50
        score = 69.0 - ((cv_pct - 10.0) / 5.0) * 19.0
        return "fair", score
    else:
        # Poor: linear map 15-30% → 49-0, capped at 0
        score = max(0.0, 49.0 - ((cv_pct - 15.0) / 15.0) * 49.0)
        return "poor", score


def _compute_metric_repro(
    name: str, display_name: str, values: list[float]
) -> MetricReproducibility:
    """Compute reproducibility stats for a single metric."""
    mr = MetricReproducibility(name=name, display_name=display_name, values=values)
    if len(values) < 2:
        return mr

    mr.mean = sum(values) / len(values)
    variance = sum((v - mr.mean) ** 2 for v in values) / (len(values) - 1)
    mr.stddev = math.sqrt(variance)

    if mr.mean > 0:
        mr.cv = mr.stddev / mr.mean
    else:
        mr.cv = 0.0
    mr.cv_pct = mr.cv * 100.0
    mr.classification, _ = classify_cv(mr.cv_pct)
    return mr


def compute_reproducibility(
    run_results: list[dict[str, Any]],
    num_runs: int | None = None,
) -> ReproducibilityResult:
    """Compute reproducibility score from multiple run results.

    Args:
        run_results: List of benchmark result dicts (JSON-serializable).
        num_runs: Requested number of runs (may differ from len(run_results)
                  if interrupted).

    Returns:
        ReproducibilityResult with per-metric stats and overall score.
    """
    result = ReproducibilityResult(
        num_runs=num_runs or len(run_results),
        completed_runs=len(run_results),
    )

    if len(run_results) < 2:
        result.overall_score = 0.0
        result.overall_classification = "insufficient_data"
        result.recommendations.append(
            "Need at least 2 completed runs for reproducibility analysis."
        )
        return result

    # Compute per-metric reproducibility
    scores: list[float] = []
    for key, display in _METRIC_KEYS:
        values = []
        for r in run_results:
            v = r.get(key)
            if v is not None and isinstance(v, (int, float)) and not math.isnan(v):
                values.append(float(v))

        if len(values) < 2:
            continue

        mr = _compute_metric_repro(key, display, values)
        result.metrics.append(mr)

        _, score = classify_cv(mr.cv_pct)
        scores.append(score)

        if mr.cv_pct >= 10.0:
            result.unstable_metrics.append(mr.display_name)

    # Overall score is mean of per-metric scores
    if scores:
        result.overall_score = sum(scores) / len(scores)
    else:
        result.overall_score = 0.0

    result.overall_classification, _ = classify_cv(
        # Reverse-map score to a pseudo-CV for classification
        # Instead, just classify based on score thresholds directly
        0.0
    )
    # Classify overall based on score
    if result.overall_score >= 90:
        result.overall_classification = "excellent"
    elif result.overall_score >= 70:
        result.overall_classification = "good"
    elif result.overall_score >= 50:
        result.overall_classification = "fair"
    else:
        result.overall_classification = "poor"

    # Generate recommendations
    result.recommendations = _generate_recommendations(result)

    return result


def _generate_recommendations(result: ReproducibilityResult) -> list[str]:
    """Generate actionable recommendations based on reproducibility analysis."""
    recs: list[str] = []

    if result.completed_runs < 5:
        recs.append(
            f"Run more iterations (currently {result.completed_runs}). "
            "5+ runs gives more reliable CV estimates."
        )

    if result.overall_classification in ("poor", "fair"):
        recs.append(
            "Consider adding warmup requests (--warmup) to reduce cold-start effects."
        )
        recs.append(
            "Increase benchmark duration or num_prompts to reduce variance "
            "from short runs."
        )
        recs.append(
            "Pin CPU frequency and disable power management to reduce "
            "hardware-induced variance."
        )
        recs.append(
            "Close other workloads on the benchmark machine to minimize "
            "resource contention."
        )

    # Check specific unstable metrics
    for m in result.metrics:
        if m.name == "mean_ttft_ms" and m.cv_pct >= 15.0:
            recs.append(
                "TTFT is highly variable — check for cold model loading or "
                "KV cache pressure."
            )
        if m.name == "request_throughput" and m.cv_pct >= 10.0:
            recs.append(
                "Throughput instability may indicate server-side batching variance "
                "or resource contention."
            )

    if not recs and result.overall_classification == "excellent":
        recs.append("Benchmark is highly reproducible. No action needed.")

    return recs


def print_reproducibility_report(result: ReproducibilityResult) -> None:
    """Print a human-readable reproducibility report."""
    print(f"\n{'=' * 60}")
    print(f"  Reproducibility Report ({result.completed_runs}/{result.num_runs} runs)")
    print(f"{'=' * 60}")

    # Overall score
    emoji = {
        "excellent": "🟢",
        "good": "🔵",
        "fair": "🟡",
        "poor": "🔴",
        "insufficient_data": "⚪",
    }.get(result.overall_classification, "⚪")

    print(
        f"\n  Overall Score: {result.overall_score:.1f}/100 "
        f"{emoji} {result.overall_classification.upper()}"
    )

    # Per-metric table
    if result.metrics:
        print(f"\n  {'Metric':<25} {'Mean':>12} {'StdDev':>12} {'CV%':>8} {'Rating':>10}")
        print("  " + "-" * 69)
        for m in result.metrics:
            rating_emoji = {
                "excellent": "🟢",
                "good": "🔵",
                "fair": "🟡",
                "poor": "🔴",
            }.get(m.classification, "⚪")
            print(
                f"  {m.display_name:<25} {m.mean:>12.2f} {m.stddev:>12.2f} "
                f"{m.cv_pct:>7.1f}% {rating_emoji} {m.classification}"
            )

    # Unstable metrics
    if result.unstable_metrics:
        print(f"\n  ⚠ Unstable metrics: {', '.join(result.unstable_metrics)}")

    # Recommendations
    if result.recommendations:
        print("\n  Recommendations:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"    {i}. {rec}")

    print()
