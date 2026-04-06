"""Benchmark diff report with statistical significance testing (M54).

Provides visual side-by-side comparison of two benchmark results with
Mann-Whitney U statistical significance testing, HTML diff reports with
highlighted regressions/improvements, and Markdown diff output for PR comments.
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Metric classification (reuse from compare.py)
# ---------------------------------------------------------------------------

_LOWER_IS_BETTER = {
    "mean_ttft_ms",
    "p50_ttft_ms",
    "p90_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p50_tpot_ms",
    "p90_tpot_ms",
    "p95_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "p50_itl_ms",
    "p90_itl_ms",
    "p95_itl_ms",
    "p99_itl_ms",
    "mean_e2el_ms",
    "p50_e2el_ms",
    "p90_e2el_ms",
    "p95_e2el_ms",
    "p99_e2el_ms",
}

_HIGHER_IS_BETTER = {
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
}

_ALL_METRICS = sorted(_LOWER_IS_BETTER | _HIGHER_IS_BETTER)


# ---------------------------------------------------------------------------
# Mann-Whitney U test (pure-Python, no scipy dependency)
# ---------------------------------------------------------------------------

def _mann_whitney_u(x: list[float], y: list[float]) -> tuple[float, float]:
    """Compute Mann-Whitney U statistic and approximate two-sided p-value.

    Uses normal approximation for n > 20; exact would require large tables.
    Returns (U_statistic, p_value).  When samples are too small (< 3),
    returns (0.0, 1.0) — not significant.
    """
    n1, n2 = len(x), len(y)
    if n1 < 3 or n2 < 3:
        return 0.0, 1.0

    # Rank all values together
    combined = [(v, 0) for v in x] + [(v, 1) for v in y]
    combined.sort(key=lambda t: t[0])

    # Assign ranks (handle ties via average rank)
    ranks: list[float] = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-indexed average
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    r1 = sum(ranks[k] for k in range(len(combined)) if combined[k][1] == 0)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)

    # Normal approximation
    mu = n1 * n2 / 2.0
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if sigma == 0:
        return u, 1.0

    z = abs(u - mu) / sigma
    # Two-sided p-value via complementary error function approximation
    p = _erfc_approx(z / math.sqrt(2))
    return u, p


def _erfc_approx(x: float) -> float:
    """Approximate erfc(x) using Abramowitz & Stegun formula 7.1.26."""
    if x < 0:
        return 2.0 - _erfc_approx(-x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (
        0.254829592
        + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
    )
    return poly * math.exp(-(x * x))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MetricDiff:
    """Diff result for a single summary metric."""

    name: str
    baseline: float
    candidate: float
    delta: float
    pct_change: float
    direction: str  # "improved" | "regressed" | "unchanged"
    significant: bool  # Mann-Whitney significant at p < alpha
    p_value: float | None  # None when per-request data unavailable


@dataclass
class DiffResult:
    """Full diff result."""

    metrics: list[MetricDiff] = field(default_factory=list)
    has_significant_regression: bool = False
    alpha: float = 0.05
    baseline_info: dict[str, Any] = field(default_factory=dict)
    candidate_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "has_significant_regression": self.has_significant_regression,
            "metrics": [
                {
                    "name": m.name,
                    "baseline": m.baseline,
                    "candidate": m.candidate,
                    "delta": m.delta,
                    "pct_change": m.pct_change,
                    "direction": m.direction,
                    "significant": m.significant,
                    "p_value": m.p_value,
                }
                for m in self.metrics
            ],
        }


# ---------------------------------------------------------------------------
# Core diff logic
# ---------------------------------------------------------------------------

def _extract_summary(data: dict[str, Any]) -> dict[str, float]:
    src = data.get("summary", data)
    out: dict[str, float] = {}
    for k in _ALL_METRICS:
        v = src.get(k)
        if v is not None:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                pass
    return out


def _extract_per_request_latencies(data: dict[str, Any]) -> list[float]:
    """Extract per-request e2e latency list for significance testing."""
    requests = data.get("requests", [])
    return [r["latency_ms"] for r in requests if isinstance(r, dict) and "latency_ms" in r]


def _extract_per_request_ttft(data: dict[str, Any]) -> list[float]:
    requests = data.get("requests", [])
    return [
        r["ttft_ms"]
        for r in requests
        if isinstance(r, dict) and r.get("ttft_ms") is not None
    ]


def diff(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    alpha: float = 0.05,
) -> DiffResult:
    """Compare two benchmark results with statistical significance testing."""
    base_s = _extract_summary(baseline)
    cand_s = _extract_summary(candidate)
    common = sorted(set(base_s) & set(cand_s))

    # Per-request data for significance tests
    base_lat = _extract_per_request_latencies(baseline)
    cand_lat = _extract_per_request_latencies(candidate)
    base_ttft = _extract_per_request_ttft(baseline)
    cand_ttft = _extract_per_request_ttft(candidate)

    metrics: list[MetricDiff] = []
    has_sig_reg = False

    for key in common:
        bval = base_s[key]
        cval = cand_s[key]
        delta = cval - bval
        if bval == 0:
            pct = 0.0 if cval == 0 else float("inf")
        else:
            pct = (delta / abs(bval)) * 100.0

        lower_better = key in _LOWER_IS_BETTER
        if abs(pct) < 0.01:
            direction = "unchanged"
        elif lower_better:
            direction = "regressed" if delta > 0 else "improved"
        else:
            direction = "regressed" if delta < 0 else "improved"

        # Significance test using per-request data when available
        p_value: float | None = None
        if "ttft" in key and base_ttft and cand_ttft:
            _, p_value = _mann_whitney_u(base_ttft, cand_ttft)
        elif ("e2el" in key or "latency" in key) and base_lat and cand_lat:
            _, p_value = _mann_whitney_u(base_lat, cand_lat)
        elif base_lat and cand_lat:
            # Use latency as proxy for tpot/itl significance
            _, p_value = _mann_whitney_u(base_lat, cand_lat)

        significant = p_value is not None and p_value < alpha
        if direction == "regressed" and significant:
            has_sig_reg = True

        metrics.append(
            MetricDiff(
                name=key,
                baseline=round(bval, 4),
                candidate=round(cval, 4),
                delta=round(delta, 4),
                pct_change=round(pct, 2),
                direction=direction,
                significant=significant,
                p_value=round(p_value, 6) if p_value is not None else None,
            )
        )

    return DiffResult(
        metrics=metrics,
        has_significant_regression=has_sig_reg,
        alpha=alpha,
        baseline_info=_run_info(baseline),
        candidate_info=_run_info(candidate),
    )


def _run_info(data: dict[str, Any]) -> dict[str, Any]:
    """Extract brief run info for display."""
    return {
        k: data.get(k)
        for k in ("model", "base_url", "endpoint", "num_prompts", "partial", "environment")
        if data.get(k) is not None
    }


# ---------------------------------------------------------------------------
# Terminal table
# ---------------------------------------------------------------------------

def format_diff_table(result: DiffResult) -> str:
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("  xPyD-bench — Benchmark Diff Report")
    lines.append(f"  Significance level (alpha): {result.alpha}")
    lines.append("=" * 100)
    lines.append("")
    hdr = (
        f"  {'Metric':<28s} {'Baseline':>10s} {'Candidate':>10s} "
        f"{'Delta':>10s} {'Change':>8s} {'p-value':>8s} {'Status':>14s}"
    )
    lines.append(hdr)
    lines.append("  " + "-" * 96)

    for m in result.metrics:
        sign = "+" if m.pct_change > 0 else ""
        p_str = f"{m.p_value:.4f}" if m.p_value is not None else "n/a"
        if m.direction == "regressed" and m.significant:
            status = "↓ SIG REGRESS"
        elif m.direction == "regressed":
            status = "↓ regressed"
        elif m.direction == "improved" and m.significant:
            status = "↑ SIG IMPROVED"
        elif m.direction == "improved":
            status = "↑ improved"
        else:
            status = "unchanged"
        lines.append(
            f"  {m.name:<28s} {m.baseline:>10.2f} {m.candidate:>10.2f} "
            f"{m.delta:>+10.2f} {sign}{m.pct_change:>6.1f}% {p_str:>8s} {status:>14s}"
        )

    lines.append("  " + "-" * 96)
    lines.append("")
    if result.has_significant_regression:
        lines.append("  ⚠️  SIGNIFICANT REGRESSION DETECTED (p < {})".format(result.alpha))
    else:
        lines.append("  ✅  No significant regressions detected")
    lines.append("=" * 100)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown export
# ---------------------------------------------------------------------------

def generate_markdown_diff(result: DiffResult) -> str:
    """Generate Markdown table for PR comments."""
    lines: list[str] = []
    lines.append("## xPyD-bench Diff Report")
    lines.append("")
    lines.append(f"Significance level (alpha): {result.alpha}")
    lines.append("")
    lines.append(
        "| Metric | Baseline | Candidate | Delta | Change | p-value | Status |"
    )
    lines.append(
        "|--------|----------|-----------|-------|--------|---------|--------|"
    )

    for m in result.metrics:
        sign = "+" if m.pct_change > 0 else ""
        p_str = f"{m.p_value:.4f}" if m.p_value is not None else "n/a"
        if m.direction == "regressed" and m.significant:
            status = "🔴 SIG REGRESS"
        elif m.direction == "regressed":
            status = "🟡 regressed"
        elif m.direction == "improved" and m.significant:
            status = "🟢 SIG IMPROVED"
        elif m.direction == "improved":
            status = "🟢 improved"
        else:
            status = "⚪ unchanged"
        lines.append(
            f"| {m.name} | {m.baseline:.2f} | {m.candidate:.2f} "
            f"| {m.delta:+.2f} | {sign}{m.pct_change:.1f}% | {p_str} | {status} |"
        )

    lines.append("")
    if result.has_significant_regression:
        lines.append("**⚠️ SIGNIFICANT REGRESSION DETECTED**")
    else:
        lines.append("**✅ No significant regressions detected**")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML export
# ---------------------------------------------------------------------------

def _sparkline_svg(values: list[float], width: int = 120, height: int = 24) -> str:
    """Generate a tiny inline SVG sparkline."""
    if not values:
        return ""
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1.0
    n = len(values)
    step = width / max(n - 1, 1)
    points = " ".join(
        f"{i * step:.1f},{height - (v - mn) / rng * (height - 2) - 1:.1f}"
        for i, v in enumerate(values)
    )
    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        f'<polyline points="{points}" fill="none" stroke="currentColor" stroke-width="1.5"/>'
        f"</svg>"
    )


def _histogram_svg(
    values: list[float], bins: int = 20, width: int = 120, height: int = 24
) -> str:
    """Generate a tiny inline SVG histogram."""
    if not values:
        return ""
    mn, mx = min(values), max(values)
    if mn == mx:
        return f'<svg width="{width}" height="{height}"></svg>'
    bin_width = (mx - mn) / bins
    counts = [0] * bins
    for v in values:
        idx = min(int((v - mn) / bin_width), bins - 1)
        counts[idx] += 1
    max_count = max(counts) or 1
    bar_w = width / bins
    bars = "".join(
        f'<rect x="{i * bar_w:.1f}" y="{height - c / max_count * height:.1f}" '
        f'width="{bar_w - 0.5:.1f}" height="{c / max_count * height:.1f}" '
        f'fill="currentColor" opacity="0.7"/>'
        for i, c in enumerate(counts)
    )
    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        f"{bars}</svg>"
    )


def generate_html_diff(
    result: DiffResult,
    baseline_data: dict[str, Any] | None = None,
    candidate_data: dict[str, Any] | None = None,
) -> str:
    """Generate self-contained HTML diff report."""

    # Extract per-request latencies for sparklines
    base_lat = _extract_per_request_latencies(baseline_data or {})
    cand_lat = _extract_per_request_latencies(candidate_data or {})

    rows = ""
    for m in result.metrics:
        sign = "+" if m.pct_change > 0 else ""
        p_str = f"{m.p_value:.4f}" if m.p_value is not None else "n/a"
        if m.direction == "regressed" and m.significant:
            cls = "sig-regress"
            badge = "SIG REGRESS"
        elif m.direction == "regressed":
            cls = "regress"
            badge = "regressed"
        elif m.direction == "improved" and m.significant:
            cls = "sig-improved"
            badge = "SIG IMPROVED"
        elif m.direction == "improved":
            cls = "improved"
            badge = "improved"
        else:
            cls = "unchanged"
            badge = "unchanged"
        rows += (
            f'<tr class="{cls}">'
            f"<td>{html_mod.escape(m.name)}</td>"
            f"<td>{m.baseline:.2f}</td>"
            f"<td>{m.candidate:.2f}</td>"
            f"<td>{m.delta:+.2f}</td>"
            f"<td>{sign}{m.pct_change:.1f}%</td>"
            f"<td>{p_str}</td>"
            f'<td><span class="badge badge-{cls}">{badge}</span></td>'
            f"</tr>\n"
        )

    # Distribution sparklines
    dist_section = ""
    if base_lat and cand_lat:
        base_hist = _histogram_svg(base_lat)
        cand_hist = _histogram_svg(cand_lat)
        dist_section = f"""
        <h2>Latency Distribution Comparison</h2>
        <div class="dist-row">
          <div class="dist-box">
            <h3>Baseline</h3>
            <div class="sparkline">{base_hist}</div>
            <p>n={len(base_lat)}, mean={sum(base_lat)/len(base_lat):.1f}ms</p>
          </div>
          <div class="dist-box">
            <h3>Candidate</h3>
            <div class="sparkline">{cand_hist}</div>
            <p>n={len(cand_lat)}, mean={sum(cand_lat)/len(cand_lat):.1f}ms</p>
          </div>
        </div>
        """

    verdict = (
        '<div class="verdict regress">⚠️ SIGNIFICANT REGRESSION DETECTED</div>'
        if result.has_significant_regression
        else '<div class="verdict ok">✅ No significant regressions detected</div>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>xPyD-bench Diff Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         margin: 2em auto; max-width: 1100px; color: #222; background: #fafafa; }}
  h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.3em; }}
  table {{ width: 100%; border-collapse: collapse; margin: 1em 0; }}
  th, td {{ padding: 8px 12px; text-align: right; border-bottom: 1px solid #ddd; }}
  th {{ background: #f0f0f0; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  tr.sig-regress {{ background: #fdd; }}
  tr.regress {{ background: #fff3cd; }}
  tr.sig-improved {{ background: #d4edda; }}
  tr.improved {{ background: #e8f5e9; }}
  .badge {{ padding: 2px 8px; border-radius: 4px; font-size: 0.85em; font-weight: 600; }}
  .badge-sig-regress {{ background: #dc3545; color: #fff; }}
  .badge-regress {{ background: #ffc107; color: #333; }}
  .badge-sig-improved {{ background: #28a745; color: #fff; }}
  .badge-improved {{ background: #8bc34a; color: #fff; }}
  .badge-unchanged {{ background: #ccc; color: #555; }}
  .verdict {{ font-size: 1.2em; font-weight: 700; padding: 1em; margin: 1em 0;
              border-radius: 6px; text-align: center; }}
  .verdict.ok {{ background: #d4edda; color: #155724; }}
  .verdict.regress {{ background: #f8d7da; color: #721c24; }}
  .dist-row {{ display: flex; gap: 2em; }}
  .dist-box {{ flex: 1; background: #fff; padding: 1em; border-radius: 6px;
               box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .sparkline svg {{ color: #555; }}
  .info {{ font-size: 0.9em; color: #666; }}
</style>
</head>
<body>
<h1>xPyD-bench Diff Report</h1>
<p class="info">Significance level (alpha): {result.alpha}</p>

{verdict}

<h2>Metric Comparison</h2>
<table>
<thead>
<tr><th>Metric</th><th>Baseline</th><th>Candidate</th><th>Delta</th>
<th>Change</th><th>p-value</th><th>Status</th></tr>
</thead>
<tbody>
{rows}
</tbody>
</table>

{dist_section}

</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def diff_main(argv: list[str] | None = None) -> None:
    """CLI entry point for ``xpyd-bench diff``."""
    parser = argparse.ArgumentParser(
        prog="xpyd-bench diff",
        description="Compare two benchmark results with statistical significance testing.",
    )
    parser.add_argument("baseline", help="Path to baseline result JSON")
    parser.add_argument("candidate", help="Path to candidate result JSON")
    parser.add_argument(
        "--html-diff", metavar="PATH", default=None, help="Write HTML diff report to PATH"
    )
    parser.add_argument(
        "--markdown-diff", metavar="PATH", default=None, help="Write Markdown diff to PATH"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for Mann-Whitney U test (default 0.05)",
    )
    parser.add_argument(
        "--json-diff", metavar="PATH", default=None, help="Write JSON diff to PATH"
    )

    args = parser.parse_args(argv)

    # Load results
    with open(args.baseline) as f:
        baseline_data = json.load(f)
    with open(args.candidate) as f:
        candidate_data = json.load(f)

    result = diff(baseline_data, candidate_data, alpha=args.alpha)

    # Terminal output
    print(format_diff_table(result))

    # HTML export
    if args.html_diff:
        html_content = generate_html_diff(result, baseline_data, candidate_data)
        Path(args.html_diff).write_text(html_content, encoding="utf-8")
        print(f"\nHTML diff report written to {args.html_diff}")

    # Markdown export
    if args.markdown_diff:
        md_content = generate_markdown_diff(result)
        Path(args.markdown_diff).write_text(md_content, encoding="utf-8")
        print(f"\nMarkdown diff written to {args.markdown_diff}")

    # JSON export
    if args.json_diff:
        Path(args.json_diff).write_text(
            json.dumps(result.to_dict(), indent=2), encoding="utf-8"
        )
        print(f"\nJSON diff written to {args.json_diff}")

    # Exit code 1 on significant regression
    if result.has_significant_regression:
        sys.exit(1)
