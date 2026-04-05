"""Benchmark result diffing by tag (M95).

Groups historical results by a tag key and computes cross-group metric
comparison with statistical significance testing.

Usage:
    xpyd-bench tag-compare --result-dir <path> --group-by <tag-key>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Metric definitions (reuse classification from compare/diff)
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

# Metrics available in history summaries (subset we can use)
_SUMMARY_METRICS = [
    "request_throughput",
    "output_throughput",
    "mean_ttft_ms",
    "mean_e2el_ms",
]


# ---------------------------------------------------------------------------
# Mann-Whitney U test (pure-Python, no scipy dependency)
# ---------------------------------------------------------------------------

def _mann_whitney_u(x: list[float], y: list[float]) -> tuple[float, float]:
    """Compute Mann-Whitney U statistic and approximate two-sided p-value.

    Uses normal approximation for n > 20; for smaller samples returns
    (0.0, 1.0) — not significant.
    """
    n1, n2 = len(x), len(y)
    if n1 < 3 or n2 < 3:
        return 0.0, 1.0

    combined = [(v, 0) for v in x] + [(v, 1) for v in y]
    combined.sort(key=lambda t: t[0])

    ranks: list[float] = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2  # 1-based average rank
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    r1 = sum(ranks[k] for k in range(len(combined)) if combined[k][1] == 0)
    u1 = r1 - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    u = min(u1, u2)

    mu = n1 * n2 / 2
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    if sigma == 0:
        return u, 1.0
    z = abs(u - mu) / sigma

    # Approximate two-sided p-value via error function
    p = math.erfc(z / math.sqrt(2))
    return u, p


# ---------------------------------------------------------------------------
# Full-result loading (reads JSON files for all metrics, not just summaries)
# ---------------------------------------------------------------------------

def _load_full_results(result_dir: Path) -> list[dict]:
    """Load all JSON result files from *result_dir* with full metric data."""
    results: list[dict] = []
    if not result_dir.is_dir():
        return results
    for p in sorted(result_dir.glob("*.json")):
        try:
            with open(p) as f:
                data = json.load(f)
            data["_file"] = p.name
            results.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return results


# ---------------------------------------------------------------------------
# Grouping & comparison
# ---------------------------------------------------------------------------

def group_results_by_tag(
    results: list[dict],
    tag_key: str,
) -> dict[str, list[dict]]:
    """Group result dicts by the value of *tag_key* in their ``tags`` field.

    Results without the tag key are silently skipped.
    """
    groups: dict[str, list[dict]] = {}
    for r in results:
        tags = r.get("tags", {}) or {}
        val = tags.get(tag_key)
        if val is not None:
            groups.setdefault(str(val), []).append(r)
    return groups


def _mean(vals: list[float]) -> float | None:
    if not vals:
        return None
    return sum(vals) / len(vals)


def compute_group_stats(
    groups: dict[str, list[dict]],
) -> dict[str, dict[str, Any]]:
    """Compute per-group aggregate metrics.

    Returns ``{group_name: {metric: mean_value, ...}, ...}``.
    """
    stats: dict[str, dict[str, Any]] = {}
    for name, results in groups.items():
        group_metrics: dict[str, Any] = {"count": len(results)}
        for metric in _ALL_METRICS:
            vals = [r[metric] for r in results if r.get(metric) is not None]
            group_metrics[metric] = _mean(vals)
        stats[name] = group_metrics
    return stats


def compute_pairwise_significance(
    groups: dict[str, list[dict]],
    metric: str,
) -> list[dict[str, Any]]:
    """Compute pairwise Mann-Whitney U test between all group pairs for *metric*.

    Returns list of ``{group_a, group_b, u_stat, p_value, significant}`` dicts.
    """
    group_names = sorted(groups.keys())
    results: list[dict[str, Any]] = []
    for i, a in enumerate(group_names):
        for b in group_names[i + 1:]:
            vals_a = [r[metric] for r in groups[a] if r.get(metric) is not None]
            vals_b = [r[metric] for r in groups[b] if r.get(metric) is not None]
            if not vals_a or not vals_b:
                continue
            u, p = _mann_whitney_u(vals_a, vals_b)
            results.append({
                "group_a": a,
                "group_b": b,
                "metric": metric,
                "u_stat": u,
                "p_value": p,
                "significant": p < 0.05,
            })
    return results


def tag_compare(
    result_dir: str | Path,
    group_by: str,
) -> dict[str, Any]:
    """Run full tag-based comparison and return structured result.

    Returns a dict with ``groups``, ``stats``, ``significance``, and metadata.
    """
    rdir = Path(result_dir)
    results = _load_full_results(rdir)
    groups = group_results_by_tag(results, group_by)

    if not groups:
        return {
            "group_by": group_by,
            "groups": {},
            "stats": {},
            "significance": [],
            "error": f"No results found with tag key '{group_by}'",
        }

    stats = compute_group_stats(groups)

    # Pairwise significance for key metrics
    significance: list[dict[str, Any]] = []
    for metric in _ALL_METRICS:
        significance.extend(compute_pairwise_significance(groups, metric))

    return {
        "group_by": group_by,
        "groups": {k: len(v) for k, v in groups.items()},
        "stats": stats,
        "significance": significance,
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_tag_compare_table(result: dict[str, Any]) -> str:
    """Format tag comparison result as a human-readable table."""
    if "error" in result:
        return f"Error: {result['error']}"

    stats = result["stats"]
    group_names = sorted(stats.keys())
    if not group_names:
        return "No groups to compare."

    lines: list[str] = []
    lines.append(f"Tag comparison by '{result['group_by']}'")
    group_labels = ", ".join(
        "{g} ({n} runs)".format(g=g, n=stats[g]["count"]) for g in group_names
    )
    lines.append("Groups: " + group_labels)
    lines.append("")

    # Metrics table
    header = f"{'Metric':<28}" + "".join(f"{g:>16}" for g in group_names)
    lines.append(header)
    lines.append("-" * len(header))

    for metric in _ALL_METRICS:
        vals = [stats[g].get(metric) for g in group_names]
        if all(v is None for v in vals):
            continue
        row = f"{metric:<28}"
        for v in vals:
            if v is None:
                row += f"{'N/A':>16}"
            else:
                row += f"{v:>16.2f}"
        lines.append(row)

    # Significance summary
    sig_results = [s for s in result["significance"] if s["significant"]]
    if sig_results:
        lines.append("")
        lines.append("Statistically significant differences (p < 0.05):")
        for s in sig_results:
            lines.append(
                f"  {s['metric']}: {s['group_a']} vs {s['group_b']} "
                f"(U={s['u_stat']:.1f}, p={s['p_value']:.4f})"
            )
    else:
        lines.append("")
        lines.append("No statistically significant differences found.")

    return "\n".join(lines)


def format_tag_compare_markdown(result: dict[str, Any]) -> str:
    """Format tag comparison result as a Markdown table."""
    if "error" in result:
        return f"**Error:** {result['error']}"

    stats = result["stats"]
    group_names = sorted(stats.keys())
    if not group_names:
        return "No groups to compare."

    lines: list[str] = []
    lines.append(f"## Tag Comparison: `{result['group_by']}`")
    lines.append("")

    # Metrics table
    header = "| Metric |" + " | ".join(f"{g} ({stats[g]['count']})" for g in group_names) + " |"
    sep = "|---|" + "|".join("---:" for _ in group_names) + "|"
    lines.append(header)
    lines.append(sep)

    for metric in _ALL_METRICS:
        vals = [stats[g].get(metric) for g in group_names]
        if all(v is None for v in vals):
            continue
        cells = []
        for v in vals:
            cells.append("N/A" if v is None else f"{v:.2f}")
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")

    # Significance
    sig_results = [s for s in result["significance"] if s["significant"]]
    if sig_results:
        lines.append("")
        lines.append("### Significant Differences (p < 0.05)")
        lines.append("")
        lines.append("| Metric | Group A | Group B | U-stat | p-value |")
        lines.append("|---|---|---|---:|---:|")
        for s in sig_results:
            lines.append(
                f"| {s['metric']} | {s['group_a']} | {s['group_b']} "
                f"| {s['u_stat']:.1f} | {s['p_value']:.4f} |"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def tag_compare_main(argv: list[str] | None = None) -> None:
    """Entry point for ``xpyd-bench tag-compare`` subcommand."""
    parser = argparse.ArgumentParser(
        prog="xpyd-bench tag-compare",
        description="Group benchmark results by tag and compare cross-group metrics",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        required=True,
        help="Directory containing benchmark result JSON files.",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        required=True,
        help="Tag key to group results by (e.g. 'gpu', 'env').",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        dest="json_output",
        help="Output comparison as JSON.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        default=False,
        help="Output comparison as Markdown.",
    )

    args = parser.parse_args(argv)

    result = tag_compare(args.result_dir, args.group_by)

    if args.json_output:
        print(json.dumps(result, indent=2, default=str))
    elif args.markdown:
        print(format_tag_compare_markdown(result))
    else:
        print(format_tag_compare_table(result))

    # Exit non-zero if no groups found
    if "error" in result:
        sys.exit(1)
