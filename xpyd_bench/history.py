"""Benchmark result storage & history (M30).

Usage:
    xpyd-bench history --result-dir <path> [--last N]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

# Sparkline characters (8 levels)
_SPARK = "▁▂▃▄▅▆▇█"


def _sparkline(values: list[float]) -> str:
    """Return a sparkline string for a list of numeric values."""
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo if hi != lo else 1.0
    return "".join(_SPARK[min(int((v - lo) / span * 7), 7)] for v in values)


def _parse_timestamp_from_filename(name: str) -> datetime | None:
    """Try to extract a YYYYMMDD-HHMMSS timestamp from a result filename."""
    # Pattern: *-YYYYMMDD-HHMMSS.json
    stem = name.removesuffix(".json")
    parts = stem.rsplit("-", 2)
    if len(parts) >= 2:
        date_part = parts[-2]
        time_part = parts[-1]
        try:
            return datetime.strptime(f"{date_part}-{time_part}", "%Y%m%d-%H%M%S")
        except ValueError:
            pass
    return None


def _load_result_summary(path: Path) -> dict | None:
    """Load a result JSON and return a summary dict, or None on failure."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    # Try to get timestamp from file or data
    ts = _parse_timestamp_from_filename(path.name)
    env = data.get("environment", {})
    ts_str = env.get("timestamp", "")
    if not ts and ts_str:
        try:
            ts = datetime.fromisoformat(ts_str)
        except ValueError:
            pass
    if not ts:
        # Fall back to file mtime
        ts = datetime.fromtimestamp(path.stat().st_mtime)

    return {
        "file": path.name,
        "timestamp": ts,
        "model": data.get("model", "?"),
        "num_prompts": data.get("num_prompts", 0),
        "completed": data.get("completed", 0),
        "failed": data.get("failed", 0),
        "request_throughput": data.get("request_throughput"),
        "output_throughput": data.get("output_throughput"),
        "mean_ttft_ms": data.get("mean_ttft_ms"),
        "mean_e2el_ms": data.get("mean_e2el_ms"),
        "partial": data.get("partial", False),
        "total_duration_s": data.get("total_duration_s", 0),
        "tags": data.get("tags", {}),
        "note": data.get("note"),
        "fingerprint": data.get("fingerprint"),
    }


def list_history(
    result_dir: str | Path,
    last_n: int | None = None,
    filter_tags: dict[str, str] | None = None,
) -> list[dict]:
    """List benchmark results in *result_dir*, sorted by timestamp (newest last).

    Parameters
    ----------
    filter_tags:
        If provided, only include results whose ``tags`` contain all the
        specified key-value pairs.

    Returns a list of summary dicts.
    """
    rdir = Path(result_dir)
    if not rdir.is_dir():
        return []

    summaries: list[dict] = []
    for p in sorted(rdir.glob("*.json")):
        s = _load_result_summary(p)
        if s is not None:
            summaries.append(s)

    # Filter by tags (M36)
    if filter_tags:
        summaries = [
            s for s in summaries
            if all(s.get("tags", {}).get(k) == v for k, v in filter_tags.items())
        ]

    # Sort by timestamp
    summaries.sort(key=lambda s: s["timestamp"])

    if last_n is not None and last_n > 0:
        summaries = summaries[-last_n:]

    return summaries


def format_history_by_fingerprint(summaries: list[dict]) -> str:
    """Group history summaries by fingerprint and format as grouped table (M72)."""
    if not summaries:
        return "No benchmark results found."

    groups: dict[str, list[dict]] = {}
    ungrouped: list[dict] = []
    for s in summaries:
        fp = s.get("fingerprint")
        if fp:
            groups.setdefault(fp, []).append(s)
        else:
            ungrouped.append(s)

    lines: list[str] = []
    for fp, runs in sorted(groups.items(), key=lambda x: x[1][0]["timestamp"]):
        short_fp = fp[:12]
        lines.append(f"\n🔑 Fingerprint: {short_fp}... ({len(runs)} run(s))")
        lines.append(f"   {'Timestamp':<20} {'Model':<16} {'Thr req/s':>10} "
                     f"{'TTFT ms':>9} {'E2E ms':>9}")
        lines.append("   " + "-" * 70)
        for s in runs:
            ts = s["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            model = (s["model"] or "?")[:15]
            thr = f"{s['request_throughput']:.1f}" if s["request_throughput"] is not None else "-"
            ttft = f"{s['mean_ttft_ms']:.1f}" if s["mean_ttft_ms"] is not None else "-"
            e2e = f"{s['mean_e2el_ms']:.1f}" if s["mean_e2el_ms"] is not None else "-"
            lines.append(f"   {ts:<20} {model:<16} {thr:>10} {ttft:>9} {e2e:>9}")

    if ungrouped:
        lines.append(f"\n⚠️  No fingerprint ({len(ungrouped)} run(s))")
        for s in ungrouped:
            ts = s["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"   {ts} - {s.get('model', '?')}")

    return "\n".join(lines)


def format_history_table(summaries: list[dict]) -> str:
    """Format history summaries as a human-readable table with sparkline trends."""
    if not summaries:
        return "No benchmark results found."

    lines: list[str] = []
    lines.append(f"{'#':>3}  {'Timestamp':<20} {'Model':<16} {'Reqs':>5} "
                 f"{'OK':>5} {'Fail':>4} {'Thr req/s':>10} {'TTFT ms':>9} "
                 f"{'E2E ms':>9} {'Partial':>7}  {'Note'}")
    lines.append("-" * 130)

    for i, s in enumerate(summaries, 1):
        ts = s["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        model = (s["model"] or "?")[:15]
        thr = f"{s['request_throughput']:.1f}" if s["request_throughput"] is not None else "-"
        ttft = f"{s['mean_ttft_ms']:.1f}" if s["mean_ttft_ms"] is not None else "-"
        e2e = f"{s['mean_e2el_ms']:.1f}" if s["mean_e2el_ms"] is not None else "-"
        partial = "yes" if s["partial"] else ""
        note = s.get("note") or ""
        lines.append(
            f"{i:>3}  {ts:<20} {model:<16} {s['num_prompts']:>5} "
            f"{s['completed']:>5} {s['failed']:>4} {thr:>10} {ttft:>9} "
            f"{e2e:>9} {partial:>7}  {note}"
        )

    # Sparkline trends (if ≥2 runs)
    if len(summaries) >= 2:
        lines.append("")
        lines.append("Trends:")
        for label, key in [
            ("Throughput (req/s)", "request_throughput"),
            ("Mean TTFT (ms)", "mean_ttft_ms"),
            ("Mean E2E (ms)", "mean_e2el_ms"),
        ]:
            vals = [s[key] for s in summaries if s[key] is not None]
            if len(vals) >= 2:
                spark = _sparkline(vals)
                lines.append(f"  {label:<22} {spark}  ({vals[0]:.1f} → {vals[-1]:.1f})")

    return "\n".join(lines)


def auto_save_result(result: dict, result_dir: str | Path, args) -> Path:
    """Auto-save a benchmark result JSON with a timestamped filename.

    Returns the path to the saved file.
    """
    rdir = Path(result_dir)
    rdir.mkdir(parents=True, exist_ok=True)

    dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    rate = getattr(args, "request_rate", "inf")
    if rate == float("inf"):
        rate = "inf"
    model = getattr(args, "model", None) or "unknown"
    backend = getattr(args, "backend", "openai")
    filename = f"{backend}-{rate}qps-{model}-{dt}.json"

    filepath = rdir / filename
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2, default=str)

    return filepath


def history_main(argv: list[str] | None = None) -> None:
    """Entry point for ``xpyd-bench history`` subcommand."""
    parser = argparse.ArgumentParser(
        prog="xpyd-bench history",
        description="List past benchmark runs and show metric trends",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        required=True,
        help="Directory containing benchmark result JSON files.",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=None,
        metavar="N",
        help="Show only the last N runs.",
    )
    parser.add_argument(
        "--filter-tag",
        action="append",
        dest="filter_tags",
        default=None,
        metavar="KEY=VALUE",
        help="Filter results by tag (repeatable). E.g. --filter-tag env=prod",
    )
    parser.add_argument(
        "--group-by-fingerprint",
        action="store_true",
        default=False,
        help="Group results by benchmark fingerprint (M72).",
    )
    args = parser.parse_args(argv)

    # Parse filter tags
    filter_tags: dict[str, str] | None = None
    if args.filter_tags:
        filter_tags = {}
        for item in args.filter_tags:
            if "=" in item:
                k, v = item.split("=", 1)
                filter_tags[k.strip()] = v.strip()

    summaries = list_history(args.result_dir, last_n=args.last, filter_tags=filter_tags)

    if args.group_by_fingerprint:
        print(format_history_by_fingerprint(summaries))
    else:
        print(format_history_table(summaries))
