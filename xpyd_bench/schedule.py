"""Benchmark scheduling & cron integration (M63).

Generate crontab entries and run on-complete commands.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime, timedelta

# Cron field definitions: (name, min, max)
_CRON_FIELDS = [
    ("minute", 0, 59),
    ("hour", 0, 23),
    ("day of month", 1, 31),
    ("month", 1, 12),
    ("day of week", 0, 7),  # 0 and 7 both = Sunday
]


def validate_cron_expression(expr: str) -> list[str]:
    """Validate a cron expression (5 fields). Returns list of errors (empty = valid)."""
    parts = expr.strip().split()
    errors: list[str] = []
    if len(parts) != 5:
        errors.append(f"Expected 5 fields, got {len(parts)}: '{expr}'")
        return errors

    for i, (name, lo, hi) in enumerate(_CRON_FIELDS):
        field = parts[i]
        try:
            _validate_cron_field(field, lo, hi)
        except ValueError as exc:
            errors.append(f"Field {i + 1} ({name}): {exc}")
    return errors


def _validate_cron_field(field: str, lo: int, hi: int) -> None:
    """Validate a single cron field."""
    if field == "*":
        return
    # Handle */N
    if field.startswith("*/"):
        step = field[2:]
        if not step.isdigit() or int(step) < 1:
            raise ValueError("Invalid step '*/{step}' — must be a positive integer")
        return
    # Handle comma-separated values
    for part in field.split(","):
        # Handle range A-B
        if "-" in part:
            range_parts = part.split("-")
            if len(range_parts) != 2:
                raise ValueError(f"Invalid range '{part}'")
            a, b = range_parts
            if not a.isdigit() or not b.isdigit():
                raise ValueError(f"Non-numeric range '{part}'")
            a_int, b_int = int(a), int(b)
            if a_int < lo or a_int > hi or b_int < lo or b_int > hi:
                raise ValueError(f"Range '{part}' out of bounds ({lo}-{hi})")
            if a_int > b_int:
                raise ValueError(f"Range start > end in '{part}'")
        else:
            if not part.isdigit():
                raise ValueError(f"Non-numeric value '{part}'")
            val = int(part)
            if val < lo or val > hi:
                raise ValueError(f"Value {val} out of bounds ({lo}-{hi})")


def next_cron_occurrences(
    expr: str, count: int = 5, after: datetime | None = None,
) -> list[datetime]:
    """Compute next N occurrences of a cron expression (simple implementation).

    Supports: *, */N, single values, comma-separated, ranges.
    """
    parts = expr.strip().split()
    if len(parts) != 5:
        return []

    matchers = []
    for i, (_, lo, hi) in enumerate(_CRON_FIELDS):
        matchers.append(_build_matcher(parts[i], lo, hi))

    now = after or datetime.now()
    # Start from the next minute
    current = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    results: list[datetime] = []
    max_iterations = 525960  # ~1 year of minutes

    for _ in range(max_iterations):
        minute_match = current.minute in matchers[0]
        hour_match = current.hour in matchers[1]
        dom_match = current.day in matchers[2]
        month_match = current.month in matchers[3]
        dow_set = matchers[4]
        # Normalize Sunday: 7 -> 0
        dow = current.weekday()  # Monday=0..Sunday=6 in Python
        # Convert to cron convention: Sunday=0, Monday=1..Saturday=6
        cron_dow = (dow + 1) % 7
        dow_match = cron_dow in dow_set or (cron_dow == 0 and 7 in dow_set)

        if minute_match and hour_match and dom_match and month_match and dow_match:
            results.append(current)
            if len(results) >= count:
                break
        current += timedelta(minutes=1)

    return results


def _build_matcher(field: str, lo: int, hi: int) -> set[int]:
    """Build a set of valid values for a cron field."""
    if field == "*":
        return set(range(lo, hi + 1))
    if field.startswith("*/"):
        step = int(field[2:])
        return set(range(lo, hi + 1, step))

    result: set[int] = set()
    for part in field.split(","):
        if "-" in part:
            a, b = part.split("-")
            result.update(range(int(a), int(b) + 1))
        else:
            result.add(int(part))
    return result


def generate_crontab_entry(
    cron_expr: str,
    config_path: str | None = None,
    extra_args: str | None = None,
) -> str:
    """Generate a crontab entry line for xpyd-bench run."""
    bench_cmd = shutil.which("xpyd-bench") or "xpyd-bench"
    parts = [bench_cmd, "run"]
    if config_path:
        parts.extend(["--config", config_path])
    if extra_args:
        parts.append(extra_args)
    cmd = " ".join(parts)
    return f"{cron_expr} {cmd}"


def run_on_complete(command: str) -> subprocess.CompletedProcess:
    """Run an on-complete shell command. Returns CompletedProcess."""
    return subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=300,
    )


def schedule_main(argv: list[str] | None = None) -> None:
    """CLI entry point for 'xpyd-bench schedule'."""
    parser = argparse.ArgumentParser(
        prog="xpyd-bench schedule",
        description="Generate crontab entries for scheduled benchmarks.",
    )
    parser.add_argument(
        "--cron",
        type=str,
        required=True,
        help='Cron expression (5 fields), e.g. "0 */6 * * *".',
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file to pass to xpyd-bench run.",
    )
    parser.add_argument(
        "--extra-args",
        type=str,
        default=None,
        help="Additional arguments to append to the generated command.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=5,
        help="Number of next occurrences to preview (default: 5).",
    )

    args = parser.parse_args(argv)

    # Validate cron expression
    errors = validate_cron_expression(args.cron)
    if errors:
        print("Invalid cron expression:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)

    # Generate crontab entry
    entry = generate_crontab_entry(args.cron, args.config, args.extra_args)
    print("Crontab entry:")
    print(f"  {entry}")
    print()

    # Next occurrences preview
    occurrences = next_cron_occurrences(args.cron, count=args.preview)
    if occurrences:
        print(f"Next {len(occurrences)} occurrences:")
        for occ in occurrences:
            print(f"  {occ.strftime('%Y-%m-%d %H:%M (%A)')}")
