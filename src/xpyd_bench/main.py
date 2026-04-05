"""Unified CLI entry point for xpyd-bench (M24).

Usage:
    xpyd-bench run [args...]          # benchmark (default if no subcommand)
    xpyd-bench compare [args...]      # compare results
    xpyd-bench multi [args...]        # multi-endpoint mode
    xpyd-bench profile [args...]      # profile recording
    xpyd-bench replay [args...]       # replay traces
    xpyd-bench config dump [args...]  # dump resolved config
    xpyd-bench config validate [args...]  # validate config
    xpyd-bench --version              # show version
"""

from __future__ import annotations

import importlib.metadata
import sys
import warnings


def _get_version() -> str:
    """Return the installed package version."""
    try:
        return importlib.metadata.version("xpyd-bench")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _deprecation_wrapper(func_path: str, legacy_name: str):
    """Return a wrapper that prints a deprecation warning then calls the real function."""

    def wrapper() -> None:
        subcmd = legacy_name.replace("xpyd-bench-", "").replace("-", " ")
        warnings.warn(
            f"'{legacy_name}' is deprecated, use 'xpyd-bench {subcmd}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        module_path, func_name = func_path.rsplit(":", 1)
        mod = __import__(module_path, fromlist=[func_name])
        getattr(mod, func_name)()

    return wrapper


# Legacy entry point wrappers
deprecated_compare_main = _deprecation_wrapper(
    "xpyd_bench.cli:compare_main", "xpyd-bench-compare"
)
deprecated_multi_main = _deprecation_wrapper(
    "xpyd_bench.cli:multi_main", "xpyd-bench-multi"
)
deprecated_profile_main = _deprecation_wrapper(
    "xpyd_bench.cli:profile_main", "xpyd-bench-profile"
)
deprecated_replay_main = _deprecation_wrapper(
    "xpyd_bench.cli:replay_main", "xpyd-bench-replay"
)
deprecated_config_dump_main = _deprecation_wrapper(
    "xpyd_bench.config_cmd:config_dump_main", "xpyd-bench-config-dump"
)
deprecated_config_validate_main = _deprecation_wrapper(
    "xpyd_bench.config_cmd:config_validate_main", "xpyd-bench-config-validate"
)


_SUBCOMMANDS = {
    "run",
    "compare",
    "multi",
    "profile",
    "replay",
    "config",
    "aggregate",
    "history",
    "distributed",
    "presets",
    "batch",
    "sweep",
    "healthcheck",
    "diff",
    "archive",
    "chain",
    "schedule",
    "discover",
    "resume",
    "model-compare",
    "stream-compare",
    "baseline",
    "cache-test",
    "lora-compare",
}


def main() -> None:
    """Unified CLI entry point."""
    argv = sys.argv[1:]

    # Handle --version anywhere
    if "--version" in argv:
        print(f"xpyd-bench {_get_version()}")
        sys.exit(0)

    # Determine subcommand
    if not argv or argv[0] not in _SUBCOMMANDS:
        # Default to 'run' — pass all args through
        from xpyd_bench.cli import bench_main

        bench_main(argv)
        return

    subcmd = argv[0]
    rest = argv[1:]

    if subcmd == "run":
        from xpyd_bench.cli import bench_main

        bench_main(rest)
    elif subcmd == "compare":
        from xpyd_bench.cli import compare_main

        compare_main(rest)
    elif subcmd == "multi":
        from xpyd_bench.cli import multi_main

        multi_main(rest)
    elif subcmd == "profile":
        from xpyd_bench.cli import profile_main

        profile_main(rest)
    elif subcmd == "replay":
        from xpyd_bench.cli import replay_main

        replay_main(rest)
    elif subcmd == "config":
        _config_subcommand(rest)
    elif subcmd == "aggregate":
        from xpyd_bench.aggregate import aggregate_main

        aggregate_main(rest)
    elif subcmd == "history":
        from xpyd_bench.history import history_main

        history_main(rest)
    elif subcmd == "distributed":
        from xpyd_bench.distributed.cli import distributed_main

        distributed_main(rest)
    elif subcmd == "presets":
        _presets_subcommand(rest)
    elif subcmd == "batch":
        from xpyd_bench.cli import batch_main

        batch_main(rest)
    elif subcmd == "sweep":
        from xpyd_bench.sweep import sweep_main

        sweep_main(rest)
    elif subcmd == "healthcheck":
        from xpyd_bench.healthcheck import healthcheck_main

        healthcheck_main(rest)
    elif subcmd == "diff":
        from xpyd_bench.diff import diff_main

        diff_main(rest)
    elif subcmd == "archive":
        from xpyd_bench.archive import archive_main

        archive_main(rest)
    elif subcmd == "chain":
        from xpyd_bench.chain import chain_main

        chain_main(rest)
    elif subcmd == "schedule":
        from xpyd_bench.schedule import schedule_main

        schedule_main(rest)
    elif subcmd == "discover":
        from xpyd_bench.discover import discover_main

        discover_main(rest)
    elif subcmd == "resume":
        from xpyd_bench.checkpoint import resume_main

        resume_main(rest)
    elif subcmd == "model-compare":
        from xpyd_bench.cli import model_compare_main

        model_compare_main(rest)
    elif subcmd == "stream-compare":
        from xpyd_bench.cli import stream_compare_main

        stream_compare_main(rest)
    elif subcmd == "baseline":
        _baseline_subcommand(rest)
    elif subcmd == "cache-test":
        from xpyd_bench.cache_test import cache_test_main

        cache_test_main(rest)
    elif subcmd == "lora-compare":
        from xpyd_bench.cli import lora_compare_main

        lora_compare_main(rest)


def _config_subcommand(argv: list[str]) -> None:
    """Handle 'xpyd-bench config <dump|validate>' subcommand."""
    if not argv:
        print("Usage: xpyd-bench config <dump|validate> [options]", file=sys.stderr)
        sys.exit(1)

    sub = argv[0]
    rest = argv[1:]

    if sub == "dump":
        from xpyd_bench.config_cmd import config_dump_main

        config_dump_main(rest)
    elif sub == "validate":
        from xpyd_bench.config_cmd import config_validate_main

        config_validate_main(rest)
    else:
        print(
            f"Unknown config subcommand '{sub}'. Use 'dump' or 'validate'.",
            file=sys.stderr,
        )
        sys.exit(1)


def _presets_subcommand(argv: list[str]) -> None:
    """Handle 'xpyd-bench presets <list|show>' subcommand."""
    if not argv:
        print("Usage: xpyd-bench presets <list|show> [options]", file=sys.stderr)
        sys.exit(1)

    sub = argv[0]
    rest = argv[1:]

    if sub == "list":
        from xpyd_bench.presets import presets_list_main

        presets_list_main(rest)
    elif sub == "show":
        from xpyd_bench.presets import presets_show_main

        presets_show_main(rest)
    else:
        print(
            f"Unknown presets subcommand '{sub}'. Use 'list' or 'show'.",
            file=sys.stderr,
        )
        sys.exit(1)


def _baseline_subcommand(argv: list[str]) -> None:
    """Handle 'xpyd-bench baseline <save|list|show|delete>' subcommand."""
    import argparse
    import json

    if not argv:
        print(
            "Usage: xpyd-bench baseline <save|list|show|delete> [options]",
            file=sys.stderr,
        )
        sys.exit(1)

    sub = argv[0]
    rest = argv[1:]

    if sub == "save":
        parser = argparse.ArgumentParser(prog="xpyd-bench baseline save")
        parser.add_argument("name", help="Name for the baseline")
        parser.add_argument("result", help="Path to result JSON file")
        parser.add_argument("--baseline-dir", default=None, help="Custom baseline directory")
        args = parser.parse_args(rest)

        from xpyd_bench.baseline import save_baseline

        entry = save_baseline(args.name, args.result, baseline_dir=args.baseline_dir)
        print(f"Baseline '{args.name}' saved.")
        print(json.dumps(entry, indent=2))

    elif sub == "list":
        parser = argparse.ArgumentParser(prog="xpyd-bench baseline list")
        parser.add_argument("--baseline-dir", default=None, help="Custom baseline directory")
        args = parser.parse_args(rest)

        from xpyd_bench.baseline import list_baselines

        baselines = list_baselines(baseline_dir=args.baseline_dir)
        if not baselines:
            print("No baselines saved.")
        else:
            for b in baselines:
                line = f"  {b['name']}  saved={b.get('saved_at', '?')}"
                if "model" in b and b["model"]:
                    line += f"  model={b['model']}"
                print(line)

    elif sub == "show":
        parser = argparse.ArgumentParser(prog="xpyd-bench baseline show")
        parser.add_argument("name", help="Baseline name")
        parser.add_argument("--baseline-dir", default=None, help="Custom baseline directory")
        args = parser.parse_args(rest)

        from xpyd_bench.baseline import show_baseline

        data = show_baseline(args.name, baseline_dir=args.baseline_dir)
        print(json.dumps(data, indent=2))

    elif sub == "delete":
        parser = argparse.ArgumentParser(prog="xpyd-bench baseline delete")
        parser.add_argument("name", help="Baseline name to delete")
        parser.add_argument("--baseline-dir", default=None, help="Custom baseline directory")
        args = parser.parse_args(rest)

        from xpyd_bench.baseline import delete_baseline

        delete_baseline(args.name, baseline_dir=args.baseline_dir)
        print(f"Baseline '{args.name}' deleted.")

    else:
        print(
            f"Unknown baseline subcommand '{sub}'. Use 'save', 'list', 'show', or 'delete'.",
            file=sys.stderr,
        )
        sys.exit(1)
