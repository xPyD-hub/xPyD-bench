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
