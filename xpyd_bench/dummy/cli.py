"""Backward-compatible dummy CLI shim — delegates to xpyd-sim.

DEPRECATED: Use `xpyd-sim` CLI directly.
"""


def dummy_main(argv: list[str] | None = None) -> None:
    """Entry point for ``xpyd-dummy`` command (deprecated)."""
    from xpyd_sim.cli import main
    main(argv)
