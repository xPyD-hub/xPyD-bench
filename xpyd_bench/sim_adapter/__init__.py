"""Thin adapter re-exporting xPyD-sim's server for bench testing."""

from xpyd_sim.server import ServerConfig, create_app

__all__ = ["ServerConfig", "create_app"]
