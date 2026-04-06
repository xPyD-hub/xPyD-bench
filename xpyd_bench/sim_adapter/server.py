"""Re-export xPyD-sim server components.

This module replaces the old xpyd_bench.dummy.server module.
All server functionality is now provided by xpyd-sim.
"""

from xpyd_sim.server import ServerConfig, create_app

__all__ = ["ServerConfig", "create_app"]
