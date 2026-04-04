"""Environment info collection for benchmark reproducibility."""

from __future__ import annotations

import platform
import socket
import sys
from datetime import datetime, timezone


def collect_env_info() -> dict[str, str]:
    """Collect environment metadata for embedding in benchmark results.

    Returns a dict with keys:
        python_version, os, platform, architecture, hostname,
        xpyd_bench_version, timestamp
    """
    from xpyd_bench import __version__

    return {
        "python_version": sys.version,
        "os": platform.system(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "hostname": socket.gethostname(),
        "xpyd_bench_version": __version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
