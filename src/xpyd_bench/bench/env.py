"""Environment info collection for benchmark reproducibility."""

from __future__ import annotations

import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone


def collect_git_info() -> dict[str, str] | None:
    """Collect git repository metadata if running inside a git repo.

    Returns a dict with keys: commit, branch, dirty, remote_url.
    Returns None if not in a git repository or git is not available.
    """
    try:
        # Check if we're in a git repo
        subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            check=True,
            timeout=5,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None

    info: dict[str, str] = {}

    def _git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()

    try:
        info["commit"] = _git("rev-parse", "HEAD")
        info["commit_short"] = _git("rev-parse", "--short", "HEAD")
        info["branch"] = _git("rev-parse", "--abbrev-ref", "HEAD")
        # dirty = any uncommitted changes
        dirty_output = _git("status", "--porcelain")
        info["dirty"] = str(bool(dirty_output))
        info["remote_url"] = _git("config", "--get", "remote.origin.url")
    except (subprocess.TimeoutExpired, Exception):
        pass

    return info if info else None


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
