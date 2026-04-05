"""Network latency decomposition — DNS, TCP connect, TLS handshake, server processing.

Provides a transport wrapper that measures per-phase network timing and utilities
to aggregate breakdown data across benchmark requests.
"""

from __future__ import annotations

import asyncio
import socket
import ssl
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import numpy as np


@dataclass
class LatencyBreakdown:
    """Per-request network latency decomposition in milliseconds."""

    dns_ms: float = 0.0
    connect_ms: float = 0.0
    tls_ms: float = 0.0
    server_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "dns_ms": round(self.dns_ms, 3),
            "connect_ms": round(self.connect_ms, 3),
            "tls_ms": round(self.tls_ms, 3),
            "server_ms": round(self.server_ms, 3),
        }


def _percentile_stats(values: list[float]) -> dict[str, float]:
    """Compute mean/P50/P99 for a list of float values."""
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p99": 0.0}
    arr = np.array(values)
    return {
        "mean": round(float(np.mean(arr)), 3),
        "p50": round(float(np.percentile(arr, 50)), 3),
        "p99": round(float(np.percentile(arr, 99)), 3),
    }


def compute_breakdown_summary(
    breakdowns: list[LatencyBreakdown],
) -> dict[str, Any]:
    """Aggregate per-request breakdowns into summary statistics.

    Returns a dict with keys ``dns_ms``, ``connect_ms``, ``tls_ms``,
    ``server_ms``, each containing ``mean``, ``p50``, ``p99``.
    """
    if not breakdowns:
        return {}
    dns = [b.dns_ms for b in breakdowns]
    connect = [b.connect_ms for b in breakdowns]
    tls = [b.tls_ms for b in breakdowns]
    server = [b.server_ms for b in breakdowns]
    return {
        "dns_ms": _percentile_stats(dns),
        "connect_ms": _percentile_stats(connect),
        "tls_ms": _percentile_stats(tls),
        "server_ms": _percentile_stats(server),
    }


async def measure_connection_phases(
    host: str,
    port: int,
    use_tls: bool = False,
    timeout: float = 10.0,
) -> LatencyBreakdown:
    """Measure DNS, TCP connect, and TLS handshake times for a host:port.

    This is an async probe used to get a single breakdown sample.
    It does NOT send an HTTP request — only measures network setup phases.
    """
    breakdown = LatencyBreakdown()
    loop = asyncio.get_running_loop()

    # DNS resolution
    t0 = time.perf_counter()
    try:
        addr_info = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return breakdown
    t1 = time.perf_counter()
    breakdown.dns_ms = (t1 - t0) * 1000.0

    if not addr_info:
        return breakdown

    family, socktype, proto, _, sockaddr = addr_info[0]

    # TCP connect
    sock = socket.socket(family, socktype, proto)
    sock.setblocking(False)
    t2 = time.perf_counter()
    try:
        await asyncio.wait_for(
            loop.sock_connect(sock, sockaddr),
            timeout=timeout,
        )
    except (OSError, asyncio.TimeoutError):
        sock.close()
        return breakdown
    t3 = time.perf_counter()
    breakdown.connect_ms = (t3 - t2) * 1000.0

    # TLS handshake
    if use_tls:
        ctx = ssl.create_default_context()
        t4 = time.perf_counter()
        try:
            _transport, _protocol = await loop.create_connection(
                asyncio.Protocol,
                sock=sock,
                ssl=ctx,
                server_hostname=host,
            )
            t5 = time.perf_counter()
            breakdown.tls_ms = (t5 - t4) * 1000.0
            _transport.close()
        except (ssl.SSLError, OSError):
            sock.close()
            return breakdown
    else:
        sock.close()

    return breakdown


def estimate_server_processing(
    total_ms: float,
    dns_ms: float,
    connect_ms: float,
    tls_ms: float,
) -> float:
    """Estimate server processing time by subtracting network phases.

    For keep-alive connections, DNS/connect/TLS are zero after the first
    request, so server_ms ≈ total_ms.
    """
    server = total_ms - dns_ms - connect_ms - tls_ms
    return max(server, 0.0)


def print_breakdown_summary(summary: dict[str, Any]) -> None:
    """Print a human-readable breakdown summary to stdout."""
    if not summary:
        return
    print()
    print("  Network Latency Breakdown:")
    print(f"  {'Phase':<12s}  {'Mean':>10s}  {'P50':>10s}  {'P99':>10s}")
    print(f"  {'-' * 12}  {'-' * 10}  {'-' * 10}  {'-' * 10}")
    for phase in ("dns_ms", "connect_ms", "tls_ms", "server_ms"):
        label = phase.replace("_ms", "").upper()
        stats = summary.get(phase, {})
        mean = stats.get("mean", 0.0)
        p50 = stats.get("p50", 0.0)
        p99 = stats.get("p99", 0.0)
        print(f"  {label:<12s}  {mean:>8.2f}ms  {p50:>8.2f}ms  {p99:>8.2f}ms")


def parse_url(url: str) -> tuple[str, int, bool]:
    """Parse a URL and return (host, port, use_tls)."""
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    use_tls = parsed.scheme == "https"
    default_port = 443 if use_tls else 80
    port = parsed.port or default_port
    return host, port, use_tls
