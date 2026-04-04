"""Per-request debug logging for long benchmark runs."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xpyd_bench.bench.models import RequestResult

# Maximum characters for payload in log entries
_MAX_PAYLOAD_LEN = 512


@dataclass
class DebugLogEntry:
    """A single debug log entry for one request."""

    timestamp: str
    url: str
    payload: str
    status: str
    latency_ms: float
    success: bool
    error: str | None = None
    retries: int = 0
    payload_bytes: int | None = None
    compressed_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "timestamp": self.timestamp,
            "url": self.url,
            "payload": self.payload,
            "status": self.status,
            "latency_ms": round(self.latency_ms, 3),
            "success": self.success,
        }
        if self.error is not None:
            d["error"] = self.error
        if self.retries > 0:
            d["retries"] = self.retries
        if self.payload_bytes is not None:
            d["payload_bytes"] = self.payload_bytes
        if self.compressed_bytes is not None:
            d["compressed_bytes"] = self.compressed_bytes
        return d


class DebugLogger:
    """Writes per-request debug log entries as JSON lines to a file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "w")  # noqa: SIM115

    def log_connection_config(
        self,
        http2: bool = False,
        max_connections: int = 100,
        max_keepalive: int = 20,
    ) -> None:
        """Write connection pool configuration as the first log entry."""
        entry = {
            "type": "connection_config",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "http2": http2,
            "max_connections": max_connections,
            "max_keepalive": max_keepalive,
        }
        self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._file.flush()

    def log(
        self,
        url: str,
        payload: dict[str, Any],
        result: RequestResult,
        payload_bytes: int | None = None,
        compressed_bytes: int | None = None,
    ) -> None:
        """Write one log entry for a completed request."""
        # Truncate payload
        payload_str = json.dumps(payload, ensure_ascii=False)
        if len(payload_str) > _MAX_PAYLOAD_LEN:
            payload_str = payload_str[:_MAX_PAYLOAD_LEN] + "...(truncated)"

        status = "ok" if result.success else "error"

        entry = DebugLogEntry(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            url=url,
            payload=payload_str,
            status=status,
            latency_ms=result.latency_ms,
            success=result.success,
            error=result.error,
            retries=result.retries,
            payload_bytes=payload_bytes,
            compressed_bytes=compressed_bytes,
        )
        self._file.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        """Close the log file."""
        self._file.close()
