"""Real-time metrics streaming via WebSocket (M33).

Starts a lightweight WebSocket server that pushes JSON metric snapshots
to all connected clients every second during a benchmark run.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricsCollector:
    """Thread-safe accumulator fed by the benchmark runner.

    Call :meth:`record` for every completed request and :meth:`snapshot`
    to get the current aggregated state as a JSON-serialisable dict.
    """

    _completed: int = 0
    _failed: int = 0
    _total_latency_ms: float = 0.0
    _latencies: list[float] = field(default_factory=list)
    _ttfts: list[float] = field(default_factory=list)
    _total_prompt_tokens: int = 0
    _total_completion_tokens: int = 0
    _start_time: float = 0.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def start(self) -> None:
        self._start_time = time.monotonic()

    def record(
        self,
        *,
        success: bool,
        latency_ms: float,
        ttft_ms: float | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Record a completed request (called from the event-loop, no await needed)."""
        if success:
            self._completed += 1
        else:
            self._failed += 1
        self._total_latency_ms += latency_ms
        self._latencies.append(latency_ms)
        if ttft_ms is not None:
            self._ttfts.append(ttft_ms)
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of current metrics."""
        elapsed = time.monotonic() - self._start_time if self._start_time else 0.0
        total = self._completed + self._failed
        rps = self._completed / elapsed if elapsed > 0 else 0.0
        avg_latency = self._total_latency_ms / total if total > 0 else 0.0

        recent_latencies = self._latencies[-50:]
        avg_recent = (
            sum(recent_latencies) / len(recent_latencies) if recent_latencies else 0.0
        )

        avg_ttft = 0.0
        if self._ttfts:
            avg_ttft = sum(self._ttfts) / len(self._ttfts)

        return {
            "timestamp": time.time(),
            "elapsed_s": round(elapsed, 2),
            "completed": self._completed,
            "failed": self._failed,
            "rps": round(rps, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "avg_recent_latency_ms": round(avg_recent, 2),
            "avg_ttft_ms": round(avg_ttft, 2),
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "output_tps": round(
                self._total_completion_tokens / elapsed if elapsed > 0 else 0.0, 2
            ),
        }


class MetricsWebSocketServer:
    """Async WebSocket server that broadcasts :class:`MetricsCollector` snapshots.

    Uses :mod:`starlette` + :mod:`uvicorn` (already project deps) so we
    don't add new dependencies.
    """

    def __init__(self, collector: MetricsCollector, port: int = 9100) -> None:
        self.collector = collector
        self.port = port
        self._clients: set[Any] = set()
        self._broadcast_task: asyncio.Task | None = None
        self._server: Any = None
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        """Start the WebSocket server and the periodic broadcast loop."""
        from starlette.applications import Starlette
        from starlette.routing import WebSocketRoute
        from starlette.websockets import WebSocket, WebSocketDisconnect

        server_self = self

        async def ws_endpoint(websocket: WebSocket) -> None:
            await websocket.accept()
            server_self._clients.add(websocket)
            try:
                # Keep connection open; client only receives
                while True:
                    # Wait for client messages (ping/close)
                    try:
                        await asyncio.wait_for(websocket.receive_text(), timeout=60)
                    except asyncio.TimeoutError:
                        pass
            except (WebSocketDisconnect, Exception):
                pass
            finally:
                server_self._clients.discard(websocket)

        app = Starlette(routes=[WebSocketRoute("/metrics", ws_endpoint)])

        import uvicorn

        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=self.port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)

        # Run server in background
        asyncio.get_event_loop().create_task(self._server.serve())
        # Start broadcast loop
        self._broadcast_task = asyncio.get_event_loop().create_task(self._broadcast_loop())

    async def _broadcast_loop(self) -> None:
        """Push snapshots to all connected clients every second."""
        while not self._stop_event.is_set():
            if self._clients:
                data = json.dumps(self.collector.snapshot())
                disconnected: set[Any] = set()
                for ws in list(self._clients):
                    try:
                        await ws.send_text(data)
                    except Exception:
                        disconnected.add(ws)
                self._clients -= disconnected
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass

    async def stop(self) -> None:
        """Shutdown the server and broadcast loop."""
        self._stop_event.set()
        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
        # Close connected clients
        for ws in list(self._clients):
            try:
                await ws.close()
            except Exception:
                pass
        self._clients.clear()
        if self._server:
            self._server.should_exit = True
