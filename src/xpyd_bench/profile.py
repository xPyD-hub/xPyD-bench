"""Profile & Replay: capture and replay request timing patterns.

Profile mode records a trace of request timings during a benchmark run.
Replay mode replays a recorded trace against a target server with
deterministic inter-request delays for reproducible benchmarks.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TraceEntry:
    """A single recorded request in the trace."""

    offset_s: float  # seconds from trace start
    prompt_len: int  # prompt token count
    output_len: int  # requested max output tokens
    endpoint: str = "/v1/completions"
    model: str = ""
    prompt: str = ""  # actual prompt text (optional, may be empty)
    temperature: float = 1.0
    max_tokens: int = 128
    stream: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceData:
    """Complete recorded trace."""

    version: int = 1
    base_url: str = ""
    total_duration_s: float = 0.0
    num_entries: int = 0
    entries: list[TraceEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> TraceData:
        entries = [TraceEntry(**e) for e in data.get("entries", [])]
        return cls(
            version=data.get("version", 1),
            base_url=data.get("base_url", ""),
            total_duration_s=data.get("total_duration_s", 0.0),
            num_entries=data.get("num_entries", len(entries)),
            entries=entries,
        )


def save_trace(trace: TraceData, path: str | Path) -> Path:
    """Save trace data to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(trace.to_dict(), f, indent=2)
    return p


def load_trace(path: str | Path) -> TraceData:
    """Load trace data from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return TraceData.from_dict(data)


class TraceRecorder:
    """Records request timings during a benchmark run."""

    def __init__(self, base_url: str = "") -> None:
        self._base_url = base_url
        self._start: float | None = None
        self._entries: list[TraceEntry] = []

    def start(self) -> None:
        self._start = time.monotonic()

    def record(
        self,
        prompt_len: int,
        output_len: int,
        endpoint: str = "/v1/completions",
        model: str = "",
        prompt: str = "",
        temperature: float = 1.0,
        max_tokens: int = 128,
        stream: bool = True,
        **extra: Any,
    ) -> None:
        if self._start is None:
            self.start()
        offset = time.monotonic() - self._start  # type: ignore[operator]
        self._entries.append(
            TraceEntry(
                offset_s=offset,
                prompt_len=prompt_len,
                output_len=output_len,
                endpoint=endpoint,
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                extra=extra if extra else {},
            )
        )

    def finish(self) -> TraceData:
        duration = time.monotonic() - self._start if self._start else 0.0
        return TraceData(
            version=1,
            base_url=self._base_url,
            total_duration_s=duration,
            num_entries=len(self._entries),
            entries=list(self._entries),
        )


def compute_delays(trace: TraceData, speed: float = 1.0) -> list[float]:
    """Compute inter-request delays from trace offsets.

    Args:
        trace: The recorded trace data.
        speed: Speed multiplier (2.0 = replay at 2x speed, 0.5 = half speed).

    Returns:
        List of delays in seconds. First entry delay is from time 0.
    """
    if not trace.entries:
        return []
    if speed <= 0:
        raise ValueError("speed must be positive")

    delays = []
    prev_offset = 0.0
    for entry in trace.entries:
        delay = max(0.0, (entry.offset_s - prev_offset) / speed)
        delays.append(delay)
        prev_offset = entry.offset_s
    return delays
