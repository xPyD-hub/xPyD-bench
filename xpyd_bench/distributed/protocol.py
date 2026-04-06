"""Distributed benchmark protocol models (M32)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class WorkerTask:
    """A chunk of benchmark work sent to a worker."""

    task_id: str
    base_url: str
    endpoint: str
    model: str
    prompts: list[dict[str, Any]]
    output_len: int = 128
    stream: bool | None = None
    api_key: str | None = None
    timeout: float = 300.0
    custom_headers: dict[str, str] = field(default_factory=dict)
    sampling_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WorkerTask:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class WorkerResult:
    """Result returned by a worker after executing a task."""

    task_id: str
    worker_url: str
    completed: int = 0
    failed: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_duration_s: float = 0.0
    request_throughput: float = 0.0
    output_throughput: float = 0.0
    mean_ttft_ms: float | None = None
    mean_tpot_ms: float | None = None
    mean_e2el_ms: float | None = None
    p50_e2el_ms: float | None = None
    p90_e2el_ms: float | None = None
    p99_e2el_ms: float | None = None
    p50_ttft_ms: float | None = None
    p90_ttft_ms: float | None = None
    p99_ttft_ms: float | None = None
    error: str | None = None
    requests: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WorkerResult:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class HeartbeatResponse:
    """Worker heartbeat response."""

    worker_url: str
    status: str = "ok"  # "ok" | "busy" | "error"
    current_task_id: str | None = None
    uptime_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HeartbeatResponse:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
