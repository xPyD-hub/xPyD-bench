"""Benchmark result data models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RequestResult:
    """Metrics for a single request."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: float | None = None  # Time to first token (streaming only)
    latency_ms: float = 0.0  # End-to-end latency
    tps: float = 0.0  # Tokens per second for this request
    success: bool = True
    error: str | None = None


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""

    # Config
    target: str = ""
    endpoint: str = "chat"
    concurrency: int = 1
    num_requests: int = 0
    max_tokens: int = 256
    stream: bool = False

    # Results
    requests: list[RequestResult] = field(default_factory=list)
    total_duration_s: float = 0.0

    # Aggregated metrics (computed after run)
    throughput_rps: float = 0.0  # Requests per second
    throughput_tps: float = 0.0  # Tokens per second (aggregate)
    latency_p50_ms: float = 0.0
    latency_p90_ms: float = 0.0
    latency_p99_ms: float = 0.0
    ttft_p50_ms: float | None = None
    ttft_p90_ms: float | None = None
    ttft_p99_ms: float | None = None
    error_count: int = 0
    error_rate: float = 0.0
