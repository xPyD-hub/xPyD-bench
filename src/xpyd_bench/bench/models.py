"""Benchmark result data models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RequestResult:
    """Metrics for a single completed request."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: float | None = None  # Time to first token (streaming only)
    tpot_ms: float | None = None  # Time per output token
    itl_ms: list[float] = field(default_factory=list)  # Inter-token latencies
    latency_ms: float = 0.0  # End-to-end latency
    retries: int = 0
    success: bool = True
    error: str | None = None


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""

    # Config echo
    backend: str = "openai"
    base_url: str = ""
    endpoint: str = "/v1/completions"
    model: str = ""
    num_prompts: int = 0
    request_rate: float = float("inf")
    max_concurrency: int | None = None
    input_len: int = 256
    output_len: int = 128

    # Per-request results
    requests: list[RequestResult] = field(default_factory=list)
    total_duration_s: float = 0.0

    # Aggregated metrics
    completed: int = 0
    failed: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    request_throughput: float = 0.0  # req/s
    output_throughput: float = 0.0  # output tok/s
    total_token_throughput: float = 0.0  # (input+output) tok/s

    # Latency percentiles
    mean_ttft_ms: float = 0.0
    median_ttft_ms: float = 0.0
    p50_ttft_ms: float = 0.0
    p90_ttft_ms: float = 0.0
    p95_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    mean_tpot_ms: float = 0.0
    median_tpot_ms: float = 0.0
    p50_tpot_ms: float = 0.0
    p90_tpot_ms: float = 0.0
    p95_tpot_ms: float = 0.0
    p99_tpot_ms: float = 0.0
    mean_itl_ms: float = 0.0
    median_itl_ms: float = 0.0
    p50_itl_ms: float = 0.0
    p90_itl_ms: float = 0.0
    p95_itl_ms: float = 0.0
    p99_itl_ms: float = 0.0
    mean_e2el_ms: float = 0.0
    median_e2el_ms: float = 0.0
    p50_e2el_ms: float = 0.0
    p90_e2el_ms: float = 0.0
    p95_e2el_ms: float = 0.0
    p99_e2el_ms: float = 0.0

    # Partial result flag (set when benchmark was interrupted)
    partial: bool = False

    # Environment metadata for reproducibility
    environment: dict[str, str] = field(default_factory=dict)
