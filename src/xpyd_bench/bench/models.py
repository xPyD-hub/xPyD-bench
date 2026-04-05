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
    start_time: float | None = None  # perf_counter timestamp when request started
    retries: int = 0
    success: bool = True
    error: str | None = None
    request_id: str | None = None
    response_text: str | None = None
    validation_errors: list[str] = field(default_factory=list)
    priority: int | None = None
    chunk_timings: list | None = None  # SSE per-chunk timing data (M53)


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
    duration_limit: float | None = None

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

    # Latency percentiles (None = metric not measured)
    mean_ttft_ms: float | None = None
    median_ttft_ms: float | None = None
    p50_ttft_ms: float | None = None
    p90_ttft_ms: float | None = None
    p95_ttft_ms: float | None = None
    p99_ttft_ms: float | None = None
    mean_tpot_ms: float | None = None
    median_tpot_ms: float | None = None
    p50_tpot_ms: float | None = None
    p90_tpot_ms: float | None = None
    p95_tpot_ms: float | None = None
    p99_tpot_ms: float | None = None
    mean_itl_ms: float | None = None
    median_itl_ms: float | None = None
    p50_itl_ms: float | None = None
    p90_itl_ms: float | None = None
    p95_itl_ms: float | None = None
    p99_itl_ms: float | None = None
    mean_e2el_ms: float | None = None
    median_e2el_ms: float | None = None
    p50_e2el_ms: float | None = None
    p90_e2el_ms: float | None = None
    p95_e2el_ms: float | None = None
    p99_e2el_ms: float | None = None

    # Partial result flag (set when benchmark was interrupted)
    partial: bool = False

    # Benchmark start time (perf_counter) for relative timestamp computation
    bench_start_time: float = 0.0

    # Environment metadata for reproducibility
    environment: dict[str, str] = field(default_factory=dict)

    # User-defined tags for annotation (M36)
    tags: dict[str, str] = field(default_factory=dict)

    # Anomaly detection results (M43)
    anomalies: dict | None = None

    # Validation summary (M47)
    validation_summary: dict | None = None

    # Warmup profiling results (M51)
    warmup_profile: dict | None = None

    # Priority metrics breakdown (M52)
    priority_metrics: dict | None = None

    # SSE streaming metrics (M53)
    sse_metrics: dict | None = None
