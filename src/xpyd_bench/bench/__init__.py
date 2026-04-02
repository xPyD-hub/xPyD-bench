"""Benchmark runner — sends requests and collects metrics."""

from __future__ import annotations

# TODO: implement
# - async request dispatcher (aiohttp/httpx)
# - concurrency control (semaphore-based, N in-flight)
# - support /v1/completions and /v1/chat/completions
# - streaming support for TTFT measurement
# - metrics collection: TTFT, TPS, latency (p50/p90/p99), throughput, error rate
# - dataset/prompt loading
# - result serialization to JSON
