# xPyD-bench — Current Iteration Status

> Updated: 2026-04-05

## Current Milestone: M88 — Speculative Decoding Metrics

M88 is complete, adding benchmark metric support for Speculative Decoding.

## Feature List

### Core Benchmark
- OpenAI-compatible API benchmark (completions / chat)
- Streaming & non-streaming response support
- Poisson / burst request scheduling (`--request-rate`, `--burstiness`)
- Custom dataset loading (random / synthetic / JSONL / CSV)
- Token bucket rate limiting
- Multi-backend plugin architecture (openai / custom plugin)

### Advanced Test Modes
- **Multi-model comparison** (M75) — compare multiple models side by side
- **Streaming vs non-streaming overhead** (M76) — streaming overhead analysis
- **Multimodal vision benchmark** (M77) — vision model support
- **Multi-turn conversation** — multi-turn dialogue testing
- **Chain benchmark** — request chain testing
- **Sweep mode** — parameter sweep
- **Distributed benchmark** — multi-node coordinated load testing

### Metrics & Analysis
- TTFT / TPOT / TPS / throughput / error rate
- Custom percentiles (P50 / P90 / P99 / user-defined) (M73)
- Rolling window metrics (M81) — real-time rolling window statistics
- Confidence intervals
- Latency breakdown
- Workload distribution statistics (M78)
- Speculative decoding metrics (M88)
- Prefix caching impact analysis (M87)
- Anomaly detection

### Reliability & Operations
- Benchmark checkpointing & resume (M74)
- Benchmark fingerprinting (M72) — unique configuration identifier
- Baseline registry (M82) — baseline registration and regression comparison
- Error threshold abort (M83) — auto-abort when error rate exceeds limit
- Request deduplication & idempotency (M85)
- Adaptive timeout auto-tuning (M86)
- Git metadata capture (M79) — bind results to git version
- Configuration inheritance via `extends` (M80)

### Reporting & Integration
- Rich CLI output
- JSON / HTML / JUnit XML reports
- Prometheus metrics export
- WebSocket real-time metrics push
- Webhook notifications
- OTLP trace export
- Heatmap visualization

### Tools
- `xpyd-dummy` — mock server for testing without a real model
- `xpyd-bench compare` — result comparison
- `xpyd-bench profile` — performance profiling
- `xpyd-bench replay` — request replay
- `xpyd-bench config-dump / config-validate` — configuration management

## Known Limitations

1. **No gRPC backend support** — currently only HTTP/1.1 and HTTP/2 (requires `h2` installation)
2. **Token counting depends on tiktoken** — requires `xpyd-bench[tokenizer]`; otherwise uses rough estimation
3. **Distributed mode has no auto-discovery** — worker addresses must be specified manually
4. **No built-in charts** — HTML reports only contain tables; use external tools (Grafana + Prometheus exporter) for visualization
5. **License TBD** — open-source license not yet determined

## Next Steps

- **M89+**: See [ROADMAP.md](../../ROADMAP.md) for the full roadmap
- Priority areas:
  - GPU utilization correlation analysis
  - A/B testing framework (automatic statistical significance testing)
  - Richer HTML reports (embedded charts)
  - gRPC backend support
  - Automatic cluster node discovery
