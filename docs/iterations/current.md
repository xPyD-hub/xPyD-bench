# xPyD-bench — Current Iteration Status

> Updated: 2026-04-05

## Current Milestone: M89 — Multi-LoRA Endpoint Benchmarking

### What was done

- Added `xpyd-bench lora-compare` CLI subcommand
- New module `src/xpyd_bench/lora_compare.py` implementing:
  - Sequential benchmarking of multiple LoRA adapters on the same base model
  - `--interleave` flag for round-robin adapter switching to measure switching overhead
  - Per-adapter metrics: TTFT, TPOT, throughput, latency percentiles
  - Pairwise comparison with regression detection and Mann-Whitney U significance testing
  - Adapter switching overhead calculation (sequential vs interleaved mean TTFT)
  - JSON and terminal/Markdown comparison output
- Registered `lora-compare` subcommand in `main.py` dispatch table
- Added `lora_compare_main` entry point in `cli.py`
- Comprehensive test suite in `tests/test_lora_compare.py` covering:
  - Data class serialization
  - Statistical significance computation
  - Sequential and interleaved orchestration (mocked)
  - Three-adapter comparison
  - Summary formatting (terminal + Markdown)
  - JSON and Markdown export

### Result: pending review

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
- **Multi-LoRA endpoint benchmarking** (M89) — compare LoRA adapters with switching overhead
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
- JSON / Markdown / HTML export
- JUnit XML for CI integration
- Webhook notifications
- OpenTelemetry (OTLP) export
