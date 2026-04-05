# xPyD-bench — Current Iteration Status

> Updated: 2026-04-06

## Current Milestone: M90 — Request Warm-up Curve Analysis

### What was done

- Added `--warmup-curve` CLI flag for warmup curve analysis
- New module `src/xpyd_bench/bench/warmup_curve.py` implementing:
  - `fit_exponential_decay()` — fits `f(x) = a * exp(-b * x) + c` to latency data
  - `detect_convergence()` — finds the request index where latency stabilizes
  - `render_ascii_curve()` — ASCII visualization with observed data points, fitted curve, and convergence marker
  - `build_warmup_curve()` — orchestrates fitting, convergence detection, and result packaging
  - `print_warmup_curve()` — terminal output with fit parameters, R², cold-start penalty
  - `WarmupCurveResult` dataclass with `to_dict()` serialization
- `BenchmarkResult` includes `warmup_curve` dict field in JSON output
- YAML config support (`warmup_curve: true`)
- Config key added to `_KNOWN_KEYS`
- Runner integration: analyzes all successful request latencies sorted by send time
- 16 tests covering:
  - Perfect and noisy exponential decay fitting
  - Flat data and edge cases (too few points, empty)
  - Convergence detection (fast, no-decay, slow)
  - ASCII rendering with and without convergence marker
  - Realistic warmup patterns and no-warmup scenarios
  - Serialization to dict

### Different from M51 (Warmup Profiling)
- M51: Simple stabilization detection using rolling CV threshold
- M90: Mathematical exponential decay curve fitting with R² goodness-of-fit, convergence point, and ASCII visualization

### Files Changed
- `src/xpyd_bench/bench/warmup_curve.py` (new)
- `src/xpyd_bench/bench/models.py` (added `warmup_curve` field)
- `src/xpyd_bench/bench/runner.py` (integration + JSON serialization)
- `src/xpyd_bench/cli.py` (added `--warmup-curve` flag)
- `src/xpyd_bench/config_cmd.py` (added to `_KNOWN_KEYS`)
- `tests/test_warmup_curve.py` (new, 16 tests)
- `docs/iterations/current.md` (this file)

---

## Capabilities Summary (M1–M90)

### Core Benchmarking
- Full vLLM bench CLI compatibility
- OpenAI API: completions, chat, embeddings, batch, function calling, multimodal (vision)
- Streaming and non-streaming modes
- Multi-turn conversation benchmarking
- Duration-based and repeat mode benchmarking

### Traffic Shaping
- Per-second, per-N-seconds, burst, ramp, Poisson rate patterns
- Token bucket rate limiting
- Adaptive concurrency
- Priority-based request scheduling
- Request dependency chains
- Concurrency sweep mode

### Metrics & Analysis
- TTFT / TPOT / TPS / throughput / error rate
- Custom percentiles (P50 / P90 / P99 / user-defined) (M73)
- Rolling window metrics (M81)
- Confidence intervals
- Latency breakdown
- Workload distribution statistics (M78)
- Speculative decoding metrics (M88)
- Prefix caching impact analysis (M87)
- Anomaly detection
- Warmup curve analysis (M90) — exponential decay curve fitting

### Reliability & Operations
- Benchmark checkpointing & resume (M74)
- Benchmark fingerprinting (M72)
- Baseline registry (M82)
- Error threshold abort (M83)
- Request deduplication & idempotency (M85)
- Adaptive timeout auto-tuning (M86)
- Git metadata capture (M79)
- Configuration inheritance via `extends` (M80)

### Reporting & Integration
- Rich CLI output
- JSON / Markdown / HTML export
- JUnit XML for CI integration
- Webhook notifications
- OpenTelemetry (OTLP) export

## Iteration History

| # | Date | Task | Result | Reviewer Comments |
|---|------|------|--------|-------------------|
| 1 | 2026-04-05 | M89: Multi-LoRA Endpoint Benchmarking | ✅ merged (PR #242) | Both approved — clean code, good test coverage |
| 2 | 2026-04-06 | M90: Request Warm-up Curve Analysis | ✅ merged | Both approved |
| 3 | 2026-04-06 | M91: Token-Level Streaming Latency CDF | ✅ merged (PR #246) | Both approved |
| 4 | 2026-04-06 | M92: Prompt Caching Cost Analysis | ✅ merged (PR #248) | Both approved |
| 5 | 2026-04-06 | M93: Request Pacing Accuracy Report | ⏳ in progress | — |
