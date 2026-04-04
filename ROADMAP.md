# xPyD-bench Roadmap

## M1: Core Benchmark Runner — vLLM Bench Compatible CLI ✅
- Study vLLM bench (`benchmarks/benchmark_serving.py`) CLI arguments thoroughly
- Implement CLI with **full compatibility** to vLLM bench arguments
- Implement async HTTP client for OpenAI API endpoints
- Support /v1/completions and /v1/chat/completions
- Streaming support for TTFT measurement
- Basic metrics collection (TTFT, TPOT, ITL, latency, throughput)
- Output format aligned with vLLM bench
- Extended/advanced options via YAML config (`--config`)

## M2: Dummy Server ✅
- Dummy prefill/decode server simulating vLLM behavior
- Support /v1/completions and /v1/chat/completions
- Streaming response simulation with configurable latency
- Decoupled from bench code (separate module in `src/xpyd_bench/dummy/`)
- Used for bench validation without real GPU hardware

## M3: Full OpenAI API Parameter Coverage ✅
- All 4 input formats (string, array of strings, array of tokens, array of mixed)
- All sampling parameters: temperature, top_k, top_p, frequency_penalty, presence_penalty
- Stop sequences, logprobs, max_tokens, n, seed
- Validate no parameter is omitted vs OpenAI API spec
- Dummy server must support all parameters too

## M4: Flexible Request Rate Patterns ✅
- Per-second (vLLM bench compatible, CLI)
- Per-N-seconds (5s, 10s, custom interval) — via YAML config
- Burst patterns, ramp-up/ramp-down — via YAML config
- Poisson distribution — via YAML config
- Custom pattern definition — via YAML config

## M5: Rich Dataset Input ✅
- JSONL input format for user-defined datasets
- JSON array input format
- CSV input format
- Synthetic dataset generation (configurable prompt/output length distribution)
- Dataset validation and stats reporting

## M6: Extended Metrics & Reporting ✅
- Per-request detailed metrics export
- Percentile breakdown (P50/P90/P95/P99)
- Time-series metrics (throughput over time)
- Rich terminal output (progress bars, live stats)
- JSON and human-readable report formats

## M7: Built-in Benchmark Scenarios ✅
- Predefined scenario presets: short, long_context, mixed, stress
- ScenarioConfig dataclass with CLI-compatible overrides
- `--scenario` CLI flag to select presets
- `--list-scenarios` flag to list available presets
- Tests covering all presets and CLI integration

## M8: Benchmark Comparison & Regression Detection ✅
- `xpyd-bench compare <baseline.json> <candidate.json>` CLI subcommand
- Side-by-side metric comparison (TTFT, TPOT, throughput, latency percentiles)
- Percentage delta and direction indicators (improved / regressed / unchanged)
- Configurable regression threshold (default 5%)
- Exit code 1 when regression detected (CI-friendly)
- Human-readable table output and JSON diff export
- Tests covering comparison logic and CLI integration

## M9: Warmup Requests ✅
- `--warmup N` CLI argument to send N warmup requests before benchmark
- Warmup requests excluded from metrics and results
- Sequential warmup with separate progress indicator
- Works with completions, chat, streaming and non-streaming
- YAML config support (`warmup: N`)
- Tests covering warmup logic and metrics exclusion
