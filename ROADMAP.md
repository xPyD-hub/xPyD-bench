# xPyD-bench Roadmap

## M1: Core Benchmark Runner ⬜
- Implement async HTTP client for OpenAI API endpoints
- Support /v1/completions and /v1/chat/completions
- Streaming support for TTFT measurement
- Basic metrics collection (TTFT, TPOT, latency, throughput)
- CLI aligned with vLLM bench arguments
- Output format aligned with vLLM bench

## M2: Dummy Server ⬜
- Dummy prefill/decode server simulating vLLM behavior
- Support /v1/completions and /v1/chat/completions
- Streaming response simulation with configurable latency
- Decoupled from bench code (separate module)
- Used for bench validation without real GPU hardware

## M3: Full OpenAI API Parameter Coverage ⬜
- All 4 input formats (string, array of strings, array of tokens, array of mixed)
- All sampling parameters: temperature, top_k, top_p, frequency_penalty, presence_penalty
- Stop sequences, logprobs, max_tokens, n, seed
- Validate no parameter is omitted vs OpenAI spec
- Dummy server must support all parameters too

## M4: Flexible Request Rate Patterns ⬜
- Per-second (vLLM bench compatible)
- Per-N-seconds (5s, 10s, custom interval)
- Burst patterns, ramp-up/ramp-down
- Poisson distribution
- Custom pattern via config file

## M5: Rich Dataset Input ⬜
- JSONL input format
- JSON array input format
- CSV input format
- Synthetic dataset generation (configurable prompt/output length distribution)
- Dataset validation and stats reporting

## M6: Extended Metrics & Reporting ⬜
- Per-request detailed metrics export
- Percentile breakdown (P50/P90/P95/P99)
- Time-series metrics (throughput over time)
- Rich terminal output (progress bars, live stats)
- JSON and human-readable report formats
