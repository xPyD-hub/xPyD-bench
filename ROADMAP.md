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

## M10: Configurable Timeouts & Retry Logic ✅
- `--timeout SECONDS` CLI flag for per-request HTTP timeout (default 300s)
- `--retries N` for automatic retry on transient errors (connection, 429, 503)
- `--retry-delay SECONDS` with exponential backoff (delay * 2^attempt)
- RequestResult tracks `retries` count
- YAML config support (`timeout`, `retries`, `retry_delay`)
- Tests covering timeout, retry success, retry exhaustion, backoff

## M11: API Key Authentication ✅
- `--api-key` CLI flag for Bearer token authentication
- `OPENAI_API_KEY` environment variable fallback when `--api-key` not provided
- Authorization header (`Bearer <key>`) added to all HTTP requests
- YAML config support (`api_key`)
- Dummy server optionally validates API key (`--require-api-key`)
- Tests covering auth header injection, env var fallback, missing key error, dummy server validation

## M12: Graceful Shutdown & Progress Persistence ✅
- SIGINT during benchmark triggers graceful shutdown instead of losing all data
- In-flight requests get a grace period (default 5s) to complete
- Partial `BenchmarkResult` computed from completed requests
- Result JSON includes `"partial": true` flag
- `--save-result` saves partial results
- Compare tool warns when comparing partial results
- Tests covering graceful shutdown, partial metrics, and compare warnings

## M13: Custom HTTP Headers ✅
- `--header "Key: Value"` CLI flag (repeatable) for arbitrary HTTP headers
- YAML config support (`headers: {"X-Custom": "value"}`)
- Headers merged with internal headers (Authorization); user headers take precedence
- Dummy server echoes received custom headers in response metadata for validation
- Tests covering CLI parsing, header injection, YAML config, precedence, and dummy echo

## M14: CSV & Markdown Export Formats ✅
- `--csv-report <path>` CLI flag to export summary metrics as CSV
- `--markdown-report <path>` CLI flag to export summary as Markdown table
- Per-request CSV export via `--export-requests-csv <path>`
- YAML config support (`csv_report`, `markdown_report`, `export_requests_csv`)
- Consistent column ordering across formats
- Tests covering all export paths, file content validation, and CLI integration

## M15: Multi-Endpoint Comparison Mode ✅
- `xpyd-bench multi --endpoints url1,url2 ...` to benchmark multiple endpoints in one run
- Side-by-side summary table with per-endpoint metrics
- Shared workload: same prompts, same ordering, same timing for fair comparison
- JSON and Markdown comparison output
- Reuse existing compare logic for pairwise regression detection
- Tests covering multi-endpoint orchestration and output

## M16: Token Bucket & Adaptive Concurrency ✅
- Token bucket rate limiter replacing simple sleep-based scheduling
- `--rate-algorithm token-bucket` CLI flag (default preserves current behavior)
- Adaptive concurrency: auto-adjust max in-flight requests based on server response latency
- `--adaptive-concurrency` flag with configurable target latency
- YAML config support for all new parameters
- Tests covering rate accuracy, adaptive scaling up/down, and edge cases

## M17: HTML Report Dashboard ✅
- `--html-report <path>` CLI flag to generate a self-contained HTML report
- Interactive charts: latency distribution histogram, throughput timeline, TTFT CDF
- Embedded CSS/JS (no external dependencies, works offline)
- Summary stats table with color-coded thresholds
- Per-request scatter plot (latency vs time)
- YAML config support (`html_report`)
- Tests covering HTML generation, file content validation, and CLI integration

## M18: Profile & Replay Mode ✅
- `xpyd-bench profile --output trace.json` to record request timing patterns from a live endpoint
- `xpyd-bench replay --trace trace.json --base-url <url>` to replay recorded patterns
- Trace format captures: timestamps, prompt lengths, endpoint, inter-request delays
- Deterministic replay for reproducible benchmarks across different servers
- Tests covering profile recording, replay accuracy, and trace format validation

## M19: Dry Run Mode ✅
- `--dry-run` CLI flag to validate configuration and dataset without sending HTTP requests
- Print execution plan: resolved base URL, endpoint, model, dataset stats, rate config
- Load and validate dataset (file or synthetic), report stats
- Validate YAML config if provided
- Print estimated benchmark duration based on num_prompts and request_rate
- Exit with code 0 on success, non-zero on validation errors
- YAML config support (`dry_run: true`)
- Works with all existing CLI flags (scenarios, custom headers, auth, etc.)
- Tests covering dry-run output, validation errors, and CLI integration

## M20: SLA Validation Mode ✅
- `--sla <path>` CLI flag to load SLA targets from a YAML file
- SLA file defines thresholds: max P99 TTFT, max P95 e2e latency, min throughput, max error rate
- After benchmark completes, validate results against SLA targets
- Print pass/fail for each SLA target with actual vs threshold
- Exit code 1 when any SLA target is violated (CI-friendly)
- JSON output includes `sla_results` section with per-target pass/fail
- YAML config support (`sla: <path>`)
- Tests covering SLA pass, SLA fail, partial SLA, missing metrics

## M21: Environment Info Capture ✅
- `collect_env_info()` utility in `src/xpyd_bench/bench/env.py`
- Captures: Python version, OS, platform, CPU architecture, hostname, xpyd-bench version, ISO 8601 timestamp
- `BenchmarkResult` includes `environment` dict field (auto-populated)
- All saved JSON results include `environment` section
- `--dry-run` output shows environment info
- Tests covering environment info structure and presence

## M22: Request Logging & Debug Mode ✅
- `--debug-log <path>` CLI flag to write per-request debug logs
- Each log entry: timestamp, URL, payload (truncated), status code, latency, error
- Useful for diagnosing failures in long benchmark runs
- YAML config support (`debug_log`)
- Tests covering log file creation and content

## M23: Configuration Dump & Validation ✅

## M24: Unified CLI Subcommand Interface ✅
- Single `xpyd-bench <subcommand>` entry point with subcommand routing
- Subcommands: `run`, `compare`, `multi`, `profile`, `replay`, `config dump`, `config validate`
- `xpyd-bench` with no subcommand defaults to `run` for backward compatibility
- `--version` flag prints package version
- Legacy standalone entry points still work but print deprecation warning
- Tests covering subcommand routing, backward compat, version flag, deprecation warnings

## M25: Result Aggregation Across Multiple Runs ✅
- `xpyd-bench aggregate <result1.json> <result2.json> ...` CLI subcommand
- Compute mean, stddev, min, max across runs for key metrics (TTFT, TPOT, throughput, latency percentiles)
- Coefficient of variation (CV) per metric to flag unstable benchmarks
- `--output <path>` to save aggregated summary as JSON
- `--min-runs N` to enforce minimum number of runs (default 2)
- Detect and warn about outlier runs (>2 stddev from mean)
- Human-readable table output with statistical summary
- Tests covering aggregation math, outlier detection, CLI integration, edge cases
- `xpyd-bench config dump` subcommand to print resolved configuration (CLI + YAML merged)
- `xpyd-bench config validate --config <path>` to validate YAML config without running
- Print warnings for deprecated or conflicting options
- Tests covering dump output and validation errors

## M26: Embeddings Endpoint Benchmarking ✅
- Support `/v1/embeddings` endpoint benchmarking
- `--endpoint /v1/embeddings` with model and input text
- Metrics: latency, throughput (requests/s, tokens/s), batch size impact
- Dummy server `/v1/embeddings` implementation with configurable vector dimensions
- Tests covering embeddings benchmarking and dummy server

## M27: Tokenizer-Accurate Token Counting ✅
- Integrate `tiktoken` for accurate token counting (instead of word-split heuristic)
- `--tokenizer <model>` CLI flag to select tokenizer (default: cl100k_base)
- Accurate prompt_tokens and completion_tokens in all metrics
- Synthetic dataset generation uses token-accurate lengths
- YAML config support (`tokenizer`)
- Fallback to word-split when tiktoken unavailable
- Tests covering token counting accuracy and fallback

## M28: Connection Pool & HTTP/2 Configuration ✅
- `--http2` flag to enable HTTP/2 multiplexing
- `--max-connections N` to configure connection pool size (default: 100)
- `--max-keepalive N` for keepalive connection limit
- Connection reuse metrics in debug log
- YAML config support for all connection parameters
- Tests covering HTTP/2 mode and pool configuration

## M29: Live Progress Dashboard (Terminal UI) ✅
- Real-time terminal dashboard using `rich.live` during benchmark execution
- Display: progress bar, current RPS, latency sparkline, error count, ETA
- `--no-live` flag to disable (for CI/non-interactive environments)
- Auto-detect non-TTY and disable gracefully
- Tests covering dashboard data feed and non-TTY fallback

## M30: Benchmark Result Storage & History ✅
- `--result-dir <path>` to auto-save results with timestamped filenames
- `xpyd-bench history --result-dir <path>` to list past runs with summary
- `xpyd-bench history --result-dir <path> --last N` to show last N runs
- Trend visualization: metric trends across recent runs (terminal sparklines)
- YAML config support (`result_dir`)
- Tests covering auto-save, history listing, and trend computation

## M31: Plugin Architecture for Custom Backends ✅
- Plugin interface for registering custom backend protocols (beyond OpenAI)
- `--backend custom --backend-plugin <module>` CLI integration
- Example plugin for vLLM native protocol
- Plugin discovery via entry points (`xpyd.backends`)
- Tests covering plugin registration, loading, and execution

## M32: Distributed Benchmark Coordination ✅
- `xpyd-bench distributed --workers worker1:8080,worker2:8080` for multi-machine load generation
- Coordinator node distributes prompts across workers
- Aggregated results from all workers
- Worker heartbeat and failure detection
- Tests covering coordination protocol and result aggregation

## M33: Real-time Metrics Streaming (WebSocket) ✅
- `--metrics-ws-port <port>` to expose live metrics via WebSocket
- JSON metrics pushed every second during benchmark
- External dashboards (Grafana, custom) can subscribe
- Tests covering WebSocket server and metric streaming

## M34: Prometheus / OpenMetrics Export ✅
- `--prometheus-export <path>` CLI flag to write metrics in Prometheus exposition format
- Metrics: `xpyd_bench_ttft_seconds`, `xpyd_bench_tpot_seconds`, `xpyd_bench_request_latency_seconds` (histograms)
- Metrics: `xpyd_bench_throughput_tokens_per_second` (gauge), `xpyd_bench_requests_total`, `xpyd_bench_errors_total` (counters)
- Labels: model, endpoint, scenario (if applicable)
- YAML config support (`prometheus_export`)
- Compatible with `node_exporter` textfile collector and `promtool check metrics`
- Tests covering export format, histogram buckets, label injection, and CLI integration

## M35: Request Latency Heatmap ✅
- Time-bucketed latency heatmap visualization in terminal (rich) and HTML report
- X-axis: benchmark elapsed time, Y-axis: latency buckets
- Color intensity: request count per bucket
- `--heatmap` flag to enable terminal heatmap after benchmark
- Automatically included in HTML report when `--html-report` is used
- Tests covering heatmap data generation and rendering

## M36: Benchmark Annotations & Tags ✅
- `--tag key=value` CLI flag (repeatable) to attach metadata tags to benchmark runs
- Tags stored in result JSON `tags` field
- `xpyd-bench history --filter-tag env=prod` to filter history by tags
- YAML config support (`tags: {env: prod, gpu: A100}`)
- Tests covering tag storage, filtering, and CLI integration

## M37: Request Body Templating ✅
- Jinja2-style template variables in prompt strings (`{{ variable }}`)
- `--template-vars <path>` to load variables from JSON/YAML file
- Variable substitution applied before sending requests
- Enables parameterized benchmarks (different user names, topics, etc.)
- YAML config support (`template_vars`)
- Tests covering template rendering, missing variables, and CLI integration

## M38: Benchmark Presets Library
- `xpyd-bench presets list` to show built-in and user-defined presets
- `xpyd-bench presets show <name>` to display preset configuration
- `--preset <name>` as shorthand for common benchmark configurations
- User preset directory: `~/.xpyd-bench/presets/` (YAML files)
- Built-in presets: throughput-max, latency-optimal, soak-test, cold-start
- Tests covering preset loading, user overrides, and CLI integration

## M39: Cost Estimation
- `--cost-model <path>` YAML file mapping model names to $/1K tokens (input/output)
- After benchmark, compute estimated cost based on actual token counts
- Cost included in JSON result, CSV export, and terminal summary
- `--dry-run` shows estimated cost before running
- YAML config support (`cost_model`)
- Tests covering cost calculation, multiple models, and dry-run estimation
