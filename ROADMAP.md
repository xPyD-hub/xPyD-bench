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

## M38: Benchmark Presets Library ✅
- `xpyd-bench presets list` to show built-in and user-defined presets
- `xpyd-bench presets show <name>` to display preset configuration
- `--preset <name>` as shorthand for common benchmark configurations
- User preset directory: `~/.xpyd-bench/presets/` (YAML files)
- Built-in presets: throughput-max, latency-optimal, soak-test, cold-start
- Tests covering preset loading, user overrides, and CLI integration

## M39: Cost Estimation ✅
- `--cost-model <path>` YAML file mapping model names to $/1K tokens (input/output)
- After benchmark, compute estimated cost based on actual token counts
- Cost included in JSON result, CSV export, and terminal summary
- `--dry-run` shows estimated cost before running
- YAML config support (`cost_model`)
- Tests covering cost calculation, multiple models, and dry-run estimation

## M40: Request Payload Compression ✅
- `--compress` CLI flag to enable gzip compression of request bodies (`Content-Encoding: gzip`)
- Useful for large-prompt benchmarks to reduce network overhead
- YAML config support (`compress: true`)
- Dummy server supports `Content-Encoding: gzip` decompression
- Metrics: track compressed vs uncompressed payload sizes in debug log
- Tests covering compression encoding, dummy server decompression, and CLI integration

## M41: Batch Inference API Benchmarking ✅
- Support `/v1/batch` endpoint for offline batch inference benchmarking
- `xpyd-bench run --endpoint /v1/batch` with batch-specific metrics (queue time, processing time)
- Batch submission, polling, and result retrieval workflow
- Dummy server `/v1/batch` implementation
- Tests covering batch submission, polling, result retrieval, and metrics

## M42: Request ID Tracking & Correlation ✅
- Auto-generate unique `X-Request-ID` header for each request
- `--request-id-prefix <prefix>` CLI flag for custom prefixes
- Request IDs in debug log, per-request export, and error messages
- Dummy server echoes `X-Request-ID` in response headers
- Correlation between client-side and server-side logs
- YAML config support (`request_id_prefix`)
- Tests covering ID generation, header injection, echo, and export

## M43: Request Latency Anomaly Detection ✅
- `detect_anomalies()` function in `src/xpyd_bench/bench/anomaly.py` using IQR method
- `--anomaly-threshold <float>` CLI flag (default 1.5, 0 disables)
- BenchmarkResult includes `anomalies` dict field
- JSON output includes `anomalies` section when anomalies found
- Terminal summary prints anomaly count and worst offenders
- YAML config support (`anomaly_threshold`)
- Tests covering detection logic, threshold customization, no-anomaly case, disabled mode

## M44: Concurrency Sweep Mode ✅
- `xpyd-bench sweep --concurrency-range 1,2,4,8,16,32` CLI subcommand
- `--concurrency-range` accepts comma-separated values or start:stop:step notation (e.g. `1:32:2x` for exponential)
- Each concurrency level runs a full benchmark (`--sweep-prompts N`, default 100)
- Summary table: concurrency vs throughput, mean latency, P99 latency, error rate
- Identify and highlight optimal concurrency (max throughput with <=5% error rate)
- JSON output with per-level results via `--sweep-output <path>`
- YAML config support (`sweep` section)
- Tests covering sweep orchestration, result aggregation, optimal detection, CLI integration

## M45: Multi-Turn Conversation Benchmarking ✅
- `xpyd-bench run --multi-turn <path>` CLI flag to load multi-turn conversation datasets
- JSONL format with `messages` array per line (OpenAI chat format)
- Sequential turn execution: send turn 1, wait for response, append to context, send turn 2, etc.
- Per-turn metrics: TTFT, TPOT, latency for each turn in the conversation
- Aggregate metrics across all conversations and per-turn-index
- Context growth impact analysis: latency vs conversation depth
- `--max-turns N` to cap conversation length
- Synthetic multi-turn generation with configurable turn count and message lengths
- Dummy server stateful conversation support (maintains context window)
- YAML config support (`multi_turn`, `max_turns`)
- Tests covering multi-turn execution, per-turn metrics, context growth, and CLI integration

## M46: Duration-based Benchmarking ✅
- `--duration SECONDS` CLI flag to run benchmark for a fixed time period
- Prompts cycle round-robin until duration expires
- When combined with `--num-prompts`, benchmark stops at whichever limit is reached first
- `BenchmarkResult` includes `duration_limit` field
- YAML config support (`duration`)
- Works with all rate patterns, token bucket, adaptive concurrency
- Dry-run shows duration mode info
- Tests covering CLI parsing, dry-run output, YAML config, model serialization

## M47: Response Validation & Content Checking ✅
- `--validate-response <mode>` CLI flag (repeatable) for response content validation
- Modes: `non-empty`, `json`, `regex:<pattern>`, `min-tokens:<N>`
- Multiple validators can be chained
- `RequestResult` includes `response_text` and `validation_errors` fields
- `BenchmarkResult` includes `validation_summary` with pass/fail counts and pass rate
- Terminal summary shows validation pass rate
- YAML config support (`validate_response: ["non-empty", "min-tokens:10"]`)
- Tests covering all validation modes, aggregation, CLI integration, and YAML config

## M48: Endpoint Health Check ✅
- `xpyd-bench healthcheck --base-url <url>` CLI subcommand
- Check connectivity, /v1/models, /v1/completions, /v1/chat/completions, /v1/embeddings
- Auto-detect model from /v1/models when not specified
- `--api-key` for authenticated endpoints (falls back to OPENAI_API_KEY env)
- `--timeout` for connection timeout configuration
- `--json` flag for machine-readable JSON output
- Human-readable summary with ✓/✗ indicators and latency
- Exit code 0 for healthy, 1 for unhealthy (CI-friendly)
- Tests covering healthy, unreachable, JSON output, API key, auto-detect scenarios

## M49: JUnit XML Export for CI Integration ✅
- `--junit-xml <path>` CLI flag to export benchmark results as JUnit XML
- Each request maps to a `<testcase>` element with latency as duration attribute
- Failed requests produce `<failure>` elements with error messages
- SLA violations (when `--sla` used) appear as separate testcases in a dedicated testsuite
- Validation failures included as testcase failures
- YAML config support (`junit_xml`)
- Compatible with Jenkins, GitHub Actions, GitLab CI JUnit report parsers
- Tests covering XML structure, request failures, SLA integration, and CLI integration

## M50: Benchmark Repeat Mode ✅
- `--repeat N` CLI flag to run the benchmark N times (default 1)
- `--repeat-delay SECONDS` CLI flag for pause between runs (default 0)
- YAML config support (`repeat`, `repeat_delay`)
- After all runs, display per-run summary table and aggregated statistics
- Reuses existing `aggregate_results()` for cross-run aggregation
- Partial repeat: if interrupted (SIGINT), aggregate completed runs
- JSON output includes `repeat_results` array and `repeat_summary` section
- `--dry-run` shows repeat configuration
- Tests covering repeat execution, delay, aggregation, CLI parsing, YAML config

## M51: Model Warmup Profiling ✅
- `--warmup-profile` flag to measure and report warmup characteristics
- Track latency curve during warmup phase (cold → steady state)
- Detect warmup completion point automatically (latency stabilization)
- Report warmup duration and cold-start penalty in JSON output
- Separate warmup metrics section in HTML report
- YAML config support (`warmup_profile: true`)
- Tests covering warmup detection, profiling output, and edge cases

## M52: Request Priority & Queuing ✅
- `--priority-levels N` CLI flag to simulate priority-based request scheduling
- Priority field in JSONL dataset format (`"priority": 0-9`)
- Metrics broken down by priority level (latency, throughput, error rate)
- Priority-aware rate limiting: higher priority requests sent first
- YAML config support (`priority_levels`)
- Tests covering priority scheduling, per-priority metrics, and CLI integration

## M53: Server-Sent Events (SSE) Metrics ✅
- Enhanced streaming metrics: inter-token latency distribution, token delivery jitter
- `--sse-metrics` flag to enable detailed SSE event analysis
- Per-chunk timing in debug log and per-request export
- Stall detection: flag periods with no tokens for >Xms
- HTML report SSE timeline visualization
- Tests covering SSE parsing, jitter calculation, and stall detection

## M54: Benchmark Diff Report ✅
- `xpyd-bench diff <result1.json> <result2.json> --html-diff <path>` subcommand
- Visual side-by-side HTML diff with highlighted regressions/improvements
- Metric sparklines showing distribution differences
- Statistical significance testing (Mann-Whitney U) for metric comparisons
- Markdown diff output for PR comments
- Tests covering diff generation, significance testing, and output formats

## M55: Load Shedding Simulation ✅
- `--load-shed-threshold <rps>` to simulate server-side load shedding
- Gradually increase request rate until target rejects requests (429/503)
- Find maximum sustainable throughput automatically
- Report saturation point, degradation curve, and recovery behavior
- JSON output with saturation analysis section
- Tests covering load shedding detection, saturation point, and CLI integration

## M56: Structured Output / Function Calling Benchmarking ✅
- `--tools <path>` CLI flag to load tool/function definitions (OpenAI function calling format)
- `--response-format json_object` and `--response-format json_schema` support
- Measure structured output overhead: latency delta vs unconstrained generation
- Validate response conforms to provided JSON schema
- Dummy server supports `tools`, `tool_choice`, and `response_format` parameters
- Per-request tracking of tool call extraction success/failure
- Summary metrics: schema conformance rate, structured output latency overhead
- YAML config support (`tools`, `response_format`)
- Tests covering function calling, JSON mode, schema validation, and dummy server

## M57: Network Latency Decomposition ✅
- Measure and report DNS resolution, TCP connect, TLS handshake, and server processing times separately
- `--latency-breakdown` flag to enable detailed network timing
- Per-request breakdown in debug log and per-request export
- Summary with mean/P50/P99 for each phase
- HTML report includes latency waterfall chart
- Tests covering timing decomposition and reporting

## M58: Benchmark Result Archival & Cloud Storage ✅
- `--archive <backend>` flag to push results to S3, GCS, or local archive
- Archive manifest with run metadata for querying historical results
- `xpyd-bench archive list` and `xpyd-bench archive fetch <run-id>` subcommands
- Plugin interface for custom storage backends
- Tests covering archival workflow and retrieval

## M59: Request Dependency Chains ✅
- Define request sequences where output of request N feeds into request N+1
- `--chain <path>` JSONL file defining extraction rules between requests
- Measure end-to-end chain latency and per-step contribution
- Useful for benchmarking RAG pipelines and agent workflows
- Tests covering chain execution, extraction, and metrics

## M60: Noise Injection & Chaos Testing ✅
- `--inject-delay <ms>` to add artificial client-side delay (simulate slow networks)
- `--inject-error-rate <float>` to randomly abort requests client-side
- `--inject-payload-corruption <float>` to send malformed payloads
- Measure server resilience and error handling under adverse conditions
- Tests covering injection modes and metric impact

## M61: Webhook Notifications ✅
- `--webhook-url <url>` CLI flag (repeatable) to POST benchmark results to webhook endpoints
- `--webhook-secret <secret>` for HMAC-SHA256 signature in `X-Webhook-Signature` header
- Timeout and retry on delivery failure (max 3 attempts with backoff)
- Non-blocking: webhook failure does not affect benchmark exit code
- YAML config support (`webhook_url`, `webhook_secret`)
- Tests covering delivery, signature, retry, server error, and CLI integration

## M62: Request Trace Export (OpenTelemetry) ✅
- `--otlp-endpoint <url>` to export per-request spans to an OpenTelemetry collector
- Span attributes: prompt_tokens, completion_tokens, TTFT, TPOT, model, endpoint
- Parent span for entire benchmark run with child spans per request
- YAML config support (`otlp_endpoint`)
- Tests covering span generation and attribute correctness

## M63: Benchmark Scheduling & Cron Integration ✅
- `xpyd-bench schedule --cron "0 */6 * * *" --config bench.yaml` to generate crontab entries
- `--on-complete <command>` to run shell command after benchmark (e.g., notify, upload)
- Schedule validation and human-readable next-run preview
- Tests covering cron generation and on-complete execution

## M64: Endpoint Capability Discovery ✅
- `xpyd-bench discover --base-url <url>` probes endpoint to auto-detect capabilities
- Detect: /v1/completions, /v1/chat/completions, /v1/embeddings, /v1/batch, streaming, function calling
- Auto-discover available models via /v1/models
- `--api-key` for authenticated endpoints (OPENAI_API_KEY env fallback)
- `--timeout` for probe timeout configuration
- `--json` for machine-readable JSON output
- `--generate-config <path>` writes recommended YAML config based on discovered capabilities
- Human-readable summary with ✓/✗ indicators
- Tests covering discovery logic, JSON output, config generation, CLI integration

## M65: Output Verbosity Control ✅
- `--quiet`/`-q` CLI flag to suppress non-essential output (errors and final JSON only)
- `--verbose`/`-v` CLI flag for extra detail (config summary, per-request progress)
- Default behavior unchanged (normal verbosity)
- `Verbosity` enum and `VerbosityPrinter` utility in `src/xpyd_bench/bench/verbosity.py`
- `parse_verbosity()` helper for string-to-enum conversion
- YAML config support (`verbosity: quiet|normal|verbose`)
- CLI flags override YAML config
- Tests covering all three verbosity levels, CLI flags, and YAML config

## M66: Rate-Limit Header Tracking & Backpressure Reporting ✅
- `--track-ratelimits` CLI flag to parse `X-RateLimit-*` and standard `RateLimit-*` response headers
- Track remaining quota, reset time, and limit values from each response
- Per-request `ratelimit_headers` field in `RequestResult`
- `BenchmarkResult` includes `ratelimit_summary` with min remaining, throttle events (429 count), time under pressure
- Terminal summary prints rate-limit pressure indicators when tracking enabled
- Dummy server returns configurable rate-limit headers (`--ratelimit-rpm N`)
- YAML config support (`track_ratelimits: true`)
- Tests covering header parsing, summary aggregation, dummy server headers, and CLI integration

## M67: Request/Response Payload Size Tracking ✅
- `--track-payload-size` CLI flag to enable bandwidth tracking
- Track `request_bytes` and `response_bytes` per request in `RequestResult`
- Aggregate `payload_summary` in `BenchmarkResult` (total, mean, min, max, P50/P99)
- Works with both streaming and non-streaming responses
- YAML config support (`track_payload_size: true`)
- Tests covering tracking logic, aggregation, CLI flag, and config key

## M68: Output Token Speed Benchmarking & Comparison ✅
- `--measure-generation-speed` CLI flag to compute tokens-per-second for each request's output generation phase
- Per-request `generation_tps` field in `RequestResult` (completion_tokens / generation_time)
- `BenchmarkResult` includes `generation_speed_summary` with mean, P50, P90, P99 tokens/s
- Terminal summary prints generation speed alongside existing throughput metrics
- Compare tool includes generation speed in regression detection
- HTML report adds generation speed distribution chart
- YAML config support (`measure_generation_speed: true`)
- Tests covering speed calculation, summary aggregation, compare integration, and CLI flag

## M69: Benchmark Metadata & Notes ✅
- `--note "description"` CLI flag to attach a human-readable note to a benchmark run
- Note stored in `BenchmarkResult.note` field and included in JSON output
- `xpyd-bench history` displays the note column
- YAML config support (`note: "description"`)
- `note` added to known config keys for validation
- Tests covering CLI parsing, JSON output, history display, YAML config

## M70: Request Timeout Classification & Reporting ✅
- Classify timed-out vs completed vs errored requests separately in metrics
- `BenchmarkResult` includes `timeout_summary` with count, percentage, and latency-at-timeout stats
- Terminal summary shows timeout breakdown when timeouts occur
- Per-request `timeout_detected` boolean field in `RequestResult`
- Timeout classification based on configured `--timeout` value
- JSON output includes `timeout_summary` section
- Tests covering timeout classification, summary aggregation, and edge cases

## M71: Request Queuing Time Measurement ✅
- Measure client-side queuing time (time between request creation and actual send)
- Per-request `queue_time` field in `RequestResult`
- `BenchmarkResult` includes `queue_time_summary` with mean/P50/P99 stats
- Distinguishes scheduling delay from actual server processing
- Useful for understanding rate limiter and concurrency bottlenecks
- Tests covering queue time measurement, summary stats, and high-concurrency scenarios

## M72: Benchmark Fingerprinting ✅
- Generate deterministic fingerprint hash from benchmark configuration (CLI args + config)
- `BenchmarkResult` includes `fingerprint` field (SHA-256 of normalized config)
- `xpyd-bench history` can group runs by fingerprint to track same-config performance over time
- `xpyd-bench aggregate --by-fingerprint` to auto-group results by config
- Tests covering fingerprint determinism, config normalization, and grouping

## M73: Custom Percentile Configuration ✅
- `--percentiles 50,90,95,99,99.9` CLI flag to specify which latency percentiles to compute
- Default: `50,90,95,99` (preserves current behavior)
- BenchmarkResult includes `custom_percentiles` dict mapping metric prefix to {pN: value}
- JSON output includes `custom_percentiles` section
- Terminal summary shows all requested percentiles
- YAML config support (`percentiles: [50, 90, 95, 99, 99.9]`)
- Tests covering custom percentile computation, CLI parsing, YAML config, edge cases

## M74: Benchmark Checkpointing & Resume ✅
- `--checkpoint-dir <path>` CLI flag to enable periodic checkpointing during benchmark
- Save completed request results every N requests (default 50) to checkpoint file
- `xpyd-bench resume --checkpoint <path>` to resume an interrupted benchmark from checkpoint
- Resume skips already-completed prompts and continues from where it stopped
- Final result merges checkpoint data with new results seamlessly
- Checkpoint file includes config snapshot for validation on resume
- YAML config support (`checkpoint_dir`, `checkpoint_interval`)
- Tests covering checkpoint creation, resume logic, config mismatch detection, and CLI integration

## M75: Multi-Model Comparison Mode ✅
- `xpyd-bench model-compare --models model1,model2 --base-url <url>` CLI subcommand
- Benchmark same prompts against multiple models on same endpoint
- Side-by-side per-model metrics (TTFT, TPOT, throughput, latency percentiles)
- Statistical significance testing between models (reuse diff logic)
- JSON and Markdown comparison output
- Tests covering multi-model orchestration, result comparison, and CLI integration

## M76: Streaming vs Non-Streaming Overhead Analysis ✅
- `xpyd-bench stream-compare --base-url <url> --model <model>` CLI subcommand
- Auto-run same prompts in streaming and non-streaming modes
- Report streaming overhead: TTFT delta, total latency delta, throughput impact
- Identify when streaming is beneficial vs detrimental
- JSON and terminal summary output
- Tests covering dual-mode execution, overhead calculation, and CLI integration

## M77: Multimodal (Vision) Benchmarking ✅
- `--image-url <url>` CLI flag to include an HTTP image in every chat prompt
- `--image-dir <path>` to randomly sample images from a directory per request
- `--synthetic-images N` to generate N synthetic PNG images for benchmarking
- `--synthetic-image-size WxH` for synthetic image dimensions (default 64x64)
- `--image-detail auto|low|high` to control vision detail level
- Multimodal content array in chat messages (OpenAI vision API format)
- Dummy server handles multimodal content in token estimation
- YAML config support (`image_url`, `image_dir`, `synthetic_images`, `synthetic_image_size`, `image_detail`)
- Tests covering image generation, content building, CLI parsing, payload construction, and config keys

## M78: Workload Distribution Statistics ✅
- `--workload-stats` CLI flag to compute and display prompt/output token length distributions
- Per-request `prompt_tokens` and `completion_tokens` already tracked; aggregate into distribution stats
- Report mean, stddev, min, max, P50/P90/P99 for both prompt and output token lengths
- Terminal summary table with distribution overview
- JSON output includes `workload_stats` section
- HTML report includes prompt/output length distribution histograms
- Useful for understanding actual workload characteristics vs configured parameters
- YAML config support (`workload_stats: true`)
- Tests covering stats computation, CLI flag, YAML config, and edge cases

## M79: Git Metadata Capture in Benchmark Results ✅
- `collect_git_info()` utility in `src/xpyd_bench/bench/env.py`
- Auto-detect git repository and capture: commit hash, short commit, branch, dirty status, remote URL
- `BenchmarkResult` includes `git_info` dict field (auto-populated, None if not in git repo)
- All saved JSON results include `git_info` section when available
- `--dry-run` output shows git metadata section
- Graceful fallback: returns None when git unavailable, not in repo, or command times out
- Tests covering git info collection, fallback scenarios, serialization, and model field
