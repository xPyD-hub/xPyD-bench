# xPyD-bench — Current Iteration Status

> Updated: 2026-04-06

## Current Milestone: M95 — Benchmark Result Diffing by Tag

### What was done

- Added `xpyd-bench tag-compare --result-dir <path> --group-by <tag-key>` CLI subcommand
- New module `src/xpyd_bench/tag_compare.py` implementing:
  - `group_results_by_tag()` — groups JSON results by tag value
  - `compute_group_stats()` — per-group mean for all standard metrics
  - `compute_pairwise_significance()` — Mann-Whitney U test between all group pairs
  - `tag_compare()` — full orchestration returning structured result
  - `format_tag_compare_table()` — human-readable terminal output
  - `format_tag_compare_markdown()` — Markdown table output
  - `tag_compare_main()` — CLI entry point
- `--json` flag for machine-readable JSON output
- `--markdown` flag for Markdown table output
- Exit code 1 when no groups found
- Registered `tag-compare` subcommand in `main.py`
- 33 tests covering grouping, stats, significance, formatting, CLI, edge cases

### Files Changed
- `src/xpyd_bench/tag_compare.py` (new)
- `src/xpyd_bench/main.py` (registered `tag-compare` subcommand)
- `tests/test_tag_compare.py` (new, 33 tests)
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
| 5 | 2026-04-06 | M93: Request Pacing Accuracy Report | ✅ merged (PR #250) | Both approved |
| 6 | 2026-04-06 | M94: Model Output Quality Scoring | ✅ merged (PR #252) | Both approved |
| 7 | 2026-04-06 | M95: Benchmark Result Diffing by Tag | ✅ merged (PR #254) | Both approved |
| 8 | 2026-04-06 | M96: Endpoint Response Consistency Check | ✅ merged (PR #256) | Both approved |
| 9 | 2026-04-06 | M97: Request Latency Heatmap Data Export | ✅ merged (PR #258) | Both approved |
| 10 | 2026-04-06 | M98: Auto-Tuning Optimal Configuration | ⏳ in progress | — |
