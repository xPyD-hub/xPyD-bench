<!-- CRITICAL: DO NOT SUMMARIZE OR COMPRESS THIS FILE -->
<!-- This file contains precise rules that must be read in full. -->

# xPyD-bench Design Principles

## Core Positioning
A comprehensive benchmarking tool for LLM inference endpoints, built as an enhancement on top of vLLM bench.

## CLI Compatibility
- **CLI arguments must be fully compatible with vLLM bench** — users can switch from `python benchmark_serving.py` to `xpyd-bench` without changing their command line
- Extended features beyond vLLM bench CLI should be configured via **YAML config file** (`--config config.yaml`)
- Basic usage = CLI only (vLLM bench compatible), advanced usage = CLI + YAML

## Alignment with vLLM Bench
- CLI arguments must align with vLLM bench where applicable
- Output format must align with vLLM bench
- We build incremental improvements on top of vLLM bench, not a replacement

## Areas of Enhancement
- **Full OpenAI API coverage**: every parameter matters — all 4 input formats, temperature, top_k, top_p, frequency_penalty, presence_penalty, stop sequences, logprobs, etc. No omissions.
- **Flexible request rate patterns**: vLLM bench only supports per-second rate. Support per-5s, per-10s, burst patterns, ramp-up/ramp-down, Poisson distribution, custom patterns.
- **Rich dataset input**: support JSONL, JSON, CSV — let users bring their own data easily.
- **Extended metrics**: beyond what vLLM bench provides.

## Architecture
- Bench is a **pure client tool** — it sends requests and measures responses.
- No built-in server or simulator. Backend simulation is handled by [xPyD-sim](https://github.com/xPyD-hub/xPyD-sim).
- Integration tests (bench + sim, bench + proxy + sim) live in [xPyD-integration](https://github.com/xPyD-hub/xPyD-integration).
- Code in `xpyd_bench/`, tests in `tests/`.

## Code Organization
- `xpyd_bench/bench/` — core benchmark runner, metrics, rate patterns
- `xpyd_bench/reporting/` — output formats (JSON, CSV, HTML, Prometheus)
- `xpyd_bench/scenarios/` — preset configurations
- `xpyd_bench/distributed/` — multi-worker coordination
- `xpyd_bench/plugins/` — backend plugin system
- `tests/` — unit tests only (no external dependencies)
