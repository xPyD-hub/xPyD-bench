# xPyD-bench Design Principles

## Core Positioning
A comprehensive benchmarking tool for LLM inference endpoints, built as an enhancement on top of vLLM bench.

## Alignment with vLLM Bench
- CLI arguments must align with vLLM bench where applicable
- Output format must align with vLLM bench
- We build incremental improvements on top of vLLM bench, not a replacement

## Areas of Enhancement (think creatively, these are examples)
- **Full OpenAI API coverage**: every parameter matters — all 4 input formats, temperature, top_k, top_p, frequency_penalty, presence_penalty, stop sequences, logprobs, etc. No omissions.
- **Flexible request rate patterns**: vLLM bench only supports per-second rate. Support per-5s, per-10s, burst patterns, ramp-up/ramp-down, Poisson distribution, custom patterns.
- **Rich dataset input**: support JSONL, JSON, CSV — let users bring their own data easily.
- **Extended metrics**: beyond what vLLM bench provides.
- Think about what else users need that vLLM bench doesn't offer.

## Dummy Server (for testing)
- A dummy prefill/decode server that simulates vLLM behavior for bench validation
- Must support all OpenAI API endpoints that bench tests against
- Code must be decoupled from bench — separate module, no imports between them
- Goal: when dummy is mature, it migrates to the xPyD-simulator repo

## Principles
- **Independent thinking**: reference vLLM bench for alignment, but design our own enhancements
- **Data-driven**: all metrics from real measurements
- **User-friendly**: easy CLI, sensible defaults, clear output
- **Rigorous**: every parameter, every edge case matters

## Rules
- Committer must be `hlin99 <hlin99@gmail.com>`
- All code, docs, issues, PRs in English
- Commit messages: conventional commits format
- Code in `src/xpyd_bench/`, tests in `tests/`
- Dummy server in `src/xpyd_bench/dummy/` (decoupled from bench code)
- Follow pyproject.toml ruff/isort config
