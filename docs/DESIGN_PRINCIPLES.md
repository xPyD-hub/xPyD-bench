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

## Areas of Enhancement (think creatively, these are examples)
- **Full OpenAI API coverage**: every parameter matters — all 4 input formats, temperature, top_k, top_p, frequency_penalty, presence_penalty, stop sequences, logprobs, etc. No omissions.
- **Flexible request rate patterns**: vLLM bench only supports per-second rate. Support per-5s, per-10s, burst patterns, ramp-up/ramp-down, Poisson distribution, custom patterns.
- **Rich dataset input**: support JSONL, JSON, CSV — let users bring their own data easily.
- **Extended metrics**: beyond what vLLM bench provides.
- Think about what else users need that vLLM bench doesn't offer.

## Dummy Server — vLLM Boundary Rule (IMPORTANT)

The dummy server simulates a **vLLM backend** for bench validation. It must stay strictly within vLLM's API surface:

### Hard Rules
1. **Only implement features that vLLM actually supports.** If vLLM doesn't have it, the dummy server must not have it.
2. **OpenAI API parameters**: only those that vLLM's OpenAI-compatible server accepts (including vLLM extensions like `best_of`, `top_k`, `min_p`, etc.).
3. **Response format**: must match vLLM's response structure, including vLLM-specific fields (`stop_reason`, `service_tier`, `kv_transfer_params`).
4. **No test-only hacks**: features like gzip decompression, rate-limit simulation (429/X-RateLimit headers), custom header echo, speculative decoding metadata injection, or online /v1/batches API do NOT belong in the dummy server — they are not vLLM behaviors.

### What Belongs Here vs. Elsewhere
| Need | Where to implement |
|---|---|
| Simulating vLLM inference behavior | ✅ Dummy server |
| Testing bench's own features (compression, rate-limit tracking, header injection) | ❌ NOT dummy server — use a separate test fixture/middleware |
| Features vLLM doesn't support | ❌ NOT dummy server |

### Co-Evolution with xPyD-sim
- The dummy server will eventually be **replaced by xPyD-sim** as the canonical vLLM simulator.
- Any feature added to the dummy must also exist (or be planned) in xPyD-sim.
- When in doubt, check vLLM's source: `vllm/entrypoints/openai/` is the reference.
- Code must be decoupled from bench — separate module, no imports between them.

## Principles
- **Independent thinking**: reference vLLM bench for alignment, but design our own enhancements
- **Data-driven**: all metrics from real measurements
- **User-friendly**: easy CLI, sensible defaults, clear output
- **Rigorous**: every parameter, every edge case matters

## Rules
- Committer must be `hlin99 <tony.lin@intel.com>`
- All code, docs, issues, PRs in English
- Commit messages: conventional commits format
- Code in `src/xpyd_bench/`, tests in `tests/`
- Dummy server in `src/xpyd_bench/dummy/` (decoupled from bench code)
- Follow pyproject.toml ruff/isort config
