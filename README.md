# xPyD-bench

**Benchmarking & PD ratio planning tool for LLM inference endpoints.**

xPyD-bench measures the performance of OpenAI-compatible LLM serving endpoints with detailed latency, throughput, and quality metrics. Built as a superset of vLLM bench with full CLI compatibility.

## Key Features

- **vLLM bench compatible CLI** — drop-in replacement, same arguments
- **Rich metrics** — TTFT, TPOT, ITL, P50/P90/P95/P99, throughput
- **Flexible load patterns** — constant, burst, ramp, poisson, custom
- **Multiple datasets** — JSONL, CSV, JSON, synthetic generation
- **Advanced analysis** — comparison, regression detection, SLA validation, cost estimation
- **Reports** — JSON, CSV, Markdown, HTML dashboard, JUnit XML, Prometheus

## Install

```bash
pip install xpyd-bench
```

Or as part of the full xPyD toolkit:

```bash
pip install xpyd
```

## Quick Start

```bash
# Benchmark a running endpoint
xpyd-bench --base-url http://localhost:8080 \
           --model my-model \
           --dataset-name random \
           --num-prompts 100

# Compare two runs
xpyd-bench compare baseline.json candidate.json
```

## Part of xPyD

xPyD-bench is part of the [xPyD ecosystem](https://github.com/xPyD-hub/xPyD) for PD-disaggregated LLM serving:

| Component | Description |
|-----------|-------------|
| [xpyd-proxy](https://github.com/xPyD-hub/xPyD-proxy) | Prefill-Decode disaggregated proxy |
| [xpyd-sim](https://github.com/xPyD-hub/xPyD-sim) | OpenAI-compatible inference simulator |
| **xpyd-bench** | Benchmarking & planning tool |

📖 **[Full Guide →](docs/guide.md)** | 💡 **[Examples →](examples/)** | 🏗️ **[Contributing →](CONTRIBUTING.md)**

## License

Apache 2.0 — see [LICENSE](LICENSE)
