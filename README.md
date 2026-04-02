# xPyD-bench

Benchmarking & PD ratio planning tool for [xPyD-proxy](https://github.com/xPyD-hub/xPyD-proxy).

## Features

- **`xpyd-bench`** — Benchmark xPyD proxy with configurable concurrency, request patterns, and both `/v1/completions` and `/v1/chat/completions` endpoints

For PD ratio planning, see [xPyD-plan](https://github.com/xPyD-hub/xPyD-plan).

## Install

```bash
pip install xpyd-bench
```

## Quick Start

### Benchmark

```bash
# Run benchmark against a running xPyD proxy
xpyd-bench --target http://localhost:8080 \
           --endpoint chat \
           --concurrency 16 \
           --num-requests 200 \
           --output results.json

# Use completion endpoint
xpyd-bench --target http://localhost:8080 \
           --endpoint completion \
           --concurrency 8 \
           --num-requests 100
```

## Configuration

See [examples/](examples/) for sample configs and scenarios.

## Output Metrics

- **TTFT** — Time to first token
- **TPS** — Tokens per second (per request & aggregate)
- **Latency** — P50 / P90 / P99 end-to-end latency
- **Throughput** — Total requests/sec and tokens/sec
- **Error rate** — Failed requests count and percentage

## License

TBD
