# xPyD-bench User Guide

> Benchmarking tool for [xPyD-proxy](https://github.com/xPyD-hub/xPyD-proxy) — measure latency, throughput, and PD-disaggregated inference performance.

## Installation

```bash
pip install xpyd-bench

# Optional dependencies
pip install xpyd-bench[tokenizer]   # tiktoken for accurate token counting
pip install xpyd-bench[http2]       # HTTP/2 support
pip install xpyd-bench[dev]         # Development & testing
```

Install from source:

```bash
git clone https://github.com/xPyD-hub/xPyD-bench.git
cd xPyD-bench
pip install -e ".[dev]"
```

## Core Commands

### xpyd-bench

Main entry point for running benchmarks.

```bash
xpyd-bench [OPTIONS]
```

### Main Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--base-url` | `http://127.0.0.1:8000` | Target server URL |
| `--model` | *(auto-detect)* | Model name; auto-fetched from server if omitted |
| `--endpoint` | `/v1/completions` | API endpoint path |
| `--num-prompts` | `1000` | Total number of requests |
| `--request-rate` | `inf` | Requests per second; `inf` sends all concurrently |
| `--max-concurrency` | *(unlimited)* | Maximum concurrent requests |
| `--input-len` | `256` | Input prompt token length |
| `--output-len` | `128` | Maximum output token count |
| `--stream` / `--no-stream` | *(auto)* | Enable/disable streaming responses |
| `--duration` | *(none)* | Fixed run duration (seconds); auto-stops when elapsed |
| `--dataset-name` | `random` | Dataset type: `random` / `synthetic` |
| `--dataset-path` | *(none)* | Custom dataset file path (.jsonl/.json/.csv) |
| `--seed` | `0` | Random seed |
| `--burstiness` | `1.0` | Burstiness factor (1.0 = Poisson distribution) |
| `--repeat` | `1` | Number of repeat runs |
| `--repeat-delay` | `0` | Delay between repeat runs (seconds) |
| `--output` / `-o` | *(stdout)* | Output file path for results |
| `--backend` | `openai` | Backend type |
| `--backend-plugin` | *(none)* | Custom backend plugin module path |

#### Sampling Parameters

| Parameter | Description |
|-----------|-------------|
| `--temperature` | Sampling temperature |
| `--top-p` | Nucleus sampling |
| `--frequency-penalty` | Frequency penalty |
| `--presence-penalty` | Presence penalty |
| `--stop` | Stop sequence |

### Other Subcommands

```bash
xpyd-bench compare    # Compare multiple benchmark results
xpyd-bench profile    # Performance profiling mode
xpyd-bench replay     # Replay recorded requests
xpyd-bench config-dump      # Export current configuration
xpyd-bench config-validate  # Validate configuration file
xpyd-dummy             # Start dummy server for testing
```

## Typical Use Cases

### 1. Single-Machine Test

Run a benchmark against a locally running vLLM / xPyD instance:

```bash
xpyd-bench \
  --base-url http://localhost:8000 \
  --model Qwen/Qwen2.5-7B \
  --num-prompts 500 \
  --max-concurrency 32 \
  --input-len 512 \
  --output-len 256 \
  --stream \
  -o results.json
```

### 2. PD Disaggregation (Prefill-Decode Disaggregation) Test

Route through xpyd-proxy to separate prefill / decode nodes:

```bash
# 1) Start prefill node (xpyd-sim or real vLLM)
xpyd-sim --role prefill --port 8100

# 2) Start decode node
xpyd-sim --role decode --port 8200

# 3) Start proxy
xpyd-proxy --prefill http://localhost:8100 --decode http://localhost:8200 --port 8080

# 4) Run benchmark (against proxy)
xpyd-bench \
  --base-url http://localhost:8080 \
  --num-prompts 1000 \
  --request-rate 50 \
  --stream \
  -o pd_results.json
```

### 3. Multi-Model Comparison

Use the built-in multi-model comparison mode:

```bash
# Compare two models
xpyd-bench \
  --base-url http://localhost:8000 \
  --model Qwen/Qwen2.5-7B \
  --num-prompts 200 \
  -o model_a.json

xpyd-bench \
  --base-url http://localhost:8000 \
  --model Qwen/Qwen2.5-14B \
  --num-prompts 200 \
  -o model_b.json

# Compare results
xpyd-bench compare model_a.json model_b.json
```

### 4. Duration Mode

Run for a fixed duration without limiting request count:

```bash
xpyd-bench \
  --base-url http://localhost:8000 \
  --duration 300 \
  --request-rate 20 \
  --stream
```

### 5. Quick Validation with Dummy Server

Start a mock server without a real model:

```bash
# Terminal 1: Start dummy server
xpyd-dummy --port 8000

# Terminal 2: Run benchmark
xpyd-bench --base-url http://localhost:8000 --num-prompts 100
```

## Understanding Results

### Core Metrics

| Metric | Description |
|--------|-------------|
| **TTFT** (Time To First Token) | Time from sending a request to receiving the first token. Reflects prefill stage latency. |
| **TPOT** (Time Per Output Token) | Average time to generate each output token. Reflects decode stage speed. |
| **TPS** (Tokens Per Second) | Tokens generated per second (per-request / aggregate). |
| **Throughput** | Total throughput: requests/sec (req/s) and tokens/sec (tok/s). |
| **Error Rate** | Percentage of failed requests. |

### Percentile Metrics

| Metric | Description |
|--------|-------------|
| **P50** | Median — 50% of requests are below this value |
| **P90** | 90% of requests are below this value |
| **P99** | 99% of requests are below this value; reflects tail latency |
| **Mean** | Arithmetic mean |
| **Std** | Standard deviation; reflects latency stability |

### How to Evaluate Results

- **TTFT < 200ms**: Good prefill performance (7B model, 512 token input)
- **TPOT < 30ms**: Normal decode speed
- **P99/P50 < 3x**: Healthy latency distribution with no severe tail latency
- **Error Rate = 0%**: Stable service
- **Throughput**: Should scale near-linearly with concurrency, plateauing at saturation

### Result File

Output JSON contains:

```json
{
  "config": { ... },
  "results": {
    "total_requests": 1000,
    "successful_requests": 998,
    "failed_requests": 2,
    "total_duration_s": 45.2,
    "requests_per_second": 22.1,
    "tokens_per_second": 2834,
    "ttft_ms": { "mean": 152, "p50": 140, "p90": 210, "p99": 380 },
    "tpot_ms": { "mean": 22, "p50": 20, "p90": 28, "p99": 45 },
    "latency_ms": { "mean": 2950, "p50": 2800, "p90": 3500, "p99": 4200 }
  }
}
```

## Using with xpyd-sim / xpyd-proxy

### Architecture

```
Client (xpyd-bench)
        │
        ▼
   xpyd-proxy (routing layer)
     ┌──┴──┐
     ▼     ▼
  Prefill  Decode
  (xpyd-sim / vLLM)
```

### Full Example

See [`scripts/run_benchmark.sh`](../scripts/run_benchmark.sh) for an all-in-one launch script.

```bash
# Manual steps
pip install xpyd-sim xpyd-proxy xpyd-bench

# Start sim nodes
xpyd-sim --role prefill --port 8100 &
xpyd-sim --role decode  --port 8200 &

# Start proxy
xpyd-proxy \
  --prefill http://localhost:8100 \
  --decode  http://localhost:8200 \
  --port 8080 &

# Wait for services to be ready
sleep 3

# Run benchmark
xpyd-bench \
  --base-url http://localhost:8080 \
  --num-prompts 500 \
  --request-rate 30 \
  --max-concurrency 64 \
  --input-len 256 \
  --output-len 128 \
  --stream \
  -o benchmark_results.json

echo "Results saved to benchmark_results.json"
```

## Advanced Features

- **Checkpoint & Resume** (`--checkpoint`): Resume long-running benchmarks after interruption
- **Benchmark Fingerprint** (`--fingerprint`): Uniquely identify benchmark configurations for easy comparison
- **Configuration Inheritance** (`--extends`): Configuration file inheritance
- **Rolling Window Metrics**: Real-time rolling window statistics
- **Baseline Registry**: Register baseline results for automatic regression comparison
- **Speculative Decoding Metrics**: Metrics related to speculative decoding
- **Prefix Caching Impact**: Analyze prefix caching effectiveness
- **Adaptive Timeout**: Automatically adjust timeout based on observed latency
- **Multimodal Vision Benchmark**: Support for vision model testing
- **SLA Validation**: Define SLA rules and automatically check compliance
- **Distributed Benchmark**: Multi-node coordinated distributed load testing
