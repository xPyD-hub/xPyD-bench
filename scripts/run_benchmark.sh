#!/usr/bin/env bash
# run_benchmark.sh — One-click PD-disaggregated benchmark
# Starts xpyd-sim (prefill + decode) + xpyd-proxy, runs benchmark, cleans up.
set -euo pipefail

# --- Configuration (override via environment) ---
PREFILL_PORT="${PREFILL_PORT:-8100}"
DECODE_PORT="${DECODE_PORT:-8200}"
PROXY_PORT="${PROXY_PORT:-8080}"
NUM_PROMPTS="${NUM_PROMPTS:-500}"
REQUEST_RATE="${REQUEST_RATE:-30}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-64}"
INPUT_LEN="${INPUT_LEN:-256}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
OUTPUT_FILE="${OUTPUT_FILE:-benchmark_results.json}"

PIDS=()

cleanup() {
    echo ""
    echo "🧹 Cleaning up background processes..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    echo "✅ All processes stopped."
}
trap cleanup EXIT

# --- Dependency checks ---
echo "🔍 Checking dependencies..."

for cmd in xpyd-sim xpyd-proxy xpyd-bench; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "❌ $cmd not found. Install it first:"
        case "$cmd" in
            xpyd-sim)   echo "   pip install xpyd-sim" ;;
            xpyd-proxy) echo "   pip install xpyd-proxy" ;;
            xpyd-bench) echo "   pip install xpyd-bench" ;;
        esac
        exit 1
    fi
done

echo "✅ All dependencies found."
echo ""

# --- Start services ---
echo "🚀 Starting prefill simulator on port $PREFILL_PORT..."
xpyd-sim --role prefill --port "$PREFILL_PORT" &
PIDS+=($!)

echo "🚀 Starting decode simulator on port $DECODE_PORT..."
xpyd-sim --role decode --port "$DECODE_PORT" &
PIDS+=($!)

echo "🚀 Starting proxy on port $PROXY_PORT..."
xpyd-proxy \
    --prefill "http://localhost:$PREFILL_PORT" \
    --decode "http://localhost:$DECODE_PORT" \
    --port "$PROXY_PORT" &
PIDS+=($!)

echo "⏳ Waiting for services to be ready..."
sleep 3

# Health check
for port in "$PREFILL_PORT" "$DECODE_PORT" "$PROXY_PORT"; do
    if ! curl -sf "http://localhost:$port/health" >/dev/null 2>&1; then
        echo "⚠️  Port $port not responding to /health, continuing anyway..."
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Running benchmark: $NUM_PROMPTS prompts @ ${REQUEST_RATE} req/s"
echo "  Target: http://localhost:$PROXY_PORT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# --- Run benchmark ---
xpyd-bench \
    --base-url "http://localhost:$PROXY_PORT" \
    --num-prompts "$NUM_PROMPTS" \
    --request-rate "$REQUEST_RATE" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --input-len "$INPUT_LEN" \
    --output-len "$OUTPUT_LEN" \
    --stream \
    -o "$OUTPUT_FILE"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ Benchmark complete!"
echo "  📄 Results: $OUTPUT_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# cleanup runs via trap EXIT
