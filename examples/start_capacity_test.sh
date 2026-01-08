#!/usr/bin/env bash
# Start multiple TEI servers with different --batch-channel-capacity values
# for testing channel capacity impact

set -euo pipefail

MODEL_ID="${MODEL_ID:-BAAI/bge-small-en-v1.5}"
LOG_DIR="${LOG_DIR:-/tmp/tei-logs}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Use ORT backend by default (can be changed)
BIN="${BIN:-/tmp/tei-ort/bin/text-embeddings-router}"

# Ports for different capacity servers
PORT1="${PORT1:-8084}"  # Capacity 1
PORT2="${PORT2:-8085}"  # Capacity 4
PORT3="${PORT3:-8086}"  # Capacity 8

mkdir -p "${LOG_DIR}"

# Check if binary exists
if [[ ! -x "${BIN}" ]]; then
    echo "❌ Error: Binary not found at ${BIN}"
    echo "   Please set BIN environment variable or ensure the binary exists"
    echo "   Example: BIN=/tmp/tei-ort/bin/text-embeddings-router $0"
    exit 1
fi

echo "=" * 80
echo "Starting TEI servers with different channel capacities"
echo "=" * 80
echo ""
echo "📦 Model: ${MODEL_ID}"
echo "🔧 Binary: ${BIN}"
echo ""
echo "Server Configuration:"
echo "  Port ${PORT1}: --batch-channel-capacity 1 (Low Latency)"
echo "  Port ${PORT2}: --batch-channel-capacity 4 (Balanced)"
echo "  Port ${PORT3}: --batch-channel-capacity 8 (High Throughput)"
echo ""

# Function to start a server
start_server() {
    local port=$1
    local capacity=$2
    local log_file="${LOG_DIR}/capacity-${capacity}.log"
    local pid_file="${LOG_DIR}/capacity-${capacity}.pid"

    if [[ -f "${pid_file}" ]]; then
        local old_pid=$(cat "${pid_file}")
        if kill -0 "${old_pid}" 2>/dev/null; then
            echo "⚠️  Server on port ${port} (capacity ${capacity}) already running (PID: ${old_pid})"
            return
        fi
    fi

    echo "🚀 Starting server on port ${port} with capacity=${capacity}..."

    "${BIN}" \
        --model-id "${MODEL_ID}" \
        --port "${port}" \
        --batch-channel-capacity "${capacity}" \
        > "${log_file}" 2>&1 &

    local pid=$!
    echo "${pid}" > "${pid_file}"
    echo "   ✅ Started (PID: ${pid}, Log: ${log_file})"

    # Wait a bit for server to start
    sleep 2

    # Check if it's healthy
    if curl -s "http://127.0.0.1:${port}/health" > /dev/null 2>&1; then
        echo "   ✅ Health check passed"
    else
        echo "   ⚠️  Health check failed (server may still be starting)"
    fi
}

# Start all three servers
start_server "${PORT1}" 1
start_server "${PORT2}" 4
start_server "${PORT3}" 8

echo ""
echo "=" * 80
echo "✅ All servers started!"
echo "=" * 80
echo ""
echo "Server URLs:"
echo "  Capacity 1: http://127.0.0.1:${PORT1}"
echo "  Capacity 4: http://127.0.0.1:${PORT2}"
echo "  Capacity 8: http://127.0.0.1:${PORT3}"
echo ""
echo "To run the benchmark:"
echo "  python examples/benchmark_batching.py \\"
echo "    --test-channel-capacity \\"
echo "    --capacity-urls http://127.0.0.1:${PORT1} http://127.0.0.1:${PORT2} http://127.0.0.1:${PORT3}"
echo ""
echo "To stop servers:"
echo "  bash examples/stop_capacity_test.sh"
echo ""

