#!/usr/bin/env bash
# Stop TEI servers for ORT, Candle, and Python backends.
# This script will stop Docker containers and local processes.

set -euo pipefail

# Container names (must match start_all_backends.sh)
ORT_CONTAINER="${ORT_CONTAINER:-tei-ort}"
CANDLE_CONTAINER="${CANDLE_CONTAINER:-tei-candle}"
PY_CONTAINER="${PY_CONTAINER:-tei-python}"

echo "🛑 Stopping all TEI backend servers..."

# Stop Docker containers
echo "🐳 Stopping Docker containers..."
docker stop "${ORT_CONTAINER}" "${CANDLE_CONTAINER}" "${PY_CONTAINER}" 2>/dev/null || true
docker rm "${ORT_CONTAINER}" "${CANDLE_CONTAINER}" "${PY_CONTAINER}" 2>/dev/null || true

# Also stop any other TEI containers that might be running
if command -v docker >/dev/null 2>&1; then
  # Find and stop any containers with "tei-" prefix
  docker ps -a --format '{{.Names}}' | grep -E '^tei-' | while read -r container; do
    echo "  Stopping container: ${container}"
    docker stop "${container}" 2>/dev/null || true
    docker rm "${container}" 2>/dev/null || true
  done
fi

# Stop by process name (for locally run processes, backwards compatibility)
echo "🔄 Stopping local processes..."
pkill -f "text-embeddings-router" 2>/dev/null || true

# Also attempt to stop the python-text-embeddings-server if lingering
pkill -f "python-text-embeddings-server" 2>/dev/null || true

# Wait for processes to terminate
sleep 2

# Check if any processes are still running
if ps aux | grep -v grep | grep -q text-embeddings-router; then
  echo "⚠️  Some TEI processes are still running. Attempting force kill..."
  pkill -9 -f "text-embeddings-router" 2>/dev/null || true
  sleep 1

  if ps aux | grep -v grep | grep -q text-embeddings-router; then
    echo "❌ Some TEI processes could not be stopped:"
    ps aux | grep text-embeddings-router | grep -v grep
  else
    echo "✅ All TEI backends stopped."
  fi
else
  echo "✅ All TEI backends stopped."
fi

