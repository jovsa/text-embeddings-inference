#!/usr/bin/env bash
# Stop TEI servers for ORT, Candle, and Python backends.
# This script will stop ALL text-embeddings-router processes, regardless of port.

set -euo pipefail

echo "🛑 Stopping all TEI backend servers..."

# Stop by process name (more reliable than port matching)
pkill -f "text-embeddings-router" 2>/dev/null || true

# Also attempt to stop the python-text-embeddings-server if lingering
pkill -f "python-text-embeddings-server" 2>/dev/null || true

# Wait for processes to terminate
sleep 2

# Check if any are still running
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

