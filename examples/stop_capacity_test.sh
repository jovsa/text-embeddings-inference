#!/usr/bin/env bash
# Stop TEI servers started for channel capacity testing

LOG_DIR="${LOG_DIR:-/tmp/tei-logs}"

echo "🛑 Stopping channel capacity test servers..."

for capacity in 1 4 8; do
    pid_file="${LOG_DIR}/capacity-${capacity}.pid"
    if [[ -f "${pid_file}" ]]; then
        pid=$(cat "${pid_file}")
        if kill -0 "${pid}" 2>/dev/null; then
            echo "  Stopping server with capacity=${capacity} (PID: ${pid})..."
            kill "${pid}" 2>/dev/null || true
            sleep 1
            # Force kill if still running
            if kill -0 "${pid}" 2>/dev/null; then
                kill -9 "${pid}" 2>/dev/null || true
            fi
            echo "  ✅ Stopped"
        fi
        rm -f "${pid_file}"
    fi
done

echo "✅ Done!"

