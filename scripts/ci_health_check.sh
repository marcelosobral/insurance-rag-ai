#!/usr/bin/env bash
set -euo pipefail

# Local CI-style health check for the FastAPI app.
# Usage: scripts/ci_health_check.sh

PORT=8000
HOST=0.0.0.0

uvicorn app.main:app --host "$HOST" --port "$PORT" &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for i in {1..10}; do
  if curl -fsS "http://localhost:${PORT}/" >/dev/null; then
    echo "Health check passed."
    exit 0
  fi
  sleep 1
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "Server exited early." >&2
    exit 1
  fi
done

echo "Health check failed after retries." >&2
exit 1
