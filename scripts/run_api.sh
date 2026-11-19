#!/usr/bin/env bash
# scripts/run_api.sh — Start the FastAPI server locally
# Usage: bash scripts/run_api.sh
set -e

cd "$(dirname "$0")/.."

UVICORN=/Users/leonida/Library/Python/3.9/bin/uvicorn
export MPLCONFIGDIR=/tmp/mpl

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " 🚀  Kenya Land Readiness Atlas — FastAPI"
echo "     http://localhost:8000"
echo "     http://localhost:8000/docs  (Swagger UI)"
echo "     http://localhost:8000/health"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

$UVICORN api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --reload-dir api
