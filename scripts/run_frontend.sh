#!/usr/bin/env bash
# scripts/run_frontend.sh - Start the Streamlit dashboard locally
# Usage: bash scripts/run_frontend.sh
set -e

cd "$(dirname "$0")/.."

STREAMLIT=/Users/leonida/Library/Python/3.9/bin/streamlit
export MPLCONFIGDIR=/tmp/mpl
export API_URL=${API_URL:-http://localhost:8000}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " 🗺️   Kenya Land Readiness Atlas - Streamlit"
echo "     http://localhost:8501"
echo "     API: $API_URL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

$STREAMLIT run frontend/app.py \
    --server.port 8501 \
    --server.headless true

