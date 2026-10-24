#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

export STREAMLIT_CONFIG_DIR="$DIR/.streamlit"
export YOLO_CONFIG_DIR="$DIR/data/.ultralytics"

# Select available port (default 8501, fallback 8502)
PORT=8501
if lsof -Pi :8501 -sTCP:LISTEN -t >/dev/null 2>&1; then
    PORT=8502
fi

echo "🚀 Launching MIRAGE Vision Workbench on port $PORT..."
streamlit run app.py --server.port "$PORT"
