#!/usr/bin/env bash
set -e

# Reddit Data Science Analysis Container Entrypoint
echo "=================================================="
echo "  Reddit Data Science Container Workspace"
echo "=================================================="

# Export Matplotlib and NLTK cache directories to avoid permission warnings
export MPLCONFIGDIR="/tmp/matplotlib"
export NLTK_DATA="/workspace/data/nltk_data"

MODE="${1:-pipeline}"

case "$MODE" in
  pipeline)
    echo "Running headless automated pipeline..."
    shift 1 2>/dev/null || true
    exec python main.py "$@"
    ;;

  jupyter)
    echo "Starting JupyterLab server on port 8888..."
    shift 1 2>/dev/null || true
    exec jupyter lab \
      --ip=0.0.0.0 \
      --port=8888 \
      --no-browser \
      --allow-root \
      --ServerApp.token='' \
      --ServerApp.password='' \
      --ServerApp.allow_origin='*' \
      "$@"
    ;;

  test)
    echo "Running test suite..."
    shift 1 2>/dev/null || true
    exec python -m unittest discover tests -v "$@"
    ;;

  bash|sh)
    exec /bin/bash "$@"
    ;;

  *)
    exec "$@"
    ;;
esac

