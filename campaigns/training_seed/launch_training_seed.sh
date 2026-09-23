#!/usr/bin/env bash
# Verify a frozen representation, accept the production path, and train a new seed.
set -euo pipefail
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$code_root"
export PYTHONPATH="$code_root"
: "${PYTHON:?Set the production interpreter}"
: "${CONFIG:?Set the selected training-seed configuration}"
: "${RUN_DIR:?Set a fresh training result directory}"
: "${EXPERIMENT_DIR:?Set the independent experiment root}"
: "${LAUNCH_DIR:?Set the launcher evidence directory}"
: "${ACCEPTANCE_DIR:?Set a fresh acceptance directory}"
mkdir -p "$LAUNCH_DIR"
exec "$PYTHON" "$code_root/graph_dit/portable_lock.py" \
    --lock "$LAUNCH_DIR/launcher.lock" -- \
    bash "$code_root/scripts/launch_training_seed_body.sh"
