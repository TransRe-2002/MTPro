#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/python/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing project Python: $PYTHON_BIN" >&2
  exit 1
fi

# Avoid leaking an activated Conda environment or user-site packages into MTPro.
unset CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT_DIR/python/src"

exec "$PYTHON_BIN" "$ROOT_DIR/python/src/main.py" "$@"
