#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY_DIR="$ROOT_DIR/python"
SRC_MAIN="$PY_DIR/src/main.py"
BUILD_DIR="$PY_DIR/build/nuitka"

mkdir -p "$BUILD_DIR"

python -m nuitka \
  --standalone \
  --follow-imports \
  --enable-plugin=pyside6 \
  --include-package=ui \
  --include-package=core \
  --include-package=base \
  --include-package=processor \
  --include-package=io_utils \
  --include-package=utils \
  --output-dir="$BUILD_DIR" \
  --assume-yes-for-downloads \
  --show-progress \
  --show-memory \
  "$SRC_MAIN"
