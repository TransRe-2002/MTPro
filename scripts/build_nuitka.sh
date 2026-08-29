#!/usr/bin/env bash
set -euo pipefail

# 用法: build_nuitka.sh [linux|windows]
#   linux   — 原生 Linux 构建 (默认)
#   windows — 从 Linux 交叉编译 Windows .exe (需要 MinGW-w64 + Wine)
#
# 注意: Windows 分发现由 scripts/build_nuitka.ps1(本地)或 CI 的
# build-windows job(原生 Windows runner)负责,交叉分支仅供实验。
# 交叉编译必须先有 Windows 目标 CPython(WINPYTHON / nuitka
# --windows-download-python),否则 main.dist 只会是空目录。

PLATFORM="${1:-linux}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_DIR="$ROOT_DIR/python"
PYTHON_BIN="$PY_DIR/.venv/bin/python"
SRC_MAIN="$PY_DIR/src/main.py"
OUTPUT_DIR="$ROOT_DIR/outputs"

mkdir -p "$OUTPUT_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing project Python: $PYTHON_BIN" >&2
  echo "Create/install the project venv before building." >&2
  exit 1
fi

# Packages observed in local development environments but not used by MTPro.
# Keeping these explicit prevents accidental dynamic inclusion from a dirty venv.
EXCLUDED_UNUSED_PACKAGES=(
  annotated_types
  astropy
  astropy_iers_data
  aurora
  certifi
  cftime
  charset_normalizer
  dns
  email_validator
  greenlet
  idna
  llvmlite
  loguru
  lxml
  mt_io
  mt_timeseries
  mt_metadata
  mth5
  numba
  obspy
  openpyxl
  pandas_stubs
  psutil
  pydantic
  pydantic_core
  pyerfa
  pyproj
  requests
  scipy_stubs
  sqlalchemy
  typing_inspection
  urllib3
  yaml
)

EXCLUDE_FLAGS=()
for package in "${EXCLUDED_UNUSED_PACKAGES[@]}"; do
  EXCLUDE_FLAGS+=(--nofollow-import-to="$package")
done

# 公共 Nuitka 参数
COMMON_FLAGS=(
  --standalone
  --follow-imports
  --enable-plugin=pyside6
  --static-libpython=no
  --include-package=ui
  --include-package=core
  --include-package=base
  --include-package=processor
  --include-package=io_utils
  --include-package=utils
  --include-package=emtf
  --include-data-dir="$PY_DIR/src/bin=bin"
  --include-module=rust_lib
  --remove-output
  --lto=yes
  --noinclude-pytest-mode=nofollow
  --noinclude-unittest-mode=nofollow
  --noinclude-pydoc-mode=nofollow
  --noinclude-IPython-mode=nofollow
  --noinclude-setuptools-mode=nofollow
  "${EXCLUDE_FLAGS[@]}"
  --assume-yes-for-downloads
  --show-progress
)

case "$PLATFORM" in
  windows)
    BUILD_DIR="$PY_DIR/build/nuitka-windows"
    DIST_NAME="mtpro-windows-x86_64"
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"

    echo "=== Cross-compiling for Windows ==="
    "$PYTHON_BIN" -m nuitka \
      "${COMMON_FLAGS[@]}" \
      --mingw64 \
      --output-dir="$BUILD_DIR" \
      "$SRC_MAIN"

    # 移至 outputs
    rm -rf "$OUTPUT_DIR/$DIST_NAME"
    mv "$BUILD_DIR/main.dist" "$OUTPUT_DIR/$DIST_NAME"
    rm -rf "$BUILD_DIR"
    echo "Output: $OUTPUT_DIR/$DIST_NAME/main.exe"
    ;;

  linux|*)
    BUILD_DIR="$PY_DIR/build/nuitka"
    DIST_NAME="mtpro-linux-x86_64"
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"

    echo "=== Building for Linux ==="
    "$PYTHON_BIN" -m nuitka \
      "${COMMON_FLAGS[@]}" \
      --output-dir="$BUILD_DIR" \
      "$SRC_MAIN"

    # 移至 outputs
    rm -rf "$OUTPUT_DIR/$DIST_NAME"
    mv "$BUILD_DIR/main.dist" "$OUTPUT_DIR/$DIST_NAME"
    rm -rf "$BUILD_DIR"
    echo "Output: $OUTPUT_DIR/$DIST_NAME/main.bin"
    ;;
esac
