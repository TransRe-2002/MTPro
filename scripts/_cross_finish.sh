#!/usr/bin/env bash
# 内部脚本: 等孤儿化的 Nuitka 交叉编译进程结束后,装配 outputs dist。
# 由 build 包装器被杀后的场景手动补偿使用。用法: bash scripts/_cross_finish.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="$ROOT_DIR/outputs"
BUILD_DIR="python/build/nuitka-cross"
DIST_NAME="mtpro-windows-x86_64"

echo "=== waiting for orphaned nuitka build to finish ==="
# 轮询直到没有进程引用该构建目录(main.dist 的 main.exe 也出现即认为完成)
for i in $(seq 1 600); do  # max ~5h
  if [[ -f "$BUILD_DIR/main.dist/main.exe" ]]; then
    if ! pgrep -f "build/nuitka-cross|main.build" >/dev/null 2>&1; then
      echo "build finished (main.exe present, no build procs)"
      break
    fi
  fi
  if ! pgrep -f "build/nuitka-cross|main.build" >/dev/null 2>&1; then
    echo "no build procs running (finished or died)"
    break
  fi
  sleep 30
done

if [[ ! -f "$BUILD_DIR/main.dist/main.exe" ]]; then
  echo "FATAL: main.exe not produced. state:"
  ls -la "$BUILD_DIR/main.dist" 2>/dev/null || true
  exit 1
fi

echo "=== assembling dist ==="
rm -rf "$OUTPUT_DIR/$DIST_NAME"
mkdir -p "$OUTPUT_DIR/$DIST_NAME"
cp -r "$BUILD_DIR/main.dist"/. "$OUTPUT_DIR/$DIST_NAME/"
mv "$OUTPUT_DIR/$DIST_NAME/main.exe" "$OUTPUT_DIR/$DIST_NAME/MTPro.exe"
cp python/requirements.txt "$OUTPUT_DIR/$DIST_NAME/" 2>/dev/null || true

echo "OUTPUT_READY: $OUTPUT_DIR/$DIST_NAME/MTPro.exe"