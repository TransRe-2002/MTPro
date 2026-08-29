#!/usr/bin/env bash
set -euo pipefail

# 用法: build_matlab.sh
#
# 用 MATLAB Compiler (mcc) 将 matlab/ 下的 TS 格式转换工具编译为独立可执行
# 文件 mtpro_ts_convert，输出到 matlab/build/：
#   mtpro_ts_convert.bin   — mcc 生成的原生可执行文件
#   mtpro_ts_convert       — 启动脚本（自动定位 MATLAB Runtime）
#   run_mtpro_ts_convert.sh — mcc 生成的 Runtime 引导脚本
#
# 运行编译产物需要 MATLAB Runtime（本机装有完整 MATLAB 时可直接复用）。
# Windows 版本需在 Windows 上用 MATLAB Compiler 另行构建。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MATLAB_DIR="$ROOT_DIR/matlab"
BUILD_DIR="$MATLAB_DIR/build"

MCC="${MCC:-mcc}"
if ! command -v "$MCC" >/dev/null 2>&1; then
  echo "Missing MATLAB Compiler (mcc). Install MATLAB Compiler first." >&2
  exit 1
fi

# 从 matlab 可执行文件的符号链接推断 MATLAB 安装根目录（用作默认 MCR 路径）
MATLAB_BIN="$(command -v matlab || true)"
if [[ -z "$MATLAB_BIN" ]]; then
  echo "Cannot locate matlab on PATH; needed to infer the default Runtime root." >&2
  exit 1
fi
MATLAB_ROOT_GUESS="$(dirname "$(dirname "$(readlink -f "$MATLAB_BIN")")")"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

echo "=== Compiling mtpro_ts_convert with mcc ==="
"$MCC" \
  -m "$MATLAB_DIR/main.m" \
  -o mtpro_ts_convert \
  -d "$BUILD_DIR" \
  -a "$MATLAB_DIR/TS.m" \
  -a "$MATLAB_DIR/struct_to_ts.m" \
  -a "$MATLAB_DIR/ts_to_struct.m"

# 原生二进制改名为 .bin，腾出 mtpro_ts_convert 名字给启动脚本
mv -f "$BUILD_DIR/mtpro_ts_convert" "$BUILD_DIR/mtpro_ts_convert.bin"

# 注意：不能复用 mcc 生成的 run_mtpro_ts_convert.sh —— 它硬编码执行
# 名为 mtpro_ts_convert 的文件，而该名字已被本启动脚本占用，会无限递归。
# 这里直接照搬 run 脚本的环境变量逻辑，然后执行 .bin 本体。
cat > "$BUILD_DIR/mtpro_ts_convert" <<EOF
#!/bin/sh
# mtpro_ts_convert 启动脚本：为编译产物定位 MATLAB Runtime。
# 默认使用构建时的 MATLAB 安装路径，可用环境变量 MTPRO_MCR_ROOT 覆盖。
SELF_DIR="\$(dirname "\$0")"
MCR_ROOT="\${MTPRO_MCR_ROOT:-$MATLAB_ROOT_GUESS}"
if [ ! -d "\$MCR_ROOT" ]; then
  echo "MATLAB Runtime not found at \$MCR_ROOT" >&2
  echo "Set MTPRO_MCR_ROOT to a valid MATLAB/MCR install root." >&2
  exit 1
fi
LD_LIBRARY_PATH=".:\$MCR_ROOT/runtime/glnxa64:\$MCR_ROOT/bin/glnxa64:\$MCR_ROOT/sys/os/glnxa64:\$MCR_ROOT/sys/opengl/lib/glnxa64\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH
exec "\$SELF_DIR/mtpro_ts_convert.bin" "\$@"
EOF
chmod +x "$BUILD_DIR/mtpro_ts_convert"

echo "Output: $BUILD_DIR/mtpro_ts_convert"
echo "MCR root baked in: $MATLAB_ROOT_GUESS (override with MTPRO_MCR_ROOT)"
