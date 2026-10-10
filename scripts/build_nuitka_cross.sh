#!/usr/bin/env bash
# 从 Linux 用 Wine 交叉构建 Windows 版 MTPro (Nuitka AOT)
#
# 原理: Nuitka 4.x 没有 Linux→Windows 的 --target 交叉模式,必须在
# Windows 目标解释器下运行才能产出 PE。这里用 Wine 跑 python-build-standalone
# 的 Windows CPython 3.14(默认 ~/.cross/python,可用 WIN_PY 覆盖),Nuitka
# 的 C 编译器用 winlibs gcc(Nuitka --assume-yes-for-downloads 自动下载,
# 在 Wine 内原生运行)。
#
# 前置(与 build_nuitka.ps1 相同的依赖集合):
#   - wine + ~/.cross/python 下的 Windows CPython 3.14
#   - Windows wheels 已装入该 Python(scripts 依赖 + nuitka 4.2.x)
#   - rust_lib.pyd 已放入其 site-packages(用 maturin --target
#     x86_64-pc-windows-gnu 的产物改名,或 cargo 同样目标构建)
#
# 用法: scripts/build_nuitka_cross.sh
# 环境变量: WIN_PY 覆盖 Windows python.exe 路径

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WIN_PY="${WIN_PY:-$HOME/.cross/python/python.exe}"
OUTPUT_DIR="$ROOT_DIR/outputs"
BUILD_DIR="python/build/nuitka-cross"
DIST_NAME="mtpro-windows-x86_64"

if [[ ! -f "$WIN_PY" ]]; then
  echo "Missing Windows target Python: $WIN_PY" >&2
  echo "Set WIN_PY to a Windows CPython 3.14 python.exe" >&2
  exit 1
fi

# 当前桌面构建组合固定使用 SciPy 1.17.1；升级时需同时复测 Nuitka 和 scipy.signal。
# 环境必须钉 scipy==1.17.1,否则编出来的 exe 启动即崩。
WIN_SCIPY="$(cd "$(dirname "$WIN_PY")" 2>/dev/null; wine "$WIN_PY" -c "import scipy;print(scipy.__version__)" 2>/dev/null | tail -1 | tr -d '\r')"
if [[ "$WIN_SCIPY" != "1.17.1" ]]; then
  echo "Windows target Python scipy 版本为 '$WIN_SCIPY',需要 1.17.1:" >&2
  echo "  wine \"$WIN_PY\" -m pip install 'scipy==1.17.1'" >&2
  exit 1
fi

# 按系统可用内存自适应并行度:桌面不忙时可冲更高并发,swap 边缘自动降低。
AVAIL_MB=$(( $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 4000) ))
if   [ "$AVAIL_MB" -ge 12000 ]; then JOBS=12
elif [ "$AVAIL_MB" -ge 8000 ];  then JOBS=8
elif [ "$AVAIL_MB" -ge 4000 ];  then JOBS=6
else                                 JOBS=4
fi
echo "=== avail ${AVAIL_MB} MB → --jobs=$JOBS"

# RUST_REBUILD=1 时重建 rust_lib(改过 Rust 才需要;纯 Python 改动跳过,
# 直接复用已有 rust_lib.pyd)
if [[ "${RUST_REBUILD:-0}" == "1" ]]; then
  echo "=== rebuilding rust_lib for x86_64-pc-windows-gnu ==="
  (cd rust_lib && cargo build --release --target x86_64-pc-windows-gnu)
  SITE_PY="$(dirname "$WIN_PY")/Lib/site-packages"
  cp rust_lib/target/x86_64-pc-windows-gnu/release/rust_lib.dll "$SITE_PY/rust_lib.pyd"
fi

PKGS=(ui core base processor io_utils utils emtf)
INCLUDE_PACKAGES=()
for p in "${PKGS[@]}"; do
  INCLUDE_PACKAGES+=(--include-package="$p")
done

# 排除开发环境里有、MTPro 不用的包(与 build_nuitka.sh / ps1 一致)
EXCLUDED_UNUSED_PACKAGES=(
  annotated_types astropy astropy_iers_data aurora certifi cftime
  charset_normalizer dns email_validator greenlet idna llvmlite loguru lxml
  mt_io mt_timeseries mt_metadata mth5 numba obspy openpyxl pandas_stubs
  psutil pydantic pydantic_core pyerfa pyproj requests scipy_stubs sqlalchemy
  typing_inspection urllib3 yaml
)
EXCLUDE_FLAGS=()
for pkg in "${EXCLUDED_UNUSED_PACKAGES[@]}"; do
  EXCLUDE_FLAGS+=(--nofollow-import-to="$pkg")
done

echo "=== Wine target Python: $(wine "$WIN_PY" --version 2>/dev/null | tail -1)"

# 关键: 路径一律相对 —— Wine 里的 Windows Python 不认绝对 unix 路径,
# 会当成当前盘符的根路径解析。
# 注: 不删 build 目录,让 scons 增量续编已编译的 .o;全量重来可手动 rm。

PYTHONPATH="python/src" \
NUITKA_EXPERIMENTAL="force-mingw64" \
WINEDEBUG=-all \
wine "$WIN_PY" -m nuitka \
  --standalone \
  --follow-imports \
  --enable-plugin=pyside6 \
  --static-libpython=no \
  "${INCLUDE_PACKAGES[@]}" \
  --include-module=rust_lib \
  --include-package=matio \
  --include-data-dir=python/src/bin=bin \
  --mingw64 \
  --experimental=force-mingw64 \
  --experimental=force-dependencies-pefile \
  --low-memory \
  --jobs="$JOBS" \
  --noinclude-pytest-mode=warning \
  --noinclude-unittest-mode=warning \
  --noinclude-pydoc-mode=warning \
  --noinclude-IPython-mode=warning \
  --noinclude-setuptools-mode=warning \
  "${EXCLUDE_FLAGS[@]}" \
  --assume-yes-for-downloads \
  --output-dir="$BUILD_DIR" \
  python/src/main.py

echo "=== Nuitka done, assembling dist ==="
DIST_SRC="$BUILD_DIR/main.dist"
if [[ ! -f "$DIST_SRC/main.exe" ]]; then
  echo "Expected $DIST_SRC/main.exe not found" >&2
  exit 1
fi

rm -rf "$OUTPUT_DIR/$DIST_NAME"
mkdir -p "$OUTPUT_DIR/$DIST_NAME"
# -l 硬链接装配:800M dist 到 outputs 从"整盘拷贝"变成"瞬时", build 目录
# 本身不被改动,硬链接安全(需要与 build 同分区,实际都在 /home)。
cp -rl "$DIST_SRC"/. "$OUTPUT_DIR/$DIST_NAME/"
mv "$OUTPUT_DIR/$DIST_NAME/main.exe" "$OUTPUT_DIR/$DIST_NAME/MTPro.exe"
# Nuitka 的 --include-data-dir 会滤掉 .exe 数据文件,win64 转换器显式补拷
mkdir -p "$OUTPUT_DIR/$DIST_NAME/bin"
cp -rl python/src/bin/win64/. "$OUTPUT_DIR/$DIST_NAME/bin/" 2>/dev/null || true
rm -rf "$OUTPUT_DIR/$DIST_NAME/bin/linux"   # Linux 版转换器是 Windows 包的死重
cp python/requirements.txt "$OUTPUT_DIR/$DIST_NAME/" 2>/dev/null || true

echo "Output: $OUTPUT_DIR/$DIST_NAME/MTPro.exe"
