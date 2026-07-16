#!/usr/bin/env bash
# 在 Wine Windows Python 下安装 Nuitka
set -euo pipefail
export DISPLAY=
export WINEDLLOVERRIDES="mscoree,mshtml="
WIN_PYTHON="$HOME/windows-python/python.exe"
"$WIN_PYTHON" -m pip install nuitka ordered-set
echo "✓ Nuitka installed"
