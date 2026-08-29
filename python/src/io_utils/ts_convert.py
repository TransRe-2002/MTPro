"""External converter for MATLAB TS-object .mat files.

MATLAB ``TS`` class objects serialized into .mat files cannot be parsed by
scipy.io (they come back as opaque ``MatlabOpaque`` placeholders and the
variable name is lost).  The standalone executable ``mtpro_ts_convert``,
compiled from the MATLAB sources in ``matlab/`` via
``scripts/build_matlab.sh``, normalizes such files into the plain
``tsStruct`` form that scipy.io reads.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent   # python/src
_REPO_ROOT = _SRC_DIR.parent.parent
_BUNDLED_BIN_DIR = _SRC_DIR / "bin"

_PLATFORM_MAP = {
    "linux": "linux",
    "linux2": "linux",
    "win32": "win64",
    "cygwin": "win64",
}

_CONVERT_TIMEOUT_SEC = 300


def _platform_tag() -> str:
    tag = _PLATFORM_MAP.get(sys.platform)
    if tag is None:
        raise RuntimeError(
            f"Unsupported platform: {sys.platform}. Only Linux and Windows are supported."
        )
    return tag


def find_converter() -> Path:
    """Locate the ``mtpro_ts_convert`` executable.

    Lookup order (mirrors ``emtf/_binaries.py:resolve_binary``):

    1. ``MTPRO_TS_CONVERT`` environment variable (full path to the executable)
    2. ``<repo>/matlab/build/`` (local build from ``scripts/build_matlab.sh``)
    3. ``python/src/bin/<platform>/`` (bundled with packaged builds)

    Raises :class:`FileNotFoundError` with guidance if no candidate exists.
    """
    name = "mtpro_ts_convert"
    if _platform_tag() == "win64":
        name += ".exe"

    candidates = []
    if env_path := os.environ.get("MTPRO_TS_CONVERT"):
        candidates.append(Path(env_path))
    candidates.append(_REPO_ROOT / "matlab" / "build" / name)
    candidates.append(_BUNDLED_BIN_DIR / _platform_tag() / name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(
        "Cannot find the 'mtpro_ts_convert' executable needed to read "
        f"MATLAB TS-object .mat files. Searched:\n{searched}\n"
        "Build it with scripts/build_matlab.sh or set MTPRO_TS_CONVERT."
    )


def convert_to_struct(
    src_path: str | Path,
    dst_path: str | Path | None = None,
) -> Path:
    """Convert a TS-object .mat file to plain tsStruct form.

    When *dst_path* is ``None`` a temporary file is created; the caller then
    owns its cleanup.  Returns the output path.
    """
    return _run_converter("ts2struct", src_path, dst_path)


def convert_to_ts_object(
    src_path: str | Path,
    dst_path: str | Path | None = None,
) -> Path:
    """Convert a plain tsStruct .mat file to a serialized TS-object file.

    When *dst_path* is ``None`` a temporary file is created; the caller then
    owns its cleanup.  Returns the output path.
    """
    return _run_converter("struct2ts", src_path, dst_path)


def _run_converter(
    mode: str,
    src_path: str | Path,
    dst_path: str | Path | None,
) -> Path:
    """Run ``mtpro_ts_convert <mode> <src> <dst>`` in a subprocess.

    Raises
    ------
    FileNotFoundError
        Input missing or converter executable not found.
    RuntimeError
        Converter failed or timed out (stderr/stdout included in the message).
    """
    converter = find_converter()
    src = Path(src_path)
    if not src.is_file():
        raise FileNotFoundError(f"Input file does not exist: {src}")

    if dst_path is None:
        fd, tmp_name = tempfile.mkstemp(suffix=".mat", prefix="mtpro_ts_")
        os.close(fd)
        dst = Path(tmp_name)
    else:
        dst = Path(dst_path)

    try:
        result = subprocess.run(
            [str(converter), mode, str(src), str(dst)],
            capture_output=True,
            text=True,
            timeout=_CONVERT_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"mtpro_ts_convert timed out after {_CONVERT_TIMEOUT_SEC}s on {src}"
        ) from exc

    if result.returncode != 0 or not dst.is_file():
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"mtpro_ts_convert failed for {src} (exit {result.returncode}): {detail}"
        )
    return dst
