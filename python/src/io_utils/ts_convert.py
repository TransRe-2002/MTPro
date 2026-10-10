"""Runtime-free conversion between MATLAB TS objects and plain tsStruct.

find_converter is retained for legacy callers; conversion never invokes it.
"""

from __future__ import annotations

import os
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
    return _convert("ts2struct", src_path, dst_path)


def convert_to_ts_object(
    src_path: str | Path,
    dst_path: str | Path | None = None,
) -> Path:
    """Convert a plain tsStruct .mat file to a serialized TS-object file.

    When *dst_path* is ``None`` a temporary file is created; the caller then
    owns its cleanup.  Returns the output path.
    """
    return _convert("struct2ts", src_path, dst_path)


def _convert(
    mode: str,
    src_path: str | Path,
    dst_path: str | Path | None,
) -> Path:
    """Convert locally; no external process or MATLAB library is required."""
    from io_utils.ts_object_io import read_ts_object, read_ts_struct, write_ts_object, write_ts_struct

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
        if mode == 'ts2struct':
            write_ts_struct(read_ts_object(src), dst)
        else:
            write_ts_object(read_ts_struct(src), dst)
    except Exception:
        if dst_path is None:
            dst.unlink(missing_ok=True)
        raise
    return dst


def main() -> None:
    """CLI: python -m io_utils.ts_convert [ts2struct|struct2ts] input.mat output.mat."""
    import argparse
    from io_utils.ts_object_io import read_ts_struct

    parser = argparse.ArgumentParser(description='Convert MATLAB TS objects without MATLAB Runtime')
    parser.add_argument('files', nargs='+', metavar='ARG')
    args = parser.parse_args()
    if len(args.files) == 2:
        source, destination = args.files
        try:
            read_ts_struct(source)
        except ValueError:
            mode = 'ts2struct'
        else:
            mode = 'struct2ts'
    elif len(args.files) == 3:
        mode, source, destination = args.files
        aliases = {'ts_to_struct': 'ts2struct', 'struct_to_ts': 'struct2ts'}
        mode = aliases.get(mode, mode)
        if mode not in ('ts2struct', 'struct2ts'):
            parser.error('mode must be ts2struct or struct2ts')
    else:
        parser.error('expected input output, or mode input output')
    _convert(mode, source, destination)
    print(f'Done: {source} -> {destination}')


if __name__ == '__main__':
    main()
