"""Platform-adaptive binary resolution for EMTF Fortran executables.

Resolves paths to ``dnff`` and ``tranmt`` (plus future binaries) based
on the runtime platform.  Binaries are bundled in ``python/src/bin``.
"""

import os
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent
_BIN_DIR = _SRC_DIR / "bin"
_EMTF_BIN_DIR = _SRC_DIR / "EMTF" / "bin"

_PLATFORM_MAP = {
    "linux": "linux",
    "linux2": "linux",
    "win32": "win64",
    "cygwin": "win64",
}


def get_platform() -> str:
    """Return the platform tag (``"linux"`` or ``"win64"``) for binary lookup."""
    plat = sys.platform
    tag = _PLATFORM_MAP.get(plat)
    if tag is None:
        raise RuntimeError(f"Unsupported platform: {plat}.  Only Linux and Windows are supported.")
    return tag


def resolve_binary(name: str) -> Path:
    """Resolve the path to a Fortran binary.

    Lookup order:
    1. ``EMTF_BIN_DIR`` environment variable (if set, takes priority).
    2. ``bin/<platform>/`` in the project root (bundled binaries).
    3. ``EMTF/bin/<platform>/`` (legacy upstream binaries).

    On Windows, ``.exe`` is appended to *name* automatically.

    Raises :class:`FileNotFoundError` if no candidate exists.
    """
    platform = get_platform()
    if platform == "win64":
        name = f"{name}.exe"

    # 1) Environment override
    if env_dir := os.environ.get("EMTF_BIN_DIR"):
        candidate = Path(env_dir) / name
        if candidate.exists():
            return candidate

    # 2) Bundled
    candidate = _BIN_DIR / platform / name
    if candidate.exists():
        return candidate

    # 3) Legacy upstream
    candidate = _EMTF_BIN_DIR / platform / name
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Cannot find '{name}' for platform '{platform}'. "
        f"Set EMTF_BIN_DIR or ensure binaries are in {_BIN_DIR / platform}/."
    )
