"""Execute Fortran binaries in isolated temporary workspaces.

EMTF Fortran programs read/write files on disk (not stdin/stdout),
so each invocation creates a temporary directory, writes configuration
and input files, runs the binary, and returns the result — providing
a "virtual filesystem" abstraction to the caller.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Sequence


class Workspace:
    """An isolated directory for running an EMTF Fortran binary.

    The workspace acts as a "virtual filesystem": config files and
    input data are staged inside, the binary runs with the workspace
    as the current directory, and output files land there.

    By default the directory is removed automatically when the
    :class:`Workspace` object is garbage-collected.  Pass
    ``cleanup=False`` to persist files for inspection.
    """

    def __init__(self, cleanup: bool = True, prefix: str = "emtf_"):
        self.cleanup = cleanup
        if cleanup:
            self._tmpdir = tempfile.TemporaryDirectory(prefix=prefix)
            self.path = Path(self._tmpdir.name)
        else:
            self.path = Path(tempfile.mkdtemp(prefix=prefix))

    def write(self, relpath: str, content: str) -> Path:
        """Write a text file into the workspace, creating parent directories."""
        dest = self.path / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)
        return dest

    def write_binary(self, relpath: str, source: Path) -> Path:
        """Copy a binary file into the workspace."""
        dest = self.path / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        return dest

    def resolve(self, relpath: str) -> Path:
        """Return the full path for *relpath* inside the workspace."""
        return self.path / relpath

    def cleanup_now(self):
        """Explicitly remove the workspace (no-op if already cleaned up)."""
        if self.cleanup:
            self._tmpdir.cleanup()


def run(
    binary: Path,
    stdin: Optional[str] = None,
    workspace: Optional[Workspace] = None,
    timeout: int = 300,
    args: Sequence[str] = (),
) -> subprocess.CompletedProcess:
    """Run an EMTF binary inside a :class:`Workspace`.

    Parameters
    ----------
    binary:
        Full path to the Fortran executable (from :func:`resolve_binary`).
    args:
        Additional command-line flags passed to the executable.
    stdin:
        Text to feed to the process's stdin.  EMTF programs are interactive
        — they prompt for filenames and options.
    workspace:
        A pre-configured workspace.  If ``None``, a fresh temporary one
        is created and cleaned up after the run.
    timeout:
        Maximum runtime in seconds (default 5 minutes).

    Returns
    -------
    :class:`subprocess.CompletedProcess` with an extra ``workspace``
    attribute pointing to the :class:`Workspace` used.
    """
    ws = workspace if workspace is not None else Workspace()
    try:
        result = subprocess.run(
            [str(binary), *args],
            cwd=str(ws.path),
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        result.workspace = ws  # type: ignore[attr-defined]
        return result
    finally:
        if workspace is None:
            ws.cleanup_now()
