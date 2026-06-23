## Nuitka Build

This project does not yet have a full production packaging pipeline. The script below is a minimal
starting point for building the PySide6 desktop app as a standalone application with Nuitka.

### Prerequisites

- Python environment with the project dependencies installed
- `nuitka`
- A working C/C++ toolchain on the target platform
- The Rust extension built and importable if the app depends on it

### Build

From the repository root:

```bash
bash python/scripts/build_nuitka.sh
```

The output will be placed under:

```text
python/build/nuitka/
```

### Notes

- The current script uses `--standalone` and the `pyside6` plugin.
- It includes the main application packages explicitly to reduce missed imports.
- This is only a starter pipeline. Platform-specific polishing such as icons, data-file bundling,
  onefile mode, and Windows metadata still needs to be added later.
