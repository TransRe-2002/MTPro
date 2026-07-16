# AGENTS.md

Guidance for agents working in this repo. MTPro = PySide6 desktop app for magnetotelluric time-series processing, plus a small PyO3 Rust extension.

## Environment (get this wrong and nothing runs)

- **Use the project venv at `python/.venv`** (Anaconda Python 3.13) for every Python/pip/pytest command. Use `python/.venv/bin/python` / `bin/pip` / `bin/pytest` directly, or `source python/.venv/bin/activate` (does not persist across separate bash calls).
- **`PYTHONPATH=python/src` is required** to run the app or tests — sources live in `python/src/` and are imported as top-level packages (`from ui.mainwindow import MainWindow`).
- No linter/typecheck/codegen is configured. There is no formatter; match surrounding style.

## Commands

```bash
# Install deps
python/.venv/bin/pip install -r python/requirements.txt

# Launch app
PYTHONPATH=python/src python/.venv/bin/python python/src/main.py

# Full test suite
PYTHONPATH=python/src python/.venv/bin/pytest python/tests

# Single test file / single test
PYTHONPATH=python/src python/.venv/bin/pytest python/tests/test_save_mat.py
PYTHONPATH=python/src python/.venv/bin/pytest python/tests/test_save_mat.py::test_name

# Synthetic hotspot profiling for Python/Rust migration candidates
PYTHONPATH=python/src python/.venv/bin/python python/scripts/profile_hotspots.py

# Rust extension (release wheel, then install) — required before Nuitka build
cd rust_lib && maturin build --release && pip install --force-reinstall target/wheels/*.whl
# Or debug dev install:
cd rust_lib && maturin develop
# Rust tests:
cd rust_lib && cargo test

# Standalone desktop binary (Nuitka) -> outputs/mtpro-{linux,windows}-x86_64
bash scripts/build_nuitka.sh linux     # native Linux
bash scripts/build_nuitka.sh windows   # cross-compile from Linux (MinGW-w64 + Wine)
```

CI (`.github/workflows/`): `CI.yml` only builds/uploads `rust_lib` wheels via maturin (tagged pushes publish to PyPI). `build-executables.yml` builds Nuitka standalone binaries for Linux and Windows. CI does **not** run the Python test suite.

## Architecture (not obvious from filenames)

- **Data model**: `core/em_data.py` `EMData` stores channels as an `xarray.Dataset` keyed by channel name (`Ex`, `Hy`, ...). `pd.Series` with `DatetimeIndex` is the exchange format between `EMData` and processors — use `get_channel_series()` / `set_channel_series()`.
- **Registry**: `base/data_manager.py` `DataManager(QObject)` owns `Dict[int, EMData]` and emits `data_added` / `data_removed` / `active_changed`. **Key `0` means "no active data"** — never use it for a dataset. All UI references datasets by integer key.
- **MAT format**: `io_utils/mat_io.py` reads/writes the legacy `tsStruct` dict with nested `(1,1)` MATLAB cell arrays (see `TS_STRUCT_TEMPLATE`). `MatSaver.save()` calls `em_data.update_meta()` to sync changes back into the raw `mat_meta` dict before writing. `core/mat_data.py` auto-fetches Kp-index data on construction from a local HDF5 file (`utils/fetch_kp.py`).
- **Signal-driven flow**: `DataManager` signals → `MainWindow` slots → viewer/processor updates. UI is MDI-based; tools open as `QMdiSubWindow` children.
- **Processing tools** (`processor/_base_processor.py` `BaseProcessorWidget`) share a pattern: load channel → interpolate NaN → history stack for undo → `_confirm_and_close()` restores NaN positions and emits `result_signal(ch_name, ndarray)`.
- **Pipeline mode is disabled**: `ui/data_process.py` has `PIPELINE_ENABLED = False`; Classic mode is the active path.
- `core/db_data.py` `EMDBData` is a stub for future DB-backed loading.

## Conventions

- 4-space indent; `snake_case` for files/functions/vars, `PascalCase` for Qt widgets and data classes.
- `from __future__ import annotations` at the top of files using forward references.
- Default timezone is `Asia/Shanghai` (CST); timestamps are timezone-aware `pd.Timestamp`.
- Rust: `rustfmt` defaults, snake_case; edition 2024, PyO3 0.28.
- Keep new runtime code in `python/src/{ui,core,io_utils,processor,utils,base,emtf}` — not in `python/scripts/` or `python/old/`.

## Testing quirks

- Tests live in `python/tests/`; some require local `.mat` fixtures (e.g. `test1.mat`, `test2.mat`) or a running Qt event loop (`QApplication`).
- Prefer deterministic unit tests over interactive GUI loops. When adding tests, use repo-local fixtures and note any external data needs in the docstring/PR.
- The repo root contains sample data files (`*.mat`, `*.h5`, `*.csv`) — do not hard-code machine-specific paths.

## Commits / PRs

Git history mixes short version subjects (`0.0.1`) and imperative subjects. For new commits use concise imperative subjects describing scope (e.g. `Add MAT save regression test`). PRs: include summary, test notes, and screenshots/recordings for visible PySide6 UI changes.
