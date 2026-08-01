# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MTPro** — a PySide6 desktop application for magnetotelluric (MT) time series processing and visualization. It loads MAT-format EM data files, displays multi-channel waveform plots with Kp-index overlays, and provides interactive processing tools (spike removal, wavelet denoising, step correction, etc.) via an MDI workspace.

There's also a small PyO3 Rust extension in `rust_lib/` (currently a stub) built with maturin.

## Python Virtual Environment

**All Python commands MUST use the project venv** at `python/.venv` (created from Anaconda Python 3.13).

Prefix every Python / pip / pytest command with the venv Python path, or activate first:

```bash
# Option 1: use full path
python/.venv/bin/python ...
python/.venv/bin/pip ...
python/.venv/bin/pytest ...

# Option 2: activate (not persisted across shell commands)
source python/.venv/bin/activate
```

## Build, Test, and Development Commands

```bash
# Activate venv first (if using option 2)
source python/.venv/bin/activate

# Install Python dependencies
pip install -r python/requirements.txt

# Launch the application (PYTHONPATH required)
PYTHONPATH=python/src python python/src/main.py

# Run the Python test suite
PYTHONPATH=python/src pytest python/tests

# Build standalone desktop app with Nuitka
bash scripts/build_nuitka.sh [linux|windows]

# Build and install the Rust extension (release)
cd rust_lib && maturin build --release && pip install --force-reinstall target/wheels/*.whl

# Or with develop mode (debug)
cd rust_lib && maturin develop

# Run Rust tests
cd rust_lib && cargo test
```

## Architecture

### Data layer

- **`base/data_manager.py`** — Central `DataManager(QObject)` registry. Owns `Dict[int, EMData]`, emits `data_added` / `data_removed` / `active_changed` signals. Key 0 means "no active data." All UI components reference datasets by integer key.
- **`core/em_data.py`** — Abstract `EMData` base class. Data is stored as an `xarray.Dataset` keyed by channel name (e.g. `Ex`, `Hy`). Channels are accessed via `get_channel_series()` / `set_channel_series()` which return/accept `pd.Series` with `DatetimeIndex`. Properties `chid`, `npts`, `NCh` shadow from the xarray Dataset when present, with setter fallbacks.
- **`core/mat_data.py`** — `MatEMData(EMData)` concrete subclass for MAT files. Parses the legacy `tsStruct` format with nested MATLAB arrays. Stores raw `mat_meta` dict for round-trip save. Auto-fetches Kp-index data on construction.
- **`core/db_data.py`** — `EMDBData(EMData)` stub for future database-backed data loading.

### I/O layer

- **`io_utils/em_io.py`** — Abstract `EMDataLoader` / `EMDataSaver` interfaces.
- **`io_utils/mat_io.py`** — `MatLoader.load(path)` and `MatSaver.save(data, path)`. Uses `scipy.io.loadmat`/`savemat`. The MAT format uses a `tsStruct` dict with nested `(1,1)` cell arrays — see `TS_STRUCT_TEMPLATE` for the full schema. `MatSaver.save()` calls `em_data.update_meta()` on `MatEMData` to sync changes back into the raw dict before writing.

### UI layer

- **`ui/mainwindow.py`** — `MainWindow(QMainWindow)`. Owns the `DataManager`. Layout: welcome screen → workspace splitter (sidebar data tree + tab widget). Tabs: "波形浏览" (waveform view via `DataViewWidget`) and "数据处理" (processing via `DataProcessWidget`). File open/save/save-as menu actions. Persists splitter sizes, active tab, and recent files to `~/.config/mtpro/app_state.toml`. **Viewer caching is active-only**: `data_viewers` holds at most the active dataset's `DataViewWidget`; on dataset switch the old viewer's interaction state (viewport, channel order, hidden channels, row heights) is snapshotted into `viewer_state[key]` and the widget is destroyed. Re-activation rebuilds the viewer and restores the state. Batch MAT open warns when the estimated in-memory footprint (~2× on-disk size for v7.3) approaches available RAM.
- **`ui/data_view_widget.py`** — `DataViewWidget(TimeViewportMixin, QWidget)`. The waveform viewer. One `BriefPlot` per channel in a vertical `QSplitter`, plus Kp bar chart. Channels are ordered E → B/H → Kp. Supports drag-reorder and per-channel visibility checkboxes. Display uses **LOD pyramids** (`utils/decimate.py`, Rust-backed min/max envelope, truncated at `PYRAMID_MAX_FINEST_POINTS` to bound memory on 5s-sampling year-long datasets); viewport changes render only ~4×pixel-width points per channel. Pyramid construction runs **async on the shared io pool** (`start_display_build()` → `display_ready` signal); the synchronous `init_data(build_plots=True)` path is kept for tests.
- **`ui/data_process.py`** — `DataProcessWidget` with Classic/Pipeline mode toggle (Pipeline currently disabled via `PIPELINE_ENABLED = False`). Classic mode uses a multi-workspace tab system (`WorkspaceTabPanel` → `WorkspaceMdiArea`) where tools open as `QMdiSubWindow` children. Tools are drag-dropped from the sidebar tool model. The `DataProcessLegacyWidget` handles tool window lifecycle: creation, dedup, cleanup on data close.
- **`ui/data_tree_view.py`** — `DataTreeViewer` sidebar widget showing loaded datasets and their channels.
- **`base/time_viewport_mixin.py`** — `TimeViewportMixin` providing shared time navigation logic: 16ms debounced viewport updates (`x_view_start` / `x_view_size` as `pd.Timedelta`), slider/scrollbar sync, time jump with boundary clamping, and non-modal auto-dismiss warnings. Subclasses must implement `_on_viewport_changed()`.

### Processor layer

- **`processor/_base_processor.py`** — `BaseProcessorWidget(QWidget)`. Common pattern for all processing tools:
  1. On init: loads channel series, interpolates NaN, stores `origin_y_data`
  2. History stack for undo (`_save_history()` / `_undo()`)
  3. `_revert()` restores original interpolated data
  4. `_confirm_and_close()` restores NaN positions and emits `result_signal(ch_name, ndarray)`
  5. Subclasses implement `_update_plot()` and `_log()`
- Processing tools: `RemoveSpike`, `RemoveWaveletDenoise`, `RemoveDiurnalSinusoid`, `RemoveStepByBaseline`, `RemoveStepByDiff`, `RemoveStepByWindow`, `RobustEstimateWidget`, `RemoteReferenceWidget`.

### Utilities

- **`utils/concurrency.py`** — App-wide shared thread pools (Qt-free): `compute_pool` (max(1, cpu//2) workers) for CPU-bound orchestration, `io_pool` (cap 32, lazily created) for file I/O. Rules: never block-wait on a future from within its own pool (cross-pool waits are fine); cancellation is cooperative; pool threads must not touch Qt widgets.
- **`utils/qt_bridge.py`** — `FutureBridge` / `run(kind, fn, ..., on_done, on_error, owner)`: relays pool future results to the main thread via queued signals. Note: `on_done`/`on_error` must be QObject bound methods (closures degrade to DirectConnection and run on the pool thread).
- **`utils/signals.py`** — `block_signals(*widgets)` context manager for temporarily blocking Qt signals.
- **`utils/app_state.py`** — `AppState` dataclass persisted as TOML. `remember_recent_file()` / `remove_recent_file()` helpers. Path: `$MTPRO_APP_STATE_PATH` or `~/.config/mtpro/app_state.toml`. Max 8 recent files.
- **`utils/series.py`**, **`utils/time_convert.py`**, **`utils/timestamp.py`** — pandas/DatetimeIndex/numpy conversion helpers.
- **`utils/fetch_kp.py`** — Fetches Kp-index data from a local HDF5 file.

## Coding Conventions

- Python: 4-space indent, `snake_case` for files/functions/variables, `PascalCase` for Qt widgets and data classes
- Imports are absolute from `python/src` packages: `from ui.mainwindow import MainWindow`
- Default timezone is `Asia/Shanghai` (CST); timestamps are timezone-aware `pd.Timestamp`
- `from __future__ import annotations` at the top of files that use forward references
- Rust: `rustfmt` defaults, snake_case naming

## Key Design Patterns

- **Signal-driven data flow**: `DataManager` signals → `MainWindow` slots → viewer/processor updates
- **Mixin for shared behavior**: `TimeViewportMixin` encapsulates time navigation that both `DataViewWidget` and other components share
- **Active-only viewer cache**: per-dataset view *state* is permanent (dict), per-dataset *widgets + LOD pyramids* exist only for the active dataset — memory no longer scales with the number of open datasets (matters: 5s-sampling year-long MAT files are ~350 MB each in RAM)
- **MIME-type drag-and-drop**: Tool selection uses custom MIME types (`CLASSIC_TOOL_MIME`, `DATA_TREE_SELECTION_MIME`); channel reordering uses `application/x-channel-reorder`
- **Undo/redo via history stack**: Processing tools maintain `list[np.ndarray]` snapshots
- **xarray.Dataset** is the canonical in-memory data store; `pd.Series` is the exchange format between EMData and processors

## Tests

Tests live in `python/tests/`. Some tests require local `.mat` fixture files or a running Qt event loop (`QApplication`). Run with `PYTHONPATH=python/src pytest python/tests`. When adding tests, use repo-local fixtures and document any external data dependencies.
