# Repository Guidelines

## Project Structure & Module Organization
This repository has two active parts:

- `python/`: the main desktop application. Source lives in `python/src/` with feature areas split into `ui/`, `core/`, `io_utils/`, `processor/`, `utils/`, and `base/`. Keep new runtime code in these packages, not in `python/scripts/` or `python/old/`.
- `rust_lib/`: a small PyO3 extension built with `maturin`. Rust sources are in `rust_lib/src/`.

Tests currently live in `python/tests/`. Sample and local data files such as `*.mat`, `*.h5`, and `python/data/*.db` are used for development; avoid hard-coding new machine-specific paths.

## Build, Test, and Development Commands
- `python -m pip install -r python/requirements.txt`: install the Python GUI and data dependencies.
- `PYTHONPATH=python/src python python/src/main.py`: launch the PySide6 application locally.
- `PYTHONPATH=python/src pytest python/tests`: run the Python test suite.
- `cd rust_lib && maturin develop`: build and install the Rust extension into the active Python environment.
- `cd rust_lib && cargo test`: run Rust tests when Rust code changes.

If you add a new developer command, document it here and prefer commands that work from a clean checkout.

## Coding Style & Naming Conventions
Use 4-space indentation in Python and follow the existing PEP 8-style module layout: `snake_case` for files, functions, and variables; `PascalCase` for Qt widgets, data objects, and other classes. Keep imports absolute from `python/src` packages, for example `from ui.mainwindow import MainWindow`.

Rust should follow `rustfmt` defaults and standard snake_case naming. No formatter or linter is configured in-repo, so keep changes consistent with surrounding code and avoid unrelated cleanup.

## Testing Guidelines
Name Python tests `test_*.py` and keep them under `python/tests/`. Prefer deterministic unit tests over interactive GUI loops. Several existing tests depend on local MAT files or a running Qt event loop; when adding tests, use repository-local fixtures and state any external data requirements in the test docstring or PR.

## Commit & Pull Request Guidelines
Recent Git history uses short version-style subjects such as `0.0.1`, so there is no strong existing convention to preserve. For new commits, use concise imperative subjects that describe the change scope, for example `Add MAT save regression test`.

PRs should include a clear summary, test notes, and linked issues when relevant. Include screenshots or a short screen recording for visible PySide6 UI changes.
