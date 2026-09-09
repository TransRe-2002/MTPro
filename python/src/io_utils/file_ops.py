from __future__ import annotations

import logging
import os
from pathlib import Path

from core.em_data import EMData
from io_utils.mat_io import MatLoader, MatSaver

logger = logging.getLogger(__name__)


def load_mat(path: str | Path, tz: str = "Asia/Shanghai") -> EMData:
    """Load a single MAT file into an EMData using the configured timezone."""
    normalized = str(Path(path).expanduser().resolve(strict=False))
    return MatLoader.load(normalized, tz=tz)


def save_mat(em_data: EMData, path: str | Path) -> None:
    """Write an EMData back to a MAT file at *path*."""
    MatSaver.save(em_data, str(path))


def unique_mat_path(folder: str, name: str, suffix: str) -> str:
    """Return a non-colliding ``{name}{suffix}.mat`` path inside *folder*."""
    base = f"{name}{suffix}"
    path = os.path.join(folder, f"{base}.mat")
    if not os.path.exists(path):
        return path
    n = 1
    while True:
        path = os.path.join(folder, f"{base}({n}).mat")
        if not os.path.exists(path):
            return path
        n += 1


def batch_save_mat(
    items: list[tuple[int, EMData]],
    folder: str,
    suffix: str,
    progress_callback: callable | None = None,
) -> tuple[int, int]:
    """Save every dataset in *items* into *folder* as ``{name}{suffix}.mat``.

    Returns ``(saved, failed)`` counts. *progress_callback* (if given) is
    called per item with ``(data_id, path)``.
    """
    saved = 0
    failed = 0
    for data_id, data in items:
        name = data.name if data.name else f"dataset_{data_id}"
        path = unique_mat_path(folder, name, suffix)
        try:
            save_mat(data, path)
            logger.info("Batch saved #%s to %s", data_id, path)
            saved += 1
        except Exception:
            logger.exception("Batch save failed for #%s", data_id)
            failed += 1
        if progress_callback is not None:
            progress_callback(data_id, path)
    return saved, failed
