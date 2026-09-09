from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from plugins.common.base_processor import BaseProcessorWidget, _HistoryBuffer


def test_history_buffer_bounds_retained_array_memory():
    history = _HistoryBuffer(max_bytes=32, max_steps=3)
    first = np.arange(2, dtype=np.float64)
    second = first + 10
    third = first + 20

    history.append(first)
    history.append(second)
    history.append(third)

    assert len(history) == 2
    assert history[0] is second
    assert history.total_bytes == 32

    assert history.pop() is third
    assert history.total_bytes == 16
    history.clear()
    assert history.total_bytes == 0


def test_history_buffer_always_retains_latest_snapshot():
    history = _HistoryBuffer(max_bytes=8, max_steps=2)
    oversized = np.arange(4, dtype=np.float64)

    history.append(oversized)

    assert history == [oversized]
    assert history.total_bytes == oversized.nbytes


def test_restore_nan_applies_boolean_mask_without_index_list():
    processor = SimpleNamespace(
        _nan_mask=np.array([False, True, False, True], dtype=bool),
    )

    restored = BaseProcessorWidget._restore_nan(
        processor, np.array([1.0, 2.0, 3.0, 4.0]),
    )

    np.testing.assert_array_equal(
        restored, np.array([1.0, np.nan, 3.0, np.nan]),
    )
