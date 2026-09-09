from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication, QSpinBox

from ui.data_process_pipeline import PreviewChannelSlice
from ui.filter_preview_editor import FilterPreviewEditor
from ui.tools import registry


PREVIEW_TOOL_IDS = (
    "median_filter",
    "morphological_filter",
    "detrend",
    "iir_filter",
)


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _preview_slice() -> PreviewChannelSlice:
    index = pd.date_range(
        "2025-01-01",
        periods=512,
        freq="s",
        tz="Asia/Shanghai",
    )
    positions = np.arange(index.size, dtype=np.float64)
    values = np.sin(positions / 13.0) + 0.25 * np.sin(positions / 2.0)
    values[180:184] = np.nan
    return PreviewChannelSlice("Ex", pd.Series(values, index=index, name="Ex"))


@pytest.mark.parametrize("tool_id", PREVIEW_TOOL_IDS)
def test_lightweight_filter_uses_common_curve_preview(qtbot, tool_id):
    _app()
    spec = registry.get(tool_id)
    assert spec is not None
    assert spec.kind == "interactive"

    editor = registry.create_pipeline_editor(
        tool_id,
        _preview_slice(),
        None,
        params={},
    )
    assert editor is not None
    qtbot.addWidget(editor)
    preview_editor = editor._editor
    assert isinstance(preview_editor, FilterPreviewEditor)

    qtbot.waitUntil(lambda: preview_editor.preview_result is not None, timeout=5000)
    original_x, original_y = preview_editor.original_curve.getData()
    preview_x, preview_y = preview_editor.preview_curve.getData()
    difference_x, difference_y = preview_editor.difference_curve.getData()
    assert len(original_x) == len(original_y) > 0
    assert len(preview_x) == len(preview_y) > 0
    assert len(difference_x) == len(difference_y) > 0
    assert "RMS 输入" in preview_editor.metrics_label.text()
    assert preview_editor.apply_button.isEnabled()
    preview_editor.display_mode_combo.setCurrentText("仅参数预览")
    assert not preview_editor.original_curve.isVisible()
    assert preview_editor.preview_curve.isVisible()
    editor.close()


def test_filter_preview_recomputes_after_debounced_parameter_change(qtbot):
    _app()
    editor = registry.create_pipeline_editor(
        "median_filter",
        _preview_slice(),
        None,
        params={"window_points": 5, "min_segment_points": 5},
    )
    assert editor is not None
    qtbot.addWidget(editor)
    preview_editor = editor._editor
    assert isinstance(preview_editor, FilterPreviewEditor)
    qtbot.waitUntil(lambda: preview_editor.preview_result is not None, timeout=5000)

    window_editor, _spec = preview_editor._editors["window_points"]
    assert isinstance(window_editor, QSpinBox)
    window_editor.setValue(7)
    assert not preview_editor.apply_button.isEnabled()
    qtbot.waitUntil(
        lambda: (
            preview_editor.preview_result is not None
            and preview_editor.preview_params is not None
            and preview_editor.preview_params["window_points"] == 7
        ),
        timeout=5000,
    )

    applied = []
    editor.params_applied.connect(lambda params, _broadcast: applied.append(params))
    preview_editor.apply_button.click()
    qtbot.waitUntil(lambda: bool(applied), timeout=5000)
    assert applied[-1]["window_points"] == 7
    editor.close()
