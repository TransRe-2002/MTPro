from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication, QGridLayout, QSpinBox, QSplitter

from core.pipeline_model import PreviewChannelSlice
from ui.widgets.filter_preview_editor import FilterPreviewEditor
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


@pytest.mark.parametrize("kind,active", [
    ("lowpass", {"high_fraction", "order"}),
    ("highpass", {"low_fraction", "order"}),
    ("bandpass", {"low_fraction", "high_fraction", "order"}),
    ("bandstop", {"low_fraction", "high_fraction", "order"}),
    ("notch", {"notch_fraction", "quality_factor"}),
])
def test_iir_fields_follow_filter_kind_without_losing_values(qtbot, kind, active):
    wrapper = registry.create_pipeline_editor(
        "iir_filter", _preview_slice(), None,
        params={"filter_kind": kind, "low_fraction": 0.12, "high_fraction": 0.42,
                "notch_fraction": 0.55, "quality_factor": 25.0, "order": 6},
    )
    qtbot.addWidget(wrapper)
    editor = wrapper._editor
    assert isinstance(editor.layout().itemAt(1).widget(), QSplitter)
    assert isinstance(editor.layout().itemAt(2).layout(), QGridLayout)
    dependent = {"low_fraction", "high_fraction", "notch_fraction", "quality_factor", "order"}
    for name in dependent:
        assert editor._editors[name][0].isEnabled() == (name in active)
        assert editor._field_labels[name].isEnabled() == (name in active)
    assert editor._editors["min_segment_points"][0].isEnabled()
    values = editor.params()
    assert values["low_fraction"] == pytest.approx(0.12)
    assert values["high_fraction"] == pytest.approx(0.42)
    assert values["notch_fraction"] == pytest.approx(0.55)
    assert values["quality_factor"] == pytest.approx(25.0)
    assert values["order"] == 6
    assert editor._editors["low_fraction"][0].minimum() > 0


def test_iir_switching_restores_values_and_applies_preview_once(qtbot, monkeypatch):
    wrapper = registry.create_pipeline_editor(
        "iir_filter", _preview_slice(), None,
        params={"filter_kind": "bandpass", "low_fraction": 0.12, "high_fraction": 0.42},
    )
    qtbot.addWidget(wrapper)
    editor = wrapper._editor
    plugin = wrapper.plugin
    original_process = plugin.process
    calls = []

    def process(*args):
        calls.append(1)
        return original_process(*args)

    monkeypatch.setattr(plugin, "process", process)
    qtbot.waitUntil(lambda: editor.preview_result is not None, timeout=5000)
    kind = editor._editors["filter_kind"][0]
    kind.setCurrentText("notch")
    assert not editor._editors["low_fraction"][0].isEnabled()
    assert editor._editors["notch_fraction"][0].isEnabled()
    kind.setCurrentText("bandpass")
    assert editor._editors["low_fraction"][0].value() == pytest.approx(0.12)
    assert editor._editors["high_fraction"][0].value() == pytest.approx(0.42)
    qtbot.waitUntil(lambda: editor.preview_result is not None and editor.preview_params["filter_kind"] == "bandpass", timeout=5000)
    x, _y = editor.original_curve.getData()
    assert x[0] == pytest.approx(_preview_slice().cts.index[0].timestamp())
    ready = []
    wrapper.params_applied.connect(lambda params, _broadcast: ready.append(params))
    count = len(calls)
    editor.apply_button.click()
    assert ready and ready[0]["filter_kind"] == "bandpass"
    assert len(calls) == count
