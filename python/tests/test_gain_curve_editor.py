from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from ui.widgets.gain_curve_editor import GainCurveEditor


def test_gain_drag_updates_label_without_replacing_target_label():
    app = QApplication.instance() or QApplication([])
    editor = GainCurveEditor()
    editor.set_bands(["D1", "D2"], [1.0, 1.0])
    target = editor._targets[0]
    label = target._label

    editor.set_gain(0, 2.25)

    assert target._label is label
    assert "2.25" in label.toPlainText()
    editor.close()
    app.processEvents()
