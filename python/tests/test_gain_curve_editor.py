from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
import numpy as np
import pytest

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


def test_eq_curve_is_smooth_bounded_and_view_is_fixed(qtbot):
    editor = GainCurveEditor(maximum=2.0)
    qtbot.addWidget(editor)
    editor.set_bands(["D3", "D2", "D1"], [0.0, 2.0, 0.5])
    editor.resize(600, 250)
    editor.show()
    qtbot.waitExposed(editor)
    x, y = editor.curve.getData()
    assert len(x) > 3
    np.testing.assert_allclose(y[[0, 24, 48]], [0.0, 2.0, 0.5])
    assert np.min(y) >= 0.0
    assert np.max(y) <= 2.0
    view = editor.plot_item.getViewBox()
    assert view.viewRange()[1] == pytest.approx([0.0, 2.0])
    assert view.state["mouseEnabled"] == [False, False]
    assert editor.plot_item.buttonsHidden
    editor._targets[1].setPos(9, 3)
    assert editor._targets[1].pos().x() == 1
    assert editor._targets[1].pos().y() == 2
