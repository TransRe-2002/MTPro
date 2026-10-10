from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from plugins.common.plotting import ProcessingPlotWidget, StepPlotWidget
from plugins.remove_region.editor import RegionPlotWidget
from plugins.remove_spike.editor import RemoveSpike, SpikePlotWidget


PLOT_WIDGET_TYPES = (
    ProcessingPlotWidget,
    StepPlotWidget,
    SpikePlotWidget,
    RegionPlotWidget,
)


def _show_plot(qtbot, plot):
    qtbot.addWidget(plot)
    plot.resize(640, 480)
    plot.show()
    qtbot.wait(10)
    return plot


def _send_wheel(plot, modifiers):
    view_box = plot.getViewBox()
    view_box.setRange(xRange=(0.0, 10.0), yRange=(0.0, 10.0), padding=0)
    QtWidgets.QApplication.processEvents()
    before = view_box.viewRange()

    viewport = plot.viewport()
    local_pos = QtCore.QPointF(viewport.rect().center())
    global_pos = QtCore.QPointF(viewport.mapToGlobal(local_pos.toPoint()))
    event = QtGui.QWheelEvent(
        local_pos,
        global_pos,
        QtCore.QPoint(),
        QtCore.QPoint(0, 120),
        QtCore.Qt.MouseButton.NoButton,
        modifiers,
        QtCore.Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QtWidgets.QApplication.sendEvent(viewport, event)
    QtWidgets.QApplication.processEvents()
    return before, view_box.viewRange()


def _range_width(view_range, axis):
    return view_range[axis][1] - view_range[axis][0]


@pytest.mark.parametrize("plot_type", PLOT_WIDGET_TYPES)
def test_ctrl_wheel_only_zooms_x_axis(qtbot, plot_type):
    plot = _show_plot(qtbot, plot_type())

    before, after = _send_wheel(
        plot, QtCore.Qt.KeyboardModifier.ControlModifier,
    )

    assert not np.isclose(_range_width(after, 0), _range_width(before, 0))
    assert np.isclose(_range_width(after, 1), _range_width(before, 1))
    assert tuple(plot.getViewBox().state["mouseEnabled"]) == (True, True)


@pytest.mark.parametrize("plot_type", PLOT_WIDGET_TYPES)
def test_shift_wheel_only_zooms_y_axis(qtbot, plot_type):
    plot = _show_plot(qtbot, plot_type())

    before, after = _send_wheel(
        plot, QtCore.Qt.KeyboardModifier.ShiftModifier,
    )

    assert np.isclose(_range_width(after, 0), _range_width(before, 0))
    assert not np.isclose(_range_width(after, 1), _range_width(before, 1))
    assert tuple(plot.getViewBox().state["mouseEnabled"]) == (True, True)


@pytest.mark.parametrize("plot_type", PLOT_WIDGET_TYPES)
@pytest.mark.parametrize(
    ("modifier", "expected"),
    (
        (QtCore.Qt.KeyboardModifier.ControlModifier, (True, False)),
        (QtCore.Qt.KeyboardModifier.ShiftModifier, (False, True)),
    ),
)
def test_middle_drag_uses_same_axis_constraints(qtbot, plot_type, modifier, expected):
    plot = _show_plot(qtbot, plot_type())
    view_box = plot.getViewBox()
    view_box.setRange(xRange=(0.0, 10.0), yRange=(0.0, 10.0), padding=0)
    QtWidgets.QApplication.processEvents()
    before = view_box.viewRange()

    viewport = plot.viewport()
    start = viewport.rect().center()
    end = start + QtCore.QPoint(40, 40)
    QtTest.QTest.mousePress(
        viewport, QtCore.Qt.MouseButton.MiddleButton, modifier, start,
    )
    assert tuple(view_box.state["mouseEnabled"]) == expected

    move_event = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseMove,
        QtCore.QPointF(end),
        QtCore.QPointF(viewport.mapToGlobal(end)),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.MouseButton.MiddleButton,
        modifier,
    )
    QtWidgets.QApplication.sendEvent(viewport, move_event)
    after = view_box.viewRange()
    if expected == (True, False):
        assert not np.isclose(after[0][0], before[0][0])
        assert np.isclose(after[1][0], before[1][0])
    else:
        assert np.isclose(after[0][0], before[0][0])
        assert not np.isclose(after[1][0], before[1][0])

    QtTest.QTest.mouseRelease(
        viewport, QtCore.Qt.MouseButton.MiddleButton, modifier, end,
    )
    assert tuple(view_box.state["mouseEnabled"]) == (True, True)


@pytest.mark.parametrize("plot_type", (ProcessingPlotWidget, StepPlotWidget))
def test_double_click_restores_ordinary_plot_view(qtbot, plot_type):
    plot = _show_plot(qtbot, plot_type())
    auto_range = Mock()
    plot.getViewBox().autoRange = auto_range

    QtTest.QTest.mouseDClick(
        plot.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        plot.viewport().rect().center(),
    )

    auto_range.assert_called_once_with()


def test_spike_operation_guide_tracks_current_mode(qtbot):
    editor = RemoveSpike(1, "Ex")
    qtbot.addWidget(editor)

    assert "框选放大模式" in editor.instruction_label.text()
    assert "左键框选放大" in editor.instruction_label.text()
    assert "Ctrl+滚轮/中键拖动 → 仅操作X轴" in editor.instruction_label.text()
    assert "Shift+滚轮/中键拖动 → 仅操作Y轴" in editor.instruction_label.text()

    editor._toggle_selection_mode()

    assert "框选选点模式" in editor.instruction_label.text()
    assert "左键框选添加点" in editor.instruction_label.text()
    assert "右键框选取消点" in editor.instruction_label.text()
