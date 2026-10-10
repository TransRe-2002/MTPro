from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt

from ui.theme import configure_matplotlib_canvas


def _configured_canvas(qtbot, *, logarithmic: bool = False):
    figure = Figure()
    canvas = FigureCanvas(figure)
    qtbot.addWidget(canvas)
    axis = figure.subplots(1, 1)
    axis.plot([1.0, 10.0], [2.0, 20.0])
    if logarithmic:
        axis.set_xscale("log")
        axis.set_yscale("log")
    configure_matplotlib_canvas(canvas, figure)
    canvas.draw()
    return figure, canvas, axis, canvas._mtpro_interaction_controller


def _scroll_event(axis, *, key: str | None = None):
    return SimpleNamespace(
        inaxes=axis,
        xdata=3.0,
        ydata=6.0,
        button="up",
        step=1.0,
        key=key,
    )


def _mouse_event(
    axis,
    *,
    x: float,
    y: float,
    key: str | None = None,
    button: int | None = 1,
    dblclick: bool = False,
):
    return SimpleNamespace(
        inaxes=axis,
        x=x,
        y=y,
        button=button,
        key=key,
        dblclick=dblclick,
    )


def test_qtagg_wheel_zoom_reset_button_and_double_click(qtbot):
    _figure, canvas, axis, controller = _configured_canvas(qtbot)
    canvas.resize(640, 480)
    controller._position_button()
    home_x = axis.get_xlim()
    home_y = axis.get_ylim()

    assert controller.reset_button.x() == 640 - controller.reset_button.width() - 8
    assert controller.reset_button.y() == 480 - controller.reset_button.height() - 8

    controller._on_scroll(_scroll_event(axis))
    assert np.ptp(axis.get_xlim()) < np.ptp(home_x)
    assert np.ptp(axis.get_ylim()) < np.ptp(home_y)

    controller.reset_button.click()
    np.testing.assert_allclose(axis.get_xlim(), home_x)
    np.testing.assert_allclose(axis.get_ylim(), home_y)

    controller._on_scroll(_scroll_event(axis))
    controller._on_button_press(SimpleNamespace(
        inaxes=axis,
        button=1,
        dblclick=True,
    ))
    np.testing.assert_allclose(axis.get_xlim(), home_x)
    np.testing.assert_allclose(axis.get_ylim(), home_y)


def test_qtagg_modifier_keys_limit_zoom_axis(qtbot):
    _figure, _canvas, axis, controller = _configured_canvas(qtbot)
    home_x = axis.get_xlim()
    home_y = axis.get_ylim()

    controller._on_scroll(_scroll_event(axis, key="control"))
    assert axis.get_xlim() != home_x
    np.testing.assert_allclose(axis.get_ylim(), home_y)

    controller.reset()
    controller._on_scroll(_scroll_event(axis, key="shift"))
    np.testing.assert_allclose(axis.get_xlim(), home_x)
    assert axis.get_ylim() != home_y


def test_qtagg_log_zoom_stays_positive_and_shared_axes_follow(qtbot):
    figure = Figure()
    canvas = FigureCanvas(figure)
    qtbot.addWidget(canvas)
    axes = figure.subplots(2, 2, sharex=True, sharey="row")
    for axis in axes.flat:
        axis.loglog([1.0, 10.0], [2.0, 20.0])
    configure_matplotlib_canvas(canvas, figure)
    canvas.draw()
    controller = canvas._mtpro_interaction_controller

    controller._on_scroll(_scroll_event(axes[0, 0], key="control"))

    assert all(limit > 0 for limit in axes[0, 0].get_xlim())
    np.testing.assert_allclose(axes[0, 0].get_xlim(), axes[1, 1].get_xlim())


def test_qtagg_new_render_replaces_reset_baseline(qtbot):
    figure, canvas, axis, controller = _configured_canvas(qtbot)
    controller._on_scroll(_scroll_event(axis))

    figure.clear()
    replacement = figure.subplots(1, 1)
    replacement.set_xlim(100.0, 200.0)
    replacement.set_ylim(-5.0, 5.0)
    configure_matplotlib_canvas(canvas, figure)
    controller._on_scroll(SimpleNamespace(
        inaxes=replacement,
        xdata=150.0,
        ydata=0.0,
        button="up",
        step=1.0,
        key=None,
    ))
    controller.reset()

    np.testing.assert_allclose(replacement.get_xlim(), (100.0, 200.0))
    np.testing.assert_allclose(replacement.get_ylim(), (-5.0, 5.0))


def test_qtagg_left_drag_pans_both_axes_and_restores_cursor(qtbot):
    _figure, canvas, axis, controller = _configured_canvas(qtbot)
    home_x = axis.get_xlim()
    home_y = axis.get_ylim()

    controller._on_button_press(_mouse_event(axis, x=100.0, y=100.0))
    assert canvas.cursor().shape() == Qt.CursorShape.ClosedHandCursor
    controller._on_motion(_mouse_event(axis, x=140.0, y=130.0, button=None))

    assert axis.get_xlim() != home_x
    assert axis.get_ylim() != home_y
    controller._on_button_release(_mouse_event(axis, x=140.0, y=130.0))
    assert canvas.cursor().shape() == Qt.CursorShape.OpenHandCursor


def test_qtagg_drag_modifier_keys_limit_pan_axis(qtbot):
    _figure, _canvas, axis, controller = _configured_canvas(qtbot)
    home_x = axis.get_xlim()
    home_y = axis.get_ylim()

    controller._on_button_press(_mouse_event(axis, x=100.0, y=100.0, key="control"))
    controller._on_motion(_mouse_event(axis, x=140.0, y=130.0, button=None))
    controller._on_button_release(_mouse_event(axis, x=140.0, y=130.0))
    assert axis.get_xlim() != home_x
    np.testing.assert_allclose(axis.get_ylim(), home_y)

    controller.reset()
    controller._on_button_press(_mouse_event(axis, x=100.0, y=100.0, key="shift"))
    controller._on_motion(_mouse_event(axis, x=140.0, y=130.0, button=None))
    controller._on_button_release(_mouse_event(axis, x=140.0, y=130.0))
    np.testing.assert_allclose(axis.get_xlim(), home_x)
    assert axis.get_ylim() != home_y


def test_qtagg_log_drag_stays_positive_and_preserves_axis_ratio(qtbot):
    _figure, _canvas, axis, controller = _configured_canvas(qtbot, logarithmic=True)
    home_x = axis.get_xlim()
    home_ratio = home_x[1] / home_x[0]

    controller._on_button_press(_mouse_event(axis, x=100.0, y=100.0))
    controller._on_motion(_mouse_event(axis, x=180.0, y=100.0, button=None))
    controller._on_button_release(_mouse_event(axis, x=180.0, y=100.0))

    assert all(limit > 0 for limit in axis.get_xlim())
    assert np.isclose(axis.get_xlim()[1] / axis.get_xlim()[0], home_ratio)


def test_qtagg_dense_zoom_events_are_coalesced(qtbot):
    _figure, canvas, axis, controller = _configured_canvas(qtbot)
    canvas.draw_idle = Mock()

    for _ in range(20):
        controller._on_scroll(_scroll_event(axis))

    assert controller._redraw_timer.isActive()
    qtbot.waitUntil(lambda: canvas.draw_idle.call_count == 1, timeout=500)
    assert canvas.draw_idle.call_count == 1
