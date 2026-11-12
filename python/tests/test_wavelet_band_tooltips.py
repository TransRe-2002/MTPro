from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent, QPointF
from PySide6.QtGui import QHelpEvent
from PySide6.QtWidgets import QToolTip

from plugins.remove_wavelet_denoise.editor import RemoveWaveletDenoise
from test_remove_step_widget import PulseDemoData
from ui.widgets.band_axis import band_tooltip


def test_frequency_and_period_ranges_include_zero_frequency():
    assert band_tooltip('D1', 0.05, 0.1) == 'D1\n频率：0.05–0.1 Hz\n周期：10–20 s'
    assert band_tooltip('B1', 0.0, 0.05) == 'B1\n频率：0–0.05 Hz\n周期：20–∞ s'


def test_energy_axis_shows_tooltip_on_the_label(qtbot, monkeypatch):
    widget = RemoveWaveletDenoise(PulseDemoData(), 'Ex1')
    qtbot.addWidget(widget)
    widget.resize(1200, 800)
    widget.show()
    widget.analysis_tabs.setCurrentWidget(widget.detail_plot_widget)
    widget._update_detail_energy_plot(SimpleNamespace(
        detail_energy_before=np.array([1.0, 2.0]),
        detail_energy_after=np.array([0.5, 1.0]),
        band_labels=['D1 [0.05-0.1 Hz]', 'D2 [0.025-0.05 Hz]'],
        band_ranges_hz=[(0.05, 0.1), (0.025, 0.05)],
    ))
    qtbot.wait(100)
    axis = widget.detail_band_axis
    assert [name for _, name, _ in axis._band_ticks] == ['D1', 'D2']
    view = axis.linkedView()
    scene_tick = view.mapViewToScene(QPointF(1, 0))
    axis_tick = axis.mapFromScene(scene_tick)
    assert '周期：10–20 s' in axis.tooltip_at(axis_tick.x())
    axis_point = QPointF(axis_tick.x(), axis.boundingRect().center().y())
    viewport_point = widget.detail_plot_widget.mapFromScene(axis.mapToScene(axis_point))
    captured = []
    monkeypatch.setattr(QToolTip, 'showText', lambda _pos, text, *_args: captured.append(text))
    event = QHelpEvent(QEvent.Type.ToolTip, viewport_point,
                       widget.detail_plot_widget.viewport().mapToGlobal(viewport_point))
    QCoreApplication.sendEvent(widget.detail_plot_widget.viewport(), event)
    assert captured and captured[-1] == 'D1\n频率：0.05–0.1 Hz\n周期：10–20 s'
