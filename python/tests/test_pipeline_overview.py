from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from ui.data_process_pipeline import (
    ChannelOverviewWidget,
    points_to_pixels,
    ruler_tick_multipliers,
)
from utils.decimate import LODPyramid


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_pipeline_overview_queries_lod_for_the_painted_range():
    _app()
    samples = np.sin(np.arange(100_000, dtype=np.float64) / 17.0)
    samples[50_000] = 50.0
    overview = ChannelOverviewWidget(samples, len(samples))
    overview.set_zoom(6.0)

    assert isinstance(overview._pyramid, LODPyramid)
    x_values, y_values = overview._visible_lod(0, 200)
    assert len(x_values) == len(y_values)
    assert len(x_values) <= 804
    overview.close()


def test_pipeline_overview_hides_zero_baseline_by_default_and_can_toggle_it():
    _app()
    overview = ChannelOverviewWidget(np.array([-2.0, 0.0, 3.0]), 3)

    assert overview._show_baseline is False
    overview.set_baseline_visible(True)
    assert overview._show_baseline is True
    overview.set_baseline_visible(False)
    assert overview._show_baseline is False
    overview.close()


def test_pipeline_ruler_widens_tick_and_label_spacing_at_low_zoom():
    grid_points = 50
    low_tick, low_label = ruler_tick_multipliers(grid_points, 1.0)
    high_tick, high_label = ruler_tick_multipliers(grid_points, 20.0)

    assert points_to_pixels(grid_points * low_tick, 1.0) >= 8.0
    assert points_to_pixels(grid_points * low_label, 1.0) >= 48.0
    assert low_tick > high_tick
    assert low_label > high_label
