"""PlotPanel 对空值和退化幅值范围的回归测试（offscreen Qt）。"""

import os
import warnings

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QApplication

from ui.plot_panel import PlotPanel


def _app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _index(npts: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=npts, freq="1s", tz="UTC")


def test_all_nan_channel_shows_placeholder_without_warning():
    _app()
    y_data = pd.Series(np.full(20, np.nan), index=_index(20))

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        panel = PlotPanel(x_data=y_data.index, y_data=y_data, label="Hy")

    assert panel.plot_curve is None
    assert not panel._has_plottable_data
    assert not panel.zoom_in_action.isEnabled()
    assert not panel.zoom_out_action.isEnabled()
    panel.close()


def test_constant_channel_has_nonzero_y_viewport():
    _app()
    y_data = pd.Series(np.full(20, 3.5), index=_index(20))

    panel = PlotPanel(x_data=y_data.index, y_data=y_data, label="Ex")

    assert panel._has_plottable_data
    assert panel.y_viewport_size > 0
    panel.close()
