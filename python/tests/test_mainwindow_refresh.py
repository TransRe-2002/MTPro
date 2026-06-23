from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import xarray as xr
from PySide6.QtWidgets import QApplication

from base.data_manager import DataManager
from core.em_data import EMData
from ui.mainwindow import MainWindow


class DemoData(EMData):
    def __init__(
        self,
        name: str,
        channels: list[str],
        start: str = "2024-01-01 00:00:00",
        values: list[float] | None = None,
    ):
        super().__init__("/tmp/demo.mat")
        self.name = name
        samples = values or [1.0, 2.0, 3.0]
        self.npts = len(samples)
        self.start_time = pd.Timestamp(start)
        self.dt = pd.Timedelta(seconds=5)
        self.datetime_index = pd.date_range(self.start_time, periods=self.npts, freq="5s")
        self.end_time = self.datetime_index[-1]
        self.kp_data = None
        self.data = xr.Dataset(coords={"time": self.datetime_index})
        for ch_name in channels:
            self.data[ch_name] = xr.DataArray(
                np.asarray(samples, dtype=np.float64),
                dims=("time",),
                coords={"time": self.datetime_index},
            )

    def restore_data(self, ch: str):
        return None


def _app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_mainwindow_refreshes_waveform_view_when_processing_confirms():
    _app()
    manager = DataManager()
    data = DemoData("demo", ["Ex1", "Hy1"])
    key = manager.add(data)

    window = MainWindow(manager)
    viewer = window._ensure_data_viewer(key)
    assert viewer is not None
    viewer.on_change_data_finished = MagicMock()

    window.data_process_widget.change_finished_signal.emit(key, "Ex1")

    viewer.on_change_data_finished.assert_called_once_with("Ex1")
    window.close()
