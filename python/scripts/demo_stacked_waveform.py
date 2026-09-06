"""Standalone preview for the stacked waveform viewer."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import xarray as xr
from PySide6.QtWidgets import QApplication

from core.em_data import EMData
from ui.data_view_widget import DataViewWidget


class DemoEMData(EMData):
    def __init__(self) -> None:
        super().__init__("/tmp/mtpro-stacked-demo.mat")
        self.name = "Stacked waveform demo"
        self.start_time = pd.Timestamp("2024-07-01", tz="UTC")
        self.datetime_index = pd.date_range(self.start_time, periods=7200, freq="30s")
        self.end_time = self.datetime_index[-1]
        self.dt = pd.Timedelta(seconds=30)
        self.kp_data = pd.DataFrame({
            "Kp_datetime": pd.date_range(self.start_time, periods=48, freq="3h"),
            "Kp": ([2, 3, 4, 5, 3, 2, 6, 7, 5, 4, 3, 2, 4, 5, 6, 3] * 3),
        })

        t = np.arange(len(self.datetime_index), dtype=float) / 120.0
        rng = np.random.default_rng(42)
        signals = {
            "Ex1": 2.0 * np.sin(t / 3.0) + 0.15 * rng.normal(size=t.size),
            "Ey1": 0.6 * np.sin(t * 1.4) + 0.08 * rng.normal(size=t.size),
            "Ey2": 8.0 * np.sin(t / 7.0) + 0.4 * rng.normal(size=t.size),
            "Hx": 30.0 * np.sin(t / 2.0) + 2.0 * rng.normal(size=t.size),
            "Hy": 1.2 * np.sin(t * 2.1) + 0.1 * rng.normal(size=t.size),
        }
        self.data = xr.Dataset(coords={"time": self.datetime_index})
        for name, values in signals.items():
            self.data[name] = xr.DataArray(
                values, dims=("time",), coords={"time": self.datetime_index}
            )

    def restore_data(self, ch: str):
        return None


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    viewer = DataViewWidget()
    viewer.resize(1400, 820)
    viewer.setWindowTitle("MTPro — 多通道堆叠波形预览")
    viewer.init_data(DemoEMData())
    viewer.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
