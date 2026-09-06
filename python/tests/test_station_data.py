import numpy as np
import pandas as pd
import xarray as xr
from PySide6.QtWidgets import QApplication

from base.data_manager import DataManager
from core.em_data import EMData
from output_tools.station_data import station_data_package
from ui.station_output_host import StationOutputHost


class DemoData(EMData):
    def __init__(self):
        super().__init__("/tmp/demo.mat")
        self.name = "024BE"
        self.npts = 3
        self.start_time = pd.Timestamp("2024-01-01 00:00:00")
        self.end_time = pd.Timestamp("2024-01-01 00:00:10")
        self.dt = pd.Timedelta(seconds=5)
        self.datetime_index = pd.date_range(self.start_time, periods=3, freq="5s")
        self.kp_data = None
        self.latitude = 31.123456
        self.longitude = 103.654321
        self.elevation = 1705.5
        self.data = xr.Dataset(coords={"time": self.datetime_index})
        for channel in ("Ex", "Ey", "Hx", "Hy"):
            self.data[channel] = xr.DataArray(
                np.arange(3, dtype=np.float64),
                dims=("time",),
                coords={"time": self.datetime_index},
            )

    def restore_data(self, ch: str):
        return None


def test_station_data_package_includes_station_location_metadata():
    manager = DataManager()
    data_id = manager.add(DemoData())

    package = station_data_package(manager, data_id)

    assert package is not None
    assert package["name"] == "024BE"
    assert package["latitude"] == 31.123456
    assert package["longitude"] == 103.654321
    assert package["elevation"] == 1705.5
    assert package["stations"][0]["elevation"] == 1705.5
    assert package["station_metadata"].name == "024BE"
    assert package["station_metadata"].latitude == 31.123456
    assert package["station_metadata"].longitude == 103.654321
    assert package["station_metadata"].elevation == 1705.5


def test_station_data_package_keeps_datetime_index():
    manager = DataManager()
    data = DemoData()
    data_id = manager.add(data)

    package = station_data_package(manager, data_id)

    assert package is not None
    assert package["datetime_index"].equals(data.datetime_index)


def test_station_output_host_opens_a_station_tool_and_delivers_data(qtbot):
    QApplication.instance() or QApplication([])
    manager = DataManager()
    data = DemoData()
    data_id = manager.add(data)
    host = StationOutputHost(manager)
    host.init_data(data, data_id)
    qtbot.addWidget(host)

    window = host.open_output_window("robust estimate", data_id)

    assert window is not None
    assert window.widget().__class__.__name__ == "RobustEstimateWidget"
    assert window.widget()._pkg["station_metadata"].name == "024BE"
    assert host.open_output_window("robust estimate", data_id) is window
    host.close_windows_for_data(data_id)
    assert host.mdi_area.subWindowList() == []
    host.close()


def test_station_output_host_delivers_initial_data_to_remote_reference(qtbot):
    QApplication.instance() or QApplication([])
    manager = DataManager()
    data = DemoData()
    data_id = manager.add(data)
    host = StationOutputHost(manager)
    host.init_data(data, data_id)
    qtbot.addWidget(host)

    window = host.open_output_window("remote reference", data_id)

    assert window is not None
    assert window.widget().__class__.__name__ == "RemoteReferenceWidget"
    assert window.widget()._local_pkg["station_metadata"].name == "024BE"
    host.close()
