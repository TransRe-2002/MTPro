import numpy as np
import pandas as pd
import xarray as xr

from base.data_manager import DataManager
from base.processing_broker import ProcessingBroker
from core.em_data import EMData


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


def test_processing_broker_includes_station_location_metadata():
    manager = DataManager()
    data_id = manager.add(DemoData())
    broker = ProcessingBroker(manager)
    captured = []
    broker.data_ready.connect(lambda _data_id, _channel, pkg: captured.append(pkg))

    broker.request_data(data_id, "")

    assert len(captured) == 1
    pkg = captured[0]
    assert pkg["name"] == "024BE"
    assert pkg["latitude"] == 31.123456
    assert pkg["longitude"] == 103.654321
    assert pkg["elevation"] == 1705.5
    assert pkg["stations"][0]["elevation"] == 1705.5
    assert pkg["station_metadata"].name == "024BE"
    assert pkg["station_metadata"].latitude == 31.123456
    assert pkg["station_metadata"].longitude == 103.654321
    assert pkg["station_metadata"].elevation == 1705.5


def test_processing_broker_includes_datetime_index_for_channel_payload():
    manager = DataManager()
    data = DemoData()
    data_id = manager.add(data)
    broker = ProcessingBroker(manager)
    captured = []
    broker.data_ready.connect(lambda _data_id, _channel, pkg: captured.append(pkg))

    broker.request_data(data_id, "Ex")

    assert len(captured) == 1
    assert captured[0]["datetime_index"].equals(data.datetime_index)
