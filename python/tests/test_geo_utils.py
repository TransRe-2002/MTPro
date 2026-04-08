import pandas as pd

from core.em_data import Channel, EMData
from utils.geo import great_circle_distance, station_distance


class DemoData(EMData):
    def __init__(self, name: str, latitude: float | None, longitude: float | None):
        super().__init__("/tmp/demo.mat")
        self.name = name
        self.npts = 3
        self.start_time = pd.Timestamp("2024-01-01 00:00:00")
        self.end_time = pd.Timestamp("2024-01-01 00:00:10")
        self.dt = pd.Timedelta(seconds=5)
        self.datetime_index = pd.date_range(self.start_time, periods=3, freq="5s")
        self.chid = ["Ex1"]
        self.kp_data = None
        self.latitude = latitude
        self.longitude = longitude
        self.data["Ex1"] = Channel(
            "Ex1",
            pd.Series([1.0, 2.0, 3.0], index=self.datetime_index),
            self,
        )

    def restore_data(self, ch: str):
        return None


def test_great_circle_distance_returns_zero_for_same_point():
    assert great_circle_distance(39.9, 116.4, 39.9, 116.4) == 0.0


def test_great_circle_distance_between_beijing_and_shanghai_is_reasonable():
    distance_km = great_circle_distance(39.9042, 116.4074, 31.2304, 121.4737, unit="km")
    assert 1050.0 < distance_km < 1100.0


def test_station_distance_uses_em_data_lat_lon():
    station_a = DemoData("A", 39.9042, 116.4074)
    station_b = DemoData("B", 31.2304, 121.4737)

    distance_km = station_distance(station_a, station_b)
    assert 1050.0 < distance_km < 1100.0


def test_station_distance_raises_when_station_has_no_location():
    station_a = DemoData("A", None, 116.4074)
    station_b = DemoData("B", 31.2304, 121.4737)

    try:
        station_distance(station_a, station_b)
    except ValueError as exc:
        assert "station_a" in str(exc)
    else:
        raise AssertionError("expected ValueError when latitude/longitude is missing")
