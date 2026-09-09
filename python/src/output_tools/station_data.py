from __future__ import annotations

import numpy as np

from base.data_manager import DataManager
from core.station_metadata import StationMetadata
from utils.series import dti_to_numpy


def station_data_package(data_manager: DataManager, data_id: int) -> dict | None:
    """Build the read-only station payload required by impedance-output tools."""

    em_data = data_manager.get(data_id)
    if em_data is None:
        return None

    channel_names = em_data.channel_names()
    datetime_index = em_data.datetime_index
    x_values = (
        dti_to_numpy(em_data.get_channel_series(channel_names[0]).index)
        if channel_names
        else np.array([])
    )
    stations = []
    for station_id, station_data in data_manager.items():
        station_channels = station_data.channel_names()
        stations.append({
            "id": station_id,
            "name": station_data.name,
            "has_magnetic": any(channel.startswith(("H", "B")) for channel in station_channels),
            "has_electric": any(channel.startswith("E") for channel in station_channels),
            "latitude": station_data.latitude,
            "longitude": station_data.longitude,
            "elevation": station_data.elevation,
        })

    return {
        "x": x_values,
        "y": np.array([]),
        "channel_names": channel_names,
        # Output tools treat these arrays as immutable.  Keeping views avoids a
        # full multi-channel copy on the Qt thread when a large station opens.
        "all_y": {
            channel: em_data.get_channel_series(channel).to_numpy(dtype=np.float64, copy=False)
            for channel in channel_names
        },
        "npts": em_data.npts,
        "start_time": em_data.start_time.timestamp(),
        "dt": em_data.dt.total_seconds(),
        "kp_data": em_data.kp_data,
        "stations": stations,
        "datetime_index": datetime_index,
        "name": em_data.name,
        "latitude": em_data.latitude,
        "longitude": em_data.longitude,
        "elevation": em_data.elevation,
        "station_metadata": StationMetadata(
            name=em_data.name,
            latitude=em_data.latitude,
            longitude=em_data.longitude,
            elevation=em_data.elevation,
        ),
    }
