from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication

from base.data_manager import DataManager
from test_impedance_split_stations import Station
from ui.station_output_host import StationOutputHost


def _station(name, channels, unit):
    data = Station(name, channels, start='2026-06-08 01:24:13', count=4096, dt=1)
    data.datetime_index = data.datetime_index.as_unit(unit)
    data.start_time, data.end_time = data.datetime_index[0], data.datetime_index[-1]
    data.data = data.data.assign_coords(time=data.datetime_index)
    return data


@pytest.mark.parametrize('unit', ['ns', 'us', 'ms'])
@pytest.mark.parametrize('tool', ['robust estimate', 'cross_station_impedance', 'remote reference'])
def test_output_time_axis_uses_real_dates_and_selection(qtbot, unit, tool):
    manager = DataManager()
    local = _station('local', ['Ex', 'Ey', 'Bx', 'By'], unit)
    remote = _station('remote', ['Bx', 'By'], unit)
    local_id, remote_id = manager.add(local), manager.add(remote)
    host = StationOutputHost(manager)
    qtbot.addWidget(host)
    host.init_data(local, local_id)
    widget = host.open_output_window(tool, local_id).widget()
    QApplication.processEvents()
    expected = np.array([stamp.timestamp() for stamp in local.datetime_index])
    np.testing.assert_allclose(widget._ts_array, expected, rtol=0, atol=1e-6)
    assert widget._ts_range_to_indices(expected[0], expected[-1]) == (0, len(expected))
    assert widget._ts_range_to_indices(expected[100], expected[200]) == (100, 200)
    plot = widget.local_plot if tool == 'remote reference' else widget.plot_widget
    bounds = plot.getViewBox().viewRange()[0]
    assert bounds[0] <= expected[0] and bounds[1] >= expected[-1]
    assert plot.getAxis('bottom').utcOffset == 0
    assert widget.start_dt.dateTime().toSecsSinceEpoch() == int(expected[0])
    if tool == 'remote reference':
        assert widget.remote_station_combo.currentData() == remote_id
        np.testing.assert_allclose(widget._ts_remote_array, expected, rtol=0, atol=1e-6)
        assert widget.local_plot.getAxis('top').style['showValues'] is False
        assert widget.remote_plot.getAxis('top').style['showValues'] is False
        remote_bounds = widget.remote_plot.getViewBox().viewRange()[0]
        np.testing.assert_allclose(remote_bounds, bounds, rtol=0, atol=1e-3)
    widget._set_region_from_indices(100, 200)
    assert widget._ts_range_to_indices(*widget.region_item.getRegion()) == (100, 200)
    widget._set_dt_from_indices(100, 200)
    assert widget.start_dt.dateTime().toSecsSinceEpoch() == int(expected[100])
    assert widget.end_dt.dateTime().toSecsSinceEpoch() == int(expected[199])
    host.close()
