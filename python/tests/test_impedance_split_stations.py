from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from PySide6.QtWidgets import QMessageBox

from base.data_manager import DataManager
from core.em_data import EMData
from ui.station_output_host import StationOutputHost


class Station(EMData):
    def __init__(self, name, channels, start='2025-01-01', count=4096, dt=5):
        super().__init__('/tmp/synthetic-station.mat')
        self.name = name
        self.datetime_index = pd.date_range(start, periods=count, freq=f'{dt}s', tz='UTC')
        self.start_time, self.end_time = self.datetime_index[0], self.datetime_index[-1]
        self.dt = pd.Timedelta(seconds=dt)
        self.kp_data = None
        self.latitude = self.longitude = self.elevation = None
        self.data = xr.Dataset({ch: (('time',), np.arange(count, dtype=float) + i * 10000)
                                for i, ch in enumerate(channels)}, coords={'time': self.datetime_index})

    def restore_data(self, channel):
        pass


def _open(qtbot, electric, magnetic):
    manager = DataManager()
    electric_id = manager.add(electric)
    magnetic_id = manager.add(magnetic)
    host = StationOutputHost(manager)
    qtbot.addWidget(host)
    host.init_data(electric, electric_id)
    widget = host.open_output_window('cross_station_impedance', electric_id).widget()
    return manager, host, widget, electric_id, magnetic_id


def test_cross_station_uses_distinct_electric_and_magnetic_stations(qtbot, monkeypatch):
    electric = Station('electric', ['Ex', 'Ey'])
    magnetic = Station('magnetic', ['Bx', 'By'], start='2025-01-01 00:08:20')
    _, host, widget, eid, mid = _open(qtbot, electric, magnetic)
    assert widget.station_combo.currentData() == eid
    assert widget.magnetic_station_combo.currentData() == mid
    assert widget.ch_combo_ex.currentData() == 'Ex'
    assert widget.ch_combo_hx.currentData() == 'Bx'
    assert (widget.start_spin.value(), widget.end_spin.value()) == (100, 4096)
    submitted = []
    monkeypatch.setattr('output_tools.cross_station_impedance.run_in_pool', lambda *args, **kw: submitted.append(args) or object())
    widget._run_estimation()
    assert submitted
    ex, ey, hx, hy, hz = submitted[0][3]
    np.testing.assert_array_equal(ex, electric.data['Ex'].values[100:])
    np.testing.assert_array_equal(hx, magnetic.data['Bx'].values[:3996])
    assert hz is None
    assert widget._pending_context['parameters']['electric_station_id'] == eid
    assert widget._pending_context['parameters']['magnetic_station_id'] == mid
    widget._estimation_future = None
    host.close()


def test_cross_station_refuses_shifted_sample_grid(qtbot, monkeypatch):
    electric = Station('electric', ['Ex', 'Ey'])
    magnetic = Station('magnetic', ['Bx', 'By'], start='2025-01-01 00:00:01')
    _, host, widget, _, _ = _open(qtbot, electric, magnetic)
    errors = []
    monkeypatch.setattr(QMessageBox, 'critical', lambda *args: errors.append(args[2]))
    widget._run_estimation()
    assert errors and '未逐点时间对齐' in errors[0]
    assert widget._estimation_future is None
    host.close()



def test_cross_station_button_is_independent_from_original_tools(qtbot):
    from ui.impedance_output import ImpedanceOutputWidget
    from output_tools.robust_estimate import RobustEstimateWidget
    from output_tools.cross_station_impedance import CrossStationImpedanceWidget

    manager = DataManager()
    data = Station('combined', ['Ex', 'Ey', 'Bx', 'By'])
    data_id = manager.add(data)
    page = ImpedanceOutputWidget(manager)
    qtbot.addWidget(page)
    page.init_data(data, data_id)
    page.robust_button.click()
    page.cross_station_button.click()
    windows = page.host.mdi_area.subWindowList()
    assert len(windows) == 2
    assert type(windows[0].widget()) is RobustEstimateWidget
    assert isinstance(windows[1].widget(), CrossStationImpedanceWidget)
    assert not hasattr(windows[0].widget(), 'magnetic_station_combo')
    assert windows[1].widget().magnetic_station_combo.currentData() == data_id
    page.discard_and_close_all()


def test_cross_station_refreshes_after_magnetic_station_is_removed(qtbot):
    manager, host, widget, eid, mid = _open(qtbot, Station('E', ['Ex', 'Ey']), Station('H', ['Bx', 'By']))
    manager.remove(mid)
    assert widget.station_combo.currentData() == eid
    assert widget.magnetic_station_combo.count() == 0
    assert not widget._magnetic_pkg
    host.close()
