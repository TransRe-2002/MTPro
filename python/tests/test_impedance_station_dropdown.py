from __future__ import annotations

from PySide6.QtWidgets import QApplication

from base.data_manager import DataManager
from test_impedance_split_stations import Station
from ui.station_output_host import StationOutputHost


def test_all_output_station_menus_refresh_on_open_without_loading_waveforms(qtbot, monkeypatch):
    manager = DataManager()
    first = Station('first', ['Ex', 'Ey', 'Bx', 'By'])
    second = Station('second', ['Ex', 'Ey', 'Hx', 'Hy'])
    first_id, second_id = manager.add(first), manager.add(second)
    host = StationOutputHost(manager)
    qtbot.addWidget(host)
    host.init_data(first, first_id)
    robust = host.open_output_window('robust estimate', first_id).widget()
    remote = host.open_output_window('remote reference', first_id).widget()
    cross = host.open_output_window('cross_station_impedance', first_id).widget()
    third = Station('third', ['Ex', 'Ey'])
    third_id = manager.add(third)
    # Simulate a changed label without emitting a data-manager update. The popup
    # must query the manager itself, not rely on its old station package.
    second.name = 'renamed'
    monkeypatch.setattr('ui.station_output_host.station_data_package',
                        lambda *_args: (_ for _ in ()).throw(AssertionError('loaded waveform')))
    combos = [robust.station_combo, remote.local_station_combo, remote.remote_station_combo,
              cross.station_combo, cross.magnetic_station_combo]
    for combo in combos:
        before = combo.currentData()
        combo.showPopup()
        QApplication.processEvents()
        assert combo.currentData() == before
        assert 'renamed' in combo.itemText(combo.findData(second_id))
        combo.hidePopup()
    assert cross.station_combo.findData(third_id) >= 0
    assert cross.magnetic_station_combo.findData(third_id) == -1
    assert remote.local_station_combo.findData(third_id) == -1
    host.close()


def test_removed_remote_station_clears_old_input_and_requires_reselection(qtbot):
    manager = DataManager()
    first = Station('local', ['Ex', 'Ey', 'Bx', 'By'])
    second = Station('remote', ['Hx', 'Hy'])
    first_id, second_id = manager.add(first), manager.add(second)
    host = StationOutputHost(manager)
    qtbot.addWidget(host)
    host.init_data(first, first_id)
    widget = host.open_output_window('remote reference', first_id).widget()
    assert widget.remote_station_combo.currentData() == second_id
    assert widget._remote_pkg
    manager.remove(second_id)
    assert widget.remote_station_combo.currentData() is None
    assert not widget._remote_pkg
    assert '重新选择' in widget.status_label.text()
    host.close()


def test_cross_station_can_reselect_removed_electric_and_magnetic_sources(qtbot):
    manager = DataManager()
    electric = Station('electric', ['Ex', 'Ey'])
    magnetic = Station('magnetic', ['Bx', 'By'])
    eid, mid = manager.add(electric), manager.add(magnetic)
    host = StationOutputHost(manager)
    qtbot.addWidget(host)
    host.init_data(electric, eid)
    widget = host.open_output_window('cross_station_impedance', eid).widget()
    new_e = Station('new electric', ['Ex', 'Ey'])
    new_h = Station('new magnetic', ['Bx', 'By'])
    new_eid, new_mid = manager.add(new_e), manager.add(new_h)
    manager.remove(eid)
    manager.remove(mid)
    assert widget.station_combo.currentData() is None
    assert widget.magnetic_station_combo.currentData() is None
    widget.magnetic_station_combo.setCurrentIndex(widget.magnetic_station_combo.findData(new_mid))
    widget.station_combo.setCurrentIndex(widget.station_combo.findData(new_eid))
    assert widget._pkg['name'] == 'new electric'
    assert widget._magnetic_pkg['name'] == 'new magnetic'
    assert widget.ch_combo_ex.currentData() == 'Ex'
    assert widget.ch_combo_hx.currentData() == 'Bx'
    host.close()
