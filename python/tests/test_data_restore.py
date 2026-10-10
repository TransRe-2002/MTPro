"""MAT restore regression tests using generated local files, no external data/network."""

import numpy as np
import pandas as pd
import pytest
from concurrent.futures import Future
from pathlib import Path
from scipy.io import savemat
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox, QDialog

from base.data_manager import DataManager
from base.data_restore import load_restore_source, restore_mat_data, station_metadata_differs
from core.mat_data import MatEMData
from core.pipeline_model import PipelineBlock
from io_utils.mat_io import MatLoader
from ui.data_process import DataProcessWidget
from ui.data_tree_view import DataTreeViewer
from ui.mainwindow import MainWindow
from ui.station_output_host import StationOutputHost


@pytest.fixture
def mat_source(tmp_path, monkeypatch):
    monkeypatch.setattr("core.mat_data.fetch_kp", lambda *args, **kwargs: None)
    path = tmp_path / "source.mat"
    values = np.arange(31, dtype=float)
    values[[2, 3, 8, 17]] = np.nan
    savemat(path, {"tsStruct": {
        "name": "original", "dt": 1.0, "npts": len(values), "NCh": 2,
        "startTime": [[2024, 1, 1, 0, 0, 0]],
        "timeZone": "UTC", "chid": np.array([["Ex", "Hy"]], dtype=object),
        "data": np.vstack([values, values + 100]),
        "latitude": 30.0, "longitude": 120.0, "elevation": 50.0,
    }})
    return MatLoader.load(str(path))


def _manager(data):
    manager = DataManager()
    key = manager.add(data)
    return manager, key


def test_channel_restore_replays_trim_and_repeated_decimation(mat_source):
    data = mat_source
    original = load_restore_source(data)
    data.trim_time_range(data.datetime_index[1], data.datetime_index[28])
    data.decimate(2, max_missing_fraction=0.5)
    data.trim_time_range(data.datetime_index[1], data.datetime_index[-2])
    data.decimate(3, max_missing_fraction=0.75)
    expected = data.get_channel_series("Ex").copy()
    other = data.get_channel_series("Hy").copy()
    data.rename_channel("Ex", "renamed")
    data.rename_channel("renamed", "custom")
    data.set_channel_data("custom", np.zeros(data.npts))
    manager, key = _manager(data)
    assert manager.capture_channel_processing_baseline(key, "custom")
    events = []
    manager.data_restored.connect(lambda *args: events.append(("restored", *args)))
    manager.data_changed.connect(lambda *args: events.append(("changed", *args)))

    restore_mat_data(manager, key, original, channel="custom")

    pd.testing.assert_series_equal(data.get_channel_series("custom"), expected.rename("custom"))
    pd.testing.assert_series_equal(data.get_channel_series("Hy"), other)
    assert data.dt == pd.Timedelta(seconds=6)
    assert not manager.has_channel_processing_baseline(key, "custom")
    assert events == [("restored", key, "custom"), ("changed", key, False)]


@pytest.mark.parametrize("keep_metadata", [False, True])
def test_dataset_restore_preserves_identity_and_restores_structure(mat_source, keep_metadata):
    data = mat_source
    disk_bytes = Path(data.path).read_bytes()
    original = load_restore_source(data)
    manager, key = _manager(data)
    assert manager.capture_channel_processing_baseline(key, "Ex")
    data.rename_channel("Ex", "custom")
    data.remove_channel("Hy")
    data.decimate(3)
    data.trim_time_range(data.datetime_index[2], data.datetime_index[-2])
    data.set_metadata(name="edited", latitude=35.0, longitude=110.0, elevation=99.0)
    restore_mat_data(manager, key, original, keep_station_metadata=keep_metadata)
    assert manager.get(key) is data
    assert data.channel_names() == ["Ex", "Hy"]
    assert data.dt == original.dt
    assert data.datetime_index.equals(original.datetime_index)
    np.testing.assert_equal(data.data["Ex"].values, original.data["Ex"].values)
    assert data.name == ("edited" if keep_metadata else "original")
    assert data.latitude == (35.0 if keep_metadata else 30.0)
    assert data.elevation == (99.0 if keep_metadata else 50.0)
    assert data._source_time_operations == []
    assert data._source_channels == {"Ex": "Ex", "Hy": "Hy"}
    assert not manager.has_channel_processing_baseline(key, "Ex")
    assert Path(data.path).read_bytes() == disk_bytes


@pytest.mark.parametrize("failure", ["missing_channel", "missing_time", "upsampling", "fractional"])
def test_incompatible_source_never_mutates_current_data(mat_source, failure):
    source = load_restore_source(mat_source)
    if failure == "missing_channel":
        source.remove_channel("Ex")
    elif failure == "missing_time":
        source.trim_time_range(source.datetime_index[2], None)
    elif failure == "upsampling":
        source.decimate(2)
    else:
        source.dt = pd.Timedelta(milliseconds=600)
    before = mat_source.data.copy(deep=True)
    manager, key = _manager(mat_source)
    events = []
    manager.data_restored.connect(lambda *args: events.append(args))
    with pytest.raises(ValueError):
        restore_mat_data(manager, key, source, channel="Ex")
    assert mat_source.data.identical(before)
    assert events == []


def test_source_reload_uses_disk_not_mutable_mat_meta(mat_source):
    expected = mat_source.get_channel_series("Ex").copy()
    mat_source.set_channel_data("Ex", np.zeros(mat_source.npts))
    mat_source.update_meta()
    mat_source.restore_data("Ex", source=load_restore_source(mat_source))
    pd.testing.assert_series_equal(mat_source.get_channel_series("Ex"), expected)


def test_metadata_missing_values_compare_equal(mat_source):
    source = load_restore_source(mat_source)
    mat_source.latitude = np.nan
    source.latitude = None
    assert not station_metadata_differs(mat_source, source)
    source.name = "changed"
    assert station_metadata_differs(mat_source, source)


@pytest.mark.parametrize("answer", [QMessageBox.StandardButton.Yes, QMessageBox.StandardButton.No])
def test_channel_dialog_confirms_mismatched_rate(mat_source, qtbot, monkeypatch, answer):
    data = mat_source
    data.decimate(2)
    expected = data.get_channel_series("Ex").copy()
    data.set_channel_data("Ex", np.full(data.npts, -1.0))
    before = data.data.copy(deep=True)
    manager, key = _manager(data)
    viewer = DataTreeViewer(manager)
    qtbot.addWidget(viewer)
    prompts = []
    monkeypatch.setattr(QMessageBox, "question", lambda *args: prompts.append(args[2]) or answer)
    viewer._restore_channel(key, "Ex")
    assert "采样率和原始数据不匹配，是否进行处理？" in prompts[0]
    if answer == QMessageBox.StandardButton.Yes:
        pd.testing.assert_series_equal(data.get_channel_series("Ex"), expected)
    else:
        assert data.data.identical(before)


@pytest.mark.parametrize("failure", ["missing", "corrupt"])
def test_unreadable_source_shows_warning_without_mutation(mat_source, qtbot, monkeypatch, tmp_path, failure):
    if failure == "missing":
        mat_source.path = str(tmp_path / "not_found.mat")
    else:
        path = tmp_path / "corrupt.mat"
        path.write_bytes(b"not a mat file")
        mat_source.path = str(path)
        monkeypatch.setattr("io_utils.ts_convert.convert_to_struct", lambda *args: (_ for _ in ()).throw(ValueError("bad MAT")))
    before = mat_source.data.copy(deep=True)
    manager, key = _manager(mat_source)
    viewer = DataTreeViewer(manager)
    qtbot.addWidget(viewer)
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    viewer._restore_dataset(key)
    assert warnings
    assert mat_source.data.identical(before)


def test_restore_clears_only_affected_pipeline_and_rejects_late_results(mat_source, qtbot):
    manager, key = _manager(mat_source)
    widget = DataProcessWidget(manager)
    qtbot.addWidget(widget)
    widget.init_data(mat_source, key)
    pipeline = widget._create_pipeline_window()
    affected = pipeline._find_pipeline(key, "Ex")
    other = pipeline._find_pipeline(key, "Hy")
    affected_block = PipelineBlock(tool_id="remove_spike", start=0, end=10)
    other_block = PipelineBlock(tool_id="remove_spike", start=0, end=10)
    affected.layers[0].blocks.append(affected_block)
    other.layers[0].blocks.append(other_block)
    pipeline._plugin_tasks[affected_block.block_id] = 99
    editor = QDialog(pipeline)
    editor.setProperty("pipeline_block_id", affected_block.block_id)
    pipeline._register_editor_dialog(editor)
    editor.show()
    mat_source.set_channel_data("Ex", np.zeros(mat_source.npts))
    restore_mat_data(manager, key, load_restore_source(mat_source), channel="Ex")
    restored = pipeline._find_pipeline(key, "Ex")
    assert all(not layer.blocks for layer in restored.layers)
    np.testing.assert_equal(restored.baseline_snapshot, mat_source.data["Ex"].values)
    assert pipeline._find_pipeline(key, "Hy") is other
    assert other.layers[0].blocks == [other_block]
    assert not editor.isVisible()
    assert affected_block.block_id not in pipeline._plugin_tasks
    pipeline._on_plugin_task_done(((affected_block.block_id, 99, None, None), None))


@pytest.mark.parametrize("button", ["恢复源文件信息", "保留当前信息", "取消还原"])
def test_dataset_metadata_dialog_choices_are_atomic(mat_source, qtbot, button):
    manager, key = _manager(mat_source)
    widget = DataTreeViewer(manager)
    qtbot.addWidget(widget)
    mat_source.set_metadata(name="edited", elevation=200.0)
    mat_source.decimate(2)
    before = mat_source.data.copy(deep=True)

    def choose():
        dialog = QApplication.activeModalWidget()
        assert isinstance(dialog, QMessageBox)
        next(item for item in dialog.buttons() if item.text() == button).click()

    QTimer.singleShot(0, choose)
    widget._restore_dataset(key)
    if button == "取消还原":
        assert mat_source.data.identical(before)
        assert mat_source.dt == pd.Timedelta(seconds=2)
    else:
        assert mat_source.npts == 31
        assert mat_source.dt == pd.Timedelta(seconds=1)
    keep = button != "恢复源文件信息"
    assert mat_source.name == ("edited" if keep else "original")
    assert mat_source.elevation == (200.0 if keep else 50.0)


def test_restore_menu_is_only_available_for_mat_sources(mat_source, qtbot):
    from core.xlsx_data import XlsxEMData

    manager, key = _manager(mat_source)
    other = XlsxEMData(
        {"Ex": mat_source.get_channel_series("Ex")},
        timezone="UTC", dt=mat_source.dt, path="other.xlsx",
    )
    other_key = manager.add(other)
    widget = DataTreeViewer(manager)
    qtbot.addWidget(widget)
    for builder in (widget._build_dataset_context_menu, widget._build_channel_context_menu):
        menu, _ = builder(key)
        assert "从源文件还原…" in [action.text() for action in menu.actions()]
        menu, _ = builder(other_key)
        assert "从源文件还原…" not in [action.text() for action in menu.actions()]


@pytest.mark.parametrize("tool", ["robust estimate", "remote reference"])
def test_restore_refreshes_impedance_plot_and_cancels_old_estimate(mat_source, qtbot, tool):
    manager, key = _manager(mat_source)
    mat_source.decimate(2)
    expected = mat_source.get_channel_series("Ex").to_numpy(copy=True)
    mat_source.set_channel_data("Ex", np.zeros(mat_source.npts))
    host = StationOutputHost(manager)
    qtbot.addWidget(host)
    widget = host.open_output_window(tool, key).widget()
    pyramids = lambda: widget._lod_pyramids if tool == "robust estimate" else widget._lod_local
    curves = lambda: widget._curves if tool == "robust estimate" else widget._curves_local
    qtbot.waitUntil(lambda: "Ex" in pyramids())
    future = Future()
    widget._estimation_future = future
    generation = widget._task_generation

    restore_mat_data(manager, key, load_restore_source(mat_source), channel="Ex")

    # Both impedance tools intentionally remove the mean for display only.
    centered = expected - np.nanmean(expected)
    qtbot.waitUntil(lambda: "Ex" in pyramids() and np.array_equal(pyramids()["Ex"].y_raw, centered, equal_nan=True))
    np.testing.assert_equal(curves()["Ex"].getData()[1], centered)
    package = widget._pkg if tool == "robust estimate" else widget._local_pkg
    np.testing.assert_equal(package["all_y"]["Ex"], expected)
    assert widget._estimation_future is None
    assert future.cancelled()
    assert widget._task_generation > generation
    assert "还原" in widget.status_label.text()
    # Stale results must be discarded before they access result contents.
    widget._on_estimation_done((generation, None, None))
    host.close()


def test_restore_refreshes_active_waveform(mat_source, qtbot):
    manager, key = _manager(mat_source)
    expected = mat_source.data["Ex"].values.copy()
    mat_source.set_channel_data("Ex", np.zeros(mat_source.npts))
    window = MainWindow(manager)
    qtbot.addWidget(window)
    manager.set_active(key)
    viewer = window.data_viewers[key]
    qtbot.waitUntil(lambda: "Ex" in viewer._pyramids)
    restore_mat_data(manager, key, load_restore_source(mat_source), channel="Ex")
    qtbot.waitUntil(lambda: np.array_equal(viewer._pyramids["Ex"].y_raw, expected, equal_nan=True))
    assert window.data_viewers[key] is viewer
    window.close()


def test_dataset_restore_resets_all_open_pipelines_but_preserves_other_datasets(mat_source, qtbot):
    manager, key = _manager(mat_source)
    other_data = load_restore_source(mat_source)
    other_key = manager.add(other_data)
    mat_source.rename_channel("Ex", "custom")
    mat_source.remove_channel("Hy")
    mat_source.decimate(2)
    host = DataProcessWidget(manager)
    qtbot.addWidget(host)
    host.init_data(mat_source, key)
    first = host._create_pipeline_window()
    first.add_channels([(other_key, "Ex")])
    unrelated = first._find_pipeline(other_key, "Ex")
    block = PipelineBlock(tool_id="remove_spike", start=0, end=3)
    unrelated.layers[0].blocks.append(block)
    host._create_pipeline_window()
    second = host.pipeline_widget
    assert second is not first
    host.pipeline_panel._create_workspace()
    third = host._create_pipeline_window()
    assert third is not second
    empty_workspace = host.pipeline_panel._create_workspace()
    restore_mat_data(manager, key, load_restore_source(mat_source))
    assert empty_workspace.subWindowList() == []
    for widget in (first, second, third):
        assert widget._find_pipeline(key, "custom") is None
        for channel in ("Ex", "Hy"):
            lane = widget._find_pipeline(key, channel)
            assert all(not layer.blocks for layer in lane.layers)
            assert len(lane.baseline_snapshot) == mat_source.npts
    assert first._find_pipeline(other_key, "Ex") is unrelated
    assert unrelated.layers[0].blocks == [block]
