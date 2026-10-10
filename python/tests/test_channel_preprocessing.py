from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from PySide6.QtWidgets import QMessageBox

from base.batch_runner import BatchTarget, BatchTargetResult
from base.data_manager import DataManager
from core.em_data import EMData
from core.pipeline_model import PipelineBlock
from ui.batch_process_dialog import BatchProcessDialog
from ui.data_process_pipeline import DataProcessPipelineWidget, PipelineTrackWidget
from ui.data_tree_view import DataTreeViewer


class _Data(EMData):
    def __init__(self):
        super().__init__("/tmp/channel-preprocessing.mat")
        self.name = "channel preprocessing"
        self.start_time = pd.Timestamp("2026-01-01", tz="Asia/Shanghai")
        self.datetime_index = pd.date_range(self.start_time, periods=12, freq="s")
        self.end_time = self.datetime_index[-1]
        self.dt = pd.Timedelta(seconds=1)
        self.kp_data = None
        values = 1000.0 + 3.0 * np.arange(12, dtype=np.float64)
        values[[3, 8]] = np.nan
        self.data = xr.Dataset(
            {"Ex": (("time",), values)},
            coords={"time": self.datetime_index},
        )

    def restore_data(self, channel: str):
        del channel


def _menu_action(menu, text: str):
    for action in menu.actions():
        if action.text() == text:
            return action
        submenu = action.menu()
        if submenu is not None:
            found = _menu_action(submenu, text)
            if found is not None:
                return found
    return None


class _PendingPipeline:
    def __init__(self, pending: bool = True):
        self.pending = pending
        self.resets: list[tuple[int, str | None]] = []

    def has_blocks_for_channel(self, data_id: int, channel: str) -> bool:
        del data_id, channel
        return self.pending

    def reset_restored_data(self, data_id: int, channel: str | None) -> None:
        self.resets.append((data_id, channel))
        self.pending = False


def test_immediate_preprocessing_captures_once_and_undoes_all(qtbot):
    manager = DataManager()
    data = _Data()
    data_id = manager.add(data)
    original = data.get_channel_series("Ex").copy()
    viewer = DataTreeViewer(manager)
    qtbot.addWidget(viewer)

    menu, _actions = viewer._build_channel_context_menu(data_id, "Ex")
    assert not _menu_action(menu, "撤销该通道全部处理").isEnabled()

    viewer._demean_channel(data_id, "Ex")
    demeaned = data.get_channel_series("Ex")
    assert abs(float(np.nanmean(demeaned))) < 1e-12
    np.testing.assert_array_equal(np.isnan(demeaned), np.isnan(original))
    assert manager.has_channel_processing_baseline(data_id, "Ex")

    viewer._detrend_channel(data_id, "Ex")
    detrended = data.get_channel_series("Ex")
    finite = np.isfinite(detrended)
    assert abs(float(np.polyfit(np.flatnonzero(finite), detrended[finite], 1)[0])) < 1e-10

    menu, _actions = viewer._build_channel_context_menu(data_id, "Ex")
    assert _menu_action(menu, "撤销该通道全部处理").isEnabled()
    viewer._undo_all_channel_processing(data_id, "Ex")
    pd.testing.assert_series_equal(data.get_channel_series("Ex"), original)
    assert not manager.has_channel_processing_baseline(data_id, "Ex")


def test_immediate_preprocessing_confirmation_controls_pending_blocks(qtbot, monkeypatch):
    manager = DataManager()
    data = _Data()
    data_id = manager.add(data)
    original = data.get_channel_series("Ex").copy()
    viewer = DataTreeViewer(manager)
    qtbot.addWidget(viewer)
    pending = _PendingPipeline()
    viewer.data_process_widget = pending

    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.No,
    )
    viewer._demean_channel(data_id, "Ex")
    pd.testing.assert_series_equal(data.get_channel_series("Ex"), original)
    assert pending.resets == []
    assert not manager.has_channel_processing_baseline(data_id, "Ex")

    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    viewer._demean_channel(data_id, "Ex")
    assert pending.resets == [(data_id, "Ex")]
    assert manager.has_channel_processing_baseline(data_id, "Ex")


def test_confirmed_preprocessing_clears_pending_blocks_when_values_do_not_change(
    qtbot,
    monkeypatch,
):
    manager = DataManager()
    data = _Data()
    data_id = manager.add(data)
    zero = data.get_channel_series("Ex").copy()
    zero.loc[:] = 0.0
    data.set_channel_series("Ex", zero)
    viewer = DataTreeViewer(manager)
    qtbot.addWidget(viewer)
    pending = _PendingPipeline()
    viewer.data_process_widget = pending
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )

    viewer._demean_channel(data_id, "Ex")

    assert pending.resets == [(data_id, "Ex")]
    assert not manager.has_channel_processing_baseline(data_id, "Ex")


def test_pipeline_commit_captures_first_processing_baseline(qtbot, monkeypatch):
    manager = DataManager()
    data = _Data()
    data_id = manager.add(data)
    original = data.get_channel_series("Ex").copy()
    widget = DataProcessPipelineWidget(manager)
    qtbot.addWidget(widget)
    widget.init_data(data, data_id)
    pipeline = widget._find_pipeline(data_id, "Ex")
    assert pipeline is not None
    source = pipeline.baseline_snapshot.copy()
    pipeline.layers[0].blocks.append(PipelineBlock(
        tool_id="median_filter",
        start=0,
        end=source.size,
        params={},
        status="ready",
        input_slice=source.copy(),
        output_slice=source + 5.0,
    ))
    monkeypatch.setattr(QMessageBox, "information", lambda *_args, **_kwargs: None)

    widget._commit_preview()

    assert manager.has_channel_processing_baseline(data_id, "Ex")
    manager.undo_all_channel_processing(data_id, "Ex")
    pd.testing.assert_series_equal(data.get_channel_series("Ex"), original)


def test_batch_processing_captures_first_processing_baseline(qtbot):
    manager = DataManager()
    data = _Data()
    data_id = manager.add(data)
    original = data.get_channel_series("Ex").copy()
    dialog = BatchProcessDialog(manager)
    qtbot.addWidget(dialog)
    target = BatchTarget(data_id, "Ex", original)
    result = BatchTargetResult(
        target=target,
        status="ok",
        series=original + 7.0,
    )

    dialog._on_result(result)

    assert manager.has_channel_processing_baseline(data_id, "Ex")
    assert manager.undo_all_channel_processing(data_id, "Ex")
    pd.testing.assert_series_equal(data.get_channel_series("Ex"), original)


def test_channel_rename_carries_processing_baseline():
    manager = DataManager()
    data = _Data()
    data_id = manager.add(data)
    original = data.get_channel_series("Ex").copy()
    assert manager.capture_channel_processing_baseline(data_id, "Ex")
    data.rename_channel("Ex", "ElectricX")

    manager.rename_channel_processing_baseline(data_id, "Ex", "ElectricX")

    assert not manager.has_channel_processing_baseline(data_id, "Ex")
    assert manager.has_channel_processing_baseline(data_id, "ElectricX")
    changed = data.get_channel_series("ElectricX") + 2.0
    data.set_channel_series("ElectricX", changed)
    assert manager.undo_all_channel_processing(data_id, "ElectricX")
    pd.testing.assert_series_equal(
        data.get_channel_series("ElectricX"),
        original.rename("ElectricX"),
    )


def test_global_processing_block_spans_and_locks_full_timeline(qtbot):
    manager = DataManager()
    data = _Data()
    data_id = manager.add(data)
    widget = DataProcessPipelineWidget(manager)
    qtbot.addWidget(widget)
    widget.init_data(data, data_id)
    pipeline = widget._find_pipeline(data_id, "Ex")
    assert pipeline is not None

    widget._on_tool_dropped(
        data_id,
        "Ex",
        pipeline.layers[0].layer_id,
        "detrend",
        7,
    )
    block = pipeline.layers[0].blocks[0]
    assert (block.start, block.end) == (0, data.npts)

    track = PipelineTrackWidget(pipeline, pipeline.layers[0], data.npts, widget._snap, widget)
    qtbot.addWidget(track)
    track.set_zoom(6.0, pixels_per_point=20.0)
    found, mode = track._hit_test(100)
    assert found is block
    assert mode == "locked"
