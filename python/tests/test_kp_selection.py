from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from PySide6.QtWidgets import QMessageBox

from base.data_manager import DataManager
from base.plugin_runtime import run_plugin_process
from core.algorithms.kp_selection import (
    format_kp_condition,
    normalize_kp_selection_params,
    select_kp_samples,
)
from core.em_data import EMData
from plugin_api import ChannelBlock
from plugins.demean.plugin import DemeanPlugin
from plugins.kp_selection.plugin import KpSelectionPlugin
from ui.data_process_pipeline import DataProcessPipelineWidget
from ui.data_tree_view import DataTreeViewer
from ui.models.processing_tools import create_processing_tool_model


def _params(**changes):
    values = {
        "lower_enabled": True,
        "lower_value": 2.0,
        "lower_inclusive": True,
        "upper_enabled": True,
        "upper_value": 5.0,
        "upper_inclusive": False,
        "minimum_duration_hours": 0.0,
        "trim_start_hours": 0.0,
        "trim_end_hours": 0.0,
        "missing_policy": "exclude",
    }
    values.update(changes)
    return values


def _kp_data():
    return pd.DataFrame({
        "Kp_datetime": pd.to_datetime(
            [
                "2026-01-01 03:00",
                "2026-01-01 06:00",
                "2026-01-01 09:00",
                "2026-01-01 12:00",
            ],
            utc=True,
        ),
        "Kp": [1.0, 3.0, 5.0, np.nan],
    })


def test_kp_range_uses_preceding_half_open_three_hour_intervals():
    index = pd.date_range("2026-01-01", periods=13, freq="h", tz="UTC")

    result = select_kp_samples(index, _kp_data(), _params())

    np.testing.assert_array_equal(
        np.flatnonzero(result.keep_mask),
        np.array([3, 4, 5]),
    )
    assert result.raw_duration_hours == 3.0
    assert result.selected_duration_hours == 3.0
    assert format_kp_condition(_params()) == "2 ≤ Kp < 5"


def test_kp_trim_then_minimum_duration_and_missing_policy():
    index = pd.date_range("2026-01-01", periods=13, freq="h", tz="UTC")
    kp = _kp_data()
    kp.loc[0, "Kp"] = 3.0

    retained = select_kp_samples(
        index,
        kp,
        _params(
            trim_start_hours=1.0,
            trim_end_hours=1.0,
            minimum_duration_hours=4.0,
        ),
    )
    np.testing.assert_array_equal(
        np.flatnonzero(retained.keep_mask),
        np.array([1, 2, 3, 4]),
    )

    dropped = select_kp_samples(
        index,
        kp,
        _params(
            trim_start_hours=1.0,
            trim_end_hours=1.0,
            minimum_duration_hours=4.1,
        ),
    )
    assert not dropped.keep_mask.any()

    missing_kept = select_kp_samples(
        index,
        _kp_data(),
        _params(
            lower_value=8.0,
            upper_enabled=False,
            missing_policy="keep",
        ),
    )
    np.testing.assert_array_equal(
        np.flatnonzero(missing_kept.keep_mask),
        np.array([9, 10, 11, 12]),
    )


def test_kp_equal_bounds_require_closed_interval():
    values = _params(
        lower_value=3.0,
        upper_value=3.0,
        lower_inclusive=True,
        upper_inclusive=True,
    )
    normalized = normalize_kp_selection_params(values)
    assert format_kp_condition(normalized) == "3 ≤ Kp ≤ 3"

    with pytest.raises(ValueError, match="同时包含等号"):
        normalize_kp_selection_params({
            **values,
            "upper_inclusive": False,
        })


def test_kp_plugin_masks_outside_and_downstream_demean_preserves_nan():
    index = pd.date_range("2026-01-01", periods=13, freq="h", tz="UTC")
    source = pd.Series(np.arange(13, dtype=np.float64), index=index, name="Ex")
    block = ChannelBlock(
        "Ex",
        source,
        auxiliary={"kp_data": _kp_data()},
    )

    selected, _ = run_plugin_process(KpSelectionPlugin(), block, _params())
    finite = np.flatnonzero(np.isfinite(selected.series.to_numpy()))
    np.testing.assert_array_equal(finite, np.array([3, 4, 5]))

    demeaned, _ = run_plugin_process(
        DemeanPlugin(),
        ChannelBlock("Ex", selected.series),
        {},
    )
    np.testing.assert_array_equal(
        np.isnan(demeaned.series.to_numpy()),
        np.isnan(selected.series.to_numpy()),
    )


class _KpData(EMData):
    def __init__(self):
        super().__init__("/tmp/kp-selection.mat")
        self.name = "kp selection"
        self.start_time = pd.Timestamp("2026-01-01", tz="UTC")
        self.datetime_index = pd.date_range(
            self.start_time,
            periods=13,
            freq="h",
        )
        self.end_time = self.datetime_index[-1]
        self.dt = pd.Timedelta(hours=1)
        self.kp_data = _kp_data()
        self.data = xr.Dataset(
            {
                "Ex": (("time",), np.arange(13, dtype=np.float64)),
                "Hy": (("time",), np.arange(13, dtype=np.float64) + 100.0),
            },
            coords={"time": self.datetime_index},
        )

    def restore_data(self, channel: str):
        del channel


class _FailingKpData(_KpData):
    def __init__(self):
        super().__init__()
        self.fail_next_hy_write = True

    def set_channel_series(self, channel_name, series):
        if channel_name == "Hy" and self.fail_next_hy_write:
            self.fail_next_hy_write = False
            raise RuntimeError("injected write failure")
        super().set_channel_series(channel_name, series)


def test_immediate_kp_selection_is_dataset_wide_and_undoable(qtbot):
    manager = DataManager()
    data = _KpData()
    data_id = manager.add(data)
    originals = {
        channel: data.get_channel_series(channel).copy(deep=True)
        for channel in data.channel_names()
    }
    viewer = DataTreeViewer(manager)
    qtbot.addWidget(viewer)

    assert viewer._apply_kp_selection_to_dataset(data_id, _params())

    for channel in data.channel_names():
        values = data.get_channel_series(channel).to_numpy()
        np.testing.assert_array_equal(
            np.flatnonzero(np.isfinite(values)),
            np.array([3, 4, 5]),
        )
        assert manager.has_channel_processing_baseline(data_id, channel)

    assert manager.undo_all_dataset_processing(data_id)
    for channel, original in originals.items():
        pd.testing.assert_series_equal(data.get_channel_series(channel), original)


def test_immediate_kp_selection_rolls_back_every_channel_on_write_failure(
    qtbot,
    monkeypatch,
):
    manager = DataManager()
    data = _FailingKpData()
    data_id = manager.add(data)
    originals = {
        channel: data.get_channel_series(channel).copy(deep=True)
        for channel in data.channel_names()
    }
    viewer = DataTreeViewer(manager)
    qtbot.addWidget(viewer)
    monkeypatch.setattr(QMessageBox, "critical", lambda *_args, **_kwargs: None)

    assert not viewer._apply_kp_selection_to_dataset(data_id, _params())

    for channel, original in originals.items():
        pd.testing.assert_series_equal(data.get_channel_series(channel), original)
        assert not manager.has_channel_processing_baseline(data_id, channel)


def test_kp_tool_is_global_and_spans_full_pipeline(qtbot):
    manager = DataManager()
    data = _KpData()
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
        "kp_selection",
        7,
    )

    block = pipeline.layers[0].blocks[0]
    assert block.tool_id == "kp_selection"
    assert (block.start, block.end) == (0, data.npts)

    widget._compose_channel_preview(
        data_id,
        "Ex",
        update_status=True,
    )
    qtbot.waitUntil(
        lambda: block.status in {"ready", "error"},
        timeout=5000,
    )
    assert block.status == "ready", block.last_summary
    preview = widget._compose_channel_preview(data_id, "Ex")
    np.testing.assert_array_equal(
        np.flatnonzero(np.isfinite(preview)),
        np.arange(6),
    )

    model = create_processing_tool_model()
    global_item = model.item(1)
    ids = [
        global_item.child(row).data(257)
        for row in range(global_item.rowCount())
    ]
    assert "kp_selection" in ids
