from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import xarray as xr

from core.em_data import EMData
from ui.data_view_widget import DataViewWidget


def test_data_view_pyramid_reuses_raw_channel_values():
    x = np.arange(20_000, dtype=np.float64)
    values = 12.5 + np.sin(x / 20.0)

    pyramid = DataViewWidget._build_channel_pyramid(x, values)
    center, scale = DataViewWidget._display_parameters(values)

    assert np.shares_memory(pyramid.y_raw, values)
    assert center == np.mean(values)
    assert scale > 0.0



class DemoData(EMData):
    def __init__(self, npts: int = 128):
        super().__init__("/tmp/data-view-performance.mat")
        self.name = "demo"
        self.dt = pd.Timedelta(seconds=1)
        self.start_time = pd.Timestamp("2024-01-01", tz="UTC")
        self.datetime_index = pd.date_range(
            self.start_time, periods=npts, freq=self.dt,
        )
        self.end_time = self.datetime_index[-1]
        self.kp_data = None
        self.data = xr.Dataset(
            {
                "Ex": (("time",), np.linspace(0.0, 1.0, npts)),
                "Hy": (("time",), np.linspace(1.0, 2.0, npts)),
            },
            coords={"time": self.datetime_index},
        )

    def restore_data(self, ch: str):
        return None


def test_changed_channel_rebuild_keeps_other_pyramid(monkeypatch, qtbot):
    data = DemoData()
    widget = DataViewWidget()
    qtbot.addWidget(widget)
    widget.init_data(data, build_plots=True)

    old_ex = widget._pyramids["Ex"]
    old_hy = widget._pyramids["Hy"]

    def run_immediately(
        _pool, task, *args, on_done=None, on_error=None, owner=None, **kwargs,
    ):
        try:
            result = task(*args)
        except BaseException as exc:
            if on_error is not None:
                on_error(exc)
            return None
        if on_done is not None:
            on_done(result)
        return None

    monkeypatch.setattr("ui.data_view_widget.run_in_pool", run_immediately)
    changed = np.linspace(10.0, 20.0, data.npts)
    data.set_channel_data("Ex", changed)

    widget.on_change_data_finished("Ex")

    assert widget._pyramids["Ex"] is not old_ex
    assert widget._pyramids["Hy"] is old_hy
    np.testing.assert_array_equal(widget._pyramids["Ex"].y_raw, changed)

def _queued_pool(jobs):
    def queue(
        _pool, task, *args, on_done=None, on_error=None, owner=None, **kwargs,
    ):
        jobs.append((task, args, on_done, on_error))
        return None

    return queue


def _complete_job(job):
    task, args, on_done, on_error = job
    try:
        result = task(*args)
    except BaseException as exc:
        if on_error is not None:
            on_error(exc)
    else:
        if on_done is not None:
            on_done(result)


def test_stale_channel_build_cannot_overwrite_newer_values(monkeypatch, qtbot):
    data = DemoData()
    widget = DataViewWidget()
    qtbot.addWidget(widget)
    widget.init_data(data, build_plots=True)

    jobs = []
    monkeypatch.setattr("ui.data_view_widget.run_in_pool", _queued_pool(jobs))

    first = np.linspace(10.0, 20.0, data.npts)
    data.set_channel_data("Ex", first)
    widget.on_change_data_finished("Ex")

    latest = np.linspace(30.0, 40.0, data.npts)
    data.set_channel_data("Ex", latest)
    widget.on_change_data_finished("Ex")

    _complete_job(jobs[1])
    _complete_job(jobs[0])

    np.testing.assert_array_equal(widget._pyramids["Ex"].y_raw, latest)


def test_stale_full_build_cannot_overwrite_newer_generation(monkeypatch, qtbot):
    data = DemoData()
    widget = DataViewWidget()
    qtbot.addWidget(widget)
    widget.init_data(data, build_plots=False)

    jobs = []
    monkeypatch.setattr("ui.data_view_widget.run_in_pool", _queued_pool(jobs))

    widget.start_display_build()
    latest = np.linspace(50.0, 60.0, data.npts)
    data.set_channel_data("Ex", latest)
    widget.start_display_build()

    _complete_job(jobs[1])
    _complete_job(jobs[0])

    np.testing.assert_array_equal(widget._pyramids["Ex"].y_raw, latest)

