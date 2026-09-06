from __future__ import annotations

import os
from pathlib import Path
import time
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import xarray as xr
from PySide6.QtWidgets import QApplication

from base.data_manager import DataManager
from base.plugin_loader import load_all_plugins
from core.em_data import EMData
from core.pipeline_model import PipelineBlock
from core.recipe import load_recipe, recipe_from_pipeline, save_recipe
from ui.data_process_pipeline import DataProcessPipelineWidget, refresh_tool_meta
from ui.tools import registry


class DemoData(EMData):
    def __init__(self):
        super().__init__("/tmp/plugin-e2e.mat")
        self.npts = 9
        self.start_time = pd.Timestamp("2025-01-01", tz="Asia/Shanghai")
        self.dt = pd.Timedelta(seconds=1)
        self.datetime_index = pd.date_range(
            self.start_time,
            periods=self.npts,
            freq="s",
        )
        self.end_time = self.datetime_index[-1]
        self.kp_data = None
        self.data = xr.Dataset(coords={"time": self.datetime_index})
        self.data["Ex"] = xr.DataArray(
            np.arange(self.npts, dtype=np.float64),
            dims=("time",),
            coords={"time": self.datetime_index},
        )

    def restore_data(self, _channel: str):
        return None


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_third_party_plugin_runs_through_pipeline_commit_and_recipe(tmp_path):
    app = _app()
    docs_root = Path(__file__).resolve().parents[2] / "docs"
    report = load_all_plugins(
        scan_paths=[docs_root],
        builtin_classes=[],
        include_entry_points=False,
    )
    assert report.failed == []
    assert [item.info.plugin_id for item in report.loaded] == [
        "example.rolling_median"
    ]

    registry.register_loaded_plugins(report.loaded)
    refresh_tool_meta()
    spec = registry.get("example.rolling_median")
    assert spec is not None
    assert spec.kind == "param"
    assert registry.get_plugin("example.rolling_median") is not None

    manager = DataManager()
    data = DemoData()
    data_id = manager.add(data)
    widget = DataProcessPipelineWidget(manager)
    widget.init_data(data, data_id)
    pipeline = widget.pipelines[0]
    block = PipelineBlock(
        tool_id="example.rolling_median",
        start=0,
        end=data.npts,
        params={"window": 3},
        status="configured",
    )
    pipeline.layers[0].blocks.append(block)

    deadline = time.monotonic() + 5.0
    while block.status != "ready" and time.monotonic() < deadline:
        widget._collect_preview_results(update_status=True)
        app.processEvents()
        time.sleep(0.01)

    assert block.status == "ready", block.last_summary
    preview = widget._collect_preview_results(update_status=False)[pipeline.ref]
    np.testing.assert_allclose(
        preview,
        [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.5],
    )

    recipe = recipe_from_pipeline(pipeline, data.datetime_index, "rolling median")
    recipe_path = save_recipe(recipe, tmp_path / "rolling-median.toml")
    assert load_recipe(recipe_path) == recipe

    with patch("ui.data_process_pipeline.QMessageBox.information"):
        widget._commit_preview()
    np.testing.assert_allclose(
        data.get_channel_series("Ex").to_numpy(dtype=np.float64),
        [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.5],
    )
    widget.close()
