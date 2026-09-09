from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QApplication

from ui.data_process_pipeline import PreviewChannelSlice
from ui.tools import registry


BUILTIN_PLUGIN_IDS = (
    "remove_region",
    "remove_spike",
    "auto_spike_filter",
    "emd_denoise",
    "remove_step_diff",
    "remove_step_window",
    "remove_step_baseline",
    "remove_wavelet_denoise",
    "remove_diurnal_sinusoid",
    "remove_secular_variation",
    "vmd_denoise",
    "median_filter",
    "morphological_filter",
    "detrend",
    "iir_filter",
)


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_all_builtin_plugin_editors_create_from_pipeline_slice():
    app = _app()
    index = pd.date_range(
        "2025-01-01",
        periods=4096,
        freq="s",
        tz="Asia/Shanghai",
    )
    series = pd.Series(
        np.sin(np.arange(len(index), dtype=np.float64) / 17.0),
        index=index,
        name="Ex",
    )
    preview = PreviewChannelSlice("Ex", series.iloc[500:1_500].copy())

    for plugin_id in BUILTIN_PLUGIN_IDS:
        editor = registry.create_pipeline_editor(
            plugin_id,
            preview,
            None,
            params={},
        )
        assert editor is not None, plugin_id
        assert editor._loaded, plugin_id
        assert editor._editor is not None, plugin_id
        assert editor._host.block.series.index.equals(preview.cts.index), plugin_id

        # Visual editors now receive this range directly;
        # no DataManager/Broker selection is involved in their construction.
        slice_editor = getattr(editor._editor, "editor", None)
        if plugin_id in {
            "remove_region",
            "remove_spike",
            "auto_spike_filter",
            "remove_step_diff",
            "remove_step_window",
            "remove_step_baseline",
            "remove_wavelet_denoise",
            "remove_diurnal_sinusoid",
        }:
            samples = getattr(slice_editor, "y_data", None)
            if samples is None:
                samples = getattr(slice_editor, "_source", None)
            assert samples is not None, plugin_id
            assert len(samples) == len(preview.cts), plugin_id
            assert slice_editor.data_id == 0, plugin_id
        editor.close()
        editor.deleteLater()
        app.processEvents()
