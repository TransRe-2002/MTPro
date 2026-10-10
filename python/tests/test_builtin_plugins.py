from __future__ import annotations

import json

import numpy as np
import pandas as pd

from base.plugin_loader import load_all_plugins, validate_plugin_info
from base.plugin_runtime import run_plugin_process
from builtin_plugins.edit_codec import apply_operations, encode_edit
from plugin_api import ChannelBlock, complete_params


def _block(values) -> ChannelBlock:
    index = pd.date_range("2025-01-01", periods=len(values), freq="s", tz="Asia/Shanghai")
    return ChannelBlock("Ex", pd.Series(values, index=index, name="Ex"))


def _plugins():
    report = load_all_plugins(scan_paths=[], include_entry_points=False)
    assert report.failed == []
    return report.by_id()


def test_all_eighteen_builtin_plugins_validate_and_have_json_defaults():
    plugins = _plugins()

    assert set(plugins) == {
        "remove_region",
        "remove_spike",
        "auto_spike_filter",
        "emd_denoise",
        "demean",
        "detrend",
        "kp_selection",
        "median_filter",
        "morphological_filter",
        "iir_filter",
        "svd_denoise",
        "remove_secular_variation",
        "remove_step_diff",
        "remove_step_window",
        "remove_step_baseline",
        "remove_wavelet_denoise",
        "remove_diurnal_sinusoid",
        "vmd_denoise",
    }
    for loaded in plugins.values():
        validate_plugin_info(loaded.info)
        json.dumps(complete_params(loaded.info, {}))


def test_builtin_plugin_entries_live_in_their_feature_packages():
    plugins = _plugins()

    expected_modules = {
        "remove_region": "plugins.remove_region.plugin",
        "remove_spike": "plugins.remove_spike.plugin",
        "auto_spike_filter": "plugins.auto_spike_filter.plugin",
        "emd_denoise": "plugins.emd_denoise.plugin",
        "demean": "plugins.demean.plugin",
        "detrend": "plugins.detrend.plugin",
        "kp_selection": "plugins.kp_selection.plugin",
        "median_filter": "plugins.median_filter.plugin",
        "morphological_filter": "plugins.morphological_filter.plugin",
        "iir_filter": "plugins.iir_filter.plugin",
        "svd_denoise": "plugins.svd_denoise.plugin",
        "remove_secular_variation": "plugins.remove_secular_variation.plugin",
        "remove_step_diff": "plugins.remove_step_diff.plugin",
        "remove_step_window": "plugins.remove_step_window.plugin",
        "remove_step_baseline": "plugins.remove_step_baseline.plugin",
        "remove_wavelet_denoise": "plugins.remove_wavelet_denoise.plugin",
        "remove_diurnal_sinusoid": "plugins.remove_diurnal_sinusoid.plugin",
        "vmd_denoise": "plugins.vmd_denoise.plugin",
    }

    assert {
        plugin_id: loaded.plugin.__class__.__module__
        for plugin_id, loaded in plugins.items()
    } == expected_modules


def test_mutable_parameter_defaults_are_not_shared():
    plugin = _plugins()["remove_spike"].plugin
    first = complete_params(plugin.info, {})
    second = complete_params(plugin.info, {})

    first["indices"].append(3)

    assert second["indices"] == []


def test_spike_and_region_plugins_replay_serialized_decisions():
    plugins = _plugins()
    block = _block([1.0, 2.0, 3.0, 4.0])

    spike, _ = run_plugin_process(
        plugins["remove_spike"].plugin, block, {"indices": [1, 99]}
    )
    region, _ = run_plugin_process(
        plugins["remove_region"].plugin,
        block,
        {"time_ranges": [[block.series.index[2].isoformat(), block.series.index[3].isoformat()]]},
    )

    np.testing.assert_allclose(spike.series.to_numpy(), [1.0, np.nan, 3.0, 4.0], equal_nan=True)
    np.testing.assert_allclose(region.series.to_numpy(), [1.0, 2.0, np.nan, np.nan], equal_nan=True)


def test_diff_plugin_replays_selected_compact_diff_indices():
    plugin = _plugins()["remove_step_diff"].plugin
    block = _block([1.0, 2.0, 12.0, 13.0])

    result, params = run_plugin_process(plugin, block, {"indices": [1]})

    np.testing.assert_allclose(result.series.to_numpy(), [1.0, 2.0, 2.0, 3.0])
    assert params["indices"] == [1]


def test_edit_codec_exactly_replays_finite_and_nan_changes():
    source = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    changed = np.array([2.5, np.nan, 7.0, 8.0, 5.0])
    edit = encode_edit(source, changed)

    replayed = apply_operations(source, [edit])

    np.testing.assert_allclose(replayed, changed, equal_nan=True)
    json.dumps(edit)


def test_auto_plugins_accept_exact_editor_operation_replay():
    plugins = _plugins()
    source = np.array([1.0, 2.0, 10.0, 11.0])
    target = np.array([1.0, 2.0, 2.0, 3.0])
    edit = encode_edit(source, target)
    block = _block(source)

    for plugin_id in (
        "remove_step_window",
        "remove_step_baseline",
        "remove_wavelet_denoise",
        "remove_diurnal_sinusoid",
    ):
        result, params = run_plugin_process(
            plugins[plugin_id].plugin,
            block,
            {"operations": [edit]},
        )
        np.testing.assert_allclose(result.series.to_numpy(), target)
        json.dumps(params)
