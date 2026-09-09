from __future__ import annotations

import numpy as np
import pandas as pd

from base.batch_runner import BatchRunner, BatchTarget
from base.plugin_loader import load_all_plugins
from core.pipeline_model import ChannelPipeline, PipelineBlock
from core.recipe import Recipe, RecipeStep, load_recipe, recipe_from_pipeline, save_recipe


def _plugin_map():
    report = load_all_plugins(scan_paths=[], include_entry_points=False)
    return {item.info.plugin_id: item.plugin for item in report.loaded}


def _series(values, freq="s"):
    return pd.Series(
        values,
        index=pd.date_range("2025-01-01", periods=len(values), freq=freq, tz="Asia/Shanghai"),
        name="Ex",
    )


def test_recipe_toml_roundtrip_handles_nested_editor_operations(tmp_path):
    recipe = Recipe(
        "cleanup",
        (
            RecipeStep(
                "remove_step_window",
                {
                    "threshold": 10.0,
                    "operations": [{"nan_indices": [2], "delta_runs": [[3, [-4.0, -4.0]]], "absolute": []}],
                },
                ("2025-01-01T00:00:00+08:00", "2025-01-01T00:00:05+08:00"),
                1.0,
            ),
        ),
    )
    path = save_recipe(recipe, tmp_path / "cleanup.toml")

    assert load_recipe(path) == recipe


def test_recipe_exports_pipeline_in_execution_order():
    index = _series(range(10)).index
    pipeline = ChannelPipeline(1, "Ex")
    first = pipeline.add_layer()
    first.blocks.extend([
        PipelineBlock(tool_id="remove_spike", start=5, end=8, params={"indices": [0]}),
        PipelineBlock(tool_id="remove_spike", start=0, end=3, params={"indices": [1]}),
    ])
    second = pipeline.add_layer()
    second.blocks.append(PipelineBlock(tool_id="remove_region", start=2, end=4, params={"time_ranges": []}))

    recipe = recipe_from_pipeline(pipeline, index, "from pipeline")

    assert [step.time_range for step in recipe.steps] == [
        (index[0].isoformat(), index[2].isoformat()),
        (index[5].isoformat(), index[7].isoformat()),
        (index[2].isoformat(), index[3].isoformat()),
    ]


def test_batch_runner_applies_steps_serially_and_isolates_target_failure():
    plugins = _plugin_map()
    recipe = Recipe(
        "remove points",
        (
            RecipeStep("remove_spike", {"indices": [1]}),
            RecipeStep("remove_spike", {"indices": [2]}),
        ),
    )
    good = BatchTarget(1, "Ex", _series([1.0, 2.0, 3.0, 4.0]))
    broken_recipe = Recipe("broken", (RecipeStep("missing.plugin", {}),))
    runner = BatchRunner(plugins)

    good_result = runner.run_target(recipe, good)
    bad_result = runner.run_target(broken_recipe, good)

    assert good_result.status == "ok"
    np.testing.assert_allclose(good_result.series, [1.0, np.nan, np.nan, 4.0], equal_nan=True)
    assert bad_result.status == "error"
    assert "missing.plugin" in bad_result.error


def test_batch_runner_skips_point_indices_on_sample_interval_mismatch():
    runner = BatchRunner(_plugin_map())
    target = BatchTarget(1, "Ex", _series([1.0, 2.0, 3.0], freq="2s"))
    recipe = Recipe(
        "aligned source",
        (RecipeStep("remove_spike", {"indices": [1]}, sample_interval_seconds=1.0),),
    )

    result = runner.run_target(recipe, target)

    assert result.status == "ok"
    np.testing.assert_allclose(result.series, target.series)
    assert result.skipped_steps


def test_batch_submit_runs_multiple_targets():
    runner = BatchRunner(_plugin_map())
    recipe = Recipe("remove", (RecipeStep("remove_spike", {"indices": [0]}),))
    targets = [
        BatchTarget(1, "Ex", _series([1.0, 2.0])),
        BatchTarget(2, "Hy", _series([3.0, 4.0]).rename("Hy")),
    ]

    futures = runner.submit(recipe, targets)
    results = [future.result(timeout=5) for future in futures.values()]

    assert {result.target.key for result in results} == {(1, "Ex"), (2, "Hy")}
    assert all(np.isnan(result.series.iloc[0]) for result in results)


def test_batch_runner_uses_a_contiguous_slice_for_sorted_time_ranges():
    runner = BatchRunner(_plugin_map())
    target = BatchTarget(1, "Ex", _series([1.0, 2.0, 3.0, 4.0]))
    selection = runner._selection(
        target.series.index,
        (
            target.series.index[1].isoformat(),
            target.series.index[2].isoformat(),
        ),
    )
    recipe = Recipe(
        "remove one point in range",
        (
            RecipeStep(
                "remove_spike",
                {"indices": [0]},
                (
                    target.series.index[1].isoformat(),
                    target.series.index[2].isoformat(),
                ),
            ),
        ),
    )

    result = runner.run_target(recipe, target)

    assert selection == slice(1, 3)
    np.testing.assert_allclose(
        result.series,
        [1.0, np.nan, 3.0, 4.0],
        equal_nan=True,
    )
    np.testing.assert_allclose(target.series, [1.0, 2.0, 3.0, 4.0])
