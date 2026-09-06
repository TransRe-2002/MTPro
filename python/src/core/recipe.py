"""Serializable processing recipes built from plugin parameter mappings."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import tomllib
from typing import Any, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class RecipeStep:
    plugin: str
    params: dict[str, Any] = field(default_factory=dict)
    time_range: tuple[str, str] | None = None
    sample_interval_seconds: float | None = None


@dataclass(frozen=True)
class Recipe:
    name: str
    steps: tuple[RecipeStep, ...] = ()


def recipe_from_pipeline(pipeline, index: pd.DatetimeIndex, name: str) -> Recipe:
    steps: list[RecipeStep] = []
    sample_interval = None
    if len(index) > 1:
        sample_interval = float((index[1] - index[0]).total_seconds())
    for layer in pipeline.layers:
        for block in layer.sorted_blocks():
            start = max(0, min(int(block.start), len(index)))
            end = max(start, min(int(block.end), len(index)))
            time_range = None
            if start > 0 or end < len(index):
                if start < end:
                    time_range = (index[start].isoformat(), index[end - 1].isoformat())
            steps.append(
                RecipeStep(
                    plugin=block.tool_id,
                    params=dict(block.params),
                    time_range=time_range,
                    sample_interval_seconds=sample_interval,
                )
            )
    return Recipe(name=name, steps=tuple(steps))


def load_recipe(path: str | Path) -> Recipe:
    with Path(path).expanduser().open("rb") as fh:
        raw = tomllib.load(fh)
    section = raw.get("recipe")
    if not isinstance(section, Mapping):
        raise ValueError("recipe file must contain a [recipe] table")
    name = str(section.get("name", "")).strip()
    if not name:
        raise ValueError("recipe.name must not be empty")
    raw_steps = raw.get("steps", [])
    if not isinstance(raw_steps, list):
        raise ValueError("steps must be an array of tables")
    steps: list[RecipeStep] = []
    for raw_step in raw_steps:
        if not isinstance(raw_step, Mapping):
            raise ValueError("each recipe step must be a table")
        plugin = str(raw_step.get("plugin", "")).strip()
        if not plugin:
            raise ValueError("recipe step plugin must not be empty")
        raw_range = raw_step.get("time_range", [])
        if raw_range in (None, []):
            time_range = None
        elif isinstance(raw_range, list) and len(raw_range) == 2:
            time_range = (str(raw_range[0]), str(raw_range[1]))
        else:
            raise ValueError("time_range must be empty or contain two timestamps")
        raw_params = raw_step.get("params", {})
        if not isinstance(raw_params, Mapping):
            raise ValueError("step.params must be a table")
        interval = raw_step.get("sample_interval_seconds")
        steps.append(
            RecipeStep(
                plugin=plugin,
                params=dict(raw_params),
                time_range=time_range,
                sample_interval_seconds=(float(interval) if interval is not None else None),
            )
        )
    return Recipe(name=name, steps=tuple(steps))


def save_recipe(recipe: Recipe, path: str | Path) -> Path:
    target = Path(path).expanduser()
    lines = ["[recipe]", f"name = {_toml_value(recipe.name)}", ""]
    for step in recipe.steps:
        lines.append("[[steps]]")
        lines.append(f"plugin = {_toml_value(step.plugin)}")
        lines.append(f"time_range = {_toml_value(list(step.time_range or ())) }")
        if step.sample_interval_seconds is not None:
            lines.append(f"sample_interval_seconds = {float(step.sample_interval_seconds)!r}")
        lines.append(f"params = {_toml_value(step.params)}")
        lines.append("")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def _toml_value(value: Any) -> str:
    if value is None:
        raise TypeError("TOML has no null value")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("recipe parameters must not contain non-finite floats")
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, Mapping):
        items = []
        for key, item in value.items():
            key_text = str(key)
            if not key_text:
                raise ValueError("recipe parameter keys must not be empty")
            items.append(f"{json.dumps(key_text, ensure_ascii=False)} = {_toml_value(item)}")
        return "{ " + ", ".join(items) + " }"
    raise TypeError(f"unsupported recipe value: {type(value).__name__}")
