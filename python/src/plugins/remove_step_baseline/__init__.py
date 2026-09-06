"""Robust-baseline step removal feature plugin package."""

from plugins.remove_step_baseline.plugin import StepBaselinePlugin
from plugins.remove_step_baseline.editor import RemoveStepByBaseline

PLUGIN_CLASS = StepBaselinePlugin
EDITOR_CLASS = RemoveStepByBaseline

__all__ = ["StepBaselinePlugin", "RemoveStepByBaseline", "PLUGIN_CLASS", "EDITOR_CLASS"]
