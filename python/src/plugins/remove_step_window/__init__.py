"""Windowed step removal feature plugin package."""

from plugins.remove_step_window.plugin import StepWindowPlugin
from plugins.remove_step_window.editor import RemoveStepByWindow

PLUGIN_CLASS = StepWindowPlugin
EDITOR_CLASS = RemoveStepByWindow

__all__ = ["StepWindowPlugin", "RemoveStepByWindow", "PLUGIN_CLASS", "EDITOR_CLASS"]
