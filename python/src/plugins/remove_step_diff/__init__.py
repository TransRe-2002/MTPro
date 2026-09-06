"""Step-difference removal feature plugin package."""

from plugins.remove_step_diff.plugin import StepDiffPlugin
from plugins.remove_step_diff.editor import RemoveStepByDiff

PLUGIN_CLASS = StepDiffPlugin
EDITOR_CLASS = RemoveStepByDiff

__all__ = ["StepDiffPlugin", "RemoveStepByDiff", "PLUGIN_CLASS", "EDITOR_CLASS"]
