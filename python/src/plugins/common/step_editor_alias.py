from plugins.remove_step_diff.editor import RemoveStepByDiff
from plugins.remove_step_window.editor import RemoveStepByWindow
from plugins.common.plotting import StepPlotWidget

# Backward-compatible exports for existing imports.
RemoveStep = RemoveStepByDiff
RemoveStepPlotWidget = StepPlotWidget
WindowedDeStep = RemoveStepByWindow
