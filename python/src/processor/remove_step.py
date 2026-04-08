from processor.remove_step_by_diff import RemoveStepByDiff
from processor.remove_step_by_window import RemoveStepByWindow
from processor.step_plot_widget import StepPlotWidget

# Backward-compatible exports for existing imports.
RemoveStep = RemoveStepByDiff
RemoveStepPlotWidget = StepPlotWidget
WindowedDeStep = RemoveStepByWindow
