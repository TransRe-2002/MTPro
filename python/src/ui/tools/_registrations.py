from __future__ import annotations

from ui.tools.registry import ToolSpec, register
from processor.remove_region import RemoveRegion
from processor.remove_spike import RemoveSpike
from processor.remove_wavelet_denoise import RemoveWaveletDenoise
from processor.remove_diurnal_sinusoid import RemoveDiurnalSinusoid
from processor.remove_step_by_baseline import RemoveStepByBaseline
from processor.remove_step_by_diff import RemoveStepByDiff
from processor.remove_step_by_window import RemoveStepByWindow
from processor.robust_estimate import RobustEstimateWidget
from processor.remote_reference import RemoteReferenceWidget


register(ToolSpec(
    tool_id="remove_region",
    display_name="remove region",
    category="manual",
    kind="interactive",
    requires_channel=True,
    has_channel_sync=True,
    has_region_sync=True,
    classic_factory=RemoveRegion,
    pipeline_editor_factory=RemoveRegion,
))

register(ToolSpec(
    tool_id="remove_spike",
    display_name="remove spike",
    category="manual",
    kind="interactive",
    requires_channel=True,
    has_channel_sync=True,
    classic_factory=RemoveSpike,
    pipeline_editor_factory=RemoveSpike,
))

register(ToolSpec(
    tool_id="remove_step_diff",
    display_name="remove step diff",
    category="manual",
    kind="interactive",
    requires_channel=True,
    aliases=("remove step", "remove_step"),
    classic_factory=RemoveStepByDiff,
    pipeline_editor_factory=RemoveStepByDiff,
))

register(ToolSpec(
    tool_id="remove_step_window",
    display_name="remove step window",
    category="auto",
    kind="interactive",
    requires_channel=True,
    classic_factory=RemoveStepByWindow,
    pipeline_editor_factory=RemoveStepByWindow,
))

register(ToolSpec(
    tool_id="remove_step_baseline",
    display_name="remove step baseline",
    category="auto",
    kind="interactive",
    requires_channel=True,
    classic_factory=RemoveStepByBaseline,
    pipeline_editor_factory=RemoveStepByBaseline,
))

register(ToolSpec(
    tool_id="remove_wavelet_denoise",
    display_name="remove wavelet denoise",
    category="auto",
    kind="interactive",
    requires_channel=True,
    aliases=("remove pulse wavelet",),
    classic_factory=RemoveWaveletDenoise,
    pipeline_editor_factory=RemoveWaveletDenoise,
))

register(ToolSpec(
    tool_id="remove_diurnal_sinusoid",
    display_name="remove diurnal sinusoid",
    category="auto",
    kind="interactive",
    requires_channel=True,
    classic_factory=RemoveDiurnalSinusoid,
    pipeline_editor_factory=RemoveDiurnalSinusoid,
))

register(ToolSpec(
    tool_id="robust_estimate",
    display_name="robust estimate",
    category="output",
    kind="param",
    requires_channel=False,
    fields=(
        ("low", "low", float, 0.01),
        ("high", "high", float, 1.0),
    ),
    classic_factory=RobustEstimateWidget,
))

register(ToolSpec(
    tool_id="remote_reference",
    display_name="remote reference",
    category="output",
    kind="param",
    requires_channel=False,
    fields=(
        ("order", "order", int, 1),
    ),
    classic_factory=RemoteReferenceWidget,
))