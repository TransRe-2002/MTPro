from __future__ import annotations

from ui.tools.registry import ToolSpec, register
from output_tools.robust_estimate import RobustEstimateWidget
from output_tools.remote_reference import RemoteReferenceWidget
from output_tools.cross_station_impedance import CrossStationImpedanceWidget

register(ToolSpec(
    tool_id="cross_station_impedance",
    display_name="跨台站阻抗估计",
    category="output",
    kind="param",
    requires_channel=False,
    output_factory=CrossStationImpedanceWidget,
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
    output_factory=RobustEstimateWidget,
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
    output_factory=RemoteReferenceWidget,
))
