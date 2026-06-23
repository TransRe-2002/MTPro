from __future__ import annotations

"""共享的修饰键轴向约束逻辑，供 StepPlotWidget 和 RegionSelectionPlotWidget 使用。"""

from PySide6 import QtCore


def apply_modifier_axis_constraint(view_box, modifiers) -> None:
    """根据 Shift/Alt 修饰键决定是双轴缩放还是仅 X/Y 轴缩放。"""
    if modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier:
        view_box.setMouseEnabled(x=True, y=False)
    elif modifiers & QtCore.Qt.KeyboardModifier.AltModifier:
        view_box.setMouseEnabled(x=False, y=True)
    else:
        view_box.setMouseEnabled(x=True, y=True)
