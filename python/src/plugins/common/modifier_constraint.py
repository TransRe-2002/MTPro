from __future__ import annotations

"""共享的修饰键轴向约束逻辑，供数据处理 PlotWidget 使用。"""

from PySide6 import QtCore


def apply_modifier_axis_constraint(view_box, modifiers) -> None:
    """Ctrl constrains interaction to X; Shift constrains it to Y."""
    if modifiers & QtCore.Qt.KeyboardModifier.ControlModifier:
        view_box.setMouseEnabled(x=True, y=False)
    elif modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier:
        view_box.setMouseEnabled(x=False, y=True)
    else:
        view_box.setMouseEnabled(x=True, y=True)
