"""
time_slider_qslider.py
──────────────────────
继承 QSlider 实现的非等距时间刻度滑块。

刻度（按时间顺序）：
  1h · 3h · 6h · 9h · 12h · 18h · 24h · 2d · 3d · 7d · 10d · 30d
  60d · 90d · 120d · 180d · 240d · 360d

用法
────
    slider = TimeRangeSlider(parent)
    slider.timeChanged.connect(lambda label, secs: ...)

依赖
────
    pip install PySide6
"""

import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout,
    QLabel, QFrame, QHBoxLayout, QSlider, QStyleOptionSlider, QStyle
)
from PySide6.QtCore import Qt, Signal, QRect
from PySide6.QtGui import QPainter, QColor, QFont, QFontMetrics, QPen

# ─── 刻度表（标签, 秒数）───────────────────────────────────────────────────────
TICKS = [
    ("3h",    3   * 3_600),
    ("6h",    6   * 3_600),
    ("9h",    9   * 3_600),
    ("12h",   12  * 3_600),
    ("18h",   18  * 3_600),
    ("24h",   24  * 3_600),
    ("2d",    2   * 86_400),
    ("3d",    3   * 86_400),
    ("7d",    7   * 86_400),
    ("10d",   10  * 86_400),
    ("30d",   30  * 86_400),
    ("60d",   60  * 86_400),
    ("90d",   90  * 86_400),
    ("120d",  120 * 86_400),
    ("180d",  180 * 86_400),
    ("240d",  240 * 86_400),
    ("360d",  360 * 86_400),
]
N = len(TICKS)   # 17

# ─── 主组件 ───────────────────────────────────────────────────────────────────
class TimeRangeSlider(QSlider):
    """
    继承 QSlider 的时间范围滑块。
    """

    timeChanged = Signal(int)

    _LABEL_GAP   = 6    # 刻度线底部到标签顶部的距离
    _EXTRA_BOT   = 28   # 为标签额外保留的底部高度

    def __init__(self, parent=None, default_index: int = 3):
        super().__init__(Qt.Orientation.Horizontal, parent)

        # QSlider 用整数索引 0~N-1 映射每个刻度
        self.setRange(0, N - 1)
        self.setSingleStep(1)
        self.setPageStep(1)
        self.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.setTickInterval(1)
        self.setValue(max(0, min(default_index, N - 1)))

        # 内部信号 → 对外信号
        super().valueChanged.connect(self._on_value_changed)

    # ── 公开 API ──────────────────────────────────────────────────────────────
    @property
    def current_seconds(self) -> int:
        return TICKS[self.value()][1]

    @property
    def current_label(self) -> str:
        return TICKS[self.value()][0]

    def set_index(self, index: int):
        self.setValue(max(0, min(index, N - 1)))

    # ── 内部 ──────────────────────────────────────────────────────────────────
    def _on_value_changed(self, index: int):
        secs = TICKS[index][1]
        self.timeChanged.emit(secs)

    def sizeHint(self):
        sh = super().sizeHint()
        return sh.__class__(sh.width(), sh.height() + self._EXTRA_BOT)

    def minimumSizeHint(self):
        msh = super().minimumSizeHint()
        return msh.__class__(msh.width(), msh.height() + self._EXTRA_BOT)

    # ── 绘制标签 ─────────────────────────────────────────────────────────────
    def paintEvent(self, event):
        # 1. 先让 QSlider 画它自己（groove + handle + 原生刻度线）
        super().paintEvent(event)

        # 2. 再叠加自定义标签
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        font = QFont("Consolas", 8)
        if not font.exactMatch():
            font = QFont("Courier New", 8)
        font.setWeight(QFont.Weight.Medium)
        p.setFont(font)
        fm = QFontMetrics(font)

        # 通过 QStyleOptionSlider 算出 groove 的实际像素范围
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        groove_rect: QRect = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            opt,
            QStyle.SubControl.SC_SliderGroove,
            self,
        )
        handle_rect: QRect = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            opt,
            QStyle.SubControl.SC_SliderHandle,
            self,
        )

        # groove 可用的左右像素边界（排除 handle 宽度的一半）
        half_h = handle_rect.width() // 2
        x_left = groove_rect.left() + half_h
        x_right = groove_rect.right() - half_h
        span = x_right - x_left

        # --- 修改点 1：定义上下两个基准 Y 坐标 ---
        # 计算字体高度，用于调整间距
        font_height = fm.height()
        # 上方标签的基线位置（在滑块轨道上方）
        y_top_base = groove_rect.top() - self._LABEL_GAP
        # 下方标签的基线位置（在滑块轨道下方）
        y_bottom_base = groove_rect.bottom() + self._LABEL_GAP + fm.ascent()

        cur = self.value()
        for i, (label, _) in enumerate(TICKS):
            # 每个刻度的像素 x
            px = x_left + round(i / (N - 1) * span)
            active = (i == cur)

            # --- 修改点 2：根据索引奇偶性切换位置 ---
            # 偶数索引 (0,2,4...) 显示在下方，奇数索引 (1,3,5...) 显示在上方
            if i % 2 == 0:
                y_pos = y_bottom_base
            else:
                y_pos = y_top_base

            # 标签颜色：激活刻度用高亮色，其余用柔和色
            if active:
                p.setPen(self.palette().text().color())
            else:
                muted = self.palette().text().color()
                muted.setAlphaF(0.45)
                p.setPen(muted)

            lw = fm.horizontalAdvance(label)
            lx = px - lw // 2

            # 端点标签防裁剪（如果标签跑到窗口上方，需要修正）
            # 这里简单处理，或者你可以根据 y_pos 动态调整
            lx = max(0, min(lx, self.width() - lw))

            p.drawText(lx, y_pos, label)

        p.end()


# ─── 演示窗口 ────────────────────────────────────────────────────────────────
class DemoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TimeRangeSlider（基于 QSlider）")
        self.setMinimumWidth(640)

        root = QVBoxLayout(self)
        root.setContentsMargins(32, 24, 32, 24)
        root.setSpacing(20)

        # 标题
        title = QLabel("时间范围选择器")
        title.setStyleSheet("font-size: 14px; font-weight: 600;")
        root.addWidget(title)

        # 滑块
        self.slider = TimeRangeSlider(self, default_index=3)
        root.addWidget(self.slider)

        # 信息面板
        panel = QFrame()
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        panel_layout = QHBoxLayout(panel)
        panel_layout.setContentsMargins(16, 10, 16, 10)

        self._sec = self._stat("对应秒数", panel_layout)
        self._tip = self._stat("操作", panel_layout, "拖拽 · 滚轮 · ← →")

        root.addWidget(panel)

        self.slider.timeChanged.connect(self._refresh)
        self._refresh(self.slider.current_seconds)

    def _stat(self, caption, layout, init=""):
        col = QVBoxLayout()
        cap = QLabel(caption)
        cap.setStyleSheet("font-size: 10px; color: gray;")
        val = QLabel(init)
        val.setStyleSheet("font-size: 14px; font-weight: 600;")
        col.addWidget(cap)
        col.addWidget(val)
        layout.addLayout(col)
        return val

    def _refresh(self, secs: int):
        self._sec.setText(f"{secs:,} 秒")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = DemoWindow()
    win.show()
    sys.exit(app.exec())