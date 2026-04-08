"""
Pipeline Lane Editor — PySide6 Prototype
=========================================
运行: python pipeline_prototype.py
依赖: PySide6

交互:
  - 左侧工具列表拖拽到泳道上放置工具块
  - 拖动工具块移动，拖两端调整范围
  - 双击工具块打开参数 MdiSubWindow
  - 右键工具块删除
  - "+ add step" 增加处理层
  - "▶ Preview" 模拟执行
  - "✔ Commit" 提交写回数据
"""
from __future__ import annotations
import sys, uuid, math
from dataclasses import dataclass, field
from typing import Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout, QScrollArea, QLabel,
    QPushButton, QToolBar, QSlider, QCheckBox,
    QSpinBox, QListWidget, QListWidgetItem,
    QMdiArea, QMdiSubWindow, QFormLayout,
    QComboBox, QDoubleSpinBox, QDialog,
    QDialogButtonBox, QGroupBox, QRadioButton,
    QSizePolicy, QFrame, QMessageBox
)
from PySide6.QtCore import (
    Qt, Signal, QPoint, QRect, QSize,
    QMimeData, QByteArray, QTimer
)
from PySide6.QtGui import (
    QPainter, QColor, QBrush, QPen, QFont,
    QDrag, QCursor, QPainterPath
)

# ─── 常量 ────────────────────────────────────────────────────
DATA_LEN   = 20000
LAYER_H    = 40
CH_HDR_H   = 28
ADD_ROW_H  = 24
HANDLE_W   = 7
HEADER_W   = 140

PLUGIN_META: dict[str, dict] = {
    "remove_spike": dict(name="remove spike", color="#378ADD",
        params=dict(method=dict(type="select", opts=["linear","zero"], default="linear"),
                    threshold=dict(type="float", default=3.0))),
    "detrend":      dict(name="detrend",      color="#639922",
        params=dict(degree=dict(type="int", default=1))),
    "notch_filter": dict(name="notch filter", color="#BA7517",
        params=dict(freq=dict(type="float", default=50.0),
                    q=dict(type="float", default=30.0))),
    "bandpass":     dict(name="bandpass",     color="#993556",
        params=dict(low=dict(type="float", default=0.01),
                    high=dict(type="float", default=100.0))),
    "robust":       dict(name="robust",       color="#993C1D",
        params=dict(iterations=dict(type="int", default=5),
                    confidence=dict(type="float", default=0.95))),
    "remote_ref":   dict(name="remote ref",   color="#534AB7",
        params=dict(ref_ch=dict(type="select",
                    opts=["Ex1","Ex2","Bx","By"], default="Bx"))),
}

# ─── 数据模型 ─────────────────────────────────────────────────
def uid() -> str:
    return uuid.uuid4().hex[:8]

@dataclass
class ToolBlock:
    block_id:  str  = field(default_factory=uid)
    plugin_id: str  = ""
    start:     int  = 0
    end:       int  = 0
    params:    dict = field(default_factory=dict)
    done:      bool = False

@dataclass
class ProcessLayer:
    layer_id: str = field(default_factory=uid)
    label:    str = "step 1"
    blocks:   list[ToolBlock] = field(default_factory=list)

    def sorted_blocks(self):
        return sorted(self.blocks, key=lambda b: b.start)

@dataclass
class ChannelPipeline:
    channel: str
    expanded: bool = True
    layers: list[ProcessLayer] = field(default_factory=list)

    def add_layer(self):
        l = ProcessLayer(label=f"step {len(self.layers)+1}")
        self.layers.append(l)
        return l

    def remove_layer(self, layer_id: str):
        self.layers = [l for l in self.layers if l.layer_id != layer_id]
        for i, l in enumerate(self.layers):
            l.label = f"step {i+1}"

    def find_block(self, block_id: str):
        for l in self.layers:
            for b in l.blocks:
                if b.block_id == block_id:
                    return l, b
        return None

# ─── 吸附引擎 ─────────────────────────────────────────────────
class SnapEngine:
    THRESHOLD_PX = 8

    def __init__(self):
        self.grid_pts = 500
        self.enabled  = True
        self._edges: list[int] = []

    def update_edges(self, pipeline: list[ChannelPipeline],
                     exclude_id: str = ""):
        edges = set()
        for ch in pipeline:
            for ly in ch.layers:
                for b in ly.blocks:
                    if b.block_id == exclude_id:
                        continue
                    edges.add(b.start)
                    edges.add(b.end)
        self._edges = sorted(edges)

    def snap(self, pts: int, px_per_pt: float) -> tuple[int, list[int]]:
        if not self.enabled:
            return max(0, min(DATA_LEN, pts)), []
        grid = round(pts / self.grid_pts) * self.grid_pts
        best, best_dist = grid, abs(grid - pts) * px_per_pt
        guide = []
        for e in self._edges:
            d = abs(e - pts) * px_per_pt
            if d < best_dist:
                best_dist = d; best = e
        if best_dist < self.THRESHOLD_PX:
            guide = [best]
        return max(0, min(DATA_LEN, best)), guide

# ─── 单条泳道轨道 (QWidget + QPainter) ───────────────────────
class LayerTrack(QWidget):
    block_dropped       = Signal(str, int, int)   # plugin_id, start, end
    block_double_click  = Signal(str)             # block_id
    block_removed       = Signal(str)             # block_id
    guide_changed       = Signal(list)            # guide line positions

    def __init__(self, ch_pipeline: ChannelPipeline,
                 layer: ProcessLayer,
                 snap: SnapEngine, parent=None):
        super().__init__(parent)
        self.ch   = ch_pipeline
        self.layer = layer
        self.snap  = snap
        self.zoom  = 6.0
        self.setFixedHeight(LAYER_H)
        self.setMinimumWidth(600)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)

        self._drag_block:  Optional[ToolBlock] = None
        self._drag_mode:   str  = ""
        self._drag_start_x: int = 0
        self._drag_origin_s: int = 0
        self._drag_origin_e: int = 0
        self._hover_block: Optional[ToolBlock] = None
        self._guides: list[int] = []
        self._selected_id: str = ""

    # ── 坐标换算 ──────────────────────────────────────────────
    @property
    def _px_per_pt(self) -> float:
        return self.zoom * 8 / 1000

    def _pts2px(self, pts: int) -> float:
        return pts * self._px_per_pt

    def _px2pts(self, px: int) -> int:
        return int(px / self._px_per_pt)

    def set_zoom(self, z: float):
        self.zoom = z
        self.setMinimumWidth(int(self._pts2px(DATA_LEN)) + 100)
        self.update()

    def set_selected(self, block_id: str):
        self._selected_id = block_id
        self.update()

    def set_guides(self, guides: list[int]):
        self._guides = guides
        self.update()

    # ── 绘制 ──────────────────────────────────────────────────
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()

        # 背景
        p.fillRect(0, 0, W, H, QColor("#1e1e22"))

        # 网格
        if self.snap.enabled:
            pen = QPen(QColor("#2a2a32"), 0.5, Qt.DotLine)
            p.setPen(pen)
            pts = 0
            while pts <= DATA_LEN:
                x = int(self._pts2px(pts))
                p.drawLine(x, 0, x, H)
                pts += self.snap.grid_pts

        # 工具块
        for b in self.layer.sorted_blocks():
            self._draw_block(p, b, H)

        # 对齐参考线
        if self._guides:
            pen = QPen(QColor("#ffcc00"), 1, Qt.DashLine)
            p.setPen(pen)
            for g in self._guides:
                x = int(self._pts2px(g))
                p.drawLine(x, 0, x, H)

        p.end()

    def _draw_block(self, p: QPainter, b: ToolBlock, H: int):
        meta   = PLUGIN_META.get(b.plugin_id, {})
        x1     = int(self._pts2px(b.start))
        x2     = int(self._pts2px(b.end))
        w      = max(x2 - x1, 12)
        is_sel = b.block_id == self._selected_id
        is_drg = self._drag_block and self._drag_block.block_id == b.block_id

        color = QColor("#3a7a40") if b.done else QColor(meta.get("color", "#378ADD"))
        p.setOpacity(0.55 if is_drg else 1.0)

        # 主体
        path = QPainterPath()
        path.addRoundedRect(x1, 3, w, H - 6, 4, 4)
        p.fillPath(path, color)

        # 选中边框
        if is_sel:
            pen = QPen(QColor(255, 255, 255, 100), 1.5)
            p.setPen(pen)
            p.drawPath(path)

        # 拉伸手柄
        p.fillRect(x1,          3, HANDLE_W, H - 6, QColor(255, 255, 255, 30))
        p.fillRect(x2 - HANDLE_W, 3, HANDLE_W, H - 6, QColor(255, 255, 255, 30))

        # 标签
        p.setOpacity(p.opacity())
        p.setPen(QPen(QColor(255, 255, 255, 230)))
        font = QFont("Courier New", 10, QFont.Bold)
        p.setFont(font)
        text = meta.get("name", b.plugin_id)
        p.drawText(QRect(x1 + HANDLE_W + 4, 0, w - HANDLE_W*2 - 8, H),
                   Qt.AlignVCenter | Qt.AlignLeft, text)

        if b.done:
            p.setPen(QPen(QColor("#7fdd7f")))
            small = QFont("Courier New", 9)
            p.setFont(small)
            p.drawText(QRect(x2 - 18, 0, 16, H), Qt.AlignVCenter, "✓")

        p.setOpacity(1.0)

    # ── 命中测试 ──────────────────────────────────────────────
    def _hit_test(self, mx: int):
        for b in reversed(self.layer.sorted_blocks()):
            x1 = int(self._pts2px(b.start))
            x2 = int(self._pts2px(b.end))
            if x1 <= mx <= x1 + HANDLE_W:
                return b, "resize_left"
            if x2 - HANDLE_W <= mx <= x2:
                return b, "resize_right"
            if x1 <= mx <= x2:
                return b, "move"
        return None, ""

    # ── 鼠标事件 ──────────────────────────────────────────────
    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        b, mode = self._hit_test(e.pos().x())
        if not b:
            return
        self._drag_block   = b
        self._drag_mode    = mode
        self._drag_start_x = e.pos().x()
        self._drag_origin_s = b.start
        self._drag_origin_e = b.end
        self._selected_id  = b.block_id
        self.snap.update_edges(
            [self.ch], exclude_id=b.block_id
        )
        self.update()
        e.accept()

    def mouseMoveEvent(self, e):
        mx = e.pos().x()
        if not self._drag_block:
            b, mode = self._hit_test(mx)
            if mode in ("resize_left", "resize_right"):
                self.setCursor(Qt.SizeHorCursor)
            elif mode == "move":
                self.setCursor(Qt.SizeAllCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            return

        dx = self._px2pts(mx - self._drag_start_x)
        ppp = self._px_per_pt

        if self._drag_mode == "move":
            span = self._drag_origin_e - self._drag_origin_s
            ns, guides = self.snap.snap(self._drag_origin_s + dx, ppp)
            ns = max(0, min(DATA_LEN - span, ns))
            self._drag_block.start = ns
            self._drag_block.end   = ns + span
        elif self._drag_mode == "resize_left":
            ns, guides = self.snap.snap(self._drag_origin_s + dx, ppp)
            self._drag_block.start = min(
                ns, self._drag_block.end - 100
            )
        else:
            ne, guides = self.snap.snap(self._drag_origin_e + dx, ppp)
            self._drag_block.end = max(
                ne, self._drag_block.start + 100
            )

        self._guides = guides
        self.guide_changed.emit(guides)
        self.update()

    def mouseReleaseEvent(self, _):
        self._drag_block = None
        self._drag_mode  = ""
        self._guides     = []
        self.guide_changed.emit([])
        self.update()

    def mouseDoubleClickEvent(self, e):
        b, _ = self._hit_test(e.pos().x())
        if b:
            self.block_double_click.emit(b.block_id)

    def contextMenuEvent(self, e):
        b, _ = self._hit_test(e.pos().x())
        if not b:
            return
        self.layer.blocks = [
            x for x in self.layer.blocks
            if x.block_id != b.block_id
        ]
        if self._selected_id == b.block_id:
            self._selected_id = ""
        self.block_removed.emit(b.block_id)
        self.update()

    # ── 拖放接收 ──────────────────────────────────────────────
    def dragEnterEvent(self, e):
        if e.mimeData().hasFormat("application/x-plugin-id"):
            e.acceptProposedAction()

    def dropEvent(self, e):
        pid = e.mimeData().data(
            "application/x-plugin-id"
        ).data().decode()
        drop_x = e.position().toPoint().x()
        start, _ = self.snap.snap(self._px2pts(drop_x), self._px_per_pt)
        start = max(0, min(DATA_LEN - 1000, start))
        end   = min(DATA_LEN, start + 2000)
        meta  = PLUGIN_META.get(pid, {})
        params = {k: v["default"]
                  for k, v in meta.get("params", {}).items()}
        b = ToolBlock(plugin_id=pid, start=start, end=end, params=params)
        self.layer.blocks.append(b)
        self._selected_id = b.block_id
        self.block_dropped.emit(pid, start, end)
        self.update()
        # 自动打开参数面板
        self.block_double_click.emit(b.block_id)
        e.acceptProposedAction()

# ─── 单通道泳道组（标签 + 多条 LayerTrack） ───────────────────
class ChannelLaneWidget(QWidget):
    block_double_click = Signal(str)
    request_add_layer  = Signal(str)
    guide_changed      = Signal(list)

    def __init__(self, cp: ChannelPipeline, snap: SnapEngine,
                 parent=None):
        super().__init__(parent)
        self.cp   = cp
        self.snap = snap
        self._tracks: list[LayerTrack] = []
        self._build()

    def _build(self):
        # 清空旧布局
        old = self.layout()
        if old:
            while old.count():
                item = old.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            QWidget().setLayout(old)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 左侧标签列
        left = QWidget()
        left.setFixedWidth(HEADER_W)
        left.setStyleSheet("background:#252528;border-right:1px solid #333;")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        # 通道标题
        ch_row = QWidget()
        ch_row.setFixedHeight(CH_HDR_H)
        ch_row.setStyleSheet("background:#2e2e32;border-bottom:1px solid #333;")
        ch_rl = QHBoxLayout(ch_row)
        ch_rl.setContentsMargins(8, 0, 4, 0)
        lbl = QLabel(self.cp.channel)
        lbl.setStyleSheet("font-weight:bold;font-size:12px;color:#ddd;")
        toggle = QPushButton("▾" if self.cp.expanded else "▸")
        toggle.setFixedSize(22, 22)
        toggle.setStyleSheet(
            "background:none;border:none;color:#666;font-size:10px;"
        )
        toggle.clicked.connect(self._toggle)
        ch_rl.addWidget(lbl, 1)
        ch_rl.addWidget(toggle)
        left_layout.addWidget(ch_row)

        # 右侧轨道列
        right = QWidget()
        right.setStyleSheet("background:#1e1e22;")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # 通道标题对应空行
        spacer = QWidget()
        spacer.setFixedHeight(CH_HDR_H)
        spacer.setStyleSheet(
            "background:#2e2e32;border-bottom:1px solid #333;"
        )
        right_layout.addWidget(spacer)

        self._tracks.clear()

        if self.cp.expanded:
            for i, layer in enumerate(self.cp.layers):
                # 标签行
                layer_row = QWidget()
                layer_row.setFixedHeight(LAYER_H)
                layer_row.setStyleSheet(
                    "background:#222226;border-bottom:1px solid #2a2a2a;"
                )
                lrl = QHBoxLayout(layer_row)
                lrl.setContentsMargins(18, 0, 4, 0)
                llbl = QLabel(layer.label)
                llbl.setStyleSheet("font-size:11px;color:#777;")
                rm_btn = QPushButton("✕")
                rm_btn.setFixedSize(18, 18)
                rm_btn.setStyleSheet(
                    "background:none;border:none;color:#444;"
                    "font-size:10px;"
                )
                rm_btn.clicked.connect(
                    lambda _, lid=layer.layer_id: self._remove_layer(lid)
                )
                lrl.addWidget(llbl, 1)
                lrl.addWidget(rm_btn)
                left_layout.addWidget(layer_row)

                # 轨道
                track = LayerTrack(self.cp, layer, self.snap, self)
                track.block_double_click.connect(self.block_double_click)
                track.guide_changed.connect(self._on_guide)
                self._tracks.append(track)
                right_layout.addWidget(track)

            # + add step 行
            add_row = QWidget()
            add_row.setFixedHeight(ADD_ROW_H)
            add_row.setStyleSheet(
                "background:#1a1a1e;border-bottom:1px solid #2a2a2a;"
            )
            arl = QHBoxLayout(add_row)
            arl.setContentsMargins(8, 0, 4, 0)
            add_btn = QPushButton("+ add step")
            add_btn.setStyleSheet(
                "background:none;border:none;color:#555;"
                "font-size:10px;text-align:left;"
            )
            add_btn.clicked.connect(
                lambda: self.request_add_layer.emit(self.cp.channel)
            )
            arl.addWidget(add_btn)
            left_layout.addWidget(add_row)

            add_spacer = QWidget()
            add_spacer.setFixedHeight(ADD_ROW_H)
            add_spacer.setStyleSheet(
                "background:#1a1a1e;border-bottom:1px solid #2a2a2a;"
            )
            right_layout.addWidget(add_spacer)

        left_layout.addStretch()
        right_layout.addStretch()

        layout.addWidget(left)
        layout.addWidget(right, 1)

    def _toggle(self):
        self.cp.expanded = not self.cp.expanded
        self._build()

    def _remove_layer(self, layer_id: str):
        if len(self.cp.layers) <= 1:
            QMessageBox.information(self, "提示", "至少保留一个步骤")
            return
        self.cp.remove_layer(layer_id)
        self._build()

    def add_layer(self):
        self.cp.add_layer()
        self._build()

    def set_zoom(self, z: float):
        for t in self._tracks:
            t.set_zoom(z)

    def set_guides(self, guides: list[int]):
        for t in self._tracks:
            t.set_guides(guides)

    def _on_guide(self, guides: list[int]):
        self.guide_changed.emit(guides)

    def set_selected(self, block_id: str):
        for t in self._tracks:
            t.set_selected(block_id)

    def find_block(self, block_id: str):
        return self.cp.find_block(block_id)

# ─── 时间轴刻度 ──────────────────────────────────────────────
class RulerWidget(QWidget):
    def __init__(self, snap: SnapEngine, parent=None):
        super().__init__(parent)
        self.snap = snap
        self.zoom = 6.0
        self.setFixedHeight(24)
        self.setMinimumWidth(600)
        self.setStyleSheet("background:#222;")

    @property
    def _px_per_pt(self) -> float:
        return self.zoom * 8 / 1000

    def set_zoom(self, z: float):
        self.zoom = z
        self.setMinimumWidth(
            int(DATA_LEN * self._px_per_pt) + 100
        )
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#222222"))
        p.setPen(QPen(QColor("#555"), 0.5))
        font = QFont("Courier New", 9)
        p.setFont(font)
        p.setPen(QColor("#666"))
        step = self.snap.grid_pts
        pts = 0
        while pts <= DATA_LEN:
            x = int(pts * self._px_per_pt)
            p.setPen(QPen(QColor("#555"), 0.5))
            p.drawLine(x, 16, x, 24)
            if pts % (step * 2) == 0 or step >= 2000:
                p.setPen(QColor("#666"))
                p.drawText(x + 2, 13, str(pts))
            pts += step
        p.end()

# ─── 左侧工具面板 ─────────────────────────────────────────────
class ToolPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(150)
        self.setStyleSheet("background:#1e1e22;border-right:1px solid #333;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(3)

        for section, pids in [
            ("TIME DOMAIN",
             ["remove_spike","detrend","notch_filter","bandpass"]),
            ("OUTPUT",
             ["robust","remote_ref"]),
        ]:
            sec_lbl = QLabel(section)
            sec_lbl.setStyleSheet(
                "font-size:9px;color:#555;letter-spacing:0.08em;"
            )
            layout.addWidget(sec_lbl)
            for pid in pids:
                meta = PLUGIN_META.get(pid, {})
                item = _DraggableToolItem(pid, meta)
                layout.addWidget(item)
            layout.addSpacing(6)

        layout.addStretch()

class _DraggableToolItem(QLabel):
    def __init__(self, plugin_id: str, meta: dict):
        color = meta.get("color", "#888")
        name  = meta.get("name", plugin_id)
        super().__init__(f"● {name}")
        self._pid = plugin_id
        self.setStyleSheet(
            f"font-size:11px;color:#bbb;padding:5px 8px;"
            f"background:#2a2a2e;border:1px solid #3a3a3a;"
            f"border-radius:4px;font-family:monospace;"
        )
        self.setCursor(Qt.OpenHandCursor)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.setCursor(Qt.ClosedHandCursor)

    def mouseReleaseEvent(self, e):
        self.setCursor(Qt.OpenHandCursor)

    def mouseMoveEvent(self, e):
        if not (e.buttons() & Qt.LeftButton):
            return
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData(
            "application/x-plugin-id",
            QByteArray(self._pid.encode())
        )
        drag.setMimeData(mime)
        drag.exec(Qt.CopyAction)

# ─── 参数编辑 MdiSubWindow ────────────────────────────────────
class ParamWindow(QMdiSubWindow):
    params_applied = Signal(str, dict)   # block_id, params

    def __init__(self, block: ToolBlock, mdi: QMdiArea):
        super().__init__()
        self._block = block
        meta  = PLUGIN_META.get(block.plugin_id, {})
        self.setWindowTitle(
            f"{meta.get('name', block.plugin_id)}  "
            f"[{block.start}~{block.end}]"
        )
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.resize(320, 280)

        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(12, 12, 12, 12)
        form.setSpacing(8)
        self._widgets: dict[str, QWidget] = {}

        for k, defn in meta.get("params", {}).items():
            if defn["type"] == "select":
                cb = QComboBox()
                cb.addItems(defn["opts"])
                cb.setCurrentText(str(block.params.get(k, defn["default"])))
                form.addRow(k, cb)
                self._widgets[k] = cb
            elif defn["type"] == "int":
                sb = QSpinBox()
                sb.setRange(0, 10000)
                sb.setValue(int(block.params.get(k, defn["default"])))
                form.addRow(k, sb)
                self._widgets[k] = sb
            else:
                sb = QDoubleSpinBox()
                sb.setRange(0, 10000)
                sb.setDecimals(4)
                sb.setSingleStep(0.1)
                sb.setValue(float(block.params.get(k, defn["default"])))
                form.addRow(k, sb)
                self._widgets[k] = sb

        # 范围
        form.addRow(QLabel("── range ──"))
        self._start_sb = QSpinBox()
        self._start_sb.setRange(0, DATA_LEN)
        self._start_sb.setSingleStep(500)
        self._start_sb.setValue(block.start)
        self._end_sb = QSpinBox()
        self._end_sb.setRange(0, DATA_LEN)
        self._end_sb.setSingleStep(500)
        self._end_sb.setValue(block.end)
        form.addRow("start (pts)", self._start_sb)
        form.addRow("end   (pts)", self._end_sb)

        btn = QPushButton("apply")
        btn.clicked.connect(self._apply)
        form.addRow(btn)

        self.setWidget(w)
        mdi.addSubWindow(self)
        self.show()

    def _apply(self):
        params = {}
        meta = PLUGIN_META.get(self._block.plugin_id, {})
        for k, wgt in self._widgets.items():
            defn = meta.get("params", {}).get(k, {})
            if isinstance(wgt, QComboBox):
                params[k] = wgt.currentText()
            elif defn.get("type") == "int":
                params[k] = wgt.value()
            else:
                params[k] = wgt.value()
        s = self._start_sb.value()
        e = self._end_sb.value()
        if s < e:
            self._block.start = s
            self._block.end   = e
        self._block.params = params
        self.params_applied.emit(self._block.block_id, params)

# ─── 提交对话框 ───────────────────────────────────────────────
class CommitDialog(QDialog):
    def __init__(self, pipeline: list[ChannelPipeline], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Commit — choose save mode")
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Choose output mode per channel:"))

        self._rows: dict[str, tuple[QRadioButton, QRadioButton]] = {}
        for cp in pipeline:
            grp = QGroupBox(cp.channel)
            grl = QHBoxLayout(grp)
            rb_ow = QRadioButton("overwrite original")
            rb_nw = QRadioButton("new channel (_processed)")
            rb_ow.setChecked(True)
            grl.addWidget(rb_ow)
            grl.addWidget(rb_nw)
            layout.addWidget(grp)
            self._rows[cp.channel] = (rb_ow, rb_nw)

        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_modes(self) -> dict[str, str]:
        return {
            ch: ("overwrite" if rb_ow.isChecked() else "new_channel")
            for ch, (rb_ow, _) in self._rows.items()
        }

# ─── 主 MdiArea ───────────────────────────────────────────────
class PipelineMdiArea(QMdiArea):
    """
    顶层 MdiArea，持有：
      - PipelineSubWindow（默认最大化）
      - ParamWindow（双击 block 后弹出，每个 block 最多一个）
    """
    data_committed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground(QBrush(QColor("#141418")))

        self.snap     = SnapEngine()
        self.pipeline: list[ChannelPipeline] = []
        self._lane_widgets: dict[str, ChannelLaneWidget] = {}
        self._param_wins:   dict[str, ParamWindow] = {}

        self._pipeline_sub = self._build_pipeline_sub()

    # ── 流水线子窗口 ──────────────────────────────────────────
    def _build_pipeline_sub(self) -> QMdiSubWindow:
        self._pipeline_widget = PipelineWidget(
            self.snap, self
        )
        self._pipeline_widget.block_double_click.connect(
            self._open_param_win
        )
        sub = QMdiSubWindow()
        sub.setWidget(self._pipeline_widget)
        sub.setWindowTitle("Pipeline")
        sub.setWindowFlags(
            sub.windowFlags() & ~Qt.WindowCloseButtonHint
        )
        self.addSubWindow(sub)
        sub.showMaximized()
        return sub

    def load_channels(self, channels: list[str]):
        self.pipeline = [
            ChannelPipeline(channel=ch, layers=[
                ProcessLayer(label="step 1")
            ])
            for ch in channels
        ]
        self._pipeline_widget.load_pipeline(self.pipeline)

    def _open_param_win(self, block_id: str):
        # 已有 → 激活
        if block_id in self._param_wins:
            win = self._param_wins[block_id]
            if win.isMinimized():
                win.showNormal()
            self.setActiveSubWindow(win)
            return
        # 找到 block
        for cp in self.pipeline:
            result = cp.find_block(block_id)
            if result:
                _, block = result
                win = ParamWindow(block, self)
                win.params_applied.connect(
                    self._pipeline_widget.refresh_block
                )
                win.destroyed.connect(
                    lambda _, bid=block_id:
                        self._param_wins.pop(bid, None)
                )
                self._param_wins[block_id] = win
                return

    def run_preview(self):
        """模拟逐块执行（带延迟，视觉反馈）"""
        all_blocks = [
            b
            for cp in self.pipeline
            for ly in cp.layers
            for b  in ly.blocks
        ]
        if not all_blocks:
            QMessageBox.information(self, "提示", "没有工具块可执行")
            return
        for b in all_blocks:
            b.done = False
        self._pipeline_widget.refresh_all()

        i = [0]
        def step():
            if i[0] >= len(all_blocks):
                self._pipeline_widget.refresh_all()
                return
            all_blocks[i[0]].done = True
            self._pipeline_widget.refresh_all()
            i[0] += 1
            QTimer.singleShot(120, step)
        step()

    def open_commit(self):
        dlg = CommitDialog(self.pipeline, self)
        if dlg.exec() == QDialog.Accepted:
            modes = dlg.get_modes()
            # 实际项目：根据 modes 写回 EMData
            summary = "\n".join(
                f"  {ch}: {mode}"
                for ch, mode in modes.items()
            )
            QMessageBox.information(
                self, "Committed",
                f"处理结果已写回：\n{summary}\n\n"
                f"（文件保存请使用菜单栏 文件 → 保存）"
            )
            self.data_committed.emit()

# ─── 流水线主控件（PipelineSubWindow 的内容） ─────────────────
class PipelineWidget(QWidget):
    block_double_click = Signal(str)

    def __init__(self, snap: SnapEngine, parent=None):
        super().__init__(parent)
        self.snap = snap
        self._lane_widgets: dict[str, ChannelLaneWidget] = []
        self._zoom = 6.0
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 滚动区
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOn
        )
        self._scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )
        self._scroll.setStyleSheet(
            "QScrollArea{border:none;background:#141418;}"
        )

        self._content = QWidget()
        self._content.setStyleSheet("background:#141418;")
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(0)

        # 刻度尺（吸附在顶部）
        ruler_row = QWidget()
        ruler_layout = QHBoxLayout(ruler_row)
        ruler_layout.setContentsMargins(0, 0, 0, 0)
        ruler_layout.setSpacing(0)
        spacer = QWidget()
        spacer.setFixedWidth(HEADER_W)
        spacer.setStyleSheet(
            "background:#222;border-right:1px solid #333;"
            "border-bottom:1px solid #333;"
        )
        spacer.setFixedHeight(24)
        self._ruler = RulerWidget(self.snap)
        ruler_layout.addWidget(spacer)
        ruler_layout.addWidget(self._ruler, 1)
        self._content_layout.addWidget(ruler_row)

        self._lanes_layout = QVBoxLayout()
        self._lanes_layout.setContentsMargins(0, 0, 0, 0)
        self._lanes_layout.setSpacing(0)
        self._content_layout.addLayout(self._lanes_layout)
        self._content_layout.addStretch()

        self._scroll.setWidget(self._content)
        layout.addWidget(self._scroll)

    def load_pipeline(self, pipeline: list[ChannelPipeline]):
        self._pipeline = pipeline
        self._lane_widgets = []
        # 清空旧控件
        while self._lanes_layout.count():
            item = self._lanes_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        # 创建通道泳道
        for cp in pipeline:
            lw = ChannelLaneWidget(cp, self.snap, self)
            lw.block_double_click.connect(self.block_double_click)
            lw.request_add_layer.connect(self._add_layer)
            lw.guide_changed.connect(self._broadcast_guides)
            self._lane_widgets.append(lw)
            self._lanes_layout.addWidget(lw)
        self._apply_zoom()

    def _add_layer(self, ch_name: str):
        for lw in self._lane_widgets:
            if lw.cp.channel == ch_name:
                lw.add_layer()
                break

    def _broadcast_guides(self, guides: list[int]):
        for lw in self._lane_widgets:
            lw.set_guides(guides)

    def set_zoom(self, z: float):
        self._zoom = z
        self._apply_zoom()

    def _apply_zoom(self):
        for lw in self._lane_widgets:
            lw.set_zoom(self._zoom)
        self._ruler.set_zoom(self._zoom)

    def refresh_block(self, block_id: str, _params: dict = None):
        for lw in self._lane_widgets:
            for t in lw._tracks:
                t.update()

    def refresh_all(self):
        for lw in self._lane_widgets:
            for t in lw._tracks:
                t.update()

# ─── 主窗口 ───────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pipeline Lane Editor — Prototype")
        self.resize(1280, 720)
        self.setStyleSheet("background:#141418;color:#ccc;")

        # 工具栏
        tb = QToolBar("main")
        tb.setMovable(False)
        tb.setStyleSheet(
            "QToolBar{background:#252528;border-bottom:1px solid #333;"
            "spacing:4px;padding:4px 8px;}"
            "QToolButton{background:#333;border:1px solid #444;"
            "border-radius:3px;color:#ccc;padding:3px 10px;"
            "font-size:11px;font-family:monospace;}"
            "QToolButton:hover{background:#444;}"
        )
        self.addToolBar(tb)

        btn_prev = QPushButton("▶  preview")
        btn_prev.setStyleSheet(
            "background:#2e4a2e;border:1px solid #3a6a3a;"
            "color:#aaffaa;font-size:11px;padding:3px 10px;"
            "border-radius:3px;font-family:monospace;"
        )
        btn_commit = QPushButton("✔  commit")
        btn_commit.setStyleSheet(
            "background:#2e3a5e;border:1px solid #3a5a9e;"
            "color:#aac4ff;font-size:11px;padding:3px 10px;"
            "border-radius:3px;font-family:monospace;"
        )
        tb.addWidget(btn_prev)
        tb.addWidget(btn_commit)
        tb.addSeparator()

        zoom_lbl = QLabel("zoom")
        zoom_lbl.setStyleSheet(
            "font-size:11px;color:#888;font-family:monospace;"
        )
        zoom_sl = QSlider(Qt.Horizontal)
        zoom_sl.setRange(1, 20)
        zoom_sl.setValue(6)
        zoom_sl.setFixedWidth(90)
        self._zoom_val_lbl = QLabel("6")
        self._zoom_val_lbl.setStyleSheet(
            "font-size:11px;color:#aaa;font-family:monospace;"
            "min-width:16px;"
        )
        tb.addWidget(zoom_lbl)
        tb.addWidget(zoom_sl)
        tb.addWidget(self._zoom_val_lbl)
        tb.addSeparator()

        snap_cb = QCheckBox("snap")
        snap_cb.setChecked(True)
        snap_cb.setStyleSheet(
            "font-size:11px;color:#888;font-family:monospace;"
        )
        grid_lbl = QLabel("grid")
        grid_lbl.setStyleSheet(
            "font-size:11px;color:#888;font-family:monospace;"
        )
        grid_sb = QSpinBox()
        grid_sb.setRange(50, 5000)
        grid_sb.setSingleStep(50)
        grid_sb.setValue(500)
        grid_sb.setFixedWidth(64)
        grid_sb.setStyleSheet(
            "font-size:11px;background:#2a2a2a;border:1px solid #3a3a3a;"
            "color:#ccc;font-family:monospace;"
        )
        grid_pts_lbl = QLabel("pts")
        grid_pts_lbl.setStyleSheet(
            "font-size:11px;color:#666;font-family:monospace;"
        )
        tb.addWidget(snap_cb)
        tb.addWidget(grid_lbl)
        tb.addWidget(grid_sb)
        tb.addWidget(grid_pts_lbl)

        # 中央：工具面板 + MdiArea
        central = QWidget()
        h_layout = QHBoxLayout(central)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(0)

        self._tool_panel = ToolPanel()
        self._mdi = PipelineMdiArea()
        h_layout.addWidget(self._tool_panel)
        h_layout.addWidget(self._mdi, 1)
        self.setCentralWidget(central)

        # 加载演示数据
        self._mdi.load_channels(["Ex1", "Ex2", "Bx", "By"])

        # 信号连接
        btn_prev.clicked.connect(self._mdi.run_preview)
        btn_commit.clicked.connect(self._mdi.open_commit)
        zoom_sl.valueChanged.connect(self._on_zoom)
        snap_cb.toggled.connect(self._on_snap)
        grid_sb.valueChanged.connect(self._on_grid)

    def _on_zoom(self, v: int):
        self._zoom_val_lbl.setText(str(v))
        self._mdi._pipeline_widget.set_zoom(float(v))
        self._mdi.snap.grid_pts = int(
            self._mdi._pipeline_widget._ruler.snap.grid_pts
        )

    def _on_snap(self, on: bool):
        self._mdi.snap.enabled = on
        self._mdi._pipeline_widget.refresh_all()

    def _on_grid(self, v: int):
        self._mdi.snap.grid_pts = v
        self._mdi._pipeline_widget._ruler.snap.grid_pts = v
        self._mdi._pipeline_widget._ruler.update()
        self._mdi._pipeline_widget.refresh_all()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
