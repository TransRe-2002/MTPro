from typing import Any, Optional, List

from PySide6.QtGui import QIntValidator, QAction, QIcon, QRegularExpressionValidator
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QSplitter, QLineEdit,
    QSizePolicy, QSlider, QDateTimeEdit, QScrollBar,
    QToolBar, QLabel, QPushButton, QMessageBox, QHBoxLayout,
    QGridLayout, QTextEdit
)
from PySide6.QtCore import Qt, Signal, QRegularExpression
import numpy as np
import pandas as pd
import pyqtgraph as pg

from core.em_data import Channel
from utils.series import dti_to_numpy

pg.setConfigOptions(antialias=True)
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

class RemoveStep(QWidget):
    result_signal = Signal(str, np.ndarray)

    def __init__(self,
                 channel: Channel,
                 parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.plot_widget = None
        self.plot_item = None

        self.ch: Channel = channel
        self.ch_name = channel.name
        x = channel.datetime_index()
        self.x_data = dti_to_numpy(x)
        y = channel.cts
        self.nan_index = y[y.isna()].index.tolist()
        y = y.interpolate(method='linear')
        self.y_data = y.to_numpy(dtype=np.float64)
        self.origin_y_data = self.y_data.copy()
        self.diff_y_data = np.diff(self.y_data)

        self.history: List[np.ndarray] = []

        self.de_step_btn = QPushButton("阈值差分去除")
        self.de_step_threshold = QLineEdit()
        float_validator = QRegularExpressionValidator(
            QRegularExpression(r"^\d*\.?\d+$")
        )
        self.de_step_threshold.setValidator(float_validator)
        int_validator = QRegularExpressionValidator(QRegularExpression(r"^\d+$"))
        self.one_step_btn = QPushButton("多步差分去除")
        self.one_step_times = QLineEdit()
        self.one_step_times.setValidator(int_validator)
        self.undo_btn = QPushButton("撤销")
        self.revert_btn = QPushButton("还原原始数据")
        self.confirm_btn = QPushButton("确认")

        self.one_step_times.setText("5")
        max_abs_diff = np.max(np.abs(self.diff_y_data))
        self.de_step_threshold.setText(f"{max_abs_diff:.4f}")

        self._init_ui()
        self._connect_signal()
        self._init_plot()

    def _init_ui(self):
        self.instruction_label = QLabel(
            "操作说明: 首先计算差分，然后根据差值选择去除方式，最后点击确认"
        )
        self.instruction_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed
        )

        self.plot_widget = pg.PlotWidget(axisItems={"bottom": pg.DateAxisItem()})
        self.plot_widget.getAxis('left').setStyle(tickLength=0)
        self.plot_widget.getAxis('bottom').setStyle(tickLength=0)
        self.plot_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.plot_item = self.plot_widget.getPlotItem()

        self.plain_text = QTextEdit()
        self.plain_text.setReadOnly(True)
        self.plain_text.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Expanding
        )

        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        button_layout.addWidget(QLabel("差分阈值："))
        button_layout.addWidget(self.de_step_threshold)
        button_layout.addWidget(self.de_step_btn)
        button_layout.addStretch(1)
        button_layout.addWidget(QLabel("多步差分次数："))
        button_layout.addWidget(self.one_step_times)
        button_layout.addWidget(self.one_step_btn)
        button_layout.addStretch(1)
        button_layout.addWidget(self.undo_btn)
        button_layout.addWidget(self.revert_btn)
        button_layout.addStretch(1)
        button_layout.addWidget(self.confirm_btn)
        button_layout.addStretch(1)

        glay = QGridLayout()
        glay.addWidget(self.instruction_label, 0, 0, 1, 2)
        glay.addWidget(self.plain_text, 1, 0)
        glay.addWidget(self.plot_widget, 1, 1)
        glay.addLayout(button_layout, 2, 0, 1, 2)
        self.setLayout(glay)

    def _connect_signal(self):
        self.de_step_btn.clicked.connect(self._de_step_by_threshold)
        self.one_step_btn.clicked.connect(self._de_step_multi)
        self.undo_btn.clicked.connect(self._undo)
        self.revert_btn.clicked.connect(self._revert)
        self.confirm_btn.clicked.connect(self._confirm_and_close)

    def _init_plot(self):
        if self.x_data is None or self.y_data is None:
            return

        y_centered = self.y_data - np.nanmean(self.y_data)
        self.y_line = pg.PlotCurveItem(
            self.x_data,
            y_centered,
            pen=pg.mkPen(255, 0, 0, width=1),
            name="序列数值",
        )
        self.dy_line = pg.PlotCurveItem(
            self.x_data[1:],
            self.diff_y_data,
            pen=pg.mkPen(0, 0, 255, width=1),
            name="序列一阶差分",
        )
        legend = self.plot_item.addLegend()
        legend.anchor((1, 0), (1, 0))
        legend.setBrush(pg.mkBrush(255, 255, 255, 255))
        legend.setPen(pg.mkPen('k', width=1))
        legend.show()

        grid_pen = pg.mkPen(
            color=pg.mkColor(100, 100, 100),
            width=1,
            alpha=1,
            antialiased=True,
            style=Qt.PenStyle.DashLine
        )
        self.plot_item.showGrid(x=True, y=True, alpha=0.7)
        self.plot_item.getAxis('bottom').setPen(grid_pen)
        self.plot_item.getAxis('left').setPen(grid_pen)
        self.plot_item.addItem(self.y_line)
        self.plot_item.addItem(self.dy_line)

        self._show_top_diffs()

    def _show_top_diffs(self):
        """在文本框中显示一阶差分绝对值前10大的位置和数值"""
        abs_diff = np.abs(self.diff_y_data)
        top_n = min(10, len(abs_diff))
        top_indices = np.argpartition(abs_diff, -top_n)[-top_n:]
        top_indices = top_indices[np.argsort(abs_diff[top_indices])[::-1]]

        self._log(f"===== 一阶差分(绝对值) Top {top_n} =====")
        for rank, idx in enumerate(top_indices, 1):
            ts = pd.Timestamp(self.x_data[idx], unit='s')
            self._log(f"  {rank}. 位置={idx}, 时间={ts}, |差分|={abs_diff[idx]:.4f}")
        self._log("")

    def _log(self, msg: str):
        """输出日志到文本框"""
        self.plain_text.append(msg)

    def _save_history(self):
        """保存当前 y_data 快照"""
        self.history.append(self.y_data.copy())

    def _update_plot(self):
        """更新数据曲线和差分曲线"""
        y_centered = self.y_data - np.nanmean(self.y_data)
        self.y_line.setData(self.x_data, y_centered)
        self.dy_line.setData(self.x_data[1:], self.diff_y_data)

    def _de_step_by_threshold(self):
        """根据阈值去除台阶：|diff| > threshold 的位置视为台阶"""
        text = self.de_step_threshold.text().strip()
        if not text:
            self._log("请输入差分阈值")
            return

        threshold = float(text)
        if threshold <= 0:
            self._log("阈值必须大于 0")
            return

        self._save_history()

        diff = self.diff_y_data.copy()
        step_mask = np.abs(diff) > threshold
        step_indices = np.where(step_mask)[0]

        if len(step_indices) == 0:
            self.history.pop()
            self._log(f"阈值 {threshold:.4f} 未找到台阶")
            return

        diff[step_mask] = 0
        self.diff_y_data = diff.copy()
        diff = np.insert(diff, 0, self.y_data[0])
        self.y_data = np.cumsum(diff)

        self._update_plot()
        self._log(f"阈值 {threshold:.4f} 去除了 {len(step_indices)} 个台阶\n")
        self._show_top_diffs()

    def _de_step_multi(self):
        """多步差分去除：每次找最大的台阶去除，重复 N 次"""
        text = self.one_step_times.text().strip()
        if not text:
            self._log("请输入去除次数")
            return

        times = int(text)
        if times <= 0:
            self._log("次数必须大于 0")
            return

        self._save_history()

        diff = self.diff_y_data.copy()
        removed = 0
        for _ in range(times):
            idx = np.argmax(np.abs(diff))
            step_val = diff[idx]
            if step_val == 0:
                break
            diff[idx] = 0
            removed += 1
            self._log(f"  去除第 {removed} 个台阶: 位置={idx}, 幅度={step_val:.4f}")

        self.diff_y_data = diff.copy()
        diff = np.insert(diff, 0, self.y_data[0])
        self.y_data = np.cumsum(diff)
        self._update_plot()
        self._log(f"多步去除完成，共去除 {removed} 个台阶\n")
        self._show_top_diffs()

    def _undo(self):
        """撤销上一步操作"""
        if not self.history:
            self._log("没有可撤销的操作")
            return

        self.y_data = self.history.pop()
        self._update_plot()
        self._log("已撤销上一步操作")

    def _revert(self):
        """还原到原始数据"""
        self._save_history()
        self.y_data = self.origin_y_data.copy()
        self._update_plot()
        self._log("已还原到原始数据")

    def _confirm_and_close(self):
        """确认修改并发送结果信号"""
        result = self.y_data.copy()
        for idx in self.nan_index:
            if 0 <= idx < len(result):
                result[idx] = np.nan

        self._log(f"确认修改，返回 {len(result)} 个数据点")
        self.result_signal.emit(self.ch_name, result)
        self.hide()
        self.close()
        if self.parent() is not None:
            self.parent().close()


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    from io_utils.mat_io import MatLoader

    em_data = MatLoader().load("/home/transen5/Project/atm_rpc/039BE-20240501-20240515-dt5_struct.mat")
    app = QApplication([])
    window = RemoveStep(channel=em_data.data['Ex1'])
    window.setWindowFlag(Qt.WindowType.Window)
    window.show()
    app.exec()
