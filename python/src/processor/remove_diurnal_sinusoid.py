from __future__ import annotations

"""“正弦相减去日变”可视化工具窗口。

本文件负责：
- 从 EMData 中取出某个通道的数据
- 提供可调参数界面
- 调用 diurnal_sinusoid_subtraction 中的算法
- 用上下双图展示原始/预览/已应用结果
- 在用户确认后把结果通过信号返回给上层工作台

本文件不直接实现最小二乘拟合细节，真正的数值算法在
processor.diurnal_sinusoid_subtraction 中。
"""

import os
from typing import Optional

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import numpy as np
import pandas as pd
from PySide6.QtCore import QRegularExpression, Qt
from PySide6.QtGui import QKeySequence, QRegularExpressionValidator, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
)
import pyqtgraph as pg

from core.em_data import EMData
from processor._base_processor import BaseProcessorWidget
from processor.diurnal_sinusoid_subtraction import (
    PeriodicSubtractionResult,
    subtract_diurnal_signal_series,
)
from processor.step_plot_widget import StepPlotWidget
from utils.validators import read_nonnegative_int, read_positive_float, read_positive_int


class RemoveDiurnalSinusoid(BaseProcessorWidget):
    """去日变工具窗口。

    上图显示：
    - 原始序列
    - 当前序列
    - 预览后的序列

    下图显示：
    - 拟合出的日变噪声

    用户典型操作流程：
    1. 选择谐波阶数组合。
    2. 调整窗口天数、步长天数等参数。
    3. 点击"预览拟合"观察结果。
    4. 满意后点击"应用相减"。
    5. 点击"确认"把结果写回主数据对象。
    """

    HARMONIC_PRESETS: list[tuple[str, tuple[int, ...]]] = [
        ("1阶: 24h", (1,)),
        ("2阶: 24h + 12h", (1, 2)),
        ("3阶: 24h + 12h + 8h", (1, 2, 3)),
        ("4阶: 24h + 12h + 8h + 6h", (1, 2, 3, 4)),
        ("5阶: 加 4.8h (17280s)", (1, 2, 3, 4, 5)),
        ("6阶: 加 4h (14400s)", (1, 2, 3, 4, 5, 6)),
    ]

    def __init__(self, data_id: int, channel_name: str, parent=None):
        # 日变工具使用 time+linear 双重插值（在 _on_data_loaded 中覆写）
        super().__init__(data_id, channel_name, parent)

        self.preview_result: Optional[PeriodicSubtractionResult] = None
        self._syncing_x_range = False

        self.signal_plot_widget = StepPlotWidget()
        self.signal_plot_item = self.signal_plot_widget.getPlotItem()
        self.noise_plot_widget = StepPlotWidget()
        self.noise_plot_item = self.noise_plot_widget.getPlotItem()
        self.plot_splitter = None

        float_validator = QRegularExpressionValidator(QRegularExpression(r"^\d*\.?\d+$"))
        int_validator = QRegularExpressionValidator(QRegularExpression(r"^\d+$"))
        self.harmonics_combo = QComboBox()
        for label, harmonics in self.HARMONIC_PRESETS:
            self.harmonics_combo.addItem(label, harmonics)
        self.harmonics_combo.setToolTip(
            "选择要同时拟合并减去的日变谐波组合。"
            "阶数越高，去掉的周期分量越多。"
        )
        self.window_days_edit = QLineEdit("3.0")
        self.window_days_edit.setValidator(float_validator)
        self.window_days_edit.setToolTip(
            "每次拟合使用的数据长度，单位为天。"
            "窗口越长拟合越稳，但过长可能削弱时变特征。"
        )
        self.step_days_edit = QLineEdit("1.0")
        self.step_days_edit.setValidator(float_validator)
        self.step_days_edit.setToolTip(
            "相邻拟合窗口之间的间隔，单位为天。"
            "步长越小，窗口重叠越多，拼接越平滑。"
        )
        self.trend_degree_edit = QLineEdit("1")
        self.trend_degree_edit.setValidator(int_validator)
        self.trend_degree_edit.setToolTip(
            "拟合谐波时允许同时拟合的背景趋势阶数。"
            "0 表示常数背景，1 表示线性趋势。"
        )
        self.min_valid_fraction_edit = QLineEdit("0.6")
        self.min_valid_fraction_edit.setValidator(float_validator)
        self.min_valid_fraction_edit.setToolTip(
            "一个窗口中有效数据所占的最小比例。"
            "低于该比例时，该窗口会跳过拟合。"
        )
        self.min_valid_samples_edit = QLineEdit("128")
        self.min_valid_samples_edit.setValidator(int_validator)
        self.min_valid_samples_edit.setToolTip(
            "一个窗口至少需要的有效采样点数。"
            "点数不足时，该窗口会跳过拟合。"
        )
        self.center_view_checkbox = QCheckBox("显示去中心化波形")
        self.center_view_checkbox.setChecked(True)
        self.center_view_checkbox.setToolTip(
            "只影响显示，不影响计算。"
            "勾选后会先减去均值，更方便观察日变起伏。"
        )

        self.preview_btn = QPushButton("预览拟合")
        self.preview_btn.setToolTip("按当前参数执行一次预览拟合，但不立刻写回结果。")
        self.apply_btn = QPushButton("应用相减")
        self.apply_btn.setToolTip("把最近一次预览结果正式应用到当前序列。")
        self.undo_btn = QPushButton("撤销")
        self.undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self.undo_btn.setToolTip("撤销上一次已应用的修改。")
        self.revert_btn = QPushButton("还原原始数据")
        self.revert_btn.setToolTip("恢复到工具刚打开时的原始插值序列。")
        self.confirm_btn = QPushButton("确认")
        self.confirm_btn.setToolTip("确认修改并把结果写回主数据对象。")
        self.help_btn = QPushButton("帮助")
        self.help_btn.setToolTip("打开参数说明弹窗。")

        self.param_frame = None
        self.action_frame = None

        self.plain_text = QTextEdit()
        self.plain_text.setReadOnly(True)
        self.plain_text.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.plain_text.setMinimumWidth(280)
        self.plain_text.setMaximumWidth(360)

        self._init_ui()
        self._connect_signal()


    def _on_data_loaded(self):
        # 日变工具使用 time+linear 双重插值（覆写基类的 linear-only）
        series = pd.Series(self._pkg['y'])
        interpolated = series.interpolate(method="time").interpolate(method="linear")
        self.y_data = interpolated.to_numpy(dtype=np.float64)
        self.origin_y_data = self.y_data.copy()
        self._init_plot()
        self._log_initial_summary()

    def _init_ui(self):
        """搭建参数区、日志区和上下双图的整体布局。"""
        instruction_label = QLabel(
            "正弦相减去日变: 对全通道按重叠时间窗拟合 24h 日变及其谐波，再从原始序列中相减。\n"
            "建议先预览，再根据下方拟合噪声图和日志调整窗口天数、步长和谐波阶次。"
        )

        for plot_widget in (self.signal_plot_widget, self.noise_plot_widget):
            plot_widget.getAxis("left").setStyle(tickLength=0)
            plot_widget.getAxis("bottom").setStyle(tickLength=0)

        self.param_frame = QFrame(self)
        self.param_frame.setFrameShape(QFrame.Shape.StyledPanel)
        param_layout = QHBoxLayout(self.param_frame)
        param_layout.setContentsMargins(8, 6, 8, 6)
        param_layout.addWidget(QLabel("正弦相减去日变"))
        param_layout.addSpacing(8)
        param_layout.addWidget(QLabel("谐波："))
        param_layout.addWidget(self.harmonics_combo)
        param_layout.addWidget(QLabel("窗口天数："))
        param_layout.addWidget(self.window_days_edit)
        param_layout.addWidget(QLabel("步长天数："))
        param_layout.addWidget(self.step_days_edit)
        param_layout.addWidget(QLabel("趋势阶数："))
        param_layout.addWidget(self.trend_degree_edit)
        param_layout.addWidget(QLabel("最小有效比例："))
        param_layout.addWidget(self.min_valid_fraction_edit)
        param_layout.addWidget(QLabel("最小有效点数："))
        param_layout.addWidget(self.min_valid_samples_edit)
        param_layout.addWidget(self.center_view_checkbox)
        param_layout.addWidget(self.preview_btn)
        param_layout.addWidget(self.apply_btn)

        self.action_frame = QFrame(self)
        self.action_frame.setFrameShape(QFrame.Shape.StyledPanel)
        action_layout = QHBoxLayout(self.action_frame)
        action_layout.setContentsMargins(8, 6, 8, 6)
        action_layout.addWidget(QLabel("操作"))
        action_layout.addSpacing(8)
        action_layout.addWidget(self.undo_btn)
        action_layout.addWidget(self.revert_btn)
        action_layout.addWidget(self.confirm_btn)
        action_layout.addStretch(1)
        action_layout.addWidget(self.help_btn)

        self.plot_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.plot_splitter.setChildrenCollapsible(False)
        self.plot_splitter.setOpaqueResize(False)
        self.plot_splitter.addWidget(self.signal_plot_widget)
        self.plot_splitter.addWidget(self.noise_plot_widget)
        self.plot_splitter.setStretchFactor(0, 3)
        self.plot_splitter.setStretchFactor(1, 2)
        self.plot_splitter.setSizes([620, 340])

        control_layout = QVBoxLayout()
        control_layout.addWidget(self.param_frame)
        control_layout.addWidget(self.action_frame)

        layout = QGridLayout(self)
        layout.addWidget(instruction_label, 0, 0, 1, 2)
        layout.addWidget(self.plain_text, 1, 0)
        layout.addWidget(self.plot_splitter, 1, 1)
        layout.addLayout(control_layout, 2, 0, 1, 2)

    def _connect_signal(self):
        """连接按钮与图形控件的交互信号。"""
        self.preview_btn.clicked.connect(self._preview)
        self.apply_btn.clicked.connect(self._apply_preview)
        self.undo_btn.clicked.connect(self._undo)
        self.undo_shortcut.activated.connect(self._undo)
        self.revert_btn.clicked.connect(self._revert)
        self.confirm_btn.clicked.connect(self._confirm_and_close)
        self.help_btn.clicked.connect(self._show_help)
        self.center_view_checkbox.toggled.connect(self._refresh_signal_plot)
        self.signal_plot_item.vb.sigXRangeChanged.connect(self._sync_noise_plot_x_range)
        self.noise_plot_widget.getViewBox().sigXRangeChanged.connect(self._sync_signal_plot_x_range)

    def _init_plot(self):
        """初始化曲线对象与图例。

        这里先把所有曲线都创建好，后面只更新数据，不重复创建图元。
        """
        curve_opts = dict(
            autoDownsample=True, downsampleMethod='peak',
            clipToView=True, useOpenGL=True, connect='finite',
        )
        self.origin_line = pg.PlotCurveItem(
            self.x_data, self._display_values(self.origin_y_data),
            pen=pg.mkPen(120, 120, 120, width=1, style=Qt.PenStyle.DashLine),
            name="原始序列", **curve_opts,
        )
        self.current_line = pg.PlotCurveItem(
            self.x_data, self._display_values(self.y_data),
            pen=pg.mkPen(215, 48, 39, width=1),
            name="当前序列", **curve_opts,
        )
        self.preview_line = pg.PlotCurveItem(
            pen=pg.mkPen(0, 114, 178, width=1),
            name="预览结果", **curve_opts,
        )
        signal_legend = self.signal_plot_item.addLegend()
        signal_legend.anchor((1, 0), (1, 0))
        signal_legend.setBrush(pg.mkBrush(255, 255, 255, 255))
        signal_legend.setPen(pg.mkPen("k", width=1))
        self.signal_plot_item.showGrid(x=True, y=True, alpha=0.7)
        self.signal_plot_item.addItem(self.origin_line)
        self.signal_plot_item.addItem(self.current_line)
        self.signal_plot_item.addItem(self.preview_line)

        self.current_noise_line = pg.PlotCurveItem(
            pen=pg.mkPen(215, 48, 39, width=1),
            name="当前拟合噪声", **curve_opts,
        )
        self.preview_noise_line = pg.PlotCurveItem(
            pen=pg.mkPen(0, 114, 178, width=1),
            name="预览拟合噪声", **curve_opts,
        )
        noise_legend = self.noise_plot_item.addLegend()
        noise_legend.anchor((1, 0), (1, 0))
        noise_legend.setBrush(pg.mkBrush(255, 255, 255, 255))
        noise_legend.setPen(pg.mkPen("k", width=1))
        self.noise_plot_item.showGrid(x=True, y=True, alpha=0.7)
        self.noise_plot_item.addItem(self.current_noise_line)
        self.noise_plot_item.addItem(self.preview_noise_line)

        self._update_noise_plot(None, None)
        self._apply_shared_x_range(self.signal_plot_widget, self.noise_plot_widget)

    # ------------------------------------------------------------------
    # BaseProcessorWidget 接口
    # ------------------------------------------------------------------

    def _log(self, msg: str):
        self.plain_text.append(msg)

    def _reset_result_state(self):
        self.preview_result = None

    def _update_plot(self):
        self._refresh_signal_plot()

    # ------------------------------------------------------------------
    # X 轴同步（双图）
    # ------------------------------------------------------------------

    def _apply_shared_x_range(self, source_plot: pg.PlotWidget, target_plot: pg.PlotWidget) -> None:
        """同步上下两张图的 X 轴范围。"""
        if self._syncing_x_range:
            return
        source_view = source_plot.getViewBox()
        target_view = target_plot.getViewBox()
        x_min, x_max = source_view.viewRange()[0]
        self._syncing_x_range = True
        try:
            target_view.setXRange(float(x_min), float(x_max), padding=0.0)
        finally:
            self._syncing_x_range = False

    def _sync_noise_plot_x_range(self, _view_box, _x_range) -> None:
        self._apply_shared_x_range(self.signal_plot_widget, self.noise_plot_widget)

    def _sync_signal_plot_x_range(self, _view_box, _x_range) -> None:
        self._apply_shared_x_range(self.noise_plot_widget, self.signal_plot_widget)

    # ------------------------------------------------------------------
    # 日志和显示
    # ------------------------------------------------------------------

    def _log_initial_summary(self):
        self._log("===== 日变正弦相减工具 =====")
        self._log(f"通道: {self.ch_name}")
        self._log(f"采样点数: {len(self.y_data)}")
        self._log(f"采样间隔: {self.em_data.dt}")
        self._log("默认目标周期: 24h")
        self._log("")

    def _display_values(self, values: np.ndarray) -> np.ndarray:
        """根据界面勾选状态决定是否做"去中心化显示"。

        去中心化只影响看图，不影响实际处理数据。
        """
        arr = np.asarray(values, dtype=np.float64)
        if not self.center_view_checkbox.isChecked():
            return arr
        mean_value = float(np.nanmean(arr)) if arr.size else 0.0
        return arr - mean_value

    def _refresh_signal_plot(self):
        self.origin_line.setData(self.x_data, self._display_values(self.origin_y_data))
        self.current_line.setData(self.x_data, self._display_values(self.y_data))
        if self.preview_result is not None:
            self.preview_line.setData(
                self.x_data,
                self._display_values(self.preview_result.cleaned.to_numpy(dtype=np.float64)),
            )
        else:
            self.preview_line.setData([], [])

    def _update_noise_plot(
        self,
        current_noise: Optional[np.ndarray],
        preview_noise: Optional[np.ndarray],
    ):
        """更新下方"拟合噪声图"中的两条曲线。"""
        if current_noise is None:
            self.current_noise_line.setData([], [])
        else:
            self.current_noise_line.setData(self.x_data, current_noise)
        if preview_noise is None:
            self.preview_noise_line.setData([], [])
        else:
            self.preview_noise_line.setData(self.x_data, preview_noise)

    # ------------------------------------------------------------------
    # 业务逻辑
    # ------------------------------------------------------------------

    def _parse_harmonics(self) -> Optional[tuple[int, ...]]:
        """从下拉框读出当前选中的谐波组合。"""
        data = self.harmonics_combo.currentData()
        harmonics = tuple(int(part) for part in data) if data is not None else ()
        if not harmonics or any(h <= 0 for h in harmonics):
            self._log("当前谐波下拉选项无效")
            return None
        return harmonics

    def _run_subtraction(self, samples: np.ndarray) -> Optional[PeriodicSubtractionResult]:
        """读取界面参数并调用底层日变相减算法。"""
        harmonics = self._parse_harmonics()
        if harmonics is None:
            return None
        window_days = read_positive_float(self.window_days_edit, "窗口天数", self._log)
        if window_days is None:
            return None
        step_days = read_positive_float(self.step_days_edit, "步长天数", self._log)
        if step_days is None:
            return None
        trend_degree = read_nonnegative_int(self.trend_degree_edit, "趋势阶数", self._log)
        if trend_degree is None:
            return None
        min_valid_fraction = read_positive_float(self.min_valid_fraction_edit, "最小有效比例", self._log)
        if min_valid_fraction is None:
            return None
        if min_valid_fraction > 1.0:
            self._log("最小有效比例必须小于等于 1")
            return None
        min_valid_samples = read_positive_int(self.min_valid_samples_edit, "最小有效点数", self._log)
        if min_valid_samples is None:
            return None

        series = pd.Series(samples, index=self.em_data.datetime_index, name=self.ch_name)
        try:
            return subtract_diurnal_signal_series(
                series,
                harmonics=harmonics,
                window_days=window_days,
                step_days=step_days,
                trend_degree=trend_degree,
                min_valid_fraction=min_valid_fraction,
                min_valid_samples=min_valid_samples,
            )
        except Exception as exc:
            self._log(f"正弦相减失败: {exc}")
            return None

    def _preview(self):
        """执行一次"只预览、不落盘"的拟合。"""
        result = self._run_subtraction(self.y_data)
        if result is None:
            return
        self.preview_result = result
        self.preview_line.setData(
            self.x_data,
            self._display_values(result.cleaned.to_numpy(dtype=np.float64)),
        )
        self._update_noise_plot(None, result.estimated_noise.to_numpy(dtype=np.float64))
        self._apply_shared_x_range(self.signal_plot_widget, self.noise_plot_widget)
        self._log_preview_summary(result)

    def _log_preview_summary(self, result: PeriodicSubtractionResult):
        """把算法层返回的窗口诊断摘要写到日志框。"""
        diagnostics = result.window_diagnostics
        ok_rows = diagnostics[diagnostics["status"] == "ok"] if not diagnostics.empty else diagnostics
        skipped_rows = diagnostics[diagnostics["status"] != "ok"] if not diagnostics.empty else diagnostics
        self._log(
            f"预览完成 | 窗口数={len(diagnostics)} | 生效窗口={len(ok_rows)} | "
            f"跳过窗口={len(skipped_rows)}"
        )
        if not ok_rows.empty:
            amplitude_columns = [col for col in ok_rows.columns if col.endswith("_amplitude")]
            for column in amplitude_columns:
                mean_amp = float(ok_rows[column].dropna().mean())
                self._log(f"  {column}: 平均拟合振幅={mean_amp:.4f}")
            if "harmonic_rms" in ok_rows:
                self._log(f"  平均拟合噪声RMS={float(ok_rows['harmonic_rms'].mean()):.4f}")
            if "residual_rms" in ok_rows:
                self._log(f"  平均残差RMS={float(ok_rows['residual_rms'].mean()):.4f}")
        self._log("")

    def _apply_preview(self):
        """把预览结果正式应用到当前工作序列。"""
        if self.preview_result is None:
            self._log("请先执行预览拟合")
            return
        self._save_history()
        self.y_data = self.preview_result.cleaned.to_numpy(dtype=np.float64)
        applied_noise = self.preview_result.estimated_noise.to_numpy(dtype=np.float64)
        self.current_line.setData(self.x_data, self._display_values(self.y_data))
        self.preview_line.setData([], [])
        self._update_noise_plot(applied_noise, None)
        self._log(f"已应用正弦相减 | 样本数={len(self.y_data)}")
        self._log("")
        self.preview_result = None

    def _show_help(self):
        """弹出参数帮助说明。"""
        QMessageBox.information(
            self,
            "正弦相减去日变 - 参数说明",
            "谐波：选择要同时拟合并减去的 24h 日变谐波组合。阶数越高，去掉的周期成分越多。\n\n"
            "窗口天数：每次拟合所使用的数据长度。窗口越长，拟合越稳；过长可能削弱时变特征。\n\n"
            "步长天数：相邻两个拟合窗口之间的间隔。步长越小，窗口重叠越多，结果拼接越平滑。\n\n"
            "趋势阶数：拟合谐波时允许同时拟合的背景趋势阶数。0 表示常数背景，1 表示线性趋势。\n\n"
            "最小有效比例：一个窗口中有效数据所占的最小比例。低于该比例时，该窗口跳过拟合。\n\n"
            "最小有效点数：一个窗口至少需要多少个有效采样点。点数不足时，该窗口跳过拟合。\n\n"
            "显示去中心化波形：只影响绘图显示。勾选后会先减去均值，便于观察日变起伏，不影响实际计算。",
        )
