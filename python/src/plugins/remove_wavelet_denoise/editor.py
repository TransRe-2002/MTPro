from __future__ import annotations

"""小波多尺度去噪界面，处理传入的完整数据区间。"""

import argparse
import os
import sys
import tomllib

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import numpy as np
import pandas as pd
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg

from core.em_data import EMData
from core.algorithms.wavelet import wavelet_full_channel_denoise
from plugins.common.base_processor import BaseProcessorWidget
from plugins.common.demo import DemoEMData, run_standalone_demo
from plugins.common.region import RegionManagementMixin
from plugins.common.step_algorithms import WaveletDenoiseResult
from plugins.common.plotting import ProcessingPlotWidget, StepPlotWidget
from ui.widgets.gain_curve_editor import GainCurveEditor
from utils.series import dti_to_numpy
from utils.theme import configure_pyqtgraph_theme, pg_background, subtle_pen, transparent_legend_brush


class RemoveWaveletDenoise(BaseProcessorWidget, RegionManagementMixin):
    """小波多尺度去噪窗口。"""
    AUTO_PRESETS = {
        "自定义": None,
        "自适应保守（推荐）": {
            "transform": "DWT", "wavelet": "sym6", "mode": "soft",
            "threshold_method": "sure", "threshold": 1.0,
            "level_policy": "auto_by_period", "protect_period_seconds": 3600.0,
            "padding_unit": "seconds", "padding_seconds": 120.0,
        },
        "常规宽带降噪": {
            "transform": "DWT", "wavelet": "sym4", "mode": "soft",
            "threshold_method": "sure", "threshold": 1.0,
            "level_policy": "auto_by_period", "protect_period_seconds": 300.0,
            "padding_unit": "seconds", "padding_seconds": 60.0,
        },
        "边界安全去噪": {
            "transform": "SWT（平移稳定）", "wavelet": "sym6", "mode": "soft",
            "threshold_method": "sure", "threshold": 1.0,
            "level_policy": "auto_by_period", "protect_period_seconds": 900.0,
            "padding_unit": "seconds", "padding_seconds": 120.0,
        },
        "LMT：保守去噪（优先保长周期）": {
            "transform": "SWT（平移稳定）", "wavelet": "sym6", "mode": "soft",
            "threshold_method": "sure", "threshold": 0.70,
            "level_policy": "auto_by_period", "protect_period_seconds": 3600.0,
            "padding_unit": "seconds", "padding_seconds": 600.0,
        },
        "LMT：抑制高频粗噪声": {
            "transform": "DWT", "wavelet": "sym6", "mode": "soft",
            "threshold_method": "sure", "threshold": 1.20,
            "level_policy": "auto_by_period", "protect_period_seconds": 1800.0,
            "padding_unit": "seconds", "padding_seconds": 300.0,
        },
        "LMT：局部脉冲轻度压制": {
            "transform": "SWT（平移稳定）", "wavelet": "sym6", "mode": "soft",
            "threshold_method": "sure", "threshold": 1.35,
            "level_policy": "auto_by_period", "protect_period_seconds": 1800.0,
            "padding_unit": "seconds", "padding_seconds": 300.0,
        },
        "脉冲（旧版固定）": {
            "mode": "soft", "threshold_method": "universal", "threshold": 1.20,
            "padding_unit": "points", "padding": 256, "level_policy": "manual", "level": 6,
            "legacy": True,
        },
        "尖峰（旧版固定）": {
            "mode": "hard", "threshold_method": "universal", "threshold": 2.20,
            "padding_unit": "points", "padding": 64, "level_policy": "manual", "level": 4,
            "legacy": True,
        },
        "平滑降噪（旧版固定）": {
            "mode": "soft", "threshold_method": "universal", "threshold": 0.80,
            "padding_unit": "points", "padding": 512, "level_policy": "manual", "level": 6,
            "legacy": True,
        },
    }
    PRESET_NAME_ALIASES = {
        "脉冲": "脉冲（旧版固定）",
        "尖峰": "尖峰（旧版固定）",
        "平滑降噪": "平滑降噪（旧版固定）",
    }

    @staticmethod
    def _toml_value(value) -> str:
        """把 Python 值转成 TOML 文本，用于导出预设。"""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            text = f"{value:.12g}"
            return text if "." in text or "e" in text or "E" in text else f"{text}.0"
        if isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        if isinstance(value, list):
            return "[" + ", ".join(RemoveWaveletDenoise._toml_value(item) for item in value) + "]"
        raise TypeError(f"unsupported TOML value type: {type(value)!r}")

    @classmethod
    def _preset_dict_to_toml(cls, preset: dict) -> str:
        """把预设字典序列化为 TOML 字符串。"""
        lines = []
        nested_tables: list[tuple[str, dict]] = []
        for key, value in preset.items():
            if isinstance(value, dict):
                nested_tables.append((key, value))
            else:
                lines.append(f"{key} = {cls._toml_value(value)}")
        for table_name, table_values in nested_tables:
            lines.append("")
            lines.append(f"[{table_name}]")
            for key, value in table_values.items():
                lines.append(f"{key} = {cls._toml_value(value)}")
        return "\n".join(lines) + "\n"

    def apply_plugin_params(self, params: dict) -> None:
        preset = dict(params.get("editor_state", {}))
        preset.update({
            "transform_kind": params.get("transform", "dwt"),
            "wavelet": params.get("wavelet", "db4"),
            "level": params.get("level", 6),
            "level_policy": params.get("level_policy", "manual"),
            "protect_period_seconds": params.get("protect_period_seconds", 3600.0),
            "threshold_scale": params.get("threshold", 1.0),
            "threshold_method": params.get("threshold_method", "universal"),
            "padding_points": params.get("padding", 256),
            "padding_unit": params.get("padding_unit", "points"),
            "padding_seconds": params.get("padding_seconds", 0.0),
            "mode": params.get("mode", "soft"),
            "detail_gains": params.get("detail_gains", []),
        })
        self._apply_preset_dict(preset)

    def __init__(self, source, channel_name: str, parent=None):
        super().__init__(source, channel_name, parent)

        self.last_result: WaveletDenoiseResult | None = None
        self.selected_indices: list[int] = []
        self.gain_spins: list[QDoubleSpinBox] = []
        self.log_messages: list[str] = []

        self.signal_plot_widget = StepPlotWidget()
        self.plot_widget = self.signal_plot_widget
        self.signal_plot_item = self.signal_plot_widget.getPlotItem()
        self.detail_plot_widget = ProcessingPlotWidget()
        self.detail_plot_item = self.detail_plot_widget.getPlotItem()
        self.gain_curve_editor = GainCurveEditor(self, maximum=2.0)
        self.auto_eq_preview_checkbox = QCheckBox("松手后自动预览")
        self.auto_eq_preview_checkbox.setChecked(True)
        self.analysis_tabs = QTabWidget(self)
        self.eq_panel = QWidget(self)
        self.eq_preview_timer = QTimer(self)
        self.eq_preview_timer.setSingleShot(True)
        self.eq_preview_timer.setInterval(250)
        self.region_item: pg.LinearRegionItem | None = None
        self.plot_splitter = None

        self.param_frame = None
        self.strategy_frame = None
        self.gain_frame = None
        self.region_frame = None
        self.action_frame = None
        self.dynamic_gain_widget = None
        self.dynamic_gain_layout = None
        self.gain_layout = None

        self.wavelet_combo = QComboBox()
        self.transform_combo = QComboBox()
        self.level_spin = QSpinBox()
        self.level_policy_combo = QComboBox()
        self.protect_period_spin = QDoubleSpinBox()
        self.scope_combo = QComboBox()
        self.preset_combo = QComboBox()
        self.mode_combo = QComboBox()
        self.threshold_method_combo = QComboBox()
        self.threshold_scale_spin = QDoubleSpinBox()
        self.padding_unit_combo = QComboBox()
        self.padding_spin = QSpinBox()
        self.padding_seconds_spin = QDoubleSpinBox()
        self.global_gain_spin = QDoubleSpinBox()
        self.wp_low_gain_spin = QDoubleSpinBox()
        self.wp_mid_gain_spin = QDoubleSpinBox()
        self.wp_high_gain_spin = QDoubleSpinBox()
        self.wp_low_label = QLabel("低")
        self.wp_mid_label = QLabel("中")
        self.wp_high_label = QLabel("高")
        self.preview_btn = QPushButton("预览重构")
        self.apply_btn = QPushButton("应用去噪")
        self.add_region_btn = QPushButton("添加连续区域")
        self.remove_region_btn = QPushButton("移除最后区域")
        self.clear_region_btn = QPushButton("清空选择")
        self.reset_gains_btn = QPushButton("增益重置")
        self.scale_gains_btn = QPushButton("统一缩放")
        self.propagate_gains_btn = QPushButton("复制上一层")
        self.apply_wp_groups_btn = QPushButton("应用分组")
        self.export_preset_btn = QPushButton("导出预设")
        self.import_preset_btn = QPushButton("导入预设")
        self.selection_status_label = QLabel("当前选中: 0 个采样点")
        self.processing_status = QLabel("准备就绪；将处理全部输入数据。")
        self.processing_status.setWordWrap(True)
        self.undo_btn = QPushButton("撤销")
        self.undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self.revert_btn = QPushButton("还原原始数据")
        self.confirm_btn = QPushButton("确认")

        self._init_ui()
        self._connect_signal()
        self._load_input_data()


    def _on_data_loaded(self):
        self._sync_gain_controls()
        self._sync_scope_controls()
        self._sync_strategy_controls()
        self._update_level_limit()
        self._init_plot()

    def _init_ui(self):
        instruction_label = QLabel(
            "对输入数据整体进行小波去噪（Pipeline 中为当前 Block）；调整参数并预览。"
        )
        instruction_label.setWordWrap(True)
        instruction_label.setToolTip(
            "连续不超过 10 点的缺失值会插值，较长缺口会自动分段；过短片段保持原样。"
            "Ctrl 或 Shift 配合滚轮/中键拖动可分别操作 X 轴或 Y 轴。"
        )

        self.signal_plot_widget.getAxis("left").setStyle(tickLength=0)
        self.signal_plot_widget.getAxis("bottom").setStyle(tickLength=0)
        self.detail_plot_widget.getAxis("left").setStyle(tickLength=0)
        self.detail_plot_widget.getAxis("bottom").setStyle(tickLength=0)
        configure_pyqtgraph_theme(self.detail_plot_widget)
        self.detail_plot_widget.setBackground(pg_background(self.detail_plot_widget))

        self.wavelet_combo.addItems(["db4", "db6", "sym4", "sym6"])
        self.transform_combo.addItems(["DWT", "SWT（平移稳定）", "Wavelet Packet"])
        self.transform_combo.setCurrentText("DWT")
        self.level_spin.setRange(1, 12)
        self.level_spin.setValue(6)
        self.level_policy_combo.addItem("手动层数", "manual")
        self.level_policy_combo.addItem("按保护周期", "auto_by_period")
        self.protect_period_spin.setRange(0.001, 1_000_000_000.0)
        self.protect_period_spin.setDecimals(3)
        self.protect_period_spin.setSingleStep(60.0)
        self.protect_period_spin.setSuffix(" s")
        self.protect_period_spin.setValue(3600.0)
        self.scope_combo.addItem("全通道")
        self.preset_combo.addItems(list(self.AUTO_PRESETS.keys()))
        self.preset_combo.setCurrentText("自定义")
        preset_notes = {
            "LMT：保守去噪（优先保长周期）": (
                "低阈值、SWT 平移稳定，并保留保护周期以上的近似项。建议先框选短段预览，"
                "重点检查长周期趋势和阶跃边缘。低频 MT 文献指出常规时序编辑可能损失 1 Hz 以下慢变有效信号，"
                "因此这组参数采用偏保守设置（Li et al., 2020, doi:10.1016/j.jappgeo.2019.103919）。"
            ),
            "LMT：抑制高频粗噪声": (
                "逐层 SureShrink，较强地压制细节系数，同时保留长周期近似项。适用于宽带高频粗噪声的试探；"
                "若有效高频信号也被削弱，请降低阈值。MT 研究验证了小波分层阈值对尖脉冲和噪声段的用途，"
                "但阈值需随数据调整（Trad & Travassos, 2000, doi:10.1190/1.1444742）。"
            ),
            "LMT：局部脉冲轻度压制": (
                "适用于已框选的局部脉冲附近，采用 SWT 减少平移敏感性并保留慢变化成分。"
                "长周期 MT 常见噪声还包括尖峰和阶跃（Mair, 2003）；2026 年 LMT 研究使用噪声区段识别、"
                "Hampel 和阶跃补偿。此预设只做小波阈值处理，并不包含这些识别/补偿算法。"
            ),
        }
        for index in range(self.preset_combo.count()):
            preset_name = self.preset_combo.itemText(index)
            if preset_name in preset_notes:
                self.preset_combo.setItemData(index, preset_notes[preset_name], Qt.ItemDataRole.ToolTipRole)
        self.mode_combo.addItems(["soft", "hard", "zero", "manual"])
        self.mode_combo.setCurrentText("soft")
        for combo in (self.wavelet_combo, self.transform_combo, self.scope_combo,
                      self.preset_combo, self.mode_combo):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)
        self.threshold_method_combo.addItem("SureShrink（逐层自适应）", "sure")
        self.threshold_method_combo.addItem("通用阈值（兼容）", "universal")
        self.threshold_method_combo.setCurrentIndex(1)
        self.threshold_scale_spin.setRange(0.0, 20.0)
        self.threshold_scale_spin.setDecimals(2)
        self.threshold_scale_spin.setSingleStep(0.1)
        self.threshold_scale_spin.setValue(1.0)
        self.padding_spin.setRange(32, 200000)
        self.padding_spin.setSingleStep(32)
        self.padding_spin.setSuffix(" 点")
        self.padding_spin.setValue(256)
        self.padding_unit_combo.addItem("采样点", "points")
        self.padding_unit_combo.addItem("秒", "seconds")
        self.padding_seconds_spin.setRange(0.0, 1_000_000_000.0)
        self.padding_seconds_spin.setDecimals(3)
        self.padding_seconds_spin.setSingleStep(10.0)
        self.padding_seconds_spin.setSuffix(" s")
        self.padding_seconds_spin.setValue(120.0)
        self.global_gain_spin.setRange(0.0, 2.0)
        self.global_gain_spin.setDecimals(2)
        self.global_gain_spin.setSingleStep(0.1)
        self.global_gain_spin.setValue(1.0)
        for spin in (self.wp_low_gain_spin, self.wp_mid_gain_spin, self.wp_high_gain_spin):
            spin.setRange(0.0, 2.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.1)
            spin.setValue(1.0)

        self.param_frame = QFrame(self)
        self.param_frame.setFrameShape(QFrame.Shape.StyledPanel)
        param_layout = QFormLayout(self.param_frame)
        param_layout.setContentsMargins(8, 6, 8, 6)
        param_layout.addRow(QLabel("<b>小波参数</b>"))
        param_layout.addRow("小波基", self.wavelet_combo)
        param_layout.addRow("变换", self.transform_combo)
        param_layout.addRow("分解层数", self.level_spin)
        param_layout.addRow("自动预设", self.preset_combo)
        param_layout.addRow("模式", self.mode_combo)

        self.strategy_frame = QFrame(self)
        self.strategy_frame.setFrameShape(QFrame.Shape.StyledPanel)
        strategy_layout = QFormLayout(self.strategy_frame)
        strategy_layout.setContentsMargins(8, 6, 8, 6)
        for combo in (self.threshold_method_combo, self.level_policy_combo, self.padding_unit_combo):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)
        strategy_layout.addRow(QLabel("<b>去噪策略</b>"))
        strategy_layout.addRow("阈值", self.threshold_method_combo)
        strategy_layout.addRow("阈值倍数", self.threshold_scale_spin)
        strategy_layout.addRow("层数策略", self.level_policy_combo)
        strategy_layout.addRow("保护周期", self.protect_period_spin)
        strategy_layout.addRow("扩展单位", self.padding_unit_combo)
        strategy_layout.addRow("扩展采样点", self.padding_spin)
        strategy_layout.addRow("扩展时间", self.padding_seconds_spin)

        self.gain_frame = QFrame(self)
        self.gain_frame.setFrameShape(QFrame.Shape.NoFrame)
        self.gain_layout = QVBoxLayout(self.gain_frame)
        self.gain_layout.setContentsMargins(8, 6, 8, 6)
        eq_actions = QHBoxLayout()
        eq_actions.addWidget(QLabel("统一系数"))
        eq_actions.addWidget(self.global_gain_spin)
        eq_actions.addWidget(self.scale_gains_btn)
        eq_actions.addWidget(self.propagate_gains_btn)
        eq_actions.addWidget(self.reset_gains_btn)
        eq_actions.addWidget(self.export_preset_btn)
        eq_actions.addWidget(self.import_preset_btn)
        eq_actions.addStretch(1)
        self.gain_layout.addLayout(eq_actions)
        wp_actions = QHBoxLayout()
        for control in (self.wp_low_label, self.wp_low_gain_spin, self.wp_mid_label,
                        self.wp_mid_gain_spin, self.wp_high_label, self.wp_high_gain_spin,
                        self.apply_wp_groups_btn):
            wp_actions.addWidget(control)
        wp_actions.addStretch(1)
        self.gain_layout.addLayout(wp_actions)
        self.dynamic_gain_widget = QWidget(self.gain_frame)
        self.dynamic_gain_layout = QHBoxLayout(self.dynamic_gain_widget)
        self.dynamic_gain_layout.setContentsMargins(0, 0, 0, 0)
        self.dynamic_gain_layout.setSpacing(6)
        # Per-band spin boxes remain as the compatible parameter store for
        # recipes and presets, but the EQ curve is now their only visible
        # editor.  Keeping the widgets internal avoids changing the existing
        # detail_gains serialization contract.
        self.dynamic_gain_widget.hide()

        self.region_frame = QFrame(self)
        self.region_frame.hide()
        self.region_frame.setFrameShape(QFrame.Shape.StyledPanel)
        region_layout = QGridLayout(self.region_frame)
        region_layout.setContentsMargins(8, 6, 8, 6)
        region_layout.addWidget(QLabel("<b>区域选择</b>"), 0, 0, 1, 2)
        region_layout.addWidget(self.selection_status_label, 1, 0, 1, 2)
        region_layout.addWidget(self.add_region_btn, 2, 0)
        region_layout.addWidget(self.remove_region_btn, 2, 1)
        region_layout.addWidget(self.clear_region_btn, 3, 0, 1, 2)

        self.action_frame = QFrame(self)
        self.action_frame.setFrameShape(QFrame.Shape.StyledPanel)
        action_layout = QVBoxLayout(self.action_frame)
        action_layout.setContentsMargins(8, 6, 8, 6)
        preview_actions = QHBoxLayout()
        preview_actions.addWidget(self.preview_btn)
        preview_actions.addWidget(self.apply_btn)
        action_layout.addLayout(preview_actions)
        undo_actions = QHBoxLayout()
        undo_actions.addWidget(self.undo_btn)
        undo_actions.addWidget(self.revert_btn)
        action_layout.addLayout(undo_actions)
        action_layout.addWidget(self.confirm_btn)

        eq_layout = QVBoxLayout(self.eq_panel)
        eq_layout.setContentsMargins(6, 4, 6, 4)
        eq_header_layout = QHBoxLayout()
        eq_header_layout.setContentsMargins(0, 0, 0, 0)
        eq_hint = QLabel("拖动节点调整各频带增益；曲线用于展示节点之间的变化。")
        eq_hint.setWordWrap(True)
        self.auto_eq_preview_checkbox.setToolTip(
            "拖动结束后自动按当前增益重新计算预览；数据较大时可取消勾选。"
        )
        eq_header_layout.addWidget(eq_hint, 1)
        eq_header_layout.addWidget(self.auto_eq_preview_checkbox)
        eq_layout.addLayout(eq_header_layout)
        eq_layout.addWidget(self.gain_frame)
        eq_layout.addWidget(self.gain_curve_editor, 1)

        self.analysis_tabs.addTab(self.eq_panel, "频带 EQ")
        self.analysis_tabs.addTab(self.detail_plot_widget, "能量对比")
        self.analysis_tabs.setCurrentWidget(self.eq_panel)

        self.plot_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.plot_splitter.setChildrenCollapsible(False)
        self.plot_splitter.setOpaqueResize(False)
        self.plot_splitter.addWidget(self.signal_plot_widget)
        self.plot_splitter.addWidget(self.analysis_tabs)
        self.plot_splitter.setStretchFactor(0, 4)
        self.plot_splitter.setStretchFactor(1, 2)
        self.plot_splitter.setSizes([620, 260])

        controls = QWidget(self)
        control_layout = QVBoxLayout(controls)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.addWidget(self.param_frame)
        control_layout.addWidget(self.strategy_frame)
        control_layout.addStretch(1)
        control_scroll = QScrollArea(self)
        control_scroll.setWidgetResizable(True)
        control_scroll.setFrameShape(QFrame.Shape.NoFrame)
        control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        control_scroll.setWidget(controls)
        sidebar = QWidget(self)
        sidebar.setMinimumWidth(300)
        sidebar.setMaximumWidth(380)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.addWidget(control_scroll, 1)
        sidebar_layout.addWidget(self.processing_status)
        sidebar_layout.addWidget(self.action_frame)

        layout = QGridLayout(self)
        layout.addWidget(instruction_label, 0, 0, 1, 2)
        layout.addWidget(self.plot_splitter, 1, 0)
        layout.addWidget(sidebar, 1, 1)
        layout.setColumnStretch(0, 1)

    def _connect_signal(self):
        self.preview_btn.clicked.connect(self._preview_fit)
        self.apply_btn.clicked.connect(self._apply_correction)
        self.add_region_btn.clicked.connect(self._add_region)
        self.remove_region_btn.clicked.connect(self._remove_region)
        self.clear_region_btn.clicked.connect(self._clear_selection)
        self.reset_gains_btn.clicked.connect(self._reset_gains)
        self.scale_gains_btn.clicked.connect(self._scale_gains)
        self.propagate_gains_btn.clicked.connect(self._propagate_gains)
        self.apply_wp_groups_btn.clicked.connect(self._apply_wp_groups)
        self.export_preset_btn.clicked.connect(self._export_preset)
        self.import_preset_btn.clicked.connect(self._import_preset)
        self.wavelet_combo.currentTextChanged.connect(self._update_level_limit)
        self.transform_combo.currentTextChanged.connect(self._on_transform_changed)
        self.level_spin.valueChanged.connect(self._sync_gain_controls)
        self.level_policy_combo.currentIndexChanged.connect(self._on_level_policy_changed)
        self.scope_combo.currentTextChanged.connect(self._sync_scope_controls)
        self.scope_combo.currentTextChanged.connect(self._update_level_limit)
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        self.mode_combo.currentTextChanged.connect(self._sync_mode_controls)
        self.padding_unit_combo.currentIndexChanged.connect(self._on_padding_unit_changed)
        self.padding_spin.valueChanged.connect(self._update_level_limit)
        self.padding_seconds_spin.valueChanged.connect(self._update_level_limit)
        self.protect_period_spin.valueChanged.connect(self._update_level_limit)
        self.undo_btn.clicked.connect(self._undo)
        self.undo_shortcut.activated.connect(self._undo)
        self.revert_btn.clicked.connect(self._revert)
        self.confirm_btn.clicked.connect(self._confirm_and_close)
        self.gain_curve_editor.gain_changed.connect(self._on_eq_gain_changed)
        self.gain_curve_editor.editing_finished.connect(self._schedule_eq_preview)
        self.eq_preview_timer.timeout.connect(self._preview_fit)

    def _transform_kind(self) -> str:
        current = self.transform_combo.currentText()
        if current == "Wavelet Packet":
            return "wp"
        if current == "SWT（平移稳定）":
            return "swt"
        return "dwt"

    def _threshold_method(self) -> str:
        return str(self.threshold_method_combo.currentData() or "universal")

    def _level_policy(self) -> str:
        return str(self.level_policy_combo.currentData() or "manual")

    def _padding_unit(self) -> str:
        return str(self.padding_unit_combo.currentData() or "points")

    def _padding_points(self) -> int:
        if self._padding_unit() == "seconds":
            return int(np.ceil(float(self.padding_seconds_spin.value()) * self._sample_rate_hz()))
        return int(self.padding_spin.value())

    def _on_transform_changed(self, _value: str) -> None:
        self._sync_strategy_controls()
        self._update_level_limit()

    def _on_level_policy_changed(self, _index: int) -> None:
        self._sync_strategy_controls()
        self._update_level_limit()

    def _on_padding_unit_changed(self, _index: int) -> None:
        self._sync_strategy_controls()
        self._update_level_limit()

    def _sync_strategy_controls(self) -> None:
        is_wp = self._transform_kind() == "wp"
        if is_wp and self._level_policy() == "auto_by_period":
            self.level_policy_combo.blockSignals(True)
            self.level_policy_combo.setCurrentIndex(0)
            self.level_policy_combo.blockSignals(False)
        is_auto_level = self._level_policy() == "auto_by_period"
        self.level_spin.setEnabled(not is_auto_level)
        self.protect_period_spin.setEnabled(is_auto_level)
        is_seconds = self._padding_unit() == "seconds"
        self.padding_spin.setEnabled(not is_seconds)
        self.padding_seconds_spin.setEnabled(is_seconds)
        self.level_policy_combo.setEnabled(not is_wp)
        self.protect_period_spin.setToolTip(
            "近似项将保留不短于此值的周期；仅适用于 DWT/SWT。"
        )
        self.threshold_method_combo.setToolTip(
            "SureShrink 对每个细节层独立选阈值；通用阈值用于复现旧预设。"
        )

    def _sync_gain_controls(self):
        previous_values = [float(spin.value()) for spin in self.gain_spins]
        while self.dynamic_gain_layout.count() > 0:
            item = self.dynamic_gain_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.gain_spins.clear()
        band_count = self._current_band_count()
        for level in range(1, band_count + 1):
            label_prefix = "B" if self.transform_combo.currentText() == "Wavelet Packet" else "D"
            label = QLabel(f"{label_prefix}{level}")
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 2.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.1)
            spin.setValue(previous_values[level - 1] if level - 1 < len(previous_values) else 1.0)
            spin.setEnabled(self.mode_combo.currentText() == "manual")
            self.dynamic_gain_layout.addWidget(label)
            self.dynamic_gain_layout.addWidget(spin)
            spin.valueChanged.connect(
                lambda value, index=level - 1: self._on_manual_gain_changed(index, value)
            )
            spin.editingFinished.connect(self._schedule_eq_preview)
            self.gain_spins.append(spin)
        self._sync_mode_controls()
        self._update_wp_group_labels()
        self._sync_gain_curve()

    def _current_band_count(self) -> int:
        if self.transform_combo.currentText() == "Wavelet Packet":
            level = max(1, int(self.level_spin.value()))
            return max(1, min(64, 2 ** level))
        return max(1, int(self.level_spin.value()))

    def _current_wp_group_ranges(self) -> list[tuple[float, float]]:
        if self.transform_combo.currentText() != "Wavelet Packet":
            return [(0.0, 0.0)] * 3
        band_count = self._current_band_count()
        sample_rate_hz = self._sample_rate_hz()
        nyquist = 0.5 * sample_rate_hz
        band_width = nyquist / float(max(1, band_count))
        group_edges = np.array_split(np.arange(band_count, dtype=int), 3)
        ranges: list[tuple[float, float]] = []
        for indices in group_edges:
            if len(indices) == 0:
                ranges.append((0.0, 0.0))
                continue
            low = float(indices[0]) * band_width
            high = float(indices[-1] + 1) * band_width
            ranges.append((low, high))
        return ranges

    def _update_wp_group_labels(self) -> None:
        ranges = self._current_wp_group_ranges()
        base_labels = ["低", "中", "高"]
        widgets = [self.wp_low_label, self.wp_mid_label, self.wp_high_label]
        is_wp = self.transform_combo.currentText() == "Wavelet Packet"
        for base, widget, (low, high) in zip(base_labels, widgets, ranges):
            if is_wp:
                widget.setText(f"{base}[{low:.4g}-{high:.4g}Hz]")
            else:
                widget.setText(base)

    @staticmethod
    def _format_eq_frequency(value: float) -> str:
        value = float(value)
        if value == 0.0:
            return "0"
        if abs(value) >= 1000.0:
            return f"{value / 1000.0:.4g}k"
        return f"{value:.4g}"

    def _gain_band_descriptors(self) -> tuple[list[str], list[int]]:
        """Return display labels and gain indices ordered from low to high."""
        band_count = len(self.gain_spins)
        if band_count == 0:
            return [], []
        nyquist = 0.5 * self._sample_rate_hz()
        if self._transform_kind() == "wp":
            band_width = nyquist / float(band_count)
            logical_indices = list(range(band_count))
            ranges = [
                (index * band_width, (index + 1) * band_width)
                for index in logical_indices
            ]
            prefix = "B"
        else:
            logical_indices = list(reversed(range(band_count)))
            ranges = [
                (
                    nyquist / (2 ** (logical_index + 1)),
                    nyquist / (2 ** logical_index),
                )
                for logical_index in logical_indices
            ]
            prefix = "D"
        labels = [
            (
                f"{prefix}{logical_index + 1}\n"
                f"{self._format_eq_frequency(low)}–{self._format_eq_frequency(high)} Hz"
            )
            for logical_index, (low, high) in zip(logical_indices, ranges)
        ]
        return labels, logical_indices

    def _sync_gain_curve(self) -> None:
        labels, logical_indices = self._gain_band_descriptors()
        gains = [float(self.gain_spins[index].value()) for index in logical_indices]
        self.gain_curve_editor.set_bands(labels, gains, logical_indices)

    def _sync_mode_controls(self):
        is_manual = self.mode_combo.currentText() == "manual"
        is_wp = self.transform_combo.currentText() == "Wavelet Packet"
        self.threshold_scale_spin.setEnabled(not is_manual)
        self.threshold_method_combo.setEnabled(not is_manual)
        self.preset_combo.setEnabled(not is_manual)
        self.global_gain_spin.setEnabled(True)
        self.scale_gains_btn.setEnabled(True)
        self.propagate_gains_btn.setEnabled(True)
        for widget in (
            self.wp_low_label,
            self.wp_low_gain_spin,
            self.wp_mid_label,
            self.wp_mid_gain_spin,
            self.wp_high_label,
            self.wp_high_gain_spin,
            self.apply_wp_groups_btn,
        ):
            widget.setVisible(is_wp)
            widget.setEnabled(is_wp)
        for spin in self.gain_spins:
            spin.setEnabled(True)
        if is_manual:
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentText("自定义")
            self.preset_combo.blockSignals(False)

    def _sync_scope_controls(self):
        is_local = self.scope_combo.currentText() == "局部选区"
        self.add_region_btn.setEnabled(is_local)
        self.remove_region_btn.setEnabled(is_local)
        self.clear_region_btn.setEnabled(is_local)
        if not is_local:
            self.selection_status_label.setText(f"当前范围: 全部输入数据 ({self.y_data.size} 个采样点)")
        elif self.region_item is None:
            self.selection_status_label.setText("当前选中: 0 个采样点")
        else:
            self._update_region_selection()

    def _reset_gains(self):
        for spin in self.gain_spins:
            spin.setValue(1.0)
        self._log("已将所有细节层增益重置为 1.00")

    def _scale_gains(self):
        factor = float(self.global_gain_spin.value())
        self._ensure_manual_mode()
        for spin in self.gain_spins:
            spin.setValue(factor)
        self._log(f"已将所有细节层增益统一设置为 {factor:.2f}")

    def _propagate_gains(self):
        if not self.gain_spins:
            return
        self._ensure_manual_mode()
        last_value = float(self.gain_spins[0].value())
        for spin in self.gain_spins[1:]:
            spin.setValue(last_value)
            last_value = float(spin.value())
        self._log("已将前一层增益逐层复制到后一层")

    def _apply_wp_groups(self):
        if self.transform_combo.currentText() != "Wavelet Packet" or not self.gain_spins:
            return
        self._ensure_manual_mode()
        band_count = len(self.gain_spins)
        group_edges = np.array_split(np.arange(band_count, dtype=int), 3)
        group_values = [
            float(self.wp_low_gain_spin.value()),
            float(self.wp_mid_gain_spin.value()),
            float(self.wp_high_gain_spin.value()),
        ]
        group_labels = ["低频", "中频", "高频"]
        group_ranges = self._current_wp_group_ranges()
        for indices, value in zip(group_edges, group_values):
            for index in indices:
                self.gain_spins[int(index)].setValue(value)
        self._log(
            "已应用 Wavelet Packet 分组增益 | "
            + " | ".join(
                f"{label}={value:.2f}[{low:.4g}-{high:.4g}Hz]"
                for label, value, (low, high) in zip(group_labels, group_values, group_ranges)
            )
        )

    def _ensure_manual_mode(self):
        if self.mode_combo.currentText() != "manual":
            self.mode_combo.setCurrentText("manual")

    def _on_manual_gain_changed(self, logical_index: int, value: float):
        self.gain_curve_editor.set_gain(logical_index, value)
        if self.mode_combo.currentText() != "manual":
            current_values = [spin.value() for spin in self.gain_spins]
            if any(abs(v - 1.0) > 1e-9 for v in current_values):
                self._ensure_manual_mode()

    def _on_eq_gain_changed(self, logical_index: int, value: float) -> None:
        if not 0 <= logical_index < len(self.gain_spins):
            return
        spin = self.gain_spins[logical_index]
        spin.blockSignals(True)
        try:
            spin.setValue(value)
        finally:
            spin.blockSignals(False)
        actual_value = float(spin.value())
        if abs(actual_value - float(value)) > 1e-12:
            self.gain_curve_editor.set_gain(logical_index, actual_value)
        self._ensure_manual_mode()

    def _schedule_eq_preview(self) -> None:
        if not self.auto_eq_preview_checkbox.isChecked():
            return
        if not getattr(self, "y_data", np.array([])).size:
            return
        if self.scope_combo.currentText() == "局部选区" and not self.selected_indices:
            return
        if getattr(self, "_busy_dialog", None) is not None:
            return
        self.eq_preview_timer.start()

    def _apply_preset(self, preset_name: str):
        preset = self.AUTO_PRESETS.get(preset_name)
        if preset is None:
            return
        transform = str(preset.get("transform", self.transform_combo.currentText()))
        if self.transform_combo.findText(transform) >= 0:
            self.transform_combo.setCurrentText(transform)
        wavelet = str(preset.get("wavelet", self.wavelet_combo.currentText()))
        if self.wavelet_combo.findText(wavelet) >= 0:
            self.wavelet_combo.setCurrentText(wavelet)
        self.mode_combo.setCurrentText(str(preset.get("mode", self.mode_combo.currentText())))
        threshold_index = self.threshold_method_combo.findData(
            str(preset.get("threshold_method", self._threshold_method()))
        )
        if threshold_index >= 0:
            self.threshold_method_combo.setCurrentIndex(threshold_index)
        policy_index = self.level_policy_combo.findData(
            str(preset.get("level_policy", self._level_policy()))
        )
        if policy_index >= 0:
            self.level_policy_combo.setCurrentIndex(policy_index)
        padding_index = self.padding_unit_combo.findData(
            str(preset.get("padding_unit", self._padding_unit()))
        )
        if padding_index >= 0:
            self.padding_unit_combo.setCurrentIndex(padding_index)
        self.threshold_scale_spin.setValue(float(preset.get("threshold", self.threshold_scale_spin.value())))
        self.padding_spin.setValue(int(preset.get("padding", self.padding_spin.value())))
        self.padding_seconds_spin.setValue(float(preset.get("padding_seconds", self.padding_seconds_spin.value())))
        self.protect_period_spin.setValue(
            float(preset.get("protect_period_seconds", self.protect_period_spin.value()))
        )
        self._sync_strategy_controls()
        self._update_level_limit()
        if self._level_policy() == "manual":
            self.level_spin.setValue(min(int(preset.get("level", self.level_spin.value())), self.level_spin.maximum()))
        legacy_text = "（旧版通用阈值，仅用于兼容复现）" if preset.get("legacy") else ""
        self._log(
            f"已应用自动预设: {preset_name}{legacy_text} | transform={self._transform_kind()} | "
            f"threshold={self._threshold_method()}×{self.threshold_scale_spin.value():.2f} | "
            f"level={self.level_spin.value()} | padding={self._padding_points()} points"
        )

    def _state_to_preset_dict(self) -> dict:
        return {
            "version": 2,
            "transform_kind": self._transform_kind(),
            "wavelet": self.wavelet_combo.currentText(),
            "level": int(self.level_spin.value()),
            "level_policy": self._level_policy(),
            "protect_period_seconds": float(self.protect_period_spin.value()),
            "scope": self.scope_combo.currentText(),
            "preset_name": self.preset_combo.currentText(),
            "mode": self.mode_combo.currentText(),
            "threshold_method": self._threshold_method(),
            "threshold_scale": float(self.threshold_scale_spin.value()),
            "padding_unit": self._padding_unit(),
            "padding_points": int(self.padding_spin.value()),
            "padding_seconds": float(self.padding_seconds_spin.value()),
            "global_gain": float(self.global_gain_spin.value()),
            "wp_group_gains": {
                "low": float(self.wp_low_gain_spin.value()),
                "mid": float(self.wp_mid_gain_spin.value()),
                "high": float(self.wp_high_gain_spin.value()),
            },
            "detail_gains": [float(spin.value()) for spin in self.gain_spins],
        }

    def _apply_preset_dict(self, preset: dict) -> None:
        transform_kind = str(preset.get("transform_kind", "dwt")).lower()
        transform_label = {
            "dwt": "DWT",
            "swt": "SWT（平移稳定）",
            "wp": "Wavelet Packet",
        }.get(transform_kind, "DWT")
        self.transform_combo.setCurrentText(transform_label)

        wavelet = str(preset.get("wavelet", self.wavelet_combo.currentText()))
        if self.wavelet_combo.findText(wavelet) >= 0:
            self.wavelet_combo.setCurrentText(wavelet)

        scope = str(preset.get("scope", self.scope_combo.currentText()))
        if self.scope_combo.findText(scope) >= 0:
            self.scope_combo.setCurrentText(scope)

        preset_name = self.PRESET_NAME_ALIASES.get(str(preset.get("preset_name", "自定义")), str(preset.get("preset_name", "自定义")))
        if self.preset_combo.findText(preset_name) >= 0 and preset_name != "自定义":
            self.preset_combo.setCurrentText(preset_name)
        else:
            self.preset_combo.setCurrentText("自定义")

        mode = str(preset.get("mode", self.mode_combo.currentText()))
        if self.mode_combo.findText(mode) >= 0:
            self.mode_combo.setCurrentText(mode)

        threshold_index = self.threshold_method_combo.findData(
            str(preset.get("threshold_method", "universal"))
        )
        if threshold_index >= 0:
            self.threshold_method_combo.setCurrentIndex(threshold_index)
        policy_index = self.level_policy_combo.findData(
            str(preset.get("level_policy", "manual"))
        )
        if policy_index >= 0:
            self.level_policy_combo.setCurrentIndex(policy_index)
        padding_index = self.padding_unit_combo.findData(
            str(preset.get("padding_unit", "points"))
        )
        if padding_index >= 0:
            self.padding_unit_combo.setCurrentIndex(padding_index)
        self.threshold_scale_spin.setValue(float(preset.get("threshold_scale", self.threshold_scale_spin.value())))
        self.padding_spin.setValue(int(preset.get("padding_points", preset.get("padding", self.padding_spin.value()))))
        self.padding_seconds_spin.setValue(float(preset.get("padding_seconds", self.padding_seconds_spin.value())))
        self.protect_period_spin.setValue(
            float(preset.get("protect_period_seconds", self.protect_period_spin.value()))
        )
        self.global_gain_spin.setValue(float(preset.get("global_gain", self.global_gain_spin.value())))
        group_gains = preset.get("wp_group_gains", {})
        if isinstance(group_gains, dict):
            self.wp_low_gain_spin.setValue(float(group_gains.get("low", self.wp_low_gain_spin.value())))
            self.wp_mid_gain_spin.setValue(float(group_gains.get("mid", self.wp_mid_gain_spin.value())))
            self.wp_high_gain_spin.setValue(float(group_gains.get("high", self.wp_high_gain_spin.value())))
        self._sync_strategy_controls()
        self._update_level_limit()
        if self._level_policy() == "manual":
            self.level_spin.setValue(min(int(preset.get("level", self.level_spin.value())), self.level_spin.maximum()))

        detail_gains = preset.get("detail_gains", [])
        if isinstance(detail_gains, list):
            for spin, value in zip(self.gain_spins, detail_gains):
                spin.setValue(float(value))

    def _export_preset(self):
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "导出小波去噪预设",
            "",
            "TOML Files (*.toml)",
        )
        if not filename:
            return
        try:
            if not filename.lower().endswith(".toml"):
                filename = f"{filename}.toml"
            with open(filename, "w", encoding="utf-8") as fh:
                fh.write(self._preset_dict_to_toml(self._state_to_preset_dict()))
        except Exception as exc:
            self._log(f"导出预设失败: {exc}")
            return
        self._log(f"已导出小波去噪预设: {filename}")

    def _import_preset(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "导入小波去噪预设",
            "",
            "TOML Files (*.toml)",
        )
        if not filename:
            return
        try:
            with open(filename, "rb") as fh:
                preset = tomllib.load(fh)
            if not isinstance(preset, dict):
                raise ValueError("预设文件格式无效")
            self._apply_preset_dict(preset)
        except Exception as exc:
            self._log(f"导入预设失败: {exc}")
            return
        self._log(f"已导入小波去噪预设: {filename}")

    def _processing_region(self) -> tuple[int, int] | None:
        if self.y_data is None or self.y_data.size < 2:
            return None
        return 0, self.y_data.size - 1

    def _sample_rate_hz(self) -> float:
        try:
            dt_seconds = float(self._pkg.get("dt", 0.0))
        except Exception:
            dt_seconds = 0.0
        if dt_seconds <= 0.0:
            try:
                dt_seconds = float(self._source.dt.total_seconds())
            except Exception:
                dt_seconds = 0.0
        if not np.isfinite(dt_seconds) or dt_seconds <= 0.0:
            return 1.0
        return 1.0 / dt_seconds

    def _effective_window_length(self) -> int:
        region = self._processing_region()
        if region is None:
            return self.y_data.size
        left, right = region
        if self.scope_combo.currentText() == "全通道":
            return self.y_data.size
        selected_width = right - left + 1
        half_window = max(self._padding_points(), int(np.ceil(selected_width * 0.5)))
        local_start = max(0, left - half_window)
        local_stop = min(self.y_data.size, right + half_window + 1)
        return max(1, local_stop - local_start)

    def _update_level_limit(self):
        current_value = int(self.level_spin.value())
        new_max = 12
        try:
            import pywt  # type: ignore

            wavelet = pywt.Wavelet(self.wavelet_combo.currentText())
            effective_length = self._effective_window_length()
            if self.transform_combo.currentText() == "Wavelet Packet":
                computed = int(pywt.dwt_max_level(int(effective_length), wavelet.dec_len))
                new_max = max(1, min(6, computed if computed > 0 else 1))
            elif self._transform_kind() == "swt":
                computed = int(np.floor(np.log2(max(2, effective_length))))
                new_max = max(1, min(10, computed))
            else:
                computed = int(pywt.dwt_max_level(int(effective_length), wavelet.dec_len))
                new_max = max(1, min(12, computed if computed > 0 else 1))
        except Exception:
            new_max = 12

        self.level_spin.blockSignals(True)
        self.level_spin.setMaximum(new_max)
        if self._level_policy() == "auto_by_period":
            protected_samples = max(2.0, self._sample_rate_hz() * float(self.protect_period_spin.value()))
            automatic_level = max(1, int(np.ceil(np.log2(protected_samples))) - 1)
            self.level_spin.setValue(min(automatic_level, new_max))
        else:
            self.level_spin.setValue(min(current_value, new_max))
        self.level_spin.blockSignals(False)
        self._sync_gain_controls()
        self._update_wp_group_labels()

    def _init_plot(self):
        reference_level = np.nanmean(self.origin_y_data)
        y_centered = self.y_data - reference_level
        origin_centered = self.origin_y_data - reference_level
        curve_opts = dict(
            autoDownsample=True, downsampleMethod='peak',
            clipToView=True, useOpenGL=True, connect='finite',
        )
        self.origin_line = StepPlotWidget.make_curve(
            self.x_data, origin_centered,
            pen=pg.mkPen(120, 120, 120, width=1, style=Qt.PenStyle.DashLine),
            name="原始序列", **curve_opts,
        )
        self.y_line = StepPlotWidget.make_curve(
            self.x_data, y_centered,
            pen=pg.mkPen(210, 70, 70, width=1),
            name="当前序列", **curve_opts,
        )
        self.preview_line = StepPlotWidget.make_curve(
            self.x_data, y_centered,
            pen=pg.mkPen(40, 90, 200, width=1),
            name="去噪预览", **curve_opts,
        )
        legend = self.signal_plot_item.addLegend()
        legend.anchor((1, 0), (1, 0))
        legend.setBrush(transparent_legend_brush(self))
        legend.setPen(subtle_pen(self))
        self.signal_plot_item.showGrid(x=True, y=True, alpha=0.7)
        self.signal_plot_item.addItem(self.origin_line)
        self.signal_plot_item.addItem(self.y_line)
        self.signal_plot_item.addItem(self.preview_line)
        self.local_left_line = pg.InfiniteLine(
            pos=float(self.x_data[0]) if self.x_data.size else 0.0,
            angle=90,
            pen=pg.mkPen(220, 140, 40, width=1, style=Qt.PenStyle.DashLine),
            movable=False,
        )
        self.local_right_line = pg.InfiniteLine(
            pos=float(self.x_data[0]) if self.x_data.size else 0.0,
            angle=90,
            pen=pg.mkPen(220, 140, 40, width=1, style=Qt.PenStyle.DashLine),
            movable=False,
        )
        self.local_left_line.hide()
        self.local_right_line.hide()
        self.signal_plot_item.addItem(self.local_left_line)
        self.signal_plot_item.addItem(self.local_right_line)

        self.detail_plot_item.setLabel("bottom", "细节层级")
        self.detail_plot_item.setLabel("left", "能量")
        detail_legend = self.detail_plot_item.addLegend()
        detail_legend.anchor((1, 0), (1, 0))
        detail_legend.setBrush(transparent_legend_brush(self))
        detail_legend.setPen(subtle_pen(self))
        self.detail_plot_item.showGrid(x=False, y=True, alpha=0.7)
        self.energy_before_bars = pg.BarGraphItem(
            x=np.array([1.0]),
            height=np.array([0.0]),
            width=0.36,
            brush=pg.mkBrush(180, 180, 180, 180),
            pen=pg.mkPen(120, 120, 120, width=1),
            name="处理前能量",
        )
        self.energy_after_bars = pg.BarGraphItem(
            x=np.array([1.4]),
            height=np.array([0.0]),
            width=0.36,
            brush=pg.mkBrush(80, 130, 220, 180),
            pen=pg.mkPen(40, 90, 200, width=1),
            name="处理后能量",
        )
        self.detail_plot_item.addItem(self.energy_before_bars)
        self.detail_plot_item.addItem(self.energy_after_bars)
        self.detail_plot_item.getAxis("bottom").setTicks([[(1.2, "D1")]])

    # ------------------------------------------------------------------
    # BaseProcessorWidget 接口
    # ------------------------------------------------------------------

    def _log(self, msg: str):
        self.log_messages.append(msg)

    def _reset_result_state(self):
        self.last_result = None

    def _update_plot(self):
        reference_level = np.nanmean(self.origin_y_data)
        y_centered = self.y_data - reference_level
        origin_centered = self.origin_y_data - reference_level
        self.origin_line.setData(self.x_data, origin_centered)
        self.y_line.setData(self.x_data, y_centered)

        if self.last_result is None:
            self.preview_line.setData(self.x_data, y_centered)
            self.local_left_line.hide()
            self.local_right_line.hide()
            self._update_detail_energy_plot(None)
            return

        preview_centered = self.last_result.corrected - reference_level
        self.preview_line.setData(self.x_data, preview_centered)
        self.local_left_line.hide()
        self.local_right_line.hide()
        self._update_detail_energy_plot(self.last_result)

    def _update_detail_energy_plot(self, result: WaveletDenoiseResult | None) -> None:
        if result is None or result.detail_energy_before.size == 0:
            self.energy_before_bars.setOpts(x=np.array([1.0]), height=np.array([0.0]), width=0.36)
            self.energy_after_bars.setOpts(x=np.array([1.4]), height=np.array([0.0]), width=0.36)
            self.detail_plot_item.getAxis("bottom").setTicks([[(1.2, "D1")]])
            self.detail_plot_item.setXRange(0.5, 1.9, padding=0.0)
            return

        levels = np.arange(1, result.detail_energy_before.size + 1, dtype=np.float64)
        before_x = levels - 0.18
        after_x = levels + 0.18
        self.energy_before_bars.setOpts(x=before_x, height=result.detail_energy_before, width=0.32)
        self.energy_after_bars.setOpts(x=after_x, height=result.detail_energy_after, width=0.32)
        short_labels = [label.replace(" [", "\n[") for label in result.band_labels]
        ticks = [[(float(level), label) for level, label in zip(levels, short_labels)]]
        self.detail_plot_item.getAxis("bottom").setTicks(ticks)
        self.detail_plot_item.setXRange(0.4, float(levels[-1]) + 0.6, padding=0.0)

    # ------------------------------------------------------------------
    # RegionManagementMixin 钩子
    # ------------------------------------------------------------------

    def _on_region_changed(self) -> None:
        self._update_level_limit()

    def _update_status_label(self, count: int) -> None:
        if self.scope_combo.currentText() == "局部选区":
            self.selection_status_label.setText(f"当前选中: {count} 个采样点")

    # ------------------------------------------------------------------
    # 区域选择（读）和业务逻辑
    # ------------------------------------------------------------------

    def _read_region(self) -> tuple[int, int] | None:
        return self._processing_region()

    def _clear_selection(self):
        self._remove_region()

    def _current_detail_gains(self) -> list[float]:
        return [float(spin.value()) for spin in self.gain_spins]

    def _read_denoise_args(self) -> dict | None:
        """在 UI 线程读取去噪所需的全部参数（含切片数据），返回纯数据 dict。

        返回 ``None`` 表示参数/区域不合法。返回的 dict 只含 numpy/标量，
        可以安全地捕获进闭包丢到 compute 池里执行。
        """
        region = self._read_region()
        if region is None:
            return None
        return {
            "y": self.y_data,
            "region": region,
            "transform_kind": self._transform_kind(),
            "wavelet_name": self.wavelet_combo.currentText(),
            "level": int(self.level_spin.value()),
            "sample_rate_hz": self._sample_rate_hz(),
            "threshold_scale": float(self.threshold_scale_spin.value()),
            "threshold_method": self._threshold_method(),
            "level_policy": self._level_policy(),
            "protect_period_seconds": (
                float(self.protect_period_spin.value())
                if self._level_policy() == "auto_by_period" else None
            ),
            "padding_points": self._padding_points(),
            "padding_seconds": (
                float(self.padding_seconds_spin.value())
                if self._padding_unit() == "seconds" else None
            ),
            "mode": self.mode_combo.currentText(),
            "detail_gains": self._current_detail_gains(),
        }

    @staticmethod
    def _denoise_compute(args: dict) -> WaveletDenoiseResult:
        """纯计算：在 compute 池里执行，不碰任何 Qt 对象。"""
        y = args["y"]
        kwargs = {k: v for k, v in args.items() if k not in ("y", "region")}
        return wavelet_full_channel_denoise(y, **kwargs)

    def _log_preview_summary(self, result: WaveletDenoiseResult):
        left, right = result.region
        local_start, local_stop = result.local_window
        local_before = self.y_data[local_start:local_stop + 1]
        local_after = result.corrected[local_start:local_stop + 1]
        finite_before = local_before[np.isfinite(local_before)]
        finite_after = local_after[np.isfinite(local_after)]
        rms_before = float(np.std(finite_before)) if finite_before.size else float("nan")
        rms_after = float(np.std(finite_after)) if finite_after.size else float("nan")
        std_before = rms_before
        std_after = rms_after
        jump_text = ""
        if left > 0 and right + 1 < self.y_data.size:
            before_left = np.nanmean(self.y_data[max(0, left - 40):left])
            before_right = np.nanmean(self.y_data[right + 1:min(self.y_data.size, right + 41)])
            after_left = np.nanmean(result.corrected[max(0, left - 40):left])
            after_right = np.nanmean(result.corrected[right + 1:min(result.corrected.size, right + 41)])
            jump_text = f" | jump before={before_right - before_left:.4f} | jump after={after_right - after_left:.4f}"

        self._log(
            f"小波多尺度去噪预览完成 | region=[{left}, {right}] | local window=[{local_start}, {local_stop}] | "
            f"transform={result.transform_kind} | wavelet={result.wavelet_name} | level={result.level} | mode={result.mode} | "
            f"threshold={result.threshold_method}×{result.threshold_scale:.2f} | "
            f"level_policy={result.level_policy} | changed coeffs={result.changed_coefficients}"
        )
        if result.thresholds.size:
            threshold_text = ", ".join(
                f"{label}={value:.4g}" for label, value in zip(result.band_labels, result.thresholds)
            )
            self._log(f"  各尺度阈值: {threshold_text}")
        if result.mode == "manual":
            gains_text = ", ".join(f"{label}={gain:.2f}" for label, gain in zip(result.band_labels, result.detail_gains))
            self._log(f"  手动增益: {gains_text}")
        if result.detail_energy_before.size:
            energy_before = float(np.sum(result.detail_energy_before))
            energy_after = float(np.sum(result.detail_energy_after))
            self._log(
                f"  细节能量 before={energy_before:.6f} | after={energy_after:.6f} | "
                f"local rms before={rms_before:.6f} | after={rms_after:.6f} | "
                f"local std before={std_before:.6f} | after={std_after:.6f}{jump_text}"
            )
            for label, before, after in zip(result.band_labels, result.detail_energy_before, result.detail_energy_after):
                self._log(f"    {label}: energy before={before:.6f} | after={after:.6f}")
        else:
            self._log(
                f"  local rms before={rms_before:.6f} | after={rms_after:.6f} | "
                f"local std before={std_before:.6f} | after={std_after:.6f}{jump_text}"
            )
        self._log("")

    def _preview_fit(self):
        args = self._read_denoise_args()
        if args is None:
            return
        self._run_heavy(
            lambda: self._denoise_compute(args),
            self._finish_preview,
            "小波多尺度去噪（预览）",
        )

    def _finish_preview(self, result):
        if result is None:
            return
        self.last_result = result
        self.processing_status.setText(f"预览完成：{result.message}")
        self._update_plot()
        self._log_preview_summary(result)

    def _apply_correction(self):
        args = self._read_denoise_args()
        if args is None:
            return
        self._run_heavy(
            lambda: self._denoise_compute(args),
            self._finish_apply,
            "小波多尺度去噪（应用）",
        )

    def _finish_apply(self, result):
        if result is None:
            return
        self._save_history()
        self.y_data = result.corrected.copy()
        self.last_result = result
        self.processing_status.setText(f"已应用：{result.message}")
        self._update_plot()
        self._log_preview_summary(result)
        self._log("已将小波多尺度去噪结果应用到当前输入数据")
        self._log("")

    def _on_heavy_failed(self, exc) -> None:
        super()._on_heavy_failed(exc)
        self.processing_status.setText(f"处理失败：{exc}")


# ------------------------------------------------------------------
# Standalone demo
# ------------------------------------------------------------------

def _build_demo_channel() -> tuple[EMData, str]:
    npts = 500_000
    dt = pd.Timedelta(seconds=5)
    index = pd.date_range("2024-05-01 00:00:00", periods=npts, freq=dt)
    x = np.arange(npts, dtype=np.float64)
    baseline = 4.0 + 0.00001 * x + 0.10 * np.sin(x / 4000.0)
    noise = 0.02 * np.random.default_rng(7).standard_normal(npts)
    pulse = np.zeros(npts, dtype=np.float64)

    pulse_center = 210_000
    rise_width = 40
    decay_width = 240
    amplitude = 6.0
    rise_start = pulse_center - rise_width // 2
    rise_stop = rise_start + rise_width
    decay_stop = rise_stop + decay_width

    rise_x = np.linspace(0.0, 1.0, rise_stop - rise_start, endpoint=False, dtype=np.float64)
    decay_x = np.linspace(0.0, 1.0, decay_stop - rise_stop, endpoint=False, dtype=np.float64)
    pulse[rise_start:rise_stop] = amplitude * rise_x
    pulse[rise_stop:decay_stop] = amplitude * np.exp(-4.0 * decay_x)

    samples = baseline + noise + pulse
    series = pd.Series(samples, index=index, name="Ex1")
    demo = DemoEMData("remove-wavelet-denoise-demo", "/tmp/remove_wavelet_denoise_demo.mat", "Ex1", series)
    return demo, "Ex1"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone preview for the wavelet multiscale denoise widget.",
    )
    parser.add_argument("--mat", help="可选：加载真实 MAT 文件进行预览")
    parser.add_argument("--channel", help="可选：指定 MAT 文件中的通道名；不指定则使用第一个通道")
    args = parser.parse_args()

    run_standalone_demo(RemoveWaveletDenoise, "小波多尺度去噪演示", args.mat, args.channel, _build_demo_channel)


if __name__ == "__main__":
    main()
