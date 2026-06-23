from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QStandardItem, QStandardItemModel

DATA_TREE_SELECTION_MIME = "application/x-data-tree-selection"
CLASSIC_TOOL_MIME = "application/x-widgettype"
PIPELINE_TOOL_MIME = "application/x-pipeline-tool"


def create_classic_tool_model() -> QStandardItemModel:
    tree_model = QStandardItemModel()
    tree_model.setHorizontalHeaderLabels(["数据处理工具"])

    cat_manual_process = QStandardItem("手动数据处理")
    cat_auto_process = QStandardItem("自动数据处理")
    cat_output = QStandardItem("阻抗输出")

    remove_region_item = QStandardItem("框选区域去噪")
    remove_region_item.setData("remove region", Qt.UserRole + 1)
    cat_manual_process.appendRow(remove_region_item)

    remove_spike_item = QStandardItem("框选尖峰去噪")
    remove_spike_item.setData("remove spike", Qt.UserRole + 1)
    cat_manual_process.appendRow(remove_spike_item)

    remove_step_diff_item = QStandardItem("差分去阶跃")
    remove_step_diff_item.setData("remove step diff", Qt.UserRole + 1)
    cat_manual_process.appendRow(remove_step_diff_item)

    remove_step_window_item = QStandardItem("窗口均值去阶跃")
    remove_step_window_item.setData("remove step window", Qt.UserRole + 1)
    cat_auto_process.appendRow(remove_step_window_item)

    remove_step_baseline_item = QStandardItem("鲁棒基线去阶跃")
    remove_step_baseline_item.setData("remove step baseline", Qt.UserRole + 1)
    cat_auto_process.appendRow(remove_step_baseline_item)

    remove_wavelet_denoise_item = QStandardItem("小波多尺度去噪")
    remove_wavelet_denoise_item.setData("remove wavelet denoise", Qt.UserRole + 1)
    cat_auto_process.appendRow(remove_wavelet_denoise_item)

    remove_diurnal_sinusoid_item = QStandardItem("正弦相减去日变")
    remove_diurnal_sinusoid_item.setData("remove diurnal sinusoid", Qt.UserRole + 1)
    cat_auto_process.appendRow(remove_diurnal_sinusoid_item)

    robust_estimate_item = QStandardItem("Robust估计")
    robust_estimate_item.setData("robust estimate", Qt.UserRole + 1)
    cat_output.appendRow(robust_estimate_item)

    remote_reference_item = QStandardItem("远参考法")
    remote_reference_item.setData("remote reference", Qt.UserRole + 1)
    cat_output.appendRow(remote_reference_item)

    root = tree_model.invisibleRootItem()
    root.appendRow(cat_manual_process)
    root.appendRow(cat_auto_process)
    root.appendRow(cat_output)
    return tree_model
