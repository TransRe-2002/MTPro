from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QStandardItem, QStandardItemModel

DATA_TREE_SELECTION_MIME = "application/x-data-tree-selection"
PROCESSING_TOOL_MIME = "application/x-processing-tool"
PIPELINE_TOOL_MIME = "application/x-pipeline-tool"


def create_processing_tool_model() -> QStandardItemModel:
    tree_model = QStandardItemModel()
    tree_model.setHorizontalHeaderLabels(["数据处理工具"])

    cat_manual_process = QStandardItem("手动数据处理")
    cat_global_process = QStandardItem("全局自动处理")
    cat_auto_process = QStandardItem("局部自动处理")

    remove_region_item = QStandardItem("框选区域去噪")
    remove_region_item.setData("remove region", Qt.UserRole + 1)
    cat_manual_process.appendRow(remove_region_item)

    remove_spike_item = QStandardItem("框选尖峰去噪")
    remove_spike_item.setData("remove spike", Qt.UserRole + 1)
    cat_manual_process.appendRow(remove_spike_item)

    remove_step_diff_item = QStandardItem("差分去阶跃")
    remove_step_diff_item.setData("remove step diff", Qt.UserRole + 1)
    cat_manual_process.appendRow(remove_step_diff_item)

    auto_spike_filter_item = QStandardItem("自动跳点筛选")
    auto_spike_filter_item.setData("auto_spike_filter", Qt.UserRole + 1)
    cat_auto_process.appendRow(auto_spike_filter_item)

    emd_denoise_item = QStandardItem("EMD 经验模态分解去噪")
    emd_denoise_item.setData("emd_denoise", Qt.UserRole + 1)
    cat_auto_process.appendRow(emd_denoise_item)

    demean_item = QStandardItem("去均值")
    demean_item.setData("demean", Qt.UserRole + 1)
    cat_global_process.appendRow(demean_item)

    detrend_item = QStandardItem("线性去趋势")
    detrend_item.setData("detrend", Qt.UserRole + 1)
    cat_global_process.appendRow(detrend_item)

    kp_selection_item = QStandardItem("Kp 阈值选取")
    kp_selection_item.setData("kp_selection", Qt.UserRole + 1)
    cat_global_process.appendRow(kp_selection_item)

    median_filter_item = QStandardItem("滑动中值滤波")
    median_filter_item.setData("median_filter", Qt.UserRole + 1)
    cat_auto_process.appendRow(median_filter_item)

    morphological_filter_item = QStandardItem("形态滤波")
    morphological_filter_item.setData("morphological_filter", Qt.UserRole + 1)
    cat_auto_process.appendRow(morphological_filter_item)

    iir_filter_item = QStandardItem("零相位频率滤波")
    iir_filter_item.setData("iir_filter", Qt.UserRole + 1)
    cat_auto_process.appendRow(iir_filter_item)

    svd_denoise_item = QStandardItem("Hankel-SVD 去噪")
    svd_denoise_item.setData("svd_denoise", Qt.UserRole + 1)
    cat_auto_process.appendRow(svd_denoise_item)

    secular_variation_item = QStandardItem("去除缓慢背景变化（SV）")
    secular_variation_item.setData("remove_secular_variation", Qt.UserRole + 1)
    cat_auto_process.appendRow(secular_variation_item)

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

    vmd_denoise_item = QStandardItem("VMD 模态分解去噪")
    vmd_denoise_item.setData("vmd_denoise", Qt.UserRole + 1)
    cat_auto_process.appendRow(vmd_denoise_item)

    root = tree_model.invisibleRootItem()
    root.appendRow(cat_manual_process)
    root.appendRow(cat_global_process)
    root.appendRow(cat_auto_process)
    return tree_model
