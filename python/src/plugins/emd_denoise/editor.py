"""Parameter editor for EMD selected-IMF reconstruction."""
from __future__ import annotations

from typing import Mapping

from PySide6 import QtWidgets

from plugin_api import complete_params


class EMDDenoiseEditor(QtWidgets.QWidget):
    def __init__(self, host, params: Mapping[str, object], parent=None):
        super().__init__(parent)
        self._host = host
        values = complete_params(host.plugin.info, params)

        self.max_imfs_spin = QtWidgets.QSpinBox(self)
        self.max_imfs_spin.setRange(1, 32)
        self.max_imfs_spin.setValue(int(values["max_imfs"]))
        self.max_siftings_spin = QtWidgets.QSpinBox(self)
        self.max_siftings_spin.setRange(1, 10000)
        self.max_siftings_spin.setValue(int(values["max_siftings"]))
        self.tolerance_spin = QtWidgets.QDoubleSpinBox(self)
        self.tolerance_spin.setRange(1.0e-12, 1.0)
        self.tolerance_spin.setDecimals(12)
        self.tolerance_spin.setValue(float(values["sift_tolerance"]))
        self.min_segment_spin = QtWidgets.QSpinBox(self)
        self.min_segment_spin.setRange(4, 10_000_000)
        self.min_segment_spin.setValue(int(values["min_segment_points"]))
        self.keep_imfs_edit = QtWidgets.QLineEdit(str(values["keep_imfs"]), self)
        self.keep_residue_check = QtWidgets.QCheckBox(self)
        self.keep_residue_check.setChecked(bool(values["keep_residue"]))

        note = QtWidgets.QLabel(
            "IMF 0 是最先提取的高频分量。保留 IMF 可填 all 或 1,2,3；"
            "如按 LMT 的高频 IMF 抑制思路，可保留 1,2,... 并保留低频残差。"
            "含 NaN 的数据按连续有效段分别处理。",
            self,
        )
        note.setWordWrap(True)
        form = QtWidgets.QFormLayout()
        form.addRow("最大 IMF 数:", self.max_imfs_spin)
        form.addRow("单 IMF 最大筛分次数:", self.max_siftings_spin)
        form.addRow("筛分相对变化阈值:", self.tolerance_spin)
        form.addRow("最短有效段点数:", self.min_segment_spin)
        form.addRow("保留 IMF:", self.keep_imfs_edit)
        form.addRow("保留低频残差:", self.keep_residue_check)

        self.apply_button = QtWidgets.QPushButton("运行 EMD 并应用重构", self)
        self.apply_button.clicked.connect(self._apply)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(note)
        layout.addLayout(form)
        layout.addWidget(self.apply_button)

    def _apply(self) -> None:
        self._host.apply_params(
            {
                "max_imfs": int(self.max_imfs_spin.value()),
                "max_siftings": int(self.max_siftings_spin.value()),
                "sift_tolerance": float(self.tolerance_spin.value()),
                "min_segment_points": int(self.min_segment_spin.value()),
                "keep_imfs": self.keep_imfs_edit.text().strip() or "all",
                "keep_residue": bool(self.keep_residue_check.isChecked()),
            }
        )
