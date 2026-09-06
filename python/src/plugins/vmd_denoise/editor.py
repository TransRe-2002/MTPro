"""Parameter editor for VMD selected-mode reconstruction."""
from __future__ import annotations

from typing import Mapping

from PySide6 import QtWidgets

from plugin_api import complete_params


class VMDDenoiseEditor(QtWidgets.QWidget):
    """Keep VMD configuration explicit instead of silently discarding modes."""

    def __init__(self, host, params: Mapping[str, object], parent=None):
        super().__init__(parent)
        self._host = host
        info = host.plugin.info
        values = complete_params(info, params)

        self.mode_count_spin = QtWidgets.QSpinBox(self)
        self.mode_count_spin.setRange(1, 32)
        self.mode_count_spin.setValue(int(values["mode_count"]))
        self.alpha_spin = QtWidgets.QDoubleSpinBox(self)
        self.alpha_spin.setRange(0.001, 1.0e9)
        self.alpha_spin.setDecimals(4)
        self.alpha_spin.setValue(float(values["alpha"]))
        self.tau_spin = QtWidgets.QDoubleSpinBox(self)
        self.tau_spin.setRange(0.0, 1.0e6)
        self.tau_spin.setDecimals(6)
        self.tau_spin.setValue(float(values["tau"]))
        self.dc_check = QtWidgets.QCheckBox(self)
        self.dc_check.setChecked(bool(values["dc"]))
        self.init_combo = QtWidgets.QComboBox(self)
        self.init_combo.addItems(["uniform", "zero"])
        self.init_combo.setCurrentText(str(values["init"]))
        self.tolerance_spin = QtWidgets.QDoubleSpinBox(self)
        self.tolerance_spin.setRange(1.0e-12, 1.0)
        self.tolerance_spin.setDecimals(12)
        self.tolerance_spin.setValue(float(values["tolerance"]))
        self.max_iter_spin = QtWidgets.QSpinBox(self)
        self.max_iter_spin.setRange(1, 10000)
        self.max_iter_spin.setValue(int(values["max_iter"]))
        self.min_segment_spin = QtWidgets.QSpinBox(self)
        self.min_segment_spin.setRange(4, 10_000_000)
        self.min_segment_spin.setValue(int(values["min_segment_points"]))
        self.keep_modes_edit = QtWidgets.QLineEdit(str(values["keep_modes"]), self)

        note = QtWidgets.QLabel(
            "模态按中心频率由低到高编号。保留模态可填 all 或 0,1,3；"
            "默认 all 不会自动丢弃任何模态。VMD 的约束重构仍可能存在小残差；"
            "含 NaN 的数据按连续有效段分别处理。",
            self,
        )
        note.setWordWrap(True)
        form = QtWidgets.QFormLayout()
        form.addRow("模态数 K:", self.mode_count_spin)
        form.addRow("带宽约束 α:", self.alpha_spin)
        form.addRow("拉格朗日步长 τ:", self.tau_spin)
        form.addRow("首模态固定直流:", self.dc_check)
        form.addRow("中心频率初始化:", self.init_combo)
        form.addRow("收敛阈值:", self.tolerance_spin)
        form.addRow("最大迭代次数:", self.max_iter_spin)
        form.addRow("最短有效段点数:", self.min_segment_spin)
        form.addRow("保留模态:", self.keep_modes_edit)

        self.apply_button = QtWidgets.QPushButton("运行 VMD 并应用重构", self)
        self.apply_button.clicked.connect(self._apply)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(note)
        layout.addLayout(form)
        layout.addWidget(self.apply_button)

    def _params(self) -> dict[str, object]:
        return {
            "mode_count": int(self.mode_count_spin.value()),
            "alpha": float(self.alpha_spin.value()),
            "tau": float(self.tau_spin.value()),
            "dc": bool(self.dc_check.isChecked()),
            "init": self.init_combo.currentText(),
            "tolerance": float(self.tolerance_spin.value()),
            "max_iter": int(self.max_iter_spin.value()),
            "min_segment_points": int(self.min_segment_spin.value()),
            "keep_modes": self.keep_modes_edit.text().strip() or "all",
        }

    def _apply(self) -> None:
        self._host.apply_params(self._params())
