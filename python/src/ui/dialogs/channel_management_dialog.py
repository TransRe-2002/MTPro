"""通道管理对话框（数据管理器就地预处理）。

以表格列出全部通道：行内直接改名、勾选删除，accept 时原子
应用（先删后改，改名按依赖顺序执行，检测到循环改名则拒绝）。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.em_data import EMData


class ChannelManagementDialog(QDialog):
    """批量重命名/删除通道。结构性变更。"""

    def __init__(self, em_data: EMData, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("通道管理 — MTPro")
        self.setMinimumSize(480, 360)

        self._em_data = em_data
        self._original_names = list(em_data.chid)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 16, 20, 16)

        info_label = QLabel(
            "双击通道名可改名（前缀决定场类型：E* 电场，B*/H* 磁场）；"
            "勾选“删除”后点确定一并应用。至少保留一个通道。",
            self,
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # ── 通道表格 ──────────────────────────────────────────
        n = len(self._original_names)
        self._table = QTableWidget(n, 2, self)
        self._table.setHorizontalHeaderLabels(["通道名", "删除"])
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self._table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.verticalHeader().setVisible(False)

        self._delete_checks: list[QCheckBox] = []
        for row, name in enumerate(self._original_names):
            self._table.setItem(row, 0, QTableWidgetItem(name))

            check = QCheckBox()
            check.stateChanged.connect(self._update_summary)
            self._delete_checks.append(check)
            container = QWidget(self._table)
            check_layout = QHBoxLayout(container)
            check_layout.addWidget(check)
            check_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            check_layout.setContentsMargins(0, 0, 0, 0)
            self._table.setCellWidget(row, 1, container)

        layout.addWidget(self._table, 1)

        self._summary_label = QLabel(self)
        layout.addWidget(self._summary_label)

        self._error_label = QLabel(self)
        self._error_label.setStyleSheet("QLabel { color: red; font-size: 11px; }")
        self._error_label.setWordWrap(True)
        self._error_label.hide()
        layout.addWidget(self._error_label)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self._update_summary()

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def apply_to(self, em_data: EMData) -> tuple[int, int]:
        """原子应用删除与改名。返回 (改名数, 删除数)。

        先删除后改名；改名按“目标名当前空闲”的顺序执行，
        无环时总能完成（校验阶段已拒绝循环改名）。
        """
        deleted = self._deleted_names()
        renames = dict(self._rename_pairs())

        for old in deleted:
            em_data.remove_channel(old)

        applied = 0
        while renames:
            progressed = False
            for old, new in list(renames.items()):
                if not em_data.has_channel(new):
                    em_data.rename_channel(old, new)
                    del renames[old]
                    applied += 1
                    progressed = True
            if not progressed:
                raise ValueError("通道改名存在循环依赖，无法应用")
        return applied, len(deleted)

    # ------------------------------------------------------------------
    # 行状态读取
    # ------------------------------------------------------------------

    def _row_states(self) -> list[tuple[str, str, bool]]:
        """返回 [(原通道名, 新通道名, 是否删除)]。"""
        states = []
        for row, old in enumerate(self._original_names):
            item = self._table.item(row, 0)
            new = item.text().strip() if item is not None else ""
            deleted = self._delete_checks[row].isChecked()
            states.append((old, new, deleted))
        return states

    def _deleted_names(self) -> list[str]:
        return [old for old, _, deleted in self._row_states() if deleted]

    def _rename_pairs(self) -> list[tuple[str, str]]:
        return [
            (old, new)
            for old, new, deleted in self._row_states()
            if not deleted and new != old
        ]

    # ------------------------------------------------------------------
    # 私有槽
    # ------------------------------------------------------------------

    def _update_summary(self, *_args) -> None:
        kept = sum(1 for check in self._delete_checks if not check.isChecked())
        self._summary_label.setText(
            f"共 {len(self._original_names)} 个通道，将保留 {kept} 个。"
        )

    def _on_accept(self) -> None:
        states = self._row_states()
        kept = [(old, new) for old, new, deleted in states if not deleted]

        if not kept:
            self._show_error("不能删除全部通道，至少保留一个。")
            return

        for old, new in kept:
            if not new:
                self._show_error(f"通道 {old!r} 的新名称不能为空。")
                return

        names = [new for _, new in kept]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            self._show_error(f"通道名重复：{'、'.join(sorted(duplicates))}")
            return

        if self._has_rename_cycle(dict(self._rename_pairs())):
            self._show_error("改名存在循环依赖（如 A→B 且 B→A），请分多次操作。")
            return

        self._error_label.hide()
        self.accept()

    @staticmethod
    def _has_rename_cycle(rename_map: dict[str, str]) -> bool:
        """检测 old→new 改名映射中的环（深度优先三色标记）。"""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {old: WHITE for old in rename_map}

        def visit(start: str) -> bool:
            node = start
            path = []
            while node in rename_map and color.get(node) == WHITE:
                color[node] = GRAY
                path.append(node)
                node = rename_map[node]
            in_cycle = color.get(node) == GRAY
            for n in path:
                color[n] = BLACK
            return in_cycle

        return any(
            color[old] == WHITE and visit(old)
            for old in list(rename_map)
        )

    def _show_error(self, message: str) -> None:
        self._error_label.setText(message)
        self._error_label.show()
