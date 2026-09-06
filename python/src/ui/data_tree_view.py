from __future__ import annotations

import logging
import json

import numpy as np

from PySide6.QtWidgets import (
    QApplication, QDialog, QTreeView, QVBoxLayout, QWidget, QSplitter,
    QMenu, QPlainTextEdit, QLabel, QFrame, QGroupBox, QStackedWidget,
    QMessageBox,
)
from PySide6.QtCore import QByteArray, QMimeData, Qt, QModelIndex, Signal
from PySide6.QtGui import QDrag, QFont
from typing import Callable, Dict, Optional, Tuple

from core.em_data import EMData
from ui.data_process_shared import (
    CLASSIC_TOOL_MIME,
    DATA_TREE_SELECTION_MIME,
    PIPELINE_TOOL_MIME,
    create_classic_tool_model,
)
from ui.data_tree_model import DataTreeModel
from ui.compare_widget import CompareWidget
from ui.dialogs.channel_management_dialog import ChannelManagementDialog
from ui.dialogs.decimate_dialog import DecimateDialog
from ui.dialogs.rename_channel_dialog import RenameChannelDialog
from ui.dialogs.station_metadata_dialog import StationMetadataDialog
from ui.dialogs.trim_dialog import TrimDialog
from ui.plot_panel import PlotPanel
from ui.energy_spectrum_widget import EnergySpectrumWidget
from utils.signal_decimate import decimated_length
from base.data_manager import DataManager

class DataSelectionTreeView(QTreeView):
    def __init__(self, owner: "DataTreeViewer", parent=None):
        super().__init__(parent)
        self._owner = owner
        self.setDragEnabled(True)

    def startDrag(self, supported_actions):
        payload = self._owner.current_drag_payload()
        if payload is None:
            return

        mime_data = QMimeData()
        mime_data.setData(
            DATA_TREE_SELECTION_MIME,
            QByteArray(json.dumps(payload, ensure_ascii=True).encode("utf-8")),
        )
        drag = QDrag(self)
        drag.setMimeData(mime_data)
        drag.exec(supported_actions, Qt.DropAction.CopyAction)


class ToolTreeView(QTreeView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setHeaderHidden(True)

    def startDrag(self, supported_actions):
        index = self.currentIndex()
        if not index.isValid():
            return

        tool_type = index.data(Qt.UserRole + 1)
        if not isinstance(tool_type, str):
            return

        mime_data = QMimeData()
        mime_data.setData(CLASSIC_TOOL_MIME, QByteArray(tool_type.encode("utf-8")))
        mime_data.setData(PIPELINE_TOOL_MIME, QByteArray(tool_type.encode("utf-8")))
        drag = QDrag(self)
        drag.setMimeData(mime_data)
        drag.exec(supported_actions, Qt.DropAction.CopyAction)


class DataTreeViewer(QWidget):
    tool_requested = Signal(str)

    def __init__(self, data_manager: DataManager, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.tree_view = None
        self.status_text = None
        self.summary_stack = None
        self.info_group = None
        self.tool_group = None
        self.tool_tree = None
        self.tool_model = None
        self.data_manager = data_manager
        self._plot_panels: Dict[tuple[int, str], PlotPanel] = {}
        self._spectrum_panels: Dict[tuple[int, str], EnergySpectrumWidget] = {}

        self.activate_id = 0
        self.init_ui()
        self.connect_signal()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # ── 数据浏览器组 ──────────────────────────────────────
        browser_group = QGroupBox("数据管理器 / Data Manager", self)
        browser_group.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        browser_layout = QVBoxLayout(browser_group)
        browser_layout.setContentsMargins(4, 8, 4, 4)
        browser_layout.setSpacing(2)

        # ── 台站摘要组 ────────────────────────────────────────
        self.summary_stack = QStackedWidget(self)

        self.info_group = QGroupBox("台站摘要 / Station Summary", self)
        self.info_group.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        info_layout = QVBoxLayout(self.info_group)
        info_layout.setContentsMargins(4, 8, 4, 4)
        info_layout.setSpacing(2)

        spliter = QSplitter(Qt.Orientation.Vertical)
        spliter.setChildrenCollapsible(False)
        self.tree_view = DataSelectionTreeView(self, browser_group)
        self.tree_view.setFont(QFont("Arial", 9))
        self.model = DataTreeModel(self.data_manager, self)
        self.tree_view.setModel(self.model)
        self.tree_view.setUniformRowHeights(True)
        self.tree_view.setHeaderHidden(False)
        self.tree_view.setAlternatingRowColors(True)
        browser_layout.addWidget(self.tree_view, 1)

        self.status_text = QPlainTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.status_text.setFont(QFont("Courier New", 9))
        self.status_text.setPlainText('请在数据管理器中选择一个台站数据集。')
        info_layout.addWidget(self.status_text, 1)

        self.tool_group = QGroupBox("数据处理工具 / Processing Tools", self)
        self.tool_group.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        tool_layout = QVBoxLayout(self.tool_group)
        tool_layout.setContentsMargins(4, 8, 4, 4)
        tool_layout.setSpacing(2)
        self.tool_tree = ToolTreeView(self.tool_group)
        self.tool_model = create_classic_tool_model()
        self.tool_tree.setModel(self.tool_model)
        self.tool_tree.expandAll()
        tool_layout.addWidget(self.tool_tree, 1)

        self.summary_stack.addWidget(self.info_group)
        self.summary_stack.addWidget(self.tool_group)

        spliter.addWidget(browser_group)
        spliter.addWidget(self.summary_stack)
        spliter.setSizes([430, 220])
        main_layout.addWidget(spliter)
        self.setLayout(main_layout)
        self.tree_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

    def connect_signal(self):
        self.tree_view.doubleClicked.connect(self.on_item_double_click)
        self.tree_view.customContextMenuRequested.connect(self.on_context_menu)
        self.tree_view.selectionModel().currentChanged.connect(self.on_current_changed)
        self.tool_tree.doubleClicked.connect(self._on_tool_activated)
        self.data_manager.active_changed.connect(self.on_active_changed)
        self.data_manager.data_removed.connect(self.on_data_removed)
        self.data_manager.data_changed.connect(self._on_data_changed)

    def _on_data_changed(self, key: int, structural: bool):
        """结构变更后关闭该数据集的独立曲线窗口（数据已过期）。"""
        if not structural:
            return
        for panel_key in [pk for pk in self._plot_panels if pk[0] == key]:
            plot_panel = self._plot_panels.pop(panel_key)
            plot_panel.close()
        for panel_key in [pk for pk in self._spectrum_panels if pk[0] == key]:
            spectrum_panel = self._spectrum_panels.pop(panel_key)
            spectrum_panel.close()

    def on_current_changed(self, current: QModelIndex, previous: QModelIndex):
        del previous
        if not current.isValid():
            self.data_manager.set_active(0)
            return

        if self.model.is_dataset_index(current):
            key = self.model.dataset_key(current)
        elif self.model.is_channel_index(current):
            key = self.model.dataset_key(current)
        else:
            key = None

        if key is not None:
            self.data_manager.set_active(key)

    def on_data_removed(self, key: int):
        for panel_key in [panel_key for panel_key in self._plot_panels if panel_key[0] == key]:
            plot_panel = self._plot_panels.pop(panel_key)
            plot_panel.close()
        for panel_key in [panel_key for panel_key in self._spectrum_panels if panel_key[0] == key]:
            spectrum_panel = self._spectrum_panels.pop(panel_key)
            spectrum_panel.close()

    def on_active_changed(self, key: int):
        self.activate_id = key
        if key == 0:
            self.status_text.setPlainText('请在数据管理器中选择一个台站数据集。')
            self.tree_view.clearSelection()
        else:
            em_data = self.data_manager.get(key)
            if em_data is None:
                return

            self.status_text.setPlainText(self._format_data_summary(em_data))
            current_index = self.tree_view.currentIndex()
            if not self.model.is_channel_index(current_index):
                index = self.model.index_for_key(key)
            else:
                index = current_index
            if index.isValid() and self.tree_view.currentIndex() != index:
                self.tree_view.setCurrentIndex(index)

    def current_dataset_key(self) -> Optional[int]:
        if self.tree_view is None or self.model is None:
            return None
        current = self.tree_view.currentIndex()
        if not current.isValid():
            return None
        return self.model.dataset_key(current)

    def current_dataset(self) -> Optional[EMData]:
        key = self.current_dataset_key()
        if key is None:
            return None
        return self.data_manager.get(key)

    def checked_channel_refs(self) -> list[tuple[int, str]]:
        if self.model is None:
            return []
        return self.model.checked_channel_refs()

    def clear_checked_channels(self, refs: list[tuple[int, str]]) -> None:
        if self.model is None:
            return
        for data_id, channel_name in refs:
            self.model.set_channel_checked(data_id, channel_name, False)

    def current_channel_ref(self) -> Optional[tuple[int, str]]:
        if self.tree_view is None or self.model is None:
            return None

        current = self.tree_view.currentIndex()
        if not self.model.is_channel_index(current):
            return None

        key = self.model.dataset_key(current)
        channel_name = self.model.channel_name(current)
        if key is None or channel_name is None:
            return None
        return key, channel_name

    def current_drag_payload(self) -> Optional[dict]:
        if self.tree_view is None or self.model is None:
            return None

        current = self.tree_view.currentIndex()
        if not current.isValid():
            return None

        if self.model.is_dataset_index(current):
            data_id = self.model.dataset_key(current)
            if data_id is None:
                return None
            return {
                "kind": "dataset",
                "data_id": data_id,
            }

        if not self.model.is_channel_index(current):
            return None

        data_id = self.model.dataset_key(current)
        channel_name = self.model.channel_name(current)
        if data_id is None or channel_name is None:
            return None

        checked_refs = self.checked_channel_refs()
        if (data_id, channel_name) in checked_refs:
            refs = checked_refs
        else:
            refs = [(data_id, channel_name)]

        return {
            "kind": "channels",
            "refs": [
                {"data_id": ref_data_id, "channel_name": ref_channel_name}
                for ref_data_id, ref_channel_name in refs
            ],
        }

    def show_station_summary(self):
        if self.summary_stack is not None and self.info_group is not None:
            self.summary_stack.setCurrentWidget(self.info_group)

    def show_processing_tools(self):
        if self.summary_stack is not None and self.tool_group is not None:
            self.summary_stack.setCurrentWidget(self.tool_group)

    def _on_tool_activated(self, index: QModelIndex):
        if not index.isValid() or self.tool_model is None:
            return

        item = self.tool_model.itemFromIndex(index)
        if item is None:
            return

        tool_type = item.data(Qt.UserRole + 1)
        if isinstance(tool_type, str):
            self.tool_requested.emit(tool_type)

    @staticmethod
    def _format_data_summary(em_data) -> str:
        if getattr(em_data, 'timezone', None):
            import pytz
            station_tz = pytz.timezone(em_data.timezone)
            start_display = em_data.start_time.tz_convert(station_tz).strftime('%Y-%m-%d %H:%M:%S %Z')
            end_display = em_data.end_time.tz_convert(station_tz).strftime('%Y-%m-%d %H:%M:%S %Z')
        else:
            start_display = em_data.start_time.strftime('%Y-%m-%d %H:%M:%S %Z')
            end_display = em_data.end_time.strftime('%Y-%m-%d %H:%M:%S %Z')

        return (
            f"数据名称：{em_data.name}\n"
            f"开始时间：{start_display}\n"
            f"结束时间：{end_display}\n"
            f"采样周期：{em_data.dt.total_seconds()}s\n"
            f"采集点数：{em_data.npts}\n"
            f"采集站纬度：{em_data.latitude}\n"
            f"采集站经度：{em_data.longitude}\n"
            f"电场单位：{em_data.e_units}\n"
            f"磁场单位：{em_data.m_units}\n"
            f"数据路径：{em_data.path}"
        )

    def on_item_double_click(self, index: QModelIndex):
        if not self.model.is_channel_index(index):
            return

        key = self.model.dataset_key(index)
        ch_name = self.model.channel_name(index)
        if key is None or ch_name is None:
            return
        self._open_channel_plot(key, ch_name)

    def _open_channel_plot(self, key: int, ch_name: str):
        plot_key = (key, ch_name)
        plot_panel = self._plot_panels.get(plot_key)
        if plot_panel is not None:
            plot_panel.show()
            plot_panel.activateWindow()
            plot_panel.raise_()
            self.logger.info("Reused plot panel for dataset #%s channel %s", key, ch_name)
            return

        em_data = self.data_manager.get(key)
        if em_data is None:
            return
        data_name = em_data.name
        title = f"{data_name} - {ch_name}"

        x_values = em_data.datetime_index
        y_values = em_data.get_channel_series(ch_name)
        plot_panel = PlotPanel(
            x_data=x_values,
            y_data=y_values,
            label=f"{data_name} - {ch_name}",
            parent=self,
        )
        plot_panel.setWindowTitle(title)

        if ch_name.startswith('E'):
            e_units = em_data.e_units
            if e_units is not None:
                plot_panel.set_label('Time', e_units)
            else:
                plot_panel.set_label('Time', None)

        if ch_name.startswith('H') or ch_name.startswith('B'):
            m_units = em_data.m_units
            if m_units is not None:
                plot_panel.set_label('Time', m_units)
            else:
                plot_panel.set_label('Time', None)

        plot_panel.destroyed.connect(
            lambda *_, panel_key=plot_key: self._plot_panels.pop(panel_key, None)
        )
        plot_panel.show()
        self._plot_panels[plot_key] = plot_panel
        self.logger.info("Opened plot panel for dataset #%s channel %s", key, ch_name)

    # ── 右键菜单 ────────────────────────────────────────────────

    def _build_dataset_context_menu(self) -> Tuple[QMenu, Dict[object, Callable[[int], None]]]:
        """构建数据集级右键菜单，返回 (菜单, action→处理器) 映射。"""
        menu = QMenu(self)
        actions: Dict[object, Callable[[int], None]] = {}

        for label, handler in (
            ("保存", self._action_save_dataset),
            ("另存为", self._action_save_dataset_as),
            ("关闭数据", self._action_close_dataset),
        ):
            actions[menu.addAction(label)] = handler

        menu.addSeparator()

        for label, handler in (
            ("编辑台站元数据…", self._open_station_metadata_dialog),
            ("降采样…", self._open_decimate_dialog),
            ("裁剪时间范围…", self._open_trim_dialog),
            ("通道管理…", self._open_channel_management_dialog),
        ):
            actions[menu.addAction(label)] = handler

        return menu, actions

    def _build_channel_context_menu(self) -> Tuple[QMenu, Dict[object, Callable[[int, str], None]]]:
        """构建通道级右键菜单，返回 (菜单, action→处理器) 映射。"""
        menu = QMenu(self)
        actions: Dict[object, Callable[[int, str], None]] = {}

        for label, handler in (
            ("显示曲线", self._open_channel_plot),
            ("查看能量谱", self._open_energy_spectrum),
            ("序列对比", self._action_compare_channel),
            ("重命名通道…", self._open_rename_channel_dialog),
            ("删除通道", self._confirm_delete_channel),
        ):
            actions[menu.addAction(label)] = handler

        return menu, actions

    def _open_energy_spectrum(self, key: int, ch_name: str) -> None:
        panel_key = (key, ch_name)
        existing = self._spectrum_panels.get(panel_key)
        if existing is not None:
            existing.show()
            existing.activateWindow()
            existing.raise_()
            return
        em_data = self.data_manager.get(key)
        if em_data is None or not em_data.has_channel(ch_name):
            return
        dt = em_data.dt.total_seconds()
        if dt <= 0:
            QMessageBox.warning(self, "无法计算能量谱", "采样间隔无效。")
            return
        units = em_data.e_units if ch_name.startswith("E") else em_data.m_units
        panel = EnergySpectrumWidget(
            em_data.get_channel_series(ch_name).to_numpy(dtype=np.float64, copy=True),
            1.0 / dt,
            f"#{key} {em_data.name} - {ch_name}",
            units or "",
            self,
        )
        panel.destroyed.connect(lambda *_, item=panel_key: self._spectrum_panels.pop(item, None))
        self._spectrum_panels[panel_key] = panel
        panel.show()

    def on_context_menu(self, pos):
        index = self.tree_view.indexAt(pos)

        if not index.isValid():
            return

        if self.model.is_dataset_index(index):
            data_id = self.model.dataset_key(index)
            if data_id is None:
                return
            menu, actions = self._build_dataset_context_menu()
            action = menu.exec(self.tree_view.viewport().mapToGlobal(pos))
            if action is None:
                return
            handler = actions.get(action)
            if handler is not None:
                handler(data_id)

        elif self.model.is_channel_index(index):
            key = self.model.dataset_key(index)
            ch_name = self.model.channel_name(index)
            if key is None or ch_name is None:
                return
            menu, actions = self._build_channel_context_menu()
            action = menu.exec(self.tree_view.viewport().mapToGlobal(pos))
            if action is None:
                return
            handler = actions.get(action)
            if handler is not None:
                handler(key, ch_name)

    # ── 数据集级动作 ───────────────────────────────────────────

    def _action_save_dataset(self, data_id: int) -> None:
        main_window = self.window()
        if hasattr(main_window, "save_file_for"):
            main_window.save_file_for(data_id)

    def _action_save_dataset_as(self, data_id: int) -> None:
        main_window = self.window()
        if hasattr(main_window, "save_file_as_for"):
            main_window.save_file_as_for(data_id)

    def _action_close_dataset(self, data_id: int) -> None:
        self.data_manager.remove(data_id)

    def _close_processor_windows_for(self, data_id: int) -> None:
        """结构变更前关闭该数据集的处理工具窗口（其缓存已过期）。"""
        process_widget = getattr(self.window(), "data_process_widget", None)
        if process_widget is not None:
            process_widget.close_windows_for_data(data_id)

    def _open_station_metadata_dialog(self, data_id: int) -> None:
        em_data = self.data_manager.get(data_id)
        if em_data is None:
            return

        dialog = StationMetadataDialog(em_data, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        dialog.apply_to(em_data)
        self.logger.info("Station metadata updated for dataset #%s", data_id)
        self.data_manager.notify_changed(data_id, structural=False)

    def _open_decimate_dialog(self, data_id: int) -> None:
        em_data = self.data_manager.get(data_id)
        if em_data is None:
            return

        dialog = DecimateDialog(em_data, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        factor = dialog.factor()
        max_missing_fraction = dialog.max_missing_fraction()

        new_npts = decimated_length(em_data.npts, factor)
        if max_missing_fraction == 1.0:
            missing_policy = (
                "TS 兼容规则：窗口至少含 1 个有效样点即保留；"
                "全缺失窗口输出 NaN。"
            )
        else:
            missing_policy = (
                f"每个窗口最多允许缺失 {max_missing_fraction * 100:g}%；"
                "超过时输出 NaN。"
            )
        reply = QMessageBox.question(
            self,
            "确认降采样",
            f"将数据集「{em_data.name}」降采样 {factor} 倍：\n"
            f"{em_data.npts} → {new_npts} 点，采样周期 "
            f"{em_data.dt.total_seconds():g}s → "
            f"{(em_data.dt * factor).total_seconds():g}s\n\n"
            f"{missing_policy}\n\n"
            "此操作就地修改且不可撤销，是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._close_processor_windows_for(data_id)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            em_data.decimate(
                factor,
                max_missing_fraction=max_missing_fraction,
            )
        finally:
            QApplication.restoreOverrideCursor()
        self.logger.info(
            "Dataset #%s decimated by %d with max missing %.1f%%: "
            "npts=%d dt=%ss",
            data_id, factor, max_missing_fraction * 100,
            em_data.npts, em_data.dt.total_seconds(),
        )
        self.data_manager.notify_changed(data_id, structural=True)

    def _open_trim_dialog(self, data_id: int) -> None:
        em_data = self.data_manager.get(data_id)
        if em_data is None:
            return

        dialog = TrimDialog(em_data, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        start, end = dialog.time_range()

        reply = QMessageBox.question(
            self,
            "确认裁剪",
            f"将数据集「{em_data.name}」裁剪到：\n"
            f"{start} ～ {end}\n\n"
            "范围外的数据将被删除，此操作不可撤销，是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._close_processor_windows_for(data_id)
        try:
            em_data.trim_time_range(start, end)
        except ValueError as exc:
            QMessageBox.warning(self, "裁剪失败", str(exc))
            return
        self.logger.info(
            "Dataset #%s trimmed: npts=%d range=%s..%s",
            data_id, em_data.npts, em_data.start_time, em_data.end_time,
        )
        self.data_manager.notify_changed(data_id, structural=True)

    def _open_channel_management_dialog(self, data_id: int) -> None:
        em_data = self.data_manager.get(data_id)
        if em_data is None:
            return

        dialog = ChannelManagementDialog(em_data, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        self._close_processor_windows_for(data_id)
        try:
            renamed, deleted = dialog.apply_to(em_data)
        except (KeyError, ValueError) as exc:
            QMessageBox.warning(self, "通道管理失败", str(exc))
            return
        self.logger.info(
            "Dataset #%s channels managed: %d renamed, %d deleted",
            data_id, renamed, deleted,
        )
        self.data_manager.notify_changed(data_id, structural=True)

    # ── 通道级动作 ─────────────────────────────────────────────

    def _action_compare_channel(self, key: int, ch_name: str) -> None:
        channels = self.get_check_state()
        if len(channels) == 0:
            em_data = self.data_manager.get(key)
            if em_data is None:
                return
            channels = [(em_data, ch_name)]
        self.open_compare_widget(channels)

    def _open_rename_channel_dialog(self, key: int, ch_name: str) -> None:
        em_data = self.data_manager.get(key)
        if em_data is None:
            return

        dialog = RenameChannelDialog(em_data, ch_name, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        new_name = dialog.new_name()
        if new_name == ch_name:
            return

        self._close_processor_windows_for(key)
        try:
            em_data.rename_channel(ch_name, new_name)
        except (KeyError, ValueError) as exc:
            QMessageBox.warning(self, "重命名失败", str(exc))
            return
        self.logger.info(
            "Dataset #%s channel renamed: %s → %s", key, ch_name, new_name,
        )
        self.data_manager.notify_changed(key, structural=True)

    def _confirm_delete_channel(self, key: int, ch_name: str) -> None:
        em_data = self.data_manager.get(key)
        if em_data is None:
            return
        if em_data.NCh <= 1:
            QMessageBox.warning(self, "删除通道", "不能删除最后一个通道。")
            return

        reply = QMessageBox.question(
            self,
            "删除通道",
            f"从数据集「{em_data.name}」中删除通道 {ch_name!r}？\n"
            "此操作就地修改且不可撤销。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._close_processor_windows_for(key)
        try:
            em_data.remove_channel(ch_name)
        except (KeyError, ValueError) as exc:
            QMessageBox.warning(self, "删除失败", str(exc))
            return
        self.logger.info("Dataset #%s channel deleted: %s", key, ch_name)
        self.data_manager.notify_changed(key, structural=True)

    def get_check_state(self) -> list[tuple[EMData, str]]:
        refs = self.model.checked_channel_refs()
        result: list[tuple[EMData, str]] = []
        for data_id, channel_name in refs:
            em_data = self.data_manager.get(data_id)
            if em_data is not None and em_data.has_channel(channel_name):
                result.append((em_data, channel_name))
        return result

    def open_compare_widget(self, channels: list[tuple[EMData, str]]):
        if len(channels) == 0:
            return
        compare_widget = CompareWidget(channels, parent=self)
        compare_widget.setWindowFlags(Qt.WindowType.Window)
        compare_widget.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        compare_widget.setWindowTitle(f"Series Compare Widget")
        compare_widget.show()
        compare_widget.raise_()
        self.logger.info("Opened compare widget for %s channels", len(channels))


if __name__ == '__main__':
    import sys
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    data_manager = DataManager()
    window = DataTreeViewer(data_manager)
    window.show()
    sys.exit(app.exec())
