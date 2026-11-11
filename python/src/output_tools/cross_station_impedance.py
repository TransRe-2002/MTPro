from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QMessageBox, QWidget

from emtf.time_alignment import matching_remote_slice
from output_tools.robust_estimate import (
    RobustEstimateWidget, _detect_channels, _is_e_channel, _build_lod_task,
    _estimate_single_task, build_spectrum_kwargs, run_in_pool, pts_to_qdt, logger,
)


class CrossStationImpedanceWidget(RobustEstimateWidget):
    """Impedance estimation with independent electric and magnetic stations."""

    def __init__(self, data_id=0, parent=None):
        self._magnetic_pkg = {}
        super().__init__(data_id, parent)
        panel = self.findChild(QWidget, 'robustControlWidget').layout()
        station_row = panel.itemAt(0).layout()
        station_row.itemAt(0).widget().setText('电场台站:')
        row = QHBoxLayout()
        row.addWidget(QLabel('磁场台站:'))
        self.magnetic_station_combo = QComboBox()
        self.magnetic_station_combo.currentIndexChanged.connect(self._on_magnetic_station_changed)
        row.addWidget(self.magnetic_station_combo, 1)
        panel.insertLayout(1, row)

    def _on_estimation_done(self, completed):
        super()._on_estimation_done(completed)
        self._mark_parameters_changed()

    def _on_data_ready(self, data_id: int, channel: str, pkg: dict) -> None:
        del channel
        if self._pkg and data_id in (self.station_combo.currentData(), self.magnetic_station_combo.currentData()):
            self._cancel_estimation()
            if self._calculation_context:
                self.save_preview_btn.setEnabled(False)
                self.status_label.setText("输入数据已变化，请重新计算。")
        electric_id = self.station_combo.currentData()
        magnetic_id = self.magnetic_station_combo.currentData()
        if data_id != self._data_id and data_id not in (electric_id, magnetic_id):
            return
        if not self._pkg and data_id == magnetic_id and data_id != electric_id:
            self._magnetic_pkg = pkg
            return
        if not self._pkg:
            self._pkg = pkg
            self._station_metadata = pkg['station_metadata']
            self._populate_stations()
            electric_id = self.station_combo.currentData()
            magnetic_id = self.magnetic_station_combo.currentData()
            if electric_id != data_id and electric_id is not None:
                self.data_requested.emit(electric_id, "")
            if magnetic_id is not None and magnetic_id != data_id:
                self.data_requested.emit(magnetic_id, "")
        if data_id == electric_id:
            self._data_id = data_id
            self._pkg = pkg
            self._station_metadata = pkg['station_metadata']
        if data_id == magnetic_id:
            self._magnetic_pkg = pkg
        if data_id in (electric_id, magnetic_id):
            self._apply_station_data()


    def on_source_restored(self, data_id: int) -> None:
        if data_id not in (self.station_combo.currentData(), self.magnetic_station_combo.currentData()):
            return
        self._cancel_estimation()
        self.status_label.setText("输入数据已从源文件还原；已有阻抗结果基于还原前的数据，请重新计算。")


    def _populate_stations(self):
        prev_key = self.station_combo.currentData()
        target_key = prev_key if prev_key is not None else self._data_id
        magnetic_key = self.magnetic_station_combo.currentData()
        self.station_combo.blockSignals(True)
        self.station_combo.clear()
        for s in self._pkg.get('stations', []):
            if s['has_electric']:
                self.station_combo.addItem(f"#{s['id']} {s['name']}", s['id'])
        if target_key is not None and self.station_combo.findData(target_key) >= 0:
            self.station_combo.setCurrentIndex(self.station_combo.findData(target_key))
        elif self.station_combo.count() > 0:
            self.station_combo.setCurrentIndex(0)
        self.station_combo.blockSignals(False)
        self.magnetic_station_combo.blockSignals(True)
        self.magnetic_station_combo.clear()
        for station in self._pkg.get('stations', []):
            if station['has_magnetic']:
                self.magnetic_station_combo.addItem(f"#{station['id']} {station['name']}", station['id'])
        preferred = magnetic_key if magnetic_key is not None else self.station_combo.currentData()
        index = self.magnetic_station_combo.findData(preferred)
        if index >= 0:
            self.magnetic_station_combo.setCurrentIndex(index)
        self.magnetic_station_combo.blockSignals(False)


    def _on_station_changed(self, _idx: int):
        """台站切换时请求新数据。"""
        new_id = self.station_combo.currentData()
        if new_id is not None:
            self._data_id = new_id
            self._cancel_estimation()
            self._mark_parameters_changed()
            self.data_requested.emit(new_id, "")


    def _on_magnetic_station_changed(self, _idx: int):
        self._magnetic_pkg = {}
        self._cancel_estimation()
        self._mark_parameters_changed()
        new_id = self.magnetic_station_combo.currentData()
        if new_id is not None:
            self.data_requested.emit(new_id, "")


    def _apply_station_data(self):
        """根据 _pkg 更新通道映射、时间范围和波形。"""
        pkg = self._pkg
        channels: list[str] = pkg.get('channel_names', [])
        magnetic_channels = self._magnetic_pkg.get('channel_names', [])
        mapping = _detect_channels(channels)
        magnetic_mapping = _detect_channels(magnetic_channels)

        for combo, key in [
            (self.ch_combo_ex, "Ex"), (self.ch_combo_ey, "Ey"),
            (self.ch_combo_hx, "Hx"), (self.ch_combo_hy, "Hy"),
        ]:
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("(请选择)", None)
            source_channels = channels if key in ("Ex", "Ey") else magnetic_channels
            for ch in source_channels:
                if key in ("Ex", "Ey") and not _is_e_channel(ch):
                    continue
                if key in ("Hx", "Hy") and not ch.lower().startswith(("h", "b")):
                    continue
                combo.addItem(ch, ch)
            auto_ch = (mapping if key in ("Ex", "Ey") else magnetic_mapping).get(key)
            if auto_ch is not None:
                idx = combo.findText(auto_ch)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            combo.blockSignals(False)

        self.ch_combo_hz.blockSignals(True)
        self.ch_combo_hz.clear()
        self.ch_combo_hz.addItem("(不使用)", None)
        for ch in magnetic_channels:
            if ch.lower().startswith(("h", "b")):
                self.ch_combo_hz.addItem(ch, ch)
        self.ch_combo_hz.setCurrentIndex(0)
        self.ch_combo_hz.blockSignals(False)

        dt = pkg.get('dt', 1.0)
        n = pkg.get('npts', 0)
        start_ts = pkg.get('start_time', 0)
        end_ts = start_ts + dt * n if n > 0 else start_ts

        self.sample_rate_label.setText(f"{1.0 / dt:.4f}" if dt > 0 else "N/A")
        self.start_spin.blockSignals(True)
        self.start_spin.setRange(0, n - 1)
        self.start_spin.setValue(0)
        self.start_spin.blockSignals(False)
        self.end_spin.blockSignals(True)
        self.end_spin.setRange(1, n)
        self.end_spin.setValue(n)
        self.end_spin.blockSignals(False)

        start_dt_val = pts_to_qdt(pd.Timestamp(start_ts, unit='s', tz='Asia/Shanghai'))
        end_dt_val = pts_to_qdt(pd.Timestamp(end_ts, unit='s', tz='Asia/Shanghai'))
        self.start_dt.blockSignals(True)
        self.start_dt.setDateTime(start_dt_val)
        self.start_dt.setMinimumDateTime(start_dt_val)
        self.start_dt.setMaximumDateTime(end_dt_val)
        self.start_dt.blockSignals(False)
        self.end_dt.blockSignals(True)
        self.end_dt.setDateTime(end_dt_val)
        self.end_dt.setMinimumDateTime(start_dt_val)
        self.end_dt.setMaximumDateTime(end_dt_val)
        self.end_dt.blockSignals(False)

        magnetic_index = self._magnetic_pkg.get('datetime_index')
        electric_index = pkg.get('datetime_index')
        if magnetic_index is not None and electric_index is not None and len(magnetic_index) and len(electric_index):
            electric_index = pd.DatetimeIndex(electric_index)
            magnetic_index = pd.DatetimeIndex(magnetic_index)
            if electric_index.is_monotonic_increasing and magnetic_index.is_monotonic_increasing:
                start = int(electric_index.searchsorted(magnetic_index[0], side='left'))
                end = int(electric_index.searchsorted(magnetic_index[-1], side='right'))
                if start < end:
                    self.start_spin.setValue(start)
                    self.end_spin.setValue(end)

        self._refresh_plot()


    def _refresh_plot(self):
        """刷新波形预览：清除旧曲线，按当前通道选择重新绑图。

        每通道构建 LODPyramid（Rust 加速），曲线初始填入概览级数据；
        后续 pan/zoom 时由 sigXRangeChanged → _lod_update_curves 动态查询。
        """
        pkg = self._pkg
        if not pkg:
            return
        all_y = pkg.get('all_y', {})
        x = pkg.get('x', np.array([]))
        if len(x) == 0:
            return

        # 清除旧曲线、金字塔和图例
        for curve in list(self._curves.values()):
            curve_view = curve.getViewBox()
            if curve_view is self._b_viewbox:
                self._b_viewbox.removeItem(curve)
            else:
                self.plot_item.vb.removeItem(curve)
        self._curves.clear()
        self._lod_pyramids.clear()
        self._legend.clear()

        # 构建时间戳数组（同时缓存，供拖拽区域时二分查找使用）
        datetime_index = pd.DatetimeIndex(pkg['datetime_index'])
        t = datetime_index.asi8.astype(np.float64) / 1_000_000_000.0
        self._ts_array = t

        channels = self._selected_channels()
        pw = max(self.plot_widget.width(), 800)

        entries = []
        for tag, ch_name in channels.items():
            if ch_name is None:
                continue
            source_pkg = pkg if tag in ("Ex", "Ey") else self._magnetic_pkg
            y = source_pkg.get('all_y', {}).get(ch_name)
            if y is None:
                continue
            source_times = pd.DatetimeIndex(source_pkg['datetime_index']).asi8.astype(np.float64) / 1_000_000_000.0
            entries.append((tag, ch_name, tag in ("Ex", "Ey"), source_times, y))

        self._plot_generation += 1
        run_in_pool(
            "compute", _build_lod_task, entries, pw, self._plot_generation,
            on_done=self._on_lod_build_done,
            on_error=self._on_lod_build_failed,
            owner=self,
        )

        # Keep the shaded range consistent with the selected sample indices.
        if len(t) > 1:
            self.region_item.blockSignals(True)
            self.region_item.setRegion((t[min(self.start_spin.value(), len(t) - 1)],
                                        t[min(max(0, self.end_spin.value() - 1), len(t) - 1)]))
            self.region_item.blockSignals(False)

        self._update_range_info()
        self._update_warning()


    def _run_estimation(self):
        """运行 robust 阻抗估计。

        流程：从选中范围提取各通道数据 → 预处理（去趋势）→ 调用 emtf_rs 估计
        → 展示结果表格和对比图。

        注意：这里只做去趋势（detrend），不做插值。emtf_rs 内部有自己的插值逻辑，
        此处重复插值反而会掩盖数据质量问题。
        """
        if self._estimation_future is not None:
            return
        if self._bands is not None and not self._saved_current:
            reply = QMessageBox.question(
                self, "替换未保存结果", "当前结果尚未保存到阻抗预览，重新计算会替换它。继续吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        pkg = self._pkg
        if not pkg:
            QMessageBox.warning(self, "错误", "请先选择一个台站。")
            return
        backend = self._current_backend()
        all_y = pkg.get('all_y', {})
        magnetic_pkg = self._magnetic_pkg
        if not magnetic_pkg:
            QMessageBox.warning(self, "错误", "请选择有磁场数据的台站。")
            return

        ch_ex = self.ch_combo_ex.currentData()
        ch_ey = self.ch_combo_ey.currentData()
        ch_hx = self.ch_combo_hx.currentData()
        ch_hy = self.ch_combo_hy.currentData()
        ch_hz = self.ch_combo_hz.currentData()

        for label, ch in [("Ex", ch_ex), ("Ey", ch_ey), ("Hx", ch_hx), ("Hy", ch_hy)]:
            if ch is None:
                QMessageBox.warning(self, "错误", f"请选择 {label} 通道。")
                return
        if ch_ex == ch_ey or ch_hx == ch_hy or (ch_hz is not None and ch_hz in (ch_hx, ch_hy)):
            QMessageBox.warning(self, "错误", "电场和磁场的各分量必须分别选择不同通道。")
            return

        try:
            start = self.start_spin.value()
            end = self.end_spin.value()
            if end <= start:
                QMessageBox.warning(self, "错误", "结束索引必须大于起始索引。")
                return

            ex_arr = all_y.get(ch_ex, np.array([]))[start:end]
            ey_arr = all_y.get(ch_ey, np.array([]))[start:end]
            if not np.isclose(float(pkg['dt']), float(magnetic_pkg['dt']), rtol=1e-9, atol=0.0):
                raise ValueError("电场台站与磁场台站的采样间隔不一致，无法计算。")
            try:
                magnetic_start, magnetic_end = matching_remote_slice(pkg['datetime_index'], magnetic_pkg['datetime_index'], start, end)
            except ValueError as exc:
                raise ValueError("电场与磁场数据未逐点时间对齐，请选择两个台站共有且采样时刻一致的时间范围。") from exc
            magnetic_y = magnetic_pkg.get('all_y', {})
            hx_arr = magnetic_y.get(ch_hx, np.array([]))[magnetic_start:magnetic_end]
            hy_arr = magnetic_y.get(ch_hy, np.array([]))[magnetic_start:magnetic_end]
            hz_arr = None
            if ch_hz is not None:
                hz_arr = magnetic_y.get(ch_hz, np.array([]))[magnetic_start:magnetic_end]
            if any(len(arr) != end - start for arr in (ex_arr, ey_arr, hx_arr, hy_arr)) or (hz_arr is not None and len(hz_arr) != end - start):
                raise ValueError("所选通道的数据长度与时间索引不一致。")

            sample_rate = 1.0 / pkg['dt']
            window_size = self.window_combo.currentData()
            n_samples = end - start
            if window_size > n_samples:
                QMessageBox.warning(
                    self, "参数错误",
                    f"窗口大小 ({window_size}) 超过选中数据点数 ({n_samples})，"
                    f"请减小窗口大小或扩大数据范围。"
                )
                return

            self.status_label.setText("正在计算...")
            self.run_button.setEnabled(False)
            self.cancel_button.setEnabled(True)

            station_name = self._station_metadata.name if self._station_metadata else "STATION"
            kwargs = build_spectrum_kwargs(
                sample_rate=sample_rate,
                window_size=window_size,
                step_pct=self.step_pct_spin.value(),
                huber_threshold=self.huber_spin.value(),
                max_iterations=self.max_iter_spin.value(),
                tolerance=self.tol_spin.value(),
                hz_arr=hz_arr,
                backend=backend,
                fft_window_size=self.fft_window_combo.currentData(),
                n_bands=self.bands_spin.value(),
                station_name=station_name,
            )
            idx = pkg.get("datetime_index")
            start_time = idx[start] if idx is not None and start < len(idx) else ""
            end_time = idx[end - 1] if idx is not None and end > 0 and end <= len(idx) else ""
            context = {
                "method": "cross-station impedance",
                "station_name": station_name,
                "latitude": getattr(self._station_metadata, "latitude", 0.0) or 0.0,
                "longitude": getattr(self._station_metadata, "longitude", 0.0) or 0.0,
                "elevation": getattr(self._station_metadata, "elevation", 0.0) or 0.0,
                "range_start": start_time,
                "range_end": end_time,
                "channels": {"Ex": ch_ex, "Ey": ch_ey, "Hx": ch_hx, "Hy": ch_hy, "Hz": ch_hz},
                "parameters": {
                    "start": start, "end": end, "backend": backend, "window_size": window_size,
                    "electric_station_id": self.station_combo.currentData(),
                    "magnetic_station_id": self.magnetic_station_combo.currentData(),
                    "magnetic_station_name": magnetic_pkg.get('name', ''),
                    "fft_window_size": self.fft_window_combo.currentData(),
                    "processing_scheme": "emtf_fc_bands_v1",
                    "ar_filter_length": 3,
                    "solver_settings": {"huber_threshold": 1.5, "max_iterations": 50, "tolerance": 0.002},
                    "step_pct": self.step_pct_spin.value(), "bands": self.bands_spin.value(),
                    "huber_threshold": self.huber_spin.value(),
                    "max_iterations": self.max_iter_spin.value(), "tolerance": self.tol_spin.value(),
                },
            }
            self._pending_context = context
            self._task_generation += 1
            generation = self._task_generation
            self._estimation_future = run_in_pool(
                "compute", _estimate_single_task, backend,
                (ex_arr, ey_arr, hx_arr, hy_arr, hz_arr),
                (ch_ex, ch_ey, ch_hx, ch_hy, ch_hz), kwargs, generation,
                on_done=self._on_estimation_done,
                on_error=self._on_estimation_error,
                owner=self,
            )

        except Exception as e:
            logger.exception("Robust estimation failed")
            QMessageBox.critical(self, "估计失败", str(e))
            self.status_label.setText("失败")
            self.run_button.setEnabled(True)
            self.cancel_button.setEnabled(False)


    def _current_calculation_signature(self) -> tuple:
        return (
            self.start_spin.value(), self.end_spin.value(), self._current_backend(),
            self.window_combo.currentData(), self.fft_window_combo.currentData(),
            self.step_pct_spin.value(), self.huber_spin.value(), self.max_iter_spin.value(),
            self.tol_spin.value(), self.bands_spin.value(), self.ch_combo_ex.currentData(),
            self.ch_combo_ey.currentData(), self.ch_combo_hx.currentData(),
            self.ch_combo_hy.currentData(), self.ch_combo_hz.currentData(),
            self.station_combo.currentData(), self.magnetic_station_combo.currentData(),
        )


    def _saved_calculation_signature(self) -> tuple:
        params = self._calculation_context.get("parameters", {})
        channels = self._calculation_context.get("channels", {})
        return (
            params.get("start"), params.get("end"), params.get("backend"),
            params.get("window_size"), params.get("fft_window_size"), params.get("step_pct"),
            params.get("huber_threshold"), params.get("max_iterations"), params.get("tolerance"),
            params.get("bands"), channels.get("Ex"), channels.get("Ey"), channels.get("Hx"),
            channels.get("Hy"), channels.get("Hz"),
            params.get("electric_station_id"), params.get("magnetic_station_id"),
        )
