from __future__ import annotations

"""处理器窗口基类，提供统一的两段式数据初始化、撤销/恢复/确认流程。

子类只需实现 _init_ui() + _init_plot() + _log() 即可获得完整的 undo/revert/confirm 支持。
工具不持有数据管理器引用；Pipeline 编辑器通过切片注入数据。
"""

from typing import Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QProgressDialog, QSizePolicy, QWidget

from utils.series import dti_to_numpy


def build_pkg_from_series(channel_name: str, series: pd.Series) -> dict:
    """把一段 ``pd.Series`` 切片打包成 processor 期望的数据字典（pkg）。

    Pipeline 模式下 block 只携带 ``[start:end]`` 的切片，这里把切片伪装成
    单通道 pkg，让既有 ``_on_data_ready`` 无需改动。
    """
    y = series.to_numpy(dtype=np.float64)
    index = series.index
    n = len(y)
    return {
        'x': dti_to_numpy(index),
        'y': y,
        'channel_names': [channel_name],
        'all_y': {channel_name: y.copy()},
        'npts': n,
        'start_time': float(index[0].timestamp()) if n else 0.0,
        'dt': float((index[1] - index[0]).total_seconds()) if n > 1 else 1.0,
        'datetime_index': index,
        'kp_data': None,
    }


class BaseProcessorWidget(QWidget):
    """处理器工具窗口基类。

    统一管理：
    - 两段式数据初始化（__init__ 搭 UI → _on_data_ready 建图）
    - 撤销 / 恢复原始数据 / 确认并关闭
    - 通过信号与外部通信（不直接依赖数据管理器）

    子类需要：
    - 实现 _init_ui() 搭 UI 壳（不依赖数据）
    - 重写 _on_data_loaded() 做数据依赖的初始化（建图等）
    - 实现 _log(msg) 记录日志
    - 可选重写 _update_derived_data() / _reset_result_state()
    """

    # 供编辑器宿主使用的结果信号。
    data_requested = Signal(int, str)              # 请求数据: data_id, channel
    result_submitted = Signal(int, str, object)     # 提交结果: data_id, channel, ndarray
    # pipeline 编辑器用的二参结果信号: (channel, ndarray)，不携带 data_id
    result_signal = Signal(str, object)

    def __init__(
        self,
        source,
        channel_name: str,
        parent: Optional[QWidget] = None,
    ):
        """``source`` 可以是：

        - ``int`` data_id —— 延迟注入路径，构造时先不加载数据；
        - ``EMData`` —— 直接取 ``channel_name`` 对应通道；
        - ``pd.Series`` —— pipeline block 的 ``[start:end]`` 切片。

        后两种会在子类 ``__init__`` 结尾调用 ``_load_input_data()`` 时同步加载。
        """
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.data_id = source if isinstance(source, int) else 0
        self._source = source
        self.ch_name = channel_name

        # 数据字段 — 由 _on_data_ready 填充
        self.x_data: Optional[np.ndarray] = None
        self.y_data: Optional[np.ndarray] = None
        self.origin_y_data: Optional[np.ndarray] = None
        self.nan_index: list[int] = []
        self.history: list[np.ndarray] = []
        self._pkg: dict = {}

    # ─── 外部接线（data_process 调）─────────────────────

    def wire_data_ready(self, data_ready_signal) -> None:
        """连接外部数据信号到 _on_data_ready。由 data_process 调。"""
        data_ready_signal.connect(self._on_data_ready)

    # ─── 同步切片加载（pipeline 编辑器 / 测试入口）──────

    def _resolve_series(self) -> Optional[pd.Series]:
        source = self._source
        if isinstance(source, pd.Series):
            return source
        if hasattr(source, "has_channel") and hasattr(source, "get_channel_series"):
            if source.has_channel(self.ch_name):
                return source.get_channel_series(self.ch_name)
        return None

    def _load_input_data(self) -> None:
        """子类 ``__init__`` 结尾调用：source 能提供数据就同步加载。

        source 为 int 时不做任何事，等待后续切片注入。
        """
        series = self._resolve_series()
        if series is None:
            return
        self.load_series(self.ch_name, series)

    def load_series(self, channel_name: str, series: pd.Series) -> None:
        """以 ``pd.Series`` 切片直接注入数据（pipeline block 使用）。

        这里同步构造 pkg 并走 ``_on_data_ready``，
        适合切片数据已经在手、无需跨线程请求的场景。
        """
        self.ch_name = channel_name
        pkg = build_pkg_from_series(channel_name, series)
        self._on_data_ready(self.data_id, channel_name, pkg)

    # ─── 两段式初始化 ────────────────────────────────────

    def _on_data_ready(self, data_id: int, channel: str, pkg: dict) -> None:
        if data_id != self.data_id or channel != self.ch_name:
            return
        self._pkg = pkg
        self.x_data = pkg['x']
        raw_y = pkg['y']
        self.nan_index = np.flatnonzero(pd.isna(raw_y)).tolist()
        interpolated = pd.Series(raw_y).interpolate(method="linear")
        self.y_data = interpolated.to_numpy(dtype=np.float64)
        self.origin_y_data = self.y_data.copy()

        self._on_data_loaded()

    def _on_data_loaded(self) -> None:
        """子类重写：数据就绪后的初始化（建图等）。"""
        pass

    # ─── 子类必须实现 ────────────────────────────────────

    def _init_ui(self) -> None:
        raise NotImplementedError

    def _log(self, msg: str) -> None:
        raise NotImplementedError

    # ─── 子类可选重写 ────────────────────────────────────

    def _update_derived_data(self) -> None:
        pass

    def _reset_result_state(self) -> None:
        pass

    # ─── 统一的撤销 / 还原 / 确认流程 ──────────────────

    def _save_history(self) -> None:
        self.history.append(self.y_data.copy())

    def _undo(self) -> None:
        if not self.history:
            self._log("没有可撤销的操作")
            return
        self.y_data = self.history.pop()
        self._reset_result_state()
        self._update_derived_data()
        self._update_plot()
        self._log("已撤销上一步操作")

    def _revert(self) -> None:
        self._save_history()
        self.y_data = self.origin_y_data.copy()
        self._reset_result_state()
        self._update_derived_data()
        self._update_plot()
        self._log("已还原到原始数据")

    def _restore_nan(self, result: np.ndarray) -> np.ndarray:
        result = result.copy()
        for idx in self.nan_index:
            if 0 <= idx < len(result):
                result[idx] = np.nan
        return result

    def _confirm_and_close(self) -> None:
        result = self._restore_nan(self.y_data)
        self._log(f"确认修改，返回 {len(result)} 个数据点")
        self.result_submitted.emit(self.data_id, self.ch_name, result)
        self.result_signal.emit(self.ch_name, result)

    def _update_plot(self) -> None:
        raise NotImplementedError

    # ─── 后台重计算（compute 池 + 进度条）─────────────────

    def _run_heavy(self, compute_fn, apply_fn, description: str) -> None:
        """把重计算挪到共享 compute 池，主线程弹不定进度条。

        - ``compute_fn``：纯函数（只碰 numpy，禁止碰 Qt），在池线程执行；
        - ``apply_fn(result)``：拿到结果后在主线程执行的收尾（更新图/日志/y_data）。

        用 ``utils.qt_bridge.run`` 把结果排队回主线程；``on_done``/``on_error``
        必须是 QObject 绑定方法（这里用 ``_on_heavy_done``/``_on_heavy_failed``），
        真正的 apply 逻辑暂存在 ``_pending_apply`` 里，回到主线程后再调用。
        """
        from utils.qt_bridge import run as run_in_pool

        self._pending_apply = apply_fn
        busy = QProgressDialog(description, None, 0, 0, self)
        busy.setWindowTitle("处理中")
        busy.setWindowModality(Qt.WindowModality.WindowModal)
        busy.setCancelButton(None)
        busy.setMinimumDuration(0)
        busy.show()
        self._busy_dialog = busy
        run_in_pool(
            "compute", compute_fn,
            on_done=self._on_heavy_done,
            on_error=self._on_heavy_failed,
            owner=self,
        )

    def _close_busy(self) -> None:
        busy = getattr(self, "_busy_dialog", None)
        if busy is not None:
            busy.close()
            self._busy_dialog = None

    def _on_heavy_done(self, result) -> None:
        self._close_busy()
        apply_fn = getattr(self, "_pending_apply", None)
        self._pending_apply = None
        if apply_fn is not None:
            apply_fn(result)

    def _on_heavy_failed(self, exc) -> None:
        self._close_busy()
        self._pending_apply = None
        self._log(f"后台计算失败: {exc!r}")

    # ─── 焦点管理 ────────────────────────────────────────

    @property
    def _focus_widget(self) -> QWidget:
        for attr in ("signal_plot_widget", "plot_widget"):
            widget = getattr(self, attr, None)
            if widget is not None:
                return widget
        return self

    def showEvent(self, ev) -> None:
        super().showEvent(ev)
        self.activateWindow()
        self._focus_widget.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
