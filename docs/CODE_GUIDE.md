# MTPro 代码功能指南

> 本文档由通读 `python/src` 与 `rust_lib/src` 全部源码后整理，按包记录每个文件的用途、关键结构、算法与非显然行为。约 70 个文件、2.4 万行。

## 项目定位

MTPro 是一个基于 **PySide6** 的大地电磁（MT）时间序列处理与可视化桌面应用，配套一个 **PyO3 Rust 扩展**（`rust_lib`）。功能闭环：

1. 加载 MAT / 二进制 `.dat` 格式的 EM 时间序列；
2. 多通道波形浏览，叠加 Kp 地磁指数；
3. 交互式处理工具：去尖峰、小波多尺度去噪、框选区域去噪、差分/窗口/鲁棒基线去阶跃、正弦相减去日变；
4. 阻抗张量估计：单台 Robust 估计、远参考法，输出 Z 文件 / EDI；
5. 可选 Nuitka 打包为桌面二进制。

`emtf` 包是 EMTF（Egbert's MT processing）Fortran 程序的 Python 包装层，兼作 `emtf_rs` 的 drop-in 替代。

---

## main.py — 应用入口

- 设置 `PYQTGRAPH_QT_LIB=PySide6`，强制 pyqtgraph 使用 PySide6。
- `Main(QObject)` 组合 `DataManager` 与 `MainWindow`；`main()` 显示最大化窗口并进入 Qt 事件循环。
- 启动前 `install_standard_stream_logging` 把 stdout/stderr 重定向到 logger。

---

## core 包 — 数据模型

### `em_data.py` `EMData(ABC)`
- 通道数据核心容器：`data: xr.Dataset`，按通道名为变量，`time` 维对齐 `pd.DatetimeIndex`。
- 元数据：`name/path/zero_time/start_time/end_time/dt/lat/lon/elevation/e_units/m_units/timezone/kp_data`。
- `_chid_shadow/_npts_shadow/_NCh_shadow`：`data` 为空时的回退值。
- **核心交换格式**：`get_channel_series(name) -> pd.Series(DatetimeIndex)`、`set_channel_series(name, series)`（强制 `np.float64`，索引/长度不一致时 `reindex`）。
- 属性 `chid`(通道名列表)/`npts`/`NCh` 优先取 `data` 真实值，空时取 shadow。
- 抽象方法 `restore_data(ch)` 预留通道恢复钩子（子类均空实现）。

### `mat_data.py` `MatEMData(EMData)`
- 从 MATLAB `.mat`（scipy 或 v7.3 HDF5）解析。自动发现顶层 key（`tsStruct`/`ts`），存 `_struct_key`。
- 时区：文件内 `timeZone` 字段优先，否则用传入 `tz`（默认 `Asia/Shanghai`）；时间戳按该时区 localize 后 `tz_convert('UTC')`。
- 数据矩阵方向自适应（`shape[0]!=NCh` 时转置）；`npts` 与实际不符以实际为准并 warning。
- `units` cell 解析：`(1,2)` 视为 `[m_units, e_units]`；单元素按通道首字母 `E`/`B,H` 判定。
- 构造末尾自动 `fetch_kp(start, end)` 取 Kp 指数。
- **`update_meta()`**：裁剪通道首尾 NaN，同步 `npts/start_time/end_time/datetime_index`，用 `pts_to_array` 反推 `startTime/endTime/zeroTime`（按原时区），重写 `data/dt/npts/NCh`。是 `MatSaver.save` 的前置步骤。
- 原始 `mat_meta` dict 保留并在保存时回写。

### `bin_data.py` `BinEMData(EMData)`
- 从二进制 `.dat` 导入的容器。接收已应用缩放因子的 `dict[str, pd.Series]`，仪器参数在导入时已消耗。
- 时间轴取第一条 series 索引；`end_time = start + dt*(npts-1)`（忽略记录间隙）。
- `zero_time` = 数据年份 1 月 1 日 00:00:00（同 tz）。单位固定 `mV/km` / `nT`。
- `kp_data = None`（不自动获取，与 MatEMData 不同）。

### `db_data.py` `EMDBData(EMData)`
- 占位类，数据库加载预留，未实现。

---

## base 包 — 注册表与中介

### `plugin_loader.py` / `plugin_runtime.py`
- 插件按一方插件、`mtpro.plugins` entry point、`MTPRO_PLUGIN_PATH`、用户扫描路径、默认目录的顺序加载；目录和 zip 都支持，单个插件失败只进入 `LoadReport.failed`。
- `plugin_runtime.run_plugin_process` 补齐默认参数并强制检查 `TransformResult` 与输入同长度、同 index，供 Classic、Pipeline 和批处理共用。

### `batch_runner.py`
- `BatchRunner` 对目标间并行、目标内按 recipe step 串行；单目标失败隔离。带点索引的步骤遇到采样间隔不一致时跳过并写诊断。

### `data_manager.py` `DataManager(QObject)`
- 全局数据集注册表：`_em_datas: Dict[int, EMData]`，`_next_key` 自增，`_active_id` 活跃键。
- 信号：`data_added(int)` / `data_removed(int)` / `active_changed(int)`。
- API：`add(data)->key`、`remove(key)`、`get(key)`、`items()`、`set_active(key)`、`active_id()`、`active_data()`。
- **key=0 表示"无活跃数据"**，`get(0)` 永远返回 `None`。`remove` 当前活跃项时自动 `set_active(0)`。`set_active` 相同 key 不发信号。

### `processing_broker.py` `ProcessingBroker(QObject)`
- 数据处理工具与 `DataManager` 的中介，跑在后台 `QThread`，统一读写。
- `StationMetadata(name, latitude, longitude, elevation)` (frozen dataclass)。
- 信号：`data_ready(int, str, object)` / `result_applied(int, str)` / `data_changed(int, str, object)`。
- `request_data(data_id, channel)`：`channel==""` 返回台站级包（含 `stations` 列表 + 全通道 `all_y` + `StationMetadata`）；否则单通道 `{x, y, channel_names, all_y, npts, start_time, dt, kp_data}`。`x` 经 `dti_to_numpy` 转 float 秒数组。
- `apply_result`：长度匹配用原 index 写 Series，否则 `set_channel_data`；发 `result_applied` + `data_changed`。
- `apply_result_all_channels`：跨通道同步 NaN 位置（mask 与 result 的 NaN 一致），逐通道发 `data_changed`。
- `apply_region_to_all`：`RemoveRegion` 跨数据集同步；接收 `time_ranges` 列表，对每个数据集按自身 `start_time/dt` 计算索引区间置 NaN，**用绝对时间戳而非点索引**，跨数据集健壮。`QMutexLocker` 保护。

### `time_viewport_mixin.py` `TimeViewportMixin`
- 波形浏览器的视口状态管理混入，提供 x 轴视口与 Qt 控件双向同步。
- 状态 `duration`/`x_view_start`/`x_view_size`（`pd.Timedelta`）；控件 `start_time/end_time: QDateTimeEdit`、`scroll_x`、`slide_time`、`btn_plot`。
- **16ms（≈60fps）防抖**单次定时器触发 `_on_viewport_changed()`。
- `_clamp_jump`：时间跳转边界限幅，自动交换颠倒起止、限幅至数据边界、保证最小 1 分钟视口。
- `show_warning(msg)`：非模态 QMessageBox，3 秒自动关闭。
- 子类必须实现 `_on_viewport_changed()`。

---

## io_utils 包 — 文件 I/O

### `em_io.py`
- 抽象基类 `EMDataLoader` / `EMDataSaver`，静态接口 `load(path)->EMData` / `save(data, path)`。

### `mat_io.py`
- 兼容 scipy 与 MATLAB v7.3 (HDF5)。
- `_load_mat_v73`：递归把 HDF5 Group/Dataset 转 scipy 兼容结构；处理 char 数组（UTF-16 解码）、cell 引用、2D 转置（MATLAB 列优先 vs HDF5 行优先）、float 全整数转 int64（避免 Timestamp 误解为 epoch）。
- `TS_STRUCT_TEMPLATE`：legacy `tsStruct` 完整字段模板（`NCh/npts/data/startTime/zeroTime/endTime/dt/timeStamp/name/lat/lon/elev/chid/units/missVals/...`）。
- `_make_timestamp_(start, dt, npts)`：生成 "years" 小数时间戳行向量（按闰年算 `secs_in_year`）。
- `MatLoader.load(path, tz)`：读 128 字节头判断格式，构造 `MatEMData`。
- `MatSaver.save`：`MatEMData` 调 `update_meta()` 同步回 `mat_meta` 再 `savemat`；其他子类用 `new_ts_struct()` 构造新结构保存。
- **注意**：`_save_mat_v73` 已写但未被调用，统一走 scipy `savemat`。

### `bin_io.py`
- 大端序二进制 `.dat`(46B/记录: 7×`>u2` 时间 + 8×`>f4` 通道) 与 `.inf`(26B/记录: 时间 + 3×`>f4` GPS)。GPS 值 ÷`GPS_DIVISOR=100`。
- `_build_datetime_index`：时间字段 → naive DatetimeIndex + 毫秒偏移 → localize → convert UTC。
- `_infer_dt`：用 second/minute/hour/day 总秒数差值的**众数**推采样间隔，修正跨天回绕；fallback 中位数。
- `read_bin_header` / `read_gps_header`：读前 200 条用于向导预览。
- `BinLoader.load(path, tz, channel_configs(8项), gps_path, ex_distance, ey_distance, manual_lat/lon/alt, name, dt_override)`：
  - 默认 8 通道 `Ch1..Ch8` scale=1.0。
  - 时间轴只用第一条记录时间 + `pd.date_range(periods=npts, freq=dt)`（避开不可靠毫秒字段）。
  - 电场偶极距换算：`mV / (m/1000) = mV/km`；方向翻转：Ex-S / Ey-W 取负。
  - GPS：手动值优先，否则读 `.inf`（lat≠0 首条）。
- `BinSaver.save`：`NotImplementedError`，提示改用 MAT。

### `db_io.py`
- 仅 `import sqlite3`，空文件，预留。

### `database_manager.py`
- 极简 SQLite 包装：`connect/disconnect/execute_query(自动 commit)/execute_read_query`。无事务/上下文管理。

---

## utils 包 — 工具函数

### `app_state.py`
- 应用持久化状态（TOML）。`APP_NAME="MTPro"`、`MAX_RECENT_FILES=8`。
- `AppState` dataclass：`kp_hdf5_path/last_open_dir/active_tab(默认"波形浏览")/workspace_splitter_sizes/log_visible/recent_files/plugin_paths` 等。
- 路径解析：优先 env `MTPRO_APP_STATE_PATH`；Windows 用 `LOCALAPPDATA`；其他 `XDG_CONFIG_HOME` 或 `~/.config`；最终 `<base>/mtpro/app_state.toml`。
- `load_app_state`：读 `[app]/[paths]/[recent]` 三段，路径 expanduser+resolve，recent 去重限 8。
- `save_app_state`：手写 TOML（json.dumps 转义），自动建目录，失败静默。

### `fetch_kp.py`
- 获取 Kp 地磁指数，优先本地 HDF5，缺失从 GFZ 网络补齐并写回，网络失败填 NaN。
- `align_to_3hour_boundary(ts, 'floor'|'ceil')`：对齐到 UTC 3 小时整点。
- `fetch_kp(start, end, hdf5_path='kp.h5')`：`hdf5_path=='kp.h5'` 时自动从 `load_app_state().kp_hdf5_path` 取真实路径。统一 UTC，ceil 对齐 start、floor 对齐 end；生成 3h 频率期望序列；本地有且完整直接返回，有缺失从首个缺失点请求网络到末点。
- 网络：GET `https://kp.gfz.de/app/json/?start=...&end=...&index=Kp&status=all`，15s 超时。
- HDF5 I/O：用 h5py 直写 pandas fixed 格式（`axis0/axis1(block0_values)/block0_items`），索引存 ns 整数。

### `geo.py`
- WGS84 Haversine：`great_circle_distance(lat1,lon1,lat2,lon2,unit)`，`EARTH_RADIUS_M=6_371_000`，`asin(min(1,...))` 防溢出。
- `station_distance(a, b)`：从两个 `EMData` 取 lat/lon，缺失抛 `ValueError`。

### `decimate.py`
- 波形降采样与 LOD 金字塔，支撑百万点流畅浏览。
- Rust 加速：检测 `rust_lib`，可用调 `minmax_decimate_rs` / `build_lod_pyramid_rs`，失败回退 Python。
- `minmax_decimate(x, y, target_bins=1000)`：分桶取 min/max 交错，保证尖峰不丢，输出 ≤ 2×target_bins。
- `build_lod_pyramid`：从粗到精多层降采样，每级 `target=len//5`，下限 256。
- `LODPyramid` 类：预计算金字塔，`get(x_start, x_end, pixel_width)` 按可见点数 ≤ 2×像素选层或切原始。
- `slice_visible(x, y, x_start, x_end, margin_ratio=0.1)`：切可见范围含 10% 边距。

### `series.py`
- `dti_to_numpy(dti)`：DatetimeIndex/Series → float64 秒数组。统一 `as_unit('ns')` 再 `asi8/1e9`，兼容 pandas 各版本分辨率。

### `signals.py`
- `block_signals(*widgets)` contextmanager：临时屏蔽多个 QObject 信号。

### `time_convert.py`
- `pts_to_qdt(pts, tz=None)`：`pd.Timestamp` → `QDateTime`，用 `QTimeZone(str(pts.tz).encode())`，`fromSecsSinceEpoch`（秒精度，丢亚秒）。
- `qdt_to_pts(qdt, tz='UTC')`：`QDateTime` → `pd.Timestamp`。

### `timestamp.py`
- `pts_to_array(timestamp)`：Timestamp → `[year, month, day, hour, minute, second]` 整数列表（MATLAB dateVec 风格）。

### `validators.py`
- QLineEdit 输入校验，失败通过 `log` 回调报告并返回 `None`：`read_positive_float` / `read_nonnegative_int` / `read_positive_int`。
- 注意：`int(text)`/`float(text)` 不捕获异常，非法输入会抛 `ValueError`（仅空字符串和符号校验被处理）。

---

## ui 包 — 用户界面

### 插件宿主
- `plugin_adapter.py`：自动参数对话框、Classic 接线壳与 compute-pool 执行桥；插件 Editor 只提交 JSON 参数，数据统一由 `process()` 产生。
- `plugin_manager_dialog.py`：展示加载成功/失败、管理扫描路径、安装本地目录或 zip、重新扫描。
- `batch_process_dialog.py`：加载 recipe，勾选多个已打开通道，执行后把成功结果写回内存数据集。
- `plugins/`：按功能拆分的插件包（7 个单通道变换插件及 `output/` 下的输出工具）；`builtin_plugins/` 仅作为内置实现集合被加载器引用。

### 第三方插件开发
- 只从 `plugin_api.py` 导入 `PluginInfo`、`ParamSpec`、`ChannelBlock`、`TransformResult` 等稳定 API；`process()` 禁止触碰 Qt，必须容忍 NaN。
- 目录形态需要根层 `plugin.toml`；打包安装可声明 `[project.entry-points."mtpro.plugins"]`。完整可运行示例见 `docs/example_plugin/`。
- 参数必须可 JSON 序列化。点级决定用 block 相对索引；时间区间用带时区 ISO 字符串。

### `mainwindow.py` `MainWindow(QMainWindow)`
- 应用顶层窗口：菜单/工具栏/状态栏、欢迎页↔工作区切换、文件打开/保存、最近文件、日志面板显隐、数据集激活/移除联动。
- 持有 `DataManager`、`DataTreeViewer`、`DataProcessWidget`、`LogConsoleWidget`、`Welcome`、`AppState`。
- 布局：`central_stack`(QStackedWidget) = Welcome / 工作区。工作区为竖向 splitter：上=横向 splitter（数据树 | tab_widget[波形浏览/数据处理]），下=LogConsole。波形浏览用 `data_view_stack`（占位 + 各 dataset 的 DataViewWidget）。
- 信号接线：`welcome.open_button → open_file`；`data_manager.data_added/active_changed/data_removed → ...`；`data_tree.tool_requested → _open_processing_tool_from_sidebar`；`data_process.change_finished_signal → on_data_change_finished`。
- 打开 MAT 必须先经 `TimezoneDialog`；批量打开共用一次时区选择，逐文件带进度条。
- `_ensure_data_viewer` 惰性创建并缓存每个 key 的 DataViewWidget。
- `_resolve_processing_target`：台站级工具（robust estimate / remote reference）只需 dataset；否则优先用树中选中通道，无则弹 `QInputDialog`。
- `_BatchSaveDialog`：批量保存对话框（目标文件夹 + 后缀 + 前5条预览）。
- AppState 持久化 splitter 尺寸、log_visible、active_tab、recent_files、last_open_dir。

### `welcome.py` `Welcome(QWidget)`
- 启动首屏：左侧品牌区（MTPro 标题/版本/支持格式/功能列表），右侧"开始/最近文件"。
- 信号 `recent_file_requested(str)`。最近文件树双击 → 取 UserRole 完整路径 → emit。
- `set_recent_files` 重建模型，列=文件名/父目录，tooltip 存完整路径。

### `data_view_widget.py` `DataViewWidget(TimeViewportMixin, QWidget)`
- 单数据集多通道波形浏览：每通道一条 `BriefPlot`，支持时间窗口导航、通道可见性勾选、**通道拖拽重排序**。
- `PlotRowContainer(QFrame)`：整体可拖拽，MIME=`application/x-channel-reorder`，超 10px 曼哈顿距离启动 QDrag；drop 解码被拖通道名并 emit `drop_reorder`。
- 通道排序：E → B/H → Kp（Kp 永远最后）。默认勾选：总通道 ≤8 全勾，否则前7个非Kp + Kp。
- 每通道独立 `QThread` + `BriefWorker`，跨线程排队调用 `widget.update_plot`。
- `_refresh_plots`：E/B/H 去均值折线，Kp 三色柱状图。拖拽时去抖 50ms，释放后恢复 16ms 并立即刷新。
- `shutdown`/`closeEvent` 安全退出所有工作线程。

### `data_process.py` — 数据处理面板
- Classic/Pipeline 模式切换 + MDI。**`PIPELINE_ENABLED = True`**；pipeline block 通过插件 `process()` 异步执行并可导出 recipe。
- `DataProcessLegacyWidget`：Classic 宿主，管理 MDI 子窗口 + `ProcessingBroker`。`open_tool_window` 实例化具体 processor 并 `_wire_tool` 接到 broker。
- `WorkspaceMdiArea`：单工作区 MDI，拖放 classic 工具（`CLASSIC_TOOL_MIME`）或 pipeline 通道（`DATA_TREE_SELECTION_MIME`，JSON payload）。
- `WorkspaceTabPanel`：多工作区 tab，末尾常驻"+"tab 新建工作区。
- 信号链：`broker.data_changed → on_change_data_finished_for`；`broker.result_applied → _on_tool_result_applied`；`tool.data_requested → broker.request_data`；`tool.result_submitted → broker.apply_result`。
- `_canonical_tool_type`："remove step"→"remove step diff"，"remove pulse wavelet"→"remove wavelet denoise"。
- 工具窗口去重：相同 (tool_key, data_id, channel_name) 复用。
- `RemoveRegion` 的 `region_sync_requested → _on_region_sync`：ThreadPoolExecutor 并行把时间段置 NaN 到其他数据集，预计算 decimate 显示数据，主线程直接 setData。

### `data_process_pipeline.py` — Pipeline 编辑器（当前禁用）
- 通道级 pipeline 编辑器：每通道一条带多层 step 轨道的可视化编辑器，可放置/拖动/调整 block，双击编辑，preview/commit 串联结果。
- 数据类：`PipelineBlock`(block_id/tool_id/start/end/params/status/input_snapshot/output_snapshot)、`PipelineLayer`、`ChannelPipeline`。
- `SnapEngine`：网格吸附 + 邻接 block 边吸附（阈值 8px）。
- `PipelineTrackWidget`：单层轨道，自绘 block（状态色/选中边框/把手），支持拖动/resize/跨层移动/吸附/右键删除/双击编辑。
- 通道必须有时间重叠（`_common_time_range`）才能加入同一 pipeline。
- block 状态：empty/configured/ready/stale，各有 accent 色。
- interactive block 双击打开真实 processor（RemoveSpike/RemoveStepByWindow/RemoveStepByDiff），结果缓存为 snapshot；param block 只存参数。
- commit：把 preview snapshot 写回 `data.set_channel_series`，烘焙进 baseline。

### `data_process_shared.py`
- 拖放 MIME 常量：`DATA_TREE_SELECTION_MIME` / `CLASSIC_TOOL_MIME` / `PIPELINE_TOOL_MIME`。
- `create_classic_tool_model()`：三级工具树——手动数据处理（框选区域/框选尖峰/差分去阶跃）、自动数据处理（窗口均值/鲁棒基线/小波多尺度/正弦相减去日变）、阻抗输出（Robust估计/远参考法）。

### `data_tree_view.py` `DataTreeViewer(QWidget)`
- 左侧面板：数据集/通道树 + 台站摘要/处理工具列表（按 tab 切换），支持拖放、右键菜单、双击打开单通道图、序列对比。
- 信号 `tool_requested(str)`。
- `DataSelectionTreeView`：拖出 `DATA_TREE_SELECTION_MIME`（dataset → {kind,data_id}；channel → {kind:"channels", refs:[...]}，refs 取勾选或当前通道）。
- `ToolTreeView`：拖出 `CLASSIC_TOOL_MIME`，载荷=tool_type 字符串。
- 右键菜单：dataset 级=保存/另存为/关闭；channel 级=显示曲线/序列对比/编辑元数据（占位）。
- 单通道 PlotPanel 缓存在 `_plot_panels[(key,ch)]`，重复双击复用；E 通道用 e_units，B/H 用 m_units 作 Y 轴标签。

### `data_tree_model.py`
- 两级树 QAbstractItemModel（dataset → channel），通道多选勾选（CheckState），与 DataManager 双向同步。
- 角色：`KEY_ROLE`/`CHANNEL_ROLE`/`NODE_KIND_ROLE`；`internalId()==_ROOT_ID(0)` 区分节点类型。
- `_checked_channels: Dict[int, Set[str]]` 维护每数据集勾选通道；空集合从 dict 移除。
- 通道名优先取 `em_data.chid`（自定义顺序），否则 `channel_names()`。

### `brief_plot.py` `BriefPlot(PlotWidget)`
- pyqtgraph 封装，支持折线与柱状图，多曲线管理，跨线程视口更新。
- `BriefWorker(QObject)`：信号 `update(start,end)` / `visible(label,bool)` / `range_change`。
- 禁用鼠标拖拽/缩放（外部统一控制）。
- 折线颜色按 label 的 **MD5 哈希 → HSL 色相**（S=0.40 L=0.85），同 label 颜色稳定。
- Kp 柱状图三色：<4 蓝、4-6 黄、≥6 红；柱宽=相邻间隔，x0 左移一个宽度（Kp 代表前3小时区间）。
- X 轴 `DateAxisItem(utcOffset=0)`；`autoDownsample + clipToView + connect='finite'`（NaN 断开）。

### `plot_panel.py` `PlotPanel(TimeViewportMixin, QWidget)`
- 独立窗口式单通道绘图，集成 LOD 金字塔、X/Y 双轴滚动条、Y 轴缩放、时间窗口导航。
- `LODPyramid` 预抽稀多级 LOD，按 PlotWidget 像素宽度查询对应分辨率，只 setData 可见范围。
- Y 滚动条把浮点幅值 ×10000 映射为 int（4 位小数精度）。
- `zoom_ratio<1` 放大（启用 Y 滚动条）；`=1` 原始；`>1` 缩小（中心锚点扩展，禁用滚动条）。
- 拖动时去抖 50ms（20fps），结束恢复 16ms。
- 有父窗口时 closeEvent 隐藏而非销毁（复用）。

### `time_slider.py`
- 继承 QSlider，把整数索引 0~N 映射到非等距时间刻度（1h→360d），自绘刻度标签。
- 标签按索引奇偶性上下交替避免重叠；激活刻度高亮。
- 信号 `timeChanged(int)` 发送当前刻度秒数。

### `timezone_dialog.py` `TimezoneDialog(QDialog)`
- 打开 MAT 时弹窗选时区（MAT 时间字段无时区信息）。
- 常见时区按 MT 使用频率排序，默认 Asia/Shanghai。
- `_format_utc_offset` 用当前时刻偏移（考虑 DST）。
- `_on_accept` 用 `pytz.timezone` 校验，无效时显示红色错误并阻止关闭。

### `import_bin_wizard.py` `ImportBinWizard(QWizard)`
- 四步向导：文件选择 → 通道与仪器配置 → GPS/时区 → 确认导入。
- `FileSelectPage`：实时显示记录数/文件大小/时间范围/采样率。
- `ChannelConfigPage`：8 通道表格 + 预设（LMT-1/MT5/默认）+ 磁场缩放 + Ex/Ey 偶极距 + 电场方向同步复选框 + 采样周期。
  - 通道名首字母 H/h/B/b 视为磁场通道，自动用 mag_scale 并禁用电极对下拉。
  - 电场方向同步：所有 Ex 共用方向、Ey 共用方向；S/W 方向表示反向（×-1）。
- `ConfirmPage`：`validatePage` 调 `BinLoader.load` 真实导入，带进度条，成功后注册到 DataManager。

### `compare_widget.py` `CompareWidget(TimeViewportMixin, QWidget)`
- 多通道序列对比：按 E/B 分组绘制到上下两个 BriefPlot 子图，统一时间窗口导航。
- E 开头→上方，B/H 开头→下方，其余跳过。每条曲线去均值对齐。无曲线的子图自动隐藏。

### `map_widget.py` `MapWidget(QWidget)`
- QWebEngineView 内嵌 Leaflet.js，默认高德卫星图+注记，支持添加标记/清除/飞行/切换底图。
- `TileSource` Enum：SATELLITE/NORMAL/TERRAIN。
- `_MapHandler(QObject)`：`@Slot` 暴露方法，emit Signal 给 JS。
- 坐标系提示：高德/天地图用 GCJ-02，WGS-84 输入需先转换否则偏移 100-700m。

### `log_console.py`
- 底部日志面板 + 应用日志配置 + stdout/stderr 重定向到 logger。
- `_LogEmitter(QObject)` 信号 `message_ready(level, message)`，跨线程安全。
- `QtTextEditHandler`：通过 emitter 转发（解决 Qt 不允许子线程直接操作 widget）。
- `LogConsoleWidget`：QTextEdit（maxBlockCount=3000，NoWrap，Courier New）+ 清空/复制按钮；`append_log` 按级别着色（深/浅色主题不同色板），HTML 转义。
- `StreamToLogger`：把 stdout/stderr 按行转发到 logger，同时透传给原始 stream。
- `configure_application_logging`：root logger INFO，FileHandler 写 `<tmp>/mtpro_logs/mtpro.log`。

---

## processor 包 — 处理工具

### `_base_processor.py` `BaseProcessorWidget(QWidget)`
- 所有处理器工具窗口基类，统一两段式数据初始化、撤销/还原/确认流程。
- 信号 `data_requested(int, str)` / `result_submitted(int, str, object)`，工具不持有 `ProcessingBroker`，只通过信号通信。
- `wire_data_ready()` 连外部数据信号到 `_on_data_ready`。
- `_on_data_ready`：检查 id/通道匹配，取 `pkg['x'/'y']`，`pd.Series.interpolate(method="linear")` 插值 NaN，缓存 `nan_index`，`origin_y_data` = 当前 y 副本。
- `_save_history / _undo / _revert / _restore_nan / _confirm_and_close`：统一机制；`_confirm_and_close` 还原 NaN 位置后发 `result_submitted`。
- 子类实现 `_init_ui / _log / _update_plot`，可选重写 `_on_data_loaded / _update_derived_data / _reset_result_state`。

### `_demo_utils.py`
- standalone demo 共享工具。`DemoEMData(EMData)` 轻量子类，由 `pd.Series` 直接构造 `xarray.Dataset`。
- `run_standalone_demo(widget_class, demo_name, mat_path, channel_name, build_demo_fn)`：复用 QApplication，无 mat 时调 `build_demo_fn` 构造演示数据。

### `_display_utils.py`
- `decimate_for_display(x, y, max_points=100_000)`：等步长 `x[::stride]` 降采样，配合 `autoDownsample='peak'`。
- `apply_nan_intervals(y, intervals)`：rust_lib 可用时调 `rust_lib.apply_nan_intervals_rs`，异常回退 Python。

### `_modifier_constraint.py`
- `apply_modifier_axis_constraint(view_box, modifiers)`：Shift→仅 X，Alt→仅 Y，否则双轴。

### `_region_mixin.py` `RegionManagementMixin`
- 给处理器窗口提供基于 `LinearRegionItem` 的区域选择交互。
- `_current_view_region_bounds`：取当前可视 X 范围并裁剪到数据范围。
- `_region_bounds_around(center_x)`：可视范围 20% 宽度构造区域。
- `_update_region_selection`：用 `np.searchsorted` 把 region 转成 `selected_indices`，至少 1 个采样点。
- 右键区域内点击删除区域。子类需提供 `signal_plot_item / x_data / region_item / selected_indices / selection_status_label`。

### `remove_spike.py` — 框选尖峰去噪
- `SpikePlotWidget(pg.PlotWidget)`：两种模式 `zoom`（框选放大）/`select`（框选选点，右键取消）。undo_stack 存每次删除索引数组。`modify_selected_data` 选中点置 NaN 压栈；`undo_last_deletion` 从 `origin_y_data` 还原。
- `RemoveSpike(QWidget)`：快捷键 Delete/Ctrl+S/Ctrl+Z/Tab。**不继承 `BaseProcessorWidget`**，自己实现两段式初始化。
- "应用到所有通道"复选框：确认时发 `channel_sync_requested` 同步其他通道。

### `remove_wavelet_denoise.py` — 小波多尺度去噪
- 继承 `BaseProcessorWidget + RegionManagementMixin`。
- 预设 `AUTO_PRESETS`：自适应保守（推荐）/常规宽带降噪/边界安全去噪，以及明确标记为旧版兼容的脉冲/尖峰/平滑降噪。
- 参数：小波基、变换(DWT/SWT/Wavelet Packet)、分解层数或保护周期、处理范围(局部/全通道)、模式(soft/hard/zero/manual)、阈值策略(SureShrink/通用阈值)、秒或采样点 padding、统一增益、WP 分组增益(低/中/高)。
- `_sync_gain_controls`：根据层数动态生成 D1..Dn / B1..Bn 增益 SpinBox。
- `_update_level_limit`：用 `pywt.dwt_max_level` 计算 DWT/WP 最大层数；SWT 按窗口长度限制。自动策略按采样率和保护周期换算层数。
- `_run_denoise` 调 `wavelet_multiscale_denoise`；`_preview_fit` 不落盘，`_apply_correction` 走 `_save_history`。
- 支持 TOML 预设导入/导出。
- 主图三条曲线（原始/当前/预览）+ 橙色 InfiniteLine 标局部窗 + detail 能量柱状图。

### `remove_diurnal_sinusoid.py` — 正弦相减去日变
- 继承 `BaseProcessorWidget`。谐波预设 1~6 阶（24h/12h/8h/6h/4.8h/4h）。
- `_on_data_loaded` 重写：使用 `time`+`linear` 双重插值（基类 linear-only）。
- 上下双图（信号 + 拟合噪声），X 轴双向同步（`_syncing_x_range` 防递归）。
- `_run_subtraction` 调 `subtract_diurnal_signal_series`；`_preview` 不写回，`_apply_preview` 写回并把噪声曲线挪到"当前"。

### `diurnal_sinusoid_subtraction.py` — 日变算法层（无 GUI）
- `PeriodicSubtractionConfig`(frozen dataclass)：基础周期(默认 24h)、谐波元组、窗口/步长、趋势阶数、最小有效比例/点数。
- `subtract_periodic_signal_series`：要求 `DatetimeIndex` 且等采样；`_window_weights` 用 Hann 型权重(`np.hanning(window+2)[1:-1]`)降低接缝。
- 每窗口：`_design_matrices` 构造归一化时间趋势项 + 每谐波 cos/sin 列，`np.linalg.lstsq` 拟合，只取谐波系数重建 `harmonic_signal`。
- `estimated_noise = noise_sum / weight_sum`，`cleaned = values - estimated_noise`，NaN 位置保持 NaN。
- 诊断列：window_id/start/end/valid_samples/valid_fraction/status + 每谐波 amplitude + residual_rms/harmonic_rms。

### `remove_step.py`
- 向后兼容别名：`RemoveStep = RemoveStepByDiff`、`WindowedDeStep = RemoveStepByWindow`。

### `step_algorithms.py` — 共用纯数值算法
- **差分**：`detect_step_indices(diff_abs, min_offset, max_steps)` 循环取 argmax；`zero_diff_indices` 置零后 cumsum 重建；`remove_diff_steps_by_count / _by_threshold`。
- **窗口均值** `windowed_mean_destep_with_diagnostics`：差分找候选，前后窗口均值差估台阶，超阈值才应用，对后续段 shift 累计；返回 applied_steps + candidates 诊断。
- **鲁棒基线** `robust_baseline_destep`：`least_squares`(`soft_l1`) 两步拟合——pre-step 多项式基线 + post-step `_saturated_transition`(tⁿ 饱和) + 可选二阶平滑正则；f_scale 用 MAD×1.4826 估计。
- **小波去噪** `wavelet_multiscale_denoise`：
  - 取局部窗口(≥16 点)，NaN 用 `np.interp` 填；`pywt.dwt_max_level` 限层数。
  - DWT/SWT：保留近似项；每细节层按 mode 处理。自动模式使用背景 MAD 估 σ，再使用逐层 SureShrink 或通用阈值；手动模式乘增益。SWT 对称补齐后以 `pywt.swt/iswt` 重构。
  - Wavelet Packet：`WaveletPacket.get_level(level, order='freq')`，每节点类似处理。
  - `auto_by_period` 以保护周期和采样率选层，`padding_seconds` 优先换算为点数；重构后余弦 ramp(`blend_width`) 边界混合避免接缝。输入 NaN 只在计算时插值，输出仍保留 NaN。返回实际阈值、策略及 corrected/correction/detail_energy/band_labels/band_ranges_hz。
- `wavelet_local_depulse` 是 DWT 简化封装。

### `step_plot_widget.py` `StepPlotWidget(pg.PlotWidget)`
- `make_curve`：自动 `decimate_for_display` + `autoDownsample='peak'/clipToView/connect='finite'`。
- 滚轮缩放临时应用修饰键约束，结束恢复双轴，触发 `zoom_callback`。
- 双击/右键回调由外部注入（返回 True 拦截）。

### `remove_step_by_baseline.py` — 鲁棒基线去阶跃
- 继承 `BaseProcessorWidget + RegionManagementMixin`。适合平滑过渡阶跃。
- 控件：基线阶数(0/1)、过渡阶数(1~10)、上下文点数、损失函数(linear/soft_l1/huber/cauchy/arctan)。
- `_fit_current_region → robust_baseline_destep`；双击添加/移动区域，右键区域内删除。

### `remove_step_by_diff.py` — 差分去阶跃
- 继承 `BaseProcessorWidget`。主图 + 差分图(`RegionSelectionPlotWidget`)，X 轴双向同步。
- `_on_data_loaded` 计算 `diff_y_data = np.diff(y_data)`，阈值默认 `max(|diff|)`，打印 Top10 差分。
- 三种操作：按阈值 / 按次数 / 手动框选差分点 `zero_diff_indices`。
- 找不到台阶时弹掉刚压入的历史（避免空撤销）。

### `remove_step_by_window.py` — 窗口均值去阶跃
- 继承 `BaseProcessorWidget`。参数：差分阈值、平均窗口、最多台阶。
- `_de_step_by_threshold → windowed_mean_destep_with_diagnostics`；候选被拒绝时 `_log_rejected_step_candidates`。

### `remove_region.py` — 连续区域框选去噪
- `RegionPlotWidget(pg.PlotWidget)`：多个 `LinearRegionItem`，`_get_selected_intervals` 用 searchsorted 转区间并合并重叠/相邻。
- `modify_selected_data`：区间内置 NaN 压栈索引+时间范围；`undo_last_deletion` 用 `origin_y_data` 还原并重建区域。
- `RemoveRegion(QWidget)`：快捷键 Delete/Ctrl+S/Ctrl+Z。"应用到所有通道" + "应用到所有数据"，后者自动勾选并禁用前者；确认时发 `region_sync_requested`（携带 `time_ranges + source_data_id`）。
- **Kp 柱状图叠加**：勾选"显示Kp"建右轴 ViewBox，三色柱；`_kp_density_exceeds` 判断"可视点数 > 像素宽/3"时自动隐藏 Kp 并禁用复选框，缩放后过密消失再恢复。

### `remote_reference.py` `RemoteReferenceWidget(QWidget)` — 远参考法
- 1720 行。信号 `result_ready / data_requested / result_submitted`。
- `_PkgWrapper`：把 Broker 数据包 dict 包装为 EMData 兼容接口。
- `RemoteReferenceControlDialog`：独立浮窗调参（本地/远参考台选择、距离、时间范围、通道映射、估计参数）。
- `_detect_channels`：按名称前缀(e/h/b)+x/y/z 自动映射 Ex/Ey/Hx/Hy/Hz。
- 上方本地站双 Y 轴(左 E 右 B) + 远参考站单轴(仅磁场)并排，X 轴双向同步；两图各一 LinearRegionItem，拖一个同步另一个。
- 后端选择：`emtf_pywrap`(本仓库 emtf 包) / `emtf_rs`(未安装禁用)。
- `_run_estimation`：用 `searchsorted` 找远参考起止对齐，NaN 跳过 detrend，调 `emtf_rs.estimate_remote_reference_spectrum` 或 `emtf.estimate_remote_reference_spectrum`。
- 结果：阻抗表、视电阻率/相位表、matplotlib 误差棒图；Hz 选中时动态加倾子响应表与图。
- 表格 Ctrl+C 复制 TSV；`_export_edi` 调 `format_edi_str`。

### `robust_estimate.py` `RobustEstimateWidget(QWidget)` — 单台 Robust 估计
- 1387 行，结构与 remote_reference 高度相似但单台。
- 单台站选择器 + 时间范围 + 通道映射(Ex/Ey/Hx/Hy/Hz，Hz 可选) + 估计参数 + 后端切换。
- 双 Y 轴绘图(左 E 右 B)，滚轮缩放等比同步 B 轴。
- `_run_estimation`：调 `estimate_single_station_spectrum`，pywrap 传 `decimation_levels/bands/station_name`(由 `_emtf_period_grid` 用 `period_grid_bands` 生成)。
- 警告：选中样点 < 26×窗口 时提示。
- 结果区一致（表 + 误差棒图 + 倾子动态页 + EDI 导出，模式 `ss`）。

---

## emtf 包 — EMTF Python 包装

### `__init__.py`
- 整合子模块。典型流程：`dnff` 生成 FC 文件 → `tranmt` 估计阻抗 → `ZFile.read` → `to_edi`。
- `__version__ = "0.1.0"`，作为 `emtf_rs` 的 drop-in Python 替代。

### `api.py`
- `BandConfig`(frozen dataclass)：name/min_period/max_period。
- `SpectrumResult` dataclass：frequencies_hz/阻抗张量及标准误/rho/phi/segment_count/n_observations/tipper/station_name/mode。`periods` 属性 = 1/freq。
- `to_bands(band_configs)`：按 period 范围 mask，每 band 求 `mean_complex`(有限值复数均值)、`combine_se`(sqrt(mean(se²)/N))，组装 `BandedResult`。
- `estimate_single_station_spectrum / estimate_remote_reference_spectrum`：与 `emtf_rs` 同签名，转调 `emtf.estimate.*`，再包装为 `SpectrumResult`。

### `estimate.py`
- 默认 1 Hz 6 级 4 倍 decimation + 24 个 FrequencyBand(30s–10000s)。
- `estimate_single_station / estimate_remote_reference`：接受 `window_size/step_size/...` 等兼容参数（大多忽略，仅 `huber_threshold/max_iterations` 透传）。
- `_estimate`：校验长度 → 构 Workspace → `write_asc + write_sp`(本地 5 通道，远参考另写 2 通道 Hx/Hy) → 写 cfg → `dnff -A` → 收集 FC → 写 tranmt cfg → `tranmt` → 解析 `MT/Z.*`/`.zss`/`.zrr` → 返回 `ZFileData`。
- `decimation_with_fft_window`：按用户 fft_window_size 重写 nwin/overlap/nfft。

### `types.py`
- `DecimationLevel`(nwin/window_length/decimation_factor/nfilt/fil_1..4/nfft/offset)，`to_cfg_line` 输出 decset.cfg 一行。
- `FrequencyBand`(decimation_level(1-indexed)/fc_start/fc_end 闭区间)。
- `ProcessingOptions`(robust/remote_reference/e_field_ref/coherence/rotations)。
- `Station`、`TransferFunction`（`apparent_resistivity = 0.2*T*|Z|²`、`phase = degrees(atan(im/real))`）。

### `band_presets.py`
- `standard_bands / fine_bands / half_decade_bands / preset_bands(name)`。
- `period_grid_bands`：把对数均分 period bin 映射到 EMTF `bs.cfg` 的 (decimation_level, fc_start, fc_end)。对每 level 累乘 decimation_factor 得 sample_rate，`period = nwin/(fc*level_sample_rate)`；按连续段合并为 FrequencyBand，按中心周期排序去重。

### `_config.py`
- 渲染 EMTF 各 `.cfg` 文本：`render_paths / render_decset / render_pwset / render_bands / render_options / render_tranmt_control / render_dnff_stdin`。

### `_binaries.py`
- 平台映射 linux→linux、win32→win64。`resolve_binary` 顺序：env `EMTF_BIN_DIR` → `python/src/bin/<plat>` → `python/src/EMTF/bin/<plat>`。Windows 自动加 `.exe`。

### `_runner.py`
- `Workspace`：临时目录虚拟文件系统，`cleanup=True` 用 `TemporaryDirectory`，`False` 用 `mkdtemp`。`run(binary, stdin, workspace, timeout=300)`。

### `io/binary.py`
- EMTF 标准二进制 + ASCII + SP 读写。
- `read_binary`：80 字节 station 头 + 块交错 int32(NBLK=256×nch)。
- `write_asc`：3 行头 + 每样本一行 9 字段整数(NaN→44444444 sentinel) + footer。scales 默认 H=100、E=1000。
- `write_sp`：站名/坐标/declination/nch/dt + 每通道 2-3 行。

### `io/edi.py`
- `EdiMetadata` / `BandedResult`（派生 `center_frequencies_hz / rho / phi / rho_se / phi_se`）。
- `format_edi`：生成 SEG EDI 文本（>HEAD/>INFO/>DEFINEMEAS/>=MTSECT/>FREQ/>ZROT/ZXXR..ZYY.VAR/可选 TX/TY/>END），缺失值 `EMPTY=1e32`，6 个一行。
- `format_z`：紧凑 EMTF Z 文件（TF + Inverse Coherent Signal Power Matrix + Residual Covariance），缺失 -999。
- 误差传播：`_rho = 0.2*T*|Z|²`；`_rho_se = 2*rho*z_se/|z|`；`_phase_se = degrees(z_se/|z|`。

### `io/zfile.py` `ZFileData`
- port `readZfile.m` / `writeZfile.m`。正则解析头/坐标/通道方位；逐 band 读 period/ndf/TF(4×nche 实数)/Inverse Coherent Signal Power Matrix(2×2 Hermitian 下三角 6 实数)/Residual Covariance(下三角)。`_RNAN=-999` 转 NaN。
- 派生：`Zxx/Zxy/Zyx/Zyy/Tzx/Tzy`、`apparent_resistivity`、`phase`(`emtf_phase_deg` = MATLAB 约定 `degrees(atan(im/real))`，real=0 退到 atan2)、阻抗误差。
- write 按 E12.4 格式输出。

### `io/fc.py` / `io/fc_reader.py`
- `FCOutput` + `read` + `load_spectrum(level, freq_idx)`：从 `.fN` 二进制读 Fourier 系数。
- `_unpack`（port unpack.m）：packed int32 按 M=65536 拆 → k1/k2 (mod 8) → 还原复数 → 乘 channel scale。
- `_read_fc_open`：512 字节 ASCII 头 regex 抽 nch/nd/nfmx/nwin/dr；逐 level 扫描记录头建 `start_freqs[nd, nf]`。

### `io/sp.py`
- port `readSp.m`。`SystemParams` + 嵌套 `Channel/Filter`。H 通道第 2 行只方位+tilt；E 通道含 line_len/azimuth/tilt/gain。

### `pipeline/dnff.py`
- `dnff(...)`：写 cfg，复制输入到 `data/`，`render_dnff_stdin` 喂交互输入，`-A` 开 ASCII。返回 Workspace(FC 在 `FC/`)。
- `dnff_from_arrays(channels, dt, decimation_levels, ...)`：直接从 numpy 走 `write_asc + write_sp` → `dnff -a`。

### `pipeline/tranmt.py`
- `tranmt(fc_files, bands, station_name, options, ...)`：写 cfg(tranmt.cfg 每 run 6 行)，复制 FC，RR 模式追加 `{station}_RR` run，输出 Z 文件在 `MT/`。

---

## rust_lib — PyO3 扩展

### `src/lib.rs`
- 模块 `rust_lib`，导出 4 个 Python 函数：`minmax_decimate_rs` / `build_lod_pyramid_rs` / `apply_nan_intervals_rs` / `probe_mat_v73`。

### `src/decimate.rs`
- `decimate_core(x, y, target_bins)`：均分 `bins=min(target_bins, n/2)` 块，**rayon 并行**对每块求 min/max 索引，按"min 在 max 之前"顺序返回，拼接得 min/max 包络降采样。
- `build_lod_pyramid_rs`：循环 `decimate_core`，初始 `target=n/5`(下限 256)，压栈继续直到收敛，返回多层 LOD 金字塔。
- `apply_nan_intervals_rs`：可写 numpy 数组 + 区间列表，`slice[start..end].fill(NAN)`，自动钳制越界。

### `src/mat_loader.rs`
- `probe_mat_v73(path)`：用 `hdf5_pure::File` 打开 MATLAB v7.3，返回根下 datasets+groups 名（过滤 `#` 开头内部名）。完整 loader 仍是 WIP，Python h5py 作为回退。

---

## 关键设计模式

- **信号驱动数据流**：`DataManager` 信号 → `MainWindow` 槽 → viewer/processor 更新。
- **xarray.Dataset 是规范内存存储**；`pd.Series(DatetimeIndex)` 是 `EMData` 与处理器之间的交换格式。
- **Mixin 共享行为**：`TimeViewportMixin`(时间导航)、`RegionManagementMixin`(区域选择)。
- **MIME 拖放**：工具选择 `CLASSIC_TOOL_MIME`，通道选择 `DATA_TREE_SELECTION_MIME`(JSON)，通道重排 `application/x-channel-reorder`，pipeline 工具 `PIPELINE_TOOL_MIME`。
- **撤销/重做**：处理器维护 `list[np.ndarray]` 快照历史栈。
- **LOD 金字塔**：百万点流畅浏览，Rust 加速降采样，Python 回退。
- **后台线程**：每通道独立 QThread 绘图；`ProcessingBroker` 跑在独立 QThread；`RemoveRegion` 跨数据集同步用 ThreadPoolExecutor。
