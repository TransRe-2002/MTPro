# MTPro 未完成工作清单

> 生成日期：2026-08-31
> 来源：全量代码扫描（`python/src` 113 个模块、`rust_lib`、`matlab/`、`scripts/`、36 个测试文件）+ git 历史核验。所有文件路径与行号均为扫描时实测。
> 用途：供人工审核后，作为后续实施 AI 的工单底稿。
>
> **环境硬要求**（见 `AGENTS.md` / `CLAUDE.md`）：一切 Python / pip / pytest 命令必须用 `python/.venv`（Anaconda Python 3.13），运行应用与测试必须 `PYTHONPATH=python/src`。测试套件 `pytest.ini` 带 `--forked`（进程隔离），不要移除。

---

## P0 · 未完成的真功能

### 1. Pipeline 处理模式 —— 已建成但禁用，缺执行引擎与确定性重放

**现状**：约 2000 行管线代码（`ui/data_process_pipeline.py` + `core/pipeline_model.py`：blocks/layers/快照引擎/preview/commit/交互式编辑器）与测试均已存在，但整个模式被功能开关屏蔽，界面不可达：

- `ui/data_process.py:1006` `PIPELINE_ENABLED = False`；
- 同文件 1057–1058 行按钮置灰，tooltip "Pipeline 功能暂时屏蔽"；1143–1144 行 `set_mode("pipeline")` 被强制钳回 classic；
- git 历史 HEAD~2 即 `wip(pipeline): 管线快照模型重构（input_snapshot→input_slice）与处理器接入`，本身就是 WIP。

**缺口 A — param block 从不执行**：

- `PipelineParamDialog` 文案自认："这类 block 先保存参数，不直接执行。后续接入真实 processor 后再参与 preview/commit"（`data_process_pipeline.py:147-148`）；
- preview 摘要打印 "param block 目前只保存参数，还不参与实际计算"（同文件 1875 行）；
- block 状态字面量 `"param block waiting to run"`（1807 行）；
- 目前 `kind="param"` 的工具只有 `robust_estimate` / `remote_reference`（`ui/tools/_registrations.py`），即 param block 今天是纯死配置。

**缺口 B — interactive block 无法确定性重放**：

- 只缓存编辑器产出的 ndarray 切片；上游一变 → `stale`，必须人工重开编辑器；
- 提交对话框文案自认："interactive block 的缓存结果已经烘焙进 baseline，后续继续编辑需要重新打开对应 block"（1943 行）。

**依赖提示**：插件框架（见 P1 第 3 项）的设计目标就是补这两个缺口（插件 `process()` = 执行引擎，"编辑器产出参数、process 消费参数" = 确定性重放）。**除非决定放弃插件化路线，否则不要为 Pipeline 单独实现执行引擎**，以免重复建设。

**验证方法**：`grep -n "PIPELINE_ENABLED" python/src/ui/data_process.py`；在 `python/src` 内搜索上述三段中文文案。

---

### 2. 数据库数据通路 —— 0%，且当前不可实例化

- `core/db_data.py`：`EMDBData(EMData)` 只有 `super().__init__()`。未实现抽象方法 `restore_data` → **实测无法实例化**（`TypeError: Can't instantiate abstract class EMDBData`）；且 `EMData.__init__(self, path)` 的必需参数 `path` 也没传。
- `io_utils/db_io.py`：整个文件只有一句 `import sqlite3`（CODE_GUIDE 自注"空文件，预留"）。
- `io_utils/database_manager.py`：26 行 sqlite 包装（connect / disconnect / execute_query / execute_read_query），无事务/上下文管理，**无任何调用方**。
- `python/data/emdata.db` 0 字节、`python/data/test.db` 20 MB：均为废弃 DB 方向的遗留物，无代码引用。
- 参照物：`python/old/em_data.py`（旧数据层，含 zstd/pickle + `DatabaseManager` 的原始设计）。

**处理建议**：要么按规划实现（先让 `EMDBData` 可实例化：传 `path`、实现 `restore_data`），要么连遗留 `.db` 文件一起清退。两者都不做的话至少删掉 0 字节的 `emdata.db`。

---

## P1 · 设计完成、零代码、待评审

### 3. 数据处理插件框架

- 设计文档：`docs/plugin_api_design.md`，**状态：待评审**。
- 规划 5 个阶段，目标文件全部不存在（已核验）：
  `plugin_api.py`、`base/plugin_loader.py`、`ui/plugin_adapter.py`、`ui/plugin_manager_dialog.py`、`builtin_plugins/`（7 个变换工具迁移为一方插件）、`core/recipe.py`、`base/batch_runner.py`、`ui/batch_process_dialog.py`，外加 `AppState.plugin_paths` 字段。
- `python/src` 中 `PluginInfo` / `ParamSpec` / `API_VERSION` 零命中 → 实施进度 ~0%。
- 该框架明确以补 Pipeline 缺口 A/B 为交付目标（见其 §1.1 与 §7）。
- **实施前提**：文档 §12 有 4 个开放问题需评审人先表态（一方插件目录位置、`error` 状态是否进 `PipelineBlock`、params 索引基准、zip 直扔是否纳入 v1）。未评审前不要开工。

---

## P2 · 零散缺口

### 4. `BinSaver.save` 未实现

- `io_utils/bin_io.py:367-371`：`raise NotImplementedError("二进制格式保存暂未实现，请使用另存为 MAT 格式")`。
- Bin 格式只有导入向导，没有保存，格式支持不对称（~50%）。

### 5. Rust `mat_loader.rs` 是 WIP

- 文件自注："Full loader is WIP — uses Python h5py as fallback for now."
- 目前只导出 `probe_mat_v73`（列顶层变量名），且 `python/src` 与测试**无任何调用方**；v7.3 读取实际全走 `io_utils/mat_io.py` 的 Python h5py 路径。
- 要么实现并接线，要么删掉以免误导。
- 顺带勘误：`CLAUDE.md` 说 rust_lib "currently a stub" 已过时——现有 8 个导出函数（decimate / step / diurnal / xlsx / mat_loader，共 782 行）；`docs/CODE_GUIDE.md` 的 Rust 小节（只列 4 个导出）同样过时。修文档时一并更正。

### 6. `emtf_rs` 原生后端不存在

- `processor/_mt_common.py:96-106` 用 `importlib.util.find_spec("emtf_rs")` 探测；失败时 `processor/robust_estimate.py:230` 与 `processor/remote_reference.py:368` 的下拉项置灰为 "emtf-rs（未安装）"。
- 仓库里没有这个 crate，该选项当前**永远**回退到 Python `emtf` 包。宣传的第二后端实际缺席。
- 处理建议：实现该 crate，或从 UI 移除该选项避免误导。

### 7. MATLAB TS 转换器没有 Windows 构建路径

- `scripts/build_matlab.sh` 头部注明："Windows 版本需在 Windows 上用 MATLAB Compiler 另行构建"——没有脚本/CI 做这件事，打包产物里也不附带。
- 后果：打包版（尤其 Windows）`MatSaver.save` 永远降级为普通 tsStruct 并打警告（`mat_io.py:406` 附近 "转换为 TS 对象失败（%s），按普通 tsStruct 保存"）。
- `python/src/bin/<platform>/` 目录里只有 `dnff`/`tranmt`，没有 `mtpro_ts_convert`。

### 8. `ui/map_widget.py` 是孤儿

- 254 行 Leaflet / QWebEngineView 地图（marker、底图切换、`__main__` 调试 demo），写完了但**全仓库无人 import**。
- 要么接入主界面（如测站位置展示），要么移入 `python/draft/`。

### 9. `restore_data(ch)` 预留钩子全员空实现

- 抽象于 `core/em_data.py:275`；`mat_data.py:269`、`bin_data.py:77`、`xlsx_data.py:61`、`_demo_utils.py:35` 全是空实现；无任何调用方。
- 与数据库通路（第 2 项）关系不明，实现或清退时应一并决定。

---

## P3 · 构建 / CI / 文档杂项

### 10. CI 从不运行 Python 测试套件

- `.github/workflows/CI.yml` 只构建/上传 `rust_lib` wheel；`build-executables.yml` 只打包。无任何一个 job 跑 `pytest`。

### 11. CI 打包与本地脚本不一致：漏 `emtf`

- `build-executables.yml` 的 Linux（53–60 行）与 Windows（118–125 行）Nuitka 命令都**没有** `--include-package=emtf`，而 `scripts/build_nuitka.sh` / `build_nuitka.ps1` 有。CI 产物的打包面与本地不一致。

### 12. Windows 交叉构建是实验性质

- `scripts/build_nuitka.sh` 头部自注"交叉分支仅供实验"；`build_nuitka_cross.sh` 依赖 Wine + 手工准备的 Windows CPython 3.13，并硬钉 `scipy==1.17.1`（"否则编出来的 exe 启动即崩"）；`scripts/_cross_finish.sh` 是孤儿构建的手动补偿脚本。

### 13. 文档与杂项

- `python/Nuitka.md` 过时：自称"尚无完整生产打包流水线"，并指向已不存在的 `python/scripts/build_nuitka.sh`（脚本已搬到 `scripts/`）。
- `python/draft/` 与 `python/scripts/` 是**字节级重复**的草稿目录（`pipeline_prototype.py`、小波/pyqtgraph 实验、`mat_generator.py` 等）；`pipeline_prototype.py` 是禁用中 Pipeline 模式的祖先。二者应合并或清退其一。
- `python/old/` 为旧数据层参考代码，去留自行决定。
- `tests/smoke_*.py`（8 个探针：UI/rust/webengine/scipy…）是手工诊断脚本，无构建脚本或 CI 引用。

---

## P4 · 测试覆盖薄弱处

- **环境门槛跳过**（本机之外恒跳过）：`test_read_mat.py` / `test_save_mat.py` / `test_tree_view.py` 硬编码依赖 `/home/transen5/Project/atm_rpc/039BE-20240501-20240515-dt5_struct.mat`（仓库未包含）；`test_ts_convert.py:124` 依赖"东北数据" MAT；依赖 `mtpro_ts_convert` 的用例在未构建时跳过；`test_remove_step_widget.py` 有 7 处 `importorskip("pywt")`。
- **无专项测试的模块**：`io_utils/kp_index.py`（仅间接覆盖）、`emtf/io/fc.py`、`fc_reader.py`、`sp.py`、`binary.py`、`emtf/pipeline/tranmt.py`（仅间接）、`ui/map_widget.py`、`ui/log_console.py`、`ui/theme.py`、`ui/timezone_dialog.py`、`utils/validators.py`、`utils/matplotlib_export.py`、`io_utils/file_ops.py`（仅间接）。
- **处理器覆盖薄弱**：`remove_spike` / `remove_region` 无专项测试文件（只在 `test_data_process_widget_state` 的管线测试里被间接碰到）；`robust_estimate` / `remote_reference` 只有"能打开"级测试 + API 层的 `test_emtf_rs_compat`。
- **遗留残留**：`test_splitter_resize_policy.py` 里 `sys.modules.setdefault("gfz_client", ...)` 桩的模块在 `python/src` 中已不存在（旧 Kp 下载设计的残留），可清。

---

## 附录 · 可选小改进（本会话中提出）

**A. 降采样阈值与 xlsx 导入上限互通 —— 已完成（2026-08-31，无需处理）**

实现：`utils/limits.py`（`MAX_POINTS_PER_CHANNEL` + `suggested_decimate_factor()` + `format_points()`，`xlsx_io.py` re-export 保持旧导入路径兼容）；`ui/dialogs/decimate_dialog.py` 预览标注阈值状态与建议因子；`ui/import_xlsx_wizard.py` 两处超限提示加入"导入后右键「数据降采样」"指引。测试：`test_limits.py`、`test_preprocessing_dialogs.py::test_decimate_dialog_preview_shows_limit_hint`。

**B. 本次会话已完成的修复（无需处理，仅存档）**

`nanmean` 全 NaN 时的 "Mean of empty slice" RuntimeWarning（`np.errstate` 拦不住 `warnings.warn` 路径）：新增 `utils/series.safe_nanmean()`，替换了 `ui/data_view_widget.py`（2 处）、`ui/compare_widget.py`、`ui/data_process.py` 共 4 处调用。

---

## 实施顺序建议（供评审参考）

1. 先评审插件框架文档（第 3 项，§12 开放问题表态）——它是 Pipeline 缺口 A/B 的正解；
2. 插件框架 Phase 1–2 落地后，再决定 Pipeline 模式解禁时机；
3. 数据库通路（第 2 项）与 `restore_data` 钩子（第 9 项）绑在一起决策：实现或清退；
4. P2 其余各项与 P3/P4 相互独立，可随手领取。
