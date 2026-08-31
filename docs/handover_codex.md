# 交接文档（给 Codex）

> 日期：2026-08-31 · 上游会话：Claude Code（MTPro 项目）
> 你接手前的状态、已完成事项、剩余工单与防坑约定都在这里。
> 项目背景与完整命令见 `CLAUDE.md` / `AGENTS.md`，本文不重复，只写接手必需的信息。

---

## 1. 环境速览（错了什么都跑不了）

- 一切 Python / pip / pytest 命令必须用项目 venv：`python/.venv/bin/python`（Anaconda Python 3.13）。
- 运行应用与测试必须 `PYTHONPATH=python/src`（源码是顶层包式导入）。
- 测试套件通过 `pytest.ini` 的 `--forked` **逐用例进程隔离**（Qt/pyqtgraph 包装体跨用例残留会触发 GC 竞态崩溃）——不要为了"提速"移除。
- Qt 相关测试在用例内自行 `os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")`，无头可跑。
- 无 linter / typecheck / formatter；匹配周围风格即可，注释与 docstring 习惯用中文。
- 基线：全量测试 **308 passed, 3 skipped**（3 个跳过是本机 `.mat` 测量数据缺失的环境门槛跳过，属预期）。

## 2. 工作树状态（未提交，勿丢弃）

以下改动在上游会话中完成并全量测试通过，但**尚未提交**（用户未要求提交）：

**改动一：nanmean 空切片警告修复**
- `python/src/utils/series.py` — 新增 `safe_nanmean()`
- `python/src/ui/data_view_widget.py`（2 处）、`python/src/ui/compare_widget.py`、`python/src/ui/data_process.py` — 替换裸 `np.nanmean` + `np.errstate` 模式

**改动二：降采样阈值与 xlsx 导入上限互通**
- 新增 `python/src/utils/limits.py`、`python/tests/test_limits.py`
- `python/src/io_utils/xlsx_io.py`、`python/src/ui/dialogs/decimate_dialog.py`、`python/src/ui/import_xlsx_wizard.py`、`python/tests/test_preprocessing_dialogs.py`

**新增文档**：`docs/unfinished_work.md`（剩余工单，见 §4）、`docs/plugin_api_design.md`（插件框架设计，**待用户评审**）。

## 3. 已完成事项（存档，不要重复做）

1. **"Mean of empty slice" RuntimeWarning**：新版 numpy 中 `nanmean` 全 NaN 时经 `warnings.warn` 抛警告，`np.errstate` 拦不住。修法：`utils/series.safe_nanmean()` 先判空再调用，空输入返回 `nan`，由调用方既有的 `np.isfinite(mean)` 兜底。4 处调用点已替换，行为零变化。`processor/` 里其余裸 `np.nanmean` 跑在已插值数据上，无需动。
2. **降采样 ↔ xlsx 上限互通**：`MAX_POINTS_PER_CHANNEL`（500 万）提升到 `utils/limits.py`，`xlsx_io.py` re-export 保持旧导入路径；降采样对话框预览标注阈值状态 + 建议因子；xlsx 向导两处超限警告加了"导入后右键「数据降采样」"指引。注意建议因子公式是 `ceil(npts / max_points)`（由 `(n-1)//f+1 ≤ m ⟺ n ≤ f·m` 反解）——上游第一版用 `ceil((n-1)/(m-1))` 被随机最小性测试抓出不是最小因子，别退回错误公式。

## 4. 剩余工作：以 `docs/unfinished_work.md` 为唯一工单

全量扫描产物，含文件/行号定位、验证方法、实施顺序建议。接手前先通读。要点：

- **P0**：① Pipeline 处理模式（已建成但 `PIPELINE_ENABLED = False` 屏蔽；缺 param block 执行引擎与 interactive block 确定性重放）；② 数据库通路（`EMDBData` 当前连实例化都做不到）。
- **P1**：插件框架（`docs/plugin_api_design.md` 5 阶段，零代码，**状态：待评审**）。
- **P2–P4**：BinSaver.save 未实现、Rust `mat_loader` WIP、`emtf_rs` 后端缺席、MATLAB 转换器无 Windows 构建、孤儿 `map_widget.py`、CI 不跑 Python 测试、测试覆盖薄弱处等零散项。

**两条硬依赖，开工前必读**：

1. 插件框架是 Pipeline 两个缺口的**正解**（插件 `process()` = 执行引擎，"编辑器产出参数" = 确定性重放）。**除非用户明确放弃插件化路线，不要给 Pipeline 单独实现执行引擎**，避免重复建设。
2. `docs/plugin_api_design.md` §12 有 4 个开放问题等用户表态；**未评审通过前不要开工实施插件框架**。

## 5. 防坑约定（代码库里不显眼但踩了会疼的）

- **线程**：`utils/concurrency.py` 双池（compute / io）。禁止在池线程里阻塞等待本池 future（跨池可以）；池线程禁止碰 Qt；取消是协作式的。
- **`utils/qt_bridge.run()`**：`on_done` / `on_error` 必须传 QObject 绑定方法；传闭包会退化成 DirectConnection，在池线程里执行、碰控件即崩。
- **数据流**：`xarray.Dataset` 是 canonical 存储，`pd.Series`（DatetimeIndex）是交换格式；`DataManager` 的 key `0` 表示"无激活数据"，永远别拿来当数据集键。
- **降采样三套机制别混淆**：显示用 `utils/decimate.py::LODPyramid`（不改数据）、处理器显示用 `processor/_display_utils.py::decimate_for_display`、真信号抽取是 `utils/signal_decimate.py`（破坏性，抗混叠 + NaN 保持）。`CLAUDE.md` 说 "rust_lib currently a stub" 已过时（实际 8 个导出函数），引用时以代码为准。
- **可选依赖一律优雅降级**：`pywt`、`emtf_rs`、`mtpro_ts_convert` 可执行文件、Rust 扩展（`_HAS_RUST*` 模式）缺失时都有 fallback 路径——改动时保持降级行为，测试缺位时跳过而非失败。
- **时区**：时间戳一律 `Asia/Shanghai` 时区、timezone-aware `pd.Timestamp`。
- **MAT 格式**：嵌套 `(1,1)` cell 的 `tsStruct`，schema 见 `io_utils/mat_io.py::TS_STRUCT_TEMPLATE`；保存优先产出 TS 对象文件，转换器缺席时降级普通 tsStruct——这是有意行为，不是 bug。

## 6. 验证命令

```bash
# 全量测试（预期 308 passed, 3 skipped）
PYTHONPATH=python/src python/.venv/bin/pytest python/tests

# 单文件 / 单用例
PYTHONPATH=python/src python/.venv/bin/pytest python/tests/test_limits.py
PYTHONPATH=python/src python/.venv/bin/pytest python/tests/test_preprocessing_dialogs.py::test_decimate_dialog_preview_shows_limit_hint

# 启动应用
PYTHONPATH=python/src python/.venv/bin/python python/src/main.py
```
