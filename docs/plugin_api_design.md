# MTPro 数据处理插件 API 设计文档

> 状态：v1 已实施（2026-09-01）
> 范围：数据处理工具的插件化框架（VST 式宿主 + 插件模型），以及 7 个变换类内置工具的迁移方案。
> 读者：评审人（用户）与后续实施 AI。实施前请先通读"现状盘点"与"决议记录"两节。

---

## 1. 背景与目标

### 1.1 目标

- 定义一套稳定的**数据处理插件 API**：第三方（及内置）处理工具以插件形式注册进宿主，出现在工具面板、classic MDI 工作区与 pipeline 模式里。
- 把 7 个"变换类"内置工具全部迁移为**一方插件**，使插件 API 得到持续的内部使用。
- 借此把"交互决策序列化为参数"落地，使 pipeline block 获得**确定性重放**能力（当前 interactive block 只缓存编辑器产出的 ndarray，上游一变就 stale，必须人工重开编辑器）。
- 补上 pipeline 缺失的**执行引擎**：当前代码自己承认 "param block 目前只保存参数，还不参与实际计算"（见 `data_process_pipeline.py::_show_preview_summary`）。插件的 `process()` 就是这个执行引擎。

### 1.2 非目标（v1 明确不做）

| 项 | 理由 |
|---|---|
| `robust_estimate` / `remote_reference` 两个 **output 类**工具插件化 | 它们是多通道输入、产出新物理量（阻抗张量等），与单通道变换不是同一种形状。等插件框架站稳后另立 `OutputResult` 设计（v2），避免为凑数把 API 撑大 |
| 跨进程 / 沙箱插件 | Python 进程内插件做不了真隔离，DAW 的 Python 插件也都是进程内。加载失败隔离（见 §5.4）即可 |
| 插件热加载 / 卸载 | 启动时扫描一次；插件管理器提供"重新扫描"需重启提示即可 |
| 改动 `EMData` / `DataManager` / 数据层 | 插件只消费 `pd.Series` 切片，不接触数据层 |
| Rust 扩展的插件化 | `rust_lib` 是宿主内部实现细节；插件是 Python 接口，内部可以自由调用 `rust_lib`（一方插件的小波/日变算法就这么干） |

### 1.3 VST 类比

| VST 概念 | MTPro 对应 |
|---|---|
| 插件包（.vst3 bundle） | Python 包 + `plugin.toml` 清单 |
| 工厂 / 类信息 | `PluginInfo`（插件类的 `info` 属性） |
| Processor（引擎，跑在音频线程） | `process(block, params, ctx) -> TransformResult`，纯函数、无 Qt，跑在 compute 池 |
| Editor（参数界面） | `create_editor(host, block, params) -> QWidget`，可选；职责是**产出参数** |
| Parameter 模型 | `ParamSpec` 声明 + `params: dict`（JSON 可序列化） |
| processBlock(buffer) | 处理一段通道切片 |
| API 版本协商 | `PluginInfo.api_version` vs 宿主 `API_VERSION` |

---

## 2. 现状盘点（实施的事实基础）

以下均为代码核验过的现状，实施时不要凭印象推翻。

### 2.1 工具注册表

- `python/src/ui/tools/registry.py`：`ToolSpec` frozen dataclass（`tool_id / display_name / category / kind / requires_channel / aliases / fields / has_channel_sync / has_region_sync / classic_factory / pipeline_editor_factory`），全局 `_REGISTRY` dict，`register()` / `get()` / `all_specs()` / `create_classic_widget()` / `create_pipeline_editor()`。
- `python/src/ui/tools/_registrations.py`：手工清单，硬编码 import 全部 9 个工具并逐个 `register()`。`kind` 二值：`"interactive"`（带编辑器 widget）/ `"param"`（`fields` 元组 → 自动参数对话框）。
- 9 个工具 = 7 个变换类（remove_region、remove_spike、remove_step_diff、remove_step_window、remove_step_baseline、remove_wavelet_denoise、remove_diurnal_sinusoid）+ 2 个 output 类（robust_estimate、remote_reference，`requires_channel=False` 的 station 工具，**本期不动**）。

### 2.2 处理器基类与例外

- `python/src/processor/_base_processor.py::BaseProcessorWidget`：两段式初始化（构造搭 UI → `_on_data_ready` 建图）、NaN 记录与还原、历史栈 undo / revert、`_run_heavy()`（compute 池 + 模态进度条）、`_confirm_and_close()` 发 `result_submitted(data_id, ch, ndarray)` 与 `result_signal(ch, ndarray)`。
- **注意**：`RemoveSpike`（`processor/remove_spike.py`）与 `RemoveRegion` **不继承** `BaseProcessorWidget`，是独立 QWidget，自带同款信号集（`data_requested / result_submitted / result_signal / channel_sync_requested` 等）与 `wire_data_ready()` / `load_series()`。迁移时以"满足接线契约"为准，不以继承关系为准。

### 2.3 Classic 模式接线契约（data_process.py 实测）

宿主对工具 widget 的要求（第 298–332 行附近）：

1. `tool_registry.create_classic_widget(key, data_id, channel_name)` 构造；
2. 若 `spec.has_channel_sync`，连 `widget.channel_sync_requested → broker.apply_result_all_channels`；
3. 若 `spec.has_region_sync`，连 `widget.region_sync_requested → self._on_region_sync`；
4. `widget.data_requested.connect(broker.request_data)`、`widget.result_submitted.connect(broker.apply_result)`；
5. `widget.wire_data_ready(broker.data_ready)`（broker 的 `data_ready(int, str, pkg)` 信号）；
6. `widget.data_requested.emit(data_id, channel_name)` 启动异步送数。

`pkg` 字典字段见 `build_pkg_from_series()`：`x / y / channel_names / all_y / npts / start_time / dt / datetime_index / kp_data`。

### 2.4 Pipeline 模式现状（data_process_pipeline.py + core/pipeline_model.py）

- `PipelineBlock`：`tool_id / params: dict / start / end / input_slice / output_slice / status / last_summary`。**所有 block（interactive 与 param 统一）都走"切片缓存"模型**：`_apply_block_preview()` 只在 `current_snapshot[start:end]` 与 `input_slice` 逐点相等（`equal_nan=True`）时把 `output_slice` 拼回，否则置 `stale`。
- interactive block：双击 → `_open_interactive_editor()` 组出"截止本 block 之前"的输入快照切片 → `create_pipeline_editor()` 造编辑器 → 编辑器发 `result_signal(ndarray)` → `_on_interactive_result()` 校验长度后写缓存。**结果由编辑器直接算出，`params` 不是完整状态，无法重放。**
- param block：双击 → `PipelineParamDialog`（按 `TOOL_META[tool_id]["fields"]` 生成表单）→ 存 `params`、状态 `configured`。**没有任何代码执行它。**
- 编辑器同步能力：`channel_sync_requested(data_id, ch, result)` → 按 NaN 时间戳同步同数据集其他通道；`region_sync_requested(params)` → `core/region_sync.py` 跨数据集按时间区间置 NaN。
- 提交（commit）后 `_reset_pipeline_after_commit()` 清缓存、把结果并入 baseline。

### 2.5 可复用的既有拼图

- 纯算法已在 `core/algorithms/`（step / diurnal / wavelet 等），Rust 加速版在 `rust_lib`，Python 侧已做优雅降级（`_HAS_RUST_STEP` 模式）。
- `utils/qt_bridge.py::run()` + `utils/concurrency.py` 双池：`process()` 的天然执行场所。
- 参数对话框 `PipelineParamDialog`：`ParamSpec → 表单` 的自动对话框可以直接演化复用。

---

## 3. 总体设计

### 3.1 核心判断：交互产出参数，process 消费参数

这是全部设计的中枢，来自 VST 的 processor / editor 分离：

- **编辑器不产出数据，只产出参数。** 用户点掉三个尖峰 → 编辑器确认时交出 `{"indices": [123, 456, 789]}`；
- **`process()` 是参数的确定性函数。** `(输入切片, 参数) → 输出切片`，任何时候可重放；
- 推论：`PipelineBlock.params` 就是 block 的完整持久状态；上游变化导致 stale 时，宿主**自动重跑 `process()`** 即可，不再需要"重开编辑器"。

### 3.2 分层

```
第三方插件 / 一方插件（plugins/）
        │ 只依赖
        ▼
  plugin_api（瘦 API：dataclass + Protocol，无 Qt、无宿主内部模块）
        ▲ 宿主侧实现
        │
  ├─ base/plugin_loader   发现 / 加载 / 校验 / 包成 ToolSpec
  ├─ ui/plugin_adapter    EditorHost 实现、编辑器宿主壳、自动参数对话框
  └─ ui/tools/registry    ToolSpec 注册表（保留，成为加载结果）
```

- `plugin_api` **不 import PySide6**（widget 类型标注用 `typing.Any` + TYPE_CHECKING），保证 `process()` 可在无 Qt 环境（池线程、测试、脚本）中 import。
- `BaseProcessorWidget` **不删除、不强迫插件使用**：它降级为"宿主提供的可选编辑器工具层"，一方插件的编辑器可以继续用它的设施（undo、`_run_heavy`、两段式初始化）。API 瘦、工具层肥，各取所需。

### 3.3 参数序列化规则（强约束）

`params` 必须 JSON 可序列化，只允许 `str / int / float / bool / list / dict / None`：

- **点级选择用切片相对索引**（如 spike indices）——block 切片 `[start, end)` 长度固定，相对索引稳定；
- **区间级选择用时间戳**（ISO 字符串，如 `["2024-05-01T03:00:00+08:00", ...]` 成对）——跨数据集区域同步本来就是按时间戳做的（`region_sync` 现状），且对 baseline 漂移稳健；
- 时间统一 `Asia/Shanghai` 时区（项目约定），`pd.Timestamp.isoformat()` 进出。

---

## 4. API 定义 —— `python/src/plugin_api.py`（完整草案）

插件作者唯一需要 import 的模块。**表面积就这些，实施时不要擅自扩大。**

```python
"""MTPro 数据处理插件 API（API version 1）。

插件 = 一个类，带 ``info: PluginInfo`` 类属性、实现 ``process()``，
可选实现 ``create_editor()``。本模块不依赖 Qt。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Protocol

import pandas as pd

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget

API_VERSION = 1

SUPPORTED_CAPABILITIES = ("channel_sync", "region_sync")


# ─── 参数声明 ───────────────────────────────────────────────

@dataclass(frozen=True)
class ParamSpec:
    """单个参数的声明。宿主据此自动生成参数对话框（无编辑器的插件）。

    ``type`` 取值：float / int / bool / str / list。
    - float / int 可带 ``minimum`` / ``maximum``；
    - str 带 ``options`` 时渲染为下拉框，否则文本框；
    - list 仅用于插件自定义编辑器，自动对话框不渲染 list 参数。
    """
    name: str
    label: str
    type: type
    default: Any
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    options: Optional[tuple[str, ...]] = None


# ─── 插件元信息 ─────────────────────────────────────────────

@dataclass(frozen=True)
class PluginInfo:
    """插件静态描述。``plugin_id`` 全局唯一，建议反向域名风格。"""
    plugin_id: str
    display_name: str
    version: str
    api_version: int = API_VERSION
    category: str = "auto"              # "manual"（人主导）/ "auto"（算法主导）
    description: str = ""
    capabilities: tuple[str, ...] = ()  # ⊆ SUPPORTED_CAPABILITIES
    params_schema: tuple[ParamSpec, ...] = ()


# ─── 数据交换类型 ───────────────────────────────────────────

@dataclass(frozen=True)
class ChannelMeta:
    chid: int = -1
    sample_rate: float = 0.0    # Hz；0 = 未知
    units: str = ""


@dataclass(frozen=True)
class ChannelBlock:
    """process() 的唯一数据输入。

    ``series``：DatetimeIndex + float64；NaN = 数据缺口，插件必须容忍，
    且除非有意修改（如删点置 NaN）应当原样保留 NaN 位置。
    """
    channel: str
    series: pd.Series
    meta: ChannelMeta = field(default_factory=ChannelMeta)


@dataclass(frozen=True)
class TransformResult:
    """process() 的返回。``series`` 必须与输入同长同 index。

    ``diagnostics``：诊断量（逐窗 rms、检测到的台阶位置等），
    宿主写入日志/状态摘要，不参与数据流。
    """
    series: pd.Series
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


# ─── 宿主注入的执行上下文 ───────────────────────────────────

class ProcessContext(Protocol):
    def progress(self, fraction: float, message: str = "") -> None:
        """汇报进度，fraction ∈ [0, 1]。宿主据此刷新进度指示。"""
    def is_cancelled(self) -> bool:
        """协作式取消检查。长循环里周期性调用，为 True 时尽早抛出
        ``ProcessCancelled``（或任何异常）终止。"""
    def log(self, message: str) -> None: ...


class ProcessCancelled(Exception):
    """插件检测到取消时抛出的约定异常（抛别的也行，宿主统一按失败处理）。"""


# ─── 插件必须 / 可以实现的东西 ─────────────────────────────

class Processor(Protocol):
    """必须实现。"""
    info: PluginInfo

    def process(
        self,
        block: ChannelBlock,
        params: Mapping[str, Any],
        ctx: ProcessContext,
    ) -> TransformResult:
        """确定性变换。在 compute 池线程执行，禁止触碰 Qt。"""


class EditorHost(Protocol):
    """宿主提供给编辑器的服务（主线程）。"""

    def apply_params(
        self,
        params: Mapping[str, Any],
        broadcast: frozenset[str] = frozenset(),
    ) -> None:
        """确认。宿主随后调 process() 并把结果写回（classic）或
        缓存（pipeline）。``broadcast`` ⊆ info.capabilities：
        "channel_sync" = 结果同步到同数据集其他通道；
        "region_sync" = 按时间区间同步到所有数据集。"""

    def run_heavy(
        self,
        compute_fn: Callable[[], Any],
        apply_fn: Callable[[Any], None],
        description: str = "",
    ) -> None:
        """编辑器内的重计算：compute_fn 进 compute 池（禁 Qt），
        apply_fn 回主线程执行。语义同 BaseProcessorWidget._run_heavy。"""

    def log(self, message: str) -> None: ...


class EditorFactory(Protocol):
    """可选实现。不实现 = 纯参数插件，宿主自动弹参数对话框。"""

    def create_editor(
        self,
        host: EditorHost,
        block: ChannelBlock,
        params: Mapping[str, Any],
    ) -> Optional["QWidget"]:
        """构造交互编辑器。``params`` 是当前已保存参数：
        编辑器 **SHOULD** 从中重建交互状态（管线里重开已确认的 block 时）；
        最低要求是只读展示已保存参数。返回 None 则宿主退回参数对话框。"""


class PluginProcessor(Processor, EditorFactory, Protocol):
    """便捷别名：一个插件类同时是 Processor 和（可选）EditorFactory。
    加载器只强制 Processor 部分。"""
```

### 4.1 插件类约定

- 入口是一个**类**（不是实例、不是工厂函数），宿主无参实例化一次，长期持有。
- `info` 是类属性；`process` / `create_editor` 是实例方法。
- `params` 的默认值集合 = `{spec.name: spec.default for spec in info.params_schema}`；`process()` 收到的 `params` 宿主保证**已用默认值补全**，插件可放心 `params["x"]` 直取。
- `process()` 抛异常 = 处理失败，宿主隔离并上报（§9.3），不得静默吞掉返回原数据。

---

## 5. 打包、发现与加载

### 5.1 插件目录结构（目录扫描式，VST 味）

```
<plugin_root>/
  despiker/                    # 插件文件夹，名随意
    plugin.toml                # 清单（必需）
    despiker_pkg/              # 普通 Python 包
      __init__.py
      ...
```

`plugin.toml`：

```toml
[plugin]
id = "example.despiker"            # 必须与 PluginInfo.plugin_id 一致，加载时校验
name = "Despiker Pro"
version = "1.2.0"
api = 1                            # 声明所需宿主 API 版本
entry = "despiker_pkg:SpikePlugin" # module:attr，attr 是插件类
```

### 5.2 扫描来源与顺序

1. **内置一方插件**（代码内清单，不读 toml）：`python/src/plugins/` 按功能目录组织，`builtin_plugins/__init__.py` 导出 `BUILTIN_PLUGINS: list[type]` 作为集合，最优先加载。
2. **entry points**：`importlib.metadata.entry_points(group="mtpro.plugins")`，value 即 `module:attr`。第三方插件的 `pyproject.toml`：
   ```toml
   [project.entry-points."mtpro.plugins"]
   despiker = "despiker_pkg:SpikePlugin"
   ```
3. **目录扫描**——扫描路径列表，按以下顺序（靠前者优先）：
   1. `$MTPRO_PLUGIN_PATH` 环境变量（`os.pathsep` 分隔多路径）；
   2. 用户在插件管理器里配置的扫描路径（持久化到 `AppState.plugin_paths`，见 §6.3）；
   3. 默认插件目录 `~/.config/mtpro/plugins/`（与 `app_state` 同根，始终扫描、不可移除）。

   每个路径下含 `plugin.toml` 的子目录是一个候选（以及 `.zip` 直扔形态，见 §5.5）；`sys.path` 临时插入该目录后 import 入口。

**不扫描程序安装目录**（明确决议）：Windows `Program Files` 写入需管理员权限，per-user 安装虽可绕开但应用升级会连带清除安装目录内的插件。VST3 标准目录同样定义在操作系统/用户目录而非 DAW 安装目录，是同一个教训。MTPro 的插件目录全部位于用户可写空间。

重复 `plugin_id`：**先到先得**，后来者拒绝加载并记录原因。

> Nuitka 冻结构建注意：entry points 元数据可能缺失，目录扫描是冻结包的主分发路径。加载器两条路都要实现且互不依赖。

### 5.5 Zip 直扔形态（游戏模组式单文件分发）

第三方插件可直接打包为 `<名字>.zip`（zip 根层含 `plugin.toml` + Python 包），扔进任一扫描路径即被发现。加载器识别后缀 `.zip` 的候选：`zipfile` 读清单，import 时把该 zip 路径插入 `sys.path`（Python 原生支持从 zip import 模块）。

限制：zip 形态仅限**纯 Python 插件**——扩展模块（.pyd/.so）无法从 zip 加载，带真实文件系统路径需求的数据文件同理；此类插件必须走目录形态。加载失败时 `LoadFailure.reason` 需说明。

### 5.3 加载器 —— `python/src/base/plugin_loader.py`

```python
@dataclass(frozen=True)
class LoadFailure:
    source: str        # "builtin" | "entry_point:xxx" | "dir:/path/to/folder"
    plugin_id: str     # 未知时为 ""
    reason: str        # 异常 repr / 校验失败原因

@dataclass
class LoadReport:
    loaded: list[LoadedPlugin]
    failed: list[LoadFailure]

class LoadedPlugin:
    plugin: object          # 实例化后的插件对象
    info: PluginInfo
    source: str

def load_all_plugins() -> LoadReport: ...
```

校验清单（任一失败 → 记入 `failed`，跳过，继续加载其余）：

1. toml 可解析、必需字段齐全；`api` 与 `PluginInfo.api_version` 一致且 ≤ 宿主 `API_VERSION`；
2. 入口可 import、attr 是类、可无参实例化；
3. 实例有 `info: PluginInfo`，字段合法（`plugin_id` 非空且与 toml 一致、`capabilities ⊆ SUPPORTED_CAPABILITIES`、`params_schema` 名字不重复）；
4. `process` 可调用；有 `create_editor` 时其可调用。

加载完成后，把每个 `LoadedPlugin` 包成 `ToolSpec` 注册（§6.1）。**加载器永不向调用方抛异常。**

### 5.4 失败隔离

插件管理器对话框（§6.3）展示 `LoadReport`：成功列表（id / 名称 / 版本 / 来源）与失败列表（来源 + 原因）。任何插件加载失败不影响宿主启动，也不影响其他插件。

---

## 6. 宿主侧改造

### 6.1 registry 动态化

- `ui/tools/registry.py` 保留 `ToolSpec` 与全部现有查询函数（宿主内部描述符不动）。新增：

```python
def toolspec_from_plugin(lp: "LoadedPlugin") -> ToolSpec:
    """kind = 'interactive' 若插件实现了 create_editor，否则 'param'；
    fields = 由 params_schema 映射的 (name, label, type, default) 元组
    （供自动参数对话框与 _default_params_for_tool 使用）；
    has_channel_sync / has_region_sync 由 capabilities 映射；
    classic_factory / pipeline_editor_factory 指向 plugin_adapter 的通用工厂。"""
```

- `_registrations.py` 的 9 条静态注册中：7 个变换类随迁移逐条删除（对应一方插件顶上），**2 个 output 类保留原样**；全部迁完后该文件只剩 output 两条。
- 启动时序（`mainwindow.py`）：`load_all_plugins()` → 注册 → 报告交给插件管理器（菜单"工具 → 插件管理器"）。**必须在 `DataProcessWidget` 构造之前完成**（工具面板模型在构造时读取注册表）。

### 6.2 插件适配器 —— `python/src/ui/plugin_adapter.py`

一个文件承担"把插件接进两种模式"的全部胶水：

1. **`PluginEditorHost(QWidget)`**：`EditorHost` 的主线程实现。构造入参 `(plugin, block, mode, ...)`；`apply_params` 的行为按 mode 分叉：
   - classic：调 `process()`（经 compute 池）→ 还原调用方语义 → 发 `result_submitted(data_id, ch, ndarray)`，broadcast 含 `channel_sync` 时另发 `channel_sync_requested`、含 `region_sync` 时发 `region_sync_requested(params)` —— **信号签名与 §2.3 契约逐字一致**，这样 `data_process.py` 的现有接线代码零改动；
   - pipeline：调 `process()` → 写 `block.input_slice / output_slice` 缓存（替代现在 `_on_interactive_result` 信任编辑器 ndarray 的路径）。
2. **`PluginClassicWidget`**：满足 §2.3 全部接线契约的壳。`data_requested →` 等数据就绪后构造 `ChannelBlock` 并调 `plugin.create_editor()`；无编辑器的纯参数插件 → 弹自动参数对话框。
3. **`PluginParamDialog`**：由现 `PipelineParamDialog` 演化，按 `ParamSpec` 渲染（float/int 带 min/max 的 spin、bool 复选、str+options 下拉、list 跳过并提示"需自定义编辑器"）。
4. `create_pipeline_editor()` 路径：工厂用"截止本 block 的输入快照切片"构造 `ChannelBlock` 后调 `plugin.create_editor()`——与现有 `PreviewChannelSlice` 流程对接，`data_process_pipeline.py::_open_interactive_editor` 的对话框壳不变。

### 6.3 插件管理器 —— `python/src/ui/plugin_manager_dialog.py`

DAW 式插件管理（参照 REAPER / FL Studio 的 VST 管理器形态）：

- **已加载 / 加载失败**两个表格（id / 名称 / 版本 / 来源路径；失败附原因）；
- **扫描路径管理**：当前扫描路径列表（默认插件目录为固定项、不可删除），"添加目录…"（文件夹选择器）/ "移除"；改动持久化到 `AppState.plugin_paths`（`utils/app_state.py` 增加 `plugin_paths: list[str]` 字段，沿用现有 TOML 持久化机制）；
- **"从文件安装…"**：选**本地已下载好**的 `.zip` 或插件文件夹 → 复制/解压到默认插件目录 → 重新扫描。纯离线功能，不涉及联网下载；这是"由程序安装插件"的入口，覆盖不想手动移动文件的习惯（联网仓库是另一回事，见 §11 第 7 条）；
- **"重新扫描"**：重跑 `load_all_plugins()`，新插件提示重启生效（v1 不做热插拔）。

不做启用/禁用开关（v1）。

---

## 7. Pipeline 执行引擎与批处理

### 7.1 单 block 执行引擎

插件化后，`_apply_block_preview()` 的"param block 等待执行"分支获得真实语义：

```
block 状态机：
  empty（无 params）
    → configured（params 已设；纯参数插件双击即配，交互式编辑器确认后写入）
    → ready（process() 执行成功，缓存有效）
    → stale（上游/区间变化，缓存失效）
```

- **执行时机**：`_refresh_all_pipeline_statuses()` 遍历到 `configured` 或 `stale` 且 params 完整的 block 时，把 `(plugin, ChannelBlock(输入快照切片), params)` 投递到 compute 池（`utils.qt_bridge.run`），完成后写缓存、置 `ready`、刷新 UI。投递去重（同一 block 在途只允许一个任务），取消语义：上游再变则丢弃过期结果（比对任务携带的输入指纹）。
- **交互式编辑器确认**：不再接收编辑器算出的 ndarray。编辑器 `apply_params(params)` → 宿主存 `block.params` → 走上一条的同一执行路径。`_on_interactive_result` 的长度校验逻辑移到执行引擎对 `TransformResult` 做（同长同 index，否则 block 置 error 摘要）。
- **取消**：`PipelineBlock` 增加 `status = "error"`（`BLOCK_STATUS_STYLE` 加一条），`last_summary` 记异常 repr。
- `_show_preview_summary` 里"param block 目前只保存参数"的文案随之删除。

> 本节是插件化红利的兑现点：stale block 自动复活，不再"重开编辑器"。

### 7.2 批处理：执行引擎的横向扩展

场景：一次导入十几个 MAT 数据集后，要把同一套处理流程自动施加到多个数据集/多个通道——不开任何编辑器。插件化让这成为纯组合问题：**批处理 = 对目标列表循环调用 `process()`**，不引入新的插件类别、不改 API。

**配方（recipe）**：可保存/加载的处理步骤清单，TOML 格式：

```toml
[recipe]
name = "标准去台阶+去噪"

[[steps]]
plugin = "mtpro.remove_step_window"
time_range = []          # 空 = 整段；或 ["2024-05-01T00:00:00+08:00", "2024-05-02T00:00:00+08:00"]

[steps.params]
window = 1000
indices = []             # 空 = 自动检测（见"参数二态约定"）
```

来源两条：手工编辑文件；或从现有 `ChannelPipeline` 一键导出（block 按 layer 顺序、起点顺序展平为 steps——与 `_compose_channel_preview` 语义一致——索引区间按当前通道的时间索引换算为时间戳）。

**参数二态约定**（对 auto 类工具是硬要求）：`process()` 对"结果类参数"（`indices`、`fit_window` 等检测产物）必须支持两种输入：

- **显式值 = 精确重放**——管线里用户确认过的决策，原样复现；
- **空值/默认值 = 自动检测**——用配置类参数（`window`、阈值等）现场检测。

交互编辑器确认时存显式值；批处理配方通常存空值，让每个目标通道各自检测。检测本身也是输入的确定性函数，不破坏可重放性。

**执行语义**：

- 目标 = `(data_id, channel)` 列表，来自批处理对话框的数据树勾选；
- 每个目标一个 compute 池任务；**目标内部 steps 严格串行**（每步消费上一步输出）；
- 时间区间按时间戳匹配到目标通道 index（同 §3.3 区间约定）；**点级索引参数仅在源/目标采样严格对齐时有意义**，不对齐时该步记入诊断并跳过、不算目标失败；
- 失败隔离：单个目标抛异常记入报告，不打断其余目标；
- 结果写回内存数据集（与 `broker.apply_result` 同路径，保持后续手工编辑/撤销的一致性），**批处理不落盘**——导出走现有保存流程。

**批处理对话框**（`python/src/ui/batch_process_dialog.py`）：选配方（当前会话的某个通道管线 / 从文件加载）→ 数据树勾选目标通道 → 运行 → 进度/结果表（目标、状态、`diagnostics` 摘要、失败原因）。

---

## 8. 工具迁移指南

### 8.1 样板：remove_spike（完整推演）

现状（`processor/remove_spike.py`，591 行）：`SpikePlotWidget`（框选/删点/撤销）+ `RemoveSpike` 窗口，删除 = 选中点置 NaN，确认 = 直接发结果 ndarray，另有"应用到所有通道"复选框。

一方插件形态（`plugins/remove_spike/` 与 `builtin_plugins/spike_removal.py`）：

```python
class SpikeRemovalPlugin:
    info = PluginInfo(
        plugin_id="mtpro.remove_spike",
        display_name="remove spike",
        version="1.0.0",
        category="manual",
        capabilities=("channel_sync",),
        params_schema=(
            ParamSpec("indices", "删除点索引", list, default=[]),
        ),
    )

    def process(self, block, params, ctx):
        y = block.series.to_numpy(dtype=np.float64, copy=True)
        idx = [i for i in params["indices"] if 0 <= i < len(y)]
        y[idx] = np.nan
        return TransformResult(
            series=pd.Series(y, index=block.series.index),
            diagnostics={"removed": len(idx)},
        )

    def create_editor(self, host, block, params):
        return SpikeEditor(host, block, params)   # 复用现有 SpikePlotWidget
```

编辑器改造点（现有 `RemoveSpike` 窗口改扮 `SpikeEditor`）：

- 构造时从 `params["indices"]` 恢复：先对输入应用这些 NaN，再把它们画成红色高亮（重建交互状态，满足 §4 EditorFactory 约定）；
- 内部 `undo_stack` / 追加选点照旧，但维护一个累积的 `deleted_indices: set[int]`；
- "确定"按钮：不再 `result_signal.emit(ndarray)`，改为
  `host.apply_params({"indices": sorted(deleted_indices)}, broadcast=frozenset({"channel_sync"} if 复选框勾选 else ()))`；
- classic 模式由 `PluginEditorHost` 把 `apply_params` 翻译成 §2.3 的信号序列（含 `channel_sync_requested`），`data_process.py` 不动。

迁移验收：
1. classic 模式行为与旧版逐操作一致（框选、撤销、还原、通道同步）；
2. pipeline 里确认后 `block.params == {"indices": [...]}`，**改动上游后 stale 自动重跑且结果一致**（这是旧版做不到的，必须有测试）；
3. `params` 可 `json.dumps`。

### 8.2 其余 6 个变换工具的参数草案

迁移每个工具的核心问题只有一个：**确认那一刻，把哪些"决策"写进 params 才能让 process 复现结果？** 草案如下（实施时按工具现状细化）：

| tool_id | params 草案 | process 复用 |
|---|---|---|
| `remove_region` | `{"time_ranges": [[iso, iso], ...]}`（沿用现有 region_sync 的时间戳约定） | 区间置 NaN |
| `remove_step_diff` | `{"indices": [...], ...}`（差分检测出的台阶位置 + 校正方式） | `core/algorithms.step` 差分去阶跃 |
| `remove_step_window` | `{"indices": [...], "window": int}` | 窗口均值去阶跃（Rust 可用） |
| `remove_step_baseline` | `{"indices": [...], "fit_window": [i0, i1], "region": [i0, i1]}`（`RobustBaselineStepResult` 里已有这些字段，正是为序列化准备的） | 鲁棒基线拟合 |
| `remove_wavelet_denoise` | `{"levels": ..., "region": [i0, i1], ...}`（按 `WaveletDenoiseResult` 字段核对） | 小波多尺度去噪 |
| `remove_diurnal_sinusoid` | `{"window": ..., ...}`（按 diurnal 拟合器的输入参数核对） | 日变两步最小二乘（Rust 可用） |

> 实施提醒：这三个 step 工具与日变工具当前是"自动检测 + 人工确认"混合体。迁移后**检测留在编辑器里**（编辑器把检测结果展示给用户、允许修改），**确认后的最终决策**进 params。检测本身若依赖完整上下文，可把检测算法也放进 params 生成路径，但 process 必须只依赖 params。另按 §7.2 参数二态约定：每个 auto 工具的 `process()` 都要实现"显式结果参数 = 重放、空结果参数 = 自动检测"两个分支。

每个工具迁移完成即从 `_registrations.py` 删除对应静态条目；7 个迁完后静态清单只剩 2 个 output 工具。

---

## 9. 契约：线程、错误、版本

### 9.1 线程

- `process()` 在 compute 池线程执行：禁 Qt、禁阻塞等待本池 future（`utils/concurrency.py` 现有规矩）；
- 编辑器在主线程；重计算走 `host.run_heavy()`；
- 插件不得自己起线程/进程池。需要并行用 `numpy` / `rust_lib`（rayon）。

### 9.2 数据

- 输入 `series` 只读语义：`process()` 必须拷贝后再改；
- 输出与输入同长同 index，否则宿主按失败处理；
- NaN 语义见 §4 `ChannelBlock` 注释。

### 9.3 错误

- 加载期：§5.3 校验失败 → 报告，不崩；
- 运行期：`process()` 抛异常 → classic 模式宿主弹日志/消息框；pipeline 模式 block 置 `error`；**宿主捕获范围是 Exception 而非 BaseException**（KeyboardInterrupt 等透传）。

### 9.4 版本策略

- 宿主 v1 只接受 `api_version == 1`；未来新增能力走**只增不改**（新 capability、新可选 ParamSpec 字段），`API_VERSION` 只在破坏性变更时升；
- `params` 向前兼容规则：新增参数必须带默认值；`process()` 应忽略不认识的键（宿主透传整个 dict）。

---

## 10. 实施阶段划分

每阶段独立可验收、可单独交付评审。**所有命令遵守 `AGENTS.md`：venv 前缀 + `PYTHONPATH=python/src`。**

### Phase 1 —— API、加载器、插件管理器

产出文件：
- `python/src/plugin_api.py`（§4 全文）
- `python/src/base/plugin_loader.py`（§5，含多路径目录扫描与 §5.5 zip 直扔）
- `python/src/ui/plugin_manager_dialog.py` + mainwindow 菜单项（§6.3 四件套）
- `python/src/utils/app_state.py`：`AppState` 增加 `plugin_paths: list[str]` 字段（默认空列表 = 仅默认目录）
- 测试：`python/tests/test_plugin_api.py`（ParamSpec/PluginInfo 校验、参数补全）、`python/tests/test_plugin_loader.py`（fixture 插件目录放 `python/tests/fixtures/plugins/`：一个好插件、一个坏入口、一个 api 版本不符、一个 id 冲突 → 断言 `LoadReport` 内容；全程无 Qt）

验收：`PYTHONPATH=python/src pytest python/tests/test_plugin_api.py python/tests/test_plugin_loader.py` 绿；插件管理器能列出 fixture 结果；**现有功能零变化**（此时还没有任何真实插件注册）。

### Phase 2 —— 适配器 + remove_spike 样板 + 执行引擎

产出文件：
- `python/src/ui/plugin_adapter.py`（§6.2）
- `python/src/builtin_plugins/__init__.py` + `spike_removal.py`（§8.1）
- `processor/remove_spike.py` 改造为编辑器（或新文件承接，旧窗口类删除前确认无引用）
- `data_process_pipeline.py`：interactive 确认改走 `apply_params` + 执行引擎（§7）；`_registrations.py` 删除 remove_spike 静态条目
- 测试：参数往返（`process(block, params)` 与旧行为对拍）、pipeline stale 自动重放确定性、`params` JSON 可序列化

验收：`PYTHONPATH=python/src pytest python/tests` 全绿（含既有 forked 套件）；手工走查 classic 框选删点全流程 + pipeline 确认→改上游→自动重放。

### Phase 3 —— 其余 6 个变换工具迁移

按 §8.2 逐个迁移，顺序建议：remove_region（结构简单）→ step×3（共享算法层）→ wavelet → diurnal。每个工具一次提交，附对拍测试。`_registrations.py` 逐条收敛。

验收：7 个工具全部以一方插件身份注册；`_registrations.py` 只剩 output 两条；全量测试绿。

### Phase 4 —— 批处理

产出文件：
- `python/src/core/recipe.py`：配方数据结构 + TOML 存读 + 从 `ChannelPipeline` 导出（含索引→时间戳换算）
- `python/src/base/batch_runner.py`：目标展开、compute 池调度（目标间并行、目标内串行）、失败隔离、进度汇报
- `python/src/ui/batch_process_dialog.py` + 数据处理页入口
- auto 类工具的 `process()` 补齐"空结果参数 → 自动检测"分支（§7.2 参数二态约定）
- 测试：配方存读往返、多目标批处理结果与逐通道手工管线对拍、单目标失败隔离、采样不对齐时点级步骤的跳过行为

验收：同一配方对 ≥2 个数据集批处理，结果与逐通道走管线一致；单目标失败不中断整体；全量测试绿。

### Phase 5 —— 收尾

- `docs/CODE_GUIDE.md` 更新（新包/模块条目、插件开发指引小节）；
- 附一个**独立的第三方示例插件**（放 `docs/` 或 `examples/`，不进内置清单）：~50 行的简单变换（如滑动中值），带 `plugin.toml` 与 pyproject entry point 两种形态，作为插件作者的参考实现；
- `CLAUDE.md` / `AGENTS.md` 补一句插件层的位置。

---

## 11. 决议记录（讨论中已定，实施时不再翻案）

1. **7 个变换类工具全部插件化；2 个 output 类（robust_estimate / remote_reference）v1 不动**——避免为它们撑大 API；
2. **编辑器产出参数、不产出数据**；`process()` 是参数的确定性函数；
3. **`BaseProcessorWidget` 保留为宿主可选工具层**，插件 API 本身保持瘦；
4. **编辑器 SHOULD 从 params 重建交互状态**（`create_editor` 会收到已存参数）；最低要求只读展示；
5. 加载失败隔离、进程内运行、无沙箱、不做热插拔；
6. 同步能力（通道/区域同步）是**宿主能力**，插件通过 `capabilities` 声明 + `apply_params(broadcast=...)` 触发，不自己实现。
7. **插件不装进程序安装目录**：默认目录 `~/.config/mtpro/plugins/`（用户可写）+ 可配置扫描路径列表（DAW 式）+ "从文件安装…"（宿主复制进插件目录，纯离线）。在线插件仓库（Factorio 式）需后端，推迟到 v2——届时只是给加载器加下载来源，架构不变。
8. **批处理基于插件 API 组合实现，不引入新插件类别**：配方 = 步骤清单（plugin_id + params + 时间区间），执行 = 对目标列表循环调 `process()`；auto 工具必须支持参数二态（显式结果参数精确重放 / 空结果参数自动检测）；批处理结果只写回内存数据集，不落盘。

## 12. 已决议开放问题

实施采用文档倾向方案：一方插件位于 `builtin_plugins/`；Pipeline 增加
`error` 状态；点选择使用 block 相对索引、区间使用 ISO 时间戳；v1 支持
zip 直扔。

1. 一方插件位于 `python/src/builtin_plugins/`，与宿主设施分离。
2. 接受 `PipelineBlock.status = "error"`，用于区分执行失败和待刷新。
3. 点级参数使用 block 相对索引，区间参数使用带时区 ISO 时间戳。
4. v1 纳入 zip 直扔形态；仅纯 Python 插件可从 zip 加载。
