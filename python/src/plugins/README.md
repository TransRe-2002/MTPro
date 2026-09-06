# MTPro 功能插件包

每个目录对应一个可注册的功能插件。插件入口和编辑器放在同一个功能包内。

典型结构：

~~~
remove_step_diff/
├── __init__.py       # 对外导出插件类和编辑器类
├── plugin.py         # PluginInfo + process()
└── editor.py         # 可选的 Qt 交互编辑器
~~~

纯算法优先放在 plugin.py 或功能包内的 core/，process() 不应触碰 Qt。

- `remove_region`
- `remove_spike`
- `auto_spike_filter`
- `emd_denoise`：Rust EMD 内核；IMF 从高频到低频编号，可选择性重构
- `detrend`
- `median_filter`
- `morphological_filter`
- `iir_filter`：低/高/带通/带阻/陷波；频率参数为奈奎斯特频率比例
- `svd_denoise`：Hankel 轨迹矩阵的低秩 SVD 重构
- `remove_secular_variation`：复现 `TS.removeSV` 的局部均值、平滑锚点和样条背景相减
- `remove_step_diff`
- `remove_step_window`
- `remove_step_baseline`
- `remove_wavelet_denoise`
- `remove_diurnal_sinusoid`
- `vmd_denoise`：Rust VMD 内核；按连续有效段分解，并按用户选择的模态重构

阻抗估计和远参考法暂时位于独立的 `output_tools/`，不纳入本轮插件化，
后续可以按多通道 `OutputResult` 方案单独拆分。

宿主只从 `plugins.*` 导入功能入口。旧的 Qt 编辑器代码已经按功能移动到各自
插件包的 `editor.py`，公共基类和算法位于 `plugins.common`；`processor` 仅保留
兼容转发，便于已有工程和第三方代码平滑迁移；
新的插件代码应实现 `plugin_api` 中的 `PluginInfo` 与 `process()`，不要再依赖
`processor` 命名空间。

## 交互式 Pipeline 工具

交互工具由 Pipeline 注入当前 block 的 `ChannelBlock` 切片：它已经包含上游步骤
的结果、当前通道和时间索引。编辑器不得自行选择数据集、请求完整通道或直接写回
`DataManager`；确认后只返回同长度的切片结果和可回放参数。所有可视化工具只实现
这条 Pipeline 切片路径。
