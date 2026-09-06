# Rolling Median 示例插件

目录本身就是可扫描插件：在 MTPro 的“工具 → 插件管理器”中添加
`docs/example_plugin` 的父目录，或把本目录复制到默认插件目录
`~/.config/mtpro/plugins/`。也可以通过 `pip install docs/example_plugin` 安装，
此时由 `mtpro.plugins` entry point 发现。

`process()` 不接触 Qt，并且必须返回与输入同长度、同 DatetimeIndex 的
`TransformResult`。本例没有 `create_editor()`，所以宿主根据 `ParamSpec`
自动生成参数对话框。
