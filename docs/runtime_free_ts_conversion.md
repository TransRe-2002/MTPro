# TS 对象转换免 MATLAB Runtime：可行性验证

日期：2026-10-10。

## 结论

可以在不调用 MATLAB、MATLAB Engine 或 MATLAB Runtime 的情况下读取和写入真正的 MATLAB `TS` 类对象 MAT 文件。已验证的路线是 `mat-io 1.0.0` 的 MCOS 序列化实现。它不只是重命名变量或保存普通结构体。

研究原型通过后，已将该路线接入 MTPro；MATLAB 只用作独立的兼容性验证器，不参与运行时转换。

## 已接入路径

- `python/src/io_utils/ts_object_io.py` 实现 TS 属性归一化、v7/v7.3 对象读写与原子保存。
- `MatSaver` 直接写真正的 `TS` 对象；失败时报告错误并保留原文件，不再静默退化为普通结构体。
- `MatLoader` 保留轻量 v7 MCOS 读取器，复杂对象和 v7.3 使用 mat-io。双向转换接口不再运行外部程序。
- `python/requirements.txt` 固定 `mat-io==1.0.0`；本地及 CI 的 Nuitka 配置显式包含 `matio`。
- `python -m io_utils.ts_convert [ts2struct|struct2ts] input.mat output.mat` 提供免 Runtime 命令行转换。

普通 v7 结构体输入由 SciPy 读取：实测 mat-io 1.0.0 对 SciPy 写出的某些 UTF-8 中文字符数组读取失败，保留 SciPy 路径可避免该问题。

接入验证：TS 相关 20 个 Python 回归测试通过；命令行双向转换通过。正式写入适配器生成的 v7/v7.3 文件已由 MATLAB R2025b 独立验证 TS/handle 身份、中文台站名、嵌套 metadata、NaN、日期及派生结束时间，getTime 与原生构造对象一致，setTimeStamp/demean 和再次保存通过。Windows 安装包仍需在 Windows 上重新构建验证。

## 实验与结果

环境：项目 `python/.venv`，CPython 3.14，现有 NumPy/SciPy/h5py；`mat-io 1.0.0` 安装到隔离目录；MATLAB R2025b 25.2.0.2998904 用于验证。

输入样本：`python/tests/fixtures/ts_object.mat`、`ts_object_nondefault.mat`，以及普通结构体文件 `python/tests/test1.mat`。

| 实验 | Python 检查 | MATLAB 独立检查 |
| --- | --- | --- |
| 默认属性 TS 修改后保存为 v7 | 通过 | 通过 |
| 默认属性 TS 修改后保存为 v7.3 | 通过 | 通过 |
| 非默认属性 TS 修改后保存为 v7 | 通过 | 通过 |
| 非默认属性 TS 修改后保存为 v7.3 | 通过 | 通过 |
| 普通 tsStruct 新建 TS 对象并保存为 v7 | 写入成功，现有 MCOS 读取器可读 | 通过 |

前四项把数据截为 7×137，修改首个值为 12345.25，并更新 npts。MATLAB 比较了全部 TS 属性及数据，包括 NaN 和非默认 missVals、timeStampType、elevation、badRec。

五项均通过 `isa(obj,'TS')` 和 `isa(obj,'handle')`，可执行 `getTime`、`setTimeStamp`、`demean`，且可由 MATLAB 再保存为 v7。新建对象的属性与输入结构体逐字段比较通过。

原型及生成文件保留在 `/tmp/mtpro-matio-research.H1PvSx/`：`probe_ts.py` 生成文件，`verify_ts.m` 完成 MATLAB 验证。临时目录不是正式分发路径。

## 技术路线

写入使用 `matio.utils.MatlabOpaque(properties=property_map, classname='TS')`，再调用 `matio.save_to_mat(..., version='v7')` 或 `version='v7.3'`。库负责对象引用、MCOS 子系统与属性表，不依赖 MATLAB 库。

从结构体构造属性时，需保持旧转换器的语义：将 NCh、npts、dt 和数值日期字段转为 double；处理默认字段；需要时重算 endTime/timeStamp。原型验证了日期类型要求：整数 datevec 会让 MATLAB 的 datenum 方法报错，转为 float64 后通过。

当前 `mcos_reader.py` 能读回三种新写出的 v7 样本。v7.3 对象读取应使用新的库路径，不能假定当前读取器已有支持。

## 打包与兼容性限制

- `mat-io 1.0.0` 在本机 CPython 3.14 源码构建成功，含 Cython 扩展；依赖为 NumPy、SciPy、pandas、h5py。检查的扩展没有 MATLAB 动态库依赖。
- 本次 PyPI 查询没有找到 1.0.0 的 Windows CPython 3.14 wheel；Windows 需要源码构建验证，并让 Nuitka 包含 `matio` 与其扩展模块。
- 0.4.2 有通用 wheel，但在当前 SciPy 组合下写入失败：`no field of name _Class`，且不实现 v7.3 写入。不能直接作为免编译替代。
- 仅验证了当前 TS 类和仓库样本。包含嵌套对象、自定义 saveobj/loadobj、不同 MATLAB 版本或大规模数据的文件仍需专项测试。
- 在 MATLAB 中使用 TS 方法仍需要对应 `TS.m` 类定义；这与转换器部署时不需要 Runtime 是不同问题。

## 接入建议

先新增 `python/src/io_utils/` 下的 TS 对象写入适配器，沿用已有字段归一化，再接入 `MatSaver` 和双向转换接口。保留明确的格式错误，避免将普通结构体宣称为 TS 对象。完成 Windows wheel、Nuitka 和无 MATLAB 环境测试后，再移除外部转换器依赖。

## 来源

- [mat-io 源码、对象读写接口与 BSD-3-Clause 许可](https://github.com/foreverallama/matio)
- [MATLAB 对象保存和加载机制](https://www.mathworks.com/help/matlab/matlab_oop/understanding-the-save-and-load-process.html)

MATLAB Coder 仍不能直接替换当前动态 load/save 对象路径；本方案通过第三方文件格式实现绕过这个限制。
