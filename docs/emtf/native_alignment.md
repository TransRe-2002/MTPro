# emtf-rs 与原版 EMTF 的处理链对齐

本次完成的是 **DNFF 前处理和 FC 选带对齐**，并在频带内直接进行稳健回归。
不是浮点逐位复现；稳健误差估计和有效自由度仍未对齐。原有单层 FFT API 保留。

## 已接入 MTPro 的行为

- 两后端使用同一份 `(降采样层, FC 起点, FC 终点)` 频带表。
- Rust 使用原版中心在前的对称 FIR 系数逐层降采样，匹配缺失值处理。
- 每个短窗使用 `pwset` 长度 3 的 AR 预白化（AR(2)），原版 pi-prolate 窗函数，随后修正 AR 和降采样滤波响应。
- 复现零时钟条件下的首窗定位及相位补偿；这部分已逐层与实际 DNFF 输出核对。
- 一个频带内的 FC × 时间窗合并后求解阻抗，不再先求每个频点的 Z 再平均 Z。
- 中心周期为 `2 * dt_level * nwin / (fc_start + fc_end)`，与 `bset.f` 相同。
- 少于 3 个 FC 观测时终止后续频带，与 `tranmt.f` 的处理顺序相同。
- 界面向两个后端展示相同的 FFT 短窗参数；隐藏不参与兼容路径的旧长窗步长/迭代参数。
- 原版兼容路径固定 Huber=1.5、最大迭代=50、容差=0.002；结果快照记录处理方案和实际设置。

原生接口：`emtf_rs.estimate_emtf_bands(...)`；逐层 FC 诊断接口：`emtf_rs.emtf_spectra(...)`。
MTPro 适配层位于 `python/src/base/emtf_native.py`。它使用 emtf 的配置/结果数据类，但不调用 Fortran。
目前支持同时起始、零时钟偏移、整数样点重叠、固定 AR 滤波长度 3；非零偏移会明确拒绝。

另外修复了 Python FC 读取器的记录偏移：每组 FC 的头之后还有一个完整缩放记录，且缩放记录不含前置 set 编号。
该问题影响 Python 读取 FC 的诊断，不影响原版 tranmt 自己读取 FC。

## 039BE 实测

输入为 `039BE-20240501-20240515-dt5_struct.mat` 的完整 259201 样点，
Ex1/Ey1/Bx/By/Bz，dt=5 s。远参考使用 031BE 的逐样点对齐重叠区间。
这次使用原始 MAT 通道和应用公共预处理，不代表截图中可能已经编辑过的通道。

设置：目标周期窗=2048 样点，FFT 短窗=128，频带边界数=40。
两种模式、两个后端均输出 35 个频带，周期范围约 **22.45614–9929.69697 s**。
相对周期差最大为 `9.38e-8`，来自原版结果文本/浮点精度。

| 比较指标（相对误差中位数） | 单站 SS | 远参考 RR |
| --- | ---: | ---: |
| 复阻抗 Zxy | 0.115% | 0.549% |
| 复阻抗 Zyx | 0.129% | 0.596% |
| Zxy 标准误 | 27.8% | 66.2% |
| Zyx 标准误 | 25.7% | 59.7% |

单站最短周期 Zxy/Zyx 仍分别约有 8.86%/4.34% 差异；远参考最大复阻抗差异约为 6.10%/15.11%。
不能用中位数掩盖这些边缘频带差异。标准误差异尤其说明：**误差棒尚不能视作原版等价结果**。
Rust 的观测数表示进入回归的 FC 数，原版结果中的 ndf 已受稳健剔除影响；当前这两项不能直接等同。

### 精度因素的独立验证

原版高层路径写 ASC 时会将 H 取整到 0.01 nT、E 取整到 0.001 mV/km，
随后使用单精度运算和有损 FC 打包。原生路径保留双精度输入，不主动引入这些精度损失。
对 039BE 人为施加同样的 ASC 取整后，单站最大复阻抗差异从 8.86%/4.34% 降至 2.24%/1.55%。
这表明取整解释了部分差异，不能据此断言剩余差异都来自舍入。

生成的随机量化输入，在 64/128/256 点 FFT、4 个降采样层（包括首窗）及带缺失值情况下，
原生 FC 与 DNFF 解包 FC 的归一化误差均通过 `<0.002` 的回归阈值。

### 单次运行耗时（非严谨性能基准）

本次单站：原生约 0.077 s，原版约 0.848 s；远参考：原生约 0.155 s，原版约 1.127 s。
原版耗时包含临时文件与子进程开销，不应直接解释为相同求解器的纯计算加速比。

## 重现与后续

验证记录：Rust 全特性测试 **136 通过**；仅启用 ss 的编译检查通过；格式检查通过。
Python 最后一次全量回归 **465 通过、3 跳过**，随后新增的 2 个 FC 读取格式测试也通过。
MATLAB Runtime 相关测试需要放开本地端口初始化权限，已在授权后通过。
一次离屏全量回归出现过已有 pyqtgraph InfiniteLine 删除时序崩溃，单项及最后一次全量重跑均通过；本次未修改该控件。
Clippy 严格检查尚未通过：全目标检查遇到开发依赖 simba 构建错误，库检查还存在既有 dead-code / too-many-arguments 诊断。
新增窗函数常数的精度提示已单独标注保留原版字面值，没有全局屏蔽检查。

在仓库根目录，用项目虚拟环境运行：

```bash
PYTHONPATH=python/src python/.venv/bin/python python/scripts/compare_emtf_alignment.py LOCAL.mat --output outputs/alignment_ss
PYTHONPATH=python/src python/.venv/bin/python python/scripts/compare_emtf_alignment.py LOCAL.mat --remote REMOTE.mat --output outputs/alignment_rr
QT_QPA_PLATFORM=offscreen PYTHONPATH=python/src python/.venv/bin/pytest python/tests/test_emtf_native_alignment.py
```

对比脚本目前面向带 Ex1/Ey1/Bx/By/Bz 的 legacy tsStruct；生成 JSON 明细及曲线 PNG，不修改输入数据。
实测产物位于 `outputs/emtf_alignment_039be_ss/` 和 `outputs/emtf_alignment_039be_rr/`。

Rust 构建时显式指定 MTPro 环境，避免 maturin 自动选择 emtf-rs 目录内的旧虚拟环境：

```bash
VIRTUAL_ENV=/path/to/atm_rpc/python/.venv /path/to/atm_rpc/python/.venv/bin/maturin develop --release --features python
```

下一阶段应固定同一批 FC，分别核对：最终 redescending 保留集合、psi 导数修正、协方差/自由度、远参考权重。
在这些统计量对齐之前，本实现称为“处理链兼容”，不宣称原版完整数值复刻。
已打开的 MTPro 需重启后重新计算；旧快照不会自动变为新算法结果。
