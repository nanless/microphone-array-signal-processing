# 按章查找教学实验

这里是单章实验的**真实实现**。共享的 STFT、阵列、AEC、WPE、追踪等数值核仍在 [`array_tutorial/`](../array_tutorial/)；跨章练习、音频生成器和固定上游源码探针仍在 [`examples/`](../examples/)。这样既能按章节学习，也不会把一份算法复制成多份。

| 目录 | 主要内容 |
|---|---|
| [ch01](ch01/README.md) | 阵列有效性的基础模型与边界 |
| [ch02](ch02/README.md) | 时延、声学模型和数字口径 |
| [ch03](ch03/README.md) | 几何、协同阵与校准 |
| [ch04](ch04/README.md) | 定位及双源分辨率实验 |
| [ch05](ch05/README.md) | 波束约束、状态和同输入比较 |
| [ch06](ch06/README.md) | AEC 状态、双讲与算法对照 |
| [ch07](ch07/README.md) | WPE 时间和预测边界 |
| [ch08](ch08/README.md) | 分离模型与活动错误 |
| [ch09](ch09/README.md) | 追踪、交叉与缺测 |
| [ch10](ch10/README.md) | 实时、数值、谱减和时钟工程 |
| [ch11](ch11/README.md) | 约束、评分与方案选择 |
| [appendix_a](appendix_a/README.md) | 数学、FFT 和统计边界 |
| [appendix_b](appendix_b/README.md) | 房间结果、相位与来源证据题 |

从仓库根目录使用模块形式运行，例如：

```bash
.venv/bin/python -m codes.chapters.ch04.chapter04_experiments
.venv/bin/python -m codes.chapters.appendix_b.appendix_b_experiments
```

原来的 `codes.examples.<同名模块>` 导入、`-m` 命令及 `codes/examples/<同名文件>.py` 直接脚本入口继续转交同一实现。兼容文件不应新增算法逻辑；修改题目时改本目录里的真实源文件，再核对旧入口与新入口的输出、全书 228 个稳定练习 ID、对应章节和覆盖表。

房间、主音频、GSS、移动和追踪资产的清单，以及某些工业报告，会校验生成器的**路径和完整源码摘要**。这些生成器和需要同目录 C/C++ 文件的探针目前保留在 `examples/`；不要为目录整齐而只改清单路径。若以后迁移，应先生成到新目录，核对参数、逐文件摘要和报告，再同步构建与测试。

仍在 `examples/` 的单章文件按学习章查找：第 4 章有 MDL 重复试验、doatools/SBL/SMP-PHAT 原实现探针；第 5 章有波束上游与 SOF 设计审查；第 6 章有 AEC3、真实配对录音、同输入接口和上游审查；第 7 章有 WPE 外部对照、静音边界与上游审查；第 8 章有 GSS 音频生成、AuxIVA、Stream.FM/TF-Locoformer 和分离上游审查；第 9 章有追踪与移动音频生成、上游审查；第 10 章有工程综合基线、真实录音准备、工业接口与原生 C/C++ 探针；第 11 章有会议评分接口审查；附录 B 有房间仿真生成器。这些脚本的精确文件名与原始验证范围见[算法覆盖表](../COVERAGE.md)和[研究手册](../research/README.md)。

`ch02_05_baselines.py`、`ch06_09_baselines.py`、三个 `exercises_*` 及空间模型、增强步骤、工程边界、时间状态等跨章练习继续留在 `examples/`；其稳定 ID 由[练习目录](../research/05_exercises_and_audio.md)逐题映射。后续拆分时须保持一个 ID 只有一个真实实现，并同步目录测试，不能复制一份后让两个实现各自漂移。
