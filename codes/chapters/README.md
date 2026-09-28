# 按章查找教学实验

这里是全书教学代码与资产的唯一目录。每章的 `core/` 放算法数值核，`examples/` 放对应实验或外部接口探针；同一个数值核只归一个章节，其他章节直接导入。导读 `ch00/` 保存跨章练习、主音频总清单、全书索引和上游获取工具。

| 目录 | 主要内容 |
|---|---|
| [ch00](ch00/research/README.md) | 全书索引、研究手册、外部源码清单和复现记录 |
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

旧的 `codes.examples.*` 和 `codes.array_tutorial.*` 导入路径已经退出仓内接口。运行时使用上表中的章节模块；[旧新路径映射](../../scripts/code_layout_map.json)帮助定位原文件。修改题目时改唯一真实源文件，再核对全书 228 个稳定练习 ID、对应章节和覆盖表。

房间、主音频、GSS、移动和追踪资产的清单，以及某些工业报告，会校验生成器的**路径和完整源码摘要**。生成器已归入相应章节的 `examples/`；原生 C/C++ 探针与调用它的 Python 文件放在同一目录。更改这些文件后，应从真实新源重生资产并核对参数、逐文件摘要和报告，再同步构建与测试。

各章 `examples/` 的文件按学习章查找：第 2 章有真实录音准备；第 4 章有 MDL 重复试验、doatools/SBL/SMP-PHAT 原实现探针；第 5 章有波束上游与 SOF 设计审查；第 6 章有 AEC3、真实配对录音、同输入接口和上游审查；第 7 章有 WPE 外部对照、静音边界与上游审查；第 8 章有 GSS 音频生成、AuxIVA、Stream.FM/TF-Locoformer 和分离上游审查；第 9 章有追踪与移动音频生成、上游审查；第 10 章有工程综合基线、工业接口与原生 C/C++ 探针；第 11 章有会议评分接口审查；附录 B 有房间仿真生成器。这些脚本的精确文件名与原始验证范围见[算法覆盖表](ch00/COVERAGE.md)和[研究手册](ch00/research/README.md)。

`ch02_05_baselines.py`、`ch06_09_baselines.py`、三个 `exercises_*` 及空间模型、增强步骤、工程边界、时间状态等跨章练习集中在 [`ch00/cross_chapter/`](ch00/cross_chapter/)；其稳定 ID 由[练习目录](ch00/research/05_exercises_and_audio.md)逐题映射。后续拆分时须保持一个 ID 只有一个真实实现，并同步目录测试，不能复制一份后让两个实现各自漂移。

## 全书代码、实验资产与复现边界

这里的程序用于把正文公式变成可检查形状、单位、时延符号与边界情况的最小实验，不能直接当作产品音频前端。实时音频线程、设备驱动、调度、定点优化、模型权重和现场标定仍需按第 10 章另行完成。函数对明显不合法的维度或参数做输入检查，也不表示具备产品级防御能力。

全书的[算法覆盖表](ch00/COVERAGE.md)把正文方法对应到教学代码、测试与外部实现；[研究手册](ch00/research/README.md)说明源码入口、适用条件和已执行实验。阅读一个实现时，先核对输入输出形状、角度零点、传播时延与导向矢量相位，再比较数字。部分公共模块使用通道 × 频点 × 帧，WPE 与分离模块使用频点 × 通道 × 帧；跨模块传值需显式换轴。一个确定性例子数值相符，不能证明任意输入都正确。

本书的主[合成音频清单](ch00/audio/MANIFEST.json)记录 27 组、109 个分章存放的 PCM16 WAV 的输入、所属章节、共同增益、种子、运行环境、生成源码与逐文件 SHA-256。另有附录 B 的 18 个[房间合成 WAV 与结果报告](appendix_b/room_audio/RESULTS.json)、[GSS 教学链](ch08/gss_audio/MANIFEST.json)的 5 个 WAV 和中间状态、[连续移动声源](ch09/moving_audio/MANIFEST.json)的 3 个 WAV，以及[观测到追踪](ch09/tracking_audio/MANIFEST.json)的 2 个 WAV；这些是彼此独立的实验，不并入主 109 个样本。相同实验组使用共同导出增益，不逐文件做峰值归一化。生成物出现问题应修改生成源码并重新生成、只读核对清单，再重建图和站点；不得手改单个 WAV、清单或报告。

[DEMAND 数据说明](ch02/real_audio/README.md)记录 10 秒同步 16 通道真实环境录音摘录和 3 个派生文件的来源、通道及 CC BY-SA 3.0 条件。它们没有干净语音或位置真值，不用于声学增强性能结论。另一个 AEC 真实成对录音实验只使用本地忽略的 Microsoft AEC Challenge 缓存；其文件与可选输出不进入本书发布音频。正确、全零及错位参考的结果只是固定片段的接口观察，不是真值 ERLE。

外部代码由[固定 Git 来源清单](ch00/SOURCES.lock.json)和[独立归档清单](ch00/ARCHIVE_SOURCES.lock.json)管理，来源、许可证与用途见[第三方记录](ch00/THIRD_PARTY.md)。截至 2026-09-28，96 个 Git 项目中有 83 个本地工作区；[状态报告](ch00/SOURCE_STATUS.json)中的 82 项通过、13 项仅索引和 1 项未通过必须分别阅读。AEC Challenge 缓存中有本地修改的真实录音，核验没有把它当成通过。取得源码、构建依赖、实际运行和复现论文性能是不同层级；固定版诊断的执行条件与失败记录见[复现手册](ch00/research/04_source_reproduction.md)。

上游获取工具放在[源码工具目录](ch00/upstream/README.md)。下载源码留在 Git 忽略的独立工作区，不随本书推送；已有工作区及修改必须保留。HARKTOOL5 与 Vo RFS 的归档摘要和选择范围单独记录，不混入 96 个 Git 项目数。本仓库根目录目前没有明确的 `LICENSE` 或 `COPYING`，源码可见不等于已经获得复制、修改或再分发许可；第三方项目的许可证也不自动覆盖本书、模型或数据。
