# 按章查找教学实验

这里是全书教学代码与资产的唯一目录。每章的 `core/` 放算法数值核，`examples/` 放对应实验或外部接口探针；同一个数值核只归一个章节，其他章节直接导入。导读 `ch00/` 保存跨章练习、主音频总清单、全书索引和上游获取工具。

例如第 4 章 Capon 与第 5 章 MVDR/LCMV 共用的协方差验证和相对对角加载，只在首次用到的 [`ch04/core/covariance.py`](ch04/core/covariance.py) 实现，第 5 章直接导入。

| 目录 | 主要实验模块（统一前缀 `codes.chapters.`） | 复算范围与边界 |
|---|---|---|
| [ch00](ch00/) | `ch00.cross_chapter.*` | 全书索引、跨章练习、主音频清单与上游来源 |
| [ch01](ch01/) | `ch01.chapter01_experiments` | E01-04～09；有限记录、残余延迟、球头、混合功率、ILD与WNG；数学输入及实际PCM读回 |
| [ch02](ch02/) | `ch02.chapter02_experiments` | E02-09～18；传播、频谱、协方差与采样，不是设备测量 |
| [ch03](ch03/) | `ch03.chapter03_experiments`、`ch03.coarray_covariance_exercise`、`ch03.examples.self_calibration_demo` | E03-07～17 与单独的受外部相位锚约束标定示例；几何、模糊、校准和虚拟滞后统计，不是全盲设备标定 |
| [ch04](ch04/) | `ch04.chapter04_experiments`、`ch04.doa_resolution_trials` | E04-08、E04-12～18；分辨率试验保留分类计数与统计分母 |
| [ch05](ch05/) | `ch05.chapter05_experiments`、`ch05.beamformer_common_input_demo` | E05-01～22练习；同输入波束比较与独立导数约束PCM限于所声明条件；`ch05.examples.generate_derivative_audio --check`只读核验七源/实际PCM，原方法审计另记 |
| [ch06](ch06/) | `ch06.chapter06_experiments`、`ch06.aec_algorithm_minicases`、`ch06.aec_partitioned_demo` | E06-22～33 及 AEC 算法缩例；外部库或录音示例另有依赖 |
| [ch07](ch07/) | `ch07.chapter07_experiments`、`ch07.wpe_temporal_contract` | E07 练习及在线 WPE 分块连续性、未来帧影响 |
| [ch08](ch08/) | `ch08.chapter08_experiments`、`ch08.gss_activity_error_demo` | E08 练习及固定密度下活动标注误差；不是官方 GPU 整链 |
| [ch09](ch09/) | `ch09.chapter09_experiments`、`ch09.tracking_crossing_dropout_demo` | E09-10～19；轨迹交叉、缺测、方向限速的合成反例 |
| [ch10](ch10/) | `ch10.chapter10_experiments`、`ch10.spectral_subtraction_demo`、`ch10.sro_closed_loop_demo` | E10-13、E10-18～27；合成时间戳及有状态插值，不是声卡实时实测 |
| [ch11](ch11/) | `ch11.chapter11_experiments` | E11-10～19；硬约束、评分和 FIR 取舍，示意分数不代表产品测量 |
| [appendix_a](appendix_a/) | `appendix_a.appendix_a_experiments` | E12-06～13；短向量、矩阵和合成脉冲的数学边界 |
| [appendix_b](appendix_b/) | `appendix_b.appendix_b_experiments`、`appendix_b.interpolation_exercise` | E13-02～10；E13-02 读回主音频 PCM，E13-03～10 核查房间结果、PCM、来源证据或解析反例，不重跑房间仿真 |

从仓库根目录使用模块形式运行，例如：

```bash
.venv/bin/python -m codes.chapters.ch04.chapter04_experiments
.venv/bin/python -m codes.chapters.appendix_b.appendix_b_experiments
```

第一次运行可先按[导读的动手复现](../../chapters/00_overview.md#sec-u-6d43c0086f)准备环境，再运行三组跨章基线。
它们只打印计算结果，不生成或改写音频。读输出时先找输入与对应约束：`GCC-PHAT tau12 (samples): 3.0`
表示第一个输入相对第二个晚 3 点，在 16 kHz 下是 187.5 μs；`DSB target response` 和
`MVDR target response` 应接近 1；STFT 重建误差接近浮点舍入。极高的匹配无噪声 AEC ERLE 是算术
收敛检查，队列示例中的超期次数则来自构造的负载，二者均不代表设备测量。完整核对步骤见导读。

旧的 `codes.examples.*` 和 `codes.array_tutorial.*` 导入路径已经退出仓内接口。运行时使用上表中的章节模块；修改题目时改唯一真实源文件，再核对全书 250 个稳定练习 ID、对应章节和覆盖表。

双耳、STFT卷积、多频几何、房间、主音频、GSS、移动和追踪资产的清单，以及某些工业报告，会校验生成器的**路径和完整源码摘要**。生成器已归入相应章节的 `examples/`；原生 C/C++ 探针与调用它的 Python 文件放在同一目录。更改这些文件后，应从真实新源重生资产并核对参数、逐文件摘要和报告，再同步构建与测试。

各章 `examples/` 的文件按学习章查找：第 1 章有双耳线索音频生成和 libmysofa 原始归一化方法探针；第 2 章有真实录音准备、独立STFT卷积试听和原方法提取对照；第 4 章有 MDL 重复试验、doatools/SBL/SMP-PHAT 原实现探针；第 5 章有波束上游与 SOF 设计审查；第 6 章有 AEC3、真实配对录音、同输入接口和上游审查；第 7 章有 WPE 外部对照、静音边界与上游审查；第 8 章有 GSS 音频生成、AuxIVA、Stream.FM/TF-Locoformer 和分离上游审查；第 9 章有追踪与移动音频生成、上游审查；第 10 章有工程综合基线、工业接口与原生 C/C++ 探针；第 11 章有会议评分接口审查；附录 B 有房间仿真生成器。这些脚本的精确文件名与原始验证范围见[算法覆盖表](ch00/COVERAGE.md)和[研究手册](ch00/research/README.md)。

`ch02_05_baselines.py`、`ch06_09_baselines.py`、三个 `exercises_*` 及空间模型、增强步骤、工程边界、时间状态等跨章练习集中在 [`ch00/cross_chapter/`](ch00/cross_chapter/)；其稳定 ID 由[练习目录](ch00/research/05_exercises_and_audio.md)逐题映射。后续拆分时须保持一个 ID 只有一个真实实现，并同步目录测试，不能复制一份后让两个实现各自漂移。

第 6 章[同输入 AEC 接口脚本](ch06/examples/aec_same_input_truth.py)默认只运行两份已记录摘要的历史二进制；另行构建时须用 `--build-manifest` 提交两份二进制的实际 SHA-256、源码提交与构建配置，脚本重新测固定延迟并评分。历史数值和新运行分开记录，具体字段与证据边界见[AEC 研究记录](ch00/research/02_aec_wpe_separation.md#aec)。

## 全书代码、实验资产与复现边界

这里的程序用于把正文公式变成可检查形状、单位、时延符号与边界情况的最小实验，不能直接当作产品音频前端。实时音频线程、设备驱动、调度、定点优化、模型权重和现场标定仍需按第 10 章另行完成。函数对明显不合法的维度或参数做输入检查，也不表示具备产品级防御能力。

全书的[算法覆盖表](ch00/COVERAGE.md)把正文方法对应到教学代码、测试与外部实现；[研究手册](ch00/research/README.md)说明源码入口、适用条件和已执行实验。阅读一个实现时，先核对输入输出形状、角度零点、传播时延与导向矢量相位，再比较数字。部分公共模块使用通道 × 频点 × 帧，WPE 与分离模块使用频点 × 通道 × 帧；跨模块传值需显式换轴。一个确定性例子数值相符，不能证明任意输入都正确。

`ch00/` 集中保存[跨章练习](ch00/cross_chapter/)、[主音频生成器](ch00/examples/generate_audio_samples.py)、[Git 来源锁表](ch00/SOURCES.lock.json)、[归档锁表](ch00/ARCHIVE_SOURCES.lock.json)、[获取状态](ch00/SOURCE_STATUS.json)和[归档状态](ch00/ARCHIVE_SOURCE_STATUS.json)。状态由[获取工具](ch00/upstream/README.md)核验生成；忽略的 `_downloads/` 工作区可能含本地修改，不能覆盖或纳入提交。

本书的主[合成音频清单](ch00/audio/MANIFEST.json)记录 27 组、109 个分章存放的 PCM16 WAV 的输入、所属章节、共同增益、种子、运行环境、生成源码与逐文件 SHA-256。另有第 1 章的 5 个[双耳线索 WAV](ch01/binaural_audio/MANIFEST.json)、第 2 章的 3 个[STFT卷积 WAV](ch02/stft_audio/MANIFEST.json)、第3章的3个[多频几何WAV](ch03/geometry_audio/MANIFEST.json)、第4章的4个[已知聚焦WAV](ch04/focus_audio/MANIFEST.json)、附录 B 的 18 个[房间合成 WAV 与结果报告](appendix_b/room_audio/RESULTS.json)、[GSS 教学链](ch08/gss_audio/MANIFEST.json)的 5 个 WAV 和中间状态、[连续移动声源](ch09/moving_audio/MANIFEST.json)的 3 个 WAV，以及[观测到追踪](ch09/tracking_audio/MANIFEST.json)的 2 个 WAV；这些是彼此独立的实验，不并入主 109 个样本。相同实验组使用共同导出增益，不逐文件做峰值归一化。生成物出现问题应修改生成源码并重新生成、只读核对清单，再重建图和站点；不得手改单个 WAV、清单或报告。

主音频由 `.venv/bin/python codes/chapters/ch00/examples/generate_audio_samples.py` 生成，附加 `--check` 时只读重算并核对现有清单与 WAV。它们是数学合成样本，没有真人录音、模型权重或下载素材，不用于证明真实语音或设备效果；试听前先调低播放音量。各组的信号模型、参考、评分窗口和代码题见[音频实验手册](ch00/research/05_exercises_and_audio.md)。

[DEMAND 数据说明](ch02/real_audio/README.md)记录 10 秒同步 16 通道真实环境录音摘录和 3 个派生文件的来源、通道及 CC BY-SA 3.0 条件。它们没有干净语音或位置真值，不用于声学增强性能结论。另一个 AEC 真实成对录音实验只使用本地忽略的 Microsoft AEC Challenge 缓存；其文件与可选输出不进入本书发布音频。正确、全零及错位参考的结果只是固定片段的接口观察，不是真值 ERLE。

外部代码由[固定 Git 来源清单](ch00/SOURCES.lock.json)和[独立归档清单](ch00/ARCHIVE_SOURCES.lock.json)管理，来源、许可证与用途见[第三方记录](ch00/THIRD_PARTY.md)。截至 2026-10-01，98 个 Git 项目中有 85 个本地工作区；[状态报告](ch00/SOURCE_STATUS.json)中的 84 项通过、13 项仅索引和 1 项未通过必须分别阅读。AEC Challenge 缓存中有本地修改的真实录音，核验没有把它当成通过。取得源码、构建依赖、实际运行和复现论文性能是不同层级；固定版诊断的执行条件与失败记录见[复现手册](ch00/research/04_source_reproduction.md)。

上游获取工具放在[源码工具目录](ch00/upstream/README.md)。下载源码留在 Git 忽略的独立工作区，不随本书推送；已有工作区及修改必须保留。HARKTOOL5 与 Vo RFS 的归档摘要和选择范围单独记录，不混入 98 个 Git 项目数。本仓库根目录目前没有明确的 `LICENSE` 或 `COPYING`，源码可见不等于已经获得复制、修改或再分发许可；第三方项目的许可证也不自动覆盖本书、模型或数据。


第二章的`E02-16～18`分别检查有限窗卷积、源噪声二阶交叉项与窗的幅度/功率归一化。
独立[STFT试听生成器](ch02/examples/generate_stft_convolution.py)支持严格只读`--check`；
[原方法提取调用](ch02/examples/audit_upstream_models.py)运行PRA、Acoular和doatools的指定原方法，
其报告不表示三个整包或设备系统已经运行。默认不安装额外依赖：

```bash
.venv/bin/python -m codes.chapters.ch02.chapter02_experiments
.venv/bin/python -m codes.chapters.ch02.examples.generate_stft_convolution --check
.venv/bin/python -m codes.chapters.ch02.examples.audit_upstream_models --report codes/chapters/ch02/reports/upstream_models.json
```

第三章的[多频几何实验](ch03/geometry_audio/MANIFEST.json)由 `ch03.examples.generate_geometry_audio` 生成；加 `--check` 时只读核对实际PCM、五个生成源及分母。双频源参考为单声道，两个方向观测各为六声道；六声道试听可能下混，方向证据使用稳定窗逐频拟合。`ch03.examples.audit_upstream_coarray` 对固定doatools原方法作有限范围提取调用，报告包括兼容适配和真实失败，不代表整包或未知误差校准。

第四章 `ch04.chapter04_experiments` 的 E04-19～23 分别复算非酉聚焦噪声、AIC/MDL、相关误差GLS、双频酉聚焦秩和root-MUSIC解析多项式。`ch04.examples.generate_focus_audio` 生成四份独立合成WAV，加 `--check` 只读核对真实PCM、五个源摘要和评分分母。`ch04.examples.audit_upstream_doa` 对固定PRA/doatools原方法作限定模型复算；上游完整包、方法提取、辅助函数和兼容适配的结论分别阅读，不把局部诊断作为整链评测。
