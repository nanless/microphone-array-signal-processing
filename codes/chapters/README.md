# 按章查找教学实验

这里是全书教学代码与资产的统一检索入口。每章的 `core/` 放算法数值核，`examples/` 放对应实验或外部接口探针；同一个数值核只归一个章节，其他章节直接导入。导读 `ch00/` 保存跨章练习、主音频总清单、全书索引和上游获取工具。

例如第 4 章 Capon 与第 5 章 MVDR/LCMV 共用的协方差验证和相对对角加载，只在首次用到的 [`ch04/core/covariance.py`](ch04/core/covariance.py) 实现，第 5 章直接导入。

| 目录 | 主要实验模块（统一前缀 `codes.chapters.`） | 复算范围与边界 |
|---|---|---|
| [ch00](ch00/) | `ch00.cross_chapter.*` | 全书索引、跨章练习、主音频清单与上游来源 |
| [ch01](ch01/) | `ch01.chapter01_experiments` | E01-04～10；有限记录、残余延迟、球头、混合功率、ILD/WNG与频率相关响应；数学输入及实际PCM读回 |
| [ch02](ch02/) | `ch02.chapter02_experiments` | E02-09～20；传播、频谱、协方差、采样、已知激励辨识与STFT一致性，不是设备测量 |
| [ch03](ch03/) | `ch03.chapter03_experiments`、`ch03.coarray_covariance_exercise`、`ch03.examples.self_calibration_demo` | 单章入口E03-08～18，协同阵入口E03-07，与单独的受外部相位锚约束标定示例；几何、模糊、校准和虚拟滞后统计，不是全盲设备标定 |
| [ch04](ch04/) | `ch04.chapter04_experiments`、`ch04.doa_resolution_trials` | E04-08、E04-12～25；包括非酉聚焦噪声、AIC/MDL、相关误差GLS、双频酉秩与解析root-MUSIC；分辨率试验保留分类计数与统计分母 |
| [ch05](ch05/) | `ch05.chapter05_experiments`、`ch05.beamformer_common_input_demo` | 单章入口E05-08～24，E05-01～07由跨章入口提供；同输入波束、独立导数约束与逐频相位PCM限于所声明条件；`ch05.examples.generate_derivative_audio --check`只读核验八源/实际PCM，`ch05.examples.generate_phase_audio --check`核五源及三PCM；当前原方法审计与历史报告分开 |
| [ch06](ch06/) | `ch06.chapter06_experiments`、`ch06.aec_algorithm_minicases`、`ch06.aec_partitioned_demo`、`ch06.aec_affine_projection_demo`、`ch06.examples.generate_apa_audio`、`ch06.examples.generate_reference_audio` | E06-22～42及AEC算法缩例；已知增益/历史尾声六PCM严格只读核验与冻结预测，APA训练/留出另计；原方法current报告与历史报告分开，外部库/录音有独立依赖 |
| [ch07](ch07/) | `ch07.chapter07_experiments`、`ch07.wpe_temporal_contract`、`ch07.examples.mint_teaching_demo`、`ch07.examples.generate_delay_audio`、`ch07.examples.audit_upstream_wpe_contracts` | 单章入口E07-08～24，E07-01～07由跨章入口提供；在线WPE时间/排列、设计矩阵求解、已知到达与历史PCM控制及逆滤波噪声权衡；`mint_teaching_demo --check`只读核验，NeMo仅固定源码静态检查 |
| [ch08](ch08/) | `ch08.chapter08_experiments`、`ch08.gss_activity_error_demo`、`ch08.examples.mask_representation_demo`、`ch08.examples.generate_css_audio`、`ch08.examples.audit_upstream_separation_contracts` | 单章入口E08-12～32，E08-01～11由跨章入口提供；另有固定密度下活动标注误差；不是官方 GPU 整链 |
| [ch09](ch09/) | `ch09.chapter09_experiments`、`ch09.tracking_crossing_dropout_demo` | E09-10～26；轨迹交叉、缺测、模式密度、双时钟生命周期与球面方向的限定例 |
| [ch10](ch10/) | `ch10.chapter10_experiments`、`ch10.spectral_subtraction_demo`、`ch10.sro_closed_loop_demo` | E10-13、E10-18～34；合成时间戳及有状态插值，不是声卡实时实测 |
| [ch11](ch11/) | `ch11.chapter11_experiments` | E11-10～27；硬约束、评分和 FIR 取舍，示意分数不代表产品测量 |
| [ch12](ch12/) | `ch12.chapter12_exercises`、`ch12.examples.generate_imaging_audio`、`ch12.examples.audit_upstream_imaging_contracts` | E12-01～18；球面CSM/PSF、DAMAS前/双向、有限小矩阵NNLS、作者full-CSM CLEAN-SC及目标失配；五独立PCM及固定原方法差异分开 |
| [ch13](ch13/) | `ch13.chapter13_exercises`、`ch13.examples.generate_distributed_audio`、`ch13.examples.audit_upstream_distributed_contracts` | E13-01～25；指定MWF任务与压缩、真实广播后接收状态、GEVD/树控制；17独立PCM及限定原helper合同分开 |
| [appendix_a](appendix_a/) | `appendix_a.appendix_a_experiments` | E14-06～20；复投影、已知噪声加权、截断与正则、相关共轭、EVD前提、实际PCM及完整相关噪声GLS/同步白化；旧01～05仍复用跨章唯一实现 |
| [appendix_b](appendix_b/) | `appendix_b.appendix_b_experiments`、`appendix_b.interpolation_exercise` | E15-02～15；E15-02 读回主音频 PCM，E15-03～15 核查房间结果、PCM、来源证据、TAC共享结构、能量尺度、时间条件化与同DRR输出，不重跑房间仿真 |

从仓库根目录使用模块形式运行，例如：

```bash
.venv/bin/python -m codes.chapters.ch04.chapter04_experiments
.venv/bin/python -m codes.chapters.appendix_b.appendix_b_experiments
```

所有命令从仓库根目录运行，使用根目录主虚拟环境。第一次运行可先按[导读的动手复现](../../chapters/00_overview.md#8-动手复现)准备环境，再运行三组跨章基线。
它们只打印计算结果，不生成或改写音频。读输出时先找输入与对应约束：`GCC-PHAT tau12 (samples): 3.0`
表示第一个输入相对第二个晚 3 点，在 16 kHz 下是 187.5 μs；`DSB target response` 和
`MVDR target response` 应接近 1；STFT 重建误差接近浮点舍入。极高的匹配无噪声 AEC ERLE 是算术
收敛检查，队列示例中的超期次数则来自构造的负载，二者均不代表设备测量。完整核对步骤见导读。

旧的 `codes.examples.*` 和 `codes.array_tutorial.*` 导入路径已经退出仓内接口。运行时使用上表中的章节模块；修改题目时改唯一真实源文件，再核对全书 360 个稳定练习 ID、对应章节和覆盖表。

第4章E04-24用同源直达与反射说明高相干和秩一不足以确认直达方向；E04-25逐行解人工Q2帧CTF首比，区分整路径比与真实STFT近似。四份[独立PCM控制](ch04/reflection_audio/MANIFEST.json)由`codes.chapters.ch04.examples.generate_reflection_audio`生成，附加`--check`严格只读重放；原源当前报告与历史报告分开保存，详见[研究58/59](ch00/research/01_spatial_and_tracking.md#58-直达路径证据dpd筛选与dp-rtf不是同一计算)。

第3章E03-18从四已知方向反求三维基线及固定相对通道时延；同俯仰反例检查的是增广矩阵秩。先运行`.venv/bin/python -m codes.chapters.ch03.examples.generate_baseline_audio --check`只读核对六份[独立音频](ch03/baseline_audio/MANIFEST.json)，首次生成去掉`--check`。500Hz单音、2400:29600稳定窗和已知总时差小于1ms是相位反推前提；实际PCM训练拟合与第五方向的留出预测分别记录。原协同阵工具默认stdout，新[当前合同报告](ch03/reports/upstream_coarray_contracts.json)保留完整选集不匹配及限定原方法成功两个状态，不能覆盖旧历史报告。

第2章E02-19用已知激励辨识数字FIR，E02-20检查三重叠帧的STFT一致性。先用`.venv/bin/python -m codes.chapters.ch02.examples.generate_sweep_audio --check`只读核对四份[独立WAV](ch02/sweep_audio/MANIFEST.json)，再运行单章入口；首次生成去掉`--check`。源32000点、三响应32160点、共同增益1，实际PCM与浮点的全65536点IR评分分开；真值支持不用于裁剪估计。原方法合同工具默认stdout，新[当前报告](ch02/reports/upstream_model_contracts.json)与旧历史报告分开，取得源码和限定方法运行不等于完整上游或设备验收。

双耳、方向谱形、STFT卷积、多频几何、房间、主音频、GSS、移动和追踪资产的清单，以及某些工业报告，会校验生成器的**路径和完整源码摘要**。生成器已归入相应章节的 `examples/`；原生 C/C++ 探针与调用它的 Python 文件放在同一目录。更改这些文件后，应从真实新源重生资产并核对参数、逐文件摘要和报告，再同步构建与测试。

各章 `examples/` 的文件按学习章查找：第 1 章有双耳线索与方向谱形音频生成、libmysofa 原始归一化及插值方法探针；第 2 章有真实录音准备、独立STFT卷积与数字扫频试听，以及原方法提取对照；第 4 章有 MDL 重复试验、doatools/SBL/SMP-PHAT 原实现探针；第 5 章有波束上游与 SOF 设计审查；第 6 章有 AEC3、真实配对录音、同输入接口和上游审查；第 7 章有 WPE 外部对照、静音边界与上游审查；第 8 章有 GSS 音频生成、已知槽位极性/增益音频、两个一致性次序与给定背景约束、AuxIVA、Stream.FM/TF-Locoformer 和分离上游审查；第 9 章有追踪与移动音频生成、只读单槽生命周期和上游审查；第 10 章有工程综合基线、工业接口与原生 C/C++ 探针；第 11 章有会议评分接口审查；附录 B 有房间仿真生成器。这些脚本的精确文件名与原始验证范围见[算法覆盖表](ch00/COVERAGE.md)和[研究手册](ch00/research/README.md)。

`ch02_05_baselines.py`、`ch06_09_baselines.py`、三个 `exercises_*` 及空间模型、增强步骤、工程边界、时间状态等跨章练习集中在 [`ch00/cross_chapter/`](ch00/cross_chapter/)；其稳定 ID 由[练习目录](ch00/research/05_exercises_and_audio.md)逐题映射。后续拆分时须保持一个 ID 只有一个真实实现，并同步目录测试，不能复制一份后让两个实现各自漂移。

第 6 章[同输入 AEC 接口脚本](ch06/examples/aec_same_input_truth.py)默认只运行两份已记录摘要的历史二进制；另行构建时须用 `--build-manifest` 提交两份二进制的实际 SHA-256、源码提交与构建配置，脚本重新测固定延迟并评分。历史数值和新运行分开记录，具体字段与证据边界见[AEC 研究记录](ch00/research/02_aec_wpe_separation.md#aec)。

<a id="audio-inventory"></a>

## 完整音频目录与运行方式

发布源共267个WAV：主清单109个、以下27套独立合成154个、DEMAND真实摘录及派生4个。网站另有267份同内容副本；忽略的上游缓存中音频不计入发布资产。文件数不是独立实验次数，也不是算法数。

主清单由 `codes.chapters.ch00.examples.generate_audio_samples` 生成，27组分章存放；其唯一定义见 [MANIFEST.json](ch00/audio/MANIFEST.json) 和 [`code_layout.py`](../../scripts/code_layout.py)。以下模块统一加前缀 `codes.chapters.`，从仓库根目录运行。

| 独立目录与清单 | WAV数 | 生成/实验模块 | 模型与解释边界 |
|---|---:|---|---|
| [ch01/binaural_audio](ch01/binaural_audio/MANIFEST.json) | 5 | `ch01.examples.generate_binaural_cues` | 双耳时间差/声级差；人工声像，非实测HRTF |
| [ch01/spectral_audio](ch01/spectral_audio/MANIFEST.json) | 4 | `ch01.examples.generate_spectral_cues` | 已知两抽头FIR与两源谱；非盲方向估计 |
| [ch02/stft_audio](ch02/stft_audio/MANIFEST.json) | 3 | `ch02.examples.generate_stft_convolution` | 完整卷积与逐帧近似；完整尾部单列 |
| [ch02/sweep_audio](ch02/sweep_audio/MANIFEST.json) | 4 | `ch02.examples.generate_sweep_audio` | 已知数字扫频辨识；完整IR与真值支持分开 |
| [ch03/geometry_audio](ch03/geometry_audio/MANIFEST.json) | 3 | `ch03.examples.generate_geometry_audio` | 双频六麦几何控制；稳定窗相位，不用试听定方向 |
| [ch03/baseline_audio](ch03/baseline_audio/MANIFEST.json) | 6 | `ch03.examples.generate_baseline_audio` | 已知方向/主值范围标定；训练与留出分开 |
| [ch04/reflection_audio](ch04/reflection_audio/MANIFEST.json) | 4 | `ch04.examples.generate_reflection_audio` | 同源相干反射假峰；不运行盲DP-RTF |
| [ch04/focus_audio](ch04/focus_audio/MANIFEST.json) | 4 | `ch04.examples.generate_focus_audio` | 已知酉聚焦分量；不是完整CSSM |
| [ch05/derivative_audio](ch05/derivative_audio/MANIFEST.json) | 4 | `ch05.examples.generate_derivative_audio` | 三麦约束/导数约束；目标损伤与噪声代价分开 |
| [ch05/phase_audio](ch05/phase_audio/MANIFEST.json) | 3 | `ch05.examples.generate_phase_audio` | 已知逐频相位；等功率不保证参考误差相同 |
| [ch06/apa_audio](ch06/apa_audio/MANIFEST.json) | 6 | `ch06.examples.generate_apa_audio` | 训练后冻结NLMS/APA；留出窗与13点尾分开 |
| [ch06/reference_audio](ch06/reference_audio/MANIFEST.json) | 6 | `ch06.examples.generate_reference_audio` | 已知播放增益/尾声；正确残差可全零 |
| [ch07/delay_audio](ch07/delay_audio/MANIFEST.json) | 6 | `ch07.examples.generate_delay_audio` | 已知到达与单实回归；误消控制可全零 |
| [ch07/mint_audio](ch07/mint_audio/MANIFEST.json) | 6 | `ch07.examples.mint_teaching_demo` | 已知稀疏路径逆与噪声权衡；非一般MINT整链 |
| [ch08/mask_audio](ch08/mask_audio/MANIFEST.json) | 6 | `ch08.examples.mask_representation_demo` | 已知实/复掩码表示；未从录音估计掩码 |
| [ch08/css_audio](ch08/css_audio/MANIFEST.json) | 4 | `ch08.examples.generate_css_audio` | 既定槽位极性/增益；重叠拟合，不是盲CSS |
| [ch08/gss_audio](ch08/gss_audio/MANIFEST.json) | 5 | `ch08.examples.gss_teaching_demo` | 受控GSS教学链；另有STATE.npz，不是GPU整链 |
| [ch09/moving_audio](ch09/moving_audio/MANIFEST.json) | 3 | `ch09.examples.moving_source_audio` | 自由场连续移动与轨迹真值；非实录 |
| [ch09/tracking_audio](ch09/tracking_audio/MANIFEST.json) | 2 | `ch09.examples.chapter09_tracking_audio` | 实际PCM→GCC→门控/KF；两种时刻分开 |
| [ch10/noise_audio](ch10/noise_audio/MANIFEST.json) | 6 | `ch10.examples.generate_noise_mismatch` | 固定/污染/已知方差；不运行盲自适应估计 |
| [ch10/channel_audio](ch10/channel_audio/MANIFEST.json) | 6 | `ch10.examples.generate_channel_audio` | 已知第0麦失效；全零坏麦是模型输入 |
| [ch11/scenario_audio](ch11/scenario_audio/MANIFEST.json) | 8 | `ch11.examples.generate_selection_audio` | 两场景3/9抽头；权重/逐场景最坏分开 |
| [ch12/imaging_audio](ch12/imaging_audio/MANIFEST.json) | 5 | `ch12.examples.generate_imaging_audio` | 指定快拍相位与CSM；音频混合不能证明源量反演 |
| [ch13/distributed_audio](ch13/distributed_audio/MANIFEST.json) | 17 | `ch13.examples.generate_distributed_audio` | 已知统计/速率/缺口；非盲DANSE或实网络 |
| [appendix_a/weighted_audio](appendix_a/weighted_audio/MANIFEST.json) | 5 | `appendix_a.examples.generate_weighted_audio` | 已知噪声权重；纯音正交不等于随机独立 |
| [appendix_b/response_audio](appendix_b/response_audio/MANIFEST.json) | 5 | `appendix_b.examples.generate_response_audio` | 同RIR DRR两短FIR；源加权反射功率另命名 |
| [appendix_b/room_audio](appendix_b/room_audio/MANIFEST.json) | 18 | `appendix_b.examples.room_srp_exercise` | 固定PRA六位置白噪声仿真；隔离依赖、非真实房间 |

前26项在主环境运行，附加 `--check` 时只读核现有资产；生成模式会重建所属清单和音频。房间项不同：`room_srp_exercise --check`只核固定输入，现有18WAV的完整只读检查用 `codes.chapters.appendix_b.examples.check_room_assets`；`--replay`才在固定PRA环境实际重跑。真正生成用 `room_srp_exercise --run`，输出到不存在的新目录。顺序、完整命令和隔离环境见[构建说明](../../scripts/README.md#audio-regeneration)。

真实数据另用 `codes.chapters.ch02.examples.prepare_real_recordings --check`，不联网或重写。来源、固定归档、四份处理和独立许可见 [DEMAND说明](ch02/real_audio/README.md)。

比较组共用明确导出增益。评分先固定参考、已知对齐与半开采样窗，再分别阅读解析、浮点分量和实际PCM整数结果；不同组参数不能混用。多声道播放可能下混，不能以耳听代替逐通道相位或CSM检查。已知零残差和坏麦全零是实验控制；技术核验与真人听测分别记录。

## 全书代码、实验资产与复现边界

这里的程序用于把正文公式变成可检查形状、单位、时延符号与边界情况的最小实验，不能直接当作产品音频前端。实时音频线程、设备驱动、调度、定点优化、模型权重和现场标定仍需按第 10 章另行完成。函数对明显不合法的维度或参数做输入检查，也不表示具备产品级防御能力。

全书的[算法覆盖表](ch00/COVERAGE.md)把正文方法对应到教学代码、测试与外部实现；[研究手册](ch00/research/README.md)说明源码入口、适用条件和已执行实验。阅读一个实现时，先核对输入输出形状、角度零点、传播时延与导向矢量相位，再比较数字。部分公共模块使用通道 × 频点 × 帧，WPE 与分离模块使用频点 × 通道 × 帧；跨模块传值需显式换轴。一个确定性例子数值相符，不能证明任意输入都正确。

`ch00/` 集中保存[跨章练习](ch00/cross_chapter/)、[主音频生成器](ch00/examples/generate_audio_samples.py)、[Git 来源锁表](ch00/SOURCES.lock.json)、[归档锁表](ch00/ARCHIVE_SOURCES.lock.json)、[获取状态](ch00/SOURCE_STATUS.json)和[归档状态](ch00/ARCHIVE_SOURCE_STATUS.json)。状态由[获取工具](ch00/upstream/README.md)核验生成；忽略的 `_downloads/` 工作区可能含本地修改，不能覆盖或纳入提交。

来源索引扩充后，旧运行报告仍保留当时的整表摘要。[历史快照](ch00/source_snapshots/)保存真实原始字节；[只读核验核](ch00/core/source_history.py)检查已登记快照的完整SHA，以及报告实际使用项目的全部来源/获取记录是否仍与当前一致。它不改报告、不重跑算法，也不把旧选集失败改成通过；具体配对和覆盖范围见[复现手册](ch00/research/04_source_reproduction.md#historical-source-bindings)。

本书的主[合成音频清单](ch00/audio/MANIFEST.json)记录 27 组、109 个分章存放的 PCM16 WAV 的输入、所属章节、共同增益、种子、运行环境、生成源码与逐文件 SHA-256。另有二十七套独立合成实验，共154个WAV；双耳、方向谱形、STFT卷积、数字扫频、几何、基线标定、相干反射、聚焦、导数约束、逐频相位、APA、播放增益与尾声、已知到达与历史、已知逆、掩码、既定槽位增益、GSS、移动、追踪、噪声失配、已知坏麦、选型、成像、分布式、已知权重、同DRR和房间的逐套清单及用途见[导读的完整音频表](../../chapters/00_overview.md#audio-assets)。它们不并入主109，GSS状态与房间数值报告也不能按WAV计数。

相同合成实验组使用共同导出增益，不逐文件做峰值归一化。生成物出现问题应修改生成源码并重新生成、只读核对清单，再重建图和站点；不得手改单个 WAV、清单或报告。

主音频由 `.venv/bin/python codes/chapters/ch00/examples/generate_audio_samples.py` 生成，附加 `--check` 时只读重算并核对现有清单与 WAV。它们是数学合成样本，没有真人录音、模型权重或下载素材，不用于证明真实语音或设备效果；试听前先调低播放音量。各组的信号模型、参考、评分窗口和代码题见[音频实验手册](ch00/research/05_exercises_and_audio.md)。

[DEMAND 数据说明](ch02/real_audio/README.md)记录 10 秒同步 16 通道真实环境录音摘录和 3 个派生文件的来源、通道及 CC BY-SA 3.0 条件。它们没有干净语音或位置真值，不用于声学增强性能结论。另一个 AEC 真实成对录音实验只使用本地忽略的 Microsoft AEC Challenge 缓存；其文件与可选输出不进入本书发布音频。正确、全零及错位参考的结果只是固定片段的接口观察，不是真值 ERLE。

外部代码由[固定 Git 来源清单](ch00/SOURCES.lock.json)和[独立归档清单](ch00/ARCHIVE_SOURCES.lock.json)管理，来源、许可证与用途见[第三方记录](ch00/THIRD_PARTY.md)。截至 2026-10-05，115 个 Git 项目中有 98 个本地工作区；按完整32条排除规则核验，[状态报告](ch00/SOURCE_STATUS.json)分别记录75项通过、22项旧稀疏规则不匹配、17项仅索引和1项本地修改失败。22个工作区仍只有20条排除规则，保留原样，不自动修复；AEC Challenge 缓存中有本地修改的真实录音，核验没有把它当成通过。取得源码、构建依赖、实际运行和复现论文性能是不同层级；固定版诊断的执行条件与失败记录见[复现手册](ch00/research/04_source_reproduction.md)。

已取得的WASN同步源码与LibriCSS评测工具分别保存在独立忽略目录，固定提交、许可与选集见[复现入口](ch00/research/04_source_reproduction.md#overview-source-entrypoints)。它们仍未安装、运行设备或执行官方评分；LibriCSS原评分包装与脚本的接口限制保留。

上游获取工具放在[源码工具目录](ch00/upstream/README.md)。下载源码留在 Git 忽略的独立工作区，不随本书推送；已有工作区及修改必须保留。HARKTOOL5 与 Vo RFS 的归档摘要和选择范围单独记录，不混入 115 个 Git 项目数。本仓库根目录目前没有明确的 `LICENSE` 或 `COPYING`，源码可见不等于已经获得复制、修改或再分发许可；第三方项目的许可证也不自动覆盖本书、模型或数据。


第二章的`E02-16～18`分别检查有限窗卷积、源噪声二阶交叉项与窗的幅度/功率归一化。
独立[STFT试听生成器](ch02/examples/generate_stft_convolution.py)支持严格只读`--check`；
[原方法提取调用](ch02/examples/audit_upstream_models.py)运行PRA、Acoular和doatools的指定原方法，
并以人工适配调用六个pyfar原函数；当前报告分列方法执行与源码完整选集，未运行整包或设备系统。默认不安装额外依赖：

```bash
.venv/bin/python -m codes.chapters.ch02.chapter02_experiments
.venv/bin/python -m codes.chapters.ch02.examples.generate_stft_convolution --check
.venv/bin/python -m codes.chapters.ch02.examples.generate_sweep_audio --check
.venv/bin/python -m codes.chapters.ch02.examples.audit_upstream_models
```

第三章的[多频几何实验](ch03/geometry_audio/MANIFEST.json)由 `ch03.examples.generate_geometry_audio` 生成；加 `--check` 时只读核对实际PCM、六个生成源及分母。双频源参考为单声道，两个方向观测各为六声道；六声道试听可能下混，方向证据使用稳定窗逐频拟合。`ch03.examples.audit_upstream_coarray` 对固定doatools原方法作有限范围提取调用，报告包括兼容适配和真实失败，不代表整包或未知误差校准。

第四章 `ch04.chapter04_experiments` 的 E04-19～23 分别复算非酉聚焦噪声、AIC/MDL、相关误差GLS、双频酉聚焦秩和root-MUSIC解析多项式。`ch04.examples.generate_focus_audio` 生成四份独立合成WAV，加 `--check` 只读核对真实PCM、六个源摘要和评分分母。`ch04.examples.audit_upstream_doa` 对固定PRA/doatools原方法作限定模型复算；上游完整包、方法提取、辅助函数和兼容适配的结论分别阅读，不把局部诊断作为整链评测。

第八章 E08-24～29 分别复算白化与独立性、联合合同对角化的边界、FastMNMF雅可比项、加权混合一致性、已知掩码表示及CSS增益/极性。`ch08.examples.mask_representation_demo --check`只读检查六实际PCM与五真实源；`ch08.examples.gss_teaching_demo --check`严格核验七普通资产和状态。上游分离合同报告分别记录原包、原函数提取、静态检查与导入失败，不能合称完整论文复现。

E08-30沿用第2章唯一STFT核，逐共享样本核对两个一致性次序；E08-31区分PCA投掉弱目标与给定目标行的OverIVA背景约束；E08-32按同钟观测重叠拟合真实标量，分开主窗与后窗的解析、浮点和实际PCM整数误差。`ch08.examples.generate_css_audio --check`严格只读重放四WAV与独立清单，不代表盲CSS或身份追踪。作者OverIVA四MIT原文件与MeCo十六个许可明确的部分原文件已在受管理源码目录取得，固定身份与未运行边界见[分离研究手册](ch00/research/02_aec_wpe_separation.md)。

第9章共26题，E09-24～26逐步计算完整模式密度与连续缺测、实际PCM单槽生命周期、已知姿态与静态球面概率。`tracking.py` 是连续白角加速度 Q 的唯一实现；`set_prior_weights` 明确重设数学支持，显示下溢与真实零分开记录。E09-19核验实际文件后读回PCM，不用内存样本替代正式音频。当前外部源码合同见[原方法报告](ch09/reports/upstream_tracking_contracts_current.json)与[接口报告](ch09/reports/tracking_upstream_interfaces_current.json)和[来源研究](ch00/research/01_spatial_and_tracking.md#tracking-contract-audit)，替身控制流与完整算法运行分栏记录。

第10章共34题，扩展入口计算E10-18～34；[六个独立噪声失配WAV](ch10/noise_audio/MANIFEST.json)由 `ch10.examples.generate_noise_mismatch` 管理，不并入主109。纯前奏固定、目标污染与已知方差对照使用同一输入和导出增益；浮点分量与两个6400点实际PCM窗口分开。[工业合同工具](ch10/examples/audit_industrial_contracts.py)只运行明确限定的固定原源码接口，模型与ARM性能不在执行范围，历史报告不改写。

第11章 E11-10～27 从题设与正式文件计算：E19先只读核主清单真实源和四WAV，再按已知延迟作整数误差评分；E25使用 `ch11.examples.generate_selection_audio` 的八个独立两场景WAV，完整尾部与共同增益0.8保留，不并入主109个。`--check`完整回放且不修复资产。`ch11.examples.audit_meeting_kernel_contracts` 只执行固定MeetEval的两个原C++核，默认输出到终端，显式 `--report 路径` 才写当前报告，仓内仅接受 `reports/meeting_kernel_contracts_current.json`，仓外仅接受普通文件。当前13例是历史9例加4个点时间控制，不覆盖完整Python评分器；两历史报告保留。评分接口工具另执行32原模块导入与限定函数，缺扩展和完整选集不匹配如实保留。E26实际记录空输出、缺失、格式异常和回调异常；E27复用第5章波束核计算同阵列候选的噪声与硬条件，未提供设备延迟或宽带音频。

附录A共20题。E14-08通过 `appendix_a.examples.check_main_math_audio` 核当前20个真实源与主清单，再读三个正式脉冲WAV；内存浮点模型分开列出。E14-19的[五个独立已知噪声权重WAV](appendix_a/weighted_audio/MANIFEST.json)由 `appendix_a.examples.generate_weighted_audio` 管理，`--check`只读完整回放，不并入主109。目标和两个不同频率干扰共同包络/增益1，OLS/GLS/反权用同一双通道输入，三权重和均1；稳态28800点的解析、浮点分量和实际PCM整数评分分开。E14-20另用已知完整协方差复算相关噪声下的负权重与同步白化，不借用这五个互正交纯音WAV证明随机相关模型。`appendix_a.examples.audit_upstream_solver_contracts` 从已核原源码字节执行完整模块，保留原四例、另列四个形状/复数批次控制和两个独立NumPy例。仓内显式 `--report` 仅可写 `reports/upstream_solver_contracts_current.json`，历史报告保留原字节。

附录B共15题。E15-15以固定指数模型比较有限尾部与无限衰减，复用唯一房间指标核，不生成新WAV。E15-03～15入口统一为 `appendix_b.appendix_b_experiments`；主插值四WAV先核20真实源与完整PCM回放。E15-11～13分别拆解TAC共享聚合、安全能量尺度和T20时间条件化，E15-14使用[五个独立同DRR对照WAV](appendix_b/response_audio/MANIFEST.json)，四真实源、共同增益1及完整两点尾。`appendix_b.examples.generate_response_audio --check`只读核全部清单/字节/实际整数评分；不并入主109。房间资产的 `appendix_b.examples.check_room_assets` 核21普通成员、七真实源、固定DOA和81抽头/40点延迟声明、报告绑定和18PCM；`--replay`需隔离PRA环境真实重跑。`appendix_b.examples.audit_tac_contracts`默认stdout核固定原源的10个静态合同和当前完整选集，四项直接执行源前后绑定；显式报告仅仓内新`tac_contracts_current.json`或仓外普通路径，不导入Torch或执行原网络。

第12章为扩展专题Ⅰ，接第11章后；第13章为扩展专题Ⅱ，附录A/B分别为第14、15章。`ch12.chapter12_exercises`只读复算18题；`ch12.examples.generate_imaging_audio --check`严格核六普通成员及当前真实源、完整PCM回放，五份WAV不并入主109。`ch12.examples.audit_upstream_imaging_contracts`默认终端，只在显式`--report`时写当前合同；固定Acoular源码身份、整选集不匹配、方法相符/已确认差异/未执行分开。没有运行完整Acoular包、JIT/HDF5或风洞录音。

第13章扩展专题Ⅱ接第12章后。25题入口只读；17个16kHz数学WAV共同增益1、32000点即时混合，评分窗1600:30400及缺包窗16000:16800分开；九真实源与整数E/D随清单保存，`--check`严格核18普通成员并完整只读重放。已知100ppm线性SRC改变频响与噪声统计，不是盲时钟估计。当前原合同仅执行paderwasn三个AST原helper和MATLAB静态审读；没有原MATLAB、完整DWACD或无线设备成绩。

E10-34的[通道选集核](ch10/core/channel_selection.py)同时处理观测轴、目标响应与协方差双轴，复用第5章唯一MVDR；[六WAV独立清单](ch10/channel_audio/MANIFEST.json)使用确定性正交音与固定实权重。[生成入口](ch10/examples/generate_channel_audio.py)的`--check`完整只读重放。图78和手册分开目标损伤、噪声与真实PCM总误差，不把失效后的两项解析NMSE相同写成PCM严格相同。

扩展专题ⅠE12-17/18，分别检查完整CSM Gram目标与扫描目标、不同传播列的集体不可辨识性。图80保存全部151×101局部网格的真实计算。作者DAMAS原模块的[限定入口](ch12/examples/audit_damas_author_contracts.py)完整执行固定Python原模块的9个小控制；8项数值吻合、零CSM保留原NameError。21个GPL文本已取得，但MATLAB、MEX、数据和大规模实验未运行。当前报告与旧报告分开。


## 音频技术检查与真人听辨记录

[逐文件技术报告](../../reviews/2026-10-05-audio-technical-audit.json)记录267个发布源的实际PCM格式、帧数、声道、峰值、RMS、直流、静音、相邻跳变与谱诊断，并独立复核51个整数评分窗。满刻度端点样本为零不能证明没有削波：主清单工程削波控制在较低固定阈值产生平台，属于题设故障。

[听辨计划](../../reviews/2026-10-05-human-listening-plan.json)覆盖55组、267文件与396声道路由；[登记表](../../reviews/2026-10-05-human-listening-register.csv)当前全部 `not_run`，真实已听文件数为0。使用登记表副本记录实际听者、设备、时间、文件摘要、听完的采样区间、声道路由和观察，不预填通过。

先保持组内共同播放增益、关闭自动响度处理，完整听起音、稳态、切换和尾部；多声道逐路记录，不用未说明的下混代替。比较轮次可按计划固定随机顺序遮蔽文件身份，记录后再揭示。已知静音残差、坏麦零路、缺口、增益切换和相位差须与异常分别判断。发现可疑问题后复听并回查实际PCM/模型，再修改生成源。

这是项目音频质量核查计划，不宣称符合MUSHRA或其他标准主观评价流程。数学纯音/噪声不填写自然语音可懂度成绩；DEMAND没有干净目标真值，不以听感或数字功率下降给出SNR提升。
