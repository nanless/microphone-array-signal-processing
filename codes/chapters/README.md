# 按章查找教学实验

这里是全书教学代码与资产的唯一目录。每章的 `core/` 放算法数值核，`examples/` 放对应实验或外部接口探针；同一个数值核只归一个章节，其他章节直接导入。导读 `ch00/` 保存跨章练习、主音频总清单、全书索引和上游获取工具。

例如第 4 章 Capon 与第 5 章 MVDR/LCMV 共用的协方差验证和相对对角加载，只在首次用到的 [`ch04/core/covariance.py`](ch04/core/covariance.py) 实现，第 5 章直接导入。

| 目录 | 主要实验模块（统一前缀 `codes.chapters.`） | 复算范围与边界 |
|---|---|---|
| [ch00](ch00/) | `ch00.cross_chapter.*` | 全书索引、跨章练习、主音频清单与上游来源 |
| [ch01](ch01/) | `ch01.chapter01_experiments` | E01-04～10；有限记录、残余延迟、球头、混合功率、ILD/WNG与频率相关响应；数学输入及实际PCM读回 |
| [ch02](ch02/) | `ch02.chapter02_experiments` | E02-09～20；传播、频谱、协方差、采样、已知激励辨识与STFT一致性，不是设备测量 |
| [ch03](ch03/) | `ch03.chapter03_experiments`、`ch03.coarray_covariance_exercise`、`ch03.examples.self_calibration_demo` | E03-07～18 与单独的受外部相位锚约束标定示例；几何、模糊、校准和虚拟滞后统计，不是全盲设备标定 |
| [ch04](ch04/) | `ch04.chapter04_experiments`、`ch04.doa_resolution_trials` | E04-08、E04-12～25；包括非酉聚焦噪声、AIC/MDL、相关误差GLS、双频酉秩与解析root-MUSIC；分辨率试验保留分类计数与统计分母 |
| [ch05](ch05/) | `ch05.chapter05_experiments`、`ch05.beamformer_common_input_demo` | E05-01～24练习；同输入波束、独立导数约束与逐频相位PCM限于所声明条件；`ch05.examples.generate_derivative_audio --check`只读核验八源/实际PCM，`ch05.examples.generate_phase_audio --check`核五源及三PCM；当前原方法审计与历史报告分开 |
| [ch06](ch06/) | `ch06.chapter06_experiments`、`ch06.aec_algorithm_minicases`、`ch06.aec_partitioned_demo`、`ch06.aec_affine_projection_demo`、`ch06.examples.generate_apa_audio`、`ch06.examples.generate_reference_audio` | E06-22～42及AEC算法缩例；已知增益/历史尾声六PCM严格只读核验与冻结预测，APA训练/留出另计；原方法current报告与历史报告分开，外部库/录音有独立依赖 |
| [ch07](ch07/) | `ch07.chapter07_experiments`、`ch07.wpe_temporal_contract`、`ch07.examples.mint_teaching_demo`、`ch07.examples.audit_upstream_wpe_contracts` | E07-01～21；在线WPE时间/排列、设计矩阵求解与实际PCM逆滤波噪声权衡；`mint_teaching_demo --check`只读核验，NeMo仅固定源码静态检查 |
| [ch08](ch08/) | `ch08.chapter08_experiments`、`ch08.gss_activity_error_demo`、`ch08.examples.mask_representation_demo`、`ch08.examples.audit_upstream_separation_contracts` | E08-01～29及固定密度下活动标注误差；不是官方 GPU 整链 |
| [ch09](ch09/) | `ch09.chapter09_experiments`、`ch09.tracking_crossing_dropout_demo` | E09-10～23；轨迹交叉、缺测、方向限速的合成反例 |
| [ch10](ch10/) | `ch10.chapter10_experiments`、`ch10.spectral_subtraction_demo`、`ch10.sro_closed_loop_demo` | E10-13、E10-18～33；合成时间戳及有状态插值，不是声卡实时实测 |
| [ch11](ch11/) | `ch11.chapter11_experiments` | E11-10～25；硬约束、评分和 FIR 取舍，示意分数不代表产品测量 |
| [ch14](ch14/) | `ch14.chapter14_exercises`、`ch14.examples.generate_imaging_audio`、`ch14.examples.audit_upstream_imaging_contracts` | E14-01～16；球面CSM/PSF、DAMAS前/双向、有限小矩阵NNLS、作者full-CSM CLEAN-SC及目标失配；五独立PCM及固定原方法差异分开 |
| [ch15](ch15/) | `ch15.chapter15_exercises`、`ch15.examples.generate_distributed_audio`、`ch15.examples.audit_upstream_distributed_contracts` | E15-01～24；指定MWF任务与压缩、真实广播后接收状态、GEVD/树控制；17独立PCM及限定原helper合同分开 |
| [appendix_a](appendix_a/) | `appendix_a.appendix_a_experiments` | E12-06～19；复投影、已知噪声加权、截断与正则、相关共轭、EVD前提及实际PCM；旧01～05仍复用跨章唯一实现 |
| [appendix_b](appendix_b/) | `appendix_b.appendix_b_experiments`、`appendix_b.interpolation_exercise` | E13-02～14；E13-02 读回主音频 PCM，E13-03～14 核查房间结果、PCM、来源证据、TAC共享结构、能量尺度、时间条件化与同DRR输出，不重跑房间仿真 |

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

旧的 `codes.examples.*` 和 `codes.array_tutorial.*` 导入路径已经退出仓内接口。运行时使用上表中的章节模块；修改题目时改唯一真实源文件，再核对全书 343 个稳定练习 ID、对应章节和覆盖表。

第4章E04-24用同源直达与反射说明高相干和秩一不足以确认直达方向；E04-25逐行解人工Q2帧CTF首比，区分整路径比与真实STFT近似。四份[独立PCM控制](ch04/reflection_audio/MANIFEST.json)由`codes.chapters.ch04.examples.generate_reflection_audio`生成，附加`--check`严格只读重放；原源当前报告与历史报告分开保存，详见[研究58/59](ch00/research/01_spatial_and_tracking.md#sec-u-1ca23edba5)。

第3章E03-18从四已知方向反求三维基线及固定相对通道时延；同俯仰反例检查的是增广矩阵秩。先运行`.venv/bin/python -m codes.chapters.ch03.examples.generate_baseline_audio --check`只读核对六份[独立音频](ch03/baseline_audio/MANIFEST.json)，首次生成去掉`--check`。500Hz单音、2400:29600稳定窗和已知总时差小于1ms是相位反推前提；实际PCM训练拟合与第五方向的留出预测分别记录。原协同阵工具默认stdout，新[当前合同报告](ch03/reports/upstream_coarray_contracts.json)保留完整选集不匹配及限定原方法成功两个状态，不能覆盖旧历史报告。

第2章E02-19用已知激励辨识数字FIR，E02-20检查三重叠帧的STFT一致性。先用`.venv/bin/python -m codes.chapters.ch02.examples.generate_sweep_audio --check`只读核对四份[独立WAV](ch02/sweep_audio/MANIFEST.json)，再运行单章入口；首次生成去掉`--check`。源32000点、三响应32160点、共同增益1，实际PCM与浮点的全65536点IR评分分开；真值支持不用于裁剪估计。原方法合同工具默认stdout，新[当前报告](ch02/reports/upstream_model_contracts.json)与旧历史报告分开，取得源码和限定方法运行不等于完整上游或设备验收。

双耳、方向谱形、STFT卷积、多频几何、房间、主音频、GSS、移动和追踪资产的清单，以及某些工业报告，会校验生成器的**路径和完整源码摘要**。生成器已归入相应章节的 `examples/`；原生 C/C++ 探针与调用它的 Python 文件放在同一目录。更改这些文件后，应从真实新源重生资产并核对参数、逐文件摘要和报告，再同步构建与测试。

各章 `examples/` 的文件按学习章查找：第 1 章有双耳线索与方向谱形音频生成、libmysofa 原始归一化及插值方法探针；第 2 章有真实录音准备、独立STFT卷积与数字扫频试听，以及原方法提取对照；第 4 章有 MDL 重复试验、doatools/SBL/SMP-PHAT 原实现探针；第 5 章有波束上游与 SOF 设计审查；第 6 章有 AEC3、真实配对录音、同输入接口和上游审查；第 7 章有 WPE 外部对照、静音边界与上游审查；第 8 章有 GSS 音频生成、AuxIVA、Stream.FM/TF-Locoformer 和分离上游审查；第 9 章有追踪与移动音频生成、上游审查；第 10 章有工程综合基线、工业接口与原生 C/C++ 探针；第 11 章有会议评分接口审查；附录 B 有房间仿真生成器。这些脚本的精确文件名与原始验证范围见[算法覆盖表](ch00/COVERAGE.md)和[研究手册](ch00/research/README.md)。

`ch02_05_baselines.py`、`ch06_09_baselines.py`、三个 `exercises_*` 及空间模型、增强步骤、工程边界、时间状态等跨章练习集中在 [`ch00/cross_chapter/`](ch00/cross_chapter/)；其稳定 ID 由[练习目录](ch00/research/05_exercises_and_audio.md)逐题映射。后续拆分时须保持一个 ID 只有一个真实实现，并同步目录测试，不能复制一份后让两个实现各自漂移。

第 6 章[同输入 AEC 接口脚本](ch06/examples/aec_same_input_truth.py)默认只运行两份已记录摘要的历史二进制；另行构建时须用 `--build-manifest` 提交两份二进制的实际 SHA-256、源码提交与构建配置，脚本重新测固定延迟并评分。历史数值和新运行分开记录，具体字段与证据边界见[AEC 研究记录](ch00/research/02_aec_wpe_separation.md#aec)。

## 全书代码、实验资产与复现边界

这里的程序用于把正文公式变成可检查形状、单位、时延符号与边界情况的最小实验，不能直接当作产品音频前端。实时音频线程、设备驱动、调度、定点优化、模型权重和现场标定仍需按第 10 章另行完成。函数对明显不合法的维度或参数做输入检查，也不表示具备产品级防御能力。

全书的[算法覆盖表](ch00/COVERAGE.md)把正文方法对应到教学代码、测试与外部实现；[研究手册](ch00/research/README.md)说明源码入口、适用条件和已执行实验。阅读一个实现时，先核对输入输出形状、角度零点、传播时延与导向矢量相位，再比较数字。部分公共模块使用通道 × 频点 × 帧，WPE 与分离模块使用频点 × 通道 × 帧；跨模块传值需显式换轴。一个确定性例子数值相符，不能证明任意输入都正确。

`ch00/` 集中保存[跨章练习](ch00/cross_chapter/)、[主音频生成器](ch00/examples/generate_audio_samples.py)、[Git 来源锁表](ch00/SOURCES.lock.json)、[归档锁表](ch00/ARCHIVE_SOURCES.lock.json)、[获取状态](ch00/SOURCE_STATUS.json)和[归档状态](ch00/ARCHIVE_SOURCE_STATUS.json)。状态由[获取工具](ch00/upstream/README.md)核验生成；忽略的 `_downloads/` 工作区可能含本地修改，不能覆盖或纳入提交。

来源索引扩充后，旧运行报告仍保留当时的整表摘要。[历史快照](ch00/source_snapshots/)保存真实原始字节；[只读核验核](ch00/core/source_history.py)检查已登记快照的完整SHA，以及报告实际使用项目的全部来源/获取记录是否仍与当前一致。它不改报告、不重跑算法，也不把旧选集失败改成通过；具体配对和覆盖范围见[复现手册](ch00/research/04_source_reproduction.md#historical-source-bindings)。

本书的主[合成音频清单](ch00/audio/MANIFEST.json)记录 27 组、109 个分章存放的 PCM16 WAV 的输入、所属章节、共同增益、种子、运行环境、生成源码与逐文件 SHA-256。另有二十四套独立合成实验，共138个WAV；双耳、方向谱形、STFT卷积、数字扫频、几何、基线标定、相干反射、聚焦、导数约束、逐频相位、APA、播放增益与尾声、已知逆、掩码、GSS、移动、追踪、噪声失配、选型、成像、分布式、已知权重、同DRR和房间的逐套清单及用途见[导读的完整音频表](../../chapters/00_overview.md#audio-assets)。它们不并入主109，GSS状态与房间数值报告也不能按WAV计数。

相同合成实验组使用共同导出增益，不逐文件做峰值归一化。生成物出现问题应修改生成源码并重新生成、只读核对清单，再重建图和站点；不得手改单个 WAV、清单或报告。

主音频由 `.venv/bin/python codes/chapters/ch00/examples/generate_audio_samples.py` 生成，附加 `--check` 时只读重算并核对现有清单与 WAV。它们是数学合成样本，没有真人录音、模型权重或下载素材，不用于证明真实语音或设备效果；试听前先调低播放音量。各组的信号模型、参考、评分窗口和代码题见[音频实验手册](ch00/research/05_exercises_and_audio.md)。

[DEMAND 数据说明](ch02/real_audio/README.md)记录 10 秒同步 16 通道真实环境录音摘录和 3 个派生文件的来源、通道及 CC BY-SA 3.0 条件。它们没有干净语音或位置真值，不用于声学增强性能结论。另一个 AEC 真实成对录音实验只使用本地忽略的 Microsoft AEC Challenge 缓存；其文件与可选输出不进入本书发布音频。正确、全零及错位参考的结果只是固定片段的接口观察，不是真值 ERLE。

外部代码由[固定 Git 来源清单](ch00/SOURCES.lock.json)和[独立归档清单](ch00/ARCHIVE_SOURCES.lock.json)管理，来源、许可证与用途见[第三方记录](ch00/THIRD_PARTY.md)。截至 2026-10-04，109 个 Git 项目中有 92 个本地工作区；按完整32条排除规则核验，[状态报告](ch00/SOURCE_STATUS.json)分别记录69项通过、22项旧稀疏规则不匹配、17项仅索引和1项本地修改失败。22个工作区仍只有20条排除规则，保留原样，不自动修复；AEC Challenge 缓存中有本地修改的真实录音，核验没有把它当成通过。取得源码、构建依赖、实际运行和复现论文性能是不同层级；固定版诊断的执行条件与失败记录见[复现手册](ch00/research/04_source_reproduction.md)。

本轮补取的WASN同步源码与LibriCSS评测工具分别保存在独立忽略目录，固定提交、许可与选集见[复现入口](ch00/research/04_source_reproduction.md#overview-source-entrypoints)。它们仍未安装、运行设备或执行官方评分；LibriCSS原评分包装与脚本的接口限制保留。

上游获取工具放在[源码工具目录](ch00/upstream/README.md)。下载源码留在 Git 忽略的独立工作区，不随本书推送；已有工作区及修改必须保留。HARKTOOL5 与 Vo RFS 的归档摘要和选择范围单独记录，不混入 109 个 Git 项目数。本仓库根目录目前没有明确的 `LICENSE` 或 `COPYING`，源码可见不等于已经获得复制、修改或再分发许可；第三方项目的许可证也不自动覆盖本书、模型或数据。


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

第八章新增 E08-24～29 分别复算白化与独立性、联合合同对角化的边界、FastMNMF雅可比项、加权混合一致性、已知掩码表示及CSS增益/极性。`ch08.examples.mask_representation_demo --check`只读检查六实际PCM与五真实源；`ch08.examples.gss_teaching_demo --check`严格核验七普通资产和状态。上游分离合同报告分别记录原包、原函数提取、静态检查与导入失败，不能合称完整论文复现。

第9章的23题包括相关观测、纯方位尺度、存在概率与迟到观测四个新增解析例。`tracking.py` 是连续白角加速度 Q 的唯一实现；`set_prior_weights` 明确重设数学支持，显示下溢与真实零分开记录。E09-19核验实际文件后读回PCM，不用内存样本替代正式音频。当前外部源码合同见[独立报告](ch09/reports/upstream_tracking_contracts.json)和[来源研究](ch00/research/01_spatial_and_tracking.md#tracking-contract-audit)，替身控制流与完整算法运行分栏记录。

第10章共33题，扩展入口计算E10-18～33；[六个独立噪声失配WAV](ch10/noise_audio/MANIFEST.json)由 `ch10.examples.generate_noise_mismatch` 管理，不并入主109。纯前奏固定、目标污染与已知方差对照使用同一输入和导出增益；浮点分量与两个6400点实际PCM窗口分开。[工业合同工具](ch10/examples/audit_industrial_contracts.py)只运行明确限定的固定原源码接口，模型与ARM性能不在执行范围，历史报告不改写。

第11章 E11-10～25 从题设与正式文件计算：E19先只读核主清单真实源和四WAV，再按已知延迟作整数误差评分；E25使用 `ch11.examples.generate_selection_audio` 的八个独立两场景WAV，完整尾部与共同增益0.8保留，不并入主109个。`--check`完整回放且不修复资产。`ch11.examples.audit_meeting_kernel_contracts` 只执行固定MeetEval的两个原C++核，默认输出到终端，显式 `--report` 才写报告；不代表完整Python评分器运行。

附录A共19题。E12-08通过 `appendix_a.examples.check_main_math_audio` 核当前20个真实源与主清单，再读三个正式脉冲WAV；内存浮点模型分开列出。E12-19的[五个独立已知噪声权重WAV](appendix_a/weighted_audio/MANIFEST.json)由 `appendix_a.examples.generate_weighted_audio` 管理，`--check`只读完整回放，不并入主109。目标和两个不同频率干扰共同包络/增益1，OLS/GLS/反权用同一双通道输入，三权重和均1；稳态28800点的解析、浮点分量和实际PCM整数评分分开。`appendix_a.examples.audit_upstream_solver_contracts` 只执行固定pb_bss完整原模块中的辅助函数，保留整数dtype失败与独立NumPy比较，只有显式 `--report` 写当前报告。

附录B共14题。E13-03～14入口统一为 `appendix_b.appendix_b_experiments`；主插值四WAV先核20真实源与完整PCM回放。E11～13分别拆解TAC共享聚合、安全能量尺度和T20时间条件化，E14使用[五个独立同DRR对照WAV](appendix_b/response_audio/MANIFEST.json)，四真实源、共同增益1及完整两点尾。`appendix_b.examples.generate_response_audio --check`只读核全部清单/字节/实际整数评分；不并入主109。房间资产的 `appendix_b.examples.check_room_assets` 核21普通成员、七真实源、报告绑定和18PCM；`--replay`需隔离PRA环境真实重跑。`appendix_b.examples.audit_tac_contracts`只核固定原源的10个静态合同，不导入Torch或执行原网络。

扩展专题Ⅰ采用文件/代码身份14，阅读时放在第11章后、附录前，不改附录身份12/13。`ch14.chapter14_exercises`只读复算16题；`ch14.examples.generate_imaging_audio --check`严格核六普通成员及当前真实源、完整PCM回放，五份WAV不并入主109。`ch14.examples.audit_upstream_imaging_contracts`默认终端，只在显式`--report`时写当前合同；固定Acoular源码身份、整选集不匹配、方法相符/已确认差异/未执行分开。没有运行完整Acoular包、JIT/HDF5或风洞录音。

扩展专题Ⅱ采用稳定身份15，接专题Ⅰ后。24题入口只读；17个16kHz数学WAV共同增益1、32000点即时混合，评分窗1600:30400及缺包窗16000:16800分开；九真实源与整数E/D随清单保存，`--check`严格核18普通成员并完整只读重放。已知100ppm线性SRC改变频响与噪声统计，不是盲时钟估计。当前原合同仅执行paderwasn三个AST原helper和MATLAB静态审读；没有原MATLAB、完整DWACD或无线设备成绩。
