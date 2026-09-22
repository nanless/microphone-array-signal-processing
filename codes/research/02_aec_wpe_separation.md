# AEC、去混响与语音分离：算法、工业实现与源码研究

核实日期：2026-09-22。对应正文 [§6.1](../../chapters/06_aec.md#sec-6-1)、[§7.1](../../chapters/07_wpe-dereverberation.md#sec-7-1) 与 [§8.1～§8.2](../../chapters/08_speech-separation.md#sec-8-1)。本篇按处理对象分开说明算法、源码位置和复现实验；正式版本和许可边界由 [SOURCES.lock.json](../SOURCES.lock.json) 固定。

“外部实现”表示可以找到承担该算法计算的代码，不表示本书已经训练、编译或测完该系统。本篇实际运行的结果单独列出，包括[教学基线测试](../../tests/test_codes_aec_wpe_sep_track.py)与 W01 的独立实现对照；未附执行结果的外部实验均为复现设计。外部代码、权重和数据分别遵守各自条款。

## 1. 先确定信号接口和要解决的问题

AEC 使用已知播放参考，估计并减去由它产生的声学回声；WPE 使用麦克风自己的延迟历史，预测晚期混响；盲分离依靠独立性、源谱或空间模型区分多个未知声源；监督分离还需要训练参考或目标条件。把这些模块都称作“消噪”，会遗漏不同的输入和验收条件。

| 任务 | 必需输入 | 必须保留的状态或条件 | 最小验收对象 |
|---|---|---|---|
| AEC | 同步麦克风与播放参考 | 参考缓冲、滤波器、延迟、自适应控制 | 远端单讲回声、双讲近端保护、路径突变 |
| WPE | 保留复相位的单/多通道 STFT | 延迟历史、预测系数、功率及相关统计 | 拖尾变化、早期目标损伤、启动与静音 |
| 盲分离 | 多通道观测及已声明源数 | 解混矩阵或源 SCM、尺度、跨频点对应 | 分离、尺度恢复、局部最优、病态输入 |
| GSS | 多通道录音与说话人活动标注 | 活动约束、空间聚类、SCM、参考麦 | 活动错误、上下文、跨频点簇标签 |
| 神经分离/TSE | 混合音频，必要时附注册语音 | 模型权重、归一化、上下文、输出槽位 | 域外输入、漏源、目标误抑制、因果性 |
| CSS | 长录音及分块方案 | 重叠窗口、跨块排列、流数 | 换人、静音、重复输出、长时内存 |

同一数值数组的轴序也不能凭名称猜测。本仓库常用 `(通道, 频点, 帧)`；教学 WPE 与掩码 SCM 的接口使用 `(频点, 通道, 帧)`；ESPnet DNN-WPE 接受 `(批, 帧, 通道, 频点)`。转换时应显式标注各轴，并用两通道不同信号检查，不能仅靠形状恰好匹配。

<a id="aec"></a>

## 2. AEC：从自适应滤波到完整音频处理链

### A01　LMS、NLMS 与泄漏更新

NLMS 以参考回归向量的能量归一化更新幅度，便于在不同播放电平间控制步长。泄漏更新额外把滤波器向零收缩，可以限制系数长期漂移，也会引入稳态偏差。它们都要求参考确实覆盖产生回声的播放内容。正文推导与可运行入口是 [§6.1.2](../../chapters/06_aec.md#sec-6-1-2) 和 [aec.py](../array_tutorial/aec.py) 的 `nlms()`。

读码顺序为：参考延迟向量 → 回声估计 → 误差 → 归一化分母 → 冻结条件 → 系数更新。最小实验使用已知短 FIR；先让近端静音检查系数，再在中段加入近端信号并冻结更新。失败实验把纯延迟移到滤波器长度之外：继续增加迭代次数不能恢复不在模型内的抽头。泄漏系数扫描应另报系数偏差，不能只看输出噪声降低。

### A02　FDAF、MDF、PBFDAF 与重叠保存

单块频域滤波把长卷积转成 FFT 乘法；MDF/PBFDAF 把长路径分成多个短分区，在保持较短输入块的同时覆盖长尾。频域乘法天然计算循环卷积，所以丢弃无效输出区与限制滤波器有效时域长度是算法的一部分。[Soo 与 Pang 原论文](https://doi.org/10.1109/29.103078)和 [SpeexDSP `libspeexdsp/mdf.c`](https://gitlab.xiph.org/xiph/speexdsp/-/blob/8e29a256ef0235ebbe7fcb8417b5ac7731eb8307/libspeexdsp/mdf.c)用于分别阅读结构与工程实现；具体检出版本以锁定清单为准。

建议沿 `speex_echo_state_init_mc()`、`speex_echo_cancellation()` 阅读分区数量、参考频谱历史、估计输出、误差频谱和梯度约束。最小实验先关闭更新，把已知 FIR 分区后与直接线性卷积逐样本比较，再启用更新。失败实验分别移除重叠保存的丢弃区、时域约束，观察块边界误差。不要把两种约束错误误判成步长过大。

### A03　AUMDF、分区约束调度与比例分配

Speex 的 `mdf.c` 注释明确采用交替更新的 MDF（AUMDF）。它调度分区约束以控制变换成本，并在代码中调节比例自适应率。比例分配倾向于给较大的路径系数更多更新量，适合考察稀疏路径；分散拖尾的收敛不能由稀疏路径结果代替。

读代码中的 `proportional adaptation rate` 与 `MDF / AUMDF` 段落时，区分“某个分区是否施加约束”和“该分区是否做梯度更新”。最小实验使用相同长度、相同总能量的稀疏与稠密 FIR，记录相对系数误差和每块耗时。失败实验在大抽头位置突变后检查小抽头是否迟迟得不到足够更新。不要把 Speex 的这一具体控制器直接称作所有 PNLMS/IPNLMS 变体的实现。

### A04　FDKF、PFDKF 与对角近似

频域卡尔曼滤波（FDKF）把路径系数当作随时间变化的状态，把近端语音与噪声视为观测干扰，通过误差统计调节更新增益。分区 FDKF 还需管理跨分区状态。工程简化常省略部分频点或系数间相关性，减少矩阵计算；这改变了统计近似，不能用普通 NLMS 的步长参数代替解释。

原始依据为 [Enzner 与 Vary，Signal Processing 2006](https://doi.org/10.1016/j.sigpro.2005.09.013)。本次未确认该原始方法具有可自由再分发的作者官方完整软件，因此保留原理索引，不能把 WebRTC AEC3 或 SpeexDSP 自动标作 FDKF。复现应记录状态转移、过程/观测噪声估计、增益下限、约束、分区和重置。最小实验先固定已知噪声统计，再改变回声路径；失败实验故意低估双讲干扰方差，检查增益与系数误更新。

### A05　DTD：二值判决与连续自适应控制

双讲检测（DTD）用于保护近端语音，不直接生成回声副本。Geigel、归一化相关和相干性方法依赖不同假设；高参考相关性可以是残余回声，也可能伴随路径失配。Speex 的 [`mdf.c`](https://github.com/xiph/speexdsp/blob/master/libspeexdsp/mdf.c)明确使用连续变化学习率，而非独立的二值 DTD。[Valin 2007 原论文](https://people.xiph.org/~jm/papers/valin_taslp2006.pdf)解释它如何利用残余回声和干扰估计。

最小实验由远端单讲、双讲、近端单讲与静音四段组成，分别保存检测量、步长与系数误差。失败实验保持远端播放同时改变路径，比较“误判双讲导致冻结”和“漏判双讲导致误更新”。即使两者输出能量相近，恢复办法也不同。不要仅用一个总体准确率验收 DTD。

### A06　AEC3 的延迟、线性抵消、残余抑制与舒适噪声

WebRTC 的 [`modules/audio_processing/aec3/`](https://webrtc.googlesource.com/src/+/0467d2b91cc20b9b001c2bbb73d43ea6b2491f3e/modules/audio_processing/aec3/)是按模块读工业 AEC 的入口。阅读顺序建议为 `echo_canceller3.cc` → `block_processor.cc` → `render_delay_buffer.cc`/`render_delay_controller.cc` → `subtractor.cc` → `residual_echo_estimator.cc` → `suppression_gain.cc` → `comfort_noise_generator.cc`。这些模块共同处理参考到达、线性残差、剩余回声估计与抑制听感。

复现应先使用工程自带测试和录制/回放入口，固定 APM 配置、采样率、帧组织、输入电平、延迟设置以及参考和采集调用次序。最小实验只更改参考延迟；失败实验引入一次参考丢块，观察延迟状态、线性残差和后滤输出。最终输出安静不等于线性滤波器收敛，至少保留线性段与抑制后两个取点。

### A07　RES/NLP 与近端保护

残余回声抑制（RES）或非线性后处理（NLP）根据未消除回声的估计给输出施加增益。线性路径失配、扬声器非线性与延迟错位都能形成残余，但需要不同诊断。AEC3 的 `residual_echo_estimator.cc` 和 `suppression_gain.cc` 可用于区分估计器与执行增益；Speex 的 [`preprocess.c`](https://github.com/xiph/speexdsp/blob/master/libspeexdsp/preprocess.c)则展示预处理与残余回声信息的连接。

最小实验把线性抵消器输出固定，只扫描后处理强度，分别试听和计量回声泄漏、近端语音损伤及噪声起伏。失败实验让低电平近端辅音与残余回声同时出现，检查词尾是否被截断。此类错误不能靠远端单讲 ERLE 发现，必须同时报告双讲语音的保真与任务指标。

本书的 [E06-04、E06-05](../../chapters/06_aec.md#sec-6-1-17)补充两个可运行的取点检查：同幅副本若反相，相减会把回声能量增为四倍；背景噪声保留时，麦克风输入/输出能量比仍可能为有限值，即使回声分量已经完全抵消。这些是给定模型下的代数结果，不是对 RES 的设备性能测量。运行入口为 [`exercises_enhancement.py`](../examples/exercises_enhancement.py)，固定检查见 [`test_codes_enhancement_round2.py`](../../tests/test_codes_enhancement_round2.py)。

### A08　PNLMS/IPNLMS、子带与非线性路径模型

比例归一化更新改变抽头间的学习分配；子带滤波改变输入相关性与每带更新问题；Volterra、Hammerstein、Wiener 或级联模型改变可表示的输入—回声关系。它们解决的困难不同，不能看见“频域”或“非线性”标签就视作同一种加速方案。出处与模型讨论见 [§6.1.3](../../chapters/06_aec.md#sec-6-1-3)和[§6.1.4](../../chapters/06_aec.md#sec-6-1-4)。

最小复现应每次只改一项：同一线性 FIR 比例分配实验；同一有色参考子带实验；固定路径中插入软饱和的非线性实验。分别记录抽头模型、带间延迟和非线性阶数。失败实验用未见的扬声器削波强度，检查高阶模型过拟合和计算增长。本书没有把这些类别全部伪装成已有工业实现；没有唯一对应源码时明确保留模型与选型依据。

### A09　多播放参考 AEC 与可辨识性

多参考 AEC 为每个播放通道建立一条路径再相加。Speex 的 `speex_echo_state_init_mc(frame_size, filter_length, nb_mic, nb_speakers)`是具体多输入接口。参数中的扬声器数与麦克风数不能互换，播放样本的交织顺序要按头文件核对。

最小实验使用两条独立播放参考与两条已知 FIR；失败实验把两路参考设成完全相同。此时总回声可能抵消得很好，但单独的两条路径不一定可辨识。工程上应同时检查参考相关性、路径估计与重新混音后的恢复，而不是只报告合成回声误差。正文条件见 [§6.1.10](../../chapters/06_aec.md#sec-6-1-10)。

### A10　DTLN-aec：双输入、两阶段表示与流式状态

[DTLN-aec 作者仓库](https://github.com/breizhn/DTLN-aec/tree/9d24e128b4f409db18227b8babb343016625921f)提供 `run_aec.py` 和不同大小的 TFLite 模型。它联合使用麦克风与播放参考；复现时要读取两阶段模型的输入/输出张量、循环状态和块拼接，不能把一个通用单输入降噪器当作 AEC。[ICASSP 2021 论文](https://doi.org/10.1109/ICASSP39728.2021.9413510)用于核对结构。

代码为 MIT；托管模型与示例音频应另查其分发说明。最小实验从一组匹配的 `*_mic.wav` 和 `*_lpb.wav` 开始，固定模型大小、采样率和处理长度。失败实验让参考错位、缺失或短于麦克风，再测路径突变与双讲。下载代码只证明可以检查调用流程；没有权重和运行时环境时不能声称推理已复现。

### A11　NKF-AEC：学习卡尔曼增益的线性 AEC

[NKF-AEC 官方仓库](https://github.com/fjiang9/NKF-AEC/tree/8ac58fb8fb9ced48579f9aa310745c54f98d7e1f)与[论文](https://arxiv.org/abs/2207.11388)给出利用神经网络估计更新增益的路线。仓库明确说明这是线性回声抵消，输入采样率为 16 kHz，显著延迟需要预先补偿。代码入口为 `src/nkf.py`；要追踪网络循环状态和频域滤波状态，而非只观察最后一个波形。

最小实验在已对齐的参考上比较固定路径和突变路径；失败实验加入未建模非线性与参考时移，检查线性限制和对齐依赖。当前没有确认可授予本书再分发权利的独立许可证，因此保留来源与论文索引，不复制源码和 checkpoint。NKF-AEC 与 ASRU 2023 的 NeuralKalman 是不同工作，名称不能混用。

### A12　NeuralKalman、Deep Adaptive AEC 与 DeepVQE

[NeuralKalman 原论文](https://doi.org/10.1109/ASRU57964.2023.10389780)保留频域递推并引入学习模块；[Deep Adaptive AEC](https://minjekim.com/wp-content/uploads/icassp2022_hzhang.pdf)把自适应过程放入可训练系统；[DeepVQE](https://www.isca-archive.org/interspeech_2023/ristea23_interspeech.html)联合回声、噪声和混响处理。这些论文的模块定义不同，不能合写成“网络输出一个掩码”。

截至核实日期，本篇未确认三者都有论文作者公开、许可明确、可完整复现的官方软件。搜索得到的第三方 PyTorch 复现不等于原系统，有些只实现降噪分支。研究时应逐项对应参考编码、对齐、更新器、输出目标和训练损失；最小失败实验覆盖近端静音、播放静音、双讲及非线性。没有原配置和权重时只评价结构，不复述论文速度为本机实测。

### A13　Meta-AF：学习更新器与目录级许可证

[MetaAF 官方仓库](https://github.com/adobe-research/MetaAF/tree/56c4665bdc51c2e0595a7c0cd9b1266408adceff)把更新规则作为可学习对象。读码从 `metaaf/filter.py` 的分块与缓冲开始，再到 `core.py`、`optimizer_gru.py`、`optimizer_fgru.py` 和 `meta.py`，最后看任务相关的 `zoo/`。其[AEC、WPE 与波束实验论文](https://arxiv.org/abs/2204.11942)不能被解释成任意输入都有效的通用优化器保证。

核心 `metaaf/` 使用 University of Illinois/NCSA 许可，`zoo/` 与权重使用 Adobe Research License，整库不得统一记成 MIT 或可自由商用。复现需要匹配 JAX、Haiku 与任务数据，分别记录训练时展开长度和推理时状态。失败实验更换路径速度、频谱及幅度尺度，检查学习更新规则的训练域依赖；本书只索引受限制部分。

<a id="wpe"></a>

## 3. WPE 与联合卷积滤波

### W01　离线 WPE 与 MIMO-WPE

延迟线性预测使用历史多通道复谱解释当前晚期混响，功率权重防止高能量帧完全支配回归。多通道版本的每频点回归维度为“通道数 × 抽头数”，样本不足和共线性会使方程病态。教学入口是 [dereverberation.py](../array_tutorial/dereverberation.py)，参考实现是 [nara_wpe `wpe.py`](https://github.com/fgnt/nara_wpe/blob/master/nara_wpe/wpe.py) 的 `build_y_tilde()` 与 `wpe_v6()` 等实现。

读码时核对历史排序、保护延迟、统计有效区、复共轭与功率下限。最小实验先复算正文单抽头正规方程，再把单通道扩成两个完全相同通道；失败实验使用短于历史窗口的记录、静音频点和病态 SCM。库的稳定求解策略不能替代明确的旁路条件。[原始 MIMO-WPE 论文](https://doi.org/10.1109/TASL.2012.2210879)定义了希望保留的早期成分。

本书另提供已执行的[独立实现对照](../examples/compare_wpe_reference.py)。它使用 nara-wpe 0.0.11 的原始 `wpe_v6()` 与教学 `offline_wpe()`比较，固定种子 20260922、复谱形状 `(4, 2, 192)`、2 个预测抽头、3 帧保护延迟、有效历史统计、无功率平滑且关闭对角加载。两麦、单麦、通道交换和整体缩放四种输入，各运行 1 次和 3 次迭代。

2026-09-22 的本机对照环境为 macOS arm64、Python 3.13.12、NumPy 2.5.3，输入与求解使用 `complex128`。脚本输出运行环境，便于区分后续平台或线性代数库带来的末位舍入差异；本次没有计时或硬件速度比较。

八组对照中，完整历史后的帧最大绝对复数误差约为 `2.69e-15`，最大相对 L2 误差约为 `5.28e-16`。开头 4 帧不参与比较：教学实现旁路，上游允许零补历史后滤波。这是合成复谱上回归求解的一致性检查，不是语音质量、公开数据集或实时性测量；随机复谱不对应任何指定采样率与声学房间。

教学练习 E07-04 另用前缀相同、末帧不同的四帧记录检查未来依赖：一轮离线 WPE 的首个有效输出从 0 变为 −1/9。其回归量虽只含历史，系数却使用整段统计；这说明“预测索引只向过去”不足以证明端到端因果。该题没有使用居中功率平滑，也不把未变化的某个单例当作一般因果性证明。

### W02　逐帧递推 WPE

`nara_wpe/wpe.py` 的 `OnlineWPE` 保存逆相关矩阵、预测抽头、功率和输入缓冲；`step_frame()` 接收 `(频点, 通道)`。该实现先由旧状态输出当前预测残差，再更新状态，这与每次对增长的整段录音重新运行离线 `wpe()`不同。原理出处为[Kinoshita 等，Interspeech 2017](https://www.isca-archive.org/interspeech_2017/kinoshita17_interspeech.html)。

最小实验逐帧输入一个固定记录并保留状态，再检查把同一帧流分成不同外层批次是否改变结果。失败实验中途错误重建 `OnlineWPE`，查看启动段反复出现；另测静音、设备重启和房间突变。遗忘因子与功率估计共同决定跟踪，不应仅报告 `taps` 和 `delay`。

### W03　块在线 WPE 与接口中的限制

`OnlineWPE.step_block()`处理已有历史缓冲形状的块；当前检出源码的 `_get_prediction()`还注明只支持 `block_shift=1`。TensorFlow 路线另有 `tf_wpe.py` 中的 `block_wpe_step()`、`recursive_wpe()`，二者不应凭名字直接互换。见[nara_wpe 官方源码](https://github.com/fgnt/nara_wpe/blob/master/nara_wpe/tf_wpe.py)。

最小复现要先确认块是“送入模型的新样本”还是“含全部历史的窗口”，再比较输出时间戳。失败实验把窗口长度误当帧移，检查是否重复消费或跳过帧。计算实时性时分别报块等待、历史上下文、STFT 和计算时间；有历史记忆不意味着必须等待同样长的未来数据。

### W04　DNN-WPE：掩码或功率网络与解析求解器

[ESPnet `DNN_WPE`](https://github.com/espnet/espnet/blob/master/espnet2/enh/layers/dnn_wpe.py)先估计掩码并形成跨通道功率，再调用 `wpe_one_iteration()`。外层输入为 `(批, 帧, 通道, 频点)`，内部转成 `(批, 频点, 通道, 帧)`。应把 `dnn_wpe.py`、`mask_estimator.py` 和 `wpe.py` 连起来读，才能知道网络实际改了哪个统计量。

最小实验用已知正数功率替换网络输出，检查解析求解，再接入固定 checkpoint。失败实验把掩码设为全零或加入未来语音，验证功率下限与非因果泄漏。该类支持不同掩码估计器，不能仅凭 `iterations=1` 就宣称在线；需要核查网络方向、统计窗口和 padding。

### W05　WPD：同时利用当前通道与延迟历史

WPD 把当前帧与历史帧一起放入无失真滤波问题。它的约束只施加在扩展向量的当前目标分量，历史部分用于削弱可预测晚期混响。出处为[Nakatani 与 Kinoshita](https://arxiv.org/abs/1908.02710)；实现入口为 [ESPnet `beamformer.py`](https://github.com/espnet/espnet/blob/master/espnet2/enh/layers/beamformer.py) 的 WPD 相关函数及 `dnn_beamformer.py` 的类型选择。

读码时先查 `wpd` 分支怎样构造延迟堆叠，再查功率倒数、参考通道和解线性方程。最小实验用正文块对角协方差验证历史权重为零，再加入当前—历史互相关；失败实验使功率权重接近零或增大历史阶数至统计量不足。WPD 与串联 WPE→MVDR 的计算和假设有关联，但不能把两个独立增益相加当作联合结果。

E07-05 已把第二种情形写成四维可执行手算：当前—历史相关使权重为 `[0.4, 0.6, -0.2, 0]`，无失真约束为 1。给定协方差上的目标值为 0.6，强制历史权重为零时为 2/3。这里比较的是同一矩阵下两个约束集合的最小值，不是音频去混响或识别收益；逐行求解见 [第 7 章练习](../../chapters/07_wpe-dereverberation.md#sec-7-1-2)。

### W06　AR-FastMNMF 与联合去混响/分离

AR-FastMNMF 把自回归混响模型和多源空间/谱模型联合估计，避免把分离与拖尾建模视作互不影响的两个估计问题。[ICASSP 2021 论文](https://ieeexplore.ieee.org/document/9414857)与[作者 `SoundSourceSeparation`](https://github.com/sekiguchi92/SoundSourceSeparation/tree/897fe87fea3d85a243d8a3fd36c2232bb0548ad3)中的 `src/`用于研究模型更新顺序；`FastBSSD.py`还将 AR、MA、ARMA 等模型区分。

作者仓库仅允许规定范围内的学术研究，工业用途或修改需额外许可，因此本书不把它作为可自由纳入产品的代码。最小复现应先用单源/无混响退化例，再与相同 STFT、相同初始化的级联方案比较；失败实验测欠定混合、近共线通道和移动源。联合优化增加了参数和局部最优，模型更复杂不自动保证结果更好。

<a id="bss"></a>

## 4. 盲分离、空间混合模型与 GSS

### B01　FDICA 与跨频点排列

频域独立成分分析（FDICA）逐频点估计瞬时解混；独立估计的频点可能以不同顺序输出同一组源。参考代码为[ssspy `bss/fdica.py`](https://github.com/tky823/ssspy/blob/38b9389e8b1914422561f1936d9b28d042d62d2c/ssspy/bss/fdica.py)，并需查看 `algorithm/` 的排列和尺度处理。它适合帮助理解为何单频点独立性还不足以得到完整语音。

最小实验对双源多频点混合故意交换部分频点的输出；即使每频点都完成分离，重构波形仍不属于单一说话人。失败实验让两个声源的包络很相似，检查基于相关性的排列方法。频点排列与最终两个输出槽位对参考的 PIT 排列是两个不同问题。

### B02　AuxIVA、IP、ISS 与尺度恢复

AuxIVA 用一个源跨频点的联合模型约束依赖，辅助函数更新避免手工选择普通梯度步长。迭代投影（IP）与迭代源导向（ISS）是不同更新实现，应分别记录，而不是都写成“用了 IVA”。[ssspy `bss/iva.py`](https://github.com/tky823/ssspy/blob/38b9389e8b1914422561f1936d9b28d042d62d2c/ssspy/bss/iva.py)给出 IP1、IP2、ISS1、ISS2 与 IPA 的选项，[Ono 2011 原论文](https://doi.org/10.1109/ASPAA.2011.6082320)说明辅助函数思想。

读码顺序为源代价 → 加权 SCM → 空间更新 → 归一化 → projection-back。最小实验采用确定混合和已知两源，固定初值比较代价与分离；失败实验使用秩亏混合。恢复参考麦的尺度需要独立步骤，SI-SDR 对尺度不敏感，不能用其通过来证明波形幅度已经恢复。

E08-04 对人为指定的解混矩阵演示回投影：原输出 `[2s, -3u]` 乘参考麦系数 `[1/2, -1/6]` 后成为 `[s, 0.5u]`。它恢复参考麦源图像而非统一干声尺度，且不把给定解混矩阵的代数操作冒充 AuxIVA 估计。E08-05 则单独检查掩码 SCM 的分母地板：普通正比例缩放不改变平均值，进入地板后不再保持该性质。两题的输入、答案和边界见 [第 8 章练习](../../chapters/08_speech-separation.md#sec-8-1)。

### B03　OverIVA、FIVE 与过定提取

麦克风数多于目标源数时，OverIVA 利用低维目标子空间与背景模型，FIVE 面向单目标提取，不必总是先做任意降维再调用方阵 IVA。[piva 官方仓库](https://github.com/fakufaku/piva)包含这些方法；[pyroomacoustics `auxiva.py`](https://github.com/LCAV/pyroomacoustics/blob/v0.10.0/pyroomacoustics/bss/auxiva.py)也需查看 `n_src` 分支，确认当前算法实际的源数条件。

最小实验保持两源、从两麦增加到四麦，比较背景建模与直接选两麦。失败实验把实际第三个强源误当背景，检查目标提取质量与输出身份。piva 为 GPL-3.0，本书将其与宽松许可证实现分别列出，不能把不同分发条件合并成一个“免训练库”。

### B04　ILRMA 的谱模型与空间更新

ILRMA 把独立向量分析与非负矩阵分解（NMF）结合，让每个源的频谱功率具有低秩结构。[Kitamura 等原论文](https://doi.org/10.1109/TASLP.2016.2577880)、[pyroomacoustics `bss/ilrma.py`](https://github.com/LCAV/pyroomacoustics/blob/v0.10.0/pyroomacoustics/bss/ilrma.py)和[ssspy `bss/ilrma.py`](https://github.com/tky823/ssspy/blob/38b9389e8b1914422561f1936d9b28d042d62d2c/ssspy/bss/ilrma.py)分别提供模型定义和可检查实现。

读码先辨认频率基与时间激活的维度，再读乘法更新、空间解混、功率下限与尺度补偿。最小实验固定初值扫描 NMF 基数；失败实验取很短的记录和近似相同的两个源谱，观察模型不能唯一解释混合。基数增加可能拟合噪声与样本波动，不能仅凭训练代价更低认定分离更好。

### B05　MNMF 的满秩空间协方差

MNMF 为每个源建模满秩空间协方差，能表达比单导向矢量更复杂的传播和欠定混合。[Sawada 等 2013 论文](https://doi.org/10.1109/TASL.2013.2239990)定义多通道模型；[ssspy `bss/mnmf.py`](https://github.com/tky823/ssspy/blob/38b9389e8b1914422561f1936d9b28d042d62d2c/ssspy/bss/mnmf.py)提供 `GaussMNMF`。

读码需一起查看谱参数、源 SCM 及多通道维纳重构。最小实验用两个通道、三个源检验输出形状、模型代价和重复初值的差异；失败实验让两个源方向和谱形同时接近。能够写出欠定模型不等于数据足以识别所有源，应报告初始化种子、分量重置和多次运行分散程度。

### B06　FastMNMF、FastMNMF2 与联合对角化

FastMNMF 以可联合对角化的空间协方差降低反复处理满矩阵的代价，FastMNMF2 改进相应空间参数化。可直接读取已锁定的 [pyroomacoustics `fastmnmf.py`](https://github.com/LCAV/pyroomacoustics/blob/v0.10.0/pyroomacoustics/bss/fastmnmf.py)和[`fastmnmf2.py`](https://github.com/LCAV/pyroomacoustics/blob/v0.10.0/pyroomacoustics/bss/fastmnmf2.py)，其文件许可与仅学术用途的作者整库分别判断。[原论文](https://arxiv.org/abs/1903.03237)解释联合对角化假设。

最小实验在相同 STFT、源数、NMF 基数和初值下比较迭代耗时及分离，不把 CPU 与 GPU 实现直接按秒排名。失败实验使混合 SCM 秩亏并比较不同加载/精度；双精度的较慢计算可能避免数值失效。输出 `mic_index` 决定重构哪支麦克风的源图像，不能与干声参考混淆。

### B07　TRINICON 的时域块自适应

[pyroomacoustics `bss/trinicon.py`](https://github.com/LCAV/pyroomacoustics/blob/v0.10.0/pyroomacoustics/bss/trinicon.py)提供依据[Aichner 等 2006 表 1](https://doi.org/10.1016/j.sigpro.2005.06.022)的时域块方法，并明确固定两个输出通道。输入轴序为 `(通道, 样本)`，与频域 BSS 接口不同。

最小实验以双源卷积混合检查滤波器长度、块长、在线重叠、离线内迭代和遗忘设置；失败实验在第三源出现或路径变化后检查重适应。滤波器覆盖的历史长度、每次读入块与首个可用输出的等待应分开计量。该具体实现存在于本地源码缓存，因此不应继续把 TRINICON 写成完全没有可用代码的原理索引。

### B08　cACGMM：方向统计、活动约束与数值稳定

复角中心高斯混合（cACGMM）对单位化的复向量建模，得到方向上的软聚类。[Ito 等原论文](https://doi.org/10.1109/EUSIPCO.2016.7760429)与[pb_bss `distribution/cacgmm.py`](https://github.com/fgnt/pb_bss/blob/10acc347fc9ea21e3d312806a0bd751d0d0af183/pb_bss/distribution/cacgmm.py)是模型和 EM 入口。继续看底层 cACG 密度、固定点更新和排列工具，不能仅复制 E 步。

最小实验先用两个不同复方向生成单位向量，检查后验归一和分量置换，再用未归一的原 STFT 构造 SCM。失败实验含低能量点、空活动分量和近奇异形状矩阵。用于聚类的方向外积没有原始功率，不能直接当作波束形成的功率 SCM。

### B09　GSS：活动日志、WPE、聚类与波束形成

GSS 用说话人活动约束空间聚类，随后由掩码估计 SCM 并生成每个目标的音频。它不等于只运行 cACGMM；活动标注、上下文和目标选择共同决定系统。[原始 CHiME-5 工作](https://www.isca-archive.org/chime_2018/boeddecker18_chime.pdf)与[GPU-GSS 论文](https://www.isca-archive.org/interspeech_2023/raj23_interspeech.pdf)应按各自实验解释。

读 [`desh2608/gss`](https://github.com/desh2608/gss/tree/10fad18cae85e2e4342c77421abc70c9c5da23ed)时，从 `gss/core/enhancer.py` 串到 `gss/wpe/`、`gss/cacgmm/` 和 `gss/beamformer/`，再看 `recipes/` 的清单生成。最小实验是一段同步多通道 WAV 加 RTTM；失败实验平移 RTTM、删除短活动或把噪声当成说话人。需分别报告活动错误和分离错误，不能全部归因于波束。

### B10　GPU-GSS 的批处理与资源边界

GPU-GSS 把频点、段及相同目标的计算合并，提高 GPU 利用率。批处理策略会同时影响上下文复用、显存与输出片段，`max-batch-duration`、`max-segment-length` 和 `context-duration`是不同参数。[官方使用说明](https://github.com/desh2608/gss/blob/10fad18cae85e2e4342c77421abc70c9c5da23ed/README.md)给出 Lhotse、CuPy 与 CUDA 前提。

最小实验保持音频/活动/算法迭代相同，只改变批处理量，检查音频和时间戳一致性；失败实验增加通道数与长段，记录峰值显存和 OOM 前的工作点。论文加速数值依赖指定硬件与基线，不应转写成所有会议录音的固定倍数。独立 [CuPy WPE 子项目](https://github.com/desh2608/wpe/tree/bd2857b5b8de36df4f436a93574c088bea142042)只包含 GSS 需要的 WPE 子集，不能代替 nara_wpe 全部在线接口。

<a id="neural"></a>

## 5. 神经分离、目标提取与连续输出

### N01　Conv-TasNet：编码窗、TCN 与因果配置

Conv-TasNet 学习短窗编码/解码器，并在编码域估计掩码。[正式论文](https://doi.org/10.1109/TASLP.2019.2915167)与[Asteroid `models/conv_tasnet.py`](https://github.com/asteroid-team/asteroid/blob/master/asteroid/models/conv_tasnet.py)用于核对前端和 TCN 分离器；还要读 `masknn/` 中的卷积、归一化和因果开关。

最小复现先固定源数与官方 recipe，记录采样率、编码核、步长、堆叠次数和 normalization。失败实验改变未来帧但保留过去，检查所谓因果配置是否真的不改变过去输出；另以短于编码窗的音频检查 padding。只提供网络构造类并不意味着已有匹配的训练权重。

### N02　DPRNN：块内、块间与重叠重构

DPRNN 把长序列划成重叠块，分开处理局部和跨块关系。[论文](https://doi.org/10.1109/ICASSP40776.2020.9054266)与[Asteroid `models/dprnn_tasnet.py`](https://github.com/asteroid-team/asteroid/blob/master/asteroid/models/dprnn_tasnet.py)需结合双路径 mask 网络阅读。块大小、块间循环方向和归一化决定能否在线。

最小实验用脉冲或已知索引序列检查分块/合并是否保留顺序，再做双源分离。失败实验在跨块边界换人，并测极短、非整块长度的输入。DPRNN 的分块主要是建模和计算设计，不能据此自动宣称它已经解决长会议的输出身份连续性。

### N03　SepFormer：双路径注意力与整句上下文

SepFormer 把双路径中的序列模型换为注意力结构。[正式论文](https://doi.org/10.1109/ICASSP39728.2021.9413901)与[SpeechBrain `lobes/models/dual_path.py`](https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/lobes/models/dual_path.py)应配合 `recipes/WSJ0Mix/separation/` 中的配置及训练入口阅读。配置、模型与 checkpoint 必须匹配。

最小复现只使用一个固定公开模型的匹配采样率与源数；失败实验延长音频，记录内存和输出槽位变化。注意力读到未来、全局归一化读完整句以及外层分块拼接分别检查。模型名和“低 RTF”不能代替前瞻、首帧时间和长录音峰值内存。

### N04　TF-GridNet：频率、时间与多通道输入

[ESPnet `tfgridnet_separator.py`](https://github.com/espnet/espnet/blob/master/espnet2/enh/separator/tfgridnet_separator.py)明确标为离线 `TFGridNet`，输入可为 `(批, 样本, 麦克风)`，固定 `n_imics`。其短时变换、频率/时间建模和注意力应与[论文](https://doi.org/10.1109/ICASSP49357.2023.10094992)及多通道扩展分开核对。

最小实验从官方配置检查输入标准差归一化、参考麦、输出源数和 STFT 长度；失败实验交换或缺失一个麦克风、改变阵列几何，并用静音检测归一化稳定性。源码的不同版本/变体不能共用一句“TF-GridNet 支持任意阵列”。推理返回多个单通道波形，也不等于每个输出永久绑定一个身份。

### N05　S4M：可读的骨干不等于完整训练配方

[S4M 官方仓库](https://github.com/JusperLee/S4M)提供 `S4M.py`、`s4.py`、`s4d.py`、`lssl.py` 和 `sashimi.py` 等结构文件，代码许可为 MIT。核实版本没有完整的数据准备、训练入口和配套 checkpoint，不能写成已经能一条命令复现论文全套训练。[Interspeech 2023 原论文](https://doi.org/10.21437/Interspeech.2023-696)仍是模型与实验设置的依据。

可以进行的最小检查是按论文输入维度追踪各尺度特征与状态空间模块；若补写训练器，必须明确这是本书适配代码并单列差异。失败实验包括非整倍数长度、长音频内存增长和未来信息依赖。状态空间模型可以提供不同计算结构，但本身不保证整个编码—解码系统严格因果。

### N06　SPMamba 与双向状态空间模型

[SPMamba 官方仓库](https://github.com/JusperLee/SPMamba/tree/f939f60a10db8a66aa69ec09685684307af47412)提供 `audio_train.py`、`audio_test.py`、`configs/` 和 `look2hear/`。与 S4M 相比，它基于 TF-GridNet 路线替换双向时序模块；[论文预印本](https://arxiv.org/abs/2404.02063)明确利用过去与未来上下文，不能把选择性状态空间的线性复杂度解释为流式因果性。

最小复现固定某个配置、混合生成方式与软件环境，再追踪时间和频率分支；失败实验变长输入和未来扰动，并核对 Mamba/CUDA 算子的硬件可用性。代码采用 Apache-2.0，模型资产与语料另核；CPU 可以读取代码不表示 CUDA 扩展一定能直接移植到当前 CPU。

### N07　Mamba-TasNet 与 Dual-Path Mamba

[作者仓库](https://github.com/xi-j/Mamba-TasNet/tree/a35c692f27213781a11b1606c375cda1e1f0fb62)提供 `train_wsj0mix.py`、`hparams/WSJ0Mix/`、`modules/` 与 `inference.ipynb`，对应[Mamba-TasNet/Dual-Path Mamba 论文](https://arxiv.org/abs/2403.18257)。它与 SPMamba 不是同一个实现，不能交换配置或权重。

先核对 WSJ0-2Mix 生成、动态混合、精度和模型大小，再检查双路径块和方向。失败实验比较单精度与低精度的非有限输出、长序列和源数失配。仓库 GPL-3.0 与 WSJ0 语料授权分别处理；GPL 代码可以作为独立本地研究源码取得，但与本书代码集成和再分发时须满足相应义务。checkpoint 不因代码可取得就自动进入本书分发范围。

### N08　SpeakerBeam：注册条件和评估许可

[SpeakerBeam](https://github.com/BUTSpeechFIT/speakerbeam/tree/91af02cc617afa35fedfbdbf32533012cd0a8672)以注册语音指定希望提取的说话人。[2019 论文](https://doi.org/10.1109/JSTSP.2019.2922820)与其时域扩展需要区分。仓库入口为 `src/models/convtasnet_informed.py`、`adapt_layers.py`、`src/datasets/librimix_informed.py`，配方位于 `egs/libri2mix/`。

该仓库的 `LICENSE.txt` 为评估协议，限定内部测试、分析和评估，并限制修改与再分发；它不是普通开源许可证，本书不复制代码。最小实验应固定注册语音列表和测试混合，不使用目标测试语句本身当注册条件。失败实验使用另一人、跨设备注册、目标缺席和近似声纹，分别记录目标误抑制与非目标泄漏。

### N09　CSS：跨块输出排列和长时状态

CSS 在长录音上产生若干不重叠输出流，活动人数与槽位对应会变化。[LibriCSS 论文](https://arxiv.org/abs/2001.11482)定义任务；[NOTSOFAR-1 官方系统](https://github.com/microsoft/NOTSOFAR1-Challenge)提供连续分离、识别和分割的工程研究入口，代码 MIT 与数据 CC-BY-4.0 分开标注。

NOTSOFAR-1 的固定源码从 `run_training_css_local.py` 进入 `css/training/train.py`，网络封装位于 `css/training/conformer_wrapper.py`；推理入口 [`run_inference.py`](https://github.com/microsoft/NOTSOFAR1-Challenge/blob/6f58e08b008f7530ba4141f0aeb02447c70b6fd7/run_inference.py)按配置选择单通道或多通道会话，再调用完整流水线。其默认调试配置虽只筛选一个会话，入口仍先请求下载整个指定开发集和模型，因此不能把直接运行脚本当成无下载的最小检查。本书已取得并核对源码入口，未执行这些自动下载或训练；数据、模型、识别后端与计算资源仍是额外前提。

最小复现需要同一长录音的窗口、步幅、重叠匹配、泄漏处理及最终评分规则；不能仅对每块分别取最优 PIT 后拼接。失败实验在静音之后更换说话人，让输出槽位交换，检查跨块匹配是否误连。cpWER、ORC-WER 和说话人归属 WER 的映射对象不同，应连同分段与转录协议固定。

### N10　SGMSE+、StoRM 与扩散迭代

分数模型沿噪声尺度学习条件生成，推理要经过多次采样或数值求解；预测—校正、随机种子和步数会影响速度与输出。官方入口为[SGMSE](https://github.com/sp-uhh/sgmse)和[StoRM](https://github.com/sp-uhh/storm)，相应论文与固定提交见[第三方来源说明](../THIRD_PARTY.md)。从各自 `enhancement.py` 读到 `sgmse/model.py`：SGMSE 的 `ScoreModel` 与 StoRM 的 `StochasticRegenerationModel`不是同一推理流程，不能仅按“扩散增强”标签共用 checkpoint。

最小复现应固定目标是增强还是去混响、采样器、步数、SDE/噪声日程、归一化和权重。失败实验包含静音、短脉冲、未见噪声与目标缺失，检查是否生成输入没有支持的音节。报告随机重复的离散程度与完整采样耗时；不能只计一次网络前向。代码 MIT 不自动授予训练语料和权重的再分发权。

### N11　ArrayDPS 与模型驱动的扩散分离

[ArrayDPS 官方仓库](https://github.com/ArrayDPS/ArrayDPS/tree/750ac2b7c75458f4ca5bad203dafda528f575e55)将多通道观测模型与扩散先验结合，来源及固定版本见[THIRD_PARTY.md](../THIRD_PARTY.md)。从 `separate.py` 到 `src/sampler.py` 的 `Sampler`，分别识别扩散先验、观测一致性项和阵列/传播参数；生成得像语音并不证明它来自指定声源。

最小实验固定双源混合与阵列模型，保存每次采样种子和条件参数；失败实验改变混响模型、源数或通道同步，观察数据一致性与听感是否冲突。SMS-WSJ 的生成程序不授予 WSJ 录音使用权，checkpoint 与示例音频也应独立检查。本书没有下载训练集、权重或把论文结果冒充已执行实验。

### N12　AudioSep：文本条件与音频预处理

[AudioSep 官方仓库](https://github.com/audio-agi/audiosep/tree/944583f18b84589dc965de3ad77525c945334252)的 `pipeline.py`先用 `build_audiosep()`加载分离模型和 CLAP 查询编码器，再由 `separate_audio()`将文本条件送入分离器。[正式论文](https://doi.org/10.1109/TASLP.2024.3520017)的目标是文本查询声音分离，不能代替注册语音 TSE 的身份评测。锁定入口会将音频转为单声道 32 kHz，不能把原始多通道空间信息保留作为默认假设。

最小复现用同一音频改变查询词并保存实际重采样参数、配置、checkpoint 与分块开关。失败实验查询录音中不存在的声音，以及两个同类声源，观察误提取和身份歧义；导出 `int16` 前还应检查浮点幅度，避免超范围转换带来的失真被误认为模型失真。代码 MIT 与托管主模型、媒体数据的许可分别记录，本书未运行该权重的推理。

## 6. 复现实验的共同记录表

| 记录组 | 必须写出的内容 | 缺失后不能判断的问题 |
|---|---|---|
| 输入 | 文件摘要、采样率、通道/源数、参考麦、目标定义 | 是否在处理相同任务 |
| 时间 | STFT窗/帧移、padding、参考延迟、前瞻、块拼接 | 是算法失真还是时移/截断 |
| 状态 | 初始化、跨块保留、冻结、重置、异常恢复 | 能否用于持续录音 |
| 数值 | 精度、下限、加载、停止条件、种子 | 是否由病态矩阵或随机性造成差别 |
| 模型 | 源码提交、配置、权重摘要、预处理 | 能否复现相同网络 |
| 指标 | 取点、活动区间、对齐、排列、输入基线 | 数字是否可比较 |
| 资源 | 硬件、线程、批量、计时范围、峰值内存 | 是否满足目标设备预算 |
| 权利 | 代码、权重、数据、第三方组件的独立许可 | 哪些资产可以使用或分发 |

建议按“解析小例 → 合成回归 → 固定公开录音 → 目标设备长时录音”逐步增加复杂度。合成例负责证明实现符合所写模型，公开录音用于比较可复现行为，设备录音用于覆盖参考抽头、时钟、驱动、非线性和操作场景。三类证据不能互相替代。
