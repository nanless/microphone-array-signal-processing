# AEC、去混响与语音分离：算法、工业实现与源码研究

原核实日期：2026-09-22；AEC 工业接口与配对数据说明复核日期：2026-09-23。对应正文 [§6.1](../../chapters/06_aec.md#sec-6-1)、[§7.1](../../chapters/07_wpe-dereverberation.md#sec-7-1) 与 [§8.1～§8.2](../../chapters/08_speech-separation.md#sec-8-1)。本篇按处理对象分开说明算法、源码位置和复现实验；正式版本和许可边界由 [SOURCES.lock.json](../SOURCES.lock.json) 固定。

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

### A02　FDAF、频域多抽头 NLMS、MDF/PBFDAF 与重叠保存

单块频域滤波把长卷积转成 FFT 乘法；MDF/PBFDAF 把长路径分成多个短分区，在保持较短输入块的同时覆盖长尾。固定一个频点，若“频域多抽头 NLMS”的 $P$ 个抽头正是当前和过去 $P-1$ 个参考块，且采用[正文式(6-3)](../../chapters/06_aec.md#sec-6-1-3)的同一误差谱和归一化分母，那么它与该式的 PBFDAF 更新逐分量相同，不另列为一种算法。名称本身不保证论文与产品使用相同步长；[Páez Borrallo 与 García Otero 的 1992 年 PBFDAF 原论文摘要](https://www.sciencedirect.com/science/article/pii/016516849290077A)说明该版本还考虑输入谱均匀度。

频域乘法计算循环卷积，因此重叠保存输出只保留有效区；若要求每个分区始终对应固定长度的时域 FIR，还要检查权重初始化及梯度约束。无梯度约束是可研究的合法变体，不等于连有效输出区也可以省略。[Shynk 1992，§IV、图 5 与式(30)](https://course.ece.cmu.edu/~ece792/handouts/Shynk92.pdf)明确区分有效区误差与梯度约束；[Soo 与 Pang 原论文](https://doi.org/10.1109/29.103078)和 [SpeexDSP `libspeexdsp/mdf.c`](https://gitlab.xiph.org/xiph/speexdsp/-/blob/8e29a256ef0235ebbe7fcb8417b5ac7731eb8307/libspeexdsp/mdf.c)可分别阅读结构与工程实现，具体检出版本以锁定清单为准。

读码时沿 `speex_echo_state_init_mc()`、`speex_echo_cancellation()` 检查分区数量、参考频谱历史、估计输出、误差频谱和梯度约束。若要比较两个都叫“频域多抽头 NLMS”或 PBFDAF 的实现，应逐项核对块长、FFT 标度、块边界、初值、有效区误差的补零位置、瞬时或平滑功率、归一化地板、步长/双讲控制及约束频率；只看到相同的 $E X^*/(\sum|X|^2+\delta)$ 不足以断定逐块权重轨迹相同。同配置时，也不存在由重命名带来的运算量或延迟优势；正文算例 6-2 的倍数是相对逐样本**时域** NLMS。

最小实验先关闭更新，把已知 FIR 分区后与直接线性卷积逐样本比较，再启用更新并保存每块权重。故障注入分别测试不丢弃循环卷积的无效输出区，以及关闭梯度约束：前者会破坏对应位置的线性卷积输出；后者可能改变权重轨迹和稳态误差，却不保证每组输入都出现块边界错误。另需区分一般 STFT 域同频带跨帧滤波：任意分析、合成窗下，忽略跨频带项未必与重叠保存的全带时域 FIR 逐样本等价，见 [Avargel 与 Cohen 2007，§II](https://webee.technion.ac.il/Sites/People/IsraelCohen/Publications/TASL_May2007.pdf)。

上述最小实验现有[教学实现](../array_tutorial/aec_partitioned.py)、[两块手算演示](../examples/aec_partitioned_demo.py)与[独立测试](../../tests/test_codes_aec_partitioned.py)：$N=P=2$ 时，第 1 块第一分区候选的非法第 3 个时域抽头为 $1/3$，约束后归零，第二分区合法首抽头保留 $1/3$。测试另以直接时域卷积校验固定已知路径，并覆盖跨块状态、零参考、短末分区、冻结和数值溢出。这证明教学代码在这些输入下遵循本书约定，不是 Speex AUMDF 的复现，也未提供真实语音收敛或实时性对比。

### A03　AUMDF、分区约束调度与比例分配

Speex 的 `mdf.c` 注释明确采用交替更新的 MDF（AUMDF）。它调度分区约束以控制变换成本，并在代码中调节比例自适应率。比例分配倾向于给较大的路径系数更多更新量，适合考察稀疏路径；分散拖尾的收敛不能由稀疏路径结果代替。

读代码中的 `proportional adaptation rate` 与 `MDF / AUMDF` 段落时，区分“某个分区是否施加约束”和“该分区是否做梯度更新”。最小实验使用相同长度、相同总能量的稀疏与稠密 FIR，记录相对系数误差和每块耗时。失败实验在大抽头位置突变后检查小抽头是否迟迟得不到足够更新。不要把 Speex 的这一具体控制器直接称作所有 PNLMS/IPNLMS 变体的实现。

### A04　RLS、Kalman、FDKF 与分区状态空间 AEC

**先分清估计目标。** [正文式(6-5)～式(6-7)](../../chapters/06_aec.md#sec-6-1-3)的实数时域 RLS 维护指数加权参考相关矩阵的逆，精确求解含指定初值约束的加权最小二乘。它用当前样本的先验残差更新路径，并不识别近端语音。有限的 $P_{-1}=\delta^{-1}I$ 与目标里的 $\lambda^{n+1}\delta\|w-w_{-1}\|^2$ 必须配套；只写未正则化的目标、却用有限初始逆矩阵，前几个样本就不等价。教学[逐样本实现](../array_tutorial/aec_rls.py)、[两抽头演示](../examples/aec_rls_demo.py)和[独立批量最小二乘测试](../../tests/test_codes_aec_rls.py)可核对这个对应关系，均不是产品级 AEC。

RLS 对 $L$ 抽头需要 $L\times L$ 逆相关状态，常规每样本更新为 $O(L^2)$；参考长时间为零时，按遗忘递推仍会放大逆相关状态。双讲时继续更新会把近端语音混进路径拟合，冻结更新又可能漏掉同时发生的回声路径变化。因此遗忘因子、路径突变检测和双讲控制必须一起观察，不能只报一条收敛曲线。平方根、QR 与 Householder 是数值稳定性路线，不表示它们都与原递推有相同运算量。[MathWorks 官方 RLSFilter 文档](https://www.mathworks.com/help/dsp/ref/dsp.rlsfilter-system-object.html)给出这些方法和锁定系数接口；它只证明工具箱提供 RLS，不证明 MathWorks 的 [AEC 示例](https://www.mathworks.com/help/audio/ug/acoustic-echo-cancellation-aec.html)采用了 RLS。

本仓库的短实数 RLS 为防止浮点更新后逆相关矩阵失去正定性，**每个已接纳样本**另用 Cholesky 检查整矩阵，实际教学代码因此含 $O(L^3)$ 检查成本；上段 $O(L^2)$ 只指矩阵求逆引理更新的代数主项。不可用这份安全检查开启后的运行耗时代表优化的工业 RLS。[固定种子、分段真值的 RLS/Kalman 对照](../examples/aec_rls_kalman_comparison.py)报告先验回声预测误差、路径误差及 Kalman 增益，不把已知近端注入时的总残差功率误称为 ERLE。

**快 RLS 不等于普通频域 NLMS。** 频域多抽头 NLMS/PBFDAF 使用参考能量归一化梯度；经典 RLS 使用输入相关矩阵的逆来改变更新方向。Schneider 与 Kellermann 的 [GFDAF 分析，2016，§3～4](https://doi.org/10.1186/s13634-015-0302-2)把该类广义频域算法联系到**块 RLS 的近似**，而不是证明所有 PBFDAF 都是精确 RLS。面向长回声路径还有 [Cioffi 与 Kailath 1984 的快速横向滤波器](https://doi.org/10.1109/TASSP.1984.1164334)及 [Slock 与 Maouche 1994 的 FFT/低位移秩 RLS](https://doi.org/10.1016/0165-1684(94)90018-3)等结构化路线；名称里的“快速”必须回到各论文的输入模型、初始化与复杂度条件核实，不可把本书全矩阵 RLS 的 $O(L^2)$ 原样用于这些变体。

**Kalman 的额外模型。** [正文式(6-8)～式(6-10)](../../chapters/06_aec.md#sec-6-1-3)把真实路径当作变化状态，用过程协方差 $Q$ 描述路径不确定度，用观测方差 $\Psi$ 描述近端语音和其他干扰；$P$ 是路径估计误差协方差，不是 RLS 定义下的逆参考相关矩阵。固定已知 $Q,\Psi$ 时可以复算增益，但残差变大本身不能证明应增大 $Q$：双讲和路径突变都会增大残差，前者通常要求减少对麦克风观测的信任，后者可能要求提高对新路径的适应。将 $A=I,\Psi=1,P_n^-=P_{n-1}/\lambda$ 特设于完整矩阵 Kalman，才与普通遗忘 RLS 的增益同式；这等价于依赖当前 $P$ 的 $Q_n=(\lambda^{-1}-1)P_{n-1}$，不是任意固定过程噪声 Kalman。

两个层次的可运行核查彼此独立：[式(6-9)单复频点演示](../examples/aec_kalman_scalar_demo.py)与[分数测试](../../tests/test_codes_aec_kalman_scalar_demo.py)给出 $K=3/7$、后验路径 $22/35$、方差 $3/35$；把已知 $\Psi$ 从 $0.4$ 改为 $4$，增益变为 $3/16$，复数 $X=i$ 时 $K=-i/2$。[两抽头实数矩阵演示](../examples/aec_kalman_matrix_demo.py)与[独立测试](../../tests/test_codes_aec_kalman_matrix.py)用 Joseph 形式核对 $P^-=\operatorname{diag}(1,4)$、$x=[1,1]^\top$、$\Psi=1$ 时的 $K=[1/6,2/3]^\top$ 和后验非对角项 $-2/3$。第二步将状态协方差强行改成对角，创新方差会从 $9/2$ 误成 $19/6$。它说明**近似改变计算**，并不测得真实录音性能；两个实现都不含 FFT、分区、在线噪声估计、延迟搜索、双讲检测或残余抑制。

**从矩阵 Kalman 到 FDKF/PBFDKF。** [Enzner 与 Vary，Signal Processing 2006，§2～4](https://doi.org/10.1016/j.sigpro.2005.09.013)建立声学状态空间频域滤波；[作者说明](https://homepages.ruhr-uni-bochum.de/gerald.enzner/StateSpaceFDAF.html)明确它是对精确 Kalman 的近似，并区分线性抵消输出和依据状态不确定度形成的后滤波输出。读论文时要跟踪频域路径状态、线性卷积有效区、观测扰动与状态协方差的近似，不能见到单频点式(6-9)就声称复现原论文。分区延伸可从 [Kuech、Mabande 与 Enzner，ICASSP 2014](https://doi.org/10.1109/ICASSP.2014.6853806)查起。[Zhu 等，Interspeech 2021，§2～3、图 4～5](https://www.isca-archive.org/interspeech_2021/zhu21d_interspeech.pdf)的 VD-PBFDKF 只保留同参考声道、同分区的协方差块；SD-PBFDKF 在同分区还保留不同参考声道间的块，两者仍丢弃跨分区块。多麦、多播放参考时，“对角”必须说明是频点、分区、声道还是矩阵元素层面的对角。论文实验的排序不自动外推到其他房间或双讲安排。

**可读源码的精确入口与边界。** 下表按 [锁定版本](../SOURCES.lock.json) 指向实际算法入口；“仅索引”表示本书尚未下载、运行或验证该方法的数值输出，源码可见不等于获得示例音频或模型权重的再分发权。

| 来源与入口 | 实际实现/值得看的状态 | 已核实的缺口 |
|---|---|---|
| [pyroomacoustics v0.10.0 `adaptive/rls.py`](https://github.com/LCAV/pyroomacoustics/blob/0dd39f2614b7fc44b2cc63dbe7d60f4641068890/pyroomacoustics/adaptive/rls.py) | MIT；通用实数 RLS 与 BlockRLS，已取得源码 | BlockRLS 按块末更新；无播放延迟、DTD、RES，不是完整 AEC |
| [pyaec `time_domain_adaptive_filters/rls.py`](https://github.com/ewan-xu/pyaec/blob/5b9c02c57075d790b7df8652884618189d49bbc4/time_domain_adaptive_filters/rls.py) | Apache-2.0；时域逆矩阵引理，另有时域 Kalman 与频域 FDKF/PFDKF 教学入口 | 仅索引；RLS 的循环末尾样本覆盖、频域代码旧版 `np.complex` 与跨块状态需另测；无完整前端 |
| [MetaAF `metaaf/optimizer_rls.py`](https://github.com/adobe-research/MetaAF/blob/56c4665bdc51c2e0595a7c0cd9b1266408adceff/metaaf/optimizer_rls.py) | 通用 JAX 复数频域、每频点矩阵 RLS 更新器；核心库源码已取得 | AEC 的 [`zoo/aec/`](https://github.com/adobe-research/MetaAF/tree/56c4665bdc51c2e0595a7c0cd9b1266408adceff/zoo/aec) 配方含 Kalman/RLS 基线但本地未取得；zoo 与权重受单独 Adobe Research License 约束 |
| [echocatzh/PFDKF `pfdkf.py`](https://github.com/echocatzh/PFDKF/blob/7c8c86b5691966c330015d8e0960db0733b4844f/pfdkf.py) | MIT；独立作者的分区频域 Kalman 演示 | 仅索引；默认 `res=True` 带残余处理，输出不能充当纯线性误差；不是 2014 年原作者官方实现 |
| [Subband_Kalman_AEC `main.m`、`saf_kalman.m`](https://github.com/changxuding/Subband_Kalman_AEC/tree/f0c4f7030769d94dea2da3c814837448f171d422) | MIT 代码；MATLAB 子带 Kalman、平方根、信息及平方根信息形式 | 仅索引；需要 MATLAB 与配套参数，默认可含后滤，附带录音权利独立核查 |
| [nay0648/bssaec2020 `SimulatedExperiment/`](https://github.com/nay0648/bssaec2020/tree/a3f52249ee61f19e2369823f65c6c31392dcf042/SimulatedExperiment) | 作者的 Aux-ICA/加权 RLS MATLAB 仿真线索 | 无明确再分发许可，仅保留索引；其逐样本矩阵求逆不是本书 $O(L^2)$ 逆矩阵引理实现，也不是下述完整挑战赛 C++ 系统 |

**工业证据不能越级。** [Wang 等的 AEC Challenge 论文，2021，§2、§4.1、表 1～2](https://arxiv.org/pdf/2102.08551)明确报告“GCC-PHAT 延迟补偿 → 频域加权 RLS 线性抵消 → Deep-FSMN 残余抑制”的参赛系统；其 wRLS 用 20 ms 帧、10 ms 帧移、320 点 DFT、5 个频域抽头，平滑参数 0.8、形状参数 0.2。文中 0.61 ms/帧来自内部 C++/SSE2、Surface Laptop i5-8350U 1.9 GHz 上**整链**平均计时，其中 0.19 ms 属于延迟补偿与 wRLS、0.42 ms 属于残余抑制；网络输入还用一帧未来上下文。论文也报告双讲近端语音可能被过抑制。这证明该团队做过有实现和计时的挑战赛系统，不证明内部 C++ 源码公开或商业产品已经部署，也不能把系统分数归给 RLS 单模块。

[DSP Concepts 的 `SbKalmanAEC4RefV1` 官方模块文档](https://documentation.dspconcepts.com/awe-designer/8.D.2.7/sbkalmanaec4refv1)则是商业软件功能证据：WOLA 复数子带输入、多麦及至多四参考相关模块，提供 `resetP`、`masterReset`、`freeze`、权重、`P` 和 `PXTerm` 调试输出；默认残余噪声抑制开启。它说明产品文档确实提供 Kalman-based AEC 模块，但页面不是可再分发的完整算法源码，不能断言与 Enzner 2006 或本书式(6-10)逐式一致，更不能由输出安静判断线性滤波器更优。WebRTC AEC3 与 SpeexDSP 的锁定源码另见 A03/A06；没有实现级证据时，不把两者改称 RLS 或 Kalman。

**公平复现实验顺序。** 先用长度 2 的已知 FIR 检查抽头顺序、先验输出、RLS 正则化与矩阵 Kalman 交叉协方差；再给同一长 FIR、同一随机种子和共同的参考/观测输入，分别记录权重误差、线性输出残差、运算量及资源。下一轮只改变一种条件：参考由宽带变窄带、插入已知双讲、改变真实路径、加入参考错位，或打开后滤。RLS 的 $\lambda$ 与 Kalman 的 $Q,\Psi$ 是不同模型参数，不用数值相等来定义“公平”。先在真值已知的合成段计算回声分量 ERLE 和近端保留，再在真实配对录音报告输入/输出能量与可获得的近端质量；真实配对录音没有干净回声和近端真值时，不能把总功率比冒充双讲 ERLE。每个结果同时标明线性段还是含后滤最终输出、冻结/门控来源、路径突变时间、FFT/分区与因果延迟、随机重复次数和离散程度。本节尚无这些外部 RLS/FDKF 实现的同条件声学性能测量，因此不提供伪造的统一排名。

### A05　DTD：二值判决与连续自适应控制

双讲检测（DTD）用于保护近端语音，不直接生成回声副本。Geigel、归一化相关和相干性方法依赖不同假设；高参考相关性可以是残余回声，也可能伴随路径失配。二值 DTD 决定是否冻结更新，只是控制方案之一。锁定版 Speex [`mdf.c`](https://gitlab.xiph.org/xiph/speexdsp/-/blob/8e29a256ef0235ebbe7fcb8417b5ac7731eb8307/libspeexdsp/mdf.c)的文件头明确说明没有独立的显式双讲判决，而是根据残余回声、双讲和背景噪声连续调节学习率；[Valin 2007 原论文](https://people.xiph.org/~jm/papers/valin_taslp2006.pdf)解释该更新控制。WebRTC AEC3 的 `subtractor.cc` 则分别计算 refined、coarse 滤波器的更新增益，不能把两者直接改写为同一个“检测到双讲便冻结”的接口。

最小实验由远端单讲、双讲、近端单讲与静音四段组成，分别保存检测量、步长与系数误差。失败实验保持远端播放同时改变路径，比较“误判双讲导致冻结”和“漏判双讲导致误更新”。即使两者输出能量相近，恢复办法也不同。不要仅用一个总体准确率验收 DTD。

### A06　AEC3 的延迟、线性抵消、残余抑制与舒适噪声

WebRTC 的 [`modules/audio_processing/aec3/`](https://webrtc.googlesource.com/src/+/0467d2b91cc20b9b001c2bbb73d43ea6b2491f3e/modules/audio_processing/aec3/)是按模块读工业 AEC 的入口。阅读顺序建议为 `echo_canceller3.cc` → `block_processor.cc` → `render_delay_buffer.cc`/`render_delay_controller.cc` → `echo_path_delay_estimator.cc` → `subtractor.cc` → `residual_echo_estimator.cc` → `suppression_gain.cc` → `comfort_noise_generator.cc`。这些模块分别处理参考缓冲、延迟、线性残差、剩余回声估计和输出抑制。

锁定版的延迟估计器先对麦克风信号作通道混合及降采样，再用参考缓冲上的匹配滤波器（matched filter）估计时差，并聚合候选滞后；`render_delay_controller.cc` 将估计转换为参考缓冲延迟。`subtractor.cc` 的 refined/coarse 双滤波器负责线性回声估计与更新，不是两套用于比较不同候选延迟的滤波器。源码入口分别见 [`echo_path_delay_estimator.cc`](https://webrtc.googlesource.com/src/+/0467d2b91cc20b9b001c2bbb73d43ea6b2491f3e/modules/audio_processing/aec3/echo_path_delay_estimator.cc)和 [`render_delay_controller.cc`](https://webrtc.googlesource.com/src/+/0467d2b91cc20b9b001c2bbb73d43ea6b2491f3e/modules/audio_processing/aec3/render_delay_controller.cc)。

应用接入点是 [Audio Processing Module（APM）接口](https://webrtc.googlesource.com/src/+/0467d2b91cc20b9b001c2bbb73d43ea6b2491f3e/api/audio/audio_processing.h)，不能从内部 `subtractor.cc` 直接推断上层调用协议。固定头文件要求约 10 ms 的线性 PCM 帧：启用 `config.echo_canceller.enabled` 后，将远端播放帧交给 `ProcessReverseStream()`，处理近端帧前按设备时间设置 `set_stream_delay_ms()`，再调用 `ProcessStream()`。`int16` 接口按通道交织，浮点接口按通道分列；这些格式要用不同通道的已知脉冲检查。

这个提交的 `int16` 接口只接受 8、16、32、48 kHz，且采集输入、输出与播放反向流的采样率必须相同，输出布局还须与采集输入一致；浮点接口才允许头文件所述较宽的合法采样率范围及不同的输入、输出布局。应用拿到 44.1 kHz 的整数 PCM，不能仅把采样率写入配置就交给 `int16` 接口，须先转换到受支持的速率，或按浮点接口的要求准备数据。APM 外部接口限制与锁定 AEC3 内部 `aec3_common.h` 的 16、32、48 kHz 分带速率不是同一层约束；内部每块 64 个最低频带样本，也不等于要求应用每次只交 64 点。[固定版头文件的 `Initialize()` 约束与 `NativeRate` 枚举](https://webrtc.googlesource.com/src/+/0467d2b91cc20b9b001c2bbb73d43ea6b2491f3e/api/audio/audio_processing.h#509)

`set_stream_delay_ms()` 报告的是播放帧进入 APM 到硬件实际播放、以及麦克风采样到采集帧进入 APM 的缓冲时间之和，不是房间传播时间或滤波器尾长。固定版 [`audio_processing_impl.cc`](https://webrtc.googlesource.com/src/+/0467d2b91cc20b9b001c2bbb73d43ea6b2491f3e/modules/audio_processing/audio_processing_impl.cc) 的 `set_stream_delay_ms()` 把负值截到 0 ms、超过 500 ms 的值截到 500 ms，并返回 `kBadStreamParameterWarning`。接入层应保存输入参数、返回码、`stream_delay_ms()` 读回值与两路硬件时间戳；无线或外接播放的延迟若超出该范围，不能只按传入值解释后续对齐结果。500 ms 是这个提交的接口行为，不是声学路径或其他 AEC 的通用限值。

线性段并非只能从内部调试转储取得。固定版公开的 `Config::EchoCanceller::export_linear_aec_output` 默认 `false`；需要线性取点时须在配置 AEC 时设为 `true`。处理采集帧后调用 `GetLinearAecOutput()` 并检查布尔返回值，可取得最近约 10 ms、16 kHz 的线性 AEC 输出；头文件说明多通道采集时返回其单声道表示。它与公开 `ProcessStream()` 的最终输出可能采用不同采样率及取点，必须按对应采样区间对齐后比较。若所用构建或接入层没有成功导出，记录“线性取点未取得”及配置、返回值，而不是用最终输出冒名顶替。入口见[固定版公开头文件的配置与接口](https://webrtc.googlesource.com/src/+/0467d2b91cc20b9b001c2bbb73d43ea6b2491f3e/api/audio/audio_processing.h)和[`audio_processing_impl.cc` 的缓冲分配与返回实现](https://webrtc.googlesource.com/src/+/0467d2b91cc20b9b001c2bbb73d43ea6b2491f3e/modules/audio_processing/audio_processing_impl.cc)。

复现应先使用工程自带测试和录制/回放入口，固定 APM 配置、采样率、帧组织、输入电平、延迟设置以及参考和采集调用次序。最小实验只更改参考延迟；失败实验引入一次参考丢块，观察延迟状态、线性残差和最终输出。固定版 `block_processor.cc` 在尚无播放参考时跳过采集处理；参考缓冲下溢时重置延迟控制器，溢出时刷新缓冲并重置。这些状态需要与队列事件和输出一起记录，不能把恢复期的原始或强抑制输出误判为稳态 AEC 性能。[固定版 `block_processor.cc`](https://webrtc.googlesource.com/src/+/0467d2b91cc20b9b001c2bbb73d43ea6b2491f3e/modules/audio_processing/aec3/block_processor.cc)

| 征兆 | 先观察什么 | 区分实验与应调整的部分 |
|---|---|---|
| 开始处理就有大量残余 | 播放帧是否先进入 APM、流延迟返回码与读回值、线性段输出 | 给已知宽带参考施加固定时移，检查对齐及滤波器覆盖；先修参考取点、时间戳或缓冲，不先调残余抑制 |
| 长时间运行后逐渐变差 | 分窗估计参考—麦克风滞后、硬件时间戳、队列占用与丢帧事件 | 滞后稳定但偏大先查固定错位与覆盖长度；持续斜率提示采样率偏移，需共享时钟或估计速率并连续重采样；阶跃且伴队列异常先查丢帧/设备切换 |
| 最终输出安静但线性段仍有明显回声 | 同一区间的 `GetLinearAecOutput()` 与 `ProcessStream()` 输出、近端语音损伤 | 后级抑制可能掩盖线性失配；先查参考完整性、延迟和更新控制，再单独调抑制器 |

分窗滞后估计需要充分激励的参考；周期节目、多径和近端语音会制造错误峰值。应把滞后轨迹与设备时间戳和缓冲事件联合判断，不能把单次互相关峰当作物理时钟真值。AEC3 的 `HasClockdrift()` 是状态判别，`block_processor.cc` 将它传给回声路径控制；这条路径没有把两个独立设备的 PCM 自动重采样到同一速率。[固定版 `render_delay_controller.cc`](https://webrtc.googlesource.com/src/+/0467d2b91cc20b9b001c2bbb73d43ea6b2491f3e/modules/audio_processing/aec3/render_delay_controller.cc)、[固定版 `block_processor.cc`](https://webrtc.googlesource.com/src/+/0467d2b91cc20b9b001c2bbb73d43ea6b2491f3e/modules/audio_processing/aec3/block_processor.cc)

### A07　RES/NLP 与近端保护

残余回声抑制（RES）或非线性后处理（NLP）根据未消除回声的估计给输出施加增益。线性路径失配、扬声器非线性与延迟错位都能形成残余，但需要不同诊断。AEC3 的 `residual_echo_estimator.cc` 和 `suppression_gain.cc` 分别估计残余和计算增益。Speex 的 [`preprocess.c`](https://gitlab.xiph.org/xiph/speexdsp/-/blob/8e29a256ef0235ebbe7fcb8417b5ac7731eb8307/libspeexdsp/preprocess.c)只有在预处理器绑定回声状态后，才通过 `speex_echo_get_residual()` 取得残余估计；[官方手册 §6.2](https://www.speex.org/docs/manual/speex-manual/node7.html)给出 `speex_preprocess_ctl(preprocess_state, SPEEX_PREPROCESS_SET_ECHO_STATE, echo_state)`。因此应把 `speex_echo_cancellation()` 的输出和再经 `speex_preprocess_run()` 的输出分开保存；后者的变化包含预处理器作用，不能全部归功于 MDF 线性抵消。

最小实验把线性抵消器输出固定，只扫描后处理强度，分别试听和计量回声泄漏、近端语音损伤及噪声起伏。失败实验让低电平近端辅音与残余回声同时出现，检查词尾是否被截断。此类错误不能靠远端单讲 ERLE 发现，必须同时报告双讲语音的保真与任务指标。

本书的 [E06-04～E06-06](../../chapters/06_aec.md#sec-6-1-17)补充三个可运行的取点检查：同幅副本若反相，相减会把回声能量增为四倍；背景噪声保留时，麦克风输入/输出能量比仍可能为有限值，即使回声分量已经完全抵消；纯 500 Hz 参考经过三次非线性后还会留下线性参考不能生成的 1500 Hz 分量。这些是给定模型下的代数结果，不是对 RES 的设备性能测量。运行入口为 [`exercises_enhancement.py`](../examples/exercises_enhancement.py)，固定检查见 [`test_codes_enhancement_round2.py`](../../tests/test_codes_enhancement_round2.py)与[`test_codes_exercises_enhancement.py`](../../tests/test_codes_exercises_enhancement.py)。

**三种实现的接口与状态对照。** 下表对应固定版本，解释怎样准备对照输入；表中的方法状态不是性能排名。Speex 的 10～20 ms 帧与 100～500 ms 滤波长度来自[固定头文件的接口说明](https://gitlab.xiph.org/xiph/speexdsp/-/blob/8e29a256ef0235ebbe7fcb8417b5ac7731eb8307/include/speex/speex_echo.h)，只是该接口给出的建议量级，设备仍需按实测路径配置。

| 实现 | 输入与持续状态 | 延迟、双讲和残余处理 |
|---|---|---|
| 本书 `NLMSState` | 等长实数参考与麦克风一维数组；保留参考历史和抽头；`freeze` 由调用者给出 | 没有内部延迟搜索、DTD、RES 或重采样；可用已知延迟和真值冻结建立算术基线，不能冒充设备方案 |
| WebRTC AEC3，经 APM 接入 | 约 10 ms 播放/采集帧；APM 调用、配置、参考缓冲及内部双滤波状态跨帧保留 | 上层报告流延迟，内部估计参考滞后；开启导出标志后用公开接口取得 16 kHz 线性段，最终输出另存 |
| SpeexDSP AUMDF | `spx_int16_t` 麦克风和播放帧；回声状态、频域分区与参考历史跨帧保留；多通道用 `speex_echo_state_init_mc()` | 同步 `speex_echo_cancellation()` 不额外加入两帧播放缓冲；固定版异步 `playback/capture` 辅助函数只按单通道帧宽搬运，不可直接接多通道交织帧；连续学习率而非二值 DTD，预处理残余抑制须另绑 `echo_state` |

上表的异步限制来自锁定提交 `8e29a256ef0235ebbe7fcb8417b5ac7731eb8307` 的[`mdf.c` 实际缓冲操作](https://gitlab.xiph.org/xiph/speexdsp/-/blob/8e29a256ef0235ebbe7fcb8417b5ac7731eb8307/libspeexdsp/mdf.c#L687-733)：`speex_echo_playback()` 每次只复制 `frame_size` 个标量样本，`speex_echo_capture()` 欠载时也只复制 `frame_size` 个输出样本，未按扬声器或麦克风通道数扩展。多通道应由应用按完整交织帧维护参考队列，再把已对应的播放帧交给 `speex_echo_cancellation()`；本书尚未运行这一多通道设备接口实验。

WebRTC 与 SpeexDSP 锁定源码分别以 BSD-3-Clause 许可登记在[第三方清单](../THIRD_PARTY.md)；本书 NumPy 基线是原创教学代码。WebRTC 完整构建还需要其依赖元数据，Speex 的可选预处理器也不等于核心 AUMDF。RNNoise 虽在外部清单中，但[官方 README](https://github.com/xiph/rnnoise)定义它为单输入噪声抑制，示例使用 48 kHz 单声道原始 PCM；它没有播放参考，不能当作 AEC3 或 Speex 的同类替代。锁定版 `autogen.sh` 会调用下载模型的脚本；源码核对不等于已取得模型，更不等于已完成构建或执行。

**同输入、同取点的三系统对照实验设计（尚未完成）。** 先选 16 kHz、单播放参考、单麦克风的 PCM，固定同一参考抽头、增益和无削波输入。用已知 FIR 生成纯回声及独立近端信号，保存两者和加和后的麦克风波形。前者用于检查算法接口和真值分量；真实设备数据则用于检查模型外的非线性、驱动与时钟问题，不能以本仓库的 DEMAND 环境噪声摘录代替有播放参考的 AEC 录音。

真实配对录音可从 [Microsoft AEC Challenge 固定版数据说明](https://github.com/microsoft/AEC-Challenge/blob/6c633d0a9d2a143a0e364899b91b06f127315b18/datasets/README.md)按需寻找同一 GUID 的 `*_farend_singletalk_lpb.wav` 与 `*_farend_singletalk_mic.wav`；双讲和移动场景有对应 `*_doubletalk_lpb.wav`、`*_doubletalk_mic.wav` 及 `*_with_movement_*`。这里的 `lpb` 是 Windows 播放环回，不是扬声器端子电压；官方还提示部分计算机虽使用 raw mode，收放链仍可能有 DSP。

外部样本要先核查数据来源、使用及再分发条款，再保存原文件摘要、配对 GUID、采样率、裁剪区间和实际参考抽头。本书按需把两对录音下载到本机 Git 忽略缓存，不在仓库再分发，也没有独立的近端干净语音和回声分量真值；可报告注明噪声底的远端单讲输入/输出功率近似、听测和自动评分，不能把它们写成真值 ERLE。[项目数据许可说明](https://github.com/microsoft/AEC-Challenge/blob/6c633d0a9d2a143a0e364899b91b06f127315b18/README.md#dataset-licenses)与代码 MIT 许可分列。

1. 对同一底稿分别构造远端单讲稳态、双讲、近端单讲、沉默、路径突变、固定参考延迟和一次参考丢块；每种条件只改一个因素，记录起止采样索引。测试 SRO 时另加长时、已测或明确合成的两时钟输入，不把固定时移当成速率偏移。

2. 三路保持各自合法帧协议：教学 `NLMSState.process()` 用显式真值冻结作为理想控制对照，再做不冻结反例；Speex 固定 `frame_size`、`filter_length`、采样率和同步 API，异步 API 另组测试其两帧缓冲；WebRTC 固定 APM 配置、`ProcessReverseStream()` 与 `ProcessStream()` 调用次序和按 HAL 时间定义的 `set_stream_delay_ms()`。输入转换到整数 PCM 时检查量化和削波，不能让三路使用不同增益。

3. 保存麦克风输入、播放参考、各实现的线性抵消输出和最终输出。WebRTC 先开启线性段导出，再检查 `GetLinearAecOutput()` 返回值；未取得时注明原因和缺失区间，只比较可观察的最终输出。Speex 分开“仅 MDF”和“MDF 加预处理”，教学 NLMS 没有最终后滤输出。远端单讲可在合成真值上计算回声分量 ERLE；真实录音无干净回声真值时只能报告注明噪声底的输入/输出功率近似，不将它命名为真值 ERLE。

4. 双讲报告近端语音损伤与回声泄漏，路径突变报告恢复时间和失败次数；CPU 时间、峰值内存与端到端延迟在同一硬件、相同构建与线程条件下分别测量。先检查各系统是否处理了相同采样区间和固定时移，再讨论相对趋势；不同取点、额外预处理或缺失结果的行不排统一名次。

**已运行：一对真实录音上的 SpeexDSP 同步 AEC（2026-09-23）。** 使用 Microsoft AEC Challenge 固定提交 `6c633d0a9d2a143a0e364899b91b06f127315b18` 的 `datasets/real/-0AcvGNEdEK-DQGxWmtq2Q_farend_singletalk_{lpb,mic}.wav`。同一 GUID 是官方远端单讲配对；`lpb` 是 Windows 播放环回，`mic` 是麦克风录音，不是干净回声真值。

两文件都是单声道、16 kHz、PCM16，分别有 188320 与 188480 个样本。文件 SHA-256 分别为 `9b204ad5473726526d14830103e53647897699ef89d49624964b7f2449040426` 与 `6b4c3e01b969c5cad91f248ff967cfa03df6554f3a060d6e5b06b7d20341bba6`。原始文件仅在 Git 忽略缓存中，处理输出默认不落盘；脚本仅允许把可选试听 WAV 写到同一缓存，不随本书再分发。[官方数据说明](https://github.com/microsoft/AEC-Challenge/blob/6c633d0a9d2a143a0e364899b91b06f127315b18/datasets/README.md)与[数据许可段](https://github.com/microsoft/AEC-Challenge/blob/6c633d0a9d2a143a0e364899b91b06f127315b18/README.md#dataset-licenses)不能被代码 MIT 许可替代。

对两路共同的 `[0,188320)` 样本按原样逐帧输入，舍弃麦克风末尾 160 个样本；不重采样、不时移、不调增益。锁定版 SpeexDSP `8e29a256ef0235ebbe7fcb8417b5ac7731eb8307` 在 macOS arm64 / AppleClang 21 上以 CMake `Release`、共享库、浮点配置构建，库 SHA-256 为 `c3e70172a3a9bf60b60bfd0bddfd58550a1899fef09078a9c7d5270d4a12105d`。

使用同步 `speex_echo_cancellation()`，每帧 160 样本，滤波覆盖 4096 样本（256 ms）；初始化后显式设置并读回 16 kHz。没有调用异步缓冲接口或 Speex 预处理器。`[16000,64000)` 的归一化互相关诊断峰在正 498 样本（约 31.1 ms，相关系数 0.260），只用于报告，不用它挪动音频；它并非硬件时钟或纯声学传播时延的测量。

固定 `[0,48000)` 为收敛区，评分区为 `[48000,188320)`，数字功率比定义为 $10\log_{10}(\sum d[n]^2/\sum e[n]^2)$，各条件使用相同评分样本。零参考控制从头重置一套 Speex 状态；“晚 1 秒参考”控制在参考前填 16000 个零，再接原参考的前段，也重新建状态。

| 输入到同一 Speex 同步接口 | 评分区输入/输出数字功率变化 | 解释限制 |
|---|---:|---|
| 正确配对参考 | 5.57 dB | 核心 AUMDF 输出；不是干净回声真值 ERLE |
| 全零参考 | 3.95 dB | 即使无参考，处理链也改变输出，不能把 5.57 dB 全归给路径抵消 |
| 故意晚 1 秒的错误参考 | −0.23 dB | 错配时本片段输出能量略增；不是所有设备的普遍增益 |

正确配对输出相对零参考输出的数字功率再低 1.62 dB。正确配对的 8 个完整 1 秒评分块各自为 4.55、4.99、5.42、6.41、5.19、5.79、5.61、6.38 dB；它们是同一录音相邻区间，不是八次独立实验，整段的 5.57 dB 应先合并线性功率再取对数。三路输出的 PCM SHA-256 及未舍入数值由[可运行脚本](../examples/aec_real_pair_experiment.py)打印。复做前先阅读数据来源及使用条款，再在仓库根目录按需把官方固定版的两只 Git LFS 对象取到忽略缓存；脚本会核对完整 SHA-256，若下载到的只是 131 字节指针会拒绝：

```bash
mkdir -p codes/upstream/_downloads/aec-challenge/datasets/real
curl -fL 'https://media.githubusercontent.com/media/microsoft/AEC-Challenge/6c633d0a9d2a143a0e364899b91b06f127315b18/datasets/real/-0AcvGNEdEK-DQGxWmtq2Q_farend_singletalk_lpb.wav' -o codes/upstream/_downloads/aec-challenge/datasets/real/-0AcvGNEdEK-DQGxWmtq2Q_farend_singletalk_lpb.wav
curl -fL 'https://media.githubusercontent.com/media/microsoft/AEC-Challenge/6c633d0a9d2a143a0e364899b91b06f127315b18/datasets/real/-0AcvGNEdEK-DQGxWmtq2Q_farend_singletalk_mic.wav' -o codes/upstream/_downloads/aec-challenge/datasets/real/-0AcvGNEdEK-DQGxWmtq2Q_farend_singletalk_mic.wav
cmake -S codes/upstream/_downloads/speexdsp -B /private/tmp/speexdsp-aec-20260923 -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DUSE_FIXED_POINT=OFF
cmake --build /private/tmp/speexdsp-aec-20260923 --parallel 4
.venv/bin/python codes/examples/aec_real_pair_experiment.py --speex-library /private/tmp/speexdsp-aec-20260923/libspeexdsp.dylib
```

此实验验证了固定版本、固定真实配对输入、严格帧协议下的接口可运行性和三个参考条件的数字功率变化。原录音没有独立的干净回声、干净近端或噪声分量；即使官方标作远端单讲，本书也未逐秒人工标注说话、噪声或主观听感。因此不能由功率比推出真值 ERLE、近端保护、感知质量、实时资源、设备时钟漂移，或 WebRTC/教学 NLMS 的相对优劣。

**真实双讲的第二组接口检查（2026-09-23 已运行）。** 同一固定版中另取 GUID `-2jLGNCgf0WDpKMY2iup7g` 的 `doubletalk_lpb` 与 `doubletalk_mic`。两份原 WAV 的 SHA-256 分别是 `790203e68cc8cc9b02becc4efefbde0ddf962fff4d223608b00ffc8b2682ed71`、`4e7e35290d763d2d22173976c2cfd46739c9afe76729a0c283c1b69609f22041`；均为 16 kHz、单声道、PCM16。参考 196186 点、麦克风 196320 点，共同截为 196160 点完整 10 ms 帧，不插值、不时移。参考全段没有满幅样本，麦克风有 1 个满幅样本；全段数字 RMS 分别为 0.05574、0.02760。官方[数据说明](https://github.com/microsoft/AEC-Challenge/blob/6c633d0a9d2a143a0e364899b91b06f127315b18/datasets/README.md)将 `doubletalk` 标为双讲且无路径变化；这不是本书对该片段每一帧的人工语音标注。

固定 `[0,3)` s 为适应期、`[3,8)` s 为评分区，三个条件都从新状态运行同一 Speex 同步核心。库与远端单讲实验相同，参数仍是 160 点帧、4096 点滤波覆盖。评分区的麦克风数字 RMS 是 0.03860；正确参考、零参考、故意晚 1 秒参考的输入/输出总功率变化分别是 2.234、2.204、2.200 dB，正确参考的输出相对零参考仅低 0.029 dB。这个很小的差别不能被解释为“已消掉的回声量”：近端语音、回声和噪声在麦克风里不可分，三种控制的输出哈希和未舍入数字由[双讲脚本](../examples/aec_doubletalk_experiment.py)给出。此处也未完成盲听，不能称近端音质已经验证。

复做时把上述两只原 WAV 放到 `codes/upstream/_downloads/aec-challenge/datasets/real/`，脚本在读取前核查摘要，再运行：

```bash
.venv/bin/python -m codes.examples.aec_doubletalk_experiment --speex-library /private/tmp/speexdsp-aec-20260923/libspeexdsp.dylib
```

**已知近端注入：严格合成与半合成两种检查（2026-09-23 已运行）。** 真实双讲没有干净分量，故另用[式(6-14)](../../chapters/06_aec.md#sec-6-1-16)定义两次独立运行的输出增量 $\Delta$、投影增益 $g_{\Delta}$ 和不拟合时移/增益的相对平方误差 $E_{\Delta}$。每一对运行都使用同一播放参考、各自新建的状态和完全相同的输入长度；第二次麦克风 PCM 比第一次恰好多一个已知的 $s$。即使如此，自适应器状态和预处理也可随 $s$ 改变，所以 $\Delta$ 不是算法内部单独输出的“近端声道”。$E_{\Delta}$ 是无量纲的逐样本误差，越小表示这项增量越接近原注入信号；它不是双讲 ERLE 或主观音质分。

严格合成夹具读 `codes/audio/aec_far.wav`、`aec_near.wav`、`aec_microphone.wav`，逐一核对清单中的 SHA-256。三路均为 16 kHz、32000 点 PCM16；近端只在 `[19200,28800)` 非零。令 $d_1$ 为已量化麦克风、$s$ 为已量化近端，取 $d_0=d_1-s$，先在 32 位整数中检查范围，再转回 PCM16，因此输入等式在 PCM 域精确成立；$d_0$ 不宣称等于未量化 FIR 真值。固定 `[9600,17600)` 为远端单讲检查区、`[19200,28800)` 为双讲评分区、`[28800,32000)` 为恢复观察区。教学 NLMS 用 32 抽头、步长 0.4、$\varepsilon=10^{-8}$；双讲冻结使用已知真值掩码，不是检测器。SpeexDSP 仍为 160 点帧、4096 点滤波覆盖和独立状态。

| 输入和实现 | 播放参考 | $g_{\Delta}$ | $E_{\Delta}$ | 能说明什么 |
|---|---|---:|---:|---|
| 合成；NLMS 真值冻结 | 正确 | 1.000 | 0 | 冻结且两臂状态相同，输出增量按代数逐样本等于已知近端 |
| 合成；NLMS 持续更新 | 正确 | 0.936 | 0.322 | 近端参与更新后，两臂滤波状态不同 |
| 合成；Speex 同步核心 | 正确 | 0.878 | 0.208 | 含核心预处理、连续更新和量化的总增量 |
| 合成；Speex 同步核心 | 全零 | 0.879 | 0.204 | 没有播放参考时也出现近似增量偏差，不能全归于回声路径 |

第二种夹具从[同一 GUID 的官方固定版数据](https://github.com/microsoft/AEC-Challenge/blob/6c633d0a9d2a143a0e364899b91b06f127315b18/datasets/README.md)取远端单讲 `lpb`/`mic` 配对，再取另一次 `nearend_singletalk_mic.wav`。该近端原文件 SHA-256 为 `192291e973b7dee519102f30ae75e4209712ce55d02f55a9020a7c3b37a1d0e0`，只有本机缓存；它是单讲麦克风录音，含本底噪声，不是干净语音。把它的 `[48000,128000)` 样本以 1.0 的 PCM 增益叠到远端单讲麦克风的同位置，参考不变；不重采样、不错位、不按峰值重新归一化，32 位整数加法确认未溢出。叠加波形全段峰值为 13633 PCM 计数。前 3 s 为适应期，评分 `[3,8)` s，恢复观察 `[8,11)` s。它是两次录音的**半合成叠加**，不能冒称同一时刻的真实双讲。

已知近端波形在评分区间的平方和必须大于零，否则 $g_{\Delta}$、$E_{\Delta}$ 均无定义。注入在 3 s、8 s 直接切换而没有淡入淡出，边界附近可能包含拼接瞬态；本实验只检验固定外加波形的增量，不作为自然双讲听感或总体性能证据。

| 半合成输入、Speex 同步核心 | $g_{\Delta}$ | $E_{\Delta}$ | `[8,11)` s 两次输出差的 RMS |
|---|---:|---:|---:|
| 正确参考 | 0.922 | 0.140 | 111.04 PCM 计数 |
| 全零参考 | 0.923 | 0.128 | 5.62 PCM 计数 |

两张表的输入条件不同，**不能横向排算法名次**。恢复区间没有外加近端，仍有输出差，说明先前注入改变了后续处理状态或其输出；没有独立回声真值时不能把 111.04 计数直接解释为“恢复期回声”。锁定版 [Speex `mdf.c`](https://gitlab.xiph.org/xiph/speexdsp/-/blob/8e29a256ef0235ebbe7fcb8417b5ac7731eb8307/libspeexdsp/mdf.c)还对麦克风做 DC 陷波、预加重、输出去加重和 PCM 转换，这解释了零参考控制也不能简单视作原样旁路。完整未舍入结果与输入、输出摘要可由[受控脚本](../examples/aec_controlled_doubletalk.py)复算；[测试](../../tests/test_codes_aec_controlled_doubletalk.py)另检查 PCM 加法、已知区间及指标算术，但不替代完整接口实验。复做命令为：

```bash
.venv/bin/python -m codes.examples.aec_controlled_doubletalk \
  --speex-library /private/tmp/speexdsp-aec-20260923/libspeexdsp.dylib \
  --include-real-hybrid
```

**WebRTC AEC3 对照的已核实入口与未完成项。** 固定源码提交 `0467d2b91cc20b9b001c2bbb73d43ea6b2491f3e` 的官方 [`audioproc_f`](https://webrtc.googlesource.com/src/+/0467d2b91cc20b9b001c2bbb73d43ea6b2491f3e/rtc_tools/BUILD.gn) 可接受 `--i=<mic.wav>`、`--ri=<lpb.wav>`，用 `--o=<final.wav>` 保存最终输出，并用 `--linear_aec_output=<linear.wav>` 导出 16 kHz 线性段。为了与 Speex 的“同帧先参考、后采集”一致，须给 `--custom_call_order_file` 一个内容为 `rc` 的文本文件；[WAV 模拟器源码](https://webrtc.googlesource.com/src/+/0467d2b91cc20b9b001c2bbb73d43ea6b2491f3e/modules/audio_processing/test/wav_based_simulator.cc)的默认顺序反而是 `cr`。建议显式设置 `--fixed_interface=true --aec=1 --agc=0 --agc2=0 --ns=0 --hpf=0 --ts=0 --stream_delay=0`，并保留完整构建选项、日志、输出 PCM 摘要；`stream_delay=0` 只是离线缓冲假设，`hpf=0` 也不保证关闭 AEC3 内部强制高通。最终输出与 Speex 核心线性输出不是同一处理取点，须分列而不能排性能名次。

**本机固定版本构建与运行（2026-09-23）。** Xcode 27.0（27A266a）的许可及首次启动组件已由用户同意并完成。本书在独立的 Git 忽略目录 `codes/upstream/_downloads/webrtc_aec3_checkout/` 用官方 depot_tools 提交 `db1dc923aa3c34fa748015e8c7a435d955b4c105` 执行 `gclient sync --revision src@0467d2b91cc20b9b001c2bbb73d43ea6b2491f3e --nohooks --noprehooks --no-history -j 4`；随后核对 `src` 的 HEAD 为该提交，未修改已跟踪上游源码。原先 136 MB 的稀疏检出仍单独保留，不冒充完整依赖。上游 `runhooks` 已安装编译器和符号工具，但因随后开始下载与本实验无关的大型视频测试资源而中止；不能称为完整 hooks 通过。

构建参数为 `is_debug=false rtc_include_tests=true rtc_enable_protobuf=true target_cpu="arm64" use_custom_libcxx=false mac_sdk_path="/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk"`；目标 `rtc_tools:audioproc_f` 构建成功，程序 SHA-256 为 `82f8eebaf574eb10fdee695ee7ea05a54418300a045a3ffb0de8f4c78f4592d3`。其中后两个参数是本机兼容设置：固定版自带 libc++ 在 Xcode 27 SDK 下编译报 `INFINITY` 未声明，固定版 `ld64.lld` 读取 27.0 SDK 的 `.tbd` 时又因 `arm64e.x1` 报未知架构；改用系统 libc++ 与本机已有的 15.4 SDK 后成功。它们不改变 WebRTC 源码，但此二进制不是上游默认 GN 参数的构建。下方给出构建与运行命令；程序摘要本身不能证明源码出处。

在完整检出目录的 `src/` 下，构建命令为：

```bash
PATH=/path/to/depot_tools:$PATH DEPOT_TOOLS_UPDATE=0 \
  buildtools/mac/gn gen out/aec3 \
  --args='is_debug=false rtc_include_tests=true rtc_enable_protobuf=true target_cpu="arm64" use_custom_libcxx=false mac_sdk_path="/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk"'
PATH=/path/to/depot_tools:$PATH DEPOT_TOOLS_UPDATE=0 \
  autoninja -C out/aec3 -j 4 rtc_tools:audioproc_f
```

`/path/to/depot_tools` 需替换为上述固定提交的本地目录；第二次运行 `gn gen` 会覆盖同一 `out/aec3` 的构建配置，复做前应检查已有目录用途。编译成功只证明这一工具在这套环境可构建，不证明整套 WebRTC 目标、全部 hooks 或其他 SDK 组合通过。

[AEC3 离线适配器](../examples/aec3_offline_compare.py)已固定两对录音的摘要、裁剪、`rc` 调用顺序及线性/最终双取点；只有传入官方已构建程序才会运行，缺程序直接报错，且不把两种输出混成一个分数。示例调用：

```bash
.venv/bin/python -m codes.examples.aec3_offline_compare \
  --audioproc codes/upstream/_downloads/webrtc_aec3_checkout/src/out/aec3/audioproc_f \
  --output-dir codes/upstream/_downloads/aec3-comparison-new-run
```

本次对两对真实录音分别从全新状态运行，16 kHz、单声道 PCM16，每帧 160 点、同帧先参考后采集；评分前 3 s 只用于适应。表中数字为 $10\log_{10}(P_{\mathrm{mic}}/P_{\mathrm{out}})$，功率按相同窗口的 PCM16 归一化样本平方均值计算。它是**总数字功率变化，不是真值 ERLE**。

| 原始录音与评分半开区间 | AEC3 线性输出 | AEC3 最终输出 |
|---|---:|---:|
| 远端单讲，`[48000,188320)` 点（`[3,11.77)` s） | 7.685 dB | 15.731 dB |
| 双讲，`[48000,128000)` 点（`[3,8)` s） | 3.384 dB | 3.851 dB |

线性输出 WAV 的 SHA-256 依次为 `6022c2a5354c9f760a3cac54e8e8ac9c63c1aec7bd16f3f2275f7257f27abc4e`、`9ca8a7e81bc7c07e8b4cf337832de20ff0cea5b94e3e937d7bbcb870baed8fab`；最终输出依次为 `cece4d1134285b5fe251a999d5b8915642753b1be19e63a72882d1e2e624944d`、`adcdd3b2d98181d0a9ae4c7d9a85181aa39c9bc2a50d4b51acb9a428df60c5b4`。录音原件及输出位于 Git 忽略缓存，输入摘要由适配器逐次检查；复做时须使用新的输出目录，避免覆盖旧实验。

远端单讲最终输出比线性输出的功率降得更多，符合两个取点包含不同处理的事实，却不能把差额全部归因于残余回声抑制。双讲录音没有独立干净近端或干净回声真值，因此 3.384/3.851 dB 既不证明近端保护，也不证明双讲回声抑制量。这里没有同输入、同取点的 AEC3、SpeexDSP 和教学 NLMS 性能排序，也没有真实设备声学回路或时钟漂移测量；[运行状态表](04_source_reproduction.md)按实现分别记录。

**本机设备链路的只读核查（2026-09-23）。** macOS `system_profiler SPAudioDataType -detailLevel full` 仅列出 MacBook Air 内置麦克风（1 路输入、48 kHz）和内置扬声器（2 路输出、48 kHz）；FFmpeg 的 AVFoundation 设备枚举仅显示内置麦克风，没有播放回环采集端点。CoreAudio 只读查询还给出两个设备 ID 71、76，均报告 `main` 时钟域和 `bltn` 内置传输类型。这些只说明系统报告它们属于同一域，不是两个独立设备的采样时钟漂移实测。

本次未发出探测音、打开麦克风录音或取得硬件时间戳，因此尚未验证数字播放参考是否对应真实声学播放、回声路径、端到端延迟、丢块或音频质量。内置全双工链路即使完成短时探测，也不能代替两个独立 USB 时钟域的长时漂移测试。进行实际链路验收时，应在获得录音环境许可后使用低电平非语音探测声，记录播放/采集时间戳、缓冲事件和分窗延迟，并将录音限制在本机忽略缓存；独立时钟偏移另需两个设备及足够长的连续记录。

### A08　PNLMS/IPNLMS、子带与非线性路径模型

比例归一化更新改变抽头间的学习分配；子带滤波改变输入相关性与每带更新问题；Volterra、Hammerstein、Wiener 或级联模型改变可表示的输入—回声关系。它们解决的困难不同，不能看见“频域”或“非线性”标签就视作同一种加速方案。出处与模型讨论见 [§6.1.3](../../chapters/06_aec.md#sec-6-1-3)和[§6.1.4](../../chapters/06_aec.md#sec-6-1-4)。

本书新增[实数 IPNLMS 状态](../array_tutorial/aec_ipnlms.py)：每次由旧权重构造抽头份额，按式(6-4)计算先验误差和同时更新；$\kappa=-1$ 与 NLMS 逐步对齐还须把 NLMS 正则项设为 $L\delta$。零初值、$\kappa=1$ 时所有比例份额均为零，这个失败边界有[独立测试](../../tests/test_codes_aec_ipnlms_subband.py)。[两带 Haar 代码](../array_tutorial/aec_subband.py)把两抽头已知时域路径精确改写成当前/历史块的 $2\times2$ 交叉带矩阵；另有只使用同带历史的对角子带 NLMS。前者不是自适应，后者不是一般房间路径的精确表示，也不是 Lee–Gan NSAF。[锚点输出](../examples/aec_ipnlms_subband_demo.py)和[章节 E06-11～18](../examples/aec_advanced_exercises.py)可独立复算。

固定在本仓库的 pyroomacoustics v0.10.0 源码还含 [`adaptive/subband_lms.py`](https://github.com/LCAV/pyroomacoustics/blob/0dd39f2614b7fc44b2cc63dbe7d60f4641068890/pyroomacoustics/adaptive/subband_lms.py)；该文件是**另一种实现入口**，不能把其演示中由子带模型合成的目标信号当作任意时域房间卷积的独立验证。一般临界抽样子带建模需检查交叉项，[Gilloire–Vetterli 1992 原论文](https://doi.org/10.1109/78.149989)是依据；[Lee–Gan 2004 NSAF 原论文](https://doi.org/10.1109/LSP.2004.833445)的子带误差更新全带抽头，结构与本书两带对角状态不同。图37和[四个子带结构音频](05_exercises_and_audio.md#16-两带-haar-的交叉项音频)是已知路径的代数模型；[另七个 WAV](05_exercises_and_audio.md#15-同源短路径-aec-的线性残差)对比时域 NLMS、IPNLMS、RLS 和矩阵 Kalman，不含子带自适应输出。

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

### A14　学习步长控制的线性 CTF-AEC

[Haubner 等的官方实现](https://github.com/ThomasHaubner/e2e_dnn_ad_control_for_lin_aec/tree/7a003133d742698de7acba9510d9586d7d57a584)没有让网络直接合成最终语音，而是保留卷积传递函数（Convolutive Transfer Function，CTF）线性回声模型，用网络给频率选择性步长和误差归一化项生成掩码。[TASLP 论文](https://doi.org/10.1109/TASLP.2023.3325923)说明训练目标；源码从 `main_train.py` 进入，再沿 [`libPython/class_frontend.py`](https://github.com/ThomasHaubner/e2e_dnn_ad_control_for_lin_aec/blob/7a003133d742698de7acba9510d9586d7d57a584/libPython/class_frontend.py) 的逐帧循环读到 [`class_aec_ctf.py`](https://github.com/ThomasHaubner/e2e_dnn_ad_control_for_lin_aec/blob/7a003133d742698de7acba9510d9586d7d57a584/libPython/class_aec_ctf.py) 的 `update_filter()`，可直接观察“学习控制器改变哪一个自适应量”。

代码使用 BSD 4-Clause 许可证，但当前仓库没有预训练 checkpoint；README 要求用户准备 `train_data.h5` 与 `test_data.h5`。固定版本的前端把 `center_stft=True`，并在每个序列的 `forward_batch()` 中重新初始化 AEC 参数，因此源码中的逐帧循环不等于可以任意切块、跨调用续算的部署接口。复现先固定参考对齐、CTF 长度、GRU 初值和序列边界，再比较固定路径、路径突变、双讲、播放静音与参考时移下的步长掩码、滤波器误差和近端损伤；没有训练资产时只读算法路径，不声称得到论文性能。

### A15　联合 AEC/NR 与处理顺序

[Integrated_AEC_NR 官方 MATLAB 实现](https://github.com/Arnout-Roebben/Integrated_AEC_NR/tree/23c6b567c7863a8ee9bafd38bad0d3ff2f25e185)在一般多麦、多扬声器设置下比较 MWF、扩展 MWF、AEC→NR、NR→AEC 与扩展 NR→AEC→后滤波。[论文](https://doi.org/10.1109/TASLPRO.2025.3648802)讨论线性相关参考情形。读码从 `Main.m` 进入 [`Util/Process/process.m`](https://github.com/Arnout-Roebben/Integrated_AEC_NR/blob/23c6b567c7863a8ee9bafd38bad0d3ff2f25e185/Util/Process/process.m)，再分别进入 `process_AEC.m`、`process_NR.m`、`process_MWFext.m`、`process_NRext.m` 与 `process_PF.m`；文件名大小写以仓库的 [`ReadMe.md`](https://github.com/Arnout-Roebben/Integrated_AEC_NR/blob/23c6b567c7863a8ee9bafd38bad0d3ff2f25e185/ReadMe.md) 为准。

代码为 MIT，官方说明使用 MATLAB R2024a。这个实现接收已经分解的 $s,n,e_s,e_n,l_s,l_n$，并在 `process.m` 中直接由干净期望语音 `s_f` 与干净回声 `es_f` 生成活动判决；这是用于受控比较的预言信息，不是实际设备可直接取得的输入。最小实验应先按原始分量重现各顺序，再把两路扬声器参考设为相关或秩亏，并以估计活动替换预言活动。`Audio/sig.mat` 的语音来自另行许可的数据，代码 MIT 不能替代音频许可；本书只取得源码，不把示例数据视为随代码自由再分发。

<a id="wpe"></a>

## 3. WPE 与联合卷积滤波

### W01　离线 WPE 与 MIMO-WPE

延迟线性预测使用历史多通道复谱解释当前晚期混响，功率权重防止高能量帧完全支配回归。多通道版本的每频点回归维度为“通道数 × 抽头数”，样本不足和共线性会使方程病态。教学入口是 [dereverberation.py](../array_tutorial/dereverberation.py)，参考实现是 [nara_wpe `wpe.py`](https://github.com/fgnt/nara_wpe/blob/a166779cca2088817e330481bd20af1a2c598555/nara_wpe/wpe.py) 的 `build_y_tilde()` 与 `wpe_v6()` 等实现。

读码时核对历史排序、保护延迟、统计有效区、复共轭与功率下限。最小实验先复算正文单抽头正规方程，再把单通道扩成两个完全相同通道；失败实验使用短于历史窗口的记录、静音频点和病态 SCM。库的稳定求解策略不能替代明确的旁路条件。[原始 MIMO-WPE 论文](https://doi.org/10.1109/TASL.2012.2210879)定义了希望保留的早期成分。

本书另提供已执行的[独立实现对照](../examples/compare_wpe_reference.py)。它使用 nara-wpe 0.0.11 的原始 `wpe_v6()` 与教学 `offline_wpe()`比较，固定种子 20260922、复谱形状 `(4, 2, 192)`、2 个预测抽头、3 帧保护延迟、有效历史统计、无功率平滑且关闭对角加载。两麦、单麦、通道交换和整体缩放四种输入，各运行 1 次和 3 次迭代。

2026-09-22 的本机对照环境为 macOS arm64、Python 3.13.12、NumPy 2.5.3，输入与求解使用 `complex128`。脚本输出运行环境，便于区分后续平台或线性代数库带来的末位舍入差异；本次没有计时或硬件速度比较。

2026-09-23 在同一设置下重跑，八组对照中完整历史后的帧最大绝对复数误差约为 `3.17e-15`，最大相对 L2 误差约为 `5.76e-16`。开头 4 帧不参与比较：教学实现旁路，上游允许零补历史后滤波。这是合成复谱上回归求解的一致性检查，不是语音质量、公开数据集或实时性测量；随机复谱不对应任何指定采样率与声学房间。

教学练习 E07-04 另用前缀相同、末帧不同的四帧记录检查未来依赖：一轮离线 WPE 的首个有效输出从 0 变为 −1/9。其回归量虽只含历史，系数却使用整段统计；这说明“预测索引只向过去”不足以证明端到端因果。该题没有使用居中功率平滑，也不把未变化的某个单例当作一般因果性证明。

### W02　逐帧递推 WPE

`nara_wpe/wpe.py` 的 `OnlineWPE` 保存逆相关矩阵、预测抽头、功率和输入缓冲；`step_frame()` 接收 `(频点, 通道)`。该实现先由旧状态输出当前预测残差，再更新状态，这与每次对增长的整段录音重新运行离线 `wpe()`不同。原理出处为[Kinoshita 等，Interspeech 2017](https://www.isca-archive.org/interspeech_2017/kinoshita17_interspeech.html)。

锁定的 0.0.11 版不能把三个接口的同名 `delay` 直接当作同一个时间索引。单频点、单通道、单抽头取帧 `[10,20,30,40]`，在 $t=3$ 令 `delay=1`、固定预测抽头为 1：离线 `build_y_tilde()` 取 $X_{t-1}=30$，残差为 10；无状态 `online_wpe_step()` 取 $X_{t-2}=20$，残差为 20；有状态 `OnlineWPE.step_frame()` 在写入当前帧前从旧缓冲取 $X_{t-3}=10$，残差为 30。这是该固定版本的接口行为，不是 WPE 理论规定必须相差两帧。

[在线接口对照](../examples/compare_online_wpe_reference.py)先按数组切片手算上述三项，再用一份精简的 NumPy 适配参考与未修改的 0.0.11 源码逐帧比较。参考程序保留该版本的缓冲顺序和数值保护规则，随附上游 MIT 声明；程序间一致性用于确认版本行为，独立期望来自抽头手算及另行用分数复算的实数、复数启动例。

固定种子 20260922、形状 `(48,2,1)`（帧、频点、通道）、2 抽头、`delay=2`、遗忘因子 0.95 时，NumPy 递推与上游连续输出的最大绝对误差为 `2.39e-15`。同一对象处理整段或外层分成两块，输出最大差为 0；在零起始第 24 帧重建对象，差异从第 24 帧立即出现，整段最大绝对差约为 2.56。数值为无单位合成复谱上的状态回归，不是语音质量或实时性测量。

最小实验逐帧输入一个固定记录并保留状态，再检查把同一帧流分成不同外层批次是否改变结果。失败实验中途错误重建 `OnlineWPE`，查看启动段反复出现；另测静音、设备重启和房间突变。遗忘因子与功率估计共同决定跟踪，不应仅报告 `taps` 和 `delay`；换版本或接口时还必须重新核对 `delay` 对应的实际时间戳。

### W03　块在线 WPE 与接口中的限制

`OnlineWPE.step_block()`处理已有历史缓冲形状的块；当前检出源码的 `_get_prediction()`还注明只支持 `block_shift=1`。TensorFlow 路线另有 `tf_wpe.py` 中的 `block_wpe_step()`、`recursive_wpe()`，二者不应凭名字直接互换。见[nara_wpe 固定版本源码](https://github.com/fgnt/nara_wpe/blob/a166779cca2088817e330481bd20af1a2c598555/nara_wpe/tf_wpe.py)。

最小复现要先确认块是“送入模型的新样本”还是“含全部历史的窗口”，再比较输出时间戳。失败实验把窗口长度误当帧移，检查是否重复消费或跳过帧。计算实时性时分别报块等待、历史上下文、STFT 和计算时间；有历史记忆不意味着必须等待同样长的未来数据。

### W04　DNN-WPE：掩码或功率网络与解析求解器

[ESPnet `DNN_WPE`](https://github.com/espnet/espnet/blob/be79590bb2ff26ffb01bc825c5f68cb9418b7f0d/espnet2/enh/layers/dnn_wpe.py)先估计掩码并形成跨通道功率，再调用 `wpe_one_iteration()`。外层输入为 `(批, 帧, 通道, 频点)`，内部转成 `(批, 频点, 通道, 帧)`。应把 `dnn_wpe.py`、`mask_estimator.py` 和 `wpe.py` 连起来读，才能知道网络实际改了哪个统计量。

最小实验用已知正数功率替换网络输出，检查解析求解，再接入固定 checkpoint。失败实验把掩码设为全零或加入未来语音，验证功率下限与非因果泄漏。该类支持不同掩码估计器，不能仅凭 `iterations=1` 就宣称在线；需要核查网络方向、统计窗口和 padding。

### W05　WPD：同时利用当前通道与延迟历史

WPD 把当前帧与历史帧一起放入无失真滤波问题。它的约束只施加在扩展向量的当前目标分量，历史部分用于削弱可预测晚期混响。出处为[Nakatani 与 Kinoshita](https://arxiv.org/abs/1908.02710)；实现入口为 [ESPnet `beamformer.py`](https://github.com/espnet/espnet/blob/be79590bb2ff26ffb01bc825c5f68cb9418b7f0d/espnet2/enh/layers/beamformer.py) 的 WPD 相关函数及 `dnn_beamformer.py` 的类型选择。

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

教学函数 `pit_permutation()` 为便于逐项核对而枚举全排列，只接受不超过 8 个源；这个上限是教学实现的资源边界，不是 PIT 定义的源数限制。对成对代价可加的目标，更多源应使用线性指派求解器；含跨输出耦合项的集合目标则不能直接这样替换。

### B02　AuxIVA、IP、ISS 与尺度恢复

AuxIVA 用一个源跨频点的联合模型约束依赖，辅助函数更新避免手工选择普通梯度步长。迭代投影（IP）与迭代源导向（ISS）是不同更新实现，应分别记录，而不是都写成“用了 IVA”。[ssspy `bss/iva.py`](https://github.com/tky823/ssspy/blob/38b9389e8b1914422561f1936d9b28d042d62d2c/ssspy/bss/iva.py)给出 IP1、IP2、ISS1、ISS2 与 IPA 的选项，[Ono 2011 原论文](https://doi.org/10.1109/ASPAA.2011.6082320)说明辅助函数思想。

读码顺序为源代价 → 加权 SCM → 空间更新 → 归一化 → projection-back。最小实验采用确定混合和已知两源，固定初值比较代价与分离；失败实验使用秩亏混合。恢复参考麦的尺度需要独立步骤，SI-SDR 对尺度不敏感，不能用其通过来证明波形幅度已经恢复。

E08-04 对人为指定的解混矩阵演示回投影：原输出 `[2s, -3u]` 乘参考麦系数 `[1/2, -1/6]` 后成为 `[s, 0.5u]`。它恢复参考麦源图像而非统一干声尺度，且不把给定解混矩阵的代数操作冒充 AuxIVA 估计。E08-05 单独检查掩码 SCM 的分母地板；E08-06 则把第二块的两个输出槽位交换，展示“每块 PIT 都是 20 dB”仍不能保证直接拼接后的说话人身份连续。三题的输入、答案和边界见 [第 8 章练习](../../chapters/08_speech-separation.md#sec-8-1)。

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

Conv-TasNet 学习短窗编码/解码器，并在编码域估计掩码。[正式论文](https://doi.org/10.1109/TASLP.2019.2915167)与[Asteroid `models/conv_tasnet.py`](https://github.com/asteroid-team/asteroid/blob/fce87469132760fbab41c20616ea0f0e079aad38/asteroid/models/conv_tasnet.py)用于核对前端和 TCN 分离器；还要读 `masknn/` 中的卷积、归一化和因果开关。

最小复现先固定源数与官方 recipe，记录采样率、编码核、步长、堆叠次数和 normalization。失败实验改变未来帧但保留过去，检查所谓因果配置是否真的不改变过去输出；另以短于编码窗的音频检查 padding。只提供网络构造类并不意味着已有匹配的训练权重。

### N02　DPRNN：块内、块间与重叠重构

DPRNN 把长序列划成重叠块，分开处理局部和跨块关系。[论文](https://doi.org/10.1109/ICASSP40776.2020.9054266)与[Asteroid `models/dprnn_tasnet.py`](https://github.com/asteroid-team/asteroid/blob/fce87469132760fbab41c20616ea0f0e079aad38/asteroid/models/dprnn_tasnet.py)需结合双路径 mask 网络阅读。块大小、块间循环方向和归一化决定能否在线。

最小实验用脉冲或已知索引序列检查分块/合并是否保留顺序，再做双源分离。失败实验在跨块边界换人，并测极短、非整块长度的输入。DPRNN 的分块主要是建模和计算设计，不能据此自动宣称它已经解决长会议的输出身份连续性。

### N03　SepFormer：双路径注意力与整句上下文

SepFormer 把双路径中的序列模型换为注意力结构。[正式论文](https://doi.org/10.1109/ICASSP39728.2021.9413901)与[SpeechBrain `lobes/models/dual_path.py`](https://github.com/speechbrain/speechbrain/blob/89ead74d163463d30c62329a09cfdb4c54f5abc1/speechbrain/lobes/models/dual_path.py)应配合 `recipes/WSJ0Mix/separation/` 中的配置及训练入口阅读。配置、模型与 checkpoint 必须匹配。

最小复现只使用一个固定公开模型的匹配采样率与源数；失败实验延长音频，记录内存和输出槽位变化。注意力读到未来、全局归一化读完整句以及外层分块拼接分别检查。模型名和“低 RTF”不能代替前瞻、首帧时间和长录音峰值内存。

### N04　TF-GridNet：频率、时间与多通道输入

[ESPnet `tfgridnet_separator.py`](https://github.com/espnet/espnet/blob/be79590bb2ff26ffb01bc825c5f68cb9418b7f0d/espnet2/enh/separator/tfgridnet_separator.py)明确标为离线 `TFGridNet`，输入可为 `(批, 样本, 麦克风)`，固定 `n_imics`。其短时变换、频率/时间建模和注意力应与[论文](https://doi.org/10.1109/ICASSP49357.2023.10094992)及多通道扩展分开核对。

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

最小复现需要同一长录音的窗口、步幅、重叠匹配、泄漏处理及最终评分规则；不能仅对每块分别取最优 PIT 后拼接。E08-06 是一个不依赖模型权重的四点序列反例：两块各自的最优 PIT 都为 20 dB，第二块交换槽位后直接拼接的两条长流却都约为 −3.69 dB。失败实验还应在静音之后更换说话人，检查跨块匹配是否误连。cpWER、ORC-WER 和说话人归属 WER 的映射对象不同，应连同分段与转录协议固定。

### N10　SGMSE+、StoRM 与扩散迭代

分数模型沿噪声尺度学习条件生成，推理要经过多次采样或数值求解；预测—校正、随机种子和步数会影响速度与输出。官方入口为[SGMSE](https://github.com/sp-uhh/sgmse)和[StoRM](https://github.com/sp-uhh/storm)，相应论文与固定提交见[第三方来源说明](../THIRD_PARTY.md)。从各自 `enhancement.py` 读到 `sgmse/model.py`：SGMSE 的 `ScoreModel` 与 StoRM 的 `StochasticRegenerationModel`不是同一推理流程，不能仅按“扩散增强”标签共用 checkpoint。

最小复现应固定目标是增强还是去混响、采样器、步数、SDE/噪声日程、归一化和权重。失败实验包含静音、短脉冲、未见噪声与目标缺失，检查是否生成输入没有支持的音节。报告随机重复的离散程度与完整采样耗时；不能只计一次网络前向。代码 MIT 不自动授予训练语料和权重的再分发权。

### N11　ArrayDPS 与模型驱动的扩散分离

[ArrayDPS 官方仓库](https://github.com/ArrayDPS/ArrayDPS/tree/750ac2b7c75458f4ca5bad203dafda528f575e55)将多通道观测模型与扩散先验结合，来源及固定版本见[THIRD_PARTY.md](../THIRD_PARTY.md)。从 `separate.py` 到 `src/sampler.py` 的 `Sampler`，分别识别扩散先验、观测一致性项和阵列/传播参数；生成得像语音并不证明它来自指定声源。

最小实验固定双源混合与阵列模型，保存每次采样种子和条件参数；失败实验改变混响模型、源数或通道同步，观察数据一致性与听感是否冲突。SMS-WSJ 的生成程序不授予 WSJ 录音使用权，checkpoint 与示例音频也应独立检查。本书没有下载训练集、权重或把论文结果冒充已执行实验。

### N12　AudioSep：文本条件与音频预处理

[AudioSep 官方仓库](https://github.com/audio-agi/audiosep/tree/944583f18b84589dc965de3ad77525c945334252)的 `pipeline.py`先用 `build_audiosep()`加载分离模型和 CLAP 查询编码器，再由 `separate_audio()`将文本条件送入分离器。[正式论文](https://doi.org/10.1109/TASLP.2024.3520017)的目标是文本查询声音分离，不能代替注册语音 TSE 的身份评测。锁定入口会将音频转为单声道 32 kHz，不能把原始多通道空间信息保留作为默认假设。

最小复现用同一音频改变查询词并保存实际重采样参数、配置、checkpoint 与分块开关。失败实验查询录音中不存在的声音，以及两个同类声源，观察误提取和身份歧义；导出 `int16` 前还应检查浮点幅度，避免超范围转换带来的失真被误认为模型失真。代码 MIT 与托管主模型、媒体数据的许可分别记录，本书未运行该权重的推理。

### N13　Online SpatialNet：因果网络与跨调用状态

[NBSS 官方仓库](https://github.com/Audio-WestlakeU/NBSS/tree/cc42fc8ad2e6642c09b8f4169a85b4766dc22b7e)中的 [`models/arch/OnlineSpatialNet.py`](https://github.com/Audio-WestlakeU/NBSS/blob/cc42fc8ad2e6642c09b8f4169a85b4766dc22b7e/models/arch/OnlineSpatialNet.py)实现面向静止与移动说话人的多通道长时增强，论文比较在线掩码注意力、Retention 与 Mamba 时序模块。[论文预印本](https://arxiv.org/abs/2403.07675)说明算法设计；源码末尾还给出固定配置的因果前缀自测，用来检查附加未来帧是否改变既有前缀。本书当前环境没有 PyTorch、Mamba 与 CUDA，未实际运行这项自测，不能把源码中的测试代码当作本书已复现实验。训练入口还要结合 `SharedTrainer.py`、`configs/onlineSpatialNet.yaml`、数据加载器和 `generate_rirs.py` 阅读。

该仓库为 MIT，但运行依赖 PyTorch、Lightning、`mamba-ssm`、`causal-conv1d` 及相应 CUDA 环境。更重要的是，顶层 `OnlineSpatialNet.forward()` 调用每一层时传入的 `state` 为 `None`，没有把 `CausalConv1d` 等子层的状态作为顶层输入输出暴露。因此“网络是因果的”“整段计算量随长度近似线性”和“任意外层块可连续续算”是三个不同命题；固定版本不能仅凭类名证明第三项。最小实验应比较整段、任意分块且保留状态、每块重置三种输出，再测 251/1000/1024 帧、静止/移动源、麦数变化和无 Mamba/CUDA 的 CPU 路径；没有显式跨调用状态时应记录不等价，而不是用每块真实参考重排掩盖边界差异。

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
