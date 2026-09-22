# 空间处理与声源追踪：算法、实现和工业使用条件

核实日期：2026-09-22。对应正文第 2～5 章和第 9 章。这里按算法的输入、计算步骤和可检查结果整理源码，既包括语音前端，也包括直接相关的球阵录音与工业噪声源成像。后两类任务的输出不同，不能把声源功率图或 Ambisonics 解码结果当成增强语音。

## 阅读与复现方式

先运行本书的 [`ch02_05_baselines.py`](../examples/ch02_05_baselines.py) 和 [`ch06_09_baselines.py`](../examples/ch06_09_baselines.py)，确认时延正负、共轭、通道顺序和角度零点。再按下面的文件入口阅读外部实现。外部目录位于 `codes/upstream/_downloads/<项目 ID>/`，由锁定清单和获取工具管理。下载、导入、单次演示成功、论文实验复现和设备验收是不同结果。

本文的“最小实验”是建议的独立验证方案，未附运行结果的方案均未在本文中宣称通过。固定版本源码中的静态疑点集中放在末节；它们不能被隐藏在“有官方实现”的分类里。外部包缺少编译器、求解器、MATLAB、模型或测量数据时，仍可阅读源码，但不能据此声称运行验证完成。

共用数组约定为多通道时频谱 `M × F × T`；`M` 为通道数、`F` 为频点数、`T` 为帧数。麦克风位置在本书教学包中为 `M × D`，pyroomacoustics 则为 `D × M`。输入转换只能转置坐标排列，不能顺带改变通道顺序。角度单位、方位角零点、俯仰角与余纬角、SH 系数归一化均要在适配层显式转换。

## 固定版本和源码许可

完整提交及入口保存在 [总锁定清单](../SOURCES.lock.json) 中。下表的日期是核实日，不表示项目当天发布了新版。

| 项目 ID | 固定提交 | 代码许可与额外前提 |
|---|---|---|
| pyroomacoustics | `0dd39f2614b7fc44b2cc63dbe7d60f4641068890` | MIT；本书原有 0.10.0 版本，算法与 C++ 房间模拟器的依赖不同 |
| doatools | `9469db201e0418aef6b97583ef54b6fec2769502` | MIT；理论研究工具，稀疏求解需另核优化器 |
| sound-field-analysis | `4b03ee123d98370c55f744c4f8d7c955fbc099f1` | MIT；当前代码许可不应从介绍早期版本的论文反推；SOFA/HRIR 数据另外核对 |
| spherical-array-processing | `f192aac652b023ee4ab8673adce20ec13bf5450c` | BSD-3-Clause；MATLAB，另依赖作者的阵列响应和球谐变换库 |
| spatial-audio-framework | `18fd5aba46e20787b51f28f7197a68506c965c07` | 核心 ISC；`saf_tracker`、`saf_hades` 等可选模块 GPLv2，BLAS/FFT 后端另有条款 |
| frida-original | `ff5d51e498805b862c342dd216ccfffb22444b7f` | MIT；原实验采用 Python 2.7，录音另外获取并核对许可 |
| acoular | `13d3d7df74ac1a8135c7ec71da098cbbc03d8652` | BSD-3-Clause；声学校准和测量文件需单独确认 |
| sbl | `d4bba35e9b60907d3024473ba5a41046450baae0` | GPL-3.0；Python/MATLAB，旧 Python 演示不构成现代环境兼容性保证 |
| robustsbl | `d746266a1336d4467f60b6f7b7e8b4695a01d26d` | MIT；MATLAB，部分模式和数据生成使用统计工具箱函数 |
| btk20 | `feff19ec8bcb770f6530fe280dc3ccafc2f5984a` | 根许可 MIT；C++/Python，GSL、SWIG、libsndfile 等依赖分别核对 |
| smpphat | `6fd33e6eb3251078a4cd9793dde909e2500265cc` | GPL-3.0；C/FFTW 单精度库，固定构建文件含 x86 SIMD 选项 |

许可证依据为各官方仓库的 [doatools LICENSE](https://github.com/morriswmz/doatools.py/blob/9469db201e0418aef6b97583ef54b6fec2769502/LICENSE.md)、[sfa LICENSE](https://github.com/AppliedAcousticsChalmers/sound_field_analysis-py/blob/4b03ee123d98370c55f744c4f8d7c955fbc099f1/LICENSE)、[Politis LICENSE](https://github.com/polarch/Spherical-Array-Processing/blob/f192aac652b023ee4ab8673adce20ec13bf5450c/LICENSE.md)、[SAF LICENSE](https://github.com/leomccormack/Spatial_Audio_Framework/blob/18fd5aba46e20787b51f28f7197a68506c965c07/LICENSE.md)、[FRIDA LICENSE](https://github.com/LCAV/FRIDA/blob/ff5d51e498805b862c342dd216ccfffb22444b7f/LICENSE) 和 [Acoular LICENSE](https://github.com/acoular/acoular/blob/13d3d7df74ac1a8135c7ec71da098cbbc03d8652/LICENSE)。这些条款只用于说明相应项目的源码，不能替本书所有者选择发布许可证。

SBL、RobustSBL、BTK 和 SMP-PHAT 的依据分别是固定提交的 [SBL LICENSE](https://github.com/gerstoft/SBL/blob/d4bba35e9b60907d3024473ba5a41046450baae0/LICENSE)、[RobustSBL LICENSE](https://github.com/NoiseLabUCSD/RobustSBL/blob/d746266a1336d4467f60b6f7b7e8b4695a01d26d/LICENSE)、[BTK LICENSE](https://github.com/kkumatani/distant_speech_recognition/blob/feff19ec8bcb770f6530fe280dc3ccafc2f5984a/LICENSE) 和 [SMP-PHAT LICENSE](https://github.com/FrancoisGrondin/smpphat/blob/6fd33e6eb3251078a4cd9793dde909e2500265cc/LICENSE)。独立上游目录不改变本书代码许可；取得状态以 [SOURCE_STATUS.json](../SOURCE_STATUS.json) 为准，下面没有运行记录的实验均为建议方案。

## 基础模型、统计估计与校准

### 1. 远场与近场导向矢量

对应正文 §2.2～2.3。输入是位置、频率、声速和候选方向或声源坐标，输出是每个通道相对参考麦的复数响应。远场以平面波投影计算相对距离，近场用每个麦到声源的欧氏距离，并决定是否保留距离衰减。源码先读本书 `geometry.py`，再读 pyroomacoustics 的 `doa/doa.py` 中 `ModeVector`。

最小实验采用三麦直线阵，先把源放在近距离，再逐步移远，同时保持方向不变。检查近场相对时延趋近远场值；只比较去除公共相位后的响应。工业中还需将挡板、麦指向性和通道校准乘入流形。把米误写成厘米会改变整个频段的相位，不是增加对角加载可以补救的问题。

### 2. 镜像法与射线追踪房间模拟

对应 §2.4。pyroomacoustics 的 `room.py`、`libroom_src/` 分别提供场景接口和计算实现；镜像法显式构造反射路径，射线追踪近似统计晚期传播。二者组合不代表已经模拟衍射、结构传声或设备非线性。输入还包括墙面材料、采样率、麦指向性和最大反射阶数，输出是 RIR 或多通道波形。[官方项目说明](https://github.com/LCAV/pyroomacoustics)。

建议先用无反射房间验证首达时刻和距离衰减，再只留一面反射墙验证镜像路径长度，最后增加高阶反射检查能量衰减。训练数据应改变房间、源位和通道扰动，并留下未见房间做测试。单独调整 `max_order` 不等价于准确控制某个实测房间的混响时间。

### 3. STFT 与加权重叠相加重构

对应 §2.5。本书 `spectral.py` 用于检查窗、帧移和边界补齐；外部阅读入口是 pyroomacoustics 的 `transform/stft.py`。分析窗和合成窗的乘积经过移位相加应形成非零、可补偿的包络。输出长度、启动补零和尾帧补齐应和原始采样时间一起记录。

最小实验分别用脉冲、常量和不整帧长度的随机信号重构；不能只用频点对齐的正弦。接入实时音频时把算法前瞻与累计输入帧数分开计算：块长并不自动等于全部算法延迟。过短窗会使房间卷积的逐频乘法近似变差，过长窗会降低时变跟随能力。

### 4. 批处理、加权与递推空间协方差

对应 §2.5、§5.4、§5.9。本书 `covariance.py` 的回归测试检查矩阵厄米性、非负二次型和权重归一化。掩码协方差按掩码和归一化；全零权重表示没有可用统计，接口明确拒绝。非零掩码整体乘同一正数不会改变归一化二阶矩，所以不能仅因绝对掩码和很小就判断无效；实现先按最大权重缩放以避免上下溢。E02-06 用 $10^{-200},1,10^{200}$ 三种共同尺度验证这一点，但缩放不增加独立样本，也不改善统计置信度。递推方式还需保留历史权重，否则启动阶段统计与稳态统计的尺度可能不同。

实验将同一组快拍分别整体处理和分块递推，比较约定一致时的结果；再给全零掩码、单个快拍和完全相关通道。工业中同时记录有效快拍数、条件数、遗忘因子和目标泄漏检测。矩阵维度正确、求解没有报错，仍不足以证明它描述的是噪声而非目标。

### 5. AIC 与 MDL 源数估计

对应 [§4.1.1](../../chapters/04_doa-estimation.md#sec-4-1-1) 和 [§4.10](../../chapters/04_doa-estimation.md#sec-4-10)。本书 `doa.py::mdl_source_count` 提供复高斯、空间白噪声模型的 MDL 教学基线；输入为任意顺序的严格正实特征值和独立快拍数，输出所选源数与全部候选评分。它拒绝秩亏、非法类型和不足以支持满秩样本协方差的快拍数，不静默加载，也不把评分写成概率。

外部入口仍为 doatools 的 `estimation/source_number.py` 中 `aic`、`mdl`、`ld_stat`。其接口接受协方差或**升序**特征值，并需要快拍数；本书接口不接受协方差矩阵。锁定版本的 `mdl` 保留公共惩罚 $\tfrac12\ln N$，本书去掉该常数，所以直接比较评分时要先统一口径，源数最小值不受影响。上游 `ld_stat` 直接计算特征值乘积，本书改用平移后的对数计算，以覆盖极大、极小尺度。[作者 API](https://morriswmz.github.io/doatools.py/references/doatools.estimation.source_number.html)；[锁定提交的官方源码](https://github.com/morriswmz/doatools.py/blob/9469db201e0418aef6b97583ef54b6fec2769502/doatools/estimation/source_number.py)。版本同时登记在 `SOURCES.lock.json`。

E04-05 使用指定谱 $[9,4,1.1,0.9]$ 和 $N=100$，四个 MDL 总分为约 171.355476、86.437847、28.636055、34.538776，选择 2。表格、算例、`exercises_spatial.py` 与 `test_codes_mdl.py` 对应；测试期望值另由 Decimal 高精度标量公式产生，覆盖所有排序、共同尺度 $10^{-300}$ 至 $10^{300}$、最小正浮点数、纯噪声、最后一个候选和拒绝输入。这是本书手算及数值回归，不是数据集检测正确率实验。

公式核查使用 Wax 与 Kailath 的 *Determining the number of signals by information theoretic criteria*，ICASSP **1984**，§II～IV、式(10)～(15)，[DOI](https://doi.org/10.1109/ICASSP.1984.1172389)及[作者上传原文](https://www.researchgate.net/profile/Mati-Wax/publication/3177764_Detection_of_signals_by_information_theoretic_criteria/links/56cacde408aee3cee54041cd/Detection-of-signals-by-information-theoretic-criteria.pdf)。[1985 年期刊论文](https://doi.org/10.1109/TASSP.1985.1164557)另题为 *Detection of signals by information theoretic criteria*；ResearchGate 的该期刊条目挂载的是 1984 年会议稿，引用定位以 PDF 的实际版本为准。核查日期：2026-09-22。

E04-06 已加入实际重复抽样，入口为 [`mdl_repeated_trials.py`](../examples/mdl_repeated_trials.py)，也由 `exercises_spatial.py` 执行。四麦半波长线阵，1000 Hz、343 m/s、两源 $\pm30^\circ$；每组 200 次独立圆对称复高斯快拍试验，使用 PCG64 与 `SeedSequence([20260922, 组编号])`。七组分别改变快拍数 100/16、每源功率 1/0.1、独立/完全相干源，以及无源时白噪声/协方差为 $\operatorname{diag}(9,4,1,1)$ 的有色噪声。白噪声每通道方差为 1；按阵列平均功率定义的双源 SNR 为 3.01/−6.99 dB。源和噪声每次重新生成，协方差不减样本均值、不加载；这里没有波形、STFT 或音频采样率。

完整计数与逐条件 Wilson 95% 比例区间见 [§4.10 的 E04-06](../../chapters/04_doa-estimation.md#sec-4-10)。NumPy 2.5.3 下，独立双源功率 1、100 快拍的 200 次均输出 2，而降至 16 快拍时为 163 次；有色纯噪声的 200 次均输出 2，尽管物理源数为 0。相干双源有 199 次输出 1，这与总体信号秩 1 一致，但不等于检出两个物理源。区间只描述指定条件下输出与物理源数相符的重复事件，不是单次 MDL 置信度；全相符也不保证以后不失败。[区间公式：NIST/SEMATECH §7.2.4.1](https://itl.nist.gov/div898/handbook/prc/section2/prc241.htm)。

这组实验没有检验重叠 STFT 帧、不同源强比、真实房间或时间迟滞。重叠帧不天然独立，把帧数全部当成独立快拍会改变评分口径；工程中的源数检测还应评估迟滞，避免每帧改变下游子空间维度。不能由本书七组数学仿真推出真实录音检测率。

### 6. 前向与前后向空间平滑

对应 §4.6。doatools 的 `estimation/preprocessing.py::spatial_smooth(R,l,fb)` 将平移子阵协方差平均；这里 `l` 是子阵数量，输出维度为 `M-l+1`，不是很多论文中的子阵长度定义。[作者接口与 Pillai–Kwon 文献定位](https://morriswmz.github.io/doatools.py/references/doatools.estimation.preprocessing.html)。

建议用 8 元 ULA、两条完全相关的平面波，先检查未经平滑的信号协方差秩，再取 3 个长度为 6 的子阵比较。平滑能改善特定模型的秩，但付出有效孔径和可处理源数的代价。非均匀阵、挡板方向响应或未知互耦不满足相同平移流形时，不能直接滑动矩阵下标冒充空间平滑。

## 定位算法和搜索加速

### 7. GCC-PHAT 与物理时延裁剪

对应 §4.2。本书 `doa.py::gcc_phat` 明确正时延约定、FFT 补零、物理 lag 截取和三点插值；ODAS 的 `src/signal/xcorr.c`、`src/system/freq2xcorr.c` 可作为块处理参考。补零到两段长度之和减一只保证未加权互相关的有限长度等价；PHAT 归一化改变频谱后，逆变换不再保证相同的有限支撑，FFT 长度仍可能改变相关值。PHAT 消除互谱幅度权重，但低能量频点的相位会变得不可靠，因此归一化下限与有效频点筛选都是算法的一部分。

建议给一个 0.5 ms 延迟的宽带源，随后改成单频正弦、两源和静音。分别报告主峰、次峰、峰背比和最大允许 lag。亚采样插值只细化相关峰形，不能凭空恢复窄带信号缺少的可辨识信息。先固定通道号，再固定 `t_1-t_2` 的方向，避免定位后又人工翻转符号。

### 8. SRP-PHAT 与近场三维网格

对应 §4.3、§4.7.1。先读本书 `doa.py::srp_phat`，再读 pyroomacoustics 的 `doa/srp.py`，理解麦对证据如何按候选传播时延相加。这里的两份代码作为远场方向 SRP 参考；近场三维 SRP 则保留正文球面传播与角度—距离网格的原理索引，不能仅因外部 API 列出 `mode='near'` 和 `r` 就标成已取得该变体实现。

锁定版本 pyroomacoustics 0.10.0 的静态调用链有明确限制。`doa/srp.py` 构造函数把 `mode/r` 传给 `DOA.__init__`，但 `doa/doa.py:289` 创建 `ModeVector(self.L, self.fs, self.nfft, self.c, self.grid)` 时没有传 `mode`，因此该对象仍采用第 32 行的默认 `mode='far'`。同时，候选 `r` 仅保存到 `self.r`，第 234～280 行的 `GridCircle/GridSphere` 构造没有把它形成距离维；`srp.py:116` 的评分随后直接使用这个 `self.mode_vec`。[固定提交 DOA 构造与导向源码](https://github.com/LCAV/pyroomacoustics/blob/0dd39f2614b7fc44b2cc63dbe7d60f4641068890/pyroomacoustics/doa/doa.py)；[固定提交 SRP 评分源码](https://github.com/LCAV/pyroomacoustics/blob/0dd39f2614b7fc44b2cc63dbe7d60f4641068890/pyroomacoustics/doa/srp.py)。

据此，即使构造参数写为 `mode='near', r=np.array([2.0])`，也没有将近场选项传入实际相位表。上述判断来自默认参数、调用实参与网格构造的静态核对，本次没有运行上游包或声学数值实验。若要实现近场搜索，必须另行构造包含距离的候选位置，并使球面传播时延进入 SRP 评分；不能仅补传一个 `mode` 参数就宣称完整三维搜索已完成。

建议在真值落格点、落在格点之间、落在搜索区外三个条件下比较峰值。源在区外时，算法仍可能返回边界上的最大值，因此“找到峰”不等于位置可信。部署时缓存静态时延表、限制重复麦对、记录选峰间距，并在阵列几何或采样率变化后重建表。

### 9. 分层 SRP-PHAT 与方向性麦对筛选

对应 §4.3 的工程扩展。ODAS 的 `src/module/mod_ssl.c` 组织定位流水，`src/signal/scan.c`、`src/signal/spatialindex.c` 和配置中的 `scans` 描述搜索层级；配置还包含相关插值倍率 `interpRate`。分层搜索先在粗球面上找候选，再搜索附近细方向，方向响应可用于排除缺少有效声学信息的麦对。

建议同一输入分别做全细网格和分层网格，比较漏峰率、最大角度误差、查表次数及内存。两源距离接近时，粗层可能合成一个峰，后面的细化无法恢复已经丢弃的区域。因此候选数量、邻域宽度和多个峰的保留规则应一起标定。依据为 [ODAS 官方源码](https://github.com/introlab/odas) 与 [Grondin–Michaud 的方法论文](https://arxiv.org/abs/1812.00115)。

### 10. SVD-PHAT 与多源扩展

这是 SRP 搜索加速的补充研究。固定几何与频点后，SRP 映射矩阵可先做截断 SVD；在线把 PHAT 观测投影到低维空间，再搜索候选方向。保留秩控制近似误差与每帧成本；多源扩展还需逐次投影已解释的分量。[单源原论文](https://arxiv.org/abs/1811.11785)、[多源原论文](https://sls.csail.mit.edu/publications/2019/Grondin_Interspeech-2019.PDF)。

最小实验应先对同一个 SRP 矩阵比较完整乘法与截断乘法，测量分数误差，再比较 DOA；不能只报告投影速度。高频、复杂几何、较密网格可能改变所需秩。本文尚未确认与这两篇论文唯一对应且许可明确的作者实现，因此此项保留原理索引，不以普通 SVD 库替代算法源码。

#### SMP-PHAT：合并重复基线，而非截断矩阵

另一条降低 SRP 运算量的路线是合并麦对的 PHAT 互谱。SMP-PHAT（Steered response power by Merging Pairs with PHAse Transform）利用远场下相同基线的时延相同：基线向量相同的麦对先在频域求和，再共用一次逆变换和时延查表；向量相反时，先取相应互谱的共轭。只有长度相同且相互平行才满足这一条件，不能把所有平行麦对合并。[Grondin 等 2022 预印本，§3 算法2～3](https://arxiv.org/html/2203.14409v1)。

官方 `smpphat` 的阅读顺序为 `demo/ssl.c` → `src/system.c::scmphat_call` → `smp_construct/smp_call`；同文件的 `srp_construct/srp_call` 提供未合并对照，`src/signal.c` 定义阵列和方向数据。输入须保持录音通道与坐标次序一致，候选方向用三维单位向量，几何长度与声速采用相容单位。输出是候选方向及评分，不是经过校准的方向概率。[固定提交算法源码](https://github.com/FrancoisGrondin/smpphat/blob/6fd33e6eb3251078a4cd9793dde909e2500265cc/src/system.c)。

固定版本有三项复现条件：

1. `CMakeLists.txt` 要求 `pkg-config` 和 `fftw3f`，并硬编码 `USE_SIMD`、`-msse3`、`-ffast-math`。非 x86 环境需要另行适配并记录补丁，不能把作者的硬件测速直接移到另一平台。
2. `wav_construct` 直接读取固定头结构，只接受 16 位 PCM；它不是通用 RIFF 块解析器。运行前检查 WAV 头、通道数、采样率和数据区，不能仅凭文件扩展名判断相容。
3. `smp_construct` 的合并容差为 `1e-5`，论文算法文本给出 `1e-4`。实现把同一常数用于基线长度差与点积残差，改变坐标单位可能改变分组；应保存实际坐标单位、容差和分组结果。

本书于 2026-09-23 实际编译并调用上述固定提交，使用 Apple clang 21、arm64 与 FFTW 3.3.10 单精度静态库。编译直接使用未修改的 `system.c`、`signal.c`，不经过硬编码 x86 选项的上游 CMake；这只改变构建入口，不修补算法。两个处理对象按“创建、调用、复制结果、销毁”的顺序分别运行，因为上游析构函数会执行全局 `fftwf_cleanup`，不能让另一个对象的计划跨过该调用继续存活。[FFTW 计划生命周期说明](https://fftw.org/fftw3_doc/Using-Plans.html)。

输入为半径 0.032 m 的四麦菱形、16 kHz 采样率、声速 343 m/s、64 点原频谱与 4 倍插值，扫描 24 个相隔 15° 的水平单位向量。0° 指向 +y，正角转向 +x，真值为 45°。DC 为 1，正频率 bin1～31 为单位幅度 PHAT 互谱，原 Nyquist bin32 置零，保证实信号频谱端点约定成立。逆变换不归一化，分数不是概率。固定输入、完整逐方向结果和源码摘要见 [运行报告](../reports/smpphat_reference.json)。

规则菱形的 6 个麦对合并为 4 组；将第四麦沿 y 移动 0.5 mm 后，分组变为 6 组。独立的有符号时延查表与直接离散傅里叶求和确认，规则几何下理论 SRP/SMP 分数最大差约为 $1.99\times10^{-13}$。但原 C 程序的最大差为 76.5567，SRP、SMP 相对正确基线的最大误差分别为 95.1649、171.7215。两者的峰方向碰巧都是 45°，不能据此判定等价检查通过。

问题位于 `system.c` 的两处查表索引：负的 `roundf` 结果先被转换成 `unsigned int`，随后才加中心偏移。该浮点值不在无符号类型的表示范围内，行为未定义；本机编译结果将负值转成 0。例如应取索引 3 的一项实际取了中心索引 9，规则案例共有 66 项 SRP 索引不符。使用程序实际索引再做独立傅里叶求和，SRP/SMP 分数误差仅约 $3.35\times10^{-5}$/$2.36\times10^{-5}$，因而将主要差异定位到查表而非变换约定。[Clang 浮点转换说明](https://clang.llvm.org/docs/UsersManual.html)。

扰动阵列的原 C 程序虽然给出 SRP 与 SMP 逐点相同的分数，两者相对正确基线仍有约 96.0504 的最大误差。这也说明两个程序互相吻合不能替代独立基线。当前报告明确记为 `failed_portability`，保留未修改的上游源码；不能把这一版本视为已经验证的 ARM 定位实现。

实验使用合成远场互谱，不含真实录音、噪声、混响或运行速度评测。近场时，相同基线处于不同位置会产生不同距离差，合并不再有远场等价保证；无重复基线的阵列可能没有节省。后续速度测试仍需分别记录初始化、逐帧耗时、内存与浮点选项。

### 11. Bartlett 与 Capon 空间谱

对应 §4.5。本书 `bartlett_spectrum` 计算候选响应下的输出功率；`capon_spectrum` 通过协方差线性求解计算自适应谱。两者共享导向矢量，但归一化和权重不同。外部交叉入口为 Acoular `fbeamform.py` 的 `BeamformerBase`、`BeamformerCapon`，其声压成像口径不应直接混入本书的单位幅度谱。

建议在同一个解析协方差上检查双麦手算，再分别缩放输入功率和导向矢量。工业成像还要记录 CSM 对角去除与功率归一化；这些设置改变幅值解释。Capon 的窄峰不自动意味着更准确，少快拍或目标流形失配会产生错误抑制。

### 12. MUSIC 与频点归一化 NormMUSIC

对应 §4.6。pyroomacoustics `doa/music.py` 和 `doa/normmusic.py` 共享子空间计算；频点归一化用于改变宽带伪谱合并时各频点的影响。它们输出扫描谱和峰方向，仍需要源数、噪声子空间维度和峰间距。

建议令一个频点信号很强、另一个频点很弱，比较归一化前后方向；再将弱频点换成纯噪声。按峰值归一化可能让低可靠频点获得过大权重，因此有效频带选择仍不可省略。MUSIC 的理想零分母应采用明示数值处理，不能把无穷大误认为一个有统计校准的置信度。

### 13. root-MUSIC

对应 §4.6 和总对比。doatools `estimation/music.py::RootMUSIC1D` 针对 ULA，把噪声子空间投影矩阵的对角线和变成多项式系数，从单位圆附近的根恢复相位，再转换为方向。输入是 ULA 协方差、源数、波长与麦间距；默认半波长间距必须显式检查。[作者接口与 Barabell/Rao–Hari 原文定位](https://morriswmz.github.io/doatools.py/references/doatools.estimation.music.html)。

建议先用单源无噪协方差比较 MUSIC 扫描峰和 root-MUSIC，再加入相干双源、几何扰动和空间混叠。对共轭倒数根需按算法规则选取，不可把所有近圆根都计作声源；成功找到指定数量的根也不保证方向正确。免网格只消除了网格量化，未消除特征分解和多项式数值误差。

### 14. LS-ESPRIT、TLS-ESPRIT 与几何约定

对应 §4.6。本书 `esprit_ula` 实现最小二乘旋转矩阵；外部阅读入口为 doatools `estimation/esprit.py`。LS 只把一侧当作拟合目标，TLS 同时考虑两侧子空间块误差；两者都依赖平移不变子阵，不能对任意几何直接使用。

建议用固定波长改变间距，检查旋转特征值相位到角度的反解；交换两个子阵后，应先解释相位符号变化再转换角度。TLS 的矩阵划分和截断秩需与源码接口一致。存在近场曲率、位置误差或相干源时，低残差不保证真实方向正确。

### 15. CSSM 宽带聚焦

对应 §4.6.1。固定 pyroomacoustics 的 `doa/cssm.py` 先为各频带产生候选峰，选参考频点，构造聚焦矩阵并循环聚合协方差。源码入口依次为 `_process`、`_coherent_sum`、继承的子空间分解。[Wang–Kaveh 原文](https://doi.org/10.1109/TASSP.1985.1164667)。

建议固定一组双源频域快拍，改变初值、参考频点和迭代数，报告方向误差与聚焦矩阵条件数。被剔除的频点必须与协方差、权重和候选峰同步筛选。固定 0.10.0 版本的这个关联存在静态疑点，见末节；因此可读源码不意味着该版本所有边界输入已验收。

### 16. WAVES 加权信号子空间

对应 §4.6.1。`doa/waves.py` 将各频带聚焦后的信号子空间按特征值相关权重拼接，再对拼接矩阵做 SVD。它与 CSSM 的区别在聚合对象和权重，不是简单把 CSSM 改名；初始化、频点剔除和噪声特征值估计都影响结果。[Di Claudio–Parisi 原文](https://doi.org/10.1109/78.950774)。

建议让部分频点只包含噪声，检查权重是否减弱该频点影响；再设强弱源功率差，观察弱源是否被压掉。阅读 `_construct_waves_matrix` 时逐项追踪 `freq_bins[j]` 与 `C_hat[j]` 的对应。WAVES 与 CSSM 相同的频点筛选疑点也列在末节。

### 17. TOPS 投影子空间正交性检验

对应 §4.6.1。`doa/tops.py` 为每个方向组合跨频子空间正交性矩阵，以最小奇异值构造谱；它不依赖 CSSM 的初始方向聚焦，但仍依赖正确的源数和多个频点。[Yoon–Kaplan–McClellan 原文](https://doi.org/10.1109/TSP.2006.872581)。

建议至少用两个不连续频点，并移动参考频点的位置，检查输出是否对频点排列保持相同物理含义。排列频点不会改变声场，但若代码错误地把“频点列表下标”当成“FFT bin 编号”，结果会改变。固定版本的具体静态索引证据列在末节；生产选型前需完整数值反例和修复后回归。

### 18. FRIDA 连续角度恢复

对应 §4.7。pyroomacoustics 的 `doa/frida.py` 组织协方差可见度与恢复过程，`doa/tools_fri_doa_plane.py` 实现映射与多频重建。先理解去除自功率后的互谱观测，再读 `max_four`、`max_ini`、`max_iter`、`G_iter` 与 `signal_type`，不能只调整一个“精度”参数。[原论文](https://doi.org/10.1109/ICASSP.2017.7952744)。

建议在非均匀平面阵上放两个不落网格的独立源，固定随机种子比较重建残差和角度误差。多个随机初值增加计算成本，也可能得到不同局部结果。原论文专库 `figure_doa_synthetic.py`、`figure_doa_separation.py`、`figure_doa_experiment.py` 分别定位仿真、分辨率和录音实验；旧环境与音频获取单列，不能直接执行过时安装脚本。[原实验仓库](https://github.com/LCAV/FRIDA)。

### 19. 组稀疏定位与协方差稀疏匹配

对应 §4.7。doatools `estimation/sparse.py` 的 `GroupSparseEstimator` 直接拟合多快拍，按候选方向共享稀疏支持；`SparseCovarianceMatching` 则拟合向量化协方差和非负源功率，要求源间不相关。前者的复幅度与后者的功率不可使用相同误差解释。[作者模型和求解接口](https://morriswmz.github.io/doatools.py/references/doatools.estimation.sparse.html)。

建议先用落格点双源，再把源移到相邻格点中间，比较支持泄漏和正则参数敏感性。求解器返回可行解、峰数达到要求，只表示数值过程完成。工程记录应包括字典归一化、正则目标、求解器版本、终止容差、迭代次数和残差；可行性约束太严时应保留失败状态。

#### 多快拍、多频稀疏贝叶斯学习

稀疏贝叶斯学习（Sparse Bayesian Learning，SBL）在候选方向上估计源功率超参数，与前面的组稀疏罚项不是同一个求解器。作者 `gerstoft/SBL` 提供 `SBL_MF_Python/sbl.py::SBL`，MATLAB 对应 `SBL_MF_matlab/SBL_v4.m`；演示从 `Beamforming_demo.m` 开始读，参数定义见 `SBLSet.m`。[Gerstoft 等 2016 原论文](https://doi.org/10.1109/LSP.2016.2598550)、[固定 Python 核心](https://github.com/gerstoft/SBL/blob/d4bba35e9b60907d3024473ba5a41046450baae0/SBL_MF_Python/sbl.py)。

Python 核心输入字典 `A` 为 `M × G × F`，观测 `Y` 为 `M × L × F`；`G` 是候选方向数，`L` 是快拍数。与本文通用的 `M × F × T` 相比，最后两轴要交换。字典承载传播相位约定，不能只转换数组形状而不核对角度零点和复指数符号。返回值为候选功率 `gamma` 和迭代报告，不直接是角度。

沿源码阅读时，依次检查样本二阶矩、功率初始化、相对功率剪枝、逐频求解、功率更新和噪声更新。`options.Nsource` 用于选峰，噪声估计的分母为 `M−Nsource`；该实现需要给定源数且满足相应维数条件，不能称为“自动免源数”。每频噪声模型为标量乘单位阵，空间有色噪声不属于这个模型。

核心 Python 文件依赖 NumPy；完整演示与 MATLAB 环境分别检查。作者 README 说明 Python 版本自 2020 年夏后未使用，旧示例中的外部数据链接也不属于已取得的代码。工业使用还要记录有效频带、共同方向支持的时间跨度、字典归一化、剪枝阈值、停止误差和最大迭代数；运动中的声源不一定满足同一批快拍共享支持的假设。

本书已用 [SBL 复现脚本](../examples/reproduce_sbl_reference.py)实际调用上述固定 Python 核心，没有复制或改写上游 GPL 算法。2026-09-22 的执行环境为 Python 3.13.12、NumPy 2.5.3、macOS arm64；[逐例 JSON 报告](../reports/sbl_reference.json)保存上游提交、核心与调用脚本的 SHA-256、环境、完整功率谱、实际迭代误差序列和协方差。运行前检查上游 `HEAD`、受 Git 跟踪文件的修改状态及核心文件摘要，不把未跟踪文件也称为已经核查。

实验使用 4 麦均匀线阵，声速 343 m/s、频率 1000 Hz、间距 0.1715 m，候选范围 −80°～80°、步长 1°。采用本书 0° 为宽侧、正角朝 +x 的约定，字典第 $m$ 个阵元为 $e^{+j\pi m\sin\theta}$，每列平方范数为 4。源快拍为两个相互独立的单位功率圆对称复高斯序列，共 200 帧；这些是窄带复数样本，不是音频录音，因而没有音频采样率或 STFT 窗参数。

随机种子为 20260922，各条件共享底层源与白噪声随机样本，没有逐次归一化。基本条件的每麦噪声方差为 0.02，总源功率与噪声功率的总体比为 $2/0.02=100$，即 20 dB；低信噪比条件仅将该方差改为 20，即 −10 dB。有色条件仅将噪声相关矩阵改成 $[0.9^{|i-j|}]$，仍保持每麦方差 0.02。停止条件为相对功率更新小于 $10^{-5}$，最多 1000 次更新；剪枝比为 $10^{-4}$、固定点参数为 1。完整参数见脚本与报告。

| 条件 | 真方向（°） | SBL 峰方向（°） | 更新次数 | 停止原因 |
|---|---|---|---|---|
| 网格上、20 dB 白噪声 | −30，30 | −30，30 | 309 | 满足更新阈值 |
| 两个真角均平移 0.5° | −29.5，30.5 | −30，31 | 168 | 满足更新阈值 |
| −10 dB 白噪声 | −30，30 | −21，31 | 1000 | 达到次数上限 |
| 20 dB 空间有色噪声 | −30，30 | −30，30 | 259 | 满足更新阈值 |
| 同一基本输入，错设源数为 1 | −30，30 | 30 | 1000 | 达到次数上限 |

独立校验没有用 SBL 自身生成期望值：±30° 的导向列可手算为 `[1,−j,−1,j]` 与 `[1,j,−1,−j]`，两列正交；基本条件的总体协方差特征值因此为 4.02、4.02、0.02、0.02。逐通道、逐快拍的标量求和与矩阵乘法所得样本二阶矩最大差小于 $8\times10^{-15}$。方向峰另由严格局部极大值检查，峰不足时不补索引 0；错设一个源的结果保留漏源数，不计算容易误导的双源平均误差。上游报告的迭代字段从 0 开始，本表用实际执行次数，即该字段加 1。

离网格样例的两个峰各偏离真值 0.5°，说明这次输出仍受离散候选限制。有色样例这次找对两个网格点，却不能证明白噪声模型对有色噪声普遍有效。低信噪比与错设源数两行没有满足停止阈值，不能称为收敛解。报告还保留同一输入、同一假定源数的 MUSIC 峰值；这些单次、配对结果不构成成功率估计、论文性能复现或算法优劣排名。达到小更新误差本身也不证明方向正确。

取得锁定源码后，可在仓库根目录执行；没有源码时程序明确报错，不自动联网或安装依赖：

```bash
.venv/bin/python codes/examples/reproduce_sbl_reference.py --output codes/reports/sbl_reference.json
.venv/bin/python -m unittest tests.test_codes_sbl_reference -v
```

#### RobustSBL：异常快拍的损失函数选择

少量大幅快拍可能主导普通二阶矩。RobustSBL 根据快拍相对当前散布矩阵的距离调整权重，再更新方向功率；Gauss 模式使用普通权重，t、Huber 和 Tyler 模式对大距离观测采用不同权重。它改变的是统计损失和更新中的快拍权重，不是简单把输入限幅。[Mecklenbräuker 等 2024，Signal Processing 220，109461，§3～4](https://doi.org/10.1016/j.sigpro.2024.109461)；[作者公开稿 §III-B～E](https://arxiv.org/html/2301.06213v2)。

机构仓库 `NoiseLabUCSD/RobustSBL` 的入口为 `_common/SBL_v5p12.m::SBL_v5p12`，参数由 `_common/SBLSet.m` 定义。函数输入仍为字典 `M × G × F` 与快拍 `M × L × F`；第一个输出是峰索引 `Ilocs`，不是残留头注所写的功率向量。阅读时沿 `method` 检查 `SBL-G`、`SBL-T`、`SBL-H`、`SBL-Tyl`，同时核对 `upar`、已知源数、选峰间隔与迭代报告。[固定作者源码](https://github.com/NoiseLabUCSD/RobustSBL/blob/d746266a1336d4467f60b6f7b7e8b4695a01d26d/_common/SBL_v5p12.m)。

`SingleMC_SNR_fixedDOA.m` 和 `SingleMC_SNR_randomDOA.m` 是实验入口；固定角脚本当前只执行 `for isnr=6`，直接运行不等于复现完整 SNR 曲线。Huber 参数计算和部分数据生成调用 `chi2inv`、`chi2cdf`、`chi2rnd`，需要相应 MATLAB 统计功能。Tyler 与 t 损失的一致性辅助函数在核心文件尾部，不是另一个待下载工具包。Tyler 主要确定散布形状，比较功率尺度前还须核对归一化。

最小实验建议在同一组窄带双源快拍上先保留 Gaussian 对照，再按明确比例和幅度加入异常快拍，比较 Gauss 与 Huber 模式。每个条件保留相同基底样本，记录异常位置、损失参数、检测失败、角误差和迭代数；重复次数及区间算法随结果报告。该方案尚未执行，也不能把模型中的异常快拍直接称为真实风噪、削波或混响录音。

## 波束、球阵与工业声源成像

### 20. DSB、超指向与加载 MVDR

对应 §5.2～5.4。本书 `beamforming.py` 给出从固定权重到估计噪声协方差的最小路径。对角加载改变的是矩阵的特征值和权重范数；以均值特征值归一化的加载系数与直接添加绝对功率不能混用。每次设计都同时输出目标响应、WNG 和噪声输出。

E05-05 给出已执行的有限 INR 反例：半波长双麦、目标 0°、干扰 30°、白噪声方差 1。线性 INR 为 1、10、100 时，干扰方向响应分别为 −9.03、−23.84、−43.10 dB，真正零点分别为 44.82°、32.03°、30.21°。这是指定理想协方差的单频手算，不是语音实验；有限噪声下的 MVDR 抑制不能写成指定方向的精确硬零陷。代码与解析式分别求零点，独立回归见 `test_codes_spatial_round3.py`。DSB 归一化另覆盖导向尺度 $10^{-200}$ 与 $10^{200}$，先缩放再计算范数，避免有限输入平方溢出后悄悄返回零权重。

建议使用 §5.3 的双麦解析协方差，增加增益误差与相位误差，按加载强度画出失真和降噪的关系。工业策略需要无效协方差检测、上一组权重保留、权重平滑和输出限幅。求解成功不能替代目标保持检验；过度加载时接近固定波束是可以解释的设计结果。

### 21. LCMV、Frost 与 GSC

对应 §5.5～5.6。LCMV 直接求满足多个线性约束的最小功率解；Frost 在时域抽头空间投影更新以保持约束；GSC 用固定支路和阻塞后的自适应支路实现同类约束结构。本书教学包只给 LCMV 闭式解与 GSC 阻塞基线；持续自适应 GSC 可另读 BTK2.0，不能把其子带实现直接登记为正文时域 Frost。

最小实验先检查约束矩阵独立性和阻塞残差，再将目标方向偏移少量，测量目标泄漏到参考支路后被抵消的程度。工业中冻结条件、步长、滤波长度、双讲/活动控制和状态复位必须与滤波器一起审查。约束保持不等于目标真实方向仍在约束集合中。

BTK2.0 是 Kumatani、McDonough 等作者的空间信号处理工具箱。固定 `btk20` 源码先读 `btk20_src/unit_test/test_online_beamforming.py::online_beamforming`，再读 `btk20_src/lib/pybeamformer.py::SubbandGSCLMSBeamformer` 与 `SubbandGSCRLSBeamformer`；C++ 对应入口包括 `btk20_src/beamformer/beamformer.{h,cc}::SubbandGSCRLS`。这些类包含跨帧自适应状态，不只是构造一个阻塞矩阵。[固定 Python 更新器](https://github.com/kkumatani/distant_speech_recognition/blob/feff19ec8bcb770f6530fe280dc3ccafc2f5984a/btk20_src/lib/pybeamformer.py)。

演示输入是逐通道音频、阵列位置、带时间标记的目标方向和分析/合成滤波器组系数。示例声速常数为 `343740.0`，与毫米制坐标配套；本书米制坐标必须转换。滤波器组的子带数、抽取率、原型滤波器长度决定时频处理与等待，不能直接沿用本书 Hann STFT 的帧数和延迟。

构建文件默认目标 Python 2.7，要求 SWIG 3、GSL、NumPy 和 libsndfile，CUDA 9 为可选；部分 Python 算法还使用 SciPy 或 pygsl。旧环境说明不能当作现代 Python 或目标设备的相容性保证。滤波器系数演示通过 `pickle` 载入，只应使用自己生成或可信来源的文件，不能为方便运行而加载不可信二进制对象。[固定构建条件](https://github.com/kkumatani/distant_speech_recognition/blob/feff19ec8bcb770f6530fe280dc3ccafc2f5984a/btk20_src/CMakeLists.txt)、[固定在线演示](https://github.com/kkumatani/distant_speech_recognition/blob/feff19ec8bcb770f6530fe280dc3ccafc2f5984a/btk20_src/unit_test/test_online_beamforming.py)。

建议先对同一混合录音保留固定支路输出，再开启 LMS/RLS 自适应，分别检查目标参考的增益、干扰残差与权重范数；第二组只将控制方向偏移 2°，比较持续更新与明确冻结区间。合成输入须保存各源分量、共同延迟和增益，指标对齐后计算。上述 BTK 构建和数值实验未在本书执行，不能称为已经完成产品验收。

### 22. 最坏情形稳健波束与 WNG 约束

对应 §5.4.1。稳健设计需要先定义导向误差集，例如有界范数误差，再约束误差集中目标响应的最低值；WNG 约束则限制白噪声放大。二者与“给矩阵加一个经验小数”有不同的建模目的，不能把所有稳健方法都称为加载 MVDR。

建议固定同一真实误差序列比较 DSB、加载 MVDR 和带明示误差集的优化解，报告失配集外的失败。参数来自阵列标定和环境变化，不应只按训练集效果选择。本书此项保留原理与 §5.4.1 的原始引用；未确认特定作者代码时，不把通用凸优化器单独列作完整波束实现。

### 23. Acoustic Rake 与时域多路径约束

这是 §5.4～5.5、§2.4 的联合扩展。pyroomacoustics `beamforming.py` 中 `rake_mvdr_filters`、`rake_distortionless_filters`、`rake_perceptual_filters` 将早期反射路径纳入滤波器设计。它们需要源位置、反射模型、滤波器长度与允许延迟；优化目标可选择保持响应或容许特定早期时间结构。[作者原论文](https://arxiv.org/abs/1407.5514)。

建议先用单反射的已知 RIR 比较只保留直达与加入反射，再扰动反射延迟和增益。利用已知反射不意味着实际房间中的所有混响都应保留。原始 `TimeDomainAcousticRakeReceiver` 专库采用 CC BY-NC-SA 4.0，和当前 MIT 的 pyroomacoustics 不能混同；本文不将该原始实验库作为商业可自由使用源码。[原实验许可](https://github.com/LCAV/TimeDomainAcousticRakeReceiver)。

### 24. 球谐变换与理论径向补偿

对应 §5.8。sfa 的 `process.py::spatFT` 把球面采样投影到球谐系数，`gen.py::radial_filter` 设计径向补偿，`sph.py` 定义球谐与模态相关计算。应先区分实球谐/复球谐、系数排列、归一化、方位角和余纬角，再设计开放球或刚性球的径向滤波。[官方 API](https://appliedacousticschalmers.github.io/sound_field_analysis-py/reference.html)。

最小实验以已知平面波系数合成球面麦信号再反演，检查每阶误差。增加阶数时同时测白噪声输出；低频高阶模态的逆滤波会放大自噪声。有限麦数、球面采样误差与空间混叠限制可恢复阶数，不能仅依据期望指向性无限增加阶数。

### 25. 实测球阵编码与软限制径向滤波

对应 §3.4、§5.8。Politis 库的 `arraySHTfiltersTheory_softLim.m`、`arraySHTfiltersTheory_regLS.m` 和 `arraySHTfiltersMeas_regLS.m` 分别提供软限制、理论正则最小二乘和实测响应拟合。`evaluateSHTfilters.m`、`sphArrayNoise.m` 用于检查编码误差和噪声代价。先读 `TEST_SCRIPTS.m` 的调用参数，再对照各函数输入单位。[作者库及原始文献列表](https://github.com/polarch/Spherical-Array-Processing)。

建议把校准方向分成设计集与留出方向，在相同频带上测球谐重建误差、WNG 和目标方向响应。实测拟合可以包含实际外壳与麦差异，但不能据同一校准数据上的低误差声称新方向泛化。MATLAB 中还需加入作者的 Array-Response-Simulator 和 Spherical-Harmonic-Transform，不能只下载一个库后承诺全部示例可运行。

### 26. 球谐固定波束、MVDR/LCMV 与子空间定位

对应 §5.8。Politis 库提供 `beamWeightsDolphChebyshev2Spherical.m`、`sphMVDR.m`、`sphLCMV.m`、`sphMUSIC.m`、`sphESPRIT.m`。固定波束主要取决于期望方向图和有效阶数；自适应方法仍需球谐域统计量，定位方法仍需源数和信号/噪声模型。

建议在同一个已编码声场上分别改变方向、阶数和模态噪声，比较旋转前后方向图形状以及自适应权值响应。编码误差是共同前提，不能将球谐坐标变换理解成已经消除物理麦克风误差。新增径向滤波时也必须更新噪声协方差，而不是照搬阵元域的单位白噪声假设。

### 27. C/C++ 球阵处理与空间功率图

对应 §5.8 的实现。SAF 的阅读顺序是 `framework/modules/saf_sh/saf_sh.h`，然后 `examples/src/array2sh/array2sh.c`、`beamformer/beamformer.c` 与 `powermap/powermap.c`。前者给数学接口，三个例子分别连接阵元到 SH、SH 到虚拟麦、统计量到方向图。[官方模块与构建说明](https://github.com/leomccormack/Spatial_Audio_Framework)。

建议先在无设备情况下以确定的平面波测试系数方向，再建立目标 CPU 构建，测每帧最慢时间和内存分配。CBLAS/LAPACK 后端、FFT 库、SIMD 和编译浮点选项都影响结果；不能由 C 实现推断它必然满足某个实时截止期。启用 GPLv2 可选追踪模块会改变许可义务，应在构建清单中明确。

### 28. DAMAS 非负声源功率反卷积

工业扩展：Acoular `fbeamform.py::BeamformerDamas` 从常规波束图和点扩散矩阵估计非负源功率。输入是互谱矩阵、声源网格和传播模型，输出是声源分布；不是语音时域波形。读码时先追踪 `PointSpreadFunction`，再追踪迭代次数、松弛系数和终止方式。[官方接口与论文定位](https://acoular.org/acoular/api_ref/generated/acoular.fbeamform.html)。

最小实验放两个独立单极子，比较常规功率图与反卷积图的峰和区域积分，然后改变网格间距与导向模型。更尖的图像不保证功率正确。工业中校准、CSM 对角处理、流场传播、积分区域和缓存版本必须固定；用不匹配点扩散函数迭代更多次可能放大误差。

### 29. CLEAN-SC 相干分量剥离

Acoular `BeamformerCleansc` 根据当前强峰及其相干结构逐次减去贡献。它适合研究旁瓣与多源解释，但减去量、停止条件和残余谱都影响弱源是否还能保留。不要把迭代获得的细峰直接解释成新物理声源。

建议在相同 CSM 上分别改变减去比例与迭代上限，报告残差、弱源恢复和区域积分。再把两个源设成强相干，检查分量解释是否混合。官方 [风洞实测例](https://acoular.org/acoular/v26.07/auto_examples/wind_tunnel_examples/example_airfoil_in_open_jet_freq_domain_methods.html) 展示了几何、校准和多种方法共用数据的流程，但该页面的运行时不是本机或目标硬件的性能数字。

### 30. 协方差拟合 CMF、SODIX 与移动源成像

Acoular 的 `BeamformerCMF` 直接拟合 CSM 的空间模型；`BeamformerSODIX` 进一步处理源指向性建模。`tbeamform.py` 的时域处理用于沿给定轨迹的传播延时补偿。静止源、旋转源与任意移动源不是只改网格坐标即可互换的模型。[官方项目功能说明](https://github.com/acoular/acoular)。

建议先在静止单源上核对功率尺度，再给已知移动轨迹检查传播时延；最后故意使用错误轨迹，测量失焦与谱展宽。CMF 需要记录非负/稀疏约束、单位缩放、求解器和残差；SODIX 需要足够观测区分源强与指向性。未取得运动或流场真值时，应把不确定性写入结果解释。

## 追踪、关联与生命周期

### 31. KF、EKF 与 UKF

对应 §9.2。本书 `tracking.py` 提供角度—角速度 KF；FilterPy 的 `kalman/` 和 Stone Soup 的 `predictor/kalman.py`、`updater/kalman.py` 提供不同非线性扩展。EKF 在当前状态线性化观测，UKF 传播一组 sigma 点，二者都需要与观测空间一致的均值和残差函数。

本书平面方位以 $+y$ 为零点、向 $+x$ 为正，所以位置观测函数是 `atan2(x-x0, y-y0)`，例如相对位置 `(1,2)` 对应 26.565°。常见数学极角接口按 `atan2(y,x)` 使用时必须转换。教学追踪接口的角度、状态、协方差和重采样权重要求有限实数；复数、布尔、字符串和对象数组不通过先强制转浮点来接受。无效标量观测或预测参数在修改状态、消耗随机数之前拒绝。

建议给 $179°$ 与 $-179°$ 的相邻观测、长期缺测和非等间隔时间戳；在笛卡尔状态到方位角观测时，检查零距离附近的雅可比与方位角环绕。过程噪声必须随时间模型离散化。预测方差增大不等于应该不断增加“目标存在概率”，运动不确定性和存在性是不同状态。

### 32. SIR 粒子滤波与重采样

对应 §9.3。先读本书 `CircularParticleFilter` 和 `systematic_resample`，再读 ODAS `src/system/particle2particle.c` 与 `src/module/mod_sst.c` 中的持续状态管理。权重应由一致的似然和杂波模型形成，极小概率在对数域计算；重采样只是分配粒子数量，不创造观测信息。

建议固定种子比较有无重采样的有效粒子数，再用双峰对称后验检查圆周均值是否有定义。单个均值可能落在两个真实峰之间，因此多峰后验应保留峰或混合表示。工业中记录粒子数、退化阈值、扩散噪声、出生位置范围和静默保持时间，不把所有声学峰强制解释成一个人。

### 33. GNN/PDA 与 JPDA

对应 §9.3。Stone Soup `dataassociator/neighbour.py` 给硬关联组件，`dataassociator/probability.py::JPDA` 根据满足一对一约束的联合事件计算边缘关联概率；`docs/tutorials/08_JPDATutorial.py` 展示如何连接预测器、假设器与更新器。[官方 JPDA 教程源码](https://github.com/dstl/Stone-Soup/blob/main/docs/tutorials/08_JPDATutorial.py)。

建议沿用正文两轨三观测例，比较最近邻与 JPDA 在杂波靠近某条轨迹时的分配，再设置两人交叉。门控先减少不可行配对，检测概率和杂波密度再决定事件权重；几何最近不等于语音身份正确。角度观测的杂波密度单位是每度或每弧度，不能直接填位置空间每平方米的数值。

### 34. GM-PHD 与高斯混合缩减

对应 §9.3。Stone Soup `updater/pointprocess.py::PHDUpdater`、`hypothesiser/gaussianmixture.py`、`mixturereducer/gaussianmixture.py::GaussianMixtureReducer` 分别执行强度更新、生成观测假设和合并/剪枝。阅读必须包含出生强度，不能只读更新公式。[官方 GM-PHD 教程](https://stonesoup.readthedocs.io/en/v1.9.1/auto_tutorials/filters/GMPHDTutorial.html)。

建议先复算正文两网格例，再以无观测帧检查总强度如何因存活与漏检改变；最后新增一个源，检查出生模型能否覆盖其方向。高斯权重和表示期望目标数，不要求归一为 1；剪枝会改变这个和，应报告删去质量。PHD 不保留完整身份，给每个高斯分量贴标签也不等于实现了 GLMB。

### 35. MHT、CPHD、LMB/GLMB 与检测前追踪

对应 §9.3 的进阶分类。MHT 保留多条关联历史，CPHD 还传播目标数分布，LMB/GLMB 显式保留标签，检测前追踪输入未经过硬阈值的弱信号证据。它们解决的缺口不同，不能作为同一个“高级追踪器”接口随意替换。

Stone Soup 固定版本实际包含滑窗多帧分配形式的 MHT 参考。阅读顺序是 `docs/examples/dataassociation/mht_example.py` → `stonesoup/hypothesiser/mfa.py::MFAHypothesiser` → `stonesoup/dataassociator/mfa/__init__.py::MFADataAssociator` → 同目录 `_step.py`。假设器给每个分量延续观测索引历史与权重，关联器优化多帧分配后执行 N-scan 剪枝；这比仅在基类注释中提到方法名称多了实际计算。[固定 MHT 示例](https://github.com/dstl/Stone-Soup/blob/8d1edeb07ef8505ed065cbef435cfb5e517d9bdc/docs/examples/dataassociation/mht_example.py)。

该 MIT 参考实现需要额外安装 OR-Tools；`_step.py` 的最大迭代数为 10，相对原始—对偶间隙门限为 0.02，代码将代价差除以当前最佳原始代价 `bestPrimalCost`，并非直接比较未归一化的差。示例使用长度为 3 的滑窗、二维位置/速度状态与方位/距离观测，预置三个目标且没有出生或消亡。它说明如何推迟关联决定，不是已经完成的声学多说话人系统，也不保证所有输入下得到精确全局最优。声学适配仍需角度环绕、每弧度杂波密度、静默期漏检和轨迹管理。[固定分配求解器](https://github.com/dstl/Stone-Soup/blob/8d1edeb07ef8505ed065cbef435cfb5e517d9bdc/stonesoup/dataassociator/mfa/_step.py)。

最小实验建议从同目录 `MFA_example.py` 的两目标交叉场景开始，保持观测集合不变，比较滑窗 1 与 3 的分支数量、延迟决策、身份交换和计算量，再插入连续缺测。出生和长静默应作为额外实验单独报告，不能由固定目标示例推断已经支持。这里仅完成源码检查，未运行 OR-Tools 实验。

CPHD、LMB/GLMB 另有 [Ba Tuong Vo 的作者 MATLAB 工具包](https://ba-tuong.vo-au.com/codes.html)，页面列出相应滤波器及 OSPA/OSPA²，并明确面向 academic/research 使用。该可变 ZIP 页面没有提供本书所需的固定提交和完整通用再分发许可，故保留受限来源索引，不复制或改称许可完备的开源集合。作者提供代码与本书已经取得可分发实现是两个判断。

检测前追踪仍保留原理索引。对其他家族，也不能从 Stone Soup 的基类文字推断其实现了 CPHD，或从 PHD 分量上的临时标签推断其实现了 GLMB。两人交叉、出生、长静默和强反射应分别评价身份切换、漏检、虚警、人数和成本；输入定位峰不符合点目标观测模型时，还要调整似然与杂波模型。

### 36. OSPA、身份连续性与波束控制接口

对应 §9.3～9.4。OSPA 在同一坐标和截断距离下同时惩罚位置与目标数误差；身份连续性应另外评价。将 DOA 转换为波束控制时还需预测到播放或采集消费时刻，记录 `measurement_time`、`publish_time`、`track_id`、协方差与有效期。

建议先复算正文 30°/70° 对单真值 32° 的 OSPA，再交换两条轨迹 ID：位置指标可能完全不变，身份连续性已经出错。控制器按最大角速度和时间间隔限速，在过期观测、长静默或身份不确定时降低更新可信度。轨迹 ID 不等于永久说话人身份；单凭声学位置不能维持跨房间身份。

### 37. TDOA 非线性最小二乘与可观测性

对应 §4.4、§4.7.1。TDOA 几何定位将每条观测与候选位置产生的距离差比较，按观测协方差加权，使用雅可比求局部位置更新。输入除了 TDOA，还需要明确参考麦和共享参考引入的误差相关性；输出应包含残差、局部条件性和解是否位于搜索域。

最小检查先用正文三麦近场几何复算距离差，再用有限差分独立检查雅可比。把源逐步移远，距离方向的敏感度会减小；算法收敛到一个距离不意味着距离已可观测。工业中至少保留多个初值或粗网格初始化，识别镜像多解与边界解。本文未将一般优化器索引成完整声学定位实现，具体模型与本书 §4.4 的原始论文一致。

### 38. 随机源、确定源与非相关源 CRB

对应 §4.8。doatools `performance/crb.py` 的 `crb_sto_farfield_1d`、`crb_det_farfield_1d`、`crb_stouc_farfield_1d` 对应不同观测统计模型。CRB 是在参数模型、噪声和正则条件下的估计方差下界，不是从输入录音直接产生方向的算法，也不是混响房间的实测误差预测器。[作者源码](https://github.com/morriswmz/doatools.py/blob/9469db201e0418aef6b97583ef54b6fec2769502/doatools/performance/crb.py)。

建议同一几何上比较源功率、快拍数和角度分离的变化，检查方差单位是弧度平方还是角度平方，再与同模型重复仿真比较。明显低于所选 CRB 的结果可能来自口径不符、偏差估计器、真值泄漏或方差/均方误差混写，应先查实验模型，不能直接宣称突破统计下界。

### 39. 差分协同阵与虚拟协方差重建

对应 §3.3。doatools `estimation/coarray.py::CoarrayACMBuilder1D` 将物理阵列协方差映射到差分滞后，再构造增广协方差用于虚拟阵列处理。原始输入仍是有限个物理通道；差分滞后具有重复计数和统计相关性，不是新增的独立麦克风通道。[作者源码](https://github.com/morriswmz/doatools.py/blob/9469db201e0418aef6b97583ef54b6fec2769502/doatools/estimation/coarray.py)。

建议对正文稀疏阵坐标手工枚举差分集合，核对重复滞后的平均与缺孔处理，然后分别使用独立源和完全相干源。可形成较长连续滞后区间不等于在任意声场都能识别相同数量的源。频带变化后相位字典需重算；宽带混响与互耦不能通过集合计数消除。

### 40. IMM 多运动模型交互

对应 §9.2 的运动建模扩展。FilterPy `kalman/IMM.py::IMMEstimator` 维护多个滤波器、模式概率与 Markov 转移概率，先按模式概率混合状态和协方差，再执行各模型的预测与更新。所有被混合状态应有相同维度和物理含义，不能直接把角度状态与笛卡尔位置相加。[作者源码](https://github.com/rlabbe/filterpy/blob/3b51149ebcff0401ff1e10bf08ffca7b6bbc4a33/filterpy/kalman/IMM.py)。

建议给匀速、转弯和静默三段轨迹，比较单模型与多模型的跟随延迟和协方差。必须核对转移矩阵方向及归一化，而不是凭行列名称猜测；长期缺测时不能把模式概率改变解释成观测证据。源突然停止说话是活动状态变化，并不必然代表其运动模式改变。

### 41. icoDOA、Cross3D 与神经方向追踪

对应 §4.7、§9.3。icoDOA 的 `1sourceTracking_icoCNN.py` 是训练与测试入口；`acousticTrackingModels.py` 定义 `IcoTempCNN`、`Cross3D` 等模型，`acousticTrackingModules.py` 提供输出映射，`acousticTrackingDataset.py` 生成移动源场景。二十面体方向网格与普通平面卷积网格有不同邻接结构，方向分辨率及输出变换必须与训练时一致。[作者仓库](https://github.com/DavidDiazGuerra/icoDOA)。

最小实验应先验证方向图旋转与标签同步，再用未见房间、阵列误差和静默输入测角度误差与失效状态。代码依赖 gpuRIR、icoCNN 和声学数据；只读到模型类不表示完整训练可复现。作者 README 明确提醒主脚本未调用的辅助功能可能未经测试，因此同一文件中的 `SELDnet` 类也不能直接等同于已经验证的 SELD 系统。AGPL 源码与模型、LibriSpeech/LOCATA 数据分别核对。

### 42. FN-SSL 与 IPDnet：先估计直达声相位差

对应 §4.7。Wang、Yang 与 Li 的 [IPDnet 原论文](https://doi.org/10.1109/TASLP.2024.3507560)把多个声源的直达声麦间相位差作为学习目标，再用阵列几何转换为方向；它并非直接输出一个房间无关的绝对位置。作者仓库是 [Audio-WestlakeU/FN-SSL](https://github.com/Audio-WestlakeU/FN-SSL)，本次锁定 `76fcb281be92caf068c712dfb015e354f437260f`，不是其他领域的同名 IPDNet。

读码顺序为 `IPDnet/Opt.py` 的阵列与数据设置、`Simu.py`/`Dataset.py` 的信号与标签、`FixedAarryIPDnet.py` 和 `VariableArrayIPDnet.py` 的网络、`Module.py` 的空间处理，最后读 `runIPDnetOn.py`/`runIPDnetOff.py` 的训练和评价。固定阵列文件名确实拼作 `FixedAarryIPDnet.py`。根 README 的通用 `main.py` 命令不能直接当成 IPDnet 子目录入口。[实际目录及命令](https://github.com/Audio-WestlakeU/FN-SSL/tree/76fcb281be92caf068c712dfb015e354f437260f/IPDnet)。窄带分支沿时间处理各频点，全带分支沿频率学习相位差与频率的关系；“在线”配置还要检查具体上下文和分块延迟，不能由脚本名字推出零前视。

最小实验先合成已知两麦时延的单个平面波，检查预测相位差与几何字典匹配的符号；再加入第二个源、换麦间距和调换通道。若交换麦顺序却不更新字典，错误方向来自接口失配，不能据此衡量网络泛化。记录源数、频带、阵列适配方式、输出轨数和活动阈值，且把未见房间与未见阵列分开评价。根 README 写 MIT，但当前未核实完整许可及其致谢的 Cross3D/icoCNN 派生仿真部分声明，因此暂不自动下载；模型和 LibriSpeech、Noise92、LOCATA、RealMAN 数据另核授权。

### 43. ACCDOA：活动与方向共用一个向量

对应 §4.7。[Shimada 等 ICASSP 2021 原文](https://doi.org/10.1109/ICASSP39728.2021.9413609)定义每类每帧的活动耦合方向输出。训练标签在事件不活动时为零向量，活动时为单位笛卡尔方向；预测向量的模用于活动判定，归一化后的方向用于定位。预测模不是自动校准过的存在概率，也不是声源距离。

最小手算令二维示意标签为 30° 的单位向量 `(0.866,0.5)`，预测为 `(0.433,0.25)`，模为 0.5，但方向仍为 30°。活动判定是否通过取决于明确的阈值及 `>`/`>=` 约定；不能因角度正确就忽略漏检。真实三维任务应保留 z 分量。再设置同类别同时出现在 30° 与 90°：单向量不能无损表示两个方向，这正是 multi-ACCDOA 要解决的输出容量问题，而非增加网络参数即可消除的问题。

可对照 [DCASE2022 官方基线模型](https://github.com/sharathadavanne/seld-dcase2022/blob/c8adb1d3a5a35de2d6c7b6d19e01ad455eef3986/seldnet_model.py)的单/多轨输出和特征生成阅读，但它是后续届次实现，不等于已经复现 2021 原论文。模型结构、训练集合和评价指标都应随所用基线记录。

### 44. Multi-ACCDOA、ADPIT 与 DCASE2022 基线

对应 §4.7。[Multi-ACCDOA 原论文](https://arxiv.org/abs/2110.07124)用多条活动耦合方向轨道表示同类重叠源；辅助重复排列不变训练（ADPIT）构造多种合法标签排列，使固定轨号不必永久对应同一实例。训练时的轨道置换不自动提供跨帧身份，因此这种输出不能直接替代第 9 章的持久轨迹 ID。

[2022 官方任务页](https://dcase.community/challenge2022/task-sound-event-localization-and-detection-evaluated-in-real-spatial-sound-scenes)链接到 `sharathadavanne/seld-dcase2022`，固定提交为 `c8adb1d3a5a35de2d6c7b6d19e01ad455eef3986`。从 `parameters.py` 读任务与 `quick_test`，再读 `cls_feature_class.py` 的 FOA/MIC 特征、`cls_data_generator.py` 的批数据及标签、`seldnet_model.py` 的网络与 ADPIT、`train_seldnet.py` 的输出合并和 `SELD_evaluation_metrics.py` 的评分。MIC 的 SALSA-lite 特征与 FOA 特征不同，四路音频并不意味着可互换。[基线说明](https://github.com/sharathadavanne/seld-dcase2022)。

最小实验构造一类零、一、二、三个活动方向，检查标签维度、允许的置换和解码后的源数，再让两个预测轨输出近似同方向以检查重复合并阈值。STARSS22 开发与评价划分不能混用，初始化默认的 `quick_test=True` 仅用于短流程检查，不代表完整训练或论文分数。官方页面展示的源代码未建立明确再分发许可，故只锁定索引，不自动复制；数据许可另核。

### 45. DCASE2025 立体声 SELD：方位、距离与画内/画外

对应 §4.7 的届次变化。[2025 官方任务页](https://dcase.community/challenge2025/task-stereo-sound-event-localization-and-detection-in-regular-video-content)与[作者基线](https://github.com/partha2409/DCASE2025_seld_baseline)规定的是双声道常规视频场景，不是 2022 的四通道三维定位。锁定提交 `42a48b6456b73be35ad0e1a9ffeb6ceef83ae0bd`。`model.py::SELDModel` 的音频输出每轨每类含 x、y 和距离；视听配置另加画内/画外输出。这里第三个量是距离，不能按旧的 xyz 方向向量求三维模。[实际输出层](https://github.com/partha2409/DCASE2025_seld_baseline/blob/42a48b6456b73be35ad0e1a9ffeb6ceef83ae0bd/model.py)。

读码从 `parameters.py` 的任务配置到 `model.py` 的激活函数，再到 `loss.py`、`inference.py` 和 `metrics.py`；需同时确认音视频时间对齐与米制距离标签。模型包含双向 GRU，整段执行不是已验证的因果实时推理。最小实验用相同方位、不同距离的标签检查输出解释，再构造声音活动但画面中不出现的事件，确认画外不等于静默。距离误差、角度误差、事件检出与画内/画外分类应分别评价。

本次未建立该仓库明确的再分发许可，只保留官方入口。数据生成器、预训练视觉模型和音视频数据另有来源及条款，不能由 baseline 的公共可见性推断全部可随书发布。

### 46. GEV：最大信噪比方向与未定尺度

对应 §5.9。GEV 最大化目标和噪声输出功率之比，输入两组厄米协方差，输出最大广义特征值对应的波束向量。pb_bss `extraction/beamformer.py::get_gev_vector` 调用广义特征值求解；ESPnet `enh/layers/beamformer.py::get_gev_vector` 提供张量实现。pb_bss 函数中的文献定位为 Warsitz 与 Haeb-Umbach 2007 年论文，归一化讨论见其 §III.A。[pb_bss 官方源码](https://github.com/fgnt/pb_bss/blob/master/pb_bss/extraction/beamformer.py)。pb_bss 为 MIT，ESPnet 为 Apache-2.0，使用本书总清单锁定的提交。

最小手算取目标协方差 `diag(4,1)`、噪声协方差 `diag(1,2)`，两个广义特征值为 4 和 0.5，最优方向为第一通道；该向量乘 10 后信噪比不变，输出幅度却增大 10 倍。因此广义特征向量不是直接满足参考通道幅度的增强输出。加入奇异噪声协方差测试加载、数值失败及状态输出；任意改用一般 `eig` 不能证明非厄米输入合理。

### 47. BAN：GEV 的盲解析尺度归一化

对应 §5.9。pb_bss `blind_analytic_normalization` 使用噪声协方差计算尺度，分子为 `sqrt(wᴴ Rn² w)`，分母为 `|wᴴ Rn w|`，再乘原向量。它不需要已知目标导向，但也因此不能保证对任意真实目标传递函数满足单位响应。函数显式将零分母位置的增益置零，工程中仍应把这种退化与正常增强区分。

本书手算取 `Rn=I`、任意非零向量 `w`，该比例为 `1/||w||`，得到单位范数权重；单位范数并不等于 `wᴴa=1`。最小实验比较同一 GEV 向量乘正数、负数及复相位后的 BAN 输出：功率尺度约束与复相位相干性是不同问题。连续帧/频点特征向量还要处理任意相位，ESPnet 的 `gev_phase_correction` 是可阅读入口，但不能把它的平滑方向约定当成通用目标相位恢复。[ESPnet 波束源码](https://github.com/espnet/espnet/blob/master/espnet2/enh/layers/beamformer.py)。

### 48. RTF：相对参考通道的传递向量

对应 §5.4、§5.9。单目标秩一模型下，目标 SCM 的主特征向量与目标传递向量同方向；有色噪声可结合噪声 SCM 的广义特征分解或幂迭代估计。pb_bss `get_pca_vector` 与 ESPnet `get_rtf` 分别提供这些入口；不能在包含多个同等功率目标的全秩 SCM 上无条件称主特征向量为“目标 RTF”。

ESPnet `get_rtf` 明确说明函数自身没有执行参考通道归一化。若估计传递向量为 `(2,1+j)`，以通道 0 为参考应得到 `(1,0.5+0.5j)`；若参考系数接近零，直接除法会放大误差。读码需继续追到消费该向量的 MVDR 层，检查参考向量、复共轭和是否重复归一化。最小实验固定目标，改变参考通道、加入近零参考响应及近简并特征值，记录相位、范数、目标响应和帧间跳变，而不是只检查输出张量尺寸。

### 49. MWF、SDW-MWF 与秩一化简

对应 §5.7、§5.9。以参考通道目标声像为估计对象，一般 SDW-MWF 解由 `(Rs+mu Rn)w=Rs e_ref` 得到。ESPnet `get_sdw_mwf_vector` 明确提供该解，并有 `approx_low_rank_psd_speech` 选项；`mu=1` 对应普通 MWF。其 `get_mwf_vector` 的参数虽名为 `psd_n`，docstring 指的是观测协方差，不能因为变量名含 n 就传入纯噪声 SCM。

pb_bss `get_wmwf_vector` 实际计算 `Phi/(mu+trace(Phi))` 的参考列，其中 `Phi=Rn^-1 Rs`，这是秩一目标条件下的化简。独立手算反例取 `Rn=I`、`Rs=diag(2,1)`、`mu=1`、参考 0：一般解第一项为 `2/3`，该迹化简第一项为 `2/(1+3)=1/2`。改成 `Rs=diag(2,0)` 后两者均为 `2/3`。这定位了模型边界，不说明秩一方法本身错误；本机尚未运行外部包数值测试。

最小工程实验应同时比较全秩与秩一 SCM，并记录目标失真、残留噪声和所用参考。令 `mu→0` 在非奇异全秩情形接近参考通道直通，并不普遍等于 MVDR；只有相应秩一模型和极限解释下才能作该联系。源码文档里的简短等价描述应随假设阅读，不能直接复制进算法对比表。

### 50. MCRA：由局部最小值控制噪声更新

对应 §5.7。最小值控制递归平均（MCRA）不是直接把短时谱的最小值当作当前噪声，而是由平滑谱相对于局部最小值的关系估计语音存在可能性，再控制噪声谱递推。最小值窗口、谱平滑、语音存在平滑和噪声更新是不同的时间尺度。起点文献为 [Cohen 与 Berdugo，IEEE Signal Processing Letters 2002 原文](https://israelcohen.com/wp-content/uploads/2018/05/SPL_Jan2002.pdf)，§II 给递推噪声估计，§III 给由最小值控制的语音存在估计。

建议生成噪声平台先为 1、随后跳到 4 的功率序列，并插入持续窄带语音。检查噪声上升后最小值窗口多久更新，以及语音保持时噪声估计是否被污染。这个合成序列只测试跟踪逻辑，不代表语音感知质量。工业中记录初始化静音假设、窗口长度换算到毫秒的值、突发噪声响应及增益恢复时间。

### 51. IMCRA：两阶段平滑与最小值搜索

对应 §5.7。改进 MCRA（IMCRA）进一步在语音存在判断控制下执行两阶段平滑与最小值搜索，目标是降低强语音对噪声估计的污染。[Cohen 2003 原论文](https://webee.technion.ac.il/Sites/People/IsraelCohen/Publications/SAP_Sep2003.pdf) §II～III 给出递推，§IV 讨论与 OM-LSA 的联合评价。不能把任何“语音概率控制的递归平均”统称 IMCRA。

最小实验使用与 MCRA 完全相同的输入和窗长，对照两阶段的平滑谱、最小值、语音判定及最终噪声估计，并同时测试低 SNR 弱语音和噪声突然上升。若只报告更低的噪声输出，却没有目标衰减，就不能证明改进。作者软件页明确介绍 MATLAB OM-LSA/IMCRA 软件，但本次页面中未取得可核验的下载包、版本与许可，故本条保留原论文和软件入口，不以非作者 GitHub 同名代码补齐下载状态。

### 52. OM-LSA：对数谱幅度增益与语音存在不确定性

对应 §5.7。OM-LSA 把语音存在概率用于组合语音存在时的 LSA 增益和语音缺席时的增益下限；噪声估计器可用 IMCRA，但两者是不同模块。其低增益区域的处理影响残余噪声的连续性，不能仅以硬门限谱减替代。[Cohen 与 Berdugo 2001 论文入口及作者实现说明](https://israelcohen.com/software/)；[作者合著的 OM-LSA 推导 §III](https://webee.technion.ac.il/Sites/People/IsraelCohen/Publications/IWAENC2006_Habets.pdf)。

局部手算若存在时增益为 0.8、缺席时下限为 0.1、存在概率为 0.5，几何组合得到 `sqrt(0.8×0.1)≈0.283`，不是算术平均 0.45。这个例子只说明增益组合，不包含 LSA 增益内部的先验 SNR 与指数积分计算。完整实验还需保存 decision-directed 先验 SNR、后验 SNR、语音存在概率、增益下限和重叠相加条件；缺失这些状态不能算完成 OM-LSA 复现。当前没有已核许可的作者下载包随书提供。

### 53. WebRTC 分位数噪声估计与语音概率控制

对应 §5.7 的工业对照。已下载 WebRTC `modules/audio_processing/ns/quantile_noise_estimator.cc::QuantileNoiseEstimator::Estimate` 跟踪分位数，`noise_estimator.cc::NoiseEstimator::PreUpdate/PostUpdate` 管理初始化和噪声更新，`speech_probability_estimator.cc::SpeechProbabilityEstimator::Update` 估计语音概率。它们提供可检查的产品代码路径，但不是 MCRA、IMCRA 或 OM-LSA 的逐式实现。[WebRTC 官方 NS 目录](https://webrtc.googlesource.com/src/+/refs/heads/main/modules/audio_processing/ns/)。代码许可及 PATENTS 使用总清单记录的固定提交逐项核对。

读码须继续到 `noise_suppressor.cc` 的整帧流程，区分分析、增益施加和通道状态，再检查 `suppression_params.h` 的模式参数。建议复用噪声阶跃、持续弱语音与静音启动输入，同时检查全零谱及采样率分支。单独调用分位数模块不等于运行整个 WebRTC Audio Processing；没有 APM 集成、采样帧检查和目标设备计时，不报告实时端到端收益。

## 可复算的工程配置与验收记录

以下是独立实验设计，不是已测性能表。每次只改变一个条件，保存随机种子、输入、真值、源代码提交和实际依赖版本。

| 实验 | 固定输入建议 | 改变项 | 必须检查的输出 |
|---|---|---|---|
| 双源子空间 | 8 元 ULA；每点按正文声速算波长；固定协方差真值 | 独立/相干源、16/256 快拍、源数错一位 | DOA、协方差特征值、选峰数量、失败标志 |
| 宽带索引 | 非连续频点 `[10,20,30]`；明确 FFT 长度与频率映射 | 改变频点排列、参考频点、删去一个频点 | 频点与 SCM 配对、跨频映射、方向是否一致 |
| 波束失配 | 同一目标/干扰/噪声协方差；冻结目标真值 | 通道相位偏差、加载、约束角偏差 | 真实目标响应、WNG、输出噪声和峰值 |
| 球阵编码 | 已知平面波与固定球面采样点 | 阶数、低频、径向限制、校准留出方向 | SH 系数误差、WNG、旋转方向一致性 |
| 关联与出生 | 两轨交叉，另有反射峰和静默区间 | 门控、杂波强度、出生强度、剪枝阈值 | 人数、ID 切换、OSPA、删去的强度质量 |
| 工业成像 | 同一校准数据、网格、FFT 及传播模型 | 对角去除、PSF、迭代、正则、运动模型 | 功率尺度、区域积分、残差和模型失配 |

每个实验至少做一个解析或可控的理想输入，再加入噪声与失配。对大规模随机实验，报告失败次数和分布，不仅报告成功样本均值。程序抛错、输出 NaN、找不到指定源数、残差过大和无法区分多个方向应分别记录。

## 固定版本的源码审查疑点

下列是对已下载 pyroomacoustics 0.10.0 源码的静态审查与最小下标推演，尚未运行该外部包的完整声学反例。它们用于限定采用范围，不据此推断原始论文有误。

| ID | 精确位置与证据 | 最小可复核配置 | 状态及下一步 |
|---|---|---|---|
| SP-CSSM-01 | `pyroomacoustics/doa/cssm.py:108` 删除 `freq_bins` 的 `invalid` 项，但第 156 行仍按 `C_hat[j]` 使用原协方差序列 | 若三频点为 `[10,20,30]`、协方差为 `[R10,R20,R30]`，首频点被删除后 `j=0` 的导向对应 20，协方差仍对应 10 | 已发现；静态不一致。需构造首频点峰数不足的音频/谱输入，核对输出及同步筛选修复 |
| SP-WAVES-01 | `pyroomacoustics/doa/waves.py:104` 同样删除频点，第 148 行仍使用 `C_hat[j]` | 同上，把首频点设为无有效峰，后两频点保持不同协方差 | 已发现；静态不一致。需独立声学回归，不能只验证维度相同 |
| SP-TOPS-01 | `pyroomacoustics/doa/tops.py:117` 构造真实 FFT bin 差，第 136 行却使用 `Phi[k]`，且构造但未使用剔除参考频点后的 `freq` | `[10,20,30]`、参考 bin 20 时，跨频差应取 −10 和 +10；下标 0、1 对应的却是 −20 和 −19 | 已发现；静态频率映射疑点。需对照原论文实现参考频点排除与真实 bin 取值后比较全谱 |
| SP-FRIDA-01 | `pyroomacoustics/doa/frida.py` 文档的 `n_rot` 默认值说明与构造函数值不同 | 直接比较文档默认与 `__init__` 默认，不需声学输入 | 已验证静态差异；本文要求显式记录 `n_rot`，不从旧 docstring 复制默认值 |

这些源码问题不要求修改下载的上游仓库来“让演示通过”。如需修复，应创建保留原始许可证与提交号的独立补丁、列出改动原因，并采用理想多频输入及真实录音双重回归。未完成修复和回归前，不把这三个固定版本实现列作设备可直接采用的已验证定位器。

本机实际运行检查记录：2026-09-22，在仓库 `.venv` 中分别尝试导入固定版本 doatools 的 `RootMUSIC1D` 和 FilterPy 的 `KalmanFilter`，两次均在导入阶段因缺少 `scipy` 失败，未进入数值计算。计划的独立输入分别为六元半波距阵、30° 单源协方差，以及先验均值 2、方差 4、观测 3、观测方差 1 的标量更新；后者手算后验为 2.8 与 0.8。这些预期值不是已取得的外部运行结果。当前也未安装 pyroomacoustics，CSSM/WAVES/TOPS 只完成上表所列静态核查。

另一个版本相容性检查是 doatools `estimation/music.py:153` 使用 `np.complex_`，需在其依赖支持的 NumPy 版本中运行或准备独立兼容补丁；本文未修改外部工作目录或共享依赖来绕过这些条件。

## 收录边界

稀疏贝叶斯定位、原子范数、完整最坏情形稳健波束以及本文未逐项收录的神经定位方法，仍需要逐论文确认代码与模型；不能用同名 GitHub 搜索结果替代作者来源。本文已分别记录实际取得的外部实现、官方实现存在但再分发许可未建立的索引、只核实到原理的算法和版本疑点。深度波束与 WPD 的完整增强链、分离及神经噪声抑制见本目录其他研究文档；同一函数在多个算法条目中被调用，不等于取得了多套独立工程系统。
