> ⚠️ 本篇是教程正文第 8 章（正文共 11 章，另有附录 A/B），可独立阅读，前后篇见下方导航。
>
> 🏠 首页导读：[`00_overview.md`](./00_overview.md) ｜ 上一篇：[07_wpe-dereverberation.md](./07_wpe-dereverberation.md) ｜ 下一篇：[09_source-tracking.md](./09_source-tracking.md)

## 8. 语音分离

第 7 章讨论单个声源的混响拖尾，本章讨论多个说话人同时发声时的混合与分离。主要方法包括免训练盲源分离（Blind Source Separation，BSS）、导引源分离（Guided Source Separation，GSS）和深度分离；§8.2再说明这些方法在前端系统中的顺序。

### 8.1 多说话人语音分离：鸡尾酒会问题

去混响处理单个声源的时间拖尾，语音分离处理多个声源在同一观测中的叠加。本节依次介绍免训练 BSS、GSS 和深度分离。时频掩码既可用于分离，也可用于 §5.4 的空间协方差估计。

**卷积混合模型**。第 $m$ 个麦克风接收各说话人经过相应房间冲激响应后的叠加，再加上噪声：

$$x_m(t) = \sum_{n=1}^{N} \sum_\tau h_{mn}(\tau)\, s_n(t-\tau) + v_m(t)\text{。}\tag{8-1}$$

$h_{mn}$ 是第 $n$ 个声源到第 $m$ 个麦克风的房间冲激响应，$s_n$ 是第 $n$ 个源信号。源数 $N$ 与麦克风数 $M$ 有三种关系：

- $N=M$ 是确定混合；
- $N<M$ 是超定混合，可保留冗余通道、先降维或使用超定扩展；
- $N>M$ 是欠定混合，不能只靠一个方阵解混器恢复全部源，还要利用时频稀疏性、源谱模型或满秩空间协方差等结构。

标准方阵辅助函数型独立向量分析（Auxiliary-function-based Independent Vector Analysis，AuxIVA）和独立低秩矩阵分析（Independent Low-Rank Matrix Analysis，ILRMA）通常针对 $N=M$。多通道非负矩阵分解（Multichannel Nonnegative Matrix Factorization，MNMF）能表示 $N>M$，但可靠性仍取决于源数、数据长度、初始化和模型匹配。

单麦 $M=1$ 时没有跨通道方向信息。此时多说话人分离只能依赖源信号先验或监督学习，不能使用后文的多通道方向聚类作为证据。

短时傅里叶变换（Short-Time Fourier Transform，STFT）常把卷积近似为每个频点上的瞬时混合：

$$X_m(f,k) \approx \sum_{n=1}^{N} A_{mn}(f)\,S_n(f,k) + V_m(f,k)\text{。}\tag{8-2}$$

其中 $k$ 是帧号，$A_{mn}(f)$ 是 $h_{mn}$ 在频率 $f$ 处的近似频响。写成向量形式为 $\mathbf X(f,k)=\mathbf A(f)\mathbf S(f,k)+\mathbf V(f,k)$：$\mathbf X\in\mathbb C^{M\times1}$、$\mathbf A\in\mathbb C^{M\times N}$、$\mathbf S\in\mathbb C^{N\times1}$、$\mathbf V\in\mathbb C^{M\times1}$。

这个窄带近似要求分析窗相对通道冲激响应的有效长度足够长，并且源与通道在窗内变化不快。$T_{60}$ 不能单独给出一个“超过多少就失效”的通用阈值；窗长、早晚期能量分布、帧移和允许误差都会影响近似质量。

远场平面波条件下，$\mathbf A$ 的每一列对应一个导向矢量（steering vector），表示给定方向在各麦克风上的相位差：

$$\mathbf{a}(f,\theta) = [1,\, e^{+\mathrm{j}2\pi f d\sin\theta/c},\, e^{+\mathrm{j}2\pi f 2d\sin\theta/c},\, \ldots,\, e^{+\mathrm{j}2\pi f(M-1)d\sin\theta/c}]^\top$$

例如麦间距 $d=4$ cm、声速 $c=343$ m/s、目标方向 $\theta=30°$。按第 5 章约定，麦 1 在左端，正角声源位于 $+x$ 一侧，$\tau_{21}=t_2-t_1=-d\sin\theta/c\approx-58.3\ \mu$s。在 $f=2$ kHz 时，相位差的绝对值为 $2\pi f|\tau_{21}|\approx0.733$ rad，即约 42°；以麦 1 为参考的 4 麦导向矢量约为 $[1,e^{+\mathrm j42°},e^{+\mathrm j84°},e^{+\mathrm j126°}]^\top$。

频率升到 4 kHz 时，相位差绝对值增至约 84°。2 kHz 的波长约 17 cm，4 cm 间距小于半波长；8 kHz 的波长约 4.3 cm，此间距会产生空间混叠和栅瓣。

最小方差无失真响应（Minimum Variance Distortionless Response，MVDR）和广义特征值（Generalized Eigenvalue，GEV）波束形成直接使用空间协方差矩阵（Spatial Covariance Matrix，SCM）；MNMF 也以源相关的 SCM 描述传播。观测 SCM 可写成 $\mathbf{R}(f,k)=E[\mathbf{X}\mathbf{X}^H]\approx\mathbf{A}\mathbf{R}_s\mathbf{A}^H+\mathbf{R}_v$。

复角中心高斯混合模型（Complex Angular Central Gaussian Mixture Model，cACGMM）先把每个观测归一化到复单位球面，拟合只描述方向分布的形状矩阵。它不把未归一化的观测功率 SCM 直接当作概率密度。

GSS 和部分深度方法先由 cACGMM 或网络估计时频掩码，再用原始 $\mathbf X\mathbf X^H$ 加权计算目标与干扰 SCM，做法见 §5.4。

波束形成一次输出通常针对一个目标，但可以用不同导向或掩码构造多组权重，分别输出多个人的信号。因此“波束形成”和“语音分离”不是互斥方法：GSS 正是先估掩码，再为每个目标计算一组波束权重。下面按解混矩阵、空间聚类和监督学习三条路线比较。

| 路线 | 代表 | 思想 | 局限 |
|---|---|---|---|
| 经典盲源分离（BSS） | 独立成分分析（Independent Component Analysis，ICA）/独立向量分析（Independent Vector Analysis，IVA） | 假设各说话人统计独立，找一组解混矩阵让输出互相独立；IVA 用“同一声源跨频点相关”解决逐频点排列混乱（置换问题） | 需要源数不多于麦克风数、对混响敏感 |
| 空间混合模型（免训练） | **cACGMM → GSS** | 见下文展开 | 依赖较长的上下文，计算量随通道数、源数、时频点数和迭代次数增加；GPU 并行化的适用边界见本节“GSS 的源数与计算规模” |
| 深度学习分离 | 时频掩码估计、深度聚类（deep clustering：把每个时频点映射成嵌入向量再按说话人聚类）、置换不变训练（Permutation Invariant Training，PIT：对输出与参考说话人的所有对应关系分别计算损失，取最小者反向传播）、Conv-TasNet（全卷积时域分离网络） | 直接学习从混合信号到各声源的映射 | 依赖训练数据，跨场景泛化需验证 |

置换不变训练（Permutation Invariant Training，PIT）在所有输出—参考排列中选择总损失最小者。对 $N$ 个输出，其定义为

$$\mathcal L_{\mathrm{PIT}}=\min_{\pi\in\mathcal S_N}\sum_{n=1}^{N}\mathcal L\!\left(\hat s_n,s_{\pi(n)}\right)\text{，}\tag{8-3}$$

其中 $\mathcal S_N$ 是 $N$ 个对象的排列集合。

**PIT 的 2×2 手算例**。分离器输出为 $\hat s_1,\hat s_2$，参考为 $s_1,s_2$。假设某种失真损失的矩阵为

$$
\mathbf L=
\begin{bmatrix}
L(\hat s_1,s_1)&L(\hat s_1,s_2)\\
L(\hat s_2,s_1)&L(\hat s_2,s_2)
\end{bmatrix}
=
\begin{bmatrix}0.4&3.1\\2.8&0.6\end{bmatrix}.
$$

保持对应关系时总损失是 $0.4+0.6=1.0$；交换参考时是 $3.1+2.8=5.9$，所以本次更新使用前一种对应关系。

这个例子说明 PIT 解决的是“输出槽位没有固定说话人标签”，并不自动解决未知人数或跨长录音的身份保持。逐帧独立选择排列会让同一输出在相邻帧换人；整句级置换不变训练（utterance-level PIT，uPIT）在整句上只选一次排列，可减少这种帧间交换。

只有当总目标确实分解成式(8-3)这种“每个输出—参考对各贡献一个代价，再把所选代价相加”的形式时，才能把 $N\times N$ 成对损失矩阵交给线性指派算法求最小总代价。若损失含有跨输出耦合项、集合级正则项或共同解码约束，就不能直接用匈牙利算法代替全排列目标。[PIT 正式论文](https://doi.org/10.1109/ICASSP.2017.7952154)、[uPIT 正式论文](https://doi.org/10.1109/TASLP.2017.2648230)。

**四类经典 BSS 方法**。以下按模型假设、计算代价和失效条件比较 AuxIVA、ILRMA、MNMF 与 TRINICON。

**AuxIVA（辅助函数型独立向量分析，Auxiliary-function IVA，Ono 2011）**。它假设不同说话人统计独立，并联合建模同一说话人在各频点的分量。频域 ICA 在各频点独立求解，可能得到不同的输出排列；IVA 利用跨频点包络相关性缓解该置换问题。AuxIVA 使用辅助函数进行闭式更新，不需要设置梯度步长，目标函数在理论条件下单调下降。标准实现仍是整段迭代，在线版本需要递推改写；强混响、说话人包络高度相关和短语音都会削弱统计独立性或估计稳定性。它常作为免训练分离基线。

**ILRMA（独立低秩矩阵分析，Independent Low-rank Matrix Analysis）**。ILRMA 在独立性假设之外，再用非负矩阵分解（Nonnegative Matrix Factorization，NMF）表示每个源的低秩功率谱，并联合优化解混矩阵与谱模型。它增加了基数、迭代次数等超参数，计算量和初始化敏感性也高于 AuxIVA。复杂音乐、重叠笑声和强混响可能不满足低秩或独立性假设，需用目标数据验证。

**MNMF（多通道 NMF，Multichannel NMF）**。MNMF 同样使用低秩源谱模型，但以满秩空间协方差描述传播，不要求方阵解混矩阵可逆，因此可以表示欠定混合。谱基、激活和空间协方差需要联合做最大似然估计，参数较多、迭代较慢且对初值敏感。实践中可用 AuxIVA 或 ILRMA 结果初始化。其计算量通常更适合离线分析或伪标签生成。

**TRINICON（三重 N 卷积混合盲分离，TRIple-N ICA for CONvolutive mixtures）**直接优化时域解混滤波器，用二阶去相关和高阶独立性处理卷积混合，因此不依赖逐频点瞬时混合近似。代价是滤波器与统计窗口较长、优化量大，声源或房间变化时还要重新适应。滤波器长度决定能表示多长的卷积逆，但不等于端到端延迟：因果 FIR 可逐样本输出，块实现的额外等待由分块、缓冲、FFT 和是否使用未来样本决定。

AuxIVA 适合确定混合的免训练基线；ILRMA 进一步利用源谱低秩结构；MNMF 用满秩空间协方差表示欠定混合；TRINICON 直接处理时域卷积混合。选择时需同时检查源数与麦克风数、混响强度、数据长度和实时预算。

**BSS 的排列与尺度歧义**。若 $\mathbf W(f)$ 能把混合信号分开，交换 $\mathbf W$ 的两行只会交换两个输出；把某一行乘非零复数，也不会破坏输出间的统计独立性。因此 BSS 只能在排列和复尺度因子意义下恢复声源。IVA、ILRMA 用跨频点结构缓解频点间排列错乱，但仍要固定输出尺度。方阵情形可令 $\hat{\mathbf A}(f)=\mathbf W^{-1}(f)$，选参考麦克风 $r$，再做

$$\tilde Y_n(f,k)=\hat A_{rn}(f)Y_n(f,k).$$

这一步称为回投影（projection back）：它把第 $n$ 个分离输出恢复到参考通道上的幅度和相位尺度。回投影不能修复分离错误，也不能保证不同时间块中的说话人编号一致；连续录音仍要做跨块排列对齐。

**BSS 方法的规模例子**。以下计算用于说明参数量和迭代规模；性能数字必须以各论文的测试集和评测脚本为准。

AuxIVA 的规模例子：2 源 2 麦、8 kHz、3 s 语音，STFT 取 256 点窗和 64 点帧移。名义帧率为 $8000/64=125$ 帧/s，3 s 约为 375 帧；不补零、从首样本开始取完整窗时，精确帧数是 $\lfloor(24000-256)/64\rfloor+1=372$。非负频点数为 129，每个频点维护一个 $2\times2$ 解混矩阵，共 516 个复数参数。

辅助函数迭代不需要学习率。帧数还取决于是否居中、边界补零和丢弃尾部不完整窗。算法见 [AuxIVA 原始论文](https://doi.org/10.1109/ASPAA.2011.6082320)（Ono, WASPAA 2011, pp. 189–192；公开实现见 [piva](https://github.com/fakufaku/piva)）。

ILRMA 在同一 2 源 2 麦例子中，还要为每个源维护 NMF 基矩阵和激活矩阵。若每源取 2 个基，则矩阵尺寸分别为 $129\times2$ 和 $2\times375$，并与解混矩阵交替更新。基数、迭代次数和初值都会影响计算量与结果。语音与音乐是否满足同样的低秩假设，需要分别验证。见 [ILRMA 原始论文](https://doi.org/10.1109/TASLP.2016.2577880)（Kitamura et al., IEEE/ACM TASLP 24(9), 2016, pp. 1626–1641）。

MNMF 可以表示 3 个说话人、2 个麦克风的欠定混合。每个频点、每个源对应一个 $2\times2$ Hermitian 空间协方差；129 个频点和 3 个源共需 387 个协方差矩阵，还要联合估计谱基和激活。计算量和局部最优问题使初始化十分重要，AuxIVA 或 ILRMA 结果可作为初值。见 [MNMF 原始论文](https://doi.org/10.1109/TASL.2013.2239990)（Sawada et al., IEEE TASL 21(5), 2013, pp. 971–982）。

TRINICON 的长度换算可以这样看：8 kHz 下，0.5 s 对应 4000 个采样，但这不表示解混滤波器必须取 4000 点，也不表示算法延迟就是 0.5 s。实际长度是在可表示的卷积范围、估计方差、计算量和块实现之间折中。例如 1024 点与 2048 点分别覆盖 128 ms 和 256 ms 的滤波器记忆；系统仍要单独报告块大小、帧移、前视和输入输出缓冲，才能得到算法延迟。见 [TRINICON 原始论文](https://ieeexplore.ieee.org/document/1414341)（Buchner et al., 2005）。

**GSS（导引源分离，guided source separation）**在 CHiME-5/6 的真实晚宴多通道任务中得到广泛使用（Boeddeker et al., CHiME-5 2018；CHiME-8 任务设置已经变化，引用时应按当届规则）：
1. **cACGMM 空间聚类**：记 WPE 输出、尚未做方向归一化的多通道 STFT 为 $\mathbf Y(f,k)$。先剔除 $\|\mathbf Y(f,k)\|_2\le\varepsilon_E$ 的低能量点，再对保留点严格归一化为 $\mathbf z=\mathbf Y/\|\mathbf Y\|_2$，主要保留通道间的复数比例。第 $n$ 个分量的密度满足

   $$p(\mathbf z\mid\mathbf B_n)\propto \frac{1}{\det(\mathbf B_n)}\left(\mathbf z^H\mathbf B_n^{-1}\mathbf z\right)^{-M}\text{，}\qquad \|\mathbf z\|_2=1\text{。}\tag{8-4}$$

   其中 $\mathbf B_n$ 是正定的形状矩阵。cAC 密度对 $\mathbf B_n$ 的正比例缩放不变，所以每次更新后必须固定尺度，例如令 $\operatorname{tr}(\mathbf B_n)=M$。式(8-4)建模的是归一化方向 $\mathbf z$ 的密度，不是 $\mathbf Y$ 的功率 SCM。混合模型的后验概率 $\gamma_n(f,k)$ 用作软时频掩码。“免训练”只表示聚类参数在当前录音上用期望最大化（Expectation-Maximization，EM）估计，不需要预先训练分离网络；说话人活动标注仍可能来自监督模型。
2. **导引（Guided）**：各频点独立聚类会产生簇标签置换。GSS 使用**说话人活动时间标注**（diarization 输出）限制每个说话人可出现的帧，从而统一不同频点的簇标签；
3. **级联**：WPE 去混响 → GSS 得到掩码 → 掩码估计目标/噪声协方差 → MVDR/GEV 波束形成。MVDR 是最小方差无失真响应，GEV 是广义特征值波束形成，见 §5.4。GSS 论文在 CHiME-5 的特定阵列和打分集上报告了识别结果；引用数字时必须同时给出原文表号、阵列配置、打分集和基线。[GSS 原始论文](https://www.isca-archive.org/chime_2018/boeddecker18_chime.pdf "citation")

**紧凑实现顺序**。下面的伪代码强调依赖关系，不替代具体实现中的复数矩阵更新式。

1. 从 WPE 输出 $\mathbf Y$ 中剔除 $\|\mathbf Y\|_2\le\varepsilon_E$ 的点；对其余点计算严格单位向量 $\mathbf z=\mathbf Y/\|\mathbf Y\|_2$。活动标注只初始化或约束混合权重（以及允许的分量掩码）；$\mathbf B_n$ 另用单位阵或空间统计量初始化，不能把活动标注当作形状矩阵。
2. E 步按式(8-4)计算后验 $\gamma_n(f,k)$，并把活动标注为“不发言”的分量权重置零，再对允许的分量归一化。
3. M 步按 cACG 的固定点更新形状矩阵：每个 $\mathbf z\mathbf z^H$ 外积除以当前二次型 $\mathbf z^H\mathbf B_n^{-1}\mathbf z$，再用 $\gamma_n$ 加权求和；每次更新后令 $\operatorname{tr}(\mathbf B_n)=M$（或采用等价的固定行列式约定），再加入小的正则项。有效权重接近零或条件数过大时，重置该分量，不能继续求逆。
4. 重复 E/M 步，直到对数似然相对变化小于预设阈值，或达到最大迭代次数；报告实际迭代数、停止条件和重置次数。
5. 将后验作为掩码，但用同一 WPE 阶段、未做方向归一化的 $\mathbf Y\mathbf Y^H$ 按式(8-5)估计目标/干扰 SCM，再计算每个目标的 MVDR 或 GEV 权重。这里不能把单位向量 $\mathbf z$ 的外积误当成带功率信息的 SCM。

活动标注只限制允许的簇，不会自动修复错误分割；短活动段、空分量或近奇异形状矩阵仍需要正则化和失败检测。

**GSS 的源数与计算规模**：输入通常是一段包含重叠语音的会议录音。cACGMM 的分量数通常由日志或系统配置预先给定；它不会仅凭式(8-4)可靠推断未知说话人数。

$M\ge2$ 才有跨通道方向信息。当 $N\le M$ 时，掩码波束形成通常还有空间自由度；当 $N>M$ 时，仍可为每个已知目标估计掩码和 SCM，但分离依赖时频占优、活动标注与数据量，不能获得方阵解混的可逆性保证。

WPE 先减弱晚期混响；cACGMM 在目标段附近的上下文上做 EM 聚类，为每个时频点输出说话人后验掩码；目标和非目标掩码分别用于估计 SCM，再计算 MVDR 或 GEV 权重。

上下文长度和 EM 迭代次数会共同影响估计方差、延迟与计算量。GPU 并行化的收益必须绑定硬件、数据、代码版本和计时范围，不能作为跨实现的通用加速倍数。

以 16 kHz 采样、512 点窗、128 点帧移为例，每秒有 125 帧，30 s 上下文含 3750 帧和 257 个非负频点，约 96 万个时频点。这个换算只给出待处理样本数，不能保证 EM 在固定次数内收敛。实现应监测对数似然相对变化，并设置最大迭代次数；空分量、近奇异 $\mathbf B_n$ 和低能量时频点还需要正则化或重置。

掩码换成目标协方差时，用

$$\mathbf R_{\mathrm{tar}}(f)=\frac{\sum_k\gamma(f,k)\mathbf Y(f,k)\mathbf Y^H(f,k)}{\sum_k\gamma(f,k)+\varepsilon}\text{。}\tag{8-5}$$

噪声协方差可用相应的非目标掩码估计。软权重的有效样本数不是“$\gamma>0.8$ 的帧数”，而可用 Kish 公式近似：

$$N_{\mathrm{eff}}(f)=\frac{\left(\sum_k\gamma(f,k)\right)^2}{\sum_k\gamma^2(f,k)}.$$

例如 900 帧权重全为 1 时 $N_{\mathrm{eff}}=900$；若权重集中在少数帧，有效样本数会明显更小。

估计 $M\times M$ 协方差不仅需要样本数大于矩阵维度，还要覆盖足够多的独立空间状态；高度相关的连续帧不能当作同等数量的独立快照。工程上应检查 $N_{\mathrm{eff}}$、矩阵条件数，并在必要时做对角加载。

说话人活动标注用于限制各簇可出现的时间区间，从而减少跨频点置换。见 [cACGMM 原始工作](https://doi.org/10.1109/EUSIPCO.2016.7760429)（Ito et al., EUSIPCO 2016, pp. 1153–1157）与 [GSS 原始论文](https://www.isca-archive.org/chime_2018/boeddecker18_chime.pdf)。

**掩码到波束权重的小例子**。以下实数向量只是 WPE 输出复数 STFT 的简化手算。两个时频快照取 $\mathbf Y_1=[1,1]^\top$、$\mathbf Y_2=[1,-1]^\top$，目标掩码为 $\gamma=[0.9,0.1]$。因为权重和为 1，目标协方差为

$$
\mathbf R_{\mathrm{tar}}=0.9\mathbf Y_1\mathbf Y_1^H+0.1\mathbf Y_2\mathbf Y_2^H
=\begin{bmatrix}1&0.8\\0.8&1\end{bmatrix}.
$$

若这个二分类例子中用 $1-\gamma=[0.1,0.9]$ 估计干扰协方差，则 $\mathbf R_{\mathrm{int}}=\begin{bmatrix}1&-0.8\\-0.8&1\end{bmatrix}$。正的非对角项表示目标快照中两通道同相成分占优，负的非对角项表示干扰快照中反相成分占优。

随后可从 $\mathbf R_{\mathrm{tar}}$ 估计目标相对传递函数，并与 $\mathbf R_{\mathrm{int}}$ 一起计算 MVDR。GEV 则直接求矩阵对 $(\mathbf R_{\mathrm{tar}},\mathbf R_{\mathrm{int}})$ 的主广义特征向量，并需另做尺度归一化。

真实的“目标 + 多个说话人 + 环境噪声”并非二分类，$1-\gamma$ 会把所有非目标成分合在一起，不能解释成纯噪声掩码。实际实现还要检查有效样本数、条件数和对角加载。[掩码波束形成实例](https://doi.org/10.1109/ICASSP.2016.7471664 "citation")。

定位结果可以给分离提供方向先验；分离网络输出的掩码又能用于估计 MVDR 的目标与干扰协方差。

常用的尺度不变信号失真比（scale-invariant signal-to-distortion ratio，SI-SDR）先把估计信号 $\hat s$ 投影到参考 $s$ 上：$s_{\mathrm{tar}}=\langle\hat s,s\rangle s/\|s\|^2$，再计算 $10\log_{10}(\|s_{\mathrm{tar}}\|^2/\|\hat s-s_{\mathrm{tar}}\|^2)$。SI-SDR 提升量（SI-SDRi）是输出 SI-SDR 减去混合输入的 SI-SDR，两项必须使用同一预处理和实现。

**分离指标的最小报告模板**：

- 写明目标信号定义（无混响干声、带混响参考或参考麦克风图像）、参考通道、采样率与截断长度；
- 说明是否去均值，以及固定传播时延、算法时延和整体增益怎样对齐；
- 给出活动区间、静音段处理、输出—参考排列规则和输入混合基线；
- 固定指标实现及版本，并对多次随机实验报告种子、次数和离散程度。

若其中任一项不同，SI-SDRi、SI-SNRi 或 WER 数字不能直接横向排名。

**数据集口径表**：去混响与分离的论文使用不同数据、通道和指标。下表只说明各数据集的用途，不给出脱离原文表格的性能范围。

| 数据集 | 考什么 | 房间/混响口径 | 通道/采样口径 | 怎么用 |
|---|---|---|---|---|
| REVERB 挑战赛 | 去混响（含识别端 WER） | 仿真三档混响时间约 0.25 / 0.5 / 0.7 s；另有真实房间录音（MC-WSJ-AV 基） | 8 通道 / 2 通道 / 单通道三种设置；基线语音来自 WSJCAM0 | 比较 WPE 时必须注明通道数、识别器、测试子集和 WER 的绝对/相对口径 |
| CHiME-5 | 真实晚宴多说话人（分离 + 识别端 WER） | 真实家庭：厨房、餐厅、客厅多房间自然混响加生活噪声 | 每场 6 台 Kinect、每台 4 个同步麦克风；每位参与者另戴双耳麦 | 6 台设备不能缩写成“6 麦阵列”；双耳录音用于辅助转写与时间对齐，不是干净评分参考。见 [CHiME-5 官方数据说明](https://www.chimechallenge.org/challenges/chime5/overview) |
| CHiME-6 | 重新对齐的 CHiME-5 录音；多阵列远场识别 | 与 CHiME-5 相同的真实家庭录音，但版本与划分必须写清 | 共 32 个物理通道：6 台四麦阵列的 24 通道，加 4 套双耳设备的 8 通道 | Track 2 评测中双耳通道只允许同步，不能用于分割、增强或识别；GSS 还要注明实际选用阵列、日志输入、打分集和表号。见 [CHiME-6 Track 2 官方数据规则](https://www.chimechallenge.org/challenges/chime6/track2_data) |
| WHAMR | 噪声 + 混响 + 重叠三合一 | 合成房间冲激响应；$T_{60}$ 低、中、高三档分别从 0.1～0.3 s、0.2～0.6 s、0.4～1.0 s 均匀采样，这三档表示混响程度，不表示房间大小 | 8 kHz / 16 kHz；min 版（截短对齐）/ max 版（全长）两版；底子是 WSJ0-2mix 加 WHAM 噪声 | 报告时注明 min/max、采样率、源数、目标是无混响还是带混响语音，以及 SI-SDRi 实现。见 [WHAMR 正式论文 §2 与表 1](https://doi.org/10.1109/ICASSP40776.2020.9053327) |
| WSJ0-2mix / 3mix | 干净重叠两/三人 | 无噪声、无混响（近场干声直接叠加） | 8 kHz / 16 kHz；min 版（按最短截断）/ max 版（全长补零）；30 小时训练 / 10 小时验证 / 5 小时测试 | 引用时注明采样率、min/max、源数、静态/动态混合和指标实现 |
| LibriMix | 干净或含噪重叠（WSJ0 的开源替代） | clean 版无污染；noisy 版叠加 WHAM 环境噪声 | 8 kHz / 16 kHz；train-100/train-360；min/max；另行发布的 SparseLibriMix 是稀疏重叠测试集 | 用于跨语料验证时，同时注明训练集、测试集、噪声版本和是否使用 SparseLibriMix。见 [LibriMix 论文](https://arxiv.org/abs/2005.11262 "citation")与[官方生成仓库](https://github.com/JorisCos/LibriMix "citation") |
| LibriCSS | 连续会议式重叠与长录音识别 | 扬声器在真实会议室回放 LibriSpeech 语音；共 10 个约 1 h 会话，每个分成 6 个约 10 min 小会话，条件为 0S、0L、10%、20%、30%、40% 重叠 | 7 通道圆阵录音；0S 与 0L 都无重叠，但句间静音长度不同 | 检查具体重叠条件、分块、跨块排列、输出流数和识别评测的说话人分配规则。见 [LibriCSS 正式论文](https://doi.org/10.1109/ICASSP40776.2020.9053426) |

REVERB 主要用于去混响与识别，CHiME-5/6 用于真实重叠语音的分离与识别，WHAMR 同时包含噪声、混响和重叠，WSJ0-2mix/3mix 是干净混合基准，LibriMix 还可检查跨数据集泛化，LibriCSS 则把问题扩展到连续会议。CHiME-6 重新对齐了 CHiME-5 的部分录音，引用结果时不能把两个版本混为同一数据版本。[CHiME-6 官方说明](https://www.chimechallenge.org/challenges/chime6 "citation")。不同数据集的数字不能直接比较。

**连续语音分离（Continuous Speech Separation，CSS）**不能把整句分离器机械地逐块调用。长录音中的活动人数会变化，同一输出槽位还可能在相邻块交换说话人。

实现需明确四件事：

1. 每块使用多少左、右上下文；
2. 重叠块怎样加权拼接；
3. 跨块排列依据声纹、波形相关性还是联合优化；
4. 输出流数固定还是随活动人数变化。

连续识别的指标也要写清分配规则。连接最小排列词错误率（Concatenated Minimum-Permutation WER，cpWER）、最优参考组合词错误率（Optimal Reference Combination WER，ORC-WER）和说话人归属词错误率（Speaker-Attributed WER，sa-WER）处理说话人对应的方式不同，不能只写“WER”。

CSS 解决长录音的分块与输出一致性，不保证身份永久不交换；需要稳定身份时还要结合分割或说话人跟踪。

VarArray 的正式论文发表于 ICASSP 2022（pp. 6027–6031，DOI [10.1109/ICASSP43922.2022.9746876](https://doi.org/10.1109/ICASSP43922.2022.9746876)），通过通道置换/数量不变的聚合结构适配不同麦克风数量和论文评测中的多种同步阵列几何。“阵列几何无关”不表示对任意未见孔径、异步时钟、通道失配或设备频响都有保证；这些条件仍需在目标设备上验证。

**时频稀疏性与掩码**：在某些语音混合中，同一时频点往往由一个声源占优，文献将这种近似称为 W-不重叠正交性（W-disjoint orthogonality，WDO）。掩码方法可以据此估计每个时频点属于各声源的权重。说话人在同一时频区域强烈重叠时，WDO 近似变差。波束形成的区别不是“对整个频带统一加权”：宽带 STFT 波束形成通常在每个频点使用一组复权重，利用通道间的幅度和相位关系做空间选择。

**深度分离的几类结构**。下表只比较建模方式。原论文的指标、数据生成和评测代码并不完全相同，不在表中按单个 dB 数排名。

| 结构 | 代表 | 主要结构 | 结果复现要求 | 优点与限制 |
|---|---|---|---|---|
| 时域卷积 | Conv-TasNet（Luo & Mesgarani, 2019） | 可学习的短窗编码器代替 STFT，时序卷积网络（Temporal Convolutional Network，TCN）的空洞卷积建模长上下文，再对编码系数施加掩码 | 核对数据版本、采样率、因果设置和 SI-SNRi 实现 | 端到端延迟取决于编码窗、感受野和缓冲。见 [Conv-TasNet 正式论文](https://doi.org/10.1109/TASLP.2019.2915167) |
| 双路径循环 | DPRNN（Luo et al., 2020） | 长序列分块，块内 RNN 建模局部，块间 RNN 建模较长时间关系 | 核对模型规模、块长、静态/动态混合和 SI-SNRi 实现 | RNN 顺序计算会限制并行度，块长影响内存与上下文。见 [DPRNN 正式论文](https://doi.org/10.1109/ICASSP40776.2020.9054266) |
| 双路径注意力 | SepFormer（Subakan et al., 2021） | 在双路径结构中用 Transformer 替换 RNN | 核对源数、数据生成、模型版本和 SI-SNRi 实现 | 注意力内存随序列长度增长，长句通常需分块。见 [SepFormer 正式论文](https://doi.org/10.1109/ICASSP39728.2021.9413901) |
| 状态空间 | S4M（Chen et al., 2023） | 用多尺度编码和结构化状态空间块替换分离骨干中的循环或注意力时序建模 | 原文使用固定两源的 WSJ0-2Mix、LibriMix 与 LRS2-Mix；核对 8/16 kHz 数据版本、源数和官方配置 | 状态递推提供长程建模的另一种实现，但原论文未验证严格因果的前瞻量与端到端延迟。见 [Interspeech 2023 正式论文](https://doi.org/10.21437/Interspeech.2023-696) 与[官方复现仓库](https://github.com/JusperLee/S4M) |
| 时频网格 | TF-GridNet（Wang et al., 2023） | 在时频域分别建模时间、频率与跨帧关系，并结合子带长短期记忆网络（Long Short-Term Memory，LSTM）和注意力 | 会议版是单通道分离，原文主表使用 SI-SDRi；多通道混响扩展是另一模型与实验设置 | 整句注意力需要改造后才能流式运行。见[单通道会议正式论文](https://doi.org/10.1109/ICASSP49357.2023.10094992)与[多通道混响期刊扩展](https://doi.org/10.1109/TASLP.2023.3304482 "citation") |

比较分离数字时要对齐采样率、`min`/`max` 混合方式、静态或动态混合、源数、评测脚本和指标名称。分离文献中的 SI-SNR 常采用与 SI-SDR 相同的正交投影公式，二者在相同去均值与数值实现下可以相等；也有代码把均值处理、截断长度或稳定项写得不同。不能笼统地说两者固定相差若干 dB。引用 SI-SNRi 或 SI-SDRi 时，应给出实现或至少说明是否去均值，并始终用同一指标计算输入基线和输出结果。

**深度分离的结构选择**。卷积、循环网络、注意力和状态空间模型都可以作为分离骨干。S4M 改变的是编码器—分离器—解码器中的时序建模骨干：多尺度表示送入结构化状态空间块，而不是仅更换损失函数。

论文实验使用已知的两个输出源和监督干净参考。核实至 2026-09-22，官方仓库提供 `S4M.py`、`s4.py` 等模型结构文件，但没有完整的数据准备、训练入口和配套 checkpoint，不能把可读的骨干代码写成已经可复现整套训练。未知源数、无监督会议录音或任意麦克风阵列不属于这项结果的直接覆盖范围。论文把该结构称为流式分离的潜在方案，但没有报告严格因果实现的前瞻样本数、首帧延迟或端到端延迟。[S4M 固定版本](https://github.com/JusperLee/S4M/tree/4990b3fe9d7391e59d652c5a7d2803d1354fd0ae)

部署前仍需检查双向上下文与缓冲。选择时不要只看单个榜单分数，还要检查模型是否使用整句上下文、块边界如何处理、状态能否在流式推理中延续，以及计算量和内存怎样随语句长度变化。生成式方法还要检查采样步数和语音幻觉。

近期模型若没有在相同数据、采样率、混合方式和评测脚本下比较，本章不按论文年份或模型名称给出性能排序。

目标说话人提取（target speaker extraction，TSE）不要求输出所有声源，而是用注册语音、视觉或方向等与目标说话人绑定的条件提取一条音轨。声纹网络可把注册语音变成定长嵌入，再将其广播到混合语音的各帧；嵌入可在编码器入口拼接，也可用逐层缩放与偏置调制（FiLM）或交叉注意力注入。

嵌入维度、注册时长和余弦阈值由具体模型和训练数据决定，不存在跨模型通用的“同人区间”。部署时要验证注册信道失配、非目标泄漏、目标误抑制和首帧延迟。

§6.1.6 的 pAEC 也使用声纹条件，但训练目标还包含回声消除。见 [SpeakerBeam/TSE 系列](https://arxiv.org/abs/1705.10687 "citation")；声纹嵌入的端侧存储、加密和删除要求见 §6.1.14。

文本查询通用声音分离是另一项任务：查询词描述“警报声”“狗叫”等声音类别，不保证对应某个可注册的说话人。对比语言—音频预训练（Contrastive Language-Audio Pretraining，CLAP）可提供文本与音频嵌入，但 CLAP 本身不是分离器；实际系统还需要用查询嵌入控制掩码或波形生成模块。

见 [CLAP](https://arxiv.org/abs/2206.04769 "citation")与使用语言查询的 AudioSep。AudioSep 已正式发表于 *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 33, pp. 458–471，DOI [10.1109/TASLP.2024.3520017](https://doi.org/10.1109/TASLP.2024.3520017)；[官方仓库](https://github.com/audio-agi/audiosep)提供模型与推理入口。

TSE 的身份泄漏、注册信道和声纹保护测试，不能由文本查询分离的类别级结果替代。

**可执行分离基线与代码边界。** [`codes/array_tutorial/separation.py`](../codes/array_tutorial/separation.py) 提供四个只依赖 NumPy 的小模块：`si_sdr()` 接收等长一维估计与参考；`pit_permutation()` 接收 `(源, 样本)` 数组并枚举排列，返回“第几个输出对应第几个参考”的元组和平均 SI-SDR；`masked_spatial_covariance()` 接收 `(频点, 通道, 帧)` 复谱与 `(频点, 帧)` 非负掩码；`mask_mvdr_2x2()` 用目标 SCM 的主特征向量估计相对传递函数，再以干扰 SCM 求两麦 MVDR，返回 `(频点, 帧)` 输出和 `(频点, 2)` 权重。

联合示例用 `.venv/bin/python -m codes.examples.ch06_09_baselines` 运行，测试见 [`tests/test_codes_aec_wpe_sep_track.py`](../tests/test_codes_aec_wpe_sep_track.py)。测试覆盖输出交换、静音参考和静音估计拒绝、SCM 厄米性、MVDR 无失真约束，以及目标或干扰掩码为空时回退参考麦。实际评测还必须固定去均值、时延/增益对齐、静音段、截断长度和输入基线；秩亏 SCM、单通道输入、$N>M$、跨块换人及 STFT 重构均需单独测试。教学版 MVDR 在目标或干扰统计不可用时退回参考麦克风，不应把这一回退的输出解释成成功分离。

本章的外部实现已经按具体算法定位，不能把“正文尚未实现”写成“没有可用源码”。[盲分离与 GSS 研究](../codes/research/02_aec_wpe_separation.md#bss)和[神经分离研究](../codes/research/02_aec_wpe_separation.md#neural)进一步列出读码顺序、状态、最小实验与失败条件。

| 方法 | 已定位的源码入口 | 复现与工业使用的主要限制 |
|---|---|---|
| AuxIVA、ILRMA | pyroomacoustics `bss/auxiva.py`、`ilrma.py`；ssspy `bss/iva.py`、`ilrma.py` | 区分 IP/ISS 更新、源数条件、功率下限和 projection-back |
| MNMF、FastMNMF/FastMNMF2 | ssspy `bss/mnmf.py`；pyroomacoustics `bss/fastmnmf.py`、`fastmnmf2.py` | 固定初值、NMF 基数、空间模型、参考麦与重构尺度 |
| TRINICON | pyroomacoustics `bss/trinicon.py` | 此实现固定两个输出，不能当作任意源数版本；块长和滤波器长度分开 |
| cACGMM | pb_bss `distribution/cacgmm.py` | 方向归一、空分量、密度数值稳定与功率 SCM 分开检查 |
| GPU-GSS | `gss/core/enhancer.py` → `gss/wpe/` → `gss/cacgmm/` → `gss/beamformer/` | 需活动标注、同步多通道音频及匹配的 CUDA/CuPy；活动错误会传播 |
| Conv-TasNet、DPRNN | Asteroid `models/conv_tasnet.py`、`dprnn_tasnet.py` | 配套 recipe、编码窗、归一化、源数、循环方向和权重 |
| SepFormer | SpeechBrain `lobes/models/dual_path.py` 与 WSJ0Mix 分离 recipe | 双路径注意力和整句上下文不能直接当作流式状态 |
| TF-GridNet | ESPnet `enh/separator/tfgridnet_separator.py` | 此类明确为离线，固定输入麦数；单/多通道版本分别核对 |
| S4M、SPMamba、Mamba-TasNet | 各作者模型/训练目录，详见研究文档 | 三者不是同一算法；S4M 训练资产不完整，双向模型使用未来上下文 |
| SpeakerBeam、AudioSep | 注册语音条件模型、`pipeline.py` 等各自入口 | 身份条件与文本类别条件不同；代码、权重和数据许可分别核对 |

FastMNMF 的可用实现还体现了许可需要逐来源检查：作者 `SoundSourceSeparation` 整库限定学术研究，而 pyroomacoustics 的 `fastmnmf.py` 文件有独立 MIT 许可。SpeakerBeam 作者仓库采用内部评估协议，限制修改与再分发，不能把它与 MIT/Apache-2.0 项目统一称为可自由纳入产品的开源代码。[FastMNMF 文件许可](https://github.com/LCAV/pyroomacoustics/blob/v0.10.0/pyroomacoustics/bss/fastmnmf.py)、[作者仓库许可](https://github.com/sekiguchi92/SoundSourceSeparation/blob/897fe87fea3d85a243d8a3fd36c2232bb0548ad3/LICENSE)、[SpeakerBeam 评估协议](https://github.com/BUTSpeechFIT/speakerbeam/blob/91af02cc617afa35fedfbdbf32533012cd0a8672/LICENSE.txt)

最小的源码复现实验应先选固定两源、两麦、同一 STFT 与参考麦，记录解混/SCM、代价、尺度恢复和 SI-SDR；再分别改变源数、混响、初始化及片段长度。CSS 还需测试跨块换人和静音后重新出现，不能让每块用真实答案独立排列后再拼接。cACGMM 的正文伪代码仅解释依赖关系；完整实现中的空分量处理、对数域计算与收敛检查仍须核对。

> **练习 8-1（自测三题）**
>
> 1. 说明 WDO 假设及其失效条件。
>
>     **答案要点**：同一时频点通常只有一个人占优；重叠越重，近似越差。
>
> 2. GSS 三步顺序是什么？
>
>     **答案要点**：WPE 去混响 → cACGMM 聚类得到掩码 → 用掩码估计 SCM，再计算 MVDR/GEV 权重。
>
> 3. 掩码换协方差要写哪两个矩阵？
>
>     **答案要点**：目标协方差 $\mathbf R_{\mathrm{tar}}$ 与干扰协方差 $\mathbf R_{\mathrm{int}}$；二分类例子可分别用 $\gamma$ 与 $1-\gamma$ 加权，多说话人加环境噪声时应按模型分别定义非目标掩码，不能把 $1-\gamma$ 一律解释成纯噪声。
>
### 8.2 处理模块的顺序

回声、混响和多说话人混合对应不同的信号模型。以下顺序由参考可用性、空间相位保持和统计量依赖关系决定，接口定义见 §10.1。

一种较完整的音频主链见[第 10 章 §10.1](./10_engineering-practice.md#sec-10-1)：**采集 → AEC → WPE → 解析波束形成 → 单通道增强**。定位/追踪、说话人分割/GSS/SCM 是为解析波束提供条件的支路；神经或 CSS 分离则是从多通道特征到单通道/多流输出的替代音频路径，不与 GSS 掩码或解析波束合并成一个黑箱。项目只启用满足输入条件且能改善目标指标的模块。

1. **启用声学回声消除（Acoustic Echo Cancellation，AEC）时先处理回声**：回声参考与麦克风回声之间的线性关系可能被后续自适应波束、非线性增益或单通道增强改变，见[第 10 章 §10.1](./10_engineering-practice.md#sec-10-1)；
2. **多通道 WPE 通常在波束之前**：晚期混响会污染波束形成所需的协方差估计。WPE 是线性处理，但各通道的预测滤波器不同，不能笼统保证所有跨通道相位关系完全不变。应验证处理后的目标相对传递函数和协方差是否仍适用于后续波束；
3. **定位/追踪给波束指方向**：MVDR/GEV 要目标方向或掩码才能成形，先有方向、再有波束（见[第 5 章](./05_beamforming.md)与[第 9 章](./09_source-tracking.md)）；
4. **diarization 在 GSS 聚类之前、掩码在 MVDR 之前**：diarization 提供“谁在何时说话”的标注，用于约束 cACGMM 的跨频点置换；掩码随后用于估计目标和噪声协方差，再计算波束权重（见 §5.4 和 §8.1）；
5. **DNN 的位置由输出决定**：逐通道改写复谱或波形的网络可能破坏跨通道相位，通常放在波束形成后的单通道链路；只输出共享掩码、协方差或 WPE 功率先验的网络可以放在多通道算法内部；联合多通道网络则可能同时完成分离和波束形成。还要单独检查网络是否使用未来帧。

对于“逐通道线性前端—波束形成—单通道增强”这类实现，逐通道处理通常位于前端，单通道增强位于波束形成之后。若神经网络只估计解析算法所需的统计量，或本身联合建模多通道相位，则不受这个顺序限制。

**本章模块与图 23 的对应关系**（图的信号取点与启用条件见[第 10 章 §10.1](./10_engineering-practice.md#sec-10-1)）：

| 本章模块 | 图 23 中的位置 | 作用 |
|---|---|---|
| diarization / GSS | “说话人分割 → GSS/cACGMM 掩码”支路 | diarization 约束活动时间与跨频点置换；它不直接输出 SCM 或音频 |
| 目标/干扰 SCM 与解析波束 | “GSS 掩码 + 同阶段 WPE 输出未方向归一 STFT → SCM → DSB/MVDR/GEV”支路 | 掩码只给权重，SCM 仍由 $\mathbf Y\mathbf Y^H$ 加权计算；不能改用单位向量 $\mathbf z$ |
| 神经分离 / CSS | WPE 后的替代音频路径，可绕过解析波束后接单通道增强 | 是否输出固定多流、使用未来上下文和跨块排列，都要单独说明 |
| DNN 后滤波 / DNN 后端 | 波束形成或分离后的单通道增强或后端 | 独立改写复谱的网络通常放在多通道 WPE/MVDR 之后，以免破坏空间相位 |
| 降噪（Noise Suppression，NS）/自动增益控制（Automatic Gain Control，AGC）/语音活动检测（VAD）/关键词检出（Keyword Spotting，KWS） | 单通道增强与可选 KWS 旁路；VAD 属控制面 | 各模块按目标任务启用，不在本章展开 |

回声、混响和分离使用不同的参考与统计假设，模块顺序需要服从这些条件。下一章讨论如何把逐帧定位结果连接成连续轨迹。

---
> 📄 本篇信息：配图 0 张 ｜ [回首页](./00_overview.md)
