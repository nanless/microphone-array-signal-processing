# 空间处理与声源追踪：算法、实现和工业使用条件

基础索引核实日期：2026-09-22；第 1 章相关研究核实于2026-09-30、原归一化报告执行于2026-10-01 UTC，本轮原始资料/源码复查及原插值控制执行于2026-10-04；第 2 章历史资料与原方法提取调用核实于2026-09-30，当前原始资料/源码及限定合同复核于2026-10-04，第 4 章相关原始资料及有限原方法调用核实于 2026-10-01，第 5 章相关研究小节、原论文与限定原函数审计复核于 2026-10-01（历史报告仍保留原执行日期）；第 9 章原始资料与限定接口合同复核于 2026-10-01，保留 2026-09-28 的历史运行记录；DCASE 2026 任务状态及近年候选的后续复核日期见各条 2026-09-29 记录。对应正文第 1～5 章和第 9 章。这里按算法的输入、计算步骤和可检查结果整理源码，既包括语音前端，也包括直接相关的球阵录音与工业噪声源成像。后两类任务的输出不同，不能把声源功率图或 Ambisonics 解码结果当成增强语音。

## 阅读与复现方式

先运行本书的 [`ch02_05_baselines.py`](../cross_chapter/ch02_05_baselines.py) 和 [`ch06_09_baselines.py`](../cross_chapter/ch06_09_baselines.py)，确认时延正负、共轭、通道顺序和角度零点。再按下面的文件入口阅读外部实现。外部目录位于 `codes/chapters/ch00/upstream/_downloads/<项目 ID>/`，由锁定清单和获取工具管理。下载、导入、单次演示成功、论文实验复现和设备验收是不同结果。

本文的“最小实验”是建议的独立验证方案，未附运行结果的方案均未在本文中宣称通过。固定版本源码中的静态疑点集中放在末节；它们不能被隐藏在“有官方实现”的分类里。外部包缺少编译器、求解器、MATLAB、模型或测量数据时，仍可阅读源码，但不能据此声称运行验证完成。

共用数组约定为多通道时频谱 `M × F × T`；`M` 为通道数、`F` 为频点数、`T` 为帧数。麦克风位置在本书教学包中为 `M × D`，pyroomacoustics 则为 `D × M`。输入转换只能转置坐标排列，不能顺带改变通道顺序。角度单位、方位角零点、俯仰角与余纬角、SH 系数归一化均要在适配层显式转换。

## 固定版本和源码许可

完整提交及入口保存在 [总锁定清单](../SOURCES.lock.json) 中。下表的日期是核实日，不表示项目当天发布了新版。

| 项目 ID | 固定提交 | 代码许可与额外前提 |
|---|---|---|
| pyroomacoustics | `0dd39f2614b7fc44b2cc63dbe7d60f4641068890` | MIT；本书使用 0.10.0 版本，算法与 C++ 房间模拟器的依赖不同 |
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

## 第 1 章：从听觉线索到可检验的阵列任务

对应[第 1 章](../../../../chapters/01_problem-definition.md)。本节区分三种证据：头部几何模型给出的时延、听觉实验测得的辨别能力，以及软件根据录音估计的数值。三者可以互相启发，但它们的输入、输出和误差定义不同。

### 双耳时差、声级差与方向相关频谱

双耳时间差（Interaural Time Difference，ITD）描述两耳信号的相对时间；双耳声级差（Interaural Level Difference，ILD）描述同一频率或指定频带中的相对声级。实际声音经过头、耳廓和躯干后，还会形成随方向变化的谱峰和谱谷。后者称为方向相关频谱线索较准确。

按正文的“右减左”约定，同一频率的两耳响应幅度都非零时，`ILD = 20 log10(|H_R|/|H_L|)`，单位为 dB。左右幅度为 `1` 和 `0.5` 时，ILD 约为 −6.02 dB；交换两耳后变为 +6.02 dB。若两耳同时乘同一个非零增益，这个比值不变。这个例子是幅度比的计算，不是把左右分别归一化，也不是完整的人头滤波。

头相关传递函数（Head-Related Transfer Function，HRTF）是每只耳朵的复数频率响应，其时域对应物是头相关脉冲响应（Head-Related Impulse Response，HRIR）。一对 HRTF 同时描述左右耳的幅度和相位，因此可用来分析 ILD、时差及频谱形状；它并不是与 ITD、ILD 互不相干的第三种传感器。固定版本 SAF 的 `HRIRs2HRTFs` 把左右 HRIR 分别变换到频域，`estimateITDs` 则从同一对 HRIR 提取一个时差，正好展示这种从完整响应到摘要量的关系。[SAF 头文件及数据维度](https://github.com/leomccormack/Spatial_Audio_Framework/blob/18fd5aba46e20787b51f28f7197a68506c965c07/framework/modules/saf_hrir/saf_hrir.h)

方向谱形也不能直接从一段未知声音里无条件读出。耳边频谱同时受声源本身的频谱和传播响应影响；只有先说明已知输入、参考测量或统计假设，才能解释哪些差异来自方向。例如，单个频率的纯音不能展示完整的谱谷位置；左右各乘一个常数也只改变声级，不能模拟耳廓随频率变化的滤波。

机器阵列采用同一类观测关系，但硬件响应不同。无挡板、自由场中的两个理想全向麦克风，没有人头产生的头影，也没有耳廓谱谷。若为音箱外壳、机器人头部或耳机采集多通道响应，应记录设备几何、通道顺序、校准和测量方向，再建立该设备的阵列流形；不能把一个人的 HRTF 直接作为任意麦克风阵列的导向矢量。

### 几何时延、听觉阈值与数字采样分别回答什么

第 1 章的 Woodworth 算例使用半径 0.0875 m、声速 343 m/s，并在正前方至正侧面的角度范围使用刚性球射线路径近似。由该章模型复算，正前方偏 1°的时差为 8.9045 μs，正侧面为 655.8154 μs。这里只保留计算中间量的额外位数，图示分别取约 9 μs 和 656 μs。这是几何模型的数值，不是人体平均值或听觉辨别阈值。

在 16 kHz 数字信号里，一个采样间隔为 62.5 μs。因此上述两个时差分别对应约 0.1425 和 10.4930 个采样间隔。非整数结果没有矛盾：物理传播时延是连续的，采样间隔只是记录时间网格。能否从带限信号估计分数时延还受带宽、噪声和模型误差影响；仅把整数采样间隔缩小，不能证明定位准确率已经提高。

相同的耳间距离若改成没有头部的两个自由场传感器，正侧面的最大时差由直线距离计算：`0.175 / 343` 秒，即约 510.2041 μs。两种数值不同，是因为传播模型不同。用 656 μs 合成时延的双通道文件可以说明声像时差，但在没有头部滤波时，不能声称该文件完整模拟了真实双耳录音。[模型假设研究：Aaronson 与 Hartmann，JASA 2014](https://doi.org/10.1121/1.4861243)

Brughera、Dunai 与 Hartmann 的实验使用等声级耳机纯音，起止包络同步，任务是判断两个呈现区间之间的左右变化。其图 1 比较四位听者，两个最敏感听者在 1400 Hz 仍得到收敛阈值，1450 Hz 时未得到收敛阈值。论文的阈值量是两个区间的 `ΔITD`；例如一个区间右耳领先 10 μs、另一个区间左耳领先 10 μs，比较量为 20 μs。不能把它写成单次声源方位估计误差，也不能直接套用于宽带语音或高频调制包络。[共同作者 UPV 库保存的正式原文：§II.A，页 2840；§II.B 与图 1，页 2841；§II.C.4，页 2843](https://riunet.upv.es/server/api/core/bitstreams/f65b9337-1b1e-4013-9020-8338ac77ad6b/content)

### 从产品问题确定输入、输出和验收对象

下面是基于模块定义的工程分工建议，不是某一产品的性能承诺。读者应先写清最终要得到什么，再决定需要哪些模块。

| 实际问题 | 需要的输入与输出 | 第一项应核对的条件 |
|---|---|---|
| 会议摄像头朝向正在说话的人 | 同步多路音频 → 方向候选、活动状态和连续轨迹 | 反射峰是否被当成新说话人；无声段如何保持或释放轨迹 |
| 远处语音需要送入识别器 | 多路音频、目标控制或目标统计 → 增强音频 | 目标是否被相消；识别错误与输出失真是否改善，不能只看音量 |
| 音箱边播放边收音 | 麦克风采集与播放参考 → 回声处理后的采集音频 | 播放参考是否可取得、是否同步；电视机声音未必有可用参考 |
| 多人同时讲话且都要保留 | 混合音频 → 多个输出流及各自活动范围 | 输出流的说话人身份是否跨时间一致；单束目标增强不等于完整分离 |
| 用耳机呈现一个虚拟方向 | 干声、方向和左右 HRIR → 左右耳输出 | 数据坐标及通道顺序；个体差异、耳机响应和头动是否被处理 |

这些任务有不同的软件入口。例如 SOFA 官方的 [Software and APIs](https://www.sofacoustics.org/mediawiki/index.php/Software_and_APIs) 页面列出 libmysofa 的 C 数据读取、IRCAM Spat/Panoramix 的空间创作与混音，以及 3D Tune-In 的耳机空间化用途。用途还可直接查[IRCAM Panoramix维护者手册§1与§6](https://forum.ircam.fr/media/uploads/forumnet-legacy/2016/12/Panoramix-QuickStart2.pdf)中的空间创作、混音及双耳输出，以及[3D Tune-In维护者说明](https://github.com/3DTune-In/3dti_AudioToolkit)中的耳机空间化模块。2026-10-04仅审读这些用途段落，未取得这两个项目的源码或验证当前许可、运行平台、渲染和设备性能；数据许可与代码许可仍须分别核查。第 1 章优先用已锁定的 libmysofa 检查共同增益；个体化 HRTF 与神经插值需要数据、基线误差和渲染实验，本章不再增加只有名称的模型。

ODAS 是现有源码中连接声源定位、追踪与分离的入口之一。其维护者 README 把定位、追踪、分离和后滤波列为不同功能；本书固定的 `mod_ssl.c`、`mod_sst.c`、`mod_sss.c` 也分别保留这些模块。它可以帮助读者理解方向候选如何变成连续轨迹、轨迹怎样控制输出流，但取得这些 C 源文件不能证明已经在某个设备上达到实时性能。[ODAS 固定版本说明](https://github.com/introlab/odas/blob/bcb845434495e293df3d48f1203b7a86e1852449/README.md)

对麦克风平均增益，还要分别记录目标与噪声。若两者都减半，输出功率都会降到四分之一，信噪比保持不变；若噪声通道相关，则四麦等权平均也不必得到 6.02 dB。第 1 章 E01-01～E01-03 已给出这两类反例与不等噪声权重计算。实际录音缺少目标和噪声的分量参考时，只能报告可测的总输出功率或其他明确指标，不能由声音变小反推 SNR 增益。

### 现有官方源码怎样用于双耳研究

优先复用已经取得的 Spatial Audio Framework（SAF）。本节核对的是固定提交 `18fd5aba46e20787b51f28f7197a68506c965c07` 的源码行为；尚未在本节运行其编译和音频渲染，也没有将其函数改名当成本书独立实现。相关模块源码和头文件声明 ISC 许可，其他可选模块及第三方依赖仍按上表分别核对。

| 源码入口 | 实际操作 | 与本章的关系及限制 |
|---|---|---|
| `framework/modules/saf_hrir/saf_hrir.c::estimateITDs` | 左右 HRIR 先经 750 Hz 二阶低通，再寻找互相关最大值，换算到秒并限幅 | 是特定带宽及峰值规则的时差估计，不是 Woodworth 几何公式或听觉神经模型 |
| 同文件 `HRIRs2HRTFs` | 对每个方向、每只耳朵作实数 FFT；FFT 长度不短于 HRIR 时补零，否则先截断 | 可以从 HRIR 检查频率响应；补零不增加测量信息，短 FFT 会丢失尾部 |
| 同文件 `interpHRTFs` | 可直接插值复数 HRTF；提供 ITD 时走幅度与时差相关的处理分支 | 插值规则会改变幅相；要分别测中间方向与原测量方向的误差 |
| `framework/modules/saf_hrir/saf_hrir.h` | 定义方向、左右耳、滤波器长度与输出布局 | 适配前先检查实际内存顺序，不能仅按二维数组外观猜左右声道 |

`estimateITDs` 的源码先在全部互相关延迟上找最大值，再把结果限制到约 ±707.1 μs；这与“只在允许延迟区间内寻找最大值”是不同算法。其峰值索引为整数，未执行峰间插值。在 48 kHz 输入下，限幅前的时延网格间隔为约 20.8333 μs，不能因此声称能验证 9 μs 的辨别能力。[固定版本具体实现](https://github.com/leomccormack/Spatial_Audio_Framework/blob/18fd5aba46e20787b51f28f7197a68506c965c07/framework/modules/saf_hrir/saf_hrir.c)

建议先用左右相同的人工脉冲、交换左右的整数延迟脉冲、近边界时延和全零输入检查符号、网格与退化输出，再使用数据集 HRIR。这里的人工脉冲用于检查软件接口，不是人体响应。全零输入和强干扰峰尤其应单列，因为函数返回的有限时差本身并不证明存在可靠的声源证据。

### SOFA 数据接口与新增资源的收录边界

空间定向声学格式（Spatially Oriented Format for Acoustics，SOFA）用于交换 HRTF、双耳或空间房间脉冲响应等数据。格式兼容只说明软件知道怎样组织数据；仍须读取具体文件中的坐标、采样率、接收器顺序、时延及数据许可。[SOFA 项目说明与规范入口](https://www.sofacoustics.org/mediawiki/index.php/Main_Page)

读取 SOFA、按方向获取和插值左右滤波器，可阅读维护者官方实现 [libmysofa](https://github.com/hoene/libmysofa)。SOFA 项目的 [Software and APIs](https://www.sofacoustics.org/mediawiki/index.php/Software_and_APIs) 页面直接链接该实现。本书锁定版本为 v1.3.5、提交 `6cc5b15a73e9bd97810d03767082edda7f315881`，其[许可证文件](https://github.com/hoene/libmysofa/blob/6cc5b15a73e9bd97810d03767082edda7f315881/LICENSE)给出三条款 BSD 条件。源码子集和许可证已取得到 `codes/chapters/ch00/upstream/_downloads/libmysofa/`。原归一化报告的实际执行日期为2026-10-01 UTC，2026-09-30是此前资料审读日期；该次仅对原方法的公共归一化函数进行了提取调用：直接编译原始 `loudness.c` 和 `tools.c`，在临时目录用人工结构调用；未取得 `share/` 和 `tests/` 中的 SOFA 测量数据，也未运行完整读取与渲染链。

官方接口文档说明 `mysofa_open` 会在读取时归一化，而 `mysofa_open_no_norm` 保留未归一化数据。研究方向间增益、ILD 或不同软件输出时，应明确实际调用和增益处理；归一化、重采样和插值都不是“原始文件完全未变”。固定版 `loudness.c` 计算一个公共缩放因子，并乘到整个 `DataIR` 数组，两耳不会在这一步各自归一化。[维护者接口说明](https://github.com/hoene/libmysofa/blob/6cc5b15a73e9bd97810d03767082edda7f315881/README.md)

**最小原生调用的实际结果。** [调用工具](../../ch01/examples/audit_libmysofa_loudness.py)先核对来源锁表、HEAD、干净工作树及编译所需源码与头文件的 Git 字节和 SHA；原始 C 文件直接参与编译，不修改上游。临时导出头只定义空的 `MYSOFA_EXPORT` 宏，是编译脚手架。自写调用程序设一个球坐标方向 `[0°, 0°, 1 m]`、两个接收器和每耳一个系数，结构尺寸为 `M=1、C=3、R=2、N=1`。人工系数不是测量 HRIR。

| 人工输入 `[左, 右]` | 独立计算的预期 | 实际调用结果 | 可以支持什么 |
|---|---|---|---|
| `[1, 0.5]` | 两耳能量和 `1²+0.5²=1.25`；公共因子 `sqrt(2/1.25)`，约 1.264911 | 因子约 1.264911；输出约 `[1.264911, 0.632456]`，两耳能量和约 2、幅度比约 2 | 该输入下共同缩放保持右减左 ILD 约 −6.02 dB |
| `[0, 0]` | 选定响应能量为零，公式涉及除零与无穷乘零 | 因子为正无穷，两耳输出都是 NaN | 该固定版本及编译配置下的退化；不是成功归一化 |

运行使用 Apple clang 21.0.0、C99、`-O0`，非零数值的绝对核对容差为 `2×10⁻⁶`。[实际报告](../../ch01/reports/libmysofa_loudness.json)保留编译器、实际命令、源 SHA、脚手架摘要、输入、输出与未执行范围；非有限结果以 `null` 和分类字段保存，符合严格 JSON。报告中的“符合预期”包括确认全零退化，不代表两例都成功归一化。可在已有锁定源码与 C 编译器的环境运行：

```bash
.venv/bin/python -m codes.chapters.ch01.examples.audit_libmysofa_loudness
```

参考方向的选择也有条件。固定版 `loudness.c` 先在球坐标中最小化方位角与仰角的和 `c[0]+c[1]`，相同时再比较半径；它不是一般的角距离最小化。此次只有一个方向，验证了公共缩放与全零退化，没有验证多方向网格的参考选择。2026-10-04重新执行默认只读stdout，两个结果保持；未覆盖或改写这份历史报告。使用自己的数据网格时，仍须核对哪个响应被选作参考及其能量。[固定版函数，行 30–55](https://github.com/hoene/libmysofa/blob/6cc5b15a73e9bd97810d03767082edda7f315881/src/hrtf/loudness.c)

额外延迟需单独核查。固定版 README 将浮点接口的延迟标为秒、short 接口标为采样数，但源码与 SOFA 的数据单位之间存在不一致：SOFA 将 `Data.Delay` 定义为采样数；`reader.c` 直接读取该数组，`interpolate.c` 对其取值或加权，浮点接口直接返回所得数值。重采样时 `resample.c` 按新旧采样率之比缩放延迟，而 short 接口会将插值结果再乘当前采样率。

因此，不能只按 README 的“秒”注释决定补偿量，也不能忽略额外延迟。本书记录的是固定版本的静态源码矛盾，尚未用含非零 `Data.Delay`、不同输入输出采样率的文件运行验证，不据此声称端到端渲染已经正确。[SOFA 延迟定义](https://www.sofaconventions.org/mediawiki/index.php/GeneralFIR)；[固定版接口实现](https://github.com/hoene/libmysofa/blob/6cc5b15a73e9bd97810d03767082edda7f315881/src/hrtf/easy.c)；[固定版重采样实现](https://github.com/hoene/libmysofa/blob/6cc5b15a73e9bd97810d03767082edda7f315881/src/hrtf/resample.c)。

**方向查询还须转换坐标。** 本书水平面角度以正前方为零、向右为正；固定版libmysofa的说明采用X向前、Y向左、Z向上，方位角逆时针增加。在听者朝向及上下轴一致、水平面、采用该球坐标约定时，输入方位应为本书角度的负值：本书右30°对应SOFA −30°，或同一方向的330°。原归一化案例只有0°，所以不能检出左右镜像。使用具体文件前还要读坐标类型、`ListenerView`、`ListenerUp`与接收器次序；这个负号关系不能用于朝向或坐标轴已经旋转的任意文件。[固定版坐标说明](https://github.com/hoene/libmysofa/blob/6cc5b15a73e9bd97810d03767082edda7f315881/README.md)

**公共延迟的原插值控制（2026-10-04实际运行）。** [独立工具](../../ch01/examples/audit_libmysofa_interpolation.py)编译完整原`interpolate.c`与`tools.c`，由人工调用者提供两个位置、最近点及邻点索引。两个位置为(−1,0,0)/(1,0,0)，每个接收器仅一个IR系数，两位置的左右系数分别为[2,4]/[4,6]；查询中点到两位置的距离均为1，所以IR的独立预期为[3,5]。这不是经过SOFA读取、格式检查和自动邻域查询的测量数据。

| 延迟值的存储方式与查询 | 独立数学预期：左右IR／延迟值 | 原函数实际结果 | 解释 |
|---|---|---|---|
| 公共[8,16]，准确命中第一位置 | [2,4]／[8,16] | [2,4]／[8,16] | 距离为零时直接返回 |
| 公共[8,16]，等距中点 | [3,5]／[8,16] | [3,5]／[4,8] | 公共延迟保持的不变量失败 |
| 公共[0,0]，等距中点 | [3,5]／[0,0] | [3,5]／[0,0] | 零值会掩盖上述缺陷 |
| 每方向均[8,16]，等距中点 | [3,5]／[8,16] | [3,5]／[8,16] | 相同延迟换成逐方向布局则保持 |
| 两方向[8,16]/[12,20]，等距中点 | [3,5]／[10,18] | [3,5]／[10,18] | 逐方向分支累计邻点后正确平均 |

原因在固定版原代码的公共延迟分支：先把最近点延迟乘距离倒数，加入邻点时却只在逐方向延迟布局下累计延迟，最后两种布局都除以总权重。中点的两个权重均为1，公共分支只有一个分子的[8,16]，却除以2。因此IR插值正确、延迟返回有限值，都不能证明延迟保持正确。[固定原函数](https://github.com/hoene/libmysofa/blob/6cc5b15a73e9bd97810d03767082edda7f315881/src/hrtf/interpolate.c)

[实际报告](../../ch01/reports/libmysofa_interpolation.json)把独立数学不变量失败与成功复现原缺陷分开保存，记录完整origin/提交、六所用原文件的Git blob/SHA与许可、前后洁净、当前锁表/状态及完整选集核验范围。Apple clang21、C99、`-O0`的五人工控制不验证延迟单位转换；表内是裸结构的延迟数值，不能由本次调用声称秒或采样数已经正确处理。SOFA GeneralFIR规范的`Data.Delay`定义为采样数且允许公共IR/逐方向MR布局，是另一个规范依据；前述easy/重采样单位矛盾仍保留静态范围。原SOFA解析、自动方向查询、真实数据、重采样、卷积、渲染与设备实验均未运行，上游和旧归一化报告不改。

```bash
.venv/bin/python -m codes.chapters.ch01.examples.audit_libmysofa_interpolation
```

此命令默认只写stdout；只有显式`--report codes/chapters/ch01/reports/libmysofa_interpolation.json`才写当前报告，写前拒绝符号链接、受保护路径或覆盖其他仓内文件。

SAF 提供听觉响应的处理函数，libmysofa 补充 SOFA 数据读取及方向查询入口，两者承担不同步骤。源码筛选保留 `src/hrtf/`、`src/hdf/`、`src/resampler/`（K-D 树源码位于 `src/hrtf/kdtree.c`）、必要构建说明、README 和许可证；省略测试和测量数据，不表示已经形成可直接重现官方测试的完整环境。重采样器与 K-D 树的原始源码声明也随文件保留。HRTF 个体化、神经插值及神经双耳渲染需要独立的数据、误差定义和渲染实验，本章不把这些方法名称堆入入门任务目录。

## 基础模型、统计估计与校准

### 1. 远场与近场导向矢量

对应正文 §2.1～2.3、§2.7。传播模型的输入包括麦克风位置、声源位置或方向、物理频率和声速；输出是各通道相对同一参考的复数响应。代码中的坐标、音频通道和校准因子必须逐项对应。把米误写成厘米会改变相位，增加对角加载无法修正这种几何错误。

#### 1.1 先区分传播响应、归一化响应和波束权重

在自由场点声源模型中，设源到麦 $m$ 的距离为 $r_m>0$。前向傅里叶变换采用负指数时，传播响应正比于 $e^{-j2\pi f r_m/c}/r_m$。以麦 1 为参考，除去公共传播增益后，第 $m$ 路响应为 $(r_1/r_m)e^{-j2\pi f(r_m-r_1)/c}$。其中 $r_1/r_m$ 无量纲，第一路响应等于 1；改变参考麦会改变表示形式，不会改变实际录音。

远场近似再把距离差换成位置投影，并把幅度比近似为 1。直接比较近场和远场的绝对相位，会把声源至阵列的公共传播距离混入误差。建议实验沿同一方向逐步增大距离，比较相对时延、幅度比与去除公共相位后的响应；它们是三个不同的误差量。

固定 pyroomacoustics 0.10.0 的 `doa/doa.py::ModeVector` 不能直接充当上述完整球面响应：`near` 分支计算欧氏距离，但最终只构造 `exp(+j*omega*dist/c)`，没有距离倒数衰减。它还采用算法内部的符号与参考约定。当前上层 DOA 构造未将 `mode` 传入 `ModeVector` 的限制见本文 §8；仅看函数参数存在 `near` 不能证明近场调用链已正确启用。[固定版 ModeVector](https://github.com/LCAV/pyroomacoustics/blob/0dd39f2614b7fc44b2cc63dbe7d60f4641068890/pyroomacoustics/doa/doa.py)。

Acoular 提供另一组需要明确区分的对象。固定提交的 `SteeringVector.transfer(f)` 返回传播矩阵；`fastFuncs.py::_transferCoreFunc` 实际计算相对参考距离的距离比和负指数相位。随后 `steer_vector(f)` 才依据 `steer_type` 把传播矩阵变为波束权重。它的 `classic`、`inverse`、`true level`、`true location` 使用不同的幅度归一化，不能把返回值全部理解为本书的单位参考导向矢量。[传播与权重接口](https://github.com/acoular/acoular/blob/13d3d7df74ac1a8135c7ec71da098cbbc03d8652/acoular/fbeamform.py)；[传播内核](https://github.com/acoular/acoular/blob/13d3d7df74ac1a8135c7ec71da098cbbc03d8652/acoular/fastFuncs.py)。

| 对象 | 需要固定的参考 | 核对方法 |
|---|---|---|
| 自由场传播响应 | 源端幅度与传播距离 | 单一路径时延和 $1/r$ 衰减 |
| 本书相对导向 | 参考麦 1 | 第一分量为 1，保留通道顺序 |
| Acoular `transfer` | `ref` 指定的参考位置或距离 | 比较同一个参考，不直接逐元素减去本书向量 |
| Acoular `steer_vector` | 上一行参考与 `steer_type` | 独立计算目标响应、输出功率归一化 |

本章的[原方法调用工具](../../ch02/examples/audit_upstream_models.py)已在2026-10-04重新实际运行，结果见[当前严格 JSON 合同](../../ch02/reports/upstream_model_contracts.json)；[历史报告](../../ch02/reports/upstream_models.json)保留当时的源码与运行身份，不改写。工具从已有固定源码中用 AST 选取定义，在内存中执行原函数体；不导入整包、不修改外部工作树。当前报告记录实际官方origin、固定HEAD、所用文件及许可的Git blob/SHA、前后洁净、完整锁表/状态和工具及公共依赖摘要。所用文件通过身份核验与完整稀疏选集是两栏：当前PRA、pyfar、声速项目选集通过，Acoular与doatools仍是`source_selection_mismatch`，限定方法调用没有把它们升级。

Acoular 的 NumPy 数值体保留，但在内存移除了 Numba/JIT 装饰器；四种权重直接选取原来的 lambda 字典，未实例化 Traits 管线。这种运行能检查所选方法，不能代表已运行完整工业声学成像系统。声速项目只核说明和许可身份，没有执行LabVIEW。报告默认只输出stdout；显式写入前检查普通父链和文件，仓内只允许新的当前合同路径，拒绝旧报告、源码和上游缓存。

小输入固定三麦横坐标为 $[-0.1,0,0.1]$ m，近场源为 $[0.2,0,0]$ m，距离依次为 $[0.3,0.2,0.1]$ m；以中间麦为参考，频率 1 kHz，声速 343 m/s。PRA 使用 $f_s=8$ kHz、`nfft=8` 的第 1 个频点。原 `near` 输出相对相位约为 $[+104.9563°,0,-104.9563°]$，幅度全为 1；本书物理响应的相位正负相反，幅度比为 $[2/3,1,2]$。两种 `precompute` 路线都已运行。另用 $[1,0,0]$ **单位方向**调用 `far`；它不能用近场源坐标代替。上层 DOA 构造与搜索半径的连通性仍只是静态核查。

Acoular 传播内核使用相同距离及 0.2 m 参考距离。独立期望由标量正弦、余弦和距离比计算，未用待检函数生成答案。内核先将相位转成 `float32`，因此采用 $2\times10^{-6}$ 的绝对容差。设该输入传播响应为 $\vec a$、返回权重为 $\vec w$，下面两列分别核对目标增益 $\vec w^H\vec a$ 和独立单位方差白噪声的输出功率 $\|\vec w\|^2$；它们不是同一个量。

| `steer_type` | 本小输入的目标响应 | 权重平方范数 |
|---|---:|---:|
| `classic` | $11/9\approx1.22222$ | $1/3$ |
| `inverse` | $1$ | $7/18\approx0.38889$ |
| `true level` | $1$ | $9/49\approx0.18367$ |
| `true location` | $7/(3\sqrt3)\approx1.34715$ | $1/3$ |

后两列由 $[2/3,1,2]$ 的幅度独立手算，原字典调用在上述容差内一致。本例每路响应非零；没有测试零距离、零通道响应或含流场的传播，不能外推这些除法在退化输入上也稳健。

#### 1.2 声速参数必须与实际实现一致

正文采用 $c=343$ m/s 作为近室温的教学近似。pyroomacoustics 0.10.0 的 `parameters.py::calculate_speed_of_sound(t,h,p)` 实际使用 `331.4 + 0.6*t + 0.0124*h`，温度单位为 °C、相对湿度按 0～100 的百分数输入。函数虽接收压力 `p`，计算式没有使用它；`Physics` 内部也将压力标注为未使用。这是该版本的简化经验实现，不能写成执行了 Cramer 的完整湿空气模型。[固定版环境参数](https://github.com/LCAV/pyroomacoustics/blob/0dd39f2614b7fc44b2cc63dbe7d60f4641068890/pyroomacoustics/parameters.py)。

复现实验时，应同时记录传入温度、湿度和最终采用的 `room.c`。例如显式传入 20°C、0% 会由该式得到 343.4 m/s；省略温度时则按库的声速常数反算温度。这两个设置不可因都写作“室温”就视为相同。原方法提取调用已核对 `(20°C,0%,101.325 kPa)` 得 343.4 m/s；湿度改为 50% 得 344.02 m/s，压力再改为 80 kPa 仍为 344.02 m/s。这些是函数运行结果，不是声速测量，也没有执行 `Physics` 到 `room.c` 的完整设置链。

需要环境校准与不确定度预算时，简单温湿度公式遗漏的压力、气体组分和频率效应才成为需要解决的问题。Gavioso、Astrua、Zucco、Pisani 的 2025 年原研究 *Speed of sound in humid air: Accurate thermodynamic model and experimental validation*，JPCRD 54(4), 043101，[DOI](https://doi.org/10.1063/5.0294663)及[INRiM 作者机构原文](https://iris.inrim.it/retrieve/3fb286b8-b616-4d85-9c76-d08d22e41af2/Gavioso_et.pdf)，从热力学性质与声学弛豫建立模型，并给出不确定度计算。它扩展的是传播参数的环境描述，不改变阵列几何公式，也不替代通道校准。论文 §1 的实验验证范围为 284～301 K、99.5～100 kPa、30%～60% 相对湿度、400～700 ppm CO₂；这比模型声称可计算的全域窄，不能把全域计算能力写成全域实验验证。

原文 §6 与参考文献 81 指向[原作者仓库](https://github.com/RobertoGavioso/Speed-of-sound-in-air/tree/5c7e6652cf2fd7b4c52e83d8fa3782b3c56e61de)。固定提交 `5c7e6652cf2fd7b4c52e83d8fa3782b3c56e61de` 的 43 个 VI、`ReadMe.txt` 与 GPLv3 `LICENSE` 已通过总锁表取得到 `upstream/_downloads/speed-of-sound-in-air/`，没有获取 EXE 或论文 PDF。VI 是 LabVIEW 保存的图形源码；当前只读了论文、说明和许可，没有使用 LabVIEW 查看框图或运行高精度模型。作者要求源码使用晚于 2016 的 64 位 LabVIEW；另行提供的 EXE 依赖 2016 的 32 位 Runtime，这与源码环境不同。

最小复现可固定温度、压力、水汽或相对湿度、CO₂ 和频率，以论文表 4 的一个完整条件核对声速，再逐项改变环境变量、检查不确定度预算。当前缺少 LabVIEW，故此方案保留为建议；本章已执行的仍是 PRA 简化函数。源代码 GPLv3、专有运行时与机构论文的许可分别核对，不把代码许可延伸到论文再分发。入门推导继续使用 343 m/s 教学约值。

正文引用的 ETSI ETR 273-1-1 为 1998 年电磁辐射测量技术报告。其 §7.2.3.1、印刷页 76 给出点源与孔径边缘的路径差及 $\lambda/16$ 容限；这里采用的是相同的波程几何。它不规定麦克风阵列应当接受多大的模型误差，也不替代声学条件和设备响应检查。[原报告](https://www.etsi.org/deliver/etsi_etr/200_299/2730101/01_60/etr_2730101e01p.pdf)。

#### 1.3 从理想阵列到实际设备

实际通道还包含麦克风、外壳、模拟前端和数字滤波的响应。若这些响应为 $G_m(f,\Omega)$，相对参考流形要包含 $G_m/G_1$，而不只是距离差。方向相关的外壳散射不能靠每通道一个常数修正；独立设备的时钟漂移也不是一个固定相位校准可以长期消除的量。

Acoular `Calib.result()` 将每个数据块逐通道乘以指定校准因子，因子单位为 Pa/unit；频域输入也可使用按频点、通道排列的因子。这个接口应用已有校准数据，不会自动测量灵敏度，不会推断麦克风坐标或同步偏差。其 XML 读取路径载入实数因子，不能把这种文件称为完整复频响校准。[固定版校准源码](https://github.com/acoular/acoular/blob/13d3d7df74ac1a8135c7ec71da098cbbc03d8652/acoular/calib.py)。

设备实验至少保留以下记录；缺少某项时，缩小结论到已知范围。

| 条件 | 应保存的内容 | 它影响什么 |
|---|---|---|
| 几何和通道 | 米制坐标、通道排列、缺失通道 | 距离差及方向正负号 |
| 幅度校准 | 每通道灵敏度、单位、校准频带 | 数字功率能否换算为 Pa² |
| 复响应校准 | 频率、方向、相位参考与测量方法 | 模型失配、主瓣和零陷 |
| 采样同步 | 公共时钟或偏移估计、时间戳 | 固定 TDOA 与随时间漂移的区别 |
| 环境 | 声速、温湿度、流场模型 | 路径时延与传播假设 |

Acoular 的静止均匀环境与含流场模型也应分开使用。这里只核查源码和接口含义，没有执行本机设备校准或风洞实验；官方声学成像示例不能替代某台语音设备的测量。

HARK 的传递函数文件也需要分开检查结构与复数数值。固定 HARKTOOL5 3.5.0 的 `harktoolcli-validtf.c` 调用 `harkio_TransferFunction_fromFile_withoutLoadingMatrix`，因此其中的结构检查不等于加载并核对各方向、频点、通道的复谱。[HARK 官方 FAQ 的 Transfer Function 小节](https://hark.jp/faq/)说明其 Python 包没有直接提供这种读取方法，需依文件格式处理。接口身份与许可见本文 §39；本章没有执行 HARK 复谱读取，不把方向数、通道数检查算作相位校准验证。

### 2. 镜像法与射线追踪房间模拟

对应正文 §2.4。pyroomacoustics 的 `room.py`、`libroom_src/` 分别提供场景接口和计算实现。镜像法构造离散反射路径，射线追踪用声线和能量传播描述更晚的响应，并可包含散射设置。这些几何声学方法不自动覆盖衍射、结构传声、扬声器失真或采集链路非线性。[官方房间配置说明](https://pyroomacoustics.readthedocs.io/en/stable/pyroomacoustics.room.html)。

#### 2.1 从目标混响时间到模拟器输入

`acoustics.py::inverse_sabine(rt60, room_dim, c)` 返回平均**能量吸收系数**和建议的最大镜像阶数。三维房间中，它由体积、表面积和声速代入赛宾关系得到吸收系数；阶数由路径长度覆盖的几何估计决定。输入的 `rt60` 是设计目标，并非对最终 RIR 的测量结果。[固定版反推代码](https://github.com/LCAV/pyroomacoustics/blob/0dd39f2614b7fc44b2cc63dbe7d60f4641068890/pyroomacoustics/acoustics.py)。

把能量和幅度分开手算有助于检查材料参数。在不考虑透射、反射相位且使用实幅度反射因子的简化模型中，能量吸收系数 $\alpha=0.36$ 意味着保留 0.64 的能量，对应幅度反射因子 $\sqrt{0.64}=0.8$。两次相同反射的幅度因子为 $0.8^2=0.64$，能量因子为 $0.64^2=0.4096$。不能把 0.36 直接当作幅度损失后再平方。

[Allen 与 Berkley 1979 原文](https://users.umiacs.umd.edu/~ramanid/cmsc828d_audio/AllenBerkley79.pdf)（大学提供的原论文副本）§I.A～I.B、印刷页 943～945 将离散镜像路径与房间冲激响应对应，式(9)区分吸收与幅度反射因子。本文使用 MIT 许可的独立实现 pyroomacoustics；论文中的历史 FORTRAN 清单不是该库的源代码，也不能由库的 MIT 许可推定可以再分发。

这只是参数定义例子。实际材料系数随频率、入射角和测量条件变化；“0”与“1”表示模型中的理想极限，不能据此宣称某种瓷砖或窗户在所有频率恰为极限材料。版本 0.10.0 推荐 `materials=Material(...)`；已弃用的 `absorption` 参数沿用不同历史含义，复现旧脚本应先转换口径。

模拟器输入还需保存源位、麦位、采样率、指向性、空气吸收、反射阶数，以及是否启用随机镜像扰动。规则鞋盒房间的镜像位置可能产生规律性的回声结构；随机镜像方法用于扰动该结构。它改变的是模拟假设，不证明新波形更接近任意真实房间。

#### 2.2 用生成的 RIR 检查结果

建议按由简单到复杂的顺序检查：无反射时核对首达时刻和距离衰减；单面反射时核对镜像路径的几何长度；加入多次反射后，才检查衰减曲线、直达混响比和阶数增加后的变化。每一步先确认时间原点、库引入的分数延迟滤波器延迟、高通和截断处理。

附录 B 第 16 题已经用锁定 pyroomacoustics 0.10.0 在隔离环境中实际运行[六位置房间脚本](../../appendix_b/examples/room_srp_exercise.py)，保存[18 个合成白噪声 WAV 与参数清单](../../appendix_b/room_audio/MANIFEST.json)。该版本逐条 RIR 默认使用 10 Hz 零相位高通；若用较短的零阶直达 RIR 补零后去减高阶完整 RIR，高通边界差会混入“反射能量”。脚本以相同镜像阶数和长度，令直达对照房间墙面完全吸收，并用原始单直达镜像源及同长度滤波独立核对。

六位置 $T_{20}$ 外推 $T_{60}$ 为约 0.546～0.563 s，低于反推设计值 0.6 s；定位误差为约 0.604～2°。这些数是一个房间、一次固定输入的仿真，不是实测混响时间或跨房间定位准确率。图、逐麦 DRR 和阶数收敛检查见[附录题目](../../../../chapters/13_appendix-guide.md)。

`experimental/rt60.py::measure_rt60` 对实 RIR 平方后反向累加，得到 Schroeder 剩余能量曲线，再拟合衰减斜率并外推 60 dB。`decay_db` 控制拟合跨度，默认值是 60；要使用 20 dB 跨度需显式设定。它还接受 `energy_thres` 和线性域拟合选项，改变这些参数会改变评分口径。[固定版测量代码](https://github.com/LCAV/pyroomacoustics/blob/0dd39f2614b7fc44b2cc63dbe7d60f4641068890/pyroomacoustics/experimental/rt60.py)。

原方法提取调用现已运行三个确定输入，全部显式使用 `fs=16000`、`decay_db=20`、`energy_thres=1`、`linear_domain_fit=False`、`plot=False`。有效样本为 32000 点的 $h[n]=\exp[-3\ln(10)n/(f_s\,0.6)]$；平方后的能量按每秒 100 dB 衰减，反向几何级数积分在远离截断处保持相同斜率，因此独立期望的外推 $T_{60}$ 是 0.6 s。

| 输入 | 原函数返回 | 本例判定 |
|---|---:|---|
| 上述解析指数衰减 | 约 0.6000000000 s | 与独立期望相符，绝对容差 $10^{-10}$ s |
| 8 个全零样本 | 0.0 s | 无效：能量对数和归一化未定义，实际产生 RuntimeWarning |
| `[0,0,1]` | 0.0 s | 无效：未找到 −5 dB 衰减位置 |

报告把有限的零返回与有效测量分开，记录退化分类和实际警告；严格 JSON 不输出 NaN/Infinity。数据衰减不够时函数还可能调整拟合区间，因此设备测量必须另保存原始曲线、实际区间、噪底、拟合残差和失败状态。普通房间与演出空间的适用范围分别查 [ISO 3382-2](https://www.iso.org/standard/36201.html)和 [ISO 3382-1](https://www.iso.org/standard/40979.html)。标准页面的范围说明不能代替完整条款验收，本次解析输入也不是 ISO 测量认证。

Schroeder 原研究是 1965 年 *New Method of Measuring Reverberation Time*，[DOI 10.1121/1.1909343](https://doi.org/10.1121/1.1909343)；固定源码注释中的 1968 年不能作为出版年份依据。[Brüel & Kjær Technical Review No.4 (1966)](https://www.bksv.com/media/doc/TechnicalReview1966-4.pdf) 的 Broch、Jensen《On the Measurement of Reverberation》Appendix A、印刷页 21～23 给出可读原推导：白色平稳激励在零时刻关断后，输出二阶期望对应冲激响应平方的尾积分。有限带宽激励不再有理想 δ 自相关，需要计入滤波器响应。此依据解释统计关系，不把单条实际噪声关断曲线视为无误差尾积分，也不引入历史仪器性能数字。

#### 2.3 固定版本中空气吸收公式的静态边界

pyroomacoustics 0.10.0 的 `acoustics.py::rt60_eyring` 返回值可写成 $-K V/[S\ln(1-a)+4mV]$，其中 $K=24\ln(10)/c>0$，`m` 在接口中表示空气吸收参数。固定 $0<a<1$ 时，分母首项为负；增加正的 `m` 会先让分母更接近零，使预测时间增加。这与增加吸收应加快能量衰减的单调性不符。原函数体提取调用已执行，房间仿真路径尚未执行。

小输入固定 $S=100$ m²、$V=56$ m³、$a=0.2$、$c=343$ m/s；独立正吸收参考采用 $KV/[-S\ln(1-a)+4mV]$。后者用于检查符号与单调性，没有修改上游函数，也不是新测量结果。

| `m`（m⁻¹） | 原函数返回（s） | 独立正吸收参考（s） | 本例的物理解释 |
|---:|---:|---:|---|
| 0 | 0.404330 | 0.404330 | 零空气吸收时一致 |
| 0.001 | 0.408430 | 0.400312 | 原返回随正吸收增加 |
| 0.01 | 0.449448 | 0.367445 | 同一异常趋势 |
| 0.1 | −105.346349 | 0.201778 | 原返回是无效负时间 |

表中原返回与直接代入固定表达式相符。通过检查表示准确复现了该版本行为，不表示异常返回是可用物理结果；其他版本不能据此推定存在相同问题。

`room.py::rt60_theory` 还逐面墙调用理论函数后累加倒数；当每次调用都包含同一个体积空气吸收项时，该项会随墙数重复计入。因此本文不使用该版本含空气吸收的理论返回值作为正文公式的数值基准，也不把这项静态矛盾外推成其全部 RIR 模拟器失效。[固定版理论公式](https://github.com/LCAV/pyroomacoustics/blob/0dd39f2614b7fc44b2cc63dbe7d60f4641068890/pyroomacoustics/acoustics.py)；[固定版逐墙聚合](https://github.com/LCAV/pyroomacoustics/blob/0dd39f2614b7fc44b2cc63dbe7d60f4641068890/pyroomacoustics/room.py)。

正文无空气吸收的赛宾/Eyring手算应与同样设为零空气吸收的解析模型比较。需要分析空气吸收时，先明确振幅还是能量的衰减系数，再独立核对公式、代码和生成 RIR 的衰减；这些额外路径尚未在本节运行。

#### 2.4 ESS 扫频与受限频带反卷积：pyfar

Farina 的原作者预印本 *Simultaneous measurement of impulse response and distortion with a swept-sine technique*，AES 第 108 届会议（2000）、preprint 5093，[原 PDF](https://angelofarina.it/Public/Papers/134-AES00.PDF) §2、印刷页 2～7 解释指数扫频和相配逆滤波，以及线性响应与非线性谐波响应在反卷积后分离到不同时刻的条件。它解决的是测量时的响应分离问题；不能只说“播放扫频就得到房间 RIR”，也不表示任意时变非线性都能精确恢复。pyfar 是维护者的独立通用声学工具，下面的通用正则反卷积接口不是原作者测量流程的完整复现。

正文提到的指数正弦扫频需要配套生成测试信号和反卷积实现。pyfar 补充这一入口：本书已将维护者官方仓库 v0.8.1、提交 `0bfe1e8b7d71ab83edd3ea3b5fab7b28761d114a` 的源码子集取得到 `codes/chapters/ch00/upstream/_downloads/pyfar/`，保留 MIT 许可证与作者声明；它是本地独立研究副本，不是已经并入发布仓库的第三方包。本轮已执行下面六个原函数的限定合同，没有安装运行整包测量流程，也没有播放声音或采集真实房间数据。[固定版源码](https://github.com/pyfar/pyfar/tree/0bfe1e8b7d71ab83edd3ea3b5fab7b28761d114a)；[许可证](https://github.com/pyfar/pyfar/blob/0bfe1e8b7d71ab83edd3ea3b5fab7b28761d114a/LICENSE)。

`signals/deterministic.py::exponential_sweep_time` 按指数变化的瞬时频率合成信号，输入包括样本数、起止频率、采样率、幅度和末尾淡出长度。若指定 `sweep_rate`，函数会按每秒倍频程数重新计算长度；不能同时假定传入的 `n_samples` 仍决定时长。默认淡出为 90 个样本，它在 16 kHz 下是 5.625 ms，不是接口说明中以 44.1/48 kHz 为例的约 2 ms。频域合成版本 `exponential_sweep_freq` 采用另一条生成路线，不能假定两者拥有逐样本相同的包络。[固定版扫频实现](https://github.com/pyfar/pyfar/blob/0bfe1e8b7d71ab83edd3ea3b5fab7b28761d114a/pyfar/signals/deterministic.py)。

`dsp/dsp.py::regularized_spectrum_inversion` 对输入谱构造 $X^*/(|X|^2+\epsilon)$。正则项限制低能量频点上的巨大逆增益，但也会产生偏差；它不把未受激频带的信息恢复出来。指定带内、带外参数时，该版本先构造频率过渡，再乘以输入谱最大模方；直接提供 `regu_final` 时，则按给定最终数值使用。比较结果前要固定 FFT 归一化和正则项的尺度。[固定版反卷积与逆滤波实现](https://github.com/pyfar/pyfar/blob/0bfe1e8b7d71ab83edd3ea3b5fab7b28761d114a/pyfar/dsp/dsp.py)。

同文件 `deconvolve` 的文档把 `frequency_range=None` 描述为绕过正则化，但固定源码实际将它换成 `[0,f_s/2]`，仍调用上述逆滤波函数；默认 `regu_inside=10^{-10}` 会保留很小的正则项。因此采用范围应按代码写成“默认以整个单边频带为带内”，不能写成自动识别有效扫频频带，也不能据文档注释称它完全没有正则。

`deconvolve` 还检查输入输出采样率一致，并默认补齐到两者较长的长度。这个长度规则不保证录音已经保留足够的混响尾，也不说明结果已经消除播放/采集的固定延迟。v0.8.1 源码已对该接口和旧逆滤波函数发出将来弃用的提示，推荐 `RegularizedSpectrumInversion` 与 `convolve`；本书固定版本阅读不把将来的接口变化说成已经验证的迁移。

**原函数实际调用范围。** 2026-10-04的[当前合同](../../ch02/reports/upstream_model_contracts.json)提取并执行完整`exponential_sweep_time`、`_time_domain_sweep`、`_exponential_sweep`、`regularized_spectrum_inversion`、`_cross_fade`和`deconvolve`六个原函数体。本合同仅用人工`SignalAdapter`运行实单声道、NumPy FFT和`fft_norm='none'`控制，补零、归一化匹配与警告类型也由调用脚手架提供；没有验证其多通道行为。原Signal类、整包导入、原卷积包装、Farina加权逆扫频和非线性谐波分离均未执行。

四点控制使用激励`[1,1/2]`、路径`[1,0,1/2]`和完整输出`[1,1/2,1/2,1/4]`。绝对正则项`epsilon=1/4`时，原单边估计谱为`[27/20,5/12,3/4]`，时域四抽头为`[11/15,3/20,19/60,3/20]`。这与E02-19的逐频参数目标一致；它没有施加已知三抽头支持，不能裁掉第四项后宣称得到三抽头约束解。另三项控制保留固定噪声、观测截尾及DC未受激的结果。谱零只使逐频除法失效，已知有限支持的完整线性系统还需独立检查秩。

原扫频控制采用8kHz、1024点、100～3000Hz、幅度0.2、末尾32点淡出，经过路径`[0.8,0,0,0.25]`保留1027点，FFT为2048点。显式零正则的完整IR最大误差约`1.04e-14`；默认`frequency_range=None`仍得到正则项约`2.2031e-8`，IR最大误差约`9.01e-5`。默认很小的正则也有偏差。这一控制与正文16kHz、32000点的独立音频实验使用不同参数，不能混用数字。

正文[E02-19](../../../../chapters/02_basics-signal-model.md#sec-u-30cdf46bd5)另用本书唯一[数值核](../../ch02/core/deconvolution.py)和[四份数字WAV清单](../../ch02/sweep_audio/MANIFEST.json)比较完整、后路径加噪和截尾，并把浮点与实际PCM逆解分开评分。有限扫频的设计瞬时频率范围不表示DFT带外严格为零；弱能量频点会放大量化及观测误差。全65536点IR误差与已知161点支持内误差是两个诊断，不用真值支持裁掉支持外错误。

建议的设备验证仍未执行：保留播放参考和足够混响尾，核查未削波、同步及标定，再估计路径。所得响应包含未单独校准的扬声器和麦克风传输特性，数字控制不代替这一设备链，也不与附录B已运行的房间仿真混计。

### 3. STFT 与加权重叠相加重构

对应正文 §2.5。本书 [`spectral.py`](../../ch02/core/spectral.py) 固定多通道谱形状为 `M × F × L`，分别表示通道、频点和帧。实信号单边谱的 $0\le k\le\lfloor N/2\rfloor$ 对应 $f_k=kf_s/N$，单位是 Hz；完整 DFT 的 $k>N/2$ 则对应 $(k-N)f_s/N$ 的负频率。不能把频点下标 $k$ 直接代入传播复指数。帧号、采样号和秒也应在接口边界转换。

#### 3.1 重构成立与卷积近似成立是两件事

完整信号的傅里叶变换满足卷积定理。STFT 先乘有限窗，再按帧移取样，其精确系统表示通常同时涉及跨帧卷积和频带耦合。把每个时频点独立写作“输入谱乘一个传输系数”是乘性传输函数近似（Multiplicative Transfer Function，MTF）；它不能仅由 iSTFT 往返误差小推出。

[Avargel 与 Cohen 2007 作者原文](https://webee.technion.ac.il/Sites/People/IsraelCohen/Publications/TASL_May2007.pdf)的 §II、式(7)～(11)、印刷页 1307 将 LTI 系统写成带内与跨带滤波之和。需要几个跨带项与分析/合成窗有关，跨帧滤波长度也受原始冲激响应影响。增加可估参数还会改变有限数据估计误差，不能把“更多跨带滤波”直接写作性能必然更好。

一个最小反例是延迟超出当前窗的短脉冲：当前帧没有输入脉冲，输出帧却仍可能收到先前帧的延迟响应。同一时频点只做相乘无法产生这段跨帧记忆。应先按时域线性卷积得到参考，再采用完全相同的窗、帧移和边界生成参考 STFT，最后统计近似残差；不要用待检逐点模型生成自己的正确答案。

#### 3.2 合成窗和有限信号端点

pyroomacoustics `transform/stft.py::compute_synthesis_window` 用分析窗除以所有移位分析窗的平方和，构造合成窗。这说明分母非零的重要性，但其周期移位窗计算不能自动证明某个有限录音首尾都有覆盖。[固定版合成窗函数](https://github.com/LCAV/pyroomacoustics/blob/0dd39f2614b7fc44b2cc63dbe7d60f4641068890/pyroomacoustics/transform/stft.py)。

正文 E02-03 用首点为零的 Hann 窗检查这一差别：中间区域的重叠条件成立，仍不能恢复未被任何非零窗覆盖的首点。外部 SciPy `stft` 也分别设置 `boundary` 和 `padded`；其文档将该接口标为旧式接口并推荐新应用考虑 `ShortTimeFFT`，但替换接口时仍要逐项对齐窗、尺度、时间标记和边界。[STFT 边界参数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html)；[NOLA 分母条件](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.check_NOLA.html)（文档版本 1.18.0，2026-09-28 核实）。

工程对照应使用脉冲、常量和不能整除帧移的随机信号，核对输入输出长度、启动延迟和末帧；只用稳态正弦容易漏掉边缘问题。实时系统还要记录每次调用消耗和产出的采样数、是否保留跨块状态、前瞻和结束时如何冲洗尾部。块长不会自动等于整个算法延迟。

PSD 的尺度是另一个独立检查。实信号单边功率谱须将成对的正负频率功率合并，直流和偶数长度的 Nyquist 点各只计一次；幅度谱不能直接按同一倍数解释。SciPy `periodogram` 默认 `detrend='constant'`、`scaling='density'`，功率密度单位是输入单位平方/Hz；默认去均值会删除常量信号的直流项。[NumPy `rfft` 的频点与归一化说明](https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html)；[SciPy 官方 periodogram 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html)（2026-09-30 阅读）。本书 E02-08 另用 Parseval 和频带积分核对自己的尺度，不把缺少 SciPy 的本机环境写成已执行外部 PSD 接口。

#### 3.3 修改谱还须满足跨帧一致性

NOLA检查保留样本的窗平方和非零；DC和偶数FFT的Nyquist虚部为零检查每帧是否对应实序列。两项都满足，重叠帧仍可能给共享样本指定不同值。正文[E02-20](../../../../chapters/02_basics-signal-model.md#sec-u-9fc99c5f2f)用四点矩形窗、帧移2和三个居中帧逐项展示这种冲突。

对任意待合成谱先WOLA再重新分析，得到同一分析模型能实现的谱。配套分析、合成对合法波形完美重构时，该操作幂等；是否是正交投影还需要指定内积和最小二乘合成。实谱内部频率的正负配对必须计两次，本例完整DFT距离的单边权重是`[1,2,1]`，不能把未加权rFFT距离称为同一误差。

来源为[Griffin–Lim 1984原文§II，式(4)～(6)，印刷236～237页](https://dub.ucsd.edu/CATbox/Reader/GriffinLimMSTFT.pdf)、[SciPy istft官方Notes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.istft.html)与[Wisdom等原文§3.1式(3)](https://arxiv.org/pdf/1811.08521)。本书[最小数值核](../../ch02/core/stft_consistency.py)仅执行指定三帧控制，不运行完整幅度重建迭代、神经分离器或SciPy接口。[第8章式(8-31)](../../../../chapters/08_speech-separation.md#sec-u-110769710d)讨论同一约束如何与多源混合一致性区分。

### 4. 批处理、加权与递推空间协方差

对应正文 §2.5。本书 [`covariance.py`](../../ch03/core/covariance.py) 的统计量是未中心化的空间二阶矩。零均值假设、去均值操作和无偏分母应分别核对，不能仅凭接口名含 `covariance` 就认定实现相同。单帧外积的秩至多为 1；没有去均值的 $L$ 帧平均秩至多为 $\min(M,L)$。

#### 4.1 权重归一化不增加信息量

掩码二阶矩按权重和归一化。所有非零权重共同乘正数，结果不变；实现先除以最大权重以避免过小尺度被误判为无数据。E02-06 用 $10^{-200},1,10^{200}$ 三种共同尺度核对这一点。全零权重没有统计依据，代码明确拒绝。

“权重非零”“矩阵满秩”和“快拍独立”是不同条件。很多帧若彼此成比例，平均后仍可能秩一；重叠帧也不能自动作为同样数量的独立样本。工程记录应保留权重和、权重集中程度、特征值与条件数，以及参与更新的时间范围，而不只记录矩阵形状。

#### 4.2 遗忘时间和数值正定性分别检查

递推估计要保留历史累计权重，否则从零初始化的矩阵会因权重总和不足而偏小。正文 E02-05 展示归一化，E02-07 再把遗忘因子换算为实际时间常数。语音活动检测冻结更新时，状态不会按墙钟时间自行遗忘；可变帧移也不能继续沿用固定帧移的时间解释。

收缩估计把矩阵与同尺度的单位阵混合。由特征分解直接可知，原特征值 $\lambda_i$ 会变为 $(1-\rho)\lambda_i+\rho\mu$，其中 $\mu=\operatorname{tr}(\hat{\mathbf R})/M$。只有原矩阵半正定、$\mu>0$ 且 $\rho>0$ 等条件成立时，这种混合才会把零特征值抬高；零输入产生的零矩阵不会凭空变成有效噪声模型。

这里的手选 $\rho$ 表示指定正则强度，没有执行自动收缩系数估计。将实数独立样本的收缩公式直接用于复数、重叠 STFT 快拍，也需要重新核对假设。归一化和加载可以改善计算条件，却不能纠正目标泄漏、错误通道排列或错误噪声时段。

#### 4.3 理论精度接口中的参数身份

正文式(2-14)为单源确定性模型下的 CRB。固定 doatools `performance/crb.py::crb_det_farfield_1d` 先把导向导数投影到导向张成空间的正交补，再结合源样本二阶矩和噪声方差计算下界。这一步消除了未知复幅度造成的干扰参数，解释了阵元位置为何要相对中心求二阶矩。[固定版 CRB 源码](https://github.com/morriswmz/doatools.py/blob/9469db201e0418aef6b97583ef54b6fec2769502/doatools/performance/crb.py)。

该接口的 `sigma` 表示噪声**方差**，不是标准差；`P` 是源信号样本二阶矩，`n_snapshots` 是模型中的快拍数量。相邻的 `crb_sto_farfield_1d` 和 `crb_stouc_farfield_1d` 分别采用随机源模型和不相关随机源模型，不能因为参数名类似就交换结果。源码所引 Stoica–Nehorai 1990 比较论文与正文 1989 理论论文也是不同文献。

本章工具现已选取原 `SourcePlacement`、`FarField1DSourcePlacement.phase_delay_matrix` 所在类、`projm`、`reduce_output_matrix` 和 `crb_det_farfield_1d` 实际执行。原方法与类体未修改；自写的六阵元理想 ULA 适配器把原相位及导数转换成复响应及导数，没有导入完整 `ArrayDesign`、求解器或定位器。

输入为 $M=6$、阵元间距 0.014 m、波长 0.343 m、源样本二阶矩 `P=[[10]]`、噪声方差 `sigma=1`、100 个模型独立快拍。采用相对正横的角度，分别以 rad 和 deg 输入同一方向。独立答案由正文式(2-14)的闭式解和角度单位换算得到，没有用待检投影矩阵产生参考。

| 方向 | 原接口标准差换算为度 | 独立检查 |
|---|---:|---|
| 0° | 1.19419384° | rad² 方差乘 $(180/\pi)^2$ 后与 deg² 方差一致 |
| 60° | 2.38838768° | $\cos60°=1/2$，方差为正横四倍、标准差两倍 |

四次调用在 $10^{-12}$ 相对、$10^{-14}$ 绝对方差容差内一致；一维 `P=[10]` 另被原接口拒绝，因为该确定性接口要求 $K\times K$ 矩阵。快拍独立、窄带、理想阵列、模型可辨识与局部无偏等理论前提仍须满足。这些结果是原方法提取调用与独立闭式解的对应，不是某个定位器、有限混响录音或设备的实测精度。

### 5. AIC 与 MDL 源数估计

对应 [§4.1.1](../../../../chapters/04_doa-estimation.md#sec-4-1-1) 和 [§4.9](../../../../chapters/04_doa-estimation.md#sec-4-9)。本书 `doa.py::mdl_source_count` 提供复高斯、空间白噪声模型的 MDL 教学基线；输入为任意顺序的严格正实特征值和独立快拍数，输出所选源数与全部候选评分。它拒绝秩亏、非法类型和不足以支持满秩样本协方差的快拍数，不静默加载，也不把评分写成概率。

外部入口仍为 doatools 的 `estimation/source_number.py` 中 `aic`、`mdl`、`ld_stat`。其接口接受协方差或**升序**特征值，并需要快拍数；本书接口不接受协方差矩阵。锁定版本的 `mdl` 保留公共惩罚 $\tfrac12\ln N$，本书去掉该常数，所以直接比较评分时要先统一口径，源数最小值不受影响。上游 `ld_stat` 直接计算特征值乘积，本书改用平移后的对数计算，以覆盖极大、极小尺度。[作者 API](https://morriswmz.github.io/doatools.py/references/doatools.estimation.source_number.html)；[锁定提交的官方源码](https://github.com/morriswmz/doatools.py/blob/9469db201e0418aef6b97583ef54b6fec2769502/doatools/estimation/source_number.py)。版本同时登记在 `SOURCES.lock.json`。

2026-10-01 的[原方法审计](../../ch04/examples/audit_upstream_doa.py)实际调用固定 `ld_stat/aic/mdl`，输入是 E04-20 的升序谱 `[0.8,1.2,4,9]`、100 快拍。原 AIC 选择 3，原 MDL 选择 2，与独立标量对数和及本书评分一致。`aic` 内部评分是本书去公共常数 AIC 的一半，不改变最小值；原 MDL 多出的公共项为约 2.302585。将全部特征值乘 $10^{-300}$ 后，原 `ld_stat` 的乘积下溢并返回正无穷；[严格 JSON 报告](../../ch04/reports/upstream_doa.json)以 `positive_infinity` 分类、数值字段 `null` 保存，不把它写成有效评分。本书对数计算的共同尺度不变性另由 E04-20 和教学测试核对。

E04-05 使用指定谱 $[9,4,1.1,0.9]$ 和 $N=100$，四个 MDL 总分为约 171.355476、86.437847、28.636055、34.538776，选择 2。表格、算例、`exercises_spatial.py` 与 `test_codes_mdl.py` 对应；测试期望值另由 Decimal 高精度标量公式产生，覆盖所有排序、共同尺度 $10^{-300}$ 至 $10^{300}$、最小正浮点数、纯噪声、最后一个候选和拒绝输入。这是本书手算及数值回归，不是数据集检测正确率实验。

公式核查使用 Wax 与 Kailath 的 *Determining the number of signals by information theoretic criteria*，ICASSP **1984**，§II～IV、式(10)～(15)，[DOI](https://doi.org/10.1109/ICASSP.1984.1172389)及[作者上传原文](https://www.researchgate.net/profile/Mati-Wax/publication/3177764_Detection_of_signals_by_information_theoretic_criteria/links/56cacde408aee3cee54041cd/Detection-of-signals-by-information-theoretic-criteria.pdf)。[1985 年期刊论文](https://doi.org/10.1109/TASSP.1985.1164557)另题为 *Detection of signals by information theoretic criteria*；ResearchGate 的该期刊条目挂载的是 1984 年会议稿，引用定位以 PDF 的实际版本为准。核查日期：2026-09-22。

E04-06 已加入实际重复抽样，入口为 [`mdl_repeated_trials.py`](../../ch04/examples/mdl_repeated_trials.py)，也由 `exercises_spatial.py` 执行。四麦半波长线阵，1000 Hz、343 m/s、两源 $\pm30^\circ$；每组 200 次独立圆对称复高斯快拍试验，使用 PCG64 与 `SeedSequence([20260922, 组编号])`。七组分别改变快拍数 100/16、每源功率 1/0.1、独立/完全相干源，以及无源时白噪声/协方差为 $\operatorname{diag}(9,4,1,1)$ 的有色噪声。白噪声每通道方差为 1；按阵列平均功率定义的双源 SNR 为 3.01/−6.99 dB。源和噪声每次重新生成，协方差不减样本均值、不加载；这里没有波形、STFT 或音频采样率。

完整计数与逐条件 Wilson 95% 比例区间见 [§4.9 的 E04-06](../../../../chapters/04_doa-estimation.md#sec-4-9)。NumPy 2.5.3 下，独立双源功率 1、100 快拍的 200 次均输出 2，而降至 16 快拍时为 163 次；有色纯噪声的 200 次均输出 2，尽管物理源数为 0。相干双源有 199 次输出 1，这与总体信号秩 1 一致，但不等于检出两个物理源。区间只描述指定条件下输出与物理源数相符的重复事件，不是单次 MDL 置信度；全相符也不保证以后不失败。[区间公式：NIST/SEMATECH §7.2.4.1](https://itl.nist.gov/div898/handbook/prc/section2/prc241.htm)。

这组实验没有检验重叠 STFT 帧、不同源强比、真实房间或时间迟滞。重叠帧不天然独立，把帧数全部当成独立快拍会改变评分口径；工程中的源数检测还应评估迟滞，避免每帧改变下游子空间维度。不能由本书七组数学仿真推出真实录音检测率。

正文还讨论特征值间隙这一辅助启发式：先明确特征值排序，再查相邻差或比的突变。它不是上述 AIC/MDL 的另一种名称，没有由惩罚似然自动得到的快拍数修正。强弱源、空间有色噪声和相干源都可能使最大的间隙不对应物理源数；本文没有把一般特征值分解函数登记为已复现的源数检测器，也未运行该启发式的成功率实验。

### 6. 前向与前后向空间平滑

对应 §4.6。doatools 的 `estimation/preprocessing.py::spatial_smooth(R,l,fb)` 将平移子阵协方差平均；这里 `l` 是子阵数量，输出维度为 `M-l+1`，不是很多论文中的子阵长度定义。[作者接口与 Pillai–Kwon 文献定位](https://morriswmz.github.io/doatools.py/references/doatools.estimation.preprocessing.html)。

建议用 8 元 ULA、两条完全相关的平面波，先检查未经平滑的信号协方差秩，再取 3 个长度为 6 的子阵比较。平滑能改善特定模型的秩，但付出有效孔径和可处理源数的代价。非均匀阵、挡板方向响应或未知互耦不满足相同平移流形时，不能直接滑动矩阵下标冒充空间平滑。

固定函数先形成前向平均 `Rf`；`fb=True` 时再返回 `0.5*(Rf+flip(Rf).conj())`，`flip` 同时反转两轴，对应交换矩阵作用于两侧。它不另做自动源数选择，也不检查真实阵列是否满足这种对称性。2026-10-01 已实际提取调用这个未经修改的函数：三麦输入 `v=[2,0,2]`、`R=v vᴴ`、`l=2`，浮点前向与复数前后向两路都得到手算的 `diag(2,2)`。同样数值用整数矩阵输入时，原函数因平均中的浮点结果不能原地写回整数数组而抛类型错误；报告保留该失败，没有替上游自动转换。该确定性小矩阵调用不验证真实非理想阵列。[固定预处理源码](https://github.com/morriswmz/doatools.py/blob/9469db201e0418aef6b97583ef54b6fec2769502/doatools/estimation/preprocessing.py)。

原模型的条件可核对 [Shan–Wax–Kailath 1985 原文，§II～III，印刷页806～809](https://www.researchgate.net/profile/Mati-Wax/publication/3177856_On_spatial_smoothing_of_estimation_of_coherent_signals/links/56cacb9208ae5488f0d9b392/On-spatial-smoothing-of-estimation-of-coherent-signals.pdf)：相同阵元 ULA、不同且无混叠的方向、窄带源及白噪声等前提共同支持秩恢复。全相干 $q$ 源的前向构造要求足够多的子阵和足够长的子阵，不能把矩阵平均直接用于未知流形。这里于 2026-10-01 阅读原文相关定理，不扩写为任意房间保证。

## 定位算法和搜索加速

### 7. GCC-PHAT 与物理时延裁剪

对应 §4.2。本书 `doa.py::gcc_phat` 明确正时延约定、FFT 补零、物理 lag 截取和三点插值；ODAS 的 `src/signal/xcorr.c`、`src/system/freq2xcorr.c` 可作为块处理参考。补零到两段长度之和减一只保证未加权互相关的有限长度等价；PHAT 归一化改变频谱后，逆变换不再保证相同的有限支撑，FFT 长度仍可能改变相关值。PHAT 消除互谱幅度权重，但低能量频点的相位会变得不可靠，因此归一化下限与有效频点筛选都是算法的一部分。

建议给一个 0.5 ms 延迟的宽带源，随后改成单频正弦、两源和静音。分别报告主峰、次峰、峰背比和最大允许 lag。亚采样插值只细化相关峰形，不能凭空恢复窄带信号缺少的可辨识信息。先固定通道号，再固定 `t_1-t_2` 的方向，避免定位后又人工翻转符号。

CC、Roth、SCOT 与 PHAT 改变同一互谱的频率权重。普通 CC 保留互谱幅度；Roth 用指定一路自谱作分母，交换参考路会改变权重；SCOT 使用两路自谱乘积的平方根；PHAT 使用互谱模。SCOT 若直接用单帧未经平滑的周期图，自谱乘积的平方根会等于互谱模，不能由这一次数值相同推出两种统计估计总是等价。比较时需要相同的谱平均、下限和物理时延范围。现有教学入口只实现 PHAT，另外三项按正文原理和公式阅读，未用同名函数冒充已运行实现。[Knapp–Carter 1976 原论文](https://doi.org/10.1109/TASSP.1976.1162830)。

### 8. SRP-PHAT 与近场三维网格

对应 §4.3、§4.7 的“近场模型失配”。先读本书 `doa.py::srp_phat`，再读 pyroomacoustics 的 `doa/srp.py`，理解麦对证据如何按候选传播时延相加。这里的两份代码作为远场方向 SRP 参考；近场三维 SRP 则保留正文球面传播与角度—距离网格的原理索引，不能仅因外部 API 列出 `mode='near'` 和 `r` 就标成已取得该变体实现。

锁定版本 pyroomacoustics 0.10.0 的静态调用链有明确限制。`doa/srp.py` 构造函数把 `mode/r` 传给 `DOA.__init__`，但 `doa/doa.py:289` 创建 `ModeVector(self.L, self.fs, self.nfft, self.c, self.grid)` 时没有传 `mode`，因此该对象仍采用第 32 行的默认 `mode='far'`。同时，候选 `r` 仅保存到 `self.r`，第 234～280 行的 `GridCircle/GridSphere` 构造没有把它形成距离维；`srp.py:116` 的评分随后直接使用这个 `self.mode_vec`。[固定提交 DOA 构造与导向源码](https://github.com/LCAV/pyroomacoustics/blob/0dd39f2614b7fc44b2cc63dbe7d60f4641068890/pyroomacoustics/doa/doa.py)；[固定提交 SRP 评分源码](https://github.com/LCAV/pyroomacoustics/blob/0dd39f2614b7fc44b2cc63dbe7d60f4641068890/pyroomacoustics/doa/srp.py)。

据此，即使构造参数写为 `mode='near', r=np.array([2.0])`，也没有将近场选项传入实际相位表。2026-10-01 用原类构造器进一步运行核对：半径 `[0.2,0.5]` 在构造中抛 `ValueError`；单半径 `[0.2]` 可构造，但三个候选坐标的范数仍均为 1，选定频点的相位表与相同几何的 `far` 完全一致。这是构造与导向表检查，未运行完整近场评分。附录 B 实际运行的是本书远场 SRP 教学实现与上游房间 RIR，也不构成上游近场接口的声学数值验证。若要实现近场搜索，必须另行构造包含距离的候选位置，并使球面传播时延进入 SRP 评分；不能仅补传一个 `mode` 参数就宣称完整三维搜索已完成。

建议在真值落格点、落在格点之间、落在搜索区外三个条件下比较峰值。源在区外时，算法仍可能返回边界上的最大值，因此“找到峰”不等于位置可信。部署时缓存静态时延表、限制重复麦对、记录选峰间距，并在阵列几何或采样率变化后重建表。

**两个实现的分数不能直接混用。** 本书入口接收 `M×F×T` 复谱，逐帧归一化麦对互谱，再对帧、所选频点和无序麦对平均；返回的实数得分可以为负。固定 pyroomacoustics `SRP._process` 先按每通道频谱模做 PHAT，累积互谱后计算两倍麦对实部，并加上 `T*M*F` 的方向无关常数，最后除以 `T*F*P`（`P` 为麦对数）。只有所有参与项都高于各自下限、频点和帧相同、几何及相位约定对齐时，后者才等于本书分数的两倍再加 `M/P`。它们的最大值方向此时相同，绝对谱高和峰背比却不同。静音或低幅频点受不同下限处理影响，不能沿用该关系；尤其上游常数按维度加入，并非逐项实测自功率。[固定 SRP 计算入口](https://github.com/LCAV/pyroomacoustics/blob/0dd39f2614b7fc44b2cc63dbe7d60f4641068890/pyroomacoustics/doa/srp.py)。此处是逐式静态比较，未声称完成两包同录音的运行对照。

### 9. 分层 SRP-PHAT 与方向性麦对筛选

对应 §4.3 的工程扩展。ODAS 的 `src/module/mod_ssl.c` 组织定位流水，`src/signal/scan.c`、`src/signal/spatialindex.c` 和配置中的 `scans` 描述搜索层级；配置还包含相关插值倍率 `interpRate`。分层搜索先在粗球面上找候选，再搜索附近细方向，方向响应可用于排除缺少有效声学信息的麦对。

建议同一输入分别做全细网格和分层网格，比较漏峰率、最大角度误差、查表次数及内存。两源距离接近时，粗层可能合成一个峰，后面的细化无法恢复已经丢弃的区域。因此候选数量、邻域宽度和多个峰的保留规则应一起标定。依据为 [ODAS 官方源码](https://github.com/introlab/odas) 与 [Grondin–Michaud 的方法论文](https://arxiv.org/abs/1812.00115)。

固定 `mod_ssl_process` 的具体链是频谱相位归一化、选定麦对乘积、频谱插值、逆变换相关、逐层评分与选峰；查下一个候选前会重置前一候选附近的相关项。`pots` 中每项为三维方向分量和评分，评分还乘以 `interpRate`，既不是米制坐标，也不是后端稳定轨迹或概率。定位输出进入 `mod_sst` 后才有另一套追踪状态；不能用单帧候选数量当成当前活跃说话人数。[固定 ODAS 定位模块](https://github.com/introlab/odas/blob/bcb845434495e293df3d48f1203b7a86e1852449/src/module/mod_ssl.c)。工程中要连同通道映射、阵列方向响应、采样率、声速、插值倍率和层级一起保存配置；本次仅静态读码，未执行设备录音流水线或宣称实时时延。

### 10. SVD-PHAT 与多源扩展

这是 SRP 搜索加速的补充研究。固定几何与频点后，SRP 映射矩阵可先做截断 SVD；在线把 PHAT 观测投影到低维空间，再搜索候选方向。保留秩控制近似误差与每帧成本；多源扩展还需逐次投影已解释的分量。[单源原论文](https://arxiv.org/abs/1811.11785)、[多源原论文](https://sls.csail.mit.edu/publications/2019/Grondin_Interspeech-2019.PDF)。

最小实验应先对同一个 SRP 矩阵比较完整乘法与截断乘法，测量分数误差，再比较 DOA；不能只报告投影速度。高频、复杂几何、较密网格可能改变所需秩。本文尚未确认与这两篇论文唯一对应且许可明确的作者实现，因此此项保留原理索引，不以普通 SVD 库替代算法源码。

另一个可对照的作者工程是 *Steered Response Power for Sound Source Localization: A Tutorial Review* §7 的 X-SRP。它将互相关、候选空间、麦对选择和搜索过程拆成可替换模块，适合研究同一 SRP 目标下的工程取舍；不应由框架名称推断每个扩展都实现了原论文。[作者教程 §7](https://arxiv.org/html/2405.02991v2)；[固定代码入口](https://github.com/egrinstein/xsrp/tree/5876b760c0ead781c05d4f319ed23e302478ea2d/xsrp)。固定提交的 `pyproject.toml` 只声明 MIT 标识，根目录未见完整许可文本，因此本书保留官方索引和入口，未下载或执行，不能将其计作新增已取得源码。

#### SMP-PHAT：合并重复基线，而非截断矩阵

另一条降低 SRP 运算量的路线是合并麦对的 PHAT 互谱。SMP-PHAT（Steered response power by Merging Pairs with PHAse Transform）利用远场下相同基线的时延相同：基线向量相同的麦对先在频域求和，再共用一次逆变换和时延查表；向量相反时，先取相应互谱的共轭。只有长度相同且相互平行才满足这一条件，不能把所有平行麦对合并。[Grondin 等 2022 预印本，§3 算法2～3](https://arxiv.org/html/2203.14409v1)。

官方 `smpphat` 的阅读顺序为 `demo/ssl.c` → `src/system.c::scmphat_call` → `smp_construct/smp_call`；同文件的 `srp_construct/srp_call` 提供未合并对照，`src/signal.c` 定义阵列和方向数据。输入须保持录音通道与坐标次序一致，候选方向用三维单位向量，几何长度与声速采用相容单位。输出是候选方向及评分，不是经过校准的方向概率。[固定提交算法源码](https://github.com/FrancoisGrondin/smpphat/blob/6fd33e6eb3251078a4cd9793dde909e2500265cc/src/system.c)。

固定版本有三项复现条件：

1. `CMakeLists.txt` 要求 `pkg-config` 和 `fftw3f`，并硬编码 `USE_SIMD`、`-msse3`、`-ffast-math`。非 x86 环境需要另行适配并记录补丁，不能把作者的硬件测速直接移到另一平台。
2. `wav_construct` 直接读取固定头结构，只接受 16 位 PCM；它不是通用 RIFF 块解析器。运行前检查 WAV 头、通道数、采样率和数据区，不能仅凭文件扩展名判断相容。
3. `smp_construct` 的合并容差为 `1e-5`，论文算法文本给出 `1e-4`。实现把同一常数用于基线长度差与点积残差，改变坐标单位可能改变分组；应保存实际坐标单位、容差和分组结果。

本书于 2026-09-23 实际编译并调用上述固定提交，使用 Apple clang 21、arm64 与 FFTW 3.3.10 单精度静态库。编译直接使用未修改的 `system.c`、`signal.c`，不经过硬编码 x86 选项的上游 CMake；这只改变构建入口，不修补算法。两个处理对象按“创建、调用、复制结果、销毁”的顺序分别运行，因为上游析构函数会执行全局 `fftwf_cleanup`，不能让另一个对象的计划跨过该调用继续存活。[FFTW 计划生命周期说明](https://fftw.org/fftw3_doc/Using-Plans.html)。

输入为半径 0.032 m 的四麦菱形、16 kHz 采样率、声速 343 m/s、64 点原频谱与 4 倍插值，扫描 24 个相隔 15° 的水平单位向量。0° 指向 +y，正角转向 +x，真值为 45°。DC 为 1，正频率 bin1～31 为单位幅度 PHAT 互谱，原 Nyquist bin32 置零，保证实信号频谱端点约定成立。逆变换不归一化，分数不是概率。固定输入、完整逐方向结果和源码摘要见 [运行报告](../../ch04/reports/smpphat_reference.json)。

规则菱形的 6 个麦对合并为 4 组；将第四麦沿 y 移动 0.5 mm 后，分组变为 6 组。独立的有符号时延查表与直接离散傅里叶求和确认，规则几何下理论 SRP/SMP 分数最大差约为 $1.99\times10^{-13}$。但原 C 程序的最大差为 76.5567，SRP、SMP 相对正确基线的最大误差分别为 95.1649、171.7215。两者的峰方向碰巧都是 45°，不能据此判定等价检查通过。

问题位于 `system.c` 的两处查表索引：负的 `roundf` 结果先被转换成 `unsigned int`，随后才加中心偏移。该浮点值不在无符号类型的表示范围内，行为未定义；本机编译结果将负值转成 0。例如应取索引 3 的一项实际取了中心索引 9，规则案例共有 66 项 SRP 索引不符。使用程序实际索引再做独立傅里叶求和，SRP/SMP 分数误差仅约 $3.35\times10^{-5}$/$2.36\times10^{-5}$，因而将主要差异定位到查表而非变换约定。[Clang 浮点转换说明](https://clang.llvm.org/docs/UsersManual.html)。

扰动阵列的原 C 程序虽然给出 SRP 与 SMP 逐点相同的分数，两者相对正确基线仍有约 96.0504 的最大误差。这也说明两个程序互相吻合不能替代独立基线。当前报告明确记为 `failed_portability`，保留未修改的上游源码；不能把这一版本视为已经验证的 ARM 定位实现。

**隔离适配实验（2026-09-24 已运行）。** [适配脚本](../../ch04/examples/reproduce_smpphat_portable_overlay.py)先核对锁定工作树摘要，再只在临时目录复制源码，把 `system.c` 两处 `roundf` 的结果先转为有符号整数，保留原工作树不变。对上述规则菱形与 0.5 mm 扰动阵列，适配版的 6 个麦对分别合并为 4/6 组；两组的 SRP 与 SMP 峰都在第 3 个候选方向（45°），相对独立有符号查表和直接 DFT 的最大绝对误差分别不超过 $1.34\times10^{-5}$/$1.77\times10^{-5}$，两种 C 分数之间的最大差不超过 $3.10\times10^{-5}$。逐项数据、补丁表达式、原始与适配后摘要见[适配报告](../../ch04/reports/smpphat_portable_overlay.json)。这只验证指定的两组远场合成互谱，不把原版失败改写为通过，也不说明真实录音准确率、其他几何或运算速度。

实验使用合成远场互谱，不含真实录音、噪声、混响或运行速度评测。近场时，相同基线处于不同位置会产生不同距离差，合并不再有远场等价保证；无重复基线的阵列可能没有节省。后续速度测试仍需分别记录初始化、逐帧耗时、内存与浮点选项。

### 11. Bartlett 与 Capon 空间谱

对应 §4.5。本书 `bartlett_spectrum` 计算候选响应下的输出功率；`capon_spectrum` 通过协方差线性求解计算自适应谱。两者共享导向矢量，但归一化和权重不同。外部交叉入口为 Acoular `fbeamform.py` 的 `BeamformerBase`、`BeamformerCapon`，其声压成像口径不应直接混入本书的单位幅度谱。

建议在同一个解析协方差上检查双麦手算，再分别缩放输入功率和导向矢量。工业成像还要记录 CSM 对角去除与功率归一化；这些设置改变幅值解释。Capon 的窄峰不自动意味着更准确，少快拍或目标流形失配会产生错误抑制。

Capon 的原始出处是 *High-Resolution Frequency-Wavenumber Spectrum Analysis*（1969），原文印刷页 1410 的式(18)给出逆协方差谱，页 1412 讨论估计谱矩阵可逆所需的样本条件。[原文 PDF](https://epsc.wustl.edu/~ggeuler/reading/cam_noise_biblio/capon_1969-ieee-high-resolution_frequency-wavenumber_spectrum_analysis.pdf)。本书的样本数、加载和导向归一化须另行明确；加载后可求解并不能证明少快拍协方差已经具有充分统计精度。Bartlett/Capon 都是候选方向上的功率类量，MUSIC 则是正交性伪谱，不能把三者的峰高当成同单位功率比较。

### 12. MUSIC 与频点归一化 NormMUSIC

对应 §4.6。pyroomacoustics `doa/music.py` 和 `doa/normmusic.py` 共享子空间计算；频点归一化用于改变宽带伪谱合并时各频点的影响。它们输出扫描谱和峰方向，仍需要源数、噪声子空间维度和峰间距。

建议令一个频点信号很强、另一个频点很弱，比较归一化前后方向；再将弱频点换成纯噪声。按峰值归一化可能让低可靠频点获得过大权重，因此有效频带选择仍不可省略。MUSIC 的理想零分母应采用明示数值处理，不能把无穷大误认为一个有统计校准的置信度。

固定 `MUSIC._compute_correlation_matricesvec` 把 `M×F×T` 输入转置为帧、频率、通道，形成每频二阶矩并对帧平均，未先减均值；`_subspace_decomposition` 利用 `eigh` 的升序结果按已给定的 `num_src` 切分。`NormMUSIC` 开启逐频最大谱值归一化，再在频率轴合并，不是对协方差白化，也不自动选择可靠频点。空间有色噪声仍会改变噪声子空间；应使用合适的噪声模型或另作预白化。[固定 MUSIC 源码](https://github.com/LCAV/pyroomacoustics/blob/0dd39f2614b7fc44b2cc63dbe7d60f4641068890/pyroomacoustics/doa/music.py)。这些接口本次仅静态核查。

HARK 的工业接口另有明确的数据和时间契约。[官方 LocalizeMUSIC，3.4.0 Rev9509，§6.2.14.1～4](https://www.hark.jp/document/hark-document-en/subsec-LocalizeMUSIC.html)要求 `M×(NFFT/2+1)` 复数频谱，传递函数通道列表须与输入物理通道一致。SEVD 分支忽略噪声协方差，GEVD/GSVD 分支利用它做噪声处理，不能仅换分解名字而保持任意噪声模型。默认 `WINDOW_TYPE=FUTURE` 使用当前及后续帧，`PAST` 才改变窗口方向；因此默认配置不能称作严格因果。输出 MUSIC 伪谱也不是校准声压功率。该文档角度 0° 指机器人正前、正角向左，须转换为本文坐标。这里于 2026-10-01 阅读官方接口，未运行 HARK 节点或设备录音。

### 13. root-MUSIC

对应 §4.6 和总对比。doatools `estimation/music.py::RootMUSIC1D` 针对 ULA，把噪声子空间投影矩阵的对角线和变成多项式系数，从单位圆附近的根恢复相位，再转换为方向。输入是 ULA 协方差、源数、波长与麦间距；默认半波长间距必须显式检查。[作者接口与 Barabell/Rao–Hari 原文定位](https://morriswmz.github.io/doatools.py/references/doatools.estimation.music.html)。

建议先用单源无噪协方差比较 MUSIC 扫描峰和 root-MUSIC，再加入相干双源、几何扰动和空间混叠。对共轭倒数根需按算法规则选取，不可把所有近圆根都计作声源；成功找到指定数量的根也不保证方向正确。免网格只消除了网格量化，未消除特征分解和多项式数值误差。

固定 `RootMUSIC1D.estimate` 中还使用 `np.complex_` 创建系数数组，该别名已从 NumPy 2 移除；因此下面 ESPRIT 在 NumPy 2.5.3 中运行成功进入计算，不代表同一包的 root-MUSIC 相容。2026-10-01 实调原方法确实在此抛 `AttributeError`。随后仅给提取方法的局部 `np` 门面增加 `complex_→complex128` 别名，不改方法体、上游目录或共享 NumPy；六元半波距 ULA、`a=[1,j,−1,−j,1,j]`、`R=a aᴴ+0.1I` 得到约 29.99999971°，与解析 30° 相差小于 $10^{-6}$ 度。门面成功不等于原发行包已兼容 NumPy 2，也不覆盖 E04-23 三麦理想重根的数值稳定性。此前缺少 SciPy 的导入失败另记在末节。[固定 root-MUSIC 源码](https://github.com/morriswmz/doatools.py/blob/9469db201e0418aef6b97583ef54b6fec2769502/doatools/estimation/music.py)。

### 14. LS-ESPRIT、TLS-ESPRIT 与几何约定

对应 §4.6。本书 `esprit_ula` 实现最小二乘旋转矩阵；外部阅读入口为 doatools `estimation/esprit.py`。LS 只把一侧当作拟合目标，TLS 同时考虑两侧子空间块误差；两者都依赖平移不变子阵，不能对任意几何直接使用。

建议用固定波长改变间距，检查旋转特征值相位到角度的反解；交换两个子阵后，应先解释相位符号变化再转换角度。TLS 的矩阵划分和截断秩需与源码接口一致。存在近场曲率、位置误差或相干源时，低残差不保证真实方向正确。

**固定版默认行加权的实调反例。** doatools 提交 `9469db201e0418aef6b97583ef54b6fec2769502` 的 `Esprit1D.estimate` 实际默认 `formulation='ls'`，docstring 却写 TLS；复现时必须显式给出。其第 120～125 行的 `Es1=Es[:-displacement,:]` 与 `Es2=Es[displacement:,:]` 是重叠视图，随后两次原地乘行权会相互影响。期望计算是给两块分别左乘同一对角权重矩阵；重叠内存导致其中一些原行被重复缩放，破坏本应保留的移位关系。[固定 ESPRIT 源码](https://github.com/morriswmz/doatools.py/blob/9469db201e0418aef6b97583ef54b6fec2769502/doatools/estimation/esprit.py)。

本书于 2026-09-28 用既有独立临时环境实际调用未经修改的上游。环境为 Python 3.13.12、NumPy 2.5.3、SciPy 1.18.1、macOS arm64；缺少可选稀疏求解包的导入警告完整保留，不影响这四次 ESPRIT 调用。输入为 8 元半波距 ULA，波长 1 m、间距 0.5 m，已知源数 2、移位 1；真角为 −5°、10°，导向为 `exp(+j*pi*m*sin(theta))`。直接构造总体协方差 `R=A Aᴴ+0.1I`，每源功率为 1、每麦白噪声功率 0.1，总源功率对噪声比为 20，即约 13.01 dB。这不是有限快拍或音频实验，没有随机抽样、采样率或 STFT 参数。

| 求解 | 原版 `row_weights='default'` 方向（°） | 原版 `row_weights='none'` 方向（°） | 独立安全复制后同样加权（°） |
|---|---|---|---|
| LS | −6.508543，11.527825 | −5，10 | −5，10 |
| TLS | −6.478045，11.496900 | −5，10 | −5，10 |

默认行权为 `[1,√2,√3,2,√3,√2,1]`。四次上游调用都返回 `resolved=True`，但两次默认加权的最大方向误差分别约 1.527825°、1.496900°；该标志不能判定方向正确。未加权两路与独立复制两路的误差均低于 `10⁻⁸` 度。独立 LS 使用 `lstsq`，TLS 使用拼接子阵的 SVD；再用解析导向列替代特征向量作为另一组参考，结果也恢复真角。解析基下的移位特征值直接为 `exp(j*pi*sin(theta))`，无需用上游输出定义答案。

[执行脚本](../../ch04/examples/reproduce_doatools_esprit.py)在调用前核对提交、跟踪文件状态和所调用关键源码摘要，禁止未跟踪 Python 源混入；[报告](../../ch04/reports/doatools_esprit_reference.json)绑定脚本、配置与源码摘要，保留完整协方差、原版四路结果及两组独立参考。普通[离线测试](../../../../tests/test_codes_doatools_esprit_reference.py)只核报告和 NumPy 参考，以标量三角和、±30° 四分之一周期解析基及换基不变性独立校验，不导入外部包或联网。历史运行使用 `/private/tmp/room-pra-venv/bin/python`；2026-10-01 核查该临时环境已不存在。当前重跑需另备固定源码和含 SciPy 的环境，例如本轮实际使用的 `/private/tmp/masp-appb-pra-venv/bin/python`，但不能把新环境运行写回成上述历史执行条件。

[Roy–Kailath 1989 原文，§IV-A～D，印刷页989～991](https://alumni.media.mit.edu/~aggelos/papers/roykailath89.pdf)对成对传感器的一致响应、相同平移和共同采样作出约定，并给出共同子空间基下的相似关系；任意旋转布局不能代替平移子阵。原文 TLS 同时拟合两块误差，不能由其名称推出所有失配下优于 LS。该原文相关页于 2026-10-01 实读。

本书保留原版失败，未修改上游或将独立参考包装成修复发行版。`row_weights='none'` 只是在本例中通过的明确调用配置；它与安全复制参考都不构成真实房间准确率、TLS 优于 LS、不同源数或其他几何已验证的证据。

### 15. CSSM 宽带聚焦

对应 [§4.6 的宽带聚焦](../../../../chapters/04_doa-estimation.md#sec-u-08146bdaaf)。固定 pyroomacoustics 的 `doa/cssm.py` 先为各频带产生候选峰，选参考频点，构造聚焦矩阵并循环聚合协方差。源码入口依次为 `_process`、`_coherent_sum`、继承的子空间分解。[Wang–Kaveh 原文](https://doi.org/10.1109/TASSP.1985.1164667)。

建议固定一组双源频域快拍，改变初值、参考频点和迭代数，报告方向误差与聚焦矩阵条件数。被剔除的频点必须与协方差、权重和候选峰同步筛选。固定 0.10.0 版本的这个关联已经用原 `_coherent_sum` 方法体提取调用核对，见末节。人工模拟首频点剔除后，剩余频点为 `[20,30]`、三份原协方差对角为 `[4,1,1]`、`[9,1,1]`、`[16,1,1]`；导向取相同值使聚焦为单位阵。原方法返回 `diag(13,2,2)`，独立保留频点求和应为 `diag(25,2,2)`，Frobenius 差为 12。未运行原完整 `_process`，也未自动构造触发剔除的声学输入。

聚焦误差与噪声变换也要分开核对。即使导向被精确对齐，非酉聚焦仍把白噪声变成 `σ² T Tᴴ`；[E04-19](../../../../chapters/04_doa-estimation.md#e04-19)给出两麦反例。相干源在单频为秩一，跨频源比例改变且已正确聚焦时，合并矩阵可以恢复秩；[E04-22](../../../../chapters/04_doa-estimation.md#e04-22)以已知酉变换展示这一条件。它们是本书可控教学链，不是固定 CSSM 已在真实录音中成功的证据。

### 16. WAVES 加权信号子空间

对应 [§4.6 的宽带聚焦](../../../../chapters/04_doa-estimation.md#sec-u-08146bdaaf)。`doa/waves.py` 将各频带聚焦后的信号子空间按特征值相关权重拼接，再对拼接矩阵做 SVD。它与 CSSM 的区别在聚合对象和权重，不是简单把 CSSM 改名；初始化、频点剔除和噪声特征值估计都影响结果。[Di Claudio–Parisi 原文](https://doi.org/10.1109/78.950774)。

建议让部分频点只包含噪声，检查权重是否减弱该频点影响；再设强弱源功率差，观察弱源是否被压掉。阅读 `_construct_waves_matrix` 时逐项追踪 `freq_bins[j]` 与 `C_hat[j]` 的对应。2026-10-01 同样提取调用原 `_construct_waves_matrix`，给上述对角矩阵显式特征子空间适配器。原首行是 `[3/√5,8/√10]`，保留频点的独立期望则为 `[8/√10,15/√17]`；这验证辅助方法的错位，不声称原整条 `_process` 已运行。

WAVES 方法类不能笼统写成总要初始 DOA。[TOPS 作者上传正式原文 §II-C，印刷页1980](https://www.researchgate.net/publication/3319689_TOPS_new_DOA_estimator_for_wideband_signals)明确区分 RSS 聚焦与可不需初始 DOA 的 BICSSM 聚焦，同时提示其视场与阵列条件。这里的固定 PRA 实现采用候选方向聚焦；原 WAVES 与这一路实现的条件须分开。当前未取得可完整读取的 WAVES 原论文正文，不用上述源码替代原文定义。

### 17. TOPS 投影子空间正交性检验

对应 [§4.6 的宽带聚焦](../../../../chapters/04_doa-estimation.md#sec-u-08146bdaaf)。`doa/tops.py` 为每个方向组合跨频子空间正交性矩阵，以最小奇异值构造谱；它不依赖 CSSM 的初始方向聚焦，但仍依赖正确的源数和多个频点。[Yoon–Kaplan–McClellan 原文](https://doi.org/10.1109/TSP.2006.872581)。

建议至少用两个不连续频点，并移动参考频点的位置，检查输出是否对频点排列保持相同物理含义。排列频点不会改变声场，但若代码错误地把“频点列表下标”当成“FFT bin 编号”，结果会改变。2026-10-01 已运行原完整 TOPS 类：三只非共线麦、16 kHz、256 点 FFT、343 m/s、bins `[10,20,30]`、参考 bin 20，真方向 30°；8 个精确正交的复数快拍使每频协方差为 `p a aᴴ+0.01I`，三频功率为 1、4、2。原实现峰值为 49°，依据已知真导向独立计算的跨频正交投影范数在 30° 为零（数值小于 $10^{-12}$）；全 360 个方向的差异保存在报告。这里没有随机抽样、PCM、房间或论文成功率实验。

原文 §III-A/B，印刷页1981～1982 的论证还需要满秩源协方差、跨频相对阵元增益一致和相应无混叠流形；它不保证任意三维阵列、任意相干源都可辨。[作者上传全文及公式定位](https://www.researchgate.net/publication/3319689_TOPS_new_DOA_estimator_for_wideband_signals)。固定实现的频率索引问题不能反推论文方法错误，也不能把本书独立投影参考当作已发布修复包。

### 18. FRIDA 连续角度恢复

对应 §4.7。pyroomacoustics 的 `doa/frida.py` 组织协方差可见度与恢复过程，`doa/tools_fri_doa_plane.py` 实现映射与多频重建。先理解去除自功率后的互谱观测，再读 `max_four`、`max_ini`、`max_iter`、`G_iter` 与 `signal_type`，不能只调整一个“精度”参数。[原论文](https://doi.org/10.1109/ICASSP.2017.7952744)。

建议在非均匀平面阵上放两个不落网格的独立源，固定随机种子比较重建残差和角度误差。多个随机初值增加计算成本，也可能得到不同局部结果。原论文专库 `figure_doa_synthetic.py`、`figure_doa_separation.py`、`figure_doa_experiment.py` 分别定位仿真、分辨率和录音实验；旧环境与音频获取单列，不能直接执行过时安装脚本。[原实验仓库](https://github.com/LCAV/FRIDA)。

这里“任意阵列”必须加上模型范围：作者预印本 §2.1.1 的麦坐标属于二维平面，采用远场、互不相关声源的可见度模型；其式(1)～(2)把互谱写成各方向功率贡献之和。全三维球面恢复在该文 §4 列为扩展方向，固定 pyroomacoustics 的 `FRIDA._process` 也对 `dim==3` 明确报错。[作者预印本（本段节号按此稿）](https://dokmanic.ece.illinois.edu/assets/pdf/Pan2016tv.pdf)；[固定 FRIDA 接口](https://github.com/LCAV/pyroomacoustics/blob/0dd39f2614b7fc44b2cc63dbe7d60f4641068890/pyroomacoustics/doa/frida.py)。平面内布局可以不规则，不等于任意三维阵列、相干源或近场都适用。

`signal_type='visibility'` 从选中帧的协方差取非对角元素，帧筛选受 `stft_noise_floor` 与 `stft_noise_margin` 控制；可选低秩清理还使用给定源数。`signal_type='raw'` 则先按固定表达式校正逐帧相位旋转再平均原始复谱，输入统计意义不同。论文利用更多麦对可见度讨论的源数能力，不能直接移到 raw 分支；相干源交叉项也不会因去除自功率而自动消失。固定代码的 `n_rot` docstring 写 10、构造默认却为 1，工程记录应显式保存所有实际参数。当前未运行这两种 FRIDA 接口，也未复现论文分辨率或实录数字。

### 19. 组稀疏定位与协方差稀疏匹配

对应 §4.7。doatools `estimation/sparse.py` 的 `GroupSparseEstimator` 直接拟合多快拍，按候选方向共享稀疏支持；`SparseCovarianceMatching` 则拟合向量化协方差和非负源功率，要求源间不相关。前者的复幅度与后者的功率不可使用相同误差解释。[作者模型和求解接口](https://morriswmz.github.io/doatools.py/references/doatools.estimation.sparse.html)。

建议先用落格点双源，再把源移到相邻格点中间，比较支持泄漏和正则参数敏感性。求解器返回可行解、峰数达到要求，只表示数值过程完成。工程记录应包括字典归一化、正则目标、求解器版本、终止容差、迭代次数和残差；可行性约束太严时应保留失败状态。

固定 `GroupSparseEstimator.estimate(Y,k,l)` 接收 `M×T` 快拍；`SparseCovarianceMatching.estimate(R,k,l,...)` 接收 `M×M` 协方差，将实部与虚部分开堆叠，并在噪声未知时加入 `vec(I)` 对应的标量白噪声功率列。两者的选峰接口都仍需 `k`，不能因稀疏优化而称整套接口自动免源数。后者的 `l` 在惩罚目标、限制解的 l1 范数、限制残差 l2 范数三种 formulation 中含义不同，不能搬用同一数值。可选求解器未安装，本次未运行这两路。[固定稀疏实现](https://github.com/morriswmz/doatools.py/blob/9469db201e0418aef6b97583ef54b6fec2769502/doatools/estimation/sparse.py)。

L1-SVD 是降维后再做稀疏拟合：固定 `preprocessing.py::l1_svd(Y,k)` 只返回前 `k` 列左奇异向量乘对应奇异值，即 `M×k` 的压缩观测，本身不输出方向或执行 l1 优化。将 `GroupSparseEstimator` 的 `n_snapshots` 设为这个保留维数，再把压缩观测送入 `estimate`，才连接组稀疏求解和选峰；这里“压缩列数”不等于原始统计快拍数。弱源可能因截断而丢失，保留维数和正则参数都需记录。[Malioutov–Cetin–Willsky 2005 原文](https://doi.org/10.1109/TSP.2005.850882)；[固定降维入口](https://github.com/morriswmz/doatools.py/blob/9469db201e0418aef6b97583ef54b6fec2769502/doatools/estimation/preprocessing.py)。两段外部参考源码已经取得，本书尚未运行这一组合定位链。

原子范数对应连续频率或方向参数上的稀疏约束，避免直接固定一个细角度字典，但仍受阵列结构、模型阶数和求解精度约束。[Tang 等作者稿 §2、式(2.2)](https://people.eecs.berkeley.edu/~brecht/papers/12.Tang.EtAl.ctscs.pdf)与 [Yang–Xie 2015](https://doi.org/10.1109/TSP.2015.2420541)讨论的是线谱估计；与 ULA DOA 的联系需要通过空间相位到 `sin(theta)` 的映射建立，不能泛称对所有麦阵通用的网格外算法。Tang 等的恢复保证还包含分量间隔与采样条件，不能把“连续参数”解释成无限分辨率。本项保留模型索引，未确认并取得与这两篇工作唯一对应的完整官方实现，也未运行半正定规划。

#### 多快拍、多频稀疏贝叶斯学习

稀疏贝叶斯学习（Sparse Bayesian Learning，SBL）在候选方向上估计源功率超参数，与前面的组稀疏罚项不是同一个求解器。作者 `gerstoft/SBL` 提供 `SBL_MF_Python/sbl.py::SBL`，MATLAB 对应 `SBL_MF_matlab/SBL_v4.m`；演示从 `Beamforming_demo.m` 开始读，参数定义见 `SBLSet.m`。[Gerstoft 等 2016 原论文](https://doi.org/10.1109/LSP.2016.2598550)、[固定 Python 核心](https://github.com/gerstoft/SBL/blob/d4bba35e9b60907d3024473ba5a41046450baae0/SBL_MF_Python/sbl.py)。

Python 核心输入字典 `A` 为 `M × G × F`，观测 `Y` 为 `M × L × F`；`G` 是候选方向数，`L` 是快拍数。与本文通用的 `M × F × T` 相比，最后两轴要交换。字典承载传播相位约定，不能只转换数组形状而不核对角度零点和复指数符号。返回值为候选功率 `gamma` 和迭代报告，不直接是角度。

沿源码阅读时，依次检查样本二阶矩、功率初始化、相对功率剪枝、逐频求解、功率更新和噪声更新。`options.Nsource` 用于选峰，噪声估计的分母为 `M−Nsource`；该实现需要给定源数且满足相应维数条件，不能称为“自动免源数”。每频噪声模型为标量乘单位阵，空间有色噪声不属于这个模型。

核心 Python 文件依赖 NumPy；完整演示与 MATLAB 环境分别检查。作者 README 说明 Python 版本自 2020 年夏后未使用，旧示例中的外部数据链接也不属于已取得的代码。工业使用还要记录有效频带、共同方向支持的时间跨度、字典归一化、剪枝阈值、停止误差和最大迭代数；运动中的声源不一定满足同一批快拍共享支持的假设。

本书已用 [SBL 复现脚本](../../ch04/examples/reproduce_sbl_reference.py)实际调用上述固定 Python 核心，没有复制或改写上游 GPL 算法。2026-09-22 的执行环境为 Python 3.13.12、NumPy 2.5.3、macOS arm64；[逐例 JSON 报告](../../ch04/reports/sbl_reference.json)保存上游提交、核心与调用脚本的 SHA-256、环境、完整功率谱、实际迭代误差序列和协方差。运行前检查上游 `HEAD`、受 Git 跟踪文件的修改状态及核心文件摘要，不把未跟踪文件也称为已经核查。

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
.venv/bin/python codes/chapters/ch04/examples/reproduce_sbl_reference.py --output codes/chapters/ch04/reports/sbl_reference.json
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

新增[同输入波束对照](../../ch05/beamformer_common_input_demo.py)把四麦 DSB、MVDR、相对加载为 1 的 MVDR 和 LCMV 放在同一 2 kHz、同一解析干扰加噪声协方差下。设计目标 0°、真实目标 10°、干扰 40°，同时报告真实目标幅度增益、干扰响应、WNG 和输出 SINR；结果表及独立 DSB 几何级数核对见正文 §5.10 与 `tests/test_codes_beamformer_common_input.py`。这只比较一个精确模型，没有快拍随机性、混响或语音质量；后续可在固定随机种子下增加多次协方差估计，并分别记录样本数、失败次数和离散程度。

建议使用 §5.3 的双麦解析协方差，增加增益误差与相位误差，按加载强度画出失真和降噪的关系。工业策略需要无效协方差检测、上一组权重保留、权重平滑和输出限幅。求解成功不能替代目标保持检验；过度加载时接近固定波束是可以解释的设计结果。

**经典来源的实验范围。** Kumatani、McDonough 与 Raj 的 *Microphone Array Processing for Distant Speech Recognition: From Close-Talking Microphones to Far-Field Sensors* 是 IEEE Signal Processing Magazine 29(6), 127–140，2012 年 11 月的综述，CMU 页面托管的是论文副本。[正式 DOI](https://doi.org/10.1109/MSP.2012.2205285)、[作者版](https://course.ece.cmu.edu/~ece792/handouts/KumataniEtAl12.pdf)。其中 pp.137–139 比较 32 麦、半径 4.2 cm 的刚性球阵与 64 麦、孔径 1.26 m 的线阵；TIMIT 预录语音经扬声器在真实房间重放，采样率 44.1 kHz、混响时间约 525 ms。表 4、5 分别对应 28°、68° 的位置和四轮识别处理。这不是圆阵实验，也不能用不同几何的结果给所有 DSB/超指向算法排名。

**加窗的优化对象。** [Dolph 1946](https://doi.org/10.1109/JRPROC.1946.225956)研究对称、等间距、同相激励的 broadside 阵列，控制等波纹旁瓣与首零点主瓣宽度；不能把这个结论直接用于任意几何的最小 HPBW。[SciPy 1.18 的 `chebwin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.chebwin.html)按 dB 旁瓣衰减生成对称窗，返回的最大窗值为 1，不保证窗之和为 1。用作空间加权时还须根据导向向量归一化目标响应。它和 §26 的球谐 Dolph 扩展是两个接口。

#### SOF 的固定 FIR 波束与离线设计诊断

SOF TDFB 将离线设计的滤波器系数用于运行时 FIR 波束形成；设计脚本、固件配置和设备上实际执行应分别核对。固定源码的 `src/audio/tdfb/tune/sof_bf_design.m` 使用 `sinc` 构造扩散噪声相干矩阵，再计算方向性与白噪声增益。这里 MATLAB/Octave 的标准 `sinc(x)` 是 `sin(pi*x)/(pi*x)`；本章三维扩散场相干度需要的参数是 `2*f*d/c`，不是脚本第 123 行的 `2*pi*f*d/c`。

**固定系数与方向控制是不同环节。** 2026-10-01 重读同一锁定提交 `b6c6a05d52536313fe8e8752b1c4e069b1cc4002`，其中已经有方向估计与 IPC 控制，不是仅在后来版本才出现。[`tdfb.c` 第 511～523 行](https://github.com/thesofproject/sof/blob/b6c6a05d52536313fe8e8752b1c4e069b1cc4002/src/audio/tdfb/tdfb.c#L511)把请求的方位映射到已有角度滤波器组；处理流程在 FIR 输出后调用方向估计，并在变化时通知控制端。[`tdfb_direction.c` 第 539～571 行](https://github.com/thesofproject/sof/blob/b6c6a05d52536313fe8e8752b1c4e069b1cc4002/src/audio/tdfb/tdfb_direction.c#L539)计算通道时差和方向，更新的是 `az_value_estimate` 与变化标记；[`tdfb_ipc4.c` 第 124～151 行](https://github.com/thesofproject/sof/blob/b6c6a05d52536313fe8e8752b1c4e069b1cc4002/src/audio/tdfb/tdfb_ipc4.c#L124)则把方位控制写入 `az_value`、触发系数组选取，并可开关方向更新。估计、通知、控制写入和选择固定系数共同组成可能的跟随链路，不能把“固定 FIR”解释成“没有方向控制”。

上述估计函数没有直接将估计值回写成控制方位，也没有根据实时 SCM 重新求解 MVDR 权重。[当前架构说明](https://thesofproject.github.io/latest/developer_guides/firmware/tdfb.html)与固定提交的具体行为须分别看待；仅凭网页对 autonomous DOA 或动态 IPC 的描述，不能声称本书已经运行完整自动跟随。这里是逐文件 BSD-3-Clause 源码的静态核验，没有编译固件、发出设备控制或测试方向跟随延迟。

同一脚本的 DI 循环计算 `denom2`，WNG 循环第 271 行另算 `denom=wᴴw`，但第 272 行使用的仍是 `denom2`，因此读到了上一循环最后一个频点的量。这是固定版本的设计诊断问题，不能拿它输出的 WNG 曲线验证本章定义；也不等于所有已经部署的 SOF 滤波器都失效。[独立静态诊断脚本](../../ch05/examples/audit_sof_tdfb_design.py)和[逐项报告](../../ch05/reports/sof_tdfb_design_audit.json)记录固定版本、源码摘要及独立数值对照。报告中的两频点权重是本书为隔离分母问题构造的例子，不是实际 SOF 滤波器；该检查没有执行 MATLAB/Octave 或固件。

[官方设计说明](https://thesofproject.github.io/latest/developer_guides/algorithms/tdfb/time_domain_fixed_beamformer.html)中的旧目录、频点数、窗与加载值也不能移植解释固定提交：页面使用 `tools/tune/tdfb`，说明为 512 个频点与 Kaiser 窗；所核源码使用 `src/audio/tdfb/tune`、1024 点 FFT（513 个非负频点）与 Hann 窗。加载量的 dB 值及 `10^(mu_db/20)` 转换须以实际脚本为准，不能改写成未经核对的功率转换。目标设备还须核对麦坐标、系数格式、采样率、增益与饱和处理；本书未用这次静态诊断声称固件运行或实时性能通过。

### 21. LCMV、Frost 与 GSC

对应 §5.5～5.6。LCMV 直接求满足多个线性约束的最小功率解；Frost 在时域抽头空间投影更新以保持约束；GSC 用固定支路和阻塞后的自适应支路实现同类约束结构。本书教学包给 LCMV 闭式解、GSC 阻塞基线及 `gsc.py::ScalarGSCNLMS` 的单参考状态更新。后者保留复数系数、先输出再更新、逐样本冻结和跨块状态；输入固定支路及参考支路由调用者提供，不估计方向或阻塞矩阵。音频例采用已知活动区间控制更新，只检查目标泄漏与冻结的作用，不能称为实际 VAD 或工业 GSC。持续子带自适应可另读 BTK2.0，不能把其实现直接登记为正文时域 Frost。

最小实验先检查约束矩阵独立性和阻塞残差，再将目标方向偏移少量，测量目标泄漏到参考支路后被抵消的程度。工业中冻结条件、步长、滤波长度、双讲/活动控制和状态复位必须与滤波器一起审查。约束保持不等于目标真实方向仍在约束集合中。

**求解器返回有限数不等于约束可行。** 本次[固定 pb_bss 原方法审计](#beamformer-upstream-audit)实际调用 `get_lcmv_vector`：两条约束都取 `[1,1]`，却要求 `Cᴴw=[1,0]`，本来就无解；原辅助函数遇到奇异约束矩阵改用最小二乘，返回 `[.25,.25]`，实际 `Cᴴw=[.5,.5]`，约束残差范数为 `1/sqrt(2)`。这是不相容输入下的返回行为，不是 LCMV 理论错误。与正文 E05-04 的复响应共轭检查一起，调用者应检查约束独立性、可行性和实际残差。另一个同名族接口 `get_lcmv_vector_souden` 在固定源码中立即抛 `NotImplementedError`，本次也保留了该真实异常；不能按函数名认为它已经可用。

BTK2.0 是 Kumatani、McDonough 等作者的空间信号处理工具箱。固定 `btk20` 源码先读 `btk20_src/unit_test/test_online_beamforming.py::online_beamforming`，再读 `btk20_src/lib/pybeamformer.py::SubbandGSCLMSBeamformer` 与 `SubbandGSCRLSBeamformer`；C++ 对应入口包括 `btk20_src/beamformer/beamformer.{h,cc}::SubbandGSCRLS`。这些类包含跨帧自适应状态，不只是构造一个阻塞矩阵。[固定 Python 更新器](https://github.com/kkumatani/distant_speech_recognition/blob/feff19ec8bcb770f6530fe280dc3ccafc2f5984a/btk20_src/lib/pybeamformer.py)。

演示输入是逐通道音频、阵列位置、带时间标记的目标方向和分析/合成滤波器组系数。示例声速常数为 `343740.0`，与毫米制坐标配套；本书米制坐标必须转换。滤波器组的子带数、抽取率、原型滤波器长度决定时频处理与等待，不能直接沿用本书 Hann STFT 的帧数和延迟。

构建文件默认目标 Python 2.7，要求 SWIG 3、GSL、NumPy 和 libsndfile，CUDA 9 为可选；部分 Python 算法还使用 SciPy 或 pygsl。旧环境说明不能当作现代 Python 或目标设备的相容性保证。滤波器系数演示通过 `pickle` 载入，只应使用自己生成或可信来源的文件，不能为方便运行而加载不可信二进制对象。[固定构建条件](https://github.com/kkumatani/distant_speech_recognition/blob/feff19ec8bcb770f6530fe280dc3ccafc2f5984a/btk20_src/CMakeLists.txt)、[固定在线演示](https://github.com/kkumatani/distant_speech_recognition/blob/feff19ec8bcb770f6530fe280dc3ccafc2f5984a/btk20_src/unit_test/test_online_beamforming.py)。

建议先对同一混合录音保留固定支路输出，再开启 LMS/RLS 自适应，分别检查目标参考的增益、干扰残差与权重范数；第二组只将控制方向偏移 2°，比较持续更新与明确冻结区间。合成输入须保存各源分量、共同延迟和增益，指标对齐后计算。上述 BTK 构建和数值实验未在本书执行，不能称为已经完成产品验收。

**Frost 与 BTK 的具体差别。** [Frost 原刊](https://doi.org/10.1109/PROC.1972.8817) §II、式(16)给带线性约束的最小功率闭式解；§III、式(19)～(22)在宽带时域抽头上给出保持约束的投影更新。约束修正项还处理累积数值偏差。BTK 的 `SubbandGSCLMSBeamformer` 则在子带内更新阻塞支路，默认 `beta=0.97` 平滑能量、`gamma=0.01` 控制归一化步长，并带泄漏、能量下限和活动权重范数上限；前 128 帧输出固定支路，之后才使用自适应支路输出，但此前满足能量条件时已经更新权重。每 4096 帧减小步长也是该实现的控制策略，不是 GSC 的定义。参数依赖滤波器组、输入幅度与声学变化，不能作为通用工业默认。

#### BTK 的 Zelinski 与 McCowan 后滤

已有固定 BTK 源码中的 `btk20_src/postfilter/postfilter.cc`、`.h` 和 `.i` 给出了后滤实现；根 MIT 许可及作者信息随源码保留。[固定头文件与算法文献](https://github.com/kkumatani/distant_speech_recognition/blob/feff19ec8bcb770f6530fe280dc3ccafc2f5984a/btk20_src/postfilter/postfilter.h)。`ZelinskiPostFilter` 使用对齐后的各通道自谱与互谱，按共同遗忘因子递推；REAL 分支取互谱实部，ABS 分支使用复数平均的模。两者不是同一个有限样本估计器。默认 `alpha=0.6`、`type=2`，所得增益限制在 `[1e-4,1]`；这些是源码参数，不是本书语音质量验证的结果。

[McCowan 与 Bourlard 作者报告](https://publications.idiap.ch/attachments/reports/2001/rr01-40.pdf)印刷 p.4 式(12)用平均互谱实部估计目标 PSD，并除以输入自谱均值；分母没有自动变成波束输出 PSD。p.6 式(22)引入噪声相干度的实部，式(23)再对麦对平均。前提包括目标对齐、目标与噪声不相关以及各麦噪声功率相等；相干度趋近 1 时分母病态。报告索引为 RR-40-2001，所读 PDF 封面日期为 2002 年 12 月，正式期刊版为 [IEEE TSAP 11(6), 709–716，2003](https://doi.org/10.1109/TSA.2003.818212)，不要混用版本页码。

BTK 的 `McCowanPostFilter::estimate_average_clean_PSD` 有停用的 `#if 0` 分支；实际 `#else` 分支先作复数相干修正，再按类型取实部或模，并非直接逐式复制上述实部公式。默认相干阈值为 0.99，扩散模型用 GSL 归一化 `sinc`；半频带移位路径明确不支持。增益上下限不能处理所有非法数值：若自谱和互谱同时为零，源码中的 `0/0` 可以先产生 NaN，随后普通大小比较不会把它夹回有限值。这是静态边界检查；本书未编译调用该 C++ 后滤器，不能声称已经测得静音输出。

正文的 Zelinski 输入噪声差式在一致、非负的样本权重下等于麦对差分能量均值，有限样本本身不会把它变负。可能为负的是互谱得到的目标估计，或模型失配下的 McCowan 相干修正目标 PSD。例如两路自谱均为 1、互谱为 0、假定相干度为 0.5，原报告式(22)给出目标估计 −1。工程恢复应分别处理 PSD 模型失效、零分母和增益下限，不能把三种情形统一解释成“有限样本噪声 PSD 为负”。

### 22. 最坏情形稳健波束与 WNG 约束

对应 §5.4.1。稳健设计需要先定义导向误差集，例如有界范数误差，再约束误差集中目标响应的最低值；WNG 约束则限制白噪声放大。二者与“给矩阵加一个经验小数”有不同的建模目的，不能把所有稳健方法都称为加载 MVDR。

建议固定同一真实误差序列比较 DSB、加载 MVDR 和带明示误差集的优化解，报告失配集外的失败。参数来自阵列标定和环境变化，不应只按训练集效果选择。本书此项保留原理与 §5.4.1 的原始引用；未确认特定作者代码时，不把通用凸优化器单独列作完整波束实现。

[Vorobyov、Gershman 与 Luo 的作者版](https://users.aalto.fi/~vorobys1/RobBeamformer.pdf) §III.A、式(18)～(29)把球形导向不确定集写成二阶锥问题；误差半径若使集合包含零向量，任何非零最低响应都不可满足。原文 p.316 脚注还指出该幅度约束不直接控制宽带目标的相位，不能据窄带响应下界保证语音波形无失真。§II 的特征空间方法需要选择目标加干扰的主子空间维数；弱目标被错分到噪声子空间时，投影并不能保护它。

正文的协方差重构沿 [Gu 与 Leshem 2012](https://doi.org/10.1109/TSP.2012.2194289)理解：非目标角区的谱积分依赖正确导向流形、角区和统计模型，不能把任意样本协方差去对角化都称为该方法。前后向空间平滑则沿第 4 章的平移子阵条件与固定 `doatools` 入口；它不是任意几何的通用协方差修复。上述三类方法仍登记为相应原理/已有外部入口，未新增作者优化程序的运行或性能结论。

### 23. Acoustic Rake 与时域多路径约束

这是 §5.4～5.5、§2.4 的联合扩展。pyroomacoustics `beamforming.py` 中 `rake_mvdr_filters`、`rake_distortionless_filters`、`rake_perceptual_filters` 将早期反射路径纳入滤波器设计。它们需要源位置、反射模型、滤波器长度与允许延迟；优化目标可选择保持响应或容许特定早期时间结构。[作者原论文](https://arxiv.org/abs/1407.5514)。

建议先用单反射的已知 RIR 比较只保留直达与加入反射，再扰动反射延迟和增益。利用已知反射不意味着实际房间中的所有混响都应保留。原始 `TimeDomainAcousticRakeReceiver` 专库采用 CC BY-NC-SA 4.0，和当前 MIT 的 pyroomacoustics 不能混同；本文不将该原始实验库作为商业可自由使用源码。[原实验许可](https://github.com/LCAV/TimeDomainAcousticRakeReceiver)。

三个函数的约束不同。固定 v0.10.0 的 `rake_mvdr_filters` 用时域总协方差、约束一个指定延迟处的响应；`rake_distortionless_filters` 对完整期望时域响应施加约束；`rake_perceptual_filters` 允许指定的早反射区间变化。后者默认 `d_relax=0.035` s，不能把 docstring 的“30 ms”当成实际默认值；包含干扰时 `K_nq=R_n` 后的原地加法还会修改调用者给的噪声矩阵，比较多个方法时应各传独立副本。

**原版方法实际失败。** [诊断脚本](../../ch05/examples/audit_beamformer_reference.py)于 2026-09-28 在既有 Python 3.13.12、NumPy 2.5.3、SciPy 1.18.1、pyroomacoustics 0.10.0 环境调用了未经修改的 `rake_distortionless_filters`。双麦坐标为 `(0,0)`、`(0.05,0)` m，目标 `(1,1)` m、干扰 `(−1,1)` m；采样率 8 kHz、FFT 64、每麦滤长 32、噪声协方差 `I64`、允许延迟 1 ms。这里只构造直接路径对象，没有随机音频或房间重放。

函数第 1373 行的 `/2` 产生浮点数 `L`，第 1375 行用它切片时报 `TypeError`，未得到滤波器。相邻方法使用整除，并不能使这个方法通过。[报告](../../ch05/reports/beamformer_reference_audit.json)保存完整输入、异常位置、安装文件与固定上游文件相同的摘要。此项是实际原包方法调用；同报告的 pb_bss 项是函数提取调用，执行范围不同。第 4 章房间 SRP 已运行也不能替代这次 Rake 方法检查。

### 24. 球谐变换与理论径向补偿

对应 §5.8。sfa 的 `process.py::spatFT` 把球面采样投影到球谐系数，`gen.py::radial_filter` 设计径向补偿，`sph.py` 定义球谐与模态相关计算。应先区分实球谐/复球谐、系数排列、归一化、方位角和余纬角，再设计开放球或刚性球的径向滤波。[官方 API](https://appliedacousticschalmers.github.io/sound_field_analysis-py/reference.html)。

最小实验以已知平面波系数合成球面麦信号再反演，检查每阶误差。增加阶数时同时测白噪声输出；低频高阶模态的逆滤波会放大自噪声。有限麦数、球面采样误差与空间混叠限制可恢复阶数，不能仅依据期望指向性无限增加阶数。

在固定 sfa 版本中，`array_extrapolation(..., normalize=True)` 返回的模态量已经乘了 `4*pi*i**n`；`bn_open_omni` 才只返回球 Bessel 函数。把两者都记作同一个未定义的 `b_n` 再取逆，会重复或遗漏归一化。其 `bn_rigid_omni` 使用第二类球 Hankel 函数，必须连同时间相位约定阅读，不能直接替换另一约定下的第一类函数。`sph_harm` 使用方位角与余纬角，不是方位角与仰角。

`radial_filter` 的默认最大模态放大为 40 dB，采用反正切软限制，并非简单硬裁切。这个固定版本还使用旧 API：`gen.py` 读取 `np.NAN/PINF/NINF`，`sph.py` 调用 `scipy.special.sph_harm`。本书在已有 NumPy 2.5.3、SciPy 1.18.1 环境实际检查到这四个属性均不存在，尚未配置兼容版本并运行球阵处理。源码获取成功不能被写成这些接口已经在当前环境可运行；本书也没有修改上游来掩盖兼容性限制。

### 25. 实测球阵编码与软限制径向滤波

对应 §3.4、§5.8。Politis 库的 `arraySHTfiltersTheory_softLim.m`、`arraySHTfiltersTheory_regLS.m` 和 `arraySHTfiltersMeas_regLS.m` 分别提供软限制、理论正则最小二乘和实测响应拟合。`evaluateSHTfilters.m`、`sphArrayNoise.m` 用于检查编码误差和噪声代价。先读 `TEST_SCRIPTS.m` 的调用参数，再对照各函数输入单位。[作者库及原始文献列表](https://github.com/polarch/Spherical-Array-Processing)。

建议把校准方向分成设计集与留出方向，在相同频带上测球谐重建误差、WNG 和目标方向响应。实测拟合可以包含实际外壳与麦差异，但不能据同一校准数据上的低误差声称新方向泛化。MATLAB 中还需加入作者的 Array-Response-Simulator 和 Spherical-Harmonic-Transform，不能只下载一个库后承诺全部示例可运行。

两个直接依赖已按固定提交取得最小源码选集。`Spherical-Harmonic-Transform` 的 [`getSH.m`](https://github.com/polarch/Spherical-Harmonic-Transform/blob/30ec1454ab654a0432eafd168da7bee72ecca3fc/getSH.m)返回“方向数 × 球谐数”矩阵，角度为弧度制 `[azi, inclination]`，按阶递增、同阶 `m=−n…n` 排列；复基包含 Condon–Shortley 相位，实基作相应相位抵消。此选集只包含该自包含基函数与许可/说明，不包括所有求积网格和 Gaunt 系数数据。

`Array-Response-Simulator` 的 [`sphModalCoeffs.m`](https://github.com/polarch/Array-Response-Simulator/blob/1ebfb28296736c52691c63e1aa336a7bd0d6216b/sphModalCoeffs.m)及六个 Bessel/Hankel 辅助函数支持这里的开放球、刚性球模态计算，输出已经含 `4*pi*i^n`，刚性球使用第二类 Hankel。理论径向逆函数又除以 `4*pi`，所以其输入输出归一化必须沿调用链检查。代码对零频和 NaN 的处理是实现分支，不能据此断言任意高阶结果可靠。两个选集均为 BSD-3-Clause，保留原作者与根许可；未运行 MATLAB/Octave 或整个作者测试集。

该模态函数 `directional` 分支还有头注反写：实际方向图是 `dirCoeff+(1-dirCoeff)*cos(theta)`，故 1 对应全指向、0 对应偶极；头注一处却将二者颠倒。本节采用的开放球/刚性球路径不依赖这个参数，仍保留原版说明这个边界，不把头注原样推广为接口事实。

### 26. 球谐固定波束、MVDR/LCMV 与子空间定位

对应 §5.8。Politis 库提供 `beamWeightsDolphChebyshev2Spherical.m`、`sphMVDR.m`、`sphLCMV.m`、`sphMUSIC.m`、`sphESPRIT.m`。固定波束主要取决于期望方向图和有效阶数；自适应方法仍需球谐域统计量，定位方法仍需源数和信号/噪声模型。

建议在同一个已编码声场上分别改变方向、阶数和模态噪声，比较旋转前后方向图形状以及自适应权值响应。编码误差是共同前提，不能将球谐坐标变换理解成已经消除物理麦克风误差。新增径向滤波时也必须更新噪声协方差，而不是照搬阵元域的单位白噪声假设。

`beamWeightsDolphChebyshev2Spherical` 按 Koretz 与 Rafaely 2009 的球阵扩展返回轴对称的 `N+1` 个系数；旁瓣参数是线性幅度比，不是 dB，主瓣宽度参数是角度。它与 `beamWeightsDifferential2Spherical` 一样生成理想球谐方向图系数，后者并不实现真实小间距麦之间的差分与低频补偿。`sphMVDR` 输入维度为 `(N+1)^2` 方阵、每个方向单独求一组权重；`sphLCMV` 用多个约束共同求解。接口本身不保证采样/径向补偿后的噪声矩阵可逆，加载及约束独立性须由调用者检查。

### 27. C/C++ 球阵处理与空间功率图

对应 §5.8 的实现。SAF 的阅读顺序是 `framework/modules/saf_sh/saf_sh.h`，然后 `examples/src/array2sh/array2sh.c`、`beamformer/beamformer.c` 与 `powermap/powermap.c`。前者给数学接口，三个例子分别连接阵元到 SH、SH 到虚拟麦、统计量到方向图。[官方模块与构建说明](https://github.com/leomccormack/Spatial_Audio_Framework)。

建议先在无设备情况下以确定的平面波测试系数方向，再建立目标 CPU 构建，测每帧最慢时间和内存分配。CBLAS/LAPACK 后端、FFT 库、SIMD 和编译浮点选项都影响结果；不能由 C 实现推断它必然满足某个实时截止期。启用 GPLv2 可选追踪模块会改变许可义务，应在构建清单中明确。

固定 `beamformer.c` 的处理对象接收球谐时域通道，先把 FuMa/ACN 次序与 SN3D/N3D 等归一化转换到内部 ACN、N3D。示例固定波束可选 cardioid、hypercardioid 或 max-energy-vector，转向后对旧/新权重输出作线性交叉淡化；它不是一个自动估计噪声协方差的 MVDR。处理函数还要求输入块长等于 `BEAMFORMER_FRAME_SIZE`，不足通道补零不代表缺失模态可以无误恢复。读者应区分 `array2sh` 的阵元编码、该固定虚拟麦输出和 `powermap` 的方向功率图；本书尚未在目标 CPU 实测这三个示例的最坏帧时间。

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

[E09-08](../../../../chapters/09_source-tracking.md#sec-9-6)补充状态所属时刻、发布时刻与消费时刻的区别：从 1.00 s 状态预测至 1.08 s，同时传播均值与协方差；不是把旧原始观测当作当前观测更新。对乱序融合，可读 [Stone Soup 1.9.1 固定延迟示例](https://stonesoup.readthedocs.io/en/v1.9.1/auto_examples/oosm/KalmanFilterOOSMExample.html)，区分等待重排、忽略旧数据和按到达顺序直接处理的后果。本书未实现通用乱序融合器。 [E09-23](../../../../chapters/09_source-tracking.md#e09-23)进一步用相关迟到观测区分按原时刻回放、明确丢弃和错误地当作当前独立观测三种协议；原接口调用成功不能替代时刻合同检查。

本轮在既有隔离环境实际调用了固定 FilterPy，补充了此前主环境缺少 SciPy、没有运行该类的历史记录。`ExtendedKalmanFilter.predict_update()` 与 `predict(); update()` 在非线性观测下不能直接互换：前者在状态预测前计算观测雅可比，而预测后的状态才交给观测函数；后者在预测后计算雅可比。前者也没有 `residual` 参数，角度更新宜显式检查残差入口。完整输入、函数调用顺序与数值见[§54 的上游诊断](#tracking-upstream-audit)，没有修改上游。

UKF 的 `sigma_points.py::MerweScaledSigmaPoints` 分开给均值权重 `Wm` 和协方差权重 `Wc`；负权重不是概率为负，不能用普通粒子权重归一化替代。`UKF.py` 提供 `x_mean_fn`、`z_mean_fn`、`residual_x`、`residual_z`，角度量应协调使用。固定版本在 `predict()` 加入过程协方差后重新生成 sigma 点，不能照搬旧版本“更新点不含 Q”的批评。本轮运行的是一维无迹变换小例，未运行完整非线性声学 UKF。原始方法见 Julier、Uhlmann，*Unscented Filtering and Nonlinear Estimation*，Proceedings of the IEEE 92(3), 2004, pp.401–422，[正式 DOI](https://doi.org/10.1109/JPROC.2003.823141)。

观测噪声也应检查跨帧或跨接口相关性。[E09-20](../../../../chapters/09_source-tracking.md#e09-20)的两条相关观测例说明：同一物理信息被重复提供时，逐条独立更新会过度缩小方差；它不由观测数量自动成为两份独立证据。

过程噪声的名字相近，定义却不同。固定 FilterPy 的 `common/discretization.py` 将 `Q_discrete_white_noise(dim=2,dt=.5,var=2)` 解释为每步恒定加速度样本经 `[h²/2,h]` 的外积，输出 `[[.03125,.125],[.125,.5]]`；`Q_continuous_white_noise(dim=2,dt=.5,spectral_density=2)` 则对连续白加速度积分，输出 `[[1/12,.25],[.25,1]]`。前者参数是加速度方差，后者是谱密度，不能只因数值都为 2 就互换。2026-10-01 的[合同审计工具](../../ch09/examples/audit_upstream_tracking_contracts.py)执行未修改原定义，单块 `block_diag` 由保留单矩阵的 NumPy 适配器提供；没有运行 SciPy 通用分块函数。独立期望分别来自外积与连续积分，绝对容差 $10^{-15}$。正文 [E09-04](../../../../chapters/09_source-tracking.md#sec-9-6)与 [E09-18](../../../../chapters/09_source-tracking.md#e09-18)说明模型与时间间隔的区别。

固定增益可另读 `filterpy/gh/gh_filter.py` 的 `GHFilter` 与 `GHKFilter`。其中位置修正为 `g*residual`、速度修正为 `h*residual/dt`；二阶实现的加速度修正含 `2*k/dt**2`，所以应先说明 γ 的定义再与 α-β-γ 教材公式对照。滑动平均是线性平滑，中值和直方图众数是非线性汇聚；三者不共同构成一个线性低通滤波器。它们都没有自动维护关联、人数或状态协方差，本轮未把这些接口当成完整追踪系统实跑。

### 32. SIR 粒子滤波与重采样

对应 §9.2.4。先读本书 `CircularParticleFilter` 和 `systematic_resample`，再读 ODAS `src/system/particle2particle.c` 与 `src/module/mod_sst.c` 中的持续状态管理。权重应由一致的似然和杂波模型形成，极小概率在对数域计算；重采样只是分配粒子数量，不创造观测信息。

建议固定种子比较有无重采样的有效粒子数，再用双峰对称后验检查圆周均值是否有定义。单个均值可能落在两个真实峰之间，因此多峰后验应保留峰或混合表示。工业中记录粒子数、退化阈值、扩散噪声、出生位置范围和静默保持时间，不把所有声学峰强制解释成一个人。

[E09-07](../../../../chapters/09_source-tracking.md#sec-9-6)给出两次独立观测的权重递推：不重采样时，旧权重必须进入新后验。[跨章时间练习](../cross_chapter/tracking_time_exercises.py)的两粒子例子从 `[0.5,0.5]` 经 `[0.622459,0.377541]` 变为 `[0.731059,0.268941]`；每次覆盖为似然会抹掉历史证据。一般重要性提议还需密度比，见 [Stone Soup 1.9.1 权重推导](https://stonesoup.readthedocs.io/en/v1.9.1/auto_tutorials/04_ParticleFilter.html)。该确定性例子检验概率递推，不评价真实定位性能。正文 [E09-12](../../../../chapters/09_source-tracking.md#e09-12)还区分有限对数权重与显示为零的浮点权重：显示下溢不能被误当成数学上没有支持。

APF 的“辅助”是利用下一次观测构造粒子选择或提议的辅助权重，再补上重要性修正；它不自动把离群观测识别成杂波。Pitt、Shephard 的 1999 原论文 *Filtering via Simulation: Auxiliary Particle Filters* 应与门控、重尾似然分别阅读；本章不把 APF 作为已经运行的声学鲁棒插件。正文的 SIR 数字例也不能替代 APF 实现验证。

ODAS 的 `particle2particle.c` 以位置、速度和权重维护粒子，按模式的 α、β 与帧间隔产生扩散；更新按关联后验混合观测似然，在有效粒子数低于 `Nmin*nParticles` 时重采样。它并不是把每一个峰独立送入本书单源圆周滤波器。完整生命周期、Kalman 分支与配置消费见 §55。

### 33. GNN/PDA 与 JPDA

对应 §9.3。Stone Soup `dataassociator/neighbour.py` 给硬关联组件，`dataassociator/probability.py::JPDA` 根据满足一对一约束的联合事件计算边缘关联概率；`docs/tutorials/08_JPDATutorial.py` 展示如何连接预测器、假设器与更新器。[固定提交的 JPDA 教程源码](https://github.com/dstl/Stone-Soup/blob/8d1edeb07ef8505ed065cbef435cfb5e517d9bdc/docs/tutorials/08_JPDATutorial.py)。

建议沿用正文两轨三观测例，比较最近邻与 JPDA 在杂波靠近某条轨迹时的分配，再设置两人交叉。门控先减少不可行配对，检测概率和杂波密度再决定事件权重；几何最近不等于语音身份正确。角度观测的杂波密度单位是每度或每弧度，不能直接填位置空间每平方米的数值。

本书已运行的最小交叉反例见[两轨交叉、缺测和限速脚本](../../ch09/tracking_crossing_dropout_demo.py)与第 9 章 §9.3。输入是无随机性的五帧角度数列，间隔人为设成 1 s；两轨在第 2 s 都预测 $40^\circ$，观测依次为 B 的 $39^\circ$、A 的 $41^\circ$。两种一对一分配的角残差平方和都为 $2\ \mathrm{deg}^2$，指定的并列规则使该帧两条身份都错配；未滤波的观测集合与真值集合相同，集合 OSPA 为 0。硬关联更新后的 A/B 角误差均为 $1.68^\circ$，使用真值标签作诊断关联时均为 $0.32^\circ$。这些数值分别测量身份、集合位置和滤波位置，不能互换。

第 3 s A 无观测时只预测，角度方差由约 $0.68$ 增至 $1.80\ \mathrm{deg}^2$；另一个纯缺测的控制缩例从 $20^\circ$ 起、目标速度 $10^\circ/\mathrm s$，当波束每秒最多转 $5^\circ$ 时两秒后滞后 $10^\circ$。脚本未实现 JPDA、轨迹出生或身份确认，真值标签不参与盲关联。下一轮应在真实帧率、标注说话人身份和可复现角度观测上分别统计 ID 错配、位置误差、缺测持续时间与控制滞后，并给出重复序列的离散程度。

原始依据是 Fortmann、Bar-Shalom、Scheffe，*Sonar Tracking of Multiple Targets Using Joint Probabilistic Data Association*，IEEE Journal of Oceanic Engineering OE-8(3), July 1983, pp.173–184。[正式 DOI](https://doi.org/10.1109/JOE.1983.1145560)；[作者上传全文](https://www.researchgate.net/publication/3231807_Sonar_tracking_of_multiple_targets_using_joint_probabilistic_data_association)的 §II 式(2.11)–(2.13)将组合新息与关联不确定性的协方差项分开。它讨论已建立轨迹的目标导向关联，不能据此省掉声学系统的出生确认。§III 为简化假定门内概率为 1；真实有限门控需要一致处理漏检项与门内概率。

固定 Stone Soup 的 `hypothesiser/probability.py::PDAHypothesiser` 默认检测概率 0.85、门内概率 0.95；漏检假设权重起点是 `1-prob_detect*prob_gate`，不是一律 `1-prob_detect`。提供杂波空间密度时，观测权重除以该密度；未提供时则用有效门体积与门内观测数估计。`include_all=True` 要求显式提供杂波密度。门控以平方 Mahalanobis 距离与卡方分位数比较，输入维度改变时阈值也改变。这些是该固定通用库的默认值，不是麦克风产品推荐参数。假设器仍是静态核对，未运行完整包链。2026-10-01 单独提取原 `JPDA.isvalid`，用带明确观测身份的最小假设容器检查两轨、三观测与漏检的 16 种组合：三个重复使用同一真实观测的组合被拒，剩下 13 个合法事件；两轨同时漏检合法。该方法只检查一对一约束，不计算事件权重或后验。原 `enumerate_JPDA_hypotheses` 的注释提到额外概率门控，但当前循环直接保留假设器给出的每项，不能把注释当作又执行了一遍门控。实际门控应追到 `PDAHypothesiser.hypothesise`；容器替身没有执行此路径。结果与[合同审计](../../ch09/reports/upstream_tracking_contracts.json)分栏保存。

### 34. GM-PHD 与高斯混合缩减

对应 §9.3。Stone Soup `updater/pointprocess.py::PHDUpdater`、`hypothesiser/gaussianmixture.py`、`mixturereducer/gaussianmixture.py::GaussianMixtureReducer` 分别执行强度更新、生成观测假设和合并/剪枝。阅读必须包含出生强度，不能只读更新公式。[Vo–Ma 2006 作者稿](https://www.ee.cuhk.edu.hk/~wkma/publications/gmphd_2col.pdf) §II-C 式(14)把强度积分定义为期望人数，假设 A.1～A.3 与式(15)～(16)分别给出独立点目标、Poisson 杂波及 Poisson 预测近似下的递推；§III-A 才增加线性高斯条件。本轮实际读取这份作者稿，不把剪枝接口的结果当作原递推的定理。[官方 GM-PHD 教程](https://stonesoup.readthedocs.io/en/v1.9.1/auto_tutorials/filters/GMPHDTutorial.html)。

建议先复算正文两网格例，再以无观测帧检查总强度如何因存活与漏检改变；最后新增一个源，检查出生模型能否覆盖其方向。高斯权重和表示期望目标数，不要求归一为 1；直接丢弃低权分量会改变总强度，但具体库可能另行补偿，必须报告实际输入、输出质量。PHD 不保留完整身份，给每个高斯分量贴标签也不等于实现了 GLMB。

固定 `GaussianMixtureReducer` 的实现需要特别区分：`prune()` 把低权分量的总权重均摊给幸存分量，`truncate()` 也把截掉的质量均摊给保留分量；若全部分量被剪掉则返回空列表。`merge_components()` 先按原权重做均值与协方差矩匹配，再把合并权重大于 1 的值截为 1。这些步骤不能统称为“直接删除”，也不能假定均保持 PHD 的期望人数。

本轮没有安装 Stone Soup 缺少的 `ordered_set` 依赖；仅提取上述未修改原函数，用明确的最小状态容器运行。权重 `[0.1,0.4,0.5]`、剪枝阈值 0.2 得到 `[0.45,0.55]`；两个权重 0.8、0.7 合并后却为 1，原总质量是 1.5。该反例是固定版本缩减接口行为，不是 PHD 理论禁止期望人数大于 1。详见 §54；`PHDUpdater`、真实状态类和完整合并调度均未运行。

### 35. MHT、CPHD、LMB/GLMB 与检测前追踪

对应 §9.3 的进阶分类。MHT 保留多条关联历史，CPHD 还传播目标数分布，LMB/GLMB 显式保留标签，检测前追踪输入未经过硬阈值的弱信号证据。它们解决的缺口不同，不能作为同一个“高级追踪器”接口随意替换。

Stone Soup 固定版本实际包含滑窗多帧分配形式的 MHT 参考。阅读顺序是 `docs/examples/dataassociation/mht_example.py` → `stonesoup/hypothesiser/mfa.py::MFAHypothesiser` → `stonesoup/dataassociator/mfa/__init__.py::MFADataAssociator` → 同目录 `_step.py`。假设器给每个分量延续观测索引历史与权重，关联器优化多帧分配后执行 N-scan 剪枝；这比仅在基类注释中提到方法名称多了实际计算。[固定 MHT 示例](https://github.com/dstl/Stone-Soup/blob/8d1edeb07ef8505ed065cbef435cfb5e517d9bdc/docs/examples/dataassociation/mht_example.py)。

该 MIT 参考实现需要额外安装 OR-Tools；`_step.py` 的最大迭代数为 10，相对原始—对偶间隙门限为 0.02，代码将代价差除以当前最佳原始代价 `bestPrimalCost`，并非直接比较未归一化的差。示例使用长度为 3 的滑窗、二维位置/速度状态与方位/距离观测，预置三个目标且没有出生或消亡。它说明如何推迟关联决定，不是已经完成的声学多说话人系统，也不保证所有输入下得到精确全局最优。声学适配仍需角度环绕、每弧度杂波密度、静默期漏检和轨迹管理。[固定分配求解器](https://github.com/dstl/Stone-Soup/blob/8d1edeb07ef8505ed065cbef435cfb5e517d9bdc/stonesoup/dataassociator/mfa/_step.py)。

最小实验建议从同目录 `MFA_example.py` 的两目标交叉场景开始，保持观测集合不变，比较滑窗 1 与 3 的分支数量、延迟决策、身份交换和计算量，再插入连续缺测。出生和长静默应作为额外实验单独报告，不能由固定目标示例推断已经支持。这里仅完成源码检查，未运行 OR-Tools 实验。

CPHD、LMB/GLMB 另有 [Ba Tuong Vo 的作者 MATLAB 工具包](https://ba-tuong.vo-au.com/codes.html)。2026-09-28 已按其 `readme.txt` 第 5 条“academic/research purposes only”取得 62 个源码和说明文件，以归档锁 `vo-rfs-tracking-updated` 管理；它不是宽松开源许可，也不据此允许公开再分发。取得固定归档不要求作者一定提供 Git 提交。归档 SHA、逐文件范围、第三方说明和未运行项目见 §56。

目标存在与说话活动应另设状态。[E09-22](../../../../chapters/09_source-tracking.md#e09-22)在 Bernoulli 存在模型中显式计算空观测的 Bayes 更新；改变与活动条件匹配的检测概率，会改变空观测意味着多大反证。该最小例不是把 PHD 总质量解释为单条轨迹的存在概率。

检测前追踪仍保留原理索引。对其他家族，也不能从 Stone Soup 的基类文字推断其实现了 CPHD，或从 PHD 分量上的临时标签推断其实现了 GLMB。两人交叉、出生、长静默和强反射应分别评价身份切换、漏检、虚警、人数和成本；输入定位峰不符合点目标观测模型时，还要调整似然与杂波模型。

### 36. OSPA、身份连续性与波束控制接口

对应 §9.3～9.4。OSPA 在同一坐标和截断距离下同时惩罚位置与目标数误差；身份连续性应另外评价。将 DOA 转换为波束控制时还需预测到播放或采集消费时刻，记录 `measurement_time`、`publish_time`、`track_id`、协方差与有效期。

建议先复算正文 30°/70° 对单真值 32° 的 OSPA，再交换两条轨迹 ID：位置指标可能完全不变，身份连续性已经出错。控制器按最大角速度和时间间隔限速，在过期观测、长静默或身份不确定时降低更新可信度。轨迹 ID 不等于永久说话人身份；单凭声学位置不能维持跨房间身份。

GOSPA 补充的是 OSPA 按集合基数归一后不便分解的误差解释。Rahmathullah、García-Fernández、Svensson 的[作者原文 v7](https://arxiv.org/pdf/1601.05585v7) Definition 1 要求阶数至少为 1、正截断距离，且参数 α 在 `(0,2]`；§II.B Proposition 1 在 α=2 时给出定位、漏检与虚警代价的分解。对真值 `{32°}` 和估计 `{30°,70°}`，取阶数 1、截断 10°、α=2，定位代价 2°，一个虚警代价 5°，GOSPA 为 7°；同条件 OSPA 为 6°。这是本书小例，两种指标的量值不可直接互判优劣。

原论文的 arXiv 页面还指向[作者维护的 MATLAB 实现](https://github.com/abusajana/GOSPA)。截至 2026-09-29，该仓库根目录未见 `LICENSE`，其 README 说明 `assign2D.m` 取自另一项目；这两部分的再分发授权均未在本书核实。因此它仅作原理与实现对照的来源索引，本书没有自动下载、复制、再分发或运行这份作者代码。下段固定 MIT 许可的 Stone Soup 接口和本书上面的手算小例是另外两条实现与核对路径，不能把它们的许可或运行状态移给作者仓库。

固定 `metricgenerator/ospametric.py::GOSPAMetric` 把 α 固定为 2，默认距离为普通 Euclidean；方位角必须提供适合环绕的距离，不能在 ±180° 附近机械相减。输出 `distance` 取阶数根，而 `localisation`、`missed`、`false` 保留取根前的代价，因此阶数 2 时后三者是平方单位。该版本还有 `_SwitchingLoss`，但 `switching_penalty` 默认为 0；不能沿用旧版本“没有切换项”的说法，也不能把默认单帧分数称为身份连续性验收。本轮未运行该完整指标对象。

### 37. TDOA 非线性最小二乘与可观测性

对应 §4.4、§4.7 的“近场模型失配”。TDOA 几何定位将每条观测与候选位置产生的距离差比较，按观测协方差加权，使用雅可比求局部位置更新。输入除了 TDOA，还需要明确参考麦和共享参考引入的误差相关性；输出应包含残差、局部条件性和解是否位于搜索域。

最小检查先用正文三麦近场几何复算距离差，再用有限差分独立检查雅可比。把源逐步移远，距离方向的敏感度会减小；算法收敛到一个距离不意味着距离已可观测。工业中至少保留多个初值或粗网格初始化，识别镜像多解与边界解。本文未将一般优化器索引成完整声学定位实现，具体模型与本书 §4.4 的原始论文一致。

Chan–Ho 的 1994 方法把距离差定位整理为两阶段加权代数估计；Taylor/Foy 路线则围绕当前位置线性化距离差，再迭代更新。这是初始化与求解策略不同，不能把一般 `least_squares` 调用自动命名为 Chan 方法。[Chan–Ho 原文](https://doi.org/10.1109/78.301830)；[Foy 原文](https://doi.org/10.1109/TAES.1976.308294)。当前两项按正文原理阅读，尚未确认并取得唯一对应的作者定位实现。使用共享参考麦的 TDOA 误差往往相关，权重不能随意简化为每对独立的同方差；异常相关峰、声速错误和时间同步偏差还需要先在观测层处理。

### 38. 随机源、确定源与非相关源 CRB

对应 §4.8。doatools `performance/crb.py` 的 `crb_sto_farfield_1d`、`crb_det_farfield_1d`、`crb_stouc_farfield_1d` 对应不同观测统计模型。CRB 是在参数模型、噪声和正则条件下的估计方差下界，不是从输入录音直接产生方向的算法，也不是混响房间的实测误差预测器。[作者源码](https://github.com/morriswmz/doatools.py/blob/9469db201e0418aef6b97583ef54b6fec2769502/doatools/performance/crb.py)。

建议同一几何上比较源功率、快拍数和角度分离的变化，检查方差单位是弧度平方还是角度平方，再与同模型重复仿真比较。明显低于所选 CRB 的结果可能来自口径不符、偏差估计器、真值泄漏或方差/均方误差混写，应先查实验模型，不能直接宣称突破统计下界。

### 39. 差分协同阵与虚拟协方差重建

对应[第 3 章](../../../../chapters/03_array-geometry.md)。本节同时检查几何、二阶统计量与标定接口：阵元位置给出传播模型，协同阵重排已有的统计量，标定则检验模型与设备是否一致。2026-10-01 重新核对本章原始论文与固定 doatools 源码，并执行下述原方法提取对照；MATLAB 系统、完整 doatools 包、DOA 估计器和设备测量仍未运行。其他项目的固定来源与核实日期分别保留在对应段落中。

#### 39.1 从阵列参数到坐标与连续差集

固定 doatools 的 [`model/arrays.py`](https://github.com/morriswmz/doatools.py/blob/9469db201e0418aef6b97583ef54b6fec2769502/doatools/model/arrays.py)提供 `NestedArray`、`CoPrimeArray` 和若干预设几何。它们生成坐标，不自动搜索最优布置。`d0` 是基础网格的物理间距，不能在不同频点偷偷改成该频点的半波长，否则比较的已经是不同硬件。

| 调用与参数 | 以 `d0` 为单位的阵元索引 | 阅读时要确认的条件 |
|---|---|---|
| `NestedArray(3,3,d0)` | `{0,1,2,3,7,11}` | 前三点来自 `0…n1−1`，后三点来自 `(n1+1)k−1`；与正文六麦例相符 |
| `CoPrimeArray(3,4,d0,mode='m')` | `{0,3,6,9,4,8}` | 这是图 10 的六麦基本互质阵；库返回顺序不是从小到大，必须同步保留通道次序 |
| `CoPrimeArray(3,4,d0)` | `{0,3,6,9,4,8,12,16,20}` | 默认 `mode='2m'`，是扩展的九麦配置，不能拿它替代图中的六麦结果 |

这些构造已在固定提交 `9469db201e0418aef6b97583ef54b6fec2769502` 的原定义上提取调用。统一取 `d0=0.02` m，嵌套阵、基本互质阵、默认扩展互质阵分别得到 23、17、35 个不同有符号滞后，中心连续长度分别为 23、13、29，单侧虚拟矩阵维数为 12、7、15。审计用独立的有序麦对双循环检查每个滞后与重数，没有把默认九麦配置当作六麦比较对象。doatools 是维护者的独立实现，其 MIT 许可与实现身份见[来源锁表](../SOURCES.lock.json)；它不是 Pal–Vaidyanathan 的原作者代码。

[`model/coarray.py::WeightFunction1D`](https://github.com/morriswmz/doatools.py/blob/9469db201e0418aef6b97583ef54b6fec2769502/doatools/model/coarray.py)先保存每个有符号差分滞后对应的协方差元素索引，再记录重复次数。对六麦嵌套阵，36 个有序麦对分布在 −11～11 的 23 个滞后上；零滞后对应 6 个对角元素。重复元素共享物理通道，不能按 36 次独立测量解释。

`get_central_ula_size()` 从零向正方向寻找第一个缺失整数，并利用差集对称性返回中心连续段的长度。若连续滞后为 $-L,\ldots,L$，其长度为 $2L+1$；排除负半部时返回 $L+1$。它不会对缺孔自动插值。该函数早期 docstring 的端点描述与实际循环有一位差异，读取时以 `while mv in self._index_map` 和返回表达式核对，本节采用实际代码的长度。

物理嵌套阵跨度为 $11d_0$，差集从 $-11d_0$ 到 $11d_0$ 的跨度为 $22d_0$，用于增广的单侧虚拟阵列有 12 个位置、跨度仍为 $11d_0$。三种数量分别是物理孔径、差集跨度和虚拟矩阵维度，不能把“23 个滞后”读成 23 路录音或 23 个可分辨声源。

#### 39.2 直接增广与空间平滑的实际计算

读取路线为 `WeightFunction1D` → `utils/math.py::vec` → [`estimation/coarray.py::CoarrayACMBuilder1D.transform`](https://github.com/morriswmz/doatools.py/blob/9469db201e0418aef6b97583ef54b6fec2769502/doatools/estimation/coarray.py)。输入是 $M\times M$ 物理协方差，输出是 $M_v\times M_v$ 增广矩阵，其中 $M_v=L+1$。本书采用 $R_{ij}=E[X_iX_j^*]$、滞后 $\ell=p_i-p_j$；向量化和位置差必须采用同一排列。

同一滞后的元素先求平均，得到 $z=[r[-L],\ldots,r[L]]^\top$。代码提供两个分支。

- `method='da'` 是直接增广：依次把 $z$ 的长度 $M_v$ 子段放到矩阵列中，得到 $T_{ij}=r[i-j]$。有限样本下，即使输入物理协方差半正定，$\mathbf T$ 仍可能有负特征值。
- 默认 `method='ss'` 是协同阵空间平滑：取全部 $M_v$ 个长度 $M_v$ 的连续子段 $\vec z_q$，计算 $\mathbf R_{\rm ss}=M_v^{-1}\sum_q\vec z_q\vec z_q^H$。每项外积半正定，所以结果半正定；这不自动证明源独立、模型匹配或快拍数充分。

若 $\mathbf T$ 是上述子段按列组成的矩阵，列顺序不会改变外积和，因此 $\mathbf R_{\rm ss}=\mathbf T\mathbf T^H/M_v$。在输入滞后共轭对称时，$\mathbf T$ 为厄米矩阵，于是特征值经过平方和尺度变换。直接增广的负特征值不会在平滑后保持符号，特征值大小次序也可能改变；不能据平滑矩阵半正定，反推直接增广原先没有问题。

两条分支的数值单位不同。若输入麦克风谱单位为 V，$r[\ell]$ 与 $\mathbf T$ 的元素单位为 V²，$\mathbf R_{\rm ss}$ 则为 V⁴；输入幅度整体乘 $b$ 时，两者分别乘 $|b|^2$ 与 $|b|^4$。空间平滑矩阵用于构造子空间，不应直接当作新增物理通道的声功率，或与原始协方差按元素比较功率大小。

#### 39.3 两组可独立复算的矩阵

正文 [E03-07](../../../../chapters/03_array-geometry.md#e03-07)使用物理位置 `{0,1,3}`，两个互不相关的单位功率源来自 0° 与 30°，通道白噪声功率为 0.1。其四维直接增广矩阵特征值为 $0.1,0.1,4.1,4.1$，因而空间平滑的特征值为 $0.0025,0.0025,4.2025,4.2025$，逐项来自原值平方后除以 4。

同题的一次快拍反例 $\vec x=[1,0,1]^\top$ 给出直接增广特征值 $-1/3,2/3,2/3,5/3$。空间平滑后的值则为 $1/36,1/9,1/9,25/36$。这些结果既用本书独立 NumPy 外积求和与矩阵乘法复算，也已与固定 doatools 原方法的提取调用对齐；一次快拍仍没有恢复真实的两个声源。

实际入口为[原方法审计工具](../../ch03/examples/audit_upstream_coarray.py)，当前结果保存于[执行合同报告](../../ch03/reports/upstream_coarray_contracts.json)，[原历史报告](../../ch03/reports/upstream_coarray.json)保留当时的字节与运行身份。工具执行 `da` 与默认 `ss`，对上述两个相同输入保存物理矩阵、滞后顺序、输出矩阵及特征值；期望来自独立逐滞后平均、直接 Toeplitz 填充、$\mathbf T\mathbf T^H/4$ 和手算特征值。逐元素绝对容差为 $10^{-14}$，两组实际最大矩阵差均为 0。把源幅度乘 2，即输入协方差乘 4，实际 DA 乘 4、SS 乘 16；缩放检查绝对容差为 $10^{-12}$。这些容差只用于此处固定的小规模输入，不是对任意尺度设备数据的通用验收限值。

运行环境为 Python 3.13.12、NumPy 2.5.3、macOS arm64。固定源码引用已经移除的 `np.float_` 与 `np.complex_`；工具在提取定义的局部命名空间内分别提供 `float64`、`complex128`，不改全局 NumPy、不改上游文件。AST 提取保留选中的原定义体及装饰器，跳过模块导入与其他顶层代码，因此这是有明确相容脚手架的原方法调用，不是完整原包运行。当前工具用严格 JSON 读取锁表，隔离会改变 Git 目标或配置的环境变量，核对官方 origin、固定 HEAD、七个所用源码与许可的 SHA/Git blob，以及执行前后洁净状态。完整稀疏选集另由获取工具核验：此 doatools 工作区仍使用旧排除规则，真实状态是 `source_selection_mismatch`；限定原方法身份匹配与执行成功没有把它升级为选集通过。工具还记录自身和所调用公共工具的实际摘要；[独立测试](../../../../tests/test_codes_ch03_upstream_coarray.py)另以手写矩阵和解析分数核对。

原始方法定位为 [Pal–Vaidyanathan 2010 §IV-A、p4172～4174 及 Caltech 所列勘误](https://authors.library.caltech.edu/records/e8ge0-xc746)。正式记录和勘误入口可追溯；本节不把源码对照称为原作者完整系统复现。

默认运行 `.venv/bin/python -m codes.chapters.ch03.examples.audit_upstream_coarray` 只输出报告；明确加上 `--report codes/chapters/ch03/reports/upstream_coarray_contracts.json` 才根据当前真实执行写新的当前报告；仓内其他路径、历史报告和符号链接父链在执行前拒绝，仓外普通报告路径也须通过路径检查。该有限写前检查不保证消除并发文件系统竞态。已有固定缓存缺失、HEAD 不符、工作树变脏或源文件与 Git blob 不同都会失败，不联网获取或覆盖缓存。

#### 39.4 几何秩、相位歧义与方向域

时延模型的几何秩与窄带相位混叠是两次不同的检查。共线基线只能确定一个方向投影；非共线平面基线可以确定两个投影，但法向两侧仍有镜像。增加非共面基线可补足第三个投影，却不能单独保证单频相位在整个球面上唯一。还要检查相位绕回后是否有不同方向产生相同相对导向。

[Tucker、Zhao、Ahmad 与 Potter 2022 原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9757818/)的 §II～III 用相对首阵元的几何矩阵定义方向域；平面阵的完整单侧方向域映射为二维单位圆盘。§IV 的 Corollary 4 要求最短非零歧义格点的范数严格大于 2；§V-B 给出规则六边形相邻间距 $d=\lambda/\sqrt3$ 时圆盘恰好相切。相切已经存在一对相同相位方向，因此无歧义需要严格处于临界内侧。正文的 7.5 cm 环径例由此得到约 5.28 kHz 的临界频率。

原文 §VI 也把无歧义限制为无噪声性质。近似相同的导向、高旁瓣、有限快拍和标定误差仍会造成大误差；不能把无歧义证明当成定位精度保证。若采用刚性挡板、方向性麦克风或实测流形，应重新比较完整复响应，不能直接套用各向同性传感器的纯相位判据。

[E03-16](../../../../chapters/03_array-geometry.md#e03-16)把第四麦高度依次缩小，却保持几何秩为 3：相同 $1\ \mu s$ 时差误差在无约束线性逆中引起的法向分量变化由 0.008575 增至 0.8575。这个计算诊断观测与单位方向约束的不一致，不能当作有效方向的角误差。

[E03-17](../../../../chapters/03_array-geometry.md#e03-17)先消去每频一个未知公共复源幅，再比较完整六麦流形：同一方向对在 8 kHz 残差为零、4 kHz 为 $8/9$；另一个 ULA 的 2/4 kHz 反例仍共同混叠。几何秩、单频相位唯一性、多频新增信息与带噪精度必须分别判断。

原论文脚注明确指向 [Alias-free-Arrays 作者仓库](https://github.com/Zhao-Shen/Alias-free-Arrays/tree/4c80e169518d44f8333aac7f13c935286538f670)。固定提交包含 `CreateFigure1a.m`～`CreateFigure2.m` 与 README，2026-09-28 核查未见代码许可证，因此只保留来源索引，不自动获取。其绘图代码也不等于带噪声阵列几何优化器。本书几何练习使用公开模型独立计算相对导向，不复制这些 MATLAB 脚本。

#### 39.5 前向失配模型与校准器的区别

固定 doatools 的 [`model/perturbations.py`](https://github.com/morriswmz/doatools.py/blob/9469db201e0418aef6b97583ef54b6fec2769502/doatools/model/perturbations.py)可以把指定误差施加到模型：`LocationErrors` 加到坐标，`GainErrors` 逐通道乘 $1+g_m$，`PhaseErrors` 乘 $e^{\mathrm j\phi_m}$，`MutualCoupling` 左乘矩阵 $\mathbf C$。其中 $\phi_m$ 用弧度，$g_m=-0.1$ 表示实际增益 0.9。这些类接收已给定误差，不从录音估计误差，也不是自动校准算法。

建议先用正文 E03-06 的已知双麦复增益检查前向响应，再把反向补偿和未知参数估计作为单独步骤。把全部通道乘同一非零复数可并入未知源幅度；相对校准需指定参考通道。参考通道本身含噪时，直接复谱比值会受分母噪声影响。多位置联合拟合还需检查参考能量、方向覆盖、模型残差与留出方向，不能从训练方向拟合准确推出全方向准确。

固定原方法的合法输入对照使用 $\mathbf A=\begin{bmatrix}1&\mathrm j\\1&-1\end{bmatrix}$。增益误差 `[0.1,-0.2]` 与独立左乘 `diag(1.1,0.8)` 相符；相位误差 `[0.2,-0.3]` rad 与左乘对应复指数对角阵相符；耦合矩阵 `[[1,0.1],[0.2,1]]` 与独立矩阵乘法相符。另以一维坐标 `[[0],[0.04]]` m 加二维位置误差 `[[0.001,0.002],[-0.001,0.003]]` m，原方法得到 `[[0.001,0.002],[0.039,0.003]]` m。输出及传入的导数对照都保存在同一报告，未执行未知参数估计。

[E03-15](../../../../chapters/03_array-geometry.md#e03-15)单独展示带噪参考：真实物理增益为 2，四个等权观测的普通最小二乘系数为 1.6。残差平方和由代入真增益的 4 降到 3.2，仍没有恢复物理增益；完整重复这组观测只同时放大分子与分母，系数不变。该例把参考噪声、拟合目标与校准准确性分开，不能由残差正交或更小推断参数正确。

**未知方向与任意复增益还有更深的不唯一性。** 对间距 $d$ 的 ULA，令 $u_q=\sin\theta_q$、$\kappa=2\pi f/c$，单频第 $q$ 个单源场景的通道预测可写 $x_{mqt}=g_m e^{\mathrm j\kappa md u_q}b_{qt}$。只要全部 $u_q+\delta$ 仍在允许方向域，把每个方向余弦加同一个 $\delta$，同时令 $g'_m=g_m e^{-\mathrm j\kappa md\delta}$，每个观测的复数预测便逐项不变，而且 $g'_0=g_0$。固定参考通道、增加几个未知方向或增加同频快拍，都不足以确定绝对方向；若每个频率的通道相位可自由变化，简单叠加频点也不能自动解除这项自由度。[Tashev 2004 的作者原文](https://www.microsoft.com/en-us/research/wp-content/uploads/2004/06/2004-ivantash-icme.pdf)说明交替估计方向与通道参数的思路，但该论文处理的是增益校准；[Weiss 与 Friedlander 1990](https://doi.org/10.1007/BF01201215)讨论方向与增益相位联合估计。下面的程序是本书独立写的受约束教学例，不是两篇论文的原实现。

[第 3 章受锚标定演示](../../ch03/examples/self_calibration_demo.py)先构造无锚反例：4 麦、间距 4 cm、1 kHz、声速 343 m/s，三场景真角为 −40°、0°、35°；方向余弦共同加 0.15，并让通道相位吸收反向斜率后，两套参数的逐样本复预测最大差约 $7.0\times10^{-16}$。随后给估计器一个**另行测得**的相邻通道相位 0°，固定参考麦增益为 1，在 −60°～60° 内交替剖面搜索方向、最小二乘更新各通道复增益与逐帧源谱。[唯一数值核](../../ch03/core/calibration.py)记录每轮残差目标、收敛状态和低能量、越界或混叠拒绝条件；[独立边界测试](../../../../tests/test_codes_chapter03_self_calibration.py)检查代数等价、正确锚与错误锚。固定种子 307、每场景 48 帧、复噪声 RMS 0.005 的一次合成运行于第 21 轮停止，三个角误差为约 +0.00561°、−0.01208°、−0.00282°；这不是跨条件性能分布。

错误锚比低残差更值得注意：把相邻通道相位误设为 +5°，同一无噪声数据仍可拟合到约 $10^{-28}$ 的残差目标，却把正横场景估成约 −6.84°。因此优化目标下降不能证明外部相位锚正确。示例假设同步远场、每场景一个窄带源、已知几何和准确的外部相位锚；它没有运行真实硬件、混响或多个同时说话人的实验，也不保证全局最优。运行：`.venv/bin/python -m codes.chapters.ch03.examples.self_calibration_demo`，标准输出为 JSON，不生成音频文件。

固定源码的输入检查存在下列实际触发的故障。负例直接调用未改定义，不经过本书对合法物理协方差的形状、有限性、厄米性和半正定性检查；“函数返回了值”与“接受了有效模型”分别记录。

| 原方法与输入 | 固定源码行为与实际结果 | 使用边界 |
|---|---|---|
| `GainErrors([0.1,0.2])`、`PhaseErrors([0.1,0.2])` | 列表转数组分支写入拼错的局部变量，后面仍读原列表的 `ndim`；两者均抛 `AttributeError` | 合法对照显式传正确形状的 NumPy 数组；不称列表路径已通过 |
| `MutualCoupling` 接收 2×3 数组 | 方阵校验使用 `and`，构造器实际接受非方阵 | 接受成功不构成合法耦合模型；调用前独立核对方阵与通道数 |
| `MutualCoupling` 接收一维数组 | 访问第二维长度时抛 `IndexError` | 形状错误没有得到完整的接口诊断 |
| `CoarrayACMBuilder1D` 接收 3×4 输入 | [`ensure_covariance_size`](https://github.com/morriswmz/doatools.py/blob/9469db201e0418aef6b97583ef54b6fec2769502/doatools/estimation/core.py#L14)重复检查行数；把合法3×3矩阵附加一列 `[99,99,99]` 后仍返回相同 DA，额外列被静默忽略 | 协方差必须先核对两个维度，不允许截断通道冒充输入相符 |
| 同一构造器接收 3×2 输入 | 行数检查仍放行，后续向量索引抛 `IndexError` | 不由晚到的索引异常替代入口形状检查 |

这些结果绑定上述固定提交与报告环境，不推断后续主分支是否已经修复，也没有补丁改写本地上游缓存。合法前向模型、错误输入表现、未知参数估计及设备验收是不同证据。

串扰补偿还应注明噪声加入的位置。若观测为 $\vec x=\mathbf C\vec s+\vec n_{\rm post}$，左乘逆矩阵会改变后加噪声为 $\mathbf C^{-1}\vec n_{\rm post}$，病态矩阵可大幅放大它；若噪声在混合前已与信号共同经过 $\mathbf C$，就不能把同一句结论无条件套上去。方向相关壳体散射应进入实测流形，固定串扰矩阵不能代替它。

#### 39.6 从 TOA 距离到同步麦克风 TDOA 自标定

[Kuang 等 ICASSP 2013 的大学官方记录](https://www.lunduniversity.lu.se/lup/publication/e90f122e-669e-44c8-8da8-41fae8645aae)明确讨论到达时间（Time of Arrival，TOA）给出的全部接收器—发射器距离。到达时间差（Time Difference of Arrival，TDOA）只提供距离差，未知发声时刻带来另一组未知量，不能直接套用 TOA 的最小数据要求。

进一步的作者系统见 [Zhayida 等 2016 预印本 §2、§5～6](https://arxiv.org/html/1610.02392)。该系统在未知麦克风与声源坐标之间估计几何，接收麦克风仍须同步，声速须已知。论文中的未知发声偏置不是每台独立设备的采样时钟偏移；设备时钟不同步时还要增加同步模型。论文通过多个相关峰、跨通道一致性与稳健拟合处理错误匹配，不能只取一次互相关最高峰就宣称复现了该系统。

论文 §10 指向 [StructureFromSound 作者代码](https://github.com/kalleastrom/StructureFromSound/tree/9b7db79a489d347bed9a38bd38224e65c273660a)，根 [LICENSE.md](https://github.com/kalleastrom/StructureFromSound/blob/9b7db79a489d347bed9a38bd38224e65c273660a/LICENSE.md)为 GPL-3.0。代码来源与取得状态见锁表及状态报告；录音、测量真值、模板和外部工具箱分别核对，不由根代码许可推断数据可再分发。

读码从 [`matlab/main.m`](https://github.com/kalleastrom/StructureFromSound/blob/9b7db79a489d347bed9a38bd38224e65c273660a/matlab/main.m)进入 `sfs_system_v1`。相关峰匹配后，`main_gcctracking_rex` 将采样点差乘 `settings.v/settings.sr` 得到米制 `matches.u`；后续估计偏置、麦位置与源位置，并继续寻找内点。

[`tdoa/calcresandjac.m`](https://github.com/kalleastrom/StructureFromSound/blob/9b7db79a489d347bed9a38bd38224e65c273660a/matlab/tdoa/calcresandjac.m)的残差是 $e_{ij}=\|\vec x_i-\vec y_j\|+o_j-D_{ij}$。这里 $\vec x_i$ 是麦位置、$\vec y_j$ 是第 $j$ 个声事件位置，均以米计；$D_{ij}$ 是已转换成米的距离差，不是秒或采样点数。对于以麦 1 为参考的理想距离差，$o_j=-\|\vec x_1-\vec y_j\|$，所以参考行残差为零。相同项目的部分注释采用相反的偏置符号，适配时须用实际残差表达式核对。

最小独立检查可取 $\vec x_1=(0,0,0)$ m、$\vec x_2=(3,0,0)$ m、$\vec y=(0,4,0)$ m。两距离为 4 m、5 m，距离差为 `[0,1]` m，偏置为 −4 m，两个残差都为零。整体平移、旋转或镜像后距离不变；仅凭这些观测不能恢复外部世界坐标。这个 3–4–5 手算只验证符号与单位，不满足完整三维自标定的观测数量，更没有执行 MATLAB 求解器。

#### 39.7 固定校准代码的运行前提与障碍

下面记录的是固定提交的静态检查。源码可以支持模型研究，但“取得源码”不能改写成“原作者系统已复现”。

| 入口 | 已读到的具体行为 | 对复现的影响 |
|---|---|---|
| `matlab/main.m` | 写死作者本机的 `multipolpath`、`datapath`，读取指定 `sfsdb` 记录，再调用 `sfs_system_v1` | 需另准备路径、依赖与获授权录音；原样运行不是可移植入口 |
| `sfs_system_v1`、`main_gcctracking_rex` | 内部重设声速为 340 m/s，后者写入 `channels=1:8`、参考 1、窗长 2048、步长 1000 | 论文实验声速 343 m/s 不能直接当成固定程序值；传入设置也可能被覆盖 |
| `tdoa_offset_ransac` | 抽取 7 麦、6 个完整匹配列，加载 `options76`，调用偏置求解 | 只有部分缺失数据可处理，不代表任意缺测模式都可解；模板和随机状态须记录 |
| `tdoa_offset` | 多项式分支调用 `multipol`、`polysolve`，部分路径含 `keyboard` | MATLAB 与辅助代码、模板及交互停止点需逐项处理，不能只列主文件 |
| `bundletdoa` | 构造过固定坐标的选择矩阵，但当前活跃分支对全部参数作阻尼更新 | 不能声称已使用坐标固定分支；比较几何误差前须处理刚体规范，检查实际残差与退化方向 |
| `sfs_system_v2` | 留有合并冲突标记，函数输入为 `matches,rns,settings` 却还读取 `a` | 是备用版本的静态故障；`main.m` 调用的是 v1，不能把 v2 故障说成已实测的主入口失败 |

[`bundletdoa.m`](https://github.com/kalleastrom/StructureFromSound/blob/9b7db79a489d347bed9a38bd38224e65c273660a/matlab/tdoa/bundletdoa.m)的阻尼不是补充声学信息。它可以使数值方程更容易求解，却不能消除整体坐标规范不唯一，也不能保证全局收敛。正确的验收应分别报告距离差残差、规范对齐后的坐标误差、失败次数，以及未参与拟合的声源位置结果。上述程序尚未在本机运行，表中的问题不带虚构的报错输出或运行时间。

#### 39.8 实测阵列流形怎样进入工程系统

[HARKTOOL5 官方传递函数生成手册 §1.3、§2.2、§3.1](https://www.hark.jp/document/tf/generating_transfer_functions/Generating_a_Transfer_Function_Using_HARKTOOL5.html)区分测量与几何计算两条路线。测量路线利用时间伸展脉冲（Time Stretched Pulse，TSP）或脉冲响应录音建立各方向到各通道的传递函数；几何路线根据麦与源坐标生成自由场响应。后者不能自动包含实际外壳散射。

测量文件的通道数、录音通道次序及声源方向须对应。官方表 2 对测量路线要求正确的麦数与声源位置，表 5 对几何路线还要求正确麦坐标；因此“文件能加载”并不能证明测量坐标和几何模型都正确。方向网格、采样率、频率网格、参考通道和幅相归一化应随传递函数一起保存。

[HARK Cookbook 3.5.0 的脉冲响应测量节](https://www.hark.jp/document/3.5.0/hark-cookbook-en/subsec-InputDataGeneration-002.html)要求播放扬声器、麦阵、可同时播放录音的音频设备和重复 TSP。这个流程与第 2 章的反卷积测量相衔接，但这里没有运行 HARK、采集设备录音或验证某款产品。官方示例中的角度步长、重复次数和频带设置属于该流程，不能提升为所有产品的统一验收标准。

**HARKTOOL5 有实际源码，但许可和分发形式要分别核对。** [官方源码页](https://hark.jp/download/source-code/)给出 Ubuntu 源包分发入口。本节固定使用 jammy 仓库的 `harktool5_3.5.0.tar.xz`，用[同版 `.dsc` 文件](http://archive.hark.jp/harkrepos/dists/jammy/non-free/source/harktool5_3.5.0.dsc)公布的 SHA-256 校验归档；它不是 Git 提交。下载地址、摘要和选择的源码范围记录在[归档来源锁表](../ARCHIVE_SOURCES.lock.json)，本地取得与校验结果由[归档状态报告](../ARCHIVE_SOURCE_STATUS.json)记录，和 Git 来源计数分开。

包内 `debian/copyright` 与[官方 HARK License v2.0（2020 年 7 月）](https://hark.jp/notice/HARK_License_Agreement.pdf)一致。该许可限定研究、开发、教育或学术用途，商业用途另有约定；它不等于 MIT 或 GPL 许可。本书只保留独立的本地研究源码及许可，不把它纳入教程的开放许可或重新分发。选择范围包括构建说明、相关 C 源文件和原始文档，排除样本音频、二进制与生成的文档资产。源码取得不证明编译、设备采集或算法运行成功。

静态阅读可从 `src/harktool/main.c` 进入 `calctfgeo.c`（按几何生成传递函数）、`calctfrec.c`（从 TSP 录音生成传递函数）和 `calctfimp.c`（脉冲响应输入）。`calctfrec` 读取采样率、FFT 长度、同步平均与峰值搜索范围；这些设置必须随测量记录保存。`calctfgeo.c::writeNormalizeInfo` 明确提示忽略关闭归一化的选项，几何分支始终输出归一化值。因此，把它与实测响应比较之前，应先核对归一化和参考量，不能把两者幅度之差全部归因于壳体或标定误差。源包根 `CMakeLists.txt` 仍有旧的 `PROJECT_VERSION "3.4.0"` 字段，本书依据 `.dsc` 及归档摘要识别 3.5.0 分发包，不用这个旧字段冒充新版本证据。

实际应用可先用已知坐标生成解析响应核对通道与符号，再用实测流形替换；保持同一录音与评分方法比较目标响应和错误方向峰。把测量方向分成拟合集与留出方向，另测试设备外壳装配、温度或位置变化。改进应由这些独立条件的结果支持，不能仅凭存下更多方向或训练残差变小判断标定有效。

一个固件配置实例是 [Infineon AN240916 的 BeamForming Configuration](https://documentation.infineon.com/psocedge/docs/fqo1761931960980)（2026-09-28 核实）。其双麦接口给出 20～100 mm 麦距和 0～180° 角度参数；接入前须转换本书坐标约定并核对通道次序。厂商说明波束图是示意，实际场景仍需测量；`audio-voice-core` 是单独授权交付的算法库，公开应用示例不能作为算法核心已经开源的证据。

该文档直接指向[官方 AE 应用固定源码](https://github.com/Infineon/mtb-example-psoc-edge-ae-application/tree/955a61090acf7caddd75fc74161fc0fb40aa7ae5)。仓库根 `LICENSE` 是 Infineon EULA，不能仅因公开可读就将整个应用标为 Apache-2.0。不过 `proj_cm55/source/audio_enhancement_application/audio_enhancement/GeneratedSource/` 下的 `cy_afe_configurator_settings.c` 和 `.h` 各自明确写有 Apache-2.0 许可头。本书的[Git 来源锁表](../SOURCES.lock.json)只选择这两个配置文件，并保留根许可供核对；其余应用包装代码、配置器项目和核心库不因这两个文件的许可而变成开放源码。

阅读配置头中的 `AFE_PARAM_ID_MIC_DIST`、`AFE_PARAM_ID_ANGLE_RANGE_A` 和 `AFE_PARAM_ID_ANGLE_RANGE_B`，可以追踪麦距与角域怎样进入部署接口；配置 C 文件保存生成的常量数组。它们由配置器生成，不能代替配置器算法源码，也不是可独立运行的波束形成器。本节没有构建、刷写或执行 Infineon 应用；涉及核心库的算法性能不由这两个配置文件证明。


#### 39.9 已知方向如何分开位置和固定通道时延

正文 [E03-18](../../../../chapters/03_array-geometry.md#e03-18)限定为一个远场声源、已知外部方向和声速，未知量是一条三维麦克风基线及固定相对通道延迟。正延迟表示第二通道的观测更晚；它与传播时差相加，不能由一次朝向的相位差单独分开。

把时间延迟乘声速得到长度后，设计矩阵的每行是 $[-\vec u_q^\top,1]$。四方向 $+x,-x,+y,+z$ 可解四个未知量，前两时差的平均先确定固定延迟。仅检查方向矩阵 $\mathbf U$ 的秩还不够：四方位同俯仰30°时，$\mathbf U$ 秩3，但常数列与高度列相关；高度加1cm同时延迟加约14.5773μs，全部观测不变。改变正则化不能补出没有采集到的信息。

[唯一教学核](../../ch03/core/baseline_calibration.py)使用增广矩阵最小二乘，秩不足时明确拒绝。六份[独立合成音频](../../ch03/baseline_audio/MANIFEST.json)从真实PCM相量读回四训练方向的时差，并用未参与拟合的方向检验预测；这是给定模型的数值闭环，没有从未知房间或机器人轨迹估计方向。500Hz单音只在已知总时差绝对值小于1ms时可用相位主值直接反推；整数周歧义、任意频率相关电子相位、声速未知尺度和时钟漂移须另建模型。

#### 39.10 异步多阵列标定增加了哪些观测

已知方向的一条基线线性逆问题、同步麦克风的几何自标定，以及独立设备的时钟标定有不同未知量。固定电子延迟不随记录时间变化；采样时钟速率误差则使时间差随时间积累。把首设备的相对时间偏移设为零，只是选择参考，不能据此认定所有采样时钟已经同步。

| 原始路线 | 输入与额外前提 | 求解目标和关键改变 | 本书当前证据 |
|---|---|---|---|
| [Gburrek等2021，§1～2](https://link.springer.com/article/10.1186/s13636-021-00210-x) | 各节点已知阵内几何、阵内通道同步；节点粗时间同步关联同一源段，方向及声学距离估计；本文为二维 | 联合恢复节点几何，以距离提供尺度；该路线不要求跨节点精确TDOA | 已读原文，未执行作者系统或设备 |
| [Wang等2024预印本v1，§II～III](https://arxiv.org/html/2405.19813v1) | 已知阵内几何、阵内同步，跨阵列TDOA/DOA与声源运动的里程计 | 同时建模节点位姿、固定偏移、速率漂移及源位置；雅可比秩给局部可辨识条件 | 原文模型与秩条件已读；没有将局部可辨识写成全局唯一或成功率保证 |
| [Zhang等2025预印本v1，§II～III](https://arxiv.org/html/2502.06195v1) | 时间TDOA-S、空间TDOA-M、DOA、里程计和已知事件时间间隔 | 利用同阵跨事件时差与同事件跨阵时差的不同时间项，初始化后做加权非线性拟合 | 固定作者链接源码静态核；未运行MATLAB、DOA/里程计整链或实录数据 |

表内论文年份和节号绑定所链版本，2024与2025两项按预印本介绍，不猜测最终出版版本或通用工业默认。观测更多是否有用取决于它们提供的独立信息；相同角度覆盖、共同参考误差和额外先验须一起检查。没有同一数据、坐标约定、时间基准与评分协议，不能把三条路线按论文数字排序。

2025路线的时间TDOA-S包含本节点漂移乘已知事件间隔，空间TDOA-M包含节点间相对offset和相对drift项。原 [gt_generation.m](https://github.com/AISLAB-sustech/Hybrid-TDOA-Multi-Calib/blob/4cc21cb06b9f82f83cc90d65a418f2b100748254/gt_generation.m#L24)固定首阵列位姿及相对offset，却仍估首阵列drift；已知事件间隔提供时间尺度。实录入口从 `dt.mat` 读间隔并从位移数据构造里程计，这些输入不能在“只靠录音标定”的表述里省掉。

作者预印本链接的 [Hybrid固定源码](https://github.com/AISLAB-sustech/Hybrid-TDOA-Multi-Calib/tree/4cc21cb06b9f82f83cc90d65a418f2b100748254)在GitHub标为 `zcj808` 项目的fork。核实于2026-10-04，main/HEAD为完整提交 `4cc21cb06b9f82f83cc90d65a418f2b100748254`；[来源锁表](../SOURCES.lock.json)仅登记 `hybrid-tdoa-multi-calib`，许可为 `NOASSERTION`，不自动获取或随书复制。根列表与四个普通许可候选请求未建立代码许可，不据此断言完整树没有任何授权。

读码顺序为 `sim_main.m` 或 `real_main.m` → `init_estimator.m` → `GN_Solver.m` → `compute_J.m`，再核 `gt_generation.m` 的时间基准及 `high2low.m/low2high.m` 的参考参数。DOA前端另读 [SRP-PHAT-DOA.py:351～404](https://github.com/AISLAB-sustech/Hybrid-TDOA-Multi-Calib/blob/4cc21cb06b9f82f83cc90d65a418f2b100748254/SRP-PHAT-DOA.py#L351)。该入口使用已知平面六麦环，在394行主动取方向z分量的绝对值，395行翻转y；上半球选择来自先验，不能说平面阵凭声学独立辨出上下两侧。Jacobian的局部DOA用接收传播方向，本书 $\vec u$ 指向声源，适配须核对符号和坐标变换。

固定静态身份中，README的SHA-256为 `c54031c5dd6b4537249a4e5f6933dfe121b1dc97ccdc11cb7ba3f6193eed7a77`，`compute_J.m`为 `fd82d0e8dc9f9d7937b870557c595043d96ada74c7eeb5984a2f32b04238f42b`。这些身份只追溯已读响应，不等于完成本地选集取得、依赖构建或原系统运行。

静态检查还发现几个复现前须核对的范围：仿真入口首段含 `clc7`；GN用正规方程并按时间截断；仿真只保存收敛试次，随后统计中间四分位样本；`compute_error.m`用旋转后的单个固定向量夹角，并非完整SO(3)测地旋转误差。SRP配置声速346m/s，实录数据构造/求解配置另设340m/s。这里只记录原文件配置与实现定义，没有虚构运行报错或判定设备误差；尚未运行MATLAB、PyTorch前端、OneDrive音频、`.mat`真值或完整比较。

建议实验从E03-18的满秩与同俯仰退化开始，再把方向测量噪声、错误声速、固定延迟和随时间变化的时钟项分别加入。进入非线性多阵列实验时，应先固定参考位姿、相对offset和事件时间基准，使用留出事件报告残差、对齐后几何误差及失败分母。该段是实验设计，不是已经执行的上游结果。

### 40. IMM 多运动模型交互

对应 §9.2 的运动建模扩展。FilterPy `kalman/IMM.py::IMMEstimator` 维护多个滤波器、模式概率与 Markov 转移概率，先按模式概率混合状态和协方差，再执行各模型的预测与更新。所有被混合状态应有相同维度和物理含义，不能直接把角度状态与笛卡尔位置相加。[作者源码](https://github.com/rlabbe/filterpy/blob/3b51149ebcff0401ff1e10bf08ffca7b6bbc4a33/filterpy/kalman/IMM.py)。

建议给匀速、转弯和静默三段轨迹，比较单模型与多模型的跟随延迟和协方差。必须核对转移矩阵方向及归一化，而不是凭行列名称猜测；长期缺测时不能把模式概率改变解释成观测证据。源突然停止说话是活动状态变化，并不必然代表其运动模式改变。

原始 IMM 论文为 Blom、Bar-Shalom，*The Interacting Multiple Model Algorithm for Systems with Markovian Switching Coefficients*，IEEE TAC 33(8), 1988, pp.780–783，[DOI](https://doi.org/10.1109/9.1299)。本书混合矩的数字推导独立列明行转移约定；论文身份与具体库实现应分开核对。

本轮原类调用发现两个缺测边界。`predict()` 在混合和预测子滤波器后用旧 `mu` 融合总体均值、协方差，没有按 `cbar=mu@M` 形成该时刻总体先验；连续只预测时还会沿用旧模式权重。`update(None)` 也不是“什么都不变”：子 KF 把新息设为零、清除似然缓存，但保留旧新息协方差，IMM 随后读取这个零新息下的似然更新模式概率。没有新观测时，这不等于仅传播 Markov 模式概率。§54 保留第一次预测、连续三次预测及有观测后转入缺测的全部分量和总体二阶矩；不通过修补上游来掩盖差异。

### 41. icoDOA、Cross3D 与神经方向追踪

对应 §4.7、§9.3。icoDOA 的 `1sourceTracking_icoCNN.py` 是训练与测试入口；`acousticTrackingModels.py` 定义 `IcoTempCNN`、`Cross3D` 等模型，`acousticTrackingModules.py` 提供输出映射，`acousticTrackingDataset.py` 生成移动源场景。二十面体方向网格与普通平面卷积网格有不同邻接结构，方向分辨率及输出变换必须与训练时一致。[作者仓库](https://github.com/DavidDiazGuerra/icoDOA)。锁定 README 的 PyTorch 依赖条目把版本误写成“Python 1.8.1”；不能把这个文本推断成已验证的 PyTorch 1.8.1，重跑须先独立锁定可用环境。

最小实验应先验证方向图旋转与标签同步，再用未见房间、阵列误差和静默输入测角度误差与失效状态。代码依赖 gpuRIR、icoCNN 和声学数据；只读到模型类不表示完整训练可复现。作者 README 明确提醒主脚本未调用的辅助功能可能未经测试，因此同一文件中的 `SELDnet` 类也不能直接等同于已经验证的 SELD 系统。AGPL 源码与模型、LibriSpeech/LOCATA 数据分别核对。

这类方向图网络的输入依赖上游 SRP 图的网格、角度坐标、频带与归一化；即使输出同为三维方向，也不能直接把不同麦阵生成的任意图送入已有权重。单源方向追踪脚本不自动提供同类重叠多源的轨道容量或身份关联。本书本次未运行训练、预训练推理或 LOCATA 评分，工程复现还需固定训练划分、上下文长度、模型状态复位和前端耗时，分别核对网络耗时与端到端延迟。

固定实现中 `IcoTempCNN` 输入轴是批量、通道、时间、五个网格面片及两个面片坐标，最后经 `SoftArgMax` 输出方向坐标；`Cross3D` 则在方位—俯仰方向图上分支卷积。`CausConv1d/2d/3d` 左侧补足历史长度后裁去尾部，没有自动保存跨调用的流式历史；逐帧调用仍需调用者管理上下文。还应避开未验证的任意参数组合：这些层直接使用 `:-pad`，时间核长度为 1 时 `pad=0` 会产生空切片。该点由固定源码静态识别，未执行 Torch 模型；不否定作者使用大于 1 时间核的既定配置。

ACCDOA 的向量模与方向、multi-ACCDOA 的输出槽位和持久说话人 ID 是三个不同对象。训练时允许轨道置换不会自动解决第 9 章两人交叉的身份问题；需要明确跨帧关联或身份特征后才能评价 ID 切换。有限近期候选及不纳入正文目录的原因见 §57。

#### GCC 向量、GCC-PHAT 图与 STFT 相位图：三个不同的分类输入

第 4 章的[学习式定位比较表](../../../../chapters/04_doa-estimation.md#sec-4-7)把输入特征和输出格式分开，不能把不同论文中的特征和网络随意拼成一项已验证系统。[Xiao 等的 ICASSP 2015 论文](https://doi.org/10.1109/ICASSP.2015.7178484)以 GCC 向量特征输入多层感知机并分类 DOA；目前直接核到的[作者机构论文页](https://experts.illinois.edu/en/publications/a-learning-based-approach-to-direction-of-arrival-estimation-in-n/)没有明确把该向量限定为 PHAT 加权，也未在此核定具体麦对构造。[Zhao 与 Ritz 的 APSIPA-ASC 2021 论文，§III 与图 5](https://www.apsipa.org/proceedings/2021/pdfs/0000974.pdf)则把共素阵的 GCC-PHAT 二维特征图送入 CNN。前者的 GCC 向量与后者的 PHAT 图不能仅凭名称互换。

[Chakrabarty 与 Habets 的 IEEE JSTSP 2019 论文作者稿](https://www.audiolabs-erlangen.de/resources/aps-w23/papers/sap_Chakrabarty2019.pdf)使用多通道 STFT 相位图输入 CNN，目标是多类、多标签方位分类，不是上面两条 GCC 输入路线。作者[出版页](https://soumitrochak.netlify.app/publication/chakrabarty-2019/)虽链接一个[模型仓库](https://github.com/Soumitro-Chakrabarty/Single-speaker-localization)，但该仓库 README 对应 2017 年单源论文、声学和阵列配置也不同，且本次未核到可再分发的许可；不能把它标成 2019 年多源整链复现代码。三项在[覆盖表](../COVERAGE.md)均为“原理索引”，不是本书已经训练、运行或收录源码的模型；定向检索未核到与三篇论文逐项对应且许可明确的作者实现，并不证明此类实现不存在。

若做最小实验，应固定同一组同步多通道录音、训练/验证/测试房间划分、阵列坐标、角度类别和评分协议，分别计算 GCC 向量、共素阵 GCC-PHAT 图和 STFT 相位图，再训练各自结构；解析 GCC/SRP 的同输入结果是必要基线。模型的角度误差、漏检和跨房间差异需逐项报告；本书没有这组数据和训练结果，不能用论文结论或本书合成 WAV 冒充这一比较。

### 42. FN-SSL 与 IPDnet：先估计直达声相位差

对应 §4.7。Wang、Yang 与 Li 的 [IPDnet 原论文](https://doi.org/10.1109/TASLP.2024.3507560)把多个声源的直达声麦间相位差作为学习目标，再用阵列几何转换为方向；它并非直接输出一个房间无关的绝对位置。作者仓库是 [Audio-WestlakeU/FN-SSL](https://github.com/Audio-WestlakeU/FN-SSL)，本次锁定 `76fcb281be92caf068c712dfb015e354f437260f`，不是其他领域的同名 IPDNet。

读码顺序为 `IPDnet/Opt.py` 的阵列与数据设置、`Simu.py`/`Dataset.py` 的信号与标签、`FixedAarryIPDnet.py` 和 `VariableArrayIPDnet.py` 的网络、`Module.py` 的空间处理，最后读 `runIPDnetOn.py`/`runIPDnetOff.py` 的训练和评价。固定阵列文件名确实拼作 `FixedAarryIPDnet.py`。根 README 的通用 `main.py` 命令不能直接当成 IPDnet 子目录入口。[实际目录及命令](https://github.com/Audio-WestlakeU/FN-SSL/tree/76fcb281be92caf068c712dfb015e354f437260f/IPDnet)。窄带分支沿时间处理各频点，全带分支沿频率学习相位差与频率的关系；“在线”配置还要检查具体上下文和分块延迟，不能由脚本名字推出零前视。

最小实验先合成已知两麦时延的单个平面波，检查预测相位差与几何字典匹配的符号；再加入第二个源、换麦间距和调换通道。若交换麦顺序却不更新字典，错误方向来自接口失配，不能据此衡量网络泛化。记录源数、频带、阵列适配方式、输出轨数和活动阈值，且把未见房间与未见阵列分开评价。根 README 写 MIT，但当前未核实完整许可及其致谢的 Cross3D/icoCNN 派生仿真部分声明，因此暂不自动下载；模型和 LibriSpeech、Noise92、LOCATA、RealMAN 数据另核授权。

原论文的直达声相位差目标与最终 DOA 评分应分别验收：预测相位轨正确并不意味着候选几何、活动判定和多源解码全都正确。作者预印本的模型图与直达声相位差定义可用于核对输出轨、频率轴和麦对轴，但论文实验数字必须绑定正式版的具体配置，不从预印本摘要迁移成工业保证。[作者预印本 §II～III](https://arxiv.org/pdf/2405.07021)。此项仅完成原文及公开源码入口核查，未运行模型。

### 43. ACCDOA：活动与方向共用一个向量

对应 §4.7。[Shimada 等 ICASSP 2021 原文](https://doi.org/10.1109/ICASSP39728.2021.9413609)定义每类每帧的活动耦合方向输出。训练标签在事件不活动时为零向量，活动时为单位笛卡尔方向；预测向量的模用于活动判定，归一化后的方向用于定位。预测模不是自动校准过的存在概率，也不是声源距离。

最小手算沿用本书 $+y$ 为零点、向 $+x$ 为正的约定，二维 30° 单位标签为 $(\sin30^\circ,\cos30^\circ)=(0.5,\sqrt3/2)$，预测为 $(0.25,\sqrt3/4)$。预测模是 $\sqrt{0.25^2+(\sqrt3/4)^2}=0.5$，方向仍由 `atan2(x,y)` 得到 30°。若上游数据以 $+x$ 为零点并用 `atan2(y,x)`，要先转换坐标再比较；不能把其 `(0.866,0.5)` 向量直接代入本书约定后仍称 30°。活动判定是否通过取决于明确的阈值及 `>`/`>=` 约定；不能因角度正确就忽略漏检。真实三维任务应保留 z 分量。再设置同类别同时出现在 30° 与 90°：单向量不能无损表示两个方向，这正是 multi-ACCDOA 要解决的输出容量问题，而非增加网络参数即可消除的问题。

可对照 [DCASE2022 官方基线模型](https://github.com/sharathadavanne/seld-dcase2022/blob/c8adb1d3a5a35de2d6c7b6d19e01ad455eef3986/seldnet_model.py)的单/多轨输出和特征生成阅读，但它是后续届次实现，不等于已经复现 2021 原论文。模型结构、训练集合和评价指标都应随所用基线记录。

### 44. Multi-ACCDOA、ADPIT 与 DCASE2022 基线

对应 §4.7。[Multi-ACCDOA 原论文](https://arxiv.org/abs/2110.07124)用多条活动耦合方向轨道表示同类重叠源；辅助重复排列不变训练（ADPIT）构造多种合法标签排列，使固定轨号不必永久对应同一实例。训练时的轨道置换不自动提供跨帧身份，因此这种输出不能直接替代第 9 章的持久轨迹 ID。

[2022 官方任务页](https://dcase.community/challenge2022/task-sound-event-localization-and-detection-evaluated-in-real-spatial-sound-scenes)链接到 `sharathadavanne/seld-dcase2022`，固定提交为 `c8adb1d3a5a35de2d6c7b6d19e01ad455eef3986`。从 `parameters.py` 读任务与 `quick_test`，再读 `cls_feature_class.py` 的 FOA/MIC 特征、`cls_data_generator.py` 的批数据及标签、`seldnet_model.py` 的网络与 ADPIT、`train_seldnet.py` 的输出合并和 `SELD_evaluation_metrics.py` 的评分。MIC 的 SALSA-lite 特征与 FOA 特征不同，四路音频并不意味着可互换。[基线说明](https://github.com/sharathadavanne/seld-dcase2022)。

最小实验构造一类零、一、二、三个活动方向，检查标签维度、允许的置换和解码后的源数，再让两个预测轨输出近似同方向以检查重复合并阈值。STARSS22 开发与评价划分不能混用，初始化默认的 `quick_test=True` 仅用于短流程检查，不代表完整训练或论文分数。官方页面展示的源代码未建立明确再分发许可，故只锁定索引，不自动复制；数据许可另核。

评估还需同时保存角度匹配阈值、活动阈值和重复轨合并条件；降低活动阈值可能减少漏检并增加虚警。ACCDOA/Multi-ACCDOA 输出形式本身不确定这些阈值，也不使不同届次的检出、定位与关联指标可直接互换。本次未执行 2022 基线的训练、推理或评测。

### 45. DCASE2025 立体声 SELD：方位、距离与画内/画外

对应 §4.7 的届次变化。[2025 官方任务页](https://dcase.community/challenge2025/task-stereo-sound-event-localization-and-detection-in-regular-video-content)与[作者基线](https://github.com/partha2409/DCASE2025_seld_baseline)规定的是双声道常规视频场景，不是 2022 的四通道三维定位。锁定提交 `42a48b6456b73be35ad0e1a9ffeb6ceef83ae0bd`。`model.py::SELDModel` 的音频输出每轨每类含 x、y 和距离；视听配置另加画内/画外输出。这里第三个量是距离，不能按旧的 xyz 方向向量求三维模。[实际输出层](https://github.com/partha2409/DCASE2025_seld_baseline/blob/42a48b6456b73be35ad0e1a9ffeb6ceef83ae0bd/model.py)。

读码从 `parameters.py` 的任务配置到 `model.py` 的激活函数，再到 `loss.py`、`inference.py` 和 `metrics.py`；需同时确认音视频时间对齐与米制距离标签。模型包含双向 GRU，整段执行不是已验证的因果实时推理。最小实验用相同方位、不同距离的标签检查输出解释，再构造声音活动但画面中不出现的事件，确认画外不等于静默。距离误差、角度误差、事件检出与画内/画外分类应分别评价。

本次未建立该仓库明确的再分发许可，只保留官方入口。数据生成器、预训练视觉模型和音视频数据另有来源及条款，不能由 baseline 的公共可见性推断全部可随书发布。

**2026 届的任务变化另记。** 截至 2026-09-29，[官方 Task 3 页面](https://dcase.community/challenge2026/task-semantic-acoustic-imaging-for-sound-event-localization-and-detection-from-spatial-audio-and-audiovisual-scenes)已标为结束并提供结果入口；本书只核对任务与基线，不据此编写跨系统排名。该届转向空间音频及视听场景中的语义声学成像，开发材料包含 32 通道录音，规定评价输入为其中指定 4 通道；32 通道采集配置不能写成基线推理使用 32 路，也不能沿用 2025 的立体声标签与输出格式。开发集的高分辨率参考声学图由 32 通道录音经 Latent Acoustic Mapping 超分辨方法生成，再由声事件球面多边形掩码分离；这是算法派生训练参考，并非逐像素独立实测真值。评测输入不提供这些参考图，须区分事件标注、派生强度图和模型预测图。

官方 `iranroman/DCASE2026_Task3_SAISELD_baseline` 固定到 `d4df66251f39e34bc0157be93858e5a68ec9d7c4`。从 `acoustic_features.py` 读音频特征，沿 `lam_model.py` 的 UpLAM 到 `model.py` 的实例模型，再读 `run_inference.py` 的掩模关联和 JSON 输出、`evaluate.py` 的评测。基线把 4 路音频送入声学图预测，再把九频带图与 RGB 通道送入实例模型；纯音频配置使用零值视觉通道。按掩模 IoU 进行 Hungarian 关联是这条基线的后处理，不等于由空间图自动获得永久声源身份。[固定基线入口](https://github.com/iranroman/DCASE2026_Task3_SAISELD_baseline/tree/d4df66251f39e34bc0157be93858e5a68ec9d7c4)。

该固定仓库根目录、README 和已核对模型文件未建立明确许可文本，因此仅保留索引，不复制源码或 `UpLAM.pth`；模型权重和数据许可也不由代码可见性推出。这里未运行 2025/2026 两届基线，未将不同通道、标签、数据和评分规则下的数字直接比较。

<a id="said-semantic-imaging"></a>

#### SAID：直接从四路声学特征生成语义强度图

[SAID 作者预印本 v1，2026-09-25，§2～4](https://arxiv.org/html/2609.31492v1)解决的是语义成像中的事件区域、强度和类别联合预测。相对先由 UpLAM 生成声学图再用实例模型处理的基线，它从四路幅度与相位特征开始，经 ConvNeXt 和学习的方向查询交叉注意力生成 `45×90` 中间空间表示，再用多尺度、16 个掩模查询输出 `180×360` 的实例强度图、活动和 13 类预测。16 是输出查询槽数，不能当作任意场景同时检出 16 源的保证；空间图也不是声源波形或持久身份。仓库说明稿件获 DCASE2026 Workshop 接收，本文公式定位仍使用这个预印本版本。

这一方法以训练几何和数据域为前提。作者说明使用 Eigenmike 的指定四路（按一基编号为 `[6,10,26,22]`），48 kHz、2048 点 Hann、480 点步长；两秒片段的边界帧处理与 10 fps 输出也属于复现配置。它不是无需已知几何的任意阵列定位器，两秒片段输入不能直接写成已验证实时低延迟系统。[固定 README](https://github.com/IN03X/SAID/blob/cf52ede4f38361cdbb03aa93c8254109583dfe2b/README.md)；[模型架构说明](https://github.com/IN03X/SAID/blob/cf52ede4f38361cdbb03aa93c8254109583dfe2b/docs/model_architecture.md)。

作者代码已按 `cf52ede4f38361cdbb03aa93c8254109583dfe2b` 取得；阅读入口依次为 `said/models/audio2sph.py`、`sph2imaging.py`、`said.py`、`said/inference/infer.py`、`said/evaluation/output2dcase.py`。原项目核心 MIT；组合软件还包含 Apache-2.0 来源，须保留 `THIRD_PARTY_NOTICES.md` 和 `LICENSES/`。三个作者检查点采用非商业研究许可 1.0，AudioMAE 权重另有 CC-BY-NC-4.0 条款；源码许可不能覆盖权重。此次限定取源排除了 demo 媒体和检查点。[主许可证](https://github.com/IN03X/SAID/blob/cf52ede4f38361cdbb03aa93c8254109583dfe2b/LICENSE)；[第三方说明](https://github.com/IN03X/SAID/blob/cf52ede4f38361cdbb03aa93c8254109583dfe2b/THIRD_PARTY_NOTICES.md)。原推理依赖 Python `>=3.10,<3.13`、Torch/Torchaudio 2.5.1 等；本书 Python 3.13 环境未安装该模型依赖、未取得权重、未运行预测或榜单评分。

最小已执行范围是后处理格式与体积约束。[独立调用工具](../../ch04/examples/audit_said_compression.py)直接装载未修改的 `said/utils/compression.py` 单文件，避开会载入模型的包入口；只用标准库与 NumPy。2026-10-01 运行结果见[报告](../../ch04/reports/said_compression.json)，[测试](../../../../tests/test_codes_ch04_said_compression.py)用标量坐标和手写点集另核期望。

| 人工输入／约束 | 原方法实际结果 | 支持的结论 |
|---|---|---|
| 高置信检测，能量 `[1,.5,.09,0]` | 保留 `[1,.5]`，检测 ID、类别及额外元数据保留 | 相对峰值下限为 `.1`；不删除整条检测 |
| 低置信单点 `(359,0,1)` | 原点加八个能量 `.12` 的边界支持点，横坐标绕回、纵坐标裁切 | 边界点是后处理生成，不是额外实测声学观测 |
| 两检测小 JSON，单进程目录流程 | 两检测均保留；输入 285 字节，输出 353 字节 | 边界支持可增大极小文件，压缩不保证每次更小 |
| 全零／非有限能量 | 分别抛 `ValueError` | 保留错误，未改成合法检测 |
| 人工 1 字节上限 | 逐级回退后 `RuntimeError`，未发布输出目录，无临时残留 | 执行了大小失败路径；不证明 20 MB 长录音性能 |

原默认上限为十进制 `20,000,000` 字节。论文 §4.2、Table1 的压缩/指标数据属于作者自己的开发测试预测、阈值和评分协议，本表不复现那些数字；后处理成功也不验证声学图准确率。选型时先看任务输出是否真需要实例区域，再确认阵列、数据域、推理依赖及权重条款；若只需要一个方位角，没有理由仅据名称替换本章经典 DOA 链路。

### 46. GEV：最大信噪比方向与未定尺度

对应 §5.9。GEV 最大化目标和噪声输出功率之比，输入两组厄米协方差，输出最大广义特征值对应的波束向量。pb_bss `extraction/beamformer.py::get_gev_vector` 调用广义特征值求解；ESPnet `enh/layers/beamformer.py::get_gev_vector` 提供张量实现。pb_bss 函数中的文献定位为 Warsitz 与 Haeb-Umbach 2007 年论文，归一化讨论见其 §III.A。[pb_bss 官方源码](https://github.com/fgnt/pb_bss/blob/10acc347fc9ea21e3d312806a0bd751d0d0af183/pb_bss/extraction/beamformer.py)。pb_bss 为 MIT，ESPnet 为 Apache-2.0，使用本书总清单锁定的提交。

最小手算取目标协方差 `diag(4,1)`、噪声协方差 `diag(1,2)`，两个广义特征值为 4 和 0.5，最优方向为第一通道；该向量乘 10 后信噪比不变，输出幅度却增大 10 倍。因此广义特征向量不是直接满足参考通道幅度的增强输出。加入奇异噪声协方差测试加载、数值失败及状态输出；任意改用一般 `eig` 不能证明非厄米输入合理。

固定 pb_bss 优先尝试可选 Cython 后端，否则调用 SciPy；`use_eig` 决定是否采用一般特征分解，不会修复错误的协方差模型。ESPnet 的 `get_gev_vector` 默认 `mode='power'`、3 次迭代，最后归一化并作相位连续性修正；`evd` 路径求解失败时还有替代向量分支。三次幂迭代不必等于精确最大广义特征向量，近重根、极小初始投影和参考通道都会影响结果。这里尚未用相同张量实际运行两包来比较收敛或误差。

<a id="beamformer-upstream-audit"></a>

**固定原方法实跑范围，2026-10-01。** [审计工具](../../ch05/examples/audit_upstream_beamformers.py)从所锁 pb_bss 的 `beamformer.py`、`math/solve.py` 提取 12 个原函数 AST，保留函数体和默认参数，用 NumPy 2.5.3、SciPy 1.18.1 的线性代数依赖执行 31 个确定性小输入。[历史报告](../../ch05/reports/upstream_beamformers.json)绑定当次工具与完整锁表 SHA、原 Git blob/文件与函数片段 SHA、运行前后 HEAD/clean、输入、输出、警告、异常及独立手算期望。上游目录未修改，原函数没有兼容 facade；GEV 明确关闭可选 Cython 标记，只执行 SciPy 后备路径。有效对角模型的最大特征方向为第一通道；奇异噪声模型抛出的 `ValueError` 仍记录为失败。

2026-10-04 用[当前工具](../../ch05/examples/audit_upstream_beamformers.py)再次实际调用13个未修改原函数、33个确定性输入，见[新报告](../../ch05/reports/upstream_beamformers_current.json)。增加原 `phase_correction` 的二维/三维对照；原5项异常仍保留，33项“符合预期行为”包含对原失败的核对，不能解释为33项算法成功。旧31项报告及其工具身份保持真实历史字节，独立测试从本仓库提交747ec3fec96ef20c7c128cc4291fd7bb9ee36c34读取旧工具的Git原blob，未给历史报告替换当前源摘要。

默认执行只核验并打印摘要；只有加 `--report` 才生成报告。可在已有隔离环境中复跑：

```bash
/private/tmp/masp-ch04-pra-venv/bin/python -m codes.chapters.ch05.examples.audit_upstream_beamformers
/private/tmp/masp-ch04-pra-venv/bin/python -m codes.chapters.ch05.examples.audit_upstream_beamformers --report codes/chapters/ch05/reports/upstream_beamformers_current.json
```

这个临时环境的路径不是永久安装承诺；所需包版本与真实解释器路径在报告中。该审计没有导入完整 pb_bss/ESPnet/Torch 包、运行增强波形、测论文指标或验证设备。`stable_solve` 的正常与奇异例采用它实际要求的二维列矩阵右端；直接传 NumPy 常见的一维右端会 `IndexError`，另列负例。普通数值比较使用绝对容差 `2e-14`；极小 SCM 用 `1e-303`，避免把零误判成正确的 `1e-290`。GEV 则比较特征向量张成的方向，允许任意整体符号与相位。

#### 2024～2026 年方法的有限选型补充

下列方法针对已有链路中的具体限制，不据年份或不同论文指标排序。Deng 等的正式出版元数据及 Tammen 原文相关章节于 2026-10-01 复核；其他代码和许可状态保留各条所述边界。

| 方法 | 本书收录范围 |
|---|---|
| ASA 的阵列泛化，2024 | 原理候选；未核到官方代码 |
| iDeepPE，2025 | 原理及固定源码索引；许可未明 |
| 联合学习 SCM/WNG，2026 | 已正式发表于 Interspeech 2026；arXiv v1 仍为预印本版本；未核到官方代码 |

[Tammen 等，Interspeech 2024](https://www.isca-archive.org/interspeech_2024/tammen24_interspeech.pdf) §2–3 将固定时间平均换成对瞬时 SCM 的注意力聚合，并用随机通道训练、TAC 和特征选择减弱通道数量/排列依赖。它仍通过 SCM 计算 MVDR，所解决的是移动目标下的统计量跟踪。原文式(2)可使用整段帧，不能自动声称因果；§4 是模拟移动语音叠加 CHiME-3/DEMAND 录音噪声，非真实移动说话人测试。最小选型应固定掩码，对照递归平均和注意力聚合，再改变通道排列与数目。当前未核到可取得的作者实现，故不设正文独立目录或声称已经复现。

其 §4.2 还明确：注意力估计器训练时用 oracle Wiener-like 掩码，评价时改用网络掩码。训练和推理的输入来源要分开写；原实验结果不能被解释成“训练全程没有理想掩码”。这里没有训练注意力网络或复算表 1。

[Cheong、Kim 与 Shin，SPL 2025](https://doi.org/10.1109/LSP.2025.3599455)的 iDeepPE 把波束阶段估计的语音存在概率、目标/噪声 PSD 传给后滤参数估计，补充只看单通道波束输出的信息；[作者版 §II–III](https://sapl.gist.ac.kr/wp-content/uploads/2025/08/Integrated_DNN-Based_Parameter_Estimation_for_Multichannel_Speech_Enhancement.pdf)保留 MVDR 加 LSA 结构，并采用远场的相位型 RTF 近似。最小对照应固定前级输出，仅比较独立后滤与融合估计；不能把整个网络改善归因于某个协方差公式。

[官方固定源码](https://github.com/CSeIn/iDeepPE/tree/c2cdc26ddafd33bf3bda45640c06febef9092365)中 `evaluate.py` 调用非因果 BMC-MCRA 与双网络，`make_mvdr_out.py` 用 oracle 参数准备后滤训练数据，不能当作部署推理入口。主要阅读路径是 `models/conformer_cmgan.py`、`util/oracle_test_mse.py`、`util/gain.py` 和 `evaluate.py`。固定根目录及所核核心文件未见明确代码许可，故只保留索引，不复制到本地源码集合；训练数据与权重另行核许可。未训练、推理或重测论文分数。

[Deng 等正式论文，Interspeech 2026，pp.6996–7001](https://www.isca-archive.org/interspeech_2026/deng26d_interspeech.html)，DOI [10.21437/Interspeech.2026-2212](https://doi.org/10.21437/Interspeech.2026-2212)，§2–3 将噪声估计与频率相关的 WNG 下限共同学习，并将稳健 MVDR 作为可微层。[正式 PDF](https://www.isca-archive.org/interspeech_2026/deng26d_interspeech.pdf)说明复掩码先生成噪声估计，再平均其外积得到 SCM；不是直接把任意复数掩码作为外积的权重。它针对手工固定加载/门限不能随场景调整的问题，但依赖远场、阵列几何与目标方向，仍不能消除误差集之外的失配。可选型的对照是同一噪声估计下固定 WNG 和预测 WNG 的失真、噪声及失配曲线。[arXiv:2606.24137v1](https://arxiv.org/abs/2606.24137v1)仍是预印本版本，不能与正式版页码混用。此项未核到官方代码和可复算完整配置，不进入正文独立目录；未运行网络或 QEP，不把作者结果写成本书实测。

### 47. BAN：GEV 的盲解析尺度归一化

对应 §5.9。pb_bss `blind_analytic_normalization` 使用噪声协方差计算尺度，分子为 `sqrt(wᴴ Rn² w)`，分母为 `|wᴴ Rn w|`，再乘原向量。它不需要已知目标导向，但也因此不能保证对任意真实目标传递函数满足单位响应。函数显式将零分母位置的增益置零，工程中仍应把这种退化与正常增强区分。

本书手算取 `Rn=I`、任意非零向量 `w`，该比例为 `1/||w||`，得到单位范数权重；单位范数并不等于 `wᴴa=1`。本次[原函数调用](#beamformer-upstream-audit)令 `w=[1,1]` 并分别乘 `1、2、−1、j`，前两个输出均为 `[1,1]/sqrt(2)`，后两个分别保留负号和 `j` 相位。对真实目标 `a=[1,1]`，输出响应分别是 `sqrt(2)、sqrt(2)、−sqrt(2)、−j*sqrt(2)`，没有恢复单位目标响应。全零向量返回全零，报告将其单列为退化。连续帧/频点特征向量还要处理任意相位，ESPnet 的 `gev_phase_correction` 是可阅读入口，但不能把它的平滑方向约定当成通用目标相位恢复。[ESPnet 波束源码](https://github.com/espnet/espnet/blob/be79590bb2ff26ffb01bc825c5f68cb9418b7f0d/espnet2/enh/layers/beamformer.py)。

**同名接口不一定同尺度。** 在所锁 ESPnet 提交中，`blind_analytic_normalization` 返回的是增益而不是乘完增益的向量，分母还包含通道数的平方 `C²` 及 `eps`；pb_bss 返回的则是已缩放向量，且没有这个 `C²`。忽略 `eps`、令 `Rn=I`、两通道单位范数向量，前者增益为 `1/4`、后者为 1。因此比较波形幅度前必须核对调用方施加增益的位置。ESPnet 这部分仍为固定源码静态读取，未实际调用 Torch BAN；pb_bss 的限定原函数调用不能被写成两包运行比较。

### 48. RTF：相对参考通道的传递向量

对应 §5.4、§5.9。单目标秩一模型下，目标 SCM 的主特征向量与目标传递向量同方向；有色噪声可结合噪声 SCM 的广义特征分解或幂迭代估计。pb_bss `get_pca_vector` 与 ESPnet `get_rtf` 分别提供这些入口；不能在包含多个同等功率目标的全秩 SCM 上无条件称主特征向量为“目标 RTF”。

ESPnet `get_rtf` 明确说明函数自身没有执行参考通道归一化。若估计传递向量为 `(2,1+j)`，以通道 0 为参考应得到 `(1,0.5+0.5j)`；若参考系数接近零，直接除法会放大误差。读码需继续追到消费该向量的 MVDR 层，检查参考向量、复共轭和是否重复归一化。最小实验固定目标，改变参考通道、加入近零参考响应及近简并特征值，记录相位、范数、目标响应和帧间跳变，而不是只检查输出张量尺寸。

掩码 SCM 的输入约定也不同：pb_bss 通常把时间轴放在最后，ESPnet 使用形如 `(...,F,C,T)` 的复谱与相应掩码，后者还提供通道掩码归约选项。共同的非负标量权重可保持外积和半正定；负掩码、错误通道轴或空支撑不能靠分母加 `eps` 获得物理有效性。pb_bss 的整数掩码还涉及原地浮点归一化，旧布尔转换使用 `np.asfarray`，不能未经 dtype 检查就承诺 NumPy 2 的所有输入类型可用。

本次原 `get_power_spectral_density_matrix` 调用使用一个频点、两通道和两个单位快拍，即 `X[0]=I2`，检查掩码类型与支撑。普通浮点掩码 `[1,1]` 得到 `I2/2`；整数同值掩码抛原地除法类型错误，布尔同值掩码因 NumPy 2 移除 `np.asfarray` 抛 `AttributeError`。负掩码 `[2,−1]` 被接受并得到 `diag(2,−1)`，其负特征值证明这不是有效协方差。全零支撑返回零矩阵，仍不能用于没有另行恢复策略的逆矩阵计算。

两项掩码都为 `1e-300` 时，原函数的分母下限 `1e-10` 使结果约为 `1e-290*I2`，而不是未加下限时的 `I2/2`。这是输入极小尺度和原接口下限共同导致的结果，不应套用本书另一个接口的 `1e-12`。报告保留了每个输入 dtype、输出与异常，不把强制转换后的重跑冒充原输入成功。

**Souden 与自动参考选择。** pb_bss `get_mvdr_vector_souden` 先求解 `Rn*Phi=Rs`，以 `trace(Phi)` 归一化后取参考列；参考可显式指定，也可按输出 SNR 选择。单目标秩一条件下才能把它与相应 RTF 无失真解联系起来；全秩目标 SCM 时要重新界定估计对象。其 `stable_solve` 遇奇异矩阵会逐矩阵改用最小二乘，有限返回值不自动证明无失真约束成立。ESPnet 的 Souden 路径另含加载与分母下限，极端尺度下不应假定二者逐点相等。

对应正文 [E05-20](../../../../chapters/05_beamforming.md#e05-20)，令物理目标传递向量 `b=[2,1+j]`、目标功率 3、`Rs=3bbᴴ`、`Rn=diag(2,1)`，则独立按分量求得 `q=bᴴRn⁻¹b=4`。迹化简的第 `r` 列为 `w_r=conj(b_r)*Rn⁻¹b/q`，因此 `w_rᴴb=b_r`：保持的是参考通道目标声像，不是对未归一化物理 `b` 的单位响应。

| 参考通道 | 原函数权重 | 独立目标响应 | 对应 RTF |
|---|---|---|---|
| 0 | `[.5,.5+.5j]` | `2` | `b/2` |
| 1 | `[.25−.25j,.5]` | `1+j` | `b/(1+j)` |

参考 1 的复数例子专门检验共轭因子；仅测实数参考 0 会掩盖这个错误。对 `a=b/b_r` 才有 `w_rᴴa=1`，要求 `b_r≠0`。本书独立推导和原函数实跑互证，但不把数学符号缺失的论文文本提取当作完整原式目视核验。全秩 `Rs=diag(2,1),Rn=I2` 时原迹归一化列为 `[2/3,0]`，不能要求对每个独立目标声像都无失真。自动参考的对照 `Rs=diag(1,4),Rn=I2` 实际选通道 1、输出 `[0,.8]`，说明下面 MERL 的固定缺陷不适用于 Souden 接口。

另一个名称相近的 `get_mvdr_vector_merl` 有独立问题：固定版本把每个候选参考通道的输出功率全部相加，再对单个标量 `argmax`，因此总选第 0 列。[诊断脚本](../../ch05/examples/audit_beamformer_reference.py)从固定原文件提取这个函数的原始 AST，仅提供 NumPy 依赖后执行，没有改写函数或执行整个模块；普通包导入因缺少 `paderbox` 失败。不能把这个提取调用称为原包完整运行。

输入只有一个频点，噪声 SCM 为 `I2`、目标 SCM 为 `diag(1,4)`。独立按标量功率算得两候选输出 SNR 为 1、4，应选权重 `(0,0.8)`，原函数实际返回 `(0.2,0)`；交换两路目标功率后，第 0 列恰是正确参考，得到 `(0.8,0)`。这一对测试说明固定实现的选择缺陷，不能推断同文件的 Souden 接口也有此错。[逐项报告](../../ch05/reports/beamformer_reference_audit.json)记录原文件、函数片段与诊断脚本摘要、环境和两次完整输出；上游源码保持原样，失败仍保留。

### 49. MWF、SDW-MWF 与秩一化简

对应 §5.7、§5.9。以参考通道目标声像为估计对象，一般 SDW-MWF 解由 `(Rs+mu Rn)w=Rs e_ref` 得到，其中 `mu>0`，矩阵为目标与噪声的未中心化二阶矩；零均值时也等于协方差。ESPnet `get_sdw_mwf_vector` 明确提供该解，并有 `approx_low_rank_psd_speech` 选项；当目标与噪声的互二阶矩为零时，`mu=1` 对应混合观测的普通MWF，见[正文的代价与条件](../../../../chapters/05_beamforming.md#sec-u-fb93744989)。其 `get_mwf_vector` 的参数虽名为 `psd_n`，docstring 指的是观测协方差，不能因为变量名含 n 就传入纯噪声 SCM。

pb_bss `get_wmwf_vector` 实际计算 `Phi/(mu+trace(Phi))` 的参考列，其中 `Phi=Rn^-1 Rs`，这是秩一目标条件下的化简。独立手算反例取 `Rn=I`、`Rs=diag(2,1)`、`mu=1`、参考 0：一般解第一项为 `2/3`，该迹化简第一项为 `2/(1+3)=1/2`。改成 `Rs=diag(2,0)` 后两者均为 `2/3`。本次[原函数提取调用](#beamformer-upstream-audit)得到的两个结果分别为 `[.5,0]` 和 `[2/3,0]`，与上述手算相符。一般 MWF 的期望由标量 `2/(2+1)` 独立计算，不从上游输出反造答案。这定位了模型边界，不说明秩一方法本身错误；没有运行完整外部包或增强音频链。

最小工程实验应同时比较全秩与秩一 SCM，并记录目标失真、残留噪声和所用参考。令 `mu→0` 在非奇异全秩情形接近参考通道直通，并不普遍等于 MVDR；只有相应秩一模型和极限解释下才能作该联系。源码文档里的简短等价描述应随假设阅读，不能直接复制进算法对比表。

### 50. MCRA：由局部最小值控制噪声更新

对应 §5.7。最小值控制递归平均（MCRA）不是直接把短时谱的最小值当作当前噪声，而是由平滑谱相对于局部最小值的关系估计语音存在可能性，再控制噪声谱递推。最小值窗口、谱平滑、语音存在平滑和噪声更新是不同的时间尺度。起点文献为 [Cohen 与 Berdugo，IEEE Signal Processing Letters 2002 原文](https://israelcohen.com/wp-content/uploads/2018/05/SPL_Jan2002.pdf)，§II 给递推噪声估计，§III 给由最小值控制的语音存在估计。

建议生成噪声平台先为 1、随后跳到 4 的功率序列，并插入持续窄带语音。检查噪声上升后最小值窗口多久更新，以及语音保持时噪声估计是否被污染。这个合成序列只测试跟踪逻辑，不代表语音感知质量。工业中记录初始化静音假设、窗口长度换算到毫秒的值、突发噪声响应及增益恢复时间。

### 51. IMCRA：两阶段平滑与最小值搜索

对应 §5.7。改进 MCRA（IMCRA）进一步在语音存在判断控制下执行两阶段平滑与最小值搜索，目标是降低强语音对噪声估计的污染。[Cohen 2003 作者托管原论文](https://israelcohen.com/wp-content/uploads/2018/05/SAP_Sep2003.pdf) §II～III 给出递推与两阶段搜索，§IV 总结实现，§V 评价与 OM-LSA 的联合处理。第一阶段形成粗略语音判断，第二阶段平滑排除较强语音分量，另用偏差补偿校正噪声估计；不能把任何“语音概率控制的递归平均”统称 IMCRA。

最小实验使用与 MCRA 完全相同的输入和窗长，对照两阶段的平滑谱、最小值、语音判定及最终噪声估计，并同时测试低 SNR 弱语音和噪声突然上升。若只报告更低的噪声输出，却没有目标衰减，就不能证明改进。作者软件页明确介绍 MATLAB OM-LSA/IMCRA 软件，但本次页面中未取得可核验的下载包、版本与许可，故本条保留原论文和软件入口，不以非作者 GitHub 同名代码补齐下载状态。

### 52. OM-LSA：对数谱幅度增益与语音存在不确定性

对应 §5.7。OM-LSA 把语音存在概率用于组合语音存在时的 LSA 增益和语音缺席时的增益下限；噪声估计器可用 IMCRA，但两者是不同模块。其低增益区域的处理影响残余噪声的连续性，不能仅以硬门限谱减替代。[Cohen 与 Berdugo 2001 论文入口及作者实现说明](https://israelcohen.com/software/)；[作者合著的 OM-LSA 推导 §III](https://webee.technion.ac.il/Sites/People/IsraelCohen/Publications/IWAENC2006_Habets.pdf)。

局部手算若存在时增益为 0.8、缺席时下限为 0.1、存在概率为 0.5，几何组合得到 `sqrt(0.8×0.1)≈0.283`，不是算术平均 0.45。这个例子只说明增益组合，不包含 LSA 增益内部的先验 SNR 与指数积分计算。完整实验还需保存 decision-directed 先验 SNR、后验 SNR、语音存在概率、增益下限和重叠相加条件；缺失这些状态不能算完成 OM-LSA 复现。当前没有已核许可的作者下载包随书提供。

Habets 与 Cohen 2006 的 §III 先复述标准几何组合，随后把语音缺席假设下的增益改为依赖平稳/非平稳干扰 PSD 的形式，§IV 还讨论非因果先验 SNR 估计。引用该文时应明确使用的是标准 OM-LSA 还是这个扩展，不能把所有后续公式都写成标准 OM-LSA 的定义。语音存在概率来自特定统计模型及先验，它的名称不保证在新设备、风噪或削波输入上仍校准；需同时检查语音衰减和噪声残留。

### 53. WebRTC 分位数噪声估计与语音概率控制

对应 §5.7 的工业对照。已下载 WebRTC `modules/audio_processing/ns/quantile_noise_estimator.cc::QuantileNoiseEstimator::Estimate` 跟踪分位数，`noise_estimator.cc::NoiseEstimator::PreUpdate/PostUpdate` 管理初始化和噪声更新，`speech_probability_estimator.cc::SpeechProbabilityEstimator::Update` 估计语音概率。它们提供可检查的产品代码路径，但不是 MCRA、IMCRA 或 OM-LSA 的逐式实现。[WebRTC 官方 NS 目录](https://webrtc.googlesource.com/src/+/0467d2b91cc20b9b001c2bbb73d43ea6b2491f3e/modules/audio_processing/ns/)。代码许可及 PATENTS 使用总清单记录的固定提交逐项核对。

读码须继续到 `noise_suppressor.cc` 的整帧流程，区分分析、增益施加和通道状态，再检查 `suppression_params.h` 的模式参数。建议复用噪声阶跃、持续弱语音与静音启动输入，同时检查全零谱及采样率分支。单独调用分位数模块不等于运行整个 WebRTC Audio Processing；没有 APM 集成、采样帧检查和目标设备计时，不报告实时端到端收益。

## 追踪接口、工程配置与来源核验

<a id="tracking-upstream-audit"></a>

### 54. 固定追踪上游的实际调用与失败边界

<a id="tracking-contract-audit"></a>

本节于 2026-09-28 使用既有隔离环境执行[诊断脚本](../../ch09/examples/audit_tracking_upstream_interfaces.py)，原始结果保存在[JSON 报告](../../ch09/reports/tracking_upstream_interfaces.json)。2026-10-01 的[新合同工具](../../ch09/examples/audit_upstream_tracking_contracts.py)在当前隔离环境重新调用该工具的只读函数，把当前执行结果记录在[新报告](../../ch09/reports/upstream_tracking_contracts.json)的 `historical_current_recheck`，没有改写本节的历史报告。脚本绑定自身摘要、精确输入、上游提交与被调用文件摘要，运行前核查独立 checkout、固定 HEAD 和未修改的跟踪文件。它不下载依赖，不改 FilterPy 或 Stone Soup。此前主环境没有 SciPy、未运行这些接口的记录仍然有效；本节新增的是另一环境中的有限方法调用，不能追溯性地改写旧验证范围。

2026-10-02 在同一隔离环境复验时，整工作区的稀疏选集核验与所用原文件身份分别记录。SAF、FilterPy 和 Stone Soup 的旧选集不匹配保持原状态；官方 origin、固定提交、许可与所用文件摘要、完整工作区洁净状态独立核对后，才执行上述限定方法。当前报告的 `acquisition_record` 和 `source_selection_verified` 保留取得层事实，`required_source_identity_verified` 只表示所用原文件满足执行合同。

**当前工具与历史报告分开核对。** 2026-10-05 的修订不覆盖上述两份原报告；历史工具字节固定于本仓库提交 `982ba05f7562f2e61228cbd4f6e447c8f568e55e`。新运行分别保存到[当前接口报告](../../ch09/reports/tracking_upstream_interfaces_current.json)和[当前合同报告](../../ch09/reports/upstream_tracking_contracts_current.json)。两工具默认只向终端输出，显式写报告时先检查目标，再执行计算，写入前重复有限检查。仓内只接受指定的新报告，原历史报告、源码和上游缓存均不作为输出目标。

原文件身份、完整获取选集和方法执行范围分别记录。FilterPy 预核可用的原 Python 文件，再检查实际导入模块的来源；这不表示执行了所有已核方法。Stone Soup 的三个缩减方法采用原 AST 提取调用，当前不再通过尝试导入整个包来执行无关模块。官方 origin、固定提交、所用 Git blob、许可、真实本地依赖及运行前后状态都进入当前报告；旧选集不匹配不会因限定调用成功而改变。

#### 54.1 IMM：模式先验与输出时刻

两模型都是标量 KF，初始均值 0、10，方差 1、4，模式概率 0.75、0.25；两者状态转移和观测矩阵均为 1，过程方差 0、观测方差 1。行转移矩阵第一行为 `(0.9,0.1)`，第二行为 `(0.2,0.8)`。这些是无量纲接口诊断输入，不是目标运动的标定数据。

一次交互后两模式先验为 0.725、0.275，条件均值约为 0.689655、7.272727。物理状态没有运动且没有新观测，按联合概率求和，整体均值仍应为 2.5、方差 20.5、原始二阶矩 26.75；这些是本书独立概率计算。原 `IMMEstimator.predict()` 却用旧模式概率融合，返回均值 2.335423、方差 19.600657，原始二阶矩 25.054859。再连续调用时，旧模式权重未推进，整体矩继续改变。报告逐次保留全部 `mu`、`cbar`、混合系数及子滤波器状态，不只给一个错误标签。

调用 `update(None)` 也不能假定模式证据为 1。先实际观测 0、再预测，模式先验约为 `(0.849531,0.150469)`；随后缺测调用得到模式概率 `(0.904028,0.095972)`。原因是子 KF 把新息设为零，似然却继续使用上次观测留下的新息方差。它不是直接复用上次似然标量，也不是新的观测证据。本书没有修补这一上游版本，实际系统若使用该接口，应明确缺测协议并独立验证模式概率与总体矩。

#### 54.2 EKF：同名便利接口不等于相同调用顺序

输入为均值 1、方差 1、状态转移系数 2、过程方差 0、观测方差 1，观测函数为状态平方，观测值为 4。预测均值为 2、方差为 4，观测雅可比在预测点应为 4；独立计算的增益为 `16/65`、后验方差为 `4/65`。

| 原调用 | 增益；后验方差 |
|---|---|
| `predict_update()` | 0.470588；0.235294 |
| `predict(); update()` | 0.246154；0.061538 |

合并接口先在旧状态 1 求雅可比 2，再把预测状态 2 送入平方观测函数；分步接口在预测后线性化。两条路径本例的新息均为零，所以均值同为 2，只有协方差显示差异。合并路径还把旧状态记入 `x_prior/P_prior`；因此不能只对照最终均值或字段名称。

另一例把先验设为 179°、观测设为 −179°，两方差均为 1。合并接口默认相减得到 −358° 新息；分步 `update(residual=...)` 用最短角差得到 2° 新息，更新后的局部展开均值为 180°。最终是否显示为 −180° 是另一个坐标表示步骤。

**缺测还要检查失败后的状态。** 固定原 `EKF.py::predict_update` 的说明把 `z=None` 描述为只预测，但完整函数没有该分支。对上面的平方观测模型实际调用缺测合并接口，原函数在新息相减时报 `TypeError`；此时 `x/P` 仍为 1，内部增益已为 $8/17$，新息方差已为 17。因此它既没有完成纯预测，也不能当成状态完全不变的原子失败。当前报告保留异常和失败前后字段。调用方应在缺测时明确走 `predict()`，并按自己的状态时间协议维护其余字段，不能由便利接口的名称推断缺测行为。

#### 54.3 无迹变换与高斯混合缩减

原 `MerweScaledSigmaPoints` 取一维标准正态、α=1、β=2、κ=0，sigma 点是 0、1、−1。平方变换后的 `unscented_transform` 输出均值 1、方差 2，与标准正态二阶和四阶矩独立复算一致。这只验证一个确定性变换，不代表完整 UKF 追踪已经验证。

Stone Soup 包导入实际被缺少 `ordered_set` 阻断。本节调用的是 AST 提取的未修改 `prune`、`truncate`、`merge_components`，依赖替身只保存 `mean/covar/weight/timestamp`，没有声称执行完整包、PHD 更新器或带标签分支。

| 操作与输入 | 实际输出权重 |
|---|---|
| 阈值 0.2 剪枝：0.1、0.4、0.5 | 0.45、0.55 |
| 阈值 0.2 剪枝：0.1、0.1 | 空集合 |
| 截为一项：0.4、0.6 | 1 |
| 合并两项：0.8、0.7 | 1，而非 1.5 |

合并例的位置为 0、1、方差均为 1；输出均值为 `7/15`，方差约 1.248889。均值和方差按原始相对权重算，只有输出总质量被截断。因此“位置矩匹配”与“强度质量保持”应分别验收。[离线测试](../../../../tests/test_codes_tracking_upstream_interfaces.py)不用外部源码、SciPy 或下载，按联合概率、标量 Joseph 式和解析高阶矩检查报告，并核对脚本与源配置绑定。它不把上游缺陷改成通过。

### 55. ODAS 与 SAF 的工程追踪接口

#### 55.1 ODAS：每帧参数与生命周期

[ODAS 2022 正式论文](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2022.854444/full) §2.2、§3.6 说明其追踪可选择粒子或 Kalman 分支，并将定位峰解释为已有源、新源或虚警。论文 §4 给出机器人与阵列应用，能支持该项目的部署案例，不能推出所有商业产品都使用这些默认值。本文不转引其对早期文献的 CPU 倍数；这里没有在 Raspberry Pi 或任何阵列设备上计时。

固定 `bcb845434495e293df3d48f1203b7a86e1852449` 的 `config/odaslive/respeaker.cfg` 是一份具体配置：处理采样率 16 kHz、帧移 128 样本，即 8 ms；`sst.mode="kalman"`，`add="dynamic"`。同一文件也列粒子数 1000 和有效粒子比例阈值 0.7，但在 Kalman 分支下并不执行粒子更新。`mod_sst.c` 实际用 `hopSize/fS` 构造动态模型，时间参数必须按该处理帧率解释。

`Pfalse/Pnew/Ptrack` 分别为 0.1、0.1、0.8；新源后验超过 0.9 才建立候选，候选累计 5 帧的平均活动度至少 0.8 才确认。确认源的低活动计数用 `N_inactive[nTracks-1]`，不是按轨道 ID 取固定超时。该配置为 `(150,200,250,250)` 帧，在持续低活动且当前轨道数不变时分别对应 1.2、1.6、2.0、2.0 s；轨道数变化会改变所用门限，不能写成每条轨迹固定等待 2 s。

`active/inactive` 是定位峰功率的统计模型，观测噪声另区分候选、活动源和预设目标；应由实际前端数据标定，不能把这些功率阈值换成角度置信区间。读取顺序为配置、配置解析、`mod_sst.c`、`kalman2kalman.c`/`kalman2coherence.c` 或 `particle2particle.c`。固定 C 源使用枚举分支，非法模式有进程退出路径；宿主程序不能把它当成无副作用的 Python 参数异常。生命周期与完整应用仍只读审查，未构建 ODAS 应用或运行硬件。

原 `kalman2kalman_construct` 的 `Q` 仅在三个速度对角线上放 `sigmaQ²`，`R` 的三个位置对角是 `sigmaR²`；`Q` 中没有随 `dt` 变化的连续白加速度积分项。这里的位置是方向向量坐标，不是米制声源位置。预测先计算线性 `FPFᵀ+Q`，再把均值的前三项归一到近似单位球面、后三项投影到切平面；这次非线性均值修正没有配套变换协方差。不能把它的协方差直接解释为球面约束下严格一致的后验，也不能把配置中的 `.001` 当成正文连续角加速度密度。

本轮真实编译调用原 `kalman2kalman.c`、`signal/kalman.c` 和 `utils/matrix.c`，包括原头文件；没有改源码或用替身替换矩阵运算。诊断输入为 `dt=.5`、`sigmaQ=.2`、`sigmaR=.1`、`epsilon=1e-10`、状态 `[1,0,0,1,0,0]`、协方差 `I₆`。线性预测的 x 坐标应为 1.5；原均值修正后约为 1、径向速度约为 0，而 `Pxx=1.25`、`Pxv=.5`、`Pvv≈1.04` 仍是线性预测结果。`Qvv≈.04`、`Rxx≈.01` 验证平方口径。单精度原调用按绝对容差 $10^{-6}$ 与独立代数比较；这不是关联、更新、音频整链或设备验收。

#### 55.2 SAF：关联粒子与条件 Kalman 状态

固定 SAF `18fd5aba46e20787b51f28f7197a68506c965c07` 的 `framework/modules/saf_tracker/` 以 RBMCDA 为基础：粒子保存关联及出生、死亡假设，每个目标的位置和速度由条件 KF 表示。[McCormack 等 EUSIPCO 2021 原文](https://eurasip.org/Proceedings/Eusipco/Eusipco2021/pdfs/0000206.pdf) §III 描述该分工；§IV 使用 LOCATA 任务 1–4、Eigenmike 的 32 路录音转四阶球谐，按开发集分别调各任务参数，再评价盲测集。整链结果不能归因于追踪器单模块。

`tracker3d_step` 输入为 `nObs×3` 坐标，可声明为单位方向向量；输出为目标位置、各轴方差、ID 和人数。调用者须每 `dt` 调用一次，缺测时仍传空指针或 `nObs=0`，不能停止调用而期待时钟自行推进。不过固定实现的空观测调用只增加 `incrementTime`，返回的最大权重粒子均值与方差保持旧值；下一次非空观测到来时才补做累计预测。调用时刻与输出状态所属时刻须分开记录，不能把空帧返回值直接当成新预测去控制波束。创建时实际将 `dt` 限为至少 0.0001 s、测量标准差限为至少 0.001、权重平滑系数限为不超过 0.99；最后一项与头文件注释中的 0.999 不一致，应以固定实现和输入核验为准。

近距离强制删除较年轻目标是一项可选控制策略，会影响交叉时的轨迹身份，不能与纯几何定位精度混为一谈。模块文件头授权 GPL-2.0-or-later，核心 SAF 的 ISC 许可不能覆盖它；MATLAB 封装另读 `extras/safmex/safmex_tracker3d.c/.m`。本轮未运行原粒子预测/更新、MEX 或论文 LOCATA 评分。

固定 `tracker3d_step` 在 `Neff<Np/4` 时将所有祖先索引设为当前最大权重粒子，随机重采样调用 `resampstr` 被注释。这是该版本的最大复制分支，不能描述成一般随机重采样；随后可平滑粒子权重，最终输出仍取最大权重粒子的状态与 ID，而不是对所有关联假设做混合矩。原论文 RBMCDA 的定义与这个固定分支的实现行为分别核验。

[合同工具](../../ch09/examples/audit_upstream_tracking_contracts.py)保留原 `tracker3d_step` 完整函数体，包括禁用分支；只在临时编译单元中提供结构容器、预测/更新计数器、强制 ESS 分支选择器、权重 argmax、结构复制及内存分配替身。

两个空帧均不调用预测，输出标记坐标 9、方差 1；下一帧的一条观测先触发三次累计预测，再以计数 3 调用更新。同一帧两条观测只有第一条前新增一次预测，更新的计数依次为 1、0。四粒子的标签 `[101,202,303,404]`、权重 `[.1,.1,.1,.7]` 在强制低 ESS 分支后全部复制标签 404，权重重设为 `.25`。

标记坐标只帮助观察控制流，不是物理目标运动；这些结果不证明真实 ESS、Gaussian 似然、出生/死亡或条件 KF 正确。

新报告绑定工具与完整来源锁摘要、许可、实际 HEAD、逐文件 SHA/Git blob及执行前后干净状态；[独立测试](../../../../tests/test_codes_ch09_upstream_tracking_contracts.py)分别用外积、积分、明确事件集合与计数期望核对。默认命令只输出 JSON，明确 `--report` 才保存当前报告；正式缓存缺失或未通过来源状态/摘要检查时不冒充执行成功。

### 56. Vo 作者归档：CPHD、LMB 与 GLMB 的真实取得范围

官方[工具包页面](https://ba-tuong.vo-au.com/codes.html)和包内 `readme.txt` 明确面向学术与研究；第 5 条不给运行保证，商业级代码需另联系作者。归档原链接为 `rfs_tracking_toolbox_updated.zip`，下载大小 565949 字节；本地计算 SHA-256 为：

```text
fb22c9edecb56049b7f8ede1e1522384f0c4e6bfb5e5577f3bd482f6367ef46a
```

这是本地内容摘要，不是作者签名或独立发布校验和。归档固定选集与逐文件摘要见[归档锁](../ARCHIVE_SOURCES.lock.json)及[生成的获取报告](../ARCHIVE_SOURCE_STATUS.json)。本轮取得 `cphd/glmb/lmb/jointglmb/jointlmb` 下各自 `gms/` 示例和递归依赖共 62 个文件，保存于独立的 `vo-rfs-tracking-updated` 缓存；未取得预编译 MEX、数据或图像。`_common/BFMSpathwrap.m` 自带 BSD 条件，原样保留；其他帮助函数的归属和包内研究限制也不能因“公开可下载”而省掉。

阅读每组 `gen_model → gen_truth → gen_meas → run_filter → plot_results`。CPHD 代码显式递推人数分布并计算基本对称函数，不能解释成仅对 PHD 强度换一个名字。LMB 和 GLMB 维护标签；`joint*` 是联合预测—更新实现，与分开的实现不是另一种数据集或另一个许可证。`readme` 说明本次所选线性高斯例是四维匀速状态与二维位置观测；非线性目录则是带转率的五维状态及距离—方位观测，本轮没有选取后者。

`_common/gaus_prune.m` 直接删去低权质量；`gaus_cap.m` 则按原总质量比例重标定保留分量。这与 §34 的 Stone Soup 均摊规则不同，同名缩减步骤也需要逐函数核对。GLMB 分配路径调用 `assignmentoptimal`，本轮保存了其 C 源但没有编译 MEX；绘图、随机仿真和 MATLAB 工具箱依赖未运行。取得代码使 CPHD/LMB/GLMB 从单纯原理索引增加了作者实现入口，不等于已经复现完整声学多说话人实验。

### 57. 本轮检索与有限候选决策

检索日期为 2026-09-28。关键词包括 `Julier Uhlmann 2004 unscented`、`Fortmann Bar-Shalom 1983 JPDA`、`ODAS tracking Kalman configuration`、`LOCATA evaluation toolkit`、`Neural-SRP multi-source tracking` 和 `Vo CPHD GLMB MATLAB code`。搜索摘要只用于定位，以上正式结论分别来自论文正文、作者说明或固定源码；没有用搜索排名判断先进性。

2026-10-01 的定向复核另外使用 `ODAS 2022 854444 sound source tracking Kalman particle software`、`McCormack Rao Blackwellized Monte Carlo data association EUSIPCO 2021 206`、`JPDA 1983 Fortmann Bar Shalom Scheffe Sonar tracking multiple targets joint probabilistic data association pdf`、`Vo Ma 2006 Gaussian mixture probability hypothesis density filter author pdf`、`Schuhmacher Vo Vo 2008 consistent metric performance evaluation multi object filters pdf author`、`Neural SRP 2025 2026 multi source tracking microphone code license`、`Position tracking varying number sound sources sliding permutation invariant training 2023 github license` 与 `IPDnet interchannel phase difference 2024 TASLP doi Westlake`。本次 CUHK 作者稿在 web 接口返回 502 后，经本机临时 HTTP 实际读到；OSPA 作者站该 PDF 请求为 406，Julier 的 Notre Dame PDF 为 404，JPDA 原全文未在此次取得。这些失败没有变成“已通读”；GOSPA v7 与下述神经方法原文则实际可读。固定 Neural-SRP/IPDnet README 的 web 请求本次 cache miss，未新增许可证据，因此保留原取得限制，不把旧记录写成新成功。

**GOSPA 采纳为指标扩展。** 它补充已出现的 OSPA 人数归一与漏检、虚警解释问题，现有 Stone Soup 已有接口；§36 提供可手算小例。它不是新增声源追踪网络，也不自动提供持久身份。

**Neural-SRP 保留为有明确机制的研究候选。** Grinstein 等的正式出版信息为 IEEE OJSP 5, 2024, pp.19–28，[DOI](https://doi.org/10.1109/OJSP.2023.3340057)；这里实际阅读的是[作者接受稿](https://lirias.kuleuven.be/retrieve/740352)。§III 式(9)把每个麦对的 GCC 和坐标经共享网络编码、求和，再全局解码；单向 GRU 提供时间信息。它针对固定几何网络的迁移问题，仍需已知麦位和一致特征。作者实验只支持所述源数与阵列，结尾把三个及以上同时源列为后续研究，不能把“universal”外推为任意源数或任意录音保证。

[作者仓库](https://github.com/egrinstein/neural_srp/tree/0ec639f028987ca3d9d3764331f6d65ff455f9a8)固定 HEAD 为 `0ec639f028987ca3d9d3764331f6d65ff455f9a8`。`visualize_locata.py` 与 `visualize_tau.py` 分别对应论文表 3/4 与表 5，配置通过 `params.json` 消费；这是后续最小复现入口。当前已读根说明和目录未建立代码许可，论文的 CC BY 不能代替软件授权，故未取得该源码/权重、不添加正文算法目录，也不冒称运行过预训练模型。

**沿用 icoDOA/Cross3D，暂不把 IPDnet 变成已执行追踪器。** 前者已有固定 AGPL 源码，§41 核其轴与历史边界；后者根 README 仅写 MIT，其 `IPDnet/Dataset.py` 明称修改自 Cross3D/icoCNN，却没有完整继承条件说明。这里缺少的是派生仿真代码的清楚授权记录，不是因为 AGPL 或研究限制一概禁止本地取得。网络、仿真和数据许可应分别解决，不能用已训练 DOA 输出替代跨帧身份评估。

**sPIT 仅补充训练机制线索。** [Diaz-Guerra、Politis、Virtanen 的 EUSIPCO 2023 原文](https://eurasip.org/Proceedings/Eusipco/Eusipco2023/pdfs/0000251.pdf) pp.251～255，§2～3 式(1)～(4)说明：逐帧 PIT 允许输出身份迅速交换；滑窗 PIT 先对最近若干帧的配对距离求和，再选统一排列用于当前训练损失。它改变监督训练的排列选择，不是在部署时借助真值做关联。ACCDOA 零向量可填补不活动槽位，仍有固定最大输出容量；因果历史窗和居中窗的可用信息不同。§4 的模拟多源条件不能外推为任意人数、设备或保证无身份切换。四项门槛中，问题、改变步骤和假设可由原文说明，最小两轨历史窗配对可手算，但本轮未建立作者训练代码及明确许可的复现入口，因此只留短线索，不增正文模型目录、不开新来源锁、不取权重。正文 [E09-21](../../../../chapters/09_source-tracking.md#e09-21)讨论位置尺度与可观测性，训练技巧不能消除物理观测本身的歧义。

LOCATA 官方 [I/O 框架](https://github.com/cevers/sap_locata_io)与[评价框架](https://github.com/cevers/sap_locata_eval)分开负责读写/基线和参赛输出评价；2026-09-28时未建立这两个软件仓库的完整许可，也未执行其 MATLAB 评分；2026-10-05的逐文件许可选集与静态差异见下文§61.4。已有论文中的任务、版本、阵列与真值条件仍须原样保留，不能把教学自由场移动音频称为 LOCATA 复现。上述有限清单没有声称穷尽追踪方法。

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

下表保留此前静态发现，并补充 2026-10-01 的实际方法调用。CSSM/WAVES 只执行原辅助方法，TOPS 执行原完整类；其余行按各自范围说明。它们用于限定固定实现采用范围，不据此推断原始论文有误。

| ID | 精确位置与证据 | 最小可复核配置 | 状态及下一步 |
|---|---|---|---|
| SP-CSSM-01 | `pyroomacoustics/doa/cssm.py:108` 删除 `freq_bins` 的 `invalid` 项，但第 156 行仍按 `C_hat[j]` 使用原协方差序列 | 若三频点为 `[10,20,30]`、协方差为 `[R10,R20,R30]`，首频点被删除后 `j=0` 的导向对应 20，协方差仍对应 10 | 已执行原辅助方法；人工模拟剔除首频点得 `diag(13,2,2)`，独立期望 `diag(25,2,2)`。原 `_process` 和触发剔除的声学输入未运行 |
| SP-WAVES-01 | `pyroomacoustics/doa/waves.py:104` 同样删除频点，第 148 行仍使用 `C_hat[j]` | 同上，把首频点设为无有效峰，后两频点保持不同协方差 | 已执行原辅助方法与显式对角特征子空间适配器；首行 `[3/√5,8/√10]` 与保留频点期望错位。原 `_process` 未运行 |
| SP-TOPS-01 | `pyroomacoustics/doa/tops.py:117` 构造真实 FFT bin 差，第 136 行却使用 `Phi[k]`，且构造但未使用剔除参考频点后的 `freq` | `[10,20,30]`、参考 bin 20 时，跨频差应取 −10 和 +10；下标 0、1 对应的却是 −20 和 −19 | 已执行原完整 TOPS；真方向 30°、原峰 49°、独立已知导向投影峰 30°，完整360方向谱保存。未修补上游 |
| SP-FRIDA-01 | `pyroomacoustics/doa/frida.py` 文档的 `n_rot` 默认值说明与构造函数值不同 | 直接比较文档默认与 `__init__` 默认，不需声学输入 | 已验证静态差异；本文要求显式记录 `n_rot`，不从旧 docstring 复制默认值 |
| SP-ESPRIT-01 | doatools `estimation/esprit.py:120–125` 两个移位子阵共享内存，随后原地加权 | §14 同一总体协方差的 LS/TLS 默认加权、未加权与独立复制参考 | 已实际复现默认加权失败；保留原版，未加权及独立参考只在本例通过，不是上游完整验收 |

这些源码问题不要求修改下载的上游仓库来“让演示通过”。如需修复，应创建保留原始许可证与提交号的独立补丁、列出改动原因，并采用理想多频输入及真实录音双重回归。未完成修复和回归前，不把这三个固定版本实现列作设备可直接采用的已验证定位器。

本机实际运行检查记录：2026-09-22，在仓库 `.venv` 中分别尝试导入固定版本 doatools 的 `RootMUSIC1D` 和 FilterPy 的 `KalmanFilter`，两次均在导入阶段因缺少 `scipy` 失败，未进入数值计算。计划的独立输入分别为六元半波距阵、30° 单源协方差，以及先验均值 2、方差 4、观测 3、观测方差 1 的标量更新；后者手算后验为 2.8 与 0.8。这些预期值不是已取得的外部运行结果。2026-09-24 已在**独立临时环境**安装 pyroomacoustics 0.10.0 并实际运行上述房间 RIR；仓库 `.venv` 未因此改变。截至该次历史运行，CSSM/WAVES/TOPS 只完成静态核查，不能以房间 RIR 运行替代定位接口验证。

2026-09-28 在该既有临时环境中实际执行了 §14 的 doatools ESPRIT 四路诊断，并保存独立报告；没有安装新依赖，没有修改上游源码。此次方法级计算不覆盖 root-MUSIC、稀疏求解器、FRIDA 或其他定位接口，也不改变先前两次导入失败的历史记录。源码获取状态与方法运行状态分别记录。

2026-10-01 新建的隔离环境是 `/private/tmp/masp-ch04-pra-venv`，实际安装 PRA 0.10.0 wheel、NumPy 2.5.3、SciPy 1.18.1、Cython 3.3.0、pybind11 3.1.0；主 `.venv` 和下载源码未改变。[审计脚本](../../ch04/examples/audit_upstream_doa.py)、[当次报告](../../ch04/reports/upstream_doa.json)及[独立测试](../../../../tests/test_codes_ch04_upstream_doa.py)绑定当次工具 SHA、整锁摘要、项目条目、HEAD、原 Git blob 与文件 SHA、前后洁净状态。原完整 TOPS 所用七份相关 DOA Python 文件与固定缓存逐字节一致；CSSM/WAVES 明确为 AST 方法提取，root-MUSIC 明确先记录原失败再用局部兼容门面。未运行 CSSM/WAVES 整流程、FRIDA、稀疏求解器、硬件或论文大样本基准。

```bash
/private/tmp/masp-appb-pra-venv/bin/python codes/chapters/ch04/examples/audit_upstream_doa.py --report codes/chapters/ch04/reports/upstream_doa_contracts.json
/private/tmp/masp-appb-pra-venv/bin/python -m unittest tests.test_codes_ch04_upstream_doa -v
.venv/bin/python codes/chapters/ch04/examples/audit_said_compression.py --report codes/chapters/ch04/reports/said_compression_contracts.json
.venv/bin/python -m unittest tests.test_codes_ch04_said_compression -v
```

两个工具默认只读并输出当前结果；只有显式 `--report` 才写报告，没有缺省下载、安装或上游补丁。缺少缓存或 PRA/SciPy 的环境不能声称完成原定位方法重跑；普通离线报告测试和可选原调用分开记。上述临时环境是本机路径，其他读者应另建同版本环境。此轮原始资料还实读了 Knapp–Carter 页321～323、Capon 页1409～1412、Shan–Wax–Kailath §II～III、Roy–Kailath §IV 与 TOPS §II～III；CSSM/WAVES 原论文完整正文的网络入口未能取得，Chan–Ho 和 Foy 本轮也未取得全文，因此未据摘要新增算法细节或性能数字。

## 从源代码接口到完整数学模型：四个确定性核对

2026-09-26 新增的 [`spatial_model_exercises.py`](../cross_chapter/spatial_model_exercises.py)补充以下四道题。它们调用本仓库 NumPy 教学实现，不下载新的外部算法包，不用合成音频代替协方差真值，也不声称复现了某个产品的现场性能。运行入口为 `.venv/bin/python -m codes.chapters.ch00.cross_chapter.spatial_model_exercises`，输出完整 JSON，不改写资产。

| 稳定 ID 与正文 | 先检查的模型条件 | 实际执行与独立答案 | 接到工程实现时必须保留的条件 |
|---|---|---|---|
| [E02-08：单边谱与功率](../../../../chapters/02_basics-signal-model.md) | 实样本、未缩放正向 FFT、无窗、无去均值 | 五组 DC、Nyquist、内部频点、奇数长度与补零输入；频域均方值与直接时域平方和相符 | 奇偶端点分别处理；补零保留原统计长度；PSD 还需乘 Hz 频率间距，不能与平方幅度混用 |
| [E04-11：有色噪声 MUSIC](../../../../chapters/04_doa-estimation.md) | 精确已知正定噪声协方差、一个目标、三元半波距阵 | 普通、只白化协方差、完整白化三条链的峰为 29.1°、−4.8°、0.0°；用秩一投影解析式独立核对白化矩阵 | 白化矩阵必须用于全部候选导向；噪声估计不含目标、通道和频点对应；不能把此题精确结果当作估计噪声协方差后的性能 |
| [E05-07：由 WNG 预算反推加载](../../../../chapters/05_beamforming.md) | 指定二通道协方差、无失真响应、线性 WNG 至少 1.6 | 最小绝对加载为 9，相对加载为 9/11；现有 `mvdr_weights` 与解析权重一致 | 记录加载参数的单位和尺度；用未加载协方差评分，分别报告干扰残留与白噪声增益 |
| [E12-05：奇异协方差反例](../../../../chapters/12_appendix-symbols-math.md) | 明确的 $\mathbf R=\operatorname{diag}(0,1)$、$\vec a=[1,1]^\top$ | 机械伪逆替换给出可行但非最优的 $[0,1]^\top$；直接约束解及加载极限为 $[1,0]^\top$ | 不能从“可计算且满足响应”推出最优；须检查零空间与目标导向的关系，不泛化成“伪逆均失效” |

复查代码时，特别留意行向量接口：本例候选表每一行存的是列导向的转置，所以变换写作 `dictionary @ whitener.T`。若误写成 `whitener.conj().T`，对非对角复白化矩阵会得到另一组导向。后续 MUSIC 投影再按定义共轭，不能提前多取一次共轭。

FFT 接口约定由 [NumPy `rfft`](https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html)与 [SciPy `periodogram`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html)官方文档核实；代码不依赖 SciPy。白化矩阵的定义另见 [MathWorks `whiteningmat`](https://www.mathworks.com/help/phased/ref/whiteningmat.html)。三处均于 2026-09-26 核实。本节其余数值为正文公开输入的本书推导；[独立测试](../../../../tests/test_codes_spatial_model.py)核对闭式答案、零空间反例、端点与接口拒绝行为，不能替代真实录音和有限样本实验。

## 收录边界

除 §19 已逐项列出的 SBL 与 RobustSBL 外，其他稀疏贝叶斯变体、原子范数、完整最坏情形稳健波束以及本文未逐项收录的神经定位方法，仍需要逐论文确认代码与模型；不能用同名 GitHub 搜索结果替代作者来源。本文已分别记录实际取得的外部实现、官方实现存在但再分发许可未建立的索引、只核实到原理的算法和版本疑点。深度波束与 WPD 的完整增强链、分离及神经噪声抑制见本目录其他研究文档；同一函数在多个算法条目中被调用，不等于取得了多套独立工程系统。本次没有声称穷尽定位算法领域。


<a id="imaging-contract-audit"></a>

## 声学成像：原始方法、固定源码与实际执行边界

对应[扩展专题Ⅰ](../../../../chapters/14_acoustic-imaging.md)，完整输入、推导和16道分步题在那里展开。本节用于选择原始资料、定位源码接口和区分实际执行范围。声学成像输出频率与空间网格上的源量估计；它与输出目标语音波形的增强链具有不同的输出合同。

### 原始资料怎样连接到实现

| 原始资料 | 本轮读取的位置与模型 | 正文采用与边界 |
|---|---|---|
| [Brooks、Humphreys，AIAA 2004-2954 DAMAS 原文](https://ntrs.nasa.gov/api/citations/20080015889/downloads/20080015889.pdf) | 非相干源、扫描图与点扩散矩阵、非负迭代；原外层迭代包括前向和反向扫描 | 保留一般对角除法与两种轮次；NASA 重印记录年份不改写为论文首次年份。DAMAS 截断更新不能直接称为最小化扫描平方残差的 NNLS |
| [Sijtsma，NLR-TP-2007-345](https://reports.nlr.nl/server/api/core/bitstreams/813b0521-b37c-4aff-be61-7a8bceaa06d4/content) | §4.6 式(32)～(34) 的完整CSM分量剥离，与删对角后需要额外迭代的分支分开 | 正文实现作者 full-CSM 教学式，推导剩余矩阵的半正定条件；扣出的相干分量不自动等于一个真实源 |
| [Yardibi 等，2008](https://doi.org/10.1121/1.2896754) | §V 式(22) 含非负源量、非负噪声量与源量总和约束；相关源扩展改变模型 | 本书小型无噪声/已知白噪声拟合只用于目标比较，不称为完成原 CMF-C 或全部约束优化 |
| [Chardon、Picheral、Ollivier，DAMAS 与协方差拟合分析](https://gilleschardon.fr/papers/damascmf.pdf) | 比较 DAMAS 的二次型驻点和协方差残差，等价性需要相应 Gram 模型 | 明确扫描 NNLS、DAMAS 更新与完整CSM拟合的目标差别，不由同名变量或非负输出推断等价 |
| [Sarradj，BeBeC-2012-11](https://www.bebec.eu/fileadmin/bebec/downloads/bebec-2012/papers/BeBeC-2012-11.pdf)；[2012 期刊论文](https://doi.org/10.1155/2012/292695) | 三维位置和不同导向归一化的量纲与峰行为 | 采用球面传播与显式源参考位置；单位响应不意味着多源混合峰必在真位置。两篇原文分别引用 |
| [NASA 2017 阵列校准报告](https://ntrs.nasa.gov/api/citations/20170006081/downloads/20170006081.pdf) | 通道、阵列几何及测量环境的校准条件 | 工业配置须保存幅相校准、温度、流动与坐标条件；本书数字WAV不能证明绝对声压校准 |

### 固定 Acoular 26.08 的源码身份

原项目为[Acoular](https://github.com/acoular/acoular/tree/13d3d7df74ac1a8135c7ec71da098cbbc03d8652)，固定提交 `13d3d7df74ac1a8135c7ec71da098cbbc03d8652`，BSD-3-Clause。已有下载缓存位于 `codes/chapters/ch00/upstream/_downloads/acoular/`，由来源锁表管理，不把忽略缓存重复提交到本书树。许可文件SHA-256为 `b5bc3bfa7c76d388170a8f29f8dc3047bc3ca0abcd3160781f4ec54d3e95f69f`。

[本轮合同工具](../../ch14/examples/audit_upstream_imaging_contracts.py)在执行前后分别核官方origin、固定HEAD、所用文件的原Git blob和SHA、普通文件身份与工作区洁净。`fbeamform.py`、`fastFuncs.py`、`spectra.py`、`version.py`和LICENSE逐项身份随[当前实际报告](../../ch14/reports/upstream_imaging_contracts.json)保存。完整来源选集仍为 `source_selection_mismatch`，这是获取范围状态；所用原文件身份通过不把完整选集改成成功。

本轮有限调用采用原AST方法体，移除JIT装饰器，以协议对象代替Traits构造，自定义网格驱动代替原JIT分派，NumPy FFT代替SciPy FFT。原数学方法体没有修补；这些调用没有覆盖完整包、Numba并行、HDF5缓存、完整风洞数据链或CMF估计器。工具默认只读stdout，显式 `--report` 才写当前报告。

```bash
.venv/bin/python -m codes.chapters.ch14.examples.audit_upstream_imaging_contracts
.venv/bin/python -m codes.chapters.ch14.examples.audit_upstream_imaging_contracts --report codes/chapters/ch14/reports/upstream_imaging_contracts.json
.venv/bin/python -m unittest tests.test_codes_imaging_contracts -v
```

### 二十一项限定控制的实际结果

报告记录21项数值控制：13项与所声明的独立目标一致，8项观察到原实现行为与比较目标不同；另有1个CMF估计器未运行。审计完成不代表21项都是方法正确性通过。下表保留重要差别及其使用条件。

| 合同 | 实际原方法结果与独立控制 | 使用限制 |
|---|---|---|
| 完整CSM扫描与PSF | 两格PSF为 `[[1,.25],[.25,1]]`；原DAMAS类由扫描图初始化，前向20次得到本控制源量 `(1,.2)` | 正文部分例采用零初始化和 `(1,.25)`，不能把不同输入或双向轮次混作同一运行记录 |
| GS与扫描NNLS | `b=(1,0)` 时原截断GS为 `(1,0)`，平方残差 `1/16`；独立扫描NNLS为 `(16/17,0)`，残差 `1/17` | 这是目标差异，不凭非负输出称原GS实现了扫描NNLS |
| 原full CLEAN-SC | `C=2·ones(2,2)`、扣除比例 `.6`：2步原输出 `2.10176635`，作者full式 `1.68`；20步原输出 `4.81250286`，作者式 `1.99999998` | 原 `r_diag=False` 仍保留删对角分量的内部迭代。上游不改写，正文使用作者full式；原 `r_diag=True` 对称限定控制一致，不推广到任意输入 |
| 对角删除与裁零 | 本控制原输出 `(.9,0)`，第二格裁零前为 `−.3` | 裁零会隐藏负扫描量；不证明删对角观测仍是正定功率矩阵 |
| CMF半三角目标 | `a=(1,2)`、`R=diag(1,10)`，原字典的未加权半三角独立最优为 `41/21`；完整Frobenius目标为 `41/25` | 原字典非对角实虚分量没有√2权重。只执行原字典/观测向量方法，未执行原估计器 |
| CMF默认截距 | 固定源码构造 `LinearRegression(positive=True)`，未显式取消默认截距；含截距数学解 `87/35`、截距 `−8/5` | [官方接口](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)默认 `fit_intercept=True`。本环境没有sklearn，数字是独立数学预期，不能写为原估计器实测 |
| 谱完整帧数 | 128点内部整数频余弦原积分 `.5`；200点记录实际只消费一帧，原除数1.5625导致积分 `.32` | 评分按实际完整帧数；没有把部分尾帧当完整帧运行 |
| 单边谱端点 | 同128点矩形窗的单位DC和Nyquist信号，原积分均2，时域均方均1 | 端点不能使用内部正频率的翻倍因子；未以更改上游掩盖结果 |

独立合同测试还覆盖错误origin、HEAD、文件摘要、脏工作区、路径与报告写入边界。测试使用本地Git夹具；原实际数值结果由工具调用取得，不用夹具成功替代外部运行。

### 工业入口与先进方法怎样继续研究

固定源码中的风洞入口为 `examples/wind_tunnel_examples/example_airfoil_in_open_jet_freq_domain_methods.py`，另有CMF、导向与区域选择示例。工业复现应保存测量通道次序、校准文件、CSM窗与完整帧计数、网格/声源参考、剪切层或流动模型、对角处理、正则与区域积分。源码的下载入口使用浮动分支，配套数据许可须独立核实；本轮未取得风洞数据或运行该整链，不记录设备性能排名。

DAMAS-C/CMF-C用于允许源间相关的模型，HR-CLEAN-SC关注多个分量与峰选择，SODIX引入不同的源表示和约束，移动/旋转源还需要运动轨迹与接收时刻模型。它们应逐一核观测、目标与额外假设，不能只更换算法名就沿用本章非相干PSF矩阵。上述扩展本轮仅保留研究入口，没有宣称运行。近场声全息的重建面与正则逆问题、DCASE的语义事件输出也不由本章功率成像覆盖。


## 58. 直达路径证据：DPD筛选与DP-RTF不是同一计算

定位器面对的首先是麦克风总声场，不是已经分好的直达声。高相干说明两路在该频点近似由同一个复比例解释；无噪单频信号二阶矩秩一说明该频点的有效空间分量只有一个。若同一声源的直达和反射仍相干，这两个性质都可能成立，却无法据此确认该分量的传播方向。第4章E04-24以两麦半波距、4kHz直达与同源反射给出有限假峰；[四个PCM控制](05_exercises_and_audio.md#sec-u-ed41d4e99e)让模型、实文件和评分互相核对。

### 58.1 先区分筛选、特征估计与方向匹配

直达占优检验（direct-path dominance test，DPD）筛选较符合单一主导传播方向的时频区域，然后定位器消费这些区域。它通常改变哪些观测进入方向估计，并不自动分离或输出干净语音。通过阈值只说明通过当前模型下的检验，不能称为已证明“无反射”。

直达路径相对传递函数（direct-path relative transfer function，DP-RTF）则尝试估计两麦直达响应的复比，再与候选方向响应匹配。复比同时包含相对幅度与相位；将相位直接变成角度仍需要阵列几何、传播正号、频率、可辨方向域及相应响应模型。HRTF表、通道校准和房间条件都会改变匹配关系。

| 路线 | 处理的对象 | 新增的模型条件 | 本书实际范围 |
|---|---|---|---|
| 平滑后奇异值比DPD | 球谐声场的局部空间统计 | 球阵编码、径向补偿、跨频同坐标与足够局部样本 | 原论文与后续作者说明；未实现完整DPD |
| 声场指向性DPD | 球谐系数对应的扫描声场 | 阶数、球谐声场模型、方向性度量和阈值 | 2018作者替代方案；不能机械要求同一频率平滑 |
| DP-RTF | 双通道CTF首系数比 | 跨帧模型、首段与直达关系、PSD估计及噪声条件 | E04-25只做精确人工帧CTF交叉关系 |
| 普通单频相干度/秩检查 | 麦克风总声场 | 单频二阶统计 | E04-24证明不能单凭它确认直达 |

第一行的原始入口为[Nadiri、Rafaely 2014](https://doi.org/10.1109/TASLP.2014.2337846)。本轮取得机构摘要和作者2018稿中的原方法说明，没有取得2014全文，不补写未经实读的实验细节。后续[Rafaely、Alhaiany 2018正式论文](https://doi.org/10.1016/j.sigpro.2017.08.010)及[作者上传稿](https://arxiv.org/html/2310.03688v1)§II～IV区分奇异值比检验与声场指向性替代；上传年份2023不改写方法年份2018。

### 58.2 为什么频率平滑也有坐标和噪声条件

未补偿时，不同频率的球阵模态响应各不相同。作者方法先进行球谐表示及径向处理，再在局部频率与时间区域形成统计量；不能把任意阵列的原始协方差直接相加后沿用同一导向矢量。任意阵列的推广还涉及适当聚焦，见[2018作者研究记录](https://doi.org/10.1109/ICSEE.2018.8646090)。

理想单平面波经过合适的共同坐标表示，空间系数有固定方向结构。多方向反射的频率相位发生变化时，局部平滑可能使它们不再表现为同一个合成分量，因而奇异值结构与单平面波不同。这是有条件的统计区分：频率跨度过小、延迟差过小、反射方向相近或样本不足时，都可能仍难以区分。

径向补偿、白化或一般聚焦也会变换噪声。如果变换不是酉矩阵，原白噪声不一定仍为白噪声；奇异值阈值与MUSIC噪声子空间必须消费正确的噪声模型。[E04-19](../../../../chapters/04_doa-estimation.md#e04-19)已有非酉噪声控制，[E04-22](../../../../chapters/04_doa-estimation.md#e04-22)只用已知酉置换展示跨频秩改变，二者均不是完整DPD实现。

2018声场指向性替代在相应球谐模型下避免该频率平均与特征分解。它的扫描度量和计算优势依赖阵列阶数、网格与实现；不能把论文条件下的优势写成任意两麦、任意芯片的耗时保证。本书不把这两种检验混成一个必然步骤表。

### 58.3 DP-RTF的首系数从哪里来

[Li等2016作者论文](https://www.gipsa-lab.grenoble-inp.fr/~laurent.girin/papers/Li_et_al_TASLP_2016.pdf)，DOI [10.1109/TASLP.2016.2598319](https://doi.org/10.1109/TASLP.2016.2598319)，§II～III式(3)、(7)～(12)把时域长路径近似为逐频跨帧CTF，再用两通道交叉关系估计归一化系数。跨频项与可能的非因果帧项不是因为写了CTF就消失，采用它们的近似要说明窗、帧移及路径条件。

CTF首系数是RIR第一段与分析/合成窗相应权重的组合，不是无条件等于某一个时域直达抽头。若直达和最早反射的间隔足够大，适当短窗的首段才更接近直达响应；窗覆盖早反射时，首系数比也会被污染。窗变短还影响频率分辨率及谱估计方差，不能简单宣布越短越好。

令双通道人工帧模型为x=a*s、y=b*s，其中卷积发生在帧序号上。交换卷积得a*y=b*x；令a0非零，再将其他系数除以a0，得到线性回归。这里是普通乘法与转置，不能把交叉关系误改成复内积的共轭转置。E04-25给出Q=2的三行设计矩阵、消元、行列式与条件数，最后比较首比1+j和人工帧调制频率π/2处的整路径比0.1+0.8j。

真实算法不能直接把含噪x/y当无误差回归列。原文随后形成PSD/互PSD统计，并用不同帧区间的差分处理相应噪声项；其成立依赖噪声平稳性、源噪不相关与可用观测。有限平均残余、语音活动检测失误、时变路径和病态设计矩阵仍会造成误差。单一持续复指数使设计行成比例，样本多不自动补足列秩；a0接近零时归一化也会放大误差。

### 58.4 作者实现的逐文件入口与未运行边界

固定作者仓库为[Audio-WestlakeU/DP_RTF_SSL](https://github.com/Audio-WestlakeU/DP_RTF_SSL/tree/b83e6e672f8248a11feb326d44aef6a333cbc709)，完整提交`b83e6e672f8248a11feb326d44aef6a333cbc709`。本轮静态读完整四文件树，逐文件SHA保存在来源锁的`dprtf-ssl`条目；没有建立LICENSE/SPDX授权，所以只索引，没有把作者代码复制入本书发布目录。论文许可不能替代软件许可。

本机另将这四个固定原blob导出到Git忽略的`codes/chapters/ch00/upstream/_downloads/.research-only/dprtf-ssl-b83e6e672f82/`，保留原字节及`RESEARCH_ACQUISITION.json`逐文件摘要。它是供本地静态研究的文件选集，不是Git工作树，不随本书发布，也不改变受管理来源的index_only状态；未运行MATLAB或取得方向匹配数据。

| 固定文件 | 实际承担的步骤 | 需要检查的合同 |
|---|---|---|
| `stft.m` | 使用调用方传入窗的STFT与重叠平方和归一化 | Hamming窗由DP_RTF.m选择；16kHz下256点窗、64点帧移；端点与轴序 |
| `MCMT.m` | 基于Erlang与最小值统计的语音/噪声阈值 | 输入sl/st/fn形成th_s/th_n；延迟向量与D段PSD平均在DP_RTF.m，不由此文件配对 |
| `DP_RTF.m` | 静态双通道批量特征估计 | 输入N×2、语音和噪声统计、可用帧数、首系数归一化 |
| `README.md` | 作者入口说明 | 不提供完整候选方向匹配表或一套可直接评测的HRIR数据 |

256/16000=16ms，64/16000=4ms。代码D=30的时间长度是30×4ms=120ms，原注释写120s；本书按实际配置解释，不修上游。默认T60=400ms时，路径设置的约80ms除以4ms帧移对应Q=20，另有最多50的限制；这些是该实现的参数规则，不能称为普遍正确的混响路径长度。

每个频点至少需要2Q−1=39个有效语音PSD统计段才满足该代码的最低长度控制；不足时跳过该频点并保留零，不表示整批所有频点都为零。零特征可能表示此控制触发，不能直接当“方向为零”或“无声源”。每个语音统计段另按时间索引距离选最近噪声统计段，同距时取原find顺序的第一项，再作PSD差分；缺少有效统计输入不是用多复制几个帧即可解决。本轮没有MATLAB/Octave执行、训练、HRIR匹配或作者数据实验，E04-25也没有实现这些完整步骤。

## 59. ESP-SR：从Capon接口到有效的新方向观测

[官方DOA说明](https://docs.espressif.com/projects/esp-sr/en/latest/esp32/direction_of_arrival/README.html)与固定[esp-sr源码](https://github.com/espressif/esp-sr/tree/76581015af7075681814627a5bb03d2f3f328f8a)分别承担当前产品接口说明和可追溯静态阅读。锁表ID为`esp-sr-doa`，组件声明2.5.5。本轮读公开头文件、默认参数、CMake、组件配置、原测试及根许可，没有构建设备程序或运行任何硬件。

### 59.1 声道、坐标、块长与角度先对齐

当前官方说明针对ESP32-P4/S31，支持2～8麦。麦坐标以米计，采用右手坐标；数组中第m个坐标必须与输入第m声道对应。文档URL含esp32路径，不意味着该功能支持所有名字含ESP32的芯片。

输入块每声道128点，16kHz下是8ms；接口采用按通道连续的planar布局。普通多声道WAV是逐采样交织的interleaved布局，两者不一致。两通道例中，WAV的L0,R0,L1,R1必须拆成[L0,L1,…]与[R0,R1,…]，再按接口装入块；通道次序或布局错误会改变空间关系，不能当作算法失效。

默认FFT256点，物理bin间隔16000/256=62.5Hz。默认扫描频率1500～4500Hz、名义步长100Hz形成31个配置频率，但100Hz不是62.5Hz的整数倍。公开接口不足以证明它们都映射到不同的物理bin、采用哪种插值或如何重复加权；这些细节在链接核心内，不能从配置列表推算完整实现。

方位从+x轴起逆时针，默认0～350°步长10°，共36点。若本书二维ULA角θ从+y宽侧向+x测量，则在同一平面约定内θ=90°−φ，随后按当前方向域处理折返；不能只把返回数值贴进以宽侧为零点的图。平面阵列本身的上下半空间歧义也不会因输出0～350°方位而消失。

### 59.2 VAD关闭时返回旧角，不产生新观测

固定公开头文件说明，VAD=0冻结协方差更新、逆与谱等相应自适应步骤，并返回上一角度。这个值可用于显示保持，但不能赋一个新的测量时刻交给追踪器，否则同一旧信息会被重复计权，产生虚假的置信收缩。−1错误返回则应走无效观测分支，不能当作有效的−1°方向。

接入第9章时应分别保存音频块时刻、状态最近更新时刻、返回时刻和当前VAD/有效标志。启动与暖机期间要明确何时首个角度有效；保持显示、真正新方向观测和错误三种状态应分别处理。VAD强制为1的测试也不能证明静默或双讲场景下这一状态控制正确。

### 59.3 厂商测试能支持哪些结论

固定`test_doa_accuracy.cpp`使用半径5cm的四麦阵列、距离2m、12个方向，每方向64块，前10块不计，分母为12×(64−10)=648。该测试强制VAD=1；648是准确匹配与±10°计数的评分分母，原程序只打印这两个计数，没有断言它们都等于648。准确性相关的最终断言是invalid_angles为0，检查全部12×64=768个返回值，包含暖机块。厂商文档另报告exact和±10°各648/648；软件测试通过本身不能单独证明这份全正确报告。它不能代替本书实际设备结果，也不能推断任意房间、反射、信噪比、运动或校准误差下的定位率。

文档给出的400MHz条件下约0.65ms及内存配置是厂商条件报告，不是本书测量。8ms输入块长也不等于算法端到端延迟；还需记录FFT帧历史、暖机、调度、缓冲、任务阻塞和返回时刻。平均计算用时小于8ms不能单独保证每块实时期限。

### 59.4 公开接口与核心许可分别核对

根LICENSE要求在Espressif产品范围内使用，所读部分公开头文件标Apache-2.0。不能将某个头文件的许可扩展为整个产品许可，或据Capon名称假定核心C源已公开。固定CMake链接`lib/esp32p4/libesp_audio_processor.a`；本轮公开选集没有取得与该库对应的完整算法实现。

锁表只登记七个入口的固定SHA和不同许可层，`fetch_enabled=false`、`acquisition=index_only`。没有取得二进制、权重、测试音频或设备依赖，没有运行厂商648例。读者后续应在适用许可及支持设备上核布局、坐标、VAD与状态、真实频率映射和错误码，再分别测声学效果与时间期限；本书此处是接口研究，不是新增本地Capon替代实现。

本机研究文件选集位于Git忽略的`codes/chapters/ch00/upstream/_downloads/.research-only/esp-sr-doa-76581015af70/`，只导出上述七个已核原blob，包含根LICENSE及所读头文件的原许可声明。`RESEARCH_ACQUISITION.json`记录官方origin、固定提交、各blob/SHA和非发布、非运行范围。该目录不是完整1156文件树的检出，也不含链接的算法库；七个接口文件不能算作已取得整个DOA实现。


<a id="per-channel-mask-evidence"></a>

## 60. 第5章：相位连续化、逐通道掩码与工业输出边界

本节于2026-10-04定向复核，连接正文E05-23、E05-24和[音频手册§55](05_exercises_and_audio.md#phase-reference-audio)。新增内容隔离两个模型问题：GEV逐频方向的尺度/相位自由度，以及各通道预滤波怎样改变空间统计。它们仍属于本书既有波束章节，不增设独立章节。

### 60.1 原函数的维度合同与相位锚

固定[pb_bss原源码](https://github.com/fgnt/pb_bss/blob/10acc347fc9ea21e3d312806a0bd751d0d0af183/pb_bss/extraction/beamformer.py#L517-L560)的 `phase_correction` 对相邻频率向量的内积取相角，再累乘相位因子。二维输入F×C时，`axis=0`是频率；三维B×F×C时，同一行累乘却沿batch。不能用“函数名是相位校正”来推断支持所有前导批次维度。

[当前原函数报告](../../ch05/reports/upstream_beamformers_current.json)实际提取完整原函数体，不修上游。两种输入都使用三频权重符号(+,-,+)，每频两个通道相同；二维结果成为(+,+,+)，三维单batch结果却是(+,+,-)。真实目标取全1向量时，三维输出复响应为(2,2,-2)，权重平方范数均为2。范数或功率核查不能看见最后一频的反号。此处只定位固定原函数的轴合同，没有运行完整pb_bss包、神经网络或增强波形；正文三份PCM是独立的已知相位控制。

相位连续也不自动决定目标的绝对参考相位。[Warsitz与Haeb-Umbach，2006原文§2–3](https://groups.uni-paderborn.de/nt/pubs/2006/WaHa06-2.pdf)将最大输出SNR与语音失真控制分开讨论；它的归一公式与后来的实现不能只因同名BAN就互换。[2019作者机构元数据与摘要](https://graz.elsevierpure.com/en/publications/eigenvector-based-speech-mask-estimation-for-multi-channel-speech)，DOI10.1109/TASLP.2019.2941592，可核相位感知归一这一研究路线，但本次未以摘要代替全文推导或声称运行作者网络。

[Pfeifenberger等2017原ISCA论文](https://www.isca-archive.org/interspeech_2017/pfeifenberger17_interspeech.pdf)§3.1式(12)、(13)给出相位感知归一（PAN）。将该单源秩一模型的目标向量记为 $\vec a$，要求 $\|\vec a\|_2=1$ 且已选择参考复相位；非零GEV方向记为 $\vec w$，噪声矩阵为正定 $\mathbf R_n$：

$$\begin{aligned}
C_{\rm PAN}&=\frac{\vec w^H\mathbf R_n\vec a}{\vec w^H\mathbf R_n\vec w},\\
\vec w_{\rm PAN}&=C_{\rm PAN}\vec w。
\end{aligned}$$

在 $\vec w\propto\mathbf R_n^{-1}\vec a$ 这一精确GEV方向条件下，上式可恢复该已锚定目标向量的MVDR响应；任意非GEV向量不能照搬这项等价。单位范数本身没有确定全局相位：$\vec a$ 和 $e^{\mathrm j\psi}\vec a$ 同范数，估计主特征向量后仍要选参考相位。

一个独立代数控制取 $\mathbf R_n=\mathbf I$、$\vec a=[1,1]^T/\sqrt2$、$\vec w=\mathrm j\vec a$。分母为1，分子为 $-\mathrm j$，故 $C_{\rm PAN}=-\mathrm j$，新权重等于 $\vec a$，其目标响应为1。只做单位范数缩放则仍留下 $\mathrm j\vec a$，输出目标响应为 $-\mathrm j$。这是手算控制，不是原作者网络或完整PAN音频链运行。

该2017论文的式(16)使用居中的协方差窗，§6设置250ms窗与32ms STFT、50%重叠；中心窗约需半窗未来观测，还要计入帧可用时刻，不能称零前瞻。2017的逻辑回归系统与上述2019扩展论文分别登记；本次未核到许可明确且可取得的作者网络代码，没有训练、运行网络或复算论文成绩。

### 60.2 在线逐通道掩码的原始计算

[Middelberg、Voit、Doclo与Corey，2026年7月29日作者预印本v1](https://arxiv.org/html/2607.26623v1)§3式(5)、(6)、(8)先将各通道观测乘对应掩码的平方根，再对最后P帧的处理后向量外积求均值。分母为P；它与共同标量掩码在原观测外积上的加权、除以掩码总质量是不同操作。每项外积半正定，平均仍半正定，但不能据此推断保留原RTF或单源秩一结构。

正文E05-24用固定两通道、两帧控制独立复算该差别，没有训练掩码网络、估计未知目标或实现论文完整在线Souden链。原文§4–5使用16kHz、512点STFT、256点hop、平方根Hann窗和200/500/1000ms缓冲；频率上的双向LSTM与时间上的单向LSTM应分别判断。时间单向也还要等分析帧到齐，hop的16ms不能充当全部端到端延迟。论文的紧凑/分布式阵列仿真不等于实际无线时钟、传输或硬件测量。

本次原文未提供可核对、许可明确的作者源码入口，故仅登记原理与独立教学控制，没有取得或运行完整作者实现。已有[Deng等Interspeech2026正式论文](https://www.isca-archive.org/interspeech_2026/deng26d_interspeech.html)仍归§46；其复掩码先构造噪声声像，再平均外积，不当作任意复标量协方差权重，也不重复增设算法章。

### 60.3 固定SOF设计与实际输出的取点

本轮还实际重新生成[SOF当前静态合同](../../ch05/reports/sof_tdfb_design_current.json)。固定 `sof_bf_design.m` 的归一化sinc自变量与WNG分母问题保留；新增的可选仿真分支在方位循环内重置 `nmi`，循环后只留下最后一次方位的数据。这是静态控制流与独立标量控制，不是已经执行MATLAB并生成错误WAV的证据。默认 `create_simulation_data=false`，没有运行MATLAB/Octave、设计实际FIR、刷写固件或测试设备。

固定 `tdfb_generic.c` 使用每支路FIR累加、Q格式缩放、输出位图路由和饱和；设计脚本的逐麦峰位裁切、有限抽头和板端量化也与理想频率权重不同。应分别记录理论权重、导出的FIR系数、固件输出取点和数字饱和，不能仅用理想WNG曲线证明设备效果。[固定SOF目录](https://github.com/thesofproject/sof/tree/b6c6a05d52536313fe8e8752b1c4e069b1cc4002/src/audio/tdfb)

### 60.4 当前调用、历史报告与有限收录清单

[reference当前报告](../../ch05/reports/beamformer_reference_current.json)仍记录pb_bss正常导入缺少paderbox、MERL单原函数提取的误选参考，以及PRA原 `rake_distortionless_filters` 在浮点切片处真实失败、没有输出滤波器。PRA安装目录的85份Python源与固定Git blob逐字节核对；两个安装二进制只登记本机SHA，不宣称由固定Git源码复现构建。所用原文件身份与完整选集状态分栏，运行前后源码和许可保持洁净。

三个工具默认只读stdout；只有显式 `--report` 才写指定新current报告或普通外部文件。写前拒绝历史报告、仓内其他文件、缓存、符号链接父链及非普通目标；这些有限检查不保证消除并发竞态或崩溃持久性。旧三报告与旧工具Git身份保留。重新执行命令及范围见[复现手册](04_source_reproduction.md#beamforming-current-contracts)。

本轮对[重尾spiked-MVDR预印本2609.27552](https://arxiv.org/abs/2609.27552)的候选审查没有纳入正文：其通用阵列控制使用100元ULA、200快拍与重尾纹理，没有STFT语音、房间、麦克风实录或设备证据，不能移植为本书声学性能结论。WPD已经由第7章介绍并有对应练习，这里只保留跨章引用。本轮是有限问题清单的研究，不宣称穷尽不断更新的全部阵列算法。

<a id="tracking-model-lifecycle-sphere"></a>

## 61. 第9章：模式证据、生命周期与球面方向

2026-10-05 的补充对应正文 E09-24～26。已有34式和23题分别覆盖关联、人数、相关观测、尺度与迟到时刻；新增内容补齐三个独立环节，仍归第9章。完整复算步骤见[正文练习](../../../../chapters/09_source-tracking.md#e09-24)与[音频手册](05_exercises_and_audio.md#tracking-model-lifecycle-sphere-exercises)。

### 61.1 IMM 的密度归一与缺测协议

原始 IMM 的来源身份见§40。教学核[imm_teaching.py](../../ch09/core/imm_teaching.py)只维护同义标量局部角度的低、高过程方差随机游走模式，固定每步1s；它没有角速度、转弯动力学或偏置状态。这个限定使每个模式的预测、观测密度和融合矩都能手算，而不是用外给似然代替两套滤波器。

E09-24 的共同初值均值0、方差1，过程方差分别0、4，测量方差1，模式概率为0.8、0.2，行转移矩阵为 `[[.9,.1],[.2,.8]]`。观测3到来时，模式先验0.76、0.24；两个新息相同，方差分别2、6，完整高斯密度约0.0297325723、0.0769331614。后验模式概率约0.5503254309、0.4496745691。密度含 $1/\sqrt{2\pi S_j}$，不能只比较残差指数：在零新息下，方差2的密度仍是方差6的 $\sqrt3$ 倍。

融合均值约1.9496745691，方差约0.8973588741。随后两步缺测不产生任何观测似然，只推进模式 Markov 先验和条件状态；总体均值保持，方差约增至2.5564476676、4.1178098232。这里的概率变化来自转移模型，不是新声学证据。核保留对数模式质量，显示下溢和真正零先验分别处理；非法输入或不可达模式拒绝后不提交部分新状态。

固定偏置与真实运动都可能形成持续新息。增加过程方差只能让当前状态更容易移动，不能凭这一变化证明已经识别了运动或估计了传感器偏置；需要增加相应模型或独立标定证据。

### 61.2 从实际PCM观测到确认、过期与新ID

[LOCATA 原论文](https://arxiv.org/pdf/1909.01008)§VI.A 把连续观测确认、短缺测保持及终止规则引起的延迟分开；§VI.B.4 的及时性指标还依赖发声活动区间。这些原则支持检查生命周期，但不提供本书示例的统一产品阈值，也不把局部轨迹ID定义为说话人身份。

[lifecycle.py](../../ch09/core/lifecycle.py)与[实际PCM示例](../../ch09/examples/tracking_lifecycle_demo.py)复用[音频手册§31](05_exercises_and_audio.md#sec-u-eb5a66336a)的两份合成WAV。前端仍是同一组197帧、173有效观测和24缺测；不读取未来观测完成候选确认，不额外生成一套声音来增加文件数。连续3个有效帧确认、200ms有效期是显式教学规则，均不等于贝叶斯存在概率。

时钟用每秒32000个整数单位表示：16kHz样本的一半采样周期为一单位，因此窗中心与窗末可用时刻都可精确记录。过期检查采用当前数据可用时刻减最近有效观测时刻，年龄严格大于6400单位才删除，相等时保留。检查先于新观测处理；很晚才恢复的观测不能掩盖老轨已经过期。

实际事件是第0帧建立候选、第2帧在0.052s确认、第100帧在1.032s过期；第106帧建立新候选，第108帧在1.112s确认。旧ID1不复用，新候选为ID2。第100帧年龄为6593/32000s，已超过0.2s；若明确改用状态龄期协议，删除会晚两帧；不能把该协议称作默认的可用时刻龄期。这些规则只管理单条检测链的生命周期，没有运行多人关联、人物识别、Bernoulli/LMB 或 LOCATA 评分。完整事件和图77的数值报告保留时钟字段与实际PCM来源。

### 61.3 三维方向、已知姿态与一次静态vMF更新

[Traa、Smaragdis 的2014年作者原文](https://www.mit.edu/~paris/pubs/traa-mlsp2014.pdf)§2.1式(2)采用单位球面上的 Fisher–von Mises 密度，§2.2式(3)给出右手旋转，§4的动态预测另含近似。本书[spherical_tracking.py](../../ch09/core/spherical_tracking.py)仅运行已知坐标与姿态变换、归一化的局部协方差和一次独立静态似然条件化，不能称完整论文滤波器。

本书方位从+y朝+x增大，仰角朝+z增大。标准右手z轴旋转+90°会把本书方位30°变为−60°，而不是120°；把阵列向量变到世界坐标的矩阵与其逆必须区分。阵列已知姿态没有增加距离观测，也不能替代姿态误差标定。

精确极点的方位无定义；等方位、等仰角网格也不是等球面面积，积分须包含 $\cos\phi_{\mathrm{el}}$ 的面积因子。归一化向量的协方差采用 $J=(I-uu^\top)/\|v\|$ 的一阶局部近似，去除径向扰动；它没有把任意三维Gaussian变成精确球面后验。

静态例的先验浓度3、均值方向+y，独立观测浓度4、方向+x。自然参数相加得到 `[4,3,0]`，后验浓度5、单位均值方向 `[.8,.6,0]`。期望向量长度 $A(5)\approx0.800090804$，与单位方向不同。自然参数完全抵消时输出均匀分布与无定义方向；不凭空指定方位。浓度参数也不能在宽分布、极点或任意坐标下直接解释为角度方差的精确倒数。


### 61.4 LOCATA的逐文件取得范围与静态接口差异

2026-10-05重新检查官方两个仓库的固定原文件头，按明确的逐文件许可取得31个文本文件，总计211383字节。它们在`codes/chapters/ch00/upstream/_downloads/{locata-io,locata-eval}/`中分别保留独立Git工作树；不随本书提交复制，不包含LOCATA录音、模型或评分数据。全部正选集、固定提交与入口见[来源锁表](../SOURCES.lock.json)，实际离线获取状态见[SOURCE_STATUS.json](../SOURCE_STATUS.json)。

| 固定原来源 | 实际选集 | 文件许可及边界 |
|---|---|---|
| [I/O：632468f49b13ca5a9f2f51c064d91183c06622fa](https://github.com/cevers/sap_locata_io/tree/632468f49b13ca5a9f2f51c064d91183c06622fa) | 15个`.m`、82549字节，覆盖主入口及MUSIC、STFT、CSV、真值、坐标助手 | 各自原文件头声明ODC-BY1.0；不把这一声明推广为完整仓库的软件许可或数据授权 |
| [评价：2787cdabf21bc96453c0d4f3c66db9c9c9f40434](https://github.com/cevers/sap_locata_eval/tree/2787cdabf21bc96453c0d4f3c66db9c9c9f40434) | 16个文本、128834字节，包括14个作者`.m`及munkres源码/许可 | 作者文件头分别声明ODC-BY1.0；`utils/munkres/`两文件采用保留Yi Cao版权的BSD两条款，不能混称同一种许可 |

许可不明的Hungarian、MathWorks标记为内部使用的`angdiff.m`以及若干辅助脚本未纳入选集。原主入口还引用被排除的`read_participants_results`、`stats_for_paper`等；需要MAT文件、工具箱和完整运行条件。因此这些选集用于阅读原方法，不能直接作为可独立运行的官方评分包。实际核了origin、完整HEAD、逐文件SHA与原Git blob及前后洁净状态，没有执行MATLAB、原MUSIC或LOCATA评分。

静态核读发现的具体差异如下。每项仅说明固定源码的语句，不能替代其运行路径的实测，也不修改上游来消除原差异。

| 原源码入口 | 固定语句与需要核对的合同 |
|---|---|
| [mycart2sph.m 56～59行](https://github.com/cevers/sap_locata_io/blob/632468f49b13ca5a9f2f51c064d91183c06622fa/utils/mycart2sph.m#L56-L59) | `el=acos(z/r)`是从+z计的极角；`az=atan2(y,x)-π/2`从+y朝−x增大。两者不能按名字直接套成本书朝+x的方位和水平俯仰 |
| [MUSIC参数](https://github.com/cevers/sap_locata_io/blob/632468f49b13ca5a9f2f51c064d91183c06622fa/utils/MUSIC.m#L103-L119)及[stft.m 92～104行](https://github.com/cevers/sap_locata_io/blob/632468f49b13ca5a9f2f51c064d91183c06622fa/utils/stft.m#L92-L104) | 分析窗0.03fs、FFT1024；48kHz时窗1440点长于FFT。原调用使用`fft(y,num_FFT,1)`，须核截断的有效支持；时间戳加NFFT/2，不直接等于完整窗支持中心 |
| [MUSIC块时刻](https://github.com/cevers/sap_locata_io/blob/632468f49b13ca5a9f2f51c064d91183c06622fa/utils/MUSIC.m#L130-L136)及[离线插值](https://github.com/cevers/sap_locata_io/blob/632468f49b13ca5a9f2f51c064d91183c06622fa/utils/MUSIC.m#L226-L249) | 块时刻取首尾帧时间均值；插值可能使用后续bracket。原main的[242～244行](https://github.com/cevers/sap_locata_io/blob/632468f49b13ca5a9f2f51c064d91183c06622fa/main.m#L242-L244)只计MUSIC调用耗时，不能当作数据可用时钟或端到端延迟 |
| [measures.m 152～194行](https://github.com/cevers/sap_locata_eval/blob/2787cdabf21bc96453c0d4f3c66db9c9c9f40434/functions/measures.m#L152-L194) | 关联代价遍历`all_src_idx`而非仅活动源；原极角缓存`error_el_ss`赋入方位代价。需要另行真实控制确认活动与角度指标，不能据函数名保证评价正确 |
| [measures.m 198～203行](https://github.com/cevers/sap_locata_eval/blob/2787cdabf21bc96453c0d4f3c66db9c9c9f40434/functions/measures.m#L198-L203) | OSPA距离矩阵先被`(:)`拉平再传给未取得的Hungarian依赖；本书的独立集合距离不能冒充原评分器执行 |
| [measures.m 284～307行](https://github.com/cevers/sap_locata_eval/blob/2787cdabf21bc96453c0d4f3c66db9c9c9f40434/functions/measures.m#L284-L307) | `find(this_miss,first)`取首个非零缺测项，再使用全局真值时间索引。对`[1,1,0]`，独立语句追踪会给起始缺测数0而不是2；这只是静态反例，未运行MATLAB及时性评分 |

VAP起始代码检查0→1转换，初始已经活动的区间还须独立核边界。论文§VI.B.4的指标定义与这份固定程序的行为应分别记录；不能从论文定义推断代码已经正确执行，也不能由这些静态疑点声称论文结果错误。
