# 取得源码后怎样复现

## 1. 先确定复现对象

一次复现至少要固定一个算法、一个实现版本、一个输入配置和一个评价问题。以 MVDR 为例，
“复现波束形成”还不够明确；可以改成“给定二通道目标导向矢量和噪声协方差，验证权重满足单位目标响应，
并核对输出噪声功率是否与闭式解一致”。神经掩码估计、音频读写和识别后端可以在下一层实验中接入。

| 复现层次 | 必要输入 | 输出与验收 | 不能由这一层推出的结论 |
|---|---|---|---|
| 算式 | 小矩阵、向量、确定性序列 | 与手算或独立解析结果相符 | 完整论文系统有效 |
| 模块 | 音频、采样率、配置、连续状态 | 保持输入输出约定，故障实验符合预期 | 设备实时性或端到端质量 |
| 论文实验 | 数据划分、权重、配置、评分版本 | 按原文统计口径报告结果和偏差 | 另一房间/阵列具有同等表现 |
| 设备链路 | 硬件、驱动、固件、参考取点、时钟、功耗状态 | 连续运行、延迟、语音损伤、异常恢复 | 所有产品场景均已覆盖 |

分布式波形增强还要固定节点参考、广播维数、滤波版本和共同样本标签。学习入口为[扩展专题Ⅱ](../../../../chapters/15_distributed-enhancement.md)，实际原源码执行范围见本篇[固定分布式合同](#distributed-reproduction)，网络时钟与码率口径见[工业部署§8](03_industrial_deployment.md#distributed-network-deployment)。

## 2. 固定提交与本地文件

[锁定清单](../SOURCES.lock.json) 保存官方仓库地址和完整 Git 提交。获取工具检出独立仓库，
保留许可证与作者声明，不安装依赖，也不启动维护者的训练或构建程序。

在本仓库根目录执行：

```bash
.venv/bin/python codes/chapters/ch00/upstream/fetch_upstreams.py --list
.venv/bin/python codes/chapters/ch00/upstream/fetch_upstreams.py --all --report tmp/source-acquisition.json
.venv/bin/python codes/chapters/ch00/upstream/fetch_upstreams.py --verify --report tmp/source-verification.json
```

已有目录只有在来源、提交、工作树和入口文件都符合清单时才能通过核对。发现本地修改或错误版本时，
工具会保留现场并报错；应先检查修改归属，再决定建立另一个下载目录。不要把未完成目录当成可用实现。

新下载使用稀疏工作树，跳过常见音频、权重、二进制库和压缩包扩展名。它不会递归取得子模块或下载 LFS 大对象。
该策略不能识别所有资产：文本文件也可能包含模型系数，Git 对象库也可能保留上游跟踪的资源。
此前已经存在的工作树原样保留，其资产范围不由新下载规则追溯保证。若复现确实需要被省略的文件，
应先读上游对应资产说明，再在独立环境中按其方法取得。

本地下载目录位于 `codes/chapters/ch00/upstream/_downloads/`，被本仓库 Git 忽略。提交本书时提交获取工具和锁定信息，
这些上游工作树仍各自保留原始历史和许可。

## 3. 对齐数据接口

本书 STFT、定位和通用协方差接口采用 `C × F × T`：通道、频点、帧；WPE 和分离模块采用 `F × C × T`。
从前一组转到后一组使用 `spectrum.transpose(1, 0, 2)`。外部库还可能使用 `T × F × C`。
维度大小相同也可能掩盖轴序错误，因此适配层应把轴变换写成显式操作，并用各维大小不相等的输入验证。

| 项目 | 记录内容 | 最小检查 |
|---|---|---|
| 时域输入 | 通道顺序、数值范围、采样率、起始时戳 | 一个通道给脉冲，其余静音，检查通道混排 |
| STFT | 窗函数、窗长、FFT 长度、帧移、中心填充、单边谱规则 | 完整分析/合成后检查有效区间与边缘误差 |
| 时延与相位 | 互相关定义、FFT 正负号、角度方向、导向向量 | 人为延迟已知样点，检查符号与相位一致 |
| 协方差 | 轴序、共轭位置、权重归一化、加载和功率下限 | 检查厄米性、半正定与缩放响应 |
| 波束输出 | 参考通道、权重内积、频点缩放 | 检查目标约束和已知干扰方向的响应 |
| 分离输出 | 源数、排列、尺度恢复、截断长度 | 用已知两源交换输出，验证评分排列处理 |
| 在线状态 | 上一帧滤波器、噪声功率、循环网络状态、缓冲和采样相位 | 连续输入与分块输入对照；新音频流必须重新初始化 |

正确的对照顺序是先检查约定，再检查目标函数，最后比较波形或指标。不同版本的正则化和停止条件会造成
合理差异；应报告差异来源，不能只按一个固定绝对误差阈值判断所有浮点结果。

## 4. 每类算法怎样做独立实验

### 定位与波束

先用双阵元、单平面波和无噪声输入核对方向。把入射角从正方向改为负方向，再改变阵元顺序，
预期结果必须按坐标约定变化。之后增加一个已知干扰源，分别检查源数假设、协方差秩和扫描网格。

MVDR/LCMV 可以用小矩阵直接代入约束。含特征分解的外部实现要允许特征向量整体复相位不唯一，
因此比较投影矩阵、约束残差或输出功率比逐元素比较特征向量更合适。宽带方法还要核对每个实际 FFT 频点
与工作数组索引的映射，尤其是删除频带、采用稀疏频点或选择参考频率时。

### AEC 与去混响

AEC 先给确定的短 FIR 回声路径，保留播放参考，单独划出收敛段和远端单讲评分段。
再加入近端语音、固定延迟、路径突变和非线性四种改变，每次只改一种。
ERLE 不能把双讲中的近端语音解释成残余回声，也不能用参考全静音的片段计算有效消除量。

WPE 首先验证延迟回归向量、复共轭和加权正规方程。离线与在线实现因可用帧不同，不应被要求逐样点相同。
比较时记录保护延迟、滤波阶数、初始化、功率估计是否用了未来信息，以及输出相对参考的固定时延。

### 盲分离与神经系统

盲分离先在两通道两源的已知混合上检查尺度和排列，再改变源数、输入秩或混响长度。
如果迭代目标应单调，检查的是同一目标、同一归一化下的变化；不要把一项代理目标下降解释成 SI-SDR 必然提高。

神经网络系统还需固定已训练模型的参数文件（源码常称 `checkpoint`）、特征归一化、模型状态、分块长度和推理设备。训练入口存在只能说明仓库
提供训练程序；没有训练数据权限、配置与权重版本，仍不能声称论文实验已经复现。只下载结构代码时应明确记录。

### 跟踪与工程处理

跟踪器用匀速轨迹、跨越角度边界、连续缺测、两个目标交叉和误检分别测试。数据关联正确率与定位误差分开统计；
把两条轨迹身份交换后，平均位置误差可能仍很小。粒子滤波需要固定种子，并记录重采样条件。

音频工程接口至少测试静音、过载、丢帧、设备断开、采样率切换和队列积压。离线算法测试不能代替回调线程测试。
连续重采样必须保留滤波历史和分数相位；逐块重新初始化会在块边界产生跳变，即使每块单独看起来正常。

事件边界也要有独立期望。服务时间与到达周期严格相等时，一帧完成后应立即释放容量，再接收同刻到来的帧；不能让浮点乘法与累加的舍入差制造丢帧。[工程边界练习 E10-16](../cross_chapter/engineering_boundary_exercises.py)以整数刻度手算作对照，另保留略大于周期的真实超期输入。AGC 的增益下降更新则要回到解析等式：平滑系数为 1 时增益应等于目标增益；“输出有限且不削波”并不能排除数值消减导致的错误静音。

外部时间戳与模型运行时也应先检查协议。[工业研究 I01 与 I18](03_industrial_deployment.md)分别核对 PortAudio 的测点含义、ONNX Runtime 的设备缓冲绑定与同步接口。E10-17 实际运行的是构造时间戳的减法和样本映射；I18 的 GPU 绑定对照仍是实验设计。两者不能因为有可执行教学程序就统一登记为设备实验已通过。

2026-10-01的[五项固定源码合同](03_industrial_deployment.md#industrial-controlled-contracts)进一步区分“执行原控制流”和“执行真实模型”。CMSIS只编译原标量FIR/初始化，Speex只调用四个原噪声助手；WebRTC VAD把实际分类器明确换为路由计数器，RNNoise把模型处理换为身份复制，FastEnhancer把加载、会话、计时和保存换为可检查的边界。它们能核验抽头顺序、跨块状态、合法帧、首尾计数和RTF分母，不能给出ARM周期、VAD正确率或降噪质量。

本书SRO闭环的 `fit_delay_ppm` 当前直接调用[唯一NumPy拟合核](../../ch10/core/engineering.py)，使用项目 `.venv` 环境。它仍是精确合成时间戳、恒定速率和已知可解析波形的小例；统一实现并未使其变成实际设备盲估计。历史报告保留当时的路径、环境和执行条件。

## 5. 记录结果的必要字段

获取、构建和数值实验分开记录。下表以 2026-09-23 的本机记录为底稿，已累计补入后续实验；SMP-PHAT 的隔离有符号索引适配与独立 DFT 对照于 2026-09-24 完成。它不是推荐配置的性能表：

| 对象 | 源码 | 依赖/构建 | 模块与数值检查 | 设备 |
|---|---|---|---|---|
| nara-wpe 0.0.11 离线与在线接口 | 已核对固定提交与文件摘要 | 已有可选 Python 环境；非本次安装 | 手算与接口对照见下文；不代表语音质量复现 | 未测 |
| libsoxr、libebur128、libsndfile | 已核对固定提交 | 原源码隔离构建静态库；未安装系统依赖 | 分块/排空/重置、半幅响度/静音、PCM16 短读与独立读回通过 | 未测 |
| libsamplerate 0.2.2 变比接口 | 已核对固定提交 | 原源码隔离构建静态库；未安装系统依赖 | [两段已知时钟偏差实验](../../ch10/examples/run_libsamplerate_sro.py)通过，逐调用核对消费量和尾部排空；时钟估计闭环与抗混叠未测 | 未测 |
| SpeexDSP AUMDF 同步 AEC | 已核对固定提交 `8e29a256ef0235ebbe7fcb8417b5ac7731eb8307` | macOS arm64 / AppleClang 21，CMake Release 浮点共享库已构建；采样率显式设为并读回 16 kHz | [远端单讲与双讲两对官方真实配对、合成真值及半合成已知近端注入](02_aec_wpe_separation.md#aec)均已运行；真实双讲三种参考的输入/输出总功率变化分别为 2.234、2.204、2.200 dB；合成注入的输出增量投影增益为 0.878、半合成为 0.922，均另有零参考控制，不记作双讲 ERLE 或近端保留率 | 未测 |
| WebRTC AEC3 | 已核对固定源码与接口，明确官方 `audioproc_f` 的逐帧调用顺序和线性导出入口 | 固定提交完整依赖已同步；Xcode 27.0 与本机 15.4 SDK 下构建 `rtc_tools:audioproc_f` 成功，视频资源下载钩子未全部完成；构建参数与摘要见增强研究 A06 | 两对真实配对录音的线性/最终输出均已离线运行，数字仅为总功率变化；没有干净近端/回声真值，不能作 ERLE 或算法性能排名 | 未播放／录音实测，未验证实机声学回路和时钟漂移 |
| PipeWire AEC 后端 | 已核对固定源码和接口 | 系统后端版本未构建核定 | 未在本机运行 | 不适用于仅凭当前 macOS 端点枚举得出设备链路结论 |
| pystoi | 已取得 Python 核心 | 共享项目环境缺 SciPy，未安装；另有现成隔离环境 | 2026-09-28 在隔离环境作 12 次原 Python 接口受控调用；自然语音、MATLAB 对照与内部重采样路径未测 | 不适用 |
| ViSQOL | 已取得源码子集 | 未安装 Bazel，未取模型和大型依赖 | 未运行 | 未测 |
| SBL | 已核对固定提交与核心文件摘要 | 现有 Python/NumPy，无新增依赖 | 五组单次合成实验已运行；两组达到迭代上限，未记作收敛 | 未测 |
| SMP-PHAT | 已核对固定提交；原版未修改 | 隔离 FFTW 浮点库；直接编译原 C 核心；另在临时副本做两处有符号索引适配 | 原版数值等价失败；隔离适配版在两组合成四麦输入通过独立 DFT 对照 | 未测 |
| RobustSBL、BTK | 已取得固定源码 | MATLAB 或旧构建条件分别见空间研究 | 未运行 | 未测 |
| DNN 控制 AEC、联合 AEC/NR、NBSS | 已取得源码子集 | 训练/推理依赖、数据和权重未齐备 | 未运行 | 未测 |
| lib_xcore_math v3.0.0 | 已取得，与 lib_voice 依赖一致 | 未配置 XTC 与目标板 | 未运行宿主参考或 VPU 内核 | 未测 |
| 五项工业源码合同 | 五个既有锁定项目，原文件/许可SHA及Git blob逐项核验 | 2026-10-01在项目Python/NumPy环境、C11标量编译器与临时目录执行；替身单独列出 | [新报告](../../ch10/reports/industrial_contracts.json)核CMSIS整数FIR、Speex两频点助手、VAD路由、RNNoise demo、FastEnhancer包装器；非整包/模型测试 | 未测 |

“未运行”不同于“运行失败”；“存在编译器”也不同于“依赖验证通过”。获取工具报告中的 `execution: not_run`
始终描述获取工具自身没有执行上游代码，具体方法的实验结果保存在独立研究记录，不反写成整个项目已通过。

2026-09-28 在已有隔离环境对固定版 `pystoi.stoi` 做了 12 次原 Python 接口受控调用，记录异常、警告和返回值，见[工业研究 I24](03_industrial_deployment.md#i24pystoi-的-stoiestoi-与无效评分)及[I32 原始报告入口](03_industrial_deployment.md#industrial-upstream-audit)。这些输入是零信号与非退化随机信号自比较；没有运行自然语音可懂度评测、原作者 MATLAB/Octave 对照或内部重采样路径，也没有在项目共享环境安装 SciPy。ViSQOL 评分器仍未运行。

### 工程接口实验与待扩展的评分检查

前两条已实际运行；响度的半幅、分块与静音检查也已完成。固定版 pystoi 已完成上述受控接口探针，但第三条的完整自然语音评分与失败处理流程仍是实验设计；ViSQOL 仍未运行。下文不预填自然语音得分。

1. **连续重采样**：用 48 kHz 双通道已知脉冲与 1 kHz 正弦输入，显式固定浮点格式、质量档和线程数，转为 16 kHz。
   分别按整段、127 帧和 509 帧供给，记录每次 `idone/odone`，未消费的输入留到下次；结束时排空尾部。

    比较同一质量配置的有效输出长度和对齐后误差，再做中途清状态的反例。libsoxr 的 `soxr_delay` 以输出样本计，
    不直接当毫秒。`soxr_oneshot` 默认 LQ、流式创建默认 HQ，若不显式对齐，差异可能来自滤波质量而非状态错误。
    固定速率比实验也不证明时钟漂移闭环已完成。[固定 API](https://github.com/chirlu/soxr/blob/945b592b70470e29f917f4de89b4281fbbd540c0/src/soxr.h)

2. **文件帧与标量项**：构造 7 帧、2 通道 PCM16，左右通道在不同时刻分别放正负脉冲，另含 −32768 和 32767 端点。
   每次请求 3 帧，检查返回计数依次为 3、3、1、0；按帧读取应累计 7，按标量项读取应累计 14。

    浮点归一转换在相应模式下应为 PCM 整数除以 32768，因此正端点小于 1。记录实际读取模式和文件编码，
    不把末块缓冲未覆盖的部分算进录音，也不把浮点 API 返回解释成原文件为浮点编码。
    [libsndfile API](https://libsndfile.github.io/libsndfile/api.html)

3. **评分失败处理**：给评分器分别传入已对齐活动语音、长度不符、全静音和过短活动片段。
   检查输出值、警告和失败原因；无效结果用 `null` 与原因字段表示，不填 0 或最低分加入均值。

    pystoi 去静音后不足 30 个 STFT 帧会发警告并返回 `1e-5`，因此单凭“返回了有限数”不能判定评分有效；
    应按警告和有效输入条件判断，不能仅按数值等于 `1e-5` 一刀切。响度全静音的负无穷也应单独表达。
    ViSQOL 还需固定 speech/audio 模式、采样率和映射模型，不能混合不同模式的分数。
    [pystoi 固定实现](https://github.com/mpariente/pystoi/blob/74872b000753a7a42ff51aa0868af8c82c7f9053/pystoi/stoi.py)

### 已执行实验与可重跑入口

[工业接口程序](../../ch10/examples/run_industrial_interfaces.py)在独立临时目录编译三库，
随后调用[原创 C 夹具](../../ch10/examples/industrial_interfaces.c)。它不下载或安装依赖，
要求先取得锁定源码，并已具备 CMake 与 clang。构建关闭可选编解码器、OpenMP 和 soxr SIMD 路径，
因此结论限定于记录的标量、单线程配置。

```bash
.venv/bin/python codes/chapters/ch10/examples/run_industrial_interfaces.py --report codes/chapters/ch10/reports/industrial_interfaces_current.json
.venv/bin/python codes/chapters/ch04/examples/reproduce_sbl_reference.py --output /private/tmp/sbl-rerun.json
```

工业输入为 48 kHz、双通道、1 s：一通道在第 12000 帧放幅度 0.5 的脉冲，另一通道放幅度 0.1 的
1 kHz 正弦。显式 HQ 配置重采样到 16 kHz，整段及 127/509 帧供给均得到 16000 帧，
同索引最大差均为 0。排空前尚有 529、341、378 帧，说明输入已经消费完不等于输出已经齐备。

中途在输入第 24000 帧清状态而不排空，最终只得到 15566 帧。这个反例反映历史与待输出数据丢失，
不把时移主导的同索引波形差解释成音质损伤指标。

响度使用另一段 48 kHz、单通道 CENTER、10 s 正弦，
幅度从 0.1 变为 0.05 后下降 6.0206 LU；静音记录为 `null` 并保留无有效门控能量的原因。

PCM16 的七帧双通道读回得到帧数 3/3/1/0、标量项数 6/6/2/0，归一化误差为 0。
完整条件和原始数值见[工业报告](../../ch10/reports/industrial_interfaces.json)及[工业研究](03_industrial_deployment.md)。

第10章2026-10-01的[独立历史报告](../../ch10/reports/industrial_contracts.json)由[bde483保存的原合同脚本](https://github.com/nanless/microphone-array-signal-processing/blob/bde483bcc429a553aeaf8830ca3687ab46831db0/codes/chapters/ch10/examples/audit_industrial_contracts.py)生成，不改上述历史三库报告或2026-09-28的STOI/DeepFilterNet报告。2026-10-05的当前复验见[工业研究§9](03_industrial_deployment.md#industrial-current-interface-contracts)，改后工具只显式写新的当前报告。各报告分别绑定运行时的原文件、许可证、HEAD和工具/锁表摘要，保存每项代替边界与运行前后清洁状态；CMSIS的期望来自整数卷积、Speex来自手写递推，VAD及demo来自合法帧与样本计数。

FastEnhancer例使用受控假计时：1000点输入、512点窗、256点帧移、10 ms假耗时，原打印0.125而新增源时长分母为0.16；[E10-32](../../../../chapters/10_engineering-practice.md#e10-32)解释两者。原main的保存分支也在内存接收器上核对裁256点后取原长度，没有写模型输出文件。MCRA受控软更新的[E10-28](../../../../chapters/10_engineering-practice.md#e10-28)是本书给定局部统计的递推子链；Speex原助手的硬标记不能充当该原算法复现。

[SBL 程序](../../ch04/examples/reproduce_sbl_reference.py)实际调用独立目录中未改动的作者实现，
以四麦、200 个快拍、1° 网格核对两源定位。标准条件得到 −30°、30°；将两源均移出网格 0.5° 后，
各有 0.5° 网格误差。低信噪比和源数错设的两组达到 1000 次上限，仍作为未收敛结果保存。
五组共享基础随机输入，每组只有一次实现，不给成功率或普遍优劣排名。
输入轴、解析总体谱、逐次更新量及峰缺失处理见[空间研究](01_spatial_and_tracking.md)
和[SBL 报告](../../ch04/reports/sbl_reference.json)。

SMP-PHAT 的原版复现发现了失败：在本机 Apple clang/arm64 上，固定 `system.c` 第 588、824 行
先把负的舍入时延转为 `unsigned int`，再与中心索引相加，不能得到理论需要的负偏移。
例如整数部分 −6 不在无符号整数的表示范围内；这不是按模回绕的整数到整数转换。
该转换属于语言未定义行为，其他编译器上的偶然结果不能作为可移植保证。
[Clang 官方手册的浮点到整数溢出说明](https://clang.llvm.org/docs/UsersManual.html)

原版复现入口是[Python 驱动脚本](../../ch04/examples/reproduce_smpphat_reference.py)及其调用的
[C 测试程序](../../ch04/examples/reproduce_smpphat_harness.c)。它们针对未修改的固定上游源码生成
[原版报告](../../ch04/reports/smpphat_reference.json)，与下文的隔离适配脚本分别保存失败和适配结果。

因此，程序能完成编译、两个方法在选定输入上恰好给出同一峰，并不等于全方向分数与推导相符。
实验保留原版失败，再用上游实际查表索引配合独立离散傅里叶求和定位误差来源；没有修改上游算法后
将其结果冒充原版通过。完整输入、依赖、判据与诊断见[空间研究](01_spatial_and_tracking.md)。

2026-09-24 又以[独立隔离脚本](../../ch04/examples/reproduce_smpphat_portable_overlay.py)在临时副本上仅改两处负时延索引转换，保持固定上游目录不动。两组合成四麦输入的全方向分数相对独立 DFT 最大误差均小于 $2\times10^{-4}$；[适配报告](../../ch04/reports/smpphat_portable_overlay.json)单列为 `passed_patched_teaching_case`。该结论不改变[未修补原版报告](../../ch04/reports/smpphat_reference.json)的 `failed_portability` 状态，也未测试真实录音和运行时间。

[离线 WPE 对照](../../ch07/examples/compare_wpe_reference.py)只比较对齐的有效历史帧；
[在线 WPE 对照](../../ch07/examples/compare_online_wpe_reference.py)核对三个接口的实际抽头时间点及连续状态。
配置、预期值和实测数字见[增强研究](02_aec_wpe_separation.md)。它们不测试真实房间的去混响质量，
也不要求离线估计器与逐帧估计器在不同统计窗口下输出相同。

2026-09-28 的[WPE 接口审查](../../ch07/examples/audit_wpe_upstream_interfaces.py)进一步区分真正调用固定 NARA 函数、提取上游原函数后提供受控边界，以及仅沿源码追踪 ESPnet、TensorFlow、GSS、BTK20 的参数。对应[报告](../../ch07/reports/wpe_upstream_interfaces.json)不能当成上述框架整包运行或真实语音评测。

[长静音实验](../../ch07/examples/wpe_silence_boundary.py)单独记录逆协方差首次非有限的调用序号、静音后的信号恢复和环境摘要。固定版的失败作为失败保留，不修改上游再宣称原版稳定。[图 21 报告](../../ch07/reports/figure21_wpe.json)则来自本书绘图算法，指标取点、时间轴及全频统计独立说明；它与教学音频生成器采用不同窗和功率更新，不要求两者逐点相同。

下面是一份记录结构示例，不含虚构运行结果：

```json
{
  "algorithm": "method and variant",
  "source_revision": "full Git SHA",
  "configuration": "path and content hash",
  "input": {"sample_rate_hz": null, "channels": null, "dataset_split": null},
  "environment": {"python": null, "dependencies": null, "hardware": null},
  "alignment": {"delay_samples": null, "reference_channel": null, "gain_rule": null},
  "random_seed": null,
  "evaluation": {"metric": null, "regions": null, "aggregation": null},
  "status": "not_run",
  "measurements": null
}
```

`null` 表示这份结构尚未填入实验条件，不是测量得到零。研究记录可以保留未执行状态；正文性能表只引用
条件齐全的真实结果。最终记录同时保留成功和失败实验，失败原因写到具体输入或依赖，不用删除记录制造全通过状态。

比较候选时还要保存配对关系。E11-09 用同一六次会话的两列错误计数，分别报告合并 WER、四胜一负一平，以及预定单侧符号检验的 0.1875；[题干与推导](../../../../chapters/11_selection-guide.md#sec-11-6)写明独立性、平局和检验对象。它只复算本书构造的统计例子，没有执行 ASR，也不把“未达到显著性”解释成两个系统等效。

第11章的[原词编辑核合同](03_industrial_deployment.md#meeting-kernel-contracts)另有[独立工具](../../ch11/examples/audit_meeting_kernel_contracts.py)与[报告](../../ch11/reports/meeting_kernel_contracts.json)：完整包含固定 MeetEval 原头文件，原生运行普通核与时间约束核的九个小输入，手算和独立二维矩阵分别核对期望。原方法不经 AST 提取或改写；自写 C++ 调用者只适配完整词元与给定秒区间。使用 `.venv/bin/python -B -m codes.chapters.ch11.examples.audit_meeting_kernel_contracts` 时只输出 JSON，显式 `--report` 才保存当前运行结果，且编译始终在临时目录。

这个证据能揭示同词错时、仅端点相接、正重叠与空输入的固定版本行为，对应 [E11-24 词时间](../../../../chapters/11_selection-guide.md#e11-24)；[E11-23 话段边界](../../../../chapters/11_selection-guide.md#e11-23)则由本书独立枚举解释。原核工具不运行生产 cpWER/ORC-WER/tcpWER 包装器，更不运行 ASR、身份关联或设备。2026-09-28旧接口报告中的完整 Python 调用缺扩展失败保留为历史事实；新的原生两核成功不能覆盖那次失败，也不能说明完整包依赖已齐。

<a id="appendix-solver-contracts"></a>

### 附录 A：最小二乘回退、数值秩与物理目标

矩阵求解成功后还要问两个问题：结果是否满足所需的优化目标，数组类型是否保存了求解器返回的数值。[附录 A 的 E12-03](../../../../chapters/12_appendix-symbols-math.md#sec-u-3899a8c1b7)说明零残差不保证参数唯一；[E12-05](../../../../chapters/12_appendix-symbols-math.md#sec-u-935c1036ea)说明奇异协方差中的最小范数线性方程解，不自动成为约束噪声最小化的解。这两个判断分别涉及代数目标和物理目标，不能用“输出有限”替代。

本书于2026-10-02在Python 3.13.12、NumPy 2.5.3环境中运行[限定审计工具](../../appendix_a/examples/audit_upstream_solver_contracts.py)。工具完整加载固定 `pb_bss` 提交 `10acc347fc9ea21e3d312806a0bd751d0d0af183` 的[原 `pb_bss/math/solve.py`](https://github.com/fgnt/pb_bss/blob/10acc347fc9ea21e3d312806a0bd751d0d0af183/pb_bss/math/solve.py "citation")，仅调用 `stable_solve`；不做AST提取、源码补丁或算法替身，也不导入完整 `pb_bss` 包。该模块是维护者提供的通用求解助手，按原[MIT许可证](https://github.com/fgnt/pb_bss/blob/10acc347fc9ea21e3d312806a0bd751d0d0af183/LICENSE "citation")使用，不是本书自行实现的求解器。

原函数先尝试 `numpy.linalg.solve`；矩阵奇异时，逐矩阵回退到 `numpy.linalg.lstsq`。对
$\mathbf A=\left[\begin{smallmatrix}1&1\\2&2\end{smallmatrix}\right]$、$\mathbf B=\mathbf I_2$，令两行参数之和为 $s$，第一列残差最小化为 $(s-1)^2+(2s)^2$，驻点 $s=1/5$；第二列则为 $s^2+(2s-1)^2$，驻点 $s=2/5$。在每列固定和的解中，两个分量取相等值使范数最小，因此独立期望为
$\left[\begin{smallmatrix}1/10&1/5\\1/10&1/5\end{smallmatrix}\right]$。

| 原函数输入 | 实际输出或分类 | 能说明的边界 |
|---|---|---|
| 上述奇异 $\mathbf A$，浮点 $\mathbf B=\mathbf I_2$ | 最小范数矩阵与手算一致，绝对容差 $10^{-14}$ | 这个限定浮点回退例符合最小二乘目标 |
| 同一 $\mathbf A$，整数 $\mathbf B=\mathbf I_2$ | 返回整数全零；记录为 `observed_integer_fallback_truncation` | 原回退分支以 `zeros_like(B)` 建结果数组，浮点结果赋入整数数组时被截断；没有警告也不代表有效 |
| $\mathbf A=\mathbf I_2$，整数 $\mathbf B=\mathbf I_2$ | 返回浮点单位阵 | 整数右端项不是所有路径都会失败；此处没有进入奇异回退 |
| $\mathbf A=\operatorname{diag}(0,1)$，浮点 $\mathbf B=[1,1]^\top$ | 最小二乘结果为 $[0,1]^\top$ | 不代表该向量是奇异MVDR最优权重 |

最后一行的物理对照完全由本书手算：若把 $\mathbf A$ 视为噪声协方差、目标导向取 $[1,1]^\top$，约束是 $w_1+w_2=1$，输出噪声功率为 $|w_2|^2$。权重 $[1,0]^\top$ 满足约束且功率为零；归一化后的最小二乘方向 $[0,1]^\top$ 则功率为1。工具没有调用上游波束形成器，不能把这个目标差异诊断写成已运行完整MVDR链。

另外两例直接调用NumPy接口，与上游调用分栏记录。取 $\mathbf A=\operatorname{diag}(1,10^{-8})$、$\vec b=[1,10^{-8}]^\top$，显式 `rcond=1e-6` 得秩1、解 $[1,0]^\top$、残差范数 $10^{-8}$；`rcond=1e-10` 得秩2、解 $[1,1]^\top$、残差零。两次返回的残差数组都为空，因为矩阵为方阵，接口条件是 $M\leq N$。判秩和残差各自记录，依据[NumPy 2.5接口的Parameters/Returns](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html "citation")（2026-10-02核实）。`stable_solve` 本身没有向调用者暴露这两个 `rcond` 参数。

[E12-16截断与ridge](../../../../chapters/12_appendix-symbols-math.md#e12-16)进一步区分删去弱方向与连续收缩；[E12-18厄米输入检查](../../../../chapters/12_appendix-symbols-math.md#e12-18)则提醒 `eigh` 使用指定三角且忽略对角虚部，不能把分解成功当作输入协方差有效的证明。这两个练习由本书入口实现，不属于上述原函数的执行范围。[Netlib LAPACK最小二乘驱动说明](https://www.netlib.org/lapack/lug/node27.html "citation")的式(2.1)/表2.3区分满秩QR/LQ与秩亏求解，支持先说明目标、秩假设和截断策略，再选择接口；本书没有另行编译这些Fortran驱动。

```bash
.venv/bin/python -B -m codes.chapters.appendix_a.examples.audit_upstream_solver_contracts
.venv/bin/python -B -m codes.chapters.appendix_a.examples.audit_upstream_solver_contracts --report codes/chapters/appendix_a/reports/upstream_solver_contracts.json
```

默认命令只把当前审计写到标准输出；显式 `--report` 才原子保存[独立报告](../../appendix_a/reports/upstream_solver_contracts.json)。报告核对官方origin、完整HEAD、锁表摘要、两个原文件的SHA/Git blob、MIT许可、工具摘要和运行前后洁净状态，并保存四个原函数例、两个独立NumPy例的输入类型、输出、独立期望、容差、警告和失败分类。拒绝符号链接、词法 `..`、非普通目标与上游缓存内报告路径。不覆盖旧历史报告，不取得新源码；完整包、波束音频、模型和设备都未在这个审计中运行。

<a id="appendix-b-reproduction"></a>

### 附录 B：先对齐研究方向的任务与版本

[附录B的研究速览](../../../../chapters/13_appendix-guide.md#sec-13-3)包含八类方向，但它们并不输出同一种结果。目标说话人提取输出音轨，DiCoW输出指定说话人的文字，CHiME-9 ECHI输出低延迟增强流；ArrayDPS的输出又有参考通道与混响源像的定义。选型前先固定目标，不能把不同任务的WER、SI-SDR或延迟写成统一排名。

以下定位与实现边界于2026-10-02核查。论文只用实际阅读的相关节说明方法，不转引条件不齐的性能数字；各项目的完整固定提交、许可与本地选集仍以[锁表](../SOURCES.lock.json)为准。

| 研究方向 | 对基线改变的步骤与输出 | 最小复现或选型时必须固定的条件 |
|---|---|---|
| 预训练识别后端 | CHiME-7/8若干系统在分割、GSS或波束之后使用Whisper/WavLM识别 | 作者综述v1的§4.2～4.4、表4～6分别记录系统组成；固定该系统的前端、权重、分词与评分，不能仅替换后端名称后沿用成绩 |
| CHiME-8 DASR任务与基线 | Cornell等的报告介绍任务与ESPnet/NeMo两套基线，不是一个独立参赛系统 | 原文§5～6对应数据、分割与识别链；先选其中一套完整配置，再核设备物理通道和实际使用通道 |
| 目标说话人提取 | TEA-PSE 3.0用注册条件串联幅度回归与复谱恢复；CIENet做逐帧注册谱交互；SonicSieve用结构提供的方向线索 | 注册语音、提示方向或声学结构是不同额外输入。核任务采样率、目标身份和非目标泄漏；SonicSieve还需其对齐的外接声学结构麦，不能迁移为任意手机纯软件效果 |
| 生成式增强 | SGMSE+从条件随机微分方程采样；StoRM先判别式回归，再生成式恢复；FlowSep做文本查询的隐空间通用声音分离 | 固定实际采样分派、函数求值次数、输入非零条件与内容一致性；文本事件分离不自动成为多通道语音增强。没有确定权重许可和配置时只做结构检查 |
| TF-GridNet | 从复数时频表示学习目标复谱，处理块交替沿频率、时间和跨帧路径交换信息 | 会议单通道无混响版、期刊多条件扩展与ESPnet单个分离类分别固定；必须记录输入麦数、输出源数、归一化和参考，不能把一类网络当成期刊完整两阶段链 |
| 几何无关系统 | Kamo作者v2的局部分割、GSS嵌入、全局聚类/人数与TS-VAD细化构成分割链；通道选择、WPE/GSS、SP-MWF与识别组成后续链 | 按§2、§4.1～4.3.6记录活动标签、EV/C50通道选择与输出参考；几何无关不意味着任意时钟漂移、标签错误或残余错位已被解决 |
| CHiME-9与DiCoW | ECHI处理流式助听增强；DiCoW以分割活动条件控制Whisper目标转写 | ECHI规则分别定义算法延迟与输出发射延迟；DiCoW的文字输出不能用于该增强排名。固定作者演示的v1/v2入口、分割权重与目标说话人 |
| ArrayDPS | 把扩散先验与每采样步的相对RIR/FCP混合估计结合，目标为参考麦的混响源像 | 固定已知源数、同步通道、语音先验域、谱窗/帧移与停止/样本选择；400步采样和内层优化不能直接当作流式模块 |

原始阅读入口分别为[CHiME-7/8作者综述v1](https://arxiv.org/html/2507.18161v1 "citation")、[Cornell等2024原报告§5～6](https://www.isca-archive.org/chime_2024/cornell24_chime.pdf "citation")、[TEA-PSE 3.0 §2](https://arxiv.org/pdf/2303.07704 "citation")、[CIENet §2.1～2.2，式(1)～(5)](https://arxiv.org/pdf/2402.17146 "citation")、[SonicSieve作者v3 §3～4](https://arxiv.org/html/2504.10793v3 "citation")、[TF-GridNet期刊作者稿§II～III](https://zqwang7.github.io/publications/TASLP2023_TF-GridNet.pdf "citation")及[Kamo作者v2 §2、§4](https://arxiv.org/html/2502.09859v2 "citation")。这里只引用对应选节，不称本书复现了这些模型的训练或整套会议系统。

#### 有效入口、额外资产与计算边界

[SGMSE+原论文](https://arxiv.org/pdf/2208.05830 "citation")的§II-B式(5)～(7)与§III描述条件扩散和采样；[StoRM原论文](https://arxiv.org/pdf/2212.11851 "citation")应单独对应回归与再生成流程。固定SGMSE提交 `1961cf4483e37df1bb92ccf0eb8b28bf6f44cb0e` 的 `sgmse/model.py::ScoreModel.enhance` 在OUVE/SBVE分支使用 `self.sde.sampler_type`，不能只记录同名公开形参；固定StoRM提交 `257e9636a7251ca40aa200753d5c0fe918e31879` 的 `StochasticRegenerationModel.enhance` 则按其形参选择PC/ODE。二者先按整输入峰值归一化，当前入口未显式保护零峰值，并直接使用CUDA；这些是本书的静态源码核查，不是模型运行结果。完整调用还需各自依赖、权重和数据权限，见[增强研究N10](02_aec_wpe_separation.md#sec-u-867bd32313)。

ESPnet固定 `be79590bb2ff26ffb01bc825c5f68cb9418b7f0d` 的 `espnet2/enh/separator/tfgridnet_separator.py::TFGridNet.forward` 接收批量、采样点、麦克风轴，网络配置固定 `n_imics`，并以整记录的标准差缩放输入。代码中的归一化与因果性要按原实现核对，不能把注释中的RMS或网络名字当作完整接口定义。本书没有导入其Torch网络、取得权重或运行音频推理；Apache-2.0代码许可不扩展到任意训练数据。

DiCoW演示固定 `e9326bd536bf632e823357438b210102903ba620` 的 `example.py` 使用v1，`app.py` 使用v2；二者均指向Pyannote 3.1分割条件，不能因同仓提交相同而合并成同版模型。相关训练仓 `TS-ASR-Whisper` 固定 `0ea6679d44405f5ff39188030123524686c198e9` 另有身份。代码Apache-2.0、两个DiCoW模型卡CC BY 4.0、分割模型门控条件分别核查；本书没有运行权重。详细许可与目标转写边界见[增强研究N17](02_aec_wpe_separation.md#sec-u-35c49c683c)。

FlowSep固定 `d8164db58bd461ef5bb6df8ffd372b536ee6afb4` 仍按无明确代码许可的来源索引处理，未复制其源码或下载VAE/分离权重。TEA-PSE、CIENet、SonicSieve的原文阅读也不等于已有可重用的作者代码与模型。最小实验若缺少必要资产，应先在同一任务下作选型说明，不用一个未训练结构生成假成绩。[FlowSep的独立权重/数据边界](02_aec_wpe_separation.md#sec-u-372b17bd11)随其专题保留。

ArrayDPS固定 `750ac2b7c75458f4ca5bad203dafda528f575e55` 的真实采样入口是 `src/sampler_spatial_v1_reverb_iva_8kHz.py::Sampler`；`separate.py` 是数据、采样预算与评分驾驶器。原论文§2式(1)～(5)定义参考通道1的全部混响源像，式(3)的相对RIR还依赖参考响应可逆的条件。干净语音扩散先验不会自动把输出目标改成无混响波形。[PMLR正式原文](https://proceedings.mlr.press/v267/xu25f.html "citation")的附录C.3使用512点FFT/64点帧移，固定CLI类配置默认为512/128，复现应明确选定其中一套条件。

固定驾驶器读取干净参考计算SDR，并用它调整停止预算；原采样器本体没有把该真值作为输入。这一源码观察不外推为原论文不盲：论文§4.2分别报告五次样本均值、真值最佳样本与仅根据混合模型的最大似然选择。部署对照应自行声明无需干净参考的固定预算或可用选择准则，并分别计算质量与耗时。本书对这条扩散链只有固定代码的静态证据，未运行400步神经采样、权重或完整SMS-WSJ实验，详见[增强研究N11与独立报告](02_aec_wpe_separation.md#sec-u-3b78d29bc2)。

#### ArrayDPS-Refine：单目标增强的有限研究入口

[ArrayDPS-Refine作者v1](https://arxiv.org/html/2603.24385v1 "citation")的§2～3、式(1)～(16)与算法1讨论单目标增强：先把判别输出通过前向卷积预测与混合对齐，再由残差估计高斯噪声空间协方差，用这一协方差指导扩散采样；初始化也利用判别结果。最终单抽头对齐是另一步，可能继承判别输出中的错误。它针对判别增强中的非线性失真，改变似然模型与采样起点，和2025年的多源参考麦源像分离任务分别比较。

这项候选的选型前提包括目标域的语音先验、剩余噪声的高斯/协方差近似、有效多通道混合与采样计算预算。2026-10-02核查的[ICASSP 2026正式程序单篇](https://www.cmsworkshops.com/ICASSP2026/view_paper.php?PaperNum=6235&bare=1 "citation")确认该论文的会议报告身份；方法引用仍明确使用作者v1，不补未核定的出版DOI或页码。[作者演示页](https://xzwy.github.io/ArrayDPSRefineDemo/ "citation")提供方法与试听，没有给出可核代码/权重许可入口。本书只保留这项问题、步骤与假设的研究索引，未取得或执行模型，未将它加入已复现算法或低延迟基线目录。

<a id="tac-contracts"></a>

### TAC作者源：共享结构核验与未执行原网络的界线

附录B的通道置换问题需要区分两个性质：通道顺序改变时，通道索引输出按相同顺序重排，这是置换等变；跨通道平均不随顺序变化，这是聚合量的置换不变。加入一个通道通常改变平均值，所以支持可变通道数不等于数值输出不随麦数变化。[TAC作者论文v3 §2.1式(1)～(4)](https://arxiv.org/html/1910.14104v3 "citation")用共享变换、平均、拼接与残差连接说明这一步；[E13-11](../../../../chapters/13_appendix-guide.md#e13-11)分别手算交换通道、只复制一路与复制全部通道。

本书通过获取工具取得作者[固定提交 `e3373b73358a96af6f64fdbe25327def8d6bd973`](https://github.com/yluo42/TAC/tree/e3373b73358a96af6f64fdbe25327def8d6bd973 "citation")的 `README.md`、`FaSNet.py`、`utility/__init__.py`、`utility/models.py` 四个文件，独立保存在被忽略的上游工作树。README明确声明CC-BY-NC-SA-3.0-US，源代码未随本教程再分发；没有取得音频、数据、权重或另一 `iFaSNet.py` 扩展。NumPy/PyTorch是原模块的依赖，但这一审计只使用标准库解析Python语法，不导入两模块或Torch。

[只读合同工具](../../appendix_b/examples/audit_tac_contracts.py)核对origin、完整HEAD、四文件SHA与Git blob、README许可、锁表摘要及前后洁净状态，再执行十项静态AST检查。它不编译或执行原 `forward`，也没有把本书的独立NumPy夹具作为网络运行证据。

| 核查对象 | 固定源中实际结构 | 证据所能支持的范围 |
|---|---|---|
| `DPRNN_TAC.__init__`与`forward` | 每个网络块有共享的transform/average/concat层；逐通道特征展开后使用同一组层 | 核实共享结构，而非验证已训练网络的质量或任意非法输入 |
| 通道平均与拼接 | `num_mic.max()==0`取全部通道均值；否则按每项有效通道前缀取均值，再拼回各通道 | 核实原轴序和前缀协议；未实际测试混合零计数、空前缀或网络运行恢复 |
| TAC残差 | 聚合分支经过共享变换和归一化后加入当前通道表示 | 核实残差位置；数学置换例由本书单独推导 |
| `FaSNet_TAC.forward` | 用通道0中心段作为余弦特征参考，再估计各通道滤波器并平均输出 | 必须保持参考选择一致；不能由内部TAC直接证明整个包装器任意重排不变 |
| 时域与网络范围 | 作者把原两阶段FaSNet改为单阶段TAC；段内RNN是双向，使用GroupNorm | 不是原FaSNet两阶段论文全复现，也没有因果、跨调用状态或有界延迟验证 |

```bash
.venv/bin/python -B -m codes.chapters.appendix_b.examples.audit_tac_contracts
.venv/bin/python -B -m codes.chapters.appendix_b.examples.audit_tac_contracts --report codes/chapters/appendix_b/reports/tac_contracts.json
```

默认只输出JSON；仅显式 `--report` 原子保存[当前独立报告](../../appendix_b/reports/tac_contracts.json)。报告状态是 `passed_static_contracts`，保存十项结构定位与方法片段摘要，原运行调用计数为零；工具拒绝重复JSON字段、非有限数字、词法 `..`、符号链接、非普通目标和上游缓存内报告。报告不替代分离性能、权重/数据授权或设备验收。

附录B的房间仿真、分数延迟和通道聚合数学例仍按各自的输入、脚本、PCM与统计窗口解释。对照时，[E13-12](../../../../chapters/13_appendix-guide.md#e13-12)复算共同尺度不变的DRR与极端数值边界，[E13-13](../../../../chapters/13_appendix-guide.md#e13-13)条件化时间原点与跨度后外推T20，[E13-14](../../../../chapters/13_appendix-guide.md#e13-14)按同源PCM与稳定评分窗比较同DRR的两条短RIR。这些独立练习不能由静态来源报告统一登记成原神经系统已运行。

## 6. 何时可以进入设备比较

只有两套候选实现使用同一输入、评分区域和延迟口径，性能比较才有明确含义。硬件上再分别测量启动加载、
稳定运行、最慢帧、峰值内存和功耗状态；平均实时系数小于一仍可能发生连续帧超期。

实际设备的验收条件由产品覆盖距离、房间、声音类型和可接受损伤决定。本书的手算输入与教学示例用于检验算法，
它们不能直接成为出厂阈值。工业项目的具体入口和测量方法见 [工业部署与评测](03_industrial_deployment.md)。


## 7. 固定延迟组件的实际接口对照

[STK DelayL探针](../../ch10/examples/run_stk_delay_probe.py)先核对官方地址、完整提交、稀疏选择和清洁工作区，再在临时目录编译两份上游C++源文件。它把零延迟、半采样、一采样、保留状态分块和逐块清空五项分别对照解析FIR。已执行配置、零误差边界与0.1519102855的重置差异见[工业I29](03_industrial_deployment.md#i29stkdelayl)和[报告](../../ch10/reports/stk_delay.json)。

这项检查不下载权重、不访问音频设备，也不需要重新授权本书使用整个STK工具箱。第三方源码保留在独立的被忽略目录；提交的是本书调用探针、锁表和可复核报告。运行失败应保留实际错误与来源状态，不能用静态源码说明代替运行记录。


### 第6章外部 AEC 接口诊断

[接口诊断](../../ch06/examples/audit_aec_upstream_interfaces.py)与[原 APA 审计](../../ch06/examples/audit_upstream_apa.py)不下载依赖、不修补上游。2026-10-04 实际重跑得到[接口当前报告](../../ch06/reports/aec_upstream_interfaces_current.json)和[APA 当前报告](../../ch06/reports/upstream_apa_current.json)，环境为 Python 3.13.12、NumPy 2.5.3。两者核对官方 origin、独立工作树根、固定 HEAD、所用源码及许可 SHA/Git blob、前后洁净状态，记录实际依赖摘要；完整稀疏选集状态与所用文件身份分别列出，不把获取报告的 `execution=not_run` 升级成完整算法运行。

接口报告实际调用 pyaec 的 RLS/Kalman/FDKF/PFDKF 与 echocatzh 初始块：末尾截断、`np.complex` 原失败、默认后滤改变误差仍保留。DTLN 原 `process_file()` AST 只在内存音频和假解释器上执行，没有 TensorFlow、权重或生产音频写入。APA 完整加载原单文件，四个确定数组与独立有理手算对照，保留实数缓冲丢复部的警告。工具退出零表示合同记录生成成功，不表示每个原算法执行成功，更不是声学性能或实时性验收。[原方法与逐例边界](02_aec_wpe_separation.md#aec)

[旧接口报告](../../ch06/reports/aec_upstream_interfaces.json)与[旧 APA 报告](../../ch06/reports/upstream_apa.json)原字节和修改时间保全；对应历史测试绑定真实提交 `621d727a63475e3ee8ad2a29d518889c42e92571` 的旧工具 blob。不能给旧报告换当前工具 SHA 或将当时有限身份核验追记为今天的新合同。新工具默认标准输出，只允许显式写各自当前报告或仓外普通报告，历史、源码、锁表、缓存和符号链接目标在执行前拒绝；严格 JSON 与有限写前检查的边界见[来源合同测试](../../../../tests/test_codes_ch06_source_contracts.py)。

```bash
.venv/bin/python -m codes.chapters.ch06.examples.audit_aec_upstream_interfaces --report codes/chapters/ch06/reports/aec_upstream_interfaces_current.json
.venv/bin/python -m codes.chapters.ch06.examples.audit_upstream_apa --report codes/chapters/ch06/reports/upstream_apa_current.json
```

**生产二进制与落盘边界。** [Speex 配对入口](../../ch06/examples/aec_real_pair_experiment.py)默认只打印报告，`--output-wav` 仅接受仓外新普通文件，写前保护输入与二进制。[AEC3 配对入口](../../ch06/examples/aec3_offline_compare.py)的 `--output-dir` 只接受仓外新空普通目录，执行前检查全部九个产出成员，拒绝缓存与来源路径。[同输入真值入口](../../ch06/examples/aec_same_input_truth.py)在仓外临时目录处理，逐个生产者调用前保护输入/输出；显式 build manifest 严格拒绝重复键和非有限数，核实际二进制 SHA，但声明的构建配置不是独立构建证明。这些有限检查不消除并发竞态。

本轮实际执行的是两个原方法审计和 38 项来源/写入合同测试，没有重跑 Speex/AEC3 生产声学链或设备实验。前文 2026-09-23/24 的配对和合成数字仍是历史运行；不能把今天的路径保护测试称为这些性能实验重新通过。增益通知、XMOS ALT 保持与 2026 两候选的固定原源/缺资产边界见[增强 A06/A17/A19](02_aec_wpe_separation.md#aec)及[工业 I10](03_industrial_deployment.md#i10xmos-libvoice-的-aec-与延迟控制)。

<a id="distributed-reproduction"></a>

## 8. 固定分布式源码：静态合同、原函数与完整算法

2026-10-04的[审计入口](../../ch15/examples/audit_upstream_distributed_contracts.py)和[当前报告](../../ch15/reports/upstream_distributed_contracts.json)固定两套独立上游工作树。工具不获取源码、不安装依赖、不改写上游；运行前后核对官方origin、完整HEAD、实际使用的源码/许可SHA和Git blob、洁净状态及完整锁表摘要。报告的`acquisition_scope`另存全部稀疏选集状态，`files`说明这次实际依赖的文件；不能用少数入口匹配替代完整选集核验，也不把获取过程的`execution=not_run`改成整包成功。

| 上游与完整提交 | 本次所用入口 | 许可与范围 |
|---|---|---|
| [AlexanderBertrandLab/Old_Code](https://github.com/AlexanderBertrandLab/Old_Code/tree/a24b73fcd2dc028659535d07bb08068b06108616 "citation")，`a24b73fcd2dc028659535d07bb08068b06108616` | `WOLA_DANSE1.m`，版本1.4；17586B，SHA256 `54fd630bd7f34ac3d9a97db2a30cee6c7a6a1bc496274b53f9ba9f756ce0772b` | 文件头保留2010年版权及三条再分发条件，无根LICENSE且缺标准BSD免责声明，登记为`LicenseRef-Bertrand-DANSE-3-Conditions`。不能标为BSD-3-Clause；未取得四个数据ZIP或运行MATLAB |
| [fgnt/paderwasn](https://github.com/fgnt/paderwasn/tree/cd7054fcf72da637e4a5e11f035e8979691faf70 "citation")，`cd7054fcf72da637e4a5e11f035e8979691faf70` | `synchronization/sync.py`、`time_shift_estimation.py`、`utils.py`、`sro_estimation.py`，以及根README/setup/许可 | 根[LICENSE](https://github.com/fgnt/paderwasn/blob/cd7054fcf72da637e4a5e11f035e8979691faf70/LICENSE "citation")为MIT，SHA256 `60be4f44a5baa0db1ed1c9f1bc3ea5f7274b2032bb57bd87c054fa27efaa81fe`；仅三个原助手函数受限执行，完整同步包未导入 |

### 三个原函数在16组顶层用例中的实际行为

本次环境为macOS arm64、Python 3.13.12、NumPy 2.5.3。工具从锁定原文件提取函数定义的AST，保留函数参数与数学body，保存原定义/执行定义/数学body的摘要及起止行。三个定义本来都没有装饰器；周围模块及其导入没有执行。执行命名空间只提供NumPy及先提取的原`golden_section_max_search`，没有把自写估计器冒充原函数，也没有导入完整`paderwasn`包。

| 原入口 | 顶层用例与独立期望 | 实际记录与含义 |
|---|---|---|
| [`sync.py::coarse_sync`，5～34行](https://github.com/fgnt/paderwasn/blob/cd7054fcf72da637e4a5e11f035e8979691faf70/paderwasn/synchronization/sync.py#L5-L34 "citation") | 32点脉冲的−3、0、+4点偏移及+4点反极性；独立峰位置差和保留支持长度 | 四例偏移完全匹配，输入未改动；这是整样本粗对齐，不是SRO估计 |
| 同一`coarse_sync` | 两路相同4点脉冲，请求`len_sync=8`，独立偏移应为0 | 原函数按请求长度减`len_sync-1`，返回−4并把两路裁成空数组；记`observed_original_behavior_difference`，保留原失败 |
| 同一`coarse_sync` | 两路8点全零，无可辨识相关峰 | 返回−7、两路各剩1点；原数值有限但无可辨识时延，另记无观测状态 |
| [`time_shift_estimation.py::max_time_lag_search`，9～49行](https://github.com/fgnt/paderwasn/blob/cd7054fcf72da637e4a5e11f035e8979691faf70/paderwasn/synchronization/time_shift_estimation.py#L9-L49 "citation") | 64点全谱的解析线性相位，已知延迟0、3、−5、2.25、31.75、32、33；Nyquist频点显式置零 | 七例按模64比较，误差均在原黄金搜索`1e-4`样本容差内。31.75得到约−32.25002，允许周期等价，不强行把返回值说成始终位于半开有符号区间 |
| [`utils.py::golden_section_max_search`，5～50行](https://github.com/fgnt/paderwasn/blob/cd7054fcf72da637e4a5e11f035e8979691faf70/paderwasn/synchronization/utils.py#L5-L50 "citation") | 在`(-1,1)`内最大化两条凹二次函数，解析顶点−0.3、0.6 | 两例与独立顶点匹配，使用原默认区间收缩容差 |
| 同一`max_time_lag_search` | 64点全零谱，无时延观测 | 返回约−31.50002；目标函数处处相等，数字不具有测量意义，记无观测状态 |

合计16组顶层用例：13组匹配独立期望，1组短输入差异，2组无峰却返回数值。GCC搜索内部还调用原黄金搜索助手，这个嵌套过程不另计为顶层用例。容差来自优化搜索的区间收缩，不是设备时钟精度、ppm成绩或自然语音同步质量。静音用例说明调用者需要输入/活动和可辨识性检查，不能仅以“返回有限数字”把结果纳入评分。

固定`setup.py`的`install_requires`为空，不表示完整包没有依赖。README另外列出固定的`paderbox`与`lazy_dataset`；同步模块还导入SciPy窗与`paderbox`的STFT/分帧。这里只略去三函数不使用的模块导入以限定执行范围，未安装上述依赖、运行OnlineWACD/DWACD、完整SRO/STO估计、重采样器、波束或录音数据。[原README](https://github.com/fgnt/paderwasn/blob/cd7054fcf72da637e4a5e11f035e8979691faf70/README.md "citation")与[同步源码](https://github.com/fgnt/paderwasn/tree/cd7054fcf72da637e4a5e11f035e8979691faf70/paderwasn/synchronization "citation")提供后续整包研究入口。

### 原MATLAB静态合同与五项独立控制

报告保存14项原MATLAB静态行合同，包括帧首VAD、内部/外部滤波器分工、广播版本生效次序、顺序token的作用、普通EVD、直接逆矩阵、窗函数和循环终点；配置与论文实验分开记录，详见[工业I34](03_industrial_deployment.md#distributed-network-deployment)。本机MATLAB与Octave运行时均不可用，未安装运行时、未运行原WOLA链、GEVD-DANSE、真实网络或数据集。

另有五项**数学控制或控制流复写**，不称原MATLAB运行结果：

| 独立控制 | 可复算的结果 | 它检查的边界 |
|---|---|---|
| 512点对称Hann、256点帧移 | 两窗重叠和最小约0.99692610，最大约0.99999055，偏离1最大约0.00307390 | 对称窗不自动满足此未经修正的恒等增益合同；没有模拟完整原WOLA音频 |
| 差协方差$\operatorname{diag}(-9,-1)$ | 先选代数最大−1再取绝对值，形成$\operatorname{diag}(0,1)$；正半定投影则为零 | 不是选绝对值最大9，也不是删去负特征值 |
| 初始外部值/目标1，事件后新目标3，平滑系数0.5 | 四帧广播版本为`[1,1,2,2.5]`；事件后的第二个广播帧才受新目标影响 | 明确先广播、再向旧目标平滑、最后改目标；内部两个节点仍逐帧更新 |
| 8个相同二维噪声快拍 | 协方差秩1，即使快拍数8大于维数2 | 足够帧数不保证正定，原`inv`无加载 |
| 输入24点、窗8点、帧移4点的原循环边界 | 零基起点`[0,4,8,12]`，最后处理到20；完整合法起点16未进入循环 | 末4点未处理，没有通过停止循环完成排尾 |

这五项控制中的配置用于隔离数学或时序问题，不是原论文的实验配置。正文的自适应广播教学核、独立数学音频和图报告同样有各自真实源与输入，不借用这份静态记录证明原MATLAB性能。

### 复跑与尚未执行的候选

```bash
.venv/bin/python -B -m codes.chapters.ch15.examples.audit_upstream_distributed_contracts
.venv/bin/python -B -m codes.chapters.ch15.examples.audit_upstream_distributed_contracts --report codes/chapters/ch15/reports/upstream_distributed_contracts.json
.venv/bin/python -B -m unittest tests.test_codes_distributed_contracts -v
```

默认仅输出严格JSON，显式`--report`才通过公共IO原语保存当前报告。工具拒绝上游缓存、源码、锁表、审查记录与旧历史报告作为输出，并检查普通父链/成员、符号链接、硬链接、词法`..`及非有限数值。有限写前检查与原子替换不宣称消除并发竞态或保证崩溃持久性。离线测试以独立临时Git仓库验证错误origin、完整HEAD、摘要、脏工作树和控制失败；官方缓存缺失时可选原函数实调明确跳过，版本不符则报错。当前报告状态为`audit_completed_with_original_differences_and_unobserved_peaks`，保留实际差异与无观测输出，不把整份报告改名为全部算法通过。

[p-didier/danse固定提交](https://github.com/p-didier/danse/tree/3f08ffd6e578a8c5301e9310d29dc31112eb20f7 "citation")`3f08ffd6e578a8c5301e9310d29dc31112eb20f7`的完整tree未见明确代码许可，本书只登记索引，未获取代码或其`pyANFgen`子模块；源码可见不等于取得再分发授权。它可帮助定位SRO-DANSE的作者实现，不能因有链接就登记成已运行。

[TI-DANSE+作者批处理仓](https://github.com/p-didier/tidanseplus_batch/tree/0fcc0da19ce9e50bafc978ce1f315bac610d4c0b "citation")固定为`0fcc0da19ce9e50bafc978ce1f315bac610d4c0b`。同一提交的[根LICENSE](https://github.com/p-didier/tidanseplus_batch/blob/0fcc0da19ce9e50bafc978ce1f315bac610d4c0b/LICENSE "citation")写MIT及2026版权，[README的License段](https://github.com/p-didier/tidanseplus_batch/blob/0fcc0da19ce9e50bafc978ce1f315bac610d4c0b/README.md "citation")却写GPL-3.0-or-later及2025版权，存在真实矛盾。锁表保留冲突和限定研究范围，不能任选一个标签当作完整可分发许可。该仓用于定位批处理主入口与论文配置，本书未运行TI-DANSE+、原GEVD算法或论文语音指标；2025会议稿与2026扩展预印本的不同证明范围见[工业I35](03_industrial_deployment.md#distributed-network-deployment)。

<a id="historical-source-bindings"></a>

## 9. 来源索引扩充后怎样核验旧报告

整表SHA标识报告运行时实际读取的全部字节。首次新增五个分布式项目时，锁表从100项变为105项，其摘要自然改变；这不意味着旧报告曾在105项索引下运行。随后导读复核又新增LibriCSS，并改变WASN的获取范围，当前为106项。直接给旧报告换成当前摘要会伪造执行条件，直接要求所有旧报告等于当前整表摘要也会把无关的索引扩充误判成所用算法来源改变。

本书保存提交`a215b4630c0c21a8744cf27436a2c9ffa9c00053`中两份文件的**原始字节**，文件名包含完整SHA。它们不是重新序列化的JSON，也不是手工删去五项所得的新表。

| 固定快照 | 使用范围 |
|---|---|
| [100项来源锁表](../source_snapshots/SOURCES.55ab323ba665633141c4864763095046f9c6161ce2d88ca2aa9332dde7ec23f0.json) | 第1～9章十份固定原实现报告所记录的锁表摘要；其中第4章DOA和SAID分为两份 |
| [当时的获取状态](../source_snapshots/SOURCE_STATUS.e3b3176d835837441224e4906b7c2befadcdc4fe2ce6163245b7d9a9ad0d9229.json) | 第9章追踪报告记录的状态摘要；该状态中的`lock_sha256`精确指向上一行 |
| [105项来源锁表](../source_snapshots/SOURCES.e3478006c7dbc6cec442bf6bccc4df9eca946d7d353b661e947596dd8b87608a.json) | 第15章分布式原函数报告运行时的完整锁表；其中所用`danse-wola`和`paderwasn`记录与当前完全一致 |
| [105项获取状态](../source_snapshots/SOURCE_STATUS.b113b63c97767d19b76ceb44677303961ff310ce8d96f4e9e777944b44916d3c.json) | 扩充前67项通过及原22项选集不匹配、AEC失败等完整记录；绑定上一行105项锁表，不把后来取得的源码追记为当时成功 |

100项锁表的完整SHA-256：

```text
55ab323ba665633141c4864763095046f9c6161ce2d88ca2aa9332dde7ec23f0
```

当时获取状态的完整SHA-256：

```text
e3b3176d835837441224e4906b7c2befadcdc4fe2ce6163245b7d9a9ad0d9229
```

[只读核验函数](../core/source_history.py)先核当前表真实字节SHA；历史摘要只允许明确登记的快照，并核文件完整SHA、严格JSON、普通父链/文件和唯一项目ID。随后按每份报告明确列出的使用项目，将历史与当前的**整个项目记录**逐字段、逐类型比较，而非只比仓库名或提交号。来源地址、提交、许可、源码入口、选择政策或获取状态中任一使用字段变化都会使这项沿用检查失败；与该报告无关的新项目可以存在。

第9章还核历史状态绑定历史锁表、当前状态绑定当前锁表，以及ODAS、Spatial Audio Framework、FilterPy、Stone Soup四项完整状态记录一致。原`source_selection_verified=false`与选集不匹配原样保留，不将获取失败升级为原方法运行成功。第7章报告另有更早的状态摘要；上面的状态快照**不覆盖它**，不据此补写核验结论。

这些检查验证来源身份的延续，不是重新执行旧算法。原报告的工具摘要、所用原文件/blob、官方origin、完整HEAD、前后洁净状态、执行范围、失败观察与数值断言仍分别检查；各报告原有项目摘要的JSON序列化口径也保持。若使用项目确实改变，应保留旧证据并另行运行和保存新报告，不能通过更换摘要消除差异。

新增的两份105项快照逐字节取自提交`c889808ab071ed4c5c2884f8d1fd007a9e6a0e75`的原始文件，文件名中的完整SHA就是实际原文件摘要。104个原项目的锁定记录和获取状态保持不变；WASN由只登记变为限定取得，其完整记录确实变化，因此历史核验函数对该项目会拒绝沿用，而不会仅因origin和提交相同就忽略获取政策。AEC失败原记录没有`execution`字段，核验保留原结构，不补字段制造一致。

<span id="overview-source-entrypoints" class="anchor-alias" aria-hidden="true"></span>

## 10. 导读的两个来源入口：同步接口与会议评分

导读推荐的任务路线需要能落到实际接口。2026-10-04取得以下两个固定选集，并对38个普通文件逐个比较完整Git blob、真实SHA及字节数；官方origin、完整HEAD和洁净状态也分别核对。源码保存在`codes/chapters/ch00/upstream/_downloads/`的独立忽略工作区，锁表和获取器可以重建它们；本书没有把上游源码改写后混入教学数值核。

### WASN：先区分估计时钟与按已知时钟重采样

官方[CN-UPB/WASN](https://github.com/CN-UPB/WASN/tree/9b2590eb104abcde2a35af52c74d64ce30bf5ae2 "citation")采用提交`9b2590eb104abcde2a35af52c74d64ce30bf5ae2`的11个限定文件，共52411字节，覆盖README、根LICENSE、DXCP-PhaT实现及封装、拓扑配置、管道读写和`sync_sed/system/`中的重采样/模拟入口。[根LICENSE](https://github.com/CN-UPB/WASN/blob/9b2590eb104abcde2a35af52c74d64ce30bf5ae2/LICENSE "citation")是Apache-2.0，SHA-256为`c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`；外部安装依赖的许可与兼容性仍分别判断。

独立设备的采样率有微小差异时，两路记录的相对时移随记录长度积累。估计器从观测中推测采样率偏差，重采样器则使用给定偏差改变取样位置；两者的输入和输出不同。DXCP-PhaT入口用于定位前者的实现，管道读写用于观察数据怎样传递。此次没有完整复审DXCP数学主体或运行它，不能从文件取得推断估计精度。

[`resample.py::STFTResampler`](https://github.com/CN-UPB/WASN/blob/9b2590eb104abcde2a35af52c74d64ce30bf5ae2/sync_sed/system/resample.py "citation")的构造参数`sro`以ppm输入，构造器除以一百万；该参数非空时，逐次调用传入的`sro`被忽略。原注释把这个配置用于**模拟**给定偏差，补偿已知$x$ ppm则要求构造器传入$-x$。这说明符号与单位需要从实际接口确认，不能看见“resample”就认定程序会盲估计时钟。

无固定构造值时，`__call__(..., sro)`中的参数直接参与延迟递推，原方法没有再次除以一百万。因此调用参数应是无量纲相对速差：80 ppm在这个入口传$80\times10^{-6}$，而不是80。同名参数在构造与逐块调用中有不同单位，照搬数值会产生一百万倍的尺度错误。

例如80 ppm表示相对采样率差$80\times10^{-6}$；按16 kHz名义时钟持续10 s，相对累计样本差的量级为$16000\times10\times80\times10^{-6}=12.8$点。这是单位换算的数学例子，不是原程序执行结果；取样率比值、延迟正负与接收端时间原点仍要在所选接口下解释。导读工程基线和[第15章](../../../../chapters/15_distributed-enhancement.md)提供可独立复算的教学入口。

[`sim_sro.py`](https://github.com/CN-UPB/WASN/blob/9b2590eb104abcde2a35af52c74d64ce30bf5ae2/sync_sed/system/sim_sro.py "citation")逐块调用上述重采样器，只在返回非空时向输出管道写块。重采样器保留内部缓冲；此封装没有显式排空记录尾部的步骤。实际流还需明确块长、首次可用输出、积累状态与记录尾部；等待期间的空返回或块延迟不能直接解释为音频丢失，结束时的剩余样本则要另行处理。

所取文件涉及NumPy、SciPy、sounddevice或Pyro4等导入，没有安装或执行。本次也没有取得固件、访问麦克风设备、部署MARVELO/SED或验证网络性能。未选择的包安装模板含MIT分类而根许可为Apache-2.0，其空依赖列表也不能代替直接导入核对；这里没有把选集称为完整可安装包。

### LibriCSS：评分协议也是系统输入的一部分

官方[LibriCSS工具仓](https://github.com/chenzhuo1011/libri_css/tree/9e3b7b0c9bffd8ef6da19f7056f3a2f2c2484ffa "citation")采用提交`9e3b7b0c9bffd8ef6da19f7056f3a2f2c2484ffa`的27个限定文件，共69954字节，覆盖README与许可、环境/安装说明、数据准备、VAD、ASR输入生成和评分的Python及命令入口。源码选择逐项列在[SOURCES.lock.json](../SOURCES.lock.json)，不是把所有`.py`或整仓库当作已经核过的数据闭包。

[原README](https://github.com/chenzhuo1011/libri_css/blob/9e3b7b0c9bffd8ef6da19f7056f3a2f2c2484ffa/README.md "citation")区分连续会议评测与已切分语音评测。前者要从连续输入得到分离或识别结果，再把带时间与流身份的输出送入会议评分；后者使用给定语音分段，参考信息和排列选择条件不同。因此不能拿分段结果直接证明一个连续系统能够处理未知活动边界。本选集是准备与评分工具，没有CSS神经网络主体、权重或可直接运行的整链。

[`asclite_libricss.py`](https://github.com/chenzhuo1011/libri_css/blob/9e3b7b0c9bffd8ef6da19f7056f3a2f2c2484ffa/scoring/python/asclite_libricss.py "citation")读取STM参考、CTM识别结果、GLM文本规则及SCTK路径，调用外部asclite再汇总错误数与参考词数。最终WER按总错误数除以总参考词数计算；[`report.py`](https://github.com/chenzhuo1011/libri_css/blob/9e3b7b0c9bffd8ef6da19f7056f3a2f2c2484ffa/scoring/python/report.py "citation")跨记录汇总也按参考词数加权。例如10词中2错与100词中10错，应为$(2+10)/(10+100)=10.91\%$，而非两个百分比的算术均值15%。这只是手算协议例，不是已执行的LibriCSS成绩。

参考文字、识别文字、时间戳、输出流身份和文本归一化规则都是评分输入；评分输出是错误数、参考词数及其比值，不是增强波形。若缺少参考、时标不一致或分段口径变化，数字即使能够打印，也不具备同协议比较的含义。连续会议的时间/排列问题还可对照[第11章](../../../../chapters/11_selection-guide.md)的MeetEval接口研究，不把两种评分器默认当作同一指标。

静态读取还发现原工具的使用限制：`run_asclite`封装只分Windows/Linux分支，这不是“SCTK算法不能在macOS运行”的结论；数据准备还调用外部下载包内的`segment_libricss.py`。`asr/python/get_wer.py`存在同名函数的后续覆盖及配置循环中的累计量，两份`run_wer_*_utterance.sh`还传入解析器没有的`--data_path`。这些是固定源的静态观察，尚未实际调用验证，原文件未修补；读者不能把取得选集等同于原脚本已经顺畅执行。

[根LICENSE](https://github.com/chenzhuo1011/libri_css/blob/9e3b7b0c9bffd8ef6da19f7056f3a2f2c2484ffa/LICENSE "citation")SHA-256为`e7758cf56804cdf83cde84d0d19f70df9173193f8be5831d31e52625e95b93d7`，保留项目MIT及所含py-webrtcvad MIT文本；上游WebRTC声明的最后免责声明原样截于`DAMAG`，不自行补齐或宣称第三方条款齐备。此次未取WebRTC C源；安装的外部VAD依赖还要检查其实际分发文本。715个CTM、715个STM、说话人JSONL、NIST派生GLM、音频和模型均未取得，数据许可不能由代码MIT代替。SCTK、PyKaldi、Docker环境也未安装或运行。

### 获取、核验与当前真实状态

```bash
.venv/bin/python -B codes/chapters/ch00/upstream/fetch_upstreams.py --project wasn-platform
.venv/bin/python -B codes/chapters/ch00/upstream/fetch_upstreams.py --project libricss
.venv/bin/python -B codes/chapters/ch00/upstream/fetch_upstreams.py --verify
```

前两项在本机实际取得选集，并核验为`source_verified`；当时离线全表核验真实返回非零：106项中69项通过、22项旧选集不匹配、14项仅索引、1项AEC工作树失败。全部核验不执行方法，失败的AEC记录保留原字段结构。锁定项目中92项有本地工作区，另有不计入此锁表的专用构建依赖目录；Git项目数也不包含独立的两项源码归档。

这些状态证明限定源码身份与获取边界。依赖安装、方法级运行、完整语音链路和设备验收仍是分开的工作；两个新选集此次都未运行原方法。已有48份Git跟踪的数值与方法JSON报告字节保持不变，旧失败没有被补成成功。历史105项状态与当时106项状态的关系见上一节。

第3章研究另登记Hybrid作者链接为NOASSERTION，仅索引固定提交，不建立受管理本地选集或复制出版源码。新增前完整保存106项锁表SHA `6dab41b1542cfa2cd4731ef4c8a3e807b343dc807f209c4eea17503e203ebf59`及原状态SHA `1142cc2290d93ea33e6b2dcdf72b9b52307efdd837b84c1739262c2c968186d6`的原字节，并在[source_history.py](../core/source_history.py)登记；旧报告继续绑定其真实历史摘要。第3章验收时107项由真实工具离线重核，69项通过、22项选集不匹配、15项仅索引及1项AEC本地修改失败，退出非零；此前106个项目的完整状态记录逐项不变。新增索引不会把源码未取得、原失败或未运行方法改成通过，方法条件见[空间研究§39.10](01_spatial_and_tracking.md#sec-39-10)。


### 第4章新增来源与当前限定调用

第4章新增DP-RTF与ESP-SR两个固定来源仅索引，当前109项由真实获取工具离线核验：69项通过、22项完整选集不匹配、17项index_only、1项既有AEC缓存失败，工具退出1。原107项完整状态逐项不变。新增前保存锁表SHA `5dbf0c55fac57d82915ce01c20d7a96505147aaaed550558ce5e6c1c24da1fd0`和状态SHA `230269d62683e0726efb2c87a1661f8179ada3ee041b7b7df266190e471e2652`的原字节并登记，旧历史方法报告不换工具或整锁摘要。

本机另在`codes/chapters/ch00/upstream/_downloads/.research-only/`导出DP-RTF四个原文件（4863字节）及ESP-SR七个所读入口（33830字节），逐文件对照固定Git blob及锁表SHA，原许可/版权声明保留。两个选集各带`RESEARCH_ACQUISITION.json`，明确是固定对象的文件导出，不是洁净Git工作树，不含二进制、权重、音频或数据集，未运行算法。该忽略缓存不进入提交推送，也不升级受管理109项的获取状态或92个独立工作区数量；静态研究、取得完整实现和获得再分发许可分别判断。

六个第4章原源工具现在统一使用[当前合同核](../../ch04/core/upstream_contracts.py)：sanitized Git、官方origin、固定HEAD、全部声明使用的原blob与许可SHA、跟踪/未忽略成员的前后洁净及忽略成员前后清单，真实完整选集与所用源码身份分栏。默认stdout不写报告；两个audit显式写入仅允许新[DOA当前报告](../../ch04/reports/upstream_doa_contracts.json)、[SAID当前报告](../../ch04/reports/said_compression_contracts.json)或仓外普通路径。四个旧reproduce入口的output/workdir只能仓外，历史报告/源码/上游及链接父链执行前拒绝。有限路径检查不保证消除并发竞态。

本轮使用既有`/private/tmp/masp-appb-pra-venv`真实运行PRA 0.10.0完整TOPS及限定原helper；原峰49°与独立30°分别保留，root-MUSIC原dtype错误与局部门面分开，SAID只执行四个压缩格式助手。ESPRIT与SBL另实际输出到仓外，不给旧报告改成新运行。SMP的FFTW条件不足，本轮没有重建原C；旧失败/临时适配历史与独立DFT测试仍分别说明。DP-RTF MATLAB、完整DPD、FRIDA、学习网络、核心二进制和硬件未执行；详细条件及许可见[空间研究58/59](01_spatial_and_tracking.md#sec-u-1ca23edba5)。


<a id="beamforming-current-contracts"></a>

## 11. 第5章当前原方法合同与相位控制

2026-10-04实际运行三项限定合同；所有结果由真实工具生成，没有改上游、旧三报告或原获取状态。重新调用可使用具备报告所列PRA/NumPy/SciPy版本的隔离环境；下面临时路径只描述本机本轮环境，不是永久安装位置。

```bash
/private/tmp/masp-appb-pra-venv/bin/python -m codes.chapters.ch05.examples.audit_upstream_beamformers --report codes/chapters/ch05/reports/upstream_beamformers_current.json
/private/tmp/masp-appb-pra-venv/bin/python -m codes.chapters.ch05.examples.audit_beamformer_reference --report codes/chapters/ch05/reports/beamformer_reference_current.json
/private/tmp/masp-appb-pra-venv/bin/python -m codes.chapters.ch05.examples.audit_sof_tdfb_design --report codes/chapters/ch05/reports/sof_tdfb_design_current.json
```

| 工具与当前报告 | 实际执行 | 原问题及未执行边界 |
|---|---|---|
| [pb_bss合同](../../ch05/reports/upstream_beamformers_current.json) | 13个未修改原函数AST，33个确定性用例；原stable_solve与SciPy后备 | 5项原异常保留；二维/三维相位轴对照有原符号差异；未导入完整pb_bss/ESPnet/Torch、可选Cython、增强波形或论文基准 |
| [reference合同](../../ch05/reports/beamformer_reference_current.json) | MERL单原函数AST两组控制、原PRA方法调用；安装85份Python源逐fixed blob核对 | pb_bss普通导入缺paderbox、MERL误选、PRA浮点切片失败，没有得到Rake滤波器；二进制只登记安装文件SHA，不声称可复现源码构建 |
| [SOF合同](../../ch05/reports/sof_tdfb_design_current.json) | 七份固定源码/许可静态核验与三组独立标量计算 | sinc/WNG/最后方位控制流问题；没有MATLAB/Octave、实际FIR设计、固件或设备执行 |

去掉 `--report` 时三项默认都只输出stdout。指定报告前先核普通父链与目标：仓内只接受该工具的新current路径；旧报告、其他仓内源文件、缓存与符号链接目标被拒绝。工具绑定当前真实工具、共享合同、IO与获取工具SHA，同时核官方origin、固定HEAD、所用原Git blob与逐文件许可；所用文件核验和完整选集状态分栏。写前检查有限，不保证消除并发竞态或崩溃持久性。

三份历史报告分别是[原31项pb_bss](../../ch05/reports/upstream_beamformers.json)、[原reference](../../ch05/reports/beamformer_reference_audit.json)、[原SOF](../../ch05/reports/sof_tdfb_design_audit.json)。它们保留原字节与修改时刻；旧工具身份从本仓库固定提交747ec3fec96ef20c7c128cc4291fd7bb9ee36c34的Git blob核对，不将当前工具SHA替换进去伪造重跑。

共享合同增加pb_bss/SOF固定项目身份后，两个第4章当前消费者[DOA合同](../../ch04/reports/upstream_doa_contracts.json)和[SAID合同](../../ch04/reports/said_compression_contracts.json)也实际重新执行，报告绑定更新后的真实合同源。此前六份第4章历史报告未改写；这只是依赖消费者的限定重跑，不是把其他章节提前列为完成。

本书[E05-23相位音频](05_exercises_and_audio.md#phase-reference-audio)与E05-24预滤波矩阵采用独立已知模型，不调用上述原算法来生成“理想答案”。三份PCM的实际量化功率和参考误差分开评分，图73绑定真实源；模型正确与工业系统性能是不同验证范围。


### 第7章到达延迟控制与当前来源身份

[E07-22～24的已知到达控制](05_exercises_and_audio.md#known-arrival-audio)把源时刻索引、有限单实回归、实际PCM与晚期功率自由衰减分开。六WAV不是原作者房间数据或完整STFT-WPE；固定时差384点与历史padding768点也分别命名。独立清单绑定当前五真实生成源，核验内存重放全部七个普通成员，并逐字节比较PCM。

2026-10-04取得固定[rir-encoder](https://github.com/sp-uhh/rir-encoder/tree/8ee0ba7e083c8e38170dc13632cc9937a702d0f0)的八份原文本，共20717字节，许可为MIT；包含README、许可、依赖、损失、网络、数据、训练与提取入口。官方origin、固定HEAD、逐blob/SHA与洁净状态均已核验；未导入训练脚本、取得权重或运行完整FiLM-SGMSE恢复链。训练入口含顶层目录创建与执行逻辑，静态阅读不能改称模型运行。[增强研究](02_aec_wpe_separation.md#rir-encoder-candidate)另说明训练与提取的STFT边界差异。

新增前原字节保存在[109项锁表](../source_snapshots/SOURCES.a93ae64f3d6464b5c19bfa99d93f6d001782e5d5cbcec3222b5b7f5ce45a5505.json)与[当时获取状态](../source_snapshots/SOURCE_STATUS.abc28f6f15512da8267618bca318b10033909ec2afdb6090b6aa2d0f1281f0e9.json)，由历史身份核登记。加入该项目后的110项当时由真实获取工具重核：70项通过、22项选集不匹配、17项仅索引、1项既有AEC缓存失败，退出1；旧109项完整状态逐项不变。93项有受管理源码工作区。新增取得状态不能证明方法、GPU或工业性能已运行。

[第7章三个当前报告](02_aec_wpe_separation.md#sec-u-841a085a72)分别绑定当前工具与来源身份，历史三报告保持原字节。共享来源合同变动后，第4章DOA/SAID、第5章三份接口报告、第6章AEC/APA当前报告已真实重跑；DOA首次在主环境因缺pyroomacoustics失败，随后在既有PRA0.10.0隔离环境执行，不能把首次失败省略。原报告中的限定调用、选集不匹配、静态检查及未运行条件继续分别记录。


## 12. 第8章作者OverIVA与MeCo来源及历史状态

2026-10-04取得[OverIVA作者固定提交](https://github.com/onolab-tmu/overiva/tree/1cb3189112889ebfe2ccbbde0f55b1db6020fc48 "citation")四文件选集，MIT通知、输入/滤波器轴序与PCA入口静态边界见[增强研究B03](02_aec_wpe_separation.md)。未取得语音数据或执行作者论文仿真。

新增前原110项锁表SHA为`eec80748e5f1fcc01d33e281b91a4482f52566d8e6b74bdb737e017604ec6487`，原状态SHA为`4cc8cbd48e3e98b51a24705468415d10654aa349f3946fb98ef48e9c4856f4f6`；原字节保存于[source_snapshots](../source_snapshots/)并由[source_history.py](../core/source_history.py)登记。加入OverIVA后的111项当时用真实工具离线核验，71项通过、22项原选集不匹配、17项仅索引、1项原AEC缓存失败，真实退出1。当时94项有独立受管理工作区。旧报告继续保留真实历史身份，当前方法执行与源码取得状态分别记录。


随后按固定`375ac4dee2a8e193aaa48553289499a6f43d93e6`取得MeCo的16个MIT核心/入口/配置原文本，共95305字节。部分选集不包括混合许可骨干和NVIDIA NC CUDA，缺少独立运行闭包；未执行Torch/CUDA或取得权重与数据。具体原输入、默认参数、JVP与评价边界见[增强N19](02_aec_wpe_separation.md#meco-candidate)。新增前111项原锁/状态再次按原字节保存并登记，当时112项实际离线核验为72通过、22原选集不匹配、17仅索引、1原AEC失败，真实退出1；95项有受管理工作区。所有原111项完整状态保持，来源取得不是性能成绩。

## 第9章：114项来源身份与当前追踪报告

2026-10-05按逐文件许可取得LOCATA I/O与eval的15/16个文本选集；不是完整MATLAB运行环境，未取得数据或执行官方评分。范围、原语句静态疑点与许可分别见[空间研究§61.4](01_spatial_and_tracking.md#tracking-model-lifecycle-sphere)及[第三方记录](../THIRD_PARTY.md)。完整锁表114项实际离线核验为74通过、22原选集不匹配、17索引和1原AEC缓存失败，工具退出1；97个项目有独立受管理工作区，原112项状态逐项保留。

扩充前的[112项原锁字节](../source_snapshots/SOURCES.d679c9d005768ea5f5d51b3e1f5dc33a99f317e859eee252250d343ca2ddb63a.json)和[原获取状态字节](../source_snapshots/SOURCE_STATUS.e6108565144184bdafbd9c6813cf510fb2c9ee6448ae003411bed5cacc78d8da.json)按完整SHA登记，旧执行报告不换成新锁表身份。当前第4～8章14个直接消费者实际重新执行后绑定新锁表，原故障/未运行边界保留；不能只改JSON字符串。

第9章两个工具默认stdout；接口工具显式`--output`写[当前接口报告](../../ch09/reports/tracking_upstream_interfaces_current.json)，原方法合同工具显式`--report`写[当前原方法合同](../../ch09/reports/upstream_tracking_contracts_current.json)，路径均须通过安全检查，不覆盖两旧历史报告。ODAS限定原C、SAF控制流替身、FilterPy原包/原方法和StoneSoup三个原AST方法各记执行范围；完整稀疏选集不匹配不抹去。FilterPy本轮预检63个跟踪源文件，33个实际导入模块分别记录；EKF组合接口`None`的原TypeError及部分K/S更新保留，不能改写成原包缺测路径成功。

2026-10-05第10章四组原接口复验保存为current报告，旧四报告与其历史工具身份保持原样；当前编译依赖、原失败及未执行范围见[工业研究§9](03_industrial_deployment.md#industrial-current-interface-contracts)。上文复跑命令写显式current路径；仓内其他报告目标由写前守卫拒绝，普通仓外路径可另存试验。
