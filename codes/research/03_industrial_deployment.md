# 工业音频实现：从采集、状态到部署和评分

核实日期：2026-09-22。对应正文第 10、11 章和附录 B。这里讨论本书算法进入连续音频系统后需要补上的部分：驱动、参考路由、跨块状态、定点内核、模型运行时和评分器。每项的完整提交以 [`SOURCES.lock.json`](../SOURCES.lock.json) 为准；网页文档版本只说明核实依据，不自动等于本机安装版本。

下面的试验是建议执行的设备验收步骤，不是本书已经测得的产品结果。源码下载、依赖安装、编译、运行、声学测量是不同状态。没有硬件、模型或数据时，可以完成源码核对，但不能把该项标成已经通过设备验收。

## 1. 采集、时钟与连续流

### I01：PortAudio 回调与设备时间戳

PortAudio 为不同主机音频系统提供统一入口。本书锁定 19.7.0 对应提交。先读 `include/portaudio.h` 的回调签名，再看 `examples/` 中的采集与播放例子；库为 MIT 许可，平台后端仍有自身依赖。官方[回调说明](https://portaudio.com/docs/v19-doxydocs/writing_a_callback.html)明确限制会阻塞或耗时不可预测的操作。

设备配置要记录 host API、设备 ID、采样率、样本格式、通道数、请求与实际延迟、回调块长。`frameCount` 表示每通道时间样本数，不能乘通道数后再当作帧数。保存输入 ADC 时间、输出 DAC 时间和状态位；它们不能未经校准就当成两个设备共享的时钟。

验收可以用环回脉冲测端到端延迟，再施加磁盘和 CPU 负载。把状态位、回调间隔、队列长度与音频缺口关联起来。回调中只将样本交给预分配缓冲，不应直接运行 Python 模型或同步日志。设备拔插后需重新确认通道映射，而非只恢复原设备编号。

### I02：ALSA PCM 状态与恢复

Linux 设备层可对照 `alsa-project/alsa-lib` 的 `src/pcm/pcm.c` 和 [`test/pcm.c`](https://github.com/alsa-project/alsa-lib/blob/f84cd4ced7b36fddb8e4ee24404cf7c091d27020/test/pcm.c)。项目 `COPYING` 为 LGPL 2.1 文本，具体源文件的版本选择和例外仍按其文件头核对。PCM（脉冲编码调制）接口把硬件配置、软件启动阈值和流状态分开。

[官方 PCM 文档](https://www.alsa-project.org/alsa-doc/alsa-lib/pcm.html)区分：`-EPIPE` 是播放下溢或采集上溢；`-ESTRPIPE` 与挂起有关；`-ENODEV` 可表示设备被移除；某些回声参考设备在关联播放未启动时可返回 `-ENODATA`。因此不能对所有负返回值无条件调用同一恢复函数，并假设音频仍连续。

规格要固定 period、buffer、`avail_min`、启动阈值、访问布局和时间戳模式。测试短读写、CPU 暂停、系统挂起、USB 拔插和“先开参考、后开播放”。恢复后记录真实丢失样本并通知 AEC、重采样器及追踪状态；重新 `prepare` 不会补回已经丢失的数据。

### I03：PipeWire 的 AEC 四条流

PipeWire 官方开发仓库位于 freedesktop GitLab，`PipeWire/pipewire` 是项目维护的 GitHub 镜像。入口为 [`src/modules/module-echo-cancel.c`](https://github.com/PipeWire/pipewire/blob/62316ae8cf659ca8e5f1ee866ae743d2438d47a4/src/modules/module-echo-cancel.c) 和 [`spa/plugins/aec/aec-webrtc.cpp`](https://github.com/PipeWire/pipewire/blob/62316ae8cf659ca8e5f1ee866ae743d2438d47a4/spa/plugins/aec/aec-webrtc.cpp)。项目根 `COPYING` 为 MIT；WebRTC 后端依赖另行保留许可证。

[PipeWire 1.6.9 文档](https://docs.pipewire.org/page_module_echo_cancel.html)将麦克风 capture、应用 source、应用播放 sink 和实际 playback 四条流分开。要抵消的播放必须经过选定参考路由。`monitor.mode` 改变参考获取方式；`audio.rate`、通道布局、目标节点与 `node.latency` 都要与实际图连接一起保存。

测试会议语音、通知声和音乐是否都进入参考；切换默认扬声器后再次检查。用断开参考和改变播放延迟的试验观察 AEC 恢复。系统已启用 AEC 时，应用再启用一套 AEC 会改变输入统计与近端损伤，应设置单套与双套处理对照，不能只检查虚拟麦克风能否出现。

### I04：libsamplerate 的有状态重采样

`libsndfile/libsamplerate` 的 README 标识 0.2.2，根许可为 BSD-2-Clause。核心入口 [`src/samplerate.c`](https://github.com/libsndfile/libsamplerate/blob/0844c208f683527c08ea8a80acc13b398aa9c8bf/src/samplerate.c) 的 `src_new`、`src_process` 与 `src_reset` 管理转换状态；滤波实现位于 `src/`。

[Full API](https://libsndfile.github.io/libsamplerate/api_full.html)规定 `src_ratio=输出采样率/输入采样率`，调用后分别返回实际消耗的 `input_frames_used` 和生成的 `output_frames_gen`。不能假设每次恰好消耗整个输入块，也不能把多个连续块各自交给一个新实例。流结束时要按 API 排空滤波尾部。

本书 SRO 约定中设备快 100 ppm 时，转换到参考时钟的比例为 `1/1.0001≈0.99990001`。用同一宽带信号比较整段与分块输出；补偿固定滤波延迟后检查残差，再测接近奈奎斯特频率的带外泄漏。比例更新还要测突变、平滑过渡和队列长期漂移；短录音长度看似正确不足以说明相位相干。

### I05：SpeexDSP 的流式转换与回声缓冲

SpeexDSP 的 `libspeexdsp/resample.c`、`include/speex/speex_resampler.h` 是另一组连续重采样入口。`libspeexdsp/mdf.c` 提供分块频域回声处理，可与第 6 章教学 NLMS 对照。代码根许可为 BSD-3-Clause，版本和提交沿用锁定清单。

重采样配置包括输入/输出速率、通道数、质量档位、整数或浮点输入，以及每次调用的输入/输出长度。AEC 的帧长、滤波覆盖长度、采样率和播放参考缓冲另行固定；两者共享“连续状态”要求，但不能共用一个模糊的延迟参数。

用跨块脉冲检查边界连续性；对各通道输入相同信号，确认输出相位仍一致。AEC 测试覆盖参考领先/滞后、远端单讲、双讲和路径突变，不能以重采样波形连续代替消回声验收。[官方源码](https://github.com/xiph/speexdsp)

## 2. 噪声、语音活动与增益

### I06：WebRTC 传统 VAD

入口为 [`common_audio/vad/include/webrtc_vad.h`](https://webrtc.googlesource.com/src/+/0467d2b91cc20b9b001c2bbb73d43ea6b2491f3e/common_audio/vad/include/webrtc_vad.h)，核心判断位于同目录的 `webrtc_vad.c` 和 `vad_core.c`。它与 APM 中的其他语音概率估计器不同；移植时必须写明具体接口，不能统称“WebRTC VAD”。WebRTC 源码使用 BSD-3-Clause，并保留相关第三方声明。

接口接受受支持采样率下的有符号 16 位单声道 PCM，帧长必须通过 `WebRtcVad_ValidRateAndFrameLength` 检查；常用合法帧为 10、20、30 ms。16 kHz 下分别是 160、320、480 个样本。模式控制检测的积极程度，不是以 dB 表示的能量门限。

验收应给出误报/漏报随模式的变化、词首/词尾截断及静音段误触发。输入浮点 `[-1,1]` 若只做整数类型转换，会丢失绝大多数幅度信息；应显式缩放、舍入和饱和。多通道阵列也必须先明确使用哪个信号取点。

### I07：WebRTC AGC2

[`modules/audio_processing/gain_controller2.h`](https://webrtc.googlesource.com/src/+/0467d2b91cc20b9b001c2bbb73d43ea6b2491f3e/modules/audio_processing/gain_controller2.h)与 `agc2/` 中的增益、限幅、语音电平和噪声估计模块构成工业对照。第 10 章峰值保护基线只解释增益状态和削波保护，不能代表这组控制器的完整行为。

部署时分别记录固定数字增益、自适应增益、输入音量控制是否启用、采样率和通道数。模拟输入音量与数字增益不是一个执行位置，前者变化可能影响 AEC 建模。状态要在采样率或设备改变时按上层 API 重新配置。

推荐输入电平阶跃、长静音后的短词、脉冲噪声、已经削波的波形及双讲数据。同步记录增益、输出峰值和语音保留情况。增益变大后噪声一起升高不等于噪声抑制失效；需要在相同输出响度或明确的增益口径下比较。

### I08：Silero VAD 的块与状态

`snakers4/silero-vad` 提供 MIT 许可代码与预训练模型入口；发布模型及依赖仍按锁定资产核对。[`src/silero_vad/utils_vad.py`](https://github.com/snakers4/silero-vad/blob/60b7ffa243625ebdc1070275a29f18c87843786a/src/silero_vad/utils_vad.py)的 `OnnxWrapper`、`VADIterator` 和整段分割函数解决不同层次的问题。

核实的包装器要求 16 kHz 时每块 512 点、8 kHz 时每块 256 点，均为 32 ms。它保留循环状态和上下文；同一会话不能每块重置，多条独立会话也不能共享状态。语音概率与分段结果之间还隔着起止门限、最短语音、静音等待和前后填充。

高采样率整数倍输入的便捷路径使用抽点，因此实际设备应先完成抗混叠重采样。试验应比较整段与正确连续分块、故意每块清状态、两路会话交错、设备切换及短词。对于 10 ms 驱动块，要先积累到模型合法块长，再把决策时间映射回原始采样轴。

### I09：DeepFilterNet 的完整运行路径

`Rikorose/DeepFilterNet` 把训练/离线增强、Rust 处理和插件集成分开：`DeepFilterNet/df/`、[`libDF/src/`](https://github.com/Rikorose/DeepFilterNet/tree/d375b2d8309e0935d165700c91da9de862a99c31/libDF/src)、[`ladspa/`](https://github.com/Rikorose/DeepFilterNet/tree/d375b2d8309e0935d165700c91da9de862a99c31/ladspa)。根许可允许选择 Apache-2.0 或 MIT；模型和训练数据按各自资产说明处理。

官方项目面向 48 kHz 全带语音增强。所选模型配置中的 FFT 长度、帧移、前瞻、深度滤波阶数与延迟补偿必须一并记录；离线 `--compensate-delay` 的截取或填充行为不等于流式系统消除了等待。LADSPA 插件可用于 PipeWire 过滤图，但操作系统的量子块与模型帧移可能不同，需要缓冲适配。

锁定源码的 `ladspa/README.md` 说明该插件模型没有额外前瞻，但 STFT 处理仍带来 20 ms 最小延迟，宿主还会增加延迟。因此“无前瞻”不等于“零延迟”。具体路由配置可读 `ladspa/filter-chain-configs/deepfilter-mono-source.conf`，编译入口为 `cargo build --release -p deep-filter-ladspa`；该命令需要对应 Rust 依赖，源码获取流程不会自动运行它。

测试应包含稳定噪声、非平稳噪声、另一说话人、音乐和静音，再比较下游词错误率与主观听感。延迟用环回测量，耗时在目标 CPU 的发布构建测量；只在 Python 文件增强中计时不能证明 Rust 插件的设备行为。[官方项目说明](https://github.com/Rikorose/DeepFilterNet)

## 3. DSP 固件与模块配合

### I10：XMOS lib_voice 的 AEC 与延迟控制

`xmos/fwk_voice` 已在官方 README 中声明迁移到 `xmos/lib_voice`。后者 README 版本为 1.1.0，列出 XTC Tools 15.3.1、Python 3.11 和 `lib_xcore_math` 等前提。源码入口为 `lib_voice/api/aec/`、`lib_voice/src/aec/` 与 `adec/`；ADEC 是围绕 AEC 的延迟估计与控制部分。[官方库](https://github.com/xmos/lib_voice)

配置应明确麦克风和播放参考通道数、帧移、主/影子滤波分区、内存池、线程调度和参考延迟。先确认参考完整，再测延迟估计器如何触发调整及 AEC 重置；不能只把其滤波器换成教学 NLMS 后继续沿用所有控制常数。

建议注入播放延迟阶跃、扬声器路径变化、静音参考和双讲。记录重对齐请求、滤波重置、残余回声和近端损伤。官方 [`LICENSE.rst`](https://github.com/xmos/lib_voice/blob/c9f1a9bf95cd88c7950adf4bf631c217f900ad25/LICENSE.rst)为 XMOS Public Licence v1；商用使用限定 XMOS 设备，并有特殊用途条款。它可以作为可获取的供应商源码研究，但不能标成不受硬件限制的 MIT/BSD 方案。

### I11：XMOS IC 与 VNR 的控制关系

`lib_voice/api/ic/`、`src/ic/` 和 `api/vnr/`、`src/vnr/` 分别给出干扰消除及语音噪声比估计接口。VNR（Voice-to-Noise Ratio）提供控制信息；它不等于一个无条件可靠的“是否有人说话”比特。源码沿用 I10 的版本和许可。[模块目录](https://github.com/xmos/lib_voice/tree/c9f1a9bf95cd88c7950adf4bf631c217f900ad25/lib_voice/src)

应记录主麦与参考麦含义、适配模式、滤波覆盖、状态初始化、VNR 模型及其阈值使用位置。目标语音泄漏到参考通道时，自适应消除可能损害目标，测试需改变目标角度、距离、干扰方向和通道增益。

锁定的 `lib_voice/api/ic/ic_defines.h` 定义 `IC_FRAME_ADVANCE=240`、`IC_FRAME_LENGTH=512`：16 kHz 下每次增加 15 ms 新音频，处理帧长则为 32 ms。两者分别描述更新步长和分析范围，不能把每次调用都解释成要等待 512 个新样本。

把自动适配、强制冻结和旁路输出并排保存，用来判断损伤来自滤波还是控制。近端语音消失时还要检查重新收敛速度，不能仅用一段固定方向噪声证明鲁棒性。

### I12：XMOS NS、AGC 与上游元数据

`lib_voice/api/ns/`、`api/agc/` 及对应 `src/` 是噪声抑制与自动增益的实现入口。[API 目录](https://github.com/xmos/lib_voice/tree/c9f1a9bf95cd88c7950adf4bf631c217f900ad25/lib_voice/api)还包含 `stage1`，便于查看模块拼接时传递的状态。配置、工具链和许可沿用 I10。

查接口时同时看音频数组和元数据：上游活动、回声状态、增益状态在哪一步使用，缺失时采用什么模式。把一个模块单独移植出来时，需要重新定义这些输入，不能只复制处理音频的函数。

推荐做 NS 单开、AGC 单开、两者联用三组对照；记录静音噪声上升、词尾截断、近端音量阶跃及双讲时增益变化。听感和下游识别均应与相同采集增益的未处理基线比较。

### I13：SOF 的固定时域波束形成

Sound Open Firmware（SOF）将波束形成放入 DSP 固件及拓扑系统。可从 [`src/audio/tdfb/`](https://github.com/thesofproject/sof/tree/b6c6a05d52536313fe8e8752b1c4e069b1cc4002/src/audio/tdfb)和 `tdfb_generic.c` 阅读时域滤波求和实现，再追踪配置数据与平台优化路径。仓库总体以 BSD-3-Clause 为主要许可，但 `LICENCE` 说明了混合来源，应保留每个纳入文件的声明。

固定时域波束依赖离线设计的滤波器组、麦克风几何和方向选择。部署清单应包括拓扑、输入输出通道映射、滤波系数、格式、采样率、固件/ABI 版本和方向控制。它不等价于从当前噪声协方差重新求解的 MVDR。[实现文件](https://github.com/thesofproject/sof/blob/b6c6a05d52536313fe8e8752b1c4e069b1cc4002/src/audio/tdfb/tdfb_generic.c)

用脉冲确认各麦到各输出的实际滤波器；再用已知角度声源复测方向响应。交换两个输入通道、改变麦位和施加增益误差，检查目标衰减与噪声放大。通用 C 路径和目标 DSP 路径还应在约定容差内比较。

### I14：SOF 的采样率转换与拓扑

[`src/audio/src/`](https://github.com/thesofproject/sof/tree/b6c6a05d52536313fe8e8752b1c4e069b1cc4002/src/audio/src)提供固件采样率转换模块；其模块名 `src` 表示 Sample Rate Conversion，不能与目录通常表示的“source”混淆。固定比率转换、异步时钟补偿及驱动同步是不同功能，选择时要查看实际拓扑中实例化的模块。

需要保存输入/输出速率、滤波系数组、块调度、通道数、样本格式和计算资源。增加一个 SRC 会改变滤波延迟、缓存需求与频带边缘，不能仅因输出文件采样率正确就认为链路已对齐。

测试合法与不支持的速率组合、分块边界、停止后重启、近奈奎斯特输入和多通道一致性。测量端到端延迟时应保留转换器真实滤波状态，不能在评分前任意独立对齐各通道而掩盖相位问题。

### I15：CMSIS-DSP 的 Q15 FIR

`ARM-software/CMSIS-DSP` 的 [`Source/FilteringFunctions/arm_fir_q15.c`](https://github.com/ARM-software/CMSIS-DSP/blob/83a2d7bc98c81b4bbe4a6f48b1f2ecf179868a0b/Source/FilteringFunctions/arm_fir_q15.c)是定点实现的具体对照，许可为 Apache-2.0。其源注释解释了 1.15 乘法、64 位累加和输出截位/饱和；快速或向量路径仍应分别检查实际实现。

本书 `q15_dot` 使用最近偶数舍入，因此不能默认与截位路径逐位相等。部署应固定系数顺序、状态缓冲大小、块长、内核变体、编译宏、对齐和舍入。不要把“Q15”当成已经完整定义的数值协议。

可用单脉冲、全正/负满幅、交替符号和超过向量宽度的余数长度验证输出。报告最大误差、饱和次数和目标板周期数，并对比对应标量参考；平均误差很小仍可能漏掉边界溢出或系数倒序。

## 4. 模型运行时与资源

### I16：CMSIS-NN 量化内核

`ARM-software/CMSIS-NN` 提供 Cortex-M 神经网络内核，许可为 Apache-2.0。入口为 `Include/arm_nnfunctions.h`、`Source/` 和 `Tests/UnitTest/`。其官方[量化一致性说明](https://github.com/ARM-software/CMSIS-NN)以 TensorFlow Lite Micro 的参考内核为重要对照，不能把这种内核一致性外推为任意导出模型均可运行。

参数包括量化尺度与零点、每通道权重量化、激活截断、scratch 大小、目标 DSP/MVE 指令集和编译选项。前处理的窗、滤波器组、对数和量化同样要固定；输入特征不一致会在整数算子正确时仍造成任务退化。

先以固定整数输入比较参考内核，再比较完整特征与 logits，最后在目标音频测漏报/误报。对不支持算子和临时缓冲不足应显式失败，不能静默换一个数值协议不同的实现。

### I17：TensorFlow Lite Micro 的内存与音频前端

`tensorflow/tflite-micro` 的 [`tensorflow/lite/micro/examples/micro_speech/`](https://github.com/tensorflow/tflite-micro/tree/9f638f18154dff868e7053572f089be845b2fbdf/tensorflow/lite/micro/examples/micro_speech)提供关键词任务入口。它是低资源运行时的例子，不是麦克风阵列的完整处理链。代码许可为 Apache-2.0，模型与训练音频各自记录来源。

[内存管理文档](https://github.com/tensorflow/tflite-micro/blob/9f638f18154dff868e7053572f089be845b2fbdf/tensorflow/lite/micro/docs/memory_management.md)将 tensor arena 分为非持久 head、临时分配和持久 tail。模型文件大小不是运行内存；音频缓存、特征缓存和栈也不一定全部位于 arena。应固定模型 schema、算子 resolver、arena、前端特征和状态寿命。

采用 recording allocator 或对应版本的内存记录 API 测高水位，再以故意缩小 arena 的试验确认错误路径。连续运行要覆盖暂停恢复和输入溢出；仅成功识别一段内嵌样本不能证明实时采集正确。

### I18：ONNX Runtime 的线程、提供程序与状态

本书已锁定 `microsoft/onnxruntime`，核心 C API 在 `include/onnxruntime/core/session/onnxruntime_c_api.h`，许可为 MIT。执行提供程序、模型算子集、动态轴、输入归一化和循环状态必须和源码版本一起记录。

[线程文档](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)区分算子内部 `intra_op_num_threads` 与图节点之间的并行；线程自旋可能降低等待但增加资源和功耗。多路模型各自开线程池时，应测线程竞争，而非把离线单模型最佳线程数复制到全部会话。

分别测建会话、首次推理、预热后推理及端到端流延迟。若算子回退到 CPU，要记录回退和数据传输；GPU 内核耗时不包含全部调用代价。使用相同连续音频检查完整状态保留、错误重置、NaN 输出旁路与模型版本回滚。

## 5. 数据与评分可复现性

### I19：DNSMOS 本地预测

`microsoft/DNS-Challenge` 的 [`DNSMOS/dnsmos_local.py`](https://github.com/microsoft/DNS-Challenge/tree/591184a9fcb2cbdec02520fed81a32bbbf9d73ff/DNSMOS)提供本地推理入口。`LICENSE-CODE` 为 MIT，数据许可由其他文件约束。DNSMOS、DNSMOS P.835 和个性化模型不能仅写成同一个“DNSMOS 分”。

要固定模型文件及摘要、采样率处理、窗长、短文件处理、个性化选项和聚合方式。个性化模式会把干扰说话人按目标任务处理，不能把两种模式的分数混入一张排名表。该分数是学习模型的质量预测，不是实际受试者 MOS，也不覆盖方位和身份追踪。

推荐建立包括干净语音、加噪语音、过度抑制和静音的固定集；保存逐文件分数及失败记录。不能先丢弃低分或无法评分的文件，再只报告成功子集均值。设备选择还需任务识别和主观听测对照。

### I20：AECMOS 的输入与评分区间

`microsoft/AEC-Challenge` 的 [`AECMOS/AECMOS_local/`](https://github.com/microsoft/AEC-Challenge/tree/6c633d0a9d2a143a0e364899b91b06f127315b18/AECMOS/AECMOS_local)提供 ONNX 模型与本地推理。官方 [AECMOS 说明](https://github.com/microsoft/AEC-Challenge/tree/6c633d0a9d2a143a0e364899b91b06f127315b18/AECMOS)区分回声与其他损伤，并说明 Web API 不再更新。源码根许可为 MIT；模型及挑战数据的来源仍须单列。

固定远端参考、麦克风、处理后信号的对应关系、文件长度、采样率和模型版本。2021/2022 测试文件含收敛段，官方说明按单讲/双讲情形裁取评分区域；这些规则绑定相应测试集，不是所有 AEC 文件的通用剪裁法。

分别保存远端单讲、近端单讲和双讲评分。用参考错配、固定延迟、近端削弱和静音输出检查评分流程是否揭示明显错误。不要把高 ERLE 当成 AECMOS 或近端保真已经通过，也不能把评分模型测出的变化写成人类听众的实测差值。

### I21：CHiME-8 数据准备、文本规范化与会议评分

`chimechallenge/chime-utils` 提供 CHiME-8 DASR 数据准备、清单转换和官方评分入口，代码为 MIT。[`README` 的 Scoring 部分](https://github.com/chimechallenge/chime-utils#scoring)说明 SegLST 中的会话、说话人、起止时间和文字字段，提供 cpWER/tcpWER，并在评分前执行该届文本规范化。MeetEval 的具体算法入口见其 `doc/algorithms.md` 与本书锁定清单。

必须记录届次、任务、数据版本、评测区域、规范化、缺失场景策略和评分器提交。`--ignore-missing` 会改变哪些场景计入结果，因此必须在报告中显式说明，不能用它掩盖未处理数据。许可证允许取评分代码，不代表能够直接再分发各原始语料。

建议制作小型人工 SegLST 夹具：完全正确、交换说话人、漏一段重叠语音、时间戳偏移、重复输出。检查评分变动是否符合所选指标的定义，再运行完整数据。这样能区分“算法增强失败”与“转写输出或评分映射失败”。

## 6. 怎样把项目变成可执行实验

每次只选一条具体链。例如 Linux 会议终端可以从 ALSA/PortAudio 采集、PipeWire 或应用内 AEC、固定波束、单通道 NS、VAD/AGC 与后端开始。若图中同时出现 PipeWire 和应用内 AEC，应先明确谁负责消回声及谁取得播放参考。

实验记录至少保存：源码提交、模型摘要、依赖和编译参数、设备与固件、声学条件、通道与时钟、数据集划分、块长、状态初始化/重置、指标、逐文件结果和资源测量。状态相关的回归要同时覆盖整段、分块、暂停恢复与故障注入；算法单元测试不能代替它们。

块长不同需要适配，但不必等待所有块长的最小公倍数。以 16 kHz 为例，10 ms 采集块是 160 点，15 ms 处理步长是 240 点，32 ms VAD 块是 512 点；三者最小公倍数为 7680 点，即 480 ms。若先攒满这 480 ms 才启动全部模块，会人为增加很大等待。连续队列可以在各消费者凑齐自身输入时立即调用，并保留剩余样本；VAD 的控制输出按时间戳使用。验收要检查每个消费者收到的样本无重无漏，以及输出决策对应哪个时间区间。这是缓冲调度算例，不表示上述不同硬件实现应直接串接。

本目录给出的是实现阅读和实验设计。源码实际获取状态见 `codes/upstream/` 的下载记录；构建、运行及目标硬件验收仍按各实验的真实记录报告。
