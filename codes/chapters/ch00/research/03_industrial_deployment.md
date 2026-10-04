# 工业音频实现：从采集、状态到部署和评分

原核实日期：2026-09-22；AEC 接口 I03/I05 与评分 I20 复核日期：2026-09-23；I01 时间戳、I18 输入输出绑定与 I28 源码接口核对日期：2026-09-26；第10章工业边界及 I30～I32 复核日期：2026-09-28；FastEnhancer 论文发表状态于 2026-09-29 另行复核。2026-10-01另核第10章直接接口，并执行 I32 所列五项受限原源码合同；旧报告保留原日期和环境。对应正文第 10、11 章和附录 B。这里讨论本书算法进入连续音频系统后需要补上的部分：驱动、参考路由、跨块状态、定点内核、模型运行时和评分器。每项的完整提交以 [`SOURCES.lock.json`](../SOURCES.lock.json) 为准；网页文档版本只说明核实依据，不自动等于本机安装版本。

除明确写出输入、环境和实测值的本地接口实验外，下面的试验是建议执行的设备验收步骤，不是已经测得的产品结果。源码下载、依赖安装、编译、运行、声学测量是不同状态。没有硬件、模型或数据时，可以完成源码核对，但不能把该项标成已经通过设备验收。

多个设备共同做波形增强时，可从[扩展专题Ⅱ的观测与更新模型](../../../../chapters/15_distributed-enhancement.md)进入本篇的[分布式网络时间轴](#distributed-network-deployment)，再按[固定原源码合同](04_source_reproduction.md#distributed-reproduction)区分实际运行与静态核查。

## 1. 采集、时钟与连续流

### I01：PortAudio 回调与设备时间戳

PortAudio 为不同主机音频系统提供统一入口。本书锁定 19.7.0 对应提交。先读 `include/portaudio.h` 的回调签名，再看 `examples/` 中的采集与播放例子；库为 MIT 许可，平台后端仍有自身依赖。官方[回调说明](https://portaudio.com/docs/v19-doxydocs/writing_a_callback.html)明确限制会阻塞或耗时不可预测的操作。

设备配置要记录 host API、设备 ID、采样率、样本格式、通道数、请求与实际延迟、回调块长。`frameCount` 表示每通道时间样本数，不能乘通道数后再当作帧数。保存输入 ADC 时间、输出 DAC 时间和状态位；它们不能未经校准就当成两个设备共享的时钟。

验收可以用环回脉冲测端到端延迟，再施加磁盘和 CPU 负载。把状态位、回调间隔、队列长度与音频缺口关联起来。一种保守架构是在回调中将样本交给预分配缓冲；也可执行已验证最坏耗时的有界处理，但不能把可能阻塞的 Python 推理或同步日志直接塞入回调。设备拔插后需重新确认通道映射，而非只恢复原设备编号。

固定 [`portaudio.h`](https://github.com/PortAudio/portaudio/blob/3f7bee79a65327d2e0965e8a74299723ed6f072d/include/portaudio.h)允许 `paFramesPerBufferUnspecified`，此时实际 `frameCount` 可以变化。处理耗时应除以本次新音频时长 `frameCount / sampleRate`；重叠 STFT 应使用帧移对应的新音频时长，不能用包含历史样本的分析窗长扩大预算。硬件块、算法块和播放块可通过有界队列适配，不要求把它们统一成最小公倍数大块。

`paInputOverflow` 只报告输入发生丢弃，不提供一个可移植的精确丢样计数。固定块模式甚至允许丢弃出现在块内一个或多个位置；可变块模式说明丢弃在首样本之前。状态位、时间戳、设备计数器和重启事件须分别记录。只有额外依据足够时才填具体丢样数，否则保留“发生不连续、数量未知”，不能填零并继续假定 AEC 或重采样状态完整。

**把测点和对应样本一起保存。** [官方时间戳结构体](https://files.portaudio.com/docs/v19-doxydocs/structPaStreamCallbackTimeInfo.html)分别表示输入首样本的 ADC 时刻、实际回调调用时刻与输出首样本的 DAC 时刻。只有属于同一时间基准且知道哪一个输入样本对应哪一个输出样本时，端点之差才是需要的延迟。工作线程若把结果延后一个输出块，必须跟着改样本映射；不能继续减当前回调的两个首点时刻。

[E10-17](../../../../chapters/10_engineering-practice.md#sec-10-11)给出 15 ms 输入年龄、30 ms 输出提前量与 45 ms 对应样本延迟的手算，并展示多排一块后变成 55 ms。3 ms 计算已位于端点之间，不能重复加上。PortAudio 的[缓冲与时序指南草案](https://github.com/PortAudio/portaudio/wiki/BufferingLatencyAndTimingImplementationGuidelines)也说明部分后端只能报告已知延迟；API 字段不是实际精度的保证，仍须用目标后端和环回信号核查。上述例子没有打开声卡。

### I02：ALSA PCM 状态与恢复

Linux 设备层可对照 `alsa-project/alsa-lib` 的 `src/pcm/pcm.c` 和 [`test/pcm.c`](https://github.com/alsa-project/alsa-lib/blob/f84cd4ced7b36fddb8e4ee24404cf7c091d27020/test/pcm.c)。项目 `COPYING` 为 LGPL 2.1 文本，具体源文件的版本选择和例外仍按其文件头核对。PCM（脉冲编码调制）接口把硬件配置、软件启动阈值和流状态分开。

[官方 PCM 文档](https://www.alsa-project.org/alsa-doc/alsa-lib/pcm.html)区分：`-EPIPE` 是播放下溢或采集上溢；`-ESTRPIPE` 与挂起有关；`-ENODEV` 可表示设备被移除；某些回声参考设备在关联播放未启动时可返回 `-ENODATA`。因此不能对所有负返回值无条件调用同一恢复函数，并假设音频仍连续。

规格要固定 period、buffer、`avail_min`、启动阈值、访问布局和时间戳模式。测试短读写、CPU 暂停、系统挂起、USB 拔插和“先开参考、后开播放”。恢复后记录不连续事件；仅在设备计数器或可信时间映射足以确定缺口时给出丢样数量，并同时保存推断依据。单独的 `-EPIPE` 不能推出精确数量。通知 AEC、重采样器及追踪状态按各自策略恢复；重新 `prepare` 不会补回已经丢失的数据。本节没有执行 Linux 声卡或热插拔实验。

### I03：PipeWire 的 AEC 四条流

PipeWire 官方开发仓库位于 freedesktop GitLab，`PipeWire/pipewire` 是项目维护的 GitHub 镜像。入口为 [`src/modules/module-echo-cancel.c`](https://github.com/PipeWire/pipewire/blob/62316ae8cf659ca8e5f1ee866ae743d2438d47a4/src/modules/module-echo-cancel.c) 和 [`spa/plugins/aec/aec-webrtc.cpp`](https://github.com/PipeWire/pipewire/blob/62316ae8cf659ca8e5f1ee866ae743d2438d47a4/spa/plugins/aec/aec-webrtc.cpp)。项目根 `COPYING` 为 MIT；WebRTC 后端依赖另行保留许可证。

[PipeWire 1.6.9 文档](https://docs.pipewire.org/page_module_echo_cancel.html)将麦克风 capture、应用 source、应用播放 sink 和实际 playback 四条流分开。要抵消的播放必须经过选定参考路由。`monitor.mode` 改变参考获取方式；`audio.rate`、通道布局、目标节点与 `node.latency` 都要与实际图连接一起保存。

锁定版 `meson.build` 依安装环境选择 `webrtc-audio-processing-2`、`-1` 或旧版依赖；不能因为 PipeWire 和本书分别提到 WebRTC，就推断它们使用相同 AEC3 提交。[后端 `aec-webrtc.cpp`](https://github.com/PipeWire/pipewire/blob/62316ae8cf659ca8e5f1ee866ae743d2438d47a4/spa/plugins/aec/aec-webrtc.cpp)在该版本默认开启高通和噪声抑制、关闭 AGC，并将 `(num_blocks-1)×10` ms 作为流延迟传入 APM；这不是直接读取播放/采集硬件时间戳。因此用 PipeWire 虚拟麦克风与直接 APM 做性能对照时，必须另记系统包版本、处理开关、`num_blocks`、参考路由和真正的输出取点，不能把差值都归给消回声内核。

测试会议语音、通知声和音乐是否都进入参考；切换默认扬声器后再次检查。用断开参考和改变播放延迟的试验观察 AEC 恢复。系统已启用 AEC 时，应用再启用一套 AEC 会改变输入统计与近端损伤，应设置单套与双套处理对照，不能只检查虚拟麦克风能否出现。

若应用直接接入锁定版 WebRTC APM，而不是使用 PipeWire 虚拟设备，应按 [`api/audio/audio_processing.h`](https://webrtc.googlesource.com/src/+/0467d2b91cc20b9b001c2bbb73d43ea6b2491f3e/api/audio/audio_processing.h)区分两路：启用 AEC 后，将送往播放硬件的参考帧交给 `ProcessReverseStream()`，每次处理采集帧前按硬件时间设置 `set_stream_delay_ms()`，再调用 `ProcessStream()`。APM 对外按约 10 ms 线性 PCM 帧工作；有符号 16 位接口通道交织，浮点接口按通道分列。

固定提交 `0467d2b91cc20b9b001c2bbb73d43ea6b2491f3e` 的 `int16` 接口限用 8、16、32、48 kHz，并要求采集输入、输出和播放反向流同速，采集输出布局与输入一致；浮点接口允许更广的合法速率和不同布局。44.1 kHz 整数 PCM 不能直接按 `int16` 接口送入，应先转换或改按浮点接口准备。[头文件 `Initialize()` 约束及 `NativeRate` 枚举](https://webrtc.googlesource.com/src/+/0467d2b91cc20b9b001c2bbb73d43ea6b2491f3e/api/audio/audio_processing.h#509)

`set_stream_delay_ms()` 的参数由参考进入 APM 后至实际播放、以及麦克风采样后至进入 APM 的缓冲时间构成，不能用房间传播时间或录音文件头里的采样率代替。

固定版 [`audio_processing_impl.cc`](https://webrtc.googlesource.com/src/+/0467d2b91cc20b9b001c2bbb73d43ea6b2491f3e/modules/audio_processing/audio_processing_impl.cc) 对超出 0～500 ms 的延迟值截断并返回警告；记录调用参数、返回码和 `stream_delay_ms()` 读回值，才能知道真正送入 AEC 的值。要比较线性抵消与最终输出，配置 `echo_canceller.export_linear_aec_output=true`，在 `ProcessStream()` 后调用 `GetLinearAecOutput()` 并检查返回值。该公开取点约为 10 ms、16 kHz；与最终输出对齐时不能把它按输入设备采样率直接拼接。两套 AEC 并开时，两边都要记录实际获得的参考与处理取点；只看到虚拟设备存在并不能证明应用内 APM 收到了正确参考。接口细节另见[增强研究 A06](02_aec_wpe_separation.md#a06-aec3-的延迟线性抵消残余抑制与舒适噪声)。

### I04：libsamplerate 的有状态重采样

`libsndfile/libsamplerate` 的 README 标识 0.2.2，根许可为 BSD-2-Clause。核心入口 [`src/samplerate.c`](https://github.com/libsndfile/libsamplerate/blob/0844c208f683527c08ea8a80acc13b398aa9c8bf/src/samplerate.c) 的 `src_new`、`src_process` 与 `src_reset` 管理转换状态；滤波实现位于 `src/`。

[Full API](https://libsndfile.github.io/libsamplerate/api_full.html)规定 `src_ratio=输出采样率/输入采样率`，调用后分别返回实际消耗的 `input_frames_used` 和生成的 `output_frames_gen`。不能假设每次恰好消耗整个输入块，也不能把多个连续块各自交给一个新实例。最后一个输入块设置 `end_of_input=1`；若输出缓冲尚未取尽，保持同一状态继续调用，直至不再产生输出。只在文件头写入新的采样率，或消耗完输入就停止调用，均不能完成上述转换协议。

逐次改变 `SRC_DATA.src_ratio` 时，库会在上次与本次的比例间平滑过渡；调用 `src_set_ratio` 可让下一次处理从新比例开始，绕过这一平滑。两者用于不同控制目的，不能把“更新了比例”当成已经实现时钟估计或低伪影切换。[官方 Full API 的 Set Ratio 小节](https://libsndfile.github.io/libsamplerate/api_full.html)

本书 SRO 约定中设备快 100 ppm 时，转换到参考时钟的比例为 `1/1.0001≈0.99990001`。用同一宽带信号比较整段与分块输出；补偿固定滤波延迟后检查残差，再测接近奈奎斯特频率的带外泄漏。比例更新还要测突变、平滑过渡和队列长期漂移；短录音长度看似正确不足以说明相位相干。

本书另以[可重跑的 C 接口实验](../../ch10/examples/run_libsamplerate_sro.py)核对一个受限情形。输入是 10 s 的单通道数学合成脉冲：参考时钟 16 kHz，设备时钟在前 5 s 快 100 ppm（16 001.6 样本/s），后 5 s 快 150 ppm（16 002.4 样本/s），脉冲发生于参考时间 1、3、4、6、8、9 s。按两段时钟积分并四舍五入，设备共生成 160 020 帧；从独立时钟算式可得 1～9 s 两个脉冲的未校正相对偏移增量为 16 帧。设备采样率在本实验中是**事先给定的真值**，并非由音频估计。

锁定 `libsamplerate` 0.2.2、提交 `0844c208f683527c08ea8a80acc13b398aa9c8bf`，本机 macOS arm64、Apple clang 21.0.0、CMake 4.3.2，仅在临时目录构建静态库；`SRC_SINC_MEDIUM_QUALITY`、257 帧输入块、64 帧输出缓冲、浮点单通道、状态跨块保留。未校正组的比例始终为 1；校正组在两段分别传入 `16000/16001.6≈0.99990001` 和 `16000/16002.4≈0.99985002`。两组最后一个输入块都设 `end_of_input=1`，随后继续取尽尾部。该实验使用 `SRC_DATA.src_ratio` 的跨调用平滑，不测试 `src_set_ratio` 的突变控制。

| 对照 | 输出总帧数 | 六个脉冲相对参考时间的峰值采样点偏移（帧） | 1～9 s 偏移增量（帧） | 部分消费调用 | 输入取尽后补出帧数 |
|---|---:|---|---:|---:|---:|
| 不校正，比例 1 | 160 020 | 2，5，6，10，15，18 | 16 | 1 875 | 84 |
| 按已知时钟更新比例 | 159 999 | 0，0，0，0，0，0 | 0 | 1 874 | 127 |

表中的“部分消费”由每次 `input_frames_used < input_frames` 计数，两组实际累计消耗均为 160 020 输入帧；输出计数、比例和排空顺序还由逐调用记录独立检查。校正组总帧数比 10 s×16 kHz 少 1 帧，不能因为内部六个标记对齐就写成输出长度严格等于 160 000。此处只观察孤立脉冲峰值位置，没有拟合固定时移或增益；脉冲之间的波形、带外抑制和听感不是表中测量项。

运行命令是 `.venv/bin/python codes/chapters/ch10/examples/run_libsamplerate_sro.py`。脚本先核对本地上游提交及范围，完成临时构建后输出 JSON；源文件、二进制输出和调用轨迹均不进入发布仓库。对应[独立算式与接口失败用例测试](../../../../tests/test_codes_libsamplerate_sro.py)检查错误比率、丢失输入、过早结束与遗漏排空。实际设备仍需另测 SRO 估计误差、比例更新伪影、抗混叠、时间戳闭环和长时队列稳定；本实验不能代表这些项目通过。

另一个[估计—补偿—核查小实验](../../ch10/sro_closed_loop_demo.py)使用项目 Python/NumPy 环境，不需要下载上游源码或连接声卡。当前 `fit_delay_ppm` 只是[唯一 `estimate_sro_ppm` 核](../../ch10/core/engineering.py)的薄调用，避免两份拟合实现独立变化。2026-09-28记录的旧标准库环境和下面原数值保留为当时的运行条件；当前复跑使用 `.venv`。它生成 12 s 的解析合成波形与**精确的逐样本合成时间戳**：参考为 2 000 Hz，设备为 2 000.3 Hz（+150 ppm），首样本在参考零点后 2 ms 到达，物理设备索引 12 000 的样本被删去。参考序号 $n$ 的名义时刻为 $n/2000$，未缺样设备序号 $n$ 的时刻为 $0.002+n/2000.3$ s，因此前 2 s 的九个锚点按本书延迟约定给出截距 $-2$ ms、斜率 $150\times10^{-6}/(1+150\times10^{-6})$。示例的直线拟合一阶读数为 149.9775 ppm，按 $s/(1-s)$ 反解为 150 ppm；截距另报初始偏移 2 ms。先从相邻时间戳识别倍数间隔并恢复物理索引，再拟合**缺样之前**的锚点，避免把缺样阶跃误认成稳定 SRO。

校准使用设备索引 0～4 000 的锚点；之后才把后续样本送入保留上个设备样本与输出相位的线性插值器，从参考索引 4 500 开始输出。257 样本输入块下，缺口使参考索引 12 002、12 003 标为无效，没有跨缺口连线。以解析参考波形计算无量纲幅值平方的均方误差（MSE），结果如下；未校正组直接按接收序号与参考序号比较，所以同时包含初始偏移、速率积累与缺样后的序号错位。

| 参考样本区间 | 有效校正样本数 | 未校正 MSE | 校正后 MSE |
|---|---:|---:|---:|
| 5 000～10 999，缺样前 | 6 000 | 0.02869 | 0.000002773 |
| 13 000～21 999，缺样后 | 9 000 | 0.02131 | 0.000002477 |

运行 `.venv/bin/python codes/chapters/ch10/sro_closed_loop_demo.py` 即得到参数、估计、缺口和逐段 JSON；[独立单元测试](../../../../tests/test_codes_sro_closed_loop.py)核对解析斜率、2 ms 截距、单点缺样、跨块相位及 257/509 样本分块一致性。这里是无时间戳噪声、恒定速率且合成波形可解析的教学闭环；MSE 还包括线性插值误差，不可解释为纯时间残差。本实验不验证实际声卡、盲音频估计、抗混叠、动态比例更新或长时队列稳定。上述 libsamplerate 实验验证真实 C 接口的有状态调用，但使用已知比率；两者共同仍不能替代设备验收。

2026-09-23 的本机环境核查只枚举设备，未打开麦克风或录制音频。`system_profiler SPAudioDataType -json` 在主机侧列出一台标称 48 kHz、单输入通道的“MacBook Air 麦克风”和一台内建扬声器；受限沙箱中的同一命令曾返回空列表，因此不能用该空结果断言没有设备。CoreAudio 再次枚举到这一输入和输出，其 `kAudioDevicePropertyClockDomain` 值相同且非零。按 [Apple 的时钟域说明](https://developer.apple.com/documentation/coreaudio/audiohardwareclock/clockdomain)，同一非零域的设备可在硬件上同步；这组内建设备不能充当两个已确认独立的输入时钟。本机当时缺少第二台独立输入设备，因而没有测得可报告的真实设备相对 SRO 或 ppm 值。实际测量还需核查两台设备的时钟域、驱动或聚合设备是否隐式重采样，并保存两路原始帧序号与时间戳，再从多个时间窗区分初始错位、稳定斜率和丢样阶跃。

锁定版 WebRTC AEC3 的 [`render_delay_controller.cc`](https://webrtc.googlesource.com/src/+/0467d2b91cc20b9b001c2bbb73d43ea6b2491f3e/modules/audio_processing/aec3/render_delay_controller.cc)提供 `HasClockdrift()`，`block_processor.cc` 将该状态传给回声路径控制。检测到延迟轨迹变化、调节参考缓冲或重置路径状态，不等于已经把两个独立设备时钟重采样到同一速率。若产品跨时钟域采集与播放，应在输入时间戳、长期参考队列和补偿前后残余上验证 SRO 闭环；本书已运行的已知比率重采样实验也不替代这一设备验收。

### I05：SpeexDSP 的流式转换与回声缓冲

SpeexDSP 的 `libspeexdsp/resample.c`、`include/speex/speex_resampler.h` 是另一组连续重采样入口。`libspeexdsp/mdf.c` 提供分块频域回声处理，可与第 6 章教学 NLMS 对照。代码根许可为 BSD-3-Clause，版本和提交沿用锁定清单。重采样器和回声消除器是两个独立状态；取得 SpeexDSP 源码不表示一打开 AEC 就自动纠正播放/采集采样率偏移。

重采样配置包括输入/输出速率、通道数、质量档位、整数或浮点输入，以及每次调用的输入/输出长度。AEC 的帧长、滤波覆盖长度、采样率和播放参考缓冲另行固定；两者共享“连续状态”要求，但不能共用一个模糊的延迟参数。[固定版 AEC 头文件](https://gitlab.xiph.org/xiph/speexdsp/-/blob/8e29a256ef0235ebbe7fcb8417b5ac7731eb8307/include/speex/speex_echo.h)说明 `speex_echo_cancellation()`由应用交付已经对应的播放帧，不额外加入两帧播放缓冲；`speex_echo_playback()`/`speex_echo_capture()`为异步辅助入口，内部预留两帧，实际声卡延迟不一定相等。[官方手册](https://www.speex.org/docs/manual/speex-manual/node7.html)明确要求回声进入麦克风前，对应播放帧已到达回声消除器，过长的额外延迟还会占用滤波器覆盖长度。

这一固定提交的 `mdf.c` 初始化时将采样率置为 **8000 Hz**；16 kHz 输入不能只设置 WAV 文件头。创建状态后要用 `speex_echo_ctl(..., SPEEX_ECHO_SET_SAMPLING_RATE, &rate)` 显式设置，再用 `SPEEX_ECHO_GET_SAMPLING_RATE` 读回验证。[源码初始化与控制分支](https://gitlab.xiph.org/xiph/speexdsp/-/blob/8e29a256ef0235ebbe7fcb8417b5ac7731eb8307/libspeexdsp/mdf.c)。上游 `testecho.c` 只演示接口，循环不核对每次 `fread` 的短读和文件结束；严谨评分应自己验证完整帧数、输入长度及尾部处理。本书的[真实配对 Speex 实验](02_aec_wpe_separation.md#aec)使用严格 PCM16 帧驱动，已在 macOS arm64 上运行，但不代表异步接口或设备链路也已通过。

多通道使用 `speex_echo_state_init_mc(frame_size, filter_length, nb_mic, nb_speakers)`。固定提交 `8e29a256ef0235ebbe7fcb8417b5ac7731eb8307` 的 [`mdf.c`](https://gitlab.xiph.org/xiph/speexdsp/-/blob/8e29a256ef0235ebbe7fcb8417b5ac7731eb8307/libspeexdsp/mdf.c#L687-733) 同步路径分别按 `i*K+speak`、`i*C+chan` 读取交织的扬声器与麦克风通道；异步 `speex_echo_playback()` 每次却只复制 `frame_size` 个标量播放样本，`speex_echo_capture()` 在参考欠载时也只复制 `frame_size` 个输出样本，均未按通道数扩展。因此该固定版异步辅助函数**不能直接接多通道交织帧**，否则参考缓冲和欠载输出可能只覆盖部分通道；这不是 `speex_echo_state_init_mc()` 已提供完整多通道异步队列的证据。

对多通道先由应用按完整交织帧维护参考队列，再用显式对齐的 `speex_echo_cancellation()`；以各通道在**不同采样索引**的单脉冲检查映射，给所有通道相同脉冲无法检出通道交换。该提交的 `SPEEX_ECHO_GET_IMPULSE_RESPONSE` 路径还标注多通道未实现，不能把返回的单一路径数据当作多通道真值。本书已运行的真实配对 Speex 实验只有单播放、单麦克风同步接口，未验证多通道或异步设备链路。

异步单通道路径也不能只按“内置两帧缓冲”理解。`mdf.c` 的 `speex_echo_playback()` 在首次采集调用前可丢弃播放帧；`speex_echo_capture()` 在参考缓冲欠载时直接复制麦克风输入到输出，并发出告警。[固定版 `mdf.c` 的异步入口](https://gitlab.xiph.org/xiph/speexdsp/-/blob/8e29a256ef0235ebbe7fcb8417b5ac7731eb8307/libspeexdsp/mdf.c)

可在固定播放 PCM 上分别运行显式配对的同步接口与异步接口，记录每帧的播放索引、缓冲占用、告警、输入与输出是否完全相同。同步正常而异步开场或缺帧时泄漏，先修调用顺序和应用缓冲；不能凭这一现象断定 AUMDF 收敛慢或增加滤波长度。[Speex 官方手册的调试说明](https://www.speex.org/docs/manual/speex-manual/node7.html)

Speex 文件头说明其 AUMDF 通过连续学习率提高双讲鲁棒性，没有独立二值 DTD。若还要使用残余回声抑制，[官方手册](https://www.speex.org/docs/manual/speex-manual/node7.html)要求将回声状态通过 `SPEEX_PREPROCESS_SET_ECHO_STATE` 绑定到预处理器；固定版 `preprocess.c` 才会读取 `speex_echo_get_residual()`。比较算法时应保留核心 MDF 输出和预处理后输出两个取点，不能用后者的安静程度倒推线性抽头收敛。

用跨块脉冲检查重采样边界连续性；先以各通道不同索引的脉冲检查交换、漏通道和交织顺序，再用共同输入检查跨通道相位一致性。AEC 另测参考领先/滞后、远端单讲、双讲、路径突变与异步缓冲欠载，记录每次调用的真实播放参考索引。重采样波形连续不能代替消回声验收；三种 AEC 的同输入对照口径见[增强研究 A07](02_aec_wpe_separation.md#aec)。

## 2. 噪声、语音活动与增益

第 10 章[功率谱减手算与 E10-13](../../../../chapters/10_engineering-practice.md#sec-10-3-2)先固定噪声专用帧、逐频点平均噪声功率和相对观测功率的谱地板；[本书实现](../../ch10/core/noise_suppression.py)只承担这一可复算基线。下面的 WebRTC VAD、DeepFilterNet 和模型运行时具有不同的状态与训练假设，不能拿本书谱减的固定噪声估计和单次合成试听推断它们的质量或计算量。谱减仍需在目标输入上单独核验噪声估计失配、音乐噪声、任务损伤和跨块处理；当前 WAV 不含真实语音或设备采集。

### 从固定噪声功率到受控的在线更新

噪声变大后，继续用开头的固定 $D_f$ 会留下更多残留；把所有带语音帧直接纳入噪声平均，又会将语音当噪声减掉。MCRA（Minima Controlled Recursive Averaging，最小值控制递归平均）改变的是**噪声功率估计的更新速度**。Cohen 与 Berdugo 的[2002年作者原稿](https://israelcohen.com/wp-content/uploads/2018/05/SPL_Jan2002.pdf)第13页、§II～III式(3)～(14)依次给出语音存在/不存在时的更新、存在概率平均、局部平滑、双最小值跟踪和判决；2026-10-01已阅读原页。它不是只给最终谱减增益乘一个二值开关。

若当前软存在概率是 $p_t$、无语音时平滑系数为 $\alpha_d\text{，}$噪声递推使用 $\widetilde\alpha_t=\alpha_d+(1-\alpha_d)p_t$。例如本书给定 $\alpha_d=0.8$、旧噪声功率2、当前观测功率10：$p_t=0$时更新为3.6，$p_t=0.9$时仅更新为2.16。数值均按同一频点、同一功率单位计算；它们只复算给定概率下的一步递推，尚未构成带最小值搜索和概率估计的完整MCRA。其受控连续例见[E10-28](../../../../chapters/10_engineering-practice.md#e10-28)，更完整的机制及IMCRA两轮区别见[空间研究§50、51](01_spatial_and_tracking.md#sec-u-2973a53a28)。

工业代码并不都使用同一软概率。本锁版 SpeexDSP 的 `preproc_update_prob` 在 $0.4S>S_{\min}$ 时设置逐频点硬标记；`preproc_noise_update` 在无标记**或当前功率低于旧噪声**时才更新。I32 的原函数小例说明它仍能跟踪下降的功率，但不能据此称作原MCRA/IMCRA全链。Cohen 的[作者软件页](https://webee.technion.ac.il/Sites/People/IsraelCohen/Info/Software.html)提供OM-LSA/IMCRA下载入口；本轮点击未取得可核软件包，也未发现明确再分发许可，因此只引用原文和网页，不将作者MATLAB代码或权重随仓分发。

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

#### 锁定插件的等待与延迟状态边界

以上 README 声明须和实际消费路径分开。[`ladspa/src/lib.rs`](https://github.com/Rikorose/DeepFilterNet/blob/d375b2d8309e0935d165700c91da9de862a99c31/ladspa/src/lib.rs#L213-L224)初始化时在每个输出队列放入一个 `hop` 的零样本。`run` 把输入推入共享队列，再反复加锁检查输出是否足够；不足时调用 `sleep` 等待。该路径不提供有界等待保证，不能因插件可接入 PipeWire 就认为满足 I01 的硬实时回调条件。这里没有安装 Rust 依赖、加载模型或运行宿主。

同文件 `run` 的 L434～459 在当次处理时间与音频时间之比不小于1时增加一个帧移的缓冲；内部 `proc_delay` 已达到一秒样本数时会触发 `panic`。这些是版本控制逻辑，不是本书测得的延迟或实际设备故障率。

L461～479 的减延迟分支存在可定位的轴混淆：`o_q` 是按通道排列的队列，`o_q.iter_mut().take(self.frame_size)` 截取的是通道数，每个通道只执行一次 `pop_front`；随后却将 `proc_delay` 减去整个 `frame_size`。取两个通道、帧移480、每路初始960样本且满足该分支前提，本书独立队列小例得到每路余959样本、首剩样本索引1，而状态计数从960变480。若实际丢弃一整帧移，每路应余480样本、首索引480。

这说明该分支的队列变化和元数据不一致；它不证明神经增强内核本身错误，也不能据此算出设备端总延迟。原版保持不改；[I32 的诊断](#industrial-upstream-audit)绑定原文件摘要，明确这是静态源码核查和独立 Python 算例，非 Rust 插件执行。

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

离线设计脚本还须单独核查。第5章已对同一固定版的 `sof_bf_design.m` 记录归一化 `sinc` 自变量和 WNG 分母变量的两个问题，见[独立诊断脚本](../../ch05/examples/audit_sof_tdfb_design.py)及[原始报告](../../ch05/reports/sof_tdfb_design_audit.json)。那是固定源码静态核对与 Python 数学反例，未运行 MATLAB、生成并测试真实 SOF FIR 或刷写固件；本节不把官网响应图或源码存在当作设计与板端性能已验收。

### I14：SOF 的采样率转换与拓扑

[`src/audio/src/`](https://github.com/thesofproject/sof/tree/b6c6a05d52536313fe8e8752b1c4e069b1cc4002/src/audio/src)提供固件采样率转换模块；其模块名 `src` 表示 Sample Rate Conversion，不能与目录通常表示的“source”混淆。固定比率转换、异步时钟补偿及驱动同步是不同功能，选择时要查看实际拓扑中实例化的模块。

需要保存输入/输出速率、滤波系数组、块调度、通道数、样本格式和计算资源。增加一个 SRC 会改变滤波延迟、缓存需求与频带边缘，不能仅因输出文件采样率正确就认为链路已对齐。

测试合法与不支持的速率组合、分块边界、停止后重启、近奈奎斯特输入和多通道一致性。测量端到端延迟时应保留转换器真实滤波状态，不能在评分前任意独立对齐各通道而掩盖相位问题。

### I15：CMSIS-DSP 的 Q15 FIR

`ARM-software/CMSIS-DSP` 的 [`Source/FilteringFunctions/arm_fir_q15.c`](https://github.com/ARM-software/CMSIS-DSP/blob/83a2d7bc98c81b4bbe4a6f48b1f2ecf179868a0b/Source/FilteringFunctions/arm_fir_q15.c)是定点实现的具体对照，许可为 Apache-2.0。其源注释解释了 1.15 乘法、64 位累加和输出截位/饱和；快速或向量路径仍应分别检查实际实现。

本书 `q15_dot` 使用最近偶数舍入，因此不能默认与截位路径逐位相等。部署应固定系数顺序、状态缓冲大小、块长、内核变体、编译宏、对齐和舍入。不要把“Q15”当成已经完整定义的数值协议。

原 [`arm_fir_init_q15.c`](https://github.com/ARM-software/CMSIS-DSP/blob/83a2d7bc98c81b4bbe4a6f48b1f2ecf179868a0b/Source/FilteringFunctions/arm_fir_init_q15.c)的合同要求偶数抽头且至少4，系数按时间逆序存储。无 `ARM_MATH_LOOPUNROLL` 的标量配置要求状态空间至少 `numTaps+blockSize-1` 个Q15项；开启该宏时是 `numTaps+blockSize`，Helium系数还有8项倍数的补零要求。这些是不同编译路径的前提，不能把一个分支的缓冲长度套给所有内核。某分支没有拒绝不合法抽头，也不等于该输入获得支持。

I32的新合同调用只编译未修改的标量FIR与初始化两个翻译单元，使用四抽头、明确逆序系数、自写最小类型头和受限饱和适配器。系数表示的因果响应 $[0.5,0.5,0,0]$ 对Q15整数输入 $[1,0,-1,0,1]$ 得到 $[0,0,-1,-1,0]$：此处正半LSB截成0，负半LSB右移为−1，与最近偶数舍入不同。另有负满幅饱和与非对称系数的整段/保留状态分块对照；没有运行ARM优化指令、目标板或周期计时。

可用单脉冲、全正/负满幅、交替符号和超过向量宽度的余数长度验证输出。报告最大误差、饱和次数和目标板周期数，并对比对应标量参考；平均误差很小仍可能漏掉边界溢出或系数倒序。

## 4. 模型运行时与资源

### I16：CMSIS-NN 量化内核

`ARM-software/CMSIS-NN` 提供 Cortex-M 神经网络内核，许可为 Apache-2.0。入口为 `Include/arm_nnfunctions.h`、`Source/` 和 `Tests/UnitTest/`。其官方[量化一致性说明](https://github.com/ARM-software/CMSIS-NN)以 TensorFlow Lite Micro 的参考内核为重要对照，不能把这种内核一致性外推为任意导出模型均可运行。

参数包括量化尺度与零点、每通道权重量化、激活截断、scratch 大小、目标 DSP/MVE 指令集和编译选项。前处理的窗、滤波器组、对数和量化同样要固定；输入特征不一致会在整数算子正确时仍造成任务退化。

先以固定整数输入比较参考内核，再比较完整特征与 logits，最后在目标音频测漏报/误报。对不支持算子和临时缓冲不足应显式失败，不能静默换一个数值协议不同的实现。

### I17：TensorFlow Lite Micro 的内存与音频前端

`tensorflow/tflite-micro` 的 [`tensorflow/lite/micro/examples/micro_speech/`](https://github.com/tensorflow/tflite-micro/tree/9f638f18154dff868e7053572f089be845b2fbdf/tensorflow/lite/micro/examples/micro_speech)提供关键词任务入口。它是低资源运行时的例子，不是麦克风阵列的完整处理链。代码许可为 Apache-2.0，模型与训练音频各自记录来源。

TensorFlow Lite Micro 要求调用方先划出一块内存，供推理时的张量和临时计算使用；这块预留区称为 `tensor arena`。例如模型文件即使只占 100 KiB，运行时仍可能需要另一块内存保存中间张量，不能直接用文件大小估算内存上限。这里的 100 KiB 仅为说明两种内存不是同一个量，并非该项目的实测配置。

[内存管理文档](https://github.com/tensorflow/tflite-micro/blob/9f638f18154dff868e7053572f089be845b2fbdf/tensorflow/lite/micro/docs/memory_management.md)把这块区域分成可复用的非持久 `head`、临时分配区和存放持久数据的 `tail`。模型文件大小不是运行内存；音频缓存、特征缓存和线程栈也不一定全部位于这块区域。复现时应固定模型格式版本（schema）、算子注册器（resolver）、预留区大小、音频特征和状态保留时长。

采用 recording allocator 或对应版本的内存记录 API 测高水位，再以故意缩小 arena 的试验确认错误路径。连续运行要覆盖暂停恢复和输入溢出；仅成功识别一段内嵌样本不能证明实时采集正确。

### I18：ONNX Runtime 的线程、提供程序与状态

本书已锁定 `microsoft/onnxruntime`，核心 C API 在 `include/onnxruntime/core/session/onnxruntime_c_api.h`，许可为 MIT。执行提供程序、模型算子集、动态轴、输入归一化和循环状态必须和源码版本一起记录。

[线程文档](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)区分算子内部 `intra_op_num_threads` 与图节点之间的并行；线程自旋可能降低等待但增加资源和功耗。多路模型各自开线程池时，应测线程竞争，而非把离线单模型最佳线程数复制到全部会话。

分别测建会话、首次推理、预热后推理及端到端流延迟。若算子回退到 CPU，要记录回退和数据传输；GPU 内核耗时不包含全部调用代价。使用相同连续音频检查完整状态保留、错误重置、NaN 输出旁路与模型版本回滚。

**输入输出绑定解决哪一笔开销。** 非 CPU 执行后端可能在 `Run()` 内把 CPU 输入复制到设备，再把输出复制回 CPU。[官方 I/O Binding 文档](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html)说明，可以预先绑定设备上的输入与输出，减少这些往返。对于流式增强，若上一块输出的模型缓存下一块仍在同一设备使用，反复把缓存取回主机再传回设备会增加传输；能否留在设备，取决于实际模型接口与执行后端，不能只根据模型名称判断。

锁定提交 `48795e0281bcaa6b3b63b28af5e078b4c41e995f` 的 [C API](https://github.com/microsoft/onnxruntime/blob/48795e0281bcaa6b3b63b28af5e078b4c41e995f/include/onnxruntime/core/session/onnxruntime_c_api.h)提供以下阅读顺序：

| 接口 | 需要核对的对象 | 不能省略的条件 |
|---|---|---|
| `CreateIoBinding`、`BindInput` | 模型输入名、`OrtValue`、所在设备 | 数值类型、形状和设备必须匹配；外部缓冲在使用期间保持有效 |
| `BindOutput` | 已知形状的预分配输出 | 模型输出尺寸必须适合已分配空间，不把上一块旧值当成新输出 |
| `BindOutputToDevice` | 仅指定设备的动态形状输出 | 输出形状事前未知时让会话分配；不能由此宣称每块没有分配 |
| `RunWithBinding` | 一组输入、输出及运行选项 | 固定执行提供程序及 CPU 回退范围，保留模型的连续缓存 |
| `SynchronizeBoundInputs`、`SynchronizeBoundOutputs` | 不同执行流或外部缓冲之间的完成顺序 | 固定版注释说明行为依赖后端，某些后端为空操作，不能当作统一 GPU 计时器 |

设备侧缓存不等于全链路零复制：麦克风样本来自哪里、频谱在 CPU 还是 GPU 计算、最终播放 PCM 在哪里消费，都决定仍需哪些传输。输出绑定也不自动授权输入、输出共用一块存储；原地复用必须由所选算子和运行时协议另行支持。

最小实验应保持相同模型、连续输入、线程和执行后端，分别测普通调用、正确绑定及每块强制取回缓存。比较前先确认三路输出在同一状态初始化、排空和对齐下等价，再记录主机到设备、计算、设备到主机以及输出真正可用的时间。异步后端须按其协议同步测点；只计提交请求到返回的短时间，不能称为已完成推理延迟。

这些是固定 API 的源码核对与实验设计。本书没有在 GPU 上运行这组三路对照，也未测得复制减少量、加速比或功耗改善；I/O Binding 是运行时接口，不额外算作一种增强算法。

## 5. 数据与评分可复现性

### I19：DNSMOS 本地预测

`microsoft/DNS-Challenge` 的 [`DNSMOS/dnsmos_local.py`](https://github.com/microsoft/DNS-Challenge/tree/591184a9fcb2cbdec02520fed81a32bbbf9d73ff/DNSMOS)提供本地推理入口。`LICENSE-CODE` 为 MIT，数据许可由其他文件约束。DNSMOS、DNSMOS P.835 和个性化模型不能仅写成同一个“DNSMOS 分”。

要固定模型文件及摘要、采样率处理、窗长、短文件处理、个性化选项和聚合方式。个性化模式会把干扰说话人按目标任务处理，不能把两种模式的分数混入一张排名表。该分数是学习模型的质量预测，不是实际受试者 MOS，也不覆盖方位和身份追踪。

推荐建立包括干净语音、加噪语音、过度抑制和静音的固定集；保存逐文件分数及失败记录。不能先丢弃低分或无法评分的文件，再只报告成功子集均值。设备选择还需任务识别和主观听测对照。

### I20：AECMOS 的输入与评分区间

`microsoft/AEC-Challenge` 的 [`AECMOS/AECMOS_local/`](https://github.com/microsoft/AEC-Challenge/tree/6c633d0a9d2a143a0e364899b91b06f127315b18/AECMOS/AECMOS_local)提供 ONNX 模型与本地推理。官方 [AECMOS 说明](https://github.com/microsoft/AEC-Challenge/tree/6c633d0a9d2a143a0e364899b91b06f127315b18/AECMOS)区分回声与其他损伤，并说明 Web API 不再更新。源码根许可为 MIT；模型及挑战数据的来源仍须单列。

固定远端参考、麦克风、处理后信号的对应关系、文件长度、原始采样率和模型版本。固定提交 `6c633d0a9d2a143a0e364899b91b06f127315b18` 的[本地 `aecmos.py`](https://github.com/microsoft/AEC-Challenge/blob/6c633d0a9d2a143a0e364899b91b06f127315b18/AECMOS/AECMOS_local/aecmos.py#L60-L81)用 `librosa.load(..., sr=模型采样率)` 读入三路音频，会自动重采样，并默认混成单声道；默认 `mono=True` 的行为见 [`librosa.load` 0.9.1 文档](https://librosa.org/doc-playground/0.9.1/generated/librosa.load.html)。随后三路各取共同最短长度，`run()` 对达到 20 s 的输入只保留**最前 20 s**。不能仅凭原文件头或成功返回分数，认定模型评分了原始采样率、全部通道和整段录音。

用于 Interspeech 2021 或 ICASSP 2022 测试集时，应先按[官方 AECMOS 说明](https://github.com/microsoft/AEC-Challenge/blob/6c633d0a9d2a143a0e364899b91b06f127315b18/AECMOS/README.md)在**原始时间轴上对三路同步裁取**实际评分区：远端单讲取后半段，双讲取末尾 `(时长秒数−15)/2` 秒，近端单讲取全段。这些规则只属于相应测试集，不是所有 AEC 文件的通用剪裁法。

裁取后核对三路采样率、通道混合、重采样和共同长度，再检查送入模型的片段是否超过 20 s；超出时须另行定义并验证分段及聚合口径，不能把默认脚本返回的首 20 s 分数称为整段评分。上述是固定源码的接口核对和建议评分流程，本书尚未运行 AECMOS 模型或取得其分数。

分别保存远端单讲、近端单讲和双讲评分。用参考错配、固定延迟、近端削弱和静音输出检查评分流程是否揭示明显错误。不要把高 ERLE 当成 AECMOS 或近端保真已经通过，也不能把评分模型测出的变化写成人类听众的实测差值。

### I21：CHiME-8 数据准备、文本规范化与会议评分

`chimechallenge/chime-utils` 提供 CHiME-8 DASR 数据准备、清单转换和官方评分入口，代码为 MIT。本节固定提交 `152882404f572d40769ef02bf91c5a9a9cfc9c78`；其 [`README` 的 Scoring 部分](https://github.com/chimechallenge/chime-utils/blob/152882404f572d40769ef02bf91c5a9a9cfc9c78/README.md#scoring)说明 SegLST 的会话、说话人、起止时间和文字字段。底层 MeetEval 固定为 `6e3dc81284f2d6928f7ef9e620fd3b6906daa429`，代码同为 MIT；核实日期为 2026-09-28。源码许可不代表能够再分发原始会议语料。

#### 输出槽位与说话人身份决定评分口径

[MeetEval 作者论文](https://arxiv.org/pdf/2307.11394)的 v3（2024-01-25）§3.2 式(3)、§3.3～3.5 和图1区分说话人转写与连续语音分离的输出流。cpWER 为整场参考说话人和假设说话人寻找一个全局排列；ORC-WER 把完整参考语句分配到输出流，且保留参考语句的给定顺序。MIMO-WER 保留各参考说话人内部的语句顺序，允许不同说话人的语句重新交错。选择前要固定分词、参考语句边界、排序、空流和评分区域。

两路 CSS 可以在不同时段承载多于两名说话人，[CSS 原论文](https://www.microsoft.com/en-us/research/uploads/prod/2020/04/ICASSP2020__Continuous_speech_separation__dataset_and_analysis.pdf)§1～2讨论了固定输出流与按说话人输出的区别。假设 A、B 先同时各说一个词 `a`、`b`，随后 C 说 `c`；CSS 两槽为 `a c`、`b`。内容完全保留，ORC-WER 为零；直接把槽位当身份交给 cpWER，补空流后仍有一次插入和一次删除，错误率为 $2/3$。这说明两种评分回答的问题不同，不能由 cpWER 的惩罚推出该例丢失了内容。

原始 CSS 槽位可先报 ORC-WER；需要身份结果时，先形成整场说话人轨迹，再报定义明确的 cpWER、tcpWER 或说话人归属指标。ORC 也不能任意切开参考语句以降低分数：参考语句 `a`、`b c` 与输出 `a b`、`c` 的最小错误数为 2；若事后将 `b c` 拆开，已经改变评分任务。

DI-cpWER 也不是身份准确率。固定 [`di_cp.py`](https://github.com/fgnt/meeteval/blob/6e3dc81284f2d6928f7ef9e620fd3b6906daa429/meeteval/wer/wer/di_cp.py) 的公开入口为 `greedy_di_cp_word_error_rate`：交换参考与假设送入贪心 ORC，再换回插入、删除计数，按原参考词数归一。这个实现的贪心结果不能默认等于穷举最小值；本书没有执行该生产入口。简化算法文档中的 `di_cp_lev` 另有形参与正文变量拼写不一致，提取原函数调用产生 `NameError`，不应把这段文档的故障扩大为生产实现的同类故障。

#### 话段、词时间与诊断指标的适用范围

作者后续原稿 [Word Error Rate Definitions and Algorithms for Long-Form Multi-talker Speech Recognition，2025，v1](https://arxiv.org/html/2508.02112v1)的 §IV-F 继续按完整参考话段定义 ORC；§V-A 式(10)加入词时间约束；§VI-A 讨论 DI-cpWER 的分析用途，§VII 算法1说明贪心分配。这里引用所读的明确v1版本。上述部分与固定源码的补充核实日期为 2026-10-02。

参考话段是评分输入的一部分。把 `b c` 事后拆成 `b`、`c`，就增加了可独立分配的单元，不能拿更低的 ORC 分数证明输出改善。[E11-23](../../../../chapters/11_selection-guide.md#e11-23)分别枚举两种分段，保留固定的输出与词序。

DI-cpWER 可以帮助诊断说话人归属约束对结果的影响，但作者 §VI-A 不建议用它给系统排名：改变假设分段可能改变可选分配，贪心算法还不保证全局最优。cpWER 与 DI-cpWER 的差值也不是独立测得的身份错误率。做选型时，应预先固定任务指标，把这个诊断量与正式排名指标分开保存。

普通词编辑距离只比较词序。如果参考“春天”的区间为 `[0,1]` 秒，假设同词却在 `[10,11]` 秒，普通错误数为0；下面实际运行的固定时间核只能删除并插入，错误数为2。正重叠区间 `[0,1]` 与 `[0.5,1.5]` 则可以匹配，错误数为0。[E11-24](../../../../chapters/11_selection-guide.md#e11-24)说明词时间、容差与内容评分的关系；词元是预先分好的整个词，不把“春天”自动拆成两个汉字。

触点边界须按版本另记。固定 [`levenshtein.h` 的 `overlaps`](https://github.com/fgnt/meeteval/blob/6e3dc81284f2d6928f7ef9e620fd3b6906daa429/meeteval/wer/matching/levenshtein.h#L94)使用严格正重叠，区间 `[0,1]` 与 `[1,2]` 仅端点相接，实际错误数为2。2025原稿式(10)以间隔大于容差为禁止条件；零容差下的等号边界不能直接代替这个固定实现的严格比较。该版本差异不推翻原论文的整体评分方法。实际使用 tcpWER 包装器还需固定容差及词时间生成方式；这里直接调用两核，不运行包装器，也不把较宽的容差解释为 DER 的排除评分区域。

#### 缺失会话不能随意从统计分母中消失

固定 [`apply_multi_file`](https://github.com/fgnt/meeteval/blob/6e3dc81284f2d6928f7ef9e620fd3b6906daa429/meeteval/io/seglst.py#L565) 按参考会话逐项调用评分函数。本书实际调用原分发函数，以“返回参考、假设片段数”的简单回调检查其会话选择；该回调不计算 WER。

| 输入边界 | 固定分发函数的行为 |
|---|---|
| 10 个参考会话缺 1 个假设 | 返回全部 10 项；缺失假设传入空 SegLST，并记录警告 |
| 2 个参考会话缺 1 个假设 | 超过默认缺失比例 0.1，抛出异常 |
| 假设含参考中不存在的会话 | 默认抛出异常 |
| 上一情况但显式 `partial=True` | 只处理参考会话，记录丢弃范围的警告 |
| 参考集合为空 | 抛出异常 |

因此，空假设、整个会话遗漏、主动评测子集和输入错误必须分别记录。参考会话存在而模型输出静音时，应明确输出该会话的空转写，不靠删文件改变分母；不能先取两个文件集合的交集再宣称完整评测。

#### CHiME 包装层的固定版本控制流边界

[`chime_utils/scoring/meeteval.py`](https://github.com/chimechallenge/chime-utils/blob/152882404f572d40769ef02bf91c5a9a9cfc9c78/chime_utils/scoring/meeteval.py) 的 `_wer` 调用 cpWER、带 5 s 容差的 tcpWER，或带 0.25 s 容差的 DER；这些是该届包装器的不同参数，不能互换。它先分别汇总各场景错误率，再对场景错误率作宏平均，不能称为全部词数加权的总 WER。上述评分分支为静态审读，未运行挑战赛整链。

`_load_and_prepare` 的缺文件分支还需要调用方预检。固定版本在 `ignore_missing=False` 时只记录错误而继续；原函数提取后用模拟路径和加载器实际调用，第一场景缺参考或假设均产生 `UnboundLocalError`。第二场景缺参考时，却可能继续产出 `mixer6` 的条目并沿用 `chime6` 的参考；第二场景缺假设也会沿用上一场景假设。显式允许忽略时，缺失场景才被跳过。诊断保留这些不利结果，没有修改上游。

同一函数的规范化循环写成 `if words == words`，普通字符串下立即退出，未比较第二次结果。受控输入 `aaa` 配合“每次删除第一个字符”的模拟规范化器，返回 `aa`，而继续规范化的稳定值为空串。这个反例只证明固定包装层没有执行所注释的稳定性检查，不证明其真实默认文本规范化器一定不幂等；本书没有在该实验中运行官方规范化器。

生产评测前应核对预定场景与文件集合，缺项时明确失败或事先声明子集；固定规范化规则，并对代表性文本验证重复应用是否改变结果。不要依赖错误日志来阻止函数继续，也不要用 `--ignore-missing` 掩盖未处理数据。

<a id="meeting-scoring-audit"></a>

#### 诊断的执行层次与复算入口

[诊断脚本](../../ch11/examples/audit_meeting_scoring_interfaces.py)与[固定报告](../../ch11/reports/meeting_scoring_interfaces.json)绑定源码提交、所读文件摘要、输入配置、脚本摘要和运行环境，区分四类证据：原 MeetEval 会话分发函数调用；原简化文档函数提取调用；原 CHiME 控制函数加模拟文件接口；未执行生产算法的静态核对。

2026-09-28的隔离环境中，MeetEval 可以导入，但真实 `cp_word_error_rate` 调用因缺少编译扩展 `cy_levenshtein` 而失败。文档函数在同一环境中给出三人两槽例的 cp 错误数 2、ORC 错误数 0；这不能改写成生产包已算出对应 WER。没有运行 ASR、说话人分离、CHiME 音频评测、模型推理或设备测试。

普通[离线测试](../../../../tests/test_codes_meeting_scoring_interfaces.py)不依赖下载目录、NumPy 或 SciPy，用独立全矩阵编辑距离与排列枚举核对记录。要重新执行外部诊断，需固定源码和含 NumPy/SciPy 的隔离环境；输出先放到新临时文件，再比较报告。脚本核对原文件摘要和干净工作树，不安装依赖、不编译扩展、不修正上游，也不下载会议录音。

<a id="meeting-kernel-contracts"></a>

#### 原C++词编辑核的独立执行合同

[当前核审计工具](../../ch11/examples/audit_meeting_kernel_contracts.py)完整包含固定版原 `levenshtein.h`，实际调用 `levenshtein_distance_` 与 `time_constrained_levenshtein_distance_v2_`。自写驱动只把已分词文字映射成无符号词元，并传入以秒为单位的非负 double 区间；没有改写原方法、替换算法或加入 ORC 外层分配。

九例均以插入、删除、替换代价1及正确匹配代价0运行。期望来自手算，[独立测试](../../../../tests/test_codes_meeting_kernel_contracts.py)另用完整二维编辑距离矩阵核对，不用原核输出生成期望。两词时间整体错位时，每个词都要删除并插入，总错误数为4；空输入按实际词数计错，双空为0。

| 已分词输入与时间关系 | 普通错误数 | 固定时间核错误数 |
|---|---:|---:|
| 同词、完全分离 | 0 | 2 |
| 同词、仅端点相接 | 0 | 2 |
| 同词、严格正重叠 | 0 | 0 |
| 异词、严格正重叠 | 1 | 1 |
| 异词、完全分离 | 1 | 2 |
| 空参考、假设两词 | 2 | 2 |
| 参考两词、空假设 | 2 | 2 |
| 双空 | 0 | 0 |
| 同两词、两段时间均完全分离 | 0 | 4 |

本机原核执行另存于[当前报告](../../ch11/reports/meeting_kernel_contracts.json)，2026-09-28的旧诊断保留。报告绑定100项锁表的实际摘要、固定 HEAD 与 origin、七个原文件的 Git blob/SHA、原 MIT 许可、工具和自写驱动摘要、编译器与实际命令，并核对源工作树执行前后洁净。其余六个文件只作接口、许可和来源核对；仅原头文件参与编译。中文词元不测试分词算法，时间区间也不是从录音估计的。

```bash
.venv/bin/python -B -m codes.chapters.ch11.examples.audit_meeting_kernel_contracts
.venv/bin/python -B -m codes.chapters.ch11.examples.audit_meeting_kernel_contracts --report codes/chapters/ch11/reports/meeting_kernel_contracts.json
```

默认只输出 JSON；编译仅在临时目录，退出后清理。只有显式 `--report` 才写入普通 JSON 文件；输出路径拒绝符号链接、非目录父链、字面 `..` 与上游缓存内部路径，并原子替换报告。工具不下载或安装依赖，所需环境为 Python 标准库、Git 与 C++17 编译器。可选缓存或编译器缺失时，普通测试明确跳过原核实调；版本、来源或摘要不符则失败。原核成功没有补齐旧环境缺少的 Python 编译扩展，也不代表生产 cpWER/ORC-WER/tcpWER、ASR、说话人关联、模型或硬件已运行。

## 6. 怎样把项目变成可执行实验

每次只选一条具体链。例如 Linux 会议终端可以从 ALSA/PortAudio 采集、PipeWire 或应用内 AEC、固定波束、单通道 NS、VAD/AGC 与后端开始。若图中同时出现 PipeWire 和应用内 AEC，应先明确谁负责消回声及谁取得播放参考。

实验记录至少保存：源码提交、模型摘要、依赖和编译参数、设备与固件、声学条件、通道与时钟、数据集划分、块长、状态初始化/重置、指标、逐文件结果和资源测量。状态相关的回归要同时覆盖整段、分块、暂停恢复与故障注入；算法单元测试不能代替它们。

块长不同需要适配，但不必等待所有块长的最小公倍数。以 16 kHz 为例，10 ms 采集块是 160 点，15 ms 处理步长是 240 点，32 ms VAD 块是 512 点；三者最小公倍数为 7680 点，即 480 ms。若先攒满这 480 ms 才启动全部模块，会人为增加很大等待。

连续队列可以在各消费者凑齐自身输入时立即调用，并保留剩余样本；VAD 的控制输出按时间戳使用。验收要检查每个消费者收到的样本无重无漏，以及输出决策对应哪个时间区间。这是缓冲调度算例，不表示上述不同硬件实现应直接串接。

本书的[工程练习 E10-07～08](../../../../chapters/10_engineering-practice.md#sec-10-11)给出字节口径与消费索引。运行 `.venv/bin/python -m codes.chapters.ch00.cross_chapter.exercises_engineering` 后，E10-08 中的 `events` 记录左闭右开的输入索引和回调轮次，`leftovers` 记录余量。两条消费者首次分别在 20、40 ms 得到合法块；这里假设各回调完整交付、计算耗时为零，不把它称为设备延迟实测。

同一入口的 E10-14 把六帧串行计算、在途容量、同时驻留内存和重复模式下的功率估算连起来；[第 10 章完整手算](../../../../chapters/10_engineering-practice.md#sec-10-4-3)给出每帧完成时间。第 11 章 E11-05～07 则分别以周向阵列的相邻间距筛查、单路会议链的输出/延迟硬约束和跨时钟漂移为候选过滤题。这四题的数字都是本书指定的教学输入，代码通过算术与约束，不代表源码已在某款设备通过资源或声学验收。

新增[状态与时间练习](../cross_chapter/tracking_time_exercises.py)把两个容易混淆的工程条件单独复算：E10-15 逐帧记录 VAD 预卷、触发和结束保持，输出片段采样区间 `[160,1120)`，避免触发帧重复；E11-08 从零失败的二项概率推导单侧风险上界，区分“没有观察到失败”和“风险已足够低”。它们与第 9 章 E09-08 的状态时间戳一起构成接口检查，均未使用真实设备或将连续相关帧当作独立试验。

[工程边界练习](../cross_chapter/engineering_boundary_exercises.py)给出相应的受控反例：E10-16 在每帧恰好用完周期时应无丢帧，稍微超过周期才触发拒绝或期限违约；E10-17 按实际对应样本计算延迟；E11-09 保留六个配对会话及一项平局，用五个非平局的全部符号排列复算单侧概率。合并 WER 下降幅度与会话胜负方向是不同统计对象，不能因为模型的总错误数较少就省去逐会话记录。

真实回调还应区分采集首样本时间、回调开始时间和播放首样本时间。[PortAudio 的 PaStreamCallbackTimeInfo](https://files.portaudio.com/docs/v19-doxydocs/structPaStreamCallbackTimeInfo.html)把三者分别放在 `inputBufferAdcTime`、`currentTime`、`outputBufferDacTime` 中，单位为秒，使用所属流的时间基准。它们不能未经校准就与另一设备的时钟相减。量化、时间戳与样本消费属于接口条件；更新模型或运行时版本时，也应保留这些对照。该接口文档核实于 2026-09-22。

本目录给出的是实现阅读和实验设计。源码实际获取状态见 `codes/chapters/ch00/upstream/` 的下载记录；构建、运行及目标硬件验收仍按各实验的真实记录报告。

## 7. 重采样、测量与数值接口的独立对照

下面六项分别补充采样率转换、响度测量、全参考评分、文件输入输出和块浮点运算。它们不组成一套必须串接的前端，也不都属于阵列算法。固定源码说明了可以研究什么；未附运行记录的输入与验收步骤属于建议实验；I22、I23、I26 附有各自的主机执行记录，应按记录中的输入、版本和指标解释，不外推为设备或语音质量验收。

### I22：libsoxr 的比率、渐变与输出延迟

libsoxr 是可独立使用的 SoX 重采样库。本书核对了[官方 SourceForge 的 0.1.3 提交](https://sourceforge.net/p/soxr/code/ci/945b592b70470e29f917f4de89b4281fbbd540c0/tree/)，其完整提交与 `chirlu/soxr` 镜像相同。阅读顺序是 `src/soxr.h` 的接口约定、`examples/5-variable-rate.c` 的可变比率示例，再到 `src/` 的滤波实现。[许可文件](https://sourceforge.net/p/soxr/code/ci/945b592b70470e29f917f4de89b4281fbbd540c0/tree/LICENCE)采用 LGPL-2.1-or-later，并要求另看 `pffft.c` 的文件内声明；源码可本地研究，不意味着静态链接或重新分发没有义务。

与 I04 的 libsamplerate 对照时，先核对比例方向。libsamplerate 的 `src_ratio` 是输出速率除以输入速率；libsoxr 的 `soxr_set_io_ratio` 使用输入/输出比。设备快 100 ppm、转到参考时钟时，两者分别为约 0.99990001 和 1.0001。`soxr_create` 则直接接受输入、输出两个速率，不需要把比率当采样率传进去。

可变比率需选择 `SOXR_VR`。官方示例先按将使用的最大输入/输出比创建状态，再设置当前比率；`slew_len` 描述用多少个**输出样本**渐变到新比率，0 表示立即改变。输出 16 kHz 时，160 个输出样本对应 10 ms 的比率过渡；这只是控制参数换算，不证明设备时钟已经估计准确。[固定可变比率示例，第 37～66 行](https://github.com/chirlu/soxr/blob/945b592b70470e29f917f4de89b4281fbbd540c0/examples/5-variable-rate.c)

连续调用还要分别累计实际输入消耗量 `idone`、实际输出量 `odone`，保留未消费输入。结束时以 `in=NULL` 表明不再有输入，并继续取出尾部；输入长度为 0 本身没有这个结束含义。`soxr_delay` 返回当前以输出样本计的延迟，不能直接当成毫秒；它也不包含驱动、线程和下游等待。[API 头文件](https://github.com/chirlu/soxr/blob/945b592b70470e29f917f4de89b4281fbbd540c0/src/soxr.h)

整段与分块对照必须显式统一质量、相位响应、输入输出格式及线程配置。该版本的 `soxr_create` 默认 HQ，而 `soxr_oneshot` 默认 LQ；若保留各自默认值，波形差异不能直接归因于分块状态。整数量化输出还要固定抖动选项，避免把随机化误差误判为状态不连续。

建议先固定 48 kHz→16 kHz、双通道浮点输入、同一质量配置。一路给脉冲，另一路给有明确频率的正弦，分别按整段和不等长块送入，记录消费区间、总输出长度、尾部及延迟。再在同一连续状态上加入比率阶跃和渐变，检查通道间相位关系；高于输出奈奎斯特频率的输入用于观察抗混叠行为，不能只检查文件头是否写成 16 kHz。

构建需要 C 编译器、CMake 和构建工具，OpenMP/SIMD 路径按配置选择。官方 `INSTALL` 给出构建与测试入口，但编译器、CMake 兼容性和测试结果仍须实际记录；取得源码不等于这些步骤已经通过。

本书于 2026-09-22 实际构建并运行了 I22、I23、I26 的三个固定库。环境为 macOS arm64、Apple clang 21.0.0、CMake 4.3.2；采用 Release 静态库与独立构建目录，没有安装到系统或教学虚拟环境。soxr 关闭 OpenMP 和 SIMD 分支，libsndfile 关闭外部压缩编解码后端；旧构建文件通过命令行 `CMAKE_POLICY_VERSION_MINIMUM=3.5` 兼容，未修改上游源码。三个工作树在构建前后均通过来源、提交与干净状态核查。运行入口为[隔离构建脚本](../../ch10/examples/run_industrial_interfaces.py)，调用[原创 C 实验程序](../../ch10/examples/industrial_interfaces.c)；[原始报告](../../ch10/reports/industrial_interfaces.json)绑定源码提交、实验程序与运行脚本摘要、构建选项和日志摘要。

重采样输入为 1 s、48 kHz 双通道：左声道在零起始帧 12000 放幅度 0.5 的脉冲，右声道为幅度 0.1、频率 1 kHz 的正弦。输出为 16 kHz，显式使用交织双精度浮点、HQ、质量标志 0 和单线程，每次输出缓冲限为 113 帧。整段供给、每块 127 帧和每块 509 帧均保持同一状态，按 `idone` 保留未消费输入，最后以 `in=NULL` 排空。

| 输入供给方式 | 排空前输出帧数 | 排空补出帧数 | 总输出帧数 | 相对整段的最大绝对差 |
|---|---:|---:|---:|---:|
| 整段 | 15471 | 529 | 16000 | 0 |
| 127 帧块 | 15659 | 341 | 16000 | 0 |
| 509 帧块 | 15622 | 378 | 16000 | 0 |

排空后的输出延迟均为 0。排空前的不同积压量与供给/输出缓冲方式有关，不是三个不同的固定算法延迟。脉冲峰落在输出帧 4000，符合 $12000\times16000/48000=4000$；右声道在半开区间 `[320,15680)` 与解析正弦的最大绝对差约为 $6.84\times10^{-8}$，没有拟合时移或增益。整段供给触发 141 次部分消费，说明实验确实经过了保留输入余量的分支。

反例在输入帧 24000 处未排空就调用 `soxr_clear`，最终只有 15566 帧，比连续处理少 434 帧。把两路输出按现有数组下标直接比较，共同长度内的最大绝对差约为 0.07669；这里已丢失待输出样本，因此该数只用于发现协议错误，不能解释为对齐后的音质损失。上述固定比率实验没有测抗混叠频响、设备时钟闭环或实时截止期；这些仍需单独实验。

#### 已运行：两段已知时钟下的可变比率与渐变

本书在同一固定提交上另运行了一个受控探针，不把已知速率当成盲估计结果。名义输入为 48 kHz，前 5 s 的设备速率为 $48000(1+100\times10^{-6})=48004.8$ Hz，后 5 s 为 $48000(1+150\times10^{-6})=48007.2$ Hz；所以两段输入各有 $240024$、$240036$ 帧，合计 $480060$ 帧。输出参考速率为 16 kHz。两通道以交织双精度输入：左路在真实时间 1～9 s 各放一个 0.5 幅度脉冲，右路按逐输入样本的真实时间生成幅度 0.1、3 kHz 正弦。时间标记的输入索引由上述两段时钟直接计算，不由 libsoxr 输出反推。

以 `SOXR_HQ | SOXR_VR` 创建状态，输入/输出最大比率设为 $48007.2/16000=3.00045$，当前比率先设为 $48004.8/16000=3.0003$。只有累计输出达到帧 80000 时才发出第二次 `soxr_set_io_ratio`；输出缓冲在边界前截断，避免一次调用跨过控制时刻。比较三种控制：继续保持 3.0003、立即切到 3.00045、用 160 个**输出帧**渐变到 3.00045。还用 320 帧渐变检验控制长度；160/320 帧在 16 kHz 下分别为 10/20 ms。本题的两档 ppm 与给定时间标记只是合成真值，未经过设备时钟估计器。

| 控制方式 | 排空后输出帧数 | 1 s 标记 | 5 s 标记 | 9 s 标记 |
|---|---:|---:|---:|---:|
| 保持旧比率 | 160004 | 16002 | 80002 | 144005 |
| 立即切换 | 160000 | 16002 | 80002 | 144002 |
| 160 帧渐变 | 160000 | 16002 | 80002 | 144002 |
| 320 帧渐变 | 160000 | 16002 | 80002 | 144002 |

初始共同的 2 帧偏移与库的滤波和峰值索引有关；本书比较的是同一探针内标记的**相对变化**：保持旧比率在 1～9 s 多漂移 3 帧，更新后该范围未再增加。若仅按无滤波的速率积分，保持 100 ppm 时全部 $480060$ 输入约对应 $160004$ 输出；更新后目标约 $160000$。后 5 s 的 50 ppm 差对应约 4 输出帧，具体脉冲只在整秒采样并经过输入索引取整，因此 9 s 的观测差为 3 帧。不能用两条输出的绝对索引差声称已测设备时钟误差。

160 帧渐变的整段输入与 127 帧输入块在相同输出帧控制时刻产生相同的 160000 帧，逐点最大差为 0；最后以 `in=NULL` 排空，保持待消费输入的起点。与立即切换按同下标比较，160 帧和 320 帧渐变的最大绝对波形差分别为 $0.000624623$、$0.001245994$。按线性比率日程手算，160 帧渐变相对立即切换的累计输入坐标差量级为 $(3.00045-3.0003)\times160/2=0.012$ 输入帧；320 帧为 0.024 输入帧。库返回的波形差仅验证两种长度确实进入不同控制路径，不把它们换算成自然音频可闻差或闭环精度。逐调用 `idone`/`odone` 与报告总量已由独立脚本核对。

本固定提交的 VR 路径在最后一次 `in=NULL` 调用产出 0 帧后，`soxr_delay` 仍读为 100 个输出样本；固定比率路径在同样排空协议下读为 0。这里同时记录**调用已无输出**与**接口当前读数**，不把 100 解释为仍有 100 帧漏写，也不把它并入端到端设备延迟。细节可由[原始报告](../../ch10/reports/industrial_interfaces.json)及[实验源码](../../ch10/examples/industrial_interfaces.c)复核。该探针未测抗混叠抑制、设备时间戳噪声、在线估计或硬实时性。

### I23：libebur128 的响度与真峰值

峰值保护 AGC 限制样本幅度，却不等于按感知响度调节音量。`jiixyj/libebur128` 提供响度和峰值测量，入口为 [`ebur128/ebur128.h`](https://github.com/jiixyj/libebur128/blob/67b33abe1558160ed76ada1322329b0e9e058b02/ebur128/ebur128.h) 与 `ebur128/ebur128.c`。锁定快照的版本宏为 1.2.6，根 [`COPYING`](https://github.com/jiixyj/libebur128/blob/67b33abe1558160ed76ada1322329b0e9e058b02/COPYING)为 MIT；实现和模式应绑定该提交，不因库名含 R128 就宣称所有配置已通过标准认证。

瞬时响度、短期响度和积分响度使用不同统计窗口；响度范围描述随时间变化的离散程度，采样峰值与真峰值则检查幅度。接口中积分响度以 LUFS（Loudness Units relative to Full Scale，相对数字满刻度的响度单位）返回；`ebur128_true_peak` 返回线性幅度，若转成 dBTP（decibels True Peak，真峰值分贝），需对相对满刻度幅度取 $20\log_{10}$。两种结果不能互相替代，也都不是未经声学校准即可得到的 dB SPL。

原始测量依据为 [ITU-R BS.1770-5](https://www.itu.int/rec/R-REC-BS.1770-5-202311-I)（2023-11-22批准，2026-09-28官方页仍为在力）。测量方法不自动指定每个产品的目标音量。[EBU R 128-2023 原文](https://tech.ebu.ch/docs/r/r128.pdf)第3～4页 g～m 项针对广播节目：规定整节目测量、−23 LUFS目标及线性音频生产中的−1 dBTP上限，并给出用途和容差条件。不能把这些节目交付目标直接作为全部 ASR、通话或唤醒前端的 AGC 默认值。下面正弦探针恰好约−23 LUFS，只是该输入的测量结果。

先用 `ebur128_init` 选择所需测量模式，再连续送入交织音频。`ebur128_add_frames_*` 的数量是每通道时间帧数，而非所有通道的标量数之和。通道角色影响测量：头文件的默认映射按常见节目声道设置，不能把阵列的四路原始麦克风直接当成四个节目声道。单路增强输出可单独测量；比较多个麦克风时应使用各自的单通道测量状态，而不是任意套用环绕声布局。

积分门控会影响哪些时间段进入统计；全静音等无有效能量的情况可能返回负无穷。调用者应同时保留状态、有效时长和失败或无定义原因，不能把负无穷替换为 0 LUFS 后参加平均。真峰值用于估计重建波形的幅度，可能高于已存样本的最大绝对值；采样峰值不越界不能单独证明重建后有足够余量。

建议输入一段足够长、幅度稳定且通过门控的非静音信号，再整体乘以 0.5。在门控保留区域相同的前提下，本书代数预期响度下降 $20\log_{10}2\approx6.0206$ LU，采样峰值和真峰值的线性值均减半。另测全静音、短文件、逐块清状态和错误通道映射，并分别保存结果；这些输入检验测量协议，不构成主观听测。

核心库用 C/CMake 构建，不需要神经模型。实际测试目录由根 `CMakeLists.txt` 指向 `test/`；文件读取示例所需依赖与核心测量分开核对。开启测试、编译成功、运行测试及取得标准测试音频是不同步骤，不因生成了库文件就把全部测试标为通过。

本书实际运行的合成测量采用 48 kHz、10 s、1 kHz 单声道正弦，显式把通道设为 `CENTER`，模式为 `I | TRUE_PEAK`。幅度 0.1 和 0.05 的积分响度分别为 −23.00360 LUFS、−29.02420 LUFS，相差 −6.020599913 LU，与独立代数预期 $20\log_{10}(0.5)$ 一致。两档信号均远高于绝对门限，保持相同有效门控区域；不能据此认为任意含静音或近门限节目减半都会得到相同的积分响度变化。

保持状态、每次送入 127 帧，与整段送入的积分响度差为 0 LU；采样峰值分别为 0.1、0.05，估计真峰值约为 0.1000000015、0.0500000007，线性比均为 0.5。全静音返回负无穷，报告将它记录为 `null`，同时保留 `negative_infinity_no_gated_energy` 原因；静音的两个峰值均为 0。原始值与构建条件见同一份[工业接口报告](../../ch10/reports/industrial_interfaces.json)。这组结果验证幅度缩放、连续状态和失败值处理，不构成 EBU 标准测试集认证、主观听测或任意文件格式的验收。

#### 已运行：采样点之间出现峰值的构造信号

前述 1 kHz 正弦的 48 kHz 网格很密，不能清楚展示“样本峰低于重建后峰”的情形。新增独立输入为 48 kHz、10 s、12 kHz、初相 $\pi/4$、幅度 0.95 的单声道正弦；开头与结尾各用 100 ms 升降余弦渐变，防止硬起振造成插值滤波器的边界过冲。中间稳定区的样本相位每次增加 $\pi/2$，各样本绝对值均为 $0.95/\sqrt2=0.6717514421$；连续解析正弦则可在样本间达到 0.95。两者分别约为 $-3.45583$ dBFS 与 $-0.44553$ dBFS，解析间隔为 $3.01030$ dB。渐变只降低边界处幅度，不改变稳定区这两个独立手算值。

锁定 libebur128 1.2.6 的[`ebur128.c` 插值器](https://github.com/jiixyj/libebur128/blob/67b33abe1558160ed76ada1322329b0e9e058b02/ebur128/ebur128.c#L119-L174)在 48 kHz 使用 49 抽头、4 倍多相 FIR；这是一种有限长**估计器**，并不对任何带限信号承诺精确重建解析最大值。实际 C 接口返回样本峰 $0.671751442206$（$-3.45583$ dBFS）、真峰估计 $0.946371197701$（$-0.47877$ dBTP），高出样本峰约 $2.97706$ dB。整段送入与保持同一状态、每次 127 帧送入的真峰读数逐位相同。样本峰与手算的微小差来自浮点正弦及内部类型转换；真峰与解析 0.95 相差约 0.00363，不能把其中一个数字代替另一个。

不加渐变而从第一点突起的同频正弦不是同一个连续波形边界。按上述固定 FIR 系数独立复算，起始瞬态可达约 0.96139，甚至超过平台解析幅度 0.95；所以本实验以平滑边界隔离采样间峰值结论。这个测试只验证固定软件实现与解析反例，不认证标准真峰测试集，也不等于聆听质量。报告的[源码摘要和原始测量值](../../ch10/reports/industrial_interfaces.json)可与[实验程序](../../ch10/examples/industrial_interfaces.c)核对。

### I24：pystoi 的 STOI、ESTOI 与无效评分

第 10 章列出的短时客观可懂度 STOI 可以通过 `mpariente/pystoi` 的 Python 实现研究。它支持普通 STOI 和扩展 STOI（Extended Short-Time Objective Intelligibility，ESTOI），入口均在 [`pystoi/stoi.py`](https://github.com/mpariente/pystoi/blob/74872b000753a7a42ff51aa0868af8c82c7f9053/pystoi/stoi.py)。这是软件作者维护的 Python 实现；[README](https://github.com/mpariente/pystoi/blob/74872b000753a7a42ff51aa0868af8c82c7f9053/README.md)另说明 MATLAB 测试来源于 Cees Taal 的代码，不应把两种实现的作者与环境混同。

该提交 `setup.py` 标为 0.4.1，Python 核心按根 `LICENSE` 的 MIT 条款提供，运行依赖 NumPy 和 SciPy。Octave/MATLAB 对照测试有额外依赖和测试文件，不是安装 Python 核心后自动完成的验证。本书源码获取范围、测试资产和实际运行环境分别登记。

输入是一维、等长度的干净参考与处理后语音，二者采样率和时间对齐口径必须一致。实现内部先转到 10 kHz，再按参考进行静音帧筛除、短时分析和频带处理。`extended=False` 与 `extended=True` 使用不同的归一化与相关性计算步骤，结果应分别命名，不能在同一列中混写成 STOI。

需要额外检查返回值是否确实来自有效评分。该提交在去除静音后不足 30 个 STFT 帧时发出 `RuntimeWarning`，随后返回 `1e-5`。这个数是实现的失败返回值，不是测得“可懂度极低”的有效结果。评分程序应捕获并记录警告、文件 ID、长度与原因，在总文件数中保留这次失败；不能通过丢弃警告或只检查结果有限就让它进入平均值。

#### 原接口实跑：异常、哨兵与静音有限值

2026-09-28在既有隔离环境实际调用固定版 `pystoi.stoi`，输入均为10 kHz、float64、一维且参考与待评分数组完全相同，不经过重采样。环境为 Python 3.13.12、NumPy 2.3.5、SciPy 1.17.1；没有安装新的共享依赖。报告区分以下三种零输入行为：

| 输入长度 | 普通 STOI | ESTOI |
|---|---|---|
| 256点 | `AxisError`，无分数 | 相同异常 |
| 1024点 | 警告与 `1e-5` 哨兵 | 相同警告与哨兵 |
| 10000点 | 有限0，无警告 | 随归一化随机状态变化的有限值，无警告 |

256点输入在 `utils.remove_silent_frames` 的帧数组形成后即发生轴异常，尚未走到30帧检查。长全零输入则不会被相对于参考最大能量的筛选全部移除。ESTOI 的 `utils.row_col_normalize` 对两路分别加入浮点精度量级的随机抖动；在两次调用前分别设 `np.random.seed(0)` 与 `np.random.seed(1)`，10000点全零自比较得到约0.00856519与−0.000209994。它们不能解释成静音的可懂度。本书保留原返回值，另以 `finite_zero_energy_invalid` 标明无有效评分意义。

独立非零输入由 `np.random.default_rng(4).normal(size=10000)` 生成，不另归一化；普通/扩展模式自比较均为1，与同一非退化归一向量的相关性恒等式一致。这只检查接口数值，不是自然语音可懂度、原作者 MATLAB 对照或完整 STOI 验证。见[I32脚本与报告](#industrial-upstream-audit)，其中记录每条输入摘要、模式、随机种子、异常和警告。评分前应检查形状、有限性、对齐、参考活动和有效长度；异常、警告及有限但无效的退化值都须保留失败分母。

建议先准备授权清楚且有干净参考的自然语音，分别比较自身、已知加噪版本、静音输出与错位版本，并记录普通/扩展模式、SciPy 版本和对齐方式。另用短输入验证无效评分是否被识别。自身对照只是接口检查，不能代替与原作者实现的独立数值对照；STOI 也不是逐词识别正确率，合成正弦不能用来报告自然语音可懂度。

### I25：ViSQOL 的模式、模型与构建依赖

Google 的 ViSQOL（Virtual Speech Quality Objective Listener）是全参考音频质量估计器：先比较干净参考与退化信号的时频结构，再映射为客观听音质量预测分（Mean Opinion Score—Listening Quality Objective，MOS-LQO）。它需要参考，和 I19 的无参考质量预测解决不同的输入条件；它不是 POLQA 的实现，也不替代受试者 MOS。[固定 README 的 Guidelines、License 与 FAQ](https://github.com/google/visqol/blob/38d0b0163e441047d4429bf07ad09e5b9031d02c/README.md)

该实现的 audio 模式使用 48 kHz 输入，speech 模式使用 16 kHz 输入。多通道会下混为单声道再比较，因此分数不能证明阵列方向、双耳线索或空间声场保持。speech 模式还需记录语音活动筛选、映射模型，以及是否选择 scaled/unscaled 映射；模式不同的分数不能直接混排。

质量映射依赖模型。源码目录 `src/`、协议目录 `src/proto/` 和 Python 接口足以研究调用过程，却不足以证明已经具备评分条件。`model/` 中的 TFLite 文件与文本 SVR 参数都属于模型资产，不能只因扩展名为 `.txt` 就将其当作没有模型内容的普通说明。若获取时排除了模型和 `testdata/`，报告中须明确写源码范围，之后按资产来源和许可补齐，不能运行到缺文件再把它称为算法失败。

构建从根 `WORKSPACE` 和 `BUILD` 阅读。该提交的 [WORKSPACE](https://github.com/google/visqol/blob/38d0b0163e441047d4429bf07ad09e5b9031d02c/WORKSPACE)会引入 TensorFlow 2.11.0 对应提交及 protobuf、pybind11、Abseil、LIBSVM、Armadillo、PFFFT 等依赖。README 的 Bazel 命令是构建入口，不是“只需一个小型 Python 包”的保证；包内源码量、依赖下载量、构建缓存与运行内存需要分开记录。

根 `LICENSE` 对该项目代码采用 Apache-2.0，不能以此替全部外部依赖和录音授予许可。建议先核对模型摘要、模式和依赖，再用同一批参考—退化文件对生成逐文件及逐片段结果。参考错配、静音过多、采样率错误和短文件应有明确失败或排除记录，不悄悄变成高低分。

官方 FAQ 指出，它原本针对编解码和网络退化，移用于降噪、前处理及生成类失真时表现并不一致。因此正式设备结论仍需目标数据、主观听测和识别指标共同验证；不能用一个质量预测分证明内容未改变。

[ViSQOL v3 原论文](https://arxiv.org/pdf/2004.09584)（QoMEX 2020）给出了比项目名更具体的边界：§II.C建议对已知有活动的3～10 s片段分析；§II.B讨论音乐编码复杂度差异不够敏感的情况，不主张直接把该类分数作为通用自动回归判据。§III.C在全局、片段对齐之外加入细粒度时域重对齐，因此质量分不能替代端到端延迟或逐块连续性测量。§III.B的 conformance 版本用于追踪已知文件分数变化，应与模型摘要、源码版本一起保存。本书已读这些原文条件，但未运行该评分器，也不采用未公开网络配置下的论文分数作设备基准。

### I26：libsndfile 的帧数、幅度转换与短读

算法读到的数组取决于文件解码接口。`libsndfile/libsndfile` 提供多种音频格式的 C 读写接口，本书使用它研究输入输出协议，不把它计作定位或增强算法。入口为 `include/sndfile.h`、`src/sndfile.c`、`src/pcm.c` 与测试目录；[官方 API](https://libsndfile.github.io/libsndfile/api.html)把标量项数、时间帧数和原始字节数分成不同函数族。

例如，双通道文件读取 160 个时间帧时，`sf_readf_float(...,160)` 需要容纳 320 个浮点数；对应的 `sf_read_float` 请求量是 320，而不是 160。返回值同样沿用所选接口的单位。读到尾部时，返回量可能小于请求量，未用缓冲还可能被补零；应只将实际返回的帧标为录音内容，不能把补零区当成真实采样时间继续更新统计量。

文件编码与数组类型也要分开。PCM16 WAV 可以读成浮点数组，但原始量化精度仍是 16 位。浮点归一化、写回整数时的舍入/削波、通道顺序和字节序都应显式记录；不要由 Python 或 C 数组类型反推原始设备分辨率。跨库对照应先检查 PCM 端点和简单脉冲，再比较复杂语音输出。

建议构造左右通道脉冲位置及符号不同的小型 WAV，逐块读取并核对交织顺序，再用不能整除文件长度的块长检查最后一次返回值。另比较 PCM16 的最小/最大码字、浮点读取与写回结果，记录是否启用归一化和削波。该实验能发现接口差错，不证明 ADC 的动态范围或实际声压正确。

锁定源码的 `CMakeLists.txt` 版本字段为 1.2.2；[根许可](https://github.com/libsndfile/libsndfile/blob/b9103bd48b6c8fb517ae737fe3baee0c718b804c/COPYING)与 [sndfile.c 文件头](https://github.com/libsndfile/libsndfile/blob/b9103bd48b6c8fb517ae737fe3baee0c718b804c/src/sndfile.c)给出 LGPL-2.1-or-later。CMake 或 Autotools 构建可按用途选择功能；PCM/WAV 读写不要求启用全部压缩编解码后端，Ogg/Vorbis/FLAC/Opus/MPEG 和设备播放工具的依赖则另行核对。文件读写成功不能替代设备采集验证。

本书实际写入的 PCM16 WAV 为 16 kHz、7 帧、2 通道，按每行一帧、左声道在前排列如下；它只测试整数端点、符号、通道交织与短读，不是语音样本。

```text
-32768      0
     0  32767
 16384 -16384
     1     -1
 12345 -23456
     0   1000
 32767 -32768
```

用 `sf_writef_short` 写入，再分别重开文件读取：每次请求 3 帧的 `sf_readf_short` 返回 `3,3,1,0`；每次请求 6 个标量项的 `sf_read_short` 返回 `6,6,2,0`。实际有效标量均与输入逐项相等，累计分别为 7 帧与 14 项。另一次显式启用归一化的 `sf_readf_double` 读回与整数除以 32768 完全一致，最大绝对差为 0；正端点为 0.999969482421875，不是 1。

独立验证没有再次调用 libsndfile：运行脚本使用 Python 标准库 `wave` 与小端整数解码核对 WAV 编码、7 帧长度、双通道次序和全部 14 个码字。该检查与[独立失败用例测试](../../../../tests/test_codes_industrial_interfaces.py)共同覆盖帧/项误用、分块丢失、静音误记为零和报告来源失配。实验 WAV 与完整日志留在忽略的独立构建目录，摘要保存在[原始报告](../../ch10/reports/industrial_interfaces.json)中；本书没有据此声称其他文件编码、削波策略或设备采集也已验证。

### I27：lib_xcore_math 的块浮点与参考内核

XMOS `lib_voice` 的底层依赖不是任意最新版数学库。锁定的 [`lib_voice/lib_build_info.cmake`](https://github.com/xmos/lib_voice/blob/c9f1a9bf95cd88c7950adf4bf631c217f900ad25/lib_voice/lib_build_info.cmake)明确要求 `lib_xcore_math(v3.0.0)`；本书按此固定到 `16130be45c4002a1f875a4b06ff68d2065cd8c69`。该库提供向量运算、块浮点、快速傅里叶变换（FFT）、离散余弦变换（Discrete Cosine Transform，DCT）与滤波，补充 I15 的 Q15 内核对照，不是一套独立的完整语音前端。

块浮点把一组整数尾数与一个共享指数一起保存。示意地，尾数 `[-16384,8192]` 与指数 −15 表示实值 `[-0.5,0.25]`。相同整数数组配不同指数会代表不同幅度，因此对照时不能只比较尾数字节。

位余量（headroom）描述还能左移多少位而不溢出，用来选择运算缩放；为了避免溢出而右移又会损失低位精度。这与“全部模块固定使用同一 Q15 小数位数”不同。

源码阅读可从 `lib_xcore_math/api/` 进入 `src/bfp/`、`src/fft/` 与 `src/filter/`，再比较 `src/arch/ref/` 和目标架构实现。[构建文件](https://github.com/xmos/lib_xcore_math/blob/16130be45c4002a1f875a4b06ff68d2065cd8c69/lib_xcore_math/lib_build_info.cmake)区分参考 C、XS3 汇编及其他架构路径；FFT 查找表也有使用内置表与重新生成的选项。必须保存指数、长度、表配置、架构和编译选项，不能只记录函数名。

建议先用上述两点向量及小幅脉冲，按指数还原实值后与独立浮点计算比较；再加入接近满幅、余数长度、大小量混合与 FFT 往返。测试应同时观察输出指数、舍入误差、饱和和状态，不把“多数样本相近”当成全部边界正确。参考 C 路径通过后，仍需在目标板运行相同输入并另测周期和内存；主机仿真耗时不能推算目标 VPU 的实时性。

[README](https://github.com/xmos/lib_xcore_math/blob/16130be45c4002a1f875a4b06ff68d2065cd8c69/README.rst)指定 XTC Tools 15.3.1，并说明原生构建的 VPU 仿真范围限制。[LICENSE.rst](https://github.com/xmos/lib_xcore_math/blob/16130be45c4002a1f875a4b06ff68d2065cd8c69/LICENSE.rst)为 XMOS Public Licence v1，商业硬件和特殊用途条件不能省略。取得这个依赖也不表示 `lib_voice` 的工具链、模型生成依赖和目标硬件已经全部齐备。


### I28：FastEnhancer 的显式流式状态与单通道降噪

FastEnhancer 用于单通道神经语音增强。[官方固定 README](https://github.com/aask1357/fastenhancer/blob/f85223bd546b27f39dc0744e0310dcd246f750a4/README.md)说明其模型按噪声抑制训练；本文只核对固定代码的接口，不转录跨数据集性能或实时性排名。源码提交为 `f85223bd546b27f39dc0744e0310dcd246f750a4`，已按[MIT 许可](https://github.com/aask1357/fastenhancer/blob/f85223bd546b27f39dc0744e0310dcd246f750a4/LICENSE)获取至 `codes/chapters/ch00/upstream/_downloads/fastenhancer/`，核对日期为 2026-09-26。

**模型算什么。** 默认实现 `models/fastenhancer/default/model.py` 中，`RNNFormerBlock` 组合时间递归状态与频率注意力；`ONNXModel.forward` 对实部/虚部表示的频谱先做幅度压缩，再预测复数掩码并执行复数乘法，最后还原压缩。它返回增强频谱和更新后的模型缓存。目录中还有 `noncausal` 等变体，不能只看到项目名称含 streaming 就把每种配置都当成相同因果结构。[固定模型源码](https://github.com/aask1357/fastenhancer/blob/f85223bd546b27f39dc0744e0310dcd246f750a4/models/fastenhancer/default/model.py)

**每块带什么状态。** `scripts/export_onnx.py::Model.forward` 的接口包含 `wav_in`、STFT 缓存、逆 STFT 缓存及模型缓存；返回本块波形与全部新缓存。实际推理示例 `scripts/test_onnx.py` 从会话输入名寻找 `cache_in_*`，会话起点初始化为零，每次调用后把输出缓存传给下一块。逐块清零会破坏连续流语义；换会话时忘记清零又会把旧状态带入新会话。[导出包装器](https://github.com/aask1357/fastenhancer/blob/f85223bd546b27f39dc0744e0310dcd246f750a4/scripts/export_onnx.py)、[推理循环](https://github.com/aask1357/fastenhancer/blob/f85223bd546b27f39dc0744e0310dcd246f750a4/scripts/test_onnx.py)

推理脚本默认 16 kHz、`n_fft=512`、`hop_size=256`，后者对应每步 16 ms；实际运行须与导出模型配置相符。脚本在结尾补零以排出状态，拼接输出后裁去 `n_fft-hop_size=256` 点，再取原输入长度。这个 16 ms 裁剪是默认配置的离线对齐操作，不能代替采集、凑块、模型计算、排队和播放共同构成的端到端延迟。其 RTF 计时包围 Python 推理循环并含循环开销，不能把项目表中的数字移植到另一机器。

固定 [`test_onnx.py`](https://github.com/aask1357/fastenhancer/blob/f85223bd546b27f39dc0744e0310dcd246f750a4/scripts/test_onnx.py#L42-L51)的打印分母还包含补零/排空对应的调用长度：循环上界是 `length+n_fft-hop_size`，末尾除以 `idx+hop_size`。1000点输入在默认配置下实际调用5次，打印分母1280点；受控假计时10 ms得到0.125，而按新增源音频时长计算是 $0.01/(1000/16000)=0.16$。16000点时打印分母也成为16384点，不严格等于原录音长度。两者应分别命名，见[E10-32](../../../../chapters/10_engineering-practice.md#e10-32)，不能将填充样本算成额外源音频。I32实际执行的是原 `main` 的AST与明确的加载、会话、计时及保存接收适配器；它验证循环、状态和裁剪，不提供本书测得的模型RTF。

**怎样设计最小核查。** 取得明确来源的同版本权重后，先固定单通道 WAV、模型摘要、块长、ONNX Runtime 版本、线程数和执行后端。对照官方导出器的 `--test-streaming` 路径，检查整段与保留缓存的逐块输出是否在相同对齐、尾部排空和裁剪下相符；再故意逐块清零，观察边界差异。容差须按数据类型和运行时预先确定，不能假设不同内核必然逐位一致。最后分别测启动、稳态、停流排空与新会话重置。

原2026-09-26记录只读取并保存源码；2026-10-01新增上述包装器合同调用，没有取得权重、导出ONNX或执行真实推理，所以仍没有本书测得的模型质量分、RTF或增强音频。该接口没有播放参考，也没有 WPE 的延迟多通道预测器；不能将它计作 AEC 或 WPE 实现。将其接在波束输出后作单通道 NS 是可研究的系统组合，仍须独立检查前端输出分布、目标失真和端到端任务指标。

2026-09-29 核对[论文原始记录](https://arxiv.org/abs/2509.21867)和 [ICASSP 2026 官方会议节目](https://www.cmsworkshops.com/ICASSP2026/view_paper.php?PaperNum=3646&bare=1)：FastEnhancer 已列入 SLP-P2.8，节目给出 2026-05-05 的报告场次。固定 README 的“已接收”是该源码快照的历史措辞；这里不据此猜测正式卷页或 DOI。另一个项目对其 48 kHz 发布配置做了专用 C 推理移植，见[I31](#faster-enhancer-runtime)；原模型论文的 16 kHz 实验、这里默认 16 kHz 脚本和移植的 48 kHz 配置必须分开，不能混成同一个速度/质量基准。


<a id="i29stkdelayl"></a>

### I29：STK DelayL 的分数延迟、幅度误差与跨块状态

STK（The Synthesis ToolKit in C++）提供音频合成与处理组件。本节只研究其中的固定分数延迟器，不将工具箱整体当作麦阵列产品。[维护者源码](https://github.com/thestk/stk/tree/6aacd357d76250bb7da2b1ddf675651828784bbc)固定为 `6aacd357d76250bb7da2b1ddf675651828784bbc`；2026-09-26核对[MIT-STK许可](https://github.com/thestk/stk/blob/6aacd357d76250bb7da2b1ddf675651828784bbc/LICENSE)，保留作者与许可。下载选择include/src/README/LICENSE，没有取得rawwaves素材或打开音频设备。

**入口与状态。** `src/DelayL.cpp`分配延迟线，`include/DelayL.h`中的`setDelay`按当前写指针确定读指针和分数权重。`tick`先写当前样本，再读插值结果，因此可以表示零延迟；每次调用推进读写指针。`nextOut`在下一次tick前缓存结果，不能把反复调用它当成推进一个采样。`include/Filter.h::clear`清空历史；逐音频块调用clear会破坏跨块连续性。改变最大容量可能重新分配存储，应在初始化阶段完成；本节未测动态延迟调制。

| 阅读位置 | 要核对的变量/步骤 | 独立判据 |
|---|---|---|
| `DelayL.h::setDelay` | 读指针追随写指针，单位为采样 | 延迟0/.5/1分别对应当前值、相邻平均、前一个值 |
| `DelayL.h::tick/nextOut` | 先写后读、插值权重、指针回绕 | 逐样本与独立二抽头FIR一致 |
| `Filter.h::clear` | 历史状态归零，不是正常块边界操作 | 连续状态分37点调用应相同；逐块clear作为反例 |

**本机已执行。** [原创C++探针](../../ch10/examples/stk_delay_probe.cpp)与[运行入口](../../ch10/examples/run_stk_delay_probe.py)只编译锁定版`Stk.cpp`和`DelayL.cpp`，临时生成物不写回上游。输入为512点、16 kHz、500/6000 Hz双音，两个分量幅度均0.18；无淡入淡出、初始历史为零。它与2 s试听素材使用相同频率，但长度和边界不同，不能借用试听文件代替此接口测试输入。

在Darwin arm64、Apple clang 21.0.0、C++17/O2、STK默认double下，半采样输出相对独立FIR的最大绝对误差为0；零延迟、一采样延迟的已知边界误差也为0。把同一标量tick调用分成37点组且保留对象状态，输出差为0；每组前清空对象，最大绝对差为0.1519102855。结果、输入条件、编译器和实际源码摘要保存于[STK接口报告](../../ch10/reports/stk_delay.json)。这里的0只表示这组输入在该编译环境下的计算结果，不是任意平台的零误差保证。

复跑命令为 `.venv/bin/python -m codes.chapters.ch10.examples.run_stk_delay_probe --report tmp/stk-delay-rerun.json`，需已取得锁定源码与C++编译器，不联网、不安装依赖。该检查只覆盖标量tick及固定延迟；没有测试StkFrames多通道接口、设备时间戳、实时期限或语音质量。

幅度和延迟需分别验证。一次半采样线性插值对6000 Hz的幅度比约0.382683，两次则约0.146447，即使累计相位对应一采样，仍不等同纯一采样移位。完整本书推导、PCM统计区间与图41见[附录B E13-02](../../../../chapters/13_appendix-guide.md#e13-02)。工业选型须另查目标频带误差、可变延迟的状态转换和存储上限；本节不把一个组件检查外推为完整重采样器验收。

<a id="rnnoise-stream-contract"></a>

### I30：RNNoise 的幅度单位、帧状态与演示程序首尾

RNNoise 将传统分析、基音相关处理与神经增益估计结合；原始依据是 Valin 的 [*A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement*](https://arxiv.org/pdf/1709.08243)（MMSP 2018）。固定源码为 `70f1d256acd4b34a572f999a05c87bf00b67730d`，根 `COPYING` 为 BSD-3-Clause；文件头仍单独保留。这里核对当前锁版的接口，不能把2018论文网络规模或实验数字自动套到此后更新的网络与模型文件。

#### 浮点容器不代表归一化幅度

[`src/denoise.h`](https://github.com/xiph/rnnoise/blob/70f1d256acd4b34a572f999a05c87bf00b67730d/src/denoise.h)定义每步480点、分析窗960点；官方示例按48 kHz使用，即每次增加10 ms音频。实际接入先调用 `rnnoise_get_frame_size()` 核对所用库，而不是对未知版本硬编码。`src/denoise.c::rnnoise_process_frame` 保留分析、合成、基音与网络状态，返回的标量是语音活动概率，增强样本写入 `out`。

[`examples/rnnoise_demo.c`](https://github.com/xiph/rnnoise/blob/70f1d256acd4b34a572f999a05c87bf00b67730d/examples/rnnoise_demo.c)把16位有符号 PCM 的 `short` 值直接转成 `float` 后送入，没有除以32768。这个幅度约定与 PortAudio 的归一化浮点格式不同；输入源若是后者，适配时须显式换算，输出也须按目标格式换算和检查范围。变量类型都叫 float 并不能保证单位一致。原示例只演示原始 PCM 文件，不读取 WAV 容器头。

#### 首尾样本和模型生命周期

同一示例处理首帧但不写出，后续每个完整帧才写480点；EOF前不足一帧的输入被丢弃，没有尾部补零排空循环。以三个完整输入帧为例，静态控制流推得处理1440点而写出960点。这个计数描述示例包装器，不代表任何接入方式都应丢首帧，亦不单独证明增强输出已正确补偿所有算法延迟。连续系统必须定义启动填充、最后不足帧、尾部排空以及输出对应输入的样本映射。

[`include/rnnoise.h`](https://github.com/xiph/rnnoise/blob/70f1d256acd4b34a572f999a05c87bf00b67730d/include/rnnoise.h)要求自定义模型晚于引用它的 `DenoiseState` 释放。由缓冲区或文件构造模型时，相应缓冲区或文件还需存活到模型销毁。初始化/分配与每帧处理分开；不能在每个音频块重建状态，或提前关闭模型文件。

原2026-09-28记录仅静态核对代码；2026-10-01在临时目录编译原完整演示主程序，并将三个模型接口明确替换为身份适配器，核对幅度和首尾计数，见I32新合同。没有编译原RNNoise增强核心、加载模型或取得增强音频。`autogen.sh` 的模型获取行为、发布模型版本和数据许可须另审，不能为编译方便自动执行下载脚本。最小后续核查应使用固定模型摘要、已知幅度的完整/不足帧输入、连续状态与重置对照，并独立检查总样本数；这不构成自然语音降噪性能或主观听测。

<a id="faster-enhancer-runtime"></a>

### I31：faster-enhancer.c 的专用量化运行时与节奏测试

本节收录的是部署实现研究，不新增一种语音分离或增强模型。Kim 的 [*faster-enhancer.c*](https://arxiv.org/html/2607.25350v1) 是2026-07-28的预印本。它独立移植 FastEnhancer-Medium 的48 kHz发布配置，作者明确与 FastEnhancer 原作者无隶属关系。论文§3.1绑定的[官方代码](https://github.com/kdrkdrkdr/faster-enhancer.c/tree/7d78dab11bca854fb200426c621a96d7b37a2983)完整提交为 `7d78dab11bca854fb200426c621a96d7b37a2983`。

#### 相对通用推理器改变什么

它保持模型结构，把固定形状的算子交给 C/SIMD 内核，按每个输入帧重估激活量化范围，将跨阶段状态保留为半精度，并在初始化时选择处理器指令路径。论文§2讨论将整数范围限制在−127～127，使两个乘积之和最大为32258，小于有符号16位上限32767；这属于该内核的溢出与跨指令一致性设计，不是所有 int8 模型都应采用的量化范围。

公开接口 `include/fe.h` 是初始化、320点处理、重置和释放。论文§2.1给出48 kHz下每步约6.67 ms、704点 STFT 对齐延迟约14.67 ms；无额外前瞻仍有这些分析与排队条件。它使用单个全局实例，CPU单线程且没有标量回退；支持的 arm64 或 x86_64 指令能力和编译器必须满足要求，不能直接用于任意微控制器或多会话并发。以上是论文/固定源码声明，没有本书设备测量。

#### 为什么按帧期限送数据

原文§3.1、§3.4将尽快遍历文件与按6.67 ms间隔送帧分开，并记录硬件、预热、重复和功率测点。这补充了本章“平均 RTF 合格不代表实时期限合格”的具体测试设计。本文不转录其速度排名，也不把一个平台的频率调节规律推广到全部 CPU。最小选型试验是对同一构建、模型、输入和输出对齐协议分别执行两种送帧方式，记录逐帧时间、逾期次数、输出摘要与功率，而非只给一个全文件平均 RTF。

#### 许可、源码选集和复现缺口

2026-09-28实际读取固定版 `LICENSE`、`NOTICE` 与 `CMakeLists.txt`：代码采用 MIT，NOTICE 保留 FastEnhancer 原版权并声明第三方内核设计参考，没有将参考项目整体包含进来。源码选集为 `LICENSE`、`NOTICE`、`README.md`、`CMakeLists.txt`、`include/`、`src/`、`tests/` 和 `docs/`；排除 `weights/fe.q8`、测试音频与外部下载脚本。模型是单独资产，代码取得范围不能替代模型授权和摘要核对。

CMake要求3.20和C11，库链接平台数学库；MSVC路径明确拒绝，目标架构和各编译单元还有指令集要求。该选集可研究源码与构建前提，但缺权重时不能运行完整增强；测试执行器存在也不等于跨指令输出检查通过。本书未编译此项目、未加载权重、未运行推理或复现论文计时。核对实际本地取得状态应读取[源码状态报告](../SOURCE_STATUS.json)，而非由本节的下载路径推断。

收录理由限于量化范围、显式流状态和按真实节奏计时三点，均直接对应本章已有工程问题。它不替代 I28 的模型解释，不扩写神经模型排行榜。缺少设备和模型时，可先复算32258这个溢出边界；任何增强质量、内存峰值或性能结论仍需完整运行证据。

<a id="industrial-upstream-audit"></a>

### I32：固定工业接口诊断的执行范围

[诊断脚本](../../ch10/examples/audit_industrial_upstream_interfaces.py)及[JSON报告](../../ch10/reports/industrial_upstream_interfaces.json)对应本章来源审查 SRC10-01～03。脚本先验证独立上游工作树的完整提交、干净状态、被用源码与许可摘要，再执行；结束再次检查，禁止自动获取依赖和改写上游。报告还绑定脚本摘要及输入配置摘要，JSON不写入非标准 NaN 或 Infinity。

| 对象 | 本轮实际动作 | 未执行 |
|---|---|---|
| pystoi | 原 Python 包12次调用，保留异常、警告和返回值 | MATLAB/Octave、重采样路径、自然语音数据 |
| DeepFilterNet | 固定 Rust 文件静态核查；独立 Python 队列算例 | Rust构建、网络、LADSPA/PipeWire宿主 |
| 其他工业库 | 本轮文档/源码复核；既有C探针另有报告 | 不据旧报告声称本轮重跑或设备验收 |

复跑需要 NumPy/SciPy 和已取得的固定源码，不联网安装。采用本机已有隔离环境的示例命令：

```bash
PYTHONDONTWRITEBYTECODE=1 /tmp/room-pra-venv/bin/python \
  codes/chapters/ch10/examples/audit_industrial_upstream_interfaces.py \
  --report tmp/industrial-upstream-rerun.json
```

该临时环境路径只描述本次记录；其他机器须指定自身已有环境。普通[离线测试](../../../../tests/test_codes_industrial_upstream_interfaces.py)只读取报告与本书脚本，无需上游缓存、SciPy或网络。期望值来自独立相关性恒等式、首次只读调用保留值和按通道/按时间样本的计数区别；测试通过不表示已认证这些库的全部输入和硬件。

本轮检索按“PortAudio callback flags / framesPerBuffer”“ALSA PCM errors / timestamps”“ITU P.835 / P.566 2026”“BS.1770-5 / EBU R128”“STOI ESTOI original paper”“ViSQOL v3”“FastEnhancer ICASSP2026 / faster-enhancer.c”定位官方资料，并实际打开原文或固定代码。STOI作者站PDF本轮访问失败，因此不把代码实跑说成已读并复现作者全文；已有Python接口结论依上面固定源码及原调用报告。P.835与P.566的生效状态、预发布/待发布状态分开核查，不用搜索摘要替代标准条款。

<a id="industrial-controlled-contracts"></a>

#### 2026-10-01：五项固定源码合同的受限执行

新[合同工具](../../ch10/examples/audit_industrial_contracts.py)和[对应报告](../../ch10/reports/industrial_contracts.json)另记本次运行，不修改上述2026-09-28报告。工具先核官方地址、完整HEAD、工作树清洁状态、各源码与许可证的SHA及Git blob，再在临时目录编译或提取原函数；结束重复核对。报告绑定工具和完整锁表摘要，严格JSON不写NaN/Infinity。获取工具的 `not_run` 仍表示获取过程没有执行上游，不改成整包运行成功。

| 固定对象 | 实际执行与输入 | 独立核对结果 | 代替项和未执行范围 |
|---|---|---|---|
| CMSIS-DSP Q15 FIR | 两个原C翻译单元、四抽头；半LSB、负满幅、非对称整段/2+1+4点分块 | 半LSB输出 `[0,0,-1,-1,0]`，饱和为32767；逆序系数与逐项整数卷积一致 | 最小类型/结构头、受限 `__SSAT`；未运行ARM优化内核或板上周期 |
| SpeexDSP 噪声助手 | 原四函数body和原 `arch.h` 浮点宏；两个频点、上升/下降功率及最小值重启 | 硬标记 `[0,1]`；噪声 `[2,2]→[3,2]→[2.5,1.75]`；两最小值状态分别核对 | 自写驱动，未运行完整预处理器、FFT或原MCRA/IMCRA |
| WebRTC VAD入口 | 原合法性及Process两个body；四采样率×10/20/30 ms、六不合法输入 | 12组合法；44.1 kHz、15 ms、空句柄、未初始化、空音频、0长度均返回−1；四路各调用3次 | 四CalcVad仅计数替身，最小初始化结构；不是实际语音判断或误报率 |
| RNNoise 演示程序 | 原完整demo主程序；1440、1457、479及0点原始PCM | 输出960、960、0、0点；首帧不写、尾17点不送；送入float的首值1000而非1000/32768 | create/process/destroy身份替身；未编译原增强核心、加载模型或测试其生命周期 |
| FastEnhancer 包装器 | 原main AST；1000、16000、256、1及0点，显式保存接收适配器 | 固定256点调用、cache连续；先裁256点再取原长度；1000点原打印分母1280 | 加载、会话、计时、进度和保存均列替身；没有ORT模型推理或真实耗时 |

Speex例中 $0.4\times10=4$ 与最小值4相等时不设置标记，最小值3.99时才设置。随后当前功率1低于旧噪声2，第二频点即使仍带标记也允许下降更新。参数 `beta=0.25`、`beta_1=0.75`，浮点绝对容差 $10^{-6}$；这不是自动估计得到的语音概率。最小值运行步由 `[5,7]`、暂存 `[6,4]` 和当前 `[3,8]` 得到 `[3,7]`、`[3,4]`，重启步再输入 `[9,2]` 得到 `[3,2]`、`[9,2]`。

FastEnhancer的计时器故意返回0和0.01，人工会话输出确定性斜坡、状态每次加1；保存接收器只在内存验证样本，不创建增强WAV。0点输入仍触发一次填充调用并打印有限值0.625，但源时长为0，按本书分母定义的RTF仍未定义，报告记 `null` 和原因。输出有限不是评分有效的充分条件。

复跑使用已取得的五个固定工作树、项目NumPy环境和C编译器，无联网获取和依赖安装：

```bash
.venv/bin/python -B -m codes.chapters.ch10.examples.audit_industrial_contracts \
  --report codes/chapters/ch10/reports/industrial_contracts.json
.venv/bin/python -B -m unittest tests.test_codes_industrial_contracts -v
```

不传 `--report` 时仅输出JSON，所有C编译和原始PCM保留在临时目录后清理。缓存缺失时，普通测试明确跳过可选原源码实调；固定版本或工作树不符则报错，不能以跳过伪装核验通过。新报告只证明以上合同，没有声卡、Linux路由、ARM指令、神经模型、自然语音质量或出厂验收结果。

<a id="distributed-network-deployment"></a>

## 8. 分布式节点的观测、更新与网络时间轴

2026-10-04核对以下原论文、固定源码和官方协议文档。本节把第10章§10.2的时钟接口、§10.7的多节点工程问题与[扩展专题Ⅱ](../../../../chapters/15_distributed-enhancement.md)接起来。节点特定输出沿用第5章MWF的线性估计目标；网络交换什么、滤波器何时变、样本属于哪个时刻，则是新增的条件。正文的已知统计控制、原函数小输入、原论文实验和实际网络设备分别记录。

### I33：原始DANSE的模型与收敛保证

先固定符号。原始[Part I：Sequential Node Updating](https://homes.esat.kuleuven.be/~abertran/reports/09-65.pdf "citation")和[Part II：Simultaneous and Asynchronous Node Updating](https://homes.esat.kuleuven.be/~abertran/reports/09-178.pdf "citation")均发表于2010年；DOI分别为10.1109/TSP.2010.2052612与10.1109/TSP.2010.2052613。原文中的$K$不是节点数：

| 含义 | Part I/II原文 | 本书扩展专题Ⅱ |
|---|---|---|
| 节点数 | $J$ | $U$ |
| 每节点广播/估计维数 | $K$ | $Q$ |
| 共同潜在目标维数 | $Q$ | $S$ |

因此原条件$K=Q$对应本书的$Q=S$。在共同频点和样本时间轴上，节点$u$观测$\vec x_u$，各节点目标为同一潜在向量$\vec s$的节点特定变换$\vec d_u=\mathbf B_u\vec s$。$Q=S$时，$\mathbf B_u$须为满秩方阵。集中式LMMSE的权重满足$\mathbf R_{xx}\mathbf W_u^\star=\mathbf R_{xd_u}$；DANSE节点保留自己的原观测，接收其他节点的$Q$维线性组合，先在此局部空间求解，再用解的本地块更新以后广播。广播是本地观测的组合，不是把包含远端观测的完整增强输出再循环广播。模型与更新定位在Part I §II、§III-A、§IV-A；单维与多维顺序结论分别见Theorems III.1、IV.1。[Part I原文](https://homes.esat.kuleuven.be/~abertran/reports/09-65.pdf "citation")

上述顺序结论讨论满秩的观测二阶矩矩阵 $\mathbf R_{xx}$、共同目标满秩变换、固定精确二阶统计和轮转更新。它说明合适的**自适应**广播可到达集中式最优输出，不能证明任意预先固定的少数混合都无损；本书§15.8给出了相关噪声改变最优远端方向的反例。有限快拍、失配VAD和非平稳输入使局部统计不再是定理中的精确矩阵。若广播维数低于共同目标维数，最优方向可能丢失；过估维数造成的秩亏和伪逆情形，也不能简单套用满秩权重收敛结论。[Part I §IV-C：DANSE Under Rank Deficiency](https://homes.esat.kuleuven.be/~abertran/reports/09-65.pdf "citation")

同时更新时，各节点求解所用的其他广播随后也改变。Part II §IV-B式(22)～(28)的$rS$-DANSE$^{+}$多了在全部广播空间中优化$\mathbf G$的步骤；其Theorem IV.1要求$0<\alpha_i\le1$、$\alpha_i\to0$且$\sum_i\alpha_i=\infty$，并保留上述统计和共同目标条件。去掉额外$\mathbf G$优化的简化$rS$版本在§IV-C另作经验讨论。固定$0.5$或$0.7$的混合既不满足趋零条件，也不能替代额外优化，故不援用该定理。Part II §IV-D的参数异步允许节点在不同迭代事件更新，但脚注8仍要求采样同步；异步结论还要求各节点持续获得更新机会。[Part II原文，5297～5298页](https://homes.esat.kuleuven.be/~abertran/reports/09-178.pdf "citation")

### I34：WOLA源码的滤波时序与论文配置

[IWAENC 2010作者全文](https://homes.esat.kuleuven.be/~abertran/reports/IWAENC10.pdf "citation")§III将频域语音失真加权MWF接入加权重叠相加（WOLA）分析/合成，讨论平方根Hann、50%重叠和指数统计更新；其链路模型忽略传输延迟。§IV的实验使用32kHz、512点窗、$\mu=5$、$\alpha=0.5$与理想VAD。这组论文配置不能由另一版演示文件的默认值反推。

本书取得的作者[固定 `WOLA_DANSE1.m` v1.4](https://github.com/AlexanderBertrandLab/Old_Code/blob/a24b73fcd2dc028659535d07bb08068b06108616/WOLA_DANSE1.m "citation")必须显式传入`fs`，没有采样率默认值。其示例输入使用16kHz；默认FFT长为`512*fs/16000`，故该示例对应512点。真正的默认项包括`mu=1`、`alpha=0.7`和同时模式开启；协方差与外部平滑的半衰期分别为2s、0.2s。更新门限分别累计每个VAD类别的三秒等效帧数，不代表输入三秒后必定有可用的两类统计。

静态控制流还揭示四项接入合同。其一，VAD取每帧首样本；外部`Wext`先用于广播，随后向旧目标平滑，更新事件最后才产生新目标，因而事件产生的新目标在第二个后续广播帧才开始影响消息。其二，即使选择顺序token模式，各节点的内部`Wint`仍逐帧更新，token限制的是外部目标更新。其三，原文件的对称Hann没有逐点COLA校正，循环也没有排出完整尾部。其四，直接`inv`没有加载；普通EVD先取代数最大特征值，再取其绝对值，不能当作噪声白化GEVD或正半定投影。这些是固定源的实际步骤，未执行MATLAB。[静态定位与独立控制报告](../../ch15/reports/upstream_distributed_contracts.json)

读取原源码时应分别保存内部估计权重、当前外部广播权重、外部目标权重和更新时间。把同帧谱直接拼接的演示改成网络程序，还要引入样本标签、广播版本、等待/缺口规则与队列；不能只加一个发送函数就沿用其零延迟观测模型。

### I35：拓扑、低秩统计与采样率补偿分别改变什么

| 方法与原文定位 | 改变的计算步骤 | 成立条件与采用范围 |
|---|---|---|
| [TI-DANSE，§III～IV、Theorems 1/2](https://homes.esat.kuleuven.be/~abertran/reports/15-87.pdf "citation")；正式发表于2017年，DOI含2016 | 广播使用$\mathbf P_u=\mathbf W_{uu}\mathbf G_u^{-1}$，树上求和并传播总和，节点减去自身贡献；本地维数成为$M_u+Q$ | 保留可逆坐标变换、连通拓扑、同样本准确求和及共同目标条件。只运行树上加法不等于运行TI-DANSE；输出最优与参数坐标是否收敛也须区分 |
| [GEVD-DANSE，§III式(9)～(17)、§IV](https://ftp.esat.kuleuven.be/SISTA/abertran/reports/GEVD_DS_TSP2016.pdf "citation")；2016 | 对总/噪声协方差做噪声度量下的GEVD，保留选定秩，方向增益为$1-1/\lambda$，再进行分布式更新 | 顺序讨论要求有效噪声统计、局部满秩和相应特征子空间条件。保留秩低于真实目标维数时，可能收敛到对应集中式低秩GEVD结果，目标已不同于完整LMMSE；原MATLAB普通EVD不是此算法 |
| [TI-GEVD-DANSE，作者稿§III～IV](https://ftp.esat.kuleuven.be/sista/pdidier/eusipco2024/eusipco24_pdidier_submission_v1.pdf "citation")；EUSIPCO 2024 | 在TI结构中使用局部GEVD，增加共同坐标规范化，抑制滤波尺度漂移导致的数值问题 | 同一变换须同时作用于相关协方差和滤波器，不能各节点独立削幅后仍称同一算法；欠秩情形的仿真观察不作为完整证明 |
| [TI-DANSE+会议作者v1，§III～V](https://arxiv.org/html/2506.20001v1 "citation")；2025 | 更新根分别保留邻居分支的部分和，增加局部自由度，并用最大更新节点树策略选择连接 | 局部维数与根度数有关。该版以固定统计仿真说明收敛行为，未提供完整证明，不把轮数改善换算成设备时间 |
| [TI-DANSE+扩展v2，§III-C～G、§IV](https://arxiv.org/html/2506.02797v2 "citation")；2026-03-10预印本 | 增加Theorems 1/2及通信、树选择和统计更新讨论 | 满维共同目标、可逆变换、顺序更新等前提仍需满足；跨广播版本混合旧SCM会破坏同一统计坐标。§III-G的低秩GEVD扩展没有同等完整证明；§IV-D的集中应用网络滤波器不等于运行真实WOLA网络 |
| [SRO感知DANSE，§IV-A～C式(11)～(20)](https://arxiv.org/html/2211.02489v2 "citation")；OJSP 2023 | 从跨帧相干性漂移估计SRO，累计分数相位补偿，并跟踪整数样本滑移；为逐样本发送引入WOLA卷积近似 | 基线讨论静止节点/目标、稳定SRO和理想传输。同节点通道同步，广播变化足够慢以维持声学相位近似；卷积近似仍有分析延迟，不是精确WOLA或零延迟实现 |

TI-DANSE的正式卷期是TSIPN 3(1)，130～144，2017，DOI为10.1109/TSIPN.2016.2623095；不要用DOI中的年份替代卷期年份。TI-GEVD作者稿式(11)的目标特征值差写法与式(12)增益不一致；本书白化例依据2016年GEVD原文使用$\lambda-1$与$1-1/\lambda$，不照抄该处差式。[2016年GEVD作者稿](https://ftp.esat.kuleuven.be/SISTA/abertran/reports/GEVD_DS_TSP2016.pdf "citation")

SRO估计还要区分三层输入。第10章的拟合控制使用精确合成时间戳，不能当作盲音频估计。2023年SRO-DANSE以跨节点相干性变化推测速率，再通过接收计数与整数滑移维持相位历史；丢包、启动错位和声源运动会引入其他变化。16kHz、100ppm的设备每秒多1.6个样本，约0.625s积累一个样本；若帧移为512点，积累一个**整帧移**约需320s。后一个尺度不能用作允许忽略采样偏差的时间。[SRO-DANSE §IV-B](https://arxiv.org/html/2211.02489v2 "citation")

[DWACD原文，ICASSP 2022 §II～III](https://ris.uni-paderborn.de/download/33807/48990/gburrek_icassp22.pdf "citation")允许活动说话人在不同局部片段改变位置，但依赖相关性、活动筛选和短片段内近似稳定的声学关系；其STO讨论还利用位置/传播时间信息，不能把初始错位全当作时钟偏差。其公开同步源码可作后续入口，实际仅运行三个小函数的范围见[复现合同](04_source_reproduction.md#distributed-reproduction)。这些方法纳入学习导航，是因为分别解决广播维数、拓扑、低秩统计和采样时钟问题；不将论文中的不同任务与配置排成统一设备排名。

### I36：RTP时间戳、参考时钟与Dante的适用范围

包的到达时间、首样本采样时间与播放时间是三个测点。[RFC 3550 §5.1](https://www.rfc-editor.org/rfc/rfc3550.html#section-5.1 "citation")的RTP序号按包递增；时间戳标记包内首个音频采样时刻，按采样时钟推进，起始值可以随机。两个流的裸时间戳不能直接相减，RTCP发送报告中的NTP/RTP对才提供时间映射。§6.4.1的到达间隔抖动是另一统计量，不能直接作为ADC相对ppm。序号缺口也不自动给出缺失的样本数，须结合载荷、时间戳和发送协议。

[RFC 7273 §4～6](https://www.rfc-editor.org/rfc/rfc7273.html "citation")分别声明时间戳参考时钟与媒体时钟来源；`ts-refclk`和`mediaclk`表达不同关系。操作系统墙钟由PTP校准，本身不足以证明音频ADC速率也被该时钟约束。跨节点相干增强应检查参考钟、媒体时钟、初始映射、累计采样计数和不连续事件；缺少关系声明时不能仅以两个文件头相同采样率认定同步。

Audinate的[Dante Controller官方时钟页](https://dev.audinate.com/GA/dante-controller/userguide/webhelp/content/clock_synchronization.htm "citation")描述Dante设备的领导/跟随时钟：默认使用PTPv1，设备启用RTP时也使用PTPv2；硬件设备可用板载或外部字时钟，Dante Virtual Soundcard使用计算机时钟。[官方延迟页](https://dev.audinate.com/GA/dante-controller/userguide/webhelp/content/latency.htm "citation")中的接收延迟是从输入样本时间戳到预定播放时刻的安排，实际流使用发送与接收设置中较高的值；通用计算机端点还可能需要额外延迟。这属于相应Dante平台的系统合同，不是对普通Wi-Fi设备的硬件采样锁定或任意操作系统的实时保证。部署时仍须检查具体端点、外部字时钟和实际状态；本书没有连接Dante硬件或运行PTP验收。

### I37：先算有效负载，再讨论链路带宽

沿用$U$节点、每节点$M_u$路原观测、$Q$路广播。若明确发送**实数时域PCM**，采样率$f_s$、每样本$b$位，则节点原观测有效负载为$f_s bM_u$ bit/s，压缩广播为$f_s bQ$ bit/s。以16kHz、PCM16、每节点三麦而广播一路为例，两者分别为768与256kbit/s。这是标量数量的换算；实际采用float32或复数谱时必须重算。

四节点逐接收者单播需要$U(U-1)=12$条一路流，有效负载合计3.072Mbit/s。若网络真正支持一次发送供所有其他节点接收，四条发出流的计量为1.024Mbit/s，不能把这个数当作单播结果。树上先融合到根、再传播总和，每个同样本求和使用$2(U-1)=6$条定向边消息，按一路流计为1.536Mbit/s；这个消息数控制对应[E15-23](../../../../chapters/15_distributed-enhancement.md#e15-23)，还没有包含TI的坐标变换或控制更新。交换一个$Q\times Q$的complex64矩阵另需$8Q^2$字节/次，须乘实际更新率及复制次数。

分帧频谱不能套时域码率。例如FFT512、帧移256、complex64单边257个频点，在16kHz下每秒62.5帧，每路有效负载为$257\times8\times62.5=128500$ B/s，即1.028Mbit/s，尚未加帧标签和控制信息。[E15-15](../../../../chapters/15_distributed-enhancement.md#e15-15)同时列出PCM、float32和复谱口径。

另作一个显式包头算式：单路PCM16每10ms发送160点，载荷320B；若假定无扩展/CSRC的12B RTP基本头、8B UDP头和无选项的20B IPv4头，每包360B，每秒100包，即288kbit/s，比纯载荷高12.5%。这些头长分别见[RFC 3550 §5.1](https://www.rfc-editor.org/rfc/rfc3550.html#section-5.1 "citation")、[RFC 768格式](https://www.rfc-editor.org/rfc/rfc768.html "citation")和[RFC 791 §3.1](https://www.rfc-editor.org/rfc/rfc791.html#section-3.1 "citation")。以太网、VLAN、无线竞争/重传、安全封装和PTP均未计入，所以这是声明条件下的数学预算，不是实测吞吐量。广播维数下降还须与目标信息损失、量化误差、等待和缺口分母一起比较，不能仅凭码率称增强质量无损。
