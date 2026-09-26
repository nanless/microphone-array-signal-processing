# 工业音频实现：从采集、状态到部署和评分

原核实日期：2026-09-22；AEC 接口 I03/I05 与评分 I20 复核日期：2026-09-23；I01 时间戳、I18 输入输出绑定与 I28 源码接口核对日期：2026-09-26。对应正文第 10、11 章和附录 B。这里讨论本书算法进入连续音频系统后需要补上的部分：驱动、参考路由、跨块状态、定点内核、模型运行时和评分器。每项的完整提交以 [`SOURCES.lock.json`](../SOURCES.lock.json) 为准；网页文档版本只说明核实依据，不自动等于本机安装版本。

除明确写出输入、环境和实测值的本地接口实验外，下面的试验是建议执行的设备验收步骤，不是已经测得的产品结果。源码下载、依赖安装、编译、运行、声学测量是不同状态。没有硬件、模型或数据时，可以完成源码核对，但不能把该项标成已经通过设备验收。

## 1. 采集、时钟与连续流

### I01：PortAudio 回调与设备时间戳

PortAudio 为不同主机音频系统提供统一入口。本书锁定 19.7.0 对应提交。先读 `include/portaudio.h` 的回调签名，再看 `examples/` 中的采集与播放例子；库为 MIT 许可，平台后端仍有自身依赖。官方[回调说明](https://portaudio.com/docs/v19-doxydocs/writing_a_callback.html)明确限制会阻塞或耗时不可预测的操作。

设备配置要记录 host API、设备 ID、采样率、样本格式、通道数、请求与实际延迟、回调块长。`frameCount` 表示每通道时间样本数，不能乘通道数后再当作帧数。保存输入 ADC 时间、输出 DAC 时间和状态位；它们不能未经校准就当成两个设备共享的时钟。

验收可以用环回脉冲测端到端延迟，再施加磁盘和 CPU 负载。把状态位、回调间隔、队列长度与音频缺口关联起来。回调中只将样本交给预分配缓冲，不应直接运行 Python 模型或同步日志。设备拔插后需重新确认通道映射，而非只恢复原设备编号。

**把测点和对应样本一起保存。** [官方时间戳结构体](https://files.portaudio.com/docs/v19-doxydocs/structPaStreamCallbackTimeInfo.html)分别表示输入首样本的 ADC 时刻、实际回调调用时刻与输出首样本的 DAC 时刻。只有属于同一时间基准且知道哪一个输入样本对应哪一个输出样本时，端点之差才是需要的延迟。工作线程若把结果延后一个输出块，必须跟着改样本映射；不能继续减当前回调的两个首点时刻。

[E10-17](../../chapters/10_engineering-practice.md#sec-10-11)给出 15 ms 输入年龄、30 ms 输出提前量与 45 ms 对应样本延迟的手算，并展示多排一块后变成 55 ms。3 ms 计算已位于端点之间，不能重复加上。PortAudio 的[缓冲与时序指南草案](https://github.com/PortAudio/portaudio/wiki/BufferingLatencyAndTimingImplementationGuidelines)也说明部分后端只能报告已知延迟；API 字段不是实际精度的保证，仍须用目标后端和环回信号核查。上述例子没有打开声卡。

### I02：ALSA PCM 状态与恢复

Linux 设备层可对照 `alsa-project/alsa-lib` 的 `src/pcm/pcm.c` 和 [`test/pcm.c`](https://github.com/alsa-project/alsa-lib/blob/f84cd4ced7b36fddb8e4ee24404cf7c091d27020/test/pcm.c)。项目 `COPYING` 为 LGPL 2.1 文本，具体源文件的版本选择和例外仍按其文件头核对。PCM（脉冲编码调制）接口把硬件配置、软件启动阈值和流状态分开。

[官方 PCM 文档](https://www.alsa-project.org/alsa-doc/alsa-lib/pcm.html)区分：`-EPIPE` 是播放下溢或采集上溢；`-ESTRPIPE` 与挂起有关；`-ENODEV` 可表示设备被移除；某些回声参考设备在关联播放未启动时可返回 `-ENODATA`。因此不能对所有负返回值无条件调用同一恢复函数，并假设音频仍连续。

规格要固定 period、buffer、`avail_min`、启动阈值、访问布局和时间戳模式。测试短读写、CPU 暂停、系统挂起、USB 拔插和“先开参考、后开播放”。恢复后记录真实丢失样本并通知 AEC、重采样器及追踪状态；重新 `prepare` 不会补回已经丢失的数据。

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

本书另以[可重跑的 C 接口实验](../examples/run_libsamplerate_sro.py)核对一个受限情形。输入是 10 s 的单通道数学合成脉冲：参考时钟 16 kHz，设备时钟在前 5 s 快 100 ppm（16 001.6 样本/s），后 5 s 快 150 ppm（16 002.4 样本/s），脉冲发生于参考时间 1、3、4、6、8、9 s。按两段时钟积分并四舍五入，设备共生成 160 020 帧；从独立时钟算式可得 1～9 s 两个脉冲的未校正相对偏移增量为 16 帧。设备采样率在本实验中是**事先给定的真值**，并非由音频估计。

锁定 `libsamplerate` 0.2.2、提交 `0844c208f683527c08ea8a80acc13b398aa9c8bf`，本机 macOS arm64、Apple clang 21.0.0、CMake 4.3.2，仅在临时目录构建静态库；`SRC_SINC_MEDIUM_QUALITY`、257 帧输入块、64 帧输出缓冲、浮点单通道、状态跨块保留。未校正组的比例始终为 1；校正组在两段分别传入 `16000/16001.6≈0.99990001` 和 `16000/16002.4≈0.99985002`。两组最后一个输入块都设 `end_of_input=1`，随后继续取尽尾部。该实验使用 `SRC_DATA.src_ratio` 的跨调用平滑，不测试 `src_set_ratio` 的突变控制。

| 对照 | 输出总帧数 | 六个脉冲相对参考时间的峰值采样点偏移（帧） | 1～9 s 偏移增量（帧） | 部分消费调用 | 输入取尽后补出帧数 |
|---|---:|---|---:|---:|---:|
| 不校正，比例 1 | 160 020 | 2，5，6，10，15，18 | 16 | 1 875 | 84 |
| 按已知时钟更新比例 | 159 999 | 0，0，0，0，0，0 | 0 | 1 874 | 127 |

表中的“部分消费”由每次 `input_frames_used < input_frames` 计数，两组实际累计消耗均为 160 020 输入帧；输出计数、比例和排空顺序还由逐调用记录独立检查。校正组总帧数比 10 s×16 kHz 少 1 帧，不能因为内部六个标记对齐就写成输出长度严格等于 160 000。此处只观察孤立脉冲峰值位置，没有拟合固定时移或增益；脉冲之间的波形、带外抑制和听感不是表中测量项。

运行命令是 `.venv/bin/python codes/examples/run_libsamplerate_sro.py`。脚本先核对本地上游提交及范围，完成临时构建后输出 JSON；源文件、二进制输出和调用轨迹均不进入发布仓库。对应[独立算式与接口失败用例测试](../../tests/test_codes_libsamplerate_sro.py)检查错误比率、丢失输入、过早结束与遗漏排空。实际设备仍需另测 SRO 估计误差、比例更新伪影、抗混叠、时间戳闭环和长时队列稳定；本实验不能代表这些项目通过。

另一个[估计—补偿—核查小实验](../examples/sro_closed_loop_demo.py)只用 Python 标准库，不需要下载上游源码或连接声卡。它生成 12 s 的解析合成波形与**精确的逐样本合成时间戳**：参考为 2 000 Hz，设备为 2 000.3 Hz（+150 ppm），首样本在参考零点后 2 ms 到达，物理设备索引 12 000 的样本被删去。参考序号 $n$ 的名义时刻为 $n/2000$，未缺样设备序号 $n$ 的时刻为 $0.002+n/2000.3$ s，因此前 2 s 的九个锚点按本书延迟约定给出截距 $-2$ ms、斜率 $150\times10^{-6}/(1+150\times10^{-6})$。示例的标准库直线拟合一阶读数为 149.9775 ppm，按 $s/(1-s)$ 反解为 150 ppm；截距另报初始偏移 2 ms。先从相邻时间戳识别倍数间隔并恢复物理索引，再拟合**缺样之前**的锚点，避免把缺样阶跃误认成稳定 SRO。

校准使用设备索引 0～4 000 的锚点；之后才把后续样本送入保留上个设备样本与输出相位的线性插值器，从参考索引 4 500 开始输出。257 样本输入块下，缺口使参考索引 12 002、12 003 标为无效，没有跨缺口连线。以解析参考波形计算无量纲幅值平方的均方误差（MSE），结果如下；未校正组直接按接收序号与参考序号比较，所以同时包含初始偏移、速率积累与缺样后的序号错位。

| 参考样本区间 | 有效校正样本数 | 未校正 MSE | 校正后 MSE |
|---|---:|---:|---:|
| 5 000～10 999，缺样前 | 6 000 | 0.02869 | 0.000002773 |
| 13 000～21 999，缺样后 | 9 000 | 0.02131 | 0.000002477 |

运行 `python3 codes/examples/sro_closed_loop_demo.py` 即得到参数、估计、缺口和逐段 JSON；[独立单元测试](../../tests/test_codes_sro_closed_loop.py)核对解析斜率、2 ms 截距、单点缺样、跨块相位及 257/509 样本分块一致性。这里是无时间戳噪声、恒定速率且合成波形可解析的教学闭环；MSE 还包括线性插值误差，不可解释为纯时间残差。本实验不验证实际声卡、盲音频估计、抗混叠、动态比例更新或长时队列稳定。上述 libsamplerate 实验验证真实 C 接口的有状态调用，但使用已知比率；两者共同仍不能替代设备验收。

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

第 10 章[功率谱减手算与 E10-13](../../chapters/10_engineering-practice.md#sec-10-3-2)先固定噪声专用帧、逐频点平均噪声功率和相对观测功率的谱地板；[本书实现](../array_tutorial/noise_suppression.py)只承担这一可复算基线。下面的 WebRTC VAD、DeepFilterNet 和模型运行时具有不同的状态与训练假设，不能拿本书谱减的固定噪声估计和单次合成试听推断它们的质量或计算量。谱减仍需在目标输入上单独核验噪声估计失配、音乐噪声、任务损伤和跨块处理；当前 WAV 不含真实语音或设备采集。

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

`chimechallenge/chime-utils` 提供 CHiME-8 DASR 数据准备、清单转换和官方评分入口，代码为 MIT。[`README` 的 Scoring 部分](https://github.com/chimechallenge/chime-utils#scoring)说明 SegLST 中的会话、说话人、起止时间和文字字段，提供 cpWER/tcpWER，并在评分前执行该届文本规范化。MeetEval 的具体算法入口见其 `doc/algorithms.md` 与本书锁定清单。

必须记录届次、任务、数据版本、评测区域、规范化、缺失场景策略和评分器提交。`--ignore-missing` 会改变哪些场景计入结果，因此必须在报告中显式说明，不能用它掩盖未处理数据。许可证允许取评分代码，不代表能够直接再分发各原始语料。

建议制作小型人工 SegLST 夹具：完全正确、交换说话人、漏一段重叠语音、时间戳偏移、重复输出。检查评分变动是否符合所选指标的定义，再运行完整数据。这样能区分“算法增强失败”与“转写输出或评分映射失败”。

## 6. 怎样把项目变成可执行实验

每次只选一条具体链。例如 Linux 会议终端可以从 ALSA/PortAudio 采集、PipeWire 或应用内 AEC、固定波束、单通道 NS、VAD/AGC 与后端开始。若图中同时出现 PipeWire 和应用内 AEC，应先明确谁负责消回声及谁取得播放参考。

实验记录至少保存：源码提交、模型摘要、依赖和编译参数、设备与固件、声学条件、通道与时钟、数据集划分、块长、状态初始化/重置、指标、逐文件结果和资源测量。状态相关的回归要同时覆盖整段、分块、暂停恢复与故障注入；算法单元测试不能代替它们。

块长不同需要适配，但不必等待所有块长的最小公倍数。以 16 kHz 为例，10 ms 采集块是 160 点，15 ms 处理步长是 240 点，32 ms VAD 块是 512 点；三者最小公倍数为 7680 点，即 480 ms。若先攒满这 480 ms 才启动全部模块，会人为增加很大等待。

连续队列可以在各消费者凑齐自身输入时立即调用，并保留剩余样本；VAD 的控制输出按时间戳使用。验收要检查每个消费者收到的样本无重无漏，以及输出决策对应哪个时间区间。这是缓冲调度算例，不表示上述不同硬件实现应直接串接。

本书的[工程练习 E10-07～08](../../chapters/10_engineering-practice.md#sec-10-11)给出字节口径与消费索引。运行 `.venv/bin/python -m codes.examples.exercises_engineering` 后，E10-08 中的 `events` 记录左闭右开的输入索引和回调轮次，`leftovers` 记录余量。两条消费者首次分别在 20、40 ms 得到合法块；这里假设各回调完整交付、计算耗时为零，不把它称为设备延迟实测。

同一入口的 E10-14 把六帧串行计算、在途容量、同时驻留内存和重复模式下的功率估算连起来；[第 10 章完整手算](../../chapters/10_engineering-practice.md#sec-10-4-3)给出每帧完成时间。第 11 章 E11-05～07 则分别以周向阵列的相邻间距筛查、单路会议链的输出/延迟硬约束和跨时钟漂移为候选过滤题。这四题的数字都是本书指定的教学输入，代码通过算术与约束，不代表源码已在某款设备通过资源或声学验收。

新增[状态与时间练习](../examples/tracking_time_exercises.py)把两个容易混淆的工程条件单独复算：E10-15 逐帧记录 VAD 预卷、触发和结束保持，输出片段采样区间 `[160,1120)`，避免触发帧重复；E11-08 从零失败的二项概率推导单侧风险上界，区分“没有观察到失败”和“风险已足够低”。它们与第 9 章 E09-08 的状态时间戳一起构成接口检查，均未使用真实设备或将连续相关帧当作独立试验。

[工程边界练习](../examples/engineering_boundary_exercises.py)给出相应的受控反例：E10-16 在每帧恰好用完周期时应无丢帧，稍微超过周期才触发拒绝或期限违约；E10-17 按实际对应样本计算延迟；E11-09 保留六个配对会话及一项平局，用五个非平局的全部符号排列复算单侧概率。合并 WER 下降幅度与会话胜负方向是不同统计对象，不能因为模型的总错误数较少就省去逐会话记录。

真实回调还应区分采集首样本时间、回调开始时间和播放首样本时间。[PortAudio 的 PaStreamCallbackTimeInfo](https://files.portaudio.com/docs/v19-doxydocs/structPaStreamCallbackTimeInfo.html)把三者分别放在 `inputBufferAdcTime`、`currentTime`、`outputBufferDacTime` 中，单位为秒，使用所属流的时间基准。它们不能未经校准就与另一设备的时钟相减。量化、时间戳与样本消费属于接口条件；更新模型或运行时版本时，也应保留这些对照。该接口文档核实于 2026-09-22。

本目录给出的是实现阅读和实验设计。源码实际获取状态见 `codes/upstream/` 的下载记录；构建、运行及目标硬件验收仍按各实验的真实记录报告。

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

本书于 2026-09-22 实际构建并运行了 I22、I23、I26 的三个固定库。环境为 macOS arm64、Apple clang 21.0.0、CMake 4.3.2；采用 Release 静态库与独立构建目录，没有安装到系统或教学虚拟环境。soxr 关闭 OpenMP 和 SIMD 分支，libsndfile 关闭外部压缩编解码后端；旧构建文件通过命令行 `CMAKE_POLICY_VERSION_MINIMUM=3.5` 兼容，未修改上游源码。三个工作树在构建前后均通过来源、提交与干净状态核查。运行入口为[隔离构建脚本](../examples/run_industrial_interfaces.py)，调用[原创 C 实验程序](../examples/industrial_interfaces.c)；[原始报告](../reports/industrial_interfaces.json)绑定源码提交、实验程序与运行脚本摘要、构建选项和日志摘要。

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

本固定提交的 VR 路径在最后一次 `in=NULL` 调用产出 0 帧后，`soxr_delay` 仍读为 100 个输出样本；固定比率路径在同样排空协议下读为 0。这里同时记录**调用已无输出**与**接口当前读数**，不把 100 解释为仍有 100 帧漏写，也不把它并入端到端设备延迟。细节可由[原始报告](../reports/industrial_interfaces.json)及[实验源码](../examples/industrial_interfaces.c)复核。该探针未测抗混叠抑制、设备时间戳噪声、在线估计或硬实时性。

### I23：libebur128 的响度与真峰值

峰值保护 AGC 限制样本幅度，却不等于按感知响度调节音量。`jiixyj/libebur128` 提供响度和峰值测量，入口为 [`ebur128/ebur128.h`](https://github.com/jiixyj/libebur128/blob/67b33abe1558160ed76ada1322329b0e9e058b02/ebur128/ebur128.h) 与 `ebur128/ebur128.c`。锁定快照的版本宏为 1.2.6，根 [`COPYING`](https://github.com/jiixyj/libebur128/blob/67b33abe1558160ed76ada1322329b0e9e058b02/COPYING)为 MIT；实现和模式应绑定该提交，不因库名含 R128 就宣称所有配置已通过标准认证。

瞬时响度、短期响度和积分响度使用不同统计窗口；响度范围描述随时间变化的离散程度，采样峰值与真峰值则检查幅度。接口中积分响度以 LUFS（Loudness Units relative to Full Scale，相对数字满刻度的响度单位）返回；`ebur128_true_peak` 返回线性幅度，若转成 dBTP（decibels True Peak，真峰值分贝），需对相对满刻度幅度取 $20\log_{10}$。两种结果不能互相替代，也都不是未经声学校准即可得到的 dB SPL。

先用 `ebur128_init` 选择所需测量模式，再连续送入交织音频。`ebur128_add_frames_*` 的数量是每通道时间帧数，而非所有通道的标量数之和。通道角色影响测量：头文件的默认映射按常见节目声道设置，不能把阵列的四路原始麦克风直接当成四个节目声道。单路增强输出可单独测量；比较多个麦克风时应使用各自的单通道测量状态，而不是任意套用环绕声布局。

积分门控会影响哪些时间段进入统计；全静音等无有效能量的情况可能返回负无穷。调用者应同时保留状态、有效时长和失败或无定义原因，不能把负无穷替换为 0 LUFS 后参加平均。真峰值用于估计重建波形的幅度，可能高于已存样本的最大绝对值；采样峰值不越界不能单独证明重建后有足够余量。

建议输入一段足够长、幅度稳定且通过门控的非静音信号，再整体乘以 0.5。在门控保留区域相同的前提下，本书代数预期响度下降 $20\log_{10}2\approx6.0206$ LU，采样峰值和真峰值的线性值均减半。另测全静音、短文件、逐块清状态和错误通道映射，并分别保存结果；这些输入检验测量协议，不构成主观听测。

核心库用 C/CMake 构建，不需要神经模型。实际测试目录由根 `CMakeLists.txt` 指向 `test/`；文件读取示例所需依赖与核心测量分开核对。开启测试、编译成功、运行测试及取得标准测试音频是不同步骤，不因生成了库文件就把全部测试标为通过。

本书实际运行的合成测量采用 48 kHz、10 s、1 kHz 单声道正弦，显式把通道设为 `CENTER`，模式为 `I | TRUE_PEAK`。幅度 0.1 和 0.05 的积分响度分别为 −23.00360 LUFS、−29.02420 LUFS，相差 −6.020599913 LU，与独立代数预期 $20\log_{10}(0.5)$ 一致。两档信号均远高于绝对门限，保持相同有效门控区域；不能据此认为任意含静音或近门限节目减半都会得到相同的积分响度变化。

保持状态、每次送入 127 帧，与整段送入的积分响度差为 0 LU；采样峰值分别为 0.1、0.05，估计真峰值约为 0.1000000015、0.0500000007，线性比均为 0.5。全静音返回负无穷，报告将它记录为 `null`，同时保留 `negative_infinity_no_gated_energy` 原因；静音的两个峰值均为 0。原始值与构建条件见同一份[工业接口报告](../reports/industrial_interfaces.json)。这组结果验证幅度缩放、连续状态和失败值处理，不构成 EBU 标准测试集认证、主观听测或任意文件格式的验收。

#### 已运行：采样点之间出现峰值的构造信号

前述 1 kHz 正弦的 48 kHz 网格很密，不能清楚展示“样本峰低于重建后峰”的情形。新增独立输入为 48 kHz、10 s、12 kHz、初相 $\pi/4$、幅度 0.95 的单声道正弦；开头与结尾各用 100 ms 升降余弦渐变，防止硬起振造成插值滤波器的边界过冲。中间稳定区的样本相位每次增加 $\pi/2$，各样本绝对值均为 $0.95/\sqrt2=0.6717514421$；连续解析正弦则可在样本间达到 0.95。两者分别约为 $-3.45583$ dBFS 与 $-0.44553$ dBFS，解析间隔为 $3.01030$ dB。渐变只降低边界处幅度，不改变稳定区这两个独立手算值。

锁定 libebur128 1.2.6 的[`ebur128.c` 插值器](https://github.com/jiixyj/libebur128/blob/67b33abe1558160ed76ada1322329b0e9e058b02/ebur128/ebur128.c#L119-L174)在 48 kHz 使用 49 抽头、4 倍多相 FIR；这是一种有限长**估计器**，并不对任何带限信号承诺精确重建解析最大值。实际 C 接口返回样本峰 $0.671751442206$（$-3.45583$ dBFS）、真峰估计 $0.946371197701$（$-0.47877$ dBTP），高出样本峰约 $2.97706$ dB。整段送入与保持同一状态、每次 127 帧送入的真峰读数逐位相同。样本峰与手算的微小差来自浮点正弦及内部类型转换；真峰与解析 0.95 相差约 0.00363，不能把其中一个数字代替另一个。

不加渐变而从第一点突起的同频正弦不是同一个连续波形边界。按上述固定 FIR 系数独立复算，起始瞬态可达约 0.96139，甚至超过平台解析幅度 0.95；所以本实验以平滑边界隔离采样间峰值结论。这个测试只验证固定软件实现与解析反例，不认证标准真峰测试集，也不等于聆听质量。报告的[源码摘要和原始测量值](../reports/industrial_interfaces.json)可与[实验程序](../examples/industrial_interfaces.c)核对。

### I24：pystoi 的 STOI、ESTOI 与无效评分

第 10 章列出的短时客观可懂度 STOI 可以通过 `mpariente/pystoi` 的 Python 实现研究。它支持普通 STOI 和扩展 STOI（Extended Short-Time Objective Intelligibility，ESTOI），入口均在 [`pystoi/stoi.py`](https://github.com/mpariente/pystoi/blob/74872b000753a7a42ff51aa0868af8c82c7f9053/pystoi/stoi.py)。这是软件作者维护的 Python 实现；[README](https://github.com/mpariente/pystoi/blob/74872b000753a7a42ff51aa0868af8c82c7f9053/README.md)另说明 MATLAB 测试来源于 Cees Taal 的代码，不应把两种实现的作者与环境混同。

该提交 `setup.py` 标为 0.4.1，Python 核心按根 `LICENSE` 的 MIT 条款提供，运行依赖 NumPy 和 SciPy。Octave/MATLAB 对照测试有额外依赖和测试文件，不是安装 Python 核心后自动完成的验证。本书源码获取范围、测试资产和实际运行环境分别登记。

输入是一维、等长度的干净参考与处理后语音，二者采样率和时间对齐口径必须一致。实现内部先转到 10 kHz，再按参考进行静音帧筛除、短时分析和频带处理。`extended=False` 与 `extended=True` 使用不同的归一化与相关性计算步骤，结果应分别命名，不能在同一列中混写成 STOI。

需要额外检查返回值是否确实来自有效评分。该提交在去除静音后不足 30 个 STFT 帧时发出 `RuntimeWarning`，随后返回 `1e-5`。这个数是实现的失败返回值，不是测得“可懂度极低”的有效结果。评分程序应捕获并记录警告、文件 ID、长度与原因，在总文件数中保留这次失败；不能通过丢弃警告或只检查结果有限就让它进入平均值。

建议先准备授权清楚且有干净参考的自然语音，分别比较自身、已知加噪版本、静音输出与错位版本，并记录普通/扩展模式、SciPy 版本和对齐方式。另用短输入验证无效评分是否被识别。自身对照只是接口检查，不能代替与原作者实现的独立数值对照；STOI 也不是逐词识别正确率，合成正弦不能用来报告自然语音可懂度。

### I25：ViSQOL 的模式、模型与构建依赖

Google 的 ViSQOL（Virtual Speech Quality Objective Listener）是全参考音频质量估计器：先比较干净参考与退化信号的时频结构，再映射为客观听音质量预测分（Mean Opinion Score—Listening Quality Objective，MOS-LQO）。它需要参考，和 I19 的无参考质量预测解决不同的输入条件；它不是 POLQA 的实现，也不替代受试者 MOS。[固定 README 的 Guidelines、License 与 FAQ](https://github.com/google/visqol/blob/38d0b0163e441047d4429bf07ad09e5b9031d02c/README.md)

该实现的 audio 模式使用 48 kHz 输入，speech 模式使用 16 kHz 输入。多通道会下混为单声道再比较，因此分数不能证明阵列方向、双耳线索或空间声场保持。speech 模式还需记录语音活动筛选、映射模型，以及是否选择 scaled/unscaled 映射；模式不同的分数不能直接混排。

质量映射依赖模型。源码目录 `src/`、协议目录 `src/proto/` 和 Python 接口足以研究调用过程，却不足以证明已经具备评分条件。`model/` 中的 TFLite 文件与文本 SVR 参数都属于模型资产，不能只因扩展名为 `.txt` 就将其当作没有模型内容的普通说明。若获取时排除了模型和 `testdata/`，报告中须明确写源码范围，之后按资产来源和许可补齐，不能运行到缺文件再把它称为算法失败。

构建从根 `WORKSPACE` 和 `BUILD` 阅读。该提交的 [WORKSPACE](https://github.com/google/visqol/blob/38d0b0163e441047d4429bf07ad09e5b9031d02c/WORKSPACE)会引入 TensorFlow 2.11.0 对应提交及 protobuf、pybind11、Abseil、LIBSVM、Armadillo、PFFFT 等依赖。README 的 Bazel 命令是构建入口，不是“只需一个小型 Python 包”的保证；包内源码量、依赖下载量、构建缓存与运行内存需要分开记录。

根 `LICENSE` 对该项目代码采用 Apache-2.0，不能以此替全部外部依赖和录音授予许可。建议先核对模型摘要、模式和依赖，再用同一批参考—退化文件对生成逐文件及逐片段结果。参考错配、静音过多、采样率错误和短文件应有明确失败或排除记录，不悄悄变成高低分。

官方 FAQ 指出，它原本针对编解码和网络退化，移用于降噪、前处理及生成类失真时表现并不一致。因此正式设备结论仍需目标数据、主观听测和识别指标共同验证；不能用一个质量预测分证明内容未改变。

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

独立验证没有再次调用 libsndfile：运行脚本使用 Python 标准库 `wave` 与小端整数解码核对 WAV 编码、7 帧长度、双通道次序和全部 14 个码字。该检查与[独立失败用例测试](../../tests/test_codes_industrial_interfaces.py)共同覆盖帧/项误用、分块丢失、静音误记为零和报告来源失配。实验 WAV 与完整日志留在忽略的独立构建目录，摘要保存在[原始报告](../reports/industrial_interfaces.json)中；本书没有据此声称其他文件编码、削波策略或设备采集也已验证。

### I27：lib_xcore_math 的块浮点与参考内核

XMOS `lib_voice` 的底层依赖不是任意最新版数学库。锁定的 [`lib_voice/lib_build_info.cmake`](https://github.com/xmos/lib_voice/blob/c9f1a9bf95cd88c7950adf4bf631c217f900ad25/lib_voice/lib_build_info.cmake)明确要求 `lib_xcore_math(v3.0.0)`；本书按此固定到 `16130be45c4002a1f875a4b06ff68d2065cd8c69`。该库提供向量运算、块浮点、快速傅里叶变换（FFT）、离散余弦变换（Discrete Cosine Transform，DCT）与滤波，补充 I15 的 Q15 内核对照，不是一套独立的完整语音前端。

块浮点把一组整数尾数与一个共享指数一起保存。示意地，尾数 `[-16384,8192]` 与指数 −15 表示实值 `[-0.5,0.25]`。相同整数数组配不同指数会代表不同幅度，因此对照时不能只比较尾数字节。

位余量（headroom）描述还能左移多少位而不溢出，用来选择运算缩放；为了避免溢出而右移又会损失低位精度。这与“全部模块固定使用同一 Q15 小数位数”不同。

源码阅读可从 `lib_xcore_math/api/` 进入 `src/bfp/`、`src/fft/` 与 `src/filter/`，再比较 `src/arch/ref/` 和目标架构实现。[构建文件](https://github.com/xmos/lib_xcore_math/blob/16130be45c4002a1f875a4b06ff68d2065cd8c69/lib_xcore_math/lib_build_info.cmake)区分参考 C、XS3 汇编及其他架构路径；FFT 查找表也有使用内置表与重新生成的选项。必须保存指数、长度、表配置、架构和编译选项，不能只记录函数名。

建议先用上述两点向量及小幅脉冲，按指数还原实值后与独立浮点计算比较；再加入接近满幅、余数长度、大小量混合与 FFT 往返。测试应同时观察输出指数、舍入误差、饱和和状态，不把“多数样本相近”当成全部边界正确。参考 C 路径通过后，仍需在目标板运行相同输入并另测周期和内存；主机仿真耗时不能推算目标 VPU 的实时性。

[README](https://github.com/xmos/lib_xcore_math/blob/16130be45c4002a1f875a4b06ff68d2065cd8c69/README.rst)指定 XTC Tools 15.3.1，并说明原生构建的 VPU 仿真范围限制。[LICENSE.rst](https://github.com/xmos/lib_xcore_math/blob/16130be45c4002a1f875a4b06ff68d2065cd8c69/LICENSE.rst)为 XMOS Public Licence v1，商业硬件和特殊用途条件不能省略。取得这个依赖也不表示 `lib_voice` 的工具链、模型生成依赖和目标硬件已经全部齐备。


### I28：FastEnhancer 的显式流式状态与单通道降噪

FastEnhancer 用于单通道神经语音增强。[官方固定 README](https://github.com/aask1357/fastenhancer/blob/f85223bd546b27f39dc0744e0310dcd246f750a4/README.md)将项目论文标为 ICASSP 2026 已接收，并明确其模型按噪声抑制训练；本文只核对固定代码的接口，不转录跨数据集性能或实时性排名。源码提交为 `f85223bd546b27f39dc0744e0310dcd246f750a4`，已按[MIT 许可](https://github.com/aask1357/fastenhancer/blob/f85223bd546b27f39dc0744e0310dcd246f750a4/LICENSE)获取至 `codes/upstream/_downloads/fastenhancer/`，核对日期为 2026-09-26。

**模型算什么。** 默认实现 `models/fastenhancer/default/model.py` 中，`RNNFormerBlock` 组合时间递归状态与频率注意力；`ONNXModel.forward` 对实部/虚部表示的频谱先做幅度压缩，再预测复数掩码并执行复数乘法，最后还原压缩。它返回增强频谱和更新后的模型缓存。目录中还有 `noncausal` 等变体，不能只看到项目名称含 streaming 就把每种配置都当成相同因果结构。[固定模型源码](https://github.com/aask1357/fastenhancer/blob/f85223bd546b27f39dc0744e0310dcd246f750a4/models/fastenhancer/default/model.py)

**每块带什么状态。** `scripts/export_onnx.py::Model.forward` 的接口包含 `wav_in`、STFT 缓存、逆 STFT 缓存及模型缓存；返回本块波形与全部新缓存。实际推理示例 `scripts/test_onnx.py` 从会话输入名寻找 `cache_in_*`，会话起点初始化为零，每次调用后把输出缓存传给下一块。逐块清零会破坏连续流语义；换会话时忘记清零又会把旧状态带入新会话。[导出包装器](https://github.com/aask1357/fastenhancer/blob/f85223bd546b27f39dc0744e0310dcd246f750a4/scripts/export_onnx.py)、[推理循环](https://github.com/aask1357/fastenhancer/blob/f85223bd546b27f39dc0744e0310dcd246f750a4/scripts/test_onnx.py)

推理脚本默认 16 kHz、`n_fft=512`、`hop_size=256`，后者对应每步 16 ms；实际运行须与导出模型配置相符。脚本在结尾补零以排出状态，拼接输出后裁去 `n_fft-hop_size=256` 点，再取原输入长度。这个 16 ms 裁剪是默认配置的离线对齐操作，不能代替采集、凑块、模型计算、排队和播放共同构成的端到端延迟。其 RTF 计时包围 Python 推理循环并含循环开销，不能把项目表中的数字移植到另一机器。

**怎样设计最小核查。** 取得明确来源的同版本权重后，先固定单通道 WAV、模型摘要、块长、ONNX Runtime 版本、线程数和执行后端。对照官方导出器的 `--test-streaming` 路径，检查整段与保留缓存的逐块输出是否在相同对齐、尾部排空和裁剪下相符；再故意逐块清零，观察边界差异。容差须按数据类型和运行时预先确定，不能假设不同内核必然逐位一致。最后分别测启动、稳态、停流排空与新会话重置。

本次只读取并保存了源码，没有取得权重、导出 ONNX 或执行推理，所以没有本书测得的质量分、RTF 或音频输出。该接口没有播放参考，也没有 WPE 的延迟多通道预测器；不能将它计作 AEC 或 WPE 实现。将其接在波束输出后作单通道 NS 是可研究的系统组合，仍须独立检查前端输出分布、目标失真和端到端任务指标。


<a id="i29stkdelayl"></a>

### I29：STK DelayL 的分数延迟、幅度误差与跨块状态

STK（The Synthesis ToolKit in C++）提供音频合成与处理组件。本节只研究其中的固定分数延迟器，不将工具箱整体当作麦阵列产品。[维护者源码](https://github.com/thestk/stk/tree/6aacd357d76250bb7da2b1ddf675651828784bbc)固定为 `6aacd357d76250bb7da2b1ddf675651828784bbc`；2026-09-26核对[MIT-STK许可](https://github.com/thestk/stk/blob/6aacd357d76250bb7da2b1ddf675651828784bbc/LICENSE)，保留作者与许可。下载选择include/src/README/LICENSE，没有取得rawwaves素材或打开音频设备。

**入口与状态。** `src/DelayL.cpp`分配延迟线，`include/DelayL.h`中的`setDelay`按当前写指针确定读指针和分数权重。`tick`先写当前样本，再读插值结果，因此可以表示零延迟；每次调用推进读写指针。`nextOut`在下一次tick前缓存结果，不能把反复调用它当成推进一个采样。`include/Filter.h::clear`清空历史；逐音频块调用clear会破坏跨块连续性。改变最大容量可能重新分配存储，应在初始化阶段完成；本节未测动态延迟调制。

| 阅读位置 | 要核对的变量/步骤 | 独立判据 |
|---|---|---|
| `DelayL.h::setDelay` | 读指针追随写指针，单位为采样 | 延迟0/.5/1分别对应当前值、相邻平均、前一个值 |
| `DelayL.h::tick/nextOut` | 先写后读、插值权重、指针回绕 | 逐样本与独立二抽头FIR一致 |
| `Filter.h::clear` | 历史状态归零，不是正常块边界操作 | 连续状态分37点调用应相同；逐块clear作为反例 |

**本机已执行。** [原创C++探针](../examples/stk_delay_probe.cpp)与[运行入口](../examples/run_stk_delay_probe.py)只编译锁定版`Stk.cpp`和`DelayL.cpp`，临时生成物不写回上游。输入为512点、16 kHz、500/6000 Hz双音，两个分量幅度均0.18；无淡入淡出、初始历史为零。它与2 s试听素材使用相同频率，但长度和边界不同，不能借用试听文件代替此接口测试输入。

在Darwin arm64、Apple clang 21.0.0、C++17/O2、STK默认double下，半采样输出相对独立FIR的最大绝对误差为0；零延迟、一采样延迟的已知边界误差也为0。把同一标量tick调用分成37点组且保留对象状态，输出差为0；每组前清空对象，最大绝对差为0.1519102855。结果、输入条件、编译器和实际源码摘要保存于[STK接口报告](../reports/stk_delay.json)。这里的0只表示这组输入在该编译环境下的计算结果，不是任意平台的零误差保证。

复跑命令为 `.venv/bin/python -m codes.examples.run_stk_delay_probe --report tmp/stk-delay-rerun.json`，需已取得锁定源码与C++编译器，不联网、不安装依赖。该检查只覆盖标量tick及固定延迟；没有测试StkFrames多通道接口、设备时间戳、实时期限或语音质量。

幅度和延迟需分别验证。一次半采样线性插值对6000 Hz的幅度比约0.382683，两次则约0.146447，即使累计相位对应一采样，仍不等同纯一采样移位。完整本书推导、PCM统计区间与图41见[附录B E13-02](../../chapters/13_appendix-guide.md#e13-02)。工业选型须另查目标频带误差、可变延迟的状态转换和存储上限；本节不把一个组件检查外推为完整重采样器验收。
