# 第三方实现与工业生态索引

以下条目均指向维护者的官方仓库，并固定到 2026-09-22 核实的完整提交。固定提交用于定位本轮调查所见
代码，不表示本书替这些项目承诺兼容性、安全性或长期维护。许可证列只概括**代码仓库**；模型权重、
训练集、测试集、示例录音和商标需分别检查。

| 项目 | 本书中的用途 | 核实提交 | 代码许可证 | 本仓库处理方式 |
|---|---|---|---|---|
| [pyroomacoustics](https://github.com/LCAV/pyroomacoustics) | 房间仿真、STFT、DOA、波束和 BSS 参考 | 0.10.0：`0dd39f2614b7fc44b2cc63dbe7d60f4641068890` | MIT | 可按需获取；与正文复现包一致 |
| [ODAS](https://github.com/introlab/odas) | C 语言实时定位、追踪、分离与后滤波链 | `bcb845434495e293df3d48f1203b7a86e1852449` | MIT | 可按需获取；设备配置与阵列标定另做 |
| [nara_wpe](https://github.com/fgnt/nara_wpe) | 离线、块在线和逐帧在线 WPE 参考 | 0.0.11：`a166779cca2088817e330481bd20af1a2c598555` | MIT | 可按需获取；与正文示例包一致 |
| [Asteroid](https://github.com/asteroid-team/asteroid) | 深度语音分离训练与评测框架 | `fce87469132760fbab41c20616ea0f0e079aad38` | MIT | 大型框架，只索引 |
| [SpeechBrain](https://github.com/speechbrain/speechbrain) | 语音增强、分离和训练配方参考 | `89ead74d163463d30c62329a09cfdb4c54f5abc1` | Apache-2.0 | 大型框架，只索引 |
| [ESPnet](https://github.com/espnet/espnet) | WPE、分离、增强与端到端语音系统参考 | `be79590bb2ff26ffb01bc825c5f68cb9418b7f0d` | Apache-2.0 | 大型框架，只索引 |
| [WebRTC](https://webrtc.googlesource.com/src) | AEC3、噪声抑制、增益控制和实时音频链 | AEC3 正文：`0467d2b91cc20b9b001c2bbb73d43ea6b2491f3e` | BSD-3-Clause | 大型多仓构建，只索引；当前主线快照另记入锁文件说明 |
| [SpeexDSP](https://gitlab.xiph.org/xiph/speexdsp) | 经典 AEC、预处理和重采样工程参考 | `8e29a256ef0235ebbe7fcb8417b5ac7731eb8307` | BSD-3-Clause | 可按需获取 |
| [RNNoise](https://gitlab.xiph.org/xiph/rnnoise) | 轻量循环神经网络降噪参考 | `70f1d256acd4b34a572f999a05c87bf00b67730d` | BSD-3-Clause | 可按需获取；训练数据与模型文件另核对 |
| [PortAudio](https://github.com/PortAudio/portaudio) | 跨平台音频设备与回调线程接口 | 19.7.0：`3f7bee79a65327d2e0965e8a74299723ed6f072d` | MIT | 可按需获取；遵守回调实时约束 |
| [CMSIS-DSP](https://github.com/ARM-software/CMSIS-DSP) | Arm MCU/DSP 上的滤波、FFT 和矩阵内核 | `83a2d7bc98c81b4bbe4a6f48b1f2ecf179868a0b` | Apache-2.0 | 可按需获取；目标核与编译选项另验收 |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | 边缘推理、线程和移动端部署参考 | `48795e0281bcaa6b3b63b28af5e078b4c41e995f` | MIT | 大型项目，只索引；模型算子和许可证另核对 |
| [icoDOA](https://github.com/DavidDiazGuerra/icoDOA) | 二十面体卷积网络 DOA 论文复现 | `04d1a89594c78ae3cf42f07d94c3737bdc1f7c82` | AGPL-3.0 | 只索引；历史 Python/PyTorch/gpuRIR/icoCNN 环境与许可义务另核对 |
| [torchaudio](https://github.com/pytorch/audio) | 官方 SoudenMVDR 教程和音频变换参考 | `b85c99ccac635a06b1afaf5284bf4c1a00c1f9b5` | BSD-2-Clause | 只索引；正文链接为 main/nightly，项目已进入维护阶段，不当作稳定 API 承诺 |
| [piva](https://github.com/fakufaku/piva) | AuxIVA、OverIVA、FIVE 等盲源分离 | `7fa273e9aa597aba57067aef2e7b6c999dadef26` | GPL-3.0 | 只索引；源码/二进制分发义务和原生依赖另核对 |
| [MeetEval](https://github.com/fgnt/meeteval) | cpWER、ORC-WER、MIMO-WER、DI-cpWER 会议评分 | `6e3dc81284f2d6928f7ef9e620fd3b6906daa429` | MIT | 只索引；转写和挑战数据许可另核对 |
| [FilterPy](https://github.com/rlabbe/filterpy) | Kalman、EKF、UKF、粒子滤波参考 | `3b51149ebcff0401ff1e10bf08ffca7b6bbc4a33` | MIT | 只索引；与本书角度环绕接口不同 |
| [Stone Soup](https://github.com/dstl/Stone-Soup) | 状态估计和多目标追踪框架 | `8d1edeb07ef8505ed065cbef435cfb5e517d9bdc` | MIT | 只索引；外链示例数据另核对 |
| [S4M](https://github.com/JusperLee/S4M) | 状态空间语音分离论文代码 | `4990b3fe9d7391e59d652c5a7d2803d1354fd0ae` | MIT | 只索引；没有完整训练入口、权重或数据 |
| [AudioSep](https://github.com/audio-agi/audiosep) | 自然语言查询的通用声音分离 | `944583f18b84589dc965de3ad77525c945334252` | MIT | 只索引；主权重未见独立再分发许可，评测媒体另核对 |
| [SGMSE+](https://github.com/sp-uhh/sgmse) | 基于分数/扩散的增强与去混响 | `1961cf4483e37df1bb92ccf0eb8b28bf6f44cb0e` | MIT | 只索引；托管权重和各训练集另核对 |
| [StoRM](https://github.com/sp-uhh/storm) | 回归加扩散的增强与去混响 | `257e9636a7251ca40aa200753d5c0fe918e31879` | MIT | 只索引；托管权重和各训练集另核对 |
| [ArrayDPS](https://github.com/ArrayDPS/ArrayDPS) | 扩散先验驱动的无监督多通道盲分离 | `750ac2b7c75458f4ca5bad203dafda528f575e55` | MIT | 只索引；checkpoint 许可未单列，WSJ 原始语料权利不随代码转移 |

## 怎么选参考实现

先确定要复现的层级。只核对式子时，优先运行本仓库的小规模 NumPy 基线；研究论文系统时，选择论文
作者或项目维护者的实现，并固定提交、配置、模型和数据划分；设备实时链路则要从音频 I/O、时钟、
回调线程和遥测开始，不能先在离线脚本上得到一个实时系数就结束。

PortAudio 的官方回调接口明确要求避免内存分配、锁、文件和网络等可能无界阻塞的操作。把 NumPy 教学
函数直接放入音频回调，通常会违反这一约束。可行结构是：回调只搬运固定大小缓冲和记录轻量时间戳，
处理线程消费缓冲；队列积压、欠载、超期和重同步都进入遥测。

WebRTC AEC3 是带延迟估计、双讲/近端检测、状态控制、线性与非线性处理的系统，不等于把 NLMS 换成
更长滤波器。RNNoise 是降噪参考，不替代播放参考驱动的 AEC。ODAS 提供完整空间音频链路，但阵列坐标、
设备通道顺序和现场标定仍决定结果是否可信。

## 许可证和分发检查

1. 先读目标提交内的 `LICENSE`、`COPYING`、`NOTICE`，再读依赖和子模块的许可。
2. 代码、权重、数据和示例媒体分别登记；不要从代码许可证推断权重或数据的权限。
3. 若要复制源码进本仓库，保留原始许可和版权声明，记录修改；当前仓库自身还没有明确根许可证，
   因而本书默认只提供链接和按需获取工具。
4. AGPL、非商业、研究限定、无许可证或来源不唯一的项目只作索引，除非维护者先确定兼容的分发方案。
5. 上线前重新核实版本和许可证；本表的日期不是永久结论。
