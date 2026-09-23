# 第三方实现与工业生态索引

原有索引核实于 2026-09-22，新增 RLS/Kalman AEC 来源核实于 2026-09-23。此索引包含 73 个上游项目；已在 `codes/upstream/_downloads/` 取得 63 个独立源码工作区；本轮离线核验 62 个通过、AEC Challenge 的 5 个真实录音有本地变动而未计通过，另 10 项仅登记来源。pystoi 是软件作者维护的 Python 实现，不称为原论文作者的官方 Python 程序。获取状态与完整提交见 [SOURCE_STATUS.json](SOURCE_STATUS.json) 和 [SOURCES.lock.json](SOURCES.lock.json)。状态报告由获取工具离线生成；不能用源码获取结果证明新增项目已运行。

“已取得”只说明来源、提交、工作区状态和指定入口符合清单，不表示已经安装依赖、编译、运行训练、取得权重、完成声学测试或取得产品使用资格。每项的完整入口和限制保存在锁定清单；逐算法解释、最小实验和失效条件见[研究手册](research/README.md)。

## 官方源码与用途

`examples/compare_online_wpe_reference.py` 中的 `NumpyOnlineWPE011` 是对 nara-wpe 0.0.11
`OnlineWPE` 和正数求逆保护行为的精简适配，保留缓冲顺序及更新口径，用于版本对照而非独立推导证明。
该改编范围保留 Communications Engineering Group, Paderborn University 的 2018 年版权与
[完整 MIT 声明](licenses/nara_wpe_MIT.txt)；脚本注明来源、改编内容和固定版本。
这份上游许可不等于为本仓库其他原创文件选择统一许可证。GPL 空间算法仍只在独立下载目录中调用，
没有复制进教学包。

SMP-PHAT 的本地实验额外使用 FFTW 3.3.10 单精度静态库。它是构建依赖，不另计为一种空间算法，
也不计入上面的 73 项 Git 源码索引。官方归档为
[`fftw-3.3.10.tar.gz`](https://fftw.org/pub/fftw/fftw-3.3.10.tar.gz)，SHA-256 为
`56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467`。
归档的 `COPYRIGHT` 与 `kernel/alloc.c` 声明 GPL-2.0-or-later；`api/fftw3.h` 单独采用 BSD 两条款文本，
不能把头文件的许可推广到整库。源码、许可文件与编译产物只保存在本地隔离目录，不随本书提交分发；
运行报告另记实际链接库的版本与摘要。

项目链接定位官方仓库的已核实版本。表中的许可证仅概括相应代码范围；依赖、模型、录音和数据集分别检查。

| 项目与固定版本 | 对应算法或工程功能 | 代码许可摘要 | 本地获取范围 |
|---|---|---|---|
| [pyroomacoustics](https://github.com/LCAV/pyroomacoustics/tree/0dd39f2614b7fc44b2cc63dbe7d60f4641068890) | 房间仿真、STFT、DOA、波束、盲分离及通用 RLS/BlockRLS 自适应滤波；后者不是完整 AEC 链路 | MIT | 已取得独立源码 |
| [odas](https://github.com/introlab/odas/tree/bcb845434495e293df3d48f1203b7a86e1852449) | 定位、追踪、分离与后滤波的实时 C 链路 | MIT | 已取得独立源码 |
| [nara_wpe](https://github.com/fgnt/nara_wpe/tree/a166779cca2088817e330481bd20af1a2c598555) | 离线、块在线和逐帧在线 WPE | MIT | 已取得独立源码 |
| [asteroid](https://github.com/asteroid-team/asteroid/tree/fce87469132760fbab41c20616ea0f0e079aad38) | Conv-TasNet、DPRNN 与训练配方 | MIT | 已取得独立源码 |
| [speechbrain](https://github.com/speechbrain/speechbrain/tree/89ead74d163463d30c62329a09cfdb4c54f5abc1) | SepFormer、增强与分离训练配方 | Apache-2.0 | 已取得独立源码 |
| [espnet](https://github.com/espnet/espnet/tree/be79590bb2ff26ffb01bc825c5f68cb9418b7f0d) | WPE、波束、TF-GridNet 与语音系统 | Apache-2.0 | 已取得独立源码 |
| [webrtc](https://webrtc.googlesource.com/src/+/0467d2b91cc20b9b001c2bbb73d43ea6b2491f3e) | AEC3、NS、VAD、AGC2 及音频处理状态 | BSD-3-Clause | 已取得独立源码 |
| [speexdsp](https://gitlab.xiph.org/xiph/speexdsp/-/tree/8e29a256ef0235ebbe7fcb8417b5ac7731eb8307) | MDF 回声消除、预处理及重采样 | BSD-3-Clause | 已取得独立源码 |
| [rnnoise](https://gitlab.xiph.org/xiph/rnnoise/-/tree/70f1d256acd4b34a572f999a05c87bf00b67730d) | 循环网络降噪 | BSD-3-Clause | 已取得独立源码 |
| [portaudio](https://github.com/PortAudio/portaudio/tree/3f7bee79a65327d2e0965e8a74299723ed6f072d) | 设备输入输出、回调与流状态 | MIT | 已取得独立源码 |
| [cmsis_dsp](https://github.com/ARM-software/CMSIS-DSP/tree/83a2d7bc98c81b4bbe4a6f48b1f2ecf179868a0b) | FFT、滤波与矩阵运算内核 | Apache-2.0 | 已取得独立源码 |
| [onnxruntime](https://github.com/microsoft/onnxruntime/tree/48795e0281bcaa6b3b63b28af5e078b4c41e995f) | 算子、执行提供程序与线程调度 | MIT | 已取得独立源码 |
| [icodoa](https://github.com/DavidDiazGuerra/icoDOA/tree/04d1a89594c78ae3cf42f07d94c3737bdc1f7c82) | 二十面体卷积定位及追踪 | AGPL-3.0 | 已取得独立源码 |
| [torchaudio](https://github.com/pytorch/audio/tree/b85c99ccac635a06b1afaf5284bf4c1a00c1f9b5) | SoudenMVDR 教程与多通道接口 | BSD-2-Clause | 已取得独立源码 |
| [piva](https://github.com/fakufaku/piva/tree/7fa273e9aa597aba57067aef2e7b6c999dadef26) | AuxIVA、OverIVA、FIVE 与更新规则 | GPL-3.0 | 已取得独立源码 |
| [meeteval](https://github.com/fgnt/meeteval/tree/6e3dc81284f2d6928f7ef9e620fd3b6906daa429) | 说话人、排列及时间约束的会议评分 | MIT | 已取得独立源码 |
| [filterpy](https://github.com/rlabbe/filterpy/tree/3b51149ebcff0401ff1e10bf08ffca7b6bbc4a33) | KF、EKF、UKF、IMM 与重采样 | MIT | 已取得独立源码 |
| [stonesoup](https://github.com/dstl/Stone-Soup/tree/8d1edeb07ef8505ed065cbef435cfb5e517d9bdc) | 多目标关联、点过程与 OSPA | MIT | 已取得独立源码 |
| [s4m](https://github.com/JusperLee/S4M/tree/4990b3fe9d7391e59d652c5a7d2803d1354fd0ae) | 状态空间分离模型；不是完整训练配方 | MIT | 已取得独立源码 |
| [audiosep](https://github.com/audio-agi/audiosep/tree/944583f18b84589dc965de3ad77525c945334252) | 文本查询条件的声音分离 | MIT | 已取得独立源码 |
| [sgmse](https://github.com/sp-uhh/sgmse/tree/1961cf4483e37df1bb92ccf0eb8b28bf6f44cb0e) | 分数模型语音增强与去混响 | MIT | 已取得独立源码 |
| [storm](https://github.com/sp-uhh/storm/tree/257e9636a7251ca40aa200753d5c0fe918e31879) | 回归与扩散的两阶段增强 | MIT | 已取得独立源码 |
| [arraydps](https://github.com/ArrayDPS/ArrayDPS/tree/750ac2b7c75458f4ca5bad203dafda528f575e55) | 扩散先验多通道盲分离 | MIT | 已取得独立源码 |
| [doatools](https://github.com/morriswmz/doatools.py/tree/9469db201e0418aef6b97583ef54b6fec2769502) | 源数估计、root-MUSIC、稀疏定位与下界 | MIT | 已取得独立源码 |
| [sound-field-analysis](https://github.com/AppliedAcousticsChalmers/sound_field_analysis-py/tree/4b03ee123d98370c55f744c4f8d7c955fbc099f1) | 球谐变换、径向滤波及球阵声场 | MIT | 已取得独立源码 |
| [spherical-array-processing](https://github.com/polarch/Spherical-Array-Processing/tree/f192aac652b023ee4ab8673adce20ec13bf5450c) | 球谐编码、SH-MVDR/LCMV/MUSIC/ESPRIT | BSD-3-Clause | 已取得独立源码 |
| [spatial-audio-framework](https://github.com/leomccormack/Spatial_Audio_Framework/tree/18fd5aba46e20787b51f28f7197a68506c965c07) | C/C++ 球阵处理、功率图与可选追踪 | ISC core; GPL-2.0 optional modules | 已取得独立源码 |
| [frida-original](https://github.com/LCAV/FRIDA/tree/ff5d51e498805b862c342dd216ccfffb22444b7f) | FRIDA 原论文仿真和录音实验 | MIT | 已取得独立源码 |
| [acoular](https://github.com/acoular/acoular/tree/13d3d7df74ac1a8135c7ec71da098cbbc03d8652) | DAMAS、CLEAN-SC、CMF 与移动声源成像 | BSD-3-Clause | 已取得独立源码 |
| [lib-voice](https://github.com/xmos/lib_voice/tree/c9f1a9bf95cd88c7950adf4bf631c217f900ad25) | XMOS AEC、IC、NS、AGC 语音前端 | XMOS Public Licence v1 | 已取得独立源码 |
| [sof](https://github.com/thesofproject/sof/tree/b6c6a05d52536313fe8e8752b1c4e069b1cc4002) | DSP 固件、拓扑与音频缓冲 | BSD-3-Clause and per-file licenses | 已取得独立源码 |
| [alsa-lib](https://github.com/alsa-project/alsa-lib/tree/f84cd4ced7b36fddb8e4ee24404cf7c091d27020) | PCM 设备、采样格式及环形缓冲 | LGPL-2.1; see individual file notices | 已取得独立源码 |
| [pipewire](https://github.com/PipeWire/pipewire/tree/62316ae8cf659ca8e5f1ee866ae743d2438d47a4) | 图调度、时钟域及低延迟音频 | MIT; plugins and dependencies separately | 已取得独立源码 |
| [deepfilternet](https://github.com/Rikorose/DeepFilterNet/tree/d375b2d8309e0935d165700c91da9de862a99c31) | 深度滤波、流式状态与 LADSPA | Apache-2.0 OR MIT | 已取得独立源码 |
| [silero-vad](https://github.com/snakers4/silero-vad/tree/60b7ffa243625ebdc1070275a29f18c87843786a) | 神经 VAD 与固定块状态 | MIT | 已取得独立源码 |
| [libsamplerate](https://github.com/libsndfile/libsamplerate/tree/0844c208f683527c08ea8a80acc13b398aa9c8bf) | 有状态采样率转换 | BSD-2-Clause | 已取得独立源码 |
| [tflite-micro](https://github.com/tensorflow/tflite-micro/tree/9f638f18154dff868e7053572f089be845b2fbdf) | 静态内存与 MCU 推理 | Apache-2.0 | 已取得独立源码 |
| [cmsis-nn](https://github.com/ARM-software/CMSIS-NN/tree/1e52d6833aecc075a487005fc16e75ce4c255182) | 量化神经算子 | Apache-2.0 | 已取得独立源码 |
| [dns-challenge](https://github.com/microsoft/DNS-Challenge/tree/591184a9fcb2cbdec02520fed81a32bbbf9d73ff) | 降噪挑战配方及 DNSMOS | MIT for code (LICENSE-CODE); data terms separate | 已取得独立源码 |
| [aec-challenge](https://github.com/microsoft/AEC-Challenge/tree/6c633d0a9d2a143a0e364899b91b06f127315b18) | 回声挑战与 AECMOS | MIT for repository code; assets separately | 已取得独立源码 |
| [chime-utils](https://github.com/chimechallenge/chime-utils/tree/152882404f572d40769ef02bf91c5a9a9cfc9c78) | 会议数据整理、活动与评测工具 | MIT | 已取得独立源码 |
| [ssspy](https://github.com/tky823/ssspy/tree/38b9389e8b1914422561f1936d9b28d042d62d2c) | FDICA、IVA、ILRMA、MNMF 与尺度恢复 | Apache-2.0 | 已取得独立源码 |
| [pb_bss](https://github.com/fgnt/pb_bss/tree/10acc347fc9ea21e3d312806a0bd751d0d0af183) | 空间聚类、GEV、BAN 与波束参考 | MIT | 已取得独立源码 |
| [gss](https://github.com/desh2608/gss/tree/10fad18cae85e2e4342c77421abc70c9c5da23ed) | 活动引导分离及 GPU 批处理 | MIT | 已取得独立源码 |
| [wpe_gpu](https://github.com/desh2608/wpe/tree/bd2857b5b8de36df4f436a93574c088bea142042) | CuPy 离线 WPE 与 GPU-GSS 去混响 | MIT | 已取得独立源码 |
| [dtln_aec](https://github.com/breizhn/DTLN-aec/tree/9d24e128b4f409db18227b8babb343016625921f) | 双信号处理域的神经 AEC | MIT | 已取得独立源码 |
| [speakerbeam](https://github.com/BUTSpeechFIT/speakerbeam/tree/91af02cc617afa35fedfbdbf32533012cd0a8672) | 目标说话人提取的受限评测实现 | LicenseRef-BUT-NTT-Evaluation | 仅来源索引 |
| [metaaf](https://github.com/adobe-research/MetaAF/tree/56c4665bdc51c2e0595a7c0cd9b1266408adceff) | 可学习更新规则及通用频域 RLS 的核心库；AEC Kalman/RLS 基线另在 `zoo/aec/` | 核心 University of Illinois/NCSA；`zoo/` Adobe Research License | 已取得 metaaf/ 核心与 README；未取得 zoo/ |
| [fastmnmf_author](https://github.com/sekiguchi92/SoundSourceSeparation/tree/897fe87fea3d85a243d8a3fd36c2232bb0548ad3) | FastMNMF 与自回归联合模型的作者实验 | LicenseRef-Academic-Research-Only | 仅来源索引 |
| [spmamba](https://github.com/JusperLee/SPMamba/tree/f939f60a10db8a66aa69ec09685684307af47412) | 空间域与时序状态空间分离 | Apache-2.0 | 已取得独立源码 |
| [mamba_tasnet](https://github.com/xi-j/Mamba-TasNet/tree/a35c692f27213781a11b1606c375cda1e1f0fb62) | Mamba 与 TasNet 分离实现 | GPL-3.0 | 已取得独立源码 |
| [nkf_aec](https://github.com/fjiang9/NKF-AEC/tree/8ac58fb8fb9ced48579f9aa310745c54f98d7e1f) | 神经 Kalman AEC 作者研究代码 | NOASSERTION | 仅来源索引 |
| [pyaec](https://github.com/ewan-xu/pyaec/tree/5b9c02c57075d790b7df8652884618189d49bbc4) | 时域 RLS、Kalman 与频域 FDKF/PFDKF 的教学代码；不含完整产品前端 | Apache-2.0；演示音频另核 | 仅来源索引，未运行 |
| [echocatzh/PFDKF](https://github.com/echocatzh/PFDKF/tree/7c8c86b5691966c330015d8e0960db0733b4844f) | 分块频域 Kalman 演示；默认输出含残余处理，并非单独的线性误差 | MIT；演示音频另核 | 仅来源索引，未运行 |
| [Subband_Kalman_AEC](https://github.com/changxuding/Subband_Kalman_AEC/tree/f0c4f7030769d94dea2da3c814837448f171d422) | MATLAB 子带 Kalman、平方根及信息形式；部分默认输出含非线性后处理 | MIT 代码；附带录音许可另核 | 仅来源索引，未运行 |
| [bssaec2020](https://github.com/nay0648/bssaec2020/tree/a3f52249ee61f19e2369823f65c6c31392dcf042) | 作者 MATLAB Aux-ICA/加权 RLS 仿真；不是完整工业 C++ 实现 | 未发现明确再分发许可；附带音频与 PESQ 文件另核 | 仅来源索引，不下载、不再分发 |
| [fn-ssl-ipdnet](https://github.com/Audio-WestlakeU/FN-SSL/tree/76fcb281be92caf068c712dfb015e354f437260f) | 直接路径 IPD 估计与定位 | MIT stated in README; complete license text and third-party notices not established | 仅来源索引 |
| [dcase2022-seld](https://github.com/sharathadavanne/seld-dcase2022/tree/c8adb1d3a5a35de2d6c7b6d19e01ad455eef3986) | multi-ACCDOA、ADPIT 与 SELD | No explicit redistribution license established from inspected official tree and source header | 仅来源索引 |
| [dcase2025-stereo-seld](https://github.com/partha2409/DCASE2025_seld_baseline/tree/42a48b6456b73be35ad0e1a9ffeb6ceef83ae0bd) | 双通道方位/距离与视听 SELD | No explicit redistribution license established from inspected official tree and source header | 仅来源索引 |
| [notsofar1](https://github.com/microsoft/NOTSOFAR1-Challenge/tree/6f58e08b008f7530ba4141f0aeb02447c70b6fd7) | 连续语音分离训练、会议推理与转写基线 | MIT code; DATA_LICENSE and dataset-version restrictions separate | 已取得独立源码 |

### 空间、控制与工程补充

以下源码用于填补明确的算法或接口缺项，不代表已经通过构建或论文复现。完整提交见锁表；libsoxr 使用官方 SourceForge Git 的 0.1.3 解引用提交。

| 项目与固定版本 | 对应算法或工程功能 | 代码许可摘要 | 本地获取范围 |
|---|---|---|---|
| [sbl](https://github.com/gerstoft/SBL/tree/d4bba35e9b60907d3024473ba5a41046450baae0) | 多快拍、多频稀疏贝叶斯定位 | GPL-3.0 | 已取得独立源码 |
| [robustsbl](https://github.com/NoiseLabUCSD/RobustSBL/tree/d746266a1336d4467f60b6f7b7e8b4695a01d26d) | Gauss、t、Huber、Tyler 损失的稳健 SBL | MIT | 已取得独立源码 |
| [btk20](https://github.com/kkumatani/distant_speech_recognition/tree/feff19ec8bcb770f6530fe280dc3ccafc2f5984a) | 子带 LMS/RLS 广义旁瓣抵消 | MIT; retain per-file notices | 已取得独立源码 |
| [smpphat](https://github.com/FrancoisGrondin/smpphat/tree/6fd33e6eb3251078a4cd9793dde909e2500265cc) | 合并等效麦对的 SRP-PHAT 加速 | GPL-3.0 | 已取得独立源码；本机负时延索引移植缺陷见空间研究，未通过数值复现 |
| [libsoxr](https://sourceforge.net/p/soxr/code/ci/945b592b70470e29f917f4de89b4281fbbd540c0/tree/) | 连续与可变比率重采样 | LGPL-2.1-or-later; embedded PFFFT terms separate | 已取得独立源码 |
| [libebur128](https://github.com/jiixyj/libebur128/tree/67b33abe1558160ed76ada1322329b0e9e058b02) | 响度和真峰值测量 | MIT | 已取得独立源码 |
| [pystoi](https://github.com/mpariente/pystoi/tree/74872b000753a7a42ff51aa0868af8c82c7f9053) | STOI 与 ESTOI 可懂度评分 | MIT for Python core; MATLAB test notices separate | 已取得源码子集 |
| [visqol](https://github.com/google/visqol/tree/38d0b0163e441047d4429bf07ad09e5b9031d02c) | 全参考语音/音频质量预测 | Apache-2.0 | 已取得源码子集 |
| [libsndfile](https://github.com/libsndfile/libsndfile/tree/b9103bd48b6c8fb517ae737fe3baee0c718b804c) | PCM/WAV 与音频文件读写 | LGPL-2.1-or-later; dependencies separately | 已取得独立源码 |
| [lib-xcore-math](https://github.com/xmos/lib_xcore_math/tree/16130be45c4002a1f875a4b06ff68d2065cd8c69) | 块浮点、FFT 与滤波内核 | XMOS Public Licence v1 | 已取得独立源码 |
| [e2e-ad-aec](https://github.com/ThomasHaubner/e2e_dnn_ad_control_for_lin_aec/tree/7a003133d742698de7acba9510d9586d7d57a584) | DNN 控制线性 CTF 回声消除 | BSD-4-Clause | 已取得源码子集 |
| [integrated-aec-nr](https://github.com/Arnout-Roebben/Integrated_AEC_NR/tree/23c6b567c7863a8ee9bafd38bad0d3ff2f25e185) | 联合与级联多通道 AEC/降噪 | MIT for code; recording permissions separate | 已取得源码子集 |
| [nbss](https://github.com/Audio-WestlakeU/NBSS/tree/cc42fc8ad2e6642c09b8f4169a85b4766dc22b7e) | OnlineSpatialNet 长序列多通道增强 | MIT | 已取得源码子集 |

`e2e-ad-aec` 保留 BSD-4-Clause 的广告署名条款；不能简写成三条款 BSD。`integrated-aec-nr` 的 MIT 代码许可不覆盖 VCTK/MYRiAD 录音，本地未取 `Audio/`；其许可文件名为 `LICENSE.md`。`pystoi` 仅取 Python 核心与根说明，未取得 MATLAB 测试资产。`visqol` 未取 `model/`、`testdata/`，也未启动 Bazel 拉取依赖，因此此源码子集不是完整可运行的评分包。`lib-xcore-math` 保留 XMOS 硬件使用限制，不能概括为任意硬件上的自由商用库。

## 真实录音的数据许可

DEMAND v1.0 的 NRIVER 河流场景来自 [Zenodo 1227121](https://zenodo.org/records/1227121)，作者为 Joachim Thiemann、Nobutaka Ito、Emmanuel Vincent。官方记录明确将音频和说明文档按 [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) 分发，核实日期为 2026-09-22。

本仓库只纳入前 10 s 的同步 16 通道摘录、单麦提取、双麦及 16 麦零延时均值，4 个 WAV 连同修改说明保持同一数据许可；见 [署名与修改记录](real_audio/ATTRIBUTION.txt)和[数据清单](real_audio/MANIFEST.json)。完整归档固定 SHA-256，保存在忽略的下载目录，不进入 Git。公开数据许可不等于原作者认可本书的处理结果，也不改变本仓库独立代码的许可状态。

## 不能混同的许可范围

- **MetaAF**：仅取得 `metaaf/` 核心及 README。核心许可为 University of Illinois/NCSA；[`zoo/`](https://github.com/adobe-research/MetaAF/blob/56c4665bdc51c2e0595a7c0cd9b1266408adceff/zoo/README.md)、任务配方和权重另受其 [Adobe Research License](https://github.com/adobe-research/MetaAF/blob/56c4665bdc51c2e0595a7c0cd9b1266408adceff/zoo/LICENSE) 约束，限非商业研究、教学与测试，不能据此用于商业产品开发，未随核心下载。核心里的 `optimizer_rls.py` 是通用更新器；`zoo/aec/` 的 Kalman/RLS 基线不能因此算作已经取得或运行。
- **RLS/Kalman AEC 示例**：pyroomacoustics 的 RLS/BlockRLS 是通用滤波器，不提供 AEC 所需的播放参考对齐、双讲保护和残余回声抑制。pyaec、PFDKF、Subband_Kalman_AEC 是可研究的算法演示，不能由代码可见推断工业部署；它们的示例录音许可应逐项另核。bssaec2020 没有查到明确源码再分发许可，维持只登记不下载，其仓库 README 中还有 Interspeech 2020 退稿通知，不能称作该会议已接收实现。
- **Spatial Audio Framework**：核心 ISC，部分可选模块 GPL-2.0。不能用核心许可描述全部模块。
- **icoDOA、piva、Mamba-TasNet、ALSA**：已取得各自许可下的独立源码用于研究；AGPL/GPL/LGPL 的义务不能因放在下载目录而消失。这里没有把这些项目合并或重新授权为本书代码。
- **SpeakerBeam、FastMNMF 作者整库**：所核实实现分别有评测或学术用途限制，未自动获取。算法本身与某一实现的限制要分开；FastMNMF 另有已取得的 pyroomacoustics 实现。
- **NKF-AEC、FN-SSL/IPDnet、DCASE 2022/2025 基线**：完整授权依据或第三方声明尚未建立，保留官方源码定位，不从“公开可见”推断完整使用与分发权限。
- **NOTSOFAR-1、DNS、AEC 和 CHiME**：代码与数据的条款分开。NOTSOFAR-1 的 README 还对 dev-set-2 作了移除及用途说明，不能只看根目录数据许可便假定所有历史版本均可取得。推理脚本可能自动下载模型和数据，本次未执行。
- **S4M、AudioSep、SGMSE、StoRM、ArrayDPS**：取得模型定义或推理代码，不等于取得完整训练配方、检查点与训练语料。是否因果及运行速度也必须绑定配置再测。

上述判断绑定表中提交。若要重新分发或用于产品，应读取相应提交的 LICENSE、COPYING、NOTICE、文件头及依赖许可；本表不是法律意见。本仓库尚无明确根许可证，不能据此替任何来源授予新的权利。

## 本地源码怎样保存

获取工具只处理锁定清单中的匿名 HTTPS 地址和完整提交号。下载目录被 Git 忽略，因此源码确实存在本机 `codes/` 下，但不会随主仓库普通提交一起上传。其他读者可用同一清单和获取命令重建这些独立目录；原始许可文件保留在各目录内。

新的工作区使用稀疏检出，跳过常见音频、模型和压缩包扩展名，不拉取 LFS 对象或子模块。扩展名过滤不等于识别了所有数据，Git 对象中也可能包含上游内嵌资产。既有完整工作区保留，不为了统一目录形态覆盖或删除其中内容。

WebRTC 等大型项目当前只有主源码工作区，未运行其多仓依赖工具；CMSIS、SOF 与推理运行时也没有进行目标板编译。源码入口已核对与依赖齐全是两个不同结论。

## 运行验证的范围

[SOURCE_STATUS.json](SOURCE_STATUS.json) 由源码核对工具生成，其中 `execution: not_run` 表示该工具不执行外部项目。方法级实验另行记录：本次已运行 [WPE 对照](examples/compare_wpe_reference.py)，对照的是 nara_wpe 0.0.11 的离线有效帧计算，不是对整个项目或语音质量的认证。

设备链路应依次检查采集时钟、播放参考、缓冲状态、算法状态与输出评分。回调线程不能直接承担不受控的 NumPy 分配、锁、文件或网络操作；工业实现的细节见[部署研究](research/03_industrial_deployment.md)和[复现步骤](research/04_source_reproduction.md)。
