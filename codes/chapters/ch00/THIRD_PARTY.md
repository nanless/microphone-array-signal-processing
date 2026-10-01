# 第三方实现与工业生态索引

原有索引核实于 2026-09-22，新增 RLS/Kalman AEC 来源核实于 2026-09-23。此索引包含 100 个 Git 上游项目；已在 `codes/chapters/ch00/upstream/_downloads/` 取得 87 个独立源码工作区，其中 pyaec、PFDKF 与 Subband_Kalman_AEC 于 2026-09-24 按固定提交取得指定源码和许可文件，DiCoW v1 演示与 TS-ASR-Whisper v1 训练源码于 2026-09-28 按固定提交取得。2026-09-30 新增原作者湿空气声速模型，2026-10-01取得SAID及NeMo WPE固定源码，2026-10-02取得TAC四文件并离线重核，86 个通过、AEC Challenge 的 5 个真实录音有本地变动而未计通过，另 13 项仅登记来源。pystoi 是软件作者维护的 Python 实现，不称为原论文作者的官方 Python 程序。获取状态与完整提交见 [SOURCE_STATUS.json](SOURCE_STATUS.json) 和 [SOURCES.lock.json](SOURCES.lock.json)。状态报告由获取工具离线生成；不能用源码获取结果证明新增项目已运行。

“已取得”只说明来源、提交、工作区状态和指定入口符合清单，不表示已经安装依赖、编译、运行训练、取得权重、完成声学测试或取得产品使用资格。每项的完整入口和限制保存在锁定清单；逐算法解释、最小实验和失效条件见[研究手册](research/README.md)。

## 官方源码与用途

`codes/chapters/ch07/examples/compare_online_wpe_reference.py` 中的 `NumpyOnlineWPE011` 是对 nara-wpe 0.0.11
`OnlineWPE` 和正数求逆保护行为的精简适配，保留缓冲顺序及更新口径，用于版本对照而非独立推导证明。
该改编范围保留 Communications Engineering Group, Paderborn University 的 2018 年版权与
[完整 MIT 声明](../ch07/licenses/nara_wpe_MIT.txt)；脚本注明来源、改编内容和固定版本。
这份上游许可不等于为本仓库其他原创文件选择统一许可证。GPL 空间算法仍只在独立下载目录中调用，
没有复制进教学包。

SMP-PHAT 的本地实验额外使用 FFTW 3.3.10 单精度静态库。它是构建依赖，不另计为一种空间算法，
也不计入主锁定清单的 Git 源码索引。官方归档为
[`fftw-3.3.10.tar.gz`](https://fftw.org/pub/fftw/fftw-3.3.10.tar.gz)，SHA-256 为
`56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467`。
归档的 `COPYRIGHT` 与 `kernel/alloc.c` 声明 GPL-2.0-or-later；`api/fftw3.h` 单独采用 BSD 两条款文本，
不能把头文件的许可推广到整库。源码、许可文件与编译产物只保存在本地隔离目录，不随本书提交分发；
运行报告另记实际链接库的版本与摘要。

项目链接定位官方仓库的已核实版本。表中的许可证仅概括相应代码范围；依赖、模型、录音和数据集分别检查。

| 项目与固定版本 | 对应算法或工程功能 | 代码许可摘要 | 本地获取范围 |
|---|---|---|---|
| [Speed of sound in air](https://github.com/RobertoGavioso/Speed-of-sound-in-air/tree/5c7e6652cf2fd7b4c52e83d8fa3782b3c56e61de) | 2025湿空气声速与不确定度模型，供环境校准研究 | 根GPLv3文本；未核得项目级or-later声明，LabVIEW运行时独立 | 2026-09-30取得43个VI图形源码、ReadMe和LICENSE，45文件共2147682字节；省略EXE和独立许可论文PDF。VI框图未读取、模型未执行；源使用需要LabVIEW晚于2016的64位版本 |
| [SAID](https://github.com/IN03X/SAID/tree/cf52ede4f38361cdbb03aa93c8254109583dfe2b) | Audio2Sph球面方向编码、Sph2Imaging逐源能量图与类别、DCASE 2026 Track A压缩提交 | 原创源MIT，所列第三方组件MIT/Apache-2.0；权重独立非商业研究条款，AudioMAE权重CC BY-NC 4.0 | 固定源码选集及完整LICENSES/第三方声明已取得；未取检查点或said/demo_data，未运行网络、训练或榜单评测；原压缩单文件实验范围见空间研究 |
| [TAC作者单阶段FaSNet变体](https://github.com/yluo42/TAC/tree/e3373b73358a96af6f64fdbe25327def8d6bd973) | 共享通道变换、平均和拼接；与参考通道封装分别核对 | README声明CC BY-NC-SA 3.0 US，固定树无独立LICENSE；不能误称MIT或自由商用 | 2026-10-02取得README、FaSNet.py、utility/__init__.py、utility/models.py四文件，只在忽略的独立工作区保留；静态合同核查未执行Torch网络，不取音频/模型/训练数据 |
| [pyroomacoustics](https://github.com/LCAV/pyroomacoustics/tree/0dd39f2614b7fc44b2cc63dbe7d60f4641068890) | 房间仿真、STFT、DOA、波束、盲分离及通用 RLS/BlockRLS 自适应滤波；后者不是完整 AEC 链路 | MIT | 已取得独立源码 |
| [odas](https://github.com/introlab/odas/tree/bcb845434495e293df3d48f1203b7a86e1852449) | 定位、追踪、分离与后滤波的实时 C 链路 | MIT | 已取得独立源码 |
| [NeMo v2.4.0 WPE](https://github.com/NVIDIA/NeMo/tree/2381f42f6979449b5b99538f8f80135831009b51) | 幅度掩码WPE、长度屏蔽和多轮回归输入契约 | Apache-2.0；完整LICENSE保留，固定根NOTICE不存在 | 2026-10-01取得11个精确源码/许可文件；原NeMo URL现重定向NVIDIA-NeMo/Speech；不是完整可安装框架，未装Torch/模型或运行原算子 |
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
| [kaldialign](https://github.com/pzelasko/kaldialign/tree/06ac40f03c3d368932adf8536965a088d54189b1) | MeetEval相关编辑距离路径的Python/C++编译依赖；源自Kaldi对齐代码 | Apache-2.0；构建还需另核pybind11依赖及其条款 | 已取得源码选集并核验提交/范围；未构建、安装或运行评分接口 |
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
| [spatial-audio-framework](https://github.com/leomccormack/Spatial_Audio_Framework/tree/18fd5aba46e20787b51f28f7197a68506c965c07) | C/C++ 球阵处理、功率图、HRIR/HRTF 与可选追踪 | ISC core；本轮saf_tracker文件头GPL-2.0-or-later，其他模块逐文件核对 | 已取得独立源码 |
| [libmysofa](https://github.com/hoene/libmysofa/tree/6cc5b15a73e9bd97810d03767082edda7f315881) | SOFA读取、HRIR方向插值与归一化接口 | BSD-3-Clause，第三方源码声明分别保留 | 已取得源码子集；未编译或数值运行，不含SOFA测量数据 |
| [pyfar](https://github.com/pyfar/pyfar/tree/0bfe1e8b7d71ab83edd3ea3b5fab7b28761d114a) | 指数扫频、频带受限谱反卷积、信号与频谱单位接口 | MIT，保留作者声明；测量数据许可另核 | v0.8.1固定源码已取得；未安装、未运行或实机测量 |
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
| [aec-challenge](https://github.com/microsoft/AEC-Challenge/tree/6c633d0a9d2a143a0e364899b91b06f127315b18) | 回声挑战与 AECMOS | MIT for repository code; assets separately | 已有独立工作树；5 个录音本地有改动，离线核验失败，未覆盖 |
| [chime-utils](https://github.com/chimechallenge/chime-utils/tree/152882404f572d40769ef02bf91c5a9a9cfc9c78) | 会议数据整理、活动与评测工具 | MIT | 已取得独立源码 |
| [ssspy](https://github.com/tky823/ssspy/tree/38b9389e8b1914422561f1936d9b28d042d62d2c) | FDICA、IVA、ILRMA、MNMF、FastMNMF、cACGMM 与尺度恢复 | Apache-2.0 | 已取得独立源码；cACGMM 入口仅静态核对，未运行 |
| [pb_bss](https://github.com/fgnt/pb_bss/tree/10acc347fc9ea21e3d312806a0bd751d0d0af183) | 空间聚类、GEV、BAN 与波束参考 | MIT | 已取得独立源码 |
| [gss](https://github.com/desh2608/gss/tree/10fad18cae85e2e4342c77421abc70c9c5da23ed) | 活动引导分离及 GPU 批处理 | MIT | 已取得独立源码 |
| [wpe_gpu](https://github.com/desh2608/wpe/tree/bd2857b5b8de36df4f436a93574c088bea142042) | CuPy 离线 WPE 与 GPU-GSS 去混响 | MIT | 已取得独立源码 |
| [dtln_aec](https://github.com/breizhn/DTLN-aec/tree/9d24e128b4f409db18227b8babb343016625921f) | 双信号处理域的神经 AEC | MIT | 已取得独立源码 |
| [speakerbeam](https://github.com/BUTSpeechFIT/speakerbeam/tree/91af02cc617afa35fedfbdbf32533012cd0a8672) | 目标说话人提取的受限评估实现 | LicenseRef-BUT-NTT-Evaluation | 已取得8个原样文件供内部评估；不随本书再分发 |
| [DiCoW v1 推理与同提交 v2 Web 演示](https://github.com/BUTSpeechFIT/DiCoW/tree/e9326bd536bf632e823357438b210102903ba620) | `example.py` 加载 v1，`app.py` 加载 v2；两者均依赖 Pyannote 说话人分段 | Apache-2.0 源码；模型与分段器另核 | 固定分支源码选集；不含权重、数据或服务部署 |
| [TS-ASR-Whisper v1](https://github.com/BUTSpeechFIT/TS-ASR-Whisper/tree/0ea6679d44405f5ff39188030123524686c198e9) | 原 DiCoW 训练、解码、Hydra 配置与评分；作者已标记此分支为旧版 | Apache-2.0 源码；子模块、权重和语料另核 | 固定 v1 分支源码选集；不初始化子模块或取得模型、数据 |
| [FlowSep](https://github.com/Audio-AGI/FlowSep/tree/d8164db58bd461ef5bb6df8ffd372b536ee6afb4) | 文本查询的整流流匹配声音分离；不是目标说话人转写 | 固定提交未找到明确仓库源码许可；仅索引 | 不自动取得源码、权重或演示音频，不再分发 |
| [metaaf](https://github.com/adobe-research/MetaAF/tree/56c4665bdc51c2e0595a7c0cd9b1266408adceff) | 可学习更新规则及通用频域 RLS 的核心库；AEC Kalman/RLS 基线另在 `zoo/aec/` | 核心 University of Illinois/NCSA；`zoo/` Adobe Research License | 已取得核心、zoo/aec/、zoo/wpe/ 与直接公共源码；zoo 仅限非商用研究（含教学/测试），附独立许可；未取权重、未运行网络 |
| [fastmnmf_author](https://github.com/sekiguchi92/SoundSourceSeparation/tree/897fe87fea3d85a243d8a3fd36c2232bb0548ad3) | FastMNMF 与自回归联合模型的作者实验 | LicenseRef-Academic-Research-Only | 仅来源索引 |
| [spmamba](https://github.com/JusperLee/SPMamba/tree/f939f60a10db8a66aa69ec09685684307af47412) | 空间域与时序状态空间分离 | Apache-2.0 | 已取得独立源码 |
| [mamba_tasnet](https://github.com/xi-j/Mamba-TasNet/tree/a35c692f27213781a11b1606c375cda1e1f0fb62) | Mamba 与 TasNet 分离实现 | GPL-3.0 | 已取得独立源码 |
| [nkf_aec](https://github.com/fjiang9/NKF-AEC/tree/8ac58fb8fb9ced48579f9aa310745c54f98d7e1f) | 神经 Kalman AEC 作者研究代码 | NOASSERTION | 仅来源索引 |
| [pyaec](https://github.com/ewan-xu/pyaec/tree/5b9c02c57075d790b7df8652884618189d49bbc4) | 时域 RLS、Kalman 与频域 FDKF/PFDKF 的教学代码；不含完整产品前端 | Apache-2.0；演示音频另核 | 已取得源码；RLS/Kalman 小输入实调发现截尾，FDKF/PFDKF 在 NumPy 2.5.3 入口失败 |
| [echocatzh/PFDKF](https://github.com/echocatzh/PFDKF/tree/7c8c86b5691966c330015d8e0960db0733b4844f) | 分块频域 Kalman 演示；默认输出含残余处理，并非单独的线性误差 | MIT；演示音频另核 | 已取得源码并实调首块，确认默认后处理改变线性残差；未测设备 |
| [Subband_Kalman_AEC](https://github.com/changxuding/Subband_Kalman_AEC/tree/f0c4f7030769d94dea2da3c814837448f171d422) | MATLAB 子带 Kalman、平方根及信息形式；部分默认输出含非线性后处理 | MIT 代码；附带录音许可另核 | 已取得指定 `.m` 源码、README 和许可，未取得音频或 `.mat`，未运行 |
| [bssaec2020](https://github.com/nay0648/bssaec2020/tree/a3f52249ee61f19e2369823f65c6c31392dcf042) | 作者 MATLAB Aux-ICA/加权 RLS 仿真；不是完整工业 C++ 实现 | 未发现明确再分发许可；附带音频与 PESQ 文件另核 | 仅来源索引，不下载、不再分发 |
| [fn-ssl-ipdnet](https://github.com/Audio-WestlakeU/FN-SSL/tree/76fcb281be92caf068c712dfb015e354f437260f) | 直接路径 IPD 估计与定位 | MIT stated in README; complete license text and third-party notices not established | 仅来源索引 |
| [dcase2022-seld](https://github.com/sharathadavanne/seld-dcase2022/tree/c8adb1d3a5a35de2d6c7b6d19e01ad455eef3986) | multi-ACCDOA、ADPIT 与 SELD | No explicit redistribution license established from inspected official tree and source header | 仅来源索引 |
| [dcase2025-stereo-seld](https://github.com/partha2409/DCASE2025_seld_baseline/tree/42a48b6456b73be35ad0e1a9ffeb6ceef83ae0bd) | 双通道方位/距离与视听 SELD | No explicit redistribution license established from inspected official tree and source header | 仅来源索引 |
| [notsofar1](https://github.com/microsoft/NOTSOFAR1-Challenge/tree/6f58e08b008f7530ba4141f0aeb02447c70b6fd7) | 连续语音分离训练、会议推理与转写基线 | MIT code; DATA_LICENSE and dataset-version restrictions separate | 已取得独立源码 |

TS-ASR-Whisper v1 的 `inference_pipeline` 子模块固定指向表中 DiCoW 提交；但该提交的 [`example.py`](https://github.com/BUTSpeechFIT/DiCoW/blob/e9326bd536bf632e823357438b210102903ba620/example.py) 加载 `BUT-FIT/DiCoW_v1`，[`app.py`](https://github.com/BUTSpeechFIT/DiCoW/blob/e9326bd536bf632e823357438b210102903ba620/app.py) 加载 `BUT-FIT/DiCoW_v2`，不能把整份源码中的演示都称为 v1。两者均用 `pyannote/speaker-diarization-3.1`，固定 `requirements.txt` 指定 `pyannote.audio==3.3.2`；当前主分支的 v3/SE-DiCoW 则是不同版本。两仓源码 `LICENSE` 均为 Apache-2.0；[v1](https://huggingface.co/BUT-FIT/DiCoW_v1)和[v2](https://huggingface.co/BUT-FIT/DiCoW_v2)模型卡分别标 CC BY 4.0。[Pyannote speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)与其所需的[segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)模型卡各标 MIT，但两处都要求接受访问条件并使用 Hugging Face 令牌；公开可见不等于免除取得前提。[作者当前 DiCoW 许可说明](https://github.com/BUTSpeechFIT/DiCoW#license)把后续演示采用的 DiariZen 权重列为 CC BY-NC 4.0，[DiariZen 模型许可原文](https://github.com/BUTSpeechFIT/DiariZen/blob/main/MODEL_LICENSE)再次确认非商用限制，它不是固定 v1/v2 Pyannote 路径的必需权重。FlowSep 的固定源码树没有仓库许可文件；其论文、Zenodo checkpoint 与网页演示不能代替源码许可。因此 FlowSep 仅保留固定提交及入口，不进入本地获取或本书代码分发。两份 DiCoW 源码虽已取得，也未安装依赖、取得权重、执行推理或训练。最小实验与方法差异见[神经方法研究手册 N17～N18](research/02_aec_wpe_separation.md#neural)。

### 空间、控制与工程补充

以下源码用于填补明确的算法或接口缺项，不代表已经通过构建或论文复现。完整提交见锁表；libsoxr 使用官方 SourceForge Git 的 0.1.3 解引用提交。

| 项目与固定版本 | 对应算法或工程功能 | 代码许可摘要 | 本地获取范围 |
|---|---|---|---|
| [sbl](https://github.com/gerstoft/SBL/tree/d4bba35e9b60907d3024473ba5a41046450baae0) | 多快拍、多频稀疏贝叶斯定位 | GPL-3.0 | 已取得独立源码 |
| [robustsbl](https://github.com/NoiseLabUCSD/RobustSBL/tree/d746266a1336d4467f60b6f7b7e8b4695a01d26d) | Gauss、t、Huber、Tyler 损失的稳健 SBL | MIT | 已取得独立源码 |
| [btk20](https://github.com/kkumatani/distant_speech_recognition/tree/feff19ec8bcb770f6530fe280dc3ccafc2f5984a) | 子带 LMS/RLS 广义旁瓣抵消、后置滤波与 WPE 去混响 | MIT; retain per-file notices | 已取得独立源码；WPE 入口已静态核对，未构建整链 |
| [smpphat](https://github.com/FrancoisGrondin/smpphat/tree/6fd33e6eb3251078a4cd9793dde909e2500265cc) | 合并等效麦对的 SRP-PHAT 加速 | GPL-3.0 | 已取得独立源码；未改原版在本机数值失败，临时两行适配版的合成案例通过；原版与适配版报告分开保存 |
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

本仓库只纳入前 10 s 的同步 16 通道摘录、单麦提取、双麦及 16 麦零延时均值，4 个 WAV 连同修改说明保持同一数据许可；见 [署名与修改记录](../ch02/real_audio/ATTRIBUTION.txt)和[数据清单](../ch02/real_audio/MANIFEST.json)。完整归档固定 SHA-256，保存在忽略的下载目录，不进入 Git。公开数据许可不等于原作者认可本书的处理结果，也不改变本仓库独立代码的许可状态。

## 不能混同的许可范围

- **MetaAF**：已取得 `metaaf/` 核心、`zoo/aec/`、`zoo/wpe/` 及所选公共依赖和许可。核心为 University of Illinois/NCSA；任务配方另受其 [Adobe Research License](https://github.com/adobe-research/MetaAF/blob/56c4665bdc51c2e0595a7c0cd9b1266408adceff/zoo/LICENSE) 约束，限非商业研究、教学与测试，不能由核心许可推定 zoo 可用于商业产品。核心里的 `optimizer_rls.py` 是通用更新器；AEC 与 WPE 配方的源码已取得，但未加载权重、训练或执行完整网络推理。受控原函数诊断另见增强研究 W09/W11。
- **RLS/Kalman AEC 示例**：pyroomacoustics 的 RLS/BlockRLS 是通用滤波器，不提供 AEC 所需的播放参考对齐、双讲保护和残余回声抑制。pyaec、PFDKF、Subband_Kalman_AEC 是可研究的算法演示，不能由代码可见推断工业部署；它们的示例录音许可应逐项另核。bssaec2020 没有查到明确源码再分发许可，维持只登记不下载，其仓库 README 中还有 Interspeech 2020 退稿通知，不能称作该会议已接收实现。
- **Spatial Audio Framework**：核心 ISC，部分可选模块 GPL-2.0。不能用核心许可描述全部模块。
- **icoDOA、piva、Mamba-TasNet、ALSA**：已取得各自许可下的独立源码用于研究；AGPL/GPL/LGPL 的义务不能因放在下载目录而消失。这里没有把这些项目合并或重新授权为本书代码。
- **SpeakerBeam**：固定协议允许内部测试、分析与评估；现已取得8个原样源码与许可文件，保存在忽略的独立目录，不修改或随本书再分发。现版本实际模型入口为 `src/models/td_speakerbeam.py`，许可附件仍列旧名；未安装未固定的模型依赖、取得权重或执行推理。
- **FastMNMF 作者整库**：许可将使用主体限定为大学或研究机构的学术研究，当前任务未确认这一资格，因此保留来源索引；符合主体条件或另获授权后可取得。不能仅因“研究用途”推定所有个人均获许可。FastMNMF另有已取得的MIT pyroomacoustics实现。
- **NKF-AEC、FN-SSL/IPDnet、DCASE 2022/2025 基线**：完整授权依据或第三方声明尚未建立，保留官方源码定位，不从“公开可见”推断完整使用与分发权限。
- **NOTSOFAR-1、DNS、AEC 和 CHiME**：代码与数据的条款分开。NOTSOFAR-1 的 README 还对 dev-set-2 作了移除及用途说明，不能只看根目录数据许可便假定所有历史版本均可取得。推理脚本可能自动下载模型和数据，本次未执行。
- **S4M、AudioSep、SGMSE、StoRM、ArrayDPS**：取得模型定义或推理代码，不等于取得完整训练配方、检查点与训练语料。是否因果及运行速度也必须绑定配置再测。
- **ArrayDPS 实际调用**：2026-10-01 核对固定 `separate.py`→`src/sampler_spatial_v1_reverb_iva_8kHz.py`→FCP/IVA/STFT；通用 `src/sampler.py` 不是这条分离入口。原驾驶器用干净源 SDR 控制试验停止，部署时不能照搬；研究和合同审计明确与原文仅用混合重构的 ML 选择分开。此次仅静态审计，未运行 Torch、权重或采样，代码 MIT 与各类模型/语料权利仍分别记录。

上述判断绑定表中提交。若要重新分发或用于产品，应读取相应提交的 LICENSE、COPYING、NOTICE、文件头及依赖许可；本表不是法律意见。本仓库尚无明确根许可证，不能据此替任何来源授予新的权利。

## 本地源码怎样保存

获取工具只处理锁定清单中的匿名 HTTPS 地址和完整提交号。下载目录被 Git 忽略，因此源码确实存在本机 `codes/` 下，但不会随主仓库普通提交一起上传。其他读者可用同一清单和获取命令重建这些独立目录；原始许可文件保留在各目录内。

新的工作区使用稀疏检出，跳过常见音频、模型和压缩包扩展名，不拉取 LFS 对象或子模块。扩展名过滤不等于识别了所有数据，Git 对象中也可能包含上游内嵌资产。既有完整工作区保留，不为了统一目录形态覆盖或删除其中内容。

通用获取器只检出 WebRTC 主源码；另一次专用构建已同步必要依赖并生成 `audioproc_f`，实际运行边界见[复现手册](research/04_source_reproduction.md)。两种目录用途不能混写。CMSIS、SOF 与推理运行时没有进行目标板编译；源码入口核对仍不等于设备链路验收。

## 运行验证的范围

[SOURCE_STATUS.json](SOURCE_STATUS.json) 由源码核对工具生成，其中 `execution: not_run` 表示该工具不执行外部项目。方法级实验另行记录：本次已运行 [WPE 对照](../ch07/examples/compare_wpe_reference.py)，对照的是 nara_wpe 0.0.11 的离线有效帧计算，不是对整个项目或语音质量的认证。

设备链路应依次检查采集时钟、播放参考、缓冲状态、算法状态与输出评分。回调线程不能直接承担不受控的 NumPy 分配、锁、文件或网络操作；工业实现的细节见[部署研究](research/03_industrial_deployment.md)和[复现步骤](research/04_source_reproduction.md)。

## 2026-09-26 流式算法补充

| 固定作者仓库 | 用途 | 许可与取得范围 | 验证边界 |
|---|---|---|---|
| [Stream.FM](https://github.com/sp-uhh/streamfm/tree/ab2700c1154acc5c2ce67a5344182028336413f5) | 流匹配语音恢复、逐步状态接口 | AGPL-3.0；独立源码与许可，未取权重/数据 | 源码检查，非完整推理运行；状态接口疑点见增强研究 |
| [FastEnhancer](https://github.com/aask1357/fastenhancer/tree/f85223bd546b27f39dc0744e0310dcd246f750a4) | 单通道流式降噪、显式ONNX状态 | MIT；独立源码与许可，模型二进制未取 | 非AEC/WPE，不引用设备性能；实现说明见工业研究 |
| [faster-enhancer.c](https://github.com/kdrkdrkdr/faster-enhancer.c/tree/7d78dab11bca854fb200426c621a96d7b37a2983) | FastEnhancer-Medium 的独立C11/int8运行时，48 kHz、320样本步长 | MIT源码，原作者版权见NOTICE；仅取得LICENSE、NOTICE、README、CMake与include/src/tests/docs原样源码选集，不取权重或测试音频 | 2026-07预印本的移植实现；源码已取得并核验范围，未编译、运行模型或测设备性能；与原项目不同配置的数字不可直接比较 |

三项均保存在忽略的独立下载目录，不随本书提交重新分发。FastEnhancer上游 `onnx/` 模型目录被二进制筛选省略，实际API入口是 `scripts/test_onnx.py` 和 `scripts/export_onnx.py`。faster-enhancer.c的`weights/`与`testaudio/`未取得。代码许可不替代权重和数据条款。


## 固定分数延迟与时频分离补充

2026-09-26 核对并取得以下两项独立源码。当前全量取得与核验数量以文首状态及生成的 `SOURCE_STATUS.json` 为准；方法级结果单列。

| 项目 | 许可与固定提交 | 实际范围 |
|---|---|---|
| [STK](https://github.com/thestk/stk/tree/6aacd357d76250bb7da2b1ddf675651828784bbc) | MIT-STK；`6aacd357d76250bb7da2b1ddf675651828784bbc` | include/src及许可；实际编译DelayL标量接口，保留状态与重置反例；未取rawwaves素材、未做设备测试 |
| [TF-Locoformer](https://github.com/merlresearch/tf-locoformer/tree/7a615460d347ff7334a13dbb831d16280da72cdc) | Apache-2.0及逐文件许可；`7a615460d347ff7334a13dbb831d16280da72cdc` | MERL作者仓库，源码与配置；静态接口研究，未取得权重或执行推理 |

本地源码位于`codes/chapters/ch00/upstream/_downloads/stk/`和`tf-locoformer/`，不随本书Git再分发。STK的许可说明保留作者声明；TF-Locoformer的模型与训练数据权利另行核对。细节见[工业I29](research/03_industrial_deployment.md#i29stkdelayl)与[增强研究](research/02_aec_wpe_separation.md)。


## 2026-09-28 阵列几何与标定源码

| 固定作者来源 | 许可与取得范围 | 验证边界 |
|---|---|---|
| [StructureFromSound](https://github.com/kalleastrom/StructureFromSound/tree/9b7db79a489d347bed9a38bd38224e65c273660a) | GPL-3.0；已取得README、许可与matlab目录，省略data/、tex/及常见音频/二进制 | 作者同步麦TDOA几何标定实现；主入口有作者路径与外部依赖，备用v2存在冲突标记。已核源码，未执行MATLAB或完整录音流程 |
| [Alias-Free Arrays](https://github.com/Zhao-Shen/Alias-free-Arrays/tree/4c80e169518d44f8333aac7f13c935286538f670) | 固定提交未建立代码许可证；仅索引，不自动获取或再分发 | 作者论文与MATLAB绘图入口对应；本书六边形相位相等是独立计算 |

StructureFromSound保存在Git忽略的`codes/chapters/ch00/upstream/_downloads/structure-from-sound/`，未随本书重新分发。获取工具保留原始冲突与依赖，不把静态核查记为运行成功。doatools已有固定目录新增登记几何、协同阵和失配前向模型入口；其误差施加器不估计未知校准量。逐函数解释、几何规范与工业流形采集流程见[空间研究§39](research/01_spatial_and_tracking.md#39-差分协同阵与虚拟协方差重建)。


| 工业来源 | 实际保留范围 | 许可与验证边界 |
|---|---|---|
| [HARKTOOL5 3.5.0](https://hark.jp/download/source-code/) | 独立归档锁定 README、debian 许可与变更、CMake、src、python 和两个文档源；不取生成文档目录 | HARK License v2.0，研究、开发、教育和学术用途受原文约束，商业用途另行授权；不是普通宽松开源许可。未编译或测设备 |
| [Infineon AE 配置](https://github.com/Infineon/mtb-example-psoc-edge-ae-application/tree/955a61090acf7caddd75fc74161fc0fb40aa7ae5) | `infineon-ae-config`：仅 GeneratedSource 下配置 C/H 与根 LICENSE | 两个文件逐文件 Apache-2.0；根 EULA 保留供核对，应用包装代码、配置器项目和算法核心不在取得范围内；未构建或刷写 |

HARK 不属于 Git 项目计数，另见 [ARCHIVE_SOURCES.lock.json](ARCHIVE_SOURCES.lock.json) 和由工具生成的 [ARCHIVE_SOURCE_STATUS.json](ARCHIVE_SOURCE_STATUS.json)。下载采用官方 `.dsc` 公布的固定 SHA-256，来源核实于2026-09-28；归档与源码放在忽略目录，不随本书 Git 提交重新分发。Infineon 已取得三个登记文件并通过固定提交、来源、清洁工作树和筛选范围核验；取得的配置数组不能重现闭源算法核心。

### 第4章固定实现复核（2026-09-28）

[doatools ESPRIT 实际对照](../ch04/reports/doatools_esprit_reference.json)记录固定版本默认行加权在重叠切片上原地写入产生的偏差；`row_weights="none"`与独立安全复制参考分别保留。成功标志不代替方向误差检查。本书没有修改上游工作树，也不将该诊断称为整个工具包验收。

| 固定来源 | 范围 | 许可与取得状态 |
|---|---|---|
| [X-SRP](https://github.com/egrinstein/xsrp/tree/5876b760c0ead781c05d4f319ed23e302478ea2d) | 可组合时域/频域及体积 SRP；研究手册限定当前接口 | 元数据声明 MIT，但未建立完整许可条款；仅索引，未运行 |
| [DCASE2026 SAISELD 官方基线](https://github.com/iranroman/DCASE2026_Task3_SAISELD_baseline/tree/d4df66251f39e34bc0157be93858e5a68ec9d7c4) | 声学图、分割、追踪与事件输出链 | 固定版本未建立明确源码许可；仅索引，权重许可另核，未运行 |

源码入口和版本见锁表。网站可访问、Python 包元数据的许可名称、模型权重可下载是不同事实，不能互相代替授权条款。


## 第5章的固定源码检查与直接依赖

2026-09-28 按每个算法的输入、输出、约定和失败边界复核波束/后滤实现。两项 Politis 直接依赖按以下最小选集取得；它们补齐现有球阵实现的函数来源，不增加两种算法，也不表示全部 MATLAB 示例已运行。

| 来源与固定提交 | 许可与本地范围 | 结果与边界 |
|---|---|---|
| [Spherical-Harmonic-Transform](https://github.com/polarch/Spherical-Harmonic-Transform/tree/30ec1454ab654a0432eafd168da7bee72ecca3fc) | BSD-3-Clause；LICENSE、README、getSH.m | 实/复球谐基直接依赖；未运行MATLAB，不含网格数据和完整工具箱 |
| [Array-Response-Simulator](https://github.com/polarch/Array-Response-Simulator/tree/1ebfb28296736c52691c63e1aa336a7bd0d6216b) | BSD-3-Clause；LICENSE、README、sphModalCoeffs及6个Bessel/Hankel函数 | 模态计算依赖闭合；未包含完整空间模拟器或测量；directional端点头注差异见研究 |
| [iDeepPE](https://github.com/CSeIn/iDeepPE/tree/c2cdc26ddafd33bf3bda45640c06febef9092365) | 未建立明确代码许可；仅固定索引 | 不获取/训练/推理；模型和CHiME-4/DEMAND/VCTK数据许可分别核查 |

SOF 的[审查程序](../ch05/examples/audit_sof_tdfb_design.py)与[报告](../ch05/reports/sof_tdfb_design_audit.json)读取未修改的固定源码，核对 normalized sinc 参数和 WNG 分母；数学反例是独立选择的标量/两频配置，没有执行官方 MATLAB 设计、生成 FIR 或运行固件。官网旧参数与固定版本默认值分别保存。其他外部方法的实际运行、提取调用、兼容性问题与尚未构建状态见[空间研究](research/01_spatial_and_tracking.md)；源码核验通过不能证明其中各套算法已正确运行。

### 第6章评测研究补充

[EC-Evaluation-Toolbox](https://github.com/ifnspaml/EC-Evaluation-Toolbox/tree/aec8873325ed8e4d93eadc53e5c0af84be4db4fd) 固定版本用于追溯 AEC/AES 评测与动态房间响应研究。2026-09-28 检查根目录及所列源文件头，未找到明确源码再分发许可，因此仅登记来源；不下载模型或数据，不称为已完整运行。其模型入口还依赖仓库未附的 `speechlightning`。原方法调用与教学例的差异见 [AEC 源码研究](research/02_aec_wpe_separation.md#aec)。


## 第 7 章 WPE 源码复核（2026-09-28）

| 来源与固定版本 | 本地取得范围及许可 | 已验证与未执行范围 |
|---|---|---|
| [MetaAF](https://github.com/adobe-research/MetaAF/tree/56c4665bdc51c2e0595a7c0cd9b1266408adceff) | 既有核心与声明之外，扩充 `zoo/wpe/` 四个源文件；核心 NCSA、zoo 的 Adobe 非商业研究许可分别保留 | 核心、WPE 教学研究与 AEC 的用途分别登记；原方法提取调用与静态核对见增强研究 W09/W11；未运行训练、权重或完整音频评测 |
| [TSO-VACE-WPE](https://github.com/dreadbird06/tso_vace_wpe/tree/10ee77dd020d58af508feb77251f9353208cb33a) | MIT；15 个源码/说明/许可文件，含所选 `torch_custom` 直接依赖，排除模型与数据 | 作者实现源码已取得和核验；未安装 PyTorch、载入模型、训练、推理或核验说话人验证增益 |
| [NTT WPE 评估包](https://www.rd.ntt/cs/team_project/media/signal/wpe/) | 官方 MATLAB p-code 评估包；[许可条件](https://www.rd.ntt/cs/team_project/media/signal/wpe/licence.html)与开放源码不同 | 仅网页资料索引；未下载程序、录音或模型，不加入 Git 源码锁表，不声称当代 MATLAB 兼容性已验证 |

NARA、ESPnet、BTK20 与 GSS 的 WPE 入口已有本地固定源码。详见[增强研究 W01～W11](research/02_aec_wpe_separation.md#wpe)：实际 NumPy 调用、原函数提取、静态接口追踪、长期静音反例与尚未构建的整链分别记录。某个项目取得成功，不能推断它的全部参数、默认值或数值边界已经通过运行验证。

## 第8章的固定源码与调用边界（2026-09-28）

[mfFCA 作者源码](https://github.com/nttcslab-sp/mfFCA/tree/1d6b422fc56f5f9ef612fd87ae1402e1311c5c3c)的固定版本已原样取得到本地独立目录 `codes/chapters/ch00/upstream/_downloads/mffca/`，只选择9个许可、说明、Python与合成实验notebook文件，不取录音。其 `LICENCE.txt` 是NTT内部测试、分析和评估协议，禁止修改与再分发；本书不复制其源码进公开Git，也不把它称为宽松许可证开源软件。已取得不等于已运行多帧协方差训练或论文语音实验，逐算法解释见[增强研究](research/02_aec_wpe_separation.md)。

PRA的AuxIVA、ILRMA、FastMNMF/FastMNMF2、TRINICON，ssspy的回投影，以及GSS活动约束按固定版本分别核查。原接口的短输入结果、提取函数的受控诊断与静态神经模型阅读是不同证据，不互相替代；上游边界不会通过改写缓存来隐藏。本书的严格活动门控、数值保护和双槽CSS关联有自己的实现范围，不称为官方整链复现。


### 第9章追踪源码与受限研究归档（2026-09-28）

[固定接口诊断](../ch09/reports/tracking_upstream_interfaces.json)分别记录 FilterPy 原包实际调用、Stone Soup 原方法提取调用和静态源码核对。
这些层级不能合并成“整库已验证”；外部代码保持原状，教学实现独立修正。完整输入和差异解释见[追踪研究](research/01_spatial_and_tracking.md)。

| 来源与固定范围 | 本地状态 | 许可与解释边界 |
|---|---|---|
| [Vo RFS tracking toolbox](https://ba-tuong.vo-au.com/codes.html)，`vo-rfs-tracking-updated` | 作者ZIP共565949字节；已取得62个选定MATLAB/C源码及readme，逐字节核对通过；没有执行MATLAB或编译MEX | 包readme限学术/研究用途，逐文件声明保留；不能概括为可自由再分发的开源。选集只含线性高斯CPHD、GLMB、LMB、joint变体及直接公共函数，不含预编译MEX、图或数据 |
| [SAF tracker](https://github.com/leomccormack/Spatial_Audio_Framework/tree/18fd5aba46e20787b51f28f7197a68506c965c07/framework/modules/saf_tracker) | 既有固定源码增加tracker和MEX包装入口核查，未编译执行 | 被核查tracker文件头为GPL-2.0-or-later；RBMCDA用粒子表示关联等离散变量，给定关联后用Kalman处理连续位置，不等于普通位置粒子滤波 |
| [Neural-SRP](https://github.com/egrinstein/neural_srp/tree/0ec639f028987ca3d9d3764331f6d65ff455f9a8) | 仅固定来源索引与有限选型候选；未获取源码/权重或运行 | 未建立代码许可；论文的CC BY不能代替仓库许可。训练/评测源数、阵列和任务边界不可由网络结构外推 |

Vo归档SHA-256为`fb22c9edecb56049b7f8ede1e1522384f0c4e6bfb5e5577f3bd482f6367ef46a`。
配套descriptor是作者下载页的固定快照，摘要为`a4903db625b19ed77c69b495b9655257bd3b1e7ebd3fe5d27bfcec52ef6d2984`；
这两个值均为本次取得后本地计算，不是作者公布的独立校验值或数字签名。
[归档锁表](ARCHIVE_SOURCES.lock.json)现有HARK与Vo两项，与97个Git项目分开计数。
完整压缩包仍在忽略缓存中，包含未选取资产；只有选定工作树排除了二进制、图和数据。缓存及源码工作树不随本书提交推送。
