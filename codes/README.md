# 与正文配套的教学代码

本目录把书中的公式变成可以检查形状、单位和边界情况的最小程序。目标是帮助读者复算，不是提供一个
可直接装进产品的音频前端。实时音频线程、设备驱动、线程调度、定点优化、模型权重和现场标定仍需按
第 10 章单独完成。

外部研究代码由 [锁定清单](SOURCES.lock.json) 管理：73 个项目中，63 个已在本机的
`upstream/_downloads/` 取得，另 10 个保留来源索引。本轮离线核验有 62 项通过，AEC Challenge 工作区的 5 个真实录音文件处于本地修改状态，未把该项记为通过；见 [SOURCE_STATUS.json](SOURCE_STATUS.json)。独立源码目录被 Git 忽略，不会随本书提交上传。
取得代码、完成构建和复现数值是不同状态，见[复现记录说明](research/04_source_reproduction.md)。

已有可选 nara-wpe 0.0.11 环境时，可运行
`.venv/bin/python codes/examples/compare_online_wpe_reference.py` 检查在线接口的时间索引及状态续接；
脚本校验固定源码摘要，不联网安装依赖。离线对照另见 `examples/compare_wpe_reference.py`。

已取得锁定的 libsoxr、libebur128、libsndfile 源码，并具备 CMake 与 clang 时，可显式运行
`.venv/bin/python codes/examples/run_industrial_interfaces.py --report tmp/industrial-rerun.json`。
它在忽略的临时目录构建静态库，检查分块、排空、响度及 PCM 读写，不安装系统依赖。
已取得锁定的 libsamplerate 0.2.2 源码时，可另运行 `.venv/bin/python codes/examples/run_libsamplerate_sro.py`，
复核两段已知时钟偏差下的变比调用、部分消费和尾部排空；这不是时钟估计闭环或真实设备测试。
SBL 使用 `.venv/bin/python codes/examples/reproduce_sbl_reference.py --output tmp/sbl-rerun.json`，
只调用独立目录中的固定作者实现。已有结果、失败条件与未执行范围见[复现记录](research/04_source_reproduction.md)。

SMP-PHAT 可用 `.venv/bin/python codes/examples/reproduce_smpphat_reference.py --download-fftw --output tmp/smpphat-rerun.json`
显式下载并隔离构建固定 FFTW 依赖，再编译作者 C 核心。已有单精度静态库时可用 `--fftw-prefix` 指定前缀，
脚本会记录实际库与头文件摘要，而不会自动声称该库来自固定归档。本机原版发生负时延索引转换错误，
因此默认保存失败报告并返回非零状态；`--allow-known-portability-failure` 只允许收集诊断时返回零，
不改变报告中的失败结论，也不修补上游源码。

## 目录

| 路径 | 内容 |
|---|---|
| `array_tutorial/` | 本书自行编写的 NumPy/标准库教学实现 |
| `examples/` | 按章节组织的可运行例子，打印输入口径、中间量和结果 |
| `reports/` | 显式运行外部接口与算法后保存的数值报告；记录输入、固定版本及环境，不等同于源码获取状态 |
| `audio/` | 11 组、44 个本书合成 WAV 及 `MANIFEST.json`；由音频生成器产生，不直接编辑 |
| `real_audio/` | 真实 DEMAND 河流录音摘录与 3 个派生 WAV，独立记录 CC BY-SA 3.0 数据许可 |
| `upstream/` | 第三方官方仓库的按需获取工具；下载内容默认不入 Git |
| `COVERAGE.md` | 正文算法到代码、测试和外部实现的逐项映射 |
| `THIRD_PARTY.md` | 官方项目的用途、许可证、工程边界和选择建议 |
| `SOURCES.lock.json` | 官方地址、完整提交哈希和核实日期的机器可读清单 |
| [research/](research/README.md) | 空间处理、AEC/WPE/分离、工业部署、源码复现、练习与音频实验；入口加 5 篇专题，共 6 页 |

## 运行

在仓库根目录、已安装 `requirements.txt` 的环境中运行：

```bash
.venv/bin/python codes/examples/ch02_05_baselines.py
.venv/bin/python codes/examples/ch06_09_baselines.py
.venv/bin/python codes/examples/ch10_engineering_baselines.py
.venv/bin/python -m codes.examples.aec_rls_demo
.venv/bin/python -m codes.examples.aec_kalman_matrix_demo
.venv/bin/python -m codes.examples.exercises_spatial
.venv/bin/python -m codes.examples.exercises_enhancement
.venv/bin/python -m codes.examples.exercises_engineering
.venv/bin/python -m unittest discover -s tests -p 'test_codes*.py' -v
```

例子只使用确定性输入，随机输入会固定种子。函数拒绝维度、单位或参数范围明显错误的输入；这类检查是
为了尽早暴露口径错误，不表示代码已经达到产品级防御能力。

三个 `exercises_` 模块分别有 26、22、20 道题，AEC 边界小例另有 4 道，共 72 道，使用 `E01-01` 至 `E13-01` 等稳定题号，不改原有练习编号。前三个模块的 `run_exercises()` 与 AEC 小例的 `run_demo()` 均返回可序列化为 JSON 的计算结果，导入模块不会执行练习。题目与测试映射见 [COVERAGE.md](COVERAGE.md)，逐题入口与音频对照见[练习与音频实验](research/05_exercises_and_audio.md)。练习数量与算法数量分开统计；MDL、流式 NLMS、逐样本 RLS 和短 FIR 矩阵 Kalman 等另有可运行教学实现，当前基线算法共 45 行。

第 6 章新增的 [RLS 两抽头演示](examples/aec_rls_demo.py)与[矩阵 Kalman 两抽头演示](examples/aec_kalman_matrix_demo.py)只核算带已知参数的状态递推；[研究手册 A04](research/02_aec_wpe_separation.md#aec)逐项列出外部 RLS、FDKF/PBFDKF 与商业 Kalman 模块的源码入口、许可及尚未完成的对照实验。教学代码不含延迟搜索、双讲检测、残余抑制或设备接入。

## 合成音频与图 34～36

在仓库根目录先生成音频，再绘图；图 34～36 读取写入 WAV 后的样本：

```bash
.venv/bin/python codes/examples/generate_audio_samples.py
.venv/bin/python scripts/make_figures.py
.venv/bin/python codes/examples/generate_audio_samples.py --check
```

44 个音频文件分为空间处理、四麦分数采样时差、AEC、WPE、给定矩阵解混、工程失真、追踪、相关噪声、极性错误、病态求逆和非线性回声 11 组，均为本书合成的 16 kHz、PCM16 信号，这 44 个文件不含第三方录音；真实录音使用独立的 `real_audio/` 目录。每组共用一个增益，避免逐文件归一化掩盖幅度差异；清单记录参数、随机种子、生成源文件摘要和 WAV 摘要。`--check` 检查当前生成物，不重写文件，也不自动播放音频。

样例用于观察时延、残留回声、混响、混合和削波等现象，不是自然语音质量评测。给定混合矩阵的求逆不是盲分离；已知双讲区间的冻结不是双讲检测器。这些限制及试听顺序见[音频实验说明](research/05_exercises_and_audio.md)。合成文件的来源说明不等于授予新的再分发许可，许可边界仍见下节。

## 真实录音实验 R01

`real_audio/` 保存 DEMAND 河流环境的 10 s 同步 16 通道摘录与 3 个派生文件。它们是第三方真实录音，不是上一节的合成资产；摘录、单麦提取和均值音频均保留 CC BY-SA 3.0 数据许可。没有干净语音参考、声源位置真值或麦间增益校准，实验不评价语音增强性能。

普通运行与检查均离线、只读：

```bash
.venv/bin/python codes/examples/prepare_real_recordings.py --check
```

从固定的约 99 MB 官方归档重建时，下载必须显式选择：

```bash
.venv/bin/python codes/examples/prepare_real_recordings.py --download
```

归档默认保存在不进入 Git 的 `codes/upstream/_downloads/demand/`。已有文件若摘要不符会被拒绝，不会被静默覆盖。完整输入口径、10 个连续 1 s 子段结果和限制见[数据说明](real_audio/README.md)及[练习手册](research/05_exercises_and_audio.md)。

## 真实成对 AEC 录音实验 R02

[aec_real_pair_experiment.py](examples/aec_real_pair_experiment.py)对 Microsoft AEC Challenge 官方固定版的一对播放环回/麦克风录音运行 SpeexDSP 同步 AEC，并分别设置正确、全零和故意晚 1 秒的参考。录音没有独立干净回声真值；报告的是固定评分区的输入/输出数字功率变化，不是真值 ERLE。[完整文件摘要、构建命令、数值和限制](research/02_aec_wpe_separation.md#aec)保存在研究记录。真实众包录音的再分发授权尚不明确，因此原文件和可选处理后 WAV 只放在 Git 忽略缓存中，不纳入本仓库的 44 个合成音频或 DEMAND 的 4 个已授权 WAV。

## 如何把公式和程序对上

1. 先在 `COVERAGE.md` 找到正文小节、函数和测试。
2. 看函数文档中的数组形状。`spectral.py`、`covariance.py`、`doa.py` 使用 `C × F × T`（通道 × 频点 × 帧）；`dereverberation.py` 与 `separation.py` 的多通道输入使用 `F × C × T`。跨这两组模块时要显式执行 `spectrum.transpose(1, 0, 2)`，不能直接传递同一数组。
3. 核对角度零点、时延正负、导向矢量相位和共轭约定；不要只比较最后一个数。
4. 运行相应例子，再改变一个条件观察边界。数值相同只说明当前输入下相符，不证明实现普遍正确。

## 原创代码与第三方代码

`array_tutorial/` 中的程序是为本书编写的教学基线。仓库根目录目前没有明确的 `LICENSE` 或 `COPYING`
文件，因此“能看到源码”不等于已经得到复制、修改或再分发许可。对外发布或复用前，项目维护者需要先
选择并添加明确许可证；在此之前不要把本目录的代码自动并入其他产品。

`SOURCES.lock.json` 固定已核实的官方仓库和提交；`upstream/fetch_upstreams.py` 把可获取的源码检出到
`codes/upstream/_downloads/`，每个项目保持独立工作树并保留许可。大型框架也可以在这里阅读源码，
该目录被本仓库 Git 忽略。第三方代码许可证不自动覆盖模型和数据，也不改变本仓库自身的许可状态。

列出索引：

```bash
.venv/bin/python codes/upstream/fetch_upstreams.py --list
```

按需获取一个体量适中的项目：

```bash
.venv/bin/python codes/upstream/fetch_upstreams.py --project nara_wpe
```

批量获取和离线核对：

```bash
.venv/bin/python codes/upstream/fetch_upstreams.py --all --report tmp/source-acquisition.json
.venv/bin/python codes/upstream/fetch_upstreams.py --verify --report tmp/source-verification.json
```

报告区分源码核对、缺少入口和获取失败；依赖构建与算法运行另行验收。新下载省略常见权重、音频和归档文件，
不自动取得 LFS 或子模块。已有目录完整保留。具体筛选范围见 [获取工具说明](upstream/README.md)。
WebRTC 和训练框架仍需各自的构建工具、依赖、模型或数据，取得主源码库只是复现的一步。
