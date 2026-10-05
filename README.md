# 深入浅出麦克风阵列信号处理

一套面向初学者和入门研究生的中文教程，依次讲解阵列信号模型、几何、声源定位（DOA）、波束形成、声学回声消除（AEC）、加权预测误差去混响（WPE）、语音分离、追踪、工程实现与选型。第 12、13 章分别扩展到声学成像与分布式增强，第 14、15 章为附录 A、B。

本书把概念定义、公式推导、数字例子、可执行实验和失效边界连接起来。全书有 **360 道可运行代码练习**、**80 张编号图**及 **1 张房间题补充图**。先读[导读](chapters/00_overview.md)，从[按章代码地图](codes/chapters/README.md)找实现，再用[练习与音频实验手册](codes/chapters/ch00/research/05_exercises_and_audio.md)核对逐题输入、答案、评分窗和代码入口。

音频分三类管理：**主清单 27 组、109 个数学合成 WAV**，**27 套独立实验、154 个合成 WAV**，以及 **4 个 DEMAND 真实录音摘录及派生 WAV**。[完整音频目录](codes/chapters/README.md#audio-inventory)逐套给出清单、生成入口和用途。267 个源 WAV 各有一个网站副本，副本不算新实验。已知目标、掩码、路径、统计量和时钟速率属于声明的真值控制，不能据此声称盲估计、自然语音质量或设备性能。[DEMAND 数据说明](codes/chapters/ch02/real_audio/README.md)单独保存数据许可和测量边界。

[源码研究手册](codes/chapters/ch00/research/README.md)把算法对应到固定源码、工业接口和实际实验，分别记录源码取得、静态检查、限定方法运行和完整系统验收。上游缓存不随本书提交，其中的 README 和原代码保留原貌。浏览器播放与 PCM 数值检查不能替代有记录的真人听测；PDF 存在结构标签也不等于已符合 PDF/UA。

English project guide: [README_EN.md](README_EN.md)

## 目录结构

| 目录/文件 | 说明 |
|---|---|
| `chapters/` | 教程正文 16 篇 Markdown（`00_overview.md` 是入口，`01`～`13` 是 13 章正文（第12、13章为扩展专题Ⅰ/Ⅱ），`14`/`15` 是附录 A/B；阅读顺序以导航为准） |
| `figures/` | 80 张编号图（`fig01`～`fig80_*.png`），房间题补充图另存于 `codes/chapters/appendix_b/room_audio/ROOM_RESULTS.png`；全部由脚本生成、可复现 |
| `codes/chapters/` | 导读、第 1～11 章、扩展专题Ⅰ/Ⅱ及附录 A/B 的源码、实验、报告和资产；逐章目录与命令见[代码地图](codes/chapters/README.md) |
| `codes/chapters/ch10/channel_audio/` | 六个已知坏麦数学合成WAV，8真实源、共同增益1；分开解析、浮点与27200点实际PCM整数评分 |
| `codes/chapters/ch00/audio/MANIFEST.json`、各章 `audio/` | 主清单统一管理 27 组、109 个按章节存放的本书合成 WAV；由脚本生成，不直接编辑 |
| `codes/chapters/ch01/spectral_audio/` | 两种数学源谱及已知两抽头响应，2 个单声源与2个双声完整尾输出；独立清单与解析/浮点/PCM评分 |
| `codes/chapters/ch02/sweep_audio/` | 已知数字扫频、完整/加噪/截尾三响应共4WAV；实际PCM与浮点的完整IR、真值支持及支持外能量分别评分，不是房间测量 |
| `codes/chapters/ch03/baseline_audio/` | 已知方向求三维基线及固定通道时延的6WAV；增广秩、真实PCM相量与未参与拟合方向分开核验 |
| `codes/chapters/ch04/reflection_audio/` | 同源参考/直达/反射/总观测四WAV；高相干与秩一的有限假峰，解析/浮点/实际PCM分开 |
| `codes/chapters/ch01/binaural_audio/`、`codes/chapters/ch08/gss_audio/`、`codes/chapters/ch09/moving_audio/`、`codes/chapters/ch09/tracking_audio/`、`codes/chapters/appendix_b/room_audio/` | 五套独立合成实验资产，分别为 5、5、3、2、18 个 WAV；中间状态、轨迹/逐帧观测或房间数值报告随各自清单保存 |
| `codes/chapters/ch02/stft_audio/`、`codes/chapters/ch03/geometry_audio/` | 另外两套独立合成实验，各3个WAV；完整卷积与稳定窗多频相位分别评分，生成源和PCM摘要随各自清单保存 |
| `codes/chapters/ch04/focus_audio/` | 另外四份独立已知酉聚焦合成WAV；两份单声道源、两份四声道观测，固定稳定窗检查双频池化秩 |
| `codes/chapters/ch05/derivative_audio/` | 四份独立三麦导数约束WAV；八个真实生成源，固定参考的实际PCM误差 |
| `codes/chapters/ch05/phase_audio/` | 三个已知逐频相位控制，分开相位、功率和实际 PCM 整数误差 |
| `codes/chapters/ch06/apa_audio/` | 六份有色参考APA合成WAV；训练/冻结留出、完整13点尾及浮点/PCM评分 |
| `codes/chapters/ch06/reference_audio/` | 六份已知播放增益/尾声WAV；同钟冻结路径、完整1600点尾及分窗整数评分 |
| `codes/chapters/ch07/delay_audio/` | 六份已知到达与历史WAV；同一接收时轴、单实回归、完整表示尾及整数评分 |
| `codes/chapters/ch07/mint_audio/` | 六份已知稀疏路径逆合成WAV；完整512点尾、共同增益及独立参数/评分清单 |
| `codes/chapters/appendix_b/response_audio/` | 五份同DRR已知短FIR频响对照WAV；四真实生成源、完整尾部、共同增益与实际PCM整数评分 |
| `codes/chapters/ch08/css_audio/` | 四个既定槽位极性与增益控制；重叠拟合与两个评分窗分开，不是盲 CSS |
| `codes/chapters/ch08/mask_audio/`、`codes/chapters/ch10/noise_audio/` | 两套独立数学合成实验，各6个WAV；已知掩码表示与固定噪声估计失配分别记录解析、浮点和实际PCM评分 |
| `codes/chapters/ch11/scenario_audio/`、`codes/chapters/appendix_a/weighted_audio/` | 选型场景8个WAV与已知噪声加权5个WAV；各有独立清单、共同增益、评分窗和整数分母 |
| `codes/chapters/ch13/distributed_audio/` | 17份独立16kHz数学WAV，共同增益1；28800点稳窗与800点缺口窗分别评分，绑定九个真实源 |
| `codes/chapters/ch12/imaging_audio/` | 五份独立24kHz成像快拍WAV，真实源摘要、48004点完整记录及38160点评分快拍集随清单保存 |
| `codes/chapters/ch00/io_contracts.py` | 生成器和源码获取工具共用的文件路径、普通成员、严格JSON及报告写入检查；数值模型和各资产清单仍由所属章维护 |
| `codes/chapters/ch02/real_audio/` | DEMAND 真实同步录音摘录、派生均值、独立清单与数据许可；不是合成数据 |
| `codes/chapters/ch00/research/` | 详细源码研究手册：算法步骤、状态与配置、代码入口、失败实验和工业复现 |
| `codes/chapters/ch00/upstream/_downloads/` | 本机按需取得的第三方源码缓存，已被 Git 忽略；可能含本地修改，不属于本书提交的文档或教学代码，取得与核验方法见[源码获取说明](codes/chapters/ch00/upstream/README.md) |
| `codes/chapters/*/reports/` | 随算法所属章保存的小规模运行报告；与源码获取状态、论文全量评测分开 |
| `scripts/` | 绘图与构建脚本（`make_figures.py`、`make_aec_figures.py`、`make_beamforming_figures.py`、`make_reference_figures.py`、`make_delay_figures.py`、`make_css_figures.py`、`make_tracking_figures.py`、`make_channel_figures.py`、`make_selection_figures.py`、`build_site.py`、`build_pdf.py`，说明见 `scripts/README.md`） |
| `site/` | 26 个网页：16 个当前教程页（含首页）、4 个旧路径兼容页面及 `research/` 下 6 个研究手册页；构建产物，可再生 |
| `dist/` | [当前合订 PDF](dist/microphone-array-tutorial.pdf) 与可再生的合订 HTML；PDF 含 16 个顶级、151 个二级、740 个三级书签，共 907 个 |

## 章节导览

| 篇 | 文件 | 内容 | 难度 |
|---|---|---|---|
| 导读 | `chapters/00_overview.md` | 全套导航、三条学习路径、插图地图 | 入门 |
| 第 1 章 | `chapters/01_problem-definition.md` | 噪声、混响、干扰、自噪声，阵列增益与双耳线索 | 入门 |
| 第 2 章 | `chapters/02_basics-signal-model.md` | 时延、远近场、信号模型、房间混响、STFT/协方差、波束图度量 | 进阶 |
| 第 3 章 | `chapters/03_array-geometry.md` | 线阵、圆阵、球阵、稀疏阵，端射灵敏度与阵列校准 | 进阶 |
| 第 4 章 | `chapters/04_doa-estimation.md` | GCC-PHAT、SRP、几何解算、Bartlett/Capon、MUSIC/ESPRIT、宽带聚焦、DNN 定位、CRLB | 较难 |
| 第 5 章 | `chapters/05_beamforming.md` | DSB、超指向、MVDR、LCMV、GSC、SPP、后置滤波、球谐、DNN 波束 | 较难 |
| 第 6 章 | `chapters/06_aec.md` | NLMS、PBFDAF、子带 AEC、IPNLMS、RLS、Kalman/FDKF、双讲检测、非线性、混合神经 AEC、AEC3 | 较难 |
| 第 7 章 | `chapters/07_wpe-dereverberation.md` | WPE 原理与推导、Δ/K 选择、在线 WPE、手算例 | 进阶 |
| 第 8 章 | `chapters/08_speech-separation.md` | 混合模型、BSS、GSS、深度分离、TSE、连续会议分离与数据集 | 进阶 |
| 第 9 章 | `chapters/09_source-tracking.md` | KF 手算、粒子滤波、PHD 多目标、定位—追踪—波束接口 | 进阶 |
| 第 10 章 | `chapters/10_engineering-practice.md` | 参考链路、关键路径延迟、SRO/标定、资源预算与评测 | 进阶 |
| 第 11 章 | `chapters/11_selection-guide.md` | 条件化选型、场景约束、可验证规格与练习 | 入门 |
| 第 12 章（专题Ⅰ） | `chapters/12_acoustic-imaging.md` | CSM、球面扫描、PSF、DAMAS/CLEAN-SC/简化CMF、校准与区域量；18道逐步代码题 | 较难 |
| 第 13 章（专题Ⅱ） | `chapters/13_distributed-enhancement.md` | 节点特定MWF、任务相关压缩、DANSE条件、树拓扑、GEVD、时钟与缺口；25题与17个独立音频 | 较难 |
| 第 14 章（附录 A） | `chapters/14_appendix-symbols-math.md` | 符号表、术语定义、预备数学速览 | 查阅 |
| 第 15 章（附录 B） | `chapters/15_appendix-guide.md` | 学习路径、领域地图、研究前沿、排错、17 道综合书面题与 E15-01～15 代码题、复现说明 | 查阅 |

## 快速开始

以下命令从仓库根目录运行。本机验证环境为 Python 3.13；主依赖的精确版本见 [requirements.txt](requirements.txt)。Windows 将 `.venv/bin/python` 换成 `.venv\Scripts\python.exe`。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m codes.chapters.ch00.cross_chapter.ch02_05_baselines
.venv/bin/python -m codes.chapters.ch00.cross_chapter.ch06_09_baselines
.venv/bin/python -m codes.chapters.ch10.examples.ch10_engineering_baselines
.venv/bin/python -m codes.chapters.ch04.chapter04_experiments
.venv/bin/python -m codes.chapters.ch12.chapter12_exercises
.venv/bin/python -m codes.chapters.ch13.chapter13_exercises
```

这些入口打印各自结果。其余单章和跨章入口见代码地图与练习手册，一个入口不会执行全部 360 题。部分题目读取随仓 PCM；源码、清单或音频过期时会失败。

仓库已包含发布音频与图片。使用这些资产重建网站和 PDF：

```bash
.venv/bin/python scripts/build_site.py
.venv/bin/python scripts/build_pdf.py --build-date 2026-10-05
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/quality_check.py
```

教程入口为 `site/index.html`，研究手册入口为 `site/research/index.html`。网站公式需要联网加载 MathJax；PDF 使用本地固定 MathJax 3.2.2，仍需 Google Chrome。非默认 macOS 安装位置用 `CHROME_BIN` 指向 Chrome 可执行文件。封面日期按实际出版日期设置，也可用 `SOURCE_DATE_EPOCH`；固定日期不等于 PDF 每个二进制字节都能跨环境重现。

完整重生成按[音频与绘图的执行顺序](scripts/README.md#audio-regeneration)进行：先生成主清单和 26 套 NumPy 独立实验，再运行九个绘图入口；另 1 套房间实验使用隔离的 `pyroomacoustics==0.10.0`，DEMAND 则从固定归档单独准备。不要手改单个 WAV 或清单来绕过核验。普通生成器的 `--check` 只读；房间资产检查用 `check_room_assets`，真正仿真另用 `--run`。

缺少 SciPy、固定上游缓存、编译器或 FFTW 时，可选测试会跳过。测试命令退出成功仍可能含未执行项，应查看具体原因并按[可选核验说明](scripts/README.md#optional-verification)补齐；安装主依赖不会自动安装全部上游系统。

## 原始实现与证据

```bash
.venv/bin/python codes/chapters/ch00/upstream/fetch_upstreams.py --list
.venv/bin/python codes/chapters/ch00/upstream/fetch_upstreams.py --all --report tmp/source-acquisition.json
.venv/bin/python codes/chapters/ch00/upstream/fetch_upstreams.py --verify --report tmp/source-verification.json
```

获取需要 Git 和网络。离线核验保留选集不匹配或有本地修改的缓存并报告失败，不自动修复，也不执行算法。官方非 Git 归档采用独立锁表与工具。具体命令、来源与许可见[获取说明](codes/chapters/ch00/upstream/README.md)、[算法覆盖表](codes/chapters/ch00/COVERAGE.md)和[第三方记录](codes/chapters/ch00/THIRD_PARTY.md)。

根目录没有 `LICENSE` 或 `COPYING`；公开可见不自动授予本书或教学代码的再分发许可。上游代码、模型权重与数据各有独立条款，DEMAND 派生物沿用其文档中的 CC BY-SA 3.0。

## 学习路径

- **基础**：导读 → 第 1 章 → §11.1～§11.3 → 第 2、3 章 → 第 4 章 GCC/SRP → 第 5 章 DSB/MVDR → 第 6～9 章。完成相应例子，并能解释成立条件。
- **工程**：第 11 章写任务和验收条件 → 第 5～8 章选模块 → 第 10 章核工程接口 → 输出延迟、同步、标定预算。硬件时序和声学质量仍在目标系统测量。
- **研究**：第 2 章信息界 → 第 3 章几何 → 第 4～9 章相关方法 → [§15.3](chapters/15_appendix-guide.md#153-研究前沿速览2024) 与 [§15.6](chapters/15_appendix-guide.md#156-思考与练习)。源量成像读第 12 章，分布式波形增强读第 13 章；比较结果前固定数据、版本与评分协议。

## 约定与验收边界

核心引用公式使用 `\tag{章-序}`，正文写“见式(5-1)”。计算前说明维度、单位、共轭、坐标和时差正号。功率比用 10 log10；功率与幅度平方成正比时，幅度比用 20 log10。每项数字保留输入、参考、对齐、评分区域与来源。

合订PDF使用离线SVG公式与固定英文结构说明，逐式核源身份和真实MCID；公式含义与辅助技术阅读顺序仍需人工核验。音频技术检查覆盖267份发布源，真人听辨登记尚未执行；材料见[音频验收说明](codes/chapters/README.md#音频技术检查与真人听辨记录)。

发布门禁检查结构、链接、源摘要与 PDF 目标，不据此声明主观音质、辅助技术阅读顺序或完整 PDF/UA 已通过。具体检查和限制见[构建说明](scripts/README.md)。
