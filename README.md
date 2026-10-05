# 深入浅出麦克风阵列信号处理

一套写给初学者和入门研究生的麦克风阵列信号处理中文教程。从“为什么摆一群麦克风”讲到定位（DOA）、波束形成、回声消除（AEC）、去混响（WPE）、语音分离、声源追踪，一直到工程实现与选型。

正文提供关键公式推导、可复算例子、适用边界，以及与各章公式对应的 NumPy/标准库教学代码。

全书有 360 道可运行代码练习、80 张脚本生成的编号图和 1 张房间题补充图，以及 27 组共 109 个[主清单合成 WAV](codes/chapters/ch00/audio/MANIFEST.json)。[练习与音频实验手册](codes/chapters/ch00/research/05_exercises_and_audio.md)逐题列出输入、答案、试听条件与代码入口。音频是数学合成样本，不是真实语音或正式听测；第 6 章两组合成 AEC 音频的参数也不完全等同于 E06-07～20 手算题。

第四章另有[四份已知酉聚焦合成WAV](codes/chapters/ch04/focus_audio/MANIFEST.json)：用稳定窗逐频复幅度与实际PCM检查相干双源的多频协方差秩；指定角度和频率分量已知，不作为盲定位或正式听测结果。

第五章另有[四份导数约束三麦合成WAV](codes/chapters/ch05/derivative_audio/MANIFEST.json)：同一两音和观测噪声输入比较单约束与零导数权重，分开目标失真、噪声代价和实际PCM总误差。共同因果参考不作后验增益或时移补偿；不是语音或设备实录。

第六章另有[六份有色参考APA合成WAV](codes/chapters/ch06/apa_audio/MANIFEST.json)：同源NLMS与APA的训练后冻结比较，保留13点非零卷积尾；无噪声和含噪声的浮点分量、8000点实际PCM总功率分别报告。较高投影阶数在本次含噪条件下有更大留出误差，不能据此作设备或语音质量排名。

第六章的[六份播放增益与尾声音频](codes/chapters/ch06/reference_audio/MANIFEST.json)配合E06-40～42与图74：同一已知路径冻结，比较早参考、预测后乘当前增益和晚参考。完整100ms尾声保留，三个评分窗的解析、浮点与实际PCM整数功率分别记录；当前参考静音时的`near_only`活动标签不能证明近端有人讲话。

第七章的[六份到达与历史音频](codes/chapters/ch07/delay_audio/MANIFEST.json)配合E07-22～24与图75，逐项核对历史对应的源时刻，再用固定单实回归比较目标误消。到达真值已知，不是盲对齐或完整STFT-WPE；晚期功率与幅度衰减另作单位换算。

第七章另有[六份已知路径逆合成WAV](codes/chapters/ch07/mint_audio/MANIFEST.json)：同一四音与后路径噪声比较精确逆和约束正则。保留完整512点尾，理论噪声增益与27200点实际PCM总误差分开；不是盲WPE、实测房间或一般MINT实现。

第八章另有[六份已知掩码表示合成WAV](codes/chapters/ch08/mask_audio/MANIFEST.json)：同一两音比较有界实掩码、无界实掩码和复掩码。全记录已知频点处理后使用共同包络和增益，27200点实际PCM误差与解析误差分开；未从混合录音估计掩码。

第十章另有[六份噪声估计失配WAV](codes/chapters/ch10/noise_audio/MANIFEST.json)：同一个目标与突变噪声比较纯前奏固定估计、目标污染估计和已知方差离线对照。浮点目标损伤、残余噪声及交叉项与实际PCM总误差分开；对照使用额外真值，没有运行盲自适应估计。

第十一章另有[八份双场景选型WAV](codes/chapters/ch11/scenario_audio/MANIFEST.json)：单音与双音目标共享3500 Hz干扰，比较3/9抽头FIR。保留完整8点尾，以实际PCM整数误差分别检查固定场景权重、权重区间最坏值和逐场景最坏值；不是设备性能排名。

第十一章新增[评分事件与物理候选综合题](chapters/11_selection-guide.md#e11-26)：合法空输出、缺失与失败分开计数；同一三麦模型先核WNG和复响应幅差，再比较噪声与NMSE。图79明确设备延迟未测，不能直接宣布选型完成。原评分接口与两个C++时间核的当前报告保留限定执行和失败，历史两报告不改写。

附录 A 另有[五份已知噪声权重WAV](codes/chapters/appendix_a/weighted_audio/MANIFEST.json)：同一双通道合成输入比较等权、正确方差权重和反置权重。28800点稳定窗分别保存解析、浮点分量与实际PCM整数误差；不估计协方差，也不把纯音正交当作随机独立或语音质量结果。

附录 B 另有[五份同DRR频响对照WAV](codes/chapters/appendix_b/response_audio/MANIFEST.json)：两组已知短FIR反射能量相同，在同一2/4 kHz双音上产生不同频响和误差。完整两点尾、共同增益1及28800点解析/浮点/实际PCM评分分开；不是实测房间或语音听测。

第三章另有[多频几何音频](codes/chapters/ch03/geometry_audio/MANIFEST.json)：32 kHz 的双频源及两方向六通道观测，共3个独立数学样本；稳定窗的实际PCM相位拟合与完整条件见手册。

另有十套独立管理的合成资产：[双耳时间差与声级差](codes/chapters/ch01/binaural_audio/MANIFEST.json) 5 个双声道 WAV、[方向谱形与源谱](codes/chapters/ch01/spectral_audio/MANIFEST.json) 4 个已知 FIR 数学样本、[有限窗STFT卷积](codes/chapters/ch02/stft_audio/MANIFEST.json) 3 个单声道 WAV、[已知激励路径辨识](codes/chapters/ch02/sweep_audio/MANIFEST.json) 4 个数字扫频与响应 WAV、[已知方向基线与时延](codes/chapters/ch03/baseline_audio/MANIFEST.json) 6 个源/训练/留出 WAV、[GSS 教学链](codes/chapters/ch08/gss_audio/MANIFEST.json) 5 个 WAV 与状态、[自由场移动声源](codes/chapters/ch09/moving_audio/MANIFEST.json) 3 个 WAV 与轨迹真值、[观测到追踪](codes/chapters/ch09/tracking_audio/MANIFEST.json) 2 个 WAV 与逐帧观测，[相干反射假峰](codes/chapters/ch04/reflection_audio/MANIFEST.json) 4 个独立WAV，以及[附录 B 房间题](codes/chapters/appendix_b/room_audio/MANIFEST.json) 18 个白噪声 WAV、结果图和[数值报告](codes/chapters/appendix_b/room_audio/RESULTS.json)。它们均不并入主 109 个样本。另有 DEMAND 真实同步录音摘录及 3 个派生 WAV，[数据说明与许可](codes/chapters/ch02/real_audio/README.md)独立保存。

从[按章代码地图](codes/chapters/README.md)查找每章的教学实现、实验和报告；其中的数值、合成波形与外部接口诊断各有适用边界。第 6 章的 [SpeexDSP 真实配对录音接口实验](codes/chapters/ch00/research/02_aec_wpe_separation.md#aec)仅使用本地缓存，不再分发录音或声称真值 ERLE。[源码研究手册](codes/chapters/ch00/research/README.md)详列算法实现、工业配置、原始来源、许可、已运行实验与尚未验证的范围。

English version: [README_EN.md](README_EN.md)

扩展专题Ⅱ另有[17份分布式增强WAV](codes/chapters/ch13/distributed_audio/MANIFEST.json)：同一四麦已知即时混合比较本地、集中式、压缩方向、已知采样率校正与两种缺口策略。解析、浮点分量与实际整数PCM评分分开；不是盲DANSE或联网硬件运行。

扩展专题Ⅰ另有[五份成像快拍WAV](codes/chapters/ch12/imaging_audio/MANIFEST.json)：同源、共同增益1、完整4点传播尾，比较指定快拍集交叉项抵消与完全相干。实际PCM、浮点和解析CSM分开；播放混合不能证明源强反演正确。

扩展专题Ⅰ新增[E12-17/18](chapters/12_acoustic-imaging.md#e12-17)：逐项推导DAMAS与完整CSM拟合的联系，并用不同传播列展示整体不可辨识。图80在同一输入下比较两种目标；固定作者原模块九个控制保留8数值吻合与1原异常，源码取得和完整工业实验分开。

第8章新增[四份既定槽位极性与增益音频](codes/chapters/ch08/css_audio/MANIFEST.json)，配合E08-30～32与图76，分别核对一致性顺序、PCA目标丢失和重叠回归。音频没有执行盲CSS。

第9章补充E09-24～26与图77，逐步计算完整IMM模式密度、连续缺测、实际PCM的单槽生命周期，以及已知姿态和静态vMF条件化。它们复用原五个独立WAV，不以局部ID证明说话人身份；代码与边界见[章节练习](codes/chapters/ch00/research/05_exercises_and_audio.md#tracking-model-lifecycle-sphere-exercises)。

第10章补充E10-34、六份独立已知坏麦WAV与图78，逐步重建选集和保目标约束，分开目标损伤、残余噪声与实际PCM总误差；遥测区分累计RTF与逐帧服务RTF。原工业接口改用四份当前报告，旧历史报告保留。[实验条件与整数评分](codes/chapters/ch00/research/05_exercises_and_audio.md#known-channel-failure-exercise)可复算。

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
| `codes/chapters/ch06/apa_audio/` | 六份有色参考APA合成WAV；训练/冻结留出、完整13点尾及浮点/PCM评分 |
| `codes/chapters/ch06/reference_audio/` | 六份已知播放增益/尾声WAV；同钟冻结路径、完整1600点尾及分窗整数评分 |
| `codes/chapters/ch07/delay_audio/` | 六份已知到达与历史WAV；同一接收时轴、单实回归、完整表示尾及整数评分 |
| `codes/chapters/ch07/mint_audio/` | 六份已知稀疏路径逆合成WAV；完整512点尾、共同增益及独立参数/评分清单 |
| `codes/chapters/appendix_b/response_audio/` | 五份同DRR已知短FIR频响对照WAV；四真实生成源、完整尾部、共同增益与实际PCM整数评分 |
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

```bash
# 1. 建虚拟环境并装依赖
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# 可选房间仿真使用独立环境中的 pyroomacoustics 0.10.0，见 scripts/README.md

# 2. 运行与正文对应的教学基线及回归测试
.venv/bin/python codes/chapters/ch00/cross_chapter/ch02_05_baselines.py
.venv/bin/python codes/chapters/ch00/cross_chapter/ch06_09_baselines.py
.venv/bin/python codes/chapters/ch10/examples/ch10_engineering_baselines.py
.venv/bin/python -m codes.chapters.ch00.cross_chapter.exercises_spatial
.venv/bin/python -m codes.chapters.ch00.cross_chapter.exercises_enhancement
.venv/bin/python -m codes.chapters.ch06.aec_advanced_exercises
.venv/bin/python -m codes.chapters.ch06.aec_algorithm_minicases
.venv/bin/python -m codes.chapters.ch00.cross_chapter.exercises_engineering
.venv/bin/python -m unittest discover -s tests -p 'test_codes*.py' -v

# 3. 先生成 109 个音频，再生成 80 张图（图 34～36、40～41、43～45、47、49、58～60、63～66 读取生成的音频）
.venv/bin/python codes/chapters/ch00/examples/generate_audio_samples.py
.venv/bin/python -m codes.chapters.ch06.examples.generate_apa_audio
.venv/bin/python -m codes.chapters.ch06.examples.generate_reference_audio  # 六份已知播放增益/尾声WAV；--check严格只读
.venv/bin/python -m codes.chapters.ch07.examples.mint_teaching_demo
.venv/bin/python -m codes.chapters.ch07.examples.generate_delay_audio  # six known arrival/history WAVs; --check只读
.venv/bin/python -m codes.chapters.ch08.examples.mask_representation_demo
.venv/bin/python -m codes.chapters.ch10.examples.generate_noise_mismatch
.venv/bin/python -m codes.chapters.ch10.examples.generate_channel_audio  # 已知坏麦六独立WAV与清单
.venv/bin/python -m codes.chapters.ch11.examples.generate_selection_audio  # 8 份独立场景 WAV；--check 只核对
.venv/bin/python -m codes.chapters.appendix_a.examples.generate_weighted_audio  # 5 份独立已知噪声权重 WAV；--check 只核对
.venv/bin/python -m codes.chapters.appendix_b.examples.generate_response_audio  # 5 份独立同DRR频响对照 WAV；--check 只核对
.venv/bin/python -m codes.chapters.ch12.examples.generate_imaging_audio  # 五份快拍WAV；--check只读
.venv/bin/python -m codes.chapters.ch13.examples.generate_distributed_audio  # 17 independent WAVs; --check只读
.venv/bin/python -m codes.chapters.ch05.examples.generate_phase_audio  # E05-23; --check is read-only
.venv/bin/python scripts/make_figures.py
.venv/bin/python scripts/make_aec_figures.py
.venv/bin/python scripts/make_beamforming_figures.py  # figure 73: checked actual PCM
.venv/bin/python scripts/make_reference_figures.py  # 图74：实际PCM播放增益与尾声控制
.venv/bin/python scripts/make_delay_figures.py  # 图75：已知到达时间与实际PCM历史控制
.venv/bin/python scripts/make_css_figures.py  # figure 76: fixed-slot polarity/gain and actual PCM
.venv/bin/python scripts/make_tracking_figures.py  # 图77：实际PCM生命周期与两个龄期时钟
.venv/bin/python scripts/make_channel_figures.py  # 图78：已知坏麦、重建约束与实际PCM
.venv/bin/python scripts/make_selection_figures.py  # 图79：同阵列模型的硬筛选与设备证据边界
.venv/bin/python -m codes.chapters.ch10.examples.generate_channel_audio --check  # 六独立WAV只读核验

# 4. 建多级页面站（输出 site/*.html）
.venv/bin/python scripts/build_site.py
# 浏览器打开 site/index.html（双击即可；公式需联网加载 MathJax 渲染）

# 5. 生成合订 PDF（公式资源已固定在 scripts/vendor/mathjax-3.2.2/，无需联网；
#    需本机装有 Google Chrome，非 macOS 可用 CHROME_BIN 指定 Chrome 路径）
.venv/bin/python scripts/build_pdf.py
# 需要可复现的封面日期时，加 --build-date YYYY-MM-DD，或设置 SOURCE_DATE_EPOCH

# 6. 发布前检查
.venv/bin/python codes/chapters/ch00/examples/generate_audio_samples.py --check
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/quality_check.py
```

获取官方参考源码并核对本地状态（需要 Git，获取时需要网络）：

```bash
.venv/bin/python codes/chapters/ch00/upstream/fetch_upstreams.py --all --report tmp/source-acquisition.json
.venv/bin/python codes/chapters/ch00/upstream/fetch_upstreams.py --verify --report tmp/source-verification.json
```

工具保留独立仓库和许可证，省略常见模型/音频资产，不安装或运行上游程序。源码核对与论文复现分别记录，
具体范围见 [获取工具说明](codes/chapters/ch00/upstream/README.md)。

HARKTOOL5 的官方源码归档另按 SHA-256 锁定和取得，使用独立的 `fetch_archives.py` 与状态报告；命令、许可和提取范围见同一获取说明。

## 学习路径

- **路径 A（零基础入门）**：导读 → 01 → 11.1/11.3 → 02/03 → 04（GCC+SRP）→ 05（DSB+MVDR）→ 06/07/08 → 09 → 运行所学章节的教学代码，并复现对应插图。
- **路径 B（工程实现）**：11.1/11.2/11.3 定 A/B/C 方案 → 05/06/07/08 → 10 全读 → 运行第 10 章工程基线 → 输出延迟/同步/标定三张预算表。
- **路径 C（研究前沿）**：02（CRLB）→ 03（稀疏阵）→ 04/05 前沿 → 06/07/08 → 13.3 八条前沿 + 13.6 练习。

## 约定

- 需要跨段或跨章引用的核心公式在公式块内使用 `\tag{章-序}` 编号，正文引用写“见式(5-1)”。
- 缩写首次出现给全称；"dB 换算用 10log（功率）/20log（幅度）"。
- 数字凡涉榜单均标注条件与出处，仿真数字注明实现口径。
- 外部公式、算法和数据优先链接 DOI、标准组织或官方页面；引用时核对标题、作者、年份和具体表/节，不能只检查链接能否打开。
- 本仓库教学代码与外部参考实现的边界见 `codes/chapters/README.md`；第三方代码、模型和数据的许可证分别核对。
