# Scripts：绘图与工具脚本

本目录放绘图、构建和发布检查工具。算法与工程教学代码在 `codes/`；两类程序都应在**仓库根目录**执行。

单章实验的真实实现按 [第 1～11 章及附录 A/B](../codes/chapters/README.md) 分目录保存，跨章练习、全书索引和主音频总清单归导读目录 `codes/chapters/ch00/`。旧 `codes.examples`、`codes.array_tutorial` 入口已移除。音频、房间生成器和部分上游探针与源路径或 SHA 绑定；修改真实源文件后，按当前清单的规则重新生成并核对对应资产或报告。

运行时间取决于处理器、操作系统、Python 与依赖版本和当前负载。若要报告耗时，应同时记录这些条件、运行次数和统计方式。

```bash
.venv/bin/python codes/chapters/ch00/examples/generate_audio_samples.py  # 先生成 27 组、109 个合成 WAV 及清单
.venv/bin/python -m codes.chapters.ch01.examples.generate_binaural_cues  # 独立双耳线索 5 个 WAV；--check 只核对
.venv/bin/python -m codes.chapters.ch01.examples.generate_spectral_cues  # 四份人工频率响应 WAV；--check 只读完整回放
.venv/bin/python -m codes.chapters.ch02.examples.generate_stft_convolution  # 独立有限窗卷积 3 个 WAV；--check 只核对
.venv/bin/python -m codes.chapters.ch02.examples.generate_sweep_audio  # 数字扫频辨识四WAV；--check只读完整回放
.venv/bin/python -m codes.chapters.ch08.examples.mask_representation_demo  # 六份已知掩码表示WAV
.venv/bin/python -m codes.chapters.ch08.examples.gss_teaching_demo  # 独立 GSS 教学音频与状态
.venv/bin/python -m codes.chapters.ch09.examples.chapter09_tracking_audio  # 独立PCM观测与追踪音频
.venv/bin/python -m codes.chapters.ch09.examples.moving_source_audio  # 独立连续移动双麦音频
.venv/bin/python -m codes.chapters.ch06.examples.generate_apa_audio  # 独立六WAV/浮点与实际PCM留出；图58读取此清单
.venv/bin/python -m codes.chapters.ch07.examples.mint_teaching_demo  # 独立六WAV/完整尾与实际PCM评分；图59读取此清单
.venv/bin/python -m codes.chapters.ch10.examples.generate_noise_mismatch  # 六独立噪声失配WAV
.venv/bin/python -m codes.chapters.ch11.examples.generate_selection_audio  # 8 份独立场景 WAV；--check 只核对
.venv/bin/python -m codes.chapters.appendix_a.examples.generate_weighted_audio  # 五份独立已知噪声加权WAV；图65读取
.venv/bin/python -m codes.chapters.appendix_b.examples.generate_response_audio  # 五份同DRR短FIR对照WAV；图66读取
.venv/bin/python scripts/make_figures.py      # 生成图 1～25、图 33～36、40～72 → figures/
.venv/bin/python scripts/make_aec_figures.py  # 生成图 26～32、37～39（回声消除专题）→ figures/
.venv/bin/python scripts/build_site.py        # 16 个教程页 + 6 个研究页，共 22 页 → site/
.venv/bin/python scripts/build_pdf.py         # 合订 chapters/ → dist/combined.html → dist/microphone-array-tutorial.pdf（需 Chrome）
.venv/bin/python codes/chapters/ch00/examples/generate_audio_samples.py --check  # 只核对音频、参数与摘要，不重写文件
.venv/bin/python codes/chapters/ch02/examples/prepare_real_recordings.py --check  # 真实录音及派生文件，离线核对
.venv/bin/python scripts/quality_check.py      # 发布前检查结构、公式、图片溯源、链接、书签和本地路径泄露
.venv/bin/python -m unittest discover -s tests -v  # 运行构建与算法回归测试
.venv/bin/python -m codes.chapters.ch10.examples.ch10_engineering_baselines  # 第 10 章工程基线
```

Windows 上把 `.venv/bin/python` 换成 `.venv\Scripts\python`。

| 脚本 | 作用 | 输出 |
|---|---|---|
| `../codes/chapters/ch00/examples/generate_audio_samples.py` | 生成 27 组、109 个合成音频文件；清单记录参数、所属章节、共同增益和摘要。`--check` 只检查现有生成物 | 各章的 `audio/*.wav` 与 `codes/chapters/ch00/audio/MANIFEST.json` |
| `../codes/chapters/appendix_b/examples/room_srp_exercise.py` | `--check` 只核固定几何与 Sabine 输入；`--run` 才用 pyroomacoustics 0.10.0 实算六位置 RIR、T60、DRR 和 SRP。写 `--results` 时还须同时指定 `--plot` 与 `--audio-dir`；已有目标会拒绝覆盖 | `codes/chapters/appendix_b/room_audio/` 已收入 18 个合成 WAV、清单、`ROOM_RESULTS.png` 及 `RESULTS.json`；重生成时先输出到另一个新目录核对 |
| `../codes/chapters/ch08/examples/gss_teaching_demo.py`、`../codes/chapters/ch09/examples/moving_source_audio.py` | 分别生成受控活动导引处理链和连续自由场双麦实验；数学合成，不是设备实测 | `codes/chapters/ch08/gss_audio/` 的 5 个 WAV、状态及清单；`codes/chapters/ch09/moving_audio/` 的 3 个 WAV 与真值清单 |
| `../codes/chapters/ch02/examples/prepare_real_recordings.py` | 默认及 `--check` 均离线只读；`--prepare` 从固定本地归档重建；`--download` 显式获取约 99 MB 归档并重建 | `codes/chapters/ch02/real_audio/`：4 个 WAV、清单；署名与许可独立保留 |
| `make_figures.py` | 生成图 1～25 和图 33～36、40～72。只用 numpy 和 matplotlib，不依赖 scipy；随机种子固定。图 34～36、40～41、43～45、47、49、58～60、63～70、72 读取已生成的音频，必须先运行音频生成器。图 13 的蒙特卡洛统计耗时最长 | `figures/fig01`～`fig25_*.png`、`fig33_*`～`fig36_*`、`fig40_*`～`fig72_*` |
| `make_aec_figures.py` | 10 张回声消除专题图（原 7 张另加两带子带、IPNLMS/RLS/Kalman 状态图及 PBFDAF 流程图）。风格与上一个脚本统一（六色/五级字号/dpi150） | `figures/fig26`～`fig32_*`、`fig37`～`fig39_*` |
| `build_site.py` | 生成 16 个教程页和 6 个研究页，保留旧版语义及顺序深链；编号图直接引用 `figures/`，独立音频与状态按各自清单核验并复制；站点 MathJax 在线加载 | `site/` 下的网页及独立媒体副本 |
| `build_pdf.py` | 合订本脚本。16 篇合成带封面和三级目录的 HTML，Chrome 标签化打印 A4 PDF，再以保留结构树的方式写三级书签；第 1～13 章的源 h4 进入第三级。常用 flag：`--html-only`、`--pdf-only`、`--no-bookmarks`、`--build-date YYYY-MM-DD` | `dist/combined.html` 与 `dist/microphone-array-tutorial.pdf` |
| `build_markdown_helpers.py` | 两构建器共享的Markdown数学与代码边界处理；保留代码、原始HTML及转义美元符的语义，数学暂存标识避开原文 | 供构建器导入，无独立产物 |
| `code_layout.py` | 主音频组到首讲章节的唯一映射及路径查询；生成器、构建和检查共同使用这一布局 | 各章 `audio/` 路径，不单独生成文件 |
| `heading_aliases.py`、`legacy_sequential_anchors.json` | 按主题维护已发布节号和顺序深链；构建器插入历史别名，门禁另外核对主题与唯一性 | 网页和合订HTML中的兼容锚点 |
| `../codes/chapters/ch00/io_contracts.py` | 音频生成和上游获取共用的路径、成员、严格JSON、元数据类型和报告写入原语；各调用者保留独立清单、评分与许可逻辑 | 无独立产物；实际参与生成的源摘要进入相应清单 |
| `inline_layout.js` | 站点与合订本共用的成品排版辅助。等待公式与字体完成后，只保护适合当前宽度的行内公式及紧邻短单位/标点、可见普通文字中的稳定题号；短粗体引导在纸版跟随下一段。视窗和字体变化后重新核对宽度。合订预览在打印前即采用 A4 正文几何，打印期间不重排 DOM；单页网站打印前拆除屏幕分组，结束后恢复。保留 TeX、代码、链接、辅助公式树和长式的滚动接口。脚本内容计入两种产物及独立门禁的源摘要 | 由两个构建器嵌入 HTML；无单独生成物 |
| `quality_check.py` | 发布门禁。用独立基线检查 16 篇/151 节/701 个指定子节/72 图，核对图号、alt、公式编号与引用、小节语义链接、PNG 绘图脚本摘要、网页导航和 PDF 三级书签。确定性问题阻断发布，高风险口语只提醒人工复核 | 通过、失败清单，以及不阻断发布的人工复核与可访问性提示 |

336 道稳定编号的代码题可从各章入口复算，例如：

```bash
.venv/bin/python -m codes.chapters.ch04.chapter04_experiments
.venv/bin/python -m codes.chapters.ch00.cross_chapter.exercises_spatial
.venv/bin/python -m codes.chapters.ch00.cross_chapter.exercises_enhancement
.venv/bin/python -m codes.chapters.ch06.aec_algorithm_minicases
.venv/bin/python -m codes.chapters.ch06.aec_advanced_exercises
.venv/bin/python -m codes.chapters.ch00.cross_chapter.exercises_engineering
.venv/bin/python -m codes.chapters.ch11.chapter11_experiments
.venv/bin/python -m codes.chapters.appendix_a.appendix_a_experiments
.venv/bin/python -m codes.chapters.appendix_b.appendix_b_experiments
.venv/bin/python -m codes.chapters.appendix_b.examples.room_srp_exercise --check
```

附录 B 的 E13-03～14 为只读逐步实验；最后一条 `--check` 仅核对第 16 题的固定房间几何和 Sabine 输入，**不会**重新计算房间脉冲响应或改写已发布资产。完整仿真命令见下文[附录 B 房间仿真复算](#附录-b-房间仿真复算)。改图练习须使用脚本副本或独立输出目录，记录改变的参数，不覆盖本书的发布图。

第 11 章 [`chapter11_experiments.py`](../codes/chapters/ch11/chapter11_experiments.py) 包含 E11-10～25 十六道选型计算。图 46 读取本书构造的四候选表；图 47 在四个实际导出的 FIR 音频通过摘要校验后，从 PCM 重新投影频率并核对对齐误差。图64另读两场景八个独立WAV，共同增益与完整尾部保留。三图的条件、数据与脚本入口见[第 11 章](../chapters/11_selection-guide.md)和[音频实验 §33与§44](../codes/chapters/ch00/research/05_exercises_and_audio.md)。

附录 A 的入口复算E12-06～19。E12-08先核20个真实主生成源，再读正式三WAV的整数样本；源摘要过期、文件缺失或PCM不符时直接失败。E12-19核对`weighted_audio/`严格六成员与四个真实源，在同一28800点稳定窗比较三种固定权重。图65从实际PCM读整数分子、分母，随图生成`appendix_a/reports/figure65_weighted_noise.json`。参数与试听入口见[音频实验§45](../codes/chapters/ch00/research/05_exercises_and_audio.md)。

固定pb_bss原求解器的本地合同可运行`.venv/bin/python -B -m codes.chapters.appendix_a.examples.audit_upstream_solver_contracts --report codes/chapters/appendix_a/reports/upstream_solver_contracts.json`。工具校验精确提交、许可、源码与洁净状态，完整加载原模块，仅调用`stable_solve`四例；另两例直接调用NumPy。原整数回退截断失败保留为真实结果，不改上游，不把局部调用当成完整波束形成。

附录B的 E13-11～14 分别复算共享TAC聚合、DRR/EDC安全尺度、T20时间条件化与同DRR频响对照。`appendix_b.examples.generate_response_audio --check`只读核四真实源、严格六成员及五WAV完整内存回放。图66独立读实际PCM整数，保存`appendix_b/reports/figure66_equal_drr_response.json`；解析反射能量与实际完整输出误差分别显示。参数、五个播放器和答案见[音频实验§46](../codes/chapters/ch00/research/05_exercises_and_audio.md)。

`appendix_b.examples.check_room_assets` 默认核21普通成员、七真实源、绑定报告及18PCM；`--replay`另需固定PRA环境并实际重跑，不把只读结构/PCM检查称为重跑房间。`appendix_b.examples.audit_tac_contracts`只读核固定作者原源的10个静态合同，显式`--report`才写报告，原神经网络调用为0。

题号、答案和音频对照见[练习与音频实验](../codes/chapters/ch00/research/05_exercises_and_audio.md)。主[音频清单](../codes/chapters/ch00/audio/MANIFEST.json)记录 109 个分章存放的 16 kHz、PCM16 本书合成信号；每组共用一个增益，不逐文件归一化。不能用这些短样例声称自然语音质量或正式听测结果。

`codes/chapters/appendix_b/room_audio/` 另外保存 18 个白噪声房间样本，建站时核对并复制到 `site/room_audio/`，不混用清单。

`codes/chapters/ch08/gss_audio/` 和 `codes/chapters/ch09/moving_audio/` 分别保存 5 个受控 GSS 音频及状态、3 个自由场移动声源音频及轨迹真值。建站时分别按独立清单核验并复制，不混用主 109 个 WAV。

`codes/chapters/ch02/real_audio/` 另含 DEMAND 真实环境录音摘录与派生文件，建站时复制到独立的 `site/real_audio/`，同时保留清单、署名和许可。16 通道输入仅供下载分析，三个单通道派生文件提供不自动播放的试听控件。

研究页入口为 `site/research/index.html`，对应 `codes/chapters/ch00/research/README.md`；其余五页保留研究文件名。研究页与正文互链，源码及未生成网页的代码文档链接指向 GitHub 中的原文件，不把 `.md` 猜成不存在的 `.html`。外部链接不改写。

合订本仍只有 16 篇教程：跨章链接指向内部锚点，研究文档和源码链接指向仓库原文件。两种构建摘要均纳入六篇研究源文件，修改后应重新构建。

`codes/chapters/ch10/core/engineering.py` 中的工程基线只依赖 NumPy。它包括 SRO 拟合与教学用线性重采样、VAD 迟滞与 hangover、峰值保护 AGC、固定容量环形缓冲、deadline/队列模拟、Q1.15 量化和遥测字段校验。运行 `.venv/bin/python -m unittest tests.test_codes_engineering -v` 可执行对应回归测试。代码范围、上游实现与许可证边界以 `codes/chapters/README.md`、`codes/chapters/ch00/COVERAGE.md`、`codes/chapters/ch00/THIRD_PARTY.md` 和 `codes/chapters/ch00/SOURCES.lock.json` 为准。

**发布与验收说明**

**书签与人工抽查**：合订本 PDF 顶层是导读、11 章正文、2 篇扩展专题和 2 篇附录，第二层来自各篇实际小节；第 1～11 章、两篇扩展专题与附录共有 701 个源 h4 作为第三级书签，并保持在各自父节之下。

书签使用 HTML 标题 id 对应的 PDF 命名目标，保留页内定位。命名目标缺失、越界或同名却指向不同位置时构建失败；发布门禁独立比较每项书签与正文目标的页码及视图参数。目录和正文可能出现同名标题，仅检查落页文字不能识别误跳到目录的问题。

合订本用 `scripts/vendor/mathjax-3.2.2/` 内固定版本的主脚本、`boldsymbol` 按需扩展和 23 个 WOFF 字体离线排版。构建前核对脚本摘要与资源完整性；打印后检查未渲染 TeX 和 AEC 算例页的数学字形子集。Chrome 将这些数学字形嵌为缺少可靠 ToUnicode 映射的 Type3 字体，因此文本提取时公式可能为空，即使画面正常；正式发布前仍须打开 PDF，抽查公式、宽表、长代码块、图片和分页是否存在半渲染、溢出或裁切。

PDF 正文固定为 16 px，MathJax 公式按 100% 字号打印；网页公式按 110% 显示，长式由公式容器独立横向滚动。

打印后检查长中文正文的变换矩阵：正常 CSS 像素到 PDF 点的比例为 0.75，低于 0.74 时拒绝发布，以发现过宽公式或表格触发的整书缩小。该检查不代替逐式版式检查。

竖图打印高度上限为 225 mm，章标题与首节标题使用紧凑间距；图片最终有效字号仍需按实际打印尺寸复核。

**PDF 可访问性边界**：Chrome 使用 `--export-tagged-pdf` 导出结构树，pypdf 完整克隆页面后添加书签；构建和发布门禁检查标记根、父树及页面连接。标签存在不等于公式辅助文本、阅读顺序或 PDF/UA 已完整验收，最终版仍需辅助技术实测。

**独立结构基线**：发布门禁的独立结构基线为 16 个顶级书签、151 个二级书签、701 个三级书签，共 868 个大纲项，以及图 1～72。它还检查图号与 alt、公式编号与引用、小节语义链接、每个源 h2/h3/h4 标题是否真的出现在当前页导航中（源 h1 可排除），以及 PNG 中的 `SourceScript` 和完整 `SourceScriptDigest`。

修改绘图脚本后未重画的 PNG 会使门禁失败；高风险口语命中只输出人工复核提示。

**失败时保留旧产物**：完整 PDF 构建先在临时 HTML/PDF 上完成打印、书签和链接校验，再替换正式文件。站点批量替换与 PDF 双文件替换发生可捕获异常时，会恢复已有文件并移除本批新建文件；若恢复也失败，会保留备份目录并报告路径。这不是断电、进程强杀或文件系统损坏时的原子发布保证。

`--html-only` 只替换 HTML；旧 PDF 的摘要会与新 HTML 不同，必须继续生成 PDF 后再发布。

## 附录 B 房间仿真复算

第 16 题的六位置房间重算需要 `pyroomacoustics==0.10.0`。可在临时虚拟环境安装，保留仓库 `.venv` 的主依赖集；下列命令以 Python 3.13 的 macOS/Linux 环境为例。没有该依赖仍能生成正文的 72 张编号图，并查阅随仓的附录 B 结果图和 18 个房间 WAV。固定版本说明见[官方 PyPI 页面](https://pypi.org/project/pyroomacoustics/0.10.0/)。

在有 Python 3.13 的 macOS/Linux 主机上，可从仓库根目录用新目录复算，不覆盖本书样本：

```bash
python3.13 -m venv /tmp/masp-room-pra
/tmp/masp-room-pra/bin/python -m pip install pyroomacoustics==0.10.0 matplotlib
PRA_NUM_THREADS=2 /tmp/masp-room-pra/bin/python -m codes.chapters.appendix_b.examples.room_srp_exercise --run --plot /tmp/masp-room-result.png --results /tmp/masp-room-results.json --audio-dir /tmp/masp-room-audio-new
```

目标图文件及音频目录应事先不存在。重跑后用新目录清单的 SHA-256 对照 `codes/chapters/appendix_b/room_audio/MANIFEST.json`，并记录 Python、NumPy、SciPy、pyroomacoustics 与线程数；跨平台绘图字体可能改变 PNG 字节，数值和 WAV 应分别核查。

## 其他练习与绘图报告

跨章的空间精度、增强步骤与追踪时间练习可分别运行：

```bash
.venv/bin/python -m codes.chapters.ch00.cross_chapter.spatial_precision_exercises
.venv/bin/python -m codes.chapters.ch00.cross_chapter.enhancement_step_exercises
.venv/bin/python -m codes.chapters.ch00.cross_chapter.tracking_time_exercises
```


模型与边界练习：

```bash
.venv/bin/python -m codes.chapters.ch00.cross_chapter.spatial_model_exercises
.venv/bin/python -m codes.chapters.ch00.cross_chapter.enhancement_structure_exercises
.venv/bin/python -m codes.chapters.ch00.cross_chapter.engineering_boundary_exercises
.venv/bin/python -m codes.chapters.appendix_b.interpolation_exercise
.venv/bin/python -m codes.chapters.ch10.examples.run_stk_delay_probe --report tmp/stk-delay-rerun.json
```

最后一项需已取得固定STK源码和C++编译器，只运行DelayL组件检查；其余为本书NumPy/标准库数学例子。图41从主清单4个interpolation音频读回逐频幅度，与独立解析曲线区分；先生成109个WAV再绘图。

图13在绘图时同步输出 `codes/chapters/ch04/reports/figure13_gcc_reverb.json`：9条件各150次的事件、峰对比度、种子和区间口径。测试与导入不写报告；修改绘图源后通过同一入口重生图和报告。


图44读取`codes/chapters/ch09/tracking_audio/MANIFEST.json`中的PCM逐帧分析。先运行
`.venv/bin/python -m codes.chapters.ch09.examples.chapter09_tracking_audio`生成两份独立WAV，
再绘图；`--check`只读复算，拒绝过期或额外文件。网页构建同时核对精确文件集合、摘要和PCM格式，
播放器与独立清单分别发布到`site/tracking_audio/`，不混入主音频组。
图22另由同一绘图过程保存`codes/chapters/ch09/reports/figure22_tracking.json`，包括逐帧ESS、重采样标记及观测/缺测独立分母。


## 第二章有限窗与固定源码复算

```bash
.venv/bin/python -m codes.chapters.ch02.chapter02_experiments
.venv/bin/python -m codes.chapters.ch02.examples.generate_stft_convolution --check
.venv/bin/python -m codes.chapters.ch02.examples.generate_sweep_audio --check
.venv/bin/python -m codes.chapters.ch02.examples.audit_upstream_models
```

第一条包含 E02-09～20；第二条只读核对 `codes/chapters/ch02/stft_audio/` 的三份 WAV、真实 PCM 评分、参数和六个生成源摘要。建站时独立核验后复制到 `site/stft_audio/`，不混入主 109 个样本。图 50 的六点确定性结果由 `make_figures.py` 同次写入 `codes/chapters/ch02/reports/figure50_stft_convolution.json`；逐帧补零但丢掉滤波尾部的对照不能称为完整卷积实现。

第三条只读完整重放数字扫频的四份WAV与独立清单，核五真实源SHA、参数、环境、完整字节及浮点/实际PCM两域的IR评分；不混入主109。首次生成可运行同一命令去掉`--check`。源32000点与三响应32160点保留完整传播尾，所有估计IR的65536点均计分，真值161点支持只作额外诊断。

第四条要求已取得固定PRA、Acoular、doatools、pyfar与声速项目。它独立验证官方origin、提交、所用blob/许可及洁净，完整选集状态另列；默认只输出stdout。需要保存当前报告时显式加入`--report codes/chapters/ch02/reports/upstream_model_contracts.json`；旧`upstream_models.json`保持历史身份，不能覆盖。PRA/Acoular/doatools原方法与六个pyfar原函数仅限定调用，声速项目仅核说明和许可。人工适配与未运行原模块随报告保存，不代表整包、设备或房间测量。入口、逐步计算及试听见[练习与音频实验§36](../codes/chapters/ch00/research/05_exercises_and_audio.md#36-第二章有限窗卷积二阶交叉项与窗归一化)及§50～51。

## 第三章：几何、标定与多频观测

```bash
.venv/bin/python -m codes.chapters.ch03.chapter03_experiments
.venv/bin/python -m codes.chapters.ch03.examples.generate_geometry_audio
.venv/bin/python -m codes.chapters.ch03.examples.generate_geometry_audio --check
.venv/bin/python -m codes.chapters.ch03.examples.audit_upstream_coarray --report codes/chapters/ch03/reports/upstream_coarray.json
```

章节入口复算E03-08～17，另读取随仓四份DMA PCM。独立三WAV由六个真实源生成；32 kHz、2 s、源单声道与两个六声道观测，实际读回在4800:59200稳定窗逐频拟合。六声道播放可能下混，不作为耳听定位成绩；多频检查只证明指定方向对的可辨识信息。图51的变化是无单位约束线性解的方向分量，不是角误差；图52是解析流形相干功率。固定doatools审计需要已核验的本地源码缓存，只提取明确的原方法并局部适配NumPy类型别名，不导入完整软件或运行未知误差校准。


### 第四章聚焦与固定源码诊断

```bash
.venv/bin/python -m codes.chapters.ch04.examples.generate_focus_audio
.venv/bin/python -m codes.chapters.ch04.examples.generate_focus_audio --check
.venv/bin/python -m codes.chapters.ch04.chapter04_experiments
```

四份独立WAV由六个真实源生成；16kHz、每通道32024点，共同增益1。参考/源2各单声道，未聚焦与已知分量聚焦各四声道；2400:29600稳定窗以cos/sin实最小二乘拟合后固定除以0.08取外积，两个频率等权池化。不是盲定位、完整CSSM或正式听测。图53/54从独立解析总体矩阵计算，源值和PCM值分别记录。

`ch04.examples.audit_upstream_doa` 需要已核验的PRA/doatools固定源码缓存及独立pyroomacoustics0.10.0环境，`--report`是唯一写报告入口；调用范围、源SHA、依赖版本、原始失败与适配分别记录。TOPS原类、CSSM/WAVES辅助函数和root-MUSIC提取方法不具有相同运行范围。`ch04.examples.audit_said_compression`只加载固定SAID原压缩单文件，不安装Torch、下载权重或执行神经网络；其报告与经典DOA报告独立。


## 第五章导数约束音频与固定上游审计

```bash
.venv/bin/python -m codes.chapters.ch05.examples.generate_derivative_audio
.venv/bin/python -m codes.chapters.ch05.examples.generate_derivative_audio --check
.venv/bin/python -m codes.chapters.ch05.chapter05_experiments
```

四WAV共用增益1和一采样因果参考，16kHz、每通道32002点；八个实际源摘要包含唯一波束核与协方差核。2400:29600窗的解析期望、浮点clean/noise/交叉项和实际PCM总误差分别保存，PCM不虚构可观测的干净/噪声分解。`--check`重新读回与复算但不重生文件；发布复制检查只验证集合、来源、格式、摘要和播放器，数值评分由生成器与独立回归验收。图55/56各从解析响应和高斯密度模型计算，不把期望标成实测。

[原方法审计工具](../codes/chapters/ch05/examples/audit_upstream_beamformers.py)需要既有SciPy环境和锁定的pb_bss独立源码缓存；默认只读，显式`--report`才写报告。工具提取12个原函数执行31个限定输入，保留异常、退化和约束违约；不是完整包、GPU网络或工业整链性能验收。具体运行环境、原源码片段与摘要在[真实报告](../codes/chapters/ch05/reports/upstream_beamformers.json)中，不默认假设历史临时环境仍存在。

## 第9章：信息边界与追踪合同

第9章共23道稳定编号题。E09-20～23分别计算相关观测、瞬时方位的尺度零空间、空观测下的Bernoulli存在概率和迟到观测；图61由指定解析参数生成，不读取音频，数值报告为 `codes/chapters/ch09/reports/figure61_tracking_information.json`。五个原独立WAV仍分别由移动源与PCM追踪生成器管理。两生成器的 `--check` 均严格只读，拒绝源码、清单或实际字节失配；网页发布同时检查完整普通文件集合、全部父目录和可见播放器。

[当前上游合同工具](../codes/chapters/ch09/examples/audit_upstream_tracking_contracts.py)默认打印结果，显式 `--report` 才写报告。原C调用需要C编译器，FilterPy原类复验需要报告中明确记录的隔离依赖；缺少依赖会如实记录未运行。SAF提取的原step使用计数替身，不能声称执行了粒子、条件Kalman或完整音频系统。固定锁、许可、源码blob/摘要与调用前后干净状态均随报告保存，历史9月28日记录不改写。

第10章现有33题；E28～32为限定软更新、截尾逆积分、非抢占阻塞、外部时钟尺度和填充RTF的可复算例，图62保存同次解析报告。E33及图63使用独立六WAV清单；先运行 `codes.chapters.ch10.examples.generate_noise_mismatch`，再绘图。`--check`只读完整回放源/清单/PCM，不自动修复。五项固定工业接口使用 `codes.chapters.ch10.examples.audit_industrial_contracts`，默认只读，显式 `--report` 才写当前报告；替身、编译路径和未执行范围随报告保存，旧历史报告不改。


## 扩展专题Ⅰ：成像实验与稳定章节身份

新增阅读顺序为第11章→扩展专题Ⅰ→扩展专题Ⅱ→附录A→附录B。源文件前缀14、公式14-x、练习E14与合订目标ch-14保持一致；附录仍使用原12/13身份。`build_pdf.chapter_number`从文件名取稳定身份，不由列表位置重新编号，旧附录书签与练习深链须同时验收。

```bash
.venv/bin/python -m codes.chapters.ch14.chapter14_exercises
.venv/bin/python -m codes.chapters.ch14.examples.generate_imaging_audio
.venv/bin/python -m codes.chapters.ch15.examples.generate_distributed_audio  # 17 independent WAVs; --check只读
.venv/bin/python -m codes.chapters.ch14.examples.generate_imaging_audio --check
.venv/bin/python -m codes.chapters.ch14.examples.audit_upstream_imaging_contracts
```

16题只读计算；五个独立24kHz数学WAV保留完整四点传播尾，不混入主109份。图67～69分别复算PSF与二维泄漏、非相干模型及full-CSM CLEAN-SC、校准与不同拟合目标。三报告绑定当前四个真实源；原Acoular限定调用报告保留目标差异与未执行条件。源、PCM、图和网页/PDF验收分别记，不能用其中一项代替另一项。详见[音频实验§47](../codes/chapters/ch00/research/05_exercises_and_audio.md#sec-47-1)与[原源合同](../codes/chapters/ch00/research/01_spatial_and_tracking.md#imaging-contract-audit)。


## 扩展专题Ⅱ的已知统计、音频与原源码合同

```bash
.venv/bin/python -m codes.chapters.ch15.chapter15_exercises
.venv/bin/python -m codes.chapters.ch15.examples.generate_distributed_audio --check
.venv/bin/python -m codes.chapters.ch15.examples.audit_upstream_distributed_contracts
```

24题覆盖任务相关压缩、真实广播更新、退化输入、时钟、缺口、树消息与GEVD。17个独立WAV用共同增益1和九个冻结真实源；图70/72先完整只读回放，再独立读整数PCM核E/D。图71记录广播改变后的当前有效权重，旧求解快照另列；预算含初始两次求解。原MATLAB仅静态合同及独立控制；paderwasn仅三个NumPy助手原函数，未执行完整网络或WOLA链。详见[实验§48](../codes/chapters/ch00/research/05_exercises_and_audio.md#distributed-exercises-audio)。


已知方向基线音频由`codes.chapters.ch03.examples.generate_baseline_audio`单独生成，六个WAV与清单不计入主109样本。站点发布先按当前六真实源完整只读回放，再复制到`site/baseline_audio/`；质量检查另按实际交织PCM整数复算27200点稳窗的逐通道与总功率，并核正文/研究页全部六个播放器和清单链接。单音相位反时差依赖已知小于1ms的主值范围，不是宽带盲时差估计或设备验收。
