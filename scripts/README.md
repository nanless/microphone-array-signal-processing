# Scripts：绘图与工具脚本

本目录放绘图、构建和发布检查工具。算法与工程教学代码在 `codes/`；两类程序都应在**仓库根目录**执行。

单章实验的真实实现按 [第 1～11 章及附录 A/B](../codes/chapters/README.md) 分目录保存，跨章练习、全书索引和主音频总清单归导读目录 `codes/chapters/ch00/`。旧 `codes.examples`、`codes.array_tutorial` 入口已移除。音频、房间生成器和部分上游探针与源路径或 SHA 绑定；修改真实源文件后，按当前清单的规则重新生成并核对对应资产或报告。

运行时间取决于处理器、操作系统、Python 与依赖版本和当前负载。若要报告耗时，应同时记录这些条件、运行次数和统计方式。

```bash
.venv/bin/python codes/chapters/ch00/examples/generate_audio_samples.py  # 先生成 27 组、109 个合成 WAV 及清单
.venv/bin/python -m codes.chapters.ch08.examples.gss_teaching_demo  # 独立 GSS 教学音频与状态
.venv/bin/python -m codes.chapters.ch09.examples.chapter09_tracking_audio  # 独立PCM观测与追踪音频
.venv/bin/python -m codes.chapters.ch09.examples.moving_source_audio  # 独立连续移动双麦音频
.venv/bin/python scripts/make_figures.py      # 生成图 1～25、图 33～36、40～49 → figures/
.venv/bin/python scripts/make_aec_figures.py  # 生成图 26～32、37～39（回声消除专题）→ figures/
.venv/bin/python scripts/build_site.py        # 14 个教程页 + 6 个研究页，共 20 页 → site/
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
| `make_figures.py` | 生成图 1～25 和图 33～36、40～49。只用 numpy 和 matplotlib，不依赖 scipy；随机种子固定。图 34～36、40～41、43～45、47、49 读取已生成的音频，必须先运行音频生成器。图 13 的蒙特卡洛统计耗时最长 | `figures/fig01`～`fig25_*.png`、`fig33_*`～`fig36_*`、`fig40_*`～`fig49_*` |
| `make_aec_figures.py` | 10 张回声消除专题图（原 7 张另加两带子带、IPNLMS/RLS/Kalman 状态图及 PBFDAF 流程图）。风格与上一个脚本统一（六色/五级字号/dpi150） | `figures/fig26`～`fig32_*`、`fig37`～`fig39_*` |
| `build_site.py` | 生成 14 个教程页和 6 个研究页，保留旧版语义及顺序深链；编号图直接引用 `figures/`，独立音频与状态按各自清单核验并复制；站点 MathJax 在线加载 | `site/` 下的网页及独立媒体副本 |
| `build_pdf.py` | 合订本脚本。14 篇合成带封面和三级目录的 HTML，Chrome 标签化打印 A4 PDF，再以保留结构树的方式写三级书签；第 1～13 章的源 h4 进入第三级。常用 flag：`--html-only`、`--pdf-only`、`--no-bookmarks`、`--build-date YYYY-MM-DD` | `dist/combined.html` 与 `dist/microphone-array-tutorial.pdf` |
| `quality_check.py` | 发布门禁。用独立基线检查 14 篇/121 节/500 个指定子节/49 图，核对图号、alt、公式编号与引用、小节语义链接、PNG 绘图脚本摘要、网页导航和 PDF 三级书签。确定性问题阻断发布，高风险口语只提醒人工复核 | 通过、失败清单，以及不阻断发布的人工复核与可访问性提示 |

231 道稳定编号的代码题可从各章入口复算，例如：

```bash
.venv/bin/python -m codes.chapters.ch04.chapter04_experiments
.venv/bin/python -m codes.chapters.ch00.cross_chapter.exercises_spatial
.venv/bin/python -m codes.chapters.ch00.cross_chapter.exercises_enhancement
.venv/bin/python -m codes.chapters.ch06.aec_algorithm_minicases
.venv/bin/python -m codes.chapters.ch06.aec_advanced_exercises
.venv/bin/python -m codes.chapters.ch00.cross_chapter.exercises_engineering
.venv/bin/python -m codes.chapters.ch11.chapter11_experiments
.venv/bin/python -m codes.chapters.appendix_b.appendix_b_experiments
.venv/bin/python -m codes.chapters.appendix_b.examples.room_srp_exercise --check
```

附录 B 的 E13-03～10 为只读逐步实验；最后一条 `--check` 仅核对第 16 题的固定房间几何和 Sabine 输入，**不会**重新计算房间脉冲响应或改写已发布资产。完整仿真命令见下文[附录 B 房间仿真复算](#附录-b-房间仿真复算)。改图练习须使用脚本副本或独立输出目录，记录改变的参数，不覆盖本书的发布图。

第 11 章 [`chapter11_experiments.py`](../codes/chapters/ch11/chapter11_experiments.py) 包含 E11-10～19 十道选型计算。图 46 读取本书构造的四候选表；图 47 在四个实际导出的 FIR 音频通过摘要校验后，从 PCM 重新投影频率并核对对齐误差。两图的条件、数据与脚本入口见[第 11 章](../chapters/11_selection-guide.md)和[音频实验 §33](../codes/chapters/ch00/research/05_exercises_and_audio.md)。

题号、答案和音频对照见[练习与音频实验](../codes/chapters/ch00/research/05_exercises_and_audio.md)。主[音频清单](../codes/chapters/ch00/audio/MANIFEST.json)记录 109 个分章存放的 16 kHz、PCM16 本书合成信号；每组共用一个增益，不逐文件归一化。不能用这些短样例声称自然语音质量或正式听测结果。

`codes/chapters/appendix_b/room_audio/` 另外保存 18 个白噪声房间样本，建站时核对并复制到 `site/room_audio/`，不混用清单。

`codes/chapters/ch08/gss_audio/` 和 `codes/chapters/ch09/moving_audio/` 分别保存 5 个受控 GSS 音频及状态、3 个自由场移动声源音频及轨迹真值。建站时分别按独立清单核验并复制，不混用主 109 个 WAV。

`codes/chapters/ch02/real_audio/` 另含 DEMAND 真实环境录音摘录与派生文件，建站时复制到独立的 `site/real_audio/`，同时保留清单、署名和许可。16 通道输入仅供下载分析，三个单通道派生文件提供不自动播放的试听控件。

研究页入口为 `site/research/index.html`，对应 `codes/chapters/ch00/research/README.md`；其余五页保留研究文件名。研究页与正文互链，源码及未生成网页的代码文档链接指向 GitHub 中的原文件，不把 `.md` 猜成不存在的 `.html`。外部链接不改写。

合订本仍只有 14 篇教程：跨章链接指向内部锚点，研究文档和源码链接指向仓库原文件。两种构建摘要均纳入六篇研究源文件，修改后应重新构建。

`codes/chapters/ch10/core/engineering.py` 中的工程基线只依赖 NumPy。它包括 SRO 拟合与教学用线性重采样、VAD 迟滞与 hangover、峰值保护 AGC、固定容量环形缓冲、deadline/队列模拟、Q1.15 量化和遥测字段校验。运行 `.venv/bin/python -m unittest tests.test_codes_engineering -v` 可执行对应回归测试。代码范围、上游实现与许可证边界以 `codes/chapters/README.md`、`codes/chapters/ch00/COVERAGE.md`、`codes/chapters/ch00/THIRD_PARTY.md` 和 `codes/chapters/ch00/SOURCES.lock.json` 为准。

**发布与验收说明**

**书签与人工抽查**：合订本 PDF 顶层是导读、11 章正文和 2 篇附录，第二层来自各篇实际小节；第 1～13 章共有 500 个源 h4 作为第三级书签，并保持在各自父节之下。

书签使用 HTML 标题 id 对应的 PDF 命名目标，保留页内定位。命名目标缺失、越界或同名却指向不同位置时构建失败；发布门禁独立比较每项书签与正文目标的页码及视图参数。目录和正文可能出现同名标题，仅检查落页文字不能识别误跳到目录的问题。

合订本用 `scripts/vendor/mathjax-3.2.2/` 内固定版本的主脚本、`boldsymbol` 按需扩展和 23 个 WOFF 字体离线排版。构建前核对脚本摘要与资源完整性；打印后检查未渲染 TeX 和 AEC 算例页的数学字形子集。Chrome 将这些数学字形嵌为缺少可靠 ToUnicode 映射的 Type3 字体，因此文本提取时公式可能为空，即使画面正常；正式发布前仍须打开 PDF，抽查公式、宽表、长代码块、图片和分页是否存在半渲染、溢出或裁切。

PDF 正文固定为 16 px，MathJax 公式按 100% 字号打印；网页公式按 110% 显示，长式由公式容器独立横向滚动。

打印后检查长中文正文的变换矩阵：正常 CSS 像素到 PDF 点的比例为 0.75，低于 0.74 时拒绝发布，以发现过宽公式或表格触发的整书缩小。该检查不代替逐式版式检查。

竖图打印高度上限为 225 mm，章标题与首节标题使用紧凑间距；图片最终有效字号仍需按实际打印尺寸复核。

**PDF 可访问性边界**：Chrome 使用 `--export-tagged-pdf` 导出结构树，pypdf 完整克隆页面后添加书签；构建和发布门禁检查标记根、父树及页面连接。标签存在不等于公式辅助文本、阅读顺序或 PDF/UA 已完整验收，最终版仍需辅助技术实测。

**独立结构基线**：发布门禁的独立结构基线为 14 个顶级书签、121 个二级书签、500 个三级书签，共 635 个大纲项，以及图 1～49。它还检查图号与 alt、公式编号与引用、小节语义链接、每个源 h2/h3/h4 标题是否真的出现在当前页导航中（源 h1 可排除），以及 PNG 中的 `SourceScript` 和完整 `SourceScriptDigest`。

修改绘图脚本后未重画的 PNG 会使门禁失败；高风险口语命中只输出人工复核提示。

**失败时保留旧产物**：完整 PDF 构建先在临时 HTML/PDF 上完成打印、书签和链接校验，再替换正式文件。站点批量替换与 PDF 双文件替换发生可捕获异常时，会恢复已有文件并移除本批新建文件；若恢复也失败，会保留备份目录并报告路径。这不是断电、进程强杀或文件系统损坏时的原子发布保证。

`--html-only` 只替换 HTML；旧 PDF 的摘要会与新 HTML 不同，必须继续生成 PDF 后再发布。

## 附录 B 房间仿真复算

第 16 题的六位置房间重算需要 `pyroomacoustics==0.10.0`。可在临时虚拟环境安装，保留仓库 `.venv` 的主依赖集；下列命令以 Python 3.13 的 macOS/Linux 环境为例。没有该依赖仍能生成正文的 49 张编号图，并查阅随仓的附录 B 结果图和 18 个房间 WAV。固定版本说明见[官方 PyPI 页面](https://pypi.org/project/pyroomacoustics/0.10.0/)。

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
