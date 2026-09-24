# Scripts：绘图与工具脚本

本目录放绘图、构建和发布检查工具。算法与工程教学代码在 `codes/`；两类程序都应在**仓库根目录**执行。

运行时间取决于处理器、操作系统、Python 与依赖版本和当前负载。若要报告耗时，应同时记录这些条件、运行次数和统计方式。

```bash
.venv/bin/python codes/examples/generate_audio_samples.py  # 先生成 14 组、60 个合成 WAV 及清单
.venv/bin/python -m codes.examples.gss_teaching_demo  # 独立 GSS 教学音频与状态
.venv/bin/python -m codes.examples.moving_source_audio  # 独立连续移动双麦音频
.venv/bin/python scripts/make_figures.py      # 生成图 1～25、图 33～36 → figures/
.venv/bin/python scripts/make_aec_figures.py  # 生成图 26～32、37～39（回声消除专题）→ figures/
.venv/bin/python scripts/build_site.py        # 14 个教程页 + 6 个研究页，共 20 页 → site/
.venv/bin/python scripts/build_pdf.py         # 合订 chapters/ → dist/combined.html → dist/microphone-array-tutorial.pdf（需 Chrome）
.venv/bin/python codes/examples/generate_audio_samples.py --check  # 只核对音频、参数与摘要，不重写文件
.venv/bin/python codes/examples/prepare_real_recordings.py --check  # 真实录音及派生文件，离线核对
.venv/bin/python scripts/quality_check.py      # 发布前检查结构、公式、图片溯源、链接、书签和本地路径泄露
.venv/bin/python -m unittest discover -s tests -v  # 运行构建与算法回归测试
.venv/bin/python -m codes.examples.ch10_engineering_baselines  # 第 10 章工程基线
```

Windows 上把 `.venv/bin/python` 换成 `.venv\Scripts\python`。

| 脚本 | 作用 | 输出 |
|---|---|---|
| `../codes/examples/generate_audio_samples.py` | 生成 14 组、60 个合成音频文件；清单记录参数、共同增益和摘要。`--check` 只检查现有生成物 | `codes/audio/*.wav`、`codes/audio/MANIFEST.json` |
| `../codes/examples/room_srp_exercise.py` | `--check` 只核几何；可选 pyroomacoustics 0.10.0 的 `--run` 实算六位置 RIR、T60、DRR 和 SRP，`--audio-dir` 与 `--plot` 可在新目录生成独立样本与补充图 | `codes/room_audio/` 已收入 18 个合成 WAV、清单及 `ROOM_RESULTS.png`；重生成时先输出到另一个新目录核对 |
| `../codes/examples/gss_teaching_demo.py`、`moving_source_audio.py` | 分别生成受控活动导引处理链和连续自由场双麦实验；数学合成，不是设备实测 | `codes/gss_audio/` 的 5 个 WAV、状态及清单；`codes/moving_audio/` 的 3 个 WAV 与真值清单 |
| `../codes/examples/prepare_real_recordings.py` | 默认及 `--check` 均离线只读；`--prepare` 从固定本地归档重建；`--download` 显式获取约 99 MB 归档并重建 | `codes/real_audio/`：4 个 WAV、清单；署名与许可独立保留 |
| `make_figures.py` | 生成图 1～25 和图 33～36。只用 numpy 和 matplotlib，不依赖 scipy；随机种子固定。图 34～36 读取已生成的音频，必须先运行音频生成器。图 13 的蒙特卡洛统计耗时最长 | `figures/fig01`～`fig25_*.png`、`fig33_*`～`fig36_*` |
| `make_aec_figures.py` | 10 张回声消除专题图（原 7 张另加两带子带、IPNLMS/RLS/Kalman 状态图及 PBFDAF 流程图）。风格与上一个脚本统一（六色/五级字号/dpi150） | `figures/fig26`～`fig32_*`、`fig37`～`fig39_*` |
| `renumber2.py` | 已退役。单篇长文时代的图号整理工具，留作存档，平时不用跑 | 无 |
| `build_site.py` | 生成 14 个教程页和 6 个研究页，保留旧版语义及顺序深链；编号图直接引用 `figures/`，独立音频与状态按各自清单核验并复制；站点 MathJax 在线加载 | `site/` 下的网页及独立媒体副本 |
| `build_pdf.py` | 合订本脚本。14 篇合成带封面和三级目录的 HTML，Chrome 标签化打印 A4 PDF，再以保留结构树的方式写三级书签；第 1～6、8～13 章的源 h4 进入第三级。常用 flag：`--html-only`、`--pdf-only`、`--no-bookmarks`、`--build-date YYYY-MM-DD` | `dist/combined.html` 与 `dist/microphone-array-tutorial.pdf` |
| `quality_check.py` | 发布门禁。用独立基线检查 14 篇/119 节/140 个指定子节/39 图，核对图号、alt、公式编号与引用、小节语义链接、PNG 绘图脚本摘要、网页导航和 PDF 三级书签。确定性问题阻断发布，高风险口语只提醒人工复核 | 通过、失败清单，以及不阻断发布的人工复核与可访问性提示 |

改图练习（如附录 B 习题）应使用脚本副本或独立输出目录，记录改变的参数，不覆盖本书的发布图。94 道稳定编号可运行题使用三个原有练习模块、AEC 边界小例和四法练习入口；附录 B 第 16 题另有需 pyroomacoustics 0.10.0 的房间仿真脚本：

```bash
.venv/bin/python -m codes.examples.exercises_spatial
.venv/bin/python -m codes.examples.exercises_enhancement
.venv/bin/python -m codes.examples.aec_algorithm_minicases
.venv/bin/python -m codes.examples.aec_advanced_exercises
.venv/bin/python -m codes.examples.exercises_engineering
.venv/bin/python -m codes.examples.room_srp_exercise --check
```

题号、答案和音频对照见[练习与音频实验](../codes/research/05_exercises_and_audio.md)。`codes/audio/` 的 60 个音频为 16 kHz、PCM16 的本书合成信号，每组共用一个增益，不逐文件归一化；不能用这些短样例声称自然语音质量或正式听测结果。

`codes/room_audio/` 另外保存 18 个白噪声房间样本，建站时核对并复制到 `site/room_audio/`，不混用清单。

`codes/gss_audio/` 和 `codes/moving_audio/` 分别保存 5 个受控 GSS 音频及状态、3 个自由场移动声源音频及轨迹真值。建站时分别按独立清单核验并复制，不混用主 60 个 WAV。

`codes/real_audio/` 另含 DEMAND 真实环境录音摘录与派生文件，建站时复制到独立的 `site/real_audio/`，同时保留清单、署名和许可。16 通道输入仅供下载分析，三个单通道派生文件提供不自动播放的试听控件。

研究页入口为 `site/research/index.html`，对应 `codes/research/README.md`；其余五页保留研究文件名。研究页与正文互链，源码及未生成网页的代码文档链接指向 GitHub 中的原文件，不把 `.md` 猜成不存在的 `.html`。外部链接不改写。

合订本仍只有 14 篇教程：跨章链接指向内部锚点，研究文档和源码链接指向仓库原文件。两种构建摘要均纳入六篇研究源文件，修改后应重新构建。

`codes/array_tutorial/engineering.py` 中的工程基线只依赖 NumPy。它包括 SRO 拟合与教学用线性重采样、VAD 迟滞与 hangover、峰值保护 AGC、固定容量环形缓冲、deadline/队列模拟、Q1.15 量化和遥测字段校验。运行 `.venv/bin/python -m unittest tests.test_codes_engineering -v` 可执行对应回归测试。代码范围、上游实现与许可证边界以 `codes/README.md`、`codes/COVERAGE.md`、`codes/THIRD_PARTY.md` 和 `codes/SOURCES.lock.json` 为准。

**发布与验收说明**

**书签与人工抽查**：合订本 PDF 顶层是导读、11 章正文和 2 篇附录，第二层来自各篇实际小节；第 1～6、8～13 章共有 140 个源 h4 作为第三级书签，并保持在各自父节之下。

合订本用 `scripts/vendor/mathjax-3.2.2/` 内固定版本的主脚本、`boldsymbol` 按需扩展和 23 个 WOFF 字体离线排版。构建前核对脚本摘要与资源完整性；打印后检查未渲染 TeX 和 AEC 算例页的数学字形子集。Chrome 将这些数学字形嵌为缺少可靠 ToUnicode 映射的 Type3 字体，因此文本提取时公式可能为空，即使画面正常；正式发布前仍须打开 PDF，抽查公式、宽表、长代码块、图片和分页是否存在半渲染、溢出或裁切。

PDF 正文固定为 16 px，MathJax 公式按 100% 字号打印；网页公式按 110% 显示，长式由公式容器独立横向滚动。

打印后检查长中文正文的变换矩阵：正常 CSS 像素到 PDF 点的比例为 0.75，低于 0.74 时拒绝发布，以发现过宽公式或表格触发的整书缩小。该检查不代替逐式版式检查。

竖图打印高度上限为 225 mm，章标题与首节标题使用紧凑间距；图片最终有效字号仍需按实际打印尺寸复核。

**PDF 可访问性边界**：Chrome 使用 `--export-tagged-pdf` 导出结构树，pypdf 完整克隆页面后添加书签；构建和发布门禁检查标记根、父树及页面连接。标签存在不等于公式辅助文本、阅读顺序或 PDF/UA 已完整验收，最终版仍需辅助技术实测。

**独立结构基线**：发布门禁的独立结构基线为 14 个顶级书签、119 个二级书签、140 个三级书签，共 273 个大纲项，以及图 1～39。它还检查图号与 alt、公式编号与引用、小节语义链接、每个源 h2/h3/h4 标题是否真的出现在当前页导航中（源 h1 可排除），以及 PNG 中的 `SourceScript` 和完整 `SourceScriptDigest`。

修改绘图脚本后未重画的 PNG 会使门禁失败；高风险口语命中只输出人工复核提示。

**失败时保留旧产物**：完整 PDF 构建先在临时 HTML/PDF 上完成打印、书签和链接校验，再替换正式文件。站点批量替换与 PDF 双文件替换发生可捕获异常时，会恢复已有文件并移除本批新建文件；若恢复也失败，会保留备份目录并报告路径。这不是断电、进程强杀或文件系统损坏时的原子发布保证。

`--html-only` 只替换 HTML；旧 PDF 的摘要会与新 HTML 不同，必须继续生成 PDF 后再发布。

**扩展依赖**：`pyroomacoustics==0.10.0`（房间声学仿真库）只用于第 10 章示例和附录 B 的房间仿真练习。可用另建的临时虚拟环境安装并运行房间脚本，保留仓库 `.venv` 的主依赖集；本轮实测使用 Python 3.13.12。没有该依赖仍能生成正文的 39 张编号图，已随仓附录 B 的补充结果图和 18 个房间 WAV 可直接查阅。固定版本说明见[官方 PyPI 页面](https://pypi.org/project/pyroomacoustics/0.10.0/)。

在有 Python 3.13 的 macOS/Linux 主机上，可从仓库根目录用新目录复算，不覆盖本书样本：

```bash
python3.13 -m venv /tmp/masp-room-pra
/tmp/masp-room-pra/bin/python -m pip install pyroomacoustics==0.10.0 matplotlib
PRA_NUM_THREADS=2 /tmp/masp-room-pra/bin/python -m codes.examples.room_srp_exercise --run --plot /tmp/masp-room-result.png --audio-dir /tmp/masp-room-audio-new
```

目标图文件及音频目录应事先不存在。重跑后用新目录清单的 SHA-256 对照 `codes/room_audio/MANIFEST.json`，并记录 Python、NumPy、SciPy、pyroomacoustics 与线程数；跨平台绘图字体可能改变 PNG 字节，数值和 WAV 应分别核查。
