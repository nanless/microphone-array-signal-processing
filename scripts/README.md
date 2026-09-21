# Scripts：绘图与工具脚本

本目录放教程的全部可运行脚本。在**报告根目录**（不是本目录）执行。运行时间取决于处理器、操作系统、Python 与依赖版本和当前负载；若要报告耗时，应同时记录这些条件、运行次数和统计方式。Windows 上把 `.venv/bin/python` 换成 `.venv\Scripts\python`。

```bash
.venv/bin/python scripts/make_figures.py      # 生成图 1～25、图 33 → figures/
.venv/bin/python scripts/make_aec_figures.py  # 生成图 26～32（回声消除专题）→ figures/
.venv/bin/python scripts/build_site.py        # 把 chapters/ 建成多级页面站 → site/（首页+13篇，带侧边栏跳转）
.venv/bin/python scripts/build_pdf.py         # 合订 chapters/ → dist/combined.html → dist/microphone-array-tutorial.pdf（需 Chrome）
.venv/bin/python scripts/quality_check.py      # 发布前检查占位符、图片、链接、书签和本地路径泄露
.venv/bin/python -m unittest discover -s tests -v  # 运行构建与算法回归测试
```

| 脚本 | 作用 | 输出 |
|---|---|---|
| `make_figures.py` | 生成图 1～25 和图 33。只用 numpy 和 matplotlib，不依赖 scipy；随机种子固定，结果可复现。图 13 的蒙特卡洛统计耗时最长 | `figures/fig01~fig25_*.png` + `fig33_*` |
| `make_aec_figures.py` | 7 张回声消除专题图（建模/结构/NLMS/冻结/延迟双讲/非线性/混合选型）。风格与上一个脚本统一（六色/五级字号/dpi150） | `figures/fig26~fig32_*.png` |
| `renumber2.py` | 已退役。单篇长文时代的图号整理工具，留作存档，平时不用跑 | 无 |
| `build_site.py` | 建站脚本。读 `chapters/` 14 篇 Markdown，用 markdown 库转成静态页面：首页 + 13 个内容页，左侧边栏可跳章节与小节，文末上一篇/下一篇。图片直接引用 `figures/`，数学公式使用固定版本的 MathJax 3.2.2 在线渲染 | `site/*.html`（共 14 页） |
| `build_pdf.py` | 合订本脚本。14 篇合成带封面、两级目录的单页 HTML，再调 Chrome 无头打印成 A4 PDF，最后写篇/节两级书签。输出先写临时文件，校验后再替换发布件。常用 flag：`--html-only`、`--pdf-only`、`--no-bookmarks` | `dist/combined.html`（中间产物） + `dist/microphone-array-tutorial.pdf` |
| `quality_check.py` | 发布门禁。检查正式正文无编辑占位、33 张图齐全、站内链接和锚点有效、合订 HTML 无分篇死链、PDF 无 `file://` 且书签完整 | 通过或带文件位置的失败清单 |

改图练习（如附录 B 习题）：改对应 `fig_*()` 函数里的参数，重跑本目录脚本，到 `figures/` 看效果。

备注：合订本 PDF 为篇/节两级书签：顶层是导读、11 章正文和 2 篇附录，第二层来自各篇实际小节。合订本构建需联网加载固定版本的 MathJax；正式发布前必须抽查公式已经渲染。

备注：`pyroomacoustics`（房间声学仿真库）是扩展依赖，只给附录 B 的房间仿真练习使用，不装也能生成正文的 33 张图。
