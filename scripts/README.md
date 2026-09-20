# Scripts：绘图与工具脚本

本目录放教程的全部可运行脚本。在**报告根目录**（不是本目录）执行（图 1~25 约 20 秒，图 26~32 约 11 秒，全套约半分钟，机器不同有出入；Windows 上把 `.venv/bin/python` 换成 `.venv\Scripts\python`）：

```bash
.venv/bin/python scripts/make_figures.py      # 生成图 1～25、图 33 → figures/
.venv/bin/python scripts/make_aec_figures.py  # 生成图 26～32（回声消除专题）→ figures/
.venv/bin/python scripts/build_site.py        # 把 chapters/ 建成多级页面站 → site/（首页+13篇，带侧边栏跳转）
.venv/bin/python scripts/build_pdf.py         # 合订 chapters/ → dist/combined.html → dist/microphone-array-tutorial.pdf（需 Chrome）
```

| 脚本 | 作用 | 输出 |
|---|---|---|
| `make_figures.py` | 约 1900 行。23 张旧图 + 图 24（WPE 帧条带）+ 图 25（延迟预算瀑布）+ 图 33（GCC 两种实现对照）。只用 numpy 和 matplotlib，不依赖 scipy。随机种子固定，结果可复现。图 13 蒙特卡洛最耗时 | `figures/fig01~fig25_*.png` + `fig33_*` |
| `make_aec_figures.py` | 7 张回声消除专题图（建模/结构/NLMS/冻结/延迟双讲/非线性/混合选型）。风格与上一个脚本统一（六色/五级字号/dpi150） | `figures/fig26~fig32_*.png` |
| `renumber2.py` | 已退役。单篇长文时代的图号整理工具，留作存档，平时不用跑 | 无 |
| `build_site.py` | 建站脚本。读 `chapters/` 14 篇 Markdown，用自带 markdown 库转成静态页面：首页 + 13 篇正文，左侧边栏可跳章节与小节，文末上一篇/下一篇。图片直接引用 `figures/`，数学公式走 MathJax 在线渲染 | `site/*.html`（共 14 页） |
| `build_pdf.py` | 合订本脚本。14 篇合成带目录超链接的单页 HTML，再调 Chrome 无头打印成 PDF。打印约 1 分钟（非 macOS 可用 `CHROME_BIN` 环境变量指定 Chrome 路径） | `dist/combined.html`（中间产物） + `dist/microphone-array-tutorial.pdf` |

改图练习（如附录 B 习题）：改对应 `fig_*()` 函数里的参数，重跑本目录脚本，到 `figures/` 看效果。

备注：合订本 PDF 仅一级书签（章节级，无节级锚点）。合订本构建需联网：公式经 MathJax CDN 渲染，离线时公式显示为源码落版。

备注：`daimon_runtime` 是可选的内部样式包，有没有都不影响出图，没有会自动降级成默认样式，不用装。`pyroomacoustics`（房间声学仿真库）是扩展依赖，只给附录 B 第 15 题用，不装也能出全部 33 张图。
