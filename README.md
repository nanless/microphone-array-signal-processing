# 深入浅出麦克风阵列信号处理

一套写给初学者和入门研究生的麦克风阵列信号处理中文教程：从“为什么摆一群麦克风”讲到定位（DOA）、波束形成、回声消除（AEC）、去混响（WPE）、语音分离、声源追踪，一直到工程实现与选型。正文提供关键公式推导、可复算例子、适用边界和 33 张脚本生成的图。

English version: [README_EN.md](./README_EN.md)

## 目录结构

| 目录/文件 | 说明 |
|---|---|
| `chapters/` | 教程正文 14 篇 Markdown（`00_overview.md` 是入口，`01`～`11` 是 11 章正文，`12`/`13` 是附录 A/B） |
| `figures/` | 33 张插图（`fig01`～`fig33_*.png`），全部由脚本生成、可复现 |
| `scripts/` | 绘图与构建脚本（`make_figures.py`、`make_aec_figures.py`、`build_site.py`、`build_pdf.py`，说明见 `scripts/README.md`） |
| `site/` | 多级页面站（`index.html` 首页 + 13 个内容页，构建产物，可再生） |
| `dist/` | 合订 PDF（`microphone-array-tutorial.pdf`，导读、11 章正文和 2 篇附录均有顶级书签）与合订 HTML 中间产物 |

## 章节导览

| 篇 | 文件 | 内容 | 难度 |
|---|---|---|---|
| 导读 | `chapters/00_overview.md` | 全套导航、三条学习路径、插图地图 | 入门 |
| 第 1 章 | `chapters/01_problem-definition.md` | 噪声、混响、干扰、自噪声，阵列增益与双耳线索 | 入门 |
| 第 2 章 | `chapters/02_basics-signal-model.md` | 时延、远近场、信号模型、房间混响、STFT/协方差、波束图度量 | 进阶 |
| 第 3 章 | `chapters/03_array-geometry.md` | 线阵、圆阵、球阵、稀疏阵，端射灵敏度与阵列校准 | 进阶 |
| 第 4 章 | `chapters/04_doa-estimation.md` | GCC-PHAT、SRP、几何解算、Bartlett/Capon、MUSIC/ESPRIT、宽带聚焦、DNN 定位、CRLB | 较难 |
| 第 5 章 | `chapters/05_beamforming.md` | DSB、超指向、MVDR、LCMV、GSC、SPP、后置滤波、球谐、DNN 波束 | 较难 |
| 第 6 章 | `chapters/06_aec.md` | NLMS/FDKF/PBFDAF、双讲检测、非线性、混合神经 AEC、AEC3 | 较难 |
| 第 7 章 | `chapters/07_wpe-dereverberation.md` | WPE 原理与推导、Δ/K 选择、在线 WPE、手算例 | 进阶 |
| 第 8 章 | `chapters/08_speech-separation.md` | 混合模型、BSS、GSS、深度分离、TSE、连续会议分离与数据集 | 进阶 |
| 第 9 章 | `chapters/09_source-tracking.md` | KF 手算、粒子滤波、PHD 多目标、定位—追踪—波束接口 | 进阶 |
| 第 10 章 | `chapters/10_engineering-practice.md` | 参考链路、关键路径延迟、SRO/标定、资源预算与评测 | 进阶 |
| 第 11 章 | `chapters/11_selection-guide.md` | 条件化选型、场景约束、可验证规格与练习 | 入门 |
| 附录 A | `chapters/12_appendix-symbols-math.md` | 符号表、术语定义、预备数学速览 | 查阅 |
| 附录 B | `chapters/13_appendix-guide.md` | 学习路径、领域地图、研究前沿、排错、17 道练习、复现说明 | 查阅 |

## 快速开始

```bash
# 1. 建虚拟环境并装依赖
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# 可选：附录 B 第 16 题和第 10 章房间示例按 pyroomacoustics 0.10.0 编写
.venv/bin/pip install pyroomacoustics==0.10.0

# 2. 生成 33 张图（运行时间随硬件、软件版本和负载变化）
.venv/bin/python scripts/make_figures.py
.venv/bin/python scripts/make_aec_figures.py

# 3. 建多级页面站（输出 site/*.html）
.venv/bin/python scripts/build_site.py
# 浏览器打开 site/index.html（双击即可；公式需联网加载 MathJax 渲染）

# 4. 生成合订 PDF（输出 dist/microphone-array-tutorial.pdf，需本机装有 Google Chrome，
#    非 macOS 可用 CHROME_BIN 环境变量指定 Chrome 路径）
.venv/bin/python scripts/build_pdf.py
# 需要可复现的封面日期时，加 --build-date YYYY-MM-DD，或设置 SOURCE_DATE_EPOCH

# 5. 发布前检查
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/quality_check.py
```

## 学习路径

- **路径 A（零基础入门，2～3 周）**：导读 → 01 → 11.1/11.3 → 02/03 → 04（GCC+SRP）→ 05（DSB+MVDR）→ 06/07/08 → 09 → 复现并解释 33 张图。
- **路径 B（工程实现，1 周精读）**：11.1/11.2/11.3 定 A/B/C 方案 → 05/06/07/08 → 10 全读 → 输出延迟/同步/标定三张预算表。
- **路径 C（研究前沿）**：02（CRLB）→ 03（稀疏阵）→ 04/05 前沿 → 06/07/08 → 13.3 九条前沿 + 13.6 练习。

## 约定

- 需要跨段或跨章引用的核心公式在公式块内使用 `\tag{章-序}` 编号，正文引用写“见式(5-1)”。
- 缩写首次出现给全称；"dB 换算用 10log（功率）/20log（幅度）"。
- 数字凡涉榜单均标注条件与出处，仿真数字注明实现口径。
- 外部公式、算法和数据优先链接 DOI、标准组织或官方页面；引用时核对标题、作者、年份和具体表/节，不能只检查链接能否打开。
