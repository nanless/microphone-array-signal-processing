# 2026-09-26 基础、定位与波束分区审查

本记录属于全书审查的一路，负责人为 `foundations_spatial`。先只读审查，再由主代理合并缺陷并授权写入。
本分区不负责提交、推送或生成全书产物；正文生成物的最终验收由主代理统一执行。

## 范围与方法

- 已读完整 `AGENTS.md`，逐篇通读第 1、2、3、4、5 章及附录 A（第 12 篇）。
- 阅读 `covariance.py`、`beamforming.py`，核对 `doa.py` 的 GCC、扫描谱和矩阵求解相关实现；
  查阅现有空间练习、研究手册 01 的算法及源码入口。没有声称逐行执行所有外部实现。
- 独立复算第 2 章 6 麦 CRB 示例，角度标准差为 1.194193839°；双麦端射超指向例的
  WNG 为 −10.64470438 dB、DI 为 5.99195745 dB，与正文舍入结果相符。
- 目视检查图 7、14、16 的 PNG，图 16 的阵列配置与双麦手算不是同一设定；本分区没有改图。
  未在本分区逐页验收 HTML/PDF，也未逐一核查其他图的最终 A4 字号。
- 对极端有限输入实际运行最小反例，确定归一化溢出可静默产生零权重，随后加入针对该行为的回归。

## 缺陷与处置

行号指修改前的源文件位置，便于对照本轮基线；状态没有把待构建正文标成已验收。

| ID | 文件与行号/图号 | 类别 | 严重度 | 证据 | 修改建议及实际处理 | 负责人 | 状态 | 验证结果 |
|---|---|---|---|---|---|---|---|---|
| FS01 | `chapters/04_doa-estimation.md:284` | 量化误差 | P2 | 原文称网格间距是误差下限；真值在格点时误差为零 | 改成理想最近格点量化界，明确不保证有噪 SRP 精度；E04-10 加三个反例 | foundations_spatial | 已验证 | 2° 网格下复算误差 0°、0.5°、1°；源文复读，生成物待统一验收 |
| FS02 | `chapters/04_doa-estimation.md:374` | 模型范围 | P2 | 将增广 Chan 第一阶段的三个未知量条件泛化成二维定位至少四麦 | 限定第一阶段无约束线性求解，并补列满秩条件 | foundations_spatial | 已验证 | 2×3 与 3×3 的方程数核对；未把方程数足够等同几何可解 |
| FS03 | `chapters/05_beamforming.md:124` | 归一化解释 | P2 | 逆矩阵大直接推出最终权重大，与下一段正横方向 `[0.5,0.5]` 矛盾 | 补导向投影与归一化分母，保留正横反例与端射对照 | foundations_spatial | 已验证 | 独立复算双麦正横与端射，源文一致；生成物待统一验收 |
| FS04 | `chapters/05_beamforming.md:347`、图 16 | 跨引用 | P2 | 双麦手算称完整曲线见图 16，但图 16 为其他阵列比较 | 删除“完整曲线见图16”的错误指引，保留五方向手算表 | foundations_spatial | 已验证 | 实际打开图 16，与正文配置比较；没有擅自将不同实验曲线等同 |
| FS05 | `chapters/12_appendix-symbols-math.md:123,135` | 术语 | P2/P3 | 导向矢量解释漏幅度；把高度相关等同完全相干降秩，并称 MUSIC 必失效 | 补复幅相、近场边界，区分高相关与完全相干，限定未经去相关的标准 MUSIC | foundations_spatial | 已验证 | 与第 2/4/5 章流形及秩条件静态交叉核对 |
| FS06 | `chapters/03_array-geometry.md:307`、`04_doa-estimation.md:124` | 教学段落 | P3 | 标定记录/加载拒绝/留出验证挤一段；三点互相关计算混排 | 标定分三段，互相关改三行数表并单列峰值解释 | foundations_spatial | 已验证 | 逐值核对 0、0、6；标题层级未改 |
| FS07 | `codes/array_tutorial/beamforming.py:86`、`doa.py:243` | 数值溢出 | P2 | `R=I,a=[1e200,1e200]` 有限输入下 MVDR 静默返回零；Capon 也把溢出二次型转为零谱 | 检查求解、二次型和最终结果有限性；超范围明确拒绝；正文说明此实现边界 | foundations_spatial | 已验证 | 反例现在报 ValueError；1e−100、1、1e100 导向仍满足解析权重/响应；普通回归通过 |
| FS08 | `codes/array_tutorial/beamforming.py:39,42` | 输出溢出 | P2 | 两路 `1e308` 输入、权重均为 1，返回 inf | 两种权重形状统一计算后检查输出有限性 | foundations_spatial | 已验证 | 向量/频率栈权重都拒绝溢出；1e300 等权平均正常返回 |

本分区未发现新的 P0/P1。没有删除已有算法章节、练习或改变已有编号。
FS01–FS06 的源文与计算已核对，正式“已验证”仍取决于主代理对生成物的验收结果。

## 三道增补练习与复算链

新增源文件 `codes/examples/spatial_precision_exercises.py`，运行无联网、无随机数、不写文件，
输出 NumPy 版本与确定性数学例子标记；新增 `tests/test_codes_spatial_precision.py`。

| 稳定题号 | 位置 | 解决的缺口 | 独立验算与边界 |
|---|---|---|---|
| E02-07 | 第 2 章既有练习节末 | 遗忘因子怎样对应物理更新时间 | 几何级数推导 `T=-H/log(alpha)`；8/16 ms 得 0.395987/0.791973 s；维持 0.4 s 后均衰减到 1/e；99 的权重集中度不等于重叠帧的独立样本数 |
| E04-10 | 第 4 章既有练习节末 | 麦对冗余、时差闭合与公共参考相关性 | 等权投影以约束平面的正交条件核验；独立到达误差传播得 `[[200,100],[100,200]]` μs²；说明并非所有 GCC 误差都服从该模型；含网格反例 |
| E05-06 | 第 5 章既有练习节末 | WNG 如何联系最坏目标响应失配 | 柯西不等式得 `epsilon/sqrt(WNG)`，构造负权重方向的扰动达到界；分别核对功率与幅度，不将任意复数球当成真实角度误差模型 |

未新增 h3/h4，因此不需要因本分区新增历史标题别名。共享练习目录、覆盖表和构建由主代理统一更新。
没有为纯代数题新增重复 WAV，也没有将已有对齐平均音频改称 MVDR 实验；全队音频增补另有负责人。

## 一手资料核查

核查日期为 2026-09-26。以下链接均实际打开原文，不以搜索摘要支持正文结论。

1. [Velasco 等，TDOA Matrices: Algebraic Properties and their Application to Robust Denoising with Missing Data](https://publications.idiap.ch/downloads/papers/2015/Velasco_IEEE_2015.pdf)：
   核对 §IV、式(1)–(2) 的到达时差代数定义及后续矩阵秩结构。本轮只用其支撑闭合与冗余概念，
   等权小例为本书自行推导；没有声称运行其完整鲁棒去噪与缺失值算法。
2. [Vorobyov、Gershman、Luo，IEEE TSP 51(2), 313–324，2003](https://users.aalto.fi/~vorobys1/RobBeamformer.pdf)，
   DOI 10.1109/TSP.2002.806865：核对 §III-A、式(18)–(25) 的范数有界不确定集和最坏响应约束；
   本题只验证固定权重的界，不冒称实现完整 SOCP 鲁棒波束形成器。
3. [SVD-PHAT，arXiv:1811.11785](https://arxiv.org/abs/1811.11785)、
   [多源扩展，arXiv:1906.11913](https://arxiv.org/abs/1906.11913)：核对原始条目后对照现有研究手册，
   发现已有相应覆盖，本轮不重复堆砌方法名。

另发现 2025 耳戴阵列遮挡工作
[Microphone Occlusion Mitigation for Own-Voice Enhancement in Head-Worn Microphone Arrays Using Switching-Adaptive Beamforming](https://arxiv.org/html/2507.09350v1)。
本轮仅完成检索和初读，未取得/核验可重分发代码，未复现实验，也未完成最终出版版本逐项核对。
因此不将它作为新增正文算法或工业性能结论；后续若收录，应先确定官方代码与数据可得性、
遮挡模型、VAD 错误设置及自声失真指标。该项为研究线索，不是已验证缺陷。

## 实际执行与限制

已执行：

```bash
.venv/bin/python -m unittest tests.test_codes_doa_beam tests.test_codes_spatial_round3 tests.test_codes_spatial_round4 tests.test_codes_foundations_round3 tests.test_codes_beamformer_common_input tests.test_codes_spatial_precision -q
.venv/bin/python codes/examples/spatial_precision_exercises.py
git diff --check
```

上述最终测试共 57 项通过；新例脚本三道题均运行并输出有限 JSON 数值。统一入口
`run_exercises()` 严格返回题号到结果字典的映射，元数据放在每题内部。测试用独立分数、
解析范数和约束正交性验证关键答案，不只对照函数生成的同一组数字。

本分区未执行全书全部测试、图像重生、HTML/PDF 重建或逐页排版验收；由主代理在源文件稳定后完成。
未重新下载外部代码，未新增第三方源码副本、权重或数据许可承诺。

另按主代理要求独立只读核查 `clock_drift_case()`、`test_codes_clock_audio.py` 及新增时钟音频说明。
同索引时间差、两正弦均值恒等式、1500 Hz 首次包络零点 3.333666667 s、500 Hz 的 10.001 s 均成立；
测试选取的内部样本区间同时避开两路边缘淡入淡出。发现研究手册重复第 19 节编号和
4 s 处 6.399364 样本的末位误写（应为 6.399360），已发给主代理修正，本分区未改写这些共享文件。

## 主审统一验收补记

源文件交接后已统一生成20篇网页与419页PDF，并完成全量710项测试（708通过、2项可选依赖跳过）。表内“待统一产物检查”等文字记录分工交接时的状态；最终状态以本补记和[总审查报告](FULL_REVIEW_2026-09-26.md)为准。分组已实际查看的成品页码、逐页缩略扫描与未验证范围见总报告，不把缩略检查称为逐字细读。工程组发现E10-15位置问题后已移回10.11，由主审重建并复看受影响页。
