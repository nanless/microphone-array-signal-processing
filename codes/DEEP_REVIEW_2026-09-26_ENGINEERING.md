# 2026-09-26 工程边界、工业接口与选型证据深查

## 范围、分工与原有记录

主审分配第 10、11 章、附录 B、研究手册 03/04、工程教学接口及获取工具的只读审查。先通读 `AGENTS.md`，检查初始干净工作区，阅读同日上一轮 `FULL_REVIEW_2026-09-26_ENGINEERING.md`、总报告和 `ROUND4_REVIEW.md`；已经修复的 VAD 输入、预卷、零失败上界等不重复作为本次缺陷。

全队只读阶段结束、主审确认 7 项后，开始互斥写入。工程分工只改 10/11/13 章、`engineering.py`、新边界练习与测试、研究 03/04 和本记录；共享目录、覆盖表、AGENTS、图、音频、网页、PDF、提交与推送由主审负责。附录 B 的 E13-02、图 41 与工业 I29 留给主审接续，不在本分工追加。

## 缺陷和处理

行号为只读基线位置，修改后按题 ID 和函数定位。交接时状态为“已修改”；以下表格已在统一网页/PDF验收后更新为“已验证”，证据见末节。

| ID | 文件与位置 | 类别 | 严重度 | 证据 | 修改建议及已完成处理 | 负责人 | 状态 | 验证结果 |
|---|---|---|---|---|---|---|---|---|
| EG2-01 | `array_tutorial/engineering.py` 280～294，10.1.2 | 数值与事件顺序 | P1 | 30 个服务时间与周期均为 0.1 ms 的帧、容量 1，本应无丢帧，旧函数只处理 26 帧；0.3 ms 误丢 2 帧 | 到达、完成、期限统一使用输入 float 的精确有理值；同刻完成先释放容量；增加 E10-16 | 工程分工 | 已验证 | 8 种周期含最小非正规数、1e308 均无伪丢帧；nextafter 真超期仍保留；81 种四帧整数调度独立对照通过 |
| EG2-02 | `array_tutorial/engineering.py` 222～224，10.3.4 | 数值边界 | P2 | 默认 AGC 对 `[1e17]` 返回零增益/静音，解析全攻击输出应为 0.8 | 用凸组合平滑，避免先减后加丢掉小增益；保留安全上限 | 工程分工 | 已验证 | 1、1e16、1e17、1e200、float64 最大幅度的输出均约 ±0.8；最小非正规数、静音及部分攻击测试通过 |
| EG2-03 | 10.9.1，原第 556 行 | 研究状态一致性 | P3 | 原称 27 个工业主题且统一核实于 9 月 22 日，实际已有 I28 与 9 月 26 日记录 | 改成按主题及条目日期查阅，区分文档建立、源码核对与运行日期 | 工程分工 | 已验证 | 对照 I28 和手册题目清单；不预填主审待追加 I29 的状态 |
| EG2-04 | 附录 B 原第 291 行，10.11 入口说明 | 执行入口 | P2 | 笼统声称全部工程/选型题由一个 runner 执行，E10-13/15、E11-08 已有独立入口 | 按基础、谱减、状态、边界分组说明；新增 runner 交给主审同步全局目录 | 工程分工 | 已验证 | 原有入口未删除；3 个新 ID 在题干、结果映射及独立测试一致 |
| EG2-05 | 10.1.1，研究 I01 | 测量教学 | P2 | 只有 PortAudio 三时间字段，缺少同一时钟、同一样本对应与计算重复计时反例 | E10-17 展开 15/30/45 ms，以及错后一个块变为 55 ms；明确单位不是硬件精度 | 工程分工 | 已验证 | 整数纳秒相减；±1e18 原点平移结果一致；错误顺序、类型及不可表示差值拒绝 |
| EG2-06 | 11.5/11.6 | 统计与选型 | P2 | 平均错误下降与逐会话配对方向证据缺少可算对照 | E11-09 给六会话错误计数、幅度统计、四胜一负一平、单侧符号检验式(11-2) | 工程分工 | 已验证 | 独立枚举 32 个符号排列，尾部 6 个，p=0.1875；70/600、59/600、11/6 百分点和 11/70 相对降幅复算 |
| EG2-07 | 研究 I18、复现手册 | 工业源码接口 | P2 | GPU 传输与缓存仅有总述，缺绑定对象、动态形状、同步及计时范围 | 阅读固定 ONNX Runtime C API，补 5 行接口表、状态位置与三路受控试验设计 | 工程分工 | 已验证 | 静态核对 `RunWithBinding`、绑定和同步 API；没有运行 GPU，未写性能数字 |

## 逐篇阅读、独立计算与运行范围

已通读第 10 章 745 行、第 11 章 277 行、附录 B 556 行和工业/复现手册 427/220 行的只读基线；重点检查公式、例子、练习、接口和来源层次。已读 `engineering.py`、获取工具 `fetch_upstreams.py`，工程练习、状态题和工程测试；核对锁表中 WebRTC/ONNX Runtime 的固定条目及本地源码路径。没有把这些接口阅读写成已审完全部 75 个项目实现。

独立复算范围：E10-16 的整数刻度队列，E10-17 的三个端点差及一块延后，E11-09 的逐会话差、合并 WER、百分点/相对降幅、32 个符号排列。旧文中的 Q15 两抽头、0.5 mm/4 kHz 相位上界、六帧资源预算、零失败 299 次边界也按公式与相关测试复核；没有重跑所有外部库或房间仿真来重新认可旧实测数字。

运行范围：

```bash
.venv/bin/python -m unittest tests.test_codes_engineering tests.test_codes_engineering_round2 tests.test_codes_time_state_exercises tests.test_codes_exercises_engineering -q
.venv/bin/python -m unittest tests.test_paragraph_engineering_round2 tests.test_codes_engineering_boundaries tests.test_codes_engineering tests.test_codes_engineering_round2 tests.test_codes_time_state_exercises tests.test_codes_exercises_engineering -q
.venv/bin/python -m codes.examples.engineering_boundary_exercises
```

第一条在写入前 51 项通过；第二条修改后 63 项通过，其中 10 项为新数值、事件、时间戳和统计测试，2 项是既有段落检查。新 runner 输出且只输出 E10-16、E10-17、E11-09 三个 ID。没有安装新依赖，`Fraction` 来自 Python 标准库，普通测试不联网；差异空白检查通过。

最小非正规数与有限最大值是实现边界夹具，不是可接受声压或设备时序规范。精确有理模拟保留浮点输入本身的值，不推断调用方本来想输入哪个十进制有理数，也不承诺实时运行。

## 检索和原始来源

核实日期均为 2026-09-26。搜索用于定位，下列页面实际打开并阅读；工业 API 另与本地固定源码对照。

| 检索主题 | 阅读来源 | 采用理由与边界 |
|---|---|---|
| `PortAudio inputBufferAdcTime outputBufferDacTime currentTime same time base` | [PaStreamCallbackTimeInfo](https://files.portaudio.com/docs/v19-doxydocs/structPaStreamCallbackTimeInfo.html)；[缓冲时序指南](https://github.com/PortAudio/portaudio/wiki/BufferingLatencyAndTimingImplementationGuidelines) | 明确字段时刻、秒单位、所属流时间基准；后者明确标为草案，只用于说明已知/未知延迟，不能证明当前设备精度 |
| `ONNX Runtime I/O Binding device copies` | [官方 I/O Binding](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html)；[固定 C API](https://github.com/microsoft/onnxruntime/blob/48795e0281bcaa6b3b63b28af5e078b4c41e995f/include/onnxruntime/core/session/onnxruntime_c_api.h) | 固定 API 实际存在，能展开设备复制、动态形状和按后端同步；未运行绑定实验，不填加速比 |
| `NIST sign test paired binomial` | [NIST Sign Test](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/signtest.htm) | 使用配对符号、平局与二项模型；正文明确检验非平局胜率，未误写为平均 WER 显著性 |
| `WebRTC ValidRateAndFrameLength 48000 10 20 30` | 定位官方 `webrtc_vad.c`，另见本地锁定源码 | 确认现有工业 VAD 主题已有明确接口；没有新增依赖、无必要重复增加同义主题或声称已运行 VAD |

此次不新引入上游项目。ONNX Runtime 与 PortAudio 的固定源码已在独立下载区存在，本分工读取所需接口，没有改写上游、取得模型权重或把其源码再复制进本书发布文件。

## 交接与未验证范围

标题计数仅第 11 章增加一个 h4（10→11），题名“题 9：平均错误减少，配对证据足够吗（E11-09）”；第 10 章仍 h3=12/h4=13，附录 B 仍 h3=7/h4=22，研究 03 仍 h3=28/h4=2，研究 04 仍 h3=6/h4=0。式(11-2)顺接式(11-1)，未改已有题号或标题。全局题目录、覆盖表与书签独立清单由主审更新。

未进行设备采集/播放、主观听测、GPU I/O Binding、全上游重编译、房间仿真或模型推理。分工阶段未构建网页/PDF、未目视新成品页、未生成音频或图。全部条目的最终“已验证”状态需在主审统一构建和对应页面检查后补记；当前源文件和数值通过不代替成品验收。

AGENTS 建议：其现有边界条款已经涵盖本次修复，不必逐 bug 追加专用规则；若主审整合新要求，可在时间状态检查中明确“同刻事件顺序与下一可表示值的真实越限分别测试”。避免只增加“有限/不削波”式检查，因为错误静音同样能通过这类过弱判据。


## 最终统一验收（2026-09-26）

本组独立查看20个网页首屏联系图及新增主题原图，并补拍E10-16、E10-17、E11-09、图41与I29。网页公式、段落、时间算例和图例正常；自动检查20页桌面/390px窄屏无横向溢出、MathJax错误或破图，80个音频控件元数据及标签通过。

PDF逐张查看联系图25～37，覆盖289～437页共149页缩略版式总览，不称逐字细读。110dpi详情页为346、347、361、373、377、378、391、392、433～437。E10-16/17均在377；E11-09在391～392；E13-02在434～436，图41在435。时间表、式(11-2)、式(13-3)～(13-5)、解析/PCM表和图41均清晰完整。最终分页调整没有改变此组已查详情页；主审逐页渲染摘要验证。

全回归754项（752通过、2可选外部环境测试跳过），最终构建/历史锚64项通过，质量门禁通过。工程7项与主审新增插值/STK内容均已完成授权范围的验收；未执行的设备/GPU/听测边界仍按前文保留。
