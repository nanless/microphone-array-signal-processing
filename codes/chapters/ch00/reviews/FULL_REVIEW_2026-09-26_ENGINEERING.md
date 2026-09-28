# 2026-09-26 工程、追踪与选型专项审查

本记录属于全书审查的工程分工。先只读审查并向主智能体报送 ENG-01～09，主智能体合并确认后进入互斥文件修改；ENG-10 为读图补充发现，ENG-11 为确认后追加的开源实现研究。没有提交或推送，生成物由主智能体统一构建。本记录的行号指审查时位置，修改后以主题和练习 ID 定位。

## 缺陷与验收记录

| ID | 文件与行号/图号 | 类别 | 严重度 | 证据 | 修改建议及实际处理 | 负责人 | 状态 | 验证结果 |
|---|---|---|---|---|---|---|---|---|
| ENG-01 | chapters/10_engineering-practice.md:303 | 接口、口径 | P1 | 原文说 VAD 输入均方能量；实际 update(frame) 接受波形，再算 RMS | 明确波形输入与均方能量阈值，加入 0.3² 与 0.09² 反例 | engineering | 已验证 | 独立调用复现错误；E10-15 输入幅度 sqrt(energy)，8 项新测试与原回归通过；待统一产物检查 |
| ENG-02 | codes/research/01_spatial_and_tracking.md:414 | 坐标约定 | P1 | (0.866,0.5) 按本书 atan2(x,y) 得约60°，却写30° | 改为 (1/2,sqrt(3)/2)，预测为一半；解释外部 +x 零点需转换 | engineering | 已验证 | 独立 atan2 复算30°、向量模0.5；待研究页渲染 |
| ENG-03 | chapters/09_source-tracking.md:201–211 | 概率递推假设 | P2 | 十粒子例子只写似然，未说明先验等权；代码实际乘旧权重 | 明示等权先验，E09-07 给一般 bootstrap 递推式(9-13)与双证据反例 | engineering | 已验证 | 独立似然比 e^.5/e^1 与运行结果相符；待正文产物检查 |
| ENG-04 | codes/array_tutorial/tracking.py:34–38 | 数值边界 | P2 | diag(-0.1,1) 被拒绝，乘1e-14后因绝对特征值容差被接受 | 对称性与半正定性检查先按最大模归一化，接受舍入级非对称后显式对称化 | engineering | 已验证 | 对协方差和Q分别检查1e-200、1、1e200尺度的负特征值与非对称；合法秩一和零矩阵通过；43项相关测试通过 |
| ENG-05 | chapters/09_source-tracking.md:446–470 | 时间与不确定度 | P2 | 现有80ms滞后例子缺少测量/发布/消费时刻及协方差可执行对应 | E09-08 完整传播状态和P，拒绝未来/过期状态；明确未实现乱序融合器 | engineering | 已验证 | 手算32.4°及P=[[4.217941333,1.7264],[1.7264,9.16]]与测试吻合；待产物检查 |
| ENG-06 | chapters/10_engineering-practice.md:301–303 | 组合实验 | P2 | 原有VAD与RingBuffer未通过组合例子展示预卷和事件边界 | E10-15 保留帧1–6，区间[160,1120)，触发40ms，结束事件80ms | engineering | 已验证 | 新测试核对无重复、无历史、静默、两次触发和未结束流；待产物检查 |
| ENG-07 | chapters/11_selection-guide.md:231–249 | 统计验收 | P2 | 有成功覆盖与分位数题，缺少零失败试验的风险上界 | E11-08 推导单侧精确二项上界，注明独立性、同p、固定停止规则与频率学派含义 | engineering | 已验证 | 20/100/1000次上界及298/299边界独立反代通过；待产物检查 |
| ENG-08 | codes/research/03_industrial_deployment.md:267 | 运行状态 | P3 | 总述称全部是建议实验，但I22/23/26有执行记录 | 改为无执行记录者才是建议，主机执行结果不可外推设备验收 | engineering | 已验证 | 与本节已有记录逐项核对；本次未重跑这三项外部实验；待研究页检查 |
| ENG-09 | figures/fig22_tracking.png，fig_tracking | 图表可读性 | P2 | 实际查看图22，左图图例遮住后段轨迹 | 主智能体将图例移出数据区 | 主智能体 | 本次处理 | 源图已读；重画及最终尺寸验收由主智能体记录 |
| ENG-10 | chapters/10_engineering-practice.md:88–90，图25 | 图文引用 | P2 | 实际图是关键路径条形与底部说明，没有(a)/(b)子图 | 改为图25关键路径条形、底部说明 | engineering | 已验证 | 与原图结构核对；待新版成品检查 |
| ENG-11 | codes/research/03_industrial_deployment.md，新I28 | 工业源码覆盖 | P2 | 新固定FastEnhancer源码有显式缓存、频谱掩码、ONNX流式循环，原研究未覆盖 | 增加I28源码阅读顺序、缓存/排空/裁剪、NS边界与可执行实验设计 | engineering | 已验证 | 直接读固定README、MIT LICENSE、default/model.py、export_onnx.py、test_onnx.py；权重未取得，未运行推理；待研究页检查 |

| ENG-12 | codes/examples/tracking_time_exercises.py，新时间接口 | 浮点与输入类型 | P2 | 独立复审发现1→1.1、1000→1000.1的0.1s有效期因减法舍入误拒绝；直接转float可能先抹掉复数类型 | 有效期边界允许时间戳量级2ULP，仅处理舍入；使用共享实数校验器先拒绝复数/布尔/文本/对象 | engineering | 已验证 | 加入0/1/1000/1e6时基平移、真实超期与所有新公开实数输入的类型测试；43项相关测试通过 |

## 实际阅读与复算范围

已通读 AGENTS.md；通读章节09、10、11、13及研究手册01、03，核对它们涉及的公式、表格、练习、跨章引用和工程声明。阅读 tracking.py、engineering.py、exercises_engineering.py、追踪交叉缺测示例及相关测试；查看图22、23、25。图23标签靠近边缘的问题已提醒主智能体复查，没有未经授权修改绘图脚本。

独立复算包括粒子权重两次递推、ACCDOA方向和向量模、连续白角加速度Q、测量时间外推的互协方差项、VAD波形与能量阈值、预卷采样索引、零失败二项概率和最少299次试验。原有E10-14输出三次逾期与667.5KiB峰值内存、交叉追踪的两次身份错配也已运行核对。没有把无标签位置集合正确当作身份正确。

实际打开的外部原始/官方依据：

- Stone Soup 1.9.1 粒子滤波权重教程及固定延迟乱序观测示例；用于区分等权例子、一般重要性递推和乱序观测处理。
- NIST 精确二项区间说明；单侧零计数式由本书另行反解，未混用原页双侧alpha/2。
- CHiME-10官方任务与日期页、DCASE2026官方挑战页；核对既有任务和时间范围，没有把后续挑战说明写成已完成榜单。
- ITU P.835、P.566官方目录页；核对2026年7月版本状态，没有读未取得的标准全文或新增条款结论。
- PortAudio官方回调说明；核对实时回调限制，不将接口说明当作某设备时延实测。
- FastEnhancer固定提交的本地官方源码与MIT许可；只读接口，不引用未经本次复算的性能数字。

上述网站于2026-09-26核对；正式引用链接放在对应章节/研究条目中。本记录不把读完目录等同于审完全部外部仓库实现，也不宣称全部工业算法已运行。

## 已运行验证

```bash
.venv/bin/python -m unittest tests.test_codes_engineering tests.test_codes_aec_wpe_sep_track tests.test_codes_sro_closed_loop -q
.venv/bin/python -m unittest tests.test_codes_time_state_exercises tests.test_codes_engineering tests.test_codes_aec_wpe_sep_track tests.test_codes_sro_closed_loop -q
.venv/bin/python -m codes.examples.tracking_time_exercises
```

第一次为修改前35项通过；第二次为修改后43项通过，包含新增8项具有独立期望值和边界的测试。新runner输出 E09-07、E09-08、E10-15、E11-08 四个ID。还执行了现有 exercises_engineering、exercises_enhancement 的 run_exercises() 与 tracking_crossing_dropout_demo.run_experiment()，未新增依赖或随机输入。差异空白检查通过。

## 生成物与尚未验证范围

此分工不构建共享HTML/PDF、不重新生成主音频或图片，避免并行覆盖。09/10/11/13及research01/03等待主智能体统一重建、页面与PDF分页/书签/公式/图表验收后更新对应状态。章11新增一个h4：`题 8：零次失败能证明多低的风险（E11-08）`；其他三章没有增加h3/h4。research03新增一个h3：I28。

主智能体负责图40及clock_drift组4个WAV；第10章已注明16kHz、100ppm、8s与理想同步目标的解释。这里没有独立试听或声卡测试，也没有核对其最终WAV摘要。章13的40张图、64主WAV与I28索引计数按合并计划同步，最终数量须以主智能体验收结果为准。

未运行FastEnhancer权重推理、ODAS真实设备追踪、Stone Soup完整跟踪器、外部评分模型或目标DSP；未重跑原文所有旧论文实验、房间仿真、真实录音派生流程。没有承诺补入全部可见源码；下载、固定摘要与许可状态由主智能体维护的清单和核验报告给出。

## 主审统一验收补记

源文件交接后已统一生成20篇网页与419页PDF，并完成全量710项测试（708通过、2项可选依赖跳过）。表内“待统一产物检查”等文字记录分工交接时的状态；最终状态以本补记和[总审查报告](FULL_REVIEW_2026-09-26.md)为准。分组已实际查看的成品页码、逐页缩略扫描与未验证范围见总报告，不把缩略检查称为逐字细读。工程组发现E10-15位置问题后已移回10.11，由主审重建并复看受影响页。
