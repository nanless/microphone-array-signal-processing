# 算法—源码—验证覆盖表

逐章核实日期：2026-09-28（第 1～11 章与附录 A/B 的教学基线、章节映射、练习和直接相关来源已逐章审查；表内其他上游条目的取得、依赖与运行状态仍以各自原始记录为准）。本表覆盖正文定义、推导或用于选型的方法，以及三篇研究文档中明确说明收录理由的扩展。每行限定具体计算步骤、算法变体或工业机制；同一算法的教学实现与外部对照不重复登记成不同状态。

四种覆盖状态：

- **本仓库可运行基线**：有教学实现、示例与回归测试；只覆盖该行说明的范围，不代表完整论文系统或产品。
- **外部参考实现**：已定位官方或作者实现并登记许可、版本；可能仍缺依赖、模型、数据或硬件，也可能存在已记录的实现疑点。
- **原理索引**：已有原理或来源依据，但尚未形成唯一、许可明确且承担对应计算的源码映射；代码可见而许可不明时也保留此状态，并说明原因。
- **明确排除**：指定软件的身份或许可不满足本书当前收录方式；不表示删除相应方法的学术讨论。

算法表共 349 行：本仓库可运行基线 77 行、外部参考实现 183 行、原理索引 88 行、明确排除 1 行。练习映射单独计数，不因题数增加算法行；已知激励的逐频正则反卷积新增唯一教学核，同一行保留pyfar外部对照及其限定运行边界。第6章新增一行已知播放增益顺序与历史尾声控制，复用NLMS与NCC，不计作新的自适应算法。MDL、功率谱减、受控 NCC 活动判别与 cACGMM 教学迭代属于本地基线。覆盖表仍有原理索引，不表示全书全部算法已经运行。

源码取得与入口核对见 [SOURCE_STATUS.json](SOURCE_STATUS.json)；该文件中的依赖验证和执行字段未开展时为 `not_run`，不承载方法级数值实验结果。实际运行及数值对照见[复现记录](research/04_source_reproduction.md)、[增强研究记录](research/02_aec_wpe_separation.md)和 [WPE 独立对照脚本](../ch07/examples/compare_wpe_reference.py)。工业接口与 SBL 的限定实验报告按主题放在 `codes/chapters/ch04/reports/`、`codes/chapters/ch10/reports/` 等对应章节；实际调用外部代码不将它改列为本仓库教学基线。覆盖状态不是测试结果。完整提交、官方地址、许可与来源 ID 见 [SOURCES.lock.json](SOURCES.lock.json)。表内本仓库教学源文件使用相对于仓库根目录的完整路径；第三方项目的内部路径仍相对于各自项目根目录。出现“同文件”时仅继承上一行文件，不继承其算法或验证结论。

详细模型、源码阅读与实验设计见 [空间与追踪](research/01_spatial_and_tracking.md)、[AEC、WPE 与分离](research/02_aec_wpe_separation.md)、[工业部署](research/03_industrial_deployment.md)。表中“研究扩展”不表示正文已有完整推导。第 6～8 章已按独立算法和工程任务拆节；下表按当前标题语义指向具体小节，源码取得与数值运行仍分别记录。

2026-09-29 第3章新增有外部相位锚的单频 ULA DOA—复增益教学拟合及无锚不唯一反例；这不改变未知几何自标定、歧义格点和失配前向模型等外部代码的原始取得状态，详见研究§39。2026-09-28 第 2 章复核新增 pyfar 扫频、正则反卷积与 Acoular 幅度校准三项源码映射；其他外部方法的核实范围沿用各自行内记录。

2026-09-29 第 4 章把 GCC 向量 MLP、共素阵 GCC-PHAT 图 CNN 与 STFT 相位图 CNN 分作三条原理索引。只核论文方法；未核到与论文逐项对应、许可明确且可再分发的作者实现，不能把相关但不同年份的代码当作该论文整链复现。图 11/14/33 的图文口径已按当前绘图源核对。

第9章的[当前接口报告](../ch09/reports/tracking_upstream_interfaces_current.json)与[当前原方法合同](../ch09/reports/upstream_tracking_contracts_current.json)分开记录ODAS原C、SAF原step计数替身与FilterPy/JPDA原方法。两份旧历史报告及来源锁保持原执行条件；sPIT仅为原理索引。26道代码题、图61和图77分别按解析、实际PCM与接口合同核查。新增两个限定机制行是单槽生命周期和一次静态vMF条件化；IMM原行增加完整标量教学递推，不把三道练习直接计作三个新算法。

2026-10-04 第5章复核另登记逐通道已知预滤波、原相位连续化函数及在线逐通道掩码原理三项及PAN原理索引；E23的三份相位控制不是新GEV算法，不另计算法行。当前限定执行见三份新的current报告，旧三报告保持真实原字节与747ec历史工具身份。

第5章复核新增单参考GSC状态更新、四项外部源码映射与六项原理候选；SOF静态反例、pb_bss原函数提取和PRA原包方法诊断分别记录。球阵两个直接依赖只扩充复现条件，不另计算法。

## 基础模型、几何与统计

| 正文或研究范围 | 算法/机制 | 覆盖状态 | 教学入口或主清单 ID：官方源码入口 | 关键边界 |
|---|---|---|---|---|
| §1.1；空间研究“第 1 章”专题 | 低通 HRIR 互相关时差估计 | 外部参考实现 | `spatial-audio-framework`：`framework/modules/saf_hrir/saf_hrir.c::estimateITDs` | 固定版 750 Hz 低通、整采样峰及限幅；非 Woodworth 真值或听觉阈值 |
| §1.1；空间研究“第 1 章”专题 | HRIR 到 HRTF 与方向插值 | 外部参考实现 | `spatial-audio-framework`：同文件 `HRIRs2HRTFs`、`interpHRTFs` | 已有响应的表示与插值，不从未知录音自动求方向；数据许可另核 |
| §1.1；空间研究“第 1 章”专题 | SOFA响应读取与方向插值接口 | 外部参考实现 | `libmysofa`：`src/hrtf/easy.c`、`src/hrtf/mysofa.h` | 固定v1.3.5源码已取得；原归一化两例保留零能量退化；`audit_libmysofa_interpolation.py`五人工控制确认原公共延迟保持失败，逐方向控制另列；未运行SOFA解析/自动查询/单位转换/重采样/渲染，非定位算法 |
| §2.2、§2.3 | 远场相对时延 | 本仓库可运行基线 | `codes/chapters/ch03/core/geometry.py::plane_wave_delays` | 坐标、角度零点、时延正号 |
| §2.3 | 平面波导向矢量 | 本仓库可运行基线 | `codes/chapters/ch03/core/geometry.py::plane_wave_steering` | 傅里叶符号、公共相位 |
| §2.2、§2.7 | 球面波导向与距离衰减 | 本仓库可运行基线 | `codes/chapters/ch03/core/geometry.py::near_field_steering` | 参考麦归一、零距离、远场退化 |
| §2.4；附录 B 第 16 题 | 镜像法 RIR | 外部参考实现 | `pyroomacoustics`：`pyroomacoustics/room.py`；`codes/chapters/appendix_b/examples/room_srp_exercise.py` 已调用锁定 0.10.0 生成六位置 RIR、DRR、T20 外推 T60 与 18 个试听 WAV | 全/直达 RIR 同长度和高通边界；反射阶数与设计 T60 不等于任意实测房间 |
| 研究扩展：空间 §2 | 射线追踪房间模拟 | 外部参考实现 | `pyroomacoustics`：`pyroomacoustics/libroom_src/` | 不自动包含衍射、结构传声 |
| §2.4；空间研究 §2.4 | 指数扫频生成 | 外部参考实现 | `pyfar`：`pyfar/signals/deterministic.py::exponential_sweep_time` | v0.8.1 源码已取得；信号生成不等于已测房间响应 |
| §2.4；E02-19；空间研究 §2.4 | 已知激励的逐频正则反卷积 | 本仓库可运行基线 | `codes/chapters/ch02/core/deconvolution.py::regularized_inverse`；外部对照`pyfar`：`pyfar/dsp/dsp.py::deconvolve`、`regularized_spectrum_inversion`；[当前原方法合同](../ch02/reports/upstream_model_contracts.json) | 教学核使用全频常数epsilon及循环参数空间，无短支持约束；固定pyfar原体仅人工Signal/FFT适配调用，默认仍正则。浮点/实际PCM、完整IR/真值支持分别评分，不含设备测量或原整包 |
| §2.2；空间研究 §1.3 | 通道标量幅度校准 | 外部参考实现 | `acoular`：`acoular/calib.py::Calib` | 使用给定校准因子；不是自动估计或频率相关复相位校准 |
| §2.1；空间研究 §1.2 | 湿空气声速与环境参数不确定度模型 | 外部参考实现 | `speed-of-sound-in-air`：`source labview files/SoS_AIR_2025.vi`；[原作者正式论文](https://doi.org/10.1063/5.0294663) | 固定43VI、GPLv3；需LabVIEW，未读框图/未运行，不替换正文343m/s约值 |
| §2.5 | STFT 分析 | 本仓库可运行基线 | `codes/chapters/ch02/core/spectral.py::stft` | 窗、帧移、补零与轴序 |
| §2.5；E02-20 | 加权重叠相加 iSTFT | 本仓库可运行基线 | `codes/chapters/ch02/core/spectral.py::istft`；`codes/chapters/ch02/core/stft_consistency.py::consistency_example` | 窗乘积包络、输出长度；实谱端点与跨帧一致性分开，完整DFT距离保留配对权重；幂等不自动等于任意度量的正交投影 |
| 附录 A §12.3/E12-08 | FFT 分块线性卷积及故意错误的逐块循环对照 | 本仓库可运行基线 | `codes/chapters/appendix_a/core/math_foundations.py::fft_overlap_add`、`blockwise_circular_convolution` | 块长512、FIR长121、FFT至少632；错误对照丢失跨块尾部，不能作为正确实现 |
| §2.5 | 批处理空间协方差 | 本仓库可运行基线 | `codes/chapters/ch03/core/covariance.py::spatial_covariance` | 共轭、快拍与功率归一 |
| §2.5、§5.4 | 递推空间协方差 | 本仓库可运行基线 | `codes/chapters/ch03/core/covariance.py::recursive_covariance` | 遗忘因子、启动与非平稳性 |
| §3.3 | 差分协同阵增广协方差 | 外部参考实现 | `doatools`：`doatools/estimation/coarray.py` | 固定原方法已执行DA/SS与缩放对照，见研究§39；默认ss为滞后段外积平均，da为直接增广；单位分别是功率平方与功率，有限样本da不保证半正定；未运行整包/DOA |
| §3.3 | 稀疏阵几何优化 | 原理索引 | 正文差分集合与孔径分析 | 协方差重建不是几何优化器 |
| §3.4.1 | 有外部相位锚的 DOA—复增益联合拟合与无锚歧义 | 本仓库可运行基线 | `codes/chapters/ch03/core/calibration.py::fit_anchored_ula`、`codes/chapters/ch03/examples/self_calibration_demo.py` | 单频 ULA、逐场景单源，须已独立测得相邻通道相位；无锚共同相位斜率不唯一；非全盲或原作者系统复现 |
| §3.4.2 | 互耦补偿 | 原理索引 | 正文互耦矩阵模型及E03-11 | 区分混合前/后噪声；仅小矩阵正则化演示，不是未知耦合估计器 |
| §3.4.3 | 麦位置自标定 | 外部参考实现 | `structure-from-sound`：`matlab/tdoa/calcresandjac.m`、`bundletdoa.m` | 同步麦、已知声速、足够不同源位；固定版依赖与路径需处理，未运行 |
| §3.4.3；E03-18 | 已知方向下的三维基线与固定相对时延最小二乘 | 本仓库可运行基线 | `codes/chapters/ch03/core/baseline_calibration.py::solve_baseline`；六份独立PCM的训练/留出实验 | 已知方向/声速/远场；检查增广[-U,1]秩，拒绝同俯仰退化；单音时差需主值界，不是盲校准 |
| 空间研究§39.10 | 混合时间/空间TDOA、DOA与里程计的异步多阵列标定 | 原理索引 | 2025预印本v1；`hybrid-tdoa-multi-calib`固定作者链接：`GN_Solver.m`、`compute_J.m`、`SRP-PHAT-DOA.py` | 已知阵内几何、阵内同步/事件间隔/里程计；NOASSERTION仅索引，未运行MATLAB/完整前端/实录，不录入性能基准 |
| §3.2.2 | 指定方向域相位歧义格点判别 | 原理索引 | Tucker等2022与`alias-free-arrays`固定索引 | 作者代码未建立许可，不自动获取；本书独立验证六边形方向对 |
| §3.4 | 已知位置/增益/相位/互耦扰动前向模型 | 外部参考实现 | `doatools`：`doatools/model/perturbations.py` | 施加给定误差，不估计误差；固定版列表/非方输入故障已实际触发，见原方法审计报告与研究§39 |

## 定位与搜索

| 正文或研究范围 | 算法/机制 | 覆盖状态 | 教学入口或主清单 ID：官方源码入口 | 关键边界 |
|---|---|---|---|---|
| §4.1.1、§4.9、E04-20 | AIC 源数估计 | 本仓库可运行基线 | `codes/chapters/ch04/core/doa.py::aic_source_count`；外部对照 `doatools`：`doatools/estimation/source_number.py::aic` | 复高斯独立快拍、空间白噪声、严格正特征值；固定谱手算不是抽样性能，AIC和MDL绝对分数不可互比 |
| §4.1.1、§4.9 | MDL 源数估计 | 本仓库可运行基线 | `codes/chapters/ch04/core/doa.py::mdl_source_count`；外部对照 `doatools`：`doatools/estimation/source_number.py::mdl` | 复高斯独立快拍、白噪声、正特征值；无自动加载 |
| §4.1.1 | 特征值间隙源数启发式 | 原理索引 | 正文相邻特征值差/比 | 阈值依赖噪声模型与尺度，不是自动保证正确的源数接口 |
| §4.2 | 普通互相关 CC | 原理索引 | 正文 GCC 权重 Ψ=1 与 Knapp–Carter 1976 | 带宽、能量与极性影响峰；教学PHAT接口不冒充所有加权法 |
| §4.2 | Roth 加权互相关 | 原理索引 | 正文互谱除参考通道自谱 | 不对称参考、低能量频点保护需明确 |
| §4.2 | SCOT 加权互相关 | 原理索引 | 正文互谱除两自谱几何均值 | 应先明确谱估计与平均；单帧外积的权重可能退化为PHAT |
| §4.2 | GCC-PHAT 与物理 lag 裁剪 | 本仓库可运行基线 | `codes/chapters/ch04/core/doa.py::gcc_phat` | 静音、麦序、带宽与多峰；互谱阈值相对本麦对峰值 |
| §4.2 | GCC 峰三点亚采样插值 | 本仓库可运行基线 | `codes/chapters/ch04/core/doa.py::gcc_phat` 插值选项 | 不能创造窄带缺少的信息 |
| §4.3；附录 B 第 16 题 | 远场 SRP-PHAT | 本仓库可运行基线 | `codes/chapters/ch04/core/doa.py::srp_phat`；`codes/chapters/appendix_b/examples/room_srp_exercise.py` 在六个固定房间输入上执行 | 逐帧PHAT后平均、等权正频点直接相位求和，非完整irfft查表；全零或仅直流输入拒绝给出方向；六点结果不代表跨房间失效率 |
| §4.7.1；研究扩展：空间 §8 | 近场三维 SRP | 原理索引 | 正文球面传播与角度—距离网格；空间研究 §8 的版本边界 | 锁定 pyroomacoustics 0.10.0 未将 `mode/r` 接入有效导向和距离网格，不能作为该变体实现 |
| §4.3；研究扩展：空间 §9 | 分层 SRP 搜索 | 外部参考实现 | `odas`：`src/module/mod_ssl.c`、`src/signal/scan.c` | 粗层丢峰不能由细层恢复 |
| 研究扩展：空间 §9 | 方向性麦对筛选 | 外部参考实现 | `odas`：`src/signal/spatialindex.c` 与配置 | 设备指向性及有效麦对 |
| 研究扩展：空间 §10 | SVD-PHAT | 原理索引 | 原论文与低秩实验设计 | 未确认唯一且许可明确的作者实现 |
| 研究扩展：空间 §10 | 多源 SVD-PHAT | 原理索引 | 多源原论文 | 普通 SVD 库不实现逐次投影规则 |
| §4.4 | Chan 双曲定位 | 原理索引 | Chan–Ho 1994；正文距离差变量与两阶段思想 | E04-16只实现GN一步，不是完整Chan加权与约束处理 |
| §4.4 | TDOA 加权非线性最小二乘 | 原理索引 | 正文高斯—牛顿推导 | 共享参考误差相关、多解、雅可比 |
| §4.5 | Bartlett 空间谱 | 本仓库可运行基线 | `codes/chapters/ch04/core/doa.py::bartlett_spectrum` | 导向与输出功率归一 |
| §4.5 | Capon 空间谱 | 本仓库可运行基线 | `codes/chapters/ch04/core/doa.py::capon_spectrum`；共用加载核 `codes/chapters/ch04/core/covariance.py::_load_covariance` | 加载、秩亏、失配 |
| §4.6 | MUSIC | 本仓库可运行基线 | `codes/chapters/ch04/core/doa.py::music_spectrum` | 源数、噪声子空间、选峰；非半正定及零协方差拒绝 |
| §4.6；空间研究§58 | 平滑后奇异值比直接路径优势检验 | 原理索引 | Nadiri/Rafaely 2014及作者2018说明；E04-24的相干反射反例 | 球谐/径向处理、局部频率时间统计及噪声条件；单频高相干/秩一不构成直达证明，未执行完整DPD |
| §4.6；空间研究§58 | 声场指向性直接路径优势检验 | 原理索引 | Rafaely/Alhaiany 2018正式方法及作者稿 | 对应球谐场/阶数/扫描度量，替代方案避免频率平均与特征分解；未取得完整唯一实现或设备性能 |
| §4.6；E04-25；空间研究§58 | CTF交叉关系与DP-RTF特征估计 | 原理索引 | `dprtf-ssl`：`DP_RTF.m`、`MCMT.m`、`stft.m`，固定作者四文件只索引 | NOASSERTION；E25仅精确人工帧代数，不包含真实窗内直达近似、PSD噪声差分、HRIR匹配或作者MATLAB执行 |
| §4.5；空间研究§59 | ESP-SR Capon DOA状态与VAD接口 | 原理索引 | `esp-sr-doa`：`include/esp32p4/esp_doa_capon_embedded.h`及原测试 | 根产品限制/公开头许可分开；核心为链接二进制；planar通道、坐标/VAD返回旧角与错误码，未设备运行 |
| §4.6 | NormMUSIC | 外部参考实现 | `pyroomacoustics`：`pyroomacoustics/doa/normmusic.py` | 低可靠频点可能被过度加权 |
| §4.6 | 前向空间平滑 | 外部参考实现 | `doatools`：`doatools/estimation/preprocessing.py::spatial_smooth` | 参数 `l` 是子阵数 |
| §4.6 | 前后向空间平滑 | 外部参考实现 | `doatools`：同函数 `fb` 选项 | 对称与平移模型、孔径损失 |
| §4.6 | root-MUSIC | 外部参考实现 | `doatools`：`doatools/estimation/music.py::RootMUSIC1D` | ULA、根选择、半波距与 NumPy 相容性 |
| §4.6 | LS-ESPRIT | 本仓库可运行基线 | `codes/chapters/ch04/core/doa.py::esprit_ula` | 共同全阵基截取、非零PSD与子阵秩；主值角不证明无空间混叠 |
| §4.6；研究扩展：空间 §14 | TLS-ESPRIT | 外部参考实现 | `doatools`：`doatools/estimation/esprit.py` | 双侧误差模型不同于LS；固定版默认行加权共享view错误已实际复现，none/独立复制对照见报告 |
| [§4.6“宽带MUSIC”](../../../chapters/04_doa-estimation.md#sec-u-08146bdaaf) | CSSM | 外部参考实现 | `pyroomacoustics`：`pyroomacoustics/doa/cssm.py` | 固定原辅助函数已复算剔除首频后的错位；见ch04/reports/upstream_doa.json，不是完整整链结果 |
| [§4.6“宽带MUSIC”](../../../chapters/04_doa-estimation.md#sec-u-08146bdaaf) | WAVES | 外部参考实现 | `pyroomacoustics`：`pyroomacoustics/doa/waves.py` | 固定原辅助函数已复算剔除首频后的错位；见ch04/reports/upstream_doa.json，不是完整整链结果 |
| [§4.6“宽带MUSIC”](../../../chapters/04_doa-estimation.md#sec-u-08146bdaaf) | TOPS | 外部参考实现 | `pyroomacoustics`：`pyroomacoustics/doa/tops.py` | 固定原类已在指定三麦/三频合成模型执行；真值30°却输出49°，与独立投影对照分开记录，不外推所有数据 |
| §4.7 | FRIDA | 外部参考实现 | `frida-original`：`doa/fri.py`；`pyroomacoustics`：`pyroomacoustics/doa/frida.py` | 二维远场、不相关源、平面阵列方位；visibility与raw模型分别说明，固定接口不支持三维；原实验旧环境 |
| §4.7 | L1-SVD 定位 | 外部参考实现 | `doatools`：`estimation/preprocessing.py::l1_svd`与`estimation/sparse.py::GroupSparseEstimator`显式组合 | 压缩观测M×K，求解器快拍数设K；压缩本身不是定位，未运行完整链 |
| §4.7 | 组稀疏多快拍定位 | 外部参考实现 | `doatools`：`doatools/estimation/sparse.py::GroupSparseEstimator` | 离格误差、字典尺度、求解器 |
| §4.7 | 稀疏协方差匹配 | 外部参考实现 | `doatools`：同文件 `SparseCovarianceMatching` | 不相关源功率模型 |
| §4.7；空间 §19 | 稀疏贝叶斯定位 SBL | 外部参考实现 | `sbl`：`SBL_MF_Python/sbl.py::SBL` | 多频/多快拍，已知源数；网格、剪枝与噪声模型 |
| 研究扩展：空间 §19 | RobustSBL | 外部参考实现 | `robustsbl`：`_common/SBL_v5p12.m` | Gauss/t/Huber/Tyler；MATLAB 统计函数、异常快拍模型 |
| 研究扩展：空间 §10 | SMP-PHAT | 外部参考实现 | `smpphat`：`src/system.c::smp_call`；本书隔离两行适配实验 | 等长平行基线与远场；非 SVD-PHAT；固定原版在本机 clang/arm64 数值失败，临时适配版仅通过两组合成四麦对照 |
| §4.7 | 原子范数定位 | 原理索引 | 正文连续参数模型 | 通用 SDP 求解器不是完整实现 |
| §4.7 | GCC 向量特征 MLP 定位（Xiao 等，2015） | 原理索引 | [原始论文 DOI](https://doi.org/10.1109/ICASSP.2015.7178484) | 机构论文页可核 GCC 向量输入；未核到与论文逐项对应且许可明确的作者实现，不外推为 PHAT 特征 |
| §4.7 | 共素阵 GCC-PHAT 图 CNN 定位（Zhao–Ritz，2021） | 原理索引 | [原始论文 §III、图 5](https://www.apsipa.org/proceedings/2021/pdfs/0000974.pdf) | 已核特征和分类路线；未核到许可明确的作者实现或本书训练结果 |
| §4.7 | 多通道 STFT 相位图 CNN 多源 DOA（Chakrabarty–Habets，2019） | 原理索引 | [作者版原论文](https://www.audiolabs-erlangen.de/resources/aps-w23/papers/sap_Chakrabarty2019.pdf) | 作者页所链代码 README 指向不同的 2017 单源模型，不能当作 2019 多源整链已复现 |
| §4.7；[空间研究§45](research/01_spatial_and_tracking.md#said-semantic-imaging) | SAID球面方向编码与逐源语义能量成像 | 外部参考实现 | `said-spatial-imaging`：`said/models/audio2sph.py`、`sph2imaging.py`；`said/utils/compression.py` | 2026-09预印本；固定源码和混合组件许可已取得，检查点/数据未取；仅单文件压缩与提交大小诊断，不称网络、训练或榜单复现 |
| §4.7、§9.3 | icoDOA | 外部参考实现 | `icodoa`：`1sourceTracking_icoCNN.py` | AGPL；gpuRIR、icoCNN、数据另核 |
| 研究扩展：空间 §41 | Cross3D | 外部参考实现 | `icodoa`：`acousticTrackingModels.py::Cross3D` | 类存在不等于训练已经复现 |
| §4.8；研究扩展：空间 §38 | 随机源 CRB | 外部参考实现 | `doatools`：`doatools/performance/crb.py::crb_sto_farfield_1d` | 模型下界，不是定位器 |
| 研究扩展：空间 §38 | 确定源 CRB | 外部参考实现 | `doatools`：同文件 `crb_det_farfield_1d` | 不能混用随机源模型 |
| 研究扩展：空间 §38 | 不相关随机源 CRB | 外部参考实现 | `doatools`：同文件 `crb_stouc_farfield_1d` | 独立性与方差单位 |
| 研究扩展：空间神经定位 | IPDnet 固定阵列 | 原理索引 | `fn-ssl-ipdnet`：`IPDnet/FixedAarryIPDnet.py` 来源索引 | 原文件如此拼写；未建立明确许可，不下载 |
| 研究扩展：空间神经定位 | IPDnet 可变阵列 | 原理索引 | `fn-ssl-ipdnet`：`IPDnet/VariableArrayIPDnet.py` 来源索引 | 许可未建立；几何、特征和在线状态另核 |
| §4.7；研究扩展：空间 §43 | 单轨 ACCDOA 表示 | 原理索引 | `dcase2022-seld`：`seldnet_model.py` 后续届次实现索引 | 许可未建立；向量模不等于概率或距离 |
| §4.7；研究扩展：空间 §44 | Multi-ACCDOA 多轨表示 | 原理索引 | `dcase2022-seld`：`seldnet_model.py` 来源索引 | 许可未建立；同类重叠源与轨道容量 |
| §4.7；研究扩展：空间 §44 | ADPIT 辅助重复排列训练 | 原理索引 | `dcase2022-seld`：`seldnet_model.py` 来源索引 | 许可未建立；训练置换不提供持久身份 |
| 研究扩展：空间 SELD | DCASE 2025 双声道 SELD 基线 | 原理索引 | `dcase2025-stereo-seld`：`model.py`、`loss.py` 来源索引 | 许可未建立；立体声任务不等于耳廓/HRTF 双耳录音或任意阵列 |
| 研究扩展：空间 §10 | X-SRP 可组合体积搜索 | 原理索引 | `xsrp`固定源索引：`xsrp/` | 元数据写MIT但完整许可未建立；未获取/运行，不修饰成已验证三维实现 |
| §4.7；研究扩展：空间 §45 | DCASE2026 SAISELD 声学成像链 | 原理索引 | `dcase2026-saiseld`：`acoustic_features.py`、`model.py`、`run_inference.py` | 源码许可未建立，权重另核；成像/分割/追踪不是单帧DOA接口 |

## 波束、后滤与球阵

| 正文或研究范围 | 算法/机制 | 覆盖状态 | 教学入口或主清单 ID：官方源码入口 | 关键边界 |
|---|---|---|---|---|
| §5.2 | DSB 权重 | 本仓库可运行基线 | `codes/chapters/ch05/core/beamforming.py::dsb_weights`；`codes/chapters/ch05/beamformer_common_input_demo.py` | 不含设备分数延时 FIR；同输入对照只在单频解析模型 |
| §5.2 | 等间距线阵 Dolph–Chebyshev 加权 | 原理索引 | 原论文与 SciPy `chebwin` 官方定义；SciPy源码未单独锁定 | 对称正横阵列与首零点宽度条件，不泛称任意阵列最窄HPBW；窗口峰归一另转为目标响应归一 |
| §5.3 | 弥散场相干模型 | 本仓库可运行基线 | `codes/chapters/ch05/core/beamforming.py::diffuse_coherence` | 各向同性假设 |
| §5.3 | 超指向波束 | 本仓库可运行基线 | `codes/chapters/ch05/core/beamforming.py::superdirective_weights` | 低频 WNG 与麦误差 |
| §5.3 专栏 | 差分麦克风阵 DMA | 原理索引 | 正文一阶算例与高阶模型 | 超指向接口不覆盖全部 DMA |
| §5.4 | MVDR 与相对对角加载 | 本仓库可运行基线 | `codes/chapters/ch05/core/beamforming.py::mvdr_weights`；共用加载核 `codes/chapters/ch04/core/covariance.py::_load_covariance`；`codes/chapters/ch05/beamformer_common_input_demo.py` | 加载按平均特征值缩放 |
| §5.4.1 | 最坏情形稳健波束 | 原理索引 | 正文误差集模型 | 经验加载不等于误差集优化；E05-19仅双麦白噪闭式，未实现通用SOC求解器 |
| §5.4.1 | 显式 WNG 约束设计 | 原理索引 | 正文约束与选型 | WNG 计算不等于约束优化器 |
| §5.4.1 | 特征空间波束 | 原理索引 | Chang–Yeh 1992；空间研究 §22 | 子空间维数/高SNR/导向投影；未取得作者完整实现 |
| §5.4.1 | 干扰加噪声协方差重构 | 原理索引 | Gu–Leshem 2012；空间研究 §22 | 排除目标角域与流形假设；Capon函数不是完整重构算法 |
| §5.5 | LCMV 闭式权重 | 本仓库可运行基线 | `codes/chapters/ch05/core/beamforming.py::lcmv_weights`；`codes/chapters/ch05/beamformer_common_input_demo.py` | 约束独立性、残差 |
| §5.5 后的 Frost 专题 | Frost 投影自适应 | 原理索引 | 正文时域投影更新；E05-10实际计算一步 | 由 LCMV 约束出发；一步约束核对不是完整在线Frost实现 |
| §5.6 | GSC 阻塞矩阵 | 本仓库可运行基线 | `codes/chapters/ch05/core/beamforming.py::blocking_matrix` | 不含持续自适应抵消支路 |
| §5.6；E05-16 | 单参考复数 GSC NLMS 状态更新 | 本仓库可运行基线 | `codes/chapters/ch05/core/gsc.py::ScalarGSCNLMS`；`codes/chapters/ch05/chapter05_experiments.py` | 给定固定/阻塞输出，跨块、冻结和复位；不含方向估计、自动SPP或多抽头滤波器组 |
| §5.6；空间 §21 | 完整在线自适应 GSC | 外部参考实现 | `btk20`：`btk20_src/lib/pybeamformer.py` | 子带 LMS/RLS；毫米坐标、旧依赖、泄漏与冻结；非时域 Frost |
| §5.7.2 | 标量 Wiener 增益 | 本仓库可运行基线 | `codes/chapters/ch05/core/beamforming.py::wiener_gain` | 不是完整噪声估计器 |
| §5.7；空间 §21 | Zelinski 互谱后滤 | 外部参考实现 | `btk20`：`btk20_src/postfilter/postfilter.cc` | REAL/ABS分支、递归平滑与增益下限分开；同谱非负性不等于无偏；未构建/运行 |
| §5.7；空间 §21 | McCowan 弥散相干后滤 | 外部参考实现 | `btk20`：同文件 McCowan 入口 | 固定复相干实现不等于原文实部式；Γ近1病态；未构建/运行 |
| §5.7.1 | MCRA | 原理索引 | 正文与空间研究的作者软件入口 | 未获得可核版本/许可包 |
| §5.7.1 | IMCRA | 原理索引 | 正文与空间研究的作者软件入口 | 不把普通最小值跟踪称 IMCRA |
| §5.7.2 | OM-LSA | 原理索引 | 正文及作者方法说明 | 不以 Wiener 或其他 MMSE 增益代替 |
| §5.5 后的 MWF 专题；§5.7.2 | 全秩 SDW-MWF | 外部参考实现 | `espnet`：`espnet2/enh/layers/beamformer.py::get_sdw_mwf_vector` | 独立均方误差问题；失真权重与参考通道，非 LCMV 子类 |
| 研究扩展：空间后滤 | 秩一迹化简 WMWF | 外部参考实现 | `pb_bss`：`pb_bss/extraction/beamformer.py::get_wmwf_vector` | 秩一迹化简；12原函数审计已实调，全秩反例不等于一般MWF |
| §5.9 | GEV 波束权重 | 外部参考实现 | `pb_bss`：同文件 `get_gev_vector` | 广义特征向量尺度和相位不确定；E05-23三PCM仅已知分量控制，不另计算法或称运行GEV整链 |
| §5.9；E05-23；空间研究§60 | 相邻频率特征向量相位连续化 | 外部参考实现 | `pb_bss`：`pb_bss/extraction/beamformer.py::phase_correction`；[当前原函数合同](../ch05/reports/upstream_beamformers_current.json) | 2D F×C原调用符合相邻频累乘；3D B×F×C原axis0沿batch造成符号控制差异；平滑不自动恢复目标绝对相位，无完整整链 |
| §5.9 | BAN 缩放 | 外部参考实现 | `pb_bss`：同文件 `blind_analytic_normalization` | pb_bss返回权重、ESPnet返回增益且含C²差异；不保证无失真 |
| 空间研究§60 | 相位感知归一PAN | 原理索引 | [Pfeifenberger等2017原文](https://www.isca-archive.org/interspeech_2017/pfeifenberger17_interspeech.pdf) §3.1式12/13 | 单源秩一/精确GEV方向/单位范数目标仍需参考复相位；未核到许可明确作者网络源码或运行完整方法，手算控制不升级为实现 |
| §5.9；空间 §48 | 秩一目标 SCM 的 PCA-RTF 方向 | 外部参考实现 | `pb_bss`：`pb_bss/extraction/beamformer.py::get_pca_vector` | 特征向量还需选参考/尺度；全秩目标不能直接认作唯一RTF |
| §5.9；空间 §48 | Souden 参考通道 MVDR | 外部参考实现 | `pb_bss`：同文件 `get_mvdr_vector_souden` | 秩一复参考因子/全秩边界见E05-20；[原方法审计](research/01_spatial_and_tracking.md#beamformer-upstream-audit)已实调；MERL失败另记 |
| §5.9 | RTF 幂迭代估计 | 外部参考实现 | `espnet`：`espnet2/enh/layers/beamformer.py::get_rtf` | 函数本身不完成参考通道归一 |
| §5.9、§8.4 | 掩码空间协方差 | 本仓库可运行基线 | `codes/chapters/ch08/core/separation.py::masked_spatial_covariance` | 轴序、空掩码、保留原始功率 |
| §5.9；E05-24 | 已知逐通道预滤波与有限样本外积均值 | 本仓库可运行基线 | `codes/chapters/ch05/chapter05_experiments.py::channel_mask_prefilter_model` | 原目标与处理后导向、均帧与掩码和分母分别核；两帧秩2不是单目标随机独立性或神经网络结果 |
| 空间研究§60；§5.9 | 分布式阵列的在线逐通道掩码波束 | 原理索引 | [Middelberg等2026原文v1](https://arxiv.org/html/2607.26623v1) §3–5 | 平方根掩码预滤后滑窗外积/Souden；未核到许可明确的作者实现，未运行网络、完整滑窗增强或真实网络硬件 |
| §5.9、§8.4 | 两通道掩码 MVDR | 本仓库可运行基线 | `codes/chapters/ch08/core/separation.py::mask_mvdr_2x2` | 限定 2×2；非零掩码公共缩放不应改变结果；不是完整神经系统 |
| §5.8 | 球面采样到球谐系数 | 外部参考实现 | `sound-field-analysis`：`sound_field_analysis/process.py::spatFT` | 实/复、余纬角、排列和归一 |
| §5.8 | 理论径向补偿 | 外部参考实现 | `sound-field-analysis`：`sound_field_analysis/gen.py::radial_filter` | 开放/刚性球、低频噪声；旧API与当前NumPy/SciPy不兼容 |
| §5.8；研究扩展：空间 §25 | 软限制径向滤波 | 外部参考实现 | `spherical-array-processing`：`arraySHTfiltersTheory_softLim.m` | 限幅与模态误差权衡 |
| §5.8；研究扩展：空间 §25 | 理论正则球阵编码 | 外部参考实现 | `spherical-array-processing`：`arraySHTfiltersTheory_regLS.m` | 正则量、噪声模型 |
| §3.4、§5.8 | 实测响应正则编码 | 外部参考实现 | `spherical-array-processing`：`arraySHTfiltersMeas_regLS.m` | 设计集与留出方向分开 |
| 研究扩展：空间 §26 | 球谐 Dolph–Chebyshev 波束 | 外部参考实现 | `spherical-array-processing`：`beamWeightsDolphChebyshev2Spherical.m` | 有效阶数与旁瓣 |
| §5.8 | 球谐 MVDR | 外部参考实现 | `spherical-array-processing`：`sphMVDR.m` | 径向滤波后噪声统计更新；getSH直接依赖已锁定 |
| §5.8 | 球谐 LCMV | 外部参考实现 | `spherical-array-processing`：`sphLCMV.m` | 约束和编码误差 |
| 研究扩展：空间 §26 | 球谐 MUSIC | 外部参考实现 | `spherical-array-processing`：`sphMUSIC.m` | 源数与有效阶数 |
| 研究扩展：空间 §26 | 球谐 ESPRIT | 外部参考实现 | `spherical-array-processing`：`sphESPRIT.m` | 与阵元 ULA 结构不同 |
| 研究扩展：空间 §27 | C/C++ 球阵编码块处理 | 外部参考实现 | `spatial-audio-framework`：`examples/src/array2sh/array2sh.c` | 核心 ISC、可选 GPL 模块与后端另核 |
| 研究扩展：空间 §23 | Acoustic Rake | 外部参考实现 | `pyroomacoustics`：`pyroomacoustics/beamforming.py::rake_mvdr_filters` | 需要反射模型，不是通用去混响器 |
| 专题Ⅰ §14.5、E14-03/17；空间 §28 | DAMAS 双向与前向教学控制 | 本仓库可运行基线 | `codes/chapters/ch14/core/imaging.py::damas_gauss_seidel`；外部 `acoular`：`BeamformerDamas`、`damasSolverGaussSeidel`；`damas-author`：完整`damas.py` | 完整CSM/匹配模板下Gram逐行归一化与GS相连，扫描残差NNLS是不同目标；作者九限定控制8吻合/1原NameError；GPL获取不等于MATLAB/MEX大实验，不保证全局唯一 |
| 专题Ⅰ §14.6、E14-11；空间 §29 | 完整CSM CLEAN-SC | 本仓库可运行基线 | `codes/chapters/ch14/core/imaging.py::clean_sc_full_csm`；外部 `acoular`：`BeamformerCleansc` | 教学采用作者full-CSM式；原full支路单源20轮过量分配保留在合同；DR隐式式不混用 |
| 专题Ⅰ §14.7、E14-12；空间 §30 | 原CMF与可选目标 | 外部参考实现 | `acoular`：`acoular/fbeamform.py::BeamformerCMF`；[限定合同](../ch14/reports/upstream_imaging_contracts_current.json) | 原字典已调用；半三角无√2与默认截距改变目标；sklearn估计器未执行，非原稀疏约束论文整链 |
| 专题Ⅰ §14.5、E14-08 | 扫描域小系统NNLS | 本仓库可运行基线 | `codes/chapters/ch14/core/imaging.py::finite_nnls` | 至多8列活动集枚举和KKT；不反演441未知格；与DAMAS的目标和固定点不同 |
| 专题Ⅰ §14.7、E14-12、E14-16 | 简化CSM拟合及已知白噪声辨识 | 本仓库可运行基线 | `codes/chapters/ch14/core/imaging.py::hermitian_real_vector`、`single_source_csm_fit`、`finite_nnls` | 完整Frobenius和有限非负小系统；单源/白噪声LS无约束、另报告非负性；不含原CMF全部稀疏约束；一麦源噪声拆分不可辨 |
| 研究扩展：空间 §30 | SODIX | 外部参考实现 | `acoular`：`acoular/fbeamform.py::BeamformerSODIX` | 源强和指向性可辨识性 |
| 研究扩展：空间 §30 | 移动源时域声学成像 | 外部参考实现 | `acoular`：`acoular/tbeamform.py` | 轨迹/传播真值，不直接输出增强语音 |
| 研究扩展：空间近年候选 | ASA 注意力空间协方差聚合（2024） | 原理索引 | Tammen 等 Interspeech 2024 原论文 | 注意力帧权重与通道不变性；未核实作者代码，不冒充已复现 |
| 研究扩展：空间近年候选 | iDeepPE 参数估计与后滤融合（2025） | 原理索引 | `ideeppe`固定索引：`evaluate.py` | 未建立代码许可，不获取；默认非因果噪声估计，oracle训练准备不等于推理 |
| 研究扩展：空间近年候选 | 可学习 WNG 阈值的稳健波束（2026） | 原理索引 | 正式 INTERSPEECH 2026，6996～7001页，[DOI 10.21437/Interspeech.2026-2212](https://www.isca-archive.org/interspeech_2026/deng26d_interspeech.html)；所链 [arXiv:2606.24137v1](https://arxiv.org/abs/2606.24137v1) 为预印本 | 已知远场方向、双头mask/阈值；未取得作者代码和完整训练复现；未执行作者网络与QEP |

## 回声消除与自适应控制

| 正文或研究范围 | 算法/机制 | 覆盖状态 | 教学入口或主清单 ID：官方源码入口 | 关键边界 |
|---|---|---|---|---|
| §6.1.2 | LMS | 原理索引 | 正文梯度更新 | 教学 NLMS 不是固定步长 LMS |
| §6.1.2 | NLMS | 本仓库可运行基线 | `codes/chapters/ch06/core/aec.py::nlms`；`codes/chapters/ch06/examples/aec_same_input_truth.py` 真值冻结对照 | 外部冻结掩码、零参考、有限 FIR；合成真值输入不等于设备性能 |
| §6.15 | 流式 NLMS 状态 | 本仓库可运行基线 | `codes/chapters/ch06/core/aec.py::NLMSState`、`codes/chapters/ch06/aec_streaming_demo.py` | 跨块同时续接抽头与 $L-1$ 个参考历史；不含 DTD、延迟搜索或实时接口 |
| 研究扩展：增强 A01 | 泄漏 NLMS | 原理索引 | 泄漏更新说明 | 教学函数没有泄漏参数 |
| §6.2 | 单块 FDAF | 原理索引 | 正文频域卷积与约束 | 分区 MDF 不覆盖所有单块变体 |
| §6.2 | MDF/PBFDAF 分区结构（同配置频域多抽头 NLMS） | 本仓库可运行基线 | `codes/chapters/ch06/core/aec_partitioned.py::PartitionedFDAFState`、`codes/chapters/ch06/aec_partitioned_demo.py`（手算、频点反例与冻结宽带留出实验）、`codes/chapters/ch06/examples/aec_pbfdaf_real_pair_compare.py`（两对固定真实配对录音与三种参考控制）；外部 Speex 对照另见 AUMDF 行 | 瞬时功率式(6-5)、有效区、可选梯度约束与跨块状态；真实录音结果含负值，只是总功率变化；不含自动 DTD、延迟搜索或产品级控制；同一结构不重复算算法 |
| §6.2；研究扩展：增强 A03 | AUMDF 约束调度 | 外部参考实现 | `speexdsp`：同文件；真实配对与已知近端注入实验见 `codes/chapters/ch06/examples/aec_real_pair_experiment.py`、`codes/chapters/ch06/aec_doubletalk_experiment.py`、`codes/chapters/ch06/aec_controlled_doubletalk.py` | 约束调度不同于梯度更新；真实双讲没有分量真值，半合成输出增量也不是近端保留率 |
| §6.2 | PNLMS | 原理索引 | 正文比例更新 | Speex 控制不代表全部变体 |
| §6.2 后续 IPNLMS 专题 | IPNLMS | 本仓库可运行基线 | `codes/chapters/ch06/core/aec_ipnlms.py::IPNLMSState`、`codes/chapters/ch06/aec_ipnlms_subband_demo.py`、`codes/chapters/ch06/aec_advanced_exercises.py`；`codes/chapters/ch06/aec_algorithm_minicases.py` 保留旧一步小例 | 实数单参考、先验残差、跨块状态；强抽头比例份额不保证稀疏/稠密任意条件下更快；外部给冻结掩码 |
| §6.2 后续子带专题 | 两带 Haar 对角子带 NLMS 教学近似 | 本仓库可运行基线 | `codes/chapters/ch06/core/aec_subband.py::HaarDiagonalSubbandNLMSState`、`codes/chapters/ch06/aec_ipnlms_subband_demo.py`、`codes/chapters/ch06/aec_advanced_exercises.py` | 只读取同带历史；不是任意回声路径的精确模型或完整 NSAF；两点 Haar 不代表工业滤波器组 |
| §6.2 后续子带专题 | 已知两抽头路径的精确交叉带映射 | 本仓库可运行基线 | `codes/chapters/ch06/core/aec_subband.py::two_tap_crossband_matrices`、`two_tap_subband_outputs`；时域卷积独立测试 | 固定已知路径的代数对照，不是交叉带自适应器；完美重建不保证对角辨识 |
| §6.2 后续子带专题 | 两带 Haar 全交叉自适应块 NLMS | 本仓库可运行基线 | `codes/chapters/ch06/core/aec_subband.py::HaarCrossbandNLMSState`、`fir_crossband_matrices`；`codes/chapters/ch06/aec_crossband_demo.py`；`tests/test_codes_aec_crossband.py` | 任意有限实数时域 FIR 在该两带块模型内可表示；每块四路 FIR、外部冻结、成块等待；非工业滤波器组、非 Lee–Gan NSAF |
| §6.2 后续子带专题 | 一般子带自适应 AEC/NSAF | 原理索引 | 正文模型与 Lee–Gan 原论文；固定版 `pyroomacoustics` 可查 `adaptive/subband_lms.py`（已在固定0.10.0锁清单登记，未运行此方法） | 原型滤波器、抽取/混叠、跨带耦合和延迟依实现而变；NSAF 的全带抽头更新不同于本书两带逐输出带 FIR |
| §6.2 仿射投影专题；E06-34～39 | 正则实数APA有限窗口 | 本仓库可运行基线 | `codes/chapters/ch06/core/aec_affine_projection.py::APAState`、`codes/chapters/ch06/aec_affine_projection_demo.py`与`codes/chapters/ch06/examples/generate_apa_audio.py`；外部`pyaec`：`time_domain_adaptive_filters/apa.py`，实际原函数报告见研究APA审计 | 同一旧权重重新计算全窗误差；启动只用真实观测，冻结仍保留观测窗，恢复污染须显式清窗；正则可计算不等于可辨识，无DTD/延迟搜索/完整AEC；原函数丢尾和复虚部警告保留 |
| §6.2 后续 RLS 专题 | 常规实数时域 RLS | 本仓库可运行基线 | `codes/chapters/ch06/core/aec_rls.py::RLSState`、`codes/chapters/ch06/aec_rls_demo.py`、`codes/chapters/ch06/aec_rls_kalman_comparison.py`、`tests/test_codes_aec_rls.py`；A04 另列 pyroomacoustics、pyaec、MetaAF 源码 | 含初始约束的指数加权最小二乘；逆相关矩阵 $O(L^2)$，教学代码每样本 Cholesky 正定检查另需 $O(L^3)$；无自动 DTD、延迟搜索或 RES |
| §6.2；研究扩展：增强 A04 | 快/块/广义频域 RLS | 外部参考实现 | `pyroomacoustics`：`adaptive/rls.py` 的 BlockRLS；`metaaf`：`optimizer_rls.py`；GFDAF 论文另见 A04 | BlockRLS 和 GFDAF 不是本书逐样本精确 RLS 的改名；MetaAF 核心与 AEC zoo 许可不同 |
| §6.2 | 短实数 FIR 矩阵 Kalman | 本仓库可运行基线 | `codes/chapters/ch06/core/aec_kalman_matrix.py::KalmanAECState`、`codes/chapters/ch06/aec_kalman_matrix_demo.py`、`tests/test_codes_aec_kalman_matrix.py` | 已知 $Q,\Psi$ 的 Joseph 协方差更新，未估计噪声或实现频域分区 |
| 研究扩展：增强A19；E06-38诊断 | 立体声路径先验暖启与能量差过程噪声 | 原理索引 | Wang等Interspeech2025，DOI 10.21437/Interspeech.2025-1572，原§2～3；固定位置预校准路径 | 未确认作者源码/许可，未运行完整方法；已知先验可改变初值与过程模型，却不恢复观测矩阵秩；E38仅先验选解/错误先验的独立留出诊断，非论文性能复现 |
| §6.2 | FDKF | 外部参考实现 | 增强 A04；`pyaec`：`frequency_domain_adaptive_filters/fdkf.py::fdkf`；`codes/chapters/ch06/examples/audit_aec_upstream_interfaces.py` | 固定原函数在 NumPy 2.5.3 因 `np.complex` 失败，未输出音频；标量教学递推式(6-21)不是完整 FDKF；AEC3/Speex 不自动归为卡尔曼 |
| §6.2 | 分区 FDKF 与 VD/SD-PBFDKF | 外部参考实现 | 增强 A04；`pyaec`、`echocatzh-pfdkf`、`subband-kalman-aec` 固定源码；外部接口报告 | pyaec 在 `np.complex` 失败；echocatzh 原接口已实调默认后滤与关闭后滤的不同取点；非同条件整链性能对照，后者仅 `.m`、README、许可 |
| §6.3 | Volterra 非线性 AEC | 原理索引 | 正文路径模型；`pyaec/nonlinear_adaptive_filters/volterra.py::svf` 可查受限二因子二阶级联 | 未执行该函数；受限级联不等于完整 Volterra 核；阶数、过拟合及未见削波另验 |
| §6.3 | Hammerstein 非线性路径 | 原理索引 | 正文级联模型 | 非线性位于线性动态系统之前 |
| §6.3 | Wiener 非线性路径 | 原理索引 | 正文级联模型 | 非线性位于线性动态系统之后 |
| §6.4 | Geigel DTD | 原理索引 | 正文判决模型；`codes/chapters/ch06/aec_algorithm_minicases.py` 反例 | 多径误判、零参考禁判；小例不是产品检测器 |
| §6.4 | 相关/相干性 DTD | 本仓库可运行基线 | `codes/chapters/ch06/core/double_talk.py::ncc_activity_states`、`codes/chapters/ch06/aec_dtd_demo.py` 的四状态计数及路径突变反例；E06-41/42补纯回声尾声 | 固定帧去均值RMS与NCC活动启发式；零方差NCC的0是未定义占位；`near_only`不能证明存在近端语音，真实设备未测 |
| §6.4.1、§6.10；E06-40～42 | 已知播放增益顺序与历史尾声控制 | 本仓库可运行基线 | `codes/chapters/ch06/core/reference_timing.py`、`core/reference_audio.py`与`examples/generate_reference_audio.py`；图74读取独立六WAV实际PCM | 已知稀疏FIR冻结、时变增益给定；卷积与当前增益不交换，播放停止仍保留历史；解析/浮点/PCM分开，不估计路径或增益、不运行生产AEC |
| §6.4 | 连续可变学习率 | 外部参考实现 | `speexdsp`：`libspeexdsp/mdf.c` | 不是独立二值 DTD |
| §6.8 | AEC3 参考延迟控制 | 外部参考实现 | `webrtc`：`modules/audio_processing/aec3/echo_path_delay_estimator.cc`、`render_delay_controller.cc` | 降采样匹配滤波估滞后；与 refined/coarse 线性滤波器分工不同 |
| §6.8 | AEC3 线性抵消 | 外部参考实现 | `webrtc`：`modules/audio_processing/aec3/subtractor.cc`；官方 WAV 入口的已运行适配器 `codes/chapters/ch06/examples/aec3_offline_compare.py`、`codes/chapters/ch06/examples/aec_same_input_truth.py` | 固定提交的 `audioproc_f` 已在两对真实录音和一组已知合成分量上分别导出线性/最终 WAV；同输入脚本默认锁历史二进制，新构建需显式清单并重新测延迟、评分，尚无第二构建实测；比较须区分取点，仍无设备性能排名，见研究手册 A06 |
| §6.8、§6.12 | AEC3 残余回声估计 | 外部参考实现 | `webrtc`：`modules/audio_processing/aec3/residual_echo_estimator.cc` | 与执行抑制增益分开 |
| §6.8、§6.12 | AEC3 抑制增益 | 外部参考实现 | `webrtc`：`modules/audio_processing/aec3/suppression_gain.cc` | 近端损伤与回声泄漏分别计量 |
| §6.8 | AEC3 舒适噪声 | 外部参考实现 | `webrtc`：`modules/audio_processing/aec3/comfort_noise_generator.cc` | 不掩盖滤波器未收敛 |
| §6.9 | 多播放参考 AEC | 外部参考实现 | `speexdsp`：`libspeexdsp/mdf.c::speex_echo_state_init_mc` | 交织顺序、参考相关、路径可辨识性 |
| §6.9 | 播放参考去相关 | 原理索引 | 正文多参考选型 | 不能任意加噪而忽略失真 |
| §6.5 | DTLN-aec | 外部参考实现 | `dtln_aec`：`run_aec.py` | 双输入、两阶段状态；模型另核 |
| §6.5 | NKF-AEC | 原理索引 | `nkf_aec`：`src/nkf.py` 来源索引 | 许可未明确；线性 AEC、需对齐 |
| §6.5 | NeuralKalman | 原理索引 | 增强 A12 原论文 | 与 NKF-AEC 不同，未核完整官方软件 |
| §6.5 | Deep Adaptive AEC | 原理索引 | 增强 A12 原论文 | 学习更新系统不等于 NS 分支 |
| §6.5 | DeepVQE | 原理索引 | 增强 A12 原论文 | 联合任务、参考和训练目标 |
| 增强 A19 近年候选 | LMPAN 三路软对齐与轻量混合AEC | 原理索引 | INTERSPEECH2026，4945～4949页，DOI10.21437/Interspeech.2026-191；作者版arXiv:2607.02062v1 | 作者仓AEC仍Developing，未取得专属实现/权重；126M MACs与对齐预算的原文单位边界分开，未运行网络 |
| 增强 A19 近年候选 | DiffVQE 条件判别与单步扩散 | 原理索引 | INTERSPEECH2026，4971～4976页，DOI10.21437/Interspeech.2026-2337；作者版arXiv:2605.08189v2 | 双向骨干与离线对齐，单步不保证因果；作者demo无已核算法实现/权重/再分发许可，未运行网络 |
| 研究扩展：增强 A13 | Meta-AF 核心更新器与 AEC 配方 | 外部参考实现 | `metaaf`：`metaaf/filter.py`、`metaaf/core.py`、`metaaf/meta.py`、`zoo/aec/` | 核心 NCSA；AEC选集保留Adobe非商用研究教学许可；已取得源码，未取权重或运行网络 |
| §6.7 | ERLE 与有效单讲区间 | 本仓库可运行基线 | `codes/chapters/ch06/core/aec.py::erle_db` | 双讲排除、收敛段、固定延迟；正功率地板使完美抵消读数有限 |

## WPE、盲分离与空间混合

| 正文或研究范围 | 算法/机制 | 覆盖状态 | 教学入口或主清单 ID：官方源码入口 | 关键边界 |
|---|---|---|---|---|
| §7.1；增强研究 W07 | 多通道逆定理（MINT）精确逆滤波 | 原理索引 | Miyoshi/Kaneda 1988 原文 §III–IV；`codes/chapters/ch07/chapter07_experiments.py` E07-14/20/21 为本书受限两路径解析与PCM例 | 已知单源响应、互素等前提；小例不等于一般未知房间的 MINT 求解器 |
| §7.1；E07-20/21 | 已知两稀疏FIR路径的直达约束正则逆 | 本仓库可运行基线 | `codes/chapters/ch07/core/mint_teaching.py`；六PCM与独立评分 | 指定512点反射、等方差独立后路径噪声、固定真值路径；不等于通用MINT辨识器或盲WPE |
| §7.1；增强研究 W07 | 实倒谱域加窗（liftering） | 原理索引 | Oppenheim/Schafer/Stockham 1968；正文对数功率谱约定（实倒谱的两倍） | 需要对数零点处理；实倒谱丢失相位，非无条件可逆去混响器；未锁定通用作者实现 |
| §7.1；增强研究 W07 | 晚期混响谱方差估计与抑制 | 原理索引 | Habets 等 2009 原文模型与边界 | 统计衰减与混合时间假设；不把 SpeexDSP 的配置开关误作已实现的谱方差抑制器 |
| 增强研究两阶段去混响 | 输出端PSD训练WPE与单通道维纳后滤 | 原理索引 | 2sderev作者2023论文及官方补充；研究 `#two-stage-dereverb` | 有限滤长残余；HA前40ms/CI16ms目标不同，训练代码未公开、源码许可未核，不取得代码/模型 |
| §7.10；增强研究 W09 | MetaAF 学习 WPE 滤波器更新 | 外部参考实现 | `metaaf`：`zoo/wpe/wpe.py`、`zoo/wpe/wpe_eval.py`；核心优化器在 `metaaf/` | 固定源码选集已取得，zoo 采用独立非商业许可；未训练、未载入模型、未复现质量指标 |
| §7.10；增强研究 W09 | VACE-WPE 虚拟通道估计与任务特定前端 | 外部参考实现 | `tso_vace_wpe`：`vace_wpe.py`、`torch_custom/wpe_th_utils.py`、`torch_custom/neural_wpe.py` | MIT 固定源码与直接依赖选集；未取模型和语料，未训练/推理或核验 ASV 性能 |
| §7.8；E07-22～24 | 已知到达下的源时刻与单历史回归控制 | 本仓库可运行基线 | `codes/chapters/ch07/core/prediction_delays.py`、`core/delay_audio.py`及六PCM生成/核验；图75 | 已知整数到达、一个实数回归、共同接收时轴；不估计时差，不是完整STFT/跨频WPE；T60题仅自由功率衰减换算 |
| 增强研究W10；`#rir-encoder-candidate` | RIR对比嵌入辅助编码器 | 外部参考实现 | `rir-encoder`：`network.py`、`data_module.py`、`loss_fn.py`、`train.py`、`embedding_extraction.py` | 作者MIT固定八文本已取得，顶层训练有写入；未装依赖/取权重/运行，非论文完整FiLM-SGMSE恢复管线 |
| 增强研究W10；`#usdnet-plus-candidate` | USDnet++混合与SPD重构弱监督 | 原理索引 | Pang等Interspeech2026，DOI10.21437/Interspeech.2026-2044，原§3～4式3～15 | WPE/WPD结果仍非干净真值；FCP可含未来与全句求解，二模型另计；未核可许可作者整链实现/权重/运行 |
| §7.2 | 离线单/多通道 WPE | 本仓库可运行基线 | `codes/chapters/ch07/core/dereverberation.py::offline_wpe`；外部对照 `nara_wpe`；`btk20`：`btk20_src/dereverberation/dereverberation.cc`、`.h` | 保护延迟、有效历史、相对加载；BTK20 的通道功率与加载不同，见研究 W08；数值对照见 W01/W11；BTK20 仅静态核对，未构建整链 |
| §7.4 | 逐帧在线 WPE | 外部参考实现 | `nara_wpe`：`nara_wpe/wpe.py::OnlineWPE` | 三接口 delay 索引不同；独立对照 `codes/chapters/ch07/examples/compare_online_wpe_reference.py`；未来扰动和跨块状态见 `codes/chapters/ch07/wpe_temporal_contract.py`；长静音状态溢出及恢复见 `codes/chapters/ch07/examples/wpe_silence_boundary.py` |
| §7.4；研究扩展：增强 W03 | 块在线/递推 WPE | 外部参考实现 | `nara_wpe`：`nara_wpe/tf_wpe.py` | TF 固定版权重 .7 乘当前块；块内估计后处理本块，非逐帧因果；静态核对见 W03 |
| §7.10；增强研究NeMo接口 | NeMo幅度掩码多轮WPE | 外部参考实现 | `nemo_wpe`：`masking.py::MaskBasedDereverbWPE`、`multichannel.py::WPEFilter`（精确路径见锁表） | 固定v2.4.0仅AST静态核验；迭代输入、长度屏蔽与trace加载不同，未执行Torch |
| §7.10 | DNN-WPE | 外部参考实现 | `espnet`：`espnet2/enh/layers/dnn_wpe.py`、`wpe.py`、`mask_estimator.py` | 配置值须追到实际求解器；固定版部分成员未传入，填充帧统计亦需核对；静态结果见 W04/W11 |
| §7.10 | WPD | 外部参考实现 | `espnet`：`espnet2/enh/layers/beamformer.py` WPD 分支及同目录的 `beamformer_th.py`、`dnn_beamformer.py` | 当前约束、延迟历史与功率；源码入口核对不代表整链运行 |
| 研究扩展：增强 W06 | AR-FastMNMF | 原理索引 | `fastmnmf_author`：`src/` 受限来源索引 | 学术研究限制，未纳入通用源码集合 |
| §8.5 | SI-SDR | 本仓库可运行基线 | `codes/chapters/ch08/core/separation.py::si_sdr` | 零均值、零能量、参考目标 |
| §8.5 | 小源数 PIT 排列枚举 | 本仓库可运行基线 | `codes/chapters/ch08/core/separation.py::pit_permutation` | 阶乘成本，不是长期身份关联 |
| §8.3；研究扩展：增强 B01 | FDICA | 外部参考实现 | `ssspy`：`ssspy/bss/fdica.py` | 跨频排列与尺度恢复 |
| §8.3 | AuxIVA 迭代投影 | 外部参考实现 | `ssspy`：`ssspy/bss/iva.py`；`codes/chapters/ch08/examples/reproduce_auxiva_reference.py` 调用固定源码 | 已跑一组数学合成盲估计及谐波反例；不等于语音论文复现或本书教学实现 |
| 研究扩展：增强 B02 | IVA 迭代源导向 ISS | 外部参考实现 | `ssspy`：同文件 ISS 选项 | 与 IP 不同，固定 ISS1/ISS2 |
| 研究扩展：增强 B02 | projection-back | 外部参考实现 | `ssspy`：`ssspy/algorithm/` | SI-SDR 通过不能证明尺度恢复 |
| §8.3；研究扩展：增强 B03 | OverIVA | 外部参考实现 | `overiva-author`：`overiva.py`、`auxiva_pca.py`；`piva`：`piva/auxiva.py` | MIT作者四文件选集与GPL编译路线分开；只核源码/轴序，未运行作者论文仿真 |
| §8.3；E08-31 | 给定目标行的OverIVA背景二阶约束 | 本仓库可运行基线 | `codes/chapters/ch08/core/overiva_teaching.py` | 只解所选坐标块；奇异拒绝、复共轭单独核，不是完整盲分离 |
| §8.6；E08-30 | 加权混合修正与STFT一致性操作次序 | 本仓库可运行基线 | `codes/chapters/ch08/core/consistency_teaching.py`，复用ch02唯一STFT | 六点两帧、非零给定估计；谱差与波形差分开，不评价真实源或训练网络 |
| 研究扩展：增强 B03 | FIVE | 外部参考实现 | `piva`：`piva/five.py` | 单目标提取不等于全部分离 |
| §8.3 | ILRMA | 外部参考实现 | `ssspy`：`ssspy/bss/ilrma.py` | NMF 基数与局部最优 |
| §8.3 | 满秩 MNMF | 外部参考实现 | `ssspy`：`ssspy/bss/mnmf.py::GaussMNMF` | 欠定模型不保证可辨识 |
| §8.3；研究扩展：增强 B06 | FastMNMF | 外部参考实现 | `pyroomacoustics`：`pyroomacoustics/bss/fastmnmf.py` | 与受限作者整库许可分开 |
| 研究扩展：增强 B06 | FastMNMF2 | 外部参考实现 | `pyroomacoustics`：`pyroomacoustics/bss/fastmnmf2.py` | 参数化、参考麦源图像 |
| 研究扩展：增强 B06 | Distributed FastMNMF（2026预印本） | 原理索引 | 作者原文 arXiv:2605.19388v1 §3.1～4.1 | 子阵块对角近似省略跨阵协方差，仍需同步/校准；未核实作者代码许可、未取得代码/模型 |
| 研究扩展：增强 B11 | 多帧满秩空间协方差分析 mfFCA | 外部参考实现 | `mffca`：`fca.py`、`main_synthetic.ipynb` | 固定作者源码仅内部评估，不修改或再分发；未运行整链 |
| §8.3 | TRINICON | 外部参考实现 | `pyroomacoustics`：`pyroomacoustics/bss/trinicon.py` | 该实现固定两个输出 |
| §8.4 | cACGMM | 本仓库可运行基线 | `codes/chapters/ch08/core/gss_teaching.py` 的受控 NumPy 活动导引迭代；`pb_bss`：`pb_bss/distribution/cacgmm.py` 与 `ssspy`：`ssspy/bss/cacgmm.py` 为外部对照入口 | 二麦合成复谱、给定活动；方向外积不保留原功率；ssspy 原包零次输出及给定后验形状更新已验证，未测 cACGMM 分离质量或官方 GPU/CHiME 整链 |
| §8.4 | GSS 活动约束分离 | 外部参考实现 | `gss`：`gss/core/enhancer.py` 为官方整链；本书 `codes/chapters/ch08/examples/gss_teaching_demo.py` 已运行 cACGMM→SCM→MVDR 受控子链 | 本书无混响输入中 WPE 旁路；官方 CuPy/Lhotse、RTTM 与 CHiME 语料未运行，不能外推为完整官方复现 |
| 研究扩展：增强 B09 | NeMo活动约束GSS掩码接口 | 外部参考实现 | `nemo_wpe`：`nemo/collections/audio/modules/masking.py::MaskEstimatorGSS` | 固定v2.4.0静态检查；未自动添加背景，返回掩码，不称已执行Torch整链 |
| 研究扩展：增强 B10 | GPU-GSS 批处理 | 外部参考实现 | `gss`：`gss/core/`、`recipes/` | CUDA/CuPy、批量、显存 |
| 研究扩展：增强 B10 | CuPy WPE 子实现 | 外部参考实现 | `wpe_gpu`：`wpe/` | GSS 子集，不覆盖全部在线接口 |

## 神经分离、条件提取与生成

| 正文或研究范围 | 算法/机制 | 覆盖状态 | 教学入口或主清单 ID：官方源码入口 | 关键边界 |
|---|---|---|---|---|
| §8.6 | Conv-TasNet | 外部参考实现 | `asteroid`：`asteroid/models/conv_tasnet.py` | 因果卷积、归一化与权重配置 |
| §8.6 | DPRNN | 外部参考实现 | `asteroid`：`asteroid/models/dprnn_tasnet.py` | 双路径方向、重叠重构 |
| §8.6 | SepFormer | 外部参考实现 | `speechbrain`：`speechbrain/lobes/models/dual_path.py` | 未来信息、整句归一化、峰值内存 |
| §8.6 | 离线 TF-GridNet | 外部参考实现 | `espnet`：`espnet2/enh/separator/tfgridnet_separator.py` | 固定通道数，不保证任意阵列 |
| §8.6 | S4M 骨干 | 外部参考实现 | `s4m`：`S4M.py`、`s4.py` | 缺完整训练入口/checkpoint，不是完整训练复现 |
| 研究扩展：增强 N06 | SPMamba | 外部参考实现 | `spmamba`：`audio_train.py`、`look2hear/` | 双向上下文与 CUDA 算子 |
| 研究扩展：增强 N07 | Mamba-TasNet/Dual-Path Mamba | 外部参考实现 | `mamba_tasnet`：`train_wsj0mix.py`、`modules/` | 配置变体、GPL、WSJ0 许可分别固定 |
| §8.6 | SpeakerBeam 方法 | 原理索引 | 增强 N08 原论文与注册模型 | 目标缺席、注册泄漏、设备失配 |
| 研究扩展：增强 N08 | BUTSpeechFIT/speakerbeam 软件 | 外部参考实现 | `speakerbeam`：`src/models/td_speakerbeam.py` | 原样本地内部评估；不修改、再分发或声称依赖/模型已运行 |
| §8.6；E08-23、29、32 | 双路 CSS 重叠相关排列关联 | 本仓库可运行基线 | `codes/chapters/ch08/core/css.py::match_two_source_overlap`、`overlap_application_gain` | 仅两槽实波形的重叠关联与同钟标量回归，不是分离器；静音、弱相关与并列返回不确定；四独立WAV不运行盲CSS |
| 研究扩展：增强N19 | MeCo一步平均流纠正 | 外部参考实现 | `meco`：`flowmse/model_MeCo.py`、`odes.py`、`sampling/odesolvers.py` | 16文件MIT部分选集；排除混合许可骨干/CUDA，非standalone；静态与论文分别读，未训练/前向/取模型 |
| 研究扩展：增强N20 | SIS说话人身份训练监督 | 原理索引 | MERL TR2026-137 §2～3；作者`merlresearch/sis_sep`仅发布预告 | 同身份异句辅助用于训练而非TSE推理注册；无方法实现/许可，不取得名义代码或运行模型 |
| §8.6；研究扩展：增强 N09 | 连续语音分离 CSS | 外部参考实现 | `notsofar1`：`css/css.py`、`css/css_with_conformer/separate.py` | 窗口、跨块排列、槽位、配置；不执行资产下载 |
| §13.3；研究扩展：增强 N10 | SGMSE+ | 外部参考实现 | `sgmse`：`enhancement.py`、`sgmse/model.py` | 采样器、权重和语料另核 |
| §13.3；研究扩展：增强 N10 | StoRM | 外部参考实现 | `storm`：`enhancement.py`、`sgmse/model.py` | 再生成不等于 SGMSE 推理流程 |
| §13.3；研究扩展：增强 N11 | ArrayDPS | 外部参考实现 | `arraydps`：`separate.py`、`src/sampler_spatial_v1_reverb_iva_8kHz.py` | 真值SDR参与停止/预算；论文ML选择另行记录，源码MIT与WSJ/权重许可分开 |
| §8.6；研究扩展：增强 N12 | AudioSep | 外部参考实现 | `audiosep`：`pipeline.py` | 默认单声道 32 kHz，不是注册语音 TSE |
| 附录 B §13.3/E13-11 | TAC 共享通道聚合 | 外部参考实现 | `tac`：`utility/models.py::DPRNN_TAC.forward`、`FaSNet.py::FaSNet_TAC.forward`；[当前静态审计](../appendix_b/examples/audit_tac_contracts.py) | 固定四文件与README非商业相同方式共享许可；内部逐通道等变，参考通道wrapper另判；未运行Torch网络，标量题仅证结构 |
| 附录 B §13.3；增强研究 N17 | DiCoW 条件目标说话人转写 | 外部参考实现 | `dicow-v1-inference`：`example.py`；`ts-asr-whisper-v1`：`src/main.py`、`src/train.py` | 固定源码已核，v1 推理需另取相配模型及两项 Pyannote gated 资产；`app.py` 在同一提交使用 v2 权重；未运行转写或论文评分 |
| 附录 B §13.3；增强研究 N18 | FlowSep 文本查询声音分离 | 原理索引 | `flowsep` 固定提交及 `lass_inference.py` 来源入口 | 固定源码许可未建立，未取得源码、模型或数据，也未运行推理；不将论文公开等同再分发授权 |

## 追踪、关联与控制

| 正文或研究范围 | 算法/机制 | 覆盖状态 | 教学入口或主清单 ID：官方源码入口 | 关键边界 |
|---|---|---|---|---|
| §9.2 | 角度—角速度 Kalman | 本仓库可运行基线 | `codes/chapters/ch09/core/tracking.py::ConstantVelocityKalman` | 最短角差、Joseph 更新、Q 的离散化 |
| §9.2 | EKF | 外部参考实现 | `filterpy`：`filterpy/kalman/EKF.py` | 观测雅可比、角度残差；固定版组合接口线性化点与分步调用不同，见追踪接口报告 |
| §9.2 | UKF | 外部参考实现 | `filterpy`：`filterpy/kalman/UKF.py` | sigma 点、圆周均值 |
| §9.2；E09-24；空间研究§40/61 | IMM交互与标量随机游走递推 | 本仓库可运行基线 | `codes/chapters/ch09/core/imm_teaching.py::ScalarRandomWalkIMM`；外部对照`filterpy`：`filterpy/kalman/IMM.py` | 同一局部展开角坐标、固定时间步和已知Q/R/T；完整密度、对数模式质量、连续缺测与均值间散布，不含未知偏置/环绕/通用运动模型；原接口与教学实现分开记录 |
| §9.3；E09-25；空间研究§61 | 单槽候选、确认与退役管理 | 本仓库可运行基线 | `codes/chapters/ch09/core/lifecycle.py::teaching_lifecycle`；实际PCM入口`examples/tracking_lifecycle_demo.py`归第9章 | 已关联单槽、整数半样本tick、连续命中与严格超时；两时钟协议分栏，局部ID不是永久说话人身份，不含多目标关联/概率存在推断或KF重置 |
| §9.2；E09-26；空间研究§61 | 已知姿态方向转换与静态vMF条件化 | 本仓库可运行基线 | `codes/chapters/ch09/core/spherical_tracking.py` | 指源单位向量、已知正交旋转；归一化协方差只是一阶局部近似，静态独立vMF似然按自然参数相加；非完整动态FvMFF或未知姿态估计 |
| §9.2.4 | 圆周 SIR 粒子滤波 | 本仓库可运行基线 | `codes/chapters/ch09/core/tracking.py::CircularParticleFilter` | 对数权重、多峰均值无定义 |
| §9.2.4 | 系统重采样 | 本仓库可运行基线 | `codes/chapters/ch09/core/tracking.py::systematic_resample` | 权重归一、随机种子 |
| §9.3；研究扩展：空间 §33 | 最近邻关联 | 外部参考实现 | `stonesoup`：`stonesoup/dataassociator/neighbour.py` | 单轨最近不等于全局最优 |
| §9.3；研究扩展：空间 §33 | GNN 全局关联 | 外部参考实现 | `stonesoup`：同文件；[本书两轨教学缩例](../ch09/tracking_crossing_dropout_demo.py) | 全局分配、一对一与门控；两轨缩例不代表完整关联器 |
| §9.3；研究扩展：空间 §33 | PDA | 外部参考实现 | `stonesoup`：`stonesoup/dataassociator/probability.py` | 漏检、检测概率、杂波密度 |
| §9.3 | JPDA | 外部参考实现 | `stonesoup`：同文件 `JPDA` | 联合事件，不自动维护说话人身份 |
| §9.3 | GM-PHD | 外部参考实现 | `stonesoup`：`stonesoup/updater/pointprocess.py::PHDUpdater` | 权重和是期望人数 |
| §9.3；研究扩展：空间 §34 | 高斯混合剪枝/合并 | 外部参考实现 | `stonesoup`：`stonesoup/mixturereducer/gaussianmixture.py` | 逐步核质量；固定版剪枝/截断重分配与合并权重上限见原方法诊断 |
| §9.3；空间 §35 | MHT：滑窗多帧分配形式 | 外部参考实现 | `stonesoup`：`stonesoup/hypothesiser/mfa.py`、`stonesoup/dataassociator/mfa/` | OR-Tools、N-scan；已知目标示例不是声学完整系统 |
| §9.3；空间研究 §56 | CPHD | 外部参考实现 | 归档锁表 `vo-rfs-tracking-updated`：`cphd/gms/run_filter.m` | 62文件选集已核；线性高斯目标研究模型，学术/研究用途受限；未执行MATLAB/MEX，非声学整链 |
| §9.3；空间研究 §56 | LMB | 外部参考实现 | 归档锁表 `vo-rfs-tracking-updated`：`lmb/gms/run_filter.m` | 62文件选集已核；线性高斯目标研究模型，学术/研究用途受限；未执行MATLAB/MEX，非声学整链 |
| §9.3；空间研究 §56 | δ-GLMB | 外部参考实现 | 归档锁表 `vo-rfs-tracking-updated`：`glmb/gms/run_filter.m` | 62文件选集已核；线性高斯目标研究模型，学术/研究用途受限；未执行MATLAB/MEX，非声学整链 |
| §9.3 | 检测前追踪 TBD | 原理索引 | 正文弱证据模型 | 硬阈值峰不能替代原输入 |
| §9.3 | OSPA | 外部参考实现 | `stonesoup`：`stonesoup/metricgenerator/ospametric.py` | 截断、阶数、单位；不直接评价身份 |
| §9.4 | 轨迹到波束预测/限速 | 原理索引 | 正文控制接口；[缺测与限速缩例](../ch09/tracking_crossing_dropout_demo.py) | 观测龄期、失效与最大角速度；缩例未接真实设备 |

| §9.2；空间研究 §31 | 固定增益 α-β / α-β-γ | 外部参考实现 | `filterpy`：`filterpy/gh/gh_filter.py` | 固定增益与采样间隔共同定义动态；不是自动估计协方差的Kalman |
| §9.1；空间研究 §31 | 滑动平均、中值与众数基线 | 原理索引 | 正文比较三种统计量 | 平均线性，中值/众数非线性；众数依赖离散化，圆周边界另处理 |
| §9.2；空间研究 §32 | APF 辅助粒子滤波 | 原理索引 | Pitt–Shephard观测引导祖先选择与重要性校正 | 改进提议分布不自动提供异常值鲁棒性；不可把SIR改名为APF |
| §9.3；空间研究 §36 | GOSPA | 外部参考实现 | `stonesoup`：`stonesoup/metricgenerator/ospametric.py::GOSPAMetric` | p≥1，α=2分解定位/漏检/虚警；单帧位置指标不证明身份连续 |
| 研究扩展：空间 §55.2 | RBMCDA 条件解析多目标追踪 | 外部参考实现 | `spatial-audio-framework`：`framework/modules/saf_tracker/saf_tracker.c`、`saf_tracker_internal.c` | 关联粒子与条件Kalman分开；GPL-2.0-or-later模块；原step提取运行使用计数替身，未执行关联/PF/KF整链 |

| 研究扩展：空间 §57 | 滑窗排列不变训练 sPIT（2023） | 原理索引 | [EUSIPCO作者原文](https://eurasip.org/Proceedings/Eusipco/Eusipco2023/pdfs/0000251.pdf)，§2～3式(1)～(4) | 训练时跨历史窗统一排列，固定槽位容量；未核作者训练代码及明确许可、未执行网络，不是部署时用真值关联 |

## 连续音频、噪声控制与部署

| 正文或研究范围 | 算法/机制 | 覆盖状态 | 教学入口或主清单 ID：官方源码入口 | 关键边界 |
|---|---|---|---|---|
| §10.1.2 | 环形缓冲与预卷 | 本仓库可运行基线 | `codes/chapters/ch10/core/engineering.py::RingBuffer` | Python 教学容器不是无锁硬实时队列 |
| §10.1.1 | 截止期限/队列模拟 | 本仓库可运行基线 | `codes/chapters/ch10/core/engineering.py::simulate_deadline_queue` | 等待、计算与排队分别计量 |
| §10.1.2 | PortAudio 回调/时间戳 | 外部参考实现 | `portaudio`：`include/portaudio.h`、`examples/` | 不阻塞、不分配；ADC/DAC 时钟核对 |
| §10.1.2；工业 I02 | ALSA PCM 状态恢复 | 外部参考实现 | `alsa-lib`：`src/pcm/pcm.c`、`test/pcm.c` | xrun、挂起、移除分别处理 |
| §10.1.2；工业 I03 | PipeWire AEC 四流路由 | 外部参考实现 | `pipewire`：`src/modules/module-echo-cancel.c` | 播放参考完整，避免无意双重 AEC |
| §10.2.1 | SRO 线性拟合 | 本仓库可运行基线 | `codes/chapters/ch10/core/engineering.py::estimate_sro_ppm`；`codes/chapters/ch10/sro_closed_loop_demo.py` | 初始时差与斜率分开；示例用精确合成时间戳估计 |
| §10.2.1 | 线性插值 SRO 补偿 | 本仓库可运行基线 | `codes/chapters/ch10/core/engineering.py::resample_sro_to_reference`；`codes/chapters/ch10/sro_closed_loop_demo.py` | 示例含跨块状态和单点丢样标记；不含抗混叠与真实异步控制 |
| §10.2.1 | libsamplerate 连续 SRC | 外部参考实现 | `libsamplerate`：`src/samplerate.c` | 速率比、消耗量与状态 |
| §10.2.1 | SpeexDSP 连续 SRC | 外部参考实现 | `speexdsp`：`libspeexdsp/resample.c` | 质量档、通道一致性 |
| §10.3.2 | 功率谱减 | 本仓库可运行基线 | `codes/chapters/ch10/core/noise_suppression.py::power_spectral_subtraction`；[E10-13 与音频演示](../ch10/spectral_subtraction_demo.py) | 固定噪声专用前奏、单通道 STFT；无在线噪声跟踪与语音概率，音乐噪声仍可能出现 |
| §10.3.1 | MMSE-STSA | 原理索引 | 正文谱幅度目标 | 不等于 Speex 修改后的响度域增益 |
| §10.3.1 | MMSE-LSA | 原理索引 | 正文对数谱幅度目标 | 注释可选式不是默认完整实现 |
| §10.3.1；研究扩展：空间 §53 | WebRTC 分位数噪声估计 | 外部参考实现 | `webrtc`：`modules/audio_processing/ns/quantile_noise_estimator.cc` | 不自动命名为 MCRA/IMCRA |
| §10.3.3；研究扩展：空间 §53 | WebRTC NS 语音概率估计 | 外部参考实现 | `webrtc`：`modules/audio_processing/ns/speech_probability_estimator.cc` | 不等同传统 VAD API |
| §10.3.1 | RNNoise | 外部参考实现 | `rnnoise`：`src/denoise.c` | DSP/神经状态、训练域和模型 |
| §10.3.1；工业 I09 | DeepFilterNet 深度滤波 | 外部参考实现 | `deepfilternet`：`DeepFilterNet/df/`、`libDF/src/` | 48 kHz、阶数、前瞻与延迟 |
| 研究扩展：工业 I09 | DeepFilterNet LADSPA | 外部参考实现 | `deepfilternet`：`ladspa/` | 无前瞻仍有 STFT/宿主延迟 |
| §10.3.4 | 能量迟滞/挂起 VAD | 本仓库可运行基线 | `codes/chapters/ch10/core/engineering.py::HysteresisVAD` | 能量门限，不是神经概率 |
| §10.3.4 | 峰值保护 AGC | 本仓库可运行基线 | `codes/chapters/ch10/core/engineering.py::PeakProtectAGC` | 不代表完整响度/限幅器 |
| §10.3.4；工业 I06 | WebRTC 传统 VAD | 外部参考实现 | `webrtc`：`common_audio/vad/webrtc_vad.c` | int16 单声道与合法 10/20/30 ms |
| §10.3.4；工业 I07 | WebRTC AGC2 | 外部参考实现 | `webrtc`：`modules/audio_processing/gain_controller2.h` | 数字增益与输入音量分开 |
| §10.3.4；工业 I08 | Silero VAD 流式状态 | 外部参考实现 | `silero-vad`：`src/silero_vad/utils_vad.py` | 16 kHz/512、8 kHz/256；会话隔离 |
| §10.4.1 | Q1.15 量化 | 本仓库可运行基线 | `codes/chapters/ch10/core/engineering.py::q15_quantize` | 最近偶数舍入与饱和 |
| §10.4.1 | Q15 宽累加点积 | 本仓库可运行基线 | `codes/chapters/ch10/core/engineering.py::q15_dot` | 不默认与截位 DSP 逐位相同 |
| §10.4.1；工业 I15 | CMSIS-DSP Q15 FIR | 外部参考实现 | `cmsis_dsp`：`Source/FilteringFunctions/arm_fir_q15.c` | 系数、状态、内核变体与周期 |
| §10.4.1 | PTQ 校准量化流程 | 原理索引 | 正文部署路线 | 内核存在不代表模型校准完成 |
| §10.4.1 | QAT 量化感知训练 | 原理索引 | 正文训练路线 | 不把量化推理内核当完整训练器 |
| §10.4.1；工业 I16 | CMSIS-NN 量化内核 | 外部参考实现 | `cmsis-nn`：`Include/arm_nnfunctions.h` | 零点、尺度、scratch、指令集 |
| §10.9；工业 I17 | TFLM tensor arena | 外部参考实现 | `tflite-micro`：`tensorflow/lite/micro/recording_micro_interpreter.h` | 模型大小不同于 arena/栈/缓存 |
| §10.9；工业 I17 | TFLM micro_speech 前端 | 外部参考实现 | `tflite-micro`：`tensorflow/lite/micro/examples/micro_speech/` | 关键词例子不是阵列系统 |
| §10.9；工业 I18 | ONNX Runtime 流式执行 | 外部参考实现 | `onnxruntime`：`include/onnxruntime/core/session/onnxruntime_c_api.h` | 算子集、EP、线程池与循环状态 |
| §10.9.1；工业 I10 | XMOS AEC/ADEC | 外部参考实现 | `lib-voice`：`lib_voice/src/aec/`、`lib_voice/api/adec/` | fwk_voice 已迁移；商用硬件限制 |
| §10.9.1；工业 I11 | XMOS IC | 外部参考实现 | `lib-voice`：`lib_voice/src/ic/` | 240 点步长/512 点分析、泄漏 |
| §10.9.1；工业 I11 | XMOS VNR | 外部参考实现 | `lib-voice`：`lib_voice/src/vnr/` | 比值控制不是无误活动判决 |
| §10.9.1；工业 I12 | XMOS NS | 外部参考实现 | `lib-voice`：`lib_voice/src/ns/` | 元数据和词尾保留 |
| §10.9.1；工业 I12 | XMOS AGC | 外部参考实现 | `lib-voice`：`lib_voice/src/agc/` | 活动/回声状态与增益共同测试 |
| §10.9.1；工业 I13 | SOF 固定 FIR 波束 | 外部参考实现 | `sof`：`src/audio/tdfb/tdfb_generic.c` | 滤波组/方向，不是 SCM 自适应 MVDR |
| §10.9.1；工业 I14 | SOF 固件 SRC | 外部参考实现 | `sof`：`src/audio/src/` | 固定比转换不等于异步补偿 |
| §10.7 | 分布式阵列同步/融合 | 原理索引 | 正文分布式模型 | SRO 拟合不是完整网络系统 |
| 导读复现；§10.2.1/10.7 | WASN DXCP-PhaT同步源码接口 | 外部参考实现 | `wasn-platform`：`DXCPPhaT_demo/system/DXCP_PhaT/dxcp_phat.py`、`sync_sed/system/resample.py`；[固定源集](research/04_source_reproduction.md#overview-source-entrypoints) | 限定源/许可已取得；DXCP算法、MARVELO、完整同步及设备未运行 |
| 专题Ⅱ §15.5～8、§15.13、E15-04～08/25 | 指定LMMSE任务的固定线性压缩 | 本仓库可运行基线 | `codes/chapters/ch15/core/distributed.py::compressed_mwf` | 已知总体协方差、归一接收坐标；E25仅比较可逆广播坐标下共同重映射与错误累计，复用MWF，不增加算法行；不称一般非线性充分统计或有限码率无损 |
| 专题Ⅱ §15.9～10、E15-09～11 | 有限顺序、同时与固定松弛广播更新 | 本仓库可运行基线 | `codes/chapters/ch15/core/distributed.py::distributed_updates` | 真实广播前求解与广播后当前接收分开；停止看所有实际有效权重；非rS+定理完整实现 |
| 专题Ⅱ §15.12、E15-24 | 已知统计的秩一GEVD-MWF白化控制 | 本仓库可运行基线 | `codes/chapters/ch15/core/distributed.py::gevd_control` | 两维白化与λ−1目标重构，不是完整分布式GEVD-DANSE |
| 专题Ⅱ §15.13；工业 §8 | 作者WOLA-DANSE脚本 | 外部参考实现 | `danse-wola`：`WOLA_DANSE1.m` | 固定作者三条件文件头；静态与独立控制，不运行原MATLAB/WOLA声学整链 |
| 专题Ⅱ §15.12；工业 §8 | OnlineWACD采样率偏差估计 | 外部参考实现 | `paderwasn`：`paderwasn/synchronization/sro_estimation.py::OnlineWACD` | 已取得原源；SciPy/paderbox及完整估计器未运行，三个helper调用不能替代 |
| 专题Ⅱ §15.12；工业 §8 | DWACD动态相干漂移估计 | 外部参考实现 | `paderwasn`：`paderwasn/synchronization/sro_estimation.py::DynamicWACD` | 活动/窗口与声学相位条件；原论文定位另记，不称本书盲SRO实测 |
| 专题Ⅱ §15.12；工业 §8 | TI-DANSE+ 2025批量协方差实验 | 外部参考实现 | `tidanseplus-batch`：`main.py`、`package/asc.py`、`package/online.py` | 同提交MIT/GPL声明冲突，源码仅本地忽略目录研究；原实验/依赖未执行 |
| 专题Ⅱ §15.11；工业 §8 | TI-DANSE 2017拓扑无关融合 | 原理索引 | Szurley、Bertrand、Moonen原论文§III/IV；本地E15-23仅树消息控制 | 精确同样本求和、目标维数和G可逆；四消息加法不替代完整算法 |
| 专题Ⅱ §15.12；工业 §8 | 分布式GEVD-DANSE 2016 | 原理索引 | 原论文§IV-B/F及附录B；`danse-python`仅来源身份索引 | 正定SCM、特征间隙与固定秩；没有主项目许可，不取原网络实现 |
| 专题Ⅱ §15.12；工业 §8 | SRO-GEVD-DANSE 2023 | 原理索引 | 作者arXiv:2211.02489v2、算法1/2 | 常量时钟偏差与相干漂移/WOLA条件；已知真值线性SRC不是原估计器 |
| 专题Ⅱ §15.12；工业 §8 | TI-GEVD-DANSE 2024共同规范化 | 原理索引 | EUSIPCO 2024作者稿§III/IV | 协方差与滤波器共享同一坐标变换；未运行完整网络GEVD，不能独立归一每向量冒充 |
| 专题Ⅱ §15.12；工业 §8 | TI-DANSE+ 2026扩展 | 原理索引 | 作者arXiv:2506.02797v2、定理1/2与§III-G/IV-D | 目标秩、统计重构和实验条件分别核；2025批量源码不自动复现该版 |
| 专题Ⅱ §15.12 | TI-dMWF 2026全局与局部源模型 | 原理索引 | 作者arXiv:2607.05561v1、模型与Remark1 | 只在一节点局部或全部节点共同的源条件；部分子集可见源不援引原证明 |
| §10.10 | 遥测记录校验 | 本仓库可运行基线 | `codes/chapters/ch10/core/engineering.py::validate_telemetry`、`codes/chapters/ch10/engineering/telemetry_schema.json` | 类型/范围的无状态检查；累计RTF与可选逐帧服务RTF分开；生产者声明统计窗/计时范围/时钟域，校验器不证明跨记录关系 |
| §10.10；E10-34 | 已知失效选集与约束重建 | 本仓库可运行基线 | `codes/chapters/ch10/core/channel_selection.py::select_channel_observations`、`select_mvdr_channels`；六份独立PCM与图78 | 已知删除列表同时选择观测、响应与协方差双轴；复用第5章MVDR，不计作新波束算法；无盲检测/自动加载/任意频率性能 |

## 任务评分与排除范围

| 正文或研究范围 | 算法/机制 | 覆盖状态 | 教学入口或主清单 ID：官方源码入口 | 关键边界 |
|---|---|---|---|---|
| §10.5；工业 I19 | DNSMOS | 外部参考实现 | `dns-challenge`：`DNSMOS/dnsmos_local.py` | 模型/个人化/窗口；不是受试者 MOS |
| §6.7、§10.5 | AECMOS | 外部参考实现 | `aec-challenge`：`AECMOS/AECMOS_local/` | 三路对应、场景、裁段规则 |
| §11.2 | cpWER | 外部参考实现 | `meeteval`：`meeteval/wer/wer/cp.py` | 全局说话人分组后的会话级排列；CSS复用的原始槽位不能直接当永久说话人流 |
| §11.2 | ORC-WER | 外部参考实现 | `meeteval`：`meeteval/wer/wer/orc.py` | 参考片段到流映射，偏向内容完整性；不保证身份连续 |
| §11.2 | MIMO-WER | 外部参考实现 | `meeteval`：`meeteval/wer/wer/mimo.py` | 流顺序、允许映射 |
| §11.2 | DI-cpWER | 外部参考实现 | `meeteval`：`meeteval/wer/wer/di_cp.py` | 锁定入口使用贪心匹配；与理论最优定义、简化文档示例及 sa-WER 均须区分 |
| §11.2；E11-24；§13.7；工业 I21 | tcpWER | 外部参考实现 | `meeteval`：`meeteval/wer/wer/time_constrained.py` | 时间约束、容差和词时间戳 |
| §13.7；工业 I21 | CHiME-8 文本规范化/评分 | 外部参考实现 | `chime-utils`：`chime_utils/scoring/meeteval.py`、`chime_utils/text_norm/`、`tests/test_normalizer.py` | 固定源码的缺文件分支和规范化幂等检查另有受控诊断；届次、划分、缺失场景与数据许可 |
| 导读路径C；§10.5/10.6 | LibriCSS公开评测程序 | 外部参考实现 | `libricss`：`scoring/python/asclite_libricss.py`、`asr/python/get_wer.py`；[源集与协议](research/04_source_reproduction.md#overview-source-entrypoints) | 连续评分与真值切段最低WER不同；外部SCTK/GLM/数据闭包未取，原接口限制/未运行状态保留 |
| §11.2 | sa-WER 说话人归属口径 | 原理索引 | 正文指标区别 | 不声称锁定 MeetEval 覆盖全部定义 |
| §5.9 | 未指定实现的“DNNBeamformer” | 明确排除 | 无唯一算法/项目身份 | 具体网络需按模型和官方实现另登记 |

## 补充源码研究：控制、连续处理与评分

下列项目按具体机制登记；文件读写和数值内核是工程接口，不称为新的阵列估计算法。

| 正文或研究范围 | 算法/机制 | 覆盖状态 | 教学入口或主清单 ID：官方源码入口 | 关键边界 |
|---|---|---|---|---|
| 第6章；增强 A14 | DNN 控制线性 CTF AEC | 外部参考实现 | `e2e-ad-aec`：`libPython/class_aec_ctf.py`、`libPython/class_frontend.py` | BSD-4-Clause；无权重、训练数据另取；居中 STFT |
| 第6、8章；增强研究 | 联合/级联多参考 AEC 与 NR | 外部参考实现 | `integrated-aec-nr`：`Util/Process/process.m` | 分解干净信号及 oracle VAD；不是采集即用系统 |
| 第8章；增强 N13 | OnlineSpatialNet | 外部参考实现 | `nbss`：`models/arch/OnlineSpatialNet.py` | 因果结构与跨调用状态分开验收 |
| 第10章；工业 I22 | libsoxr 连续/变比 SRC | 外部参考实现 | `libsoxr`：`src/soxr.h`、`examples/5-variable-rate.c` | 输入/输出比、显式质量、消费量与尾部排空 |
| 第10章；工业 I23 | EBU 响度与真峰值测量 | 外部参考实现 | `libebur128`：`ebur128/ebur128.c` | 通道角色、门控、静音非有限值；不等于声压 |
| §10.5；工业 I24 | STOI | 外部参考实现 | `pystoi`：`pystoi/stoi.py::stoi` | 干净参考、对齐、10 kHz；短片段警告不当有效分数 |
| §10.5；工业 I24 | ESTOI | 外部参考实现 | `pystoi`：同函数 `extended=True` | 与普通 STOI 分开报告；同样有有效帧限制 |
| §10.5；工业 I25 | ViSQOL | 外部参考实现 | `visqol`：`src/visqol_api.cc` | MOS-LQO；16/48 kHz 模式、模型/依赖未取、下混不评空间保真 |
| 第10章；工业 I26 | PCM/WAV 帧读写 | 外部参考实现 | `libsndfile`：`src/sndfile.c`、`src/pcm.c` | 帧数与标量数、短读、格式与浮点转换 |
| 第10章；工业 I27 | XMOS 块浮点与底层 DSP | 外部参考实现 | `lib-xcore-math`：`lib_xcore_math/src/bfp/`、`lib_xcore_math/src/arch/ref/` | v3.0.0 随 lib_voice 固定；硬件许可、共享指数与目标周期 |
| 研究扩展：增强 N14 | Stream.FM 流匹配逐帧恢复 | 外部参考实现 | `streamfm`：`sgmse/backbones/streaming_unet.py`、`sgmse/util/solvers.py` | AGPL-3.0；逐阶段状态须核对；源码接口疑点与推理未运行分别记录 |
| 研究扩展：工业 I28 | FastEnhancer 流式单通道降噪 | 外部参考实现 | `fastenhancer`：`models/fastenhancer/default/model.py`、`scripts/test_onnx.py` | MIT源码；未取ONNX权重、未运行；不是AEC/WPE/多麦分离 |
| 研究扩展：工业 I31 | FastEnhancer-Medium 独立 C11/int8 流式运行时 | 外部参考实现 | `faster-enhancer-c`：`include/`、`src/`；[接口研究](research/03_industrial_deployment.md#i31) | 与原 Python 项目不同作者及配置；MIT 源码与 NOTICE 已核，权重/测试音频未取，未编译推理或测硬件 |
| 研究扩展：增强 N15 | TF-Locoformer 复谱语音分离 | 外部参考实现 | `tf-locoformer`：`standalone/tflocoformer_separator.py`、`espnet2/enh/separator/tflocoformer_separator.py` | 固定作者源码；静态接口疑点单列；无权重推理或完整训练复现 |
| 附录B E13-02；工业 I29 | STK DelayL 固定分数延迟与连续状态 | 外部参考实现 | `stk`：`include/DelayL.h`、`src/DelayL.cpp`；`codes/chapters/ch10/examples/run_stk_delay_probe.py` | 已运行标量接口、零/.5/1采样与分块状态；未测设备或变时延 |

## 章节代码练习与音频映射

358 道代码练习沿用各章已有模型，稳定 ID 与原有数字题号并存。下表只登记学习入口，不改变上面的 349 行算法统计。补充的空间精算、增强步骤、时间状态模块分别提供3/5/4道题。三个原有 `exercises_` 模块各自提供 `run_exercises()`，分别有 28/23/25 道题；AEC 小实验另有 4 道，进阶 AEC 手算另有 10 道；E03-07、E04-08、E09-06 与 E10-13 由独立实验入口提供。E04-08 的 200 次独立双源抽样只说明固定模型中的分辨事件频率和 Wilson 区间。E04-04 是固定矩阵的前向空间平滑演示，不扩称为支持任意阵列的公共估计接口。

| 章节与稳定 ID | 练习入口 | 回归测试 |
|---|---|---|
| 扩展专题Ⅰ：E14-01～18（18题） | [成像逐层复算](../ch14/chapter14_exercises.py)、[五份快拍音频](../ch14/imaging_audio/MANIFEST.json)、[原源合同](../ch14/reports/upstream_imaging_contracts_current.json) | [独立教学/PCM测试](../../../tests/test_codes_imaging.py)、[原源和目标测试](../../../tests/test_codes_imaging_contracts.py) |
| 扩展专题Ⅱ：E15-01～25（25题） | [分布式逐步复算](../ch15/chapter15_exercises.py)、[17份传输控制音频](../ch15/distributed_audio/MANIFEST.json)、[固定原源合同](../ch15/reports/upstream_distributed_contracts_current.json) | [独立数学/状态测试](../../../tests/test_codes_distributed.py)、[PCM资产测试](../../../tests/test_codes_distributed_audio.py)、[上游合同测试](../../../tests/test_codes_distributed_contracts.py) |
| 第6章：E06-34～39（6题） | [APA五个完整缩例](../ch06/aec_affine_projection_demo.py)、[有色参考训练/留出](../ch06/examples/generate_apa_audio.py) | [状态/手算测试](../../../tests/test_codes_aec_affine_projection.py)、[PCM实验测试](../../../tests/test_codes_aec_apa_audio.py) |
| 第6章：E06-40～42（3题） | [增益与尾声逐样本控制](../ch06/core/reference_timing.py)、[章节入口](../ch06/chapter06_experiments.py)、[六份PCM生成/只读核验](../ch06/examples/generate_reference_audio.py) | [独立手算边界](../../../tests/test_codes_ch06_reference_timing.py)、[完整PCM回放](../../../tests/test_codes_ch06_reference_audio.py)、[发布与整数评分](../../../tests/test_ch06_reference_publication.py) |
| 第 6 章：E06-22～33（12题） | [AEC 状态、数值与指标实验](../ch06/chapter06_experiments.py) | [独立测试](../../../tests/test_codes_chapter06_experiments.py)、[数值边界](../../../tests/test_codes_aec_numerical_boundaries.py) |
| 第 5 章：E05-08～24（17题） | [约束、谱估计与状态实验](../ch05/chapter05_experiments.py) | [独立测试](../../../tests/test_codes_chapter05_experiments.py)、[GSC状态测试](../../../tests/test_codes_gsc.py)、[相位/掩码独立模型](../../../tests/test_codes_ch05_phase_mask_models.py)、[三PCM全回放](../../../tests/test_codes_ch05_phase_audio.py) |
| 第 4 章：E04-12～25（14题） | [定位逐步实验](../ch04/chapter04_experiments.py) | [原逐步实验测试](../../../tests/test_codes_chapter04_experiments.py)、[直达路径与CTF独立测试](../../../tests/test_codes_ch04_direct_path.py)、[相干反射PCM测试](../../../tests/test_codes_ch04_reflection_audio.py) |
| 第 3 章：E03-08～18（11题） | [几何与校准逐步实验](../ch03/chapter03_experiments.py)、[已知方向唯一逆核](../ch03/core/baseline_calibration.py)、[六PCM训练/留出清单](../ch03/baseline_audio/MANIFEST.json)、[当前原方法合同](../ch03/reports/upstream_coarray_contracts.json) | [独立题目测试](../../../tests/test_codes_chapter03_experiments.py)、[独立逆核测试](../../../tests/test_codes_ch03_baseline_calibration.py)、[完整音频回放](../../../tests/test_codes_baseline_audio.py)、[来源与写入合同](../../../tests/test_codes_ch03_upstream_coarray.py) |
| 第 2 章：E02-09～20（12题） | [声学模型逐步实验](../ch02/chapter02_experiments.py) | [独立测试](../../../tests/test_codes_chapter02_experiments.py) |
| 第 1 章：E01-04～10（7题） | [基础逐步实验](../ch01/chapter01_experiments.py) | [独立测试](../../../tests/test_codes_chapter01_experiments.py) |
| 附录 A：E12-06～19（14题） | [数学与边界逐步实验](../appendix_a/appendix_a_experiments.py)，[分块卷积实现](../appendix_a/core/math_foundations.py)，[实际主PCM检查](../appendix_a/examples/check_main_math_audio.py)，[五路已知权重音频](../appendix_a/weighted_audio/MANIFEST.json)，[原求解器执行合同](../appendix_a/reports/upstream_solver_contracts.json) | [独立数学测试](../../../tests/test_codes_appendix_a_experiments.py)、[FFT支持边界](../../../tests/test_codes_appendix_a_boundaries.py)、[完整音频回放](../../../tests/test_codes_weighted_audio.py)、[原核身份与执行](../../../tests/test_codes_upstream_solver_contracts.py) |
| 附录 B：E13-03～14（12题） | [房间、相位、PCM 评分窗口、频谱反例与来源证据实验](../appendix_b/appendix_b_experiments.py)，[房间数值报告](../appendix_b/room_audio/RESULTS.json) | [独立测试](../../../tests/test_codes_appendix_b_experiments.py) |
| 第 1～5 章：`E01-01`～`E01-03`、`E02-01`～`E02-06`、`E03-01`～`E03-07`、`E04-01`～`E04-09`、`E05-01`～`E05-05`（30 题） | [exercises_spatial.py](cross_chapter/exercises_spatial.py)（28 题）；[协同阵 E03-07](../ch03/coarray_covariance_exercise.py)、[分辨率 E04-08](../ch04/doa_resolution_trials.py) | [test_codes_exercises_spatial.py](../../../tests/test_codes_exercises_spatial.py)、[空间扩展测试](../../../tests/test_codes_spatial_expansion.py) |
| 第 6～9 章：`E06-01`～`E06-20`、`E07-01`～`E07-05`、`E08-01`～`E08-07`、`E09-01`～`E09-06`（38 题） | [exercises_enhancement.py](cross_chapter/exercises_enhancement.py)（23 题）；[AEC 四个边界小例](../ch06/aec_algorithm_minicases.py)（`E06-07`～`E06-10`）；[AEC 十个进阶手算](../ch06/aec_advanced_exercises.py)（`E06-11`～`E06-20`）；[交叉追踪 E09-06](../ch09/tracking_crossing_dropout_demo.py) | [test_codes_exercises_enhancement.py](../../../tests/test_codes_exercises_enhancement.py)、[test_codes_aec_minicases.py](../../../tests/test_codes_aec_minicases.py)、[test_codes_aec_advanced_exercises.py](../../../tests/test_codes_aec_advanced_exercises.py)、[test_codes_tracking_crossing_dropout.py](../../../tests/test_codes_tracking_crossing_dropout.py) |
| 第 10～11 章、附录 A/B：`E10-01`～`E10-14`、`E11-01`～`E11-07`、`E12-01`～`E12-04`、`E13-01`（26 题） | [exercises_engineering.py](cross_chapter/exercises_engineering.py)（25 题）；[谱减 E10-13](../ch10/spectral_subtraction_demo.py)（1 题） | [test_codes_exercises_engineering.py](../../../tests/test_codes_exercises_engineering.py)、[test_codes_spectral_subtraction.py](../../../tests/test_codes_spectral_subtraction.py) |
| E02-07、E04-10、E05-06（3题） | [空间精算](cross_chapter/spatial_precision_exercises.py) | [独立测试](../../../tests/test_codes_spatial_precision.py) |
| E09-10～26（17题） | [第9章逐步计算](../ch09/chapter09_experiments.py) | [独立解析测试](../../../tests/test_codes_chapter09_experiments.py)、[PCM音频](../../../tests/test_codes_tracking_audio.py) |
| E10-18～34（16题） | [第10章工程逐步计算](../ch10/chapter10_experiments.py) | [独立解析与PCM测试](../../../tests/test_codes_chapter10_experiments.py)、[数值边界](../../../tests/test_codes_engineering_ch10_boundaries.py) |
| E11-10～27（18题） | [第11章约束与选型逐步计算](../ch11/chapter11_experiments.py)；[小规模选型与评分模型](../ch11/core/selection.py)；[同一物理阵列选型](../ch11/core/selection_physics.py)；[四路同增益FIR音频](audio/MANIFEST.json)、[独立双场景八WAV](../ch11/scenario_audio/MANIFEST.json)；[原词编辑核合同](../ch11/reports/meeting_kernel_contracts_current.json) | [独立解析、整数边界与PCM测试](../../../tests/test_codes_chapter11_experiments.py)、[完整资产回放](../../../tests/test_codes_selection_audio.py)、[事件与独立物理闭式](../../../tests/test_codes_selection_evidence.py)、[写前模型保护](../../../tests/test_codes_ch11_selection_fixture_guard.py)、[原核身份与执行](../../../tests/test_codes_meeting_kernel_contracts.py) |
| E08-12～32（21题） | [第8章逐步计算](../ch08/chapter08_experiments.py) | [独立测试](../../../tests/test_codes_chapter08_experiments.py) |
| E07-08～21（14题） | [第7章逐步计算](../ch07/chapter07_experiments.py) | [独立测试](../../../tests/test_codes_chapter07_experiments.py) |
| E06-21、E07-06、E08-08～10（5题） | [增强逐步计算](cross_chapter/enhancement_step_exercises.py) | [独立测试](../../../tests/test_codes_enhancement_steps.py) |
| E09-07～08、E10-15、E11-08（4题） | [时间与状态](cross_chapter/tracking_time_exercises.py) | [独立测试](../../../tests/test_codes_time_state_exercises.py) |

在仓库根目录使用模块入口：

```bash
.venv/bin/python -m codes.chapters.ch09.chapter09_experiments
.venv/bin/python -m codes.chapters.ch11.chapter11_experiments
.venv/bin/python -m codes.chapters.ch08.chapter08_experiments
.venv/bin/python -m codes.chapters.ch07.chapter07_experiments
.venv/bin/python -m codes.chapters.ch06.chapter06_experiments
.venv/bin/python -m codes.chapters.ch05.chapter05_experiments
.venv/bin/python -m codes.chapters.ch04.chapter04_experiments
.venv/bin/python -m codes.chapters.ch03.chapter03_experiments
.venv/bin/python -m codes.chapters.ch00.cross_chapter.exercises_spatial
.venv/bin/python -m codes.chapters.ch00.cross_chapter.exercises_enhancement
.venv/bin/python -m codes.chapters.ch06.aec_algorithm_minicases
.venv/bin/python -m codes.chapters.ch06.aec_advanced_exercises
.venv/bin/python -m codes.chapters.ch00.cross_chapter.exercises_engineering
.venv/bin/python -m codes.chapters.ch09.tracking_crossing_dropout_demo
.venv/bin/python -m codes.chapters.ch10.spectral_subtraction_demo
.venv/bin/python -m codes.chapters.ch03.coarray_covariance_exercise
.venv/bin/python -m codes.chapters.ch04.doa_resolution_trials
```

题目、答案和 27 组、109 个合成音频的对应关系见[练习与音频实验](research/05_exercises_and_audio.md)。音频由 [generate_audio_samples.py](examples/generate_audio_samples.py) 生成，参数和摘要见 [MANIFEST.json](audio/MANIFEST.json)；它们只展示特定条件下的现象，不作为完整算法、工业性能或自然语音听测的新增覆盖证据。

另有二十四套独立合成资产，共138个WAV；逐套清单、用途与边界见[导读完整音频表](../../../chapters/00_overview.md#audio-assets)。它们不计入主109；GSS的STATE与房间结果报告不是额外WAV，合成模型及评分条件各自独立，不等于真实语音或设备验证。

第8章另有[六份已知掩码表示WAV](../ch08/mask_audio/MANIFEST.json)，对应E08-28与图60；全记录FFT和已知目标构造只检验表示范围，27200点实际PCM评分与解析、浮点结果分开，不计为新分离算法或工业性能覆盖。

## 未完成项怎样保留

外部参考实现可以存在构建失败、兼容性问题或算法缺陷；CSSM/WAVES/TOPS 的静态问题与 WPE 已完成的合成数值对照应分别阅读，不能把项目数量当成通过率。只有骨干结构时，覆盖的是结构源码，不是完整训练、checkpoint 推理或论文表格。

原理索引明确保留下一步所需证据：唯一作者实现、明确许可、原模型配置，或与正文模型一致的最小代码。不得仅因为框架大、copyleft 或权重未授权就将许可明确的源码降为“没有实现”；也不得因同名函数存在就将整个算法家族标为已覆盖。

本仓库不提交下载缓存、模型权重或未经授权的第三方语料。109 个自行合成的教学 WAV 按章放在 `codes/chapters/*/audio/`，单一总清单位于 `codes/chapters/ch00/audio/MANIFEST.json`；许可明确的 DEMAND 小型摘录和派生文件位于 `codes/chapters/ch02/real_audio/`，不包含完整下载归档。独立上游工作目录的取得、许可保留与未执行项目按来源状态记录报告。算法、源码或排除范围变化时，同步修改本表、研究说明、来源清单和真实验证记录。


真实数据练习 R01 使用 [prepare_real_recordings.py](../ch02/examples/prepare_real_recordings.py) 与 [real_recordings.py](../ch02/core/real_recordings.py)，测试见 [test_codes_real_recordings.py](../../../tests/test_codes_real_recordings.py)。R01 比较 DEMAND 录音的数字域二阶矩、交叉项与零延时均值，不是新增定位或增强算法，亦不计入上述 358 道合成/手算代码题。数据来源和许可另见 [real_audio/](../ch02/real_audio/README.md)。

四组模型与边界练习对应以下独立实现；主音频新增的 4 个 interpolation 文件验证固定滤波误差，不是完整采样率转换性能。

| 稳定 ID | 实现 | 独立测试 |
|---|---|---|
| E02-08、E04-11、E05-07、E12-05 | [空间模型](cross_chapter/spatial_model_exercises.py) | [空间模型测试](../../../tests/test_codes_spatial_model.py) |
| E07-07、E08-11、E09-09 | [增强结构](cross_chapter/enhancement_structure_exercises.py) | [增强结构测试](../../../tests/test_codes_enhancement_structure.py) |
| E10-16、E10-17、E11-09 | [工程边界](cross_chapter/engineering_boundary_exercises.py) | [工程边界测试](../../../tests/test_codes_engineering_boundaries.py) |
| E13-02 | [插值失真](../appendix_b/interpolation_exercise.py) | [插值测试](../../../tests/test_codes_interpolation.py) |

第10章新增E10-28～33与图62/63，复算限定噪声软更新、有限IR尾部、非抢占阻塞、外部计时尺度、填充RTF和噪声估计失配。[独立六WAV](../ch10/noise_audio/MANIFEST.json)不混入主109；相同谱减的诊断与MCRA限定子链不重复计算完整算法。[当前工业合同](../ch10/reports/industrial_contracts.json)分开CMSIS标量FIR、Speex限定统计、WebRTC路由、RNNoise示例控制流与FastEnhancer包装器；后三项替身不等于运行原分类器或神经模型。该阶段保持308项覆盖与99来源锁；附录B后的当前数量见本页开头。

附录B补充E13-11～14：共享TAC置换与复制、DRR/EDC幅度尺度、时间条件化、同DRR短FIR的五份正式PCM。唯一[response生成源](../appendix_b/examples/generate_response_audio.py)管理五WAV与严格清单；图66及[数值报告](../appendix_b/reports/figure66_equal_drr_response.json)从实际PCM取整数分子/分母。房间21成员由[只读核验](../appendix_b/examples/check_room_assets.py)检查当前源和格式，普通资产核验不等于重新执行PRA。TAC增加唯一一行外部算法，不由四新题增加四行。

2026-10-05 第10章增加一行已知失效选集重建机制，复用原MVDR，不把六WAV或一道练习登记成新算法；当前四工业报告与历史报告分开，完整选集状态不升级为已运行整链。

E14-17/18复用[唯一成像数值核](../ch14/core/imaging.py)，[独立目标与边界测试](../../../tests/test_codes_imaging_objectives.py)核Gram与集体零空间。图80逐网格复算；非负拟合不自动等于真实源恢复。新增两题不重复登记算法行。

[作者原模块限定入口](../ch14/examples/audit_damas_author_contracts.py)和[真实当前报告](../ch14/reports/damas_author_contracts_current.json)保留8数值吻合与1原NameError；21个GPL文本取得不等于MATLAB/MEX/数据实验运行。
