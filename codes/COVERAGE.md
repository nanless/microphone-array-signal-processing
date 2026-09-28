# 算法—源码—验证覆盖表

逐章核实日期：2026-09-28（本轮已复核第1至9章的教学基线、章节映射、练习和相关外部来源；其余条目沿用各自原核实记录）。本表覆盖正文定义、推导或用于选型的方法，以及三篇研究文档中明确说明收录理由的扩展。每行限定具体计算步骤、算法变体或工业机制；同一算法的教学实现与外部对照不重复登记成不同状态。

四种覆盖状态：

- **本仓库可运行基线**：有教学实现、示例与回归测试；只覆盖该行说明的范围，不代表完整论文系统或产品。
- **外部参考实现**：已定位官方或作者实现并登记许可、版本；可能仍缺依赖、模型、数据或硬件，也可能存在已记录的实现疑点。
- **原理索引**：已有原理或来源依据，但尚未形成唯一、许可明确且承担对应计算的源码映射；代码可见而许可不明时也保留此状态，并说明原因。
- **明确排除**：指定软件的身份或许可不满足本书当前收录方式；不表示删除相应方法的学术讨论。

算法表共 292 行：本仓库可运行基线 55 行、外部参考实现 172 行、原理索引 64 行、明确排除 1 行。练习映射单独计数，不因题数增加算法行；MDL、功率谱减、受控 NCC 双讲检测与 cACGMM 教学迭代属于本地基线。覆盖表仍有原理索引，不表示全书全部算法已经运行。

源码取得与入口核对见 [SOURCE_STATUS.json](SOURCE_STATUS.json)；该文件中的依赖验证和执行字段未开展时为 `not_run`，不承载方法级数值实验结果。实际运行及数值对照见[复现记录](research/04_source_reproduction.md)、[增强研究记录](research/02_aec_wpe_separation.md)和 [WPE 独立对照脚本](examples/compare_wpe_reference.py)。工业三库与 SBL 的限定实验保存在 `reports/`；实际调用外部代码不将它改列为本仓库教学基线。覆盖状态不是测试结果。完整提交、官方地址、许可与来源 ID 见 [SOURCES.lock.json](SOURCES.lock.json)。教学路径相对于 [array_tutorial/](array_tutorial/)；外部路径相对于对应项目根，出现“同文件”时仅继承上一行文件，不继承其算法或验证结论。

详细模型、源码阅读与实验设计见 [空间与追踪](research/01_spatial_and_tracking.md)、[AEC、WPE 与分离](research/02_aec_wpe_separation.md)、[工业部署](research/03_industrial_deployment.md)。表中“研究扩展”不表示正文已有完整推导。第 6～8 章已按独立算法和工程任务拆节；下表按当前标题语义指向具体小节，源码取得与数值运行仍分别记录。

2026-09-28 第3章复核登记几何自标定、歧义格点与失配前向模型；外部代码状态详见研究§39。第 2 章复核新增 pyfar 扫频、正则反卷积与 Acoular 幅度校准三项源码映射；其他外部方法的核实范围沿用各自行内记录。

第5章复核新增单参考GSC状态更新、四项外部源码映射与六项原理候选；SOF静态反例、pb_bss原函数提取和PRA原包方法诊断分别记录。球阵两个直接依赖只扩充复现条件，不另计算法。

## 基础模型、几何与统计

| 正文或研究范围 | 算法/机制 | 覆盖状态 | 教学入口或主清单 ID：官方源码入口 | 关键边界 |
|---|---|---|---|---|
| §1.1；空间研究“第 1 章”专题 | 低通 HRIR 互相关时差估计 | 外部参考实现 | `spatial-audio-framework`：`framework/modules/saf_hrir/saf_hrir.c::estimateITDs` | 固定版 750 Hz 低通、整采样峰及限幅；非 Woodworth 真值或听觉阈值 |
| §1.1；空间研究“第 1 章”专题 | HRIR 到 HRTF 与方向插值 | 外部参考实现 | `spatial-audio-framework`：同文件 `HRIRs2HRTFs`、`interpHRTFs` | 已有响应的表示与插值，不从未知录音自动求方向；数据许可另核 |
| §1.1；空间研究“第 1 章”专题 | SOFA响应读取与方向插值接口 | 外部参考实现 | `libmysofa`：`src/hrtf/easy.c`、`src/hrtf/mysofa.h` | 固定v1.3.5源码已取得；重采样、归一化、插值与数据许可分别核对，非定位算法 |
| §2.2、§2.3 | 远场相对时延 | 本仓库可运行基线 | `geometry.py::plane_wave_delays` | 坐标、角度零点、时延正号 |
| §2.3 | 平面波导向矢量 | 本仓库可运行基线 | `geometry.py::plane_wave_steering` | 傅里叶符号、公共相位 |
| §2.2、§2.7 | 球面波导向与距离衰减 | 本仓库可运行基线 | `geometry.py::near_field_steering` | 参考麦归一、零距离、远场退化 |
| §2.4；附录 B 第 16 题 | 镜像法 RIR | 外部参考实现 | `pyroomacoustics`：`pyroomacoustics/room.py`；`examples/room_srp_exercise.py` 已调用锁定 0.10.0 生成六位置 RIR、DRR、T20 外推 T60 与 18 个试听 WAV | 全/直达 RIR 同长度和高通边界；反射阶数与设计 T60 不等于任意实测房间 |
| 研究扩展：空间 §2 | 射线追踪房间模拟 | 外部参考实现 | `pyroomacoustics`：`pyroomacoustics/libroom_src/` | 不自动包含衍射、结构传声 |
| §2.4；空间研究 §2.3 | 指数扫频生成 | 外部参考实现 | `pyfar`：`pyfar/signals/deterministic.py::exponential_sweep_time` | v0.8.1 源码已取得；信号生成不等于已测房间响应 |
| §2.4；空间研究 §2.3 | 带限正则反卷积 | 外部参考实现 | `pyfar`：`pyfar/dsp/dsp.py::deconvolve`、`regularized_spectrum_inversion` | 固定版默认仍走正则分支；未运行设备测量，逆滤波不能修复削波与时变 |
| §2.2；空间研究 §1.4 | 通道标量幅度校准 | 外部参考实现 | `acoular`：`acoular/calib.py::Calib` | 使用给定校准因子；不是自动估计或频率相关复相位校准 |
| §2.5 | STFT 分析 | 本仓库可运行基线 | `spectral.py::stft` | 窗、帧移、补零与轴序 |
| §2.5 | 加权重叠相加 iSTFT | 本仓库可运行基线 | `spectral.py::istft` | 窗乘积包络、输出长度 |
| §2.5 | 批处理空间协方差 | 本仓库可运行基线 | `covariance.py::spatial_covariance` | 共轭、快拍与功率归一 |
| §2.5、§5.4 | 递推空间协方差 | 本仓库可运行基线 | `covariance.py::recursive_covariance` | 遗忘因子、启动与非平稳性 |
| §3.3 | 差分协同阵增广协方差 | 外部参考实现 | `doatools`：`doatools/estimation/coarray.py` | 默认ss为滞后段外积平均，da为直接增广；单位分别是功率平方与功率，有限样本da不保证半正定 |
| §3.3 | 稀疏阵几何优化 | 原理索引 | 正文差分集合与孔径分析 | 协方差重建不是几何优化器 |
| §3.4.1 | 增益—相位自校准 | 原理索引 | 正文交替估计流程 | 规范不唯一、锚点、留出方向 |
| §3.4.2 | 互耦补偿 | 原理索引 | 正文互耦矩阵模型及E03-11 | 区分混合前/后噪声；仅小矩阵正则化演示，不是未知耦合估计器 |
| §3.4.3 | 麦位置自标定 | 外部参考实现 | `structure-from-sound`：`matlab/tdoa/calcresandjac.m`、`bundletdoa.m` | 同步麦、已知声速、足够不同源位；固定版依赖与路径需处理，未运行 |
| §3.2.2 | 指定方向域相位歧义格点判别 | 原理索引 | Tucker等2022与`alias-free-arrays`固定索引 | 作者代码未建立许可，不自动获取；本书独立验证六边形方向对 |
| §3.4 | 已知位置/增益/相位/互耦扰动前向模型 | 外部参考实现 | `doatools`：`doatools/model/perturbations.py` | 施加给定误差，不估计误差；固定版输入校验静态问题见研究§39 |

## 定位与搜索

| 正文或研究范围 | 算法/机制 | 覆盖状态 | 教学入口或主清单 ID：官方源码入口 | 关键边界 |
|---|---|---|---|---|
| §4.1.1 | AIC 源数估计 | 外部参考实现 | `doatools`：`doatools/estimation/source_number.py::aic` | 特征值排序、快拍独立性、白噪声 |
| §4.1.1、§4.9 | MDL 源数估计 | 本仓库可运行基线 | `doa.py::mdl_source_count`；外部对照 `doatools`：`doatools/estimation/source_number.py::mdl` | 复高斯独立快拍、白噪声、正特征值；无自动加载 |
| §4.1.1 | 特征值间隙源数启发式 | 原理索引 | 正文相邻特征值差/比 | 阈值依赖噪声模型与尺度，不是自动保证正确的源数接口 |
| §4.2 | 普通互相关 CC | 原理索引 | 正文 GCC 权重 Ψ=1 与 Knapp–Carter 1976 | 带宽、能量与极性影响峰；教学PHAT接口不冒充所有加权法 |
| §4.2 | Roth 加权互相关 | 原理索引 | 正文互谱除参考通道自谱 | 不对称参考、低能量频点保护需明确 |
| §4.2 | SCOT 加权互相关 | 原理索引 | 正文互谱除两自谱几何均值 | 应先明确谱估计与平均；单帧外积的权重可能退化为PHAT |
| §4.2 | GCC-PHAT 与物理 lag 裁剪 | 本仓库可运行基线 | `doa.py::gcc_phat` | 静音、麦序、带宽与多峰；互谱阈值相对本麦对峰值 |
| §4.2 | GCC 峰三点亚采样插值 | 本仓库可运行基线 | `doa.py::gcc_phat` 插值选项 | 不能创造窄带缺少的信息 |
| §4.3；附录 B 第 16 题 | 远场 SRP-PHAT | 本仓库可运行基线 | `doa.py::srp_phat`；`examples/room_srp_exercise.py` 在六个固定房间输入上执行 | 逐帧PHAT后平均、等权正频点直接相位求和，非完整irfft查表；全零或仅直流输入拒绝给出方向；六点结果不代表跨房间失效率 |
| §4.7.1；研究扩展：空间 §8 | 近场三维 SRP | 原理索引 | 正文球面传播与角度—距离网格；空间研究 §8 的版本边界 | 锁定 pyroomacoustics 0.10.0 未将 `mode/r` 接入有效导向和距离网格，不能作为该变体实现 |
| §4.3；研究扩展：空间 §9 | 分层 SRP 搜索 | 外部参考实现 | `odas`：`src/module/mod_ssl.c`、`src/signal/scan.c` | 粗层丢峰不能由细层恢复 |
| 研究扩展：空间 §9 | 方向性麦对筛选 | 外部参考实现 | `odas`：`src/signal/spatialindex.c` 与配置 | 设备指向性及有效麦对 |
| 研究扩展：空间 §10 | SVD-PHAT | 原理索引 | 原论文与低秩实验设计 | 未确认唯一且许可明确的作者实现 |
| 研究扩展：空间 §10 | 多源 SVD-PHAT | 原理索引 | 多源原论文 | 普通 SVD 库不实现逐次投影规则 |
| §4.4 | Chan 双曲定位 | 原理索引 | Chan–Ho 1994；正文距离差变量与两阶段思想 | E04-16只实现GN一步，不是完整Chan加权与约束处理 |
| §4.4 | TDOA 加权非线性最小二乘 | 原理索引 | 正文高斯—牛顿推导 | 共享参考误差相关、多解、雅可比 |
| §4.5 | Bartlett 空间谱 | 本仓库可运行基线 | `doa.py::bartlett_spectrum` | 导向与输出功率归一 |
| §4.5 | Capon 空间谱 | 本仓库可运行基线 | `doa.py::capon_spectrum` | 加载、秩亏、失配 |
| §4.6 | MUSIC | 本仓库可运行基线 | `doa.py::music_spectrum` | 源数、噪声子空间、选峰；非半正定及零协方差拒绝 |
| §4.6 | NormMUSIC | 外部参考实现 | `pyroomacoustics`：`pyroomacoustics/doa/normmusic.py` | 低可靠频点可能被过度加权 |
| §4.6 | 前向空间平滑 | 外部参考实现 | `doatools`：`doatools/estimation/preprocessing.py::spatial_smooth` | 参数 `l` 是子阵数 |
| §4.6 | 前后向空间平滑 | 外部参考实现 | `doatools`：同函数 `fb` 选项 | 对称与平移模型、孔径损失 |
| §4.6 | root-MUSIC | 外部参考实现 | `doatools`：`doatools/estimation/music.py::RootMUSIC1D` | ULA、根选择、半波距与 NumPy 相容性 |
| §4.6 | LS-ESPRIT | 本仓库可运行基线 | `doa.py::esprit_ula` | 共同全阵基截取、非零PSD与子阵秩；主值角不证明无空间混叠 |
| §4.6；研究扩展：空间 §14 | TLS-ESPRIT | 外部参考实现 | `doatools`：`doatools/estimation/esprit.py` | 双侧误差模型不同于LS；固定版默认行加权共享view错误已实际复现，none/独立复制对照见报告 |
| §4.6.1 | CSSM | 外部参考实现 | `pyroomacoustics`：`pyroomacoustics/doa/cssm.py` | 已登记剔除频点后协方差下标疑点 |
| §4.6.1 | WAVES | 外部参考实现 | `pyroomacoustics`：`pyroomacoustics/doa/waves.py` | 已登记剔除频点后协方差下标疑点 |
| §4.6.1 | TOPS | 外部参考实现 | `pyroomacoustics`：`pyroomacoustics/doa/tops.py` | 已登记真实 bin/列表下标疑点 |
| §4.7 | FRIDA | 外部参考实现 | `frida-original`：`doa/fri.py`；`pyroomacoustics`：`pyroomacoustics/doa/frida.py` | 二维远场、不相关源、平面阵列方位；visibility与raw模型分别说明，固定接口不支持三维；原实验旧环境 |
| §4.7 | L1-SVD 定位 | 外部参考实现 | `doatools`：`estimation/preprocessing.py::l1_svd`与`estimation/sparse.py::GroupSparseEstimator`显式组合 | 压缩观测M×K，求解器快拍数设K；压缩本身不是定位，未运行完整链 |
| §4.7 | 组稀疏多快拍定位 | 外部参考实现 | `doatools`：`doatools/estimation/sparse.py::GroupSparseEstimator` | 离格误差、字典尺度、求解器 |
| §4.7 | 稀疏协方差匹配 | 外部参考实现 | `doatools`：同文件 `SparseCovarianceMatching` | 不相关源功率模型 |
| §4.7；空间 §19 | 稀疏贝叶斯定位 SBL | 外部参考实现 | `sbl`：`SBL_MF_Python/sbl.py::SBL` | 多频/多快拍，已知源数；网格、剪枝与噪声模型 |
| 研究扩展：空间 §19 | RobustSBL | 外部参考实现 | `robustsbl`：`_common/SBL_v5p12.m` | Gauss/t/Huber/Tyler；MATLAB 统计函数、异常快拍模型 |
| 研究扩展：空间 §10 | SMP-PHAT | 外部参考实现 | `smpphat`：`src/system.c::smp_call`；本书隔离两行适配实验 | 等长平行基线与远场；非 SVD-PHAT；固定原版在本机 clang/arm64 数值失败，临时适配版仅通过两组合成四麦对照 |
| §4.7 | 原子范数定位 | 原理索引 | 正文连续参数模型 | 通用 SDP 求解器不是完整实现 |
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
| §5.2 | DSB 权重 | 本仓库可运行基线 | `beamforming.py::dsb_weights`；`examples/beamformer_common_input_demo.py` | 不含设备分数延时 FIR；同输入对照只在单频解析模型 |
| §5.2 | 等间距线阵 Dolph–Chebyshev 加权 | 原理索引 | 原论文与 SciPy `chebwin` 官方定义；SciPy源码未单独锁定 | 对称正横阵列与首零点宽度条件，不泛称任意阵列最窄HPBW；窗口峰归一另转为目标响应归一 |
| §5.3 | 弥散场相干模型 | 本仓库可运行基线 | `beamforming.py::diffuse_coherence` | 各向同性假设 |
| §5.3 | 超指向波束 | 本仓库可运行基线 | `beamforming.py::superdirective_weights` | 低频 WNG 与麦误差 |
| §5.3 专栏 | 差分麦克风阵 DMA | 原理索引 | 正文一阶算例与高阶模型 | 超指向接口不覆盖全部 DMA |
| §5.4 | MVDR 与相对对角加载 | 本仓库可运行基线 | `beamforming.py::mvdr_weights`；`examples/beamformer_common_input_demo.py` | 加载按平均特征值缩放 |
| §5.4.1 | 最坏情形稳健波束 | 原理索引 | 正文误差集模型 | 经验加载不等于明确误差集优化 |
| §5.4.1 | 显式 WNG 约束设计 | 原理索引 | 正文约束与选型 | WNG 计算不等于约束优化器 |
| §5.4.1 | 特征空间波束 | 原理索引 | Chang–Yeh 1992；空间研究 §22 | 子空间维数/高SNR/导向投影；未取得作者完整实现 |
| §5.4.1 | 干扰加噪声协方差重构 | 原理索引 | Gu–Leshem 2012；空间研究 §22 | 排除目标角域与流形假设；Capon函数不是完整重构算法 |
| §5.5 | LCMV 闭式权重 | 本仓库可运行基线 | `beamforming.py::lcmv_weights`；`examples/beamformer_common_input_demo.py` | 约束独立性、残差 |
| §5.5 | Frost 投影自适应 | 原理索引 | 正文时域投影更新；E05-10实际计算一步 | 一步约束核对不是完整在线Frost实现 |
| §5.6 | GSC 阻塞矩阵 | 本仓库可运行基线 | `beamforming.py::blocking_matrix` | 不含持续自适应抵消支路 |
| §5.6；E05-16 | 单参考复数 GSC NLMS 状态更新 | 本仓库可运行基线 | `gsc.py::ScalarGSCNLMS`；`examples/chapter05_experiments.py` | 给定固定/阻塞输出，跨块、冻结和复位；不含方向估计、自动SPP或多抽头滤波器组 |
| §5.6；空间 §21 | 完整在线自适应 GSC | 外部参考实现 | `btk20`：`btk20_src/lib/pybeamformer.py` | 子带 LMS/RLS；毫米坐标、旧依赖、泄漏与冻结；非时域 Frost |
| §5.7.2 | 标量 Wiener 增益 | 本仓库可运行基线 | `beamforming.py::wiener_gain` | 不是完整噪声估计器 |
| §5.7；空间 §21 | Zelinski 互谱后滤 | 外部参考实现 | `btk20`：`btk20_src/postfilter/postfilter.cc` | REAL/ABS分支、递归平滑与增益下限分开；同谱非负性不等于无偏；未构建/运行 |
| §5.7；空间 §21 | McCowan 弥散相干后滤 | 外部参考实现 | `btk20`：同文件 McCowan 入口 | 固定复相干实现不等于原文实部式；Γ近1病态；未构建/运行 |
| §5.7.1 | MCRA | 原理索引 | 正文与空间研究的作者软件入口 | 未获得可核版本/许可包 |
| §5.7.1 | IMCRA | 原理索引 | 正文与空间研究的作者软件入口 | 不把普通最小值跟踪称 IMCRA |
| §5.7.2 | OM-LSA | 原理索引 | 正文及作者方法说明 | 不以 Wiener 或其他 MMSE 增益代替 |
| §5.7.2；研究扩展：空间后滤 | 全秩 SDW-MWF | 外部参考实现 | `espnet`：`espnet2/enh/layers/beamformer.py::get_sdw_mwf_vector` | 失真权重与参考通道 |
| 研究扩展：空间后滤 | 秩一迹化简 WMWF | 外部参考实现 | `pb_bss`：`pb_bss/extraction/beamformer.py::get_wmwf_vector` | 不代表一般全秩 MWF |
| §5.9 | GEV 波束权重 | 外部参考实现 | `pb_bss`：同文件 `get_gev_vector` | 广义特征向量尺度不确定 |
| §5.9 | BAN 缩放 | 外部参考实现 | `pb_bss`：同文件 `blind_analytic_normalization` | pb_bss返回权重、ESPnet返回增益且含C²差异；不保证无失真 |
| §5.9；空间 §48 | 秩一目标 SCM 的 PCA-RTF 方向 | 外部参考实现 | `pb_bss`：`pb_bss/extraction/beamformer.py::get_pca_vector` | 特征向量还需选参考/尺度；全秩目标不能直接认作唯一RTF |
| §5.9；空间 §48 | Souden 参考通道 MVDR | 外部参考实现 | `pb_bss`：同文件 `get_mvdr_vector_souden` | 秩一条件与参考选择分别核对；MERL入口固定版提取调用失败另记 |
| §5.9 | RTF 幂迭代估计 | 外部参考实现 | `espnet`：`espnet2/enh/layers/beamformer.py::get_rtf` | 函数本身不完成参考通道归一 |
| §5.9、§8.4 | 掩码空间协方差 | 本仓库可运行基线 | `separation.py::masked_spatial_covariance` | 轴序、空掩码、保留原始功率 |
| §5.9、§8.4 | 两通道掩码 MVDR | 本仓库可运行基线 | `separation.py::mask_mvdr_2x2` | 限定 2×2；非零掩码公共缩放不应改变结果；不是完整神经系统 |
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
| 研究扩展：空间 §28 | DAMAS | 外部参考实现 | `acoular`：`acoular/fbeamform.py::BeamformerDamas` | 输出功率图，依赖 PSF |
| 研究扩展：空间 §29 | CLEAN-SC | 外部参考实现 | `acoular`：同文件 `BeamformerCleansc` | 减去量、停止与弱源保留 |
| 研究扩展：空间 §30 | CMF | 外部参考实现 | `acoular`：同文件 `BeamformerCMF` | 约束、缩放与残差 |
| 研究扩展：空间 §30 | SODIX | 外部参考实现 | `acoular`：同文件 `BeamformerSODIX` | 源强和指向性可辨识性 |
| 研究扩展：空间 §30 | 移动源时域声学成像 | 外部参考实现 | `acoular`：`acoular/tbeamform.py` | 轨迹/传播真值，不直接输出增强语音 |
| 研究扩展：空间近年候选 | ASA 注意力空间协方差聚合（2024） | 原理索引 | Tammen 等 Interspeech 2024 原论文 | 注意力帧权重与通道不变性；未核实作者代码，不冒充已复现 |
| 研究扩展：空间近年候选 | iDeepPE 参数估计与后滤融合（2025） | 原理索引 | `ideeppe`固定索引：`evaluate.py` | 未建立代码许可，不获取；默认非因果噪声估计，oracle训练准备不等于推理 |
| 研究扩展：空间近年候选 | 可学习 WNG 阈值的稳健波束（2026） | 原理索引 | arXiv:2606.24137v1 预印本 | 已知远场方向、双头mask/阈值；未取得作者代码和完整训练复现 |

## 回声消除与自适应控制

| 正文或研究范围 | 算法/机制 | 覆盖状态 | 教学入口或主清单 ID：官方源码入口 | 关键边界 |
|---|---|---|---|---|
| §6.1.2 | LMS | 原理索引 | 正文梯度更新 | 教学 NLMS 不是固定步长 LMS |
| §6.1.2 | NLMS | 本仓库可运行基线 | `aec.py::nlms`；`examples/aec_same_input_truth.py` 真值冻结对照 | 外部冻结掩码、零参考、有限 FIR；合成真值输入不等于设备性能 |
| §6.15 | 流式 NLMS 状态 | 本仓库可运行基线 | `aec.py::NLMSState`、`examples/aec_streaming_demo.py` | 跨块同时续接抽头与 $L-1$ 个参考历史；不含 DTD、延迟搜索或实时接口 |
| 研究扩展：增强 A01 | 泄漏 NLMS | 原理索引 | 泄漏更新说明 | 教学函数没有泄漏参数 |
| §6.2 | 单块 FDAF | 原理索引 | 正文频域卷积与约束 | 分区 MDF 不覆盖所有单块变体 |
| §6.2 | MDF/PBFDAF 分区结构（同配置频域多抽头 NLMS） | 本仓库可运行基线 | `aec_partitioned.py::PartitionedFDAFState`、`examples/aec_partitioned_demo.py`（手算、频点反例与冻结宽带留出实验）、`examples/aec_pbfdaf_real_pair_compare.py`（两对固定真实配对录音与三种参考控制）；外部 Speex 对照另见 AUMDF 行 | 瞬时功率式(6-5)、有效区、可选梯度约束与跨块状态；真实录音结果含负值，只是总功率变化；不含自动 DTD、延迟搜索或产品级控制；同一结构不重复算算法 |
| §6.2；研究扩展：增强 A03 | AUMDF 约束调度 | 外部参考实现 | `speexdsp`：同文件；真实配对与已知近端注入实验见 `examples/aec_real_pair_experiment.py`、`examples/aec_doubletalk_experiment.py`、`examples/aec_controlled_doubletalk.py` | 约束调度不同于梯度更新；真实双讲没有分量真值，半合成输出增量也不是近端保留率 |
| §6.2 | PNLMS | 原理索引 | 正文比例更新 | Speex 控制不代表全部变体 |
| §6.2 后续 IPNLMS 专题 | IPNLMS | 本仓库可运行基线 | `aec_ipnlms.py::IPNLMSState`、`examples/aec_ipnlms_subband_demo.py`、`examples/aec_advanced_exercises.py`；`examples/aec_algorithm_minicases.py` 保留旧一步小例 | 实数单参考、先验残差、跨块状态；强抽头比例份额不保证稀疏/稠密任意条件下更快；外部给冻结掩码 |
| §6.2 后续子带专题 | 两带 Haar 对角子带 NLMS 教学近似 | 本仓库可运行基线 | `aec_subband.py::HaarDiagonalSubbandNLMSState`、`examples/aec_ipnlms_subband_demo.py`、`examples/aec_advanced_exercises.py` | 只读取同带历史；不是任意回声路径的精确模型或完整 NSAF；两点 Haar 不代表工业滤波器组 |
| §6.2 后续子带专题 | 已知两抽头路径的精确交叉带映射 | 本仓库可运行基线 | `aec_subband.py::two_tap_crossband_matrices`、`two_tap_subband_outputs`；时域卷积独立测试 | 固定已知路径的代数对照，不是交叉带自适应器；完美重建不保证对角辨识 |
| §6.2 后续子带专题 | 两带 Haar 全交叉自适应块 NLMS | 本仓库可运行基线 | `aec_subband.py::HaarCrossbandNLMSState`、`fir_crossband_matrices`；`examples/aec_crossband_demo.py`；`tests/test_codes_aec_crossband.py` | 任意有限实数时域 FIR 在该两带块模型内可表示；每块四路 FIR、外部冻结、成块等待；非工业滤波器组、非 Lee–Gan NSAF |
| §6.2 后续子带专题 | 一般子带自适应 AEC/NSAF | 原理索引 | 正文模型与 Lee–Gan 原论文；固定版 `pyroomacoustics` 可查 `adaptive/subband_lms.py`（尚未单独登记锁清单入口） | 原型滤波器、抽取/混叠、跨带耦合和延迟依实现而变；NSAF 的全带抽头更新不同于本书两带逐输出带 FIR |
| §6.2 后续 RLS 专题 | 常规实数时域 RLS | 本仓库可运行基线 | `aec_rls.py::RLSState`、`examples/aec_rls_demo.py`、`examples/aec_rls_kalman_comparison.py`、`test_codes_aec_rls.py`；A04 另列 pyroomacoustics、pyaec、MetaAF 源码 | 含初始约束的指数加权最小二乘；逆相关矩阵 $O(L^2)$，教学代码每样本 Cholesky 正定检查另需 $O(L^3)$；无自动 DTD、延迟搜索或 RES |
| §6.2；研究扩展：增强 A04 | 快/块/广义频域 RLS | 外部参考实现 | `pyroomacoustics`：`adaptive/rls.py` 的 BlockRLS；`metaaf`：`optimizer_rls.py`；GFDAF 论文另见 A04 | BlockRLS 和 GFDAF 不是本书逐样本精确 RLS 的改名；MetaAF 核心与 AEC zoo 许可不同 |
| §6.2 | 短实数 FIR 矩阵 Kalman | 本仓库可运行基线 | `aec_kalman_matrix.py::KalmanAECState`、`examples/aec_kalman_matrix_demo.py`、`test_codes_aec_kalman_matrix.py` | 已知 $Q,\Psi$ 的 Joseph 协方差更新，未估计噪声或实现频域分区 |
| §6.2 | FDKF | 外部参考实现 | 增强 A04；`pyaec`：`frequency_domain_adaptive_filters/fdkf.py::fdkf`；`examples/audit_aec_upstream_interfaces.py` | 固定原函数在 NumPy 2.5.3 因 `np.complex` 失败，未输出音频；标量教学递推式(6-17)不是完整 FDKF；AEC3/Speex 不自动归为卡尔曼 |
| §6.2 | 分区 FDKF 与 VD/SD-PBFDKF | 外部参考实现 | 增强 A04；`pyaec`、`echocatzh-pfdkf`、`subband-kalman-aec` 固定源码；外部接口报告 | pyaec 在 `np.complex` 失败；echocatzh 原接口已实调默认后滤与关闭后滤的不同取点；非同条件整链性能对照，后者仅 `.m`、README、许可 |
| §6.3 | Volterra 非线性 AEC | 原理索引 | 正文路径模型；`pyaec/nonlinear_adaptive_filters/volterra.py::svf` 可查受限二因子二阶级联 | 未执行该函数；受限级联不等于完整 Volterra 核；阶数、过拟合及未见削波另验 |
| §6.3 | Hammerstein 非线性路径 | 原理索引 | 正文级联模型 | 非线性位于线性动态系统之前 |
| §6.3 | Wiener 非线性路径 | 原理索引 | 正文级联模型 | 非线性位于线性动态系统之后 |
| §6.4 | Geigel DTD | 原理索引 | 正文判决模型；`examples/aec_algorithm_minicases.py` 反例 | 多径误判、零参考禁判；小例不是产品检测器 |
| §6.4 | 相关/相干性 DTD | 本仓库可运行基线 | `double_talk.py::ncc_activity_states`、`examples/aec_dtd_demo.py` 的四状态计数及路径突变反例 | 已对齐单参考、固定阈值、数学合成 PCM；真值只评分，远端静音禁作双讲判决；真实设备未测 |
| §6.4 | 连续可变学习率 | 外部参考实现 | `speexdsp`：`libspeexdsp/mdf.c` | 不是独立二值 DTD |
| §6.8 | AEC3 参考延迟控制 | 外部参考实现 | `webrtc`：`modules/audio_processing/aec3/echo_path_delay_estimator.cc`、`render_delay_controller.cc` | 降采样匹配滤波估滞后；与 refined/coarse 线性滤波器分工不同 |
| §6.8 | AEC3 线性抵消 | 外部参考实现 | `webrtc`：`modules/audio_processing/aec3/subtractor.cc`；官方 WAV 入口的已运行适配器 `examples/aec3_offline_compare.py`、`examples/aec_same_input_truth.py` | 固定提交的 `audioproc_f` 已在两对真实录音和一组已知合成分量上分别导出线性/最终 WAV；比较须补偿输出固定延迟并区分取点，仍无设备性能排名，见研究手册 A06 |
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
| 研究扩展：增强 A13 | Meta-AF 核心更新器与 AEC 配方 | 外部参考实现 | `metaaf`：`metaaf/filter.py`、`metaaf/core.py`、`metaaf/meta.py`、`zoo/aec/` | 核心 NCSA；AEC选集保留Adobe非商用研究教学许可；已取得源码，未取权重或运行网络 |
| §6.7 | ERLE 与有效单讲区间 | 本仓库可运行基线 | `aec.py::erle_db` | 双讲排除、收敛段、固定延迟；正功率地板使完美抵消读数有限 |

## WPE、盲分离与空间混合

| 正文或研究范围 | 算法/机制 | 覆盖状态 | 教学入口或主清单 ID：官方源码入口 | 关键边界 |
|---|---|---|---|---|
| §7.1；增强研究 W07 | 多通道逆定理（MINT）精确逆滤波 | 原理索引 | Miyoshi/Kaneda 1988 原文 §III–IV；`examples/chapter07_experiments.py` E07-14 为本书两路径解析例 | 已知单源响应、互素等前提；小例不等于一般未知房间的 MINT 求解器 |
| §7.1；增强研究 W07 | 实倒谱域加窗（liftering） | 原理索引 | Oppenheim/Schafer/Stockham 1968；正文对数功率谱约定（实倒谱的两倍） | 需要对数零点处理；实倒谱丢失相位，非无条件可逆去混响器；未锁定通用作者实现 |
| §7.1；增强研究 W07 | 晚期混响谱方差估计与抑制 | 原理索引 | Habets 等 2009 原文模型与边界 | 统计衰减与混合时间假设；不把 SpeexDSP 的配置开关误作已实现的谱方差抑制器 |
| §7.10；增强研究 W09 | MetaAF 学习 WPE 滤波器更新 | 外部参考实现 | `metaaf`：`zoo/wpe/wpe.py`、`zoo/wpe/wpe_eval.py`；核心优化器在 `metaaf/` | 固定源码选集已取得，zoo 采用独立非商业许可；未训练、未载入模型、未复现质量指标 |
| §7.10；增强研究 W09 | VACE-WPE 虚拟通道估计与任务特定前端 | 外部参考实现 | `tso_vace_wpe`：`vace_wpe.py`、`torch_custom/wpe_th_utils.py`、`torch_custom/neural_wpe.py` | MIT 固定源码与直接依赖选集；未取模型和语料，未训练/推理或核验 ASV 性能 |
| §7.2 | 离线单/多通道 WPE | 本仓库可运行基线 | `dereverberation.py::offline_wpe`；外部对照 `nara_wpe` | 保护延迟、有效历史、相对加载；外部 BTK20 的通道功率与加载不同，见研究 W08；数值对照见 W01/W11 |
| §7.4 | 逐帧在线 WPE | 外部参考实现 | `nara_wpe`：`nara_wpe/wpe.py::OnlineWPE` | 三接口 delay 索引不同；独立对照 `examples/compare_online_wpe_reference.py`；未来扰动和跨块状态见 `examples/wpe_temporal_contract.py`；长静音状态溢出及恢复见 `examples/wpe_silence_boundary.py` |
| §7.4；研究扩展：增强 W03 | 块在线/递推 WPE | 外部参考实现 | `nara_wpe`：`nara_wpe/tf_wpe.py` | TF 固定版权重 .7 乘当前块；块内估计后处理本块，非逐帧因果；静态核对见 W03 |
| §7.10 | DNN-WPE | 外部参考实现 | `espnet`：`espnet2/enh/layers/dnn_wpe.py` | 配置值须追到实际求解器；固定版部分成员未传入，填充帧统计亦需核对；静态结果见 W04/W11 |
| §7.10 | WPD | 外部参考实现 | `espnet`：`espnet2/enh/layers/beamformer.py` WPD 分支 | 当前约束、延迟历史与功率 |
| 研究扩展：增强 W06 | AR-FastMNMF | 原理索引 | `fastmnmf_author`：`src/` 受限来源索引 | 学术研究限制，未纳入通用源码集合 |
| §8.5 | SI-SDR | 本仓库可运行基线 | `separation.py::si_sdr` | 零均值、零能量、参考目标 |
| §8.5 | 小源数 PIT 排列枚举 | 本仓库可运行基线 | `separation.py::pit_permutation` | 阶乘成本，不是长期身份关联 |
| §8.3；研究扩展：增强 B01 | FDICA | 外部参考实现 | `ssspy`：`ssspy/bss/fdica.py` | 跨频排列与尺度恢复 |
| §8.3 | AuxIVA 迭代投影 | 外部参考实现 | `ssspy`：`ssspy/bss/iva.py`；`examples/reproduce_auxiva_reference.py` 调用固定源码 | 已跑一组数学合成盲估计及谐波反例；不等于语音论文复现或本书教学实现 |
| 研究扩展：增强 B02 | IVA 迭代源导向 ISS | 外部参考实现 | `ssspy`：同文件 ISS 选项 | 与 IP 不同，固定 ISS1/ISS2 |
| 研究扩展：增强 B02 | projection-back | 外部参考实现 | `ssspy`：`ssspy/algorithm/` | SI-SDR 通过不能证明尺度恢复 |
| 研究扩展：增强 B03 | OverIVA | 外部参考实现 | `piva`：`piva/auxiva.py` | 过定源数、背景模型；GPL |
| 研究扩展：增强 B03 | FIVE | 外部参考实现 | `piva`：`piva/five.py` | 单目标提取不等于全部分离 |
| §8.3 | ILRMA | 外部参考实现 | `ssspy`：`ssspy/bss/ilrma.py` | NMF 基数与局部最优 |
| §8.3 | 满秩 MNMF | 外部参考实现 | `ssspy`：`ssspy/bss/mnmf.py::GaussMNMF` | 欠定模型不保证可辨识 |
| §8.3；研究扩展：增强 B06 | FastMNMF | 外部参考实现 | `pyroomacoustics`：`pyroomacoustics/bss/fastmnmf.py` | 与受限作者整库许可分开 |
| 研究扩展：增强 B06 | FastMNMF2 | 外部参考实现 | `pyroomacoustics`：`pyroomacoustics/bss/fastmnmf2.py` | 参数化、参考麦源图像 |
| 研究扩展：增强 B11 | 多帧满秩空间协方差分析 mfFCA | 外部参考实现 | `mffca`：`fca.py`、`main_synthetic.ipynb` | 固定作者源码仅内部评估，不修改或再分发；未运行整链 |
| §8.3 | TRINICON | 外部参考实现 | `pyroomacoustics`：`pyroomacoustics/bss/trinicon.py` | 该实现固定两个输出 |
| §8.4 | cACGMM | 本仓库可运行基线 | `gss_teaching.py` 的受控 NumPy 活动导引迭代；`pb_bss`：`pb_bss/distribution/cacgmm.py` 为外部对照入口 | 二麦合成复谱、给定活动；方向外积不保留原功率，未复现官方 GPU/CHiME 结果 |
| §8.4 | GSS 活动约束分离 | 外部参考实现 | `gss`：`gss/core/enhancer.py` 为官方整链；本书 `examples/gss_teaching_demo.py` 已运行 cACGMM→SCM→MVDR 受控子链 | 本书无混响输入中 WPE 旁路；官方 CuPy/Lhotse、RTTM 与 CHiME 语料未运行，不能外推为完整官方复现 |
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
| §8.6；E08-23 | 双路 CSS 重叠相关排列关联 | 本仓库可运行基线 | `css.py::match_two_source_overlap` | 仅两槽实波形的重叠关联，不是分离器；静音、弱相关与并列返回不确定 |
| §8.6；研究扩展：增强 N09 | 连续语音分离 CSS | 外部参考实现 | `notsofar1`：`css/css.py`、`css/css_with_conformer/separate.py` | 窗口、跨块排列、槽位、配置；不执行资产下载 |
| §13.3；研究扩展：增强 N10 | SGMSE+ | 外部参考实现 | `sgmse`：`enhancement.py`、`sgmse/model.py` | 采样器、权重和语料另核 |
| §13.3；研究扩展：增强 N10 | StoRM | 外部参考实现 | `storm`：`enhancement.py`、`sgmse/model.py` | 再生成不等于 SGMSE 推理流程 |
| §13.3；研究扩展：增强 N11 | ArrayDPS | 外部参考实现 | `arraydps`：`separate.py`、`src/sampler.py` | 传播模型、先验、WSJ 录音权利 |
| §8.6；研究扩展：增强 N12 | AudioSep | 外部参考实现 | `audiosep`：`pipeline.py` | 默认单声道 32 kHz，不是注册语音 TSE |

## 追踪、关联与控制

| 正文或研究范围 | 算法/机制 | 覆盖状态 | 教学入口或主清单 ID：官方源码入口 | 关键边界 |
|---|---|---|---|---|
| §9.2 | 角度—角速度 Kalman | 本仓库可运行基线 | `tracking.py::ConstantVelocityKalman` | 最短角差、Joseph 更新、Q 的离散化 |
| §9.2 | EKF | 外部参考实现 | `filterpy`：`filterpy/kalman/EKF.py` | 观测雅可比、角度残差；固定版组合接口线性化点与分步调用不同，见追踪接口报告 |
| §9.2 | UKF | 外部参考实现 | `filterpy`：`filterpy/kalman/UKF.py` | sigma 点、圆周均值 |
| §9.2；研究扩展：空间 §40 | IMM | 外部参考实现 | `filterpy`：`filterpy/kalman/IMM.py` | 同维同义状态与模式转移；固定版缺测/模式先验实际诊断见追踪接口报告 |
| §9.2.4 | 圆周 SIR 粒子滤波 | 本仓库可运行基线 | `tracking.py::CircularParticleFilter` | 对数权重、多峰均值无定义 |
| §9.2.4 | 系统重采样 | 本仓库可运行基线 | `tracking.py::systematic_resample` | 权重归一、随机种子 |
| §9.3；研究扩展：空间 §33 | 最近邻关联 | 外部参考实现 | `stonesoup`：`stonesoup/dataassociator/neighbour.py` | 单轨最近不等于全局最优 |
| §9.3；研究扩展：空间 §33 | GNN 全局关联 | 外部参考实现 | `stonesoup`：同文件；[本书两轨教学缩例](examples/tracking_crossing_dropout_demo.py) | 全局分配、一对一与门控；两轨缩例不代表完整关联器 |
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
| §9.4 | 轨迹到波束预测/限速 | 原理索引 | 正文控制接口；[缺测与限速缩例](examples/tracking_crossing_dropout_demo.py) | 观测龄期、失效与最大角速度；缩例未接真实设备 |

| §9.2；空间研究 §31 | 固定增益 α-β / α-β-γ | 外部参考实现 | `filterpy`：`filterpy/gh/gh_filter.py` | 固定增益与采样间隔共同定义动态；不是自动估计协方差的Kalman |
| §9.1；空间研究 §31 | 滑动平均、中值与众数基线 | 原理索引 | 正文比较三种统计量 | 平均线性，中值/众数非线性；众数依赖离散化，圆周边界另处理 |
| §9.2；空间研究 §32 | APF 辅助粒子滤波 | 原理索引 | Pitt–Shephard观测引导祖先选择与重要性校正 | 改进提议分布不自动提供异常值鲁棒性；不可把SIR改名为APF |
| §9.3；空间研究 §36 | GOSPA | 外部参考实现 | `stonesoup`：`stonesoup/metricgenerator/ospametric.py::GOSPAMetric` | p≥1，α=2分解定位/漏检/虚警；单帧位置指标不证明身份连续 |
| 研究扩展：空间 §55.2 | RBMCDA 条件解析多目标追踪 | 外部参考实现 | `spatial-audio-framework`：`framework/modules/saf_tracker/saf_tracker.c`、`saf_tracker_internal.c` | 关联粒子与条件Kalman分开；GPL-2.0-or-later模块，静态核对未执行 |

## 连续音频、噪声控制与部署

| 正文或研究范围 | 算法/机制 | 覆盖状态 | 教学入口或主清单 ID：官方源码入口 | 关键边界 |
|---|---|---|---|---|
| §10.1.2 | 环形缓冲与预卷 | 本仓库可运行基线 | `engineering.py::RingBuffer` | Python 教学容器不是无锁硬实时队列 |
| §10.1.1 | 截止期限/队列模拟 | 本仓库可运行基线 | `engineering.py::simulate_deadline_queue` | 等待、计算与排队分别计量 |
| §10.1.2 | PortAudio 回调/时间戳 | 外部参考实现 | `portaudio`：`include/portaudio.h`、`examples/` | 不阻塞、不分配；ADC/DAC 时钟核对 |
| §10.1.2；工业 I02 | ALSA PCM 状态恢复 | 外部参考实现 | `alsa-lib`：`src/pcm/pcm.c`、`test/pcm.c` | xrun、挂起、移除分别处理 |
| §10.1.2；工业 I03 | PipeWire AEC 四流路由 | 外部参考实现 | `pipewire`：`src/modules/module-echo-cancel.c` | 播放参考完整，避免无意双重 AEC |
| §10.2.1 | SRO 线性拟合 | 本仓库可运行基线 | `engineering.py::estimate_sro_ppm`；`examples/sro_closed_loop_demo.py` | 初始时差与斜率分开；示例用精确合成时间戳估计 |
| §10.2.1 | 线性插值 SRO 补偿 | 本仓库可运行基线 | `engineering.py::resample_sro_to_reference`；`examples/sro_closed_loop_demo.py` | 示例含跨块状态和单点丢样标记；不含抗混叠与真实异步控制 |
| §10.2.1 | libsamplerate 连续 SRC | 外部参考实现 | `libsamplerate`：`src/samplerate.c` | 速率比、消耗量与状态 |
| §10.2.1 | SpeexDSP 连续 SRC | 外部参考实现 | `speexdsp`：`libspeexdsp/resample.c` | 质量档、通道一致性 |
| §10.3.2 | 功率谱减 | 本仓库可运行基线 | `noise_suppression.py::power_spectral_subtraction`；[E10-13 与音频演示](examples/spectral_subtraction_demo.py) | 固定噪声专用前奏、单通道 STFT；无在线噪声跟踪与语音概率，音乐噪声仍可能出现 |
| §10.3.1 | MMSE-STSA | 原理索引 | 正文谱幅度目标 | 不等于 Speex 修改后的响度域增益 |
| §10.3.1 | MMSE-LSA | 原理索引 | 正文对数谱幅度目标 | 注释可选式不是默认完整实现 |
| §10.3.1；研究扩展：空间 §53 | WebRTC 分位数噪声估计 | 外部参考实现 | `webrtc`：`modules/audio_processing/ns/quantile_noise_estimator.cc` | 不自动命名为 MCRA/IMCRA |
| §10.3.3；研究扩展：空间 §53 | WebRTC NS 语音概率估计 | 外部参考实现 | `webrtc`：`modules/audio_processing/ns/speech_probability_estimator.cc` | 不等同传统 VAD API |
| §10.3.1 | RNNoise | 外部参考实现 | `rnnoise`：`src/denoise.c` | DSP/神经状态、训练域和模型 |
| §10.3.1；工业 I09 | DeepFilterNet 深度滤波 | 外部参考实现 | `deepfilternet`：`DeepFilterNet/df/`、`libDF/src/` | 48 kHz、阶数、前瞻与延迟 |
| 研究扩展：工业 I09 | DeepFilterNet LADSPA | 外部参考实现 | `deepfilternet`：`ladspa/` | 无前瞻仍有 STFT/宿主延迟 |
| §10.3.4 | 能量迟滞/挂起 VAD | 本仓库可运行基线 | `engineering.py::HysteresisVAD` | 能量门限，不是神经概率 |
| §10.3.4 | 峰值保护 AGC | 本仓库可运行基线 | `engineering.py::PeakProtectAGC` | 不代表完整响度/限幅器 |
| §10.3.4；工业 I06 | WebRTC 传统 VAD | 外部参考实现 | `webrtc`：`common_audio/vad/webrtc_vad.c` | int16 单声道与合法 10/20/30 ms |
| §10.3.4；工业 I07 | WebRTC AGC2 | 外部参考实现 | `webrtc`：`modules/audio_processing/gain_controller2.h` | 数字增益与输入音量分开 |
| §10.3.4；工业 I08 | Silero VAD 流式状态 | 外部参考实现 | `silero-vad`：`src/silero_vad/utils_vad.py` | 16 kHz/512、8 kHz/256；会话隔离 |
| §10.4.1 | Q1.15 量化 | 本仓库可运行基线 | `engineering.py::q15_quantize` | 最近偶数舍入与饱和 |
| §10.4.1 | Q15 宽累加点积 | 本仓库可运行基线 | `engineering.py::q15_dot` | 不默认与截位 DSP 逐位相同 |
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
| §10.10 | 遥测记录校验 | 本仓库可运行基线 | `engineering.py::validate_telemetry`、`telemetry_schema.json` | 留存、隐私、统计窗、时钟域 |

## 任务评分与排除范围

| 正文或研究范围 | 算法/机制 | 覆盖状态 | 教学入口或主清单 ID：官方源码入口 | 关键边界 |
|---|---|---|---|---|
| §10.5；工业 I19 | DNSMOS | 外部参考实现 | `dns-challenge`：`DNSMOS/dnsmos_local.py` | 模型/个人化/窗口；不是受试者 MOS |
| §6.7、§10.5 | AECMOS | 外部参考实现 | `aec-challenge`：`AECMOS/AECMOS_local/` | 三路对应、场景、裁段规则 |
| §11.2 | cpWER | 外部参考实现 | `meeteval`：`meeteval/wer/wer/cp.py` | 会话级说话人排列 |
| §11.2 | ORC-WER | 外部参考实现 | `meeteval`：`meeteval/wer/wer/orc.py` | 参考片段到流映射 |
| §11.2 | MIMO-WER | 外部参考实现 | `meeteval`：`meeteval/wer/wer/mimo.py` | 流顺序、允许映射 |
| §11.2 | DI-cpWER | 外部参考实现 | `meeteval`：`meeteval/wer/wer/di_cp.py` | 不可与 sa-WER 互换 |
| §13.7；工业 I21 | tcpWER | 外部参考实现 | `meeteval`：`meeteval/wer/wer/time_constrained.py` | 时间约束、容差和词时间戳 |
| §13.7；工业 I21 | CHiME-8 文本规范化/评分 | 外部参考实现 | `chime-utils`：`chime_utils/`、`tests/test_normalizer.py` | 届次、划分、缺失场景与数据许可 |
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
| 研究扩展：增强 N15 | TF-Locoformer 复谱语音分离 | 外部参考实现 | `tf-locoformer`：`standalone/tflocoformer_separator.py`、`espnet2/enh/separator/tflocoformer_separator.py` | 固定作者源码；静态接口疑点单列；无权重推理或完整训练复现 |
| 附录B E13-02；工业 I29 | STK DelayL 固定分数延迟与连续状态 | 外部参考实现 | `stk`：`include/DelayL.h`、`src/DelayL.cpp`；`examples/run_stk_delay_probe.py` | 已运行标量接口、零/.5/1采样与分块状态；未测设备或变时延 |
| §11.6 E11-09 | 配对符号检验 | 本仓库可运行基线 | `examples/engineering_boundary_exercises.py` | 独立配对、同一评价方向、去除平局；不据WER汇总直接推显著性 |

## 章节代码练习与音频映射

195 道代码练习沿用各章已有模型，稳定 ID 与原有数字题号并存。下表只登记学习入口，不改变上面的 292 行算法统计。补充的空间精算、增强步骤、时间状态模块分别提供3/5/4道题。三个原有 `exercises_` 模块各自提供 `run_exercises()`，分别有 28/23/25 道题；AEC 小实验另有 4 道，进阶 AEC 手算另有 10 道；E03-07、E04-08、E09-06 与 E10-13 由独立实验入口提供。E04-08 的 200 次独立双源抽样只说明固定模型中的分辨事件频率和 Wilson 区间。E04-04 是固定矩阵的前向空间平滑演示，不扩称为支持任意阵列的公共估计接口。

| 章节与稳定 ID | 练习入口 | 回归测试 |
|---|---|---|
| 第 6 章：E06-22～33（12题） | [AEC 状态、数值与指标实验](examples/chapter06_experiments.py) | [独立测试](../tests/test_codes_chapter06_experiments.py)、[数值边界](../tests/test_codes_aec_numerical_boundaries.py) |
| 第 5 章：E05-08～17（10题） | [约束、谱估计与状态实验](examples/chapter05_experiments.py) | [独立测试](../tests/test_codes_chapter05_experiments.py)、[GSC状态测试](../tests/test_codes_gsc.py) |
| 第 4 章：E04-12～18（7题） | [定位逐步实验](examples/chapter04_experiments.py) | [独立测试](../tests/test_codes_chapter04_experiments.py) |
| 第 3 章：E03-08～14（7题） | [几何与校准逐步实验](examples/chapter03_experiments.py) | [独立测试](../tests/test_codes_chapter03_experiments.py) |
| 第 2 章：E02-09～15（7题） | [声学模型逐步实验](examples/chapter02_experiments.py) | [独立测试](../tests/test_codes_chapter02_experiments.py) |
| 第 1 章：E01-04～06（3题） | [基础逐步实验](examples/chapter01_experiments.py) | [独立测试](../tests/test_codes_chapter01_experiments.py) |
| 第 1～5 章：`E01-01`～`E01-03`、`E02-01`～`E02-06`、`E03-01`～`E03-07`、`E04-01`～`E04-09`、`E05-01`～`E05-05`（30 题） | [exercises_spatial.py](examples/exercises_spatial.py)（28 题）；[协同阵 E03-07](examples/coarray_covariance_exercise.py)、[分辨率 E04-08](examples/doa_resolution_trials.py) | [test_codes_exercises_spatial.py](../tests/test_codes_exercises_spatial.py)、[空间扩展测试](../tests/test_codes_spatial_expansion.py) |
| 第 6～9 章：`E06-01`～`E06-20`、`E07-01`～`E07-05`、`E08-01`～`E08-07`、`E09-01`～`E09-06`（38 题） | [exercises_enhancement.py](examples/exercises_enhancement.py)（23 题）；[AEC 四个边界小例](examples/aec_algorithm_minicases.py)（`E06-07`～`E06-10`）；[AEC 十个进阶手算](examples/aec_advanced_exercises.py)（`E06-11`～`E06-20`）；[交叉追踪 E09-06](examples/tracking_crossing_dropout_demo.py) | [test_codes_exercises_enhancement.py](../tests/test_codes_exercises_enhancement.py)、[test_codes_aec_minicases.py](../tests/test_codes_aec_minicases.py)、[test_codes_aec_advanced_exercises.py](../tests/test_codes_aec_advanced_exercises.py)、[test_codes_tracking_crossing_dropout.py](../tests/test_codes_tracking_crossing_dropout.py) |
| 第 10～11 章、附录 A/B：`E10-01`～`E10-14`、`E11-01`～`E11-07`、`E12-01`～`E12-04`、`E13-01`（26 题） | [exercises_engineering.py](examples/exercises_engineering.py)（25 题）；[谱减 E10-13](examples/spectral_subtraction_demo.py)（1 题） | [test_codes_exercises_engineering.py](../tests/test_codes_exercises_engineering.py)、[test_codes_spectral_subtraction.py](../tests/test_codes_spectral_subtraction.py) |
| E02-07、E04-10、E05-06（3题） | [空间精算](examples/spatial_precision_exercises.py) | [独立测试](../tests/test_codes_spatial_precision.py) |
| E09-10～19（10题） | [第9章逐步计算](examples/chapter09_experiments.py) | [独立解析测试](../tests/test_codes_chapter09_experiments.py)、[PCM音频](../tests/test_codes_tracking_audio.py) |
| E08-12～23（12题） | [第8章逐步计算](examples/chapter08_experiments.py) | [独立测试](../tests/test_codes_chapter08_experiments.py) |
| E07-08～17（10题） | [第7章逐步计算](examples/chapter07_experiments.py) | [独立测试](../tests/test_codes_chapter07_experiments.py) |
| E06-21、E07-06、E08-08～10（5题） | [增强逐步计算](examples/enhancement_step_exercises.py) | [独立测试](../tests/test_codes_enhancement_steps.py) |
| E09-07～08、E10-15、E11-08（4题） | [时间与状态](examples/tracking_time_exercises.py) | [独立测试](../tests/test_codes_time_state_exercises.py) |

在仓库根目录使用模块入口：

```bash
.venv/bin/python -m codes.examples.chapter09_experiments
.venv/bin/python -m codes.examples.chapter08_experiments
.venv/bin/python -m codes.examples.chapter07_experiments
.venv/bin/python -m codes.examples.chapter06_experiments
.venv/bin/python -m codes.examples.chapter05_experiments
.venv/bin/python -m codes.examples.chapter04_experiments
.venv/bin/python -m codes.examples.chapter03_experiments
.venv/bin/python -m codes.examples.exercises_spatial
.venv/bin/python -m codes.examples.exercises_enhancement
.venv/bin/python -m codes.examples.aec_algorithm_minicases
.venv/bin/python -m codes.examples.aec_advanced_exercises
.venv/bin/python -m codes.examples.exercises_engineering
.venv/bin/python -m codes.examples.tracking_crossing_dropout_demo
.venv/bin/python -m codes.examples.spectral_subtraction_demo
.venv/bin/python -m codes.examples.coarray_covariance_exercise
.venv/bin/python -m codes.examples.doa_resolution_trials
```

题目、答案和 24 组、98 个合成音频的对应关系见[练习与音频实验](research/05_exercises_and_audio.md)。音频由 [generate_audio_samples.py](examples/generate_audio_samples.py) 生成，参数和摘要见 [MANIFEST.json](audio/MANIFEST.json)；它们只展示特定条件下的现象，不作为完整算法、工业性能或自然语音听测的新增覆盖证据。

另外两套独立合成资产分别是 [GSS 五路 WAV 与中间状态](gss_audio/MANIFEST.json)和[移动声源三路 WAV 与轨迹真值](moving_audio/MANIFEST.json)。它们不计入主清单的 98 个音频，也不等于真实语音或设备验证。

## 未完成项怎样保留

外部参考实现可以存在构建失败、兼容性问题或算法缺陷；CSSM/WAVES/TOPS 的静态问题与 WPE 已完成的合成数值对照应分别阅读，不能把项目数量当成通过率。只有骨干结构时，覆盖的是结构源码，不是完整训练、checkpoint 推理或论文表格。

原理索引明确保留下一步所需证据：唯一作者实现、明确许可、原模型配置，或与正文模型一致的最小代码。不得仅因为框架大、copyleft 或权重未授权就将许可明确的源码降为“没有实现”；也不得因同名函数存在就将整个算法家族标为已覆盖。

本仓库不提交下载缓存、模型权重或未经授权的第三方语料；`audio/` 中的 98 个文件是本书自行合成的教学样本，`real_audio/` 中另有许可明确的 DEMAND 小型摘录和派生文件，不包含完整下载归档。独立上游工作目录的取得、许可保留与未执行项目按来源状态记录报告。算法、源码或排除范围变化时，同步修改本表、研究说明、来源清单和真实验证记录。


真实数据练习 R01 使用 [prepare_real_recordings.py](examples/prepare_real_recordings.py) 与 [real_recordings.py](array_tutorial/real_recordings.py)，测试见 [test_codes_real_recordings.py](../tests/test_codes_real_recordings.py)。R01 比较 DEMAND 录音的数字域二阶矩、交叉项与零延时均值，不是新增定位或增强算法，亦不计入上述 195 道合成/手算代码题。数据来源和许可另见 [real_audio/](real_audio/README.md)。

四组模型与边界练习对应以下独立实现；主音频新增的 4 个 interpolation 文件验证固定滤波误差，不是完整采样率转换性能。

| 稳定 ID | 实现 | 独立测试 |
|---|---|---|
| E02-08、E04-11、E05-07、E12-05 | [空间模型](examples/spatial_model_exercises.py) | [空间模型测试](../tests/test_codes_spatial_model.py) |
| E07-07、E08-11、E09-09 | [增强结构](examples/enhancement_structure_exercises.py) | [增强结构测试](../tests/test_codes_enhancement_structure.py) |
| E10-16、E10-17、E11-09 | [工程边界](examples/engineering_boundary_exercises.py) | [工程边界测试](../tests/test_codes_engineering_boundaries.py) |
| E13-02 | [插值失真](examples/interpolation_exercise.py) | [插值测试](../tests/test_codes_interpolation.py) |
