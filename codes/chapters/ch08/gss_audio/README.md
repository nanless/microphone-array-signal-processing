# GSS 教学链的独立资产

本目录的 5 个 WAV 与 `STATE.npz` 由 `codes/chapters/ch08/examples/gss_teaching_demo.py` 生成；不并入主音频清单。这是两路带活动包络的高斯数学信号和已知双麦延迟的受控实验，没有真人语音、房间混响或自动说话人分割。WPE 旁路，未运行官方 GPU-GSS 整链。

```bash
.venv/bin/python -m codes.chapters.ch08.examples.gss_teaching_demo
.venv/bin/python -m codes.chapters.ch08.examples.gss_teaching_demo --check
```

`--check` 重新计算后只读比较清单、状态和 WAV 字节，不会修复或覆盖过期资产；`--output` 可指定独立临时目录。清单保存源码摘要、Python/NumPy/平台、公共增益、帧边界、量化与评分区间。

输入采用周期 Hann 窗，256 点窗、128 点帧移，两侧各补 128 个零。重构用加权重叠相加，移除左侧补零并截取 32000 点。正确与漏标两种条件使用相同输入、窗、输出长度、导出增益和评分参考。

掩码来自固定 8 轮带活动约束的 cACG 更新，其中包含均值后验先验更新、先验下限、向单位阵收缩与小质量重置。它不是活动条件似然下的精确 EM；不宣称每轮似然增加或 8 轮已经收敛。低能量判据为每频点 `max(1e-12, 1e-7 * 该频点最大向量范数)`，低能点直接归背景。

`STATE.npz` 的 `posterior` 与 `e_step_shapes`、`e_step_priors` 属于最后一次 E 步。`shape_matrices`、`post_update_priors` 是随后最后一次更新的结果，不能拿它们直接解释前一时间点的后验。`valid_points` 区分强制背景与实际参与更新的点；`actual_iterations_per_frequency` 记录全低能量频点的旁路。

`target_scm` 和 `other_scm` 使用未作方向归一化的复谱。清单的 `beam_routes`、`missed_beam_routes` 区分求得 MVDR 权重与退回参考麦的原因。主特征方向没有特征值间隙时，返回数值方向并不能证明目标方向可辨识。

原 `si_sdr_db` 记录量化前浮点信号的分数；`pcm_si_sdr_db` 另用读回 PCM 与 PCM 目标评分。两者都使用参考麦 0 的第一路源图像、[3200,23200) 点，分别去均值，不作时延对齐。浮点输入/正确输出约为 1.280328/13.761977 dB，PCM 对应约为 1.280339/13.762022 dB。漏标目标输出是参考麦回退，不是成功分离。

播放前调低音量。客观读回与分数检查不代表人工听测，也不能外推真实会议、识别准确率或设备实时性。
