# 第 10 章：工程实现

本目录保存第 10 章练习的**真实实现**。`chapter10_experiments.py` 复算 E10-18～27；`spectral_subtraction_demo.py` 对应 E10-13；`sro_closed_loop_demo.py` 用精确的合成时间戳展示采样率偏差估计和有状态插值。它们没有接入声卡或验证设备实时性能。

从仓库根目录运行：

```bash
.venv/bin/python -m codes.chapters.ch10.chapter10_experiments
.venv/bin/python -m codes.chapters.ch10.spectral_subtraction_demo
.venv/bin/python -m codes.chapters.ch10.sro_closed_loop_demo
```

原有 `codes.examples` 同名模块保留为兼容入口。背景与边界见 [第 10 章](../../../chapters/10_engineering-practice.md)和[练习手册](../../research/05_exercises_and_audio.md)。
