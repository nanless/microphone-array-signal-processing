# 第 9 章：声源追踪

本目录保存第 9 章可运行练习的**真实实现**：`chapter09_experiments.py` 复算 E09-10～19，`tracking_crossing_dropout_demo.py` 展示两条轨迹交叉、缺测与方向限速的确定性反例。它们只覆盖题中声明的合成输入，不构成完整多目标追踪系统。

从仓库根目录运行：

```bash
.venv/bin/python -m codes.chapters.ch09.chapter09_experiments
.venv/bin/python -m codes.chapters.ch09.tracking_crossing_dropout_demo
```

原有 `codes.examples` 同名模块保留为兼容入口。公式、题干和限制见 [第 9 章](../../../chapters/09_source-tracking.md)；逐题映射见 [练习手册](../../research/05_exercises_and_audio.md)。
