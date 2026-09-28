# 附录 B：复现与证据练习

本目录保存附录 B 两组练习的**真实实现**。`appendix_b_experiments.py` 复算 E13-03～08：其中两题只读已发布房间报告和 PCM，其他题使用解析夹具；它不会重新运行房间仿真或外部系统。`interpolation_exercise.py` 复算 E13-02 的固定分数延迟幅度与 PCM 读回。

从仓库根目录运行：

```bash
.venv/bin/python -m codes.chapters.appendix_b.appendix_b_experiments
.venv/bin/python -m codes.chapters.appendix_b.interpolation_exercise
```

题干、结果及执行层级见 [附录 B](../../../chapters/13_appendix-guide.md)、[房间结果](room_audio/RESULTS.json)和[练习手册](../ch00/research/05_exercises_and_audio.md)。
