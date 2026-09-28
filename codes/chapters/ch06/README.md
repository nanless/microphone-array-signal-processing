# 第 6 章：声学回声消除

这里保存第 6 章综合练习及 NLMS、PBFDAF、RLS、Kalman、子带和双讲控制等示例的实际实现。`chapter06_experiments.py` 是本章 E06-22～33 的统一数值入口，其余文件按算法或实验条件单独运行。通用处理模块仍由 `codes/array_tutorial/` 提供；真实录音夹具和外部 SpeexDSP 库另有依赖，不随目录搬迁复制或下载。

从仓库根目录运行，例如：

```bash
.venv/bin/python -m codes.chapters.ch06.chapter06_experiments
.venv/bin/python -m codes.chapters.ch06.aec_algorithm_minicases
.venv/bin/python -m codes.chapters.ch06.aec_partitioned_demo
```

需要外部库或录音的示例保留其原有命令行参数和固定输入校验。原 `codes.examples.*` 模块名和命令保留兼容入口。
