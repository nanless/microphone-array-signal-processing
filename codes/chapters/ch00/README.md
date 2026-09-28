# 导读与全书共用资料

`ch00` 保存需要跨章节维护的索引和生成入口。各章算法的唯一数值实现仍放在相应章节的 `core/`；这里的跨章练习直接导入它们，不复制另一份实现。[按章代码总览](../README.md)说明目录分工。

| 内容 | 入口 | 维护边界 |
|---|---|---|
| 算法与练习 | [算法覆盖表](COVERAGE.md)、[跨章练习](cross_chapter/) | 每个稳定练习 ID 只对应一个真实实现；新增算法时同步正文、测试与覆盖状态。 |
| 主合成音频 | [总清单](audio/MANIFEST.json)、[音频说明](audio/README.md)、[生成器](examples/generate_audio_samples.py) | WAV 按章存放在各章 `audio/`；修改生成源后重生并核对清单，不手改 WAV。 |
| 外部来源 | [Git 锁表](SOURCES.lock.json)、[归档锁表](ARCHIVE_SOURCES.lock.json)、[第三方说明](THIRD_PARTY.md) | 固定版本、许可与获取范围分别记录；取得源码不等于已运行算法。 |
| 离线获取状态 | [Git 状态](SOURCE_STATUS.json)、[归档状态](ARCHIVE_SOURCE_STATUS.json)、[获取工具](upstream/README.md) | 状态报告由获取工具核验后生成；忽略的 `_downloads/` 工作区可能有本地修改，不能覆盖或纳入提交。 |
| 深入阅读 | [源码研究手册](research/README.md) | 区分建议实验、已执行的接口实验与论文级复现。 |

从仓库根目录生成主音频：

```bash
.venv/bin/python codes/chapters/ch00/examples/generate_audio_samples.py
```

只需核对现有主音频时使用 `--check`；它不会重生资产。外部来源核验的命令和报告口径见[获取工具说明](upstream/README.md)。

逐轮审查台账曾记录旧路径、旧页数和当时的环境状态；它们保留在 Git 历史（整理前提交 `d2edf67`），不作为当前代码和资产的使用说明。当前实验条件以正文、研究手册、清单和可运行源码为准。
