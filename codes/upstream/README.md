# 按需获取第三方参考实现

`fetch_upstreams.py` 读取上一级的 `SOURCES.lock.json`，只下载清单中允许获取的项目，并把工作树检出到
固定提交。默认目标是本目录下的 `_downloads/`，该目录已被 Git 忽略。

脚本只取得登记的主源码库，不安装依赖、运行构建脚本或递归获取专用多仓环境。
下载后仍应阅读目标提交中的许可证、NOTICE 和依赖说明。取得源码不代表已构建、已运行或已复现论文结果。

```bash
.venv/bin/python codes/upstream/fetch_upstreams.py --list
.venv/bin/python codes/upstream/fetch_upstreams.py --project nara_wpe
.venv/bin/python codes/upstream/fetch_upstreams.py --all --report tmp/source-acquisition.json
.venv/bin/python codes/upstream/fetch_upstreams.py --verify --report tmp/source-verification.json
```

`--all` 只处理 `fetch_enabled: true` 的项目。现有目录的官方地址、提交、工作树及入口文件符合清单时，
直接复用；不符合时保留目录并报告错误。单个项目失败后继续处理其他项目，最终以非零退出状态和 JSON 报告
保留失败记录。`--verify` 只检查本地，不联网；报告中的 `execution: not_run` 明确表示未执行该项目。

报告同时记录请求的源码范围和工作区实际稀疏规则。子集清单改变而旧工作区不符合时，返回
`source_selection_mismatch`，不自动删除旧内容。Git 调用清理外层仓库选择与配置环境变量，避免命令被重定向到本书仓库；
该工具也不读取用户全局或系统 Git 配置。需要网络代理时应使用环境级代理配置，不能依赖全局 Git URL 重写。

新下载使用稀疏工作树，省略常见权重、音频、动态库、NumPy 数组和归档扩展名；完整列表位于
`fetch_upstreams.py` 的 `OMITTED_EXTENSIONS`。它不递归取得子模块，并通过 `GIT_LFS_SKIP_SMUDGE=1`
阻止 LFS 资产自动展开。仍可能存在源码内嵌模型系数和 Git 跟踪资源，不能据此声称所有第三方资产均已排除。
原有下载目录保持原样，报告标为 `existing_checkout_preserved`。

独立本地下载适合阅读大型框架或 copyleft 项目的源码。再分发、商业集成、权重和数据使用仍按各自条款判断。
WebRTC 的这一份工作树只包含主源码库，不是 `depot_tools` 管理的完整构建环境；其他含子模块的项目同理。
无许可证、只许评估或学术用途的记录保持明确索引状态，不能混称自由开源软件。

更新锁定版本时，可通过 `--destination` 建立另一个目录，先检查已有修改归属；不要覆盖旧目录或把
`_downloads/` 整体加入本仓库 Git。逐算法读码方法见 [源码复现手册](../research/04_source_reproduction.md)。
