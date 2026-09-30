# 按需获取第三方参考实现

`fetch_upstreams.py` 读取上一级的 `SOURCES.lock.json`，只下载清单中允许获取的项目，并把工作树检出到
固定提交。默认目标是本目录下的 `_downloads/`，该目录已被 Git 忽略。

`_downloads/` 是本机按需取得的第三方源码缓存；其中的上游 README 和其他 Markdown 不属于本书的教程或研究手册，也不随本仓库的 Git 提交发布。两个获取脚本都支持 `--destination`，可为今后的获取或离线核验指定仓库外的新目录；核验时须传入同一目录。这个选项不会迁移现有缓存。已有工作树可能包含本地修改或数据，不能为清理目录而覆盖或删除。

脚本只取得登记的主源码库，不安装依赖、运行构建脚本或递归获取专用多仓环境。
下载后仍应阅读目标提交中的许可证、NOTICE 和依赖说明。取得源码不代表已构建、已运行或已复现论文结果。

```bash
.venv/bin/python codes/chapters/ch00/upstream/fetch_upstreams.py --list
.venv/bin/python codes/chapters/ch00/upstream/fetch_upstreams.py --project nara_wpe
.venv/bin/python codes/chapters/ch00/upstream/fetch_upstreams.py --all --report tmp/source-acquisition.json
.venv/bin/python codes/chapters/ch00/upstream/fetch_upstreams.py --verify --report tmp/source-verification.json
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

更新锁定版本前先检查已有修改归属，在新目录核验，勿将 `_downloads/` 整体加入本仓库 Git。逐算法读码方法见[源码复现手册](../research/04_source_reproduction.md)。


## 固定摘要的官方源码归档

有些官方源码使用发布归档而不是 Git。`fetch_archives.py` 读取独立的
[`ARCHIVE_SOURCES.lock.json`](../ARCHIVE_SOURCES.lock.json)，验证归档与官方描述文件的 SHA-256，
再提取登记的源码子集。工具支持固定的tar.xz与ZIP；两种格式都先验压缩字节，再检查路径、链接、重复成员、大小上限及选集。HARKTOOL5 3.5.0 保存在 `_downloads/harktool5-3.5.0/`，
原始压缩包与 `.dsc` 存于 `_downloads/.archive-cache/`，两处均不进入本书 Git 提交。

```bash
.venv/bin/python codes/chapters/ch00/upstream/fetch_archives.py --list
.venv/bin/python codes/chapters/ch00/upstream/fetch_archives.py --project harktool5-3.5.0
.venv/bin/python codes/chapters/ch00/upstream/fetch_archives.py --verify --report codes/chapters/ch00/ARCHIVE_SOURCE_STATUS.json
```

校验发生在解析和写入源码之前。工具限制压缩与展开大小，拒绝路径逃逸、链接、特殊文件和重复路径；
已有目录只能核对，不会覆盖修复。离线验证将本地每个文件与已校验归档的选集逐字节比较，检查缺失、
额外文件及目录，报告实际文件摘要和未运行状态。HTTP 来源使用已经固定的摘要校验内容，
这里没有宣称验证发布签名。摘要的出处与核实日期保存在锁表。

归档来源独立计数，不能用 Git 项目总数代表全部获取方式。HARK 的受限许可、Infineon 配置文件与
算法核心的区别见[第三方说明](../THIRD_PARTY.md)；两个获取工具都不会安装、编译或执行上游程序。


Vo RFS 追踪源码归档单独取得和核验：

```bash
.venv/bin/python codes/chapters/ch00/upstream/fetch_archives.py --project vo-rfs-tracking-updated
.venv/bin/python codes/chapters/ch00/upstream/fetch_archives.py --verify --report codes/chapters/ch00/ARCHIVE_SOURCE_STATUS.json
```

该作者ZIP仅作许可允许的学术研究阅读，保留62个选定源文件及原声明。其descriptor是官方网页快照，
并非`.dsc`、官方校验和或签名；锁表SHA-256为本次取得后本地计算。完整归档只在忽略缓存中，
选定目录不含预编译MEX或数据，未运行MATLAB/MEX。不能把两项归档与Git来源混算。


## 第二章高精度环境校准参考

原作者湿空气声速项目固定为`5c7e6652cf2fd7b4c52e83d8fa3782b3c56e61de`。
获取工具支持路径中的字面ASCII空格，仍拒绝通配符、换行、绝对路径和父目录越界。
目录`source labview files/`的43个VI、`LICENSE`与`ReadMe.txt`已按选集取得，
没有取得预编译EXE、论文PDF或LabVIEW运行时。VI是LabVIEW保存的图形源码；
此处取得文件没有证明已查看框图或执行模型。具体假设与运行依赖见
[第二章环境参数研究](../research/01_spatial_and_tracking.md#12-声速参数必须与实际实现一致)。

```bash
.venv/bin/python codes/chapters/ch00/upstream/fetch_upstreams.py --project speed-of-sound-in-air
```
