# 按需获取第三方参考实现

`fetch_upstreams.py` 读取上一级的 `SOURCES.lock.json`，只下载清单中允许获取的项目，并把工作树检出到
固定提交。默认目标是本目录下的 `_downloads/`，该目录已被 Git 忽略。

脚本不会安装依赖、运行构建脚本、下载模型或数据，也不会处理 WebRTC 等需要专用多仓工具的项目。
下载后仍应阅读目标提交中的许可证、NOTICE、依赖说明和安全公告。

```bash
.venv/bin/python codes/upstream/fetch_upstreams.py --list
.venv/bin/python codes/upstream/fetch_upstreams.py --project nara_wpe
.venv/bin/python codes/upstream/fetch_upstreams.py --all
```

`--all` 只处理 `fetch_enabled: true` 的项目。若目标目录已存在，脚本拒绝覆盖；更新时删除或移走自己的
下载目录后重新获取，不要把 `_downloads/` 提交到本仓库。
