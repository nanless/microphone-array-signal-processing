# 与正文配套的教学代码

本目录把书中的公式变成可以检查形状、单位和边界情况的最小程序。目标是帮助读者复算，不是提供一个
可直接装进产品的音频前端。实时音频线程、设备驱动、线程调度、定点优化、模型权重和现场标定仍需按
第 10 章单独完成。

## 目录

| 路径 | 内容 |
|---|---|
| `array_tutorial/` | 本书自行编写的 NumPy/标准库教学实现 |
| `examples/` | 按章节组织的可运行例子，打印输入口径、中间量和结果 |
| `upstream/` | 第三方官方仓库的按需获取工具；下载内容默认不入 Git |
| `COVERAGE.md` | 正文算法到代码、测试和外部实现的逐项映射 |
| `THIRD_PARTY.md` | 官方项目的用途、许可证、工程边界和选择建议 |
| `SOURCES.lock.json` | 官方地址、完整提交哈希和核实日期的机器可读清单 |
| [research/](research/README.md) | 空间处理、AEC/WPE/分离、工业部署与源码复现的详细研究文档 |

## 运行

在仓库根目录、已安装 `requirements.txt` 的环境中运行：

```bash
.venv/bin/python codes/examples/ch02_05_baselines.py
.venv/bin/python codes/examples/ch06_09_baselines.py
.venv/bin/python codes/examples/ch10_engineering_baselines.py
.venv/bin/python -m unittest discover -s tests -p 'test_codes*.py' -v
```

例子只使用确定性输入，随机输入会固定种子。函数拒绝维度、单位或参数范围明显错误的输入；这类检查是
为了尽早暴露口径错误，不表示代码已经达到产品级防御能力。

## 如何把公式和程序对上

1. 先在 `COVERAGE.md` 找到正文小节、函数和测试。
2. 看函数文档中的数组形状。`spectral.py`、`covariance.py`、`doa.py` 使用 `C × F × T`（通道 × 频点 × 帧）；`dereverberation.py` 与 `separation.py` 的多通道输入使用 `F × C × T`。跨这两组模块时要显式执行 `spectrum.transpose(1, 0, 2)`，不能直接传递同一数组。
3. 核对角度零点、时延正负、导向矢量相位和共轭约定；不要只比较最后一个数。
4. 运行相应例子，再改变一个条件观察边界。数值相同只说明当前输入下相符，不证明实现普遍正确。

## 原创代码与第三方代码

`array_tutorial/` 中的程序是为本书编写的教学基线。仓库根目录目前没有明确的 `LICENSE` 或 `COPYING`
文件，因此“能看到源码”不等于已经得到复制、修改或再分发许可。对外发布或复用前，项目维护者需要先
选择并添加明确许可证；在此之前不要把本目录的代码自动并入其他产品。

`SOURCES.lock.json` 固定已核实的官方仓库和提交；`upstream/fetch_upstreams.py` 把可获取的源码检出到
`codes/upstream/_downloads/`，每个项目保持独立工作树并保留许可。大型框架也可以在这里阅读源码，
该目录被本仓库 Git 忽略。第三方代码许可证不自动覆盖模型和数据，也不改变本仓库自身的许可状态。

列出索引：

```bash
.venv/bin/python codes/upstream/fetch_upstreams.py --list
```

按需获取一个体量适中的项目：

```bash
.venv/bin/python codes/upstream/fetch_upstreams.py --project nara_wpe
```

批量获取和离线核对：

```bash
.venv/bin/python codes/upstream/fetch_upstreams.py --all --report tmp/source-acquisition.json
.venv/bin/python codes/upstream/fetch_upstreams.py --verify --report tmp/source-verification.json
```

报告区分源码核对、缺少入口和获取失败；依赖构建与算法运行另行验收。新下载省略常见权重、音频和归档文件，
不自动取得 LFS 或子模块。已有目录完整保留。具体筛选范围见 [获取工具说明](upstream/README.md)。
WebRTC 和训练框架仍需各自的构建工具、依赖、模型或数据，取得主源码库只是复现的一步。
