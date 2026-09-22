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
2. 看函数文档中的数组形状。本书教学代码的多通道频谱统一为 `C × F × T`，即通道 × 频点 × 帧；接入采用其他轴序的外部项目时，在适配边界显式转换。
3. 核对角度零点、时延正负、导向矢量相位和共轭约定；不要只比较最后一个数。
4. 运行相应例子，再改变一个条件观察边界。数值相同只说明当前输入下相符，不证明实现普遍正确。

## 原创代码与第三方代码

`array_tutorial/` 中的程序是为本书编写的教学基线。仓库根目录目前没有明确的 `LICENSE` 或 `COPYING`
文件，因此“能看到源码”不等于已经得到复制、修改或再分发许可。对外发布或复用前，项目维护者需要先
选择并添加明确许可证；在此之前不要把本目录的代码自动并入其他产品。

本仓库不直接复制大型第三方项目、模型权重或数据集。`SOURCES.lock.json` 固定本轮核实过的官方仓库和
提交；`upstream/fetch_upstreams.py` 只在读者主动执行时下载允许获取的项目，并检出指定提交。第三方
代码许可证不自动覆盖模型和数据许可证，也不改变本仓库自身缺少许可证这一事实。

列出索引：

```bash
.venv/bin/python codes/upstream/fetch_upstreams.py --list
```

按需获取一个体量适中的项目：

```bash
.venv/bin/python codes/upstream/fetch_upstreams.py --project nara_wpe
```

WebRTC、训练框架等大型项目默认只索引。它们常有额外的构建工具、子模块、模型或数据要求，不能把
一次 Git 下载等同于完整复现。
