# 与正文配套的教学代码

本目录把书中的公式变成可以检查形状、单位和边界情况的最小程序。目标是帮助读者复算，不是提供一个
可直接装进产品的音频前端。实时音频线程、设备驱动、线程调度、定点优化、模型权重和现场标定仍需按
第 10 章单独完成。

## 目录

| 路径 | 内容 |
|---|---|
| `array_tutorial/` | 本书自行编写的 NumPy/标准库教学实现 |
| `examples/` | 按章节组织的可运行例子，打印输入口径、中间量和结果 |
| `audio/` | 9 组、36 个本书合成 WAV 及 `MANIFEST.json`；由音频生成器产生，不直接编辑 |
| `upstream/` | 第三方官方仓库的按需获取工具；下载内容默认不入 Git |
| `COVERAGE.md` | 正文算法到代码、测试和外部实现的逐项映射 |
| `THIRD_PARTY.md` | 官方项目的用途、许可证、工程边界和选择建议 |
| `SOURCES.lock.json` | 官方地址、完整提交哈希和核实日期的机器可读清单 |
| [research/](research/README.md) | 空间处理、AEC/WPE/分离、工业部署、源码复现、练习与音频实验；入口加 5 篇专题，共 6 页 |

## 运行

在仓库根目录、已安装 `requirements.txt` 的环境中运行：

```bash
.venv/bin/python codes/examples/ch02_05_baselines.py
.venv/bin/python codes/examples/ch06_09_baselines.py
.venv/bin/python codes/examples/ch10_engineering_baselines.py
.venv/bin/python -m codes.examples.exercises_spatial
.venv/bin/python -m codes.examples.exercises_enhancement
.venv/bin/python -m codes.examples.exercises_engineering
.venv/bin/python -m unittest discover -s tests -p 'test_codes*.py' -v
```

例子只使用确定性输入，随机输入会固定种子。函数拒绝维度、单位或参数范围明显错误的输入；这类检查是
为了尽早暴露口径错误，不表示代码已经达到产品级防御能力。

三个 `exercises_` 模块各有 20 道题，共新增 60 道，使用 `E01-01` 至 `E13-01` 等稳定题号，不改原有练习编号。每个模块的 `run_exercises()` 返回可序列化为 JSON 的计算结果，导入模块不会执行练习。题目与测试映射见 [COVERAGE.md](COVERAGE.md)，逐题入口与音频对照见[练习与音频实验](research/05_exercises_and_audio.md)。这些练习复用已有算法，不增加原有基线算法数量。

## 合成音频与图 34、35

在仓库根目录先生成音频，再绘图；图 34、35 读取写入 WAV 后的样本：

```bash
.venv/bin/python codes/examples/generate_audio_samples.py
.venv/bin/python scripts/make_figures.py
.venv/bin/python codes/examples/generate_audio_samples.py --check
```

36 个音频文件分为空间处理、AEC、WPE、给定矩阵解混、工程失真、追踪、相关噪声、极性错误和病态求逆 9 组，均为本书合成的 16 kHz、PCM16 信号，没有第三方录音。每组共用一个增益，避免逐文件归一化掩盖幅度差异；清单记录参数、随机种子、生成源文件摘要和 WAV 摘要。`--check` 检查当前生成物，不重写文件，也不自动播放音频。

样例用于观察时延、残留回声、混响、混合和削波等现象，不是自然语音质量评测。给定混合矩阵的求逆不是盲分离；已知双讲区间的冻结不是双讲检测器。这些限制及试听顺序见[音频实验说明](research/05_exercises_and_audio.md)。合成文件的来源说明不等于授予新的再分发许可，许可边界仍见下节。

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
