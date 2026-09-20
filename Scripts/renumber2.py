#!/usr/bin/env python3
"""图号整理脚本（已退役，仅存档）。

历史：单篇长文时代做过一次图号重排（纳入 fig22/fig23 后按首次出现顺序重排，
当时共 23 张图）。单篇长文已删除，教程改为拆分版 12 篇，图增至 32 张，
本脚本原有的“读单篇 md→改 md/py/png→重跑绘图”流程已不再适用，
直接运行会因找不到旧单篇文件而退出。

保留原因：重排逻辑（占位防串改、两阶段重命名）下次整理图号时可参考。
真要复用：先备份，改掉下面的 MD/PY 入口与数量断言，再跑。
注意：里面的正则（`figures/` 前缀）与 23 张断言都是单篇时代的，
拆分版重排要先改 `../figures/` 前缀与数量断言，不然对不上。

当前可用的绘图入口（在报告根目录执行）：
    .venv/bin/python Scripts/make_figures.py      # 图 1~25
    .venv/bin/python Scripts/make_aec_figures.py  # 图 26~32
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.path.join(ROOT, "多麦克风阵列信号处理教程.md")  # 旧单篇，已删除
PY = os.path.join(ROOT, "Scripts", "make_figures.py")
FIGDIR = os.path.join(ROOT, "figures")

if not os.path.exists(MD):
    print("本脚本已退役：旧单篇长文不存在，拆分版请直接改对应篇目与图注。")
    print("绘图请用 Scripts/make_figures.py 与 Scripts/make_aec_figures.py。")
    sys.exit(0)

# ---- 以下为旧单篇时代跑通时的原逻辑，原样保留，不再维护 ----
import re
import subprocess

md = open(MD, encoding="utf-8").read()

order = []
for m in re.finditer(r"!\[图(\d+)\s[^\]]*\]\(figures/fig(\d+)_", md):
    assert int(m.group(1)) == int(m.group(2)), m.group(0)
    n = int(m.group(1))
    if n not in order:
        order.append(n)
mapping = {old: i + 1 for i, old in enumerate(order)}
print("映射（旧→新）:", mapping)
assert len(mapping) == 23, len(mapping)
