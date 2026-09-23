# -*- coding: utf-8 -*-
"""回声消除专题插图：图 26～32、37～39。

用法（仓库根目录）：
    .venv/bin/python scripts/make_aec_figures.py  # 图 26～32、37～39 → figures/
函数与图号对照：fig_problem→图26、fig_concept→图27、fig_nlms→图28、
fig_erle→图29、fig_delay_dtd→图30、fig_nonlinear→图31、fig_hybrid→图32、
fig_haar_crossband→图37、fig_adaptive_state_examples→图38、
fig_pbfdaf_flow→图39。
"""
import hashlib
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib import gridspec
from pathlib import Path

# Direct ``python scripts/make_aec_figures.py`` must see the repository package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codes.array_tutorial.conventions import finite_real_array, finite_real_scalar

OUT = Path(os.environ.get("AEC_FIGURE_DIR", Path(__file__).parent.parent / "figures"))
OUT.mkdir(exist_ok=True)
C_MAIN, C_BLUE, C_RED, C_GREEN, C_ORANGE, C_PURPLE = "#1a1a2e", "#2f6db3", "#c0392b", "#2e8b57", "#e67e22", "#7d3c98"
# 9.8 in 宽图缩放到 A4 正文约 165 mm 后，11 pt 约为 7.3 pt。
# FS_TINY 只用于刻度；轴名、图例、框字和承担结论的注释不得低于 FS_SMALL。
FS_SUP, FS_TITLE, FS_LABEL, FS_SMALL, FS_TINY = 15, 13, 11, 11, 10
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
FS = 16000
SOURCE_SCRIPT = "scripts/make_aec_figures.py"

def source_script_digest():
    """返回当前绘图源文件完整字节的 SHA-256，供 PNG 溯源。"""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def figure_metadata():
    """生成确定性的 PNG 元数据；不写入构建时间。"""
    return {
        "SourceScript": SOURCE_SCRIPT,
        "SourceScriptDigest": source_script_digest(),
    }

def save(fig, name):
    fig.savefig(
        OUT / name,
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
        metadata=figure_metadata(),
    )
    plt.close(fig)
    print("saved", name)

def synth_h(taps=256, decay=60, seed=7):
    """构造含直达、三条早期反射和随机晚期尾的可复现回声路径。"""
    r = np.random.default_rng(seed)
    h = 0.08 * np.exp(-np.arange(taps) / decay) * r.standard_normal(taps)
    h[0] += 1.0
    for index, amplitude in ((25, 0.45), (62, -0.30), (103, 0.18)):
        if index < taps:
            h[index] += amplitude
    return h

def colored_x(N, seed=3):
    r = np.random.default_rng(seed)
    return np.convolve(r.standard_normal(N), np.ones(8) / 8, mode="same")

def causal_delay(x, delay):
    """将一维信号延迟指定采样数，开头补零，不把末尾样本绕回开头。"""
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("x 必须是一维信号")
    if not isinstance(delay, (int, np.integer)) or delay < 0:
        raise ValueError("delay 必须是非负整数")
    y = np.zeros_like(x)
    if delay == 0:
        y[:] = x
    elif delay < len(x):
        y[delay:] = x[:-delay]
    return y

def rms_envelope(x, win=320):
    """计算居中滑窗 RMS 包络，仅用于波形概览。"""
    if win <= 0:
        raise ValueError("win 必须为正整数")
    kernel = np.ones(int(win), dtype=float) / int(win)
    return np.sqrt(np.convolve(np.asarray(x, dtype=float) ** 2, kernel, mode="same"))

def nlms_adaptation_trace(
    x, d, taps=128, mu=0.5, freeze=None, snapshot_interval=400, true_path=None
):
    """运行 NLMS 并返回更新前快照、失配及其采样位置。

    每个快照都在相应样本的更新之前记录，所以第 0 个快照严格表示
    全零初值。若提供 ``true_path``，同时返回归一化系数欧氏失配（dB）。
    """
    x = finite_real_array(x, "x")
    d = finite_real_array(d, "d")
    if x.ndim != 1 or d.ndim != 1 or x.shape != d.shape:
        raise ValueError("x 与 d 必须是同长度一维数组")
    if isinstance(taps, (bool, np.bool_)) or not isinstance(taps, (int, np.integer)) or taps <= 0:
        raise ValueError("taps 必须是正整数")
    if (isinstance(snapshot_interval, (bool, np.bool_))
            or not isinstance(snapshot_interval, (int, np.integer))
            or snapshot_interval <= 0):
        raise ValueError("snapshot_interval 必须是正整数")
    mu = finite_real_scalar(mu, "mu")
    if not 0 <= mu < 2:
        raise ValueError("mu 必须满足 0 <= mu < 2")
    if freeze is not None:
        freeze = np.asarray(freeze)
        if freeze.dtype.kind != "b" or freeze.shape != x.shape:
            raise ValueError("freeze 必须是与 x 同形的布尔数组")
    if true_path is not None:
        true_path = finite_real_array(true_path, "true_path")
        if true_path.shape != (taps,):
            raise ValueError("true_path 的长度必须等于 taps")
        path_power = float(true_path @ true_path)
        if not np.isfinite(path_power) or path_power <= 0:
            raise ValueError("true_path 的能量必须为有限正数")

    N = len(x)
    w = np.zeros(taps)
    e = np.zeros(N)
    snaps, mismatch, snapshot_samples = [], [], []
    X = np.zeros(taps)
    for n in range(N):
        X = np.roll(X, 1); X[0] = x[n]
        if n % snapshot_interval == 0:
            snaps.append(w.copy())
            snapshot_samples.append(n)
            if true_path is not None:
                mismatch.append(10 * np.log10(float((true_path - w) @ (true_path - w)) / path_power))
        with np.errstate(over="raise", invalid="raise"):
            try:
                y = float(w @ X)
                e[n] = d[n] - y
            except FloatingPointError as error:
                raise ValueError("NLMS 预测超出 float64 范围") from error
        if mu != 0 and (freeze is None or not freeze[n]):
            with np.errstate(over="raise", invalid="raise"):
                try:
                    w = w + mu * e[n] * X / (float(X @ X) + 1e-6)
                except FloatingPointError as error:
                    raise ValueError("NLMS 更新超出 float64 范围") from error
    return (
        e,
        np.asarray(snaps),
        w,
        np.asarray(mismatch),
        np.asarray(snapshot_samples, dtype=int),
    )

def nlms_run(x, d, taps=128, mu=0.5, freeze=None):
    """兼容原绘图调用的 NLMS 包装器。"""
    e, snaps, w, _, _ = nlms_adaptation_trace(x, d, taps, mu, freeze)
    return e, snaps, w

def block_erle(echo, e, blk=400):
    """按完整块计算功率比 ERLE；零功率保留其数学语义而不加绝对地板。"""
    echo = finite_real_array(echo, "echo")
    e = finite_real_array(e, "e")
    if echo.shape != e.shape or echo.ndim != 1:
        raise ValueError("echo 与 e 必须是同长度一维数组")
    if isinstance(blk, (bool, np.bool_)) or not isinstance(blk, (int, np.integer)) or blk <= 0:
        raise ValueError("blk 必须是正整数")
    def log_power(block):
        scale = float(np.max(np.abs(block)))
        if scale == 0:
            return -np.inf
        normalized = block / scale
        return 2 * np.log10(scale) + np.log10(float(np.mean(normalized * normalized)))

    starts = range(0, len(echo) - blk + 1, blk)
    tc = np.array([i / FS for i in starts])
    erle = np.empty(tc.shape)
    for index, start in enumerate(starts):
        input_log = log_power(echo[start:start+blk])
        residual_log = log_power(e[start:start+blk])
        if input_log == -np.inf and residual_log == -np.inf:
            erle[index] = np.nan
        else:
            erle[index] = 10 * (input_log - residual_log)
    return tc, erle

def mask_metric_intervals(times, values, intervals):
    """把指标定义无效的时间区间置为 NaN，保留原数组不变。"""
    times = np.asarray(times, dtype=float)
    masked = np.asarray(values, dtype=float).copy()
    if times.shape != masked.shape:
        raise ValueError("times 与 values 必须同形")
    for start, stop in intervals:
        if stop <= start:
            raise ValueError("区间终点必须大于起点")
        masked[(times >= start) & (times < stop)] = np.nan
    return masked

# ---- 图26：回声是怎么产生的 ----
def fig_problem():
    N = int(1.6 * FS)
    r = np.random.default_rng(2601)
    x = colored_x(N, seed=2602); h = 0.5 * synth_h(seed=2603)
    echo = np.convolve(x, h, mode="full")[:N]
    s = np.zeros(N); s[int(0.8*FS):int(1.2*FS)] = 0.7 * r.standard_normal(int(0.4*FS))
    v = 0.02 * r.standard_normal(N)
    d = echo + s + v
    t = np.arange(N) / FS
    fig = plt.figure(figsize=(9.8, 8.0), layout="constrained")
    fig.suptitle("图26 回声形成：d(n) = s(n) + x(n)*h[ℓ] + v(n)", fontsize=FS_SUP)
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[0.9, 1.1])
    ax = fig.add_subplot(gs[0, :])
    ax.set_title("(a) 用脉冲响应 h[ℓ] 表示房间回声路径", fontsize=FS_TITLE)
    ax.stem(h[:128], linefmt=C_BLUE, markerfmt="o", basefmt=" ", label="h抽头")
    ax.set_xlabel("抽头索引 ℓ（采样）", fontsize=FS_LABEL); ax.set_ylabel("归一化幅度", fontsize=FS_LABEL)
    ax.annotate("0号抽头为直达声；25、62、103号为早期反射\n其余随机衰减项表示晚期尾", xy=(25, h[25]), xytext=(58, 0.43),
                fontsize=FS_SMALL + 1, arrowprops=dict(arrowstyle="->", color=C_ORANGE), color=C_ORANGE)
    ax.grid(ls=":", alpha=0.5); ax.tick_params(labelsize=FS_TINY)
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("(b) 卷积把声音拖长：x → 回声", fontsize=FS_TITLE)
    ax.plot(t, x, color=C_BLUE, lw=1, label="远端 x(n)")
    ax.plot(t, echo, color=C_BLUE, ls="--", lw=1, label="回声 x*h")
    ax.plot(t, s, color=C_RED, lw=1, label="合成近端干扰 s(n)（非语音）")
    ax.axvspan(0.8, 1.2, color=C_RED, alpha=0.10)
    ax.set_xlim(0, 1.6); ax.set_xlabel("时间 (s)", fontsize=FS_LABEL); ax.set_ylabel("归一化幅度", fontsize=FS_LABEL)
    ax.legend(fontsize=FS_SMALL); ax.grid(ls=":", alpha=0.5); ax.tick_params(labelsize=FS_TINY)
    ax = fig.add_subplot(gs[1, 1])
    ax.set_title("(c) 麦克风里三者相加得 d(n)", fontsize=FS_TITLE)
    ax.plot(t, d, color=C_MAIN, lw=1.2, label="麦克风 d(n)")
    ax.plot(t, echo, color=C_BLUE, lw=1.0, ls="--", alpha=0.8, label="回声分量")
    ax.axvspan(0.8, 1.2, color=C_RED, alpha=0.10)
    ax.set_xlim(0, 1.6); ax.set_xlabel("时间 (s)", fontsize=FS_LABEL); ax.set_ylabel("归一化幅度", fontsize=FS_LABEL)
    ax.legend(fontsize=FS_SMALL); ax.grid(ls=":", alpha=0.5); ax.tick_params(labelsize=FS_TINY)
    save(fig, "fig26_aec_problem.png")

# ---- 图27：AEC 基本结构 ----
def fig_concept():
    fig, ax = plt.subplots(figsize=(9.8, 5.4), layout="constrained")
    fig.suptitle("图27 AEC 结构：根据播放参考估计回声，再从麦克风信号中相减", fontsize=FS_SUP)
    ax.set_xlim(0, 11); ax.set_ylim(-0.55, 6); ax.axis("off")
    def box(x, y, w, h, text, fc):
        ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="k", lw=1.2))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=FS_SMALL, color=C_MAIN)
    def arrow(a, b, c="k", ls="-", w=1.4):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, color=c, lw=w, ls=ls))
    box(0.3, 2.5, 1.9, 1.4, "播放 x(n)", "#f6e5db")
    box(2.7, 2.5, 2.0, 1.4, "扬声器+房间 h", "#dbe9f6")
    box(2.7, 0.4, 2.0, 1.2, "自适应滤波 ŵ\n(NLMS)", "#dbe9f6")
    box(5.4, 0.4, 1.9, 1.2, "回声副本 ŷ", "#f6dbdb")
    box(5.4, 2.5, 1.9, 1.4, "麦克风 d\n=回声+s+v", "#f6dbdb")
    ax.add_patch(Circle((8.2, 3.2), 0.45, fc="white", ec="k", lw=1.4))
    ax.text(8.2, 3.2, "Σ−", ha="center", va="center", fontsize=FS_LABEL + 4)
    box(9.1, 2.5, 1.6, 1.4, "处理后输出 e\n→波束/识别", "#e8f6db")
    box(7.7, 4.6, 2.2, 0.9, "DTD（由 x、d 判定）\n控制系数更新", "#f6e5db")
    arrow((2.2, 3.2), (2.7, 3.2), C_BLUE)
    arrow((4.7, 3.2), (5.4, 3.2), C_BLUE)
    arrow((1.25, 2.5), (3.0, 1.6), C_BLUE)
    arrow((4.7, 1.0), (5.4, 1.0), C_BLUE)
    arrow((7.3, 1.0), (8.02, 2.72), C_BLUE)
    arrow((7.3, 3.2), (7.75, 3.2), C_BLUE)
    arrow((8.65, 3.2), (9.1, 3.2), C_GREEN, w=2.0)
    # DTD 的两路观测输入与下方的残差反馈是不同支路。
    ax.plot([1.25, 1.25, 8.1], [3.9, 5.72, 5.72], color=C_PURPLE, lw=1.2)
    arrow((8.1, 5.72), (8.1, 5.5), C_PURPLE)
    ax.plot([6.35, 6.35, 9.25], [3.9, 4.28, 4.28], color=C_PURPLE, lw=1.2)
    arrow((9.25, 4.28), (9.25, 4.6), C_PURPLE)
    # 残差 e 沿底部返回滤波器更新端；DTD 控制反馈支路上的开关。
    ax.plot([9.9, 9.9, 9.15], [2.5, 0.18, 0.18], color=C_BLUE, lw=1.4)
    ax.add_patch(Rectangle((8.35, 0.03), 0.8, 0.3, fc="white", ec=C_RED, lw=1.2, zorder=4))
    ax.plot([8.48, 8.98], [0.29, 0.08], color=C_RED, lw=1.6, zorder=5)
    ax.plot([3.7, 8.35], [0.18, 0.18], color=C_BLUE, lw=1.4)
    arrow((3.7, 0.18), (3.7, 0.4), C_BLUE)
    ax.annotate("残差 e 驱动更新", xy=(7.8, 0.18), xytext=(6.8, -0.4),
                fontsize=FS_SMALL, color=C_BLUE, ha="center",
                arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.1))
    arrow((8.8, 4.6), (8.75, 0.33), C_RED, ls="--", w=1.8)
    ax.annotate("估计回声路径", xy=(2.68, 1.0), xytext=(0.7, 0.35), fontsize=FS_SMALL + 2, color=C_PURPLE,
                arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=1.8))
    save(fig, "fig27_aec_concept.png")

# ---- 图28：NLMS 路径估计 ----
def fig_nlms():
    N = int(1.6 * FS); taps = 128
    h = np.exp(-np.arange(taps) / 25) * np.random.default_rng(2801).standard_normal(taps)
    x = colored_x(N, seed=2802); d = np.convolve(x, h, mode="full")[:N]
    e, snaps, _w, mism, snapshot_samples = nlms_adaptation_trace(
        x, d, taps, 0.5, None, snapshot_interval=400, true_path=h
    )
    tc = snapshot_samples / FS
    tcc, erle = block_erle(d, e)
    fig = plt.figure(figsize=(9.8, 8.2), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(1.05, 1.0))
    axes = [
        fig.add_subplot(grid[0, :]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]
    fig.suptitle("图28 NLMS 自适应：路径估计逐步逼近真实响应", fontsize=FS_SUP)
    ax = axes[0]; ax.set_title("(a) ŵ 逐步逼近 h", fontsize=FS_TITLE)
    ax.plot(h[:64], color="0.35", lw=2.2, marker="o", markevery=8, label="真实路径 h")
    cb_styles = [(C_BLUE, "-", "深蓝实线"), (C_ORANGE, "--", "橙色虚线"),
                 (C_GREEN, "-.", "绿色点划线"), (C_PURPLE, ":", "紫色点线")]
    pick = [0, len(snaps)//3, 2*len(snaps)//3, len(snaps)-1]
    for style_index, (j, (c, ls, _nm)) in enumerate(zip(pick, cb_styles)):
        ax.plot(snaps[j][:64], color=c, ls=ls, lw=1.6,
                marker=("o", "s", "^", "D")[style_index], markevery=10,
                label=f"{snapshot_samples[j]}步")
    ax.set_xlabel("抽头索引 ℓ（采样）", fontsize=FS_LABEL)
    ax.set_ylabel("滤波器系数（无量纲）", fontsize=FS_LABEL)
    ax.legend(fontsize=FS_SMALL, ncols=3)
    ax.grid(ls=":", alpha=0.5); ax.tick_params(labelsize=FS_TINY)
    ax = axes[1]; ax.set_title("(b) 失配下降，ERLE 上升", fontsize=FS_TITLE)
    ax.plot(tc, mism, color=C_RED, ls="-", marker="o", markevery=5, label="失配(dB)")
    ax.set_xlabel("时间 (s)", fontsize=FS_LABEL); ax.tick_params(labelsize=FS_TINY)
    ax2 = ax.twinx(); ax2.plot(tcc, erle, color=C_BLUE, ls="--", marker="s", markevery=6, label="ERLE(dB)")
    ax.tick_params(labelsize=FS_TINY, colors=C_RED)
    ax2.tick_params(labelsize=FS_TINY, colors=C_BLUE)
    ax.set_ylabel("失配 (dB)", fontsize=FS_LABEL, color=C_RED)
    ax2.set_ylabel("ERLE (dB)", fontsize=FS_LABEL, color=C_BLUE)
    ax.grid(ls=":", alpha=0.5)
    ax.text(0.99, -6.0, "系数欧氏失配与残余功率采用不同加权；\n这里只比较下降/上升趋势，不作等量换算", fontsize=FS_SMALL, color=C_MAIN, ha="right",
            bbox=dict(fc="white", ec="0.7", alpha=0.9))
    ax = axes[2]; ax.set_title("(c) 步长 μ：无噪声模型中的有限时段收敛", fontsize=FS_TITLE)
    step_styles = [(0.2, C_BLUE, "-", "o"), (0.5, C_ORANGE, "--", "s"),
                   (1.0, C_RED, "-.", "^")]
    for mu, c, ls, marker in step_styles:
        ee, _, _ = nlms_run(x, d, taps, mu, None)
        _, er = block_erle(d, ee)
        ax.plot(tcc[:len(er)], er, color=c, ls=ls, marker=marker,
                markevery=8, label=f"μ={mu}")
    ax.set_xlabel("时间 (s)", fontsize=FS_LABEL); ax.set_ylabel("ERLE (dB)", fontsize=FS_LABEL)
    ax.legend(fontsize=FS_SMALL); ax.grid(ls=":", alpha=0.5); ax.tick_params(labelsize=FS_TINY)
    ax.text(0.05, 0.08, "本图只比较 μ=0.2、0.5、1.0 的暂态。\n"
            "无近端噪声，不据此判断稳态失调。",
            transform=ax.transAxes, fontsize=FS_SMALL, color=C_RED,
            bbox=dict(fc="white", ec="0.8", alpha=0.85))
    save(fig, "fig28_aec_nlms.png")

# ---- 图29：ERLE + 合成近端干扰时的真值冻结 ----
def fig_erle():
    N = int(1.6 * FS); taps = 128
    h = np.exp(-np.arange(taps) / 25) * np.random.default_rng(2901).standard_normal(taps)
    x = colored_x(N, seed=2902); echo = np.convolve(x, h, mode="full")[:N]
    s = np.zeros(N); s[int(0.8*FS):int(1.2*FS)] = 0.7 * np.random.default_rng(2903).standard_normal(int(0.4*FS))
    d = echo + s
    freeze = np.zeros(N, bool); freeze[int(0.8*FS):int(1.2*FS)] = True
    e, _, _ = nlms_run(x, d, taps, 0.5, freeze)
    tcc, erle = block_erle(echo, e)
    plateau = float(np.nanmean(erle[(tcc > 0.45) & (tcc < 0.8)]))
    t = np.arange(N) / FS
    envs = [rms_envelope(sig) for sig in (x, d, e)]
    env_scale = max(float(np.max(env)) for env in envs)
    envs = [env / max(env_scale, 1e-12) for env in envs]
    fig = plt.figure(figsize=(9.8, 8.5), layout="constrained")
    fig.suptitle("图29 NLMS 收敛与合成近端干扰时的冻结", fontsize=FS_SUP)
    gs = gridspec.GridSpec(4, 1, figure=fig, height_ratios=[0.7, 0.7, 0.7, 1.5])
    labels = ["参考 x", "麦克风 d", "残差 e"]
    colors = [C_BLUE, C_RED, C_GREEN]
    for row, (env, label, color) in enumerate(zip(envs, labels, colors)):
        ax = fig.add_subplot(gs[row])
        ax.plot(t[::16], env[::16], color=color, lw=1.1)
        ax.axvspan(0.8, 1.2, color=C_RED, alpha=0.10)
        ax.set_xlim(0, 1.6); ax.set_ylim(0, 1.05)
        ax.set_ylabel(label, fontsize=FS_LABEL, color=color)
        ax.grid(ls=":", alpha=0.4); ax.tick_params(labelsize=FS_TINY)
        if row < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("时间 (s)；纵轴为共同尺度的归一化均方根（RMS）包络", fontsize=FS_SMALL)
    ax = fig.add_subplot(gs[3]); ax.set_title("ERLE 只在远端单讲区评价", fontsize=FS_TITLE)
    erle_view = erle.copy()
    erle_view[(tcc >= 0.8) & (tcc < 1.2)] = np.nan
    ax.plot(tcc, erle_view, color=C_BLUE, lw=1.6, label="ERLE（干扰段不绘制）")
    ax.axvspan(0.8, 1.2, color=C_RED, alpha=0.10)
    ax.text(1.0, np.nanmax(erle_view)*0.88, "合成近端干扰段（非语音）\nERLE 不评价回声抵消量", ha="center", fontsize=FS_SMALL + 1, color=C_RED,
            bbox=dict(fc="white", ec=C_RED, lw=0.7, alpha=0.9, boxstyle="round,pad=0.3"))
    ax.annotate(f"收敛后 ERLE≈{plateau:.0f} dB", xy=(0.62, plateau), xytext=(0.35, plateau+6),
                fontsize=FS_LABEL, color=C_BLUE,
                arrowprops=dict(arrowstyle="->", color=C_BLUE))
    ax.set_xlim(0, 1.6); ax.set_xlabel("时间 (s)", fontsize=FS_LABEL); ax.set_ylabel("ERLE (dB)", fontsize=FS_LABEL)
    ax.legend(fontsize=FS_SMALL, loc="lower right")
    ax.grid(ls=":", alpha=0.5); ax.tick_params(labelsize=FS_TINY)
    save(fig, "fig29_aec_erle_freeze.png")

# ---- 图30：延迟 × 合成近端干扰 ----
def fig_delay_dtd():
    N = int(1.6 * FS); taps = 128; delay = 300
    h = np.exp(-np.arange(taps) / 25) * np.random.default_rng(3001).standard_normal(taps)
    x = colored_x(N, seed=3002)
    xd = causal_delay(x, delay)
    echo = np.convolve(xd, h, mode="full")[:N]
    s = np.zeros(N); s[int(0.8*FS):int(1.2*FS)] = 0.7 * np.random.default_rng(3003).standard_normal(int(0.4*FS))
    d = echo + s
    freeze = np.zeros(N, bool); freeze[int(0.8*FS):int(1.2*FS)] = True
    e_align, _, _ = nlms_run(xd, d, taps, 0.5, freeze)
    e_mis, _, _ = nlms_run(x, d, taps, 0.5, freeze)
    e_nof, _, _ = nlms_run(xd, d, taps, 0.5, None)
    tcc, er_align = block_erle(echo, e_align)
    _, er_mis = block_erle(echo, e_mis)
    _, er_nof = block_erle(echo, e_nof)
    invalid_intervals = [(0.8, 1.2)]
    er_align_view = mask_metric_intervals(tcc, er_align, invalid_intervals)
    er_mis_view = mask_metric_intervals(tcc, er_mis, invalid_intervals)
    er_nof_view = mask_metric_intervals(tcc, er_nof, invalid_intervals)
    pa = float(np.nanmean(er_align[(tcc > 0.45) & (tcc < 0.8)]))
    pm = float(np.nanmean(er_mis[(tcc > 0.45) & (tcc < 0.8)]))
    fig = plt.figure(figsize=(9.8, 8.0), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(0.92, 1.08))
    axes = [
        fig.add_subplot(grid[0, :]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]
    fig.suptitle("图30 延迟对齐与合成近端干扰时的冻结", fontsize=FS_SUP)
    ax = axes[0]; ax.set_title("(a) 参考—麦克风有符号互相关", fontsize=FS_TITLE)
    seg = 4000
    corr = np.correlate(d[:seg], x[:seg], mode="full")
    lags = np.arange(len(corr)) - (seg - 1)
    corr = corr / np.max(np.abs(corr))
    tau_hat = int(lags[np.argmax(corr)])
    ax.plot(lags, corr, color=C_PURPLE, lw=1.0)
    ax.axvline(tau_hat, color=C_PURPLE, ls="--")
    peak_value = float(corr[np.argmax(corr)])
    ax.annotate(
        f"最大正峰约 {tau_hat} 采样",
        xy=(tau_hat, peak_value),
        xytext=(tau_hat + 55, 0.72),
        fontsize=FS_SMALL + 1,
        color=C_PURPLE,
        arrowprops=dict(arrowstyle="->", color=C_PURPLE),
    )
    ax.text(0.03, 0.06, "设置的纯延迟为300采样；\n有色参考与多径会移动相关峰",
            transform=ax.transAxes, fontsize=FS_SMALL, color="0.25",
            bbox=dict(fc="white", ec="0.7", alpha=0.9))
    ax.set_xlim(-50, delay + 200); ax.set_xlabel("滞后（采样）", fontsize=FS_LABEL)
    ax.set_ylabel("归一化有符号互相关", fontsize=FS_LABEL)
    ax.grid(ls=":", alpha=0.5); ax.tick_params(labelsize=FS_TINY)
    ax = axes[1]; ax.set_title("(b) 延迟对齐：300 采样 > 128 抽头", fontsize=FS_TITLE)
    ax.plot(tcc, er_align_view, color=C_BLUE, lw=1.6, label=f"对齐（单讲约{pa:.0f}dB）")
    ax.plot(tcc, er_mis_view, color="0.6", lw=1.4, ls="--", label=f"不对齐（单讲约{pm:.0f}dB）")
    ax.axvspan(0.8, 1.2, color=C_RED, alpha=0.08)
    ax.set_ylim(-10, 40); ax.set_xlabel("时间 (s)", fontsize=FS_LABEL); ax.set_ylabel("ERLE (dB)", fontsize=FS_LABEL)
    ax.legend(fontsize=FS_SMALL); ax.grid(ls=":", alpha=0.5); ax.tick_params(labelsize=FS_TINY)
    ax = axes[2]; ax.set_title("(c) 近端干扰时的更新与恢复", fontsize=FS_TITLE)
    ax.plot(tcc, er_align_view, color=C_BLUE, lw=1.6, label="干扰时冻结更新")
    ax.plot(tcc, er_nof_view, color=C_RED, lw=1.4, ls="--", label="干扰时继续更新")
    ax.axvspan(0.8, 1.2, color=C_RED, alpha=0.10)
    ax.text(1.0, -5, "合成近端干扰段（非语音）\nERLE 不用于评估回声", ha="center", fontsize=FS_SMALL, color=C_RED)
    ax.set_ylim(-10, 40); ax.set_xlabel("时间 (s)", fontsize=FS_LABEL); ax.set_ylabel("ERLE (dB)", fontsize=FS_LABEL)
    ax.legend(fontsize=FS_SMALL); ax.grid(ls=":", alpha=0.5); ax.tick_params(labelsize=FS_TINY)
    save(fig, "fig30_aec_delay_dtd.png")
    print(f"对齐约 {pa:.1f} dB，不对齐约 {pm:.1f} dB")

# ---- 图31：非线性失配 ----
def fig_nonlinear():
    N = int(3.0 * FS); taps = 256
    r = np.random.default_rng(3101)
    h = np.exp(-np.arange(taps) / 60) * r.standard_normal(taps)
    x = colored_x(N, seed=3102); x = x / np.std(x) * 0.5
    echo_lin = np.convolve(x, h, mode="full")[:N]
    drive = 1.1
    echo_nl = np.convolve(np.tanh(drive * x) / np.tanh(drive), h, mode="full")[:N]
    e1, _, _ = nlms_run(x, echo_lin, taps, 0.3, None)
    e2, _, _ = nlms_run(x, echo_nl, taps, 0.3, None)
    t1, er1 = block_erle(echo_lin, e1, blk=800)
    t2, er2 = block_erle(echo_nl, e2, blk=800)
    p1 = float(np.nanmean(er1[t1 > 2.2])); p2 = float(np.nanmean(er2[t2 > 2.2]))
    fig = plt.figure(figsize=(9.8, 5.8), layout="constrained")
    fig.suptitle("图31 本仿真中的线性匹配与非线性失配", fontsize=FS_SUP)
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2.2, 1])
    ax = fig.add_subplot(gs[0]); ax.set_title("(a) 线性路径与非线性路径", fontsize=FS_TITLE)
    ax.plot(t1, er1, color=C_BLUE, ls="-", marker="o", markevery=8,
            label=f"线性路径（本次仿真后段均值 {p1:.0f} dB）")
    ax.plot(t2, er2, color=C_RED, lw=2.2, ls="--", marker="s", markevery=8,
            label=f"tanh 非线性路径（后段均值 {p2:.0f} dB）")
    ax.set_ylim(-5, 45); ax.set_xlabel("时间 (s)", fontsize=FS_LABEL); ax.set_ylabel("ERLE (dB)", fontsize=FS_LABEL)
    ax.legend(fontsize=FS_SMALL, loc="upper left"); ax.grid(ls=":", alpha=0.5); ax.tick_params(labelsize=FS_TINY)
    ax.text(0.04, 0.06, "参数：256 抽头，μ=0.3，驱动系数 1.1。\n数值只描述本次仿真，不代表统一性能上限。",
            transform=ax.transAxes, fontsize=FS_SMALL, color="0.25",
            bbox=dict(fc="white", ec="0.7", alpha=0.9))
    ax = fig.add_subplot(gs[1]); ax.set_title("(b) tanh 非线性映射", fontsize=FS_TITLE)
    xx = np.linspace(-2, 2, 400)
    ax.plot(xx, xx, color="0.6", ls="--", label="线性")
    ax.plot(xx, np.tanh(1.1*xx)/np.tanh(1.1), color=C_RED, ls="-",
            marker="o", markevery=40, label="tanh 软削波")
    ax.axvspan(1, 2, color=C_RED, alpha=0.08); ax.axvspan(-2, -1, color=C_RED, alpha=0.08)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("归一化输入幅度（无量纲）", fontsize=FS_LABEL)
    ax.set_ylabel("归一化输出幅度（无量纲）", fontsize=FS_LABEL)
    ax.legend(fontsize=FS_SMALL); ax.grid(ls=":", alpha=0.5); ax.tick_params(labelsize=FS_TINY)
    ax.text(
        0.5,
        0.04,
        "输入幅度增大后，输出逐渐\n偏离线性关系",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=FS_SMALL,
        color=C_RED,
        bbox=dict(fc="white", ec="none", alpha=0.82, pad=1.5),
    )
    save(fig, "fig31_aec_nonlinear.png")

# ---- 图32：混合流程与条件化选型 ----
def fig_hybrid():
    fig = plt.figure(figsize=(9.8, 5.8), layout="constrained")
    fig.suptitle("图32 线性回声消除与残余抑制：按系统条件选择模块", fontsize=FS_SUP)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("混合处理流程：线性前级与残余后处理各有适用前提", fontsize=FS_TITLE)
    ax.set_xlim(0, 14); ax.set_ylim(-0.8, 6.8); ax.axis("off")
    def box(x, y, w, h, text, fc, fs=FS_SMALL):
        ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="k", lw=1.2))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs, color=C_MAIN)
    def arrow(a, b, c="k", w=1.4):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, color=c, lw=w))
    # 音频主链从麦克风信号开始；远端参考是线性 AEC 的第二路输入。
    box(0.2, 3.2, 2.0, 1.6, "麦克风信号 d(n)\n近端+回声+噪声", "#f6dbdb")
    box(2.8, 3.2, 2.2, 1.6, "线性 AEC\nPBFDAF / FDKF", "#dbe9f6")
    box(5.5, 3.2, 1.8, 1.6, "线性残差 e(n)", "#dbe9f6")
    box(7.8, 3.2, 2.4, 1.6, "学习型残余抑制\n输入可含 e 与 x", "#fde3c8")
    box(10.7, 3.2, 2.0, 1.6, "处理后输出\n→语音识别", "#e8f6db")
    for a, b in [((2.2, 4.0), (2.8, 4.0)), ((5.0, 4.0), (5.5, 4.0)),
                 ((7.3, 4.0), (7.8, 4.0)), ((10.2, 4.0), (10.7, 4.0))]:
        arrow(a, b, C_BLUE)
    box(0.2, 0.9, 1.9, 1.2, "远端参考 x(n)\n从播放链路末端取", "#f6e5db")
    box(2.6, 0.9, 1.8, 1.2, "延迟估计与对齐\n得到 x_a(n)", "#dbe9f6")
    arrow((2.1, 1.5), (2.6, 1.5), C_BLUE)
    arrow((3.5, 2.1), (3.9, 3.2), C_BLUE)
    # 可选条件参考从对齐框下缘绕行，避开下方更新控制的观测与输出箭头。
    ax.plot([3.5, 3.5, 9.0, 9.0], [0.9, -0.12, -0.12, 3.2],
            color=C_BLUE, lw=1.0, ls=":")
    ax.text(9.5, 1.85, "可选：x_a 作为\n学习模型输入",
            ha="left", fontsize=FS_SMALL, color=C_BLUE)
    box(5.0, 0.35, 2.0, 0.85, "DTD 与步长控制", "#f6e5db")
    # 此框示意用对齐参考及线性残差统计来约束更新；不是唯一的 DTD 判据。
    arrow((4.4, 1.5), (5.0, 0.78), C_PURPLE)
    arrow((6.4, 3.2), (6.4, 1.2), C_PURPLE)
    arrow((6.0, 1.2), (4.5, 3.2), C_ORANGE)
    ax.text(3.4, 5.75, "线性 AEC 同时接收麦克风信号 d(n)\n和对齐参考 x_a(n)",
            fontsize=FS_SMALL, color=C_BLUE, ha="center")
    ax.text(9.6, 5.75, "学习型后处理只抑制\n训练条件覆盖的残余成分",
            fontsize=FS_SMALL, color=C_ORANGE, ha="center")
    ax.text(
        7.0,
        -0.62,
        "选型核对：参考位置与对齐  •  双讲保护  •  训练条件覆盖  •  "
        "近端保真  •  算力、内存与端到端时延",
        ha="center",
        va="bottom",
        fontsize=FS_SMALL,
        color="0.25",
        bbox=dict(fc="#f4f4f4", ec="0.7", boxstyle="round,pad=0.3"),
    )
    save(fig, "fig32_aec_hybrid_select.png")


def haar_delay_example():
    """两带正交 Haar、一拍纯延迟：返回完整与仅对角子带映射。

    分析使用相邻两点求和/差并除以 sqrt(2)，合成使用其转置。
    子带索引 m 的负时刻输入为零。仅对角映射保留精确的 G00、G11，
    丢弃 G01、G10；它不是训练后独立子带自适应器的性能结果。
    """
    x = np.array([1.0, 0.0, 0.0, 0.0])
    u = np.vstack(((x[::2] + x[1::2]) / np.sqrt(2.0),
                   (x[::2] - x[1::2]) / np.sqrt(2.0)))
    previous = np.pad(u[:, :-1], ((0, 0), (1, 0)))
    y_band = np.vstack((
        (previous[0] - previous[1] + u[0] + u[1]) / 2,
        (previous[0] - previous[1] - u[0] - u[1]) / 2,
    ))
    diagonal_band = np.vstack((
        (previous[0] + u[0]) / 2,
        -(previous[1] + u[1]) / 2,
    ))

    def synthesize(bands):
        result = np.empty(x.size)
        result[::2] = (bands[0] + bands[1]) / np.sqrt(2.0)
        result[1::2] = (bands[0] - bands[1]) / np.sqrt(2.0)
        return result

    return x, u, synthesize(y_band), synthesize(diagonal_band)


def fig_haar_crossband():
    """图37：PR 滤波器组与一拍延迟后的交叉带项分开解释。"""
    x, _u, full, diagonal = haar_delay_example()
    fig = plt.figure(figsize=(9.8, 7.5), layout="constrained")
    fig.suptitle("图37 两带 Haar：完美重建不等于逐带回声路径等价", fontsize=FS_SUP)
    grid = fig.add_gridspec(2, 1, height_ratios=(1.05, 1.0))

    ax = fig.add_subplot(grid[0])
    ax.set_title("(a) 真实一拍延迟在子带域需要同带项和交叉带项", fontsize=FS_TITLE)
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.4); ax.axis("off")

    def box(left, bottom, width, height, label, fill):
        ax.add_patch(Rectangle((left, bottom), width, height,
                               fc=fill, ec=C_MAIN, lw=1.4))
        ax.text(left + width / 2, bottom + height / 2, label,
                ha="center", va="center", fontsize=FS_SMALL, color=C_MAIN)

    def arrow(start, end, color=C_BLUE):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                     mutation_scale=15, color=color, lw=1.8))

    box(0.1, 2.05, 1.15, 0.95, "播放参考\nx[n]", "#f6e5db")
    box(1.8, 2.05, 1.55, 0.95, "两带 Haar\n分析 + ↓2", "#dbe9f6")
    box(3.9, 1.78, 2.45, 1.5, "一拍路径的子带映射\nG₀₀、G₁₁ + G₀₁、G₁₀\n（含交叉带项）", "#e8f6db")
    box(6.9, 2.05, 1.55, 0.95, "两带 Haar\n合成 + ↑2", "#dbe9f6")
    box(9.0, 2.05, 0.9, 0.95, "回声\ny[n]", "#f6dbdb")
    for start, end in [((1.25, 2.52), (1.8, 2.52)),
                       ((3.35, 2.52), (3.9, 2.52)),
                       ((6.35, 2.52), (6.9, 2.52)),
                       ((8.45, 2.52), (9.0, 2.52))]:
        arrow(start, end)
    ax.text(5.0, 1.35,
            r"$G_{01}=(1-z^{-1})/2,\quad G_{10}=(z^{-1}-1)/2$：两项一般不为零",
            ha="center", va="center", fontsize=FS_SMALL, color=C_RED)
    ax.text(5.0, 0.65,
            "分析后立即合成可完美重建 x；中间加入延迟路径后，不能只保留 G₀₀、G₁₁。",
            ha="center", va="center", fontsize=FS_SMALL, color=C_MAIN)

    ax = fig.add_subplot(grid[1])
    ax.set_title("(b) 单位脉冲 x=[1,0,0,0]，真实路径 y[n]=x[n−1]", fontsize=FS_TITLE)
    positions = np.arange(x.size)
    ax.bar(positions - 0.17, full, width=0.31, color="0.25", edgecolor="black",
           label="完整子带映射＝直接延迟")
    ax.bar(positions + 0.17, diagonal, width=0.31, color="white", edgecolor=C_BLUE,
           hatch="///", lw=1.4, label="仅保留精确同带项")
    ax.set_xticks(positions)
    ax.set_xlabel("输出采样索引 n（采样）", fontsize=FS_LABEL)
    ax.set_ylabel("输出幅度（无量纲）", fontsize=FS_LABEL)
    ax.set_ylim(-0.1, 1.25)
    ax.legend(fontsize=FS_SMALL, ncols=2, loc="upper right")
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.tick_params(labelsize=FS_TINY)
    ax.text(0.02, 0.9,
            "完整：[0,1,0,0]；仅同带：[0,0.5,0,0.5]\n"
            "后者是删去交叉项的模型示意，不是训练后的 AEC 残差。",
            transform=ax.transAxes, fontsize=FS_SMALL, va="top", color=C_MAIN,
            bbox=dict(fc="white", ec="0.7", alpha=0.95))
    save(fig, "fig37_aec_subband.png")


def adaptive_state_example():
    """图38 的精确小例数据；三种方法的纵轴不作为彼此性能比较。"""
    weights = np.array([0.8, 0.2])
    kappas = np.array([-1.0, 0.0, 1.0])
    gains = np.array([
        (1 - kappa) / (2 * weights.size)
        + (1 + kappa) * np.abs(weights) / (2 * np.sum(np.abs(weights)))
        for kappa in kappas
    ])
    p_initial = np.eye(2)
    p_after_first = np.diag([2 / 3, 2.0])
    p_after_second = np.array([[20, -16], [-16, 28]], dtype=float) / 19
    rls_matrices = np.stack((p_initial, p_after_first, p_after_second))
    # 标量 X=1；三组条件分别是常规、假定观测方差增大、假定先验方差增大。
    prior_variances = np.array([1.0, 1.0, 4.0])
    observation_variances = np.array([1.0, 10.0, 1.0])
    kalman_gains = prior_variances / (prior_variances + observation_variances)
    return kappas, gains, rls_matrices, kalman_gains


def fig_adaptive_state_examples():
    """图38：抽头分配、逆相关状态及假设统计量如何改变更新。"""
    kappas, gains, rls_matrices, kalman_gains = adaptive_state_example()
    fig = plt.figure(figsize=(9.8, 8.2), layout="constrained")
    fig.suptitle("图38 IPNLMS、RLS 与 Kalman：三种不同的更新状态", fontsize=FS_SUP)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.15))

    ax = fig.add_subplot(grid[0, :])
    ax.set_title("(a) IPNLMS：同一权重 ŵ=[0.8,0.2]，κ 改变两个抽头的更新份额", fontsize=FS_TITLE)
    positions = np.arange(kappas.size)
    ax.bar(positions - 0.18, gains[:, 0], width=0.34, color=C_BLUE,
           edgecolor="black", label="抽头 0 的 g₀")
    ax.bar(positions + 0.18, gains[:, 1], width=0.34, color="white",
           edgecolor=C_RED, hatch="///", lw=1.4, label="抽头 1 的 g₁")
    ax.set_xticks(positions, ["κ=−1（均匀）", "κ=0（混合）", "κ=1（纯比例）"])
    ax.set_ylabel("归一化份额 g（无量纲）", fontsize=FS_LABEL)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=FS_SMALL, ncols=2)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.tick_params(labelsize=FS_TINY)
    ax.text(0.99, 0.91, "本例 εg→0；只表示单步份额，不是长期收敛结果。",
            transform=ax.transAxes, fontsize=FS_SMALL, ha="right", va="top",
            bbox=dict(fc="white", ec="0.7", alpha=0.95))

    ax = fig.add_subplot(grid[1, 0])
    ax.set_title("(b) RLS：P 是逆参考相关矩阵", fontsize=FS_TITLE)
    steps = np.arange(3)
    for row, col, label, color, style, marker in [
        (0, 0, "P₀₀", C_BLUE, "-", "o"),
        (1, 1, "P₁₁", C_RED, "--", "s"),
        (0, 1, "P₀₁", C_GREEN, "-.", "^"),
    ]:
        ax.plot(steps, rls_matrices[:, row, col], color=color,
                ls=style, marker=marker, lw=1.8, ms=7, label=label)
    ax.set_xticks(steps, ["初始", "第 0 步后", "第 1 步后"])
    ax.set_ylabel("P 元素（本例无量纲）", fontsize=FS_LABEL)
    ax.set_ylim(-1.05, 2.6)
    ax.legend(fontsize=FS_SMALL, ncols=3, loc="upper right")
    ax.grid(ls=":", alpha=0.5)
    ax.tick_params(labelsize=FS_TINY)
    ax.text(0.02, 0.08,
            r"$\lambda=1/2,\ P_{-1}=I,\ \mathbf{x}_0=[1,0],\ \mathbf{x}_1=[1,1]$",
            transform=ax.transAxes, fontsize=FS_SMALL, va="bottom",
            bbox=dict(fc="white", ec="0.7", alpha=0.95))

    ax = fig.add_subplot(grid[1, 1])
    ax.set_title("(c) Kalman：已给定方差时的增益", fontsize=FS_TITLE)
    labels = ["基准\n" + r"$P^-=1,\ \Psi=1$",
              "假定双讲\n" + r"$P^-=1,\ \Psi=10$",
              "假定突变\n" + r"$P^-=4,\ \Psi=1$"]
    bars = ax.bar(np.arange(3), kalman_gains, width=0.58,
                  color=["0.35", "white", "white"], edgecolor="black",
                  hatch=["", "///", "xxx"], lw=1.4)
    for bar, gain in zip(bars, kalman_gains):
        ax.text(bar.get_x() + bar.get_width() / 2, gain + 0.035,
                f"{gain:.3f}", ha="center", fontsize=FS_SMALL)
    ax.set_xticks(np.arange(3), labels)
    ax.set_ylabel("K，X=1（无量纲）", fontsize=FS_LABEL)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.tick_params(labelsize=FS_TINY)
    ax.text(0.03, 0.95, "Ψ 与 " + r"$P^-$" + " 均人为指定；本图不实现双讲或突变检测。",
            transform=ax.transAxes, fontsize=FS_SMALL, va="top",
            bbox=dict(fc="white", ec="0.7", alpha=0.95))
    save(fig, "fig38_aec_four_methods.png")


def fig_pbfdaf_flow():
    """图39：一块 PBFDAF 的先验预测与更新；两种半块操作分开画。"""
    fig = plt.figure(figsize=(9.8, 10.2), layout="constrained")
    fig.suptitle("图39 PBFDAF：参考历史、有效输出与受控权重更新", fontsize=FS_SUP)
    grid = fig.add_gridspec(2, 1, height_ratios=(1.02, 1.0))
    prediction_ax = fig.add_subplot(grid[0])
    update_ax = fig.add_subplot(grid[1])

    def setup(axis, title):
        axis.set_title(title, fontsize=FS_TITLE, pad=11)
        axis.set_xlim(0, 14)
        axis.set_ylim(0, 6.4)
        axis.axis("off")

    def box(axis, x, y, width, height, label, fill, *, edge=C_MAIN):
        axis.add_patch(Rectangle((x, y), width, height, fc=fill, ec=edge, lw=1.25))
        axis.text(x + width / 2, y + height / 2, label,
                  ha="center", va="center", fontsize=FS_SMALL, color=C_MAIN)

    def arrow(axis, start, end, *, color=C_BLUE, style="-", width=1.7):
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                        mutation_scale=14, color=color,
                                        lw=width, linestyle=style))

    setup(prediction_ax, "(a) 旧权重先预测；仅循环卷积的后 N 点是本块有效输出")
    box(prediction_ax, 0.2, 4.8, 1.7, 0.95,
        "播放参考 $x[n]$\n已对齐本块 $b_m$", "#f6e5db")
    box(prediction_ax, 2.25, 4.8, 2.4, 0.95,
        "线性等效路径 $h$\n扬声器—房间—麦克风", "#dbe9f6")
    box(prediction_ax, 5.0, 4.8, 1.55, 0.95, "真实回声\n$r_m$", "#f6dbdb")
    box(prediction_ax, 6.9, 4.8, 2.05, 0.95,
        "麦克风叠加\n$r_m+s_m+v_m$", "#f6dbdb")
    box(prediction_ax, 9.3, 4.8, 1.35, 0.95, "麦克风\n$d_m$", "#f6dbdb")
    for start, end in [((1.9, 5.275), (2.25, 5.275)),
                       ((4.65, 5.275), (5.0, 5.275)),
                       ((6.55, 5.275), (6.9, 5.275)),
                       ((8.95, 5.275), (9.3, 5.275))]:
        arrow(prediction_ax, start, end)
    prediction_ax.text(7.925, 6.02, "近端 $s_m$、噪声 $v_m$",
                       ha="center", fontsize=FS_SMALL, color=C_RED)
    arrow(prediction_ax, (7.925, 5.94), (7.925, 5.75), color=C_RED)

    box(prediction_ax, 0.2, 2.35, 2.05, 0.96,
        "上一块＋本块\n$u_m=[b_{m-1},b_m]$", "#f6e5db")
    box(prediction_ax, 2.47, 2.35, 1.1, 0.96,
        "$2N$ 点\nFFT", "#dbe9f6")
    box(prediction_ax, 3.75, 2.35, 2.25, 0.96,
        "参考谱移位历史\n" r"$X_m,\ldots,X_{m-P+1}$", "#dbe9f6")
    box(prediction_ax, 6.35, 2.35, 2.68, 0.96,
        "各分区乘积求和\n" r"$Q_m=\sum_p W_pX_{m-p}$", "#e8f6db")
    box(prediction_ax, 9.35, 2.35, 1.1, 0.96,
        "$2N$ 点\nIFFT", "#dbe9f6")
    box(prediction_ax, 10.78, 2.35, 2.1, 0.96,
        "舍弃前 $N$ 点\n取后 $N$ 点 " r"$\hat y_m$", "#e8f6db")
    for start, end in [((2.25, 2.83), (2.47, 2.83)),
                       ((3.57, 2.83), (3.75, 2.83)),
                       ((6.0, 2.83), (6.35, 2.83)),
                       ((9.03, 2.83), (9.35, 2.83)),
                       ((10.45, 2.83), (10.78, 2.83))]:
        arrow(prediction_ax, start, end)
    # 同一个播放参考分出声学回路和算法参考；移位历史不受冻结控制。
    arrow(prediction_ax, (1.05, 4.8), (1.05, 3.31))
    box(prediction_ax, 6.6, 3.72, 2.18, 0.67,
        "本块开始的旧权重 $W_p(m)$", "#f3f0fa")
    arrow(prediction_ax, (7.69, 3.72), (7.69, 3.31), color=C_PURPLE)
    prediction_ax.add_patch(Circle((11.2, 0.98), 0.30,
                                   fc="white", ec=C_MAIN, lw=1.35))
    prediction_ax.text(11.2, 0.98, "−", ha="center", va="center",
                       fontsize=FS_LABEL + 4, color=C_MAIN)
    prediction_ax.text(12.05, 0.98, r"$e_m=d_m-\hat y_m$",
                       va="center", fontsize=FS_SMALL, color=C_MAIN)
    arrow(prediction_ax, (11.8, 2.35), (11.35, 1.25), color=C_GREEN)
    prediction_ax.plot([10.65, 13.45, 13.45, 11.2],
                       [5.275, 5.275, 1.68, 1.68], color=C_RED, lw=1.45)
    arrow(prediction_ax, (11.2, 1.68), (11.2, 1.29), color=C_RED)
    arrow(prediction_ax, (11.5, 0.98), (11.95, 0.98), color=C_GREEN)
    prediction_ax.text(0.2, 0.28,
                       "频谱符号省略频点 $k$；前 $N$ 点只因循环折回而弃用。\n"
                       "等效路径 $h$ 含器件的小信号响应，不等于纯房间脉冲响应。",
                       fontsize=FS_SMALL, color="0.28", va="bottom")

    setup(update_ax, "(b) 有效误差驱动下一块的候选权重；冻结只阻止更新")
    box(update_ax, 0.2, 4.72, 1.55, 0.88,
        "有效误差\n$e_m$", "#e8f6db")
    box(update_ax, 2.12, 4.72, 2.0, 0.88,
        "前置 $N$ 个零\n$[0_N,e_m]$", "#dbe9f6")
    box(update_ax, 4.48, 4.72, 1.58, 0.88,
        "$2N$ 点 FFT\n得到 $E_m$", "#dbe9f6")
    arrow(update_ax, (1.75, 5.16), (2.12, 5.16), color=C_GREEN)
    arrow(update_ax, (4.12, 5.16), (4.48, 5.16), color=C_GREEN)

    box(update_ax, 0.2, 2.7, 1.92, 0.99,
        "已移位参考历史\n" r"$X_m,\ldots,X_{m-P+1}$", "#dbe9f6")
    box(update_ax, 2.5, 2.7, 2.43, 0.99,
        "共轭与逐频功率\n" r"$D=\sum_q|X_{m-q}|^2+\delta$", "#dbe9f6")
    box(update_ax, 5.28, 2.7, 1.57, 0.99,
        "候选更新量\n" r"$\mu X_{m-p}^*E_m/D$", "#e8f6db")
    box(update_ax, 7.2, 2.7, 1.78, 0.99,
        "加旧权重\n" r"$\widetilde W_p$", "#e8f6db")
    box(update_ax, 9.36, 2.7, 2.30, 0.99,
        "若约束：IFFT→清零\n后 $N$ 点→FFT", "#f3f0fa")
    box(update_ax, 12.0, 2.7, 1.65, 0.99,
        "下一块权重\n$W_p(m+1)$", "#e8f6db")
    for start, end in [((2.12, 3.195), (2.5, 3.195)),
                       ((4.93, 3.195), (5.28, 3.195)),
                       ((6.85, 3.195), (7.2, 3.195)),
                       ((8.98, 3.195), (9.36, 3.195)),
                       ((11.66, 3.195), (12.0, 3.195))]:
        arrow(update_ax, start, end)
    arrow(update_ax, (5.27, 4.72), (5.75, 3.69), color=C_GREEN)
    box(update_ax, 7.25, 1.20, 1.73, 0.67,
        "旧权重 $W_p(m)$", "#f3f0fa")
    arrow(update_ax, (8.115, 1.87), (8.115, 2.7), color=C_PURPLE)
    box(update_ax, 7.45, 4.72, 2.20, 0.88,
        "外部冻结控制\n仅关闭权重更新", "#f6e5db")
    arrow(update_ax, (8.05, 4.72), (6.13, 3.69),
          color=C_PURPLE, style="--", width=1.8)
    # 不执行投影时直接提交候选；冻结时保留旧权重，均不停止参考历史移位。
    update_ax.plot([8.98, 10.52, 12.83], [2.88, 1.86, 1.86],
                   color="0.35", lw=1.3, linestyle="--")
    arrow(update_ax, (12.83, 1.86), (12.83, 2.7),
          color="0.35", style="--", width=1.3)
    update_ax.text(10.55, 1.48, "不约束：直接提交候选",
                   ha="center", fontsize=FS_SMALL, color="0.28")
    update_ax.text(0.2, 0.55,
                   "冻结：$W_p(m+1)=W_p(m)$，参考历史仍移位；末分区不足 $N$ 个真实抽头时，补位也清零。",
                   fontsize=FS_SMALL, color="0.28")
    save(fig, "fig39_aec_pbfdaf_flow.png")

if __name__ == "__main__":
    fig_problem(); fig_concept(); fig_nlms(); fig_erle(); fig_delay_dtd(); fig_nonlinear(); fig_hybrid()
    fig_haar_crossband(); fig_adaptive_state_examples(); fig_pbfdaf_flow()
    print("ALL DONE")
