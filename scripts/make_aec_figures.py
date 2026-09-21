# -*- coding: utf-8 -*-
"""回声消除专题插图（入门导向，7 张）：图 26~32。

用法（仓库根目录）：
    .venv/bin/python scripts/make_aec_figures.py  # 图 26~32 → figures/
函数与图号对照：fig_problem→图26、fig_concept→图27、fig_nlms→图28、
fig_erle→图29、fig_delay_dtd→图30、fig_nonlinear→图31、fig_hybrid→图32。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib import gridspec
from pathlib import Path

OUT = Path(os.environ.get("AEC_FIGURE_DIR", Path(__file__).parent.parent / "figures"))
OUT.mkdir(exist_ok=True)
C_MAIN, C_BLUE, C_RED, C_GREEN, C_ORANGE, C_PURPLE = "#1a1a2e", "#2f6db3", "#c0392b", "#2e8b57", "#e67e22", "#7d3c98"
FS_SUP, FS_TITLE, FS_LABEL, FS_SMALL, FS_TINY = 13, 11, 10, 9, 8
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
FS = 16000

def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)

def synth_h(taps=256, decay=60, seed=7):
    r = np.random.default_rng(seed)
    return np.exp(-np.arange(taps) / decay) * r.standard_normal(taps)

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

def nlms_run(x, d, taps=128, mu=0.5, freeze=None):
    N = len(x)
    w = np.zeros(taps)
    e = np.zeros(N)
    snaps, mismatch = [], []
    X = np.zeros(taps)
    for n in range(N):
        X = np.roll(X, 1); X[0] = x[n]
        y = float(w @ X)
        e[n] = d[n] - y
        if freeze is not None and freeze[n]:
            pass
        else:
            w = w + mu * e[n] * X / (float(X @ X) + 1e-6)
        if n % 400 == 0:
            snaps.append(w.copy())
    return e, np.array(snaps), w

def block_erle(echo, e, blk=400):
    t = np.arange(len(echo)) / FS
    te = np.array([np.mean(echo[i:i+blk] ** 2) for i in range(0, len(echo) - blk, blk)])
    re = np.array([np.mean(e[i:i+blk] ** 2) for i in range(0, len(e) - blk, blk)])
    tc = np.array([i / FS for i in range(0, len(e) - blk, blk)])
    with np.errstate(divide="ignore"):
        erle = 10 * np.log10(te / np.maximum(re, 1e-12))
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

# ---- 图1：回声是怎么产生的 ----
def fig_problem():
    N = int(1.6 * FS)
    r = np.random.default_rng(2601)
    x = colored_x(N, seed=2602); h = synth_h(seed=2603)
    echo = np.convolve(x, h, mode="full")[:N] * 0.5
    s = np.zeros(N); s[int(0.8*FS):int(1.2*FS)] = 0.7 * r.standard_normal(int(0.4*FS))
    v = 0.02 * r.standard_normal(N)
    d = echo + s + v
    t = np.arange(N) / FS
    fig = plt.figure(figsize=(13.5, 5.2), layout="constrained")
    fig.suptitle("图26 回声形成：d(n) = s(n) + x(n)*h(n) + v(n)", fontsize=FS_SUP)
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1.4, 1.4])
    ax = fig.add_subplot(gs[0])
    ax.set_title("(a) 用脉冲响应 h(n) 表示房间回声路径", fontsize=FS_TITLE)
    ax.stem(h[:80], linefmt=C_BLUE, markerfmt="o", basefmt=" ", label="h抽头")
    ax.set_xlabel("抽头 n", fontsize=FS_LABEL); ax.set_ylabel("归一化幅度", fontsize=FS_LABEL)
    ax.annotate("前50抽头能量最大\n直达+早期反射", xy=(10, h[10]), xytext=(55, 0.85),
                fontsize=FS_SMALL + 1, arrowprops=dict(arrowstyle="->", color=C_ORANGE), color=C_ORANGE)
    ax.grid(ls=":", alpha=0.5); ax.tick_params(labelsize=FS_TINY)
    ax = fig.add_subplot(gs[1])
    ax.set_title("(b) 卷积把声音拖长：x → 回声", fontsize=FS_TITLE)
    ax.plot(t, x, color=C_BLUE, lw=1, label="远端 x(n)")
    ax.plot(t, echo, color=C_BLUE, ls="--", lw=1, label="回声 x*h")
    ax.plot(t, s, color=C_RED, lw=1, label="近端 s(n)")
    ax.axvspan(0.8, 1.2, color=C_RED, alpha=0.10)
    ax.text(1.0, ax.get_ylim()[1]*0.9 if len(ax.get_ylim()) else 1, "近端说话段", ha="center", fontsize=FS_SMALL, color=C_RED)
    ax.set_xlim(0, 1.6); ax.set_xlabel("时间 (s)", fontsize=FS_LABEL); ax.set_ylabel("归一化幅度", fontsize=FS_LABEL)
    ax.legend(fontsize=FS_SMALL); ax.grid(ls=":", alpha=0.5); ax.tick_params(labelsize=FS_TINY)
    ax = fig.add_subplot(gs[2])
    ax.set_title("(c) 麦克风里三者相加得 d(n)", fontsize=FS_TITLE)
    ax.plot(t, d, color=C_MAIN, lw=1, label="麦克风 d(n)")
    ax.plot(t, echo, color=C_BLUE, lw=0.8, alpha=0.7, label="回声分量")
    ax.axvspan(0.8, 1.2, color=C_RED, alpha=0.10)
    ax.set_xlim(0, 1.6); ax.set_xlabel("时间 (s)", fontsize=FS_LABEL); ax.set_ylabel("归一化幅度", fontsize=FS_LABEL)
    ax.legend(fontsize=FS_SMALL); ax.grid(ls=":", alpha=0.5); ax.tick_params(labelsize=FS_TINY)
    save(fig, "fig26_aec_problem.png")

# ---- 图2：一句话结构 ----
def fig_concept():
    fig, ax = plt.subplots(figsize=(13.5, 3.8), layout="constrained")
    fig.suptitle("图27 AEC 结构：根据播放参考估计回声，再从麦克风信号中相减", fontsize=FS_SUP)
    ax.set_xlim(0, 11); ax.set_ylim(0, 6); ax.axis("off")
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
    # 残差 e 沿底部返回滤波器更新端；DTD 控制反馈支路上的开关。
    ax.plot([9.9, 9.9, 9.15], [2.5, 0.18, 0.18], color=C_BLUE, lw=1.4)
    ax.add_patch(Rectangle((8.35, 0.03), 0.8, 0.3, fc="white", ec=C_RED, lw=1.2, zorder=4))
    ax.plot([8.48, 8.98], [0.29, 0.08], color=C_RED, lw=1.6, zorder=5)
    ax.plot([3.7, 8.35], [0.18, 0.18], color=C_BLUE, lw=1.4)
    arrow((3.7, 0.18), (3.7, 0.4), C_BLUE)
    ax.text(8.4, 0.36, "残差 e 驱动更新", fontsize=FS_TINY, color=C_BLUE, ha="center")
    arrow((8.8, 4.6), (8.75, 0.33), C_RED, ls="--", w=1.8)
    ax.annotate("估计回声路径", xy=(3.7, 1.0), xytext=(0.7, 0.35), fontsize=FS_SMALL + 2, color=C_PURPLE,
                arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=1.8, connectionstyle="arc3,rad=-0.25"))
    save(fig, "fig27_aec_concept.png")

# ---- 图3：NLMS 路径估计 ----
def fig_nlms():
    N = int(1.6 * FS); taps = 128
    h = np.exp(-np.arange(taps) / 25) * np.random.default_rng(2801).standard_normal(taps)
    x = colored_x(N, seed=2802); d = np.convolve(x, h, mode="full")[:N]
    e, snaps, w = nlms_run(x, d, taps, 0.5, None)
    tc = np.arange(0, N, 400) / FS
    mism = []
    ww = np.zeros(taps); X = np.zeros(taps)
    for n in range(N):
        X = np.roll(X, 1); X[0] = x[n]
        ww = ww + 0.5 * (d[n] - float(ww @ X)) * X / (float(X @ X) + 1e-6)
        if n % 400 == 0:
            mism.append(10*np.log10(np.sum((h-ww)**2)/np.sum(h**2)))
    mism = np.array(mism)
    tcc, erle = block_erle(d, e)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), layout="constrained")
    fig.suptitle("图28 NLMS 自适应：路径估计逐步逼近真实响应", fontsize=FS_SUP)
    ax = axes[0]; ax.set_title("(a) ŵ 逐步逼近 h", fontsize=FS_TITLE)
    ax.plot(h[:64], color="0.45", lw=2, label="真实路径 h")
    cb_styles = [(C_BLUE, "-", "深蓝实线"), (C_ORANGE, "--", "橙色虚线"),
                 (C_GREEN, "-.", "绿色点划线"), (C_PURPLE, ":", "紫色点线")]
    pick = [0, len(snaps)//3, 2*len(snaps)//3, len(snaps)-1]
    for j, (c, ls, _nm) in zip(pick, cb_styles):
        ax.plot(snaps[j][:64], color=c, ls=ls, lw=1.6, label=f"{j*400}步")
    ax.set_xlabel("抽头", fontsize=FS_LABEL); ax.legend(fontsize=FS_SMALL)
    ax.grid(ls=":", alpha=0.5); ax.tick_params(labelsize=FS_TINY)
    ax = axes[1]; ax.set_title("(b) 失配下降，ERLE 上升", fontsize=FS_TITLE)
    ax.plot(tc[:len(mism)], mism, color=C_RED, label="失配(dB)")
    ax.set_xlabel("时间 (s)", fontsize=FS_LABEL); ax.tick_params(labelsize=FS_TINY)
    ax2 = ax.twinx(); ax2.plot(tcc, erle, color=C_BLUE, label="ERLE(dB)")
    ax.tick_params(labelsize=FS_TINY, colors=C_RED)
    ax2.tick_params(labelsize=FS_TINY, colors=C_BLUE)
    ax.set_ylabel("失配 (dB)", fontsize=FS_LABEL, color=C_RED)
    ax2.set_ylabel("ERLE (dB)", fontsize=FS_LABEL, color=C_BLUE)
    ax.grid(ls=":", alpha=0.5)
    ax.text(0.99, -6.0, "本仿真中：失配每下降 10 dB，ERLE 约上升 10 dB", fontsize=FS_SMALL, color=C_MAIN, ha="right",
            bbox=dict(fc="white", ec="0.7", alpha=0.9))
    ax = axes[2]; ax.set_title("(c) 步长 μ：增大可加快收敛，也会增大波动", fontsize=FS_TITLE)
    for mu, c in [(0.2, C_BLUE), (0.5, C_ORANGE), (1.0, C_RED)]:
        ee, _, _ = nlms_run(x, d, taps, mu, None)
        _, er = block_erle(d, ee)
        ax.plot(tcc[:len(er)], er, color=c, label=f"μ={mu}")
    ax.set_xlabel("时间 (s)", fontsize=FS_LABEL); ax.set_ylabel("ERLE (dB)", fontsize=FS_LABEL)
    ax.legend(fontsize=FS_SMALL); ax.grid(ls=":", alpha=0.5); ax.tick_params(labelsize=FS_TINY)
    ax.text(0.05, 0.08, "本图只比较 μ=0.2、0.5、1.0。\n"
            "实际步长需按输入相关性、失配和双讲条件验证。",
            transform=ax.transAxes, fontsize=FS_SMALL, color=C_RED,
            bbox=dict(fc="white", ec="0.8", alpha=0.85))
    save(fig, "fig28_aec_nlms.png")

# ---- 图4：ERLE + 双讲冻结 ----
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
    fig = plt.figure(figsize=(13.5, 8.2), layout="constrained")
    fig.suptitle("图29 NLMS 收敛与双讲冻结", fontsize=FS_SUP)
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
    ax.plot(tcc, erle_view, color=C_BLUE, lw=1.6, label="ERLE（双讲区不绘制）")
    ax.axvspan(0.8, 1.2, color=C_RED, alpha=0.10)
    ax.text(1.0, np.nanmax(erle_view)*0.88, "双讲区的残差含近端语音\nERLE 不能评价回声抵消量", ha="center", fontsize=FS_SMALL + 1, color=C_RED,
            bbox=dict(fc="white", ec=C_RED, lw=0.7, alpha=0.9, boxstyle="round,pad=0.3"))
    ax.annotate(f"收敛后 ERLE≈{plateau:.0f} dB", xy=(0.62, plateau), xytext=(0.35, plateau+6),
                fontsize=FS_LABEL, color=C_BLUE,
                arrowprops=dict(arrowstyle="->", color=C_BLUE))
    ax.set_xlim(0, 1.6); ax.set_xlabel("时间 (s)", fontsize=FS_LABEL); ax.set_ylabel("ERLE (dB)", fontsize=FS_LABEL)
    ax.legend(fontsize=FS_SMALL, loc="lower right")
    ax.grid(ls=":", alpha=0.5); ax.tick_params(labelsize=FS_TINY)
    save(fig, "fig29_aec_erle_freeze.png")

# ---- 图5：延迟 × 双讲 ----
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
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), layout="constrained")
    fig.suptitle("图30 延迟对齐与双讲冻结", fontsize=FS_SUP)
    ax = axes[0]; ax.set_title("(a) 互相关找延迟", fontsize=FS_TITLE)
    seg = 4000
    corr = np.correlate(d[:seg], x[:seg], mode="full")
    lags = np.arange(len(corr)) - (seg - 1)
    corr = corr / np.max(np.abs(corr))
    tau_hat = int(lags[np.argmax(corr)])
    ax.plot(lags, corr, color=C_PURPLE, lw=1.0)
    ax.axvline(tau_hat, color=C_PURPLE, ls="--")
    ax.annotate(f"峰值约{tau_hat}采样", xy=(tau_hat, 0.55), xytext=(tau_hat, 0.68), fontsize=FS_SMALL + 1.5, color=C_PURPLE)
    ax.set_xlim(-50, delay + 200); ax.set_xlabel("延迟采样", fontsize=FS_LABEL)
    ax.grid(ls=":", alpha=0.5); ax.tick_params(labelsize=FS_TINY)
    ax = axes[1]; ax.set_title("(b) 对齐与不对齐（延迟 300 抽头 > 滤波器 128 抽头）", fontsize=FS_TITLE)
    ax.plot(tcc, er_align_view, color=C_BLUE, lw=1.6, label=f"对齐（单讲约{pa:.0f}dB）")
    ax.plot(tcc, er_mis_view, color="0.6", lw=1.4, ls="--", label=f"不对齐（单讲约{pm:.0f}dB）")
    ax.axvspan(0.8, 1.2, color=C_RED, alpha=0.08)
    ax.set_ylim(-10, 40); ax.set_xlabel("时间 (s)", fontsize=FS_LABEL); ax.set_ylabel("ERLE (dB)", fontsize=FS_LABEL)
    ax.legend(fontsize=FS_SMALL); ax.grid(ls=":", alpha=0.5); ax.tick_params(labelsize=FS_TINY)
    ax = axes[2]; ax.set_title("(c) 双讲区不计算 ERLE；比较双讲后的恢复", fontsize=FS_TITLE)
    ax.plot(tcc, er_align_view, color=C_BLUE, lw=1.6, label="双讲时冻结更新")
    ax.plot(tcc, er_nof_view, color=C_RED, lw=1.4, ls="--", label="双讲时继续更新")
    ax.axvspan(0.8, 1.2, color=C_RED, alpha=0.10)
    ax.text(1.0, -5, "残差含近端语音\nERLE 无定义", ha="center", fontsize=FS_SMALL, color=C_RED)
    ax.set_ylim(-10, 40); ax.set_xlabel("时间 (s)", fontsize=FS_LABEL)
    ax.legend(fontsize=FS_SMALL); ax.grid(ls=":", alpha=0.5); ax.tick_params(labelsize=FS_TINY)
    save(fig, "fig30_aec_delay_dtd.png")
    print(f"对齐约 {pa:.1f} dB，不对齐约 {pm:.1f} dB")

# ---- 图6：非线性失配 ----
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
    fig = plt.figure(figsize=(13.5, 5.0), layout="constrained")
    fig.suptitle("图31 本仿真中的线性匹配与非线性失配", fontsize=FS_SUP)
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2.2, 1])
    ax = fig.add_subplot(gs[0]); ax.set_title("(a) 线性路径与非线性路径", fontsize=FS_TITLE)
    ax.plot(t1, er1, color=C_BLUE, label=f"线性路径（本次仿真后段均值 {p1:.0f} dB）")
    ax.plot(t2, er2, color=C_RED, lw=2.2, label=f"tanh 非线性路径（后段均值 {p2:.0f} dB）")
    ax.set_ylim(-5, 45); ax.set_xlabel("时间 (s)", fontsize=FS_LABEL); ax.set_ylabel("ERLE (dB)", fontsize=FS_LABEL)
    ax.legend(fontsize=FS_SMALL, loc="upper left"); ax.grid(ls=":", alpha=0.5); ax.tick_params(labelsize=FS_TINY)
    ax.text(0.04, 0.06, "参数：256 抽头，μ=0.3，驱动系数 1.1。\n数值只描述本次仿真，不代表统一性能上限。",
            transform=ax.transAxes, fontsize=FS_SMALL, color="0.25",
            bbox=dict(fc="white", ec="0.7", alpha=0.9))
    ax = fig.add_subplot(gs[1]); ax.set_title("(b) tanh 非线性映射", fontsize=FS_TITLE)
    xx = np.linspace(-2, 2, 400)
    ax.plot(xx, xx, color="0.6", ls="--", label="线性")
    ax.plot(xx, np.tanh(1.1*xx)/np.tanh(1.1), color=C_RED, label="tanh 软削波")
    ax.axvspan(1, 2, color=C_RED, alpha=0.08); ax.axvspan(-2, -1, color=C_RED, alpha=0.08)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("输入幅度", fontsize=FS_LABEL); ax.set_ylabel("输出幅度", fontsize=FS_LABEL)
    ax.legend(fontsize=FS_SMALL); ax.grid(ls=":", alpha=0.5); ax.tick_params(labelsize=FS_TINY)
    ax.text(0, -1.55, "输入幅度增大后，输出逐渐偏离线性关系", ha="center", fontsize=FS_SMALL + 1, color=C_RED)
    save(fig, "fig31_aec_nonlinear.png")

# ---- 图7：混合流程与条件化选型 ----
def fig_hybrid():
    fig = plt.figure(figsize=(13.5, 8.8), layout="constrained")
    fig.suptitle("图32 线性回声消除与残余抑制：按系统条件选择模块", fontsize=FS_SUP)
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.2, 1], hspace=0.4)
    ax = fig.add_subplot(gs[0]); ax.set_title("(a) 一种常见的混合处理流程", fontsize=FS_TITLE)
    ax.set_xlim(0, 14); ax.set_ylim(0.2, 6.8); ax.axis("off")
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
    box(0.2, 0.9, 1.9, 1.2, "远端参考 x(n)\n从播放链路末端取", "#f6e5db", FS_TINY)
    box(2.6, 0.9, 1.8, 1.2, "延迟估计与对齐\n得到 x_a(n)", "#dbe9f6", FS_TINY)
    arrow((2.1, 1.5), (2.6, 1.5), C_BLUE)
    arrow((3.5, 2.1), (3.9, 3.2), C_BLUE)
    # 某些学习型后处理同时使用对齐参考；虚线表示可选输入，而非必需主链。
    ax.plot([4.4, 9.0, 9.0], [1.5, 1.5, 3.2], color=C_BLUE, lw=1.0, ls=":")
    ax.text(6.8, 1.68, "可选：对齐参考 x_a(n) 作为学习模型的条件输入",
            ha="center", fontsize=FS_SMALL, color=C_BLUE)
    box(5.0, 0.35, 2.0, 0.85, "DTD 与步长控制", "#f6e5db", FS_TINY)
    arrow((6.0, 1.2), (4.5, 3.2), C_ORANGE)
    ax.text(3.9, 5.75, "线性 AEC 同时接收麦克风信号 d(n) 和对齐参考 x_a(n)",
            fontsize=FS_SMALL, color=C_BLUE, ha="center")
    ax.text(9.0, 5.75, "学习型后处理只抑制训练条件覆盖的残余成分",
            fontsize=FS_SMALL, color=C_ORANGE, ha="center")
    gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1])
    ax = fig.add_subplot(gs2[0]); ax.set_title("(b) 每个模块都要满足前提条件", fontsize=FS_TITLE)
    ax.axis("off")
    module_rows = [
        ["线性自适应滤波", "线性回声", "参考信号已对齐", "失调、ERLE"],
        ["规则型残余抑制", "稳定残余回声", "可靠的双讲保护", "近端保真、衰减量"],
        ["学习型残余抑制", "非线性残余与噪声", "匹配数据、算力与时延预算", "近端保真、跨场景稳定性"],
    ]
    tab = ax.table(cellText=module_rows, colLabels=["模块", "主要对象", "使用前提", "重点验证"],
                   loc="center", colColours=["0.9"] * 4, colWidths=[0.22, 0.22, 0.30, 0.26])
    tab.auto_set_font_size(False); tab.set_fontsize(8.8); tab.scale(1, 2.0)
    ax.text(0.5, 0.06, "模块叠加不保证性能单调提升；必须在目标设备、房间和双讲条件下实测。",
            ha="center", fontsize=FS_TINY + 0.5, color="0.3", transform=ax.transAxes)
    ax = fig.add_subplot(gs2[1]); ax.set_title("(c) 选型时比较这些条件", fontsize=FS_TITLE)
    ax.axis("off")
    rows = [
        ["NLMS", "短到中等", "逐样本", "不直接处理", "不需要"],
        ["分区频域滤波", "适合长路径", "按块更新", "不直接处理", "不需要"],
        ["频域卡尔曼滤波", "适合长路径", "依赖状态模型", "不直接处理", "不需要"],
        ["混合学习方案", "由线性前级承担", "取决于前级与模型", "可针对性处理", "需要匹配数据"],
    ]
    tab = ax.table(cellText=rows, colLabels=["方案", "路径长度", "路径变化", "非线性残余", "训练数据"],
                   loc="center", colColours=["0.9"] * 5, colWidths=[0.20, 0.20, 0.22, 0.20, 0.18])
    tab.auto_set_font_size(False); tab.set_fontsize(8.6); tab.scale(1, 1.9)
    ax.text(0.5, 0.06, "还要同时核对算力、内存、端到端时延、参考位置和双讲策略。",
            ha="center", fontsize=FS_TINY + 0.5, color="0.3", transform=ax.transAxes)
    save(fig, "fig32_aec_hybrid_select.png")

if __name__ == "__main__":
    fig_problem(); fig_concept(); fig_nlms(); fig_erle(); fig_delay_dtd(); fig_nonlinear(); fig_hybrid()
    print("ALL DONE")
