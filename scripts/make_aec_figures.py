# -*- coding: utf-8 -*-
"""AEC 深度扩展版配套插图（入门导向，7 张） duct-taped to repo style."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib import gridspec
from pathlib import Path

OUT = Path(__file__).parent.parent / "figures"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(42)
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

# ---- 图1：回声是怎么产生的 ----
def fig_problem():
    N = int(1.6 * FS)
    x = colored_x(N); h = synth_h()
    echo = np.convolve(x, h, mode="same") * 0.5
    s = np.zeros(N); s[int(0.8*FS):int(1.2*FS)] = 0.7 * rng.standard_normal(int(0.4*FS))
    v = 0.02 * rng.standard_normal(N)
    d = echo + s + v
    t = np.arange(N) / FS
    fig = plt.figure(figsize=(13.5, 5.2))
    fig.suptitle("图26 回声形成：d(n) = s(n) + x(n)*h(n) + v(n)", fontsize=FS_SUP)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1.4, 1.4])
    ax = fig.add_subplot(gs[0])
    ax.set_title("(a) 房间就是一条脉冲响应 h(n)", fontsize=FS_TITLE)
    ax.stem(h[:80], linefmt=C_BLUE, markerfmt="o", basefmt=" ", label="h抽头")
    ax.set_xlabel("抽头 n", fontsize=FS_LABEL); ax.set_ylabel("幅度", fontsize=FS_LABEL)
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
    ax.set_xlim(0, 1.6); ax.set_xlabel("时间 (s)", fontsize=FS_LABEL)
    ax.legend(fontsize=FS_SMALL); ax.grid(ls=":", alpha=0.5); ax.tick_params(labelsize=FS_TINY)
    ax = fig.add_subplot(gs[2])
    ax.set_title("(c) 麦克风里三者相加得 d(n)", fontsize=FS_TITLE)
    ax.plot(t, d, color=C_MAIN, lw=1, label="麦克风 d(n)")
    ax.plot(t, echo, color=C_BLUE, lw=0.8, alpha=0.7, label="回声分量")
    ax.axvspan(0.8, 1.2, color=C_RED, alpha=0.10)
    ax.set_xlim(0, 1.6); ax.set_xlabel("时间 (s)", fontsize=FS_LABEL)
    ax.legend(fontsize=FS_SMALL); ax.grid(ls=":", alpha=0.5); ax.tick_params(labelsize=FS_TINY)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig26_aec_problem.png")

# ---- 图2：一句话结构 ----
def fig_concept():
    fig, ax = plt.subplots(figsize=(13.5, 3.8))
    fig.suptitle("图27 AEC结构：用已知播放做参考，抄出回声副本再相减", fontsize=FS_SUP)
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
    box(9.1, 2.5, 1.6, 1.4, "干净输出 e\n→波束/ASR", "#e8f6db")
    box(4.2, 4.6, 2.2, 0.9, "DTD 开关（双讲冻结）", "#f6e5db")
    arrow((2.2, 3.2), (2.7, 3.2), C_BLUE)
    arrow((4.7, 3.2), (5.4, 3.2), C_BLUE)
    arrow((1.25, 2.5), (3.0, 1.6), C_BLUE)
    arrow((4.7, 1.0), (5.4, 1.0), C_BLUE)
    arrow((7.3, 1.0), (8.02, 2.72), C_BLUE)
    arrow((7.3, 3.2), (7.75, 3.2), C_BLUE)
    arrow((8.65, 3.2), (9.1, 3.2), C_GREEN, w=2.0)
    arrow((5.3, 4.6), (3.7, 1.6), C_RED, ls="--", w=1.8)
    ax.annotate("抄房间指纹", xy=(3.7, 1.0), xytext=(0.7, 0.35), fontsize=FS_SMALL + 2, color=C_PURPLE,
                arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=1.8, connectionstyle="arc3,rad=-0.25"))
    fig.subplots_adjust(top=0.88)
    save(fig, "fig27_aec_concept.png")

# ---- 图3：NLMS 抄指纹 ----
def fig_nlms():
    N = int(1.6 * FS); taps = 128
    h = np.exp(-np.arange(taps) / 25) * np.random.default_rng(11).standard_normal(taps)
    x = colored_x(N); d = np.convolve(x, h, mode="full")[:N]
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
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    fig.suptitle("图28 NLMS自适应：ŵ 从0长成 h，错得多改得多", fontsize=FS_SUP)
    ax = axes[0]; ax.set_title("(a) ŵ 逐步长成 h", fontsize=FS_TITLE)
    ax.plot(h[:64], color="0.45", lw=2, label="真实 h（答案）")
    cb_styles = [(C_BLUE, "-", "深蓝实线"), (C_ORANGE, "--", "橙色虚线"),
                 (C_GREEN, "-.", "绿色点划线"), (C_PURPLE, ":", "紫色点线")]
    pick = [0, len(snaps)//3, 2*len(snaps)//3, len(snaps)-1]
    for j, (c, ls, _nm) in zip(pick, cb_styles):
        ax.plot(snaps[j][:64], color=c, ls=ls, lw=1.6, label=f"{j*400}步")
    ax.set_xlabel("抽头", fontsize=FS_LABEL); ax.legend(fontsize=FS_SMALL)
    ax.grid(ls=":", alpha=0.5); ax.tick_params(labelsize=FS_TINY)
    ax = axes[1]; ax.set_title("(b) 失配降、ERLE涨", fontsize=FS_TITLE)
    ax.plot(tc[:len(mism)], mism, color=C_RED, label="失配(dB)")
    ax.set_xlabel("时间 (s)", fontsize=FS_LABEL); ax.tick_params(labelsize=FS_TINY)
    ax2 = ax.twinx(); ax2.plot(tcc, erle, color=C_BLUE, label="ERLE(dB)")
    ax.tick_params(labelsize=FS_TINY, colors=C_RED)
    ax2.tick_params(labelsize=FS_TINY, colors=C_BLUE)
    ax.set_ylabel("失配 (dB)", fontsize=FS_LABEL, color=C_RED)
    ax2.set_ylabel("ERLE (dB)", fontsize=FS_LABEL, color=C_BLUE)
    ax.grid(ls=":", alpha=0.5)
    ax.text(0.99, -6.0, "失配每降10dB、ERLE约涨10dB", fontsize=FS_SMALL, color=C_MAIN, ha="right",
            bbox=dict(fc="white", ec="0.7", alpha=0.9))
    ax = axes[2]; ax.set_title("(c) 步长μ：大跑得快但抖", fontsize=FS_TITLE)
    for mu, c in [(0.2, C_BLUE), (0.5, C_ORANGE), (1.0, C_RED)]:
        ee, _, _ = nlms_run(x, d, taps, mu, None)
        _, er = block_erle(d, ee)
        ax.plot(tcc[:len(er)], er, color=c, label=f"μ={mu}")
    ax.set_xlabel("时间 (s)", fontsize=FS_LABEL); ax.set_ylabel("ERLE (dB)", fontsize=FS_LABEL)
    ax.legend(fontsize=FS_SMALL); ax.grid(ls=":", alpha=0.5); ax.tick_params(labelsize=FS_TINY)
    ax.text(0.05, 0.10, "μ∈(0,2)才稳定，工程取0.1~0.5",
            transform=ax.transAxes, fontsize=FS_SMALL + 1, color=C_RED)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, "fig28_aec_nlms.png")

# ---- 图4：ERLE + 双讲冻结 ----
def fig_erle():
    N = int(1.6 * FS); taps = 128
    h = np.exp(-np.arange(taps) / 25) * np.random.default_rng(11).standard_normal(taps)
    x = colored_x(N); echo = np.convolve(x, h, mode="full")[:N]
    s = np.zeros(N); s[int(0.8*FS):int(1.2*FS)] = 0.7 * np.random.default_rng(5).standard_normal(int(0.4*FS))
    d = echo + s
    freeze = np.zeros(N, bool); freeze[int(0.8*FS):int(1.2*FS)] = True
    e, _, _ = nlms_run(x, d, taps, 0.5, freeze)
    tcc, erle = block_erle(echo, e)
    plateau = float(np.nanmean(erle[(tcc > 0.45) & (tcc < 0.8)]))
    t = np.arange(N) / FS
    fig = plt.figure(figsize=(13.5, 6.0))
    fig.suptitle("图29 NLMS收敛：单讲爬升、双讲冻结", fontsize=FS_SUP)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.6], hspace=0.45)
    ax = fig.add_subplot(gs[0]); ax.set_title("(a) 三路波形（双讲段 d 里混入 s）", fontsize=FS_TITLE)
    ax.plot(t, x, color=C_BLUE, lw=0.8, label="参考 x")
    ax.plot(t, d, color=C_RED, lw=0.8, label="麦克风 d")
    ax.plot(t, e, color=C_GREEN, lw=0.8, label="残差 e")
    ax.axvspan(0.8, 1.2, color=C_RED, alpha=0.10)
    ax.set_xlim(0, 1.6); ax.legend(fontsize=FS_SMALL, loc="upper left")
    ax.grid(ls=":", alpha=0.5); ax.tick_params(labelsize=FS_TINY); ax.set_xticklabels([])
    ax = fig.add_subplot(gs[1]); ax.set_title("(b) ERLE：平时爬、双讲冻", fontsize=FS_TITLE)
    ax.plot(tcc, erle, color=C_BLUE, lw=1.6, label="ERLE")
    ax.axvspan(0.8, 1.2, color=C_RED, alpha=0.10)
    ax.text(1.0, np.nanmax(erle)*0.92, "双讲段ERLE无意义（混入近端语音）\n冻结≠发散", ha="center", fontsize=FS_SMALL + 1, color=C_RED,
            bbox=dict(fc="white", ec=C_RED, lw=0.7, alpha=0.9, boxstyle="round,pad=0.3"))
    ax.annotate(f"收敛后 ERLE≈{plateau:.0f} dB", xy=(0.62, plateau), xytext=(0.35, plateau+6),
                fontsize=FS_LABEL, color=C_BLUE,
                arrowprops=dict(arrowstyle="->", color=C_BLUE))
    ax.set_xlim(0, 1.6); ax.set_xlabel("时间 (s)", fontsize=FS_LABEL); ax.set_ylabel("ERLE (dB)", fontsize=FS_LABEL)
    ax.grid(ls=":", alpha=0.5); ax.tick_params(labelsize=FS_TINY)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig29_aec_erle_freeze.png")

# ---- 图5：延迟 × 双讲 ----
def fig_delay_dtd():
    N = int(1.6 * FS); taps = 128; delay = 300
    h = np.exp(-np.arange(taps) / 25) * np.random.default_rng(11).standard_normal(taps)
    x = colored_x(N)
    xd = np.roll(x, delay)
    echo = np.convolve(xd, h, mode="full")[:N]
    s = np.zeros(N); s[int(0.8*FS):int(1.2*FS)] = 0.7 * np.random.default_rng(5).standard_normal(int(0.4*FS))
    d = echo + s
    freeze = np.zeros(N, bool); freeze[int(0.8*FS):int(1.2*FS)] = True
    e_align, _, _ = nlms_run(xd, d, taps, 0.5, freeze)
    e_mis, _, _ = nlms_run(x, d, taps, 0.5, freeze)
    e_nof, _, _ = nlms_run(xd, d, taps, 0.5, None)
    tcc, er_align = block_erle(echo, e_align)
    _, er_mis = block_erle(echo, e_mis)
    _, er_nof = block_erle(echo, e_nof)
    pa = float(np.nanmean(er_align[(tcc > 0.45) & (tcc < 0.8)]))
    pm = float(np.nanmean(er_mis[(tcc > 0.45) & (tcc < 0.8)]))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    fig.suptitle("图30 工程两大坑：延迟对齐 × 双讲冻结", fontsize=FS_SUP)
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
    ax = axes[1]; ax.set_title("(b) 对齐 vs 不对齐（延迟300抽头>滤波器128）", fontsize=FS_TITLE)
    ax.plot(tcc, er_align, color=C_BLUE, lw=1.6, label=f"对齐（约{pa:.0f}dB）")
    ax.plot(tcc, er_mis, color="0.6", lw=1.4, ls="--", label=f"不对齐（单讲约{pm:.0f}dB）")
    ax.set_ylim(-10, 40); ax.set_xlabel("时间 (s)", fontsize=FS_LABEL); ax.set_ylabel("ERLE (dB)", fontsize=FS_LABEL)
    ax.legend(fontsize=FS_SMALL); ax.grid(ls=":", alpha=0.5); ax.tick_params(labelsize=FS_TINY)
    ax = axes[2]; ax.set_title("(c) 双讲冻结 vs 不冻结", fontsize=FS_TITLE)
    ax.plot(tcc, er_align, color=C_BLUE, lw=1.6, label="冻结（正确）")
    ax.plot(tcc, er_nof, color=C_RED, lw=1.4, ls="--", label="不冻结（学坏）")
    ax.axvspan(0.8, 1.2, color=C_RED, alpha=0.10)
    ax.set_ylim(-10, 40); ax.set_xlabel("时间 (s)", fontsize=FS_LABEL)
    ax.legend(fontsize=FS_SMALL); ax.grid(ls=":", alpha=0.5); ax.tick_params(labelsize=FS_TINY)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, "fig30_aec_delay_dtd.png")
    print(f"aligned约{pa:.1f}dB misaligned约{pm:.1f}dB")

# ---- 图6：非线性天花板 ----
def fig_nonlinear():
    N = int(3.0 * FS); taps = 256
    r = np.random.default_rng(7)
    h = np.exp(-np.arange(taps) / 60) * r.standard_normal(taps)
    x = colored_x(N, seed=9); x = x / np.std(x) * 0.5
    echo_lin = np.convolve(x, h, mode="full")[:N]
    drive = 1.1
    echo_nl = np.convolve(np.tanh(drive * x) / np.tanh(drive), h, mode="full")[:N]
    e1, _, _ = nlms_run(x, echo_lin, taps, 0.3, None)
    e2, _, _ = nlms_run(x, echo_nl, taps, 0.3, None)
    t1, er1 = block_erle(echo_lin, e1, blk=800)
    t2, er2 = block_erle(echo_nl, e2, blk=800)
    p1 = float(np.nanmean(er1[t1 > 2.2])); p2 = float(np.nanmean(er2[t2 > 2.2]))
    fig = plt.figure(figsize=(13.5, 5.0))
    fig.suptitle("图31 线性NLMS的天花板：tanh软削波下封顶约20dB（收敛后口径；与图19(a)同数据换标题）", fontsize=FS_SUP)
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.2, 1])
    ax = fig.add_subplot(gs[0]); ax.set_title("(a) 线性路径 vs 非线性路径", fontsize=FS_TITLE)
    ax.plot(t1, er1, color=C_BLUE, label=f"线性路径（≈{p1:.0f}dB）")
    ax.plot(t2, er2, color=C_RED, lw=2.2, label=f"非线性tanh（≈{p2:.0f}dB）")
    ax.axhline(20, color=C_ORANGE, ls=":", lw=1.5, label="低成本底色~20dB")
    ax.set_ylim(-5, 45); ax.set_xlabel("时间 (s)", fontsize=FS_LABEL); ax.set_ylabel("ERLE (dB)", fontsize=FS_LABEL)
    ax.legend(fontsize=FS_SMALL, loc="upper left"); ax.grid(ls=":", alpha=0.5); ax.tick_params(labelsize=FS_TINY)
    ax = fig.add_subplot(gs[1]); ax.set_title("(b) tanh长什么样", fontsize=FS_TITLE)
    xx = np.linspace(-2, 2, 400)
    ax.plot(xx, xx, color="0.6", ls="--", label="线性")
    ax.plot(xx, np.tanh(1.1*xx)/np.tanh(1.1), color=C_RED, label="tanh软削波")
    ax.axvspan(1, 2, color=C_RED, alpha=0.08); ax.axvspan(-2, -1, color=C_RED, alpha=0.08)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=FS_SMALL); ax.grid(ls=":", alpha=0.5); ax.tick_params(labelsize=FS_TINY)
    ax.text(0, -1.55, "大信号被压扁=谐波失真", ha="center", fontsize=FS_SMALL + 1.5, color=C_RED)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig31_aec_nonlinear.png")

# ---- 图7：混合管线 + 阶梯 + 选型 ----
def fig_hybrid():
    fig = plt.figure(figsize=(13.5, 8.8))
    fig.suptitle("图32 现代混合管线：线性打底 + DNN扫尾，算法怎么选", fontsize=FS_SUP)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.2, 1], hspace=0.4)
    ax = fig.add_subplot(gs[0]); ax.set_title("(a) 混合管线（线性消线性，DNN扫尾）", fontsize=FS_TITLE)
    ax.set_xlim(0, 14); ax.set_ylim(0.2, 6.8); ax.axis("off")
    def box(x, y, w, h, text, fc, fs=FS_SMALL):
        ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="k", lw=1.2))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs, color=C_MAIN)
    def arrow(a, b, c="k", w=1.4):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, color=c, lw=w))
    box(0.2, 3.2, 1.8, 1.6, "远端参考\n(链路末端取)", "#f6e5db")
    box(2.5, 3.2, 1.8, 1.6, "延迟对齐\n(GCC搜τ)", "#dbe9f6")
    box(4.8, 3.2, 2.2, 1.6, "线性AEC\nPBFDAF/FDKF", "#dbe9f6")
    box(7.5, 3.2, 2.2, 1.6, "DNN残余抑制\n(吃e+x)", "#fde3c8")
    box(10.2, 3.2, 2.0, 1.6, "干净输出\n→ASR", "#e8f6db")
    for a, b in [((2.0, 4.0), (2.5, 4.0)), ((4.3, 4.0), (4.8, 4.0)), ((7.0, 4.0), (7.5, 4.0)), ((9.7, 4.0), (10.2, 4.0))]:
        arrow(a, b, C_BLUE)
    box(4.8, 1.0, 2.2, 1.2, "DTD/步长控制\nGeigel·NCC·DVSS", "#f6e5db", FS_TINY)
    ax.plot([1.1, 1.1, 8.6, 8.6], [3.2, 2.2, 2.2, 3.2], color=C_BLUE, lw=1.0, ls=":")
    ax.text(2.7, 2.45, "参考也喂DNN（短折线，非跨图长线）", ha="center", fontsize=FS_SMALL + 1, color=C_BLUE)
    ax.text(2.5, 6.0, "DSP（蓝）：线性只消线性", fontsize=FS_SMALL, color=C_BLUE)
    ax.text(7.5, 6.0, "NN（橙）：扫尾非线性+噪声", fontsize=FS_SMALL, color=C_ORANGE)
    gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1])
    ax = fig.add_subplot(gs2[0]); ax.set_title("(b) ERLE阶梯：加DNN值不值", fontsize=FS_TITLE)
    bars = ax.bar(["线性AEC", "+NLP", "+DNN"], [20, 28, 35], color=[C_BLUE, C_ORANGE, C_GREEN])
    for b, v in zip(bars, [20, 28, 35]):
        ax.text(b.get_x()+b.get_width()/2, v+1.2, f"{v}dB", ha="center", va="bottom", fontsize=FS_LABEL, color=C_MAIN)
    ax.set_ylim(0, 42); ax.set_ylabel("ERLE (dB, 示意)", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TINY); ax.grid(axis="y", ls=":", alpha=0.5)
    ax = fig.add_subplot(gs2[1]); ax.set_title("(c) 算法选型速查（定性）", fontsize=FS_TITLE)
    ax.axis("off")
    rows = [["NLMS", "低", "kB", "中", "玩具/教学"],
            ["PBFDAF", "中", "百kB", "中", "会议/车载默认"],
            ["FDKF", "中", "百kB", "高", "跟踪快/高端"],
            ["混合DNN", "高", "MB", "最高", "旗舰/云端"]]
    tab = ax.table(cellText=rows, colLabels=["算法", "算力", "内存", "双讲鲁棒", "推荐场景"],
                   loc="center", colColours=["0.9"]*5)
    tab.auto_set_font_size(False); tab.set_fontsize(9.5); tab.scale(1, 1.8)
    ax.text(0.5, 0.02, "时间线：1970 LMS → 1992 PBFDAF → 2006 FDKF → 2019 端到端 → 2021 混合夺冠 → 2023 pAEC",
            ha="center", fontsize=FS_TINY, color="0.35", transform=ax.transAxes)
    fig.subplots_adjust(top=0.92, hspace=0.5)
    save(fig, "fig32_aec_hybrid_select.png")

if __name__ == "__main__":
    fig_problem(); fig_concept(); fig_nlms(); fig_erle(); fig_delay_dtd(); fig_nonlinear(); fig_hybrid()
    print("ALL DONE")
