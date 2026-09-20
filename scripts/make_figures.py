# -*- coding: utf-8 -*-
"""生成教程插图 26 张（图 1~25、图 33；图 26~32 见 make_aec_figures.py）。

用法（仓库根目录）：
    .venv/bin/python scripts/make_figures.py      # 图 1~25、图 33 → figures/
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
try:
    from daimon_runtime import setup_plot
    setup_plot()
except Exception:
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).parent.parent / "figures"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(42)
C_MAIN, C_BLUE, C_RED, C_GREEN, C_ORANGE, C_PURPLE = "#1a1a2e", "#2f6db3", "#c0392b", "#2e8b57", "#e67e22", "#7d3c98"
# 全局字号约定（统一全书插图）
FS_SUP = 13      # 图题 suptitle
FS_TITLE = 11    # 子图标题
FS_LABEL = 10    # 轴标签 / 主要注释
FS_SMALL = 9     # 图例 / 次要注释
FS_TINY = 8      # 刻度 / 极小标注


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


# ----------------------------------------------------------------------
# 图8 阵列几何形态大全
# ----------------------------------------------------------------------
def fig_geometries():
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    titles = ["(a) 双麦 Endfire", "(b) 双麦 Broadside", "(c) 四麦均匀线阵 ULA",
              "(d) 四麦方阵/平面阵", "(e) 六麦圆阵 UCA (Echo 6+1)",
              "(f) 均匀球阵(32麦)", "(g) 螺旋阵/对数阵", "(h) 分布式/非规则阵"]
    for ax, t in zip(axes.flat, titles):
        ax.set_title(t, fontsize=11)
        ax.set_aspect("equal")
        ax.grid(True, ls=":", alpha=0.5)
        ax.tick_params(labelsize=7)
    # (a) endfire
    ax = axes[0, 0]
    ax.scatter([0, 0], [0, 4], s=180, c=C_BLUE, zorder=5, marker="o", edgecolors="k")
    for a in np.linspace(-0.5, 0.5, 7):
        ax.plot([0, 6 * np.sin(a)], [2, 2 + 6 * np.cos(a)], color=C_ORANGE, alpha=0.5, lw=1)
    ax.annotate("拾音主轴", xy=(1.2, 6.2), fontsize=9, color=C_ORANGE)
    ax.set_xlim(-4, 6); ax.set_ylim(-1, 7)
    # (b) broadside
    ax = axes[0, 1]
    ax.scatter([-2, 2], [0, 0], s=180, c=C_BLUE, zorder=5, edgecolors="k")
    for a in np.linspace(np.pi / 2 - 0.55, np.pi / 2 + 0.55, 7):
        ax.plot([0, 6 * np.cos(a)], [0, 6 * np.sin(a)], color=C_ORANGE, alpha=0.5, lw=1)
    ax.annotate("拾音主轴", xy=(1.5, 5.5), fontsize=9, color=C_ORANGE)
    ax.set_xlim(-6.5, 6.5); ax.set_ylim(-1, 7)
    # (c) ULA
    ax = axes[0, 2]
    xs = np.arange(4)
    ax.scatter(xs, np.zeros(4), s=180, c=C_BLUE, zorder=5, edgecolors="k")
    for x in xs[1:]:
        ax.annotate("", xy=(x, 0.15), xytext=(x - 1, 0.15),
                    arrowprops=dict(arrowstyle="<->", color="k", lw=1))
    ax.text(1.5, 0.35, "d (阵元间距)", ha="center", fontsize=FS_SMALL)
    ax.set_xlim(-0.8, 3.8); ax.set_ylim(-1.9, 2.4)
    # (d) square
    ax = axes[0, 3]
    sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
    ax.scatter(*zip(*sq), s=180, c=C_BLUE, zorder=5, edgecolors="k")
    ax.add_patch(Circle((0.5, 0.5), 0.75, fill=False, ls="--", color="gray"))
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(-0.6, 1.6)
    # (e) UCA 6+1
    ax = axes[1, 0]
    th = np.linspace(0, 2 * np.pi, 7)[:-1]
    ax.scatter(np.cos(th), np.sin(th), s=180, c=C_BLUE, zorder=5, edgecolors="k")
    ax.scatter([0], [0], s=260, c=C_RED, zorder=6, marker="*", edgecolors="k")
    ax.add_patch(Circle((0, 0), 1, fill=False, ls="--", color="gray"))
    ax.text(0, -1.35, "红★为中心麦", ha="center", fontsize=9, color=C_RED)
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
    # (f) spherical
    ax = axes[1, 1]
    ax.remove()
    ax = fig.add_subplot(2, 4, 6, projection="3d")
    ax.set_title("(f) 均匀球阵(32麦)", fontsize=FS_TITLE)
    u = rng.normal(size=(32, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    ax.scatter(u[:, 0], u[:, 1], u[:, 2], s=65, c=C_BLUE, edgecolors="k", lw=0.4, depthshade=False)
    r = 1
    phi, theta = np.mgrid[0:np.pi:20j, 0:2 * np.pi:20j]
    ax.plot_wireframe(r * np.sin(phi) * np.cos(theta), r * np.sin(phi) * np.sin(theta),
                      r * np.cos(phi), color="gray", alpha=0.5, lw=0.6)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    # (g) spiral —— 真对数螺旋：r_k = r0·a^k，等角度递增
    ax = axes[1, 2]
    k = np.arange(12)
    th_g = k * np.deg2rad(42)
    r_g = 0.13 * 1.21 ** k
    tt_g = np.linspace(0, th_g[-1], 300)
    ax.plot(0.13 * 1.21 ** (tt_g / np.deg2rad(42)) * np.cos(tt_g),
            0.13 * 1.21 ** (tt_g / np.deg2rad(42)) * np.sin(tt_g),
            ls="--", color="gray", lw=0.9, alpha=0.8)
    ax.scatter(r_g * np.cos(th_g), r_g * np.sin(th_g), s=150, c=C_BLUE, zorder=5, edgecolors="k")
    ax.text(0, -1.55, "对数螺旋 $r_k=r_0\\,a^k$", ha="center", fontsize=FS_SMALL, color=C_MAIN)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.8, 1.5)
    # (h) distributed
    ax = axes[1, 3]
    pts = np.array([[0, 0], [3, 0.5], [1.2, 2.5], [-1, 1.8], [2, 2.8]])
    ax.scatter(pts[:, 0], pts[:, 1], s=160, c=C_BLUE, zorder=5, edgecolors="k")
    conv = pts[np.argsort(np.arctan2(pts[:, 1] - pts[:, 1].mean(), pts[:, 0] - pts[:, 0].mean()))]
    ax.plot(*np.vstack([conv, conv[0]]).T, ls="--", color="gray")
    ax.set_xlim(-1.8, 3.8); ax.set_ylim(-0.8, 3.4)
    fig.suptitle("图8  常见麦克风阵列几何形态示意", fontsize=14, y=1.0)
    fig.tight_layout()
    save(fig, "fig08_geometries.png")


# ----------------------------------------------------------------------
# 图3 近场球面波 vs 远场平面波
# ----------------------------------------------------------------------
def fig_near_far_field():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    src = (0, 4.0)
    ax.scatter(*src, marker="*", s=400, c=C_RED, zorder=6, edgecolors="k")
    ax.annotate("近场点声源", xy=src, xytext=(0.3, 4.35), fontsize=10, color=C_RED)
    th = np.linspace(np.pi + 0.35, 2 * np.pi - 0.35, 60)
    for rr in [0.8, 1.3, 1.8, 2.3]:
        ax.plot(src[0] + rr * np.cos(th), src[1] + rr * np.sin(th), color=C_BLUE, alpha=0.55)
    ax.scatter([-0.8, 0, 0.8], [0, 0, 0], s=150, c=C_GREEN, zorder=6, edgecolors="k")
    ax.annotate("幅度差 + 相位差\n(球面波)", xy=(1.6, 1.2), fontsize=10, color=C_GREEN)
    ax.set_title("(a) 近场模型：球面波，需联合估计距离与方向", fontsize=11)
    ax.set_xlim(-3, 3); ax.set_ylim(-0.8, 4.8); ax.set_aspect("equal"); ax.grid(ls=":", alpha=0.5)
    ax = axes[1]
    # 斜入射平面波：传播方向 245°（声源在右上 65°，自正横起算 θ=25°）
    u = np.array([np.cos(np.deg2rad(245)), np.sin(np.deg2rad(245))])  # 传播方向
    w = np.array([-u[1], u[0]])                                       # 波前方向
    for k in range(6):
        c0 = np.array([2.2, 5.4]) + k * 1.1 * u
        p1, p2 = c0 - 2.6 * w, c0 + 2.6 * w
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=C_BLUE, alpha=0.55, lw=1.4)
    for off in (-1.7, 0, 1.7):
        base = np.array([0.8, 5.8]) + off * w
        ax.annotate("", xy=tuple(base + 1.3 * u), xytext=tuple(base),
                    arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.6))
    ax.scatter([-0.8, 0, 0.8], [0, 0, 0], s=150, c=C_GREEN, zorder=6, edgecolors="k")
    ax.text(-3.05, -0.95, "波前先后到达各麦\n→ 仅有相位差（时延）", fontsize=FS_LABEL, color=C_GREEN,
            ha="left", va="bottom")
    ax.text(-3.0, 6.4, "远场平面波（斜入射）", fontsize=FS_LABEL, color=C_RED, ha="left", va="top")
    ax.text(-3.05, -1.12, "远场判据：r > 2D²/λ（D=孔径）", fontsize=FS_SMALL, color=C_RED,
            ha="left", va="top")
    ax.set_title("(b) 远场模型：平面波近似", fontsize=FS_TITLE)
    ax.set_xlim(-3.2, 3.2); ax.set_ylim(-1.4, 6.6); ax.set_aspect("equal"); ax.grid(ls=":", alpha=0.5)
    fig.suptitle("图3  近场与远场声场模型", fontsize=FS_SUP)
    fig.tight_layout()
    save(fig, "fig03_near_far_field.png")


# ----------------------------------------------------------------------
# 图16 波束图对比：DSB vs MVDR vs 超指向 (模拟)
# ----------------------------------------------------------------------
def fig_beampatterns():
    M, d = 8, 0.5
    th = np.linspace(-90, 90, 721)
    thetas_rad = np.deg2rad(th)
    m = np.arange(M) - (M - 1) / 2
    a0 = np.exp(-2j * np.pi * d * m * np.sin(0))          # 期望方向 0°
    # DSB
    w_ds = a0 / M
    # MVDR: 白噪声 + 20°方向强干扰(避开DSB自然零陷位置)
    a_int = np.exp(-2j * np.pi * d * m * np.sin(np.deg2rad(20)))[:, None]
    R = 10 * (a_int @ a_int.conj().T) + 1.0 * np.eye(M)
    w_mvdr = np.linalg.solve(R, a0) / (a0.conj() @ np.linalg.solve(R, a0))
    # 超指向: 小间距 d_sd=0.2λ, 各向同性(弥散)噪声协方差 Γ=sinc(2d|i-j|)
    d_sd = 0.2
    a0_sd = np.exp(-2j * np.pi * d_sd * m * np.sin(0))
    G = np.sinc(2 * d_sd * np.abs(np.subtract.outer(m, m)))
    w_sd = np.linalg.solve(G, a0_sd) / (a0_sd.conj() @ np.linalg.solve(G, a0_sd))
    w_sd = w_sd * (np.sqrt(M) / np.linalg.norm(w_sd))   # 归一化便于同图对比
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), subplot_kw=dict(polar=True))
    fig.subplots_adjust(wspace=0.12)
    curves = [("DSB 延迟求和 (d=λ/2)", w_ds, d, C_BLUE, "-"),
              ("MVDR 干扰零陷20° (d=λ/2)", w_mvdr, d, C_RED, "--"),
              ("超指向 (d=0.2λ)", w_sd, d_sd, C_GREEN, ":")]
    for ax, (title, ymax) in zip(axes, [("线性幅度", None), ("dB 刻度", 40)]):
        for name, w, dd, c, ls in curves:
            A = np.exp(-2j * np.pi * dd * np.outer(m, np.sin(thetas_rad)))
            b = np.abs(w.conj() @ A)
            if "dB" in title:
                b = 20 * np.log10(np.maximum(b / b.max(), 1e-4))
                b = np.maximum(b, -40) + 40      # 平移到 0~40 便于径向刻度
            else:
                b = b / b.max()
            ax.plot(thetas_rad, b, color=c, ls=ls, lw=2.0, label=name)
        ax.set_theta_zero_location("N")
        ax.set_thetamin(-90); ax.set_thetamax(90)
        if "dB" in title:
            ax.set_rlim(0, 40)
            ax.set_rgrids([0, 10, 20, 30, 40],
                          ["−40", "−30", "−20", "−10", "0 dB"], fontsize=FS_TINY)
        ax.set_title(title, fontsize=13, pad=18)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), fontsize=FS_SMALL + 0.5, framealpha=0.95)
    # 在 dB 图上用箭头标出 20° 处 MVDR 零陷（标签外移至图外空白，引线指回零陷坑）
    ax_db = axes[1]
    ax_db.annotate("MVDR 零陷 @20°", xy=(np.deg2rad(20), 2),
                   xytext=(np.deg2rad(76), 37.5), fontsize=FS_SMALL, color=C_RED,
                   bbox=dict(fc="white", ec=C_RED, lw=0.7, alpha=0.9, boxstyle="round,pad=0.25"),
                   arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.5,
                                   connectionstyle="arc3,rad=-0.1"))
    fig.suptitle("图16  8元ULA 0°指向波束图对比：DSB / MVDR(零陷@20°) / 超指向（f=2kHz, d=λ/2, c=343m/s, 上=0°正横, 模拟）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, "fig16_beampattern.png")


# ----------------------------------------------------------------------
# 图11 DOA空间谱对比: Bartlett vs Capon vs MUSIC
# ----------------------------------------------------------------------
def fig_doa_spectrum():
    M, d = 8, 0.5
    srcs = [-20, 30]
    snr_db, snaps = 20, 400
    m = np.arange(M) - (M - 1) / 2
    A = np.exp(-2j * np.pi * d * np.outer(m, np.sin(np.deg2rad(srcs))))
    sig = rng.normal(size=(2, snaps)) + 1j * rng.normal(size=(2, snaps))
    noise = (np.sqrt(10 ** (-snr_db / 10) / 2) * (rng.normal(size=(M, snaps)) + 1j * rng.normal(size=(M, snaps))))
    X = A @ sig + noise
    R = X @ X.conj().T / snaps
    th = np.linspace(-90, 90, 1801)
    tr = np.deg2rad(th)
    Av = np.exp(-2j * np.pi * d * np.outer(m, np.sin(tr)))  # M x grid
    bart = np.sum(np.abs(Av.conj() * (R @ Av)), axis=0)
    bart /= bart.max()
    Rinv = np.linalg.inv(R + 1e-6 * np.eye(M))
    capon = 1 / np.sum(Av.conj() * (Rinv @ Av), axis=0).real
    capon /= capon.max()
    evals, evecs = np.linalg.eigh(R)
    En = evecs[:, :M - 2]
    music = 1 / np.sum(np.abs(En.conj().T @ Av) ** 2, axis=0)
    music_db = 10 * np.log10(music / music.max())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].plot(th, 10 * np.log10(bart + 1e-6), color=C_BLUE, alpha=0.85, lw=1.4, label="Bartlett (常规BF)")
    axes[0].plot(th, 10 * np.log10(capon + 1e-6), color=C_RED, lw=2.2, label="Capon/MVDR 谱")
    for s in srcs:
        axes[0].axvline(s, color="0.55", ls="--", lw=1.2, alpha=0.9)
        axes[0].plot(s, 2.2, marker=11, color="0.45", ms=10)
        axes[0].annotate(f"真源{s}°", xy=(s, -2), xytext=(s - 6, -8), fontsize=FS_SMALL,
                         arrowprops=dict(arrowstyle="->", color="k", lw=0.8))
    axes[0].set_ylim(-40, 3); axes[0].set_xlabel("角度 (°)"); axes[0].set_ylabel("归一化功率 (dB)")
    axes[0].legend(fontsize=FS_SMALL); axes[0].set_title("(a) 波束扫描类空间谱", fontsize=12); axes[0].grid(ls=":", alpha=0.5)
    axes[1].plot(th, music_db, color=C_GREEN, lw=1.6, label="MUSIC")
    for s in srcs:
        axes[1].axvline(s, color="0.55", ls="--", lw=1.2, alpha=0.9)
        axes[1].plot(s, 2.2, marker=11, color="0.45", ms=10)
        axes[1].annotate(f"真源{s}°", xy=(s, -2), xytext=(s - 6, -8), fontsize=9,
                         arrowprops=dict(arrowstyle="->", color="k", lw=0.8))
    axes[1].set_ylim(-25, 3); axes[1].set_xlabel("角度 (°)"); axes[1].set_ylabel("伪谱 (dB)")
    axes[1].text(0.5, 0.12, "MUSIC伪谱非功率：峰高无意义，只看峰位", transform=axes[1].transAxes,
                 fontsize=FS_SMALL, color=C_RED, ha="center",
                 bbox=dict(fc="white", ec=C_RED, lw=0.7, alpha=0.9, boxstyle="round,pad=0.3"))
    axes[1].legend(fontsize=FS_SMALL); axes[1].set_title("(b) MUSIC 特征结构类伪谱", fontsize=12); axes[1].grid(ls=":", alpha=0.5)
    fig.suptitle("图11  两信号源(-20°, 30°) DOA空间谱估计对比（8元ULA, SNR=20dB, 400快拍, 模拟）", fontsize=12.5)
    fig.tight_layout()
    save(fig, "fig11_doa_spectrum.png")


# ----------------------------------------------------------------------
# 图12 GCC-PHAT 原理
# ----------------------------------------------------------------------
def fig_gcc_phat():
    """重新设计：d=4 cm、θ=30°（自正横起算）→ τ=d·sinθ/c≈58.3 μs≈0.93 采样点@16kHz，
    正好演示 GCC-PHAT 的亚采样（分数倍采样点）时延估计能力。"""
    fs = 16000
    c = 343.0
    d = 0.04
    theta_deg = 30.0
    tau_true = d * np.sin(np.deg2rad(theta_deg)) / c          # ≈ 58.3 μs
    n = 4096
    # 带限噪声源（300 Hz ~ 6 kHz）
    sig = rng.standard_normal(n)
    S0 = np.fft.rfft(sig)
    fr = np.fft.rfftfreq(n, 1 / fs)
    S0[(fr > 6000) | (fr < 300)] = 0
    sig = np.fft.irfft(S0)
    sig /= np.max(np.abs(sig))
    # 分数倍采样点延迟：频域相位斜坡
    F0 = np.fft.rfft(sig)
    x1 = np.fft.irfft(F0 * np.exp(-2j * np.pi * fr * tau_true)) + 0.05 * rng.standard_normal(n)
    x2 = sig + 0.05 * rng.standard_normal(n)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    # (a) 两路信号：放大到亚毫秒窗口，时移肉眼可辨
    ax = axes[0]
    t0 = 2.0e-3
    m = (np.arange(n) / fs >= t0) & (np.arange(n) / fs < t0 + 5e-4)
    tt = np.arange(n)[m] / fs * 1e3
    ax.plot(tt, x2[m], "o-", color=C_BLUE, ms=3.5, lw=1.2, label="麦克风2（先到）")
    ax.plot(tt, x1[m], "s-", color=C_RED, ms=3.5, lw=1.2, label="麦克风1（晚到 τ）")
    ax.annotate("", xy=(2.20, 0.62), xytext=(2.20 - tau_true * 1e3, 0.62),
                arrowprops=dict(arrowstyle="<->", color="k", lw=2.0))
    ax.text(2.20 - tau_true * 1e3 / 2, 0.68, "τ≈58 μs", ha="center", fontsize=FS_SMALL)
    ax.set_ylim(-0.8, 0.95)
    ax.set_title("(a) 两路信号（放大 0.5 ms 窗口）", fontsize=FS_TITLE)
    ax.set_xlabel("时间 (ms)", fontsize=FS_LABEL)
    ax.text(0.03, 0.05, "τ=58.3 μs < 采样间隔 62.5 μs\n（不足 1 个采样点，肉眼几乎难辨）",
            transform=ax.transAxes, fontsize=FS_SMALL, color=C_MAIN, va="bottom")
    ax.legend(fontsize=FS_SMALL, loc="upper right"); ax.grid(ls=":", alpha=0.5)
    # (b) 16 倍上采样 GCC-PHAT：尖锐内插峰落在 ≈58 μs
    ax = axes[1]
    up = 16
    # 真·亚采样：先把两路信号做 16× 频域零延拓内插（256 kHz），再做 PHAT
    def fft_upsample(x, up):
        X = np.fft.rfft(x)
        Xe = np.zeros(len(x) * up // 2 + 1, dtype=complex)
        Xe[:len(X)] = X
        return np.fft.irfft(Xe, len(x) * up)
    x1u, x2u = fft_upsample(x1, up), fft_upsample(x2, up)
    n_u = len(x1u)
    X1, X2 = np.fft.rfft(x1u), np.fft.rfft(x2u)
    S = X1 * np.conj(X2)
    gcc = np.fft.fftshift(np.fft.irfft(S / (np.abs(S) + 1e-10)))
    lags_us = np.arange(-n_u // 2, n_u // 2) / (fs * up) * 1e6
    win = np.abs(lags_us) <= 500
    ax.plot(lags_us[win], gcc[win], color=C_BLUE, lw=1.4)
    pk = np.argmax(gcc)
    ax.plot(lags_us[pk], gcc[pk], "r^", ms=10)
    ax.axvline(tau_true * 1e6, color="k", ls="--", lw=1, alpha=0.6)
    for s in np.arange(-3, 4) * 1e6 / fs:
        if abs(s) <= 500:
            ax.axvline(s, color="gray", ls=":", lw=1.4, alpha=0.9)
    ax.annotate(f"内插峰 = {lags_us[pk]:.1f} μs", xy=(lags_us[pk], gcc[pk]),
                xytext=(170, 0.72 * gcc[pk]), fontsize=FS_LABEL,
                arrowprops=dict(arrowstyle="->", color="k", lw=1.8))
    ax.text(0.03, 0.97, f"亚采样精度：真值 {tau_true*1e6:.1f} μs\n≈ 0.93 个采样点",
            transform=ax.transAxes, fontsize=FS_SMALL, color=C_RED, va="top")
    ax.text(0.03, 0.76, "（16×上采样内插；\n灰点线=整数采样间隔）",
            transform=ax.transAxes, fontsize=FS_TINY + 0.5, color=C_RED, va="top")
    ax.set_xlim(-500, 500)
    ax.set_title("(b) GCC-PHAT（16×上采样）互相关峰", fontsize=FS_TITLE)
    ax.set_xlabel("时延 τ (μs)", fontsize=FS_LABEL); ax.set_ylabel("GCC-PHAT值", fontsize=FS_LABEL); ax.grid(ls=":", alpha=0.5)
    # (c) 几何：θ 自正横方向（垂直于两麦连线）起算
    ax = axes[2]
    ax.scatter([0, 0.04], [0, 0], s=160, c=C_BLUE, zorder=6, edgecolors="k")
    ax.annotate("麦1", xy=(0, 0), xytext=(-0.014, -0.003), fontsize=FS_LABEL, ha="right")
    ax.annotate("麦2", xy=(0.04, 0), xytext=(0.054, -0.003), fontsize=FS_LABEL, ha="left")
    ax.annotate("", xy=(0.04, -0.011), xytext=(0, -0.011), arrowprops=dict(arrowstyle="<->", color="k", lw=1.2))
    ax.text(0.02, -0.021, "d = 4 cm", ha="center", fontsize=FS_LABEL)
    # 正横方向（虚线）
    ax.plot([0.02, 0.02], [0, 0.10], color="gray", ls="--", lw=1.2)
    ax.text(0.02, 0.104, "正横方向 (broadside)", ha="center", fontsize=FS_SMALL, color="gray")
    # 声源射线：自正横转 θ=30°
    th = np.deg2rad(theta_deg)
    ray = np.array([np.sin(th), np.cos(th)])
    ax.plot([0.02, 0.02 + 0.095 * ray[0]], [0, 0.095 * ray[1]], color=C_RED, lw=1.8)
    ax.annotate("", xy=(0.02 + 0.095 * ray[0], 0.095 * ray[1]),
                xytext=(0.02 + 0.075 * ray[0], 0.075 * ray[1]),
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.8))
    arc = matplotlib.patches.Arc((0.02, 0), 0.075, 0.075, theta1=90 - theta_deg, theta2=90,
                                 color="k", lw=1.2)
    ax.add_patch(arc)
    ax.text(0.0385, 0.049, "θ=30°", fontsize=FS_LABEL, style="italic")
    ax.text(0.02 + 0.098 * ray[0], 0.095 * ray[1], "远场声源", fontsize=FS_LABEL, color=C_RED,
            ha="left", va="center")
    ax.set_title("(c) θ=arcsin(c·τ/d)=arcsin(0.5)=30°\n（θ 自正横方向起算）", fontsize=FS_TITLE)
    ax.set_xlim(-0.05, 0.17); ax.set_ylim(-0.028, 0.16); ax.set_aspect("equal"); ax.axis("off")
    fig.suptitle("图12  GCC-PHAT 声源定位原理：亚采样时延估计（模拟）", fontsize=FS_SUP)
    fig.tight_layout()
    save(fig, "fig12_gcc_phat.png")


# ----------------------------------------------------------------------
# 图17 GSC 结构框图
# ----------------------------------------------------------------------
def fig_gsc():
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    ax.axis("off"); ax.set_xlim(0, 11.5); ax.set_ylim(0, 6.4)
    def box(x, y, w, h, text, fc="#dbe9f6"):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec="k", lw=1.2, zorder=3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=FS_LABEL, zorder=4)
    def arrow(x1, y1, x2, y2, text="", c="k", conn=None, dy=0.15):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                     color=c, lw=1.8, zorder=2,
                                     connectionstyle=conn or "arc3,rad=0"))
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, text, fontsize=FS_SMALL + 2,
                    ha="center", color=c, bbox=dict(fc="white", alpha=0.8, pad=1, ec="none"))
    # ---- 上支路（直通链路）：阵列 → 延迟补偿 → 固定波束 → d(n)
    box(0.3, 4.6, 1.7, 0.9, "麦克风阵列\n$x_1...x_M$")
    box(2.7, 4.6, 1.7, 0.9, "延迟补偿\n(转向期望方向)")
    box(5.1, 4.6, 1.6, 0.9, "固定波束形成器\n(求和)", fc="#f6e5db")
    arrow(2.0, 5.05, 2.7, 5.05); arrow(4.4, 5.05, 5.1, 5.05)
    # ---- 下支路：延迟补偿 → 阻塞矩阵 B → 多通道自适应滤波 → y(n)
    box(2.7, 1.6, 1.7, 0.9, "阻塞矩阵 B\n(陷波期望方向)")
    box(5.1, 1.6, 1.6, 0.9, "多通道自适应滤波\n$L_1...L_{M-1}$")
    arrow(3.55, 4.6, 3.55, 2.5)          # 延迟补偿 ↓ 阻塞矩阵
    arrow(4.4, 2.05, 5.1, 2.05)
    # ---- 最右：减法器汇合
    box(8.3, 2.85, 1.7, 1.0, "减法器\n$e=d-y$", fc="#e8f6db")
    arrow(6.7, 5.05, 9.15, 3.85, conn="angle,angleA=0,angleB=90,rad=8")   # d(n) → 减法器顶
    arrow(6.7, 2.05, 9.15, 2.85, conn="angle,angleA=0,angleB=90,rad=8")   # y(n) → 减法器底
    ax.text(7.5, 4.75, "d(n)\n期望+残余噪声", fontsize=FS_SMALL, color=C_MAIN, ha="center")
    ax.text(7.5, 1.62, "y(n)\n噪声估计", fontsize=FS_SMALL, color=C_MAIN, ha="center")
    arrow(10.0, 3.35, 10.8, 3.35)
    ax.text(10.55, 3.6, "e(n)", fontsize=FS_LABEL, color=C_RED, ha="center")
    # ---- 支路标签：放在各自支路最左端正上方
    ax.text(0.3, 5.68, "上支路（固定波束，目标无失真）", fontsize=FS_LABEL, color=C_ORANGE, ha="left")
    ax.text(2.7, 2.68, "下支路（自适应噪声估计）", fontsize=FS_LABEL, color=C_BLUE, ha="left")
    ax.text(5.75, 0.55, "输出 $e(n)$：期望方向无失真，干扰/噪声被自适应对消",
            ha="center", fontsize=FS_LABEL, color=C_RED)
    ax.set_title("图17  GSC（广义旁瓣对消器）结构框图", fontsize=FS_SUP)
    save(fig, "fig17_gsc.png")


# ----------------------------------------------------------------------
# 图14 SRP-PHAT 网格搜索
# ----------------------------------------------------------------------
def fig_srp_grid():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    mics = np.array([[0, 0], [4, 0], [0, 3], [4, 3]])
    ax.scatter(mics[:, 0], mics[:, 1], s=160, c=C_BLUE, zorder=6, edgecolors="k", label="麦克风")
    src = (5.5, 3.8)
    ax.scatter(*src, marker="*", s=400, c=C_RED, zorder=6, edgecolors="k", label="真实声源")
    for gx in np.linspace(0.5, 6.5, 8):
        for gy in np.linspace(0.5, 4.5, 6):
            ax.plot(gx, gy, ".", color="gray", ms=3, alpha=0.6)
    ax.plot(2.214, 2.1, marker="s", ms=8, mfc="none", mec=C_GREEN, mew=2, label="候选位置 r")
    for mic in mics:
        ax.plot([mic[0], src[0]], [mic[1], src[1]], ls=":", color=C_ORANGE, alpha=0.6, lw=1)
    ax.legend(fontsize=9); ax.set_title("(a) 空间网格 + 各麦到候选点的距离线", fontsize=11)
    ax.set_xlim(-0.5, 7); ax.set_ylim(-0.5, 5.2); ax.set_aspect("equal"); ax.grid(ls=":", alpha=0.3)
    ax = axes[1]
    gx = np.linspace(-0.5, 7.0, 90); gy = np.linspace(-0.5, 5.2, 68)
    GX, GY = np.meshgrid(gx, gy)
    SRP = np.zeros_like(GX)
    for mic in mics:
        SRP += np.exp(-((np.hypot(GX - mic[0], GY - mic[1]) - np.hypot(src[0] - mic[0], src[1] - mic[1])) ** 2) / 0.02)
    im = ax.pcolormesh(GX, GY, SRP, cmap="viridis", shading="auto")
    ax.scatter(mics[:, 0], mics[:, 1], s=120, c=C_BLUE, edgecolors="k", zorder=6)
    imax = np.unravel_index(np.argmax(SRP), SRP.shape)
    ax.plot(GX[imax], GY[imax], "r*", ms=12, mec="k")
    ax.set_title("(b) SRP-PHAT 功率累积图，峰值即位置估计", fontsize=FS_TITLE)
    ax.set_xlim(-0.5, 7); ax.set_ylim(-0.5, 5.2); ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=FS_LABEL); ax.set_ylabel("y (m)", fontsize=FS_LABEL)
    fig.colorbar(im, ax=ax, shrink=0.8, label="累积功率")
    fig.suptitle("图14  SRP-PHAT 定位过程示意（模拟；示意场景单位 m，与算例 4-1 参数不同）", fontsize=FS_SUP)
    fig.tight_layout()
    save(fig, "fig14_srp_grid.png")


# ----------------------------------------------------------------------
# 图22 声源追踪：粒子滤波 vs 直方图
# 口径说明（F12 定稿，供正文与重出核对）：
#   (a) SIR-PF：N=200 粒子；观测 = 真值 + 高斯噪声（σ=7°），另约 20% 帧替换为
#       [0,120]° 均匀野点（与第 9 章正文一致）。(b) 为独立合成的 PHD 强度示意。
#   两子图独立仿真、角度约定不同（(a) 0~120°扇区，(b) −90°~+90°，见子图标题）。
# ----------------------------------------------------------------------
def fig_tracking():
    T = 120
    t = np.arange(T)
    true = 60 + 40 * np.sin(2 * np.pi * t / T)
    obs = true + rng.normal(0, 7, T)
    is_out = rng.random(T) < 0.2
    obs = np.where(is_out, rng.uniform(0, 120, T), obs)
    # 粒子滤波
    N = 200
    particles = np.full(N, true[0]) + rng.normal(0, 10, N)
    est = np.zeros(T); n_eff = np.zeros(T)
    for k in range(T):
        particles += rng.normal(0, 1.8, N)  # 过程噪声
        w = np.exp(-0.5 * ((particles - obs[k]) / 7) ** 2); w /= w.sum()
        neff = 1 / np.sum(w ** 2); n_eff[k] = neff
        idx = rng.choice(N, N, p=w)
        particles = particles[idx] + rng.normal(0, 0.5, N) * (1 - neff / N > 0.5)
        est[k] = np.average(particles, weights=w)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].plot(t, obs, ".", color="gray", ms=4, label="DOA观测(含噪+野点)")
    axes[0].plot(t, true, color="k", lw=2, label="真实轨迹")
    axes[0].plot(t, est, color=C_RED, lw=1.6, label="粒子滤波估计")
    axes[0].set_xlabel("帧"); axes[0].set_ylabel("方位角 (°)")
    axes[0].legend(fontsize=FS_SMALL + 1, loc="lower left", framealpha=0.95); axes[0].grid(ls=":", alpha=0.5)
    axes[0].set_title("(a) 单说话人追踪：粒子滤波平滑DOA序列\n（角度约定：0–120° 扇区）", fontsize=FS_TITLE)
    ax = axes[1]
    th = np.linspace(-90, 90, 400)
    I = (np.exp(-0.5 * ((th - (-25)) / 4) ** 2) + 0.9 * np.exp(-0.5 * ((th - 40) / 5) ** 2)
         + 0.05 * rng.random(400))
    ax.plot(th, I, color=C_BLUE, lw=1.6)
    ax.fill_between(th, I, alpha=0.25, color=C_BLUE)
    for peak, lb in [(-25, "说话人1"), (40, "说话人2")]:
        ax.plot(peak, np.exp(0), "r^", ms=10)
        ax.annotate(lb, xy=(peak, 1.02), xytext=(peak - 16, 1.30), fontsize=FS_LABEL, color=C_RED,
                    arrowprops=dict(arrowstyle="->", color="r"))
    ax.set_ylim(0, 1.38)
    ax.set_xlabel("方位角 (°)"); ax.set_ylabel("PHD 强度")
    ax.set_title("(b) 多说话人：PHD滤波器输出的目标强度分布\n（角度约定：−90°~+90°）", fontsize=FS_TITLE)
    ax.grid(ls=":", alpha=0.5)
    fig.suptitle("图22  声源追踪示意（模拟）", fontsize=FS_SUP)
    fig.tight_layout()
    save(fig, "fig22_tracking.png")


# ----------------------------------------------------------------------
# 图15 白噪声增益/指向性 vs 频率
# ----------------------------------------------------------------------
def fig_wng_di():
    fs = 16000
    freqs = np.linspace(100, 8000, 200)
    M, d_phys = 6, 0.04  # 4cm间距六麦环
    th = np.linspace(0, 2 * np.pi, 3601)
    wng_dsb, di_dsb, wng_sd, di_sd = [], [], [], []
    for f in freqs:
        dl = f * d_phys / 343
        m = np.arange(M)
        a0 = np.exp(-2j * np.pi * dl * (np.cos(m * 2 * np.pi / M) * 1))  # 0°方向
        # 各向同性噪声协方差: Γ[i,j]=sinc(2 d_ij f/c)
        D = np.abs(np.subtract.outer(m, m)) * d_phys
        G = np.sinc(2 * f * D / 343)
        w_ds = a0 / M
        w_sd = np.linalg.solve(G, a0) / (a0.conj() @ np.linalg.solve(G, a0))
        # 波束图
        A = np.exp(-2j * np.pi * dl * np.cos(th[None, :] - m[:, None] * 2 * np.pi / M))
        Bds = np.abs(w_ds.conj() @ A) ** 2
        Bsd = np.abs(w_sd.conj() @ A) ** 2
        wng_dsb.append(10 * np.log10(1 / np.sum(np.abs(w_ds) ** 2)))
        di_dsb.append(10 * np.log10(Bds.max() / Bds.mean()))
        wng_sd.append(10 * np.log10(1 / np.sum(np.abs(w_sd) ** 2)))
        di_sd.append(10 * np.log10(Bsd.max() / Bsd.mean()))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.4))
    axes[0].semilogx(freqs, wng_dsb, color=C_BLUE, lw=2.0, label="DSB 延迟求和（理想无失配理论值）")
    axes[0].semilogx(freqs, wng_sd, color=C_RED, label="超指向(最大化DI)")
    axes[0].axhline(0, color="gray", ls="--", lw=1.2)
    axes[0].text(0.97, 0.93, "WNG = 0 dB 稳健门限（低于此别用）", transform=axes[0].transAxes,
                 fontsize=FS_SMALL + 1, color="dimgray", ha="right", va="bottom",
                 bbox=dict(fc="white", ec="0.7", alpha=0.9, boxstyle="round,pad=0.25"))
    axes[0].set_xlabel("频率 (Hz)"); axes[0].set_ylabel("白噪声增益 WNG (dB)")
    axes[0].legend(fontsize=FS_SMALL, loc="center left", bbox_to_anchor=(1.02, 0.5), framealpha=0.95)
    axes[0].grid(ls=":", alpha=0.5); axes[0].set_title("(a) WNG：超指向低频稳健性差", fontsize=FS_TITLE)
    axes[1].axvspan(4000, 8000, color="gray", alpha=0.07, zorder=1)
    axes[1].semilogx(freqs, di_dsb, color=C_BLUE, lw=2.0, label="DSB 延迟求和")
    axes[1].semilogx(freqs, di_sd, color=C_RED, label="超指向(最大化DI)")
    axes[1].text(500, 1.5, "高频灰底区抖动\n选型只看 <3kHz", fontsize=FS_SMALL + 0.5,
                 color="dimgray", ha="center", va="top",
                 bbox=dict(fc="white", ec="0.7", alpha=0.9, boxstyle="round,pad=0.3"))
    axes[1].set_xlabel("频率 (Hz)"); axes[1].set_ylabel("指向性指数 DI (dB)")
    axes[1].legend(fontsize=FS_SMALL, loc="center left", bbox_to_anchor=(1.02, 0.5), framealpha=0.95)
    axes[1].grid(ls=":", alpha=0.5); axes[1].set_title("(b) DI：超指向全频段更尖锐", fontsize=FS_TITLE)
    for _ax in axes:
        _ax.minorticks_on()
        _ax.grid(which="minor", ls=":", alpha=0.25)
    fig.suptitle("图15  6元圆阵(d=4cm) 超指向 vs 延迟求和：WNG与DI的权衡（模拟）", fontsize=FS_SUP - 0.5)
    fig.tight_layout()
    save(fig, "fig15_wng_di.png")


# ----------------------------------------------------------------------
# 图23 远场语音前端处理链路
# ----------------------------------------------------------------------
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(16.5, 5.2))
    ax.axis("off"); ax.set_xlim(0, 16.6); ax.set_ylim(0, 4.2)
    steps = [("多通道\n采集", "#dbe9f6"), ("AEC\n回声消除", "#dbe9f6"), ("去混响\n(WPE)", "#dbe9f6"),
             ("声源定位\nDOA估计", "#f6e5db"), ("声源追踪\n轨迹平滑", "#f6e5db"),
             ("波束形成\n(MVDR/GSC)", "#e8f6db"), ("后置滤波\n分离/降噪", "#e8f6db"),
             ("KWS\n唤醒词", "#f6dbdb"), ("ASR\n语音识别", "#f6dbdb")]
    ms_row = ["≈8ms", "≈16ms", "≈24ms", "≈32ms", "≈8ms", "≈12ms", "≈20ms", "≈50ms", "≈200ms+"]
    w, step = 1.55, 1.78
    for i, ((txt, fc), ms) in enumerate(zip(steps, ms_row)):
        x = 0.25 + i * step
        ax.add_patch(plt.Rectangle((x, 1.1), w, 1.32, fc=fc, ec="k", lw=1.2))
        ax.text(x + w / 2, 1.76, txt, ha="center", va="center", fontsize=FS_LABEL)
        ax.text(x + w / 2, 0.82, ms, ha="center", va="center", fontsize=FS_SMALL, color="dimgray")
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + w, 1.65), (x + step, 1.65), arrowstyle="-|>", mutation_scale=16, color="k", lw=1.5))
    ax.text(0.25, 0.52, "框下数字为单帧增量延迟量级示意（具体看§10.1瀑布条）", fontsize=FS_TINY, color="dimgray", ha="left", va="center")
    # 方向信息反馈：拆成两条带箭头线（定位→波束 “DOA”，追踪→波束 “平滑DOA”）
    x_doa = 0.25 + 3 * step + w / 2
    x_trk = 0.25 + 4 * step + w / 2
    x_bf = 0.25 + 5 * step + w / 2
    ax.add_patch(FancyArrowPatch((x_doa, 2.42), (x_bf - 0.15, 2.55), arrowstyle="-|>", mutation_scale=17,
                                 color=C_RED, lw=2.0, connectionstyle="arc3,rad=0.18"))
    ax.text(x_doa - 0.5, 3.35, "方向信息DOA/平滑DOA（自适应指向）", fontsize=FS_SMALL + 0.5, color=C_RED, ha="center",
            bbox=dict(fc="white", ec=C_RED, lw=0.6, alpha=0.9, boxstyle="round,pad=0.2"))
    # 三色图例：蓝色=信号级 / 粉色=信息级 / 绿色=增强级（+红色=后端识别）
    lx = 11.9
    for j, (lab, fc) in enumerate([("蓝色=信号级", "#dbe9f6"), ("粉色=信息级", "#f6e5db"),
                                   ("绿色=增强级", "#e8f6db"), ("红色=后端识别", "#f6dbdb")]):
        ax.add_patch(plt.Rectangle((lx + j * 1.18, 0.32), 0.28, 0.22, fc=fc, ec="k", lw=0.8))
        ax.text(lx + j * 1.18 + 0.32, 0.43, lab, fontsize=FS_TINY, va="center", ha="left")
    ax.set_title("图23  远场语音交互前端典型信号处理链路", fontsize=FS_SUP)
    save(fig, "fig23_pipeline.png")


# ----------------------------------------------------------------------
# 图9 麦克风数-性能-成本权衡 & 定位算法性能随T60变化
# ----------------------------------------------------------------------
def fig_tradeoffs():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    ax = axes[0]
    n = np.array([1, 2, 4, 6, 8, 16, 32])
    gain = 10 * np.log10(np.minimum(n, 16))
    cost = 100 * (1 - np.exp(-n / 8))
    ax.plot(n, gain / gain.max() * 100, "o-", color=C_BLUE, label="阵列增益/定位能力(相对)")
    ax.plot(n, 100 - cost, "s--", color=C_RED, label="成本/算力可控性(相对)")
    ax.axvspan(4, 8, color=C_GREEN, alpha=0.12)
    ax.text(5.4, 12, "消费级\n甜蜜区", ha="center", fontsize=FS_LABEL, color=C_GREEN)
    ax.annotate("M≥16 按 min(M,16) 截平：\n实际增益趋于饱和", xy=(16, 100), xytext=(17, 62),
                fontsize=FS_SMALL, color=C_BLUE,
                arrowprops=dict(arrowstyle="->", color=C_BLUE))
    ax.set_xlabel("麦克风数量 M", fontsize=FS_LABEL); ax.set_ylabel("相对指标 (%)", fontsize=FS_LABEL)
    ax.legend(fontsize=FS_SMALL); ax.grid(ls=":", alpha=0.5)
    ax.set_title("(a) 麦数 vs 性能 vs 成本", fontsize=FS_TITLE)
    ax = axes[1]
    # 有界饱和模型：传统方法从 ~3° 渐近饱和到 25~35°，DNN 保持 5~12°（定性示意）
    t60 = np.linspace(0, 1.0, 200)
    err_srp = 3 + 27 * (1 - np.exp(-t60 / 0.25))
    err_music = 3 + 30 * (1 - np.exp(-t60 / 0.20))
    err_dnn = 5 + 7 * (1 - np.exp(-t60 / 0.30))
    ax.plot(t60, err_srp, color=C_BLUE, lw=1.8, label="SRP-PHAT (信号模型)")
    ax.plot(t60, err_music, color=C_ORANGE, lw=1.8, ls="--", label="MUSIC (子空间)")
    ax.plot(t60, err_dnn, color=C_RED, lw=1.8, label="DNN类方法 (数据驱动)")
    ax.axhline(30, color="gray", ls=":", lw=0.8)
    ax.text(0.62, 31.5, "传统方法饱和区 25~35°", fontsize=FS_SMALL, color="gray")
    ax.set_xlabel("混响时间 $T_{60}$ (s)", fontsize=FS_LABEL)
    ax.set_ylabel("平均定位误差 (°)", fontsize=FS_LABEL)
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 40)
    ax.legend(fontsize=FS_SMALL, loc="upper left"); ax.grid(ls=":", alpha=0.5)
    ax.set_title("(b) 典型算法定位误差随混响变化趋势（定性示意）", fontsize=FS_TITLE)
    fig.suptitle("图9  阵列规模与算法鲁棒性权衡", fontsize=FS_SUP - 0.5)
    fig.tight_layout()
    save(fig, "fig09_tradeoffs.png")


# ----------------------------------------------------------------------
# 通用：合成语音信号（谐波 + 音节幅度调制），供图13/图16使用
# ----------------------------------------------------------------------
def synth_speech(fs=16000, dur=1.2):
    t = np.arange(int(fs * dur)) / fs
    sig = np.zeros_like(t)
    syll = [(0.05, 0.30, 120), (0.38, 0.62, 140), (0.70, 0.95, 110), (1.00, 1.15, 130)]
    for t0, t1, f0 in syll:
        idx = (t >= t0) & (t < t1)
        tt = t[idx] - t0
        env = np.sin(np.pi * (t[idx] - t0) / (t1 - t0)) ** 0.7
        s = np.zeros_like(tt)
        for h in range(1, 12):
            s += (1 / h) * np.sin(2 * np.pi * f0 * h * tt)
        form = 0.6 * np.sin(2 * np.pi * 700 * tt) + 0.3 * np.sin(2 * np.pi * 1700 * tt)
        sig[idx] = env * (0.5 * s + 0.5 * form)
    sig += 0.001 * rng.standard_normal(len(t))
    return t, sig / np.max(np.abs(sig))


# ----------------------------------------------------------------------
# 图5 房间声学：RIR三段结构 / 能量衰减与T60 / DRR与临界距离
# ----------------------------------------------------------------------
def fig_room_acoustics():
    fs = 16000
    T60 = 0.5
    L = int(0.7 * fs)
    t = np.arange(L) / fs
    tau = int(0.012 * fs)  # 直达声到达 12ms
    decay = np.exp(-6.91 * t / T60)
    tail = rng.standard_normal(L) * decay
    tail[:tau] = 0
    rir = tail.copy()
    rir[tau] = 1.0
    # 早期反射若干离散峰
    for dt, a in [(0.010, 0.55), (0.018, -0.4), (0.027, 0.3), (0.036, -0.22)]:
        rir[tau + int(dt * fs)] += a
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0))
    ax = axes[0]
    ax.plot(t * 1000, rir, color=C_BLUE, lw=0.7)
    ax.axvspan(0, 12, color=C_GREEN, alpha=0.18)
    ax.axvspan(12, 92, color=C_ORANGE, alpha=0.15)
    ax.axvspan(92, 700, color=C_RED, alpha=0.11)
    ax.axvline(12, color="gray", ls="--", lw=0.9, alpha=0.8)
    ax.axvline(92, color="gray", ls="--", lw=0.9, alpha=0.8)
    ax.set_ylim(-2.6, 3.1)
    ax.annotate("直达声", xy=(12, 1.05), xytext=(20, 2.5), fontsize=10, color=C_GREEN,
                arrowprops=dict(arrowstyle="->", color=C_GREEN))
    ax.annotate("早期反射\n(<80ms, 有助可懂度)", xy=(55, 0.7), xytext=(100, 2.3), fontsize=9.5,
                color=C_ORANGE, arrowprops=dict(arrowstyle="->", color=C_ORANGE))
    ax.annotate("晚期混响（弥散尾音, 糊化语音）", xy=(260, 0.25), xytext=(330, 1.5), fontsize=9.5,
                color=C_RED, arrowprops=dict(arrowstyle="->", color=C_RED))
    ax.set_xlabel("时间 (ms)"); ax.set_ylabel("幅度")
    ax.set_title("(a) 房间脉冲响应 RIR 的三段结构", fontsize=11)
    ax.grid(ls=":", alpha=0.5)
    ax = axes[1]
    energy = np.cumsum(rir[::-1] ** 2)[::-1]  # Schroeder 反向积分
    edb = 10 * np.log10(energy / energy.max() + 1e-12)
    ax.plot(t * 1000, edb, color=C_PURPLE, lw=1.8)
    ax.axhline(-60, color="k", ls="--", alpha=0.5)
    ax.axvline(T60 * 1000, color="k", ls="--", alpha=0.5)
    ax.annotate(f"$T_{{60}}$ = {T60*1000:.0f} ms\n(衰减60dB所需时间)", xy=(T60 * 1000, -60),
                xytext=(240, -40), fontsize=10, arrowprops=dict(arrowstyle="->", color="k"))
    ax.set_ylim(-75, 2); ax.set_xlabel("时间 (ms)"); ax.set_ylabel("剩余能量 (dB)")
    ax.set_title("(b) 能量衰减曲线与混响时间 $T_{60}$", fontsize=11)
    ax.grid(ls=":", alpha=0.5)
    ax = axes[2]
    d = np.linspace(0.2, 6, 200)
    dc = 1.0
    direct = -20 * np.log10(d / dc)
    reverb = np.zeros_like(d)
    ax.plot(d, direct, color=C_GREEN, lw=1.8, label="直达声 (每倍距离 -6 dB)")
    ax.plot(d, reverb, color=C_RED, lw=1.8, label="混响声 (近似处处相等)")
    ax.axvline(dc, color="k", ls="--", alpha=0.6)
    ax.annotate("临界距离 $d_c$≈0.057√(V/$T_{60}$)\n(典型客厅约 0.5~1.5 m)", xy=(dc, 0),
                xytext=(1.7, 14), fontsize=9.5, arrowprops=dict(arrowstyle="->", color="k"))
    ax.fill_between(d, direct, reverb, where=direct > reverb, color=C_GREEN, alpha=0.08)
    ax.text(0.27, -19.5, "直达占优\n(DRR>0)", fontsize=FS_LABEL - 0.5, color=C_GREEN)
    ax.text(3.6, 6, "混响占优\n(DRR<0)", fontsize=FS_LABEL - 0.5, color=C_RED)
    ax.set_xlabel("声源-麦克风距离 (m)"); ax.set_ylabel("相对声级 (dB)")
    ax.set_ylim(-25, 22); ax.legend(fontsize=FS_SMALL); ax.grid(ls=":", alpha=0.5)
    ax.set_title("(c) 直达混响比 DRR 与临界距离", fontsize=FS_TITLE)
    fig.suptitle("图5  房间声学基础：混响从哪来、有多强、多远开始失控（模拟）", fontsize=FS_SUP)
    fig.tight_layout()
    save(fig, "fig05_room_acoustics.png")


# ----------------------------------------------------------------------
# 图6 数据组织方式：STFT分帧 / 时频点与快拍 / 协方差矩阵 / 特征值谱
# ----------------------------------------------------------------------
def fig_stft_cov():
    fs = 16000
    t, sig = synth_speech(fs, 1.2)
    n_fft, hop = 512, 128
    win = np.hanning(n_fft)
    frames = [sig[i:i + n_fft] * win for i in range(0, len(sig) - n_fft, hop)]
    S = np.array([np.fft.rfft(fr) for fr in frames]).T  # F x T
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8))
    ax = axes[0, 0]
    tt = t[:3200] * 1000
    ax.plot(tt, sig[:3200], color="gray", lw=0.8)
    for k in range(3):
        i0 = k * hop
        seg = np.arange(n_fft)
        ax.plot((i0 + seg) / fs * 1000, win * 0.9, color=C_BLUE, lw=1.2)
        ax.add_patch(plt.Rectangle((i0 / fs * 1000, -1), n_fft / fs * 1000, 2, fill=False, ec=C_BLUE, ls="--", alpha=0.5))
    ax.annotate("窗长 32ms\n帧移 8ms\n相邻帧重叠75%", xy=(62, 0.72), fontsize=FS_LABEL, color=C_BLUE)
    ax.set_title("(a) 分帧加窗：把时间信号切成重叠小段", fontsize=FS_TITLE)
    ax.set_xlabel("时间 (ms)", fontsize=FS_LABEL); ax.set_ylabel("幅度", fontsize=FS_LABEL)
    ax.grid(ls=":", alpha=0.4)
    ax = axes[0, 1]
    Sdb = 20 * np.log10(np.abs(S) + 1e-6)
    ax.pcolormesh(np.arange(S.shape[1]) * hop / fs * 1000, np.fft.rfftfreq(n_fft, 1 / fs) / 1000,
                  Sdb, cmap="viridis", shading="auto", vmin=Sdb.max() - 60, vmax=Sdb.max())
    # 标出一个时频点（格子放大标出）
    f0, fb = 60, 128          # 第 60 帧、约 2 kHz 频点
    px, py = f0 * hop / fs * 1000, np.fft.rfftfreq(n_fft, 1 / fs)[fb] / 1000
    ax.plot(px, py, "x", color="w", ms=12, mew=2.8)
    ax.annotate("一个格子 = 一个时频点 (k, l)\n一个快拍 = 固定该时频点、跨 M 个\n麦克风取同一格的复数谱 → 见 (c)",
                xy=(px, py), xytext=(430, 5.2), fontsize=FS_LABEL, color="w",
                arrowprops=dict(arrowstyle="->", color="w", lw=1.5))
    ax.set_ylim(0, 8); ax.set_xlabel("时间 (ms)", fontsize=FS_LABEL)
    ax.set_ylabel("频率 (kHz)", fontsize=FS_LABEL)
    ax.set_title("(b) STFT 语谱图（单麦克风；格子为放大标出）", fontsize=FS_TITLE)
    ax = axes[1, 0]
    M = 8
    R = np.zeros((M, M), dtype=complex)
    srcs = [-20, 30]
    m = np.arange(M) - (M - 1) / 2
    A = np.exp(-2j * np.pi * 0.5 * np.outer(m, np.sin(np.deg2rad(srcs))))
    R = A @ np.diag([10, 5]) @ A.conj().T + np.eye(M)
    im = ax.imshow(np.abs(R), cmap="Blues")
    ax.set_xticks(range(M)); ax.set_yticks(range(M))
    ax.set_xticklabels([f"麦{i+1}" for i in range(M)], fontsize=9)
    ax.set_yticklabels([f"麦{i+1}" for i in range(M)], fontsize=9)
    ax.set_title("(c) 空间协方差矩阵 R（8麦）", fontsize=FS_TITLE)
    fig.colorbar(im, ax=ax, shrink=0.8, label="|R[i,j]|")
    ax = axes[1, 1]
    evals = np.linalg.eigvalsh(R)[::-1]
    evals_db = 10 * np.log10(evals)
    colors = [C_RED if i < 2 else C_BLUE for i in range(M)]
    ax.bar(range(1, M + 1), np.maximum(evals_db, 0), bottom=np.minimum(evals_db, 0),
           color=colors, edgecolor="k")
    ax.axhline(0, color=C_BLUE, ls="--", lw=1.2)
    ax.annotate("2个大特征值\n→ 信号子空间(2个源)", xy=(2, evals_db[1] - 1),
                xytext=(3.0, 13), fontsize=10, color=C_RED,
                arrowprops=dict(arrowstyle="->", color=C_RED))
    ax.annotate("6个小特征值≈噪声底(0 dB)\n→ 噪声子空间", xy=(6.5, 0), xytext=(3.2, 4.5),
                fontsize=10, color=C_BLUE, arrowprops=dict(arrowstyle="->", color=C_BLUE))
    ax.set_xlabel("特征值序号"); ax.set_ylabel("特征值 (dB)")
    ax.set_ylim(-4, 21)
    ax.set_title("(d) 特征值谱：MUSIC 类算法的地基", fontsize=11)
    ax.grid(ls=":", alpha=0.4)
    fig.suptitle("图6  从波形到协方差矩阵：阵列算法处理的数据形态（模拟）", fontsize=FS_SUP)
    fig.tight_layout()
    # 图级箭头：(b) 语谱图 → (c) 协方差矩阵（快拍流向），走两列之间的垂直走廊避免压字
    pos_b = axes[0, 1].get_position(); pos_c = axes[1, 0].get_position()
    x_lane = pos_c.x1 + (pos_b.x0 - pos_c.x1) * 0.55      # 垂直走廊中线
    arr = FancyArrowPatch((pos_b.x0 - 0.012, pos_b.y0 + 0.30 * pos_b.height),
                          (pos_c.x1 - 0.005, pos_c.y0 + 0.70 * pos_c.height),
                          arrowstyle="-|>", mutation_scale=18, color=C_MAIN, lw=1.4,
                          connectionstyle="angle,angleA=270,angleB=180,rad=0",
                          transform=fig.transFigure, zorder=6)
    fig.patches.append(arr)
    fig.text(x_lane - 0.012, (pos_b.y0 + pos_c.y1) / 2 + 0.005, "快拍向量\n逐帧累积 → R",
             fontsize=FS_SMALL, color=C_MAIN, ha="right", va="center")
    save(fig, "fig06_stft_cov.png")


# ----------------------------------------------------------------------
# 图10 稀疏阵列：ULA vs 嵌套阵/互质阵 与 差分协同阵
# ----------------------------------------------------------------------
def fig_sparse_array():
    ula = np.arange(6)
    nested = np.array([0, 1, 2, 3, 7, 11])
    coprime = np.array([0, 3, 4, 6, 8, 9])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    ax = axes[0]
    for y, (name, pos, c) in enumerate([("均匀线阵 ULA", ula, C_BLUE),
                                        ("嵌套阵 Nested", nested, C_RED),
                                        ("互质阵 Coprime", coprime, C_GREEN)]):
        ax.scatter(pos, np.full_like(pos, 2 - y, dtype=float), s=170, c=c, zorder=6, edgecolors="k")
        ax.text(-0.7, 2 - y, name, fontsize=10, va="center", ha="right")
        for p in pos:
            ax.plot([p, p], [2 - y - 0.16, 2 - y + 0.16], color="k", lw=0.8)
    ax.set_xlim(-5.5, 12.5); ax.set_ylim(-0.6, 2.6)
    ax.set_xticks(range(0, 13)); ax.set_yticks([])
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=FS_SMALL)
    ax.set_xlabel("阵元位置（单位 λ/2）", fontsize=FS_LABEL)
    ax.set_title("(a) 同样 6 个麦克风，三种摆法（横轴单位：半个波长 λ/2）", fontsize=FS_TITLE)
    ax = axes[1]
    def coarray(pos):
        lags = set()
        for p1 in pos:
            for p2 in pos:
                lags.add(p1 - p2)
        return sorted(lags)
    for y, (name, pos, c) in enumerate([("ULA：连续滞后 -5..5（11个）", ula, C_BLUE),
                                        ("嵌套阵：连续滞后 -11..11（23个）", nested, C_RED),
                                        ("互质阵：有“洞”但范围大（17个）", coprime, C_GREEN)]):
        lags = coarray(pos)
        ax.scatter(lags, np.full(len(lags), 2 - y, dtype=float), marker="|", s=400, c=c, lw=2.5)
        ax.text(-11.5, 2 - y + 0.32, name, fontsize=10, va="center", color=c)
    ax.set_xlim(-13, 12.5); ax.set_ylim(-0.7, 2.75)
    ax.set_yticks([]); ax.set_xlabel("差分滞后（单位 λ/2）")
    ax.grid(axis="x", ls=":", alpha=0.5)
    ax.set_title("(b) 差分协同阵：所有“麦对间距”组成一个虚拟阵列", fontsize=11)
    fig.suptitle("图10  稀疏阵列：6 个麦克风摆出 23 个虚拟阵元（模拟）", fontsize=13)
    fig.tight_layout()
    save(fig, "fig10_sparse_array.png")


# ----------------------------------------------------------------------
# 图18 声学回声消除 AEC：框图 + NLMS收敛曲线(ERLE)
# ----------------------------------------------------------------------
def fig_aec():
    fig = plt.figure(figsize=(13.5, 4.6))
    ax = fig.add_subplot(1, 2, 1)
    ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 6)
    def box(x, y, w, h, text, fc="#dbe9f6", fs=9.5):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec="k", lw=1.2, zorder=3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=4)
    def arrow(x1, y1, x2, y2, text="", c="k", dy=0.15):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color=c, lw=1.4, zorder=2))
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, text, fontsize=10.5, ha="center", color=c,
            bbox=dict(fc="white", alpha=0.85, pad=1, ec="none"))
    box(0.2, 4.6, 1.9, 0.9, "播放信号 x(n)\n(已知参考)", "#f6e5db")
    box(3.0, 4.6, 1.6, 0.9, "扬声器")
    box(5.5, 4.6, 1.9, 0.9, "房间回声路径\nh(n)")
    arrow(2.1, 5.05, 3.0, 5.05); arrow(4.6, 5.05, 5.5, 5.05)
    box(8.2, 4.6, 2.4, 0.9, "麦克风信号 d(n)\n=回声+用户语音", "#f6dbdb", 8.5)
    arrow(7.4, 5.05, 8.2, 5.05)
    box(3.0, 2.4, 2.2, 1.0, "自适应滤波器 ŵ(n)\n(NLMS, 学回声路径)")
    arrow(1.15, 4.6, 3.6, 3.4, "参考信号", C_BLUE)
    box(6.0, 2.4, 1.6, 1.0, "回声副本\nŷ(n)", "#f6dbdb")
    arrow(5.2, 2.9, 6.0, 2.9)
    box(8.2, 2.4, 2.4, 1.0, "相减 e=d−ŷ\n+残余回声抑制", "#e8f6db")
    arrow(7.6, 2.9, 8.2, 2.9)
    arrow(9.4, 4.6, 9.4, 3.4)
    arrow(9.4, 2.4, 9.4, 1.2, "干净输出→波束形成", C_GREEN, dy=-0.1)
    box(2.6, 0.6, 3.2, 0.9, "双讲检测 DTD\n(用户说话时冻结更新)", "#f6e5db", 8.5)
    arrow(4.2, 1.5, 4.2, 2.4, "双讲冻结控制", C_RED)
    ax.set_title("(a) AEC 结构：已知播放信号做参考，学出回声副本再相减", fontsize=11)
    ax = fig.add_subplot(1, 2, 2)
    fs = 16000
    N = int(1.6 * fs)
    taps = 128
    h = np.exp(-np.arange(taps) / 25) * rng.standard_normal(taps)
    x = np.convolve(rng.standard_normal(N), np.ones(8) / 8)[:N]
    echo = np.convolve(x, h)[:N]
    near = np.zeros(N)
    near[int(0.8 * fs):int(1.2 * fs)] = 0.7 * rng.standard_normal(int(0.4 * fs))
    d = echo + near
    w = np.zeros(taps)
    mu = 0.5
    e = np.zeros(N)
    freeze = np.zeros(N, dtype=bool)
    for n in range(taps, N):
        xv = x[n - taps:n][::-1]
        y = w @ xv
        e[n] = d[n] - y
        if int(0.8 * fs) <= n < int(1.2 * fs):
            freeze[n] = True
            continue  # 双讲冻结
        w += mu * xv * e[n] / (xv @ xv + 1e-6)
    blk = 400
    nb = (N - taps) // blk
    erle, t_axis = [], []
    for b in range(nb):
        seg = slice(taps + b * blk, taps + (b + 1) * blk)
        if freeze[seg].any():
            erle.append(np.nan)
        else:
            erle.append(10 * np.log10(np.mean(echo[seg] ** 2) / (np.mean(e[seg] ** 2) + 1e-12)))
        t_axis.append((taps + b * blk) / fs)
    erle = np.array(erle, dtype=float)
    t_axis = np.array(t_axis)
    ax.plot(t_axis, erle, color=C_BLUE, lw=1.6)
    ax.axvspan(0.8, 1.2, color=C_RED, alpha=0.10)
    # 双讲期：ERLE 保持冻结前的值（水平虚线延续）
    pre = np.where((t_axis < 0.8) & ~np.isnan(erle))[0]
    v_freeze = erle[pre[-1]]
    ax.plot([0.8, 1.2], [v_freeze, v_freeze], color=C_BLUE, ls="--", lw=1.4)
    ax.text(1.0, v_freeze + 2.5, "冻结，ERLE 保持（虚线）", fontsize=FS_SMALL,
            color=C_BLUE, ha="center")
    # 收敛值从 erle 数组实测
    plateau_mask = (t_axis > 0.45) & (t_axis < 0.8) & ~np.isnan(erle)
    erle_plateau = np.mean(erle[plateau_mask])
    ax.annotate(f"收敛后 ERLE≈{erle_plateau:.0f} dB\n（线性滤波器，未含残余抑制）",
                xy=(0.62, erle_plateau), xytext=(0.18, erle_plateau + 7),
                fontsize=FS_LABEL, color=C_BLUE,
                arrowprops=dict(arrowstyle="->", color=C_BLUE))
    ax.text(1.0, 5, "双讲期\n(冻结更新)", fontsize=FS_LABEL - 0.5, color=C_RED, ha="center")
    ax.set_xlabel("时间 (s)", fontsize=FS_LABEL); ax.set_ylabel("ERLE (dB)", fontsize=FS_LABEL)
    ax.set_ylim(-10, 40)
    ax.set_title("(b) NLMS 收敛过程（ERLE，模拟）", fontsize=FS_TITLE + 1)
    ax.grid(ls=":", alpha=0.5)
    fig.suptitle("图18  声学回声消除（AEC）原理", fontsize=FS_SUP)
    fig.tight_layout()
    save(fig, "fig18_aec.png")


# ----------------------------------------------------------------------
# 图21 WPE 去混响：混响语音 → WPE → 去混响语音（真实算法演示）
# ----------------------------------------------------------------------
def wpe_dereverb(Y, K=10, delay=3, iters=3):
    """单通道 WPE (Nakatani 2010)。Y: F x T 复数 STFT。"""
    F, T = Y.shape
    X = Y.copy()
    lam_floor = 1e-5 * np.abs(Y).max() ** 2
    for _ in range(iters):
        lam = np.maximum(np.abs(X) ** 2, lam_floor)
        # 沿时间轴平滑功率谱，防止静音帧权重过大
        ker = np.ones(5) / 5
        lam = np.apply_along_axis(lambda v: np.convolve(v, ker, mode="same"), 1, lam)
        G = np.zeros((F, K), dtype=complex)
        t0 = delay + K - 1
        if T <= t0 + 1:
            break
        # 构造过去帧堆叠 (F, K, T-t0)
        Ypast = np.stack([Y[:, t0 - delay - k + 1:T - delay - k + 1] for k in range(K)], axis=1)
        ycur = Y[:, t0:]
        w = 1.0 / lam[:, t0:]
        for f in range(F):
            P = Ypast[f]  # K x Tv
            ww = w[f]
            Rw = (P * ww[None, :]) @ P.conj().T
            rw = (P * ww[None, :]) @ ycur[f].conj()
            G[f] = np.linalg.solve(Rw + 1e-6 * np.eye(K), rw).conj()
        X[:, t0:] = ycur - np.einsum("fk,fkt->ft", G.conj(), Ypast)
    return X


def fig_wpe():
    fs = 16000
    t, clean = synth_speech(fs, 1.4)
    T60 = 0.6
    L = int(0.8 * fs)
    tt = np.arange(L) / fs
    rir = rng.standard_normal(L) * np.exp(-6.91 * tt / T60)
    rir[:int(0.01 * fs)] = 0
    rir[int(0.01 * fs)] = 1.0
    rir /= np.sqrt(np.sum(rir ** 2))
    rev = np.convolve(clean, rir)[:len(clean)]
    n_fft, hop = 512, 128
    win = np.hanning(n_fft)
    def stft(x):
        fr = [x[i:i + n_fft] * win for i in range(0, len(x) - n_fft, hop)]
        return np.array([np.fft.rfft(f) for f in fr]).T
    Yc, Yr = stft(clean), stft(rev)
    Xd = wpe_dereverb(Yr, K=8, delay=2, iters=3)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0))
    tms = np.arange(Yc.shape[1]) * hop / fs * 1000
    fk = np.fft.rfftfreq(n_fft, 1 / fs) / 1000
    vm = 20 * np.log10(np.abs(Yc).max())      # 三图统一色标，便于直接对比
    mesh = None
    for ax, (S, title) in zip(axes, [(Yc, "(a) 干净语音（音节边界清晰）"),
                                     (Yr, "(b) 混响语音：能量沿时间拖尾、边界糊化"),
                                     (Xd, "(c) WPE 去混响后：拖尾被抑制")]):
        Sdb = 20 * np.log10(np.abs(S) + 1e-8)
        mesh = ax.pcolormesh(tms, fk, Sdb, cmap="viridis", shading="auto",
                             vmin=vm - 55, vmax=vm)
        ax.set_ylim(0, 2.5); ax.set_xlabel("时间 (ms)", fontsize=FS_LABEL)
        ax.text(0.98, 0.04, "高频无能量故全黑", transform=ax.transAxes, fontsize=FS_TINY, color="white", ha="right", va="bottom")
        ax.set_ylabel("频率 (kHz)", fontsize=FS_LABEL)
        ax.set_title(title, fontsize=FS_TITLE)
    fig.suptitle("图21  WPE 去混响效果（$T_{60}$=0.6s 合成混响，单通道真实算法演示）", fontsize=FS_SUP)
    fig.tight_layout()
    cb = fig.colorbar(mesh, ax=axes, shrink=0.85, pad=0.015, label="幅度 (dB)")
    cb.ax.tick_params(labelsize=FS_SMALL + 1)
    cb.set_label("幅度 (dB)", fontsize=FS_LABEL + 2)
    cb.outline.set_linewidth(1.3)
    save(fig, "fig21_wpe.png")


# ----------------------------------------------------------------------
# 图1 人类双耳定位线索：ITD / ILD
# ----------------------------------------------------------------------
def fig_binaural():
    fig = plt.figure(figsize=(14, 4.2))
    ax = fig.add_subplot(1, 3, 1)
    ax.set_aspect("equal"); ax.axis("off")
    a = 1.0
    ax.add_patch(Circle((0, 0), a, fc="#f6e5db", ec="k", lw=1.5))
    ax.scatter([-a, a], [0, 0], s=180, c=C_BLUE, zorder=6, edgecolors="k")
    ax.annotate("左耳", xy=(-a, 0), xytext=(-a - 0.1, -0.35), fontsize=10, ha="center")
    ax.annotate("右耳", xy=(a, 0), xytext=(a + 0.1, -0.35), fontsize=10, ha="center")
    th = np.deg2rad(60)
    for dy in np.linspace(-0.9, 0.9, 5):
        p1 = np.array([3.2 * np.cos(th) - dy * np.sin(th), 3.2 * np.sin(th) + dy * np.cos(th)])
        p2 = np.array([0.9 * np.cos(th) - dy * np.sin(th), 0.9 * np.sin(th) + dy * np.cos(th)])
        ax.annotate("", xy=tuple(p2), xytext=tuple(p1),
                    arrowprops=dict(arrowstyle="->", color=C_BLUE, alpha=0.55, lw=1.2))
    ax.annotate("", xy=(1.15 * np.cos(th), 1.15 * np.sin(th)), xytext=(3.1 * np.cos(th), 3.1 * np.sin(th)),
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.8))
    ax.text(3.3 * np.cos(th) + 0.05, 3.3 * np.sin(th), "声源方向", fontsize=10, color=C_RED)
    ax.text(-1.85, -1.35, "头影：远侧耳被头遮挡\n路程差 → 双耳时间差 ITD\n遮挡 → 双耳声级差 ILD",
            fontsize=FS_LABEL - 0.5, ha="left", va="top", color=C_MAIN)
    ax.set_xlim(-1.9, 3.6); ax.set_ylim(-2.6, 3.6)
    ax.set_title("(a) 头 + 双耳 = 天然的 2 麦阵列（带挡板）", fontsize=11)
    ax = fig.add_subplot(1, 3, 2)
    az = np.linspace(-90, 90, 400)
    a_head, c = 0.0875, 343
    itd = (a_head / c) * (np.deg2rad(az) + np.sin(np.deg2rad(az))) * 1e6
    ax.plot(az, itd, color=C_BLUE, lw=2)
    ax.axhline(0, color="k", lw=0.6)
    ax.axhline(700, color="gray", ls="--", lw=0.8)
    ax.annotate("最大值 ≈ 700 μs（正侧面 90°）", xy=(85, 690), xytext=(-5, 520), fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.annotate("正前方 1° ≈ 10 μs\n（人类可分辨极限）", xy=(1, 10), xytext=(-60, 300), fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color=C_RED), color=C_RED)
    ax.set_xlabel("声源方位角 (°)"); ax.set_ylabel("ITD (μs)")
    ax.set_title("(b) 双耳时间差 ITD（Woodworth 模型）", fontsize=11)
    ax.grid(ls=":", alpha=0.5)
    ax = fig.add_subplot(1, 3, 3)
    f = np.logspace(np.log10(100), np.log10(10000), 300)
    for az_d, ccurve in [(90, C_RED), (60, C_ORANGE), (30, C_BLUE)]:
        ild = 25 * np.clip((f - 400) / 5600, 0, 1) ** 1.3 * np.sin(np.deg2rad(az_d))
        ax.semilogx(f, ild, color=ccurve, lw=2, label=f"方位 {az_d}°")
    ax.axvspan(100, 1500, color=C_BLUE, alpha=0.06)
    ax.axvspan(3000, 10000, color=C_RED, alpha=0.06)
    ax.text(200, 21, "低频段：ITD 为主\n(相位可用 <1.5 kHz)", fontsize=FS_LABEL - 0.5, color=C_BLUE)
    ax.text(9500, 1.2, "高频段：ILD 为主\n(头影 >3 kHz 显著)", fontsize=FS_LABEL - 0.5, color=C_RED,
            ha="right")
    ax.set_xlabel("频率 (Hz)"); ax.set_ylabel("ILD (dB)")
    ax.set_ylim(0, 27); ax.legend(fontsize=9, loc="lower left")
    ax.set_title("(c) 双耳声级差 ILD 随频率变化（示意）", fontsize=11)
    ax.grid(ls=":", alpha=0.5, which="both")
    fig.suptitle("图1  人类双耳听觉的定位线索（Rayleigh 双重理论）", fontsize=13)
    fig.tight_layout()
    save(fig, "fig01_binaural.png")


# ----------------------------------------------------------------------
# 图13 GCC-PHAT 在混响下的退化（真实 RIR 模拟）
# ----------------------------------------------------------------------
def fig_gcc_reverb():
    fs = 16000
    N = fs
    tau_true = 9  # 直达声时延 9 个采样点 ≈ 0.56 ms

    def make_src(seed):
        r = np.random.default_rng(seed)
        w = r.standard_normal(N)
        W = np.fft.rfft(w)
        W[np.fft.rfftfreq(N, 1 / fs) > 6000] = 0
        s = np.fft.irfft(W)
        return s / np.max(np.abs(s))

    def gcc_phat(x1, x2, n_fft=4096, half=64):
        S = np.fft.rfft(x1, n_fft) * np.conj(np.fft.rfft(x2, n_fft))
        g = np.fft.fftshift(np.fft.irfft(S / (np.abs(S) + 1e-10)))
        lags = np.arange(-n_fft // 2, n_fft // 2)
        m = (lags >= -half) & (lags <= half)
        return lags[m], g[m]

    def trial(T60, seed, frame=1024):
        """一次试验：返回 (是否找对峰, 峰对比度)。直达声增益固定为 1，混响尾能量比 ≈ 2.9×T60（逼近真实房间 DRR 量级）；
        与真实系统一致，GCC 在 64 ms 短帧上计算"""
        r = np.random.default_rng(seed)
        L = int(0.6 * fs)
        t = np.arange(L) / fs
        tail_scale = np.sqrt(2.9 * 13.82 / fs)  # 使 尾/直达 能量比 ≈ 2.9·T60
        rir1 = r.standard_normal(L) * np.exp(-6.91 * t / T60) * tail_scale
        rir2 = r.standard_normal(L) * np.exp(-6.91 * t / T60) * tail_scale
        rir1[0], rir2[0] = 1.0, 1.0
        sig = make_src(seed + 1000)
        x1 = np.convolve(sig, rir1)[:N]
        x2 = np.convolve(np.roll(sig, tau_true), rir2)[:N]
        i0 = r.integers(0, N - frame - 1)
        f1 = x1[i0:i0 + frame]
        f2 = x2[i0:i0 + frame]
        lags, g = gcc_phat(f1, f2)
        pk = lags[np.argmax(g)]
        contrast = np.max(g) / (np.median(np.abs(g)) + 1e-12)
        return abs(pk + tau_true) <= 1, contrast

    # (a) 示例互相关曲线
    fig = plt.figure(figsize=(14.5, 4.6))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.6])
    lags_ms = np.arange(-64, 65) / fs * 1000
    for i, (T60, ttl) in enumerate([(0.05, "(a) $T_{60}$=0.05s：单帧示例"),
                                    (0.3, "(b) $T_{60}$=0.3s：单帧示例"),
                                    (0.6, "(c) $T_{60}$=0.6s：单帧示例"),
                                    (1.0, "(d) $T_{60}$=1.0s：单帧示例")]):
        ax = fig.add_subplot(gs[i])
        r = np.random.default_rng(7)
        L = int(0.6 * fs)
        t = np.arange(L) / fs
        tail_scale = np.sqrt(2.9 * 13.82 / fs)
        rir1 = r.standard_normal(L) * np.exp(-6.91 * t / T60) * tail_scale
        rir2 = r.standard_normal(L) * np.exp(-6.91 * t / T60) * tail_scale
        rir1[0], rir2[0] = 1.0, 1.0
        sig = make_src(3)
        x1 = np.convolve(sig, rir1)[:N]
        x2 = np.convolve(np.roll(sig, tau_true), rir2)[:N]
        i0 = 4000
        f1 = x1[i0:i0 + 1024]
        f2 = x2[i0:i0 + 1024]
        lags, g = gcc_phat(f1, f2)
        g = g / np.max(g)
        ax.plot(lags_ms, g, color=C_BLUE, lw=1.1)
        ax.set_ylim(-0.6, 1.1)
        ax.axvline(-tau_true / fs * 1000, color="k", ls="--", lw=1, alpha=0.6)
        pk = lags[np.argmax(g)]
        ax.plot(lags_ms[np.argmax(g)], 1.0, "r^", ms=10)
        ok = abs(pk + tau_true) <= 1
        ax.set_title(ttl + ("\n本帧峰位置正确" if ok else "\n本帧找错峰"), fontsize=10.5,
                     color=C_GREEN if ok else C_RED)
        if T60 >= 0.6:
            ax.text(0.5, 0.88, "注意野峰抬高", transform=ax.transAxes, fontsize=FS_SMALL,
                    color=C_RED, ha="center",
                    bbox=dict(fc="white", ec=C_RED, lw=0.7, alpha=0.9, boxstyle="round,pad=0.25"))
        ax.set_xlabel("时延 (ms)")
        ax.grid(ls=":", alpha=0.5)
        if i == 0:
            ax.set_ylabel("GCC-PHAT（归一化）")
    # (e) 蒙特卡洛：峰正确率与峰对比度随 T60
    ax = fig.add_subplot(gs[4])
    T60s = [0.05, 0.2, 0.4, 0.6, 0.8, 1.0]
    acc, contrast = [], []
    for T60 in T60s:
        res = [trial(T60, 2000 + k * 17) for k in range(150)]
        acc.append(np.mean([r[0] for r in res]) * 100)
        contrast.append(np.median([r[1] for r in res]))
    ax.plot(T60s, acc, "o-", color=C_RED, lw=1.8, label="峰位置正确率（左轴）")
    ax.set_xlabel("混响时间 $T_{60}$ (s)")
    ax.set_ylabel("正确率 (%)", color=C_RED)
    ax.set_ylim(0, 105)
    ax.tick_params(axis="y", labelcolor=C_RED, labelsize=FS_SMALL)
    ax.tick_params(axis="x", labelsize=FS_SMALL)
    ax2 = ax.twinx()
    ax2.plot(T60s, contrast, "s--", color=C_BLUE, lw=1.8, label="峰对比度（右轴）")
    ax2.set_ylabel("峰/背景中位数 比值", color=C_BLUE)
    ax2.tick_params(axis="y", labelcolor=C_BLUE, labelsize=FS_SMALL)
    h1, lb1 = ax.get_legend_handles_labels()
    h2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, lb1 + lb2, fontsize=FS_SMALL, loc="upper left")
    ax.set_title("(e) 150 次随机试验统计（64ms 短帧）：\n混响越大峰越不可靠", fontsize=10.5)
    ax.grid(ls=":", alpha=0.5)
    fig.suptitle("图13  混响如何打败 GCC-PHAT\n（黑虚线=直达声时延；(a)–(d) 为 64ms 单帧示例，(e) 为 150 次随机试验统计；无传感器噪声，真实数值模拟）", fontsize=FS_SUP)
    fig.tight_layout()
    save(fig, "fig13_gcc_reverb.png")


# ----------------------------------------------------------------------
# 图4 时延 = 相位旋转：复数表示的几何直觉
# ----------------------------------------------------------------------
def fig_delay_phase():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    # (a) 时域：正弦延迟
    ax = axes[0]
    f0 = 1000.0
    t = np.linspace(0, 2.2e-3, 800)
    tau = 0.25e-3
    ax.plot(t * 1e3, np.sin(2 * np.pi * f0 * t), color=C_BLUE, lw=2, label="原始信号 s(t)")
    ax.plot(t * 1e3, np.sin(2 * np.pi * f0 * (t - tau)), color=C_RED, lw=2, ls="--",
            label="延迟 τ 后 s(t−τ)")
    ax.annotate("", xy=(1.25, -0.707), xytext=(1.0, -0.707),
                arrowprops=dict(arrowstyle="->", color="k", lw=1.5))
    ax.text(1.12, -1.15, "τ", fontsize=13, ha="center")
    ax.set_xlabel("时间 (ms)"); ax.set_ylabel("幅度")
    ax.set_title("(a) 时域看延迟：整条波形向右平移", fontsize=11)
    ax.legend(fontsize=9, loc="upper right"); ax.grid(ls=":", alpha=0.5)
    ax.set_ylim(-1.5, 1.5)
    # (b) 复平面：相位旋转
    ax = axes[1]
    ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Circle((0, 0), 1, fill=False, ec="gray", lw=1.2))
    ax.axhline(0, color="gray", lw=0.6); ax.axvline(0, color="gray", lw=0.6)
    ax.annotate("", xy=(1, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=C_BLUE, lw=2.5, mutation_scale=18))
    ang = -np.pi / 2
    ax.annotate("", xy=(np.cos(ang), np.sin(ang)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=2.5, mutation_scale=18))
    arc = matplotlib.patches.Arc((0, 0), 1.2, 1.2, theta1=-90, theta2=0, color="k", lw=1.2)
    ax.add_patch(arc)
    ax.annotate("", xy=(0.66, -0.52), xytext=(0.66, -0.3),
                arrowprops=dict(arrowstyle="->", color="k", lw=1.2))
    ax.text(1.05, 0.06, "原始 $S(f)$", fontsize=10, color=C_BLUE)
    ax.text(0.05, -1.18, "延迟后 S(f)·e^(−j2πfτ)", fontsize=10, color=C_RED)
    ax.text(0.62, -0.12, "2πfτ", fontsize=13, style="italic")
    ax.text(-1.55, 1.15, "延迟 τ 等价于复平面上\n顺时针转 2πfτ", fontsize=10, color=C_MAIN)
    ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.55, 1.5)
    ax.set_title("(b) 频域看延迟：复数相位转过一个角度", fontsize=11)
    # (c) 相位-频率直线
    ax = axes[2]
    f = np.linspace(0, 8000, 400)
    tau2 = 0.116e-3  # 4cm 麦距正侧向
    phase = -2 * np.pi * f * tau2
    ax.plot(f / 1000, np.rad2deg(phase), color=C_BLUE, lw=2)
    for fk in [1000, 2000, 3000, 4000]:
        ax.plot(fk / 1000, np.rad2deg(-2 * np.pi * fk * tau2), "o", color=C_RED, ms=6)
    ax.annotate("1 kHz → -42°\n2 kHz → -84°\n转角与频率成正比\n斜率 = -2πτ", xy=(3000, -125),
                xytext=(3600, -55), fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color="k"))
    ax.set_xlabel("频率 (kHz)"); ax.set_ylabel("相位 (°)")
    ax.set_title("(c) 同一延迟在不同频率：相位-频率是直线", fontsize=11)
    ax.grid(ls=":", alpha=0.5)
    fig.suptitle("图4  时延为什么变成 $e^{-j2\\pi f\\tau}$：几何直觉（4 cm 麦距、正侧向、τ=0.116 ms）", fontsize=12.5)
    fig.tight_layout()
    save(fig, "fig04_delay_phase.png")


# ----------------------------------------------------------------------
# 图7 波束图解剖 + 栅瓣演示
# ----------------------------------------------------------------------
def fig_beampattern_anatomy():
    M = 8
    th = np.linspace(-90, 90, 3601)
    tr = np.deg2rad(th)
    m = np.arange(M) - (M - 1) / 2

    def pattern(d_over_lam, steer_deg):
        a0 = np.exp(-2j * np.pi * d_over_lam * m * np.sin(np.deg2rad(steer_deg)))
        w = a0 / M
        A = np.exp(-2j * np.pi * d_over_lam * np.outer(m, np.sin(tr)))
        B = np.abs(w.conj() @ A)
        return 20 * np.log10(np.maximum(B / B.max(), 1e-5))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    # (a) 解剖图
    ax = axes[0]
    Bdb = pattern(0.5, 30)
    ax.plot(th, Bdb, color=C_BLUE, lw=1.8)
    ax.axhline(-3, color="gray", ls="--", lw=1)
    ax.axhline(-13.3, color=C_ORANGE, ls="--", lw=1)
    ax.set_ylim(-40, 3); ax.set_xlim(-90, 90)
    # HPBW 实测
    i0 = np.argmax(Bdb)
    left = np.where(Bdb[:i0] < -3)[0][-1]; right = i0 + np.where(Bdb[i0:] < -3)[0][0]
    ax.annotate("", xy=(th[right], -3), xytext=(th[left], -3),
                arrowprops=dict(arrowstyle="<->", color=C_RED, lw=1.5))
    ax.text(30, -1.4, f"半功率波束宽度 HPBW ≈ {th[right]-th[left]:.1f}°（-3dB 线上量）", ha="center", fontsize=10, color=C_RED,
            bbox=dict(fc="white", ec=C_RED, lw=0.7, alpha=0.9, boxstyle="round,pad=0.25"))
    ax.annotate("主瓣（想听的方向 30°）", xy=(29, -1), xytext=(-18, -4), fontsize=10,
                arrowprops=dict(arrowstyle="->", color="k"))
    ax.annotate("最高旁瓣 ≈ −13 dB\n（旁瓣电平 SLL）", xy=(54, -13.5), xytext=(30, -32), fontsize=10,
                color=C_ORANGE, arrowprops=dict(arrowstyle="->", color=C_ORANGE))
    ax.annotate("旁瓣：其他方向漏进来的通道", xy=(-32, -17), xytext=(-75, -8), fontsize=10,
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.annotate("零点（相消干涉处）", xy=(8.5, -38), xytext=(-30, -34), fontsize=10,
                arrowprops=dict(arrowstyle="->", color="k"))
    ax.set_xlabel("方向 (°)"); ax.set_ylabel("增益 (dB)")
    ax.set_title("(a) 波束图解剖：8 麦线阵、间距 λ/2、指向 30°", fontsize=11)
    ax.grid(ls=":", alpha=0.4)
    # (b) 栅瓣
    ax = axes[1]
    ax.plot(th, pattern(0.5, 60), color=C_BLUE, lw=1.8, ls="--", label="d = λ/2（安全）")
    ax.plot(th, pattern(1.0, 60), color=C_RED, lw=2.5, label="d = λ（超半波长）")
    ax.axvline(-7.7, color="gray", ls=":", lw=1.2, alpha=0.8)
    ax.text(0.03, 0.94, "栅瓣条件（独立成行）：sinθ = sin60° − λ/d", transform=ax.transAxes,
            fontsize=FS_LABEL, color=C_RED, va="top", ha="left",
            bbox=dict(fc="white", ec="0.7", alpha=0.9, boxstyle="round,pad=0.3"))
    ax.annotate("主瓣 60°", xy=(60, 0.3), xytext=(40, 5), fontsize=10,
                arrowprops=dict(arrowstyle="->", color="k"))
    ax.annotate("栅瓣 @ −7.7°：\n与主瓣等高的鬼影（分不清声源在哪边）", xy=(-7.7, 0.3), xytext=(-82, -10),
                fontsize=FS_LABEL, color=C_RED, arrowprops=dict(arrowstyle="->", color=C_RED))
    ax.set_ylim(-40, 8); ax.set_xlim(-90, 90)
    ax.set_xlabel("方向 (°)"); ax.set_ylabel("增益 (dB)")
    ax.legend(fontsize=12, loc="lower left")
    ax.set_title("(b) 栅瓣演示：同阵列指向 60°，间距超半波长后", fontsize=11)
    ax.grid(ls=":", alpha=0.4)
    fig.suptitle("图7  波束图怎么读：主瓣 / HPBW / 旁瓣 / 零点 / 栅瓣（模拟）", fontsize=13)
    fig.tight_layout()
    save(fig, "fig07_beampattern_anatomy.png")


# ----------------------------------------------------------------------
# 图2 时延的几何来源与角度约定：Δτ = d·sinθ/c
# ----------------------------------------------------------------------
def fig_dsin_geometry():
    d = 4.0                    # 麦距 4 cm（图面单位即 cm）
    theta = np.deg2rad(40)     # 示意入射角（自正横起算）
    u = np.array([np.sin(theta), np.cos(theta)])     # 指向声源的单位向量
    w = np.array([-np.cos(theta), np.sin(theta)])    # 波前方向（⊥ u）
    mic1 = np.array([0.0, 0.0]); mic2 = np.array([d, 0.0])
    cen = (mic1 + mic2) / 2
    ds = d * np.sin(theta)                            # 路程差 d·sinθ
    P = mic2 - ds * u                                 # 直角三角形顶点（垂足）

    fig = plt.figure(figsize=(13, 6.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[2.1, 1.0, 1.0], wspace=0.12, hspace=0.35)
    ax = fig.add_subplot(gs[:, :2])
    ax.set_aspect("equal"); ax.axis("off")
    # 波前（平行斜线，等间隔；间距取 d·sinθ，使相邻波前分别经过两麦）
    for k in range(3):
        c0 = k * ds * u
        p1, p2 = c0 - 4.2 * w, c0 + 4.2 * w
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=C_BLUE,
                alpha=0.85 if k == 0 else 0.4, lw=1.6, zorder=2)
    ax.text(5.75, 1.55, "波前（等相位面，平行等距）", fontsize=FS_SMALL, color=C_BLUE,
            ha="left", va="center")
    # 传播方向箭头（三条平行射线，均收在图界内）
    for off in (-1.8, 0.0, 1.8):
        base = 8.0 * u + off * w
        ax.add_patch(FancyArrowPatch(base, base - 1.5 * u, arrowstyle="-|>",
                                     mutation_scale=15, color=C_ORANGE, lw=1.6, zorder=3))
    ax.text(6.05, 4.35, "平面波传播方向", fontsize=FS_SMALL,
            color=C_ORANGE, ha="left", va="center")
    # 正横方向虚线
    ax.plot([cen[0], cen[0]], [-0.6, 6.8], color="gray", ls="--", lw=1.2, zorder=2)
    ax.text(1.8, 5.6, "正横方向 (broadside，θ=0°)", fontsize=FS_SMALL, color="gray",
            ha="right", va="center")
    # 声源射线（过阵列中心）
    ax.plot([cen[0], cen[0] + 8.4 * u[0]], [cen[1], cen[1] + 8.4 * u[1]],
            color=C_RED, lw=1.6, ls="-", zorder=3)
    ax.text(*(cen + 8.55 * u), "远场声源方向", fontsize=FS_LABEL, color=C_RED, ha="left", va="center")
    # 角度弧：正横 → 声源射线（θ 自正横起算）
    arc = matplotlib.patches.Arc(cen, 3.4, 3.4, theta1=90 - np.rad2deg(theta), theta2=90,
                                 color="k", lw=1.3)
    ax.add_patch(arc)
    ax.text(*(cen + 2.2 * np.array([np.sin(theta / 2), np.cos(theta / 2)]) + [0.12, 0.05]),
            "θ", fontsize=FS_SUP, style="italic")
    # 两个麦克风
    ax.scatter(*zip(mic1, mic2), s=200, c=C_BLUE, zorder=6, edgecolors="k")
    ax.annotate("麦1", xy=mic1, xytext=(-0.55, -0.55), fontsize=FS_LABEL, ha="right")
    ax.annotate("麦2", xy=mic2, xytext=(4.3, 0.3), fontsize=FS_LABEL, ha="left")
    ax.annotate("", xy=(4, -0.38), xytext=(0, -0.38), arrowprops=dict(arrowstyle="<->", color="k", lw=1.2))
    ax.text(2, -0.92, "d = 4 cm", ha="center", fontsize=FS_LABEL)
    # 直角三角形：麦1—垂足P 虚线，麦2—P 红色实线 = 路程差
    ax.plot([mic1[0], P[0]], [mic1[1], P[1]], color="gray", ls="--", lw=1.3, zorder=4)
    ax.plot([mic2[0], P[0]], [mic2[1], P[1]], color=C_RED, lw=2.4, zorder=5)
    ax.text(*((mic2 + P) / 2 + [0.5, -0.05]), "路程差\nd·sinθ", fontsize=FS_LABEL,
            color=C_RED, ha="left", va="center")
    # 直角符号
    s1, s2 = 0.42 * u, 0.42 * (mic1 - P) / np.linalg.norm(mic1 - P)
    sq = np.array([P, P + s1, P + s1 + s2, P + s2])
    ax.plot(sq[:, 0], sq[:, 1], color="k", lw=1.1, zorder=5)
    # 结论（左上角空白区）
    ax.text(-4.45, 7.45, "波前先到麦2，要多走 d·sinθ 才到麦1：\nΔτ = d·sinθ / c",
            fontsize=FS_LABEL + 0.5, color=C_MAIN, ha="left", va="top",
            bbox=dict(fc="#fdf6ec", ec=C_ORANGE, lw=1, boxstyle="round,pad=0.4"))
    ax.set_xlim(-4.6, 10.6); ax.set_ylim(-3.2, 7.8)
    ax.set_title("(a) 斜入射平面波：时延来自路程差 d·sinθ", fontsize=FS_TITLE)

    def inset(pos, title, mode):
        a = fig.add_subplot(pos)
        a.set_aspect("equal"); a.axis("off")
        m1, m2 = np.array([0.0, 0.0]), np.array([1.6, 0.0])
        a.scatter(*zip(m1, m2), s=140, c=C_BLUE, zorder=6, edgecolors="k")
        if mode == "broadside":   # θ=0：波前平行于两麦连线，同时到达
            for y in (1.1, 2.0, 2.9):
                a.plot([-0.9, 2.5], [y, y], color=C_BLUE, alpha=0.5, lw=1.5)
            a.add_patch(FancyArrowPatch((0.8, 3.2), (0.8, 2.2), arrowstyle="-|>",
                                        mutation_scale=13, color=C_ORANGE, lw=1.5))
            a.plot([-0.9, 2.5], [0.32, 0.32], color=C_GREEN, lw=2.2, zorder=4)
            a.text(0.8, -0.55, "波前同时压到两麦", fontsize=FS_TINY, color=C_GREEN, ha="center")
            a.set_xlim(-1.1, 2.7); a.set_ylim(-1.0, 3.4)
        else:                     # θ=90：波前垂直于连线，先后到达，Δτ 最大
            for x in (2.2, 3.0, 3.8):
                a.plot([x, x], [-1.3, 1.3], color=C_BLUE, alpha=0.5, lw=1.5)
            a.add_patch(FancyArrowPatch((4.1, 0.0), (3.2, 0.0), arrowstyle="-|>",
                                        mutation_scale=13, color=C_ORANGE, lw=1.5))
            a.annotate("", xy=(1.6, -0.55), xytext=(0, -0.55),
                       arrowprops=dict(arrowstyle="<->", color=C_RED, lw=1.2))
            a.text(0.8, -0.95, "d", fontsize=FS_SMALL, color=C_RED, ha="center")
            a.set_xlim(-0.6, 4.4); a.set_ylim(-1.5, 1.7)
        a.set_title(title, fontsize=FS_SMALL + 0.5)

    inset(gs[0, 2], "(b) 正横 θ=0°：同时到达，Δτ=0", "broadside")
    inset(gs[1, 2], "(c) 端射 θ=90°：沿连线方向，Δτ=d/c 最大", "endfire")
    fig.suptitle("图2  时延的几何来源与角度约定（示意）", fontsize=FS_SUP)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig02_dsin_geometry.png")


# ----------------------------------------------------------------------
# 图20 AEC 信号流管线：经典结构 vs 现代混合式（示意）
# ----------------------------------------------------------------------
def fig_aec_pipeline():
    fig = plt.figure(figsize=(13.5, 9.2))

    # ---- (a) 经典 AEC 信号流 ----
    ax = fig.add_subplot(2, 1, 1)
    ax.axis("off"); ax.set_xlim(0, 14); ax.set_ylim(0, 8)

    def box(x, y, w, h, text, fc="#dbe9f6", fs=FS_SMALL, ec="k", ls="-", lw=1.2):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, ls=ls, zorder=3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=4)

    def arrow(x1, y1, x2, y2, text="", c="k", dy=0.18, fs=FS_TINY):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=14, color=c, lw=1.4, zorder=2))
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, text, fontsize=fs, ha="center", color=c)

    def sum_node(cx, cy, r=0.32, label="Σ"):
        ax.add_patch(Circle((cx, cy), r, fc="white", ec="k", lw=1.4, zorder=3))
        ax.text(cx, cy, label, ha="center", va="center", fontsize=FS_LABEL, zorder=4)

    # 上排：远端 → 扬声器 → 房间 → Σ ← 近端
    box(0.3, 6.2, 2.1, 0.95, "远端播放 x(n)\n（已知参考）", "#f6e5db")
    box(3.2, 6.2, 1.7, 0.95, "数模转换\n功放+扬声器", fs=FS_TINY)
    box(5.7, 6.2, 2.5, 0.95, "房间回声路径 h(n)\n“房间的指纹”", fs=FS_TINY)
    arrow(2.4, 6.68, 3.2, 6.68); arrow(4.9, 6.68, 5.7, 6.68)
    sum_node(9.15, 6.68)
    arrow(8.2, 6.68, 8.82, 6.68)
    box(5.7, 4.3, 2.5, 0.9, "近端语音 s(n)\n+ 环境噪声 v(n)", "#f6dbdb", FS_TINY)
    arrow(6.95, 5.2, 8.95, 6.4, c=C_RED)
    box(10.3, 6.2, 2.3, 0.95, "麦克风信号 d(n)\n= s + x*h + v", "#f6dbdb", FS_TINY)
    arrow(9.48, 6.68, 10.3, 6.68)

    # 下排：参考 → 延迟对齐 → 自适应滤波 → Σ(减)
    box(0.3, 2.5, 1.9, 0.95, "延迟对齐 τ̂\n（先对表再抄）", fs=FS_TINY)
    box(3.2, 2.5, 2.6, 0.95, "自适应滤波器 ŵ(n)\n（NLMS，抄房间指纹）", fs=FS_TINY)
    box(6.7, 2.5, 1.7, 0.95, "回声副本\nŷ(n)=ŵ*x", fs=FS_TINY)
    sum_node(9.15, 2.98, label="−")
    arrow(1.25, 6.2, 1.25, 3.45, "参考", C_BLUE)
    arrow(2.2, 2.98, 3.2, 2.98)
    arrow(5.8, 2.98, 6.7, 2.98)
    arrow(8.4, 2.98, 8.82, 2.98)
    arrow(11.45, 6.2, 11.45, 3.7, "d(n)", C_RED)
    ax.plot([11.45, 11.45], [3.7, 3.7], lw=0)  # 占位
    ax.add_patch(FancyArrowPatch((11.45, 3.7), (9.5, 3.05), arrowstyle="-|>",
                                 mutation_scale=14, color=C_RED, lw=1.4, zorder=2))
    box(10.3, 2.5, 2.3, 0.95, "残差 e(n)=d−ŷ\n→ 后续模块", "#e8f6db", FS_TINY)
    arrow(9.48, 2.98, 10.3, 2.98)

    # e 反馈更新 + DTD 开关
    ax.plot([11.45, 11.45], [2.5, 1.15], color=C_BLUE, lw=1.4, zorder=2)
    ax.plot([4.5, 5.55], [1.15, 1.15], color=C_BLUE, lw=1.4, zorder=2)
    ax.plot([6.45, 11.45], [1.15, 1.15], color=C_BLUE, lw=1.4, zorder=2)
    ax.add_patch(FancyArrowPatch((4.5, 1.15), (4.5, 2.5), arrowstyle="-|>",
                                 mutation_scale=14, color=C_BLUE, lw=1.4, zorder=2))
    ax.text(8.6, 0.82, "e 越小 → 滤波器越像 h(n)（误差驱动更新）", fontsize=FS_TINY,
            color=C_BLUE, ha="center")
    # 反馈线上的“开关”（双讲时断开）
    ax.add_patch(plt.Rectangle((5.4, 0.95), 1.2, 0.4, fc="white", ec=C_RED, lw=1.4, zorder=4))
    ax.plot([5.62, 6.3], [1.3, 1.06], color=C_RED, lw=1.8, zorder=5)
    ax.text(6.0, 1.5, "开关（双讲时断开）", fontsize=FS_SMALL, color=C_RED, ha="center", zorder=5)
    # DTD 控制盒
    box(5.2, 0.0, 1.6, 0.75, "双讲检测\nDTD", "#f6e5db", FS_TINY)
    ax.add_patch(FancyArrowPatch((6.0, 0.75), (6.0, 1.0), arrowstyle="-|>",
                                 mutation_scale=12, color=C_RED, lw=1.2, zorder=4))

    # “抄指纹”标注
    ax.annotate("滤波器要抄的，就是上面这条指纹", xy=(4.5, 3.5), xytext=(4.5, 5.6),
                fontsize=FS_SMALL + 2, color=C_PURPLE, ha="center",
                arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=1.2,
                                connectionstyle="arc3,rad=-0.25"))
    ax.set_title("(a) 经典 AEC 信号流：已知参考学出回声副本再相减（示意）", fontsize=FS_TITLE)

    # ---- (b) 现代混合式管线 ----
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.axis("off"); ax2.set_xlim(0, 14); ax2.set_ylim(0, 8)

    def box2(x, y, w, h, text, fc="#dbe9f6", fs=FS_SMALL, ec="k", ls="-", lw=1.2):
        ax2.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, ls=ls, zorder=3))
        ax2.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=4)

    def arrow2(x1, y1, x2, y2, text="", c="k", dy=0.18, fs=FS_TINY):
        ax2.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                      mutation_scale=14, color=c, lw=1.4, zorder=2))
        if text:
            ax2.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, text, fontsize=fs, ha="center", color=c)

    C_DSP = "#dbe9f6"    # 传统 DSP（蓝）
    C_NN = "#fde3c8"     # 神经网络（橙）
    y0 = 4.6
    box2(0.3, y0, 1.8, 1.0, "远端参考 x(n)\n（播放链路末端取）", "#f6e5db", FS_TINY)
    box2(2.6, y0, 1.8, 1.0, "延迟对齐", C_DSP)
    box2(4.9, y0, 2.4, 1.0, "线性 AEC\n（PBFDAF / 频域 Kalman）", C_DSP, FS_TINY)
    box2(7.8, y0, 1.6, 1.0, "残余 e(n)\n+ 噪声", C_DSP, FS_TINY)
    box2(9.9, y0, 2.3, 1.0, "神经残余抑制器\n（DNN，输入 e 与 x）", C_NN, FS_TINY)
    box2(12.4, y0, 1.4, 1.0, "干净输出\n→ 波束/ASR", "#e8f6db", FS_TINY)
    for x1, x2 in [(2.1, 2.6), (4.4, 4.9), (7.3, 7.8), (9.4, 9.9), (12.2, 12.4)]:
        arrow2(x1, y0 + 0.5, x2, y0 + 0.5)
    box2(4.9, 6.3, 2.4, 0.9, "麦克风 d(n)", "#f6dbdb", FS_TINY)
    arrow2(6.1, 6.3, 6.1, y0 + 1.0)
    # 参考支路到 DNN
    ax2.plot([1.2, 1.2], [y0, 2.1], color=C_BLUE, lw=1.4, zorder=2, alpha=0.6)
    ax2.plot([1.2, 11.05], [2.1, 2.1], color=C_BLUE, lw=1.4, zorder=2, alpha=0.6)
    ax2.add_patch(FancyArrowPatch((11.05, 2.1), (11.05, y0), arrowstyle="-|>",
                                  mutation_scale=14, color=C_BLUE, lw=1.4, zorder=2))
    ax2.text(5.4, 2.28, "参考 x 也喂给 DNN（帮助分辨残余回声与近端语音）",
             fontsize=FS_SMALL + 1.5, color=C_BLUE, ha="center", alpha=0.9,
             bbox=dict(fc="white", alpha=0.7, pad=1, ec="none"))
    # DTD / 步长控制
    box2(3.4, 0.7, 5.4, 0.95,
         "DTD / 最优步长控制（经典：Geigel、NCC；现代：DNN 学习，如 DVSS）",
         "white", FS_TINY, ec=C_ORANGE, ls="--")
    arrow2(6.1, 1.65, 6.1, y0, "控制更新", C_ORANGE)
    # 分工注释
    ax2.text(6.1, 3.85, "线性 AEC 只消“线性回声”（物理问题，有免费午餐）",
             fontsize=FS_TINY, color=C_BLUE, ha="center")
    ax2.text(11.05, 3.85, "非线性残余、噪声、晚期混响交给 DNN 扫尾",
             fontsize=FS_TINY, color=C_ORANGE, ha="center")
    # 图例
    ax2.add_patch(plt.Rectangle((0.3, 7.3), 0.4, 0.4, fc=C_DSP, ec="k", lw=1.0))
    ax2.text(0.8, 7.5, "传统 DSP", fontsize=FS_TINY, va="center")
    ax2.add_patch(plt.Rectangle((2.6, 7.3), 0.4, 0.4, fc=C_NN, ec="k", lw=1.0))
    ax2.text(3.1, 7.5, "神经网络", fontsize=FS_TINY, va="center")
    ax2.set_title("(b) 现代混合式管线：线性滤波打底 + 神经网络扫尾（示意）", fontsize=FS_TITLE)

    fig.suptitle("图20  AEC 系统管线：从经典自适应结构到现代混合式（示意）", fontsize=FS_SUP)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "fig20_aec_pipeline.png")


# ----------------------------------------------------------------------
# 图19 AEC 全景：非线性天花板（真实模拟） + 算法家族演进时间线（定性）
# ----------------------------------------------------------------------
def fig_aec_landscape():
    fig = plt.figure(figsize=(13.5, 8.8))

    # ---- (a) 线性 AEC 的 20 dB 天花板（真实 NLMS 模拟） ----
    ax = fig.add_subplot(2, 1, 1)
    fs = 16000
    N = int(3.0 * fs)
    taps = 256
    rng_l = np.random.default_rng(7)
    h = np.exp(-np.arange(taps) / 60) * rng_l.standard_normal(taps)
    x = np.convolve(rng_l.standard_normal(N), np.ones(8) / 8)[:N]
    x = x / np.std(x) * 0.5
    drive = 1.1
    echo_lin = np.convolve(x, h)[:N]
    echo_nl = np.convolve(np.tanh(drive * x) / np.tanh(drive), h)[:N]

    def nlms_erle(d):
        w = np.zeros(taps); e = np.zeros(N); mu = 0.3
        for n in range(taps, N):
            xv = x[n - taps:n][::-1]
            y = w @ xv
            e[n] = d[n] - y
            w += mu * xv * e[n] / (xv @ xv + 1e-6)
        blk = 800
        erle, tt = [], []
        for b in range((N - taps) // blk):
            seg = slice(taps + b * blk, taps + (b + 1) * blk)
            erle.append(10 * np.log10(np.mean(d[seg] ** 2) / (np.mean(e[seg] ** 2) + 1e-15)))
            tt.append((taps + b * blk) / fs)
        return np.array(tt), np.array(erle)

    t1, e_lin = nlms_erle(echo_lin)
    t2, e_nl = nlms_erle(echo_nl)
    ax.plot(t1, e_lin, color=C_BLUE, lw=1.6, label="线性回声路径")
    ax.plot(t2, e_nl, color=C_RED, lw=1.6, label="非线性回声路径（tanh 软削波）")
    p_lin = np.mean(e_lin[t1 > 2.2]); p_nl = np.mean(e_nl[t2 > 2.2])
    ax.axhline(20, color=C_ORANGE, ls="--", lw=1.2)
    ax.text(2.12, 17.6, "低成本设备非线性“底色”≈20 dB", fontsize=FS_SMALL, color=C_ORANGE)
    ax.annotate(f"线性路径：平台 ≈{p_lin:.0f} dB", xy=(2.5, p_lin), xytext=(1.7, p_lin + 6),
                fontsize=FS_SMALL + 1, color=C_BLUE,
                arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.4))
    ax.annotate(f"非线性路径：封顶 ≈{p_nl:.0f} dB，再久也上不去\n（未建模失真擦不掉 → 需 NLP / 非线性建模 / 换参考）",
                xy=(1.3, p_nl - 0.5), xytext=(0.5, 7),
                fontsize=FS_SMALL + 1, color=C_RED,
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.4))
    ax.set_xlabel("时间 (s)", fontsize=FS_LABEL)
    ax.set_ylabel("ERLE (dB)", fontsize=FS_LABEL)
    ax.set_ylim(-5, 45); ax.set_xlim(0, 3)
    ax.legend(fontsize=FS_SMALL, loc="upper left")
    ax.grid(ls=":", alpha=0.5)
    ax.set_title("(a) 线性 NLMS 的天花板：非线性回声路径下 ERLE 封顶（模拟；收敛后线性≈37dB理想值、非线性封顶≈20dB）", fontsize=FS_TITLE)

    # ---- (b) AEC 算法家族演进时间线（定性） ----
    ax2 = fig.add_subplot(2, 1, 2)
    # (年份, 注释, 竖直层)——四层交错防止相邻标签叠字
    # 口径（D5 定稿，已核实）：1960 LMS（Widrow&Hoff）/ 1967 NLMS（Nagumo&Noda）；
    # 1987 MDF（Soo&Pang）→ 1992 PBFDAF 分区实现；最优步长控制记 2000 年
    # （Mäder/Puder/Schmidt, Signal Processing 2000，与 PNLMS 同年，两圆点重合，标签上下错开）。
    events = [
        (1960, "LMS（Widrow&Hoff 1960）/ NLMS（Nagumo&Noda 1967）\n随机梯度自适应滤波", 0.9),
        (1987, "MDF（Soo&Pang 1987）→ PBFDAF（1992 分区实现）\n分区块频域，长滤波器实用化", -0.9),
        (2000, "PNLMS → IPNLMS(2002)\n按系数幅度比例分配步长", 0.9),
        (2000, "最优步长控制\nDTD 从冻结走向连续调节", -1.15),
        (2006, "频域卡尔曼 FDKF\n增益 = 最优时变步长", 1.75),
        (2019, "端到端 DNN-AEC\n监督分离表述", -0.9),
        (2021, "AEC Challenge 创办\n混合式（线性+神经）夺冠", 0.9),
        (2023, "pAEC / DeepVQE\n个性化 + 联合增强", -1.75),
        (2025, "端侧小模型 / 扩散残余抑制\n生成式方法进场", 1.75),
    ]
    ax2.axhline(0, color="k", lw=1.6, zorder=2)
    for yr, txt, y_txt in events:
        ax2.scatter([yr], [0], s=64, color=C_BLUE if yr < 2010 else C_ORANGE, zorder=5,
                    edgecolors="k", lw=0.8)
        ax2.plot([yr, yr], [0, y_txt - 0.30 * np.sign(y_txt)], color="gray", lw=1.0, zorder=3)
        ax2.text(yr, y_txt, f"{yr}\n{txt}", fontsize=FS_SMALL, ha="center",
                 va="bottom" if y_txt > 0 else "top", color=C_MAIN)
    ax2.scatter([], [], s=80, color=C_BLUE, edgecolors="k", label="解析 / 自适应滤波时代")
    ax2.scatter([], [], s=80, color=C_ORANGE, edgecolors="k", label="深度学习时代")
    ax2.legend(fontsize=FS_SMALL, loc="lower left", framealpha=0.9)
    ax2.set_xlim(1956, 2031); ax2.set_ylim(-3.1, 3.1)
    ax2.set_yticks([])
    ax2.set_xticks([1960, 1970, 1980, 1990, 2000, 2010, 2020, 2030])
    ax2.tick_params(labelsize=FS_SMALL)
    ax2.set_xlabel("年份", fontsize=FS_LABEL)
    for sp in ("left", "right", "top"):
        ax2.spines[sp].set_visible(False)
    ax2.set_title("(b) AEC 算法家族演进时间线（定性）", fontsize=FS_TITLE)

    fig.suptitle("图19  AEC 的能力边界与算法版图", fontsize=FS_SUP)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "fig19_aec_landscape.png")


# ----------------------------------------------------------------------
# 图24 WPE 的 Δ/K 延迟预测结构：只用过去帧预测当前帧（示意）
# ----------------------------------------------------------------------
def fig_wpe_frames():
    K, delay = 5, 3
    n_show = 12  # 画面上显示的过去帧数
    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.axis("off"); ax.set_xlim(0, 15.4); ax.set_ylim(0, 6)
    bw, gap, x0, y0 = 0.95, 0.12, 0.6, 3.0
    # 从左到右：更过去 … t-K-delay … 历史窗K … 间隔Δ … 当前帧t
    labels = {}
    for i in range(n_show + 1):
        x = x0 + i * (bw + gap)
        if i == n_show:
            fc, ec, lw, tag = "#f6dbdb", C_RED, 2.0, "当前帧 t\n（含直达+早期+晚期）"
        elif n_show - delay <= i <= n_show - 1:
            fc, ec, lw, tag = "#eeeeee", "0.5", 1.0, None
        elif n_show - delay - K <= i <= n_show - delay - 1:
            fc, ec, lw, tag = "#dbe9f6", C_BLUE, 1.6, None
        else:
            fc, ec, lw, tag = "#f7f7f7", "0.7", 1.0, None
        ax.add_patch(plt.Rectangle((x, y0), bw, 1.1, fc=fc, ec=ec, lw=lw, zorder=3))
        if i == n_show:
            ax.text(x + bw / 2, y0 + 0.55, "t", ha="center", va="center",
                    fontsize=FS_TITLE, color=C_RED, weight="bold", zorder=4)
            ax.text(x + bw / 2, y0 - 0.35, tag, ha="center", va="top", fontsize=FS_SMALL, color=C_RED)
        labels[i] = x
    i_gap0, i_gap1 = n_show - delay, n_show - 1
    i_k0, i_k1 = n_show - delay - K, n_show - delay - 1
    x_gap = (labels[i_gap0] + labels[i_gap1] + bw) / 2
    x_k = (labels[i_k0] + labels[i_k1] + bw) / 2
    ax.annotate("", xy=(labels[i_gap0], y0 + 1.35), xytext=(labels[i_gap1] + bw, y0 + 1.35),
                arrowprops=dict(arrowstyle="<->", color="0.45", lw=1.4))
    ax.text(x_gap, y0 + 1.6, f"保护间隔 Δ={delay}帧\n（跳过强相关早期反射）", ha="center", va="bottom",
            fontsize=FS_SMALL + 2, color="0.35")
    ax.annotate("", xy=(labels[i_k0], y0 + 1.35), xytext=(labels[i_k1] + bw, y0 + 1.35),
                arrowprops=dict(arrowstyle="<->", color=C_BLUE, lw=1.6))
    ax.text(x_k, y0 + 1.6, f"历史窗 K={K}帧\n（回归器：只用过去）", ha="center", va="bottom",
            fontsize=FS_SMALL + 2, color=C_BLUE,
            bbox=dict(fc="white", ec=C_BLUE, lw=0.7, alpha=0.9, boxstyle="round,pad=0.25"))
    # 预测箭头：历史窗底部 → 当前帧底部（走条带下方，不压框）
    ax.add_patch(FancyArrowPatch((x_k, y0 - 0.12), (labels[n_show] + bw / 2, y0 - 0.12),
                                 arrowstyle="-|>", mutation_scale=14, color=C_GREEN, lw=1.5, zorder=2))
    ax.text((x_k + labels[n_show]) / 2, y0 - 0.32,
            "晚期混响估计 ŷ(t)=ΣG(k)·y(t−Δ−k)，y(t)−ŷ(t) 即去混响",
            ha="center", va="top", fontsize=FS_SMALL + 2, color=C_GREEN,
            bbox=dict(fc="white", alpha=0.8, pad=1, ec="none"))
    ax.text(7.4, 1.15, "无前瞻、因果处理，故低延迟（可流式）", ha="center", va="center",
            fontsize=FS_LABEL, color=C_MAIN,
            bbox=dict(fc="#e8f6db", ec=C_GREEN, lw=0.8, alpha=0.9, boxstyle="round,pad=0.35"))
    fig.suptitle("图24  WPE 延迟预测结构：Δ=3帧保护间隔 + K=5帧历史窗，只用过去预测当前（示意）", fontsize=FS_SUP)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, "fig24_wpe_frames.png")


# ----------------------------------------------------------------------
# 图25 §10.1 延迟预算瀑布条：解析链 + DNN 大头 + 300ms 预算线（示意）
# ----------------------------------------------------------------------
def fig_latency_budget():
    segs = [("多通道采集/组帧", 32, "#dbe9f6"),
            ("AEC", 16, C_BLUE),
            ("WPE(流式)", 24, C_GREEN),
            ("波束形成", 12, C_PURPLE),
            ("声源追踪", 8, C_ORANGE),
            ("后置/DNN降噪", 120, C_RED),
            ("ASR解码", 60, "0.45")]
    total = sum(v for _, v, _ in segs)
    fig, ax = plt.subplots(figsize=(13.5, 3.8))
    left = 0
    for name, v, c in segs:
        ax.barh(0, v, left=left, height=0.45, color=c, edgecolor="k", lw=1.0, label=f"{name} {v}ms")
        if v >= 16:
            ax.text(left + v / 2, 0, f"{name}\n{v}ms", ha="center", va="center",
                    fontsize=FS_SMALL, color="white" if c in (C_BLUE, C_RED) else C_MAIN)
        left += v
    ax.set_ylim(-0.8, 1.1)
    ax.axvline(300, color=C_RED, ls="--", lw=2.0)
    ax.text(300, 0.42, "300ms 预算线", ha="right", va="bottom", fontsize=FS_LABEL, color=C_RED,
            bbox=dict(fc="white", ec=C_RED, lw=0.7, alpha=0.9, boxstyle="round,pad=0.25"))
    ax.annotate(f"合计 ≈{total}ms（预算内余量 {300 - total}ms）", xy=(total, 0.23),
                xytext=(total - 60, 0.62), fontsize=FS_LABEL + 2, color=C_MAIN,
                arrowprops=dict(arrowstyle="->", color="k", lw=1.2),
                bbox=dict(fc="white", ec="0.7", alpha=0.9, boxstyle="round,pad=0.3"))
    ax.text(0.02, 0.95, "Options：短窗(16ms)/流式WPE/轻量DNN可再省约20~40ms", transform=ax.transAxes,
            fontsize=FS_SMALL, color="dimgray", va="top", ha="left",
            bbox=dict(fc="white", ec="0.7", alpha=0.9, boxstyle="round,pad=0.3"))
    ax.set_yticks([0]); ax.set_yticklabels(["延迟预算"])
    ax.tick_params(axis="y", labelsize=FS_LABEL)
    ax.tick_params(axis="x", labelsize=FS_TINY)
    ax.set_xlabel("延迟 (ms)", fontsize=FS_LABEL)
    ax.set_xlim(0, 345)
    ax.legend(fontsize=FS_SMALL, loc="center left", bbox_to_anchor=(1.01, 0.5), framealpha=0.95)
    ax.set_title("§10.1 延迟预算瀑布条：解析链(60~110ms)+DNN大头，总量对标300ms线", fontsize=FS_TITLE)
    fig.suptitle("图25  远场链路延迟预算瀑布条（各段为量级示意）", fontsize=FS_SUP)
    fig.tight_layout()
    save(fig, "fig25_latency_budget.png")


# ----------------------------------------------------------------------
# 图33 GCC 两种等价算法：时域滑动乘加 vs 频域共轭乘+IFFT
# ----------------------------------------------------------------------
def fig_gcc_two_ways():
    """左=时域滑动乘加（直观但慢），右=频域共轭乘+IFFT（快）；
    左右 R 曲线画的是同一数组（同形状、同归一）。
    参数同图12：d=4cm、θ=30°（自正横起算）→ τ≈58.3μs（模拟）。"""
    fs = 16000
    c = 343.0
    d = 0.04
    tau_true = d * np.sin(np.deg2rad(30.0)) / c          # ≈ 58.3 μs
    n = 4096
    r = np.random.default_rng(11)
    sig = r.standard_normal(n)
    S0 = np.fft.rfft(sig)
    fr = np.fft.rfftfreq(n, 1 / fs)
    S0[(fr > 6000) | (fr < 300)] = 0
    sig = np.fft.irfft(S0)
    sig /= np.max(np.abs(sig))
    F0 = np.fft.rfft(sig)
    x1 = np.fft.irfft(F0 * np.exp(-2j * np.pi * fr * tau_true))   # 晚到 τ
    x2 = sig                                                     # 先到
    # ---- 频域 GCC-PHAT（与图12同做法：16×上采样内插） ----
    up = 16

    def fft_upsample(x, up):
        X = np.fft.rfft(x)
        Xe = np.zeros(len(x) * up // 2 + 1, dtype=complex)
        Xe[:len(X)] = X
        return np.fft.irfft(Xe, len(x) * up)

    x1u, x2u = fft_upsample(x1, up), fft_upsample(x2, up)
    n_u = len(x1u)
    X1, X2 = np.fft.rfft(x1u), np.fft.rfft(x2u)
    S = X1 * np.conj(X2)
    gcc = np.fft.fftshift(np.fft.irfft(S / (np.abs(S) + 1e-10)))
    gcc = gcc / np.max(gcc)                                      # 归一化
    lags_us = np.arange(-n_u // 2, n_u // 2) / (fs * up) * 1e6
    win = np.abs(lags_us) <= 500
    R_lags, R_vals = lags_us[win], gcc[win]                      # 左右共用同一数组
    pk = int(np.argmax(gcc))
    tau_hat = lags_us[pk]
    # ---- 布局：左（波形+τ滑块 / R柱+同形曲线）、中缝（大等号+复杂度）、右（三频谱 / R曲线） ----
    fig = plt.figure(figsize=(15, 5.4))
    outer = fig.add_gridspec(2, 3, width_ratios=[1.15, 0.30, 1.55],
                             height_ratios=[1.05, 1], hspace=0.6, wspace=0.45)
    axL1 = fig.add_subplot(outer[0, 0])
    axL2 = fig.add_subplot(outer[1, 0])
    axM = fig.add_subplot(outer[:, 1])
    inner = outer[0, 2].subgridspec(1, 3, wspace=0.35)
    axF = [fig.add_subplot(inner[0, i]) for i in range(3)]
    axR = fig.add_subplot(outer[1, 2])
    # ---- 左上：两段 8 点波形 + τ 滑块 ----
    i0 = 500
    seg = np.arange(8)
    w_early, w_late = x2[i0:i0 + 8], x1[i0:i0 + 8]
    axL1.plot(seg, w_early, "o-", color=C_BLUE, ms=3.5, lw=1.2, label="x2 先到")
    axL1.plot(seg, w_late, "s-", color=C_RED, ms=4, lw=1.4, label="x1 晚到 τ")
    lo, hi = axL1.get_ylim()
    axL1.set_ylim(lo - 0.38 * (hi - lo), hi)
    ya = lo - 0.20 * (hi - lo)
    axL1.annotate("", xy=(3 + tau_true * fs, ya), xytext=(3, ya),
                  arrowprops=dict(arrowstyle="<->", color="k", lw=1.8))
    axL1.text(3 + tau_true * fs / 2, ya - 0.06 * (hi - lo), "τ滑块", ha="center",
              va="top", fontsize=FS_SMALL)
    axL1.set_xlim(-0.5, 7.5)
    axL1.set_title("(a) 时域：滑动乘加", fontsize=FS_TITLE)
    axL1.set_xlabel("采样点 n（8 点片段）", fontsize=FS_LABEL)
    axL1.legend(fontsize=FS_SMALL, loc="upper right")
    axL1.grid(ls=":", alpha=0.5)
    # ---- 左下：3 柱 R 值 + 同归一 R 曲线 ----
    probe = np.array([-1e6 / fs, 0.0, 1e6 / fs])                 # -62.5 / 0 / +62.5 μs
    idx = [int(np.argmin(np.abs(R_lags - v))) for v in probe]
    bars = axL2.bar(probe, R_vals[idx], width=38, color=[C_BLUE, C_BLUE, C_RED],
                    edgecolor="k", lw=0.8, label="3 个滞后 R 值（-1/0/+1采样点）")
    for b, v in zip(bars, R_vals[idx]):
        if v >= 0.9:
            axL2.text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:.2f}",
                      ha="center", va="bottom", fontsize=FS_TINY)
        elif v >= 0:
            axL2.text(b.get_x() - 4, v + 0.06, f"{v:.2f}",
                      ha="right", va="bottom", fontsize=FS_TINY)
        else:
            axL2.text(b.get_x() - 4, v - 0.06, f"{v:.2f}",
                      ha="right", va="top", fontsize=FS_TINY)
    axL2.plot(R_lags, R_vals, color=C_BLUE, lw=1.2, alpha=0.85, label="R(τ) 曲线（与右格同归一）")
    axL2.axvline(tau_true * 1e6, color="k", ls="--", lw=1, alpha=0.6)
    axL2.set_xlim(-500, 500)
    axL2.set_ylim(-0.62, 1.28)
    axL2.set_title("R(τ)：逐滞后乘加求和", fontsize=FS_TITLE)
    axL2.set_xlabel("时延 τ (μs)", fontsize=FS_LABEL)
    axL2.set_ylabel("GCC-PHAT值", fontsize=FS_LABEL)
    axL2.text(0.03, 0.88, "O(N²)：慢但直观", transform=axL2.transAxes,
              fontsize=FS_SMALL, color=C_RED, va="top",
              bbox=dict(fc="white", ec=C_RED, lw=0.7, alpha=0.9, boxstyle="round,pad=0.3"))
    axL2.legend(fontsize=FS_TINY, loc="lower left")
    axL2.grid(ls=":", alpha=0.5)
    # ---- 中缝：大等号 + 复杂度两行 ----
    axM.axis("off")
    axM.set_xlim(0, 1)
    axM.set_ylim(0, 1)
    axM.text(0.5, 0.58, "=", fontsize=24, ha="center", va="center", color="k")
    axM.text(0.5, 0.38, "时域 O(N·L)\n滑动乘加（口算）", ha="center", va="top",
              fontsize=FS_SMALL, color=C_BLUE,
              bbox=dict(fc="#dbe9f6", ec=C_BLUE, lw=0.8, alpha=0.9, boxstyle="round,pad=0.3"))
    axM.text(0.5, 0.13, "频域 O(N logN)\n共轭乘+IFFT（计算器）", ha="center", va="top",
              fontsize=FS_SMALL, color=C_RED,
              bbox=dict(fc="#f6e5db", ec=C_RED, lw=0.8, alpha=0.9, boxstyle="round,pad=0.3"))
    # ---- 右上：三小频谱（|X1|、|X2|、X1·conj(X2) 相位直线） ----
    X1f = np.fft.rfft(x1)
    X2f = np.fft.rfft(x2)
    ff = np.fft.rfftfreq(n, 1 / fs) / 1000.0                      # kHz
    cross = X1f * np.conj(X2f)
    band = (ff >= 0.3) & (ff <= 6.0)
    phase = np.unwrap(np.angle(cross))
    axF[0].plot(ff, np.abs(X1f), color=C_BLUE, lw=1.1)
    axF[0].set_title("|X₁| 频谱", fontsize=FS_SMALL)
    axF[1].plot(ff, np.abs(X2f), color=C_RED, lw=1.1)
    axF[1].set_title("|X₂| 频谱", fontsize=FS_SMALL)
    axF[2].plot(ff[band], phase[band], color=C_PURPLE, lw=1.4)
    axF[2].set_title("X₁·conj(X₂) 相位", fontsize=FS_SMALL)
    axF[2].set_ylim(phase[band].min() - 0.25, phase[band].max() + 0.12)
    for k, va, yy in ((1.0, "top", phase[band].max() + 0.06),
                      (2.0, "bottom", phase[band].min() - 0.19)):
        axF[2].axvline(k, color="gray", ls="--", lw=1.0, alpha=0.8)
        axF[2].text(k + 0.08, yy, f"{int(k)}kHz", ha="left", va=va,
                    fontsize=FS_TINY, color="gray",
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.5))
    for a in axF:
        a.set_xlim(0, 8)
        a.set_xlabel("频率 (kHz)", fontsize=FS_TINY)
        a.tick_params(labelsize=FS_TINY)
        a.grid(ls=":", alpha=0.5)
    axF[0].set_ylabel("幅度", fontsize=FS_TINY)
    axF[2].set_ylabel("相位 (rad)", fontsize=FS_TINY)
    # ---- 右下：R 曲线（与左格同一数组）+ 红三角峰 ----
    axR.plot(R_lags, R_vals, color=C_BLUE, lw=1.4, label="R(τ) 曲线（与左格同归一）")
    axR.plot(tau_hat, 1.0, "r^", ms=10, label=f"峰={tau_hat:.1f}μs")
    axR.axvline(tau_true * 1e6, color="k", ls="--", lw=1, alpha=0.6)
    axR.annotate(f"峰 ≈ {tau_hat:.1f} μs（真值 {tau_true * 1e6:.1f} μs）",
                 xy=(tau_hat, 1.0), xytext=(tau_hat + 130, 0.62),
                 fontsize=FS_LABEL, arrowprops=dict(arrowstyle="->", color="k", lw=1.5),
                 bbox=dict(fc="white", alpha=0.85, pad=1, ec="none"))
    axR.set_xlim(-500, 500)
    axR.set_title("(b) 频域：共轭乘 + IFFT", fontsize=FS_TITLE)
    axR.set_xlabel("时延 τ (μs)", fontsize=FS_LABEL)
    axR.set_ylabel("GCC-PHAT值", fontsize=FS_LABEL)
    axR.legend(fontsize=FS_SMALL, loc="lower left")
    axR.grid(ls=":", alpha=0.5)
    fig.suptitle("图33  GCC 两种等价算法：时域滑动乘加 vs 频域共轭乘+IFFT（d=4cm, θ=30°, τ≈58.3μs；模拟）",
                 fontsize=FS_SUP)
    fig.tight_layout()
    save(fig, "fig33_gcc_two_ways.png")


if __name__ == "__main__":
    fig_geometries()
    fig_near_far_field()
    fig_beampatterns()
    fig_doa_spectrum()
    fig_gcc_phat()
    fig_gsc()
    fig_srp_grid()
    fig_tracking()
    fig_wng_di()
    fig_pipeline()
    fig_tradeoffs()
    fig_room_acoustics()
    fig_stft_cov()
    fig_sparse_array()
    fig_aec()
    fig_aec_pipeline()
    fig_aec_landscape()
    fig_wpe()
    fig_binaural()
    fig_gcc_reverb()
    fig_delay_phase()
    fig_beampattern_anatomy()
    fig_dsin_geometry()
    fig_wpe_frames()
    fig_latency_budget()
    fig_gcc_two_ways()
    print("ALL DONE")
