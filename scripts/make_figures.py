# -*- coding: utf-8 -*-
"""生成教程插图 26 张（图 1~25、图 33；图 26~32 见 make_aec_figures.py）。

用法（仓库根目录）：
    .venv/bin/python scripts/make_figures.py      # 图 1~25、图 33 → figures/
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

# 绘图样式必须只由仓库代码决定，不能随外部运行时是否存在而改变。
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC",
                        "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

OUT = Path(__file__).parent.parent / "figures"
OUT.mkdir(exist_ok=True)
C_MAIN, C_BLUE, C_RED, C_GREEN, C_ORANGE, C_PURPLE = "#1a1a2e", "#2f6db3", "#c0392b", "#2e8b57", "#e67e22", "#7d3c98"
# 全局字号约定（统一全书插图）
FS_SUP = 13      # 图题 suptitle
FS_TITLE = 11    # 子图标题
FS_LABEL = 10    # 轴标签 / 主要注释
FS_SMALL = 9     # 图例 / 次要注释
FS_TINY = 8      # 刻度 / 极小标注

# 每张含随机数据的图使用独立种子。调用顺序变化不会改变其他图片。
FIGURE_SEEDS = {
    "doa_spectrum": 11001,
    "gcc_phat": 12001,
    "tracking": 22001,
    "room_acoustics": 5001,
    "stft_cov": 6001,
    "aec": 18001,
    "wpe": 21001,
}


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


def causal_delay(x, delay):
    """将一维信号延迟整数个采样；开头补零，不做循环移位。"""
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("x 必须是一维信号")
    if not isinstance(delay, (int, np.integer)) or delay < 0:
        raise ValueError("delay 必须是非负整数")
    delayed = np.zeros_like(x)
    if delay == 0:
        delayed[:] = x
    elif delay < len(x):
        delayed[delay:] = x[:-delay]
    return delayed


def wilson_interval(successes, trials, z=1.96):
    """二项比例的 Wilson 区间；返回 0 到 1 之间的下、上界。"""
    if not isinstance(successes, (int, np.integer)) or not isinstance(trials, (int, np.integer)):
        raise ValueError("successes 和 trials 必须为整数")
    if trials <= 0 or successes < 0 or successes > trials or z <= 0:
        raise ValueError("需要 0 <= successes <= trials 且 trials、z 为正")
    p = successes / trials
    denominator = 1 + z ** 2 / trials
    center = (p + z ** 2 / (2 * trials)) / denominator
    half = z * np.sqrt(p * (1 - p) / trials + z ** 2 / (4 * trials ** 2)) / denominator
    return center - half, center + half


def fibonacci_sphere(count):
    """在单位球面上返回确定性的 Fibonacci 近均匀布点。"""
    if not isinstance(count, (int, np.integer)) or count <= 0:
        raise ValueError("count 必须为正整数")
    index = np.arange(count, dtype=float)
    z = 1.0 - 2.0 * (index + 0.5) / count
    azimuth = np.pi * (3.0 - np.sqrt(5.0)) * index
    radius = np.sqrt(np.maximum(0.0, 1.0 - z ** 2))
    return np.column_stack((radius * np.cos(azimuth),
                            radius * np.sin(azimuth), z))


def random_decay_rir(t60, fs, rng, duration_factor=1.1):
    """生成单位直达项加随机指数尾的说明性冲激响应。

    振幅包络在 t=t60 时下降 60 dB；尾长默认取 1.1*t60，
    因此截断前已低于 -60 dB，而不是用固定时长截断长混响尾。
    """
    if t60 <= 0 or fs <= 0 or duration_factor <= 1.0:
        raise ValueError("t60、fs 必须为正，duration_factor 必须大于 1")
    length = int(np.ceil(duration_factor * t60 * fs)) + 1
    time = np.arange(length) / fs
    tail_scale = np.sqrt(2.9 * 13.82 / fs)
    rir = rng.standard_normal(length) * np.exp(-6.91 * time / t60) * tail_scale
    rir[0] = 1.0
    return rir


def fft_convolve_prefix(signal, impulse_response, output_length):
    """用零填充 FFT 计算线性卷积，并返回前 output_length 点。"""
    signal = np.asarray(signal)
    impulse_response = np.asarray(impulse_response)
    if signal.ndim != 1 or impulse_response.ndim != 1:
        raise ValueError("signal 和 impulse_response 必须是一维")
    if not isinstance(output_length, (int, np.integer)) or output_length < 0:
        raise ValueError("output_length 必须为非负整数")
    full_length = signal.size + impulse_response.size - 1
    n_fft = 1 << max(0, (full_length - 1).bit_length())
    result = np.fft.irfft(
        np.fft.rfft(signal, n_fft) * np.fft.rfft(impulse_response, n_fft),
        n_fft)[:full_length]
    return result[:output_length]


def gcc_peak_is_correct(peak_lag, true_delay_samples, tolerance_samples=1):
    """X1*conj(X2) 约定下，第二路延迟 true_delay 时真峰位于负 lag。"""
    return abs(peak_lag + true_delay_samples) <= tolerance_samples


def gcc_peak_contrast(correlation):
    """图13所用峰值对比度：max(g) / median(abs(g))。"""
    correlation = np.asarray(correlation)
    return float(np.max(correlation) /
                 (np.median(np.abs(correlation)) + 1e-12))


def distortionless_weights(covariance, steering, diagonal_loading=0.0):
    """求带对角加载的无失真最小方差权重，并返回绝对加载量。"""
    covariance = np.asarray(covariance, dtype=complex)
    steering = np.asarray(steering, dtype=complex)
    if covariance.shape != (steering.size, steering.size):
        raise ValueError("协方差矩阵尺寸必须与导向矢量一致")
    if diagonal_loading < 0:
        raise ValueError("diagonal_loading 不能为负")
    loading = diagonal_loading * np.trace(covariance).real / steering.size
    loaded = covariance + loading * np.eye(steering.size)
    whitened = np.linalg.solve(loaded, steering)
    weights = whitened / (steering.conj() @ whitened)
    return weights, float(loading)


def gcc_phat_interpolated(x1, x2, fs, interp=16, max_tau=None):
    """用零填充互谱得到插值 GCC-PHAT。

    返回时延轴（秒）、相关序列和最大峰时延。FFT 长度至少覆盖线性相关，
    因而不会把两路信号末尾绕回开头。interp 只细化时延网格，不增加带宽。
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if x1.ndim != 1 or x2.ndim != 1 or x1.size == 0 or x2.size == 0:
        raise ValueError("x1 和 x2 必须是非空一维信号")
    if fs <= 0 or not isinstance(interp, (int, np.integer)) or interp < 1:
        raise ValueError("fs 必须为正数，interp 必须为正整数")
    n_linear = x1.size + x2.size - 1
    n_fft = 1 << int(np.ceil(np.log2(n_linear)))
    cross = np.fft.rfft(x1, n_fft) * np.conj(np.fft.rfft(x2, n_fft))
    phat = cross / np.maximum(np.abs(cross), np.finfo(float).eps)
    n_interp = n_fft * interp
    correlation = np.fft.fftshift(np.fft.irfft(phat, n=n_interp))
    lags = (np.arange(n_interp) - n_interp // 2) / (fs * interp)
    if max_tau is not None:
        keep = np.abs(lags) <= max_tau
        lags = lags[keep]
        correlation = correlation[keep]
    peak_tau = float(lags[np.argmax(correlation)])
    return lags, correlation, peak_tau


# ----------------------------------------------------------------------
# 图8 阵列几何形态大全
# ----------------------------------------------------------------------
def fig_geometries():
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    titles = ["(a) 双麦 Endfire", "(b) 双麦 Broadside", "(c) 四麦均匀线阵 ULA",
              "(d) 四麦方阵/平面阵", "(e) 六麦圆阵 UCA + 中心麦",
              "(f) Fibonacci近均匀球阵(32麦)", "(g) 螺旋阵/对数阵", "(h) 分布式/非规则阵"]
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
    ax.annotate("假定目标主轴", xy=(1.2, 6.2), fontsize=9, color=C_ORANGE)
    ax.set_xlim(-4, 6); ax.set_ylim(-1, 7)
    # (b) broadside
    ax = axes[0, 1]
    ax.scatter([-2, 2], [0, 0], s=180, c=C_BLUE, zorder=5, edgecolors="k")
    for a in np.linspace(np.pi / 2 - 0.55, np.pi / 2 + 0.55, 7):
        ax.plot([0, 6 * np.cos(a)], [0, 6 * np.sin(a)], color=C_ORANGE, alpha=0.5, lw=1)
    ax.annotate("假定目标主轴", xy=(1.5, 5.5), fontsize=9, color=C_ORANGE)
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
    ax.set_title("(f) Fibonacci近均匀球阵(32麦)", fontsize=FS_TITLE)
    u = fibonacci_sphere(32)
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
    fig.text(0.5, 0.01, "平面子图的横纵轴为无量纲示意坐标；实际阵元间距需按目标频段和设备尺寸设计。",
             ha="center", fontsize=FS_SMALL, color="0.3")
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
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
    ax.set_xlabel("水平示意坐标（无量纲）", fontsize=FS_SMALL)
    ax.set_ylabel("竖直示意坐标（无量纲）", fontsize=FS_SMALL)
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
    ax.text(-3.0, 1.15, "相位曲率参考尺度：r > 2D²/λ（D=孔径）\n"
            r"还需检查 $r \gg D$ 及阵元间幅度差", fontsize=FS_SMALL, color=C_RED,
            ha="left", va="top", bbox=dict(fc="white", ec="none", alpha=0.82, pad=1.5))
    ax.set_xlabel("水平示意坐标（无量纲）", fontsize=FS_SMALL)
    ax.set_ylabel("竖直示意坐标（无量纲）", fontsize=FS_SMALL)
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
    w_mvdr, _ = distortionless_weights(R, a0)
    # 超指向: 小间距 d_sd=0.2λ, 各向同性(弥散)噪声协方差 Γ=sinc(2d|i-j|)
    d_sd = 0.2
    a0_sd = np.exp(-2j * np.pi * d_sd * m * np.sin(0))
    G = np.sinc(2 * d_sd * np.abs(np.subtract.outer(m, m)))
    w_sd, _ = distortionless_weights(G, a0_sd, diagonal_loading=1e-6)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), subplot_kw=dict(polar=True))
    fig.subplots_adjust(wspace=0.12)
    curves = [("DSB 延迟求和 (d=λ/2)", w_ds, d, C_BLUE, "-"),
              ("MVDR 干扰零陷20° (d=λ/2)", w_mvdr, d, C_RED, "--"),
              (r"超指向 (d=0.2λ, 加载 $10^{-6}$)", w_sd, d_sd, C_GREEN, ":")]
    for ax, (title, ymax) in zip(axes, [("线性幅度", None), ("dB 刻度", 40)]):
        for name, w, dd, c, ls in curves:
            A = np.exp(-2j * np.pi * dd * np.outer(m, np.sin(thetas_rad)))
            b = np.abs(w.conj() @ A)
            target = np.exp(-2j * np.pi * dd * m * np.sin(0))
            target_gain = np.abs(w.conj() @ target)
            if "dB" in title:
                b = 20 * np.log10(np.maximum(b / target_gain, 1e-4))
                b = np.maximum(b, -40) + 40      # 平移到 0~40 便于径向刻度
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
    fig.suptitle("图16  8元ULA 0°指向波束图：DSB/MVDR 用 d=λ/2，超指向用 d=0.2λ\n"
                 r"（各曲线保持目标方向单位响应；MVDR 干扰=20°；超指向相对对角加载=$10^{-6}$；模拟）",
                 fontsize=12.3)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, "fig16_beampattern.png")


# ----------------------------------------------------------------------
# 图11 DOA空间谱对比: Bartlett vs Capon vs MUSIC
# ----------------------------------------------------------------------
def bartlett_spectrum(covariance, steering_vectors):
    """计算 Bartlett 空间谱 a^H R a。

    covariance 为 M×M 复协方差矩阵，steering_vectors 为 M×G，
    返回 G 个实值。不在此函数内归一化，便于单元测试核对二次型。
    """
    covariance = np.asarray(covariance)
    steering_vectors = np.asarray(steering_vectors)
    return np.einsum(
        "mg,mn,ng->g", steering_vectors.conj(), covariance,
        steering_vectors).real


def fig_doa_spectrum():
    rng = np.random.default_rng(FIGURE_SEEDS["doa_spectrum"])
    M, d = 8, 0.5
    srcs = [-20, 30]
    snr_db, snaps = 20, 400
    m = np.arange(M) - (M - 1) / 2
    A = np.exp(-2j * np.pi * d * np.outer(m, np.sin(np.deg2rad(srcs))))
    # 每个源为单位复功率；snr_db 定义为“每阵元总信号功率 / 噪声功率”。
    sig = (rng.normal(size=(2, snaps)) + 1j * rng.normal(size=(2, snaps))) / np.sqrt(2)
    signal_power_per_sensor = len(srcs)
    noise_power = signal_power_per_sensor / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        rng.normal(size=(M, snaps)) + 1j * rng.normal(size=(M, snaps)))
    X = A @ sig + noise
    R = X @ X.conj().T / snaps
    th = np.linspace(-90, 90, 1801)
    tr = np.deg2rad(th)
    Av = np.exp(-2j * np.pi * d * np.outer(m, np.sin(tr)))  # M x grid
    bart = bartlett_spectrum(R, Av)
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
    axes[1].text(0.5, 0.12, "峰高不等于源功率；本图只比较峰位与谱形", transform=axes[1].transAxes,
                 fontsize=FS_SMALL, color=C_RED, ha="center",
                 bbox=dict(fc="white", ec=C_RED, lw=0.7, alpha=0.9, boxstyle="round,pad=0.3"))
    axes[1].legend(fontsize=FS_SMALL); axes[1].set_title("(b) MUSIC 特征结构类伪谱", fontsize=12); axes[1].grid(ls=":", alpha=0.5)
    fig.suptitle("图11  两信号源(-20°, 30°) DOA空间谱估计对比（8元ULA, 每阵元总输入SNR=20dB, 400快拍, 模拟）", fontsize=12.5)
    fig.tight_layout()
    save(fig, "fig11_doa_spectrum.png")


# ----------------------------------------------------------------------
# 图12 GCC-PHAT 原理
# ----------------------------------------------------------------------
def fig_gcc_phat():
    """重新设计：d=4 cm、θ=30°（自正横起算）→ τ=d·sinθ/c≈58.3 μs≈0.93 采样点@16kHz，
    正好演示 GCC-PHAT 的亚采样（分数倍采样点）时延估计能力。"""
    rng = np.random.default_rng(FIGURE_SEEDS["gcc_phat"])
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
    # (b) 对互谱做零填充，将 GCC-PHAT 的时延网格细化 16 倍。
    ax = axes[1]
    up = 16
    lags, gcc, tau_est = gcc_phat_interpolated(
        x1, x2, fs, interp=up, max_tau=500e-6)
    lags_us = lags * 1e6
    pk = np.argmax(gcc)
    ax.plot(lags_us, gcc, color=C_BLUE, lw=1.4)
    ax.plot(tau_est * 1e6, gcc[pk], "r^", ms=10)
    ax.axvline(tau_true * 1e6, color="k", ls="--", lw=1, alpha=0.6)
    for s in np.arange(-3, 4) * 1e6 / fs:
        if abs(s) <= 500:
            ax.axvline(s, color="gray", ls=":", lw=1.4, alpha=0.9)
    ax.annotate(f"内插峰 = {tau_est*1e6:.1f} μs", xy=(tau_est * 1e6, gcc[pk]),
                xytext=(170, 0.72 * gcc[pk]), fontsize=FS_LABEL,
                arrowprops=dict(arrowstyle="->", color="k", lw=1.8))
    ax.text(0.03, 0.97, f"真值 {tau_true*1e6:.1f} μs；估计 {tau_est*1e6:.1f} μs\n"
                         f"估计位置 = {tau_est*fs:.3f} 个采样点",
            transform=ax.transAxes, fontsize=FS_SMALL, color=C_RED, va="top")
    ax.text(0.03, 0.76, "（16×时延网格插值；\n灰点线=整数采样间隔）",
            transform=ax.transAxes, fontsize=FS_TINY + 0.5, color=C_RED, va="top")
    ax.set_xlim(-500, 500)
    ax.set_title("(b) GCC-PHAT（16×时延网格插值）", fontsize=FS_TITLE)
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
    ax.text(0.3, 5.68, "上支路（设计目标：期望方向单位响应）", fontsize=FS_LABEL, color=C_ORANGE, ha="left")
    ax.text(2.7, 2.68, "下支路（自适应噪声估计）", fontsize=FS_LABEL, color=C_BLUE, ha="left")
    ax.text(5.75, 0.55, "模型匹配且滤波器收敛时：保持目标响应，对消相关干扰",
            ha="center", fontsize=FS_LABEL, color=C_RED)
    ax.set_title("图17  GSC（广义旁瓣对消器）结构框图", fontsize=FS_SUP)
    save(fig, "fig17_gsc.png")


# ----------------------------------------------------------------------
# 图14 SRP-PHAT 网格搜索
# ----------------------------------------------------------------------
def srp_tdoa_score(grid_x, grid_y, mics, source, sound_speed=343.0,
                   gcc_peak_width_s=1.0 / 6000.0):
    """用理想化 GCC 峰查表计算 SRP 分数。

    每个麦对的观测 TDOA 来自 source；候选网格按同一几何计算预测 TDOA，
    再在以观测 TDOA 为中心的高斯 GCC 峰上取值并跨麦对求和。
    """
    mics = np.asarray(mics, dtype=float)
    source = np.asarray(source, dtype=float)
    score = np.zeros_like(grid_x, dtype=float)
    pair_count = 0
    true_dist = np.linalg.norm(mics - source[None, :], axis=1)
    for i in range(len(mics)):
        for j in range(i + 1, len(mics)):
            observed_tdoa = (true_dist[i] - true_dist[j]) / sound_speed
            di = np.hypot(grid_x - mics[i, 0], grid_y - mics[i, 1])
            dj = np.hypot(grid_x - mics[j, 0], grid_y - mics[j, 1])
            predicted_tdoa = (di - dj) / sound_speed
            mismatch = (predicted_tdoa - observed_tdoa) / gcc_peak_width_s
            score += np.exp(-0.5 * mismatch ** 2)
            pair_count += 1
    return score / pair_count


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
    ax.set_xlabel("x (m)", fontsize=FS_LABEL); ax.set_ylabel("y (m)", fontsize=FS_LABEL)
    ax = axes[1]
    gx = np.linspace(-0.5, 7.0, 90); gy = np.linspace(-0.5, 5.2, 68)
    GX, GY = np.meshgrid(gx, gy)
    SRP = srp_tdoa_score(GX, GY, mics, src)
    im = ax.pcolormesh(GX, GY, SRP, cmap="viridis", shading="auto")
    ax.scatter(mics[:, 0], mics[:, 1], s=120, c=C_BLUE, edgecolors="k", zorder=6)
    imax = np.unravel_index(np.argmax(SRP), SRP.shape)
    ax.plot(GX[imax], GY[imax], "r*", ms=12, mec="k")
    ax.set_title("(b) SRP-PHAT 麦对累积分数，峰值即位置估计", fontsize=FS_TITLE)
    ax.set_xlim(-0.5, 7); ax.set_ylim(-0.5, 5.2); ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=FS_LABEL); ax.set_ylabel("y (m)", fontsize=FS_LABEL)
    fig.colorbar(im, ax=ax, shrink=0.8, label="归一化麦对 GCC 累积分数")
    fig.suptitle("图14  SRP-PHAT 定位过程（独立米级分布式阵列例）\n"
                 "4 m × 3 m 四麦；源(5.5, 3.8) m；c=343 m/s；高斯峰σ=166.7 μs；90×68网格",
                 fontsize=FS_SUP)
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.11, top=0.80, wspace=0.28)
    save(fig, "fig14_srp_grid.png")


# ----------------------------------------------------------------------
# 图22 声源追踪：粒子滤波 vs 直方图
# 口径说明（F12 定稿，供正文与重出核对）：
#   (a) SIR-PF：N=200 粒子；观测 = 真值 + 高斯噪声（σ=7°），另约 20% 帧替换为
#       [0,120]° 均匀野点（与第 9 章正文一致）。(b) 为独立合成的 PHD 强度示意。
#   两子图独立仿真、角度约定不同（(a) 0~120°扇区，(b) −90°~+90°，见子图标题）。
# ----------------------------------------------------------------------
def systematic_resample(weights, rng):
    """系统重采样，返回与 weights 等长的祖先索引。"""
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    positions = (rng.random() + np.arange(len(weights))) / len(weights)
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0  # 避免浮点舍入使 searchsorted 返回越界索引。
    return np.searchsorted(cumulative, positions, side="right")


def particle_filter_doa(observations, rng, n_particles=200, process_std=1.0,
                        observation_std=7.0, resample_fraction=0.5,
                        angle_bounds=(0.0, 120.0), velocity_process_std=0.25,
                        clutter_probability=0.1):
    """角度—角速度粒子滤波，支持缺测和“高斯目标+均匀杂波”似然。

    NaN 表示本帧缺少观测，此时只执行运动模型预测。clutter_probability
    给均匀杂波似然的混合权重；设为 0 可退化为普通高斯似然。
    """
    observations = np.asarray(observations, dtype=float)
    if observations.ndim != 1 or observations.size == 0:
        raise ValueError("observations 必须是非空一维序列")
    if not 0 <= clutter_probability < 1:
        raise ValueError("clutter_probability 必须在 [0, 1) 内")
    finite = np.flatnonzero(np.isfinite(observations))
    initial_angle = (observations[finite[0]] if finite.size else
                     0.5 * (angle_bounds[0] + angle_bounds[1]))
    angles = np.clip(rng.normal(initial_angle, 10.0, n_particles), *angle_bounds)
    velocities = rng.normal(0.0, 1.0, n_particles)
    weights = np.full(n_particles, 1.0 / n_particles)
    estimates = np.empty(len(observations))
    neff_history = np.empty(len(observations))
    resampled = np.zeros(len(observations), dtype=bool)
    for k, observation in enumerate(observations):
        velocities += rng.normal(0.0, velocity_process_std, n_particles)
        proposed = angles + velocities + rng.normal(0.0, process_std, n_particles)
        hit_bound = (proposed < angle_bounds[0]) | (proposed > angle_bounds[1])
        angles = np.clip(proposed, *angle_bounds)
        velocities[hit_bound] = 0.0
        if np.isfinite(observation):
            normalizer = observation_std * np.sqrt(2 * np.pi)
            gaussian = np.exp(-0.5 * ((angles - observation) / observation_std) ** 2) / normalizer
            uniform = 1.0 / (angle_bounds[1] - angle_bounds[0])
            likelihood = ((1.0 - clutter_probability) * gaussian
                          + clutter_probability * uniform)
            weights *= likelihood
            weights /= weights.sum()
        estimates[k] = np.sum(weights * angles)
        neff_history[k] = 1.0 / np.sum(weights ** 2)
        if (np.isfinite(observation)
                and neff_history[k] < resample_fraction * n_particles):
            indices = systematic_resample(weights, rng)
            angles = angles[indices]
            velocities = velocities[indices]
            weights.fill(1.0 / n_particles)
            resampled[k] = True
    return estimates, neff_history, resampled


def normalized_phd_intensity(grid, centers, stds, weights=None, background=None):
    """生成一维 PHD 示意强度，并令曲线积分等于期望目标数。"""
    grid = np.asarray(grid, dtype=float)
    centers = np.atleast_1d(np.asarray(centers, dtype=float))
    stds = np.atleast_1d(np.asarray(stds, dtype=float))
    if centers.shape != stds.shape or np.any(stds <= 0):
        raise ValueError("centers 与 stds 必须同形，且标准差为正")
    if weights is None:
        weights = np.ones_like(centers)
    weights = np.atleast_1d(np.asarray(weights, dtype=float))
    if weights.shape != centers.shape or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("weights 必须同形、非负且总和为正")
    intensity = np.zeros_like(grid)
    for center, std, weight in zip(centers, stds, weights):
        intensity += weight * np.exp(-0.5 * ((grid - center) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    if background is not None:
        background = np.asarray(background, dtype=float)
        if background.shape != grid.shape or np.any(background < 0):
            raise ValueError("background 必须与 grid 同形且非负")
        intensity += background
    integral = np.trapezoid(intensity, grid)
    return intensity * (weights.sum() / integral)


def fig_tracking():
    rng = np.random.default_rng(FIGURE_SEEDS["tracking"])
    T = 120
    t = np.arange(T)
    true = 60 + 40 * np.sin(2 * np.pi * t / T)
    obs = true + rng.normal(0, 7, T)
    is_out = rng.random(T) < 0.2
    obs = np.where(is_out, rng.uniform(0, 120, T), obs)
    missing = np.zeros(T, dtype=bool)
    missing[48:58] = True
    obs[missing] = np.nan
    N = 200
    est, n_eff, resampled = particle_filter_doa(obs, rng, n_particles=N)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].plot(t, obs, ".", color="gray", ms=4, label="DOA观测（含噪声和均匀杂波）")
    axes[0].plot(t, true, color="k", lw=2, label="真实轨迹")
    axes[0].plot(t, est, color=C_RED, lw=1.6, label="混合似然粒子滤波估计")
    axes[0].plot(t[resampled], est[resampled], "|", color=C_ORANGE, ms=7,
                 label=r"重采样 ($N_\mathrm{eff}<N/2$)")
    axes[0].axvspan(48, 57, color=C_PURPLE, alpha=0.12, label="连续 10 帧缺测")
    axes[0].set_xlabel("帧"); axes[0].set_ylabel("方位角 (°)")
    axes[0].legend(fontsize=FS_SMALL + 1, loc="lower left", framealpha=0.95); axes[0].grid(ls=":", alpha=0.5)
    axes[0].set_title("(a) 单说话人追踪：匀速状态模型在缺测段只做预测\n"
                      "（200 粒子；高斯目标+均匀杂波似然；角度 0–120°）", fontsize=FS_TITLE)
    ax = axes[1]
    th = np.linspace(-90, 90, 400)
    I = normalized_phd_intensity(
        th, centers=[-25, 40], stds=[4, 5], weights=[1, 1],
        background=0.0005 * rng.random(th.size))
    ax.plot(th, I, color=C_BLUE, lw=1.6)
    ax.fill_between(th, I, alpha=0.25, color=C_BLUE)
    for peak, lb in [(-25, "说话人1"), (40, "说话人2")]:
        peak_value = I[np.argmin(np.abs(th - peak))]
        ax.plot(peak, peak_value, "r^", ms=10)
        ax.annotate(lb, xy=(peak, peak_value), xytext=(peak - 16, peak_value + 0.035), fontsize=FS_LABEL, color=C_RED,
                    arrowprops=dict(arrowstyle="->", color="r"))
    ax.set_ylim(0, 1.45 * I.max())
    ax.set_xlabel("方位角 (°)"); ax.set_ylabel("PHD 强度 (目标数/度)")
    ax.set_title("(b) 两目标 PHD 强度示意：曲线积分=2\n（角度约定：−90°~+90°）", fontsize=FS_TITLE)
    ax.grid(ls=":", alpha=0.5)
    fig.suptitle("图22  声源追踪示意（模拟）", fontsize=FS_SUP)
    fig.tight_layout()
    save(fig, "fig22_tracking.png")


# ----------------------------------------------------------------------
# 图15 白噪声增益/指向性 vs 频率
# ----------------------------------------------------------------------
def fig_wng_di():
    freqs = np.linspace(100, 8000, 200)
    sound_speed = 343.0
    M, radius = 6, 0.04  # 六麦圆阵；六边形相邻弦长等于半径 4 cm
    azimuth = 2 * np.pi * np.arange(M) / M
    mic_xy = radius * np.column_stack((np.cos(azimuth), np.sin(azimuth)))
    pair_distances = np.linalg.norm(mic_xy[:, None, :] - mic_xy[None, :, :], axis=2)
    look_direction = np.array([1.0, 0.0])
    wng_dsb, di_dsb, wng_sd, di_sd = [], [], [], []
    for f in freqs:
        a0 = np.exp(-2j * np.pi * f * (mic_xy @ look_direction) / sound_speed)
        # 三维各向同性弥散场：Γ_ij=sinc(2 f d_ij/c)，距离取圆阵二维坐标的欧氏弦长。
        G = np.sinc(2 * f * pair_distances / sound_speed)
        w_ds = a0 / M
        w_sd, _ = distortionless_weights(G, a0, diagonal_loading=1e-6)
        wng_dsb.append(10 * np.log10(1 / np.sum(np.abs(w_ds) ** 2)))
        di_dsb.append(10 * np.log10(1 / np.real(w_ds.conj() @ G @ w_ds)))
        wng_sd.append(10 * np.log10(1 / np.sum(np.abs(w_sd) ** 2)))
        di_sd.append(10 * np.log10(1 / np.real(w_sd.conj() @ G @ w_sd)))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
    axes[0].semilogx(freqs, wng_dsb, color=C_BLUE, lw=2.0, label="DSB 延迟求和（理想无失配理论值）")
    axes[0].semilogx(freqs, wng_sd, color=C_RED, ls="--", lw=2.0,
                     label=r"对角加载超指向（相对加载 $10^{-6}$）")
    axes[0].axhline(0, color="gray", ls="--", lw=1.2, label="0 dB 参考线")
    axes[0].set_xlabel("频率 (Hz)"); axes[0].set_ylabel("白噪声增益 WNG (dB)")
    axes[0].legend(fontsize=FS_SMALL, loc="lower right", framealpha=0.95)
    axes[0].grid(ls=":", alpha=0.5); axes[0].set_title("(a) WNG：超指向低频稳健性差", fontsize=FS_TITLE)
    axes[1].semilogx(freqs, di_dsb, color=C_BLUE, lw=2.0, label="DSB 延迟求和")
    axes[1].semilogx(freqs, di_sd, color=C_RED, ls="--", lw=2.0,
                     label=r"对角加载超指向（相对加载 $10^{-6}$）")
    axes[1].set_xlabel("频率 (Hz)"); axes[1].set_ylabel("指向性指数 DI (dB)")
    axes[1].legend(fontsize=FS_SMALL, loc="center left", bbox_to_anchor=(1.02, 0.5), framealpha=0.95)
    axes[1].grid(ls=":", alpha=0.5); axes[1].set_title("(b) DI：本理想弥散场模型下超指向不低于 DSB", fontsize=FS_TITLE)
    for _ax in axes:
        _ax.minorticks_on()
        _ax.grid(which="minor", ls=":", alpha=0.25)
    fig.suptitle("图15  6元圆阵（半径=相邻弦长=4 cm）超指向 vs 延迟求和：WNG与DI\n"
                 r"（三维各向同性弥散场；超指向相对对角加载=$10^{-6}$；目标方向单位响应；本书仿真）",
                 fontsize=FS_SUP - 1)
    fig.subplots_adjust(left=0.07, right=0.82, bottom=0.12, top=0.76, wspace=0.46)
    save(fig, "fig15_wng_di.png")


# ----------------------------------------------------------------------
# 图23 远场语音前端处理链路
# ----------------------------------------------------------------------
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(16.5, 5.6))
    ax.axis("off"); ax.set_xlim(0, 16.6); ax.set_ylim(0, 4.8)
    # VAD/远端活动/双讲属于控制面，不插进音频主链。
    main_steps = [
        ("多通道采集\n同步/标定", "#dbe9f6", "时钟源"),
        ("AEC\n回声消除", "#dbe9f6", "最先处理"),
        ("去混响\nWPE", "#dbe9f6", "历史预测"),
        ("声源定位\nSSL/DOA", "#f6e5db", "方向观测"),
        ("声源追踪\n轨迹平滑", "#f6e5db", "平滑方向"),
        ("空间增强\nBF / GSS / 分离", "#e8f6db", "按目标数选输出"),
        ("NS + AGC\n降噪/电平", "#e8f6db", "单通道增强"),
        ("KWS / ASR\n后端", "#f6dbdb", "取点可配置"),
    ]
    w, step, x0 = 1.65, 2.0, 0.25
    centers = []
    for index, (text_value, facecolor, role) in enumerate(main_steps):
        x = x0 + index * step
        centers.append(x + w / 2)
        ax.add_patch(plt.Rectangle(
            (x, 1.18), w, 1.26, fc=facecolor, ec="k", lw=1.2))
        ax.text(x + w / 2, 1.81, text_value, ha="center", va="center",
                fontsize=FS_LABEL)
        ax.text(x + w / 2, 0.90, role, ha="center", va="center",
                fontsize=FS_SMALL, color="dimgray")
        if index < len(main_steps) - 1:
            ax.add_patch(FancyArrowPatch(
                (x + w, 1.81), (x + step, 1.81), arrowstyle="-|>",
                mutation_scale=16, color="k", lw=1.5))

    # 追踪器向空间增强模块提供平滑后的方向或轨迹标签。
    x_tracking, x_beamformer = centers[4], centers[5]
    ax.add_patch(FancyArrowPatch(
        (x_tracking, 2.44), (x_beamformer, 2.44), arrowstyle="-|>",
        mutation_scale=17, color=C_RED, lw=2.0,
        connectionstyle="arc3,rad=-0.20"))
    ax.text((x_tracking + x_beamformer) / 2, 2.70, "平滑 DOA / 轨迹标签",
            fontsize=FS_SMALL, color=C_RED, ha="center",
            bbox=dict(fc="white", ec=C_RED, lw=0.6, alpha=0.9,
                      boxstyle="round,pad=0.2"))

    # 第九框是控制面：单个 VAD 比特不足以决定所有模块的更新策略。
    control_x, control_y, control_w, control_h = 11.95, 3.55, 3.55, 0.80
    ax.add_patch(plt.Rectangle(
        (control_x, control_y), control_w, control_h,
        fc="#f6e5db", ec=C_PURPLE, lw=1.4))
    ax.text(control_x + control_w / 2, control_y + control_h / 2,
            "活动状态控制\nVAD / 远端活动 / 双讲",
            ha="center", va="center", fontsize=FS_LABEL, color=C_PURPLE)
    bus_y = 3.08
    ax.plot([centers[1], centers[7]], [bus_y, bus_y],
            ls="--", color=C_PURPLE, lw=1.4)
    for target in [1, 2, 3, 4, 5, 7]:
        ax.add_patch(FancyArrowPatch(
            (centers[target], bus_y), (centers[target], 2.44),
            arrowstyle="-|>", mutation_scale=11, color=C_PURPLE,
            lw=1.1, ls="--"))
    ax.add_patch(FancyArrowPatch(
        (control_x + control_w / 2, control_y),
        (control_x + control_w / 2, bus_y),
        arrowstyle="-|>", mutation_scale=11, color=C_PURPLE,
        lw=1.2, ls="--"))
    ax.text(7.2, bus_y + 0.12, "按状态分别更新、保持或旁路",
            fontsize=FS_TINY, color=C_PURPLE, ha="center")

    # 增强后音频可直接送 KWS；是否绕过 VAD、何时启动 ASR 由产品策略决定。
    x_ns_out = x0 + 6 * step + w
    x_backend = centers[7]
    ax.add_patch(FancyArrowPatch(
        (x_ns_out, 1.18), (x_backend, 1.18), arrowstyle="-|>",
        mutation_scale=12, color=C_ORANGE, lw=1.2, ls="--",
        connectionstyle="arc3,rad=0.25"))
    ax.text((x_ns_out + x_backend) / 2, 0.36,
            "KWS 可绕过 VAD；是否常开、取点位置和 ASR 启动条件按产品配置",
            fontsize=FS_TINY, color=C_ORANGE, ha="center")

    ax.text(0.25, 3.62,
            "实线为音频/特征主链；虚线为控制或可选旁路。模块可流水并行，不能把历史窗长度直接相加成延迟。",
            fontsize=FS_TINY, color="dimgray", ha="left", va="center")
    ax.text(centers[5], 0.48, "单目标常用波束形成；同时输出多位说话人时选 GSS 或语音分离",
            fontsize=FS_TINY, color=C_GREEN, ha="center",
            bbox=dict(fc="white", ec=C_GREEN, lw=0.7, alpha=0.9, boxstyle="round,pad=0.2"))
    legend_x = 0.35
    legend_items = [
        ("蓝色=信号级", "#dbe9f6"), ("粉色=信息/控制", "#f6e5db"),
        ("绿色=增强级", "#e8f6db"), ("红色=后端", "#f6dbdb")]
    for index, (label, facecolor) in enumerate(legend_items):
        x = legend_x + index * 1.23
        ax.add_patch(plt.Rectangle(
            (x, 4.06), 0.25, 0.20, fc=facecolor, ec="k", lw=0.7))
        ax.text(x + 0.29, 4.16, label, fontsize=FS_TINY,
                va="center", ha="left")
    ax.set_title(
        "图23  远场语音前端：音频主链、方向反馈、状态控制与可选旁路",
        fontsize=FS_SUP)
    save(fig, "fig23_pipeline.png")


# ----------------------------------------------------------------------
# 图9 阵列规模的理想增益与处理量
# ----------------------------------------------------------------------
def array_scale_indicators(mic_counts):
    """返回阵列规模对应的理想 WNG、无序麦对数和完整协方差元素数。"""
    mic_counts = np.asarray(mic_counts)
    if mic_counts.ndim != 1 or np.any(mic_counts < 1) or np.any(mic_counts != mic_counts.astype(int)):
        raise ValueError("mic_counts 必须是一维正整数数组")
    mic_counts = mic_counts.astype(int)
    ideal_wng_db = 10 * np.log10(mic_counts)
    pair_counts = mic_counts * (mic_counts - 1) // 2
    covariance_entries = mic_counts ** 2
    return ideal_wng_db, pair_counts, covariance_entries


def fig_tradeoffs():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    mic_counts = np.array([2, 4, 6, 8, 16, 32])
    ideal_wng_db, pair_counts, covariance_entries = array_scale_indicators(mic_counts)

    ax = axes[0]
    ax.plot(mic_counts, ideal_wng_db, "o-", color=C_BLUE, lw=2,
            label=r"理想 WNG=$10\log_{10}M$")
    for m, gain in zip(mic_counts, ideal_wng_db):
        ax.annotate(f"{gain:.1f}", (m, gain), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=FS_TINY, color=C_BLUE)
    ax.set_xticks(mic_counts)
    ax.set_xlabel("麦克风数量 M", fontsize=FS_LABEL)
    ax.set_ylabel("理想白噪声增益 WNG (dB)", fontsize=FS_LABEL)
    ax.legend(fontsize=FS_SMALL, loc="lower right")
    ax.grid(ls=":", alpha=0.5)
    ax.set_title("(a) 目标已精确对齐、各通道噪声独立同方差", fontsize=FS_TITLE)
    ax.text(0.03, 0.94, "这是模型上限，不含失配、混响和相关噪声。",
            transform=ax.transAxes, va="top", fontsize=FS_SMALL, color="0.3")

    ax = axes[1]
    ax.plot(mic_counts, mic_counts, "o-", color=C_GREEN, lw=1.8, label="输入通道 M")
    ax.plot(mic_counts, pair_counts, "s--", color=C_ORANGE, lw=1.8,
            label=r"无序麦对 $M(M-1)/2$")
    ax.plot(mic_counts, covariance_entries, "^:", color=C_RED, lw=2,
            label=r"完整协方差元素 $M^2$")
    ax.set_yscale("log", base=2)
    ax.set_xticks(mic_counts)
    ax.set_xlabel("麦克风数量 M", fontsize=FS_LABEL)
    ax.set_ylabel("数据项数量（以 2 为底的对数刻度）", fontsize=FS_LABEL)
    ax.legend(fontsize=FS_SMALL, loc="upper left")
    ax.grid(ls=":", alpha=0.5, which="both")
    ax.set_title("(b) 通道、麦对和协方差矩阵的规模", fontsize=FS_TITLE)
    ax.text(0.97, 0.05, "数量由公式直接计算；不等同于运行时间、功耗或价格。",
            transform=ax.transAxes, ha="right", fontsize=FS_SMALL, color="0.3")

    fig.suptitle("图9  麦克风数量增加时的理想增益与处理量\n"
                 "（公式计算，不给出产品选型排名）", fontsize=FS_SUP - 0.5)
    fig.tight_layout()
    save(fig, "fig09_tradeoffs.png")


# ----------------------------------------------------------------------
# 通用：合成语音信号（谐波 + 音节幅度调制），供图13/图16使用
# ----------------------------------------------------------------------
def synth_speech(fs=16000, dur=1.2, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
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
    rng = np.random.default_rng(FIGURE_SEEDS["room_acoustics"])
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
    ax.axvspan(0, 12, color="0.8", alpha=0.12)
    ax.axvspan(12, 13, color=C_GREEN, alpha=0.24)
    ax.axvspan(13, 92, color=C_ORANGE, alpha=0.15)
    ax.axvspan(92, 700, color=C_RED, alpha=0.11)
    ax.axvline(12, color="gray", ls="--", lw=0.9, alpha=0.8)
    ax.axvline(92, color="gray", ls="--", lw=0.9, alpha=0.8)
    ax.set_ylim(-2.6, 3.1)
    ax.annotate("直达声\n（12 ms 到达）", xy=(12, 1.05), xytext=(16, 2.72), fontsize=9.5, color=C_GREEN,
                arrowprops=dict(arrowstyle="->", color=C_GREEN))
    ax.annotate("早期反射\n（直达后 0–80 ms）", xy=(55, 0.7), xytext=(105, 2.15), fontsize=9.5,
                color=C_ORANGE, arrowprops=dict(arrowstyle="->", color=C_ORANGE))
    ax.annotate("晚期随机尾\n（本图指数衰减模型）", xy=(260, 0.25), xytext=(330, 1.5), fontsize=9.5,
                color=C_RED, arrowprops=dict(arrowstyle="->", color=C_RED))
    ax.set_xlabel("时间 (ms)"); ax.set_ylabel("幅度")
    ax.set_title("(a) 房间脉冲响应 RIR 的三段结构", fontsize=11)
    ax.grid(ls=":", alpha=0.5)
    ax = axes[1]
    energy = np.cumsum(rir[::-1] ** 2)[::-1]  # Schroeder 反向积分
    edb = 10 * np.log10(energy / energy.max() + 1e-12)
    crossing = np.flatnonzero(edb <= -60)
    realized_ms = t[crossing[0]] * 1000 if crossing.size else np.nan
    ax.plot(t * 1000, edb, color=C_PURPLE, lw=1.8)
    ax.axhline(-60, color="k", ls="--", alpha=0.5)
    ax.axvline(T60 * 1000, color="0.45", ls=":", alpha=0.8)
    if np.isfinite(realized_ms):
        ax.axvline(realized_ms, color="k", ls="--", alpha=0.6)
        ax.annotate(f"固定随机种子的 EDC 首次到 −60 dB：{realized_ms:.0f} ms\n"
                    f"指数包络参数：名义 $T_{{60}}$={T60*1000:.0f} ms",
                    xy=(realized_ms, -60), xytext=(205, -35), fontsize=9.2,
                    arrowprops=dict(arrowstyle="->", color="k"))
    ax.set_ylim(-75, 2); ax.set_xlabel("时间 (ms)"); ax.set_ylabel("剩余能量 (dB)")
    ax.set_title("(b) 能量衰减曲线与混响时间 $T_{60}$", fontsize=11)
    ax.grid(ls=":", alpha=0.5)
    ax = axes[2]
    d = np.linspace(0.2, 6, 200)
    dc = 1.0
    direct = -20 * np.log10(d / dc)
    reverb = np.zeros_like(d)
    ax.plot(d, direct, color=C_GREEN, lw=1.8, label=r"直达声：$-20\log_{10}(d/d_c)$")
    ax.plot(d, reverb, color=C_RED, lw=1.8, label="混响声参考：本图设为 0 dB")
    ax.axvline(dc, color="k", ls="--", alpha=0.6)
    ax.annotate("本子图设定 $d_c=1.0$ m\n两条曲线在此相等", xy=(dc, 0),
                xytext=(1.7, 14), fontsize=9.5, arrowprops=dict(arrowstyle="->", color="k"))
    ax.fill_between(d, direct, reverb, where=direct > reverb, color=C_GREEN, alpha=0.08)
    ax.text(0.27, -19.5, "直达占优\n(DRR>0)", fontsize=FS_LABEL - 0.5, color=C_GREEN)
    ax.text(3.6, 6, "混响占优\n(DRR<0)", fontsize=FS_LABEL - 0.5, color=C_RED)
    ax.set_xlabel("声源-麦克风距离 (m)"); ax.set_ylabel("相对声级 (dB)")
    ax.set_ylim(-25, 22); ax.legend(fontsize=FS_SMALL); ax.grid(ls=":", alpha=0.5)
    ax.set_title("(c) 直达混响比 DRR 与临界距离", fontsize=FS_TITLE)
    fig.suptitle("图5  房间脉冲响应、能量衰减与直达混响比（本书仿真）", fontsize=FS_SUP)
    fig.tight_layout()
    save(fig, "fig05_room_acoustics.png")


# ----------------------------------------------------------------------
# 图6 数据组织方式：STFT分帧 / 时频点与快拍 / 协方差矩阵 / 特征值谱
# ----------------------------------------------------------------------
def fig_stft_cov():
    rng = np.random.default_rng(FIGURE_SEEDS["stft_cov"])
    fs = 16000
    t, sig = synth_speech(fs, 1.2, rng=rng)
    n_fft, hop = 512, 128
    win = np.hanning(n_fft)
    frames = [sig[i:i + n_fft] * win
              for i in range(0, len(sig) - n_fft + 1, hop)]
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
    ax.set_title("(c) 独立理论例：空间二阶矩阵 R（8麦）", fontsize=FS_TITLE)
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
    ax.set_title("(d) (c) 的特征值谱：已知白噪声模型下的示意", fontsize=11)
    ax.grid(ls=":", alpha=0.4)
    fig.suptitle("图6  从波形到协方差矩阵：阵列算法处理的数据形态（模拟）", fontsize=FS_SUP)
    fig.tight_layout()
    # (a)(b) 和 (c)(d) 使用不同数据，避免用箭头暗示它们来自同一次计算。
    pos_b = axes[0, 1].get_position(); pos_c = axes[1, 0].get_position()
    x_lane = pos_c.x1 + (pos_b.x0 - pos_c.x1) * 0.55      # 垂直走廊中线
    fig.text(x_lane, (pos_b.y0 + pos_c.y1) / 2 + 0.005,
             "上排：单通道分帧示例\n下排：独立的 8 麦理论例",
             fontsize=FS_SMALL, color=C_MAIN, ha="center", va="center",
             bbox=dict(fc="white", ec="0.75", alpha=0.95, boxstyle="round,pad=0.25"))
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
    ax.set_title("(b) 差分协同阵：所有有序麦对产生有符号差分位置", fontsize=11)
    fig.suptitle("图10  稀疏阵列：6 个物理麦克风对应不同的有符号差分位置（模拟）", fontsize=13)
    fig.tight_layout()
    save(fig, "fig10_sparse_array.png")


# ----------------------------------------------------------------------
# 图18 声学回声消除 AEC：框图 + NLMS收敛曲线(ERLE)
# ----------------------------------------------------------------------
def fig18_erle_simulation():
    """复现图18固定配置的分块 ERLE，并返回稳态窗口均值。"""
    rng = np.random.default_rng(FIGURE_SEEDS["aec"])
    fs = 16000
    sample_count = int(1.6 * fs)
    taps = 128
    h = np.exp(-np.arange(taps) / 25) * rng.standard_normal(taps)
    x = np.convolve(rng.standard_normal(sample_count), np.ones(8) / 8)[:sample_count]
    echo = np.convolve(x, h)[:sample_count]
    near = np.zeros(sample_count)
    near[int(0.8 * fs):int(1.2 * fs)] = (
        0.7 * rng.standard_normal(int(0.4 * fs)))
    microphone = echo + near
    weights = np.zeros(taps)
    step_size = 0.5
    residual = np.zeros(sample_count)
    freeze = np.zeros(sample_count, dtype=bool)
    for n in range(taps - 1, sample_count):
        # 当前样本在首位：[x[n], x[n-1], ..., x[n-L+1]]，与 np.convolve 对齐。
        reference = x[n - taps + 1:n + 1][::-1]
        estimate = weights @ reference
        residual[n] = microphone[n] - estimate
        if int(0.8 * fs) <= n < int(1.2 * fs):
            freeze[n] = True
            continue
        weights += (step_size * reference * residual[n]
                    / (reference @ reference + 1e-6))

    block_size = 400
    block_count = (sample_count - taps) // block_size
    erle, times = [], []
    for block in range(block_count):
        segment = slice(taps + block * block_size,
                        taps + (block + 1) * block_size)
        if freeze[segment].any():
            erle.append(np.nan)
        else:
            echo_power = np.mean(echo[segment] ** 2)
            residual_power = np.mean(residual[segment] ** 2)
            erle.append(10 * np.log10(
                echo_power / (residual_power + 1e-12)))
        times.append((taps + block * block_size) / fs)

    times = np.asarray(times)
    erle = np.asarray(erle, dtype=float)
    steady = (times > 0.45) & (times < 0.8) & np.isfinite(erle)
    plateau = float(np.mean(erle[steady]))
    return times, erle, plateau


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
    box(8.2, 2.4, 2.4, 1.0, "相减\ne(n)=d(n)−ŷ(n)", "#e8f6db")
    arrow(7.6, 2.9, 8.2, 2.9)
    arrow(9.4, 4.6, 9.4, 3.4)
    arrow(9.4, 2.4, 9.4, 1.2, "e→残余抑制/后端", C_GREEN, dy=-0.1)
    # 残差反馈驱动自适应更新；DTD 只控制这条更新支路，不切断音频输出。
    ax.plot([9.4, 9.4, 7.35], [2.4, 0.35, 0.35], color=C_BLUE, lw=1.4)
    ax.add_patch(plt.Rectangle((6.55, 0.15), 0.8, 0.4, fc="white", ec=C_RED, lw=1.2, zorder=4))
    ax.plot([6.68, 7.18], [0.49, 0.23], color=C_RED, lw=1.6, zorder=5)
    ax.plot([4.1, 6.55], [0.35, 0.35], color=C_BLUE, lw=1.4)
    arrow(4.1, 0.35, 4.1, 2.4, "残差 e 驱动更新", C_BLUE)
    box(6.0, 0.75, 2.2, 0.8, "双讲检测 DTD\n控制系数更新", "#f6e5db", 8.5)
    arrow(7.1, 0.75, 7.1, 0.35, "门控", C_RED, dy=0.05)
    ax.set_title("(a) AEC 结构：残差反馈更新滤波器，DTD 在双讲时停止系数更新", fontsize=11)
    ax = fig.add_subplot(1, 2, 2)
    t_axis, erle, erle_plateau = fig18_erle_simulation()
    ax.plot(t_axis, erle, color=C_BLUE, lw=1.6, label="ERLE（仅远端单讲区）")
    ax.axvspan(0.8, 1.2, color=C_RED, alpha=0.10)
    ax.annotate(f"收敛后 ERLE≈{erle_plateau:.0f} dB\n（线性滤波器，未含残余抑制）",
                xy=(0.62, erle_plateau), xytext=(0.18, erle_plateau + 7),
                fontsize=FS_LABEL, color=C_BLUE,
                arrowprops=dict(arrowstyle="->", color=C_BLUE))
    ax.text(1.0, 5, "双讲期冻结系数\n残差含近端语音，不计算 ERLE", fontsize=FS_LABEL - 0.5,
            color=C_RED, ha="center")
    ax.set_xlabel("时间 (s)", fontsize=FS_LABEL); ax.set_ylabel("ERLE (dB)", fontsize=FS_LABEL)
    ax.set_ylim(-10, 40)
    ax.set_title("(b) NLMS 收敛过程（ERLE，模拟）", fontsize=FS_TITLE + 1)
    ax.legend(fontsize=FS_SMALL, loc="lower right")
    ax.grid(ls=":", alpha=0.5)
    fig.suptitle("图18  声学回声消除（AEC）原理", fontsize=FS_SUP)
    fig.tight_layout()
    save(fig, "fig18_aec.png")


# ----------------------------------------------------------------------
# 图21 WPE 去混响：混响语音 → WPE → 去混响语音（算法演示）
# ----------------------------------------------------------------------
def wpe_past_frames(Y, K, delay):
    """构造单通道 WPE 回归器。

    对第 t 帧，K 列依次是 y[t-delay], ..., y[t-delay-K+1]。
    返回 (t0, Ypast, ycur)，其中 t0=delay+K-1。
    """
    Y = np.asarray(Y)
    if Y.ndim != 2:
        raise ValueError("Y 必须是 F×T 矩阵")
    if K < 1 or delay < 1:
        raise ValueError("K 和 delay 必须为正整数")
    _, T = Y.shape
    t0 = delay + K - 1
    if T <= t0:
        return t0, np.empty((Y.shape[0], K, 0), dtype=Y.dtype), Y[:, t0:]
    past = np.stack(
        [Y[:, t0 - delay - k:T - delay - k] for k in range(K)], axis=1)
    return t0, past, Y[:, t0:]


def solve_wpe_filter(weighted_covariance, weighted_cross, relative_loading=1e-6):
    """求单频点 WPE 滤波器；加载量相对矩阵平均对角线定义。"""
    weighted_covariance = np.asarray(weighted_covariance, dtype=complex)
    weighted_cross = np.asarray(weighted_cross, dtype=complex)
    if weighted_covariance.ndim != 2 or weighted_covariance.shape[0] != weighted_covariance.shape[1]:
        raise ValueError("weighted_covariance 必须是方阵")
    if weighted_cross.shape != (weighted_covariance.shape[0],):
        raise ValueError("weighted_cross 的长度必须与方阵阶数一致")
    if relative_loading < 0:
        raise ValueError("relative_loading 不能为负")
    scale = np.trace(weighted_covariance).real / weighted_covariance.shape[0]
    loading = relative_loading * max(scale, np.finfo(float).eps)
    filt = np.linalg.solve(
        weighted_covariance + loading * np.eye(weighted_covariance.shape[0]),
        weighted_cross)
    return filt, float(loading)


def scale_aligned_spectral_nmse_db(reference, estimate, frame_mask):
    """在指定帧上用一个复数增益对齐后计算谱域 NMSE。"""
    reference = np.asarray(reference, dtype=complex)
    estimate = np.asarray(estimate, dtype=complex)
    frame_mask = np.asarray(frame_mask, dtype=bool)
    if reference.shape != estimate.shape or frame_mask.shape != (reference.shape[1],):
        raise ValueError("谱矩阵须同形，frame_mask 长度须等于帧数")
    r = reference[:, frame_mask].ravel()
    x = estimate[:, frame_mask].ravel()
    gain = np.vdot(x, r) / max(float(np.vdot(x, x).real), np.finfo(float).eps)
    error_power = float(np.vdot(r - gain * x, r - gain * x).real)
    reference_power = max(float(np.vdot(r, r).real), np.finfo(float).eps)
    return 10 * np.log10(max(error_power / reference_power, np.finfo(float).eps))


def smooth_power_valid(power, width=5):
    """沿最后一轴做移动平均；边缘只除以实际参与平均的样本数。"""
    power = np.asarray(power, dtype=float)
    if power.ndim != 2:
        raise ValueError("power 必须是 F×T 矩阵")
    if not isinstance(width, (int, np.integer)) or width < 1:
        raise ValueError("width 必须为正整数")
    if power.shape[1] == 0:
        return power.copy()
    effective_width = min(int(width), power.shape[1])
    kernel = np.ones(effective_width, dtype=float)
    counts = np.convolve(np.ones(power.shape[1]), kernel, mode="same")
    return np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="same") / counts,
        1,
        power,
    )


def wpe_dereverb(Y, K=10, delay=3, iters=3):
    """单通道 WPE (Nakatani 2010)。Y: F x T 复数 STFT。"""
    Y = np.asarray(Y, dtype=complex)
    if Y.ndim != 2:
        raise ValueError("Y 必须是 F×T 矩阵")
    if not isinstance(K, (int, np.integer)) or K < 0:
        raise ValueError("K 必须为非负整数")
    if not isinstance(delay, (int, np.integer)) or delay < 1:
        raise ValueError("delay 必须为正整数")
    if not isinstance(iters, (int, np.integer)) or iters < 0:
        raise ValueError("iters 必须为非负整数")
    F, T = Y.shape
    X = Y.copy()
    if K == 0 or iters == 0 or T == 0:
        return X
    t0, Ypast, ycur = wpe_past_frames(Y, K, delay)
    if T <= t0:
        return X
    peak_power = np.max(np.abs(Y) ** 2, axis=1, keepdims=True)
    active = peak_power[:, 0] > 0
    if not np.any(active):
        return X
    lam_floor = 1e-5 * peak_power
    for _ in range(iters):
        # 先按有效样本数平滑，再施加相对功率下限。这样常数功率在边缘不被零填充压低。
        lam = smooth_power_valid(np.abs(X) ** 2, width=5)
        lam = np.maximum(lam, lam_floor)
        G = np.zeros((F, K), dtype=complex)
        w = np.zeros_like(lam[:, t0:])
        w[active] = 1.0 / lam[active, t0:]
        for f in np.flatnonzero(active):
            P = Ypast[f]  # K x Tv
            ww = w[f]
            Rw = (P * ww[None, :]) @ P.conj().T
            rw = (P * ww[None, :]) @ ycur[f].conj()
            G[f], _ = solve_wpe_filter(Rw, rw, relative_loading=1e-6)
        X[:, t0:] = ycur - np.einsum("fk,fkt->ft", G.conj(), Ypast)
    return X


def fig_wpe():
    rng = np.random.default_rng(FIGURE_SEEDS["wpe"])
    fs = 16000
    t, clean = synth_speech(fs, 1.4, rng=rng)
    T60 = 0.6
    L = int(0.8 * fs)
    tt = np.arange(L) / fs
    direct_index = int(0.01 * fs)
    drr_db = 6.0
    late_tail = rng.standard_normal(L) * np.exp(-6.91 * tt / T60)
    late_tail[:direct_index + 1] = 0
    # 令直达脉冲能量为 1，晚期尾声总能量为 10^(-DRR/10)。
    # 这样归一化不会把直达声淹没，三幅图的强弱关系也有明确口径。
    late_tail /= np.linalg.norm(late_tail)
    late_tail *= 10 ** (-drr_db / 20)
    rir = late_tail
    rir[direct_index] = 1.0
    rev = np.convolve(clean, rir)[:len(clean)]
    n_fft, hop = 512, 128
    win = np.hanning(n_fft)
    def stft(x):
        fr = [x[i:i + n_fft] * win for i in range(0, len(x) - n_fft, hop)]
        return np.array([np.fft.rfft(f) for f in fr]).T
    Yc, Yr = stft(clean), stft(rev)
    # RIR 的直达脉冲位于 direct_index。显示仍使用未移位的干净语音，
    # 但失真指标和活动/静音帧必须先把参考对齐到同一传播时刻。
    clean_aligned = causal_delay(clean, direct_index)
    Yc_aligned = stft(clean_aligned)
    prediction_order, prediction_delay = 10, 6
    Xd = wpe_dereverb(Yr, K=prediction_order, delay=prediction_delay, iters=3)
    aligned_frame_power = np.mean(np.abs(Yc_aligned) ** 2, axis=0)
    active = aligned_frame_power >= 0.10 * aligned_frame_power.max()
    quiet = aligned_frame_power <= 0.01 * aligned_frame_power.max()
    def quiet_energy_ratio_db(spectrum):
        quiet_energy = np.sum(np.abs(spectrum[:, quiet]) ** 2)
        total_energy = np.sum(np.abs(spectrum) ** 2)
        return 10 * np.log10(max(quiet_energy / total_energy, np.finfo(float).eps))
    rev_quiet_db = quiet_energy_ratio_db(Yr)
    wpe_quiet_db = quiet_energy_ratio_db(Xd)
    rev_nmse_db = scale_aligned_spectral_nmse_db(Yc_aligned, Yr, active)
    wpe_nmse_db = scale_aligned_spectral_nmse_db(Yc_aligned, Xd, active)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0), constrained_layout=True)
    tms = np.arange(Yc.shape[1]) * hop / fs * 1000
    fk = np.fft.rfftfreq(n_fft, 1 / fs) / 1000
    reference_amplitude = max(np.abs(S).max() for S in (Yc, Yr, Xd))
    mesh = None
    for ax, (S, title) in zip(axes, [(Yc, "(a) 干净语音（音节边界清晰）"),
                                     (Yr, "(b) 混响语音：能量沿时间拖尾、边界变模糊"),
                                     (Xd, "(c) WPE 去混响后：拖尾被抑制")]):
        # 10log10(|S|^2/ref^2) 与 20log10(|S|/ref) 等价；三图共用参考值。
        Sdb = 10 * np.log10((np.abs(S) ** 2 + 1e-16) / reference_amplitude ** 2)
        mesh = ax.pcolormesh(tms, fk, Sdb, cmap="viridis", shading="auto",
                             vmin=-55, vmax=0)
        ax.set_ylim(0, 2.5); ax.set_xlabel("时间 (ms)", fontsize=FS_LABEL)
        ax.text(0.98, 0.04, "高频无能量故全黑", transform=ax.transAxes, fontsize=FS_TINY, color="white", ha="right", va="bottom")
        ax.set_ylabel("频率 (kHz)", fontsize=FS_LABEL)
        ax.set_title(title, fontsize=FS_TITLE)
    axes[1].text(0.03, 0.05,
                 f"静音帧能量占比 {rev_quiet_db:.1f} dB\n对齐参考活跃帧 NMSE {rev_nmse_db:.1f} dB",
                 transform=axes[1].transAxes, fontsize=FS_TINY, color="white", va="bottom",
                 bbox=dict(fc="black", ec="none", alpha=0.55, pad=2))
    axes[2].text(0.03, 0.05,
                 f"静音帧能量占比 {wpe_quiet_db:.1f} dB\n对齐参考活跃帧 NMSE {wpe_nmse_db:.1f} dB",
                 transform=axes[2].transAxes, fontsize=FS_TINY, color="white", va="bottom",
                 bbox=dict(fc="black", ec="none", alpha=0.55, pad=2))
    fig.suptitle(
        "图21  WPE 去混响效果（本书单通道仿真）\n"
        "输入：16 kHz、1.4 s 合成音节、随机种子 21001；RIR：直达延迟 10 ms、"
        "0.8 s 指数随机尾、$T_{60}$=0.6 s、DRR=6 dB、无加性噪声\n"
        "STFT：Hann 窗 512 点、帧移 128 点；WPE：$\\Delta$=6 帧、$K$=10、3 次迭代；"
        "三图共用谱幅参考值；NMSE 与帧掩码按 10 ms 直达延迟对齐干净参考",
        fontsize=FS_SUP - 2)
    cb = fig.colorbar(mesh, ax=axes, shrink=0.85, pad=0.015, label="相对谱能量 (dB)")
    cb.ax.tick_params(labelsize=FS_SMALL + 1)
    cb.set_label("相对谱能量 (dB)", fontsize=FS_LABEL + 2)
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
    max_itd_us = float(itd[-1])
    ax.axhline(max_itd_us, color="gray", ls="--", lw=0.8)
    ax.annotate(f"最大值 ≈ {max_itd_us:.0f} μs（正侧面 90°）",
                xy=(90, max_itd_us), xytext=(-5, 500), fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.annotate("正前方偏 1° ≈ 9 μs\n（几何换算，不是听觉阈值）", xy=(1, 9), xytext=(-64, 300), fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color=C_RED), color=C_RED)
    ax.set_xlabel("声源方位角 (°)"); ax.set_ylabel("ITD (μs)")
    ax.set_title("(b) 双耳时间差 ITD（Woodworth 模型）", fontsize=11)
    ax.grid(ls=":", alpha=0.5)
    ax = fig.add_subplot(1, 3, 3)
    f = np.logspace(np.log10(100), np.log10(10000), 300)
    for (az_d, ccurve, line_style) in [(90, C_RED, "-"), (60, C_ORANGE, "--"), (30, C_BLUE, ":")]:
        ild = 25 * np.clip((f - 400) / 5600, 0, 1) ** 1.3 * np.sin(np.deg2rad(az_d))
        ax.semilogx(f, ild, color=ccurve, ls=line_style, lw=2, label=f"方位 {az_d}°")
    ax.axvspan(100, 1500, color=C_BLUE, alpha=0.06)
    ax.axvspan(3000, 10000, color=C_RED, alpha=0.06)
    ax.text(200, 21, "低频段：ITD 为主\n(相位可用 <1.5 kHz)", fontsize=FS_LABEL - 0.5, color=C_BLUE)
    ax.text(9500, 1.2, "高频段：ILD 为主\n(头影 >3 kHz 显著)", fontsize=FS_LABEL - 0.5, color=C_RED,
            ha="right")
    ax.set_xlabel("频率 (Hz)"); ax.set_ylabel("示意相对声级差 (dB)")
    ax.set_ylim(0, 27); ax.legend(fontsize=9, loc="lower left")
    ax.set_title("(c) ILD 定性趋势（手工示意曲线，不代表某个头部实测）", fontsize=10.5)
    ax.grid(ls=":", alpha=0.5, which="both")
    fig.suptitle("图1  人类双耳听觉的定位线索（Rayleigh 双重理论）", fontsize=13)
    fig.tight_layout()
    save(fig, "fig01_binaural.png")


# ----------------------------------------------------------------------
# 图13 GCC-PHAT 在混响下的退化（说明性随机指数衰减尾模型）
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
        """一次说明性随机模拟：返回 (是否找对峰, 峰对比度)。

        直达分量设为 1，两个通道各加独立的随机指数衰减尾。该模型不包含房间几何、
        声源—麦克风距离或经标定的 DRR，统计数字只用于检查定性趋势。
        """
        r = np.random.default_rng(seed)
        rir1 = random_decay_rir(T60, fs, r)
        rir2 = random_decay_rir(T60, fs, r)
        sig = make_src(seed + 1000)
        x1 = fft_convolve_prefix(sig, rir1, N)
        x2 = fft_convolve_prefix(causal_delay(sig, tau_true), rir2, N)
        i0 = r.integers(0, N - frame - 1)
        f1 = x1[i0:i0 + frame]
        f2 = x2[i0:i0 + frame]
        lags, g = gcc_phat(f1, f2)
        pk = lags[np.argmax(g)]
        contrast = gcc_peak_contrast(g)
        return gcc_peak_is_correct(pk, tau_true), contrast

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
        rir1 = random_decay_rir(T60, fs, r)
        rir2 = random_decay_rir(T60, fs, r)
        sig = make_src(3)
        x1 = fft_convolve_prefix(sig, rir1, N)
        x2 = fft_convolve_prefix(causal_delay(sig, tau_true), rir2, N)
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
        ok = gcc_peak_is_correct(pk, tau_true)
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
    acc, acc_low, acc_high, contrast, contrast_q1, contrast_q3 = [], [], [], [], [], []
    for T60 in T60s:
        res = [trial(T60, 2000 + k * 17) for k in range(150)]
        correct = int(np.sum([r[0] for r in res]))
        low, high = wilson_interval(correct, len(res))
        acc.append(100 * correct / len(res)); acc_low.append(100 * low); acc_high.append(100 * high)
        values = np.asarray([r[1] for r in res])
        contrast.append(np.median(values))
        q1, q3 = np.percentile(values, [25, 75])
        contrast_q1.append(q1); contrast_q3.append(q3)
    acc = np.asarray(acc); acc_low = np.asarray(acc_low); acc_high = np.asarray(acc_high)
    contrast = np.asarray(contrast)
    ax.errorbar(T60s, acc, yerr=[acc - acc_low, acc_high - acc], fmt="o-",
                capsize=3, color=C_RED, lw=1.8, label="峰正确率与95% Wilson区间（左轴）")
    ax.set_xlabel("混响时间 $T_{60}$ (s)")
    ax.set_ylabel("正确率 (%)", color=C_RED)
    ax.set_ylim(0, 105)
    ax.tick_params(axis="y", labelcolor=C_RED, labelsize=FS_SMALL)
    ax.tick_params(axis="x", labelsize=FS_SMALL)
    ax2 = ax.twinx()
    ax2.plot(T60s, contrast, "s--", color=C_BLUE, lw=1.8, label="峰对比度（右轴）")
    ax2.fill_between(T60s, contrast_q1, contrast_q3, color=C_BLUE, alpha=0.14,
                     label="峰对比度四分位区间")
    ax2.set_ylabel("峰/背景中位数 比值", color=C_BLUE)
    ax2.tick_params(axis="y", labelcolor=C_BLUE, labelsize=FS_SMALL)
    h1, lb1 = ax.get_legend_handles_labels()
    h2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, lb1 + lb2, fontsize=FS_SMALL, loc="upper left")
    ax.set_title("(e) 150 次随机模拟（64 ms 短帧）：\n"
                 "正确=真峰±1点；对比度=max(g)/median|g|", fontsize=10.5)
    ax.grid(ls=":", alpha=0.5)
    fig.suptitle("图13  随机衰减尾会扰乱 GCC-PHAT 峰\n（说明性随机尾模型；黑虚线=直达声时延；无传感器噪声；不用于报告绝对性能）", fontsize=FS_SUP)
    fig.subplots_adjust(left=0.055, right=0.94, bottom=0.14, top=0.77, wspace=0.42)
    save(fig, "fig13_gcc_reverb.png")


# ----------------------------------------------------------------------
# 图4 时延 = 相位旋转：复数表示的几何直觉
# ----------------------------------------------------------------------
def fig_delay_phase():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    sound_speed = 343.0
    spacing = 0.04
    tau = spacing / sound_speed  # 4 cm 端射入射的最大麦间时延
    # (a) 时域：正弦延迟
    ax = axes[0]
    f0 = 1000.0
    t = np.linspace(0, 2.2e-3, 800)
    ax.plot(t * 1e3, np.sin(2 * np.pi * f0 * t), color=C_BLUE, lw=2, label="原始信号 s(t)")
    ax.plot(t * 1e3, np.sin(2 * np.pi * f0 * (t - tau)), color=C_RED, lw=2, ls="--",
            label="延迟 τ 后 s(t−τ)")
    arrow_start = 1.0
    ax.annotate("", xy=(arrow_start + tau * 1e3, -0.707), xytext=(arrow_start, -0.707),
                arrowprops=dict(arrowstyle="->", color="k", lw=1.5))
    ax.text(arrow_start + tau * 500, -1.15, "τ≈0.117 ms", fontsize=11, ha="center")
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
    ang = -2 * np.pi * f0 * tau
    ax.annotate("", xy=(np.cos(ang), np.sin(ang)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=2.5, mutation_scale=18))
    angle_deg = np.rad2deg(ang)
    arc = matplotlib.patches.Arc((0, 0), 1.2, 1.2, theta1=angle_deg, theta2=0, color="k", lw=1.2)
    ax.add_patch(arc)
    ax.annotate("", xy=(0.69, -0.39), xytext=(0.70, -0.18),
                arrowprops=dict(arrowstyle="->", color="k", lw=1.2))
    ax.text(1.05, 0.06, "原始 $S(f)$", fontsize=10, color=C_BLUE)
    ax.text(0.05, -1.18, "延迟后 S(f)·e^(−j2πfτ)", fontsize=10, color=C_RED)
    ax.text(0.60, -0.12, "2πfτ≈42°", fontsize=11, style="italic")
    ax.text(-1.55, 1.15, "延迟 τ 等价于复平面上\n顺时针转 2πfτ", fontsize=10, color=C_MAIN)
    ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.55, 1.5)
    ax.set_title("(b) 频域看延迟：复数相位转过一个角度", fontsize=11)
    # (c) 相位-频率直线
    ax = axes[2]
    f = np.linspace(0, 8000, 400)
    phase = -2 * np.pi * f * tau
    ax.plot(f / 1000, np.rad2deg(phase), color=C_BLUE, lw=2)
    for fk in [1000, 2000, 3000, 4000]:
        ax.plot(fk / 1000, np.rad2deg(-2 * np.pi * fk * tau), "o", color=C_RED, ms=6)
    ax.annotate("1 kHz → -42°\n2 kHz → -84°\n转角与频率成正比\n斜率 = -2πτ", xy=(3.0, -125),
                xytext=(3.6, -55), fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color="k"))
    ax.set_xlabel("频率 (kHz)"); ax.set_ylabel("相位 (°)")
    ax.set_title("(c) 同一延迟在不同频率：相位-频率是直线", fontsize=11)
    ax.grid(ls=":", alpha=0.5)
    fig.suptitle("图4  时延为什么变成 $e^{-j2\\pi f\\tau}$：几何直觉\n"
                 "（4 cm 麦距；端射方向达到最大麦间时延 τ=d/c≈0.117 ms）", fontsize=12.5)
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
    # GridSpec 中再创建子 Axes，tight_layout 会警告；手工留出标题和间距。
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.07, top=0.88,
                        wspace=0.38, hspace=0.34)
    save(fig, "fig02_dsin_geometry.png")


# ----------------------------------------------------------------------
# 图20 AEC 信号处理链：经典结构 vs 混合式结构（示意）
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
    box(5.7, 6.2, 2.5, 0.95, "扬声器—房间—麦克风\n回声路径 h(n)", fs=FS_TINY)
    arrow(2.4, 6.68, 3.2, 6.68); arrow(4.9, 6.68, 5.7, 6.68)
    sum_node(9.15, 6.68)
    arrow(8.2, 6.68, 8.82, 6.68)
    box(5.7, 4.3, 2.5, 0.9, "近端语音 s(n)\n+ 环境噪声 v(n)", "#f6dbdb", FS_TINY)
    arrow(6.95, 5.2, 8.95, 6.4, c=C_RED)
    box(10.3, 6.2, 2.3, 0.95, "麦克风信号 d(n)\n= s + x*h + v", "#f6dbdb", FS_TINY)
    arrow(9.48, 6.68, 10.3, 6.68)

    # 下排：参考 → 延迟对齐 → 自适应滤波 → Σ(减)
    box(0.3, 2.5, 1.9, 0.95, "延迟对齐 τ̂\n（参考与回声输入对齐）", fs=FS_TINY)
    box(3.2, 2.5, 2.6, 0.95, "自适应滤波器 ŵ(n)\n（NLMS，估计线性路径）", fs=FS_TINY)
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
    ax.text(8.6, 0.82, "非双讲段用残差 e(n) 驱动滤波器系数更新", fontsize=FS_TINY,
            color=C_BLUE, ha="center")
    # 反馈线上的“开关”（双讲时断开）
    ax.add_patch(plt.Rectangle((5.4, 0.95), 1.2, 0.4, fc="white", ec=C_RED, lw=1.4, zorder=4))
    ax.plot([5.62, 6.3], [1.3, 1.06], color=C_RED, lw=1.8, zorder=5)
    ax.text(6.0, 1.5, "开关（双讲时断开）", fontsize=FS_SMALL, color=C_RED, ha="center", zorder=5)
    # DTD 控制盒
    box(5.2, 0.0, 1.6, 0.75, "双讲检测\nDTD", "#f6e5db", FS_TINY)
    ax.add_patch(FancyArrowPatch((6.0, 0.75), (6.0, 1.0), arrowstyle="-|>",
                                 mutation_scale=12, color=C_RED, lw=1.2, zorder=4))

    ax.annotate("ŵ(n) 估计上方线性回声路径 h(n)", xy=(4.5, 3.5), xytext=(4.5, 5.6),
                fontsize=FS_SMALL + 2, color=C_PURPLE, ha="center",
                arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=1.2,
                                connectionstyle="arc3,rad=-0.25"))
    ax.set_title("(a) 经典 AEC 信号流：已知参考学出回声副本再相减（示意）", fontsize=FS_TITLE)

    # ---- (b) 混合式处理链 ----
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
    ax2.text(5.4, 2.28, "参考 x 也可作为 DNN 输入（帮助区分残余回声与近端语音）",
             fontsize=FS_SMALL + 1.5, color=C_BLUE, ha="center", alpha=0.9,
             bbox=dict(fc="white", alpha=0.7, pad=1, ec="none"))
    # DTD / 步长控制
    box2(3.4, 0.7, 5.4, 0.95,
         "DTD / 步长控制（根据参考、麦克风与残差信号控制线性滤波器更新）",
         "white", FS_TINY, ec=C_ORANGE, ls="--")
    arrow2(6.1, 1.65, 6.1, y0, "控制更新", C_ORANGE)
    # 分工注释
    ax2.text(6.1, 3.85, "线性 AEC 只能消除滤波器能够表示并由参考解释的回声分量",
             fontsize=FS_TINY, color=C_BLUE, ha="center")
    ax2.text(11.05, 3.85, "DNN 抑制训练条件覆盖的非线性残余、噪声或混响",
             fontsize=FS_TINY, color=C_ORANGE, ha="center")
    # 图例
    ax2.add_patch(plt.Rectangle((0.3, 7.3), 0.4, 0.4, fc=C_DSP, ec="k", lw=1.0))
    ax2.text(0.8, 7.5, "传统 DSP", fontsize=FS_TINY, va="center")
    ax2.add_patch(plt.Rectangle((2.6, 7.3), 0.4, 0.4, fc=C_NN, ec="k", lw=1.0))
    ax2.text(3.1, 7.5, "神经网络", fontsize=FS_TINY, va="center")
    ax2.set_title("(b) 混合式处理链：线性回声估计 + 学习型残余抑制（示意）", fontsize=FS_TITLE)

    fig.suptitle("图20  AEC 信号处理链：自适应滤波与混合式结构（示意）", fontsize=FS_SUP)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "fig20_aec_pipeline.png")


# ----------------------------------------------------------------------
# 图19 AEC 全景：线性模型失配仿真 + 算法家族按假设分类
# ----------------------------------------------------------------------
def fig_aec_landscape():
    fig = plt.figure(figsize=(13.5, 8.8))

    # ---- (a) 线性 AEC 面对未建模非线性时的单次可复现仿真 ----
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
        for n in range(taps - 1, N):
            xv = x[n - taps + 1:n + 1][::-1]
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
    ax.annotate(f"线性路径后段均值：{p_lin:.1f} dB", xy=(2.5, p_lin), xytext=(1.7, p_lin + 6),
                fontsize=FS_SMALL + 1, color=C_BLUE,
                arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.4))
    ax.annotate(f"tanh 路径后段均值：{p_nl:.1f} dB\n"
                "线性滤波器不能表示该非线性映射",
                xy=(1.3, p_nl - 0.5), xytext=(0.5, 7),
                fontsize=FS_SMALL + 1, color=C_RED,
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.4))
    ax.text(0.98, 0.05, "配置：16 kHz，3 s，256 抽头，μ=0.3，tanh 驱动系数=1.1，随机种子=7。\n"
            "后段均值只描述本次仿真，不能外推为设备性能上限。",
            transform=ax.transAxes, fontsize=FS_SMALL, color="0.25", ha="right", va="bottom",
            bbox=dict(fc="white", ec="0.7", alpha=0.92))
    ax.set_xlabel("时间 (s)", fontsize=FS_LABEL)
    ax.set_ylabel("ERLE (dB)", fontsize=FS_LABEL)
    ax.set_ylim(-5, 45); ax.set_xlim(0, 3)
    ax.legend(fontsize=FS_SMALL, loc="upper left")
    ax.grid(ls=":", alpha=0.5)
    ax.set_title("(a) 同一 NLMS 在匹配的线性路径和未建模 tanh 路径上的结果", fontsize=FS_TITLE)

    # ---- (b) 按模型假设和处理对象分类；不画未经逐项引用核实的历史时间线 ----
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.axis("off"); ax2.set_xlim(0, 14); ax2.set_ylim(0, 4.6)
    families = [
        (0.25, "逐样本时域\nLMS / NLMS", "线性路径\n短到中等滤波器", "步长、DTD"),
        (3.7, "分区块频域\nMDF / PBFDAF", "线性长路径\n按块卷积与更新", "块长、约束"),
        (7.15, "状态空间\n频域 Kalman", "路径变化由\n状态模型描述", "过程/观测噪声"),
        (10.6, "残余抑制\n规则或学习模型", "输入线性 AEC 残差\n处理剩余成分", "近端保真、泛化"),
    ]
    for x0, name, assumption, checks in families:
        ax2.add_patch(plt.Rectangle((x0, 1.0), 2.9, 2.6, fc="#dbe9f6" if x0 < 10 else "#fde3c8",
                                    ec="k", lw=1.2))
        ax2.text(x0 + 1.45, 3.15, name, ha="center", va="center", fontsize=FS_LABEL)
        ax2.text(x0 + 1.45, 2.15, assumption, ha="center", va="center", fontsize=FS_SMALL)
        ax2.text(x0 + 1.45, 1.28, "需验证：" + checks, ha="center", va="center",
                 fontsize=FS_TINY, color="0.3")
    ax2.text(7.0, 0.35, "这些模块按路径长度、变化速度、非线性程度和资源约束选择；横向位置不表示年代或性能排名。",
             ha="center", fontsize=FS_SMALL, color="0.3")
    ax2.set_title("(b) 按模型假设和处理对象区分 AEC 方法", fontsize=FS_TITLE)

    fig.suptitle("图19  AEC 的模型失配与方法分类", fontsize=FS_SUP)
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
        elif n_show - delay + 1 <= i <= n_show - 1:
            fc, ec, lw, tag = "#eeeeee", "0.5", 1.0, None
        elif n_show - delay - K + 1 <= i <= n_show - delay:
            fc, ec, lw, tag = "#dbe9f6", C_BLUE, 1.6, None
        else:
            fc, ec, lw, tag = "#f7f7f7", "0.7", 1.0, None
        ax.add_patch(plt.Rectangle((x, y0), bw, 1.1, fc=fc, ec=ec, lw=lw, zorder=3))
        if i == n_show:
            ax.text(x + bw / 2, y0 + 0.55, "t", ha="center", va="center",
                    fontsize=FS_TITLE, color=C_RED, weight=600, zorder=4)
            ax.text(x + bw / 2, y0 - 0.35, tag, ha="center", va="top", fontsize=FS_SMALL, color=C_RED)
        labels[i] = x
    i_k0, i_k1 = n_show - delay - K + 1, n_show - delay
    x_k = (labels[i_k0] + labels[i_k1] + bw) / 2
    # Δ=3 指“当前帧 t 与最近预测帧 t-3 的帧索引差”；中间实际跳过 t-1、t-2。
    x_delay_left = labels[i_k1] + bw / 2
    x_delay_right = labels[n_show] + bw / 2
    ax.annotate("", xy=(x_delay_left, y0 + 2.05), xytext=(x_delay_right, y0 + 2.05),
                arrowprops=dict(arrowstyle="<->", color="0.45", lw=1.4))
    ax.text((x_delay_left + x_delay_right) / 2, y0 + 2.25,
            f"预测延迟 Δ={delay}帧\n（跳过 t−1、t−2）", ha="center", va="bottom",
            fontsize=FS_SMALL + 2, color="0.35")
    ax.annotate("", xy=(labels[i_k0], y0 + 1.35), xytext=(labels[i_k1] + bw, y0 + 1.35),
                arrowprops=dict(arrowstyle="<->", color=C_BLUE, lw=1.6))
    ax.text(x_k, y0 + 1.6, f"历史窗 K={K}帧\n（回归器：只用过去）", ha="center", va="bottom",
            fontsize=FS_SMALL + 2, color=C_BLUE,
            bbox=dict(fc="white", ec=C_BLUE, lw=0.7, alpha=0.9, boxstyle="round,pad=0.25"))
    # 预测箭头：历史窗底部 → 当前帧底部（走条带下方，不压框）
    ax.add_patch(FancyArrowPatch((x_k, y0 - 0.12), (labels[n_show] + bw / 2, y0 - 0.12),
                                 arrowstyle="-|>", mutation_scale=14, color=C_GREEN, lw=1.5, zorder=2))
    ax.text((x_k + labels[n_show]) / 2 - 0.15, y0 - 0.30,
            r"晚期估计：$\hat y(t,f)=\sum_{k=0}^{K-1}G^*(k,f)y(t-\Delta-k,f)$" "\n"
            r"去混响输出：$y(t,f)-\hat y(t,f)$",
            ha="center", va="top", fontsize=FS_SMALL + 1, color=C_GREEN,
            bbox=dict(fc="white", alpha=0.8, pad=1, ec="none"))
    ax.text(7.4, 1.15, "回归只用过去帧；端到端仍含分帧与计算延迟", ha="center", va="center",
            fontsize=FS_LABEL, color=C_MAIN,
            bbox=dict(fc="#e8f6db", ec=C_GREEN, lw=0.8, alpha=0.9, boxstyle="round,pad=0.35"))
    fig.suptitle("图24  WPE 延迟预测：Δ=3 时保护 t−1、t−2，K=5 使用 t−3…t−7", fontsize=FS_SUP)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, "fig24_wpe_frames.png")


# ----------------------------------------------------------------------
# 图25 延迟分类：关键路径 vs 因果历史状态
# ----------------------------------------------------------------------
def fig_latency_budget():
    fig, (ax, info) = plt.subplots(
        1, 2, figsize=(14.5, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})

    # 数字只是一组可替换的工作表格示例，不声称代表某产品。只有同一条关键路径上的项目才相加。
    critical_path = [("采集/分帧", 32, "#dbe9f6"),
                     ("未来帧前瞻", 16, C_PURPLE),
                     ("判决确认", 25, C_ORANGE),
                     ("计算/调度", 20, C_RED)]
    left = 0
    for name, duration_ms, color in critical_path:
        ax.barh(0, duration_ms, left=left, height=0.48, color=color,
                edgecolor="k", lw=1.0)
        ax.text(left + duration_ms / 2, 0, f"{name}\n{duration_ms} ms",
                ha="center", va="center", fontsize=FS_SMALL,
                color="white" if color == C_RED else C_MAIN)
        left += duration_ms
    ax.annotate(f"关键路径示例：{left} ms",
                xy=(left, 0.24), xytext=(left - 2, 0.68), ha="right",
                fontsize=FS_LABEL, arrowprops=dict(arrowstyle="->", lw=1.1))
    ax.set_xlim(0, 110); ax.set_ylim(-0.72, 1.0)
    ax.set_yticks([0]); ax.set_yticklabels(["同一关键路径"])
    ax.set_xlabel("与实时时钟同量纲的延迟 (ms)", fontsize=FS_LABEL)
    ax.set_title("(a) 只相加真正串行的延迟", fontsize=FS_TITLE)
    ax.grid(axis="x", ls=":", alpha=0.5)

    info.axis("off")
    info.set_xlim(0, 1); info.set_ylim(0, 1)
    info.text(0.04, 0.92, "会增加关键路径", fontsize=FS_TITLE, color=C_RED,
              weight=600, va="top")
    adds = ["等待未来帧（look-ahead）", "集齐整块数据后才开始",
            "VAD/KWS 确认或 hangover", "不能被流水隐藏的计算与调度"]
    for i, text_value in enumerate(adds):
        info.text(0.06, 0.82 - 0.09 * i, "• " + text_value, fontsize=FS_LABEL, va="top")
    info.text(0.04, 0.46, "不能直接当成串行前瞻", fontsize=FS_TITLE,
              color=C_GREEN, weight=600, va="top")
    states = ["WPE 的过去帧 K/Δ", "追踪器的历史状态",
              "协方差递归/平滑窗", "AGC 的因果时间常数"]
    for i, text_value in enumerate(states):
        info.text(0.06, 0.37 - 0.065 * i, "• " + text_value, fontsize=FS_LABEL, va="top")
    info.text(0.04, 0.01,
              "实际验收：打时间戳，测“音频入口 → 可用输出/首个判决”；\n"
              "若模块并行或流水，不要把每个内部窗长机械相加。",
              fontsize=FS_SMALL, color="dimgray", va="bottom",
              bbox=dict(fc="#f4f4f4", ec="0.7", boxstyle="round,pad=0.35"))
    info.set_title("(b) 先分清前瞻与因果历史", fontsize=FS_TITLE)
    fig.suptitle("图25  端到端延迟：找关键路径，不把历史窗当成串行等待（示例数字非产品指标）", fontsize=FS_SUP)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.84, wspace=0.28)
    save(fig, "fig25_latency_budget.png")


# ----------------------------------------------------------------------
# 图33 互相关两种等价实现：时域直接求和 vs FFT
# ----------------------------------------------------------------------
def time_domain_correlation(x, y):
    """直接按 R_xy[l]=sum_n x[n] conj(y[n-l]) 计算线性互相关。"""
    x = np.asarray(x)
    y = np.asarray(y)
    dtype = np.result_type(x, y, np.complex128)
    lags = np.arange(-(len(y) - 1), len(x))
    values = np.zeros(len(lags), dtype=dtype)
    for index, lag in enumerate(lags):
        n0 = max(0, lag)
        n1 = min(len(x), len(y) + lag)
        if n1 > n0:
            values[index] = np.sum(x[n0:n1] * np.conj(y[n0 - lag:n1 - lag]))
    return lags, np.real_if_close(values)


def fft_correlation(x, y):
    """用零填充 FFT 独立计算与 time_domain_correlation 同一定义的线性互相关。"""
    x = np.asarray(x)
    y = np.asarray(y)
    linear_length = len(x) + len(y) - 1
    n_fft = 1 << (linear_length - 1).bit_length()
    circular = np.fft.ifft(
        np.fft.fft(x, n_fft) * np.conj(np.fft.fft(y, n_fft)))
    negative = circular[n_fft - (len(y) - 1):] if len(y) > 1 else circular[:0]
    values = np.concatenate((negative, circular[:len(x)]))
    lags = np.arange(-(len(y) - 1), len(x))
    return lags, np.real_if_close(values)


def fig_gcc_two_ways():
    """独立运行直接求和与 FFT 实现，并在画图前数值断言二者等价。"""
    fs = 16000
    n = 128
    delay_samples = 4
    rng = np.random.default_rng(33001)
    x_early = rng.standard_normal(n)
    # 低通只为让波形更易读；延迟采用零填充，不使用循环移位。
    spectrum = np.fft.rfft(x_early)
    spectrum[np.fft.rfftfreq(n, 1 / fs) > 3500] = 0
    x_early = np.fft.irfft(spectrum, n)
    x_late = np.zeros_like(x_early)
    x_late[delay_samples:] = x_early[:-delay_samples]

    lags_td, corr_td = time_domain_correlation(x_late, x_early)
    lags_fft, corr_fft = fft_correlation(x_late, x_early)
    if not np.array_equal(lags_td, lags_fft):
        raise AssertionError("时域与 FFT 互相关的 lag 定义不一致")
    max_error = float(np.max(np.abs(corr_td - corr_fft)))
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(corr_td))))
    if max_error > tolerance:
        raise AssertionError(
            f"时域与 FFT 互相关不一致：max error={max_error:.3e}")

    scale = float(np.max(np.abs(corr_td)))
    corr_td_n = corr_td / scale
    corr_fft_n = corr_fft / scale
    peak_lag = int(lags_td[np.argmax(corr_td_n)])
    lags_ms = lags_td / fs * 1000
    view = np.abs(lags_td) <= 18

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 7.0))
    sample_index = np.arange(36)
    axes[0, 0].plot(sample_index, x_early[:36], "o-", ms=3, lw=1.2,
                    color=C_BLUE, label="x₂[n]（先到）")
    axes[0, 0].plot(sample_index, x_late[:36], "s-", ms=3, lw=1.2,
                    color=C_RED, label=f"x₁[n]（晚 {delay_samples} 点）")
    axes[0, 0].set_title("(a) 输入：零填充整数延迟", fontsize=FS_TITLE)
    axes[0, 0].set_xlabel("采样点 n", fontsize=FS_LABEL)
    axes[0, 0].set_ylabel("幅度", fontsize=FS_LABEL)
    axes[0, 0].legend(fontsize=FS_SMALL)
    axes[0, 0].grid(ls=":", alpha=0.5)

    axes[1, 0].stem(lags_ms[view], corr_td_n[view], linefmt=C_BLUE,
                    markerfmt="o", basefmt="0.6")
    axes[1, 0].axvline(delay_samples / fs * 1000, color=C_RED, ls="--",
                       label=f"真值 +{delay_samples} 点")
    axes[1, 0].set_title(
        "(b) 时域：逐 lag 直接求和（独立实现）", fontsize=FS_TITLE)
    axes[1, 0].set_xlabel("lag (ms)", fontsize=FS_LABEL)
    axes[1, 0].set_ylabel("归一化互相关", fontsize=FS_LABEL)
    axes[1, 0].legend(fontsize=FS_SMALL)
    axes[1, 0].grid(ls=":", alpha=0.5)

    freq_khz = np.fft.rfftfreq(n, 1 / fs) / 1000
    cross_spectrum = np.fft.rfft(x_late) * np.conj(np.fft.rfft(x_early))
    axes[0, 1].plot(freq_khz, np.abs(cross_spectrum), color=C_PURPLE, lw=1.3)
    axes[0, 1].set_title("(c) 频域乘积 X₁·conj(X₂)", fontsize=FS_TITLE)
    axes[0, 1].set_xlabel("频率 (kHz)", fontsize=FS_LABEL)
    axes[0, 1].set_ylabel("幅度", fontsize=FS_LABEL)
    axes[0, 1].set_xlim(0, 4.0)
    axes[0, 1].grid(ls=":", alpha=0.5)

    axes[1, 1].plot(lags_ms[view], corr_fft_n[view], color=C_RED, lw=1.8,
                    label="零填充 FFT + IFFT")
    axes[1, 1].plot(peak_lag / fs * 1000, np.max(corr_fft_n), "k^", ms=8,
                    label=f"峰值 lag={peak_lag} 点")
    axes[1, 1].set_title("(d) FFT：重排循环序列后得到线性互相关", fontsize=FS_TITLE)
    axes[1, 1].set_xlabel("lag (ms)", fontsize=FS_LABEL)
    axes[1, 1].set_ylabel("归一化互相关", fontsize=FS_LABEL)
    axes[1, 1].legend(fontsize=FS_SMALL)
    axes[1, 1].grid(ls=":", alpha=0.5)
    axes[1, 1].text(
        0.98, 0.08, f"max |R_td−R_fft| = {max_error:.2e}",
        transform=axes[1, 1].transAxes, ha="right", fontsize=FS_SMALL,
        bbox=dict(fc="white", ec=C_GREEN, boxstyle="round,pad=0.25"))

    fig.suptitle(
        "图33  互相关的两种等价实现（GCC 中 Φ(f)=1 的特例；非 PHAT 加权）",
        fontsize=FS_SUP)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.09, top=0.90,
                        hspace=0.36, wspace=0.25)
    save(fig, "fig33_gcc_two_ways.png")


def main():
    """生成本脚本负责的全部图片。"""
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


if __name__ == "__main__":
    main()
