# -*- coding: utf-8 -*-
"""生成教程插图 29 张（图 1~25、图 33~36；图 26~32 见 make_aec_figures.py）。

用法（仓库根目录）：
    .venv/bin/python scripts/make_figures.py      # 图 1~25、图 33~36 → figures/
"""
from pathlib import Path
import hashlib
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
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

OUT = Path(__file__).parent.parent / "figures"
OUT.mkdir(exist_ok=True)
C_MAIN, C_BLUE, C_RED, C_GREEN, C_ORANGE, C_PURPLE = "#1a1a2e", "#2f6db3", "#c0392b", "#2e8b57", "#e67e22", "#7d3c98"
# 全局字号约定（统一全书插图）
MAX_FIGURE_WIDTH = 9.5
FS_SUP = 14      # 图题 suptitle
FS_TITLE = 12    # 子图标题
FS_LABEL = 11    # 轴标签 / 主要注释
FS_SMALL = 11    # 图例 / 次要注释
FS_TINY = 11     # 最小注释；最终尺寸下不低于 11 pt

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


def source_script_digest():
    """返回本绘图脚本完整字节的 SHA-256，供成品追溯。"""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def figure_png_metadata():
    """返回稳定的 PNG 文本元数据；有意不写入生成时间。"""
    return {
        "SourceScript": "scripts/make_figures.py",
        "SourceScriptDigest": source_script_digest(),
    }


def finalize_figure(fig):
    """统一最终画布与关键文字下限；各图仍应先按内容选择合适布局。"""
    width, height = fig.get_size_inches()
    if width > MAX_FIGURE_WIDTH:
        fig.set_size_inches(MAX_FIGURE_WIDTH, height * MAX_FIGURE_WIDTH / width)
    for ax in fig.axes:
        ax.title.set_fontsize(max(ax.title.get_fontsize(), FS_TITLE))
        ax.xaxis.label.set_fontsize(max(ax.xaxis.label.get_fontsize(), FS_LABEL))
        ax.yaxis.label.set_fontsize(max(ax.yaxis.label.get_fontsize(), FS_LABEL))
        for text_item in ax.texts:
            text_item.set_fontsize(max(text_item.get_fontsize(), FS_LABEL))
        for legend in ([ax.get_legend()] if ax.get_legend() is not None else []):
            for text_item in legend.get_texts():
                text_item.set_fontsize(max(text_item.get_fontsize(), FS_LABEL))
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_fontsize(max(fig._suptitle.get_fontsize(), FS_SUP))
    return fig


def save(fig, name, extra_metadata=None):
    finalize_figure(fig)
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white",
                metadata={**figure_png_metadata(), **(extra_metadata or {})})
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


def plane_wave_steering(microphone_positions, source_direction, frequency,
                        sound_speed=343.0, reference_position=None):
    """按本书约定返回远场导向矢量。

    source_direction 从阵列指向声源，声波传播方向与其相反。采用前向傅里叶核
    exp(-j2*pi*f*t) 时，相对参考点的到达时延为 -(r_m-r_ref)^T u/c，
    所以导向相位为 exp(+j2*pi*f*(r_m-r_ref)^T u/c)。
    """
    positions = np.asarray(microphone_positions, dtype=float)
    direction = np.asarray(source_direction, dtype=float)
    if positions.ndim != 2 or direction.shape != (positions.shape[1],):
        raise ValueError("麦克风坐标须为 M×D，source_direction 须为 D 维")
    norm = np.linalg.norm(direction)
    if norm == 0 or sound_speed <= 0 or frequency < 0:
        raise ValueError("方向向量须非零，声速须为正，频率须非负")
    direction = direction / norm
    if reference_position is None:
        reference_position = np.zeros(positions.shape[1])
    relative = positions - np.asarray(reference_position, dtype=float)
    return np.exp(2j * np.pi * frequency * (relative @ direction) / sound_speed)


def ula_steering(element_positions_wavelengths, angles_deg):
    """正横为 0°、正角朝 +x 的 ULA 导向矢量，返回 M×G。"""
    positions = np.asarray(element_positions_wavelengths, dtype=float).reshape(-1)
    angles = np.atleast_1d(np.asarray(angles_deg, dtype=float))
    return np.exp(2j * np.pi * np.outer(positions, np.sin(np.deg2rad(angles))))


def random_decay_rir(t60, fs, rng, drr_db=0.0, duration_factor=1.1):
    """生成单位直达能量、指定 DRR 的随机指数尾说明模型。

    振幅包络在 t=t60 时下降 60 dB；尾长默认取 1.1*t60，
    因此截断前已低于 -60 dB。随机尾在离散采样后重新归一化，使
    sum(tail**2)=10**(-DRR/10)，从而把尾长 T60 与尾能量 DRR 分开。
    """
    if t60 <= 0 or fs <= 0 or duration_factor <= 1.0:
        raise ValueError("t60、fs 必须为正，duration_factor 必须大于 1")
    length = int(np.ceil(duration_factor * t60 * fs)) + 1
    time = np.arange(length) / fs
    if not np.isfinite(drr_db):
        raise ValueError("drr_db 必须为有限数")
    rir = rng.standard_normal(length) * np.exp(-6.91 * time / t60)
    rir[0] = 0.0
    tail_norm = np.linalg.norm(rir)
    if tail_norm == 0:
        raise RuntimeError("随机混响尾能量为零")
    rir *= 10 ** (-drr_db / 20) / tail_norm
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
    fig, grid = plt.subplots(4, 2, figsize=(9.5, 14.0))
    axes = np.array([[grid[0, 0], grid[0, 1], grid[1, 0], grid[1, 1]],
                     [grid[2, 0], grid[2, 1], grid[3, 0], grid[3, 1]]])
    titles = ["(a) 同一双麦基线：端射方向", "(b) 同一双麦基线：正横方向", "(c) 四麦均匀线阵 ULA",
              "(d) 四麦方阵/平面阵", "(e) 六麦圆阵 UCA + 中心麦",
              "(f) Fibonacci近均匀球阵(32麦)", "(g) 螺旋阵/对数阵", "(h) 分布式/非规则阵"]
    for ax, t in zip(axes.flat, titles):
        ax.set_title(t, fontsize=11)
        ax.set_aspect("equal")
        ax.grid(True, ls=":", alpha=0.5)
        # These panels compare layouts, not measurements. Numeric ticks would
        # imply a shared physical length scale that the panels do not have.
        ax.set_xticks([]); ax.set_yticks([])
    # (a) endfire
    ax = axes[0, 0]
    ax.scatter([-2, 2], [0, 0], s=180, c=C_BLUE, zorder=5, marker="o", edgecolors="k")
    for x_front in [2.8, 3.6, 4.4, 5.2]:
        ax.plot([x_front, x_front], [-1.1, 1.1], color=C_ORANGE, alpha=0.65, lw=1.2)
    ax.add_patch(FancyArrowPatch(
        (5.8, 0), (4.8, 0), arrowstyle="-|>", mutation_scale=14,
        color=C_ORANGE, lw=1.4))
    ax.annotate("竖线为平面波波前\n沿基线向左传播", xy=(2.55, 1.35),
                fontsize=FS_SMALL, color=C_ORANGE)
    ax.set_xlim(-3.5, 6.5); ax.set_ylim(-2.0, 2.5)
    # (b) broadside
    ax = axes[0, 1]
    ax.scatter([-2, 2], [0, 0], s=180, c=C_BLUE, zorder=5, edgecolors="k")
    for y_front in [1.3, 2.3, 3.3, 4.3, 5.3]:
        ax.plot([-3.2, 3.2], [y_front, y_front], color=C_ORANGE, alpha=0.65, lw=1.2)
    ax.add_patch(FancyArrowPatch(
        (0, 6.2), (0, 5.0), arrowstyle="-|>", mutation_scale=14,
        color=C_ORANGE, lw=1.4))
    ax.annotate("横线为平面波波前\n垂直基线向下传播", xy=(1.2, 5.75),
                fontsize=FS_SMALL, color=C_ORANGE)
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
    ax.text(0, -1.35, "红★为中心麦", ha="center", fontsize=FS_SMALL, color=C_RED)
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
    # (f) spherical
    ax = axes[1, 1]
    ax.remove()
    ax = fig.add_subplot(4, 2, 6, projection="3d")
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
    ax.scatter(r_g * np.cos(th_g), r_g * np.sin(th_g),
               s=np.linspace(48, 120, len(k)), c=C_BLUE, zorder=5, edgecolors="k")
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
    fig.text(0.5, 0.01, "(a)(b) 平行线为波前，箭头为传播方向；各平面子图只比较几何形态，坐标不代表统一的米制尺度。",
             ha="center", fontsize=FS_SMALL, color="0.3")
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    save(fig, "fig08_geometries.png")


# ----------------------------------------------------------------------
# 图3 近场球面波 vs 远场平面波
# ----------------------------------------------------------------------
def fig_near_far_field():
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.5))
    ax = axes[0]
    src = (0, 4.0)
    ax.scatter(*src, marker="*", s=400, c=C_RED, zorder=6, edgecolors="k")
    ax.annotate("近场点声源", xy=src, xytext=(0.3, 4.35), fontsize=FS_LABEL, color=C_RED)
    th = np.linspace(np.pi + 0.35, 2 * np.pi - 0.35, 60)
    for rr in [0.8, 1.3, 1.8, 2.3]:
        ax.plot(src[0] + rr * np.cos(th), src[1] + rr * np.sin(th), color=C_BLUE, alpha=0.55)
    ax.scatter([-0.8, 0, 0.8], [0, 0, 0], s=150, c=C_GREEN, zorder=6, edgecolors="k")
    ax.annotate("幅度差 + 相位差\n(球面波)", xy=(1.6, 1.2), fontsize=FS_LABEL, color=C_GREEN)
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
# 图16：三条曲线使用两种阵距，图内直接标明比较条件。
# ----------------------------------------------------------------------
def fig_beampatterns():
    M, d = 8, 0.5
    th = np.linspace(-90, 90, 721)
    thetas_rad = np.deg2rad(th)
    m = np.arange(M) - (M - 1) / 2
    a0 = ula_steering(d * m, 0)[:, 0]                    # 期望方向 0°
    # DSB
    w_ds = a0 / M
    # MVDR: 白噪声 + 20°方向强干扰(避开DSB自然零陷位置)
    a_int = ula_steering(d * m, 20)
    R = 10 * (a_int @ a_int.conj().T) + 1.0 * np.eye(M)
    w_mvdr, _ = distortionless_weights(R, a0)
    # 超指向: 小间距 d_sd=0.2λ, 各向同性(弥散)噪声协方差 Γ=sinc(2d|i-j|)
    d_sd = 0.2
    a0_sd = ula_steering(d_sd * m, 0)[:, 0]
    G = np.sinc(2 * d_sd * np.abs(np.subtract.outer(m, m)))
    w_sd, _ = distortionless_weights(G, a0_sd, diagonal_loading=1e-6)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.4), subplot_kw=dict(polar=True))
    fig.subplots_adjust(wspace=0.12)
    curves = [("DSB：8 麦，d=0.5λ", w_ds, d, C_BLUE, "-"),
              ("MVDR：8 麦，d=0.5λ", w_mvdr, d, C_RED, "--"),
              (r"超指向：8 麦，d=0.2λ", w_sd, d_sd, C_GREEN, ":")]
    for ax, (title, ymax) in zip(axes, [("线性幅度", None), ("dB 刻度", 40)]):
        for name, w, dd, c, ls in curves:
            A = ula_steering(dd * m, th)
            b = np.abs(w.conj() @ A)
            target = ula_steering(dd * m, 0)[:, 0]
            target_gain = np.abs(w.conj() @ target)
            if "dB" in title:
                b = 20 * np.log10(np.maximum(b / target_gain, 1e-4))
                b = np.maximum(b, -40) + 40      # 平移到 0~40 便于径向刻度
            ax.plot(thetas_rad, b, color=c, ls=ls, lw=2.0, label=name)
        ax.set_theta_zero_location("N")
        ax.set_thetamin(-90); ax.set_thetamax(90)
        if "dB" in title:
            ax.set_rlim(0, 40)
            ax.set_rgrids([0, 20, 40], ["−40", "−20", "0 dB"],
                          angle=55, fontsize=FS_SMALL)
        else:
            # 半圆极坐标图减少径向刻度并错开默认标签位置，避免缩放到网页宽度后
            # 与角度刻度、图例挤在一起。
            radial_max = ax.get_rmax()
            ax.set_rgrids([0.5, 1.0], ["0.5", "1.0"],
                          angle=55, fontsize=FS_SMALL)
            ax.set_rmax(radial_max)
        ax.set_title(title, fontsize=13, pad=18)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), fontsize=FS_SMALL + 0.5, framealpha=0.95)
    # 在 dB 图上用箭头标出 20° 处 MVDR 零陷（标签外移至图外空白，引线指回零陷坑）
    ax_db = axes[1]
    ax_db.annotate("MVDR：20° 方向的零陷", xy=(np.deg2rad(20), 2),
                   xytext=(np.deg2rad(76), 37.5), fontsize=FS_SMALL, color=C_RED,
                   bbox=dict(fc="white", ec=C_RED, lw=0.7, alpha=0.9, boxstyle="round,pad=0.25"),
                   arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.5,
                                   connectionstyle="arc3,rad=-0.1"))
    fig.suptitle("图16  8 麦线阵的三种波束：两种阵距，不能作为同条件性能排名\n"
                 r"DSB/MVDR：$d=0.5\lambda$；超指向：$d=0.2\lambda$；目标方向 0°、响应均为 1；MVDR 干扰 20°",
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


def capon_spectrum(covariance, steering_vectors, relative_loading=1e-6):
    """计算带相对对角加载的 Capon 谱，并返回实际加载量。

    加载量按 ``relative_loading * trace(R) / M`` 定义，因此协方差整体缩放时，
    归一化空间谱不变。这里用线性方程求解，避免显式求逆。
    """
    covariance = np.asarray(covariance)
    steering_vectors = np.asarray(steering_vectors)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance 必须是方阵")
    if (steering_vectors.ndim != 2
            or steering_vectors.shape[0] != covariance.shape[0]):
        raise ValueError("steering_vectors 必须为 M×G，且 M 与 covariance 一致")
    if not np.isfinite(relative_loading) or relative_loading < 0:
        raise ValueError("relative_loading 必须是非负有限数")
    scale = float(np.trace(covariance).real / covariance.shape[0])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("covariance 的平均对角功率必须为正且有限")
    loading = relative_loading * scale
    loaded = covariance + loading * np.eye(covariance.shape[0])
    try:
        solved = np.linalg.solve(loaded, steering_vectors)
    except np.linalg.LinAlgError as error:
        raise np.linalg.LinAlgError(
            "Capon 协方差奇异；请使用正的 relative_loading 或改进协方差估计") from error
    denominator = np.einsum(
        "mg,mg->g", steering_vectors.conj(), solved).real
    if np.any(denominator <= 0) or not np.all(np.isfinite(denominator)):
        raise ValueError("Capon 分母必须为正且有限")
    return 1.0 / denominator, loading


def fig_doa_spectrum():
    rng = np.random.default_rng(FIGURE_SEEDS["doa_spectrum"])
    M, d = 8, 0.5
    srcs = [-20, 30]
    snr_db, snaps = 20, 400
    m = np.arange(M) - (M - 1) / 2
    A = ula_steering(d * m, srcs)
    # 每个源为单位复功率；snr_db 定义为“每阵元总信号功率 / 噪声功率”。
    sig = (rng.normal(size=(2, snaps)) + 1j * rng.normal(size=(2, snaps))) / np.sqrt(2)
    signal_power_per_sensor = len(srcs)
    noise_power = signal_power_per_sensor / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        rng.normal(size=(M, snaps)) + 1j * rng.normal(size=(M, snaps)))
    X = A @ sig + noise
    R = X @ X.conj().T / snaps
    th = np.linspace(-90, 90, 1801)
    Av = ula_steering(d * m, th)  # M x grid
    bart = bartlett_spectrum(R, Av)
    bart /= bart.max()
    capon, _ = capon_spectrum(R, Av, relative_loading=1e-6)
    capon /= capon.max()
    evals, evecs = np.linalg.eigh(R)
    En = evecs[:, :M - 2]
    music = 1 / np.sum(np.abs(En.conj().T @ Av) ** 2, axis=0)
    music_db = 10 * np.log10(music / music.max())
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 9.0))
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
        axes[1].annotate(f"真源{s}°", xy=(s, -2), xytext=(s - 6, -8), fontsize=FS_SMALL,
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
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 12.0))
    # (a) 两路信号：放大到亚毫秒窗口，时移肉眼可辨
    ax = axes[0]
    t0 = 2.0e-3
    m = (np.arange(n) / fs >= t0) & (np.arange(n) / fs < t0 + 5e-4)
    tt = np.arange(n)[m] / fs * 1e3
    display_scale = max(np.max(np.abs(x1[m])), np.max(np.abs(x2[m])))
    ax.plot(tt, x2[m] / display_scale, "o-", color=C_BLUE, ms=3.5, lw=1.2, label="麦克风2（先到）")
    ax.plot(tt, x1[m] / display_scale, "s-", color=C_RED, ms=3.5, lw=1.2, label="麦克风1（晚到 τ）")
    ax.annotate("", xy=(2.20, 0.62), xytext=(2.20 - tau_true * 1e3, 0.62),
                arrowprops=dict(arrowstyle="<->", color="k", lw=2.0))
    ax.text(2.20 - tau_true * 1e3 / 2, 0.68, "τ≈58 μs", ha="center", fontsize=FS_SMALL)
    ax.set_ylim(-0.8, 0.95)
    ax.set_title("(a) 两路信号（放大 0.5 ms 窗口）", fontsize=FS_TITLE)
    ax.set_xlabel("时间 (ms)", fontsize=FS_LABEL)
    ax.set_ylabel("归一化幅度", fontsize=FS_LABEL)
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
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
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


def candidate_distance_segments(microphones, candidate):
    """返回每个麦克风到同一候选位置的线段端点，形状为 M×2×2。"""
    microphones = np.asarray(microphones, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if microphones.ndim != 2 or microphones.shape[1] != 2:
        raise ValueError("microphones 必须为 M×2 坐标")
    if candidate.shape != (2,):
        raise ValueError("candidate 必须是二维坐标")
    return np.stack(
        (microphones, np.broadcast_to(candidate, microphones.shape)), axis=1)


def fig_srp_grid():
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.5))
    ax = axes[0]
    mics = np.array([[0, 0], [4, 0], [0, 3], [4, 3]])
    ax.scatter(mics[:, 0], mics[:, 1], s=160, c=C_BLUE, zorder=6, edgecolors="k", label="麦克风")
    src = (5.5, 3.8)
    ax.scatter(*src, marker="*", s=400, c=C_RED, zorder=6, edgecolors="k", label="真实声源")
    for gx in np.linspace(0.5, 6.5, 8):
        for gy in np.linspace(0.5, 4.5, 6):
            ax.plot(gx, gy, ".", color="gray", ms=3, alpha=0.6)
    candidate = np.array([2.214, 2.1])
    ax.plot(*candidate, marker="s", ms=8, mfc="none", mec=C_GREEN, mew=2,
            label="候选位置 r")
    for segment in candidate_distance_segments(mics, candidate):
        ax.plot(segment[:, 0], segment[:, 1], ls=":", color=C_ORANGE,
                alpha=0.6, lw=1)
    ax.legend(fontsize=FS_SMALL); ax.set_title("(a) 空间网格 + 各麦到候选点的距离线", fontsize=11)
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
    ax.set_title("(b) SRP-PHAT 累积分数与峰值位置", fontsize=FS_TITLE)
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
    with np.errstate(over="ignore"):
        total = weights.sum()
    if (weights.ndim != 1 or weights.size == 0
            or not np.all(np.isfinite(weights)) or np.any(weights < 0)
            or not np.isfinite(total) or total <= 0):
        raise ValueError("weights 必须是一维、非空、有限、非负且总和为正")
    weights = weights / total
    positions = (rng.random() + np.arange(len(weights))) / len(weights)
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0  # 避免浮点舍入使 searchsorted 返回越界索引。
    return np.searchsorted(cumulative, positions, side="right")


def reflect_interval(positions, velocities, bounds):
    """把位置镜面反射回有限区间，并在奇数次碰壁时反转速度。"""
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    if positions.shape != velocities.shape:
        raise ValueError("positions 与 velocities 必须同形")
    lower, upper = map(float, bounds)
    if not lower < upper:
        raise ValueError("边界必须满足 lower < upper")
    width = upper - lower
    phase = np.mod(positions - lower, 2.0 * width)
    reverse = phase > width
    reflected = np.where(reverse, upper - (phase - width), lower + phase)
    reflected_velocity = np.where(reverse, -velocities, velocities)
    hit = (positions < lower) | (positions > upper)
    return reflected, reflected_velocity, hit


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
    if np.any(np.isinf(observations)):
        raise ValueError("observations 只允许有限值或表示缺测的 NaN")
    if not isinstance(n_particles, (int, np.integer)) or n_particles <= 0:
        raise ValueError("n_particles 必须是正整数")
    for name, value in (("process_std", process_std),
                        ("velocity_process_std", velocity_process_std)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} 必须是非负有限数")
    if not np.isfinite(observation_std) or observation_std <= 0:
        raise ValueError("observation_std 必须是正有限数")
    if not np.isfinite(resample_fraction) or not 0 <= resample_fraction <= 1:
        raise ValueError("resample_fraction 必须在 [0, 1] 内")
    if not np.isfinite(clutter_probability) or not 0 <= clutter_probability < 1:
        raise ValueError("clutter_probability 必须在 [0, 1) 内")
    lower, upper = map(float, angle_bounds)
    if not np.isfinite(lower) or not np.isfinite(upper) or not lower < upper:
        raise ValueError("angle_bounds 必须是递增的有限边界")
    # 未知初始方向时使用独立先验，不能偷看首个未来有效观测。
    velocities = rng.normal(0.0, 1.0, n_particles)
    angles = rng.uniform(lower, upper, n_particles)
    weights = np.full(n_particles, 1.0 / n_particles)
    estimates = np.empty(len(observations))
    neff_history = np.empty(len(observations))
    resampled = np.zeros(len(observations), dtype=bool)
    reflection_fraction = np.zeros(len(observations), dtype=float)
    for k, observation in enumerate(observations):
        velocities += rng.normal(0.0, velocity_process_std, n_particles)
        proposed = angles + velocities + rng.normal(0.0, process_std, n_particles)
        angles, velocities, hit_bound = reflect_interval(
            proposed, velocities, angle_bounds)
        reflection_fraction[k] = np.mean(hit_bound)
        if np.isfinite(observation):
            if clutter_probability == 0:
                # 只需相对似然。先减去最小绝对残差再形成平方差，避免极远观测
                # 使所有普通高斯密度同时下溢为零。
                absolute_residual = np.abs(angles - observation)
                closest = np.min(absolute_residual)
                with np.errstate(over="ignore", invalid="ignore"):
                    squared_difference = ((absolute_residual - closest)
                                          * (absolute_residual + closest)
                                          / observation_std ** 2)
                likelihood_log = -0.5 * squared_difference
            else:
                standardized = (angles - observation) / observation_std
                with np.errstate(over="ignore"):
                    gaussian_log = (-0.5 * standardized ** 2
                                    - np.log(observation_std * np.sqrt(2 * np.pi)))
                target_log = np.log1p(-clutter_probability) + gaussian_log
                clutter_log = (np.log(clutter_probability)
                               - np.log(upper - lower))
                likelihood_log = np.logaddexp(target_log, clutter_log)
            with np.errstate(divide="ignore", under="ignore"):
                posterior_log = np.log(weights) + likelihood_log
            posterior_log -= np.max(posterior_log)
            weights = np.exp(posterior_log)
            weight_sum = weights.sum()
            if not np.isfinite(weight_sum) or weight_sum <= 0:
                raise FloatingPointError("粒子权重归一化失败")
            weights /= weight_sum
        estimates[k] = np.sum(weights * angles)
        neff_history[k] = 1.0 / np.sum(weights ** 2)
        if (np.isfinite(observation)
                and neff_history[k] < resample_fraction * n_particles):
            indices = systematic_resample(weights, rng)
            angles = angles[indices]
            velocities = velocities[indices]
            weights.fill(1.0 / n_particles)
            resampled[k] = True
    return estimates, neff_history, resampled, reflection_fraction


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
    est, n_eff, resampled, reflection_fraction = particle_filter_doa(
        obs, rng, n_particles=N)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.4))
    axes[0].plot(t, obs, ".", color="gray", ms=4, label="DOA观测（含噪声和均匀杂波）")
    axes[0].plot(t, true, color="k", lw=2, label="真实轨迹")
    axes[0].plot(t, est, color=C_RED, lw=1.6, label="混合似然粒子滤波估计")
    axes[0].plot(t[resampled], est[resampled], "|", color=C_ORANGE, ms=7,
                 label=r"重采样 ($N_\mathrm{eff}<N/2$)")
    boundary_active = reflection_fraction >= 0.05
    axes[0].plot(t[boundary_active], est[boundary_active], "x", color=C_PURPLE,
                 ms=5, label="反射粒子比例≥5%")
    axes[0].axvspan(48, 57, color=C_PURPLE, alpha=0.12, label="连续 10 帧缺测")
    axes[0].set_xlabel("帧"); axes[0].set_ylabel("方位角 (°)")
    axes[0].legend(fontsize=FS_SMALL + 1, loc="lower left", framealpha=0.95); axes[0].grid(ls=":", alpha=0.5)
    axes[0].set_title("(a) 单说话人追踪：匀速状态模型在缺测段只做预测\n"
                      "（200 粒子；高斯目标+均匀杂波似然；0～120°镜面反射边界）", fontsize=FS_TITLE)
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
    ax.set_title("(b) 两目标 PHD 强度示意：曲线积分=2\n（角度约定：−90°～+90°）", fontsize=FS_TITLE)
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
        a0 = plane_wave_steering(mic_xy, look_direction, f, sound_speed)
        # 三维各向同性弥散场：Γ_ij=sinc(2 f d_ij/c)，距离取圆阵二维坐标的欧氏弦长。
        G = np.sinc(2 * f * pair_distances / sound_speed)
        w_ds = a0 / M
        w_sd, _ = distortionless_weights(G, a0, diagonal_loading=1e-6)
        wng_dsb.append(10 * np.log10(1 / np.sum(np.abs(w_ds) ** 2)))
        di_dsb.append(10 * np.log10(1 / np.real(w_ds.conj() @ G @ w_ds)))
        wng_sd.append(10 * np.log10(1 / np.sum(np.abs(w_sd) ** 2)))
        di_sd.append(10 * np.log10(1 / np.real(w_sd.conj() @ G @ w_sd)))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.8))
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
def pipeline_dependency_spec():
    """返回图23的信号/统计量依赖，供绘图与回归测试共用。"""
    return {
        "audio": {
            ("capture", "aec"), ("aec", "wpe"), ("wpe", "bf"),
            ("bf", "ns"), ("ns", "backend"),
            ("wpe", "neural_separator"), ("neural_separator", "ns"),
            ("far_end", "render"), ("render", "speaker"),
            ("speaker", "capture"), ("render", "render_tap"),
            ("render_tap", "aec"),
        },
        "information": {
            ("wpe", "ssl"), ("ssl", "tracking"), ("tracking", "bf"),
            ("diarization", "gss_mask"), ("wpe", "gss_mask"),
            ("gss_mask", "scm"), ("wpe", "scm"), ("scm", "bf"),
        },
        "control": {
            ("activity_control", "aec"), ("activity_control", "wpe"),
            ("activity_control", "tracking"), ("activity_control", "bf"),
            ("activity_control", "backend"),
        },
    }


def fig_pipeline():
    # 六层布局：播放、GSS、定位、控制、音频主链、神经旁路。
    fig, ax = plt.subplots(figsize=(9.5, 12.5))
    ax.axis("off"); ax.set_xlim(0, 16.2); ax.set_ylim(0, 11.8)
    box_w, box_h = 2.0, 0.95
    positions = {
        "capture": (0.20, 2.10), "aec": (2.60, 2.10), "wpe": (5.00, 2.10),
        "bf": (8.80, 2.10), "ns": (11.20, 2.10), "backend": (13.60, 2.10),
        "neural_separator": (7.00, 0.20),
        "ssl": (8.80, 4.75), "tracking": (11.40, 4.75),
        "diarization": (0.50, 6.55), "gss_mask": (3.20, 6.55),
        "scm": (5.90, 6.55),
        "far_end": (0.20, 10.00), "render": (2.60, 10.00),
        "render_tap": (2.60, 8.25), "speaker": (5.00, 10.00),
    }
    labels = {
        "capture": "多通道采集\n同步/标定", "aec": "AEC\n回声消除",
        "wpe": "WPE\n去混响", "bf": "波束形成\nMVDR / GEV",
        "ns": "NS + AGC\n逐流可选", "backend": "KWS / ASR\n逐流或选流",
        "neural_separator": "神经/CSS\n分离旁路",
        "ssl": "声源定位\nSSL / DOA", "tracking": "轨迹预测\n状态+协方差",
        "diarization": "说话人分割\n活动标注", "gss_mask": "GSS / cACG\n掩码",
        "scm": "目标/干扰\nSCM",
        "far_end": "远端播放", "render": "播放处理\n均衡/音量",
        "render_tap": "播放参考\n供回声消除", "speaker": "数模转换\n功放/扬声器",
    }
    colors = {
        "capture": "#dbe9f6", "aec": "#dbe9f6", "wpe": "#dbe9f6",
        "bf": "#e8f6db", "ns": "#e8f6db", "backend": "#f6dbdb",
        "neural_separator": "#e8f6db", "ssl": "#f6e5db", "tracking": "#f6e5db",
        "diarization": "#f6e5db", "gss_mask": "#f6e5db", "scm": "#f6e5db",
        "far_end": "#fff2cc", "render": "#fff2cc", "render_tap": "#fff2cc",
        "speaker": "#fff2cc",
    }
    node_rectangles, node_texts = {}, {}
    for name, (x, y) in positions.items():
        rectangle = plt.Rectangle((x, y), box_w, box_h, fc=colors[name],
                                  ec="k", lw=1.1)
        text_artist = ax.text(x + box_w / 2, y + box_h / 2, labels[name],
                              ha="center", va="center", fontsize=FS_SMALL)
        ax.add_patch(rectangle)
        node_rectangles[name] = rectangle
        node_texts[name] = text_artist

    def anchor(name, side):
        x, y = positions[name]
        return {
            "left": (x, y + box_h / 2), "right": (x + box_w, y + box_h / 2),
            "top": (x + box_w / 2, y + box_h), "bottom": (x + box_w / 2, y),
        }[side]

    dependencies = pipeline_dependency_spec()
    drawn_edges = {category: set() for category in dependencies}

    def edge(source, target, category, color="k", linestyle="-", source_side="right",
             target_side="left", rad=0.0, lw=1.5, via=None):
        points = [anchor(source, source_side), *(via or []), anchor(target, target_side)]
        for index, (start, stop) in enumerate(zip(points[:-1], points[1:])):
            ax.add_patch(FancyArrowPatch(
                start, stop, arrowstyle="-|>" if index == len(points) - 2 else "-",
                mutation_scale=14, color=color, lw=lw, ls=linestyle,
                connectionstyle=f"arc3,rad={rad if via is None else 0.0}"))
        drawn_edges[category].add((source, target))

    audio_routes = {
        ("render", "render_tap"): dict(color=C_RED, linestyle="--", source_side="bottom", target_side="top"),
        ("render_tap", "aec"): dict(
            color=C_RED, linestyle="--", source_side="bottom", target_side="top",
            via=[(2.80, 7.85), (2.80, 3.45), (3.60, 3.45)], lw=1.8),
        ("speaker", "capture"): dict(color=C_RED, linestyle=":", source_side="right", target_side="top", rad=-0.47, lw=1.8),
        ("wpe", "neural_separator"): dict(color=C_GREEN, linestyle="-.", source_side="bottom", target_side="left", rad=0.10),
        ("neural_separator", "ns"): dict(color=C_GREEN, linestyle="-.", source_side="right", target_side="bottom", rad=-0.18),
    }
    for source, target in sorted(dependencies["audio"]):
        edge(source, target, "audio", **audio_routes.get((source, target), {}))
    ax.text(1.95, 8.05, "处理后参考", color=C_RED, fontsize=FS_SMALL, ha="center")
    ax.text(10.7, 10.70, "声学回路：扬声器 → 房间/机壳 → 麦克风",
            color=C_RED, fontsize=FS_SMALL, ha="center")

    information_routes = {
        ("wpe", "ssl"): dict(color=C_ORANGE, linestyle="-.", source_side="top", target_side="left"),
        ("ssl", "tracking"): dict(color=C_ORANGE, linestyle="-."),
        ("tracking", "bf"): dict(color=C_RED, linestyle="-.", source_side="bottom", target_side="top"),
        ("diarization", "gss_mask"): dict(color=C_PURPLE, linestyle=":"),
        ("wpe", "gss_mask"): dict(color=C_PURPLE, linestyle=":", source_side="top", target_side="bottom", rad=-0.18),
        ("gss_mask", "scm"): dict(color=C_PURPLE, linestyle=":"),
        ("wpe", "scm"): dict(color=C_PURPLE, linestyle=":", source_side="top", target_side="bottom", rad=-0.27),
        ("scm", "bf"): dict(color=C_PURPLE, linestyle=":", source_side="bottom", target_side="top"),
    }
    for source, target in sorted(dependencies["information"]):
        edge(source, target, "information", **information_routes[(source, target)])
    ax.text(10.60, 4.47, "带时间戳方向", color=C_RED, fontsize=FS_SMALL, ha="center")

    ax.text(7.15, 7.82, "活动标注→掩码；WPE 未方向归一 STFT→SCM",
            color=C_PURPLE, fontsize=FS_SMALL, ha="center")
    ax.text(8.75, 0.02, "神经/CSS 旁路解析波束；多路输出分别处理，网络已增强时可跳过 NS",
            color=C_GREEN, fontsize=FS_SMALL, ha="center")

    control_y = 3.85
    ax.plot([anchor("aec", "top")[0], anchor("backend", "top")[0]],
            [control_y, control_y],
            ls="--", color=C_ORANGE, lw=1.3)
    ax.text(12.45, 4.08, "活动控制：VAD / 远端 / 双讲",
            color=C_ORANGE, fontsize=FS_SMALL, ha="center")
    for target in ("aec", "wpe", "tracking", "bf", "backend"):
        target_side = "bottom" if target == "tracking" else "top"
        x = anchor(target, target_side)[0]
        ax.add_patch(FancyArrowPatch((x, control_y), anchor(target, target_side),
                                    arrowstyle="-|>", mutation_scale=10,
                                    color=C_ORANGE, lw=1.0, ls="--"))
        drawn_edges["control"].add(("activity_control", target))
    ax.text(0.20, 11.48,
            "实线=音频；红/橙=参考、方向或控制；紫点线=GSS统计；绿点划线=神经旁路。",
            fontsize=FS_SMALL, color="dimgray", ha="left")
    ax.set_title("图23  远场语音前端：播放参考、音频主链与三条条件支路",
                 fontsize=FS_SUP)
    fig._pipeline_edges = {key: frozenset(value) for key, value in drawn_edges.items()}
    fig._pipeline_node_rectangles = node_rectangles
    fig._pipeline_node_texts = node_texts
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
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.2))
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
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 12.0))
    ax = axes[0]
    ax.plot(t * 1000, rir, color=C_BLUE, lw=0.7)
    ax.axvspan(0, 12, color="0.8", alpha=0.12)
    ax.axvspan(12, 13, color=C_GREEN, alpha=0.24)
    ax.axvspan(13, 92, color=C_ORANGE, alpha=0.15)
    ax.axvspan(92, 700, color=C_RED, alpha=0.11)
    ax.axvline(12, color="gray", ls="--", lw=0.9, alpha=0.8)
    ax.axvline(92, color="gray", ls="--", lw=0.9, alpha=0.8)
    ax.set_ylim(-2.6, 3.1)
    ax.annotate("直达声\n（12 ms 到达）", xy=(12, 1.05), xytext=(16, 2.72), fontsize=FS_SMALL, color=C_GREEN,
                arrowprops=dict(arrowstyle="->", color=C_GREEN))
    ax.annotate("早期反射\n（直达后 0–80 ms）", xy=(55, 0.7), xytext=(105, 2.15), fontsize=FS_SMALL,
                color=C_ORANGE, arrowprops=dict(arrowstyle="->", color=C_ORANGE))
    ax.annotate("晚期随机尾\n（本图指数衰减模型）", xy=(260, 0.25), xytext=(330, 1.5), fontsize=FS_SMALL,
                color=C_RED, arrowprops=dict(arrowstyle="->", color=C_RED))
    ax.set_xlabel("时间 (ms)"); ax.set_ylabel("幅度（任意单位）")
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
                    xy=(realized_ms, -60), xytext=(205, -35), fontsize=FS_SMALL,
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
                xytext=(1.7, 14), fontsize=FS_SMALL, arrowprops=dict(arrowstyle="->", color="k"))
    ax.fill_between(d, direct, reverb, where=direct > reverb, color=C_GREEN, alpha=0.08)
    ax.text(0.27, -19.5, "直达占优\n(DRR>0)", fontsize=FS_LABEL, color=C_GREEN)
    ax.text(3.6, 6, "混响占优\n(DRR<0)", fontsize=FS_LABEL, color=C_RED)
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
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.5))
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
    ax.annotate("一个格子：一个时频点 $(k,\\ell)$\n固定此格，取 $M$ 路复数谱\n组成一个快拍，见 (c)",
                xy=(px, py), xytext=(350, 5.3), fontsize=FS_LABEL, color="w",
                arrowprops=dict(arrowstyle="->", color="w", lw=1.5))
    ax.set_ylim(0, 8); ax.set_xlabel("时间 (ms)", fontsize=FS_LABEL)
    ax.set_ylabel("频率 (kHz)", fontsize=FS_LABEL)
    ax.set_title("(b) STFT 语谱图（单麦克风；格子为放大标出）", fontsize=FS_TITLE)
    ax = axes[1, 0]
    M = 8
    R = np.zeros((M, M), dtype=complex)
    srcs = [-20, 30]
    m = np.arange(M) - (M - 1) / 2
    A = ula_steering(0.5 * m, srcs)
    R = A @ np.diag([10, 5]) @ A.conj().T + np.eye(M)
    im = ax.imshow(np.abs(R), cmap="Blues")
    ax.set_xticks(range(M)); ax.set_yticks(range(M))
    ax.set_xticklabels([f"麦{i+1}" for i in range(M)], fontsize=FS_SMALL)
    ax.set_yticklabels([f"麦{i+1}" for i in range(M)], fontsize=FS_SMALL)
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
                xytext=(3.0, 13), fontsize=FS_LABEL, color=C_RED,
                arrowprops=dict(arrowstyle="->", color=C_RED))
    ax.annotate("6个小特征值≈噪声底(0 dB)\n→ 噪声子空间", xy=(6.5, 0), xytext=(3.2, 4.5),
                fontsize=FS_LABEL, color=C_BLUE, arrowprops=dict(arrowstyle="->", color=C_BLUE))
    ax.set_xlabel("特征值序号"); ax.set_ylabel("特征值 (dB)")
    ax.set_ylim(-4, 21)
    ax.set_title("(d) (c) 的特征值谱：已知白噪声模型下的示意", fontsize=11)
    ax.grid(ls=":", alpha=0.4)
    fig.suptitle("图6  从波形到协方差矩阵：阵列算法处理的数据形态（模拟）", fontsize=FS_SUP)
    fig.tight_layout()
    save(fig, "fig06_stft_cov.png")


# ----------------------------------------------------------------------
# 图10 稀疏阵列：ULA vs 嵌套阵/互质阵 与 差分协同阵
# ----------------------------------------------------------------------
def fig_sparse_array():
    ula = np.arange(6)
    nested = np.array([0, 1, 2, 3, 7, 11])
    coprime = np.array([0, 3, 4, 6, 8, 9])
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.2))
    ax = axes[0]
    for y, (name, pos, c) in enumerate([("均匀线阵 ULA", ula, C_BLUE),
                                        ("嵌套阵 Nested", nested, C_RED),
                                        ("互质阵 Coprime", coprime, C_GREEN)]):
        ax.scatter(pos, np.full_like(pos, 2 - y, dtype=float), s=170, c=c, zorder=6, edgecolors="k")
        ax.text(-0.7, 2 - y, name, fontsize=FS_LABEL, va="center", ha="right")
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
        ax.text(-11.5, 2 - y + 0.32, name, fontsize=FS_LABEL, va="center", color=c)
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
    fig = plt.figure(figsize=(9.5, 5.2))
    ax = fig.add_subplot(1, 2, 1)
    ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 6)
    def box(x, y, w, h, text, fc="#dbe9f6", fs=9.5):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec="k", lw=1.2, zorder=3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=4)
    def arrow(x1, y1, x2, y2, text="", c="k", dy=0.15):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color=c, lw=1.4, zorder=2))
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, text, fontsize=FS_LABEL, ha="center", color=c,
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
    ax.text(1.0, 5, "双讲期冻结系数\n残差含近端语音，不计算 ERLE", fontsize=FS_LABEL,
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


def stft_analysis(signal, window, hop):
    """按完整窗分帧并返回 F×T 单边 STFT；恰好贴住末尾的完整帧会被保留。"""
    signal = np.asarray(signal)
    window = np.asarray(window)
    if signal.ndim != 1 or window.ndim != 1 or window.size == 0:
        raise ValueError("signal 和 window 必须是一维数组，且 window 非空")
    if not isinstance(hop, (int, np.integer)) or hop < 1:
        raise ValueError("hop 必须为正整数")
    n_fft = window.size
    if signal.size < n_fft:
        return np.empty((n_fft // 2 + 1, 0), dtype=complex)
    frames = [signal[start:start + n_fft] * window
              for start in range(0, signal.size - n_fft + 1, hop)]
    return np.asarray([np.fft.rfft(frame) for frame in frames]).T


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
    Yc, Yr = stft_analysis(clean, win, hop), stft_analysis(rev, win, hop)
    # RIR 的直达脉冲位于 direct_index。显示仍使用未移位的干净语音，
    # 但失真指标和活动/静音帧必须先把参考对齐到同一传播时刻。
    clean_aligned = causal_delay(clean, direct_index)
    Yc_aligned = stft_analysis(clean_aligned, win, hop)
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
    # 保持三谱图的纵横比，同时缩小源画布；A4 等宽嵌入时字号随之增大。
    fig = plt.figure(figsize=(8.2, 11.65), layout="constrained")
    grid = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 0.12])
    axes = np.array([fig.add_subplot(grid[index, 0]) for index in range(3)])
    footer = fig.add_subplot(grid[3, 0])
    footer.axis("off")
    tms = np.arange(Yc_aligned.shape[1]) * hop / fs * 1000
    fk = np.fft.rfftfreq(n_fft, 1 / fs) / 1000
    reference_amplitude = max(np.abs(S).max() for S in (Yc_aligned, Yr, Xd))
    mesh = None
    for ax, (S, title) in zip(axes, [(Yc_aligned, "(a) 按 10 ms 传播时延对齐的干净参考"),
                                     (Yr, "(b) 混响语音：能量沿时间拖尾、边界变模糊"),
                                     (Xd, "(c) WPE 去混响后：拖尾被抑制")]):
        # 10log10(|S|^2/ref^2) 与 20log10(|S|/ref) 等价；三图共用参考值。
        Sdb = 10 * np.log10((np.abs(S) ** 2 + 1e-16) / reference_amplitude ** 2)
        mesh = ax.pcolormesh(tms, fk, Sdb, cmap="viridis", shading="auto",
                             vmin=-55, vmax=0)
        ax.set_ylim(0, 2.5); ax.set_xlabel("时间 (ms)", fontsize=FS_LABEL)
        ax.set_ylabel("频率 (kHz)", fontsize=FS_LABEL)
        ax.set_title(title, fontsize=FS_TITLE)
    axes[0].text(0.98, 0.94, "三图的高频暗区：合成信号能量很低",
                 transform=axes[0].transAxes, fontsize=FS_SMALL, color="white",
                 ha="right", va="top")
    axes[1].text(0.03, 0.05,
                 f"静音帧能量占比 {rev_quiet_db:.1f} dB\n对齐参考活跃帧 NMSE {rev_nmse_db:.1f} dB",
                 transform=axes[1].transAxes, fontsize=FS_TINY, color="white", va="bottom",
                 bbox=dict(fc="black", ec="none", alpha=0.55, pad=2))
    axes[2].text(0.03, 0.05,
                 f"静音帧能量占比 {wpe_quiet_db:.1f} dB\n对齐参考活跃帧 NMSE {wpe_nmse_db:.1f} dB",
                 transform=axes[2].transAxes, fontsize=FS_TINY, color="white", va="bottom",
                 bbox=dict(fc="black", ec="none", alpha=0.55, pad=2))
    fig.suptitle("图21  WPE 去混响效果（本书单通道仿真）", fontsize=FS_SUP)
    footer.text(
        0.5, 0.55,
        "配置：16 kHz、1.4 s、种子 21001；RIR 直达延迟 10 ms、$T_{60}$=0.6 s、DRR=6 dB；\n"
        "Hann 512 点、帧移 128 点；WPE $\\Delta$=6、$K$=10、3 次迭代；三图共用对齐时间与谱幅参考。",
        ha="center", va="center", fontsize=FS_SMALL)
    cb = fig.colorbar(mesh, ax=axes, shrink=0.85, pad=0.015, label="相对谱能量 (dB)")
    cb.ax.tick_params(labelsize=FS_SMALL + 1)
    cb.set_label("相对谱能量 (dB)", fontsize=FS_LABEL + 2)
    cb.outline.set_linewidth(1.3)
    fig._footer_axis = footer
    save(fig, "fig21_wpe.png")


# ----------------------------------------------------------------------
# 图1 人类双耳定位线索：ITD / ILD
# ----------------------------------------------------------------------
def fig_binaural():
    fig = plt.figure(figsize=(9.5, 12.0))
    ax = fig.add_subplot(3, 1, 1)
    ax.set_aspect("equal"); ax.axis("off")
    a = 1.0
    ax.add_patch(Circle((0, 0), a, fc="#f6e5db", ec="k", lw=1.5))
    ax.scatter([-a, a], [0, 0], s=180, c=C_BLUE, zorder=6, edgecolors="k")
    ax.annotate("左耳", xy=(-a, 0), xytext=(-a - 0.1, -0.35), fontsize=FS_LABEL, ha="center")
    ax.annotate("右耳", xy=(a, 0), xytext=(a + 0.1, -0.35), fontsize=FS_LABEL, ha="center")
    th = np.deg2rad(60)
    for dy in np.linspace(-0.9, 0.9, 5):
        p1 = np.array([3.2 * np.cos(th) - dy * np.sin(th), 3.2 * np.sin(th) + dy * np.cos(th)])
        p2 = np.array([0.9 * np.cos(th) - dy * np.sin(th), 0.9 * np.sin(th) + dy * np.cos(th)])
        ax.annotate("", xy=tuple(p2), xytext=tuple(p1),
                    arrowprops=dict(arrowstyle="->", color=C_BLUE, alpha=0.55, lw=1.2))
    ax.annotate("", xy=(1.15 * np.cos(th), 1.15 * np.sin(th)), xytext=(3.1 * np.cos(th), 3.1 * np.sin(th)),
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.8))
    ax.text(3.3 * np.cos(th) + 0.05, 3.3 * np.sin(th), "声源方向", fontsize=FS_LABEL, color=C_RED)
    ax.text(-1.85, -1.35, "头影：远侧耳被头遮挡\n路程差 → 双耳时间差 ITD\n遮挡 → 双耳声级差 ILD",
            fontsize=FS_LABEL, ha="left", va="top", color=C_MAIN)
    ax.set_xlim(-1.9, 3.6); ax.set_ylim(-2.6, 3.6)
    ax.set_title("(a) 头 + 双耳 = 天然的 2 麦阵列（带挡板）", fontsize=11)
    ax = fig.add_subplot(3, 1, 2)
    az = np.linspace(-90, 90, 400)
    a_head, c = 0.0875, 343
    itd = (a_head / c) * (np.deg2rad(az) + np.sin(np.deg2rad(az))) * 1e6
    ax.plot(az, itd, color=C_BLUE, lw=2)
    ax.axhline(0, color="k", lw=0.6)
    max_itd_us = float(itd[-1])
    ax.axhline(max_itd_us, color="gray", ls="--", lw=0.8)
    ax.annotate(f"最大值 ≈ {max_itd_us:.0f} μs（正侧面 90°）",
                xy=(90, max_itd_us), xytext=(-5, 500), fontsize=FS_SMALL,
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.annotate("正前方偏 1° ≈ 9 μs\n（几何换算，不是听觉阈值）", xy=(1, 9), xytext=(-64, 300), fontsize=FS_SMALL,
                arrowprops=dict(arrowstyle="->", color=C_RED), color=C_RED)
    ax.set_xlabel("声源方位角 (°)"); ax.set_ylabel("ITD (μs)")
    ax.set_title("(b) 双耳时间差 ITD（Woodworth 模型）", fontsize=11)
    ax.grid(ls=":", alpha=0.5)
    ax = fig.add_subplot(3, 1, 3)
    ax.set_xscale("log")
    ax.set_xlim(100, 10000); ax.set_ylim(-0.45, 2.65)
    bands = [
        (0.15, 100, 1400, C_BLUE, "细结构 ITD\n低频主要线索"),
        (1.15, 250, 10000, C_RED, "ILD\n高频头影通常更明显"),
        (2.15, 1500, 10000, C_GREEN, "包络 ITD\n限有调制包络的高频声"),
    ]
    for y, start, stop, color, label in bands:
        ax.add_patch(plt.Rectangle(
            (start, y - 0.32), stop - start, 0.64,
            fc=color, ec=color, lw=1.0, alpha=0.20))
        ax.text(np.sqrt(start * stop), y, label, ha="center", va="center",
                fontsize=FS_SMALL, color=color, fontweight="bold")
    ax.axvline(1400, color=C_BLUE, ls="--", lw=1)
    ax.text(1320, 2.57, "约 1.4 kHz（非硬分界）", color=C_BLUE,
            fontsize=FS_TINY, ha="right", va="top",
            bbox=dict(fc="white", ec="none", alpha=0.88, pad=1.5))
    ax.set_yticks([]); ax.set_xlabel("频率 (Hz)")
    ax.set_title("(c) 双重理论的线索主导区（理论示意，不是人体响应曲线）", fontsize=FS_LABEL)
    ax.grid(ls=":", alpha=0.35, which="both", axis="x")
    fig.suptitle("图1  人类双耳定位线索：几何 ITD 与双重理论的适用边界", fontsize=13)
    fig.tight_layout()
    save(fig, "fig01_binaural.png")


# ----------------------------------------------------------------------
# 图13 GCC-PHAT 在混响下的退化（说明性随机指数衰减尾模型）
# ----------------------------------------------------------------------
def gcc_reverb_trial(t60, drr_db, seed, fs=16000, signal_length=16000,
                     tau_true=9, frame=1024):
    """一次 GCC-PHAT 模拟：两通道独立随机尾均按 DRR 固定总能量。"""
    rng = np.random.default_rng(seed)
    white = np.random.default_rng(seed + 1000).standard_normal(signal_length)
    spectrum = np.fft.rfft(white)
    spectrum[np.fft.rfftfreq(signal_length, 1 / fs) > 6000] = 0
    source = np.fft.irfft(spectrum)
    source /= np.max(np.abs(source))
    rir1 = random_decay_rir(t60, fs, rng, drr_db=drr_db)
    rir2 = random_decay_rir(t60, fs, rng, drr_db=drr_db)
    x1 = fft_convolve_prefix(source, rir1, signal_length)
    x2 = fft_convolve_prefix(causal_delay(source, tau_true), rir2, signal_length)
    start = int(rng.integers(0, signal_length - frame - 1))
    s12 = (np.fft.rfft(x1[start:start + frame], 4096)
           * np.conj(np.fft.rfft(x2[start:start + frame], 4096)))
    gcc = np.fft.fftshift(np.fft.irfft(s12 / (np.abs(s12) + 1e-10)))
    lags = np.arange(-2048, 2048)
    keep = np.abs(lags) <= 64
    lags, gcc = lags[keep], gcc[keep]
    peak = int(lags[np.argmax(gcc)])
    return lags, gcc, gcc_peak_is_correct(peak, tau_true), gcc_peak_contrast(gcc)


def gcc_reverb_statistics(t60_values, drr_values, trials=150):
    """返回 T60×DRR 网格上的正确率、Wilson 区间和对比度四分位数。"""
    t60_values = np.asarray(t60_values, dtype=float)
    drr_values = np.asarray(drr_values, dtype=float)
    shape = (drr_values.size, t60_values.size)
    rate = np.empty(shape); low = np.empty(shape); high = np.empty(shape)
    median = np.empty(shape); q1 = np.empty(shape); q3 = np.empty(shape)
    for i, drr_db in enumerate(drr_values):
        for j, t60 in enumerate(t60_values):
            results = [gcc_reverb_trial(t60, drr_db, 2000 + k * 17)[2:]
                       for k in range(trials)]
            successes = int(sum(item[0] for item in results))
            lo, hi = wilson_interval(successes, trials)
            contrasts = np.asarray([item[1] for item in results])
            rate[i, j], low[i, j], high[i, j] = successes / trials, lo, hi
            median[i, j] = np.median(contrasts)
            q1[i, j], q3[i, j] = np.percentile(contrasts, [25, 75])
    return {"rate": rate, "low": low, "high": high,
            "median": median, "q1": q1, "q3": q3}


def fig_gcc_reverb():
    fs, tau_true = 16000, 9
    t60_values = np.asarray([0.1, 0.4, 0.8])
    drr_values = np.asarray([6.0, 0.0, -6.0])
    stats = gcc_reverb_statistics(t60_values, drr_values, trials=150)
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 12.0))

    ax = axes[0]
    for t60, color, linestyle in zip(t60_values, [C_GREEN, C_BLUE, C_RED], ["-", "--", ":"]):
        lags, gcc, _, _ = gcc_reverb_trial(t60, 0.0, seed=7)
        gcc = gcc / np.max(np.abs(gcc))
        ax.plot(lags / fs * 1000, gcc, color=color, ls=linestyle, lw=1.7,
                label=f"$T_{{60}}$={t60:.1f} s")
    ax.axvline(-tau_true / fs * 1000, color="k", ls="--", lw=1.1,
               label="直达声真峰")
    ax.set_xlabel("时延 (ms)"); ax.set_ylabel("归一化 GCC-PHAT")
    ax.set_title("(a) DRR=0 dB 的固定种子单帧", fontsize=FS_TITLE)
    ax.legend(fontsize=FS_SMALL); ax.grid(ls=":", alpha=0.5)

    styles = [(C_GREEN, "o", "-"), (C_BLUE, "s", "--"), (C_RED, "^", ":")]
    ax = axes[1]
    for i, (drr_db, (color, marker, linestyle)) in enumerate(zip(drr_values, styles)):
        y = 100 * stats["rate"][i]
        ax.errorbar(t60_values, y,
                    yerr=[y - 100 * stats["low"][i], 100 * stats["high"][i] - y],
                    color=color, marker=marker, ls=linestyle, capsize=3,
                    label=f"DRR={drr_db:+.0f} dB")
    ax.set_ylim(0, 105); ax.set_xlabel("$T_{60}$ (s)"); ax.set_ylabel("真峰±1点正确率 (%)")
    ax.set_title("(b) 150 次；误差棒为 95% Wilson 区间", fontsize=FS_TITLE)
    ax.legend(fontsize=FS_SMALL); ax.grid(ls=":", alpha=0.5)

    ax = axes[2]
    for i, (drr_db, (color, marker, linestyle)) in enumerate(zip(drr_values, styles)):
        ax.plot(t60_values, stats["median"][i], color=color, marker=marker,
                ls=linestyle, label=f"DRR={drr_db:+.0f} dB")
        ax.fill_between(t60_values, stats["q1"][i], stats["q3"][i],
                        color=color, alpha=0.12)
    ax.set_xlabel("$T_{60}$ (s)"); ax.set_ylabel("峰值 / |GCC|背景中位数")
    ax.set_title("(c) 峰对比度中位数与四分位区间", fontsize=FS_TITLE)
    ax.legend(fontsize=FS_SMALL); ax.grid(ls=":", alpha=0.5)
    fig.suptitle("图13  GCC-PHAT 的两因素实验：随机尾时长 $T_{60}$ 与尾能量 DRR\n"
                 "直达能量=1；两通道独立随机尾按目标 DRR 固定总能量；64 ms 帧；无加性噪声；本书仿真",
                 fontsize=FS_SUP - 1)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.07, top=0.84,
                        hspace=0.52)
    fig._panel_vertical_gap = min(
        axes[index].get_position().y0 - axes[index + 1].get_position().y1
        for index in range(2))
    save(fig, "fig13_gcc_reverb.png")


# ----------------------------------------------------------------------
# 图4 时延 = 相位旋转：复数表示的几何直觉
# ----------------------------------------------------------------------
def fig_delay_phase():
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 12.0))
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
    ax.legend(fontsize=FS_SMALL, loc="upper right"); ax.grid(ls=":", alpha=0.5)
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
    ax.text(1.05, 0.06, "原始 $S(f)$", fontsize=FS_LABEL, color=C_BLUE)
    ax.text(0.05, -1.18, r"延迟后 $S(f)e^{-j2\pi f\tau}$", fontsize=FS_LABEL,
            color=C_RED)
    ax.text(0.98, -0.38, "2πfτ≈42°", fontsize=11, style="italic")
    ax.text(-2.25, 0.95, "延迟 τ 等价于复平面上\n顺时针转 2πfτ", fontsize=FS_LABEL, color=C_MAIN)
    ax.set_xlim(-2.35, 1.8); ax.set_ylim(-1.55, 1.5)
    ax.set_title("(b) 频域看延迟：复数相位转过一个角度", fontsize=11)
    # (c) 相位-频率直线
    ax = axes[2]
    f = np.linspace(0, 8000, 400)
    phase = -2 * np.pi * f * tau
    ax.plot(f / 1000, np.rad2deg(phase), color=C_BLUE, lw=2)
    for fk in [1000, 2000, 3000, 4000]:
        ax.plot(fk / 1000, np.rad2deg(-2 * np.pi * fk * tau), "o", color=C_RED, ms=6)
    ax.annotate("1 kHz → −42°\n2 kHz → −84°\n转角与频率成正比\n斜率 = −2πτ", xy=(3.0, -125),
                xytext=(3.8, -135), fontsize=FS_SMALL,
                arrowprops=dict(arrowstyle="->", color="k"))
    ax.set_xlabel("频率 (kHz)"); ax.set_ylabel("相位 (°)")
    ax.set_title("(c) 同一延迟在不同频率：相位-频率是直线", fontsize=11)
    ax.grid(ls=":", alpha=0.5)
    fig.suptitle("图4  时延为什么变成 $e^{-j2\\pi f\\tau}$：几何直觉\n"
                 "（4 cm 麦距；端射方向达到最大麦间时延 τ=d/c≈0.117 ms）", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.89), h_pad=3.5)
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
        a0 = ula_steering(d_over_lam * m, steer_deg)[:, 0]
        w = a0 / M
        A = ula_steering(d_over_lam * m, th)
        B = np.abs(w.conj() @ A)
        return 20 * np.log10(np.maximum(B / B.max(), 1e-5))

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.5))
    # (a) 解剖图
    ax = axes[0]
    Bdb = pattern(0.5, 30)
    ax.plot(th, Bdb, color=C_BLUE, lw=1.8)
    ax.axhline(-3, color="gray", ls="--", lw=1)
    main_lobe = np.abs(np.sin(tr) - np.sin(np.deg2rad(30))) <= 2 / M
    right_sidelobe = ~main_lobe & (th > 30)
    sidelobe_index = np.argmax(np.where(right_sidelobe, Bdb, -np.inf))
    sidelobe_db = float(Bdb[sidelobe_index])
    ax.axhline(sidelobe_db, color=C_ORANGE, ls="--", lw=1)
    ax.set_ylim(-40, 13); ax.set_xlim(-90, 90)
    # HPBW 实测
    i0 = np.argmax(Bdb)
    left = np.where(Bdb[:i0] < -3)[0][-1]; right = i0 + np.where(Bdb[i0:] < -3)[0][0]
    ax.annotate("", xy=(th[right], -3), xytext=(th[left], -3),
                arrowprops=dict(arrowstyle="<->", color=C_RED, lw=1.5))
    ax.text(0.03, 0.96, f"HPBW ≈ {th[right]-th[left]:.1f}°\n两侧 −3 dB 交点间角宽",
            transform=ax.transAxes, ha="left", va="top", fontsize=FS_LABEL, color=C_RED,
            bbox=dict(fc="white", ec=C_RED, lw=0.7, alpha=0.9, boxstyle="round,pad=0.25"))
    ax.annotate("30° 主瓣", xy=(30, 0), xytext=(13, 7), fontsize=FS_LABEL,
                arrowprops=dict(arrowstyle="->", color="k"))
    ax.annotate(f"最高旁瓣 ≈ {sidelobe_db:.1f} dB\n（旁瓣电平 SLL）",
                xy=(th[sidelobe_index], sidelobe_db), xytext=(30, -32), fontsize=FS_LABEL,
                color=C_ORANGE, arrowprops=dict(arrowstyle="->", color=C_ORANGE))
    ax.annotate("旁瓣", xy=(-32, -17), xytext=(-75, -8), fontsize=FS_LABEL,
                arrowprops=dict(arrowstyle="->", color="gray"))
    first_left_null = np.rad2deg(np.arcsin(np.sin(np.deg2rad(30)) - 2 / M))
    ax.annotate("零点（相消干涉处）", xy=(first_left_null, -39), xytext=(-30, -34), fontsize=FS_LABEL,
                arrowprops=dict(arrowstyle="->", color="k"))
    ax.set_xlabel("方向 (°)"); ax.set_ylabel("增益 (dB)")
    ax.set_title("(a) 波束图解剖：8 麦线阵、间距 λ/2、指向 30°", fontsize=11)
    ax.grid(ls=":", alpha=0.4)
    # (b) 栅瓣
    ax = axes[1]
    ax.plot(th, pattern(0.5, 60), color=C_BLUE, lw=1.8, ls="--", label="d = λ/2（安全）")
    ax.plot(th, pattern(1.0, 60), color=C_RED, lw=2.5, label="d = λ（超半波长）")
    ax.axvline(-7.7, color="gray", ls=":", lw=1.2, alpha=0.8)
    ax.set_ylim(-40, 13); ax.set_xlim(-90, 90)
    ax.text(0.03, 0.96, "栅瓣条件：sinθ = sin60° − λ/d", transform=ax.transAxes,
            fontsize=FS_LABEL, color=C_RED, va="top", ha="left",
            bbox=dict(fc="white", ec="0.7", alpha=0.9, boxstyle="round,pad=0.3"))
    ax.annotate("主瓣 60°", xy=(60, 0), xytext=(42, 7), fontsize=FS_LABEL,
                arrowprops=dict(arrowstyle="->", color="k"))
    ax.annotate("−7.7° 栅瓣\n与 60° 主瓣等高", xy=(-7.7, 0), xytext=(-82, 5),
                fontsize=FS_LABEL, color=C_RED, va="top",
                arrowprops=dict(arrowstyle="->", color=C_RED))
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

    fig = plt.figure(figsize=(9.5, 8.0))
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
    ax.text(-4.45, 7.45, "麦1在 x=0，麦2在 +x；正角波前先到麦2：\nτ₁₂=t₁−t₂=d·sinθ/c > 0，τ₂₁=−τ₁₂",
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

    inset(gs[0, 2], "(b) 正横 θ=0°：同时到达，τ₁₂=0", "broadside")
    inset(gs[1, 2], "(c) 正端射 θ=90°：τ₁₂=d/c", "endfire")
    fig.suptitle("图2  时延的几何来源与角度约定（示意）", fontsize=FS_SUP)
    # GridSpec 中再创建子 Axes，tight_layout 会警告；手工留出标题和间距。
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.07, top=0.88,
                        wspace=0.38, hspace=0.34)
    save(fig, "fig02_dsin_geometry.png")


# ----------------------------------------------------------------------
# 图20 AEC 信号处理链：经典结构 vs 混合式结构（示意）
# ----------------------------------------------------------------------
def fig_aec_pipeline():
    fig = plt.figure(figsize=(9.5, 11.0))

    # ---- (a) 经典 AEC 信号流 ----
    ax = fig.add_subplot(2, 1, 1)
    ax.axis("off"); ax.set_xlim(0, 14); ax.set_ylim(0, 8)

    def box(x, y, w, h, text, fc="#dbe9f6", fs=FS_SMALL, ec="k", ls="-", lw=1.2):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, ls=ls, zorder=3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=4)

    def arrow(x1, y1, x2, y2, text="", c="k", dy=0.18, fs=FS_TINY, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=14, color=c, lw=1.4, ls=ls, zorder=2))
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
    arrow(1.25, 6.2, 1.25, 3.45, "参考", C_BLUE, ls="--")
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

    def arrow2(x1, y1, x2, y2, text="", c="k", dy=0.18, fs=FS_TINY, ls="-"):
        ax2.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                      mutation_scale=14, color=c, lw=1.4, ls=ls, zorder=2))
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
    # DTD / 步长控制：从左侧绕行，避免穿过两条职责说明。
    box2(3.4, 0.7, 5.4, 0.95,
         "DTD / 步长控制（根据参考、麦克风与残差信号控制线性滤波器更新）",
         "white", FS_TINY, ec=C_ORANGE, ls="--")
    ax2.plot([6.1, 3.8, 3.8, 4.65], [1.65, 1.65, 4.05, 4.05],
             color=C_ORANGE, lw=1.4, ls="--", zorder=2)
    ax2.add_patch(FancyArrowPatch((4.65, 4.05), (5.35, y0),
                                  arrowstyle="-|>", mutation_scale=14,
                                  color=C_ORANGE, lw=1.4, ls="--", zorder=2))
    ax2.text(3.62, 2.8, "控制更新", fontsize=FS_TINY,
             color=C_ORANGE, ha="right", rotation=90)
    # 分工注释
    linear_note = ax2.text(
        6.1, 3.62, "线性 AEC：消除参考可解释、滤波器可表示的回声",
        fontsize=FS_TINY, color=C_BLUE, ha="center",
        bbox=dict(fc="white", ec="none", alpha=0.92, pad=1.5))
    neural_note = ax2.text(
        10.95, 3.08, "神经残余抑制：处理训练覆盖的非线性残余与噪声",
        fontsize=FS_TINY, color=C_ORANGE, ha="center",
        bbox=dict(fc="white", ec="none", alpha=0.92, pad=1.5))
    # 图例
    ax2.add_patch(plt.Rectangle((0.3, 7.3), 0.4, 0.4, fc=C_DSP, ec="k", lw=1.0))
    ax2.text(0.8, 7.5, "传统 DSP", fontsize=FS_TINY, va="center")
    ax2.add_patch(plt.Rectangle((2.6, 7.3), 0.4, 0.4, fc=C_NN, ec="k", lw=1.0))
    ax2.text(3.1, 7.5, "神经网络", fontsize=FS_TINY, va="center")
    ax2.set_title("(b) 混合式处理链：线性回声估计 + 学习型残余抑制（示意）", fontsize=FS_TITLE)

    fig.suptitle("图20  AEC 信号处理链：自适应滤波与混合式结构（示意）", fontsize=FS_SUP)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig._lower_annotation_lanes = (linear_note, neural_note)
    save(fig, "fig20_aec_pipeline.png")


# ----------------------------------------------------------------------
# 图19 AEC 全景：线性模型失配仿真 + 算法家族按假设分类
# ----------------------------------------------------------------------
def fig_aec_landscape():
    fig = plt.figure(figsize=(9.5, 10.5))

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
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.axis("off"); ax.set_xlim(0, 16.5); ax.set_ylim(0, 6)
    bw, gap, x0, y0 = 0.95, 0.12, 0.6, 3.0
    # 从左到右：更过去 … t-K-delay … 历史窗K … 间隔Δ … 当前帧t
    labels = {}
    current_note = None
    for i in range(n_show + 1):
        x = x0 + i * (bw + gap)
        if i == n_show:
            fc, ec, lw, tag = "#f6dbdb", C_RED, 2.0, "当前帧 t\n含直达/早期/晚期"
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
            current_note = ax.text(
                x + bw + 0.14, y0 + 0.55, tag, ha="left", va="center",
                fontsize=FS_SMALL, color=C_RED)
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
    formula_note = ax.text(
        (x_k + labels[n_show]) / 2 - 0.15, y0 - 0.30,
        r"晚期估计：$\hat y(t,f)=\sum_{k=0}^{K-1}G^*(k,f)y(t-\Delta-k,f)$" "\n"
        r"去混响输出：$y(t,f)-\hat y(t,f)$",
        ha="center", va="top", fontsize=FS_SMALL + 1, color=C_GREEN,
        bbox=dict(fc="white", alpha=0.8, pad=1, ec="none"))
    ax.text(7.4, 1.15, "回归只用过去帧；端到端仍含分帧与计算延迟", ha="center", va="center",
            fontsize=FS_LABEL, color=C_MAIN,
            bbox=dict(fc="#e8f6db", ec=C_GREEN, lw=0.8, alpha=0.9, boxstyle="round,pad=0.35"))
    fig.suptitle("图24  WPE 延迟预测：Δ=3 时保护 t−1、t−2，K=5 使用 t−3…t−7", fontsize=FS_SUP)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig._wpe_current_note = current_note
    fig._wpe_formula_note = formula_note
    save(fig, "fig24_wpe_frames.png")


# ----------------------------------------------------------------------
# 图25 延迟分类：关键路径 vs 因果历史状态
# ----------------------------------------------------------------------
def fig_latency_budget():
    fig, ax = plt.subplots(figsize=(9.5, 4.8))

    # 数字只是一组可替换的工作表格示例，不声称代表某产品。只有同一条关键路径上的项目才相加。
    critical_path = [("采集/分帧", 32, "#dbe9f6"),
                     ("未来帧前瞻", 16, C_PURPLE),
                     ("计算", 14, C_RED),
                     ("调度", 6, "#9b4d3b"),
                     ("首个判决确认", 25, C_ORANGE)]
    left = 0
    for name, duration_ms, color in critical_path:
        ax.barh(0, duration_ms, left=left, height=0.48, color=color,
                edgecolor="k", lw=1.0)
        ax.text(left + duration_ms / 2, 0, f"{name}\n{duration_ms} ms",
                ha="center", va="center", fontsize=FS_SMALL,
                color="white" if color == C_RED else C_MAIN)
        left += duration_ms
    continuous_audio_ms = sum(item[1] for item in critical_path[:-1])
    ax.axvline(continuous_audio_ms, color=C_GREEN, ls="--", lw=1.5)
    ax.text(continuous_audio_ms, -0.42, f"连续音频 {continuous_audio_ms} ms",
            color=C_GREEN, ha="right", fontsize=FS_SMALL)
    ax.annotate(f"首个判决 {left} ms",
                xy=(left, 0.24), xytext=(left - 2, 0.68), ha="right",
                fontsize=FS_LABEL, arrowprops=dict(arrowstyle="->", lw=1.1))
    ax.set_xlim(0, 110); ax.set_ylim(-0.72, 1.0)
    ax.set_yticks([0]); ax.set_yticklabels(["同一关键路径"])
    ax.set_xlabel("与实时时钟同量纲的延迟 (ms)", fontsize=FS_LABEL)
    ax.set_title("只相加同一关键路径上的串行等待", fontsize=FS_TITLE)
    ax.grid(axis="x", ls=":", alpha=0.5)
    ax.text(0.01, -0.34,
            "连续音频：统计到可用音频输出；首个判决：另含确认窗；端点 hangover：只延后结束事件。"
            "WPE/追踪历史状态不是未来帧等待。",
            transform=ax.transAxes, fontsize=FS_SMALL, va="top")
    fig.suptitle("图25  端到端延迟关键路径（示例数字非产品指标）", fontsize=FS_SUP)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.30, top=0.82)
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
    x_early /= np.max(np.abs(x_early))
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

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.0))
    sample_index = np.arange(36)
    axes[0, 0].plot(sample_index, x_early[:36], "o-", ms=3, lw=1.2,
                    color=C_BLUE, label="x₂[n]（先到）")
    axes[0, 0].plot(sample_index, x_late[:36], "s-", ms=3, lw=1.2,
                    color=C_RED, label=f"x₁[n]（晚 {delay_samples} 点）")
    axes[0, 0].set_title("(a) 输入：零填充整数延迟", fontsize=FS_TITLE)
    axes[0, 0].set_xlabel("采样点 n", fontsize=FS_LABEL)
    axes[0, 0].set_ylabel("归一化幅度", fontsize=FS_LABEL)
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
    cross_magnitude = np.abs(cross_spectrum)
    cross_magnitude /= np.max(cross_magnitude)
    axes[0, 1].plot(freq_khz, cross_magnitude, color=C_PURPLE, lw=1.3)
    axes[0, 1].set_title("(c) 频域乘积 X₁·conj(X₂)", fontsize=FS_TITLE)
    axes[0, 1].set_xlabel("频率 (kHz)", fontsize=FS_LABEL)
    axes[0, 1].set_ylabel("归一化互谱幅度", fontsize=FS_LABEL)
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


def fig_audio_examples():
    """Read the exported PCM itself, so curves and downloadable audio agree."""
    import json
    import wave
    root = Path(__file__).resolve().parents[1]
    audio = root / "codes" / "audio"
    manifest = audio / "MANIFEST.json"
    metadata = json.loads(manifest.read_text())
    for record in metadata["files"]:
        if hashlib.sha256((audio / record["file"]).read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("音频文件与清单不符，先重新生成并核验")

    def read(stem):
        with wave.open(str(audio / (stem + '.wav')), 'rb') as wav:
            rate, channels = wav.getframerate(), wav.getnchannels()
            x = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').reshape(-1, channels).T / 32768
        return rate, x[0]

    def rms_blocks(x, size=320):
        x = x[:len(x)//size*size].reshape(-1, size)
        return np.sqrt(np.mean(x*x, axis=1))

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9))
    for stem, label, style in [('spatial_mic1', '单麦', '-'),
                               ('spatial_aligned', '对齐平均', '--'),
                               ('spatial_reference', '延迟参考', ':')]:
        fs, x = read(stem)
        sl = slice(8000, 8320)
        axes[0].plot(np.arange(sl.start, sl.stop)/fs*1000, x[sl], style, label=label, lw=1.2)
    axes[0].set(title='(a) 已知延迟 3 点；同组共同增益', xlabel='时间 (ms)', ylabel='PCM 幅度 / 满量程')
    for stem, label, style in [('aec_microphone', '回声＋近端', '-'),
                               ('aec_frozen', '冻结残差', '--'), ('aec_near', '近端参考', ':')]:
        fs, x = read(stem)
        values = rms_blocks(x)
        axes[1].plot((np.arange(len(values))+.5)*320/fs, values, style, label=label)
    axes[1].axvspan(1.2, 1.8, color=C_ORANGE, alpha=.13, label='已知双讲时段')
    axes[1].set(title='(b) AEC：冻结来自真值掩码，不是检测结果', xlabel='时间 (s)', ylabel='20 ms 块 RMS / 满量程')
    for stem, label, style in [('wpe_reverberant', '稀疏回声输入', '-'),
                               ('wpe_output', '离线 WPE 输出', '--'), ('wpe_dry', '无回声参考', ':')]:
        fs, x = read(stem)
        values = rms_blocks(x)
        axes[2].plot((np.arange(len(values))+.5)*320/fs, values, style, label=label)
    axes[2].set(title='(c) 谐波目标也可预测：功率降低不等于恢复更好', xlabel='时间 (s)', ylabel='20 ms 块 RMS / 满量程')
    for ax in axes:
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, top + .35 * (top - bottom))
        ax.legend(loc='upper right', fontsize=11, ncol=2)
        ax.grid(ls=':', alpha=.4)
    fig.suptitle('图34  可下载合成音频的波形与分段能量（16 kHz；非实测语音）', fontsize=FS_SUP)
    fig.tight_layout(rect=(0, 0, 1, .96))
    save(fig, 'fig34_audio_examples.png', {'AudioManifestDigest': hashlib.sha256(manifest.read_bytes()).hexdigest()})


def fig_audio_counterexamples():
    """Measure exported PCM; show three separate failure mechanisms, not a ranking."""
    import json
    import wave
    root = Path(__file__).resolve().parents[1] / 'codes/audio'
    manifest = root / 'MANIFEST.json'
    records = {item['file']: item for item in json.loads(manifest.read_text())['files']}

    def read(stem):
        path = root / (stem + '.wav')
        if hashlib.sha256(path.read_bytes()).hexdigest() != records[path.name]['sha256']:
            raise ValueError(f'音频文件与清单不符：{stem}')
        with wave.open(str(path), 'rb') as wav:
            return np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').reshape(-1, wav.getnchannels()).T / 32768

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.8))
    reference = read('correlation_reference')
    errors = [np.sqrt(np.mean((read('correlation_' + name)-reference)**2))
              for name in ('single', 'independent', 'common')]
    for index, (value, color, hatch) in enumerate(zip(errors, [C_MAIN, C_BLUE, C_ORANGE], ['', '//', 'xx'])):
        axes[0].bar(index, value, color=color, hatch=hatch, edgecolor='white', width=.55)
        axes[0].text(index, value+.003, f'{value:.4f}', ha='center')
    axes[0].set(xticks=range(3), xticklabels=['单麦', '两路独立噪声平均', '两路相同噪声平均'],
                ylim=(0, max(errors)*1.4), ylabel='误差 RMS / 满量程',
                title='(a) 已对齐目标：同组相减参考，无时延或增益拟合')
    sl = slice(8000, 8160)
    for stem, label, style in [('polarity_reference', '目标参考', '-'),
                               ('polarity_corrected', '已知极性纠正后平均', '--'),
                               ('polarity_uncorrected', '接反后直接平均', ':')]:
        axes[1].plot(np.arange(sl.start, sl.stop)/16000*1000, read(stem)[0, sl], style, label=label)
    axes[1].set(xlabel='时间 (ms)', ylabel='PCM 幅度 / 满量程',
                title='(b) 通道增益 1 与 −0.9：平均目标增益从 0.95 降至 0.05')
    axes[1].legend(loc='upper right', ncol=3, fontsize=11)
    low, high = axes[1].get_ylim()
    axes[1].set_ylim(low, high+.6*(high-low))
    reference = read('conditioning_reference')
    for offset, case, label, color, hatch in [(-.18, 'well', 'κ₂=3', C_BLUE, '//'),
                                            (.18, 'ill', 'κ₂=199', C_ORANGE, 'xx')]:
        error = np.sqrt(np.mean((read(f'conditioning_{case}_output')-reference)**2, axis=1))
        axes[2].bar(np.arange(2)+offset, error, .36, label=label, color=color, hatch=hatch, edgecolor='white')
    axes[2].set(xticks=[0, 1], xticklabels=['源 1', '源 2'], ylabel='恢复误差 RMS / 满量程',
                title='(c) 同一输入噪声，已知矩阵求解；两路各自对参考评分')
    axes[2].legend(loc='upper center', ncol=2)
    low, high = axes[2].get_ylim()
    axes[2].set_ylim(0, high*1.45)
    for ax in axes:
        ax.grid(axis='y', ls=':', alpha=.35)
        ax.set_axisbelow(True)
    fig.suptitle('图35  三种合成反例：相关噪声、极性错误与病态求逆', fontsize=FS_SUP)
    fig.tight_layout(rect=(0, 0, 1, .96), h_pad=1.6)
    save(fig, 'fig35_audio_counterexamples.png', {'AudioManifestDigest': hashlib.sha256(manifest.read_bytes()).hexdigest()})


def nonlinear_echo_measurements():
    """Read fixed PCM files; return full-segment Fourier amplitudes and power.

    The 2 s segment has an integer number of periods, so 2*|DFT[k]|/N is
    the one-sided peak amplitude at the non-DC, non-Nyquist tone bins.
    """
    import json
    import wave
    root = Path(__file__).resolve().parents[1] / 'codes/audio'
    manifest = root / 'MANIFEST.json'
    records = {item['file']: item for item in json.loads(manifest.read_text())['files']}
    data = {}
    for name in ('reference', 'echo', 'estimate', 'residual'):
        path = root / f'nonlinear_{name}.wav'
        if hashlib.sha256(path.read_bytes()).hexdigest() != records[path.name]['sha256']:
            raise ValueError(f'音频文件与清单不符：{path.name}')
        with wave.open(str(path), 'rb') as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()) != (16000, 1, 2, 32000):
                raise ValueError('非线性例需要 16 kHz 单声道、32000 点 PCM16')
            samples = np.frombuffer(wav.readframes(32000), dtype='<i2').astype(float) / 32768
        amplitudes = 2 * np.abs(np.fft.rfft(samples)[[1000, 3000]]) / samples.size
        data[name] = {'samples': samples, 'amplitudes': amplitudes,
                      'power': float(np.mean(samples**2))}
    return data, hashlib.sha256(manifest.read_bytes()).hexdigest()


def fig_nonlinear_echo():
    """Show the unmodelled third harmonic without changing residual gain."""
    data, digest = nonlinear_echo_measurements()
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.5))
    labels = [('echo', '非线性回声', '-', C_MAIN),
              ('estimate', '最佳标量线性估计', '--', C_BLUE),
              ('residual', '相减后的残差', ':', C_RED)]
    for name, label, style, color in labels:
        axes[0].plot(np.arange(128) / 16, data[name]['samples'][:128], style,
                     label=label, color=color, linewidth=1.7)
    axes[0].set(xlabel='时间 (ms)', ylabel='PCM 幅度 / 满量程', ylim=(-.6, .85),
                title='(a) 前 8 ms；三条曲线保留同一增益，残差未单独放大')
    axes[0].legend(ncol=3, loc='upper right', fontsize=11)
    for index, (name, label, _style, color) in enumerate(labels):
        values = data[name]['amplitudes']
        positions = np.arange(2) + (index - 1) * .24
        axes[1].bar(positions, values, .23, label=label, color=color,
                    hatch=('', '//', 'xx')[index], edgecolor='white')
        for position, value in zip(positions, values):
            axes[1].text(position, value + .016, f'{value:.4f}', ha='center', fontsize=11)
    axes[1].set(xticks=[0, 1], xticklabels=['500 Hz（基频）', '1500 Hz（三次谐波）'],
                ylabel='单边峰值幅度 / 满量程', ylim=(0, .7),
                title='(b) 全段 2 s 的 DFT；整数周期、无窗，数值来自 PCM 读回')
    axes[1].legend(ncol=3, loc='upper right', fontsize=11)
    for ax in axes:
        ax.grid(axis='y', ls=':', alpha=.35)
        ax.set_axisbelow(True)
    fig.suptitle('图36  三次非线性留下线性估计无法表示的谐波（合成反例）', fontsize=FS_SUP)
    fig.tight_layout(rect=(0, 0, 1, .95), h_pad=1.6)
    save(fig, 'fig36_nonlinear_echo.png', {'AudioManifestDigest': digest})


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
    fig_audio_examples()
    fig_audio_counterexamples()
    fig_nonlinear_echo()
    print("ALL DONE")


if __name__ == "__main__":
    main()
