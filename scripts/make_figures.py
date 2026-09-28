# -*- coding: utf-8 -*-
"""生成教程插图 34 张（图 1~25、图 33~36、40~44；图 26~32、37~39 见 make_aec_figures.py）。

用法（仓库根目录）：
    .venv/bin/python scripts/make_figures.py      # 图 1~25、图 33~36、40~44 → figures/
"""
from pathlib import Path
import hashlib
import json
import platform
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

    返回时延轴（秒）、相关序列和最大峰时延。原 FFT 长度覆盖未加权
    线性相关；PHAT 加权后的逆变换仍是周期函数，不保证有限支撑。
    interp 只细化同一周期三角插值，不增加带宽或独立观测信息。
    加长实逆变换时拆分旧 Nyquist 项，并补偿逆变换长度带来的尺度。
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
    if interp > 1 and n_fft > 1 and n_fft % 2 == 0:
        # The old self-conjugate Nyquist bin becomes a +/- frequency pair.
        # irfft supplies its negative partner, so each member gets half.
        phat[-1] *= 0.5
    correlation = interp * np.fft.fftshift(np.fft.irfft(phat, n=n_interp))
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
              "(d) 四麦方阵/平面阵", "(e) 圆环六麦 + 中心一麦（共7麦）",
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
    ax.add_patch(Circle((0.5, 0.5), np.sqrt(0.5), fill=False, ls="--", color="gray"))
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
    ax.set_title("(a) 近场模型：球面波，响应依赖距离与方向", fontsize=11)
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
def beam_pattern_comparison():
    """图16的固定总体协方差；返回权重/几何，不把有限INR当硬零陷。"""
    channels = 8
    centered = np.arange(channels) - (channels - 1) / 2
    spacing = 0.5
    target = ula_steering(spacing * centered, 0)[:, 0]
    interferer = ula_steering(spacing * centered, 20)[:, 0]
    covariance = np.eye(channels) + 10 * np.outer(interferer, interferer.conj())
    mvdr, _ = distortionless_weights(covariance, target)
    sd_spacing = 0.2
    diffuse = np.sinc(2 * sd_spacing * np.abs(np.subtract.outer(centered, centered)))
    sd, _ = distortionless_weights(diffuse, target, diagonal_loading=1e-6)
    return centered, [
        ("DSB：d=0.5λ", target / channels, spacing, C_BLUE, "-"),
        ("MVDR：d=0.5λ", mvdr, spacing, C_RED, "--"),
        ("超指向：d=0.2λ", sd, sd_spacing, C_GREEN, ":"),
    ]


def fig_beampatterns():
    centered, curves = beam_pattern_comparison()
    angles = np.linspace(-90, 90, 721)
    radians = np.deg2rad(angles)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.8), subplot_kw={"polar": True})
    for panel, ax in enumerate(axes):
        for name, weight, spacing, color, style in curves:
            gain = np.abs(weight.conj() @ ula_steering(spacing * centered, angles))
            # All three weights have unit target response; no per-curve peak normalization.
            if panel:
                gain = np.maximum(20 * np.log10(np.maximum(gain, 1e-12)), -40) + 40
            ax.plot(radians, gain, color=color, ls=style, lw=2.0, label=name)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)  # +y broadside -> +x: positive angles to the right.
        ax.set_thetamin(-90)
        ax.set_thetamax(90)
        ax.set_thetagrids([-90, -60, -30, 0, 30, 60, 90])
        if panel:
            ax.set_rlim(0, 42)  # Keep the small >0 dB peak visible.
            ax.set_rgrids([0, 20, 40], ["−40", "−20", "0 dB"], angle=-60, fontsize=11)
        else:
            ax.set_rlim(0, 1.06)
            ax.set_rgrids([0.5, 1], ["0.5", "1.0"], angle=-60, fontsize=11)
        ax.set_title(["(a) 幅度响应（无量纲）", "(b) 幅度响应（dB）"][panel], pad=15)
    interferer = ula_steering(.5 * centered, 20)[:, 0]
    rejection = 20 * np.log10(abs(np.vdot(curves[1][1], interferer)))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
               bbox_to_anchor=(.5, .12), ncol=3, frameon=False, fontsize=11)
    fig.text(.5, .09, f"MVDR 在 20° 的响应为 {rejection:.2f} dB；低于 −40 dB 的曲线截在图心。",
             ha="center", fontsize=11)
    fig.text(.5, .035, "两种阵距不能作为同条件排名；目标 0°，三条曲线的目标响应均为 1。",
             ha="center", fontsize=11)
    fig.suptitle("图16  8 麦线阵的固定与自适应方向图", fontsize=FS_SUP)
    fig.text(.5, .9, "MVDR：单通道线性 INR=10，白噪声方差=1；超指向：相对加载 $10^{-6}$", ha="center", fontsize=11)
    fig.subplots_adjust(left=.045, right=.97, bottom=.23, top=.82, wspace=.25)
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
    axes[0].plot(th, 10 * np.log10(capon + 1e-6), color=C_RED, ls="--", lw=2.2, label="Capon/MVDR 谱")
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
    axes[1].set_ylim(-25, 3); axes[1].set_xlabel("角度 (°)"); axes[1].set_ylabel("按自身峰值归一的伪谱 (dB)")
    axes[1].text(0.5, 0.30, "峰高不等于源功率\n只比较峰位与谱形", transform=axes[1].transAxes,
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
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 10.5),
                             gridspec_kw={"height_ratios": [1, 1, 0.8]})
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
    ax.set_ylim(-1.15, 1.15)
    ax.set_title("(a) 两路信号（放大 0.5 ms 窗口）", fontsize=FS_TITLE)
    ax.set_xlabel("时间 (ms)", fontsize=FS_LABEL)
    ax.set_ylabel("归一化幅度", fontsize=FS_LABEL)
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
    ax.set_xlabel("时延 τ (μs)", fontsize=FS_LABEL); ax.set_ylabel("GCC-PHAT（保持原网格幅度）", fontsize=FS_LABEL); ax.grid(ls=":", alpha=0.5)
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
    ax.set_xlim(-0.05, 0.17); ax.set_ylim(-0.028, 0.13); ax.set_aspect("equal"); ax.axis("off")
    fig.suptitle("图12  GCC-PHAT 声源定位原理：亚采样时延估计（模拟）", fontsize=FS_SUP)
    fig.tight_layout()
    save(fig, "fig12_gcc_phat.png")


# ----------------------------------------------------------------------
# 图17 GSC 结构框图
# ----------------------------------------------------------------------
def fig_gsc():
    fig, ax = plt.subplots(figsize=(9.5, 6.8))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)

    def box(x, y, width, height, label, color="#dbe9f6"):
        ax.add_patch(plt.Rectangle((x, y), width, height, fc=color, ec=C_MAIN, lw=1.2, zorder=3))
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=11, zorder=4)

    def path(points, color=C_MAIN, dashed=False):
        if len(points) > 2:
            xx, yy = zip(*points[:-1])
            ax.plot(xx, yy, color=color, lw=1.5, ls="--" if dashed else "-", zorder=1)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                    mutation_scale=13, lw=1.5, color=color, linestyle="--" if dashed else "-", zorder=2))

    box(.15, 3.85, 1.45, .85, "阵列观测\n$\\mathbf{x}(n)$")
    box(3.0, 5.6, 2.0, .9, "固定权重\n$\\mathbf{w}_q^H$", "#f6e5db")
    box(3.0, 3.45, 2.0, .9, "阻塞变换\n$\\mathbf{B}^H$")
    box(6.2, 3.45, 2.0, .9, "自适应权重\n$\\mathbf{h}^H(n)$")
    box(9.0, 4.8, 1.8, .9, "相减\n$e=d-y$", "#e8f6db")
    box(5.85, 1.5, 2.7, .9, "由 $u(n)$、$e(n)$ 更新\n下一步 $\\mathbf{h}(n+1)$", "#eee2f3")
    path([(1.6, 4.275), (2.25, 4.275), (2.25, 6.05), (3, 6.05)])
    path([(2.25, 4.275), (2.25, 3.9), (3, 3.9)])
    path([(5, 6.05), (9.9, 6.05), (9.9, 5.7)])
    ax.text(7.2, 6.23, "$d(n)$：固定支路输出", ha="center", fontsize=11)
    path([(5, 3.9), (6.2, 3.9)])
    ax.text(5.6, 4.1, "$u(n)$", ha="center", fontsize=11)
    path([(8.2, 3.9), (9.9, 3.9), (9.9, 4.8)])
    ax.text(9.2, 3.56, "$y(n)$：对消分量", ha="center", fontsize=11)
    path([(10.8, 5.25), (11.75, 5.25)])
    ax.text(11.4, 5.52, "$e(n)$", ha="center", fontsize=11)
    path([(5.5, 3.9), (5.5, 1.95), (5.85, 1.95)], C_PURPLE, True)
    path([(11.2, 5.25), (11.2, 1.95), (8.55, 1.95)], C_PURPLE, True)
    path([(7.2, 2.4), (7.2, 3.45)], C_PURPLE, True)
    ax.text(7.55, 2.85, "更新状态", fontsize=11, color=C_PURPLE)
    ax.text(.25, 7.15, "导向模型 a 决定固定权重与阻塞基：", fontsize=11)
    ax.text(7.0, 7.15, "$\\mathbf{w}_q^H\\mathbf{a}=1$,  $\\mathbf{B}^H\\mathbf{a}=0$", fontsize=12, ha="center")
    path([(3.65, 7.0), (3.65, 6.5)], C_ORANGE, True)
    path([(2.75, 7.0), (2.75, 4.5), (3.5, 4.5), (3.5, 4.35)], C_ORANGE, True)
    ax.text(.25, .77, "模型匹配时，任意有限 h 都保持约束方向响应；收敛决定噪声抑制效果。", fontsize=11)
    ax.text(.25, .27, "单频复数模型；本次输出使用更新前权重。虚线表示模型、误差和状态的控制依赖。", fontsize=11)
    ax.set_title("图17  GSC 的输出路径与自适应反馈", fontsize=FS_SUP, pad=10)
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
    ax.plot(*src, marker="o", ms=9, mfc="none", mec="white", mew=1.8,
            ls="none", label="真源 (5.500, 3.800) m")
    ax.plot(GX[imax], GY[imax], "r*", ms=12, mec="k",
            label=f"网格峰 ({GX[imax]:.3f}, {GY[imax]:.3f}) m")
    ax.legend(loc="upper left", fontsize=FS_SMALL - 1,
              facecolor="0.9", framealpha=0.95)
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
            if clutter_probability == 0 or not lower <= observation <= upper:
                # Outside the finite clutter support its density is zero. The
                # common Gaussian factor cancels, including the mixture weight.
                # Select the reference only among particles with positive prior.
                support = weights > 0
                supported = angles[support]
                reference = supported[np.argmin(np.abs(supported - np.clip(observation, lower, upper)))]
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    # Factored square difference preserves particle separation
                    # even when observation is so large that angle-z rounds to -z.
                    difference = angles - reference
                    total = (angles - observation) + (reference - observation)
                    # Form exact zero products before scale division: equal
                    # distances stay equally likely even for tiny positive σ.
                    squared_difference = difference * total
                    squared_difference[difference == 0] = 0.0
                    likelihood_log = -0.5 * (squared_difference / observation_std) / observation_std
                likelihood_log[angles == reference] = 0.0
                likelihood_log[~support] = -np.inf
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
    fig = plt.figure(figsize=(9.5, 9.0))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.25, 1))
    axes = [fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]),
            fig.add_subplot(grid[1, 1])]
    axes[0].plot(t, obs, ".", color="gray", ms=4, label="DOA观测（含噪声和均匀杂波）")
    axes[0].plot(t, true, color="k", lw=2, label="真实轨迹")
    axes[0].plot(t, est, color=C_RED, lw=1.6, ls="--", label="混合似然粒子滤波估计")
    axes[0].plot(t[resampled], est[resampled], "|", color=C_ORANGE, ms=7,
                 label=r"重采样 ($N_\mathrm{eff}<N/2$)")
    boundary_active = reflection_fraction >= 0.05
    axes[0].plot(t[boundary_active], est[boundary_active], "x", color=C_PURPLE,
                 ms=5, label="反射粒子比例≥5%")
    axes[0].axvspan(48, 57, color=C_PURPLE, alpha=0.12, label="连续 10 帧缺测")
    axes[0].set_xlabel("帧"); axes[0].set_ylabel("方位角 (°)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=FS_SMALL, loc="lower center", ncol=2,
               bbox_to_anchor=(.5, .002), framealpha=.95)
    axes[0].grid(ls=":", alpha=0.5)
    axes[0].set_title("(a) 单说话人追踪：匀速状态模型在缺测段只做预测\n"
                      "（200 粒子；高斯目标+均匀杂波似然；0～120°镜面反射边界）", fontsize=FS_TITLE)
    axes[1].plot(t, n_eff, color=C_BLUE, lw=1.5, label="更新后、重采样前")
    axes[1].axhline(N/2, color=C_RED, ls="--", label="重采样阈值 100")
    axes[1].axvspan(48, 57, color=C_PURPLE, alpha=.12)
    axes[1].set(xlabel="帧", ylabel="有效粒子数 (个)", ylim=(0, N+5),
                title="(b) 本帧重采样前 ESS\n虚线为重采样阈值100")
    axes[1].grid(ls=":", alpha=.5)
    ax = axes[2]
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
    ax.set_title("(c) 人工 PHD 强度示意：积分=2\n（角度约定：−90°～+90°）", fontsize=FS_TITLE)
    ax.grid(ls=":", alpha=0.5)
    fig.suptitle("图22  声源追踪示意（模拟）", fontsize=FS_SUP)
    fig.tight_layout(rect=(0, .16, 1, .95), h_pad=2.0)
    save(fig, "fig22_tracking.png")
    valid = ~missing
    rmse = lambda mask, values: float(np.sqrt(np.mean((values[mask]-true[mask])**2)))
    report = {
        "schema_version": 1, "seed": FIGURE_SEEDS["tracking"],
        "generator": "NumPy default_rng / PCG64; one sequential generator",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "parameters": {"frames": T, "particles": N, "observation_std_deg": 7,
            "observation_replacement_probability": .2, "likelihood_clutter_probability": .1,
            "angle_bounds_deg": [0,120], "process_std_deg_per_frame": 1,
            "velocity_process_std_deg_per_frame": .25, "resample_ess_below": N/2,
            "missing_frame_interval_half_open": [48,58]},
        "counts": {"observed": int(valid.sum()), "missing": int(missing.sum()),
            "replacement_draws": int(is_out.sum()), "observed_replacements": int((is_out&valid).sum()),
            "resampling": int(resampled.sum()), "reflection_at_least_5_percent": int(boundary_active.sum())},
        "scores": {"raw_observed_rmse_deg": rmse(valid,obs),
            "pf_observed_rmse_deg": rmse(valid,est), "pf_missing_rmse_deg": rmse(missing,est),
            "minimum_ess": float(n_eff.min())},
        "frames": {"index": t.tolist(), "truth_deg": true.tolist(),
            "observation_deg": [float(x) if np.isfinite(x) else None for x in obs],
            "estimate_deg": est.tolist(), "missing": missing.tolist(),
            "replacement_draw": is_out.tolist(), "ess_before_resampling": n_eff.tolist(),
            "resampled": resampled.tolist(), "reflection_fraction": reflection_fraction.tolist()},
        "phd_illustration": {"not_a_filter_run": True, "grid_deg": th.tolist(),
            "intensity_targets_per_degree": I.tolist(), "integral_targets": float(np.trapezoid(I,th))},
        "interpretation": "Single synthetic angular sequence, not microphone/audio input; no confidence interval or general performance claim."}
    target = Path(__file__).resolve().parents[1]/"codes/reports/figure22_tracking.json"
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


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
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.5))
    for ax, ds, sd, ylabel, title in zip(
            axes, [wng_dsb, di_dsb], [wng_sd, di_sd],
            ["白噪声增益 WNG (dB)", "指向性指数 DI (dB)"],
            ["(a) 空间白噪声", "(b) 三维各向同性弥散噪声"]):
        ax.semilogx(freqs, ds, color=C_BLUE, lw=2, label="DSB 延迟求和")
        ax.semilogx(freqs, sd, color=C_RED, ls="--", lw=2, label="对角加载超指向")
        ax.set_xlabel("频率 (Hz)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.minorticks_on()
        ax.grid(which="major", ls=":", alpha=.5)
        ax.grid(which="minor", ls=":", alpha=.25)
        ax.set_xlim(100, 8000)
    axes[0].axhline(0, color="gray", ls="-.", lw=1, zorder=0)
    axes[0].set_ylim(-48, 10)
    axes[1].set_ylim(-.5, 12)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
               bbox_to_anchor=(.5, .06), ncol=2, frameon=False, fontsize=11)
    fig.text(.5, .025, "目标响应均为 1；灰点划线为 0 dB；100～8000 Hz 共 200 个线性采样频点。",
             ha="center", fontsize=11)
    fig.suptitle("图15  6 麦圆阵的白噪声增益与指向性", fontsize=FS_SUP)
    fig.text(.5, .91, "半径与相邻弦长均为 4 cm；朝向 +x（正横角 90°）；声速 343 m/s", ha="center", fontsize=11)
    fig.text(.5, .86, "超指向采用相对对角加载 $10^{-6}$；使用未加载的弥散协方差计算 DI", ha="center", fontsize=11)
    fig.subplots_adjust(left=.08, right=.98, bottom=.23, top=.75, wspace=.32)
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
    """Two compact panels keep physical reference and optional branches distinct."""
    fig, (main, branches) = plt.subplots(
        2, 1, figsize=(9.5, 7.8), gridspec_kw={"height_ratios": [1, 1.32]})
    for ax, ymax in ((main, 4.15), (branches, 5.25)):
        ax.set_xlim(0, 16.0)
        ax.set_ylim(0, ymax)
        ax.axis("off")
    main.set_title("(a) 播放参考与音频主链", loc="left", pad=4)
    branches.set_title("(b) 按任务选择的方向、GSS 与分离支路", loc="left", pad=4)
    rectangles, node_texts = {}, {}

    def node(ax, key, x, y, label, *, width=2.35, height=.76, fill="#e7eff6"):
        rectangle = plt.Rectangle((x, y), width, height, fc=fill,
                                  ec=C_MAIN, lw=1.05, zorder=3)
        ax.add_patch(rectangle)
        artist = ax.text(x + width / 2, y + height / 2, label,
                         ha="center", va="center", fontsize=FS_SMALL, zorder=4)
        rectangles[key] = rectangle
        node_texts[key] = artist
        return (x, y, width, height)

    def point(box, side):
        x, y, width, height = box
        return {"left": (x, y + height / 2), "right": (x + width, y + height / 2),
                "top": (x + width / 2, y + height), "bottom": (x + width / 2, y)}[side]

    def arrow(ax, start, end, color=C_MAIN, style="-", width=1.45):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                     mutation_scale=12, color=color, lw=width,
                                     linestyle=style, zorder=2))

    # Top panel: the playback-reference take-off is visibly after render
    # processing and before the DAC/amplifier. S labels the physical loop.
    far = node(main, "far_end", .25, 3.08, "远端播放", width=2.25, fill="#fff2cc")
    render = node(main, "render", 3.05, 3.08, "混音 / 均衡 / 音量", width=2.62, fill="#fff2cc")
    speaker = node(main, "speaker", 6.38, 3.08, "DAC / 功放 / 扬声器", width=2.75, fill="#fff2cc")
    tap = node(main, "render_tap", 3.18, 1.89, "R：处理后播放参考", width=2.40, fill="#fff2cc")
    arrow(main, point(far, "right"), point(render, "left"))
    arrow(main, point(render, "right"), point(speaker, "left"))
    arrow(main, point(render, "bottom"), point(tap, "top"), C_RED, "--")
    arrow(main, point(speaker, "right"), (9.43, 3.46), C_RED, ":")
    main.text(9.50, 3.48, "S：扬声器 → 房间/机壳 → 麦克风",
              color=C_RED, va="center", fontsize=FS_SMALL)
    main.text(9.50, 2.90, "物理声路带回声；R 提供 AEC 参考",
              color=C_RED, va="center", fontsize=FS_SMALL)

    xs = [.18, 2.83, 5.48, 8.13, 10.78, 13.43]
    keys = ("capture", "aec", "wpe", "bf", "ns", "backend")
    labels = ("S：多麦采集\n同步/标定", "AEC\n回声消除", "WPE\n去混响",
              "解析波束\nMVDR/GEV", "NS / AGC\n逐流可选", "KWS / ASR\n逐流或选流")
    fills = ("#dbe9f6", "#dbe9f6", "#dbe9f6", "#e8f6db", "#e8f6db", "#f6dbdb")
    boxes = {key: node(main, key, x, .30, label, fill=fill)
             for key, x, label, fill in zip(keys, xs, labels, fills)}
    for src, dst in zip(keys[:-1], keys[1:]):
        arrow(main, point(boxes[src], "right"), point(boxes[dst], "left"))
    arrow(main, point(tap, "bottom"), point(boxes["aec"], "top"), C_RED, "--")
    main.text(6.25, 1.34, "S 与 R 在 AEC 汇合；主链按任务裁剪。",
              color="dimgray", fontsize=FS_SMALL)

    # Bottom panel: every lane has its own input/output contract. A named M
    # port denotes the same WPE-stage multichannel STFT, without crossing
    # arrows through the unrelated routes.
    port_fill = "#edf4fa"
    m = node(branches, "wpe_port", .16, 4.08,
             "M：WPE 后多通道\nSTFT / 特征", width=2.65, fill=port_fill)
    ssl = node(branches, "ssl", 3.25, 4.08, "SSL / DOA\n定位", width=2.35, fill="#f6e5db")
    track = node(branches, "tracking", 6.08, 4.08, "追踪预测\n方向+协方差+时刻", width=2.78, fill="#f6e5db")
    bfport = node(branches, "bf_port", 10.20, 4.08,
                  "D → 解析波束\n方向控制", width=2.85, fill="#e8f6db")
    for left, right in ((m, ssl), (ssl, track), (track, bfport)):
        arrow(branches, point(left, "right"), point(right, "left"), C_ORANGE, "-.")
    branches.text(13.25, 4.46, "D 有效期 / 坐标系", color=C_ORANGE, fontsize=FS_SMALL,
                  va="center")

    diar = node(branches, "diarization", .16, 2.77,
                "说话人活动标注", width=2.65, fill="#f6e5db")
    mask = node(branches, "gss_mask", 3.25, 2.77,
                "GSS / cACGMM\nM + 活动→掩码", width=2.35, fill="#f6e5db")
    scm = node(branches, "scm", 6.08, 2.77,
               "M + 掩码 →\n目标/干扰 SCM", width=2.78, fill="#f6e5db")
    gssout = node(branches, "gss_out", 10.20, 2.77,
                  "解析波束\n逐目标音轨", width=2.85, fill="#e8f6db")
    for left, right in ((diar, mask), (mask, scm), (scm, gssout)):
        arrow(branches, point(left, "right"), point(right, "left"), C_PURPLE, ":")
    branches.text(13.25, 3.15, "M 同时进掩码与 SCM", color=C_PURPLE,
                  fontsize=FS_SMALL, va="center")

    neuralin = node(branches, "neural_in", .16, 1.46,
                    "M：WPE 后多通道\nSTFT / 特征", width=2.65, fill=port_fill)
    neural = node(branches, "neural_separator", 3.25, 1.46,
                  "神经 / CSS\n分离旁路", width=2.35, fill="#e8f6db")
    nsport = node(branches, "ns_port", 6.08, 1.46,
                  "逐流 NS / AGC\n按需保留", width=2.78, fill="#e8f6db")
    backport = node(branches, "backend_port", 10.20, 1.46,
                    "逐流 KWS / ASR\n或先选一路", width=2.85, fill="#f6dbdb")
    for left, right in ((neuralin, neural), (neural, nsport), (nsport, backport)):
        arrow(branches, point(left, "right"), point(right, "left"), C_GREEN, "-.")
    branches.text(13.25, 1.84, "旁路解析波束", color=C_GREEN, fontsize=FS_SMALL,
                  va="center")

    control = node(branches, "activity_control", .16, .19,
                   "C：活动 / 远端\n双讲控制", width=2.65, height=.77, fill="#fff1df")
    branches.text(3.27, .58,
                  "C 分别约束 AEC、WPE、追踪、波束和后端；各模块按自身状态决定更新。",
                  color=C_ORANGE, fontsize=FS_SMALL, va="center")
    arrow(branches, point(control, "right"), (3.16, .58), C_ORANGE, "--")

    # Named ports express cross-panel contracts; the separate dependency
    # specification remains the machine-checkable graph, not a claim that
    # every edge has an individual arrow in this compact teaching diagram.
    fig._pipeline_node_rectangles = rectangles
    fig._pipeline_node_texts = node_texts
    fig._pipeline_panel_axes = (main, branches)
    fig.suptitle("图23  播放参考、音频主链与条件支路", fontsize=FS_SUP, y=.985)
    fig.subplots_adjust(left=.025, right=.985, top=.93, bottom=.025, hspace=.22)
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
    ax.text(0.03, 0.94, "单位响应、独立同方差噪声下的上限。",
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
    ax.legend(fontsize=FS_SMALL, loc="lower right")
    ax.grid(ls=":", alpha=0.5, which="both")
    ax.set_title("(b) 通道、麦对和协方差矩阵的规模", fontsize=FS_TITLE)
    fig.text(0.5, 0.015, "数据项数量由公式直接计算，不等同于运行时间、功耗或价格。",
             ha="center", fontsize=FS_SMALL, color="0.3")

    fig.suptitle("图9  麦克风数量增加时的理想增益与数据项数量\n"
                 "（公式计算，不给出产品选型排名）", fontsize=FS_SUP - 0.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
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
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9.5))
    ax = axes[0]
    ax.plot(t * 1000, rir, color=C_BLUE, lw=0.7)
    ax.axvspan(0, 12, color="0.8", alpha=0.12)
    ax.axvspan(12, 92, color=C_ORANGE, alpha=0.15)
    ax.plot(12, rir[tau], "o", color=C_GREEN, ms=5, zorder=5)
    ax.axvspan(92, 700, color=C_RED, alpha=0.11)
    ax.axvline(12, color="gray", ls="--", lw=0.9, alpha=0.8)
    ax.axvline(92, color="gray", ls="--", lw=0.9, alpha=0.8)
    ax.set_ylim(-2.6, 3.1)
    ax.annotate("指定直达脉冲\n（12 ms；非全局最大峰）", xy=(12, 1.0), xytext=(16, 2.72), fontsize=FS_SMALL, color=C_GREEN,
                arrowprops=dict(arrowstyle="->", color=C_GREEN))
    ax.annotate("早期区：离散反射 + 随机尾\n（直达后 0–80 ms）", xy=(55, 0.7), xytext=(180, 2.15), fontsize=FS_SMALL,
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
    ax.axvline((T60 + tau / fs) * 1000, color="0.45", ls=":", alpha=0.8)
    if np.isfinite(realized_ms):
        ax.axvline(realized_ms, color="k", ls="--", alpha=0.6)
        ax.annotate(f"EDC 首次 −60 dB：绝对时刻 {realized_ms:.0f} ms\n"
                    f"距直达 {realized_ms-tau/fs*1000:.0f} ms；包络名义衰减 {T60*1000:.0f} ms",
                    xy=(realized_ms, -60), xytext=(205, -21), fontsize=FS_SMALL,
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
    ax.plot(d, reverb, color=C_RED, ls="--", lw=1.8, label="混响声参考：本图设为 0 dB")
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
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
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
    reference_amplitude = np.max(np.abs(S))
    Sdb = 20 * np.log10(np.maximum(np.abs(S) / reference_amplitude, 1e-3))
    mesh = ax.pcolormesh(np.arange(S.shape[1]) * hop / fs * 1000, np.fft.rfftfreq(n_fft, 1 / fs) / 1000,
                        Sdb, cmap="viridis", shading="auto", vmin=-60, vmax=0)
    fig.colorbar(mesh, ax=ax, shrink=0.8, label="相对全图峰值 (dB)")
    # 标出一个时频点（格子放大标出）
    f0, fb = 60, 20          # 从0计：第60帧起点480ms，频点20为625Hz
    px, py = f0 * hop / fs * 1000, np.fft.rfftfreq(n_fft, 1 / fs)[fb] / 1000
    ax.plot(px, py, "x", color="w", ms=12, mew=2.8)
    ax.annotate("时频点 $(k,\\ell)$\n同一格的 $M$ 路谱组成快拍\n(c)(d) 另用独立理论数据",
                xy=(px, py), xytext=(100, 5.3), fontsize=FS_LABEL, color="w",
                arrowprops=dict(arrowstyle="->", color="w", lw=1.5))
    ax.set_ylim(0, 8); ax.set_xlabel("帧起点时间 (ms)", fontsize=FS_LABEL)
    ax.set_ylabel("频率 (kHz)", fontsize=FS_LABEL)
    ax.set_title("(b) 单麦语谱图：周期 Hann 窗", fontsize=FS_TITLE)
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
    ax.plot(np.arange(3, M + 1), evals_db[2:], "s", color=C_BLUE, ms=5,
            markerfacecolor="white", zorder=5)
    ax.axhline(0, color=C_BLUE, ls="--", lw=1.2)
    ax.annotate("2个大特征值\n→ 信号子空间(2个源)", xy=(2, evals_db[1] - 1),
                xytext=(3.0, 13), fontsize=FS_LABEL, color=C_RED,
                arrowprops=dict(arrowstyle="->", color=C_RED))
    ax.annotate("6个小特征值≈噪声底(0 dB)\n→ 噪声子空间", xy=(6.5, 0), xytext=(3.2, 4.5),
                fontsize=FS_LABEL, color=C_BLUE, arrowprops=dict(arrowstyle="->", color=C_BLUE))
    ax.set_xlabel("特征值序号"); ax.set_ylabel("相对噪声方差的特征值 (dB)")
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
                                        ("互质阵：17个滞后；中心连续13个", coprime, C_GREEN)]):
        lags = coarray(pos)
        ax.scatter(lags, np.full(len(lags), 2 - y, dtype=float), marker="|", s=400, c=c, lw=2.5)
        ax.text(-11.5, 2 - y + 0.32, name, fontsize=FS_LABEL, va="center", color=c)
    ax.scatter([-7, 7], [0, 0], facecolors="none", edgecolors="0.35", s=65, marker="o", zorder=7)
    ax.text(0, -0.36, "空圈：缺少±7；范围−9～9", ha="center", fontsize=FS_SMALL, color="0.3")
    ax.set_xticks([-11, -9, -7, -5, 0, 5, 7, 9, 11])
    ax.set_xlim(-13, 12.5); ax.set_ylim(-0.7, 2.75)
    ax.set_yticks([]); ax.set_xlabel("差分滞后（单位 λ/2）")
    ax.grid(axis="x", ls=":", alpha=0.5)
    ax.set_title("(b) 差分协同阵：所有有序麦对产生有符号差分位置", fontsize=11)
    fig.suptitle("图10  六麦几何与有符号差分位置（按坐标枚举）", fontsize=13)
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
    fig = plt.figure(figsize=(9.5, 8.6), layout="constrained")
    grid = fig.add_gridspec(2, 1, height_ratios=(1.05, 1.0))
    ax = fig.add_subplot(grid[0])
    ax.axis("off"); ax.set_xlim(0, 14); ax.set_ylim(0, 6.6)

    def box(x, y, w, h, label, color="#dbe9f6"):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=color, ec=C_MAIN, lw=1.2, zorder=3))
        ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=FS_SMALL, zorder=4)

    def arrow(a, b, color=C_BLUE, style="-"):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14,
                                    color=color, lw=1.5, ls=style, zorder=2))

    box(.2, 3.5, 2.1, 1.1, "已对齐参考\nx(n)", "#f6e5db")
    box(3.4, 3.5, 2.4, 1.1, "旧抽头 ŵ(n)\n预测线性回声")
    box(6.7, 3.5, 1.8, 1.1, "回声副本\nŷ(n)")
    box(10.5, 3.5, 3.1, 1.1, "先验残差 e(n)\n=d(n)−ŷ(n)", "#e8f6db")
    box(8.25, 5.3, 2.8, .8, "麦克风观测 d(n)", "#f6dbdb")
    ax.add_patch(Circle((9.5, 4.05), .33, fc="white", ec=C_MAIN, lw=1.3))
    ax.text(9.5, 4.05, "Σ", ha="center", va="center", fontsize=FS_LABEL)
    ax.text(9.0, 4.32, "−", fontsize=FS_SMALL)
    ax.text(9.68, 4.72, "+", fontsize=FS_SMALL)
    for a,b in [((2.3,4.05),(3.4,4.05)), ((5.8,4.05),(6.7,4.05)),
                ((8.5,4.05),(9.16,4.05)), ((9.84,4.05),(10.5,4.05)),
                ((9.5,5.3),(9.5,4.39))]:
        arrow(a,b)
    box(3.4, 1.3, 2.4, 1., "NLMS 系数更新\n读取 x 与 e")
    box(7.4, 1.3, 2.9, 1., "检测/外部控制\n决定是否冻结", "#f6e5db")
    # 输入、残差、控制分支互相独立；输出从不被冻结开关切断。
    ax.plot([1.25,1.25,3.4], [3.5,1.8,1.8], color=C_BLUE, lw=1.5)
    arrow((3.05,1.8),(3.4,1.8))
    ax.plot([12.05,12.05,5.2], [3.5,.65,.65], color=C_GREEN, lw=1.5)
    arrow((5.2,.65),(5.2,1.3),C_GREEN)
    ax.text(11.1,.85,"残差 e",fontsize=FS_SMALL,color=C_GREEN)
    arrow((4.6,2.3),(4.6,3.5),C_PURPLE)
    ax.text(4.75,2.65,"新抽头供下一样本",fontsize=FS_SMALL,color=C_PURPLE)
    arrow((7.4,1.8),(5.8,1.8),C_ORANGE,"--")
    # 简图用独立命名输入端，避免画穿其他框；同名 x/d 是上方同一信号。
    ax.text(8.25,2.95,"x",fontsize=FS_SMALL,color=C_BLUE,ha="center")
    ax.text(9.5,2.95,"d",fontsize=FS_SMALL,color=C_RED,ha="center")
    arrow((8.25,2.85),(8.25,2.3),C_BLUE)
    arrow((9.5,2.85),(9.5,2.3),C_RED)
    ax.text(7,.05,"控制框可由 x、d 检测；下图实验直接给定干扰真值，不测检测准确率。",
            ha="center",fontsize=FS_SMALL,color=".25")
    ax.set_title("(a) 先预测输出，再用参考、残差和控制更新抽头",fontsize=FS_TITLE)

    ax = fig.add_subplot(grid[1])
    t_axis, erle, window_mean = fig18_erle_simulation()
    ax.plot(t_axis,erle,color=C_BLUE,lw=1.6,label="线性残差的块 ERLE（单讲区）")
    ax.axvspan(.8,1.2,color=C_RED,alpha=.1)
    ax.annotate(f"指定窗块 dB 均值≈{window_mean:.0f} dB\n（块起点 0.45<t<0.8 s）",
                xy=(.62,window_mean),xytext=(.08,36),va="top",fontsize=FS_SMALL,color=C_BLUE,
                arrowprops=dict(arrowstyle="->",color=C_BLUE))
    ax.text(1.,4.,"合成近端干扰（非语音）\n按真值冻结；本区间不评分",fontsize=FS_SMALL,
            color=C_RED,ha="center")
    ax.set_xlabel("时间 (s)",fontsize=FS_LABEL);ax.set_ylabel("ERLE (dB)",fontsize=FS_LABEL)
    ax.set_ylim(-10,40);ax.set_xlim(0,1.6)
    ax.set_title("(b) 16 kHz、128 抽头、μ=0.5 的单次固定种子仿真",fontsize=FS_TITLE)
    ax.legend(fontsize=FS_SMALL,loc="lower right");ax.grid(ls=":",alpha=.5)
    fig.suptitle("图18 声学回声消除：输出、更新与控制分开检查",fontsize=FS_SUP)
    save(fig,"fig18_aec.png")

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
    value = np.asarray(relative_loading)
    if value.ndim != 0 or value.dtype.kind not in "iuf" or not np.isfinite(value) or value < 0:
        raise ValueError("relative_loading 必须是有限非负实标量")
    if weighted_cross.size == 0 or not np.all(np.isfinite(weighted_covariance)) or not np.all(np.isfinite(weighted_cross)):
        raise ValueError("WPE 统计量必须非空且有限")
    magnitude = max(np.max(np.abs(weighted_covariance.real)),
                    np.max(np.abs(weighted_covariance.imag)))
    if magnitude == 0:
        if np.any(weighted_cross):
            raise ValueError("零协方差不能配非零互相关")
        return np.zeros_like(weighted_cross), 0.0
    covariance = (weighted_covariance.real / magnitude
                  + 1j * (weighted_covariance.imag / magnitude))
    if not np.allclose(covariance, covariance.conj().T, rtol=1e-12, atol=1e-14):
        raise ValueError("协方差必须是厄米矩阵")
    if np.linalg.eigvalsh(covariance)[0] < -1e-12 * weighted_cross.size:
        raise ValueError("协方差必须为半正定矩阵")
    mean_diagonal = np.mean(np.diag(covariance).real)
    if mean_diagonal <= 0:
        raise ValueError("非零协方差必须具有正的平均对角线")
    loading_scaled = float(value) * mean_diagonal
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        loading = loading_scaled * magnitude
        loaded = covariance + loading_scaled * np.eye(weighted_cross.size)
        cross = weighted_cross.real / magnitude + 1j * (weighted_cross.imag / magnitude)
    if not np.isfinite(loading) or not np.all(np.isfinite(loaded)) or not np.all(np.isfinite(cross)):
        raise ValueError("加载后的统计量不能由当前浮点类型表示")
    solve_scale = max(np.max(np.abs(loaded.real)), np.max(np.abs(loaded.imag)))
    if solve_scale > np.sqrt(np.finfo(float).max):
        loaded = loaded.real / solve_scale + 1j * (loaded.imag / solve_scale)
        cross = cross.real / solve_scale + 1j * (cross.imag / solve_scale)
    filt = np.linalg.solve(loaded, cross)
    if not np.all(np.isfinite(filt)):
        raise ValueError("WPE 滤波器不能由当前浮点类型表示")
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
    if isinstance(K, (bool, np.bool_)) or not isinstance(K, (int, np.integer)) or K < 0:
        raise ValueError("K 必须为非负整数")
    if isinstance(delay, (bool, np.bool_)) or not isinstance(delay, (int, np.integer)) or delay < 1:
        raise ValueError("delay 必须为正整数")
    if isinstance(iters, (bool, np.bool_)) or not isinstance(iters, (int, np.integer)) or iters < 0:
        raise ValueError("iters 必须为非负整数")
    if not np.all(np.isfinite(Y)):
        raise ValueError("Y 必须只含有限复数")
    F, T = Y.shape
    if K == 0 or iters == 0 or T == 0 or F == 0 or T <= delay + K - 1:
        return Y.copy()
    # 每个频点先用实部/虚部的共同尺度归一化，避免求模平方时上溢或下溢。
    # X 与其功率共同缩放不改变正规方程；结果最后恢复原来的幅度单位。
    scales = np.maximum(np.max(np.abs(Y.real), axis=1), np.max(np.abs(Y.imag), axis=1))
    active = scales > 0
    safe_scales = np.where(active, scales, 1.0)
    normalized = Y.real / safe_scales[:, None] + 1j * (Y.imag / safe_scales[:, None])
    X = normalized.copy()
    t0, Ypast, ycur = wpe_past_frames(normalized, K, delay)
    peak_power = np.max(np.abs(normalized) ** 2, axis=1, keepdims=True)
    lam_floor = 1e-5 * peak_power
    for _ in range(iters):
        lam = np.maximum(smooth_power_valid(np.abs(X) ** 2, width=5), lam_floor)
        G = np.zeros((F, K), dtype=complex)
        for f in np.flatnonzero(active):
            P = Ypast[f]
            # 对该频点全部权重乘同一个正数，同时缩放相对加载，不改变解。
            ww = np.min(lam[f, t0:]) / lam[f, t0:]
            Rw = (P * ww[None, :]) @ P.conj().T
            rw = (P * ww[None, :]) @ ycur[f].conj()
            G[f], _ = solve_wpe_filter(Rw, rw, relative_loading=1e-6)
        X[:, t0:] = ycur - np.einsum("fk,fkt->ft", G.conj(), Ypast)
    with np.errstate(over="ignore", invalid="ignore"):
        output = X * safe_scales[:, None]
    if not np.all(np.isfinite(output)):
        raise ValueError("WPE 输出不能由当前浮点类型表示")
    output[:, :t0] = Y[:, :t0]
    return output


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
    Yr = stft_analysis(rev, win, hop)
    # 参考延迟至接收时轴；麦克风及 WPE 输出不前移，传播时延仍然保留。
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
    grid = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 0.62])
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
        ax.set_ylim(0, 2.5); ax.set_xlabel("接收时轴上的帧起点 (ms)", fontsize=FS_LABEL)
        ax.set_ylabel("频率 (kHz)", fontsize=FS_LABEL)
        ax.set_title(title, fontsize=FS_TITLE)
    axes[0].text(0.98, 0.94, "三图的高频暗区：合成信号能量很低",
                 transform=axes[0].transAxes, fontsize=FS_SMALL, color="white",
                 ha="right", va="top")
    fig.suptitle("图21  WPE 去混响效果（本书单通道仿真）", fontsize=FS_SUP)
    footer.text(
        0.5, 0.5,
        f"安静帧能量占本信号总能量：(b) {rev_quiet_db:.2f} dB；(c) {wpe_quiet_db:.2f} dB\n"
        f"活动帧、单一复增益对齐 NMSE：(b) {rev_nmse_db:.2f} dB；(c) {wpe_nmse_db:.2f} dB\n"
        "以上指标统计 0～8 kHz 全部频点；图仅显示 0～2.5 kHz。\n"
        "16 kHz、1.4 s、种子 21001；直达延迟 10 ms；衰减参数 0.6 s；DRR=6 dB。\n"
        "对称 Hann 512 点、帧移 128 点；Δ=6、K=10、3 次迭代；三图共用谱幅参考。",
        ha="center", va="center", fontsize=FS_SMALL)
    cb = fig.colorbar(mesh, ax=axes, shrink=0.85, pad=0.015, label="相对谱能量 (dB)")
    cb.ax.tick_params(labelsize=FS_SMALL + 1)
    cb.set_label("相对谱能量 (dB)", fontsize=FS_LABEL + 2)
    cb.outline.set_linewidth(1.3)
    fig._footer_axis = footer
    save(fig, "fig21_wpe.png")
    report = {
        "schema_version": 1,
        "generator": "scripts/make_figures.py::fig_wpe",
        "source_sha256": source_script_digest(),
        "data_type": "mathematical synthetic signal and impulse response; not real speech or measured RIR",
        "parameters": {"sample_rate_hz": fs, "duration_seconds": 1.4,
            "seed": FIGURE_SEEDS["wpe"], "rir_length_samples": L,
            "direct_delay_samples": direct_index, "decay_parameter_seconds": T60,
            "drr_db": drr_db, "fft_size": n_fft, "hop_samples": hop,
            "window": "symmetric Hann (numpy.hanning)", "center": False,
            "padding": False, "taps": prediction_order, "delay_frames": prediction_delay,
            "iterations": 3, "power_smoothing": "5 frames centered, truncate and renormalize at boundaries",
            "relative_power_floor": 1e-5, "relative_loading": 1e-6},
        "time_axis": "Reference delayed 160 samples; input and output stay on receiver timeline; no propagation compensation",
        "measurement": {"frequency_bins": int(Yr.shape[0]), "frequency_range_hz": [0, fs//2],
            "display_frequency_range_hz": [0, 2500], "frame_count": int(Yr.shape[1]),
            "last_frame_start_ms": float(tms[-1]), "active_frames": int(active.sum()),
            "quiet_frames": int(quiet.sum()), "other_frames": int((~(active|quiet)).sum()),
            "active_threshold_fraction_of_reference_peak": .10,
            "quiet_threshold_fraction_of_reference_peak": .01,
            "nmse_alignment": "one complex gain over all active frame-frequency coefficients"},
        "results": {}
    }
    for name, spectrum, ratio, nmse in [("input", Yr, rev_quiet_db, rev_nmse_db),
                                      ("output", Xd, wpe_quiet_db, wpe_nmse_db)]:
        report["results"][name] = {
            "quiet_energy": float(np.sum(np.abs(spectrum[:, quiet])**2)),
            "total_energy": float(np.sum(np.abs(spectrum)**2)),
            "quiet_energy_ratio_db": float(ratio), "active_scale_aligned_nmse_db": float(nmse)}
    destination = Path(__file__).resolve().parents[1] / "codes/reports/figure21_wpe.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


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
    ax.set_xlim(-1.9, 3.6); ax.set_ylim(-3.4, 3.6)
    ax.annotate("正前方 0°", xy=(0, 2.15), xytext=(0, 1.3),
                ha="center", fontsize=FS_SMALL,
                bbox=dict(fc="white", ec="none", alpha=0.9, pad=1),
                arrowprops=dict(arrowstyle="->", color=C_MAIN))
    ax.text(1.2, 1.25, r"向右为正 $\theta$", fontsize=FS_SMALL,
            bbox=dict(fc="white", ec="none", alpha=0.9, pad=1))
    ax.text(-1.85, -3.05, r"$\mathrm{ITD}=t_L-t_R$：右耳先到时为正", fontsize=FS_SMALL)
    ax.set_title("(a) 俯视图：方位与到达时差的正号约定", fontsize=11)
    ax = fig.add_subplot(3, 1, 2)
    az = np.linspace(-90, 90, 400)
    a_head, c = 0.0875, 343
    itd = (a_head / c) * (np.deg2rad(az) + np.sin(np.deg2rad(az))) * 1e6
    ax.plot(az, itd, color=C_BLUE, lw=2, label="刚性球射线近似")
    free_itd = 2 * a_head / c * np.sin(np.deg2rad(az)) * 1e6
    ax.plot(az, free_itd, color=C_ORANGE, lw=2, ls="--", label="无遮挡双点，间距 17.5 cm")
    ax.legend(loc="lower right", fontsize=FS_SMALL)
    ax.axhline(0, color="k", lw=0.6)
    max_itd_us = float(itd[-1])
    ax.axhline(max_itd_us, color="gray", ls="--", lw=0.8)
    ax.annotate(f"最大值 ≈ {max_itd_us:.0f} μs（正侧面 90°）",
                xy=(90, max_itd_us), xytext=(-5, 500), fontsize=FS_SMALL,
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.annotate("正前方偏 1° ≈ 9 μs\n（几何换算，不是听觉阈值）", xy=(1, 9), xytext=(-64, 300), fontsize=FS_SMALL,
                arrowprops=dict(arrowstyle="->", color=C_RED), color=C_RED)
    ax.set_xlabel("声源方位角 θ (°)，正前方为 0°"); ax.set_ylabel("左耳减右耳到达时差 (μs)")
    ax.set_title("(b) 球头与无遮挡双点：a = 8.75 cm，c = 343 m/s", fontsize=11)
    ax.grid(ls=":", alpha=0.5)
    ax = fig.add_subplot(3, 1, 3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    rows = [
        (0.76, C_BLUE, "细结构 ITD", "纯音相位差；辨别能力随频率与任务变化。\n约 1.4 kHz 是特定纯音实验的结果。"),
        (0.46, C_RED, "ILD", "比较左右耳声级；高频头影通常更明显。\n低频也可能有声级差，没有统一起始频率。"),
        (0.16, C_GREEN, "包络 ITD", "调制高频声的包络也可携带时差。\n不能用纯音的高频限制排除这种线索。"),
    ]
    for y, color, label, detail in rows:
        ax.add_patch(plt.Rectangle((0.01, y - 0.125), 0.98, 0.25,
                                  fc=color, ec=color, lw=1, alpha=0.10))
        ax.text(0.04, y, label, color=color, fontsize=FS_LABEL,
                va="center", fontweight="bold")
        ax.text(0.24, y, detail, fontsize=FS_SMALL, va="center")
    ax.set_title("(c) 三种实验线索：不能画成统一的频率开关", fontsize=FS_TITLE)
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
    """返回统计量与逐次结果；任何计算异常都中止，不静默删试次。

    同一 trial 的源在九条件共用；固定 T60 改 DRR 时还共用两条随机尾
    和截帧位置。改变 T60 会改变随机数消耗，不能称两条尾及截帧均配对。
    """
    if not isinstance(trials, (int, np.integer)) or trials < 1:
        raise ValueError("trials must be a positive integer")
    t60_values = np.asarray(t60_values, dtype=float)
    drr_values = np.asarray(drr_values, dtype=float)
    shape = (drr_values.size, t60_values.size)
    rate = np.empty(shape); low = np.empty(shape); high = np.empty(shape)
    median = np.empty(shape); q1 = np.empty(shape); q3 = np.empty(shape)
    success_count = np.empty(shape, dtype=int)
    trial_correct = np.empty((*shape, trials), dtype=bool)
    trial_contrasts = np.empty((*shape, trials))
    seeds = 2000 + np.arange(trials) * 17
    for i, drr_db in enumerate(drr_values):
        for j, t60 in enumerate(t60_values):
            results = [gcc_reverb_trial(t60, drr_db, int(seed))[2:]
                       for seed in seeds]
            successes = int(sum(item[0] for item in results))
            lo, hi = wilson_interval(successes, trials)
            contrasts = np.asarray([item[1] for item in results])
            success_count[i, j] = successes
            trial_correct[i, j] = [item[0] for item in results]
            trial_contrasts[i, j] = contrasts
            rate[i, j], low[i, j], high[i, j] = successes / trials, lo, hi
            median[i, j] = np.median(contrasts)
            q1[i, j], q3[i, j] = np.percentile(contrasts, [25, 75])
    return {"rate": rate, "low": low, "high": high,
            "median": median, "q1": q1, "q3": q3,
            "successes": success_count, "failures": trials - success_count,
            "exceptions": np.zeros(shape, dtype=int), "trials": int(trials),
            "trial_correct": trial_correct, "trial_contrasts": trial_contrasts,
            "seeds": seeds}


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
    return stats


def write_gcc_reverb_report(stats, output):
    """保存与图13同一次计算的逐次值；仅显式生成时写入。"""
    report = {
        "schema_version": 1,
        "generator": "scripts/make_figures.py::gcc_reverb_statistics",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "configuration": {"sample_rate_hz": 16000, "signal_samples": 16000,
            "frame_samples": 1024, "direct_delay_samples": 9, "true_gcc_lag_samples": -9,
            "fft_samples": 4096, "search_lag_samples": [-64, 64],
            "source_lowpass_hz": 6000, "phat_additive_epsilon": 1e-10,
            "row_drr_db": [6.0, 0.0, -6.0], "column_t60_s": [0.1, 0.4, 0.8],
            "success_tolerance_samples": 1, "wilson_z": 1.96,
            "percentile_method": "numpy default linear", "additive_noise": False,
            "source_peak_normalized": True, "frame_window": "rectangular",
            "frame_start_min": 0, "frame_start_max_inclusive": 14974,
            "tail_duration_factor": 1.1, "amplitude_decay_coefficient": 6.91,
            "contrast_definition": "max(g)/(median(abs(g))+1e-12)"},
        "randomness": "Each seed uses source seed+1000 shared across conditions. At fixed T60, DRR conditions share both random tails and frame start. Changing T60 changes RNG consumption; second tail and frame start are not paired.",
        "model": "Finite 1 s zero-start source; causal full convolution then prefix crop; independent exponential random tails with direct energy 1 and prescribed total tail energy. Not a measured room or a controlled isolation of tail duration.",
        "exception_policy": "Any computation exception aborts generation; no trial is omitted.",
        "statistics": {key: value.tolist() if isinstance(value, np.ndarray) else value
                       for key, value in stats.items()},
    }
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


# ----------------------------------------------------------------------
# 图4 时延 = 相位旋转：复数表示的几何直觉
# ----------------------------------------------------------------------
def fig_delay_phase():
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9.8))
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
    arrow_start = 0.625  # ms；原正弦的下降沿−1/√2，与延迟后的同相位点相连
    ax.annotate("", xy=(arrow_start + tau * 1e3, -0.707), xytext=(arrow_start, -0.707),
                arrowprops=dict(arrowstyle="->", color="k", lw=1.5))
    ax.text(arrow_start + tau * 500, -1.15, "τ≈0.117 ms", fontsize=11, ha="center")
    ax.set_xlabel("时间 (ms)"); ax.set_ylabel("幅度")
    ax.set_title("(a) 1 kHz 正弦：同相位点向右平移", fontsize=11)
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
    ax.text(1.05, 0.06, "归一化参考 1", fontsize=FS_LABEL, color=C_BLUE)
    ax.text(0.05, -1.18, r"相对相位因子 $e^{-j2\pi f\tau}$", fontsize=FS_LABEL,
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
    ax.annotate("1 kHz → −42°\n2 kHz → −84°\n本坐标斜率：\n−360000τ ≈ −41.98 °/kHz", xy=(3.0, -125),
                xytext=(3.8, -153), fontsize=FS_SMALL,
                arrowprops=dict(arrowstyle="->", color="k"))
    ax.set_xlabel("频率 (kHz)"); ax.set_ylabel("相位 (°)")
    ax.set_title("(c) 同一延迟在不同频率：相位-频率是直线", fontsize=11)
    ax.grid(ls=":", alpha=0.5)
    fig.suptitle("图4  时延为什么变成 $e^{-j2\\pi f\\tau}$：几何直觉\n"
                 "（4 cm 麦距；端射方向达到最大麦间时延 τ=d/c≈0.117 ms）", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.0)
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
    half_power_db = 10 * np.log10(0.5)
    ax.axhline(half_power_db, color="gray", ls="--", lw=1)
    main_lobe = np.abs(np.sin(tr) - np.sin(np.deg2rad(30))) <= 2 / M
    right_sidelobe = ~main_lobe & (th > 30)
    sidelobe_index = np.argmax(np.where(right_sidelobe, Bdb, -np.inf))
    sidelobe_db = float(Bdb[sidelobe_index])
    ax.axhline(sidelobe_db, color=C_ORANGE, ls="--", lw=1)
    ax.set_ylim(-40, 13); ax.set_xlim(-90, 90)
    # HPBW 实测
    i0 = np.argmax(Bdb)
    left = np.where(Bdb[:i0] < half_power_db)[0][-1]
    right = i0 + np.where(Bdb[i0:] < half_power_db)[0][0]
    left_angle = np.interp(half_power_db, Bdb[left:left+2], th[left:left+2])
    right_angle = np.interp(half_power_db, Bdb[right-1:right+1][::-1], th[right-1:right+1][::-1])
    ax.annotate("", xy=(right_angle, half_power_db), xytext=(left_angle, half_power_db),
                arrowprops=dict(arrowstyle="<->", color=C_RED, lw=1.5))
    ax.text(0.03, 0.96, f"HPBW ≈ {right_angle-left_angle:.1f}°\n两侧 −3.01 dB 交点间角宽",
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
    ax.plot(th, pattern(0.5, 60), color=C_BLUE, lw=1.8, ls="--", label="d = λ/2（指向60°）")
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
    fig = plt.figure(figsize=(9.5, 10.4), layout="constrained")
    grid=fig.add_gridspec(2,1,height_ratios=(1.15,1))

    def setup(axis,title):
        axis.axis("off");axis.set_xlim(0,14);axis.set_ylim(0,8)
        axis.set_title(title,fontsize=FS_TITLE)

    def box(axis,x,y,w,h,label,fill="#dbe9f6"):
        axis.add_patch(plt.Rectangle((x,y),w,h,fc=fill,ec=C_MAIN,lw=1.2,zorder=3))
        axis.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=FS_SMALL,zorder=4)

    def arrow(axis,a,b,color=C_BLUE,style="-"):
        axis.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=14,
                                      color=color,lw=1.5,ls=style,zorder=2))

    ax=fig.add_subplot(grid[0]);setup(ax,"(a) 参考取点、物理路径、相对时差与自适应更新")
    box(ax,.2,6.2,2.,1.,"播放参考 x\n已知数字取点","#f6e5db")
    box(ax,3.,6.2,4.5,1.,"等效线性路径 h\n器件小信号响应 + 房间 + 采集")
    box(ax,10.1,6.2,3.5,1.,"麦克风观测 d\n=x*h+s+v","#f6dbdb")
    arrow(ax,(2.2,6.7),(3.,6.7));arrow(ax,(7.5,6.7),(8.27,6.7))
    ax.add_patch(Circle((8.6,6.7),.32,fc="white",ec=C_MAIN,lw=1.2))
    ax.text(8.6,6.7,"Σ",ha="center",va="center",fontsize=FS_LABEL)
    arrow(ax,(8.93,6.7),(10.1,6.7))
    ax.text(8.6,5.62,"近端 s + 噪声 v",fontsize=FS_SMALL,ha="center",color=C_RED)
    arrow(ax,(8.6,5.92),(8.6,6.37),C_RED)

    box(ax,.2,3.5,2.4,1.1,"时差估计与对齐\n输出 x_a")
    box(ax,4.,3.5,2.8,1.1,"线性路径估计\nŷ = ŵ * x_a")
    box(ax,10.1,3.5,3.5,1.1,"线性残差 e=d−ŷ\n送后续抑制/任务","#e8f6db")
    arrow(ax,(1.2,6.2),(1.2,4.6))
    # 麦克风观测经独立命名端口送时差估计，并非由参考自身估时差。
    ax.text(2.3,5.5,"d",color=C_RED,ha="center",fontsize=FS_SMALL)
    arrow(ax,(2.3,5.25),(2.3,4.6),C_RED)
    arrow(ax,(2.6,4.05),(4.,4.05))
    ax.add_patch(Circle((8.6,4.05),.32,fc="white",ec=C_MAIN,lw=1.2))
    ax.text(8.6,4.05,"Σ",ha="center",va="center",fontsize=FS_LABEL)
    ax.text(8.05,4.28,"−",fontsize=FS_SMALL)
    ax.text(8.82,4.69,"+",fontsize=FS_SMALL)
    arrow(ax,(6.8,4.05),(8.27,4.05));arrow(ax,(8.93,4.05),(10.1,4.05))
    ax.plot([11.85,11.85,9.25],[6.2,5.1,5.1],color=C_RED,lw=1.5)
    arrow(ax,(9.25,5.1),(8.7,4.37),C_RED)

    box(ax,4.,1.1,2.8,1.,"自适应更新\n读取 x_a、e")
    box(ax,8.3,1.1,3.5,1.,"DTD / 步长控制\n读取 x_a、d 或 e","#f6e5db")
    ax.plot([1.4,1.4,4.],[3.5,1.6,1.6],color=C_BLUE,lw=1.5)
    arrow(ax,(3.5,1.6),(4.,1.6))
    ax.plot([12.5,12.5,6.],[3.5,.5,.5],color=C_GREEN,lw=1.5)
    arrow(ax,(6.,.5),(6.,1.1),C_GREEN)
    ax.text(12.65,1.6,"e",fontsize=FS_SMALL,color=C_GREEN)
    arrow(ax,(5.4,2.1),(5.4,3.5),C_PURPLE)
    ax.text(5.55,2.65,"下一时刻抽头",fontsize=FS_SMALL,color=C_PURPLE)
    arrow(ax,(8.3,1.6),(6.8,1.6),C_ORANGE,"--")
    for x,label,color in [(8.8,"x_a",C_BLUE),(10.,"d",C_RED),(11.2,"e",C_GREEN)]:
        ax.text(x,2.95,label,ha="center",fontsize=FS_SMALL,color=color)
        arrow(ax,(x,2.75),(x,2.1),color)
    ax.text(7.,.02,"同名端口连接同一信号；ŵ 表示对齐后路径，h 表示从原参考到麦克风的总路径。",
            ha="center",fontsize=FS_SMALL,color=".25")

    ax=fig.add_subplot(grid[1]);setup(ax,"(b) 一类混合流程：线性估计 + 学习型残余抑制")
    box(ax,.2,4.8,2.3,1.,"麦克风观测 d","#f6dbdb")
    box(ax,3.4,4.8,2.5,1.,"线性 AEC\nPBFDAF / FDKF")
    box(ax,6.8,4.8,2.8,1.,"学习型残余抑制\n使用 e 与可选参考","#fde3c8")
    box(ax,10.5,4.8,3.1,1.,"处理后输出\n另测近端损伤","#e8f6db")
    for a,b in [((2.5,5.3),(3.4,5.3)),((5.9,5.3),(6.8,5.3)),((9.6,5.3),(10.5,5.3))]:
        arrow(ax,a,b)
    ax.text(6.35,5.55,"e",ha="center",fontsize=FS_SMALL,color=C_GREEN)
    box(ax,.2,2.25,2.3,1.,"播放参考 x","#f6e5db")
    box(ax,3.4,2.25,2.5,1.,"时差估计/对齐\n使用 x 与 d")
    arrow(ax,(2.5,2.75),(3.4,2.75));arrow(ax,(4.65,3.25),(4.65,4.8))
    ax.plot([1.35,1.35,3.7],[4.8,3.9,3.9],color=C_RED,lw=1.5)
    arrow(ax,(3.7,3.9),(3.7,3.25),C_RED)
    box(ax,8.3,2.25,3.5,1.,"自适应控制\n读取 x_a、d、e","#f6e5db")
    for x,label,color in [(8.8,"x_a",C_BLUE),(10.,"d",C_RED),(11.2,"e",C_GREEN)]:
        ax.text(x,4.15,label,ha="center",fontsize=FS_SMALL,color=color)
        arrow(ax,(x,3.95),(x,3.25),color)
    arrow(ax,(8.3,2.75),(7.6,2.75),C_ORANGE,"--")
    ax.text(7.1,3.5,"更新控制",ha="center",fontsize=FS_SMALL,color=C_ORANGE)
    # 控制信号从对齐框外绕到 AEC，避免被误画成控制时差估计器。
    ax.plot([7.6,7.6,6.25,6.25],[2.75,1.65,1.65,4.35],color=C_ORANGE,lw=1.5,ls="--")
    arrow(ax,(6.25,4.35),(5.5,4.8),C_ORANGE,"--")
    # DNN 可使用原参考或对齐参考，由具体模型决定。
    ax.plot([1.35,1.35,8.],[2.25,1.25,1.25],color=C_BLUE,lw=1.2,ls=":")
    ax.plot([8.,8.],[1.25,4.55],color=C_BLUE,lw=1.2,ls=":")
    arrow(ax,(8.,4.55),(8.,4.8),C_BLUE,":")
    ax.text(4.1,1.02,"可选原参考 x（也可按模型改用 x_a）",ha="center",fontsize=FS_SMALL,color=C_BLUE)
    linear_note=ax.text(7.,.5,"线性前级：估计参考可解释的路径；后级仍可能含残余回声、噪声和近端失真。",
                        ha="center",fontsize=FS_SMALL,color=C_BLUE)
    neural_note=ax.text(7.,.02,"学习后级：按模型输入、前视与训练范围配置，在目标条件下测泛化与损伤。",
                        ha="center",fontsize=FS_SMALL,color=C_ORANGE)
    fig._lower_annotation_lanes=(linear_note,neural_note)
    fig.suptitle("图20 AEC 系统：信号、估计与控制的依赖",fontsize=FS_SUP)
    save(fig,"fig20_aec_pipeline.png")

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
    ax.plot(t1, e_lin, color=C_BLUE, lw=1.6, ls="-", marker="o", markevery=8, label="线性回声路径")
    ax.plot(t2, e_nl, color=C_RED, lw=1.6, ls="--", marker="s", markevery=8, label="非线性回声路径（tanh 软削波）")
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
    fig, (ax, support) = plt.subplots(2, 1, figsize=(9.5, 8.2),
                                     gridspec_kw={"height_ratios": [1.4, 1]})
    ax.axis("off"); ax.set_xlim(-.7, 7.7); ax.set_ylim(-1.65, 2.35)
    for index, lag in enumerate(range(7, -1, -1)):
        selected = lag >= 3
        color = C_BLUE if selected else (C_RED if lag == 0 else "0.45")
        ax.add_patch(plt.Rectangle((index-.43, .15), .86, .75,
                     fc="#dbe9f6" if selected else ("#f6dbdb" if lag == 0 else "#eeeeee"),
                     ec=color, lw=1.4, hatch="//" if lag in (1, 2) else None))
        ax.text(index, .53, "$t$" if lag == 0 else f"$t-{lag}$",
                ha="center", va="center", fontsize=FS_TITLE, color=color)
        if selected:
            ax.text(index, -.05, f"$q={lag-3}$", ha="center", va="top", fontsize=FS_SMALL)
    ax.annotate("", xy=(-.43, 1.2), xytext=(4.43, 1.2),
                arrowprops=dict(arrowstyle="<->", color=C_BLUE))
    ax.text(2, 1.4, "选入回归器的 K=5 帧：从最近的 q=0 向过去排列",
            ha="center", va="bottom", fontsize=FS_SMALL, color=C_BLUE)
    ax.annotate("", xy=(4, 2.0), xytext=(7, 2.0),
                arrowprops=dict(arrowstyle="<->", color="0.35"))
    ax.text(5.5, 2.12, "帧索引差 Δ=3", ha="center", va="bottom", fontsize=FS_SMALL)
    current_note = ax.text(7, -.08, "当前帧", ha="center", va="top",
                           color=C_RED, fontsize=FS_SMALL)
    ax.text(5.5, -.52, "不选 t−2、t−1", ha="center", fontsize=FS_SMALL, color="0.35")
    formula_note = ax.text(3.5, -.9,
        r"预测：$\hat r(t,f)=\sum_{q=0}^{K-1}G_q^*(f)X(t-\Delta-q,f)$；输出：$X(t,f)-\hat r(t,f)$",
        ha="center", va="center", fontsize=FS_SMALL, color=C_GREEN)
    ax.text(3.5, -1.42, "回归器只含过去帧；若用整段数据估计 G，算法仍使用未来信息。",
            ha="center", va="center", fontsize=FS_SMALL)
    ax.set_title("(a) 单通道索引示意：Δ=3，K=5", fontsize=FS_TITLE, loc="left", pad=18)
    support.barh(1, 32, left=-24, height=.28, color="#dbe9f6", edgecolor=C_BLUE)
    support.barh(0, 32, left=0, height=.28, color="#f6dbdb", edgecolor=C_RED)
    support.axvspan(0, 8, facecolor="none", edgecolor="0.4", hatch="///", lw=0)
    support.axvline(0, ls="--", color="0.4", lw=1)
    support.axvline(8, ls=":", color="0.4", lw=1)
    support.text(-8, 1, "[−24, 8) ms", ha="center", va="center", fontsize=FS_SMALL)
    support.text(17, 0, "[0, 32) ms", ha="center", va="center", fontsize=FS_SMALL)
    support.annotate("重叠 8 ms（128 点）", xy=(4, .45), xytext=(-22, .45),
                     fontsize=FS_SMALL, va="center", arrowprops=dict(arrowstyle="->"))
    support.set_yticks([0, 1], ["当前帧 t 的窗", "最近历史帧 t−3 的窗"])
    support.set_xticks([-24, -16, -8, 0, 8, 16, 24, 32])
    support.set_xlim(-26, 34); support.set_ylim(-.4, 1.4)
    support.set_xlabel("相对当前窗起点的采样时间 (ms)", fontsize=FS_LABEL)
    support.grid(axis="x", ls=":", alpha=.3)
    support.set_title("(b) 16 kHz，窗长 512 点（32 ms），帧移 128 点（8 ms）\n"
                      "窗起点相距 ΔH=384 点（24 ms），并不意味着波形时间段互不重叠",
                      loc="left", fontsize=FS_TITLE, pad=12)
    fig.suptitle("图24  WPE 的帧索引与分析窗支持区间", fontsize=FS_SUP)
    fig.tight_layout(rect=(0, 0, 1, .94), h_pad=2)
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


def clock_drift_measurements():
    """Measure PCM frame power; compare to a separate analytic clock model."""
    import json
    import wave
    root = OUT.parent / 'codes' / 'audio'
    manifest = root / 'MANIFEST.json'
    records = {r['file']: r for r in json.loads(manifest.read_text())['files']}
    decoded = {}
    for name in ('reference', 'index_mean', 'oracle_mean'):
        path = root / f'clock_{name}.wav'
        if hashlib.sha256(path.read_bytes()).hexdigest() != records[path.name]['sha256']:
            raise ValueError('clock PCM does not match manifest')
        with wave.open(str(path), 'rb') as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()) != (16000, 1, 2, 128000):
                raise ValueError('clock example requires 8 seconds of mono 16 kHz PCM16')
            decoded[name] = np.frombuffer(wav.readframes(128000), dtype='<i2').astype(float) / 32768
    # Interior 20 ms blocks exclude both 20 ms edge fades.
    centers = (np.arange(2, 399) + .5) * .02
    powers = {name: np.mean(x.reshape(400, 320)[2:399] ** 2, axis=1)
              for name, x in decoded.items()}
    return centers, powers, hashlib.sha256(manifest.read_bytes()).hexdigest()


def fig_clock_drift():
    centers, powers, digest = clock_drift_measurements()
    t = np.linspace(0, 8, 1601)
    delay = t * 1e-4 / 1.0001
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.3))
    axes[0].plot(t, delay * 16000, color=C_BLUE, lw=2)
    axes[0].set(xlabel='名义时间 (s)', ylabel='时间差 × 16 kHz (样点)',
                title='(a) 同一索引对应不同采样时刻；100 ppm 持续累积')
    for f, color, style in [(500, C_BLUE, '-'), (1500, C_RED, '--')]:
        envelope = np.abs(np.cos(np.pi * f * delay))
        axes[1].plot(t, envelope, color=color, ls=style, label=f'{f} Hz 解析包络', lw=1.8)
    axes[1].set(xlabel='名义时间 (s)', ylabel='单频幅度比 (无量纲)', ylim=(-.03, 1.08),
                title='(b) 相位差随时间增大；两路平均可抵消同一个目标')
    axes[1].legend(loc='upper left', bbox_to_anchor=(1.01, 1),
                   fontsize=FS_SMALL, framealpha=.92)
    axes[2].plot(centers, powers['index_mean'] / powers['reference'],
                 color=C_RED, label='未同步均值：PCM 20 ms 功率比')
    axes[2].plot(centers, powers['oracle_mean'] / powers['reference'],
                 color=C_GREEN, ls='--', label='理想共同时钟：PCM 功率比')
    axes[2].set(xlabel='名义时间 (s)', ylabel='相对参考功率 (无量纲)', ylim=(-.03, 1.15),
                title='(c) 双音合成 WAV 读回；排除两端淡入淡出，不拟合时延或增益')
    axes[2].text(.3, 1.05, '理想共同时钟：PCM 功率比（虚线）', color=C_GREEN)
    axes[2].text(4.9, .24, '未同步均值：PCM 20 ms 功率比', color=C_RED)
    for ax in axes:
        ax.grid(ls=':', alpha=.35)
        ax.set_xlim(0, 8)
    fig.suptitle('图40  先对齐起点仍会失步：采样时钟偏差的双麦反例\n'
                 '500 / 1500 Hz；16 kHz；8 s；100 ppm；共址、无噪声数学模型', fontsize=FS_SUP)
    fig.tight_layout(rect=(0, 0, .81, .94), h_pad=1.5)
    save(fig, 'fig40_clock_drift.png', {'AudioManifestDigest': digest})


def fig_interpolation_error():
    """Fixed-delay response from algebra, markers measured from published PCM."""
    import json
    import wave
    root = Path(__file__).resolve().parents[1]
    audio = root / 'codes/audio'
    manifest_path = audio / 'MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    params = manifest['groups']['interpolation']['parameters']
    fs = params['sample_rate_hz']
    start, stop = params['scoring_interval_samples']
    n = np.arange(start, stop)
    tones = np.array(params['frequencies_hz'])
    amplitudes = {}
    for label in ('ideal_half', 'linear_half', 'ideal_one', 'linear_twice'):
        with wave.open(str(audio / f'interpolation_{label}.wav'), 'rb') as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (fs, 1, 2):
                raise ValueError('Interpolation figure requires mono PCM16 at the declared rate')
            x = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2') / 32768
        amplitudes[label] = np.array([2 * abs(np.sum(x[start:stop] *
                                       np.exp(-2j*np.pi*f*n/fs))) / len(n) for f in tones])
    f = np.linspace(0, fs/2, 801)
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.2))
    axes[0].plot(f/1000, np.cos(np.pi*f/fs), color=C_BLUE, lw=2,
                 label='一次半采样：解析幅度')
    axes[0].plot(f/1000, np.cos(np.pi*f/fs)**2, color=C_RED, lw=2, ls='--',
                 label='两次半采样：解析幅度')
    ratios = []
    for label, ref, color, marker in [('linear_half', 'ideal_half', C_BLUE, 'o'),
                                     ('linear_twice', 'ideal_one', C_RED, 's')]:
        ratio = amplitudes[label] / amplitudes[ref]
        ratios.append(ratio)
        axes[0].scatter(tones/1000, ratio, color=color, marker=marker, s=55,
                        facecolors='white', zorder=5)
    axes[0].set(xlabel='频率 (kHz)', ylabel='幅度比 (无量纲)', xlim=(0, 8), ylim=(-.03, 1.08),
                title='(a) 线为解析响应；空心圆/方块为导出 PCM 的逐频幅度比')
    axes[0].legend(loc='lower left')
    x = np.arange(2)
    for i, (ratio, color, hatch, label) in enumerate(zip(ratios, (C_BLUE, C_RED), ('', '//'),
                  ('一次半采样', '两次半采样'))):
        db = 20*np.log10(ratio)
        axes[1].bar(x + (i-.5)*.34, db, width=.32, color=color, hatch=hatch, alpha=.8, label=label)
        for xpos, value in zip(x+(i-.5)*.34, db):
            axes[1].text(xpos, value-.8, f'{value:.2f} dB', ha='center', va='top')
    axes[1].set(xticks=x, xticklabels=['500 Hz', '6000 Hz'], ylabel='幅度比的分贝值 (dB)',
                ylim=(-20, 1), title='(b) PCM 读回：各自与同一目标时延的理想输出比较')
    axes[1].legend(loc='lower left')
    for ax in axes:
        ax.grid(axis='y', ls=':', alpha=.35)
        ax.set_axisbelow(True)
    fig.suptitle('图41  线性插值改变高频幅度；两次半采样不等于纯一采样延迟\n'
                 '16 kHz；500 / 6000 Hz 双音；同一导出增益；统计 0.1～1.9 s', fontsize=FS_SUP)
    fig.tight_layout(rect=(0, 0, 1, .93), h_pad=1.8)
    save(fig, 'fig41_interpolation_error.png',
         {'AudioManifestDigest': hashlib.sha256(manifest_path.read_bytes()).hexdigest()})


def fig_gss_flow():
    """Show the separate direction, raw-power and activity paths of teaching GSS."""
    fig, ax = plt.subplots(figsize=(9.5, 8.1))
    ax.set(xlim=(-.2, 9), ylim=(-.65, 9.15))
    ax.axis('off')
    width, height = 2.35, 1.05
    locations = {'pre': (0, 7.55), 'activity': (6.3, 7.55),
                 'raw': (0, 5.2), 'unit': (3.15, 5.2), 'cluster': (6.3, 5.2),
                 'raw_copy': (0, 2.7), 'scm': (3.15, 2.7), 'beam': (6.3, 2.7),
                 'output': (6.3, .05)}
    labels = {'pre': '同步多通道复谱 X\n可选 WPE → Y',
              'activity': '外部说话人活动 a\n背景类始终活动',
              'raw': '处理后的复谱 Y\n保留幅度与相位',
              'unit': '非低能量点单位化\nz = Y / ‖Y‖',
              'cluster': '导引 cACG 迭代\n形状 B、权重 π',
              'raw_copy': '同一复谱 Y\n不做单位范数化',
              'scm': '掩码加权外积\n目标 / 非目标 SCM',
              'beam': '目标主方向 + 参考麦\n加载 MVDR 权重 w',
              'output': '输出 $y = w^H Y$\n非可用频点选参考麦'}
    for name, (x, y) in locations.items():
        fill = '#fff2cc' if name == 'activity' else '#e8f6db' if name in ('cluster', 'scm', 'beam') else '#dbe9f6'
        ax.add_patch(plt.Rectangle((x, y), width, height, facecolor=fill, edgecolor=C_MAIN, lw=1.2))
        ax.text(x+width/2, y+height/2, labels[name], ha='center', va='center', fontsize=11)
    def anchor(name, side):
        x,y=locations[name]
        return {'l':(x,y+height/2),'r':(x+width,y+height/2),
                't':(x+width/2,y+height),'b':(x+width/2,y)}[side]
    def edge(a,b,s='r',t='l',via=(),control=False):
        pts=[anchor(a,s),*via,anchor(b,t)]
        for i,(start,stop) in enumerate(zip(pts[:-1],pts[1:])):
            ax.add_patch(FancyArrowPatch(start,stop,
                arrowstyle='-|>' if i==len(pts)-2 else '-', mutation_scale=14,
                color=C_ORANGE if control else C_BLUE, lw=1.6, ls='--' if control else '-'))
    edge('pre','raw','b','t')
    edge('raw','unit')
    edge('unit','cluster')
    edge('activity','cluster','b','t',control=True)
    edge('raw','raw_copy','b','t')
    edge('raw_copy','scm')
    edge('cluster','scm','b','t',via=((7.475,4.45),(4.325,4.45)))
    ax.text(5.6,4.53,'后验掩码 γ',ha='center',va='bottom',fontsize=11)
    edge('scm','beam')
    edge('beam','output','b','t')
    ax.text(7.65,1.8,'权重 w',ha='left',va='center',fontsize=11)
    edge('raw_copy','output','b','l',via=((1.175,1.55),(2.65,1.55),(2.65,.575)))
    ax.text(4.3,.75,'保留尺度的 Y',ha='center',va='bottom',fontsize=11)
    ax.text(3.05,7.8,'实线：信号 / 统计量\n虚线：外部活动控制',ha='left',va='center',fontsize=11)
    ax.text(.1,-.3,'方向聚类不能恢复声功率；掩码不直接替代空间滤波权重。',ha='left',va='center',fontsize=11)
    fig.suptitle('图42  教学 GSS 的三条依赖：空间方向、原始功率、外部活动',fontsize=FS_SUP)
    fig.tight_layout(rect=(0,0,1,.95))
    save(fig,'fig42_gss_flow.png')


def css_tone_amplitudes(waveform, sample_rate=16000, block_samples=1280):
    """Joint two-tone + DC regression, returning midpoint seconds and amplitudes."""
    x = np.asarray(waveform, dtype=float)
    if x.ndim != 1 or not np.all(np.isfinite(x)) or x.size < block_samples:
        raise ValueError('a finite mono waveform of at least one measurement block is required')
    times = np.arange(block_samples) / sample_rate
    basis = np.column_stack([np.cos(2*np.pi*f*times) for f in (250,625)] +
                            [np.sin(2*np.pi*f*times) for f in (250,625)] + [np.ones(block_samples)])
    count = x.size // block_samples
    coefficients = np.linalg.lstsq(basis,x[:count*block_samples].reshape(count,block_samples).T,rcond=None)[0]
    amplitudes = np.hypot(coefficients[:2],coefficients[2:4]).T
    return (np.arange(count)+.5)*block_samples/sample_rate, amplitudes


def fig_css_overlap():
    """Read published PCM for output amplitudes; keep float matching separate."""
    import wave
    root = Path(__file__).resolve().parents[1]
    manifest_path = root/'codes/audio/MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    config = manifest['groups']['css_overlap']['parameters']
    correlation = np.asarray(config['matching']['absolute_centered_correlation'])
    fig = plt.figure(figsize=(9.5,9.0))
    grid = fig.add_gridspec(3,2,height_ratios=(1.05,1,1),width_ratios=(1,1.2))
    matrix = fig.add_subplot(grid[0,0])
    matrix.imshow(correlation,vmin=0,vmax=1,cmap='Blues')
    for i in range(2):
        for j in range(2):
            matrix.text(j,i,f'{correlation[i,j]:.4f}',ha='center',va='center',
                        color='white' if correlation[i,j]>.6 else C_MAIN,fontsize=13)
    matrix.set(xticks=[0,1],yticks=[0,1],xticklabels=['当前槽0','当前槽1'],
               yticklabels=['前块槽0','前块槽1'],xlabel='当前块的重叠信号',ylabel='前一块的重叠信号',
               title='(a) 浮点重叠区绝对相关（无量纲）')
    note = fig.add_subplot(grid[0,1]);note.axis('off')
    identity_score, swap_score = config['matching']['candidate_mean_scores_identity_swap']
    mapping = config['matching']['current_indices_for_previous']
    note.text(0,.92,f'重叠区为 0.8～1.2 s\n\n交换排列平均分：{swap_score:.4f}\n原顺序平均分：{identity_score:.4f}\n\n只据重叠信号选择 {mapping}\n不向匹配器提供干净参考',
              va='top',fontsize=11,linespacing=1.4)
    for row,(stem,title) in enumerate([
            ('css_overlap_naive','(b) 未关联：槽0在重叠区逐渐换成另一源'),
            ('css_overlap_aligned','(c) 实际关联：槽0保持第一源与其残留串音')],start=1):
        with wave.open(str(root/'codes/audio'/f'{stem}.wav'), 'rb') as wav:
            fs = wav.getframerate()
            if wav.getsampwidth() != 2 or wav.getnchannels() != 2:
                raise ValueError('CSS figure requires two-channel PCM16')
            waves = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(float).reshape(-1,2).T / 32768
        t,amplitudes=css_tone_amplitudes(waves[0],fs)
        ax=fig.add_subplot(grid[row,:])
        ax.axvspan(.8,1.2,color='#bdbdbd',alpha=.25,label='两块重叠区')
        for i,(frequency,color,style,marker) in enumerate([(250,C_BLUE,'-','o'),(625,C_RED,'--','s')]):
            ax.plot(t,amplitudes[:,i],color=color,ls=style,marker=marker,ms=3.5,
                    label=f'{frequency} Hz：PCM测幅')
        ax.set(title=title,xlim=(0,2),ylim=(0,.14),xlabel='时间 (s)',ylabel='槽0音调幅度\n(数字满幅单位)')
        ax.grid(ls=':',alpha=.35)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=11)
    fig.suptitle('图43  相关匹配修正跨块槽位交换：受控两音调实验\n'
                 '16 kHz / 2 s；10%人为串音；下两图按80 ms分块联合测幅',fontsize=FS_SUP)
    fig.tight_layout(rect=(0,.035,1,.92),h_pad=1.8)
    save(fig,'fig43_css_overlap.png',{'AudioManifestDigest':hashlib.sha256(manifest_path.read_bytes()).hexdigest()})



def fig_tracking_audio():
    """Plot only the analysis recomputed from the independently exported PCM."""
    manifest_path = Path(__file__).resolve().parents[1]/'codes/tracking_audio/MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    frames = manifest['pcm_analysis']['frames']
    t = np.asarray(frames['state_time_s'])
    valid = np.asarray(frames['observation_valid'],dtype=bool)
    values = lambda key: np.asarray([np.nan if x is None else x for x in frames[key]],dtype=float)
    fig, axes = plt.subplots(3,1,figsize=(9.5,9),sharex=True)
    axes[0].plot(t,values('truth_angle_deg'),color='black',lw=1.8,label='阵列中心迟滞方向')
    axes[0].plot(t,values('observation_angle_deg'),'.',color=C_BLUE,ms=3,label='PCM → GCC-PHAT → 方位')
    axes[0].plot(t,values('filtered_angle_deg'),'--',color=C_RED,lw=1.6,label='Kalman 后验 / 缺测预测')
    axes[0].set(ylabel='方位角 (°)',title='(a) 音频观测形成后才进入滤波器')
    axes[0].legend(fontsize=11,loc='upper left')
    axes[1].plot(t,values('angle_variance_deg2'),color=C_PURPLE,lw=1.6)
    axes[1].set(ylabel='角度方差 (°²)',title='(b) 状态时刻的不确定度：缺测时只预测')
    axes[2].plot(t,values('truth_tau10_samples'),color='black',lw=1.8,label='同一发射事件的传播时差')
    axes[2].plot(t,values('observation_tau10_samples'),'.',color=C_BLUE,ms=3,label='GCC 亚采样峰')
    axes[2].set(ylabel='通道1−通道0时差 (样本)',xlabel='接收帧中心 / state_time (s)',
                title='(c) 左右麦时差与声源方向的符号相反')
    axes[2].legend(fontsize=11,loc='lower left')
    for ax in axes:
        ax.fill_between(t,0,1,where=~valid,transform=ax.get_xaxis_transform(),
                        color='#cfcfcf',alpha=.35,step='mid')
        ax.grid(ls=':',alpha=.4)
        ax.set_xlim(t[0],t[-1])
    fig.suptitle('图44  连续运动合成音频 → 时差观测 → 追踪\n'
                 '16 kHz / 2 s / 双麦间距10 cm；灰区为RMS门限产生的缺测',fontsize=FS_SUP)
    fig.tight_layout(rect=(0,0,1,.93),h_pad=1.8)
    save(fig,'fig44_tracking_audio.png',{'AudioManifestDigest':hashlib.sha256(manifest_path.read_bytes()).hexdigest()})


def fig_agc_blocks():
    """Plot exported PCM and the causal, block-available AGC state."""
    import json
    import wave
    audio = OUT.parent / 'codes' / 'audio'
    manifest_path = audio / 'MANIFEST.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    group = manifest['groups']['agc_blocks']
    records = {record['file']: record for record in manifest['files']}
    labels = (
        ('agc_blocks_input.wav', '输入', C_MAIN, '-'),
        ('agc_blocks_10ms.wav', '10 ms 正确系数', C_BLUE, '-'),
        ('agc_blocks_100ms.wav', '100 ms 正确系数', C_GREEN, '--'),
        ('agc_blocks_100ms_wrong_alpha.wav', '100 ms 错用 10 ms 系数', C_RED, ':'),
    )
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.8), sharex=True,
                             gridspec_kw={'height_ratios': [1.12, 1]})
    sample_rate = group['parameters']['sample_rate_hz']
    block = sample_rate // 100  # 10 ms measured windows, independently of AGC block size
    for name, label, color, style in labels:
        path = audio / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != records[name]['sha256']:
            raise ValueError(f'AGC PCM digest mismatch: {name}')
        with wave.open(str(path), 'rb') as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(),
                    wav.getnframes()) != (sample_rate, 1, 2, 2 * sample_rate):
                raise ValueError(f'AGC PCM format mismatch: {name}')
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(float) / 32768
        rms = np.sqrt(np.mean(pcm.reshape(-1, block) ** 2, axis=1))
        time = (np.arange(len(rms)) + .5) * block / sample_rate
        axes[0].plot(time, rms, style, color=color, lw=1.7, label=label)
    axes[0].set_ylabel('10 ms PCM RMS / 满量程', fontsize=FS_LABEL)
    axes[0].set_title('(a) 四路导出 WAV 用同一 0.7 增益；曲线从 PCM 重新测量',
                      loc='left', fontsize=FS_TITLE)
    axes[0].legend(loc='upper left', bbox_to_anchor=(1.01, 1),
                   fontsize=FS_SMALL)
    for key, label, color, style in (
        ('10ms', '10 ms 正确系数', C_BLUE, '-'),
        ('100ms', '100 ms 正确系数', C_GREEN, '--'),
        ('100ms_wrong_alpha', '100 ms 错用 10 ms 系数', C_RED, ':'),
    ):
        blocks = group['block_records'][key]['blocks']
        available = [0.] + [item['available_time_s'] for item in blocks]
        gain = [blocks[0]['gain_before']] + [item['gain'] for item in blocks]
        axes[1].step(available, gain, where='post', label=label,
                     color=color, linestyle=style, lw=1.8)
    axes[1].set(ylabel='整块可用后的增益 (倍)', xlabel='源时间与输出可用时间 (s)')
    axes[1].set_title('(b) 增益只能在对应输入块全部到达后更新；虚线与点线使用 100 ms 块',
                      loc='left', fontsize=FS_TITLE)
    for at in group['parameters']['amplitude_breaks_s']:
        for ax in axes:
            ax.axvline(at, color='0.5', ls=(0, (2, 3)), lw=.8)
    for ax in axes:
        ax.set_xlim(0, 2)
        ax.grid(ls=':', alpha=.3)
    axes[1].set_ylim(.8, 8.4)
    fig.suptitle('图45  同一合成载波的整块 AGC：块长、时间常数与 PCM 输出',
                 fontsize=FS_SUP)
    fig.tight_layout(rect=(0, 0, .81, .94), h_pad=1.2)
    save(fig, 'fig45_agc_blocks.png',
         {'AudioManifestDigest': hashlib.sha256(manifest_path.read_bytes()).hexdigest()})

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
    write_gcc_reverb_report(fig_gcc_reverb(),
        Path(__file__).resolve().parents[1] / "codes/reports/figure13_gcc_reverb.json")
    fig_delay_phase()
    fig_beampattern_anatomy()
    fig_dsin_geometry()
    fig_wpe_frames()
    fig_latency_budget()
    fig_gcc_two_ways()
    fig_audio_examples()
    fig_audio_counterexamples()
    fig_nonlinear_echo()
    fig_clock_drift()
    fig_interpolation_error()
    fig_gss_flow()
    fig_css_overlap()
    fig_tracking_audio()
    fig_agc_blocks()
    print("ALL DONE")


if __name__ == "__main__":
    main()
