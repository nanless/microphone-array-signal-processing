#!/usr/bin/env python3
"""把 chapters/ 16 篇及 codes/chapters/ch00/research/ 6 篇 Markdown 建成静态站。

用法（报告根目录）：
    .venv/bin/python scripts/build_site.py

产物：site/index.html（首页）+ site/01..15_*.html（15 篇正文/专题/附录），
另有 site/research/index.html 和 5 篇独立研究页、主清单 109 个与独立实验 144 个合成 WAV，
以及 4 个真实录音/派生 WAV；源码按章保存，发布 URL 保持原样。
左侧边栏 = 首页 + 15 篇 + 每篇的二级及以下小节锚点，顶部面包屑，
文末上一篇/下一篇（首页不输出该盒）。图片直接引用 ../figures/（不复制）。
数学公式用 MathJax CDN 渲染（离线时显示源码，页面顶部有提示）。
"""
import os
import sys
import re
import tempfile
import hashlib
import json
import shutil
from html import escape, unescape
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit, urlunsplit

try:
    from scripts.build_markdown_helpers import (ALLOWED_LINK_SCHEMES, protect_code,
        restore_code, validate_url_schemes, render_markdown, parsed_markdown_headings,
        map_table_cell_text)
    from scripts.code_layout import MAIN_AUDIO_GROUP_CHAPTER, main_audio_manifest_path, main_audio_path
except ModuleNotFoundError:  # direct ``python scripts/build_site.py``
    from build_markdown_helpers import (ALLOWED_LINK_SCHEMES, protect_code,
        restore_code, validate_url_schemes, render_markdown, parsed_markdown_headings,
        map_table_cell_text)
    from code_layout import MAIN_AUDIO_GROUP_CHAPTER, main_audio_manifest_path, main_audio_path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from codes.chapters.ch00.io_contracts import (
    validate_parent_chain, validate_asset_directory, strict_json_loads,
)
SRC = ROOT / "chapters"
OUT = ROOT / "site"
CODE_CHAPTERS = ROOT / "codes" / "chapters"
RESEARCH_ROOT = CODE_CHAPTERS / "ch00" / "research"
INLINE_LAYOUT_PATH = ROOT / "scripts" / "inline_layout.js"
REAL_AUDIO_ROOT = CODE_CHAPTERS / "ch02" / "real_audio"
ROOM_AUDIO_ROOT = CODE_CHAPTERS / "appendix_b" / "room_audio"
MOVING_AUDIO_ROOT = CODE_CHAPTERS / "ch09" / "moving_audio"
TRACKING_AUDIO_ROOT = CODE_CHAPTERS / "ch09" / "tracking_audio"
GSS_AUDIO_ROOT = CODE_CHAPTERS / "ch08" / "gss_audio"
BINAURAL_AUDIO_ROOT = CODE_CHAPTERS / "ch01" / "binaural_audio"
SPECTRAL_AUDIO_ROOT = CODE_CHAPTERS / "ch01" / "spectral_audio"
SPECTRAL_AUDIO_WAVS = {'flat_source.wav', 'flat_stereo.wav', 'tilted_source.wav', 'tilted_stereo.wav'}
STFT_AUDIO_ROOT = CODE_CHAPTERS / "ch02" / "stft_audio"
REFLECTION_AUDIO_ROOT = CODE_CHAPTERS / 'ch04' / 'reflection_audio'
REFLECTION_AUDIO_WAVS = {'reflection_reference.wav', 'reflection_direct.wav', 'reflection_component.wav', 'reflection_mixed.wav'}
BASELINE_AUDIO_ROOT = CODE_CHAPTERS / 'ch03' / 'baseline_audio'
BASELINE_AUDIO_WAVS = {'baseline_source.wav', 'baseline_train_px.wav', 'baseline_train_nx.wav',
                       'baseline_train_py.wav', 'baseline_train_pz.wav', 'baseline_heldout.wav'}
SWEEP_AUDIO_ROOT = CODE_CHAPTERS / 'ch02' / 'sweep_audio'
SWEEP_AUDIO_WAVS = {'sweep_source.wav', 'sweep_complete.wav', 'sweep_noisy.wav', 'sweep_cut.wav'}
PHASE_AUDIO_ROOT = CODE_CHAPTERS / "ch05" / "phase_audio"
PHASE_AUDIO_WAVS = {"phase_reference.wav", "phase_flip.wav", "phase_quadrature.wav"}
DERIVATIVE_AUDIO_ROOT = CODE_CHAPTERS / "ch05" / "derivative_audio"
APA_AUDIO_ROOT = CODE_CHAPTERS / "ch06" / "apa_audio"
REFERENCE_AUDIO_ROOT = CODE_CHAPTERS / "ch06" / "reference_audio"
REFERENCE_AUDIO_WAVS = {"reference_" + name + ".wav" for name in
                        ("early", "late", "echo", "early_residual",
                         "wrong_gain_residual", "late_residual")}
DELAY_AUDIO_ROOT = CODE_CHAPTERS / "ch07" / "delay_audio"
DELAY_AUDIO_WAVS = {"delay_" + name + ".wav" for name in
                    ("reference", "array", "common_history", "aligned_history",
                     "common_residual", "aligned_residual")}
MINT_AUDIO_ROOT = CODE_CHAPTERS / "ch07" / "mint_audio"
MASK_AUDIO_ROOT = CODE_CHAPTERS / "ch08" / "mask_audio"
SCENARIO_AUDIO_ROOT = CODE_CHAPTERS / "ch11" / "scenario_audio"
WEIGHTED_AUDIO_ROOT = CODE_CHAPTERS / "appendix_a" / "weighted_audio"
RESPONSE_AUDIO_ROOT = CODE_CHAPTERS / "appendix_b" / "response_audio"
IMAGING_AUDIO_ROOT = CODE_CHAPTERS / "ch14" / "imaging_audio"
DISTRIBUTED_AUDIO_ROOT = CODE_CHAPTERS / "ch15" / "distributed_audio"
DISTRIBUTED_AUDIO_WAVS = {
    name + ".wav" for name in (
        "reference_node1", "reference_node2", "array_white", "array_correlated",
        "local_node1", "central_white", "compressed_white", "central_correlated",
        "compressed_correlated", "stale_correlated", "central_node2_correlated",
        "remote_scalar_white", "transport_pcm16_white", "clock_misaligned_white",
        "clock_linear_corrected_white", "packet_zerofill_white", "packet_local_fallback_white",
    )
}
IMAGING_AUDIO_WAVS = {"source_1.wav", "source_2_phase_code.wav", "source_2_coherent.wav",
                     "array_phase_code.wav", "array_coherent.wav"}
RESPONSE_AUDIO_WAVS = {"response_" + name + ".wav" for name in
                       ("source", "reflection_a", "reflection_b", "full_a", "full_b")}
WEIGHTED_AUDIO_WAVS = {"weighted_" + name + ".wav" for name in
                       ("target", "array", "ols", "gls", "reversed")}
SCENARIO_AUDIO_WAVS = {f"selection_{scene}_{kind}.wav" for scene in ("single", "dual")
                       for kind in ("target", "mixture", "fir3", "fir9")}
NOISE_AUDIO_ROOT = CODE_CHAPTERS / "ch10" / "noise_audio"
NOISE_AUDIO_WAVS = {"noise_" + name + ".wav" for name in
                    ("reference", "component", "mixture", "fixed", "polluted", "known_variance")}
MASK_AUDIO_WAVS = {"mask_" + name + ".wav" for name in
                   ("target", "other", "mixture", "bounded_real", "unbounded_real", "complex_oracle")}
MINT_AUDIO_WAVS = {"mint_reference.wav": 1, "mint_well_array.wav": 2,
                   "mint_near_array.wav": 2, "mint_well_exact.wav": 1,
                   "mint_near_exact.wav": 1, "mint_near_regularized.wav": 1}
APA_AUDIO_WAVS = {"apa_" + name + ".wav" for name in ("reference", "true_echo", "microphone", "nlms_residual", "apa2_residual", "apa4_residual")}
DERIVATIVE_AUDIO_WAVS = {"derivative_reference.wav": 1, "derivative_array.wav": 3, "derivative_single.wav": 1, "derivative_constrained.wav": 1}
FOCUS_AUDIO_ROOT = CODE_CHAPTERS / "ch04" / "focus_audio"
FOCUS_AUDIO_WAVS = {"focus_reference.wav": 1, "focus_delayed_source.wav": 1, "focus_array.wav": 4, "focus_known_focused.wav": 4}
GEOMETRY_AUDIO_ROOT = CODE_CHAPTERS / "ch03" / "geometry_audio"
GEOMETRY_AUDIO_WAVS = {"geometry_reference.wav": 1, "geometry_u.wav": 6, "geometry_v.wav": 6}
STFT_AUDIO_WAVS = {"stft_roundtrip.wav", "full_convolution.wav", "framewise_mtf.wav"}


def _preflight_asset_stage(source, destination, members):
    """Inspect the complete source and a new destination before any copying."""
    try:
        validate_asset_directory(source, members, check=True)
    except ValueError as error:
        raise ValueError("媒体源目录必须恰含指定普通WAV、清单及必要许可/状态文件：" + str(error)) from error
    destination = validate_parent_chain(destination)
    if destination.exists():
        raise ValueError("media staging destination already exists")


def _read_asset_manifest(path, *, per_channel_rms=False):
    """Strict JSON with explicit structural numeric types, not model scoring."""
    validate_parent_chain(path)
    manifest = strict_json_loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("音频清单根必须为对象")
    integer_fields = {"schema_version", "sample_rate_hz", "channels",
                      "samples_per_channel", "sample_width_bytes"}
    numeric_fields = {"common_export_gain", "peak", "rms", "duration_s",
                      "quantization_max_abs_error", "quantization_max_abs_error_bound"}
    def types(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in integer_fields and type(item) is not int:
                    raise ValueError(f"音频元数据字段必须为整数：{key}")
                if key in numeric_fields:
                    if key == "rms" and per_channel_rms:
                        if (not isinstance(item, list) or not item
                                or type(value.get("channels")) is not int
                                or len(item) != value["channels"]):
                            raise ValueError("每通道RMS须与该WAV通道数一致")
                        numbers = item
                    else:
                        numbers = [item]
                    if any(type(number) not in (int, float) for number in numbers):
                        raise ValueError(f"音频元数据字段必须为真实数值：{key}")
                types(item)
        elif isinstance(value, list):
            for item in value:
                types(item)
    types(manifest)
    return manifest


def main_audio_sources():
    """Map only declared chapter-owned WAVs to their stable published names."""
    manifest = _read_asset_manifest(main_audio_manifest_path(CODE_CHAPTERS))
    result = {}
    for record in manifest["files"]:
        chapter = MAIN_AUDIO_GROUP_CHAPTER[record["group"]]
        if record.get("chapter") != chapter:
            raise ValueError(f"音频章节归属不符：{record['file']}")
        path = validate_parent_chain(main_audio_path(CODE_CHAPTERS, record["group"], record["file"]))
        if not path.is_file() or path.resolve() in result:
            raise ValueError("主音频文件缺失或重复")
        result[path.resolve()] = record["file"]
    return result

CHAPTERS = [
    ("01_problem-definition.md", "第 1 章 · 问题定义与双耳启示"),
    ("02_basics-signal-model.md", "第 2 章 · 声音到达阵列时发生了什么"),
    ("03_array-geometry.md", "第 3 章 · 阵列几何形态"),
    ("04_doa-estimation.md", "第 4 章 · 声源定位（DOA 估计）"),
    ("05_beamforming.md", "第 5 章 · 波束形成"),
    ("06_aec.md", "第 6 章 · 声学回声消除（AEC）"),
    ("07_wpe-dereverberation.md", "第 7 章 · 去混响（WPE）"),
    ("08_speech-separation.md", "第 8 章 · 语音分离"),
    ("09_source-tracking.md", "第 9 章 · 声源追踪"),
    ("10_engineering-practice.md", "第 10 章 · 工程实现、评测与产业实践"),
    ("11_selection-guide.md", "第 11 章 · 总结与选型指南"),
    ("14_acoustic-imaging.md", "扩展专题Ⅰ · 声学成像与噪声源诊断"),
    ("15_distributed-enhancement.md", "扩展专题Ⅱ · 分布式麦克风协同增强"),
    ("12_appendix-symbols-math.md", "附录 A · 符号术语数学"),
    ("13_appendix-guide.md", "附录 B · 路径地图与练习"),
]
HOME_FNAME = "00_overview.md"
RESEARCH = [
    ("README.md", "源码研究导读"),
    ("01_spatial_and_tracking.md", "空间处理与追踪源码研究"),
    ("02_aec_wpe_separation.md", "AEC、WPE 与分离源码研究"),
    ("03_industrial_deployment.md", "工业音频实现研究"),
    ("04_source_reproduction.md", "源码获取与独立复现记录"),
    ("05_exercises_and_audio.md", "章节代码练习与音频实验"),
]
REPOSITORY_BLOB_BASE = "https://github.com/nanless/microphone-array-signal-processing/blob/main/"
LABEL_BY_FNAME = {f: l for f, l in CHAPTERS}
LABEL_BY_FNAME[HOME_FNAME] = "🏠 导读与导航（首页）"

# Exact source/header contracts, not table positions. Budgets apply only on
# narrow screens; print tables retain the existing A4 column layout.
NARROW_TABLE_POLICIES = {
    SRC / "00_overview.md": {
        ("编号", "文件", "配图", "难度", "建议学习单元"): (60, {1: 3, 4: 6}),
        ("图", "文件", "所在文档", "类型", "内容"): (60, {1: 5, 4: 6}),
    },
    SRC / "03_array-geometry.md": {
        ("失配类型", "示例量级", "首先受影响的量", "按该示例计算", "可采取的措施"): (70, {1: 7, 2: 9, 3: 9, 4: 18, 5: 16}),
    },
    SRC / "04_doa-estimation.md": {
        ("路线", "输入怎样形成", "输出是什么", "使用前要检查什么"): (44, {1: 8, 2: 12, 3: 10, 4: 14}),
        ("方法", "主要输入与输出", "是否预先给源数", "主要模型条件", "典型失败模式", "主要计算项"): (66, {1: 9, 2: 11, 3: 8, 4: 12, 5: 12, 6: 14}),
    },
    SRC / "05_beamforming.md": {
        ("权重", "总质量", "正支持数", "集中程度对应数量", "输出矩阵"): (54, {1: 8, 2: 12, 3: 9, 4: 12, 5: 13}),
        ("参考麦", r"参考目标传递 $b_r$", r"Souden 权重 $\vec w_{S,r}$", r"输出目标响应 $\vec w^H\vec b$", "输出噪声功率"): (64, {1: 8, 2: 12, 3: 19, 4: 13, 5: 12}),
        ("权重", "线性 WNG", "WNG／dB", "复响应误差上界", "可保证的响应幅度下界"): (54, {1: 8, 2: 10, 3: 10, 4: 12, 5: 14}),
        (r"绝对加载 $\varepsilon$", r"相对加载 $\alpha=\varepsilon/11$", "线性 WNG", r"干扰功率增益 $|\vec w^H\vec b|^2$", "原协方差输出噪声功率"): (63, {1: 9, 2: 13, 3: 11, 4: 13, 5: 17}),
        ("", "Frost（1972）", "GSC"): (46, {1: 8, 2: 19, 3: 19}),
        ("设计变化", "可能获得的能力", "同时增加的风险", "必须固定的比较条件"): (52, {1: 8, 2: 14, 3: 15, 4: 15}),
        (r"$\alpha$ 取值", "加载相对平均特征值的大小", "白噪声增益", "指向性/零陷深度", "稳健性", "适用场合"): (71, {1: 9, 2: 13, 3: 10, 4: 12, 5: 13, 6: 14}),
        ("现象", "主要误差来源", "首选", "备选/联合", "验证重点"): (76, {1: 15, 2: 10, 3: 18, 4: 16, 5: 17}),
        ("方法", "优化目标或固定规则", "需要的统计量/先验", "目标保持条件", "主要失败模式", "主要计算"): (85, {1: 10, 2: 15, 3: 16, 4: 14, 5: 16, 6: 14}),
    },
    SRC / "06_aec.md": {
        ("#", "征兆", "量测", "工具/信号", "判断依据"): (70, {1: 3, 2: 9, 3: 10, 4: 20, 5: 18}),
        ("算法", "每样本主要计算", "收敛受什么影响", "双讲更新控制", "适用条件"): (48, {1: 8, 2: 13, 3: 10, 4: 9, 5: 8}),
        ("路线", "代表方案", "可检查的内部量", "主要风险", "资源验证"): (52, {1: 8, 2: 11, 3: 13, 4: 11, 5: 9}),
        ("模型", "结构", "可辨识性", "资源关系", "适用条件"): (50, {1: 9, 2: 15, 3: 10, 4: 8, 5: 8}),
        ("抽头点", "位置", "抽头之后的变换", "主要风险", "检查方法"): (48, {1: 7, 2: 8, 3: 13, 4: 9, 5: 11}),
        ("#", "测试项", "条件", "记录结果", "阈值来源"): (54, {1: 3, 2: 12, 3: 13, 4: 12, 5: 14}),
        ("处理输出", "脉冲测得固定延迟", "[2,3) s 回声功率／对齐后总输出功率", r"[3,4) s $g_\Delta$", r"[3,4) s $E_\Delta$"): (54, {1: 10, 2: 10, 3: 16, 4: 9, 5: 9}),
        ("文件分量", "切换窗 $E$", "稳定窗 $E$", "尾窗 $E$"): (52, {1: 8, 2: 15, 3: 15, 4: 14}),
    },
    SRC / "11_selection-guide.md": {
        ("场景", "条件变化", "方案怎样变化"): (36, {1: 6}),
        ("项目", "应写内容"): (30, {1: 6}),
        ("候选", "1500 Hz目标幅度保留", "3500 Hz噪声衰减（dB）", "对齐NMSE（dB）"): (44, {1: 6}),
        ("场景与候选", "目标整数能量D", "误差整数能量E", "实际PCM线性NMSE"): (50, {1: 8}),
    },
    SRC / "12_appendix-symbols-math.md": {
        ("符号", "含义"): (34, {1: 8}),
        ("术语", "解释"): (32, {1: 8}),
        ("权重", "误差均方的逐项计算", "解析 MSE", "解析 NMSE（除以 $0.02$）"): (52, {1: 6}),
        ("输出", "实际整数误差平方和 $E_I$", "实际 PCM MSE", "实际 PCM NMSE"): (50, {1: 6}),
    },
    SRC / "13_appendix-guide.md": {
        ("声源位置", "距离 (m)", "真方位 (°)", "$T_{20}$ 外推 $T_{60}$ (s)", "DRR (dB)", "SRP 方位 (°)", "绝对误差 (°)"): (64, {1: 8}),
        ("声源位置", "完整 WAV 帧数", "前 1 秒 SRP 方位", "整段 SRP 方位", "后者减前者"): (60, {1: 8}),
        ("完整输出", "整数误差平方和 $E$", "实际PCM MSE", "实际PCM NMSE"): (48, {1: 6}),
    },
    RESEARCH_ROOT / "01_spatial_and_tracking.md": {
        ("原始路线", "输入与额外前提", "求解目标和关键改变", "本书当前证据"): (54, {1: 8, 2: 15, 3: 16, 4: 15}),
        ("路线", "处理的对象", "新增的模型条件", "本书实际范围"): (54, {1: 10, 2: 12, 3: 17, 4: 15}),
        ("源码入口", "实际操作", "与本章的关系及限制"): (64, {1: 32, 2: 18, 3: 22}),
        ("人工输入 [左, 右]", "独立计算的预期", "实际调用结果", "可以支持什么"): (52, {1: 8, 2: 16, 3: 16, 4: 12}),
        ("延迟值的存储方式与查询", "独立数学预期：左右IR／延迟值", "原函数实际结果", "解释"): (50, {1: 16, 2: 12, 3: 12, 4: 12}),
    },
    RESEARCH_ROOT / "05_exercises_and_audio.md": {
        ("文件", "取点或输出", "查看重点"): (60, {1: 20, 2: 18, 3: 22}),
        ("取点或输出", "切换窗整数E", "稳定窗整数E", "尾窗整数E"): (52, {1: 16, 2: 12, 3: 12, 4: 12}),
        ("文件", "输出整数能量E_y", "误差整数能量E_e", "实际PCM功率E_y/D", "实际PCM总MSE E_e/D", "实际PCM NMSE"): (72, {1: 8, 2: 13, 3: 13, 4: 13, 5: 13, 6: 12}),
        ("实际文件/通道", "整数能量E", "每通道分母D", "实际PCM均方"): (50, {1: 14, 2: 11, 3: 13, 4: 12}),
        ("条件", "单声道源", "双通道人工响应输出"): (46, {1: 6}),
        ("题号", "输入和计算", "能支持的结论"): (36, {1: 6}),
        ("输出", "解析MSE", "实际PCM误差整数平方和$E$", "实际PCM MSE", "实际PCM NMSE"): (60, {1: 6}),
        ("完整输出", "解析稳态MSE", "实际PCM整数误差平方和 $E$", "实际PCM MSE", "实际PCM NMSE"): (60, {1: 6}),
    },
}

CSS = """
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;line-height:1.75;color:#1a1a2e;background:#fafbfc}
.topbar{position:sticky;top:0;z-index:10;background:#1a1a2e;color:#fff;padding:10px 20px;font-size:15px}
.topbar a{color:#9ec5f0;text-decoration:none}.topbar a:hover{text-decoration:underline}
h1,h2,h3,h4{scroll-margin-top:60px}
@media screen{.main a[id^="e06-"],.main a[id^="e07-"]{scroll-margin-top:60px}}
a:focus-visible,summary:focus-visible{outline:3px solid #e67e22;outline-offset:3px}
.skip-link{position:absolute;left:10px;top:-60px;z-index:30;background:#fff;color:#1a1a2e;padding:8px 12px;border:2px solid #e67e22}
.skip-link:focus{top:8px}
.wrap{display:flex;max-width:1280px;margin:0 auto}
.side{width:300px;flex-shrink:0;padding:20px 14px;position:sticky;top:47px;height:calc(100vh - 47px);overflow-y:auto;background:#fff;border-right:1px solid #e5e8ee;font-size:13.5px}
.side a{color:#2f6db3;text-decoration:none}.side a:hover{text-decoration:underline}
.side .chap{margin:10px 0 2px;font-weight:700}.side .chap.cur{color:#c0392b}
.side ul{margin:2px 0 6px;padding-left:16px;color:#666}.side li{margin:2px 0}
.main{flex:1;min-width:0;padding:28px 36px;background:#fff;overflow-wrap:anywhere}
.main p{margin:0 0 1.05em}.main li>p{margin:.35em 0}
.main img{max-width:100%;height:auto;display:block;margin:14px auto;border:1px solid #eee;min-height:40px;background:#f6f8fb}
.audio-sample{display:block;width:min(100%,460px);margin:8px 0 18px}.audio-sample:focus-visible{outline:3px solid #e67e22}
/* 表格列宽不能压掉原生播放按钮；手机上的表格可横向滚动。 */
td .audio-sample{width:280px;min-width:280px}
table:has(td .audio-sample) td:last-child{min-width:180px}
table{border-collapse:collapse;margin:14px 0;max-width:100%}
.table-scroll{max-width:100%;overflow-x:auto}
.table-scroll:focus-visible{outline:3px solid #e67e22;outline-offset:2px}
th,td{border:1px solid #dfe3ea;padding:6px 10px;font-size:14px;text-align:left}
th{background:#f0f4f9}code{background:#f0f3f7;padding:1px 5px;border-radius:4px;font-size:13.5px}
pre{background:#1a1a2e;color:#e8ecf3;padding:14px;border-radius:8px;overflow-x:auto}
pre code{background:none;color:inherit;padding:0}
blockquote{border-left:3px solid #2f6db3;margin:14px 0;padding:8px 14px;background:#f2f7fd;color:#333}
mjx-container[jax="CHTML"]{font-size:110%!important;overflow-x:auto;overflow-y:hidden;max-width:100%;min-width:0!important}
body mjx-assistive-mml{width:1px!important;max-width:1px!important;min-width:0!important;height:1px!important;overflow:hidden!important}
pre,.table-scroll,mjx-container[jax="CHTML"]{overflow-wrap:normal}
.pn{display:flex;justify-content:space-between;margin:30px 0 10px;padding-top:16px;border-top:1px solid #e5e8ee}
.pn a{color:#2f6db3;text-decoration:none}.pn .off{color:#aaa}
.foot{color:#888;font-size:13px;margin:20px 0 40px}
h1{font-size:26px;border-bottom:2px solid #1a1a2e;padding-bottom:8px}
h2{font-size:21px;margin-top:34px;border-bottom:1px solid #e5e8ee;padding-bottom:6px}
h3{font-size:17px;margin-top:24px}
h4{font-size:15.5px;margin-top:20px;color:#333}
.toc-mobile{display:none;margin:0 0 16px;border:1px solid #e5e8ee;border-radius:8px;padding:10px 14px;background:#f7fafd;font-size:14px}
.toc-mobile summary{cursor:pointer;font-weight:700}
.toc-mobile ul{padding-left:18px;margin:8px 0 2px}.toc-mobile a{color:#2f6db3;text-decoration:none}
.topbtn{display:block;margin:20px auto;background:#1a1a2e;color:#fff;border-radius:50%;width:42px;height:42px;text-align:center;line-height:42px;text-decoration:none;font-size:18px}
.offline-note{display:none;background:#fff7e6;border:1px solid #e6c87a;color:#7a5b00;padding:8px 14px;font-size:13.5px}
.anchor-alias{display:block;position:relative;top:-60px;visibility:hidden}
.tutorial-math-tail{display:inline-block;white-space:nowrap;overflow-wrap:normal;vertical-align:baseline;overflow:visible}
.tutorial-math-tail mjx-assistive-mml{max-width:1px!important;min-width:0!important;white-space:normal}
.tutorial-exercise-id{white-space:nowrap;overflow-wrap:normal}
.tutorial-table-label{white-space:nowrap;overflow-wrap:normal}
@media(max-width:900px){.selection-readable-table{min-width:760px}.selection-readable-table th,.selection-readable-table td{min-width:10em}.selection-readable-table th:first-child,.selection-readable-table td:first-child{min-width:6em}}
@media(max-width:900px){.side{display:none}.main{padding:20px}.toc-mobile{display:block}.topbar{font-size:14px}mjx-container[jax="CHTML"]:not([display="true"]){display:inline-block;vertical-align:middle}.aec-readable-table th,.aec-readable-table td,.wpe-readable-table th,.wpe-readable-table td,.separation-readable-table th,.separation-readable-table td{min-width:8em}.tracking-readable-table th,.tracking-readable-table td{min-width:8em}.industrial-readable-table th,.industrial-readable-table td{min-width:8em}}
@media(max-width:900px){.noise-readable-table th:first-child,.noise-readable-table td:first-child{min-width:6em}}
@media screen and (max-width:900px){.source-contract-readable-table th,.source-contract-readable-table td{min-width:10em}}
@media screen and (max-width:900px){.tutorial-budget-table{min-width:var(--tutorial-table-width)}.tutorial-budget-table .tutorial-short-column{min-width:var(--tutorial-column-width);overflow-wrap:normal}}
@media print{.topbar,.side,.pn,.topbtn,.toc-mobile{display:none}.main{padding:0}.table-scroll{overflow:visible}table{display:table}a{color:#000;text-decoration:none}pre{white-space:pre-wrap;background:#fff;color:#000;border:1px solid #ccc}}
"""

PAGE = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · 麦克风阵列信号处理教程</title>
<meta name="source-digest" content="{source_digest}"><style>{css}</style>
<script>
{inline_layout_script}
window.MathJax = {{tex: {{inlineMath: [['$', '$'], ['\\\\(', '\\\\)']], displayMath: [['$$', '$$']]}}, startup: {{pageReady: () => MathJax.startup.defaultPageReady().then(() => document.fonts.ready).then(() => ArrayTutorialLayout.apply())}}}};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js"
 onerror="document.getElementById('offnote').style.display='block';document.getElementById('offnote').textContent='公式渲染脚本加载失败：当前显示的是公式源码。';"></script>
</head><body id="top"><a class="skip-link" href="#main-content">跳到正文</a>
<header class="topbar"><a href="{home_href}">🏠 首页</a> &nbsp;/&nbsp; {crumb}</header>
<div class="wrap"><nav class="side" aria-label="全书目录">{sidebar}</nav>
<main class="main" id="main-content" tabindex="-1"><div class="offline-note" id="offnote">当前离线：公式显示为源码，正文讲解不受影响。</div>
<details class="toc-mobile"><summary>本页目录</summary><nav aria-label="本页目录">{toc}</nav></details>
{body}{pn}
<footer class="foot">麦克风阵列信号处理教程 · 静态站由 scripts/build_site.py 生成</footer>
</main></div><a class="topbtn" href="#top" title="回顶部" aria-label="回到页面顶部">↑</a>
<script>if(!navigator.onLine)document.getElementById('offnote').style.display='block';</script>
</body></html>
"""


REAL_AUDIO_WAVS = {
    "demand_nriver_16ch_10s.wav", "demand_nriver_ch01_10s.wav",
    "demand_nriver_mean02_10s.wav", "demand_nriver_mean16_10s.wav",
}
REAL_AUDIO_FILES = REAL_AUDIO_WAVS | {"MANIFEST.json", "ATTRIBUTION.txt", "LICENSE.txt", "README.md"}
REAL_AUDIO_SOURCE_LINKS = {
    "../core/real_recordings.py": REPOSITORY_BLOB_BASE + "codes/chapters/ch02/core/real_recordings.py",
    "../examples/prepare_real_recordings.py": REPOSITORY_BLOB_BASE + "codes/chapters/ch02/examples/prepare_real_recordings.py",
}
ROOM_AUDIO_EXTRA = {"MANIFEST.json", "ROOM_RESULTS.png", "RESULTS.json"}
TRACKING_AUDIO_WAVS = {"source.wav": 1, "array_noisy.wav": 2}
MOVING_AUDIO_WAVS = {"source.wav": 1, "static_array.wav": 2, "moving_array.wav": 2}
GSS_AUDIO_WAVS = {"source_1.wav": 1, "source_2.wav": 1, "mixture.wav": 2,
                  "enhanced_correct.wav": 1, "enhanced_missed.wav": 1}
BINAURAL_AUDIO_WAVS = {"reference.wav", "itd_only.wav", "ild_only.wav",
                       "consistent.wav", "conflicting.wav"}


def publish_real_audio_readme(source_text, destination):
    """Retain local media links while making chapter code links work on the site."""
    import markdown

    published = source_text
    for local, remote in REAL_AUDIO_SOURCE_LINKS.items():
        token = f"]({local})"
        if published.count(token) != 1:
            raise ValueError(f"真实录音说明缺少唯一的源码链接：{local}")
        published = published.replace(token, f"]({remote})")

    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.hrefs = []

        def handle_starttag(self, tag, attrs):
            if tag == "a":
                self.hrefs.extend(value for key, value in attrs if key == "href")

    links = Links()
    links.feed(markdown.markdown(published, extensions=["tables"]))
    for href in links.hrefs:
        parsed = urlsplit(href)
        if parsed.netloc and not parsed.scheme:
            raise ValueError(f"真实录音发布说明含无协议外部链接：{href}")
        if parsed.scheme:
            if parsed.scheme not in ALLOWED_LINK_SCHEMES:
                raise ValueError(f"真实录音发布说明含不安全链接：{href}")
            continue
        if not parsed.path:
            continue
        target = (destination / unquote(parsed.path)).resolve()
        if (parsed.path.startswith("/") or
                not target.is_relative_to(destination.parent.resolve()) or
                not target.is_file()):
            raise ValueError(f"真实录音发布说明含失效的本地链接：{href}")
    return published


def stage_real_audio(source, destination):
    """Stage a fixed data release with attribution, separate from synthetic audio."""
    _preflight_asset_stage(source, destination, REAL_AUDIO_FILES)
    manifest = _read_asset_manifest(source / "MANIFEST.json", per_channel_rms=True)
    records = manifest["files"]
    if len(records) != 4 or {r["file"] for r in records} != REAL_AUDIO_WAVS:
        raise ValueError("真实录音清单必须匹配四个独立发布文件")
    if {p.name for p in source.iterdir() if p.is_file()} != REAL_AUDIO_FILES:
        raise ValueError("真实录音文件或许可集合不符")
    for record in records:
        path = source / record["file"]
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("真实录音摘要不符")
    for name in ("ATTRIBUTION.txt", "LICENSE.txt"):
        if not (source / name).read_text(encoding="utf-8").strip():
            raise ValueError("真实录音署名或许可为空")
    destination.mkdir()
    for name in sorted(REAL_AUDIO_FILES):
        if (source / name).is_symlink():
            raise ValueError("真实录音发布文件不能为符号链接")
        if name != "README.md":
            shutil.copy2(source / name, destination / name)
    published_readme = publish_real_audio_readme(
        (source / "README.md").read_text(encoding="utf-8"), destination)
    (destination / "README.md").write_text(published_readme, encoding="utf-8")
    return REAL_AUDIO_FILES.copy()


def stage_room_audio(source, destination):
    """Stage the separately generated, fixed-room synthetic experiment."""
    from codes.chapters.appendix_b.examples.check_room_assets import check_assets
    from codes.chapters.appendix_b.examples.room_srp_exercise import ordinary_path
    ordinary_path(destination, directory=True)
    if destination.exists():
        raise ValueError("room staging destination already exists")
    check_assets(source)
    manifest = _read_asset_manifest(source / "MANIFEST.json")
    records = manifest["files"]
    names = [record["file"] for record in records]
    if (len(records) != 18 or len(names) != len(set(names)) or
            any(not re.fullmatch(r"[a-z0-9_]+\.wav", name) for name in names)):
        raise ValueError("房间音频清单必须列出18个安全且不重复的WAV文件")
    roles = {(record["case"], record["role"]) for record in records}
    cases = {record["case"] for record in records}
    if len(cases) != 6 or roles != {(case, role) for case in cases
                                     for role in ("source", "direct", "full")}:
        raise ValueError("房间音频必须为六组源/直达/完整三联样本")
    names = set(names) | ROOM_AUDIO_EXTRA
    if {path.name for path in source.iterdir() if path.is_file()} != names:
        raise ValueError("房间音频目录文件集合与清单不符")
    if not (source / "ROOM_RESULTS.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("房间结果图不是 PNG")
    report = json.loads((source / "RESULTS.json").read_text(encoding="utf-8"))
    if (report.get("schema_version") != 1 or
            report.get("status") != "pyroomacoustics_simulation_executed" or
            report.get("actual_max_order") != manifest.get("max_order") or
            report.get("generator", {}).get("path") != "codes/chapters/appendix_b/examples/room_srp_exercise.py" or
            report["generator"].get("sha256") != hashlib.sha256(
                (ROOT / "codes/chapters/appendix_b/examples/room_srp_exercise.py").read_bytes()).hexdigest() or
            report.get("assets", {}).get("figure", {}).get("sha256") != hashlib.sha256(
                (source / "ROOM_RESULTS.png").read_bytes()).hexdigest() or
            report["assets"].get("audio_manifest", {}).get("sha256") != hashlib.sha256(
                (source / "MANIFEST.json").read_bytes()).hexdigest() or
            len(report.get("results", [])) != 6):
        raise ValueError("房间结果报告与生成源、结果图或音频清单不符")
    for record in records:
        path = source / record["file"]
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"房间音频摘要不符：{path.name}")
    destination.mkdir()
    for name in sorted(names):
        if (source / name).is_symlink():
            raise ValueError("房间样本发布文件不能为符号链接")
        shutil.copy2(source / name, destination / name)
    return names


def stage_moving_audio(source, destination):
    """按独立清单核对连续运动合成 PCM，避免发布缺失或错配样本。"""
    import wave
    expected = set(MOVING_AUDIO_WAVS) | {"MANIFEST.json"}
    _check_tracking_members(source, expected)
    manifest = _read_tracking_manifest(source / "MANIFEST.json")
    records = manifest["files"]
    if set(records) != set(MOVING_AUDIO_WAVS):
        raise ValueError("移动声源清单文件集合不符")
    if {path.name for path in source.iterdir() if path.is_file()} != expected:
        raise ValueError("移动声源目录文件集合不符")
    if (manifest["sample_rate_hz"] != 16000
            or manifest["model"].find("free field") < 0):
        raise ValueError("移动声源采样率或模型说明不符")
    source_paths = {"codes/chapters/ch09/examples/moving_source_audio.py",
                    "codes/chapters/ch09/core/moving_source.py",
                    "codes/chapters/ch00/core/audio_samples.py",
                    "codes/chapters/ch02/core/conventions.py"}
    if set(manifest.get("source_sha256", {})) != source_paths:
        raise ValueError("移动声源生成源码清单不完整")
    for name, digest in manifest["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"移动声源生成源码已变化：{name}")
    for name, channels in MOVING_AUDIO_WAVS.items():
        path = source / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != records[name]["sha256"]:
            raise ValueError(f"移动声源音频摘要不符：{name}")
        with wave.open(str(path), "rb") as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(),
                    wav.getnframes(), wav.getcomptype()) != (
                        16000, channels, 2, records[name]["samples_per_channel"], "NONE"):
                raise ValueError(f"移动声源 PCM 格式不符：{name}")
    from codes.chapters.ch09.examples.moving_source_audio import generate, _validate_directory
    generate(source, check=True)
    _validate_directory(destination, expected, check=False)
    _validate_publish_targets([destination / name for name in expected], boundary=destination.parent)
    destination.mkdir()
    for name in sorted(expected):
        shutil.copy2(source / name, destination / name)
    return expected



def stage_binaural_audio(source, destination):
    """Publish five independent time/level cue fixtures with complete tails."""
    _preflight_asset_stage(source, destination, BINAURAL_AUDIO_WAVS | {"MANIFEST.json"})
    import wave
    expected = BINAURAL_AUDIO_WAVS | {"MANIFEST.json"}
    if (source.is_symlink() or {p.name for p in source.iterdir()} != expected
            or any(p.is_symlink() or not p.is_file() for p in source.iterdir())):
        raise ValueError("双耳线索目录必须恰含五份普通WAV及独立清单")
    manifest = _read_asset_manifest(source / "MANIFEST.json")
    if (set(manifest["files"]) != BINAURAL_AUDIO_WAVS
            or manifest["sample_rate_hz"] != 16000
            or manifest["common_export_gain"] != 1):
        raise ValueError("双耳线索清单集合、采样率或共同增益不符")
    required_sources = {"codes/chapters/ch01/core/binaural_cues.py",
                        "codes/chapters/ch01/examples/generate_binaural_cues.py",
                        "codes/chapters/ch00/core/audio_samples.py",
                        'codes/chapters/ch00/io_contracts.py'}
    if set(manifest["source_sha256"]) != required_sources:
        raise ValueError("双耳线索必须记录真实生成源与PCM编码源")
    for name, digest in manifest["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"双耳线索生成源摘要过期：{name}")
    for name in BINAURAL_AUDIO_WAVS:
        path, record = source / name, manifest["files"][name]
        if (record["channels"] != 2 or record["samples_per_channel"] != 32008
                or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]):
            raise ValueError(f"双耳线索摘要或参数不符：{name}")
        with wave.open(str(path), "rb") as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(),
                    wav.getnframes(), wav.getcomptype()) != (16000, 2, 2, 32008, "NONE"):
                raise ValueError(f"双耳线索PCM格式不符：{name}")
            if len(wav.readframes(32008)) != 32008 * 2 * 2:
                raise ValueError(f"双耳线索PCM数据截断：{name}")
    destination.mkdir()
    for name in sorted(expected):
        shutil.copy2(source / name, destination / name)
    return expected


def stage_spectral_audio(source, destination):
    """Publish four complete known-FIR/source-spectrum controls after read-only replay."""
    expected = SPECTRAL_AUDIO_WAVS | {'MANIFEST.json'}
    _preflight_asset_stage(source, destination, expected)
    from codes.chapters.ch01.examples.generate_spectral_cues import check_assets
    check_assets(source)
    validate_asset_directory(destination, expected, check=False)
    destination.mkdir()
    for name in sorted(expected):
        shutil.copy2(source / name, destination / name)
    return expected


def stage_sweep_audio(source, destination):
    """Publish four digital excitation/response controls after current-source replay."""
    expected = SWEEP_AUDIO_WAVS | {'MANIFEST.json'}
    _preflight_asset_stage(source, destination, expected)
    from codes.chapters.ch02.examples.generate_sweep_audio import check_assets
    check_assets(source)
    validate_asset_directory(destination, expected, check=False)
    destination.mkdir()
    for name in sorted(expected):
        shutil.copy2(source / name, destination / name)
    return expected


def stage_baseline_audio(source, destination):
    """Publish six known-direction geometry/time controls after full source replay."""
    expected = BASELINE_AUDIO_WAVS | {'MANIFEST.json'}
    _preflight_asset_stage(source, destination, expected)
    from codes.chapters.ch03.examples.generate_baseline_audio import check_assets
    check_assets(source)
    validate_asset_directory(destination, expected, check=False)
    destination.mkdir()
    for name in sorted(expected):
        shutil.copy2(source / name, destination / name)
    return expected


def stage_reflection_audio(source, destination):
    """Publish four coherent-reflection controls after full source replay."""
    expected = REFLECTION_AUDIO_WAVS | {'MANIFEST.json'}
    _preflight_asset_stage(source, destination, expected)
    from codes.chapters.ch04.examples.generate_reflection_audio import check_assets
    check_assets(source)
    validate_asset_directory(destination, expected, check=False)
    destination.mkdir()
    for name in sorted(expected):
        shutil.copy2(source / name, destination / name)
    return expected


def stage_phase_audio(source, destination):
    """Publish the exact known-component phase controls after read-only replay."""
    _preflight_asset_stage(source, destination, PHASE_AUDIO_WAVS | {'MANIFEST.json'})
    from codes.chapters.ch05.examples.generate_phase_audio import check_assets
    check_assets(source)
    expected = PHASE_AUDIO_WAVS | {'MANIFEST.json'}
    validate_asset_directory(destination, expected, check=False)
    destination.mkdir()
    for name in sorted(expected):
        shutil.copy2(source / name, destination / name)
    check_assets(destination)
    return expected


def stage_stft_audio(source, destination):
    """Publish three independent finite-window convolution fixtures with complete tails."""
    _preflight_asset_stage(source, destination, STFT_AUDIO_WAVS | {"MANIFEST.json"})
    import wave
    expected = STFT_AUDIO_WAVS | {"MANIFEST.json"}
    if (source.is_symlink() or {p.name for p in source.iterdir()} != expected
            or any(p.is_symlink() or not p.is_file() for p in source.iterdir())):
        raise ValueError("STFT卷积目录必须恰含三份普通WAV及独立清单")
    manifest = _read_asset_manifest(source / "MANIFEST.json")
    if (set(manifest["files"]) != STFT_AUDIO_WAVS
            or manifest["sample_rate_hz"] != 16000
            or manifest["common_export_gain"] != 1):
        raise ValueError("STFT卷积清单集合、采样率或共同增益不符")
    required_sources = {"codes/chapters/ch02/core/stft_convolution.py",
                        "codes/chapters/ch02/examples/generate_stft_convolution.py",
                        "codes/chapters/ch02/core/spectral.py",
                        "codes/chapters/ch02/core/conventions.py",
                        "codes/chapters/ch00/core/audio_samples.py",
                        'codes/chapters/ch00/io_contracts.py'}
    if set(manifest["source_sha256"]) != required_sources:
        raise ValueError("STFT卷积必须记录真实生成源与PCM编码源")
    for name, digest in manifest["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"STFT卷积生成源摘要过期：{name}")
    for name in STFT_AUDIO_WAVS:
        path, record = source / name, manifest["files"][name]
        if (record["channels"] != 1 or record["samples_per_channel"] != 32320
                or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]):
            raise ValueError(f"STFT卷积摘要或参数不符：{name}")
        with wave.open(str(path), "rb") as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(),
                    wav.getnframes(), wav.getcomptype()) != (16000, 1, 2, 32320, "NONE"):
                raise ValueError(f"STFT卷积PCM格式不符：{name}")
            if len(wav.readframes(32320)) != 32320 * 1 * 2:
                raise ValueError(f"STFT卷积PCM数据截断：{name}")
    destination.mkdir()
    for name in sorted(expected):
        shutil.copy2(source / name, destination / name)
    return expected


def stage_geometry_audio(source, destination):
    """Publish three multichannel geometry fixtures with full propagation envelopes."""
    _preflight_asset_stage(source, destination, set(GEOMETRY_AUDIO_WAVS) | {"MANIFEST.json"})
    import wave
    expected = set(GEOMETRY_AUDIO_WAVS) | {"MANIFEST.json"}
    if (source.is_symlink() or {p.name for p in source.iterdir()} != expected
            or any(p.is_symlink() or not p.is_file() for p in source.iterdir())):
        raise ValueError("多频几何目录必须恰含三份普通WAV及独立清单")
    manifest = _read_asset_manifest(source / "MANIFEST.json")
    if (set(manifest["files"]) != set(GEOMETRY_AUDIO_WAVS)
            or manifest["sample_rate_hz"] != 32000
            or manifest["common_export_gain"] != 1):
        raise ValueError("多频几何清单集合、采样率或共同增益不符")
    required_sources = {"codes/chapters/ch03/core/geometry_audio.py",
                        "codes/chapters/ch03/examples/generate_geometry_audio.py",
                        "codes/chapters/ch03/core/geometry.py",
                        "codes/chapters/ch02/core/conventions.py",
                        "codes/chapters/ch00/core/audio_samples.py",
                        'codes/chapters/ch00/io_contracts.py'}
    if set(manifest["source_sha256"]) != required_sources:
        raise ValueError("多频几何必须记录真实生成源与PCM编码源")
    for name, digest in manifest["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"多频几何生成源摘要过期：{name}")
    for name in set(GEOMETRY_AUDIO_WAVS):
        path, record = source / name, manifest["files"][name]
        if (record["channels"] != GEOMETRY_AUDIO_WAVS[name] or record["samples_per_channel"] != 64000
                or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]):
            raise ValueError(f"多频几何摘要或参数不符：{name}")
        with wave.open(str(path), "rb") as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(),
                    wav.getnframes(), wav.getcomptype()) != (32000, GEOMETRY_AUDIO_WAVS[name], 2, 64000, "NONE"):
                raise ValueError(f"多频几何PCM格式不符：{name}")
            if len(wav.readframes(64000)) != 64000 * GEOMETRY_AUDIO_WAVS[name] * 2:
                raise ValueError(f"多频几何PCM数据截断：{name}")
    destination.mkdir()
    for name in sorted(expected):
        shutil.copy2(source / name, destination / name)
    return expected


def stage_focus_audio(source, destination):
    """Publish four known-unitary focusing fixtures with complete propagation tails."""
    _preflight_asset_stage(source, destination, set(FOCUS_AUDIO_WAVS) | {"MANIFEST.json"})
    import wave
    expected = set(FOCUS_AUDIO_WAVS) | {"MANIFEST.json"}
    if (source.is_symlink() or {p.name for p in source.iterdir()} != expected
            or any(p.is_symlink() or not p.is_file() for p in source.iterdir())):
        raise ValueError("已知酉聚焦目录必须恰含四份普通WAV及独立清单")
    manifest = _read_asset_manifest(source / "MANIFEST.json")
    if (set(manifest["files"]) != set(FOCUS_AUDIO_WAVS)
            or manifest["sample_rate_hz"] != 16000
            or manifest["common_export_gain"] != 1):
        raise ValueError("已知酉聚焦清单集合、采样率或共同增益不符")
    required_sources = {"codes/chapters/ch04/core/focus_audio.py",
                        "codes/chapters/ch04/examples/generate_focus_audio.py",
                        "codes/chapters/ch03/core/geometry.py",
                        "codes/chapters/ch02/core/conventions.py",
                        "codes/chapters/ch00/core/audio_samples.py",
                        'codes/chapters/ch00/io_contracts.py'}
    if set(manifest["source_sha256"]) != required_sources:
        raise ValueError("已知酉聚焦必须记录真实生成源与PCM编码源")
    for name, digest in manifest["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"已知酉聚焦生成源摘要过期：{name}")
    for name in set(FOCUS_AUDIO_WAVS):
        path, record = source / name, manifest["files"][name]
        if (record["channels"] != FOCUS_AUDIO_WAVS[name] or record["samples_per_channel"] != 32024
                or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]):
            raise ValueError(f"已知酉聚焦摘要或参数不符：{name}")
        with wave.open(str(path), "rb") as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(),
                    wav.getnframes(), wav.getcomptype()) != (16000, FOCUS_AUDIO_WAVS[name], 2, 32024, "NONE"):
                raise ValueError(f"已知酉聚焦PCM格式不符：{name}")
            if len(wav.readframes(32024)) != 32024 * FOCUS_AUDIO_WAVS[name] * 2:
                raise ValueError(f"已知酉聚焦PCM数据截断：{name}")
    destination.mkdir()
    for name in sorted(expected):
        shutil.copy2(source / name, destination / name)
    return expected


def stage_derivative_audio(source, destination):
    """Validate and copy E05-22 assets; numerical scores are checked by the generator."""
    _preflight_asset_stage(source, destination, set(DERIVATIVE_AUDIO_WAVS) | {"MANIFEST.json"})
    import wave
    expected = set(DERIVATIVE_AUDIO_WAVS) | {"MANIFEST.json"}
    if (source.is_symlink() or {p.name for p in source.iterdir()} != expected
            or any(p.is_symlink() or not p.is_file() for p in source.iterdir())):
        raise ValueError("导数约束目录必须恰含四份普通WAV及独立清单")
    manifest = _read_asset_manifest(source / "MANIFEST.json")
    if (set(manifest["files"]) != set(DERIVATIVE_AUDIO_WAVS)
            or manifest["sample_rate_hz"] != 16000 or manifest["samples_per_channel"] != 32002
            or manifest["common_export_gain"] != 1):
        raise ValueError("导数约束清单集合、采样率、完整长度或共同增益不符")
    required_sources = {"codes/chapters/ch05/core/derivative_audio.py",
                        "codes/chapters/ch05/examples/generate_derivative_audio.py",
                        "codes/chapters/ch05/core/beamforming.py",
                        "codes/chapters/ch04/core/covariance.py",
                        "codes/chapters/ch03/core/geometry.py",
                        "codes/chapters/ch02/core/conventions.py",
                        "codes/chapters/ch00/core/audio_samples.py",
                        'codes/chapters/ch00/io_contracts.py'}
    if set(manifest["source_sha256"]) != required_sources:
        raise ValueError("导数约束必须记录七个真实生成源")
    for name, digest in manifest["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"导数约束生成源摘要过期：{name}")
    for name, channels in DERIVATIVE_AUDIO_WAVS.items():
        path, record = source / name, manifest["files"][name]
        if (record["sample_rate_hz"] != 16000 or record["channels"] != channels
                or record["samples_per_channel"] != 32002
                or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]):
            raise ValueError(f"导数约束摘要或参数不符：{name}")
        with wave.open(str(path), "rb") as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(),
                    wav.getnframes(), wav.getcomptype()) != (16000, channels, 2, 32002, "NONE"):
                raise ValueError(f"导数约束PCM格式不符：{name}")
            if len(wav.readframes(32002)) != 32002*channels*2:
                raise ValueError(f"导数约束PCM数据截断：{name}")
    destination.mkdir()
    for name in sorted(expected):
        shutil.copy2(source / name, destination / name)
    return expected


def stage_apa_audio(source, destination):
    """Read and validate six real PCM files before atomic publication."""
    _preflight_asset_stage(source, destination, APA_AUDIO_WAVS | {"MANIFEST.json"})
    import sys
    if str(ROOT.resolve()) not in sys.path:
        sys.path.insert(0, str(ROOT.resolve()))
    from codes.chapters.ch06.examples.generate_apa_audio import check_assets
    check_assets(source, replay=True)
    expected = APA_AUDIO_WAVS | {"MANIFEST.json"}
    destination.mkdir()
    for name in sorted(expected):
        shutil.copy2(source / name, destination / name)
    return expected


def stage_reference_audio(source, destination):
    """Publish six known playback-gain/tail controls after complete read-only replay."""
    expected = REFERENCE_AUDIO_WAVS | {"MANIFEST.json"}
    _preflight_asset_stage(source, destination, expected)
    from codes.chapters.ch06.examples.generate_reference_audio import check_assets
    check_assets(source)
    destination.mkdir()
    for name in sorted(expected):
        shutil.copy2(source / name, destination / name)
    check_assets(destination)
    return expected


def stage_delay_audio(source, destination):
    """Publish known source-time controls after complete read-only replay."""
    expected = DELAY_AUDIO_WAVS | {"MANIFEST.json"}
    _preflight_asset_stage(source, destination, expected)
    from codes.chapters.ch07.examples.generate_delay_audio import check_assets
    check_assets(source)
    destination.mkdir()
    for name in sorted(expected):
        shutil.copy2(source / name, destination / name)
    check_assets(destination)
    return expected


def stage_mint_audio(source, destination):
    """Validate independent known-path inverse PCM before publication."""
    _preflight_asset_stage(source, destination, set(MINT_AUDIO_WAVS) | {"MANIFEST.json"})
    import sys
    if str(ROOT.resolve()) not in sys.path:
        sys.path.insert(0, str(ROOT.resolve()))
    from codes.chapters.ch07.examples.mint_teaching_demo import check_assets
    # Actual PCM scoring is followed by pure in-memory replay so all published
    # float decomposition and numerical metadata bind to the current sources.
    check_assets(source, replay=True)
    expected = set(MINT_AUDIO_WAVS) | {"MANIFEST.json"}
    destination.mkdir()
    for name in sorted(expected):
        shutil.copy2(source / name, destination / name)
    return expected


def stage_mask_audio(source, destination):
    """Validate actual PCM and replay the fixed FFT representation experiment."""
    _preflight_asset_stage(source, destination, MASK_AUDIO_WAVS | {"MANIFEST.json"})
    import sys
    if str(ROOT.resolve()) not in sys.path:
        sys.path.insert(0, str(ROOT.resolve()))
    from codes.chapters.ch08.examples.mask_representation_demo import check_assets
    check_assets(source, replay=True)
    expected = MASK_AUDIO_WAVS | {"MANIFEST.json"}
    destination.mkdir()
    for name in sorted(expected):
        shutil.copy2(source / name, destination / name)
    return expected


def stage_noise_audio(source, destination):
    """Publish only the fully replayed six-file noise-mismatch experiment."""
    _preflight_asset_stage(source, destination, NOISE_AUDIO_WAVS | {"MANIFEST.json"})
    from codes.chapters.ch10.examples.generate_noise_mismatch import check_assets, validate_asset_directory
    check_assets(source)
    validate_asset_directory(destination, check=False)
    expected = NOISE_AUDIO_WAVS | {"MANIFEST.json"}
    destination.mkdir()
    for name in sorted(expected):
        shutil.copy2(source / name, destination / name)
    return expected


def stage_scenario_audio(source, destination):
    """Publish the replayed eight-WAV chapter-11 scenario comparison."""
    _preflight_asset_stage(source, destination, SCENARIO_AUDIO_WAVS | {"MANIFEST.json"})
    from codes.chapters.ch11.examples.generate_selection_audio import check_assets, validate_asset_directory
    check_assets(source)
    validate_asset_directory(destination, check=False)
    expected = SCENARIO_AUDIO_WAVS | {"MANIFEST.json"}
    destination.mkdir()
    for name in sorted(expected):
        shutil.copy2(source / name, destination / name)
    return expected


def stage_weighted_audio(source, destination):
    """Publish only the replayed known-noise, five-WAV Appendix-A fixture."""
    _preflight_asset_stage(source, destination, WEIGHTED_AUDIO_WAVS | {"MANIFEST.json"})
    from codes.chapters.appendix_a.examples.generate_weighted_audio import (
        check_assets, validate_asset_directory,
    )
    check_assets(source)
    validate_asset_directory(destination, check=False)
    expected = WEIGHTED_AUDIO_WAVS | {"MANIFEST.json"}
    destination.mkdir()
    for name in sorted(expected):
        shutil.copy2(source / name, destination / name)
    return expected


def stage_response_audio(source, destination):
    """Publish the replayed five-WAV, equal-RIR-DRR Appendix-B fixture."""
    _preflight_asset_stage(source, destination, RESPONSE_AUDIO_WAVS | {"MANIFEST.json"})
    from codes.chapters.appendix_b.examples.generate_response_audio import (
        check_assets, validate_asset_directory,
    )
    check_assets(source)
    validate_asset_directory(destination, check=False)
    expected = RESPONSE_AUDIO_WAVS | {"MANIFEST.json"}
    destination.mkdir()
    for name in sorted(expected):
        shutil.copy2(source / name, destination / name)
    return expected


def stage_imaging_audio(source, destination):
    """Publish the five complete, replayed imaging snapshot fixtures."""
    expected = IMAGING_AUDIO_WAVS | {"MANIFEST.json"}
    _preflight_asset_stage(source, destination, expected)
    from codes.chapters.ch14.examples.generate_imaging_audio import check_assets
    check_assets(source)
    from codes.chapters.ch00.io_contracts import validate_asset_directory as validate_imaging
    validate_imaging(destination, expected, check=False)
    destination.mkdir()
    for name in sorted(expected):
        shutil.copy2(source / name, destination / name)
    return expected


def stage_distributed_audio(source, destination):
    """Publish seventeen fully replayed covariance/transport teaching WAVs."""
    expected = DISTRIBUTED_AUDIO_WAVS | {"MANIFEST.json"}
    _preflight_asset_stage(source, destination, expected)
    from codes.chapters.ch15.examples.generate_distributed_audio import check_assets
    check_assets(source)
    validate_asset_directory(destination, expected, check=False)
    destination.mkdir()
    for name in sorted(expected):
        shutil.copy2(source / name, destination / name)
    return expected


def _check_tracking_members(folder, expected):
    """Reject unexpected directories as well as linked or special members."""
    if (folder.is_symlink() or not folder.is_dir()
            or {p.name for p in folder.iterdir()} != set(expected)
            or any(p.is_symlink() or not p.is_file() for p in folder.iterdir())):
        raise ValueError(f"移动/追踪音频目录文件集合或普通文件类型不符：{folder}")


def _read_tracking_manifest(path):
    """Strict JSON: reject duplicate fields, non-finite values and overflow."""
    import math

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"音频清单重复字段：{key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"音频清单含非有限数：{value}")

    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("音频清单数值溢出或非有限")
        if isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, list):
            for item in value:
                finite(item)

    result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                        parse_constant=constant)
    if not isinstance(result, dict):
        raise ValueError("音频清单根必须为对象")
    finite(result)
    return result


def stage_tracking_audio(source, destination):
    """Validate the independent PCM-to-observation experiment before publishing."""
    import wave
    expected = set(TRACKING_AUDIO_WAVS) | {"MANIFEST.json"}
    _check_tracking_members(source, expected)
    manifest = _read_tracking_manifest(source/"MANIFEST.json")
    if set(manifest["files"]) != set(TRACKING_AUDIO_WAVS) or manifest["sample_rate_hz"] != 16000:
        raise ValueError("追踪音频清单集合或采样率不符")
    for name, channels in TRACKING_AUDIO_WAVS.items():
        path, record = source/name, manifest["files"][name]
        if (record["channels"] != channels or record["samples_per_channel"] != 32000
                or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]):
            raise ValueError(f"追踪音频摘要或参数不符：{name}")
        with wave.open(str(path),"rb") as wav:
            if (wav.getframerate(),wav.getnchannels(),wav.getsampwidth(),
                    wav.getnframes(),wav.getcomptype()) != (16000,channels,2,32000,"NONE"):
                raise ValueError(f"追踪音频PCM格式不符：{name}")
    from codes.chapters.ch09.examples.chapter09_tracking_audio import generate, _validate_directory
    generate(source, check=True)
    _validate_directory(destination, expected, check=False)
    _validate_publish_targets([destination / name for name in expected], boundary=destination.parent)
    destination.mkdir()
    for name in sorted(expected):
        shutil.copy2(source/name,destination/name)
    return expected

def stage_gss_audio(source, destination):
    """核对受控 GSS 教学链的五路 PCM 与可复算中间状态。"""
    _preflight_asset_stage(source, destination, set(GSS_AUDIO_WAVS) | {"STATE.npz", "MANIFEST.json"})
    import sys
    if str(ROOT.resolve()) not in sys.path:
        sys.path.insert(0, str(ROOT.resolve()))
    from codes.chapters.ch08.examples.gss_teaching_demo import generate
    generate(source, check=True)
    import wave
    import zipfile
    manifest = _read_asset_manifest(source / "MANIFEST.json")
    records = manifest["files"]
    expected = set(GSS_AUDIO_WAVS) | {"STATE.npz", "MANIFEST.json"}
    if (set(records) != expected - {"MANIFEST.json"}
            or {p.name for p in source.iterdir() if p.is_file()} - {"README.md"} != expected
            or manifest["sample_rate_hz"] != 16000):
        raise ValueError("GSS 独立清单、文件集合或采样率不符")
    source_paths = {"codes/chapters/ch08/examples/gss_teaching_demo.py",
                    "codes/chapters/ch08/core/gss_teaching.py",
                    "codes/chapters/ch08/core/separation.py",
                    "codes/chapters/ch02/core/spectral.py",
                    "codes/chapters/ch02/core/conventions.py",
                    "codes/chapters/ch00/core/audio_samples.py",
                    "codes/chapters/ch07/core/dereverberation.py",
                    'codes/chapters/ch00/io_contracts.py'}
    if set(manifest.get("generator_inputs", {})) != source_paths:
        raise ValueError("GSS 生成源码清单不完整")
    for name, digest in manifest["generator_inputs"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"GSS 生成源码已变化：{name}")
    for name in expected - {"MANIFEST.json"}:
        path = source / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != records[name]["sha256"]:
            raise ValueError(f"GSS 资产摘要不符：{name}")
        if name == "STATE.npz":
            if not zipfile.is_zipfile(path):
                raise ValueError("GSS 中间状态不是 NPZ")
            continue
        with wave.open(str(path), "rb") as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(),
                    wav.getnframes(), wav.getcomptype()) != (
                        16000, GSS_AUDIO_WAVS[name], 2,
                        records[name]["samples_per_channel"], "NONE"):
                raise ValueError(f"GSS PCM 格式不符：{name}")
    destination.mkdir()
    for name in sorted(expected):
        shutil.copy2(source / name, destination / name)
    return expected


def clean_label(text):
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"[`*_~]", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def source_digest():
    """站点正文与构建器的稳定摘要，用于拒绝陈旧生成物。"""
    digest = hashlib.sha256()
    paths = sorted(SRC.glob("*.md"))
    paths += [RESEARCH_ROOT / name for name, _ in RESEARCH]
    paths += [main_audio_manifest_path(CODE_CHAPTERS)]
    paths += sorted(main_audio_sources())
    for asset_root in (REAL_AUDIO_ROOT, ROOM_AUDIO_ROOT, MOVING_AUDIO_ROOT,
                       TRACKING_AUDIO_ROOT, GSS_AUDIO_ROOT, BINAURAL_AUDIO_ROOT, SPECTRAL_AUDIO_ROOT, STFT_AUDIO_ROOT, SWEEP_AUDIO_ROOT, BASELINE_AUDIO_ROOT, REFLECTION_AUDIO_ROOT, PHASE_AUDIO_ROOT, GEOMETRY_AUDIO_ROOT, FOCUS_AUDIO_ROOT, DERIVATIVE_AUDIO_ROOT, APA_AUDIO_ROOT, REFERENCE_AUDIO_ROOT, DELAY_AUDIO_ROOT, MINT_AUDIO_ROOT, MASK_AUDIO_ROOT, NOISE_AUDIO_ROOT, SCENARIO_AUDIO_ROOT, WEIGHTED_AUDIO_ROOT, RESPONSE_AUDIO_ROOT, IMAGING_AUDIO_ROOT, DISTRIBUTED_AUDIO_ROOT):
        paths += sorted(asset_root.glob("*"))
    paths += sorted((ROOT / "figures").glob("fig*.png"))
    paths += [Path(__file__), ROOT / "scripts" / "build_markdown_helpers.py",
              INLINE_LAYOUT_PATH,
              ROOT / "scripts" / "heading_aliases.py",
              ROOT / "scripts" / "legacy_sequential_anchors.json",
              ROOT / "scripts" / "code_layout.py",
              ROOT / "scripts" / "make_figures.py",
              ROOT / "scripts" / "make_aec_figures.py",
              ROOT / "scripts" / "make_beamforming_figures.py",
              ROOT / "scripts" / "make_reference_figures.py",
              ROOT / "scripts" / "make_delay_figures.py", ROOT / "requirements.txt"]
    paths.append(ROOT / "codes/chapters/ch00/io_contracts.py")
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def parse_headings(md):
    """Use the pinned Markdown grammar while retaining our label/anchor policy."""
    return [(level, clean_label(label)) for level, label in parsed_markdown_headings(md)
            if level <= 4]


def heading_anchor(text, fallback_index):
    """编号标题使用 sec-x-y；无编号标题使用内容摘要。"""
    label = clean_label(re.sub(r"<[^>]+>", "", text))
    numbered = re.match(r"^(\d+(?:\.\d+)+)(?=\s|$)", label)
    if numbered:
        return "sec-" + numbered.group(1).replace(".", "-")
    if label:
        token = hashlib.sha1(label.encode("utf-8")).hexdigest()[:10]
        return f"sec-u-{token}"
    return f"sec-{fallback_index}"


def heading_records(heads):
    """返回 (level, text, primary_id, legacy_id)，稳定处理重名标题。"""
    seen = Counter()
    records = []
    for index, (level, text) in enumerate(heads, 1):
        base = heading_anchor(text, index)
        seen[base] += 1
        primary = base if seen[base] == 1 else f"{base}-{seen[base]}"
        records.append((level, text, primary, f"sec-{index}"))
    return records


def source_outputs():
    """显式发布清单：未发布的源码不能仅按后缀猜成 HTML。"""
    return {
        **{(SRC / name).resolve(): name.replace(".md", ".html")
           for name, _ in CHAPTERS},
        (SRC / HOME_FNAME).resolve(): "index.html",
        **{(RESEARCH_ROOT / name).resolve():
           "research/" + ("index.html" if name == "README.md" else name.replace(".md", ".html"))
           for name, _ in RESEARCH},
    }


def local_link_target(href, source_path):
    """返回 URI 与仓库内绝对目标；外部 URI、页内锚点和越界路径不解析。"""
    parsed = urlsplit(href)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return parsed, None
    target = (Path(source_path).parent / unquote(parsed.path)).resolve()
    if not target.is_relative_to(ROOT.resolve()):
        return parsed, None
    return parsed, target


def repository_url(parsed, target):
    path = quote(target.relative_to(ROOT.resolve()).as_posix(), safe="/")
    return urlunsplit(("https", "github.com",
                      "/nanless/microphone-array-signal-processing/blob/main/" + path,
                      parsed.query, parsed.fragment))


def rewrite_href_targets(html, transform):
    """仅改真正链接的 href，不改外部网址、文本或代码中的 .md。"""
    def replace(match):
        value = transform(unescape(match.group(3)))
        return match.group(1) + match.group(2) + escape(value, quote=True) + match.group(2)
    return re.sub(r'(<a\b[^>]*?\bhref=)([\"\'])(.*?)\2', replace, html, flags=re.S)


def rewrite_room_image_src(html, source_path, current):
    """Map actual image sources to published room assets or repository figures."""
    def replace(match):
        parsed, target = local_link_target(unescape(match.group(3)), source_path)
        if target == (ROOM_AUDIO_ROOT / "ROOM_RESULTS.png").resolve():
            output = "room_audio/ROOM_RESULTS.png"
        elif (target is not None and target.parent == (ROOT / "figures").resolve()
              and target.suffix.lower() == ".png" and target.is_file()):
            output = "../figures/" + target.name
        else:
            return match.group(0)
        relative = os.path.relpath(output, Path(current).parent).replace(os.sep, "/")
        value = urlunsplit(("", "", relative, parsed.query, parsed.fragment))
        return match.group(1) + match.group(2) + escape(value, quote=True) + match.group(2)
    return re.sub(r'(<img\b[^>]*?\s+src\s*=\s*)(["\x27])(.*?)\2', replace, html, flags=re.S)


def rewrite_site_links(html, source_path):
    outputs = source_outputs()
    current = outputs.get(Path(source_path).resolve(), "index.html")

    def transform(href):
        parsed, target = local_link_target(href, source_path)
        if target is None:
            return href
        if target in outputs:
            relative = os.path.relpath(outputs[target], Path(current).parent).replace(os.sep, "/")
            return urlunsplit(("", "", relative, parsed.query, parsed.fragment))
        if target.parent == (ROOT / "figures").resolve() and target.suffix.lower() == ".png" and target.is_file():
            relative = os.path.relpath("../figures/" + target.name, Path(current).parent).replace(os.sep, "/")
            return urlunsplit(("", "", relative, parsed.query, parsed.fragment))
        main_audio = main_audio_sources()
        if target in main_audio:
            relative = os.path.relpath("audio/" + main_audio[target], Path(current).parent).replace(os.sep, "/")
            return urlunsplit(("", "", relative, parsed.query, parsed.fragment))
        if target.parent == REAL_AUDIO_ROOT.resolve() and target.name in REAL_AUDIO_FILES:
            relative = os.path.relpath("real_audio/" + target.name, Path(current).parent).replace(os.sep, "/")
            return urlunsplit(("", "", relative, parsed.query, parsed.fragment))
        if target.parent == ROOM_AUDIO_ROOT.resolve():
            relative = os.path.relpath("room_audio/" + target.name, Path(current).parent).replace(os.sep, "/")
            return urlunsplit(("", "", relative, parsed.query, parsed.fragment))
        if target.parent == TRACKING_AUDIO_ROOT.resolve() and target.name in (set(TRACKING_AUDIO_WAVS) | {"MANIFEST.json"}):
            relative = os.path.relpath("tracking_audio/" + target.name, Path(current).parent).replace(os.sep, "/")
            return urlunsplit(("", "", relative, parsed.query, parsed.fragment))
        if target.parent == MOVING_AUDIO_ROOT.resolve() and target.name in (set(MOVING_AUDIO_WAVS) | {"MANIFEST.json"}):
            relative = os.path.relpath("moving_audio/" + target.name, Path(current).parent).replace(os.sep, "/")
            return urlunsplit(("", "", relative, parsed.query, parsed.fragment))
        if target.parent == GSS_AUDIO_ROOT.resolve() and target.name in (set(GSS_AUDIO_WAVS) | {"MANIFEST.json", "STATE.npz"}):
            relative = os.path.relpath("gss_audio/" + target.name, Path(current).parent).replace(os.sep, "/")
            return urlunsplit(("", "", relative, parsed.query, parsed.fragment))
        if target.parent == BINAURAL_AUDIO_ROOT.resolve() and target.name in (BINAURAL_AUDIO_WAVS | {"MANIFEST.json"}):
            relative = os.path.relpath("binaural_audio/" + target.name, Path(current).parent).replace(os.sep, "/")
            return urlunsplit(("", "", relative, parsed.query, parsed.fragment))
        if target.parent == SPECTRAL_AUDIO_ROOT.resolve() and target.name in (SPECTRAL_AUDIO_WAVS | {"MANIFEST.json"}):
            relative = os.path.relpath("spectral_audio/" + target.name, Path(current).parent).replace(os.sep, "/")
            return urlunsplit(("", "", relative, parsed.query, parsed.fragment))
        if target.parent == STFT_AUDIO_ROOT.resolve() and target.name in (STFT_AUDIO_WAVS | {"MANIFEST.json"}):
            relative = os.path.relpath("stft_audio/" + target.name, Path(current).parent).replace(os.sep, "/")
            return urlunsplit(("", "", relative, parsed.query, parsed.fragment))
        if target.parent == SWEEP_AUDIO_ROOT.resolve() and target.name in (SWEEP_AUDIO_WAVS | {'MANIFEST.json'}):
            relative = os.path.relpath('sweep_audio/' + target.name, Path(current).parent).replace(os.sep, '/')
            return urlunsplit(('', '', relative, parsed.query, parsed.fragment))
        if target.parent == BASELINE_AUDIO_ROOT.resolve() and target.name in (BASELINE_AUDIO_WAVS | {'MANIFEST.json'}):
            relative = os.path.relpath('baseline_audio/' + target.name, Path(current).parent).replace(os.sep, '/')
            return urlunsplit(('', '', relative, parsed.query, parsed.fragment))
        if target.parent == REFLECTION_AUDIO_ROOT.resolve() and target.name in (REFLECTION_AUDIO_WAVS | {'MANIFEST.json'}):
            relative = os.path.relpath('reflection_audio/' + target.name, Path(current).parent).replace(os.sep, '/')
            return urlunsplit(('', '', relative, parsed.query, parsed.fragment))
        if target.parent == PHASE_AUDIO_ROOT.resolve() and target.name in (PHASE_AUDIO_WAVS | {'MANIFEST.json'}):
            relative = os.path.relpath('phase_audio/' + target.name, Path(current).parent).replace(os.sep, '/')
            return urlunsplit(('', '', relative, parsed.query, parsed.fragment))
        if target.parent == GEOMETRY_AUDIO_ROOT.resolve() and target.name in (set(GEOMETRY_AUDIO_WAVS) | {"MANIFEST.json"}):
            relative = os.path.relpath("geometry_audio/" + target.name, Path(current).parent).replace(os.sep, "/")
            return urlunsplit(("", "", relative, parsed.query, parsed.fragment))
        if target.parent == DERIVATIVE_AUDIO_ROOT.resolve() and target.name in (set(DERIVATIVE_AUDIO_WAVS) | {"MANIFEST.json"}):
            relative = os.path.relpath("derivative_audio/" + target.name, Path(current).parent).replace(os.sep, "/")
            return urlunsplit(("", "", relative, parsed.query, parsed.fragment))
        if target.parent == APA_AUDIO_ROOT.resolve() and target.name in (APA_AUDIO_WAVS | {"MANIFEST.json"}):
            relative = os.path.relpath("apa_audio/" + target.name, Path(current).parent).replace(os.sep, "/")
            return urlunsplit(("", "", relative, parsed.query, parsed.fragment))
        if target.parent == REFERENCE_AUDIO_ROOT.resolve() and target.name in (REFERENCE_AUDIO_WAVS | {"MANIFEST.json"}):
            relative = os.path.relpath("reference_audio/" + target.name, Path(current).parent).replace(os.sep, "/")
            return urlunsplit(("", "", relative, parsed.query, parsed.fragment))
        if target.parent == DELAY_AUDIO_ROOT.resolve() and target.name in (DELAY_AUDIO_WAVS | {"MANIFEST.json"}):
            relative = os.path.relpath("delay_audio/" + target.name, Path(current).parent).replace(os.sep, "/")
            return urlunsplit(("", "", relative, parsed.query, parsed.fragment))
        if target.parent == MINT_AUDIO_ROOT.resolve() and target.name in (set(MINT_AUDIO_WAVS) | {"MANIFEST.json"}):
            relative = os.path.relpath("mint_audio/" + target.name, Path(current).parent).replace(os.sep, "/")
            return urlunsplit(("", "", relative, parsed.query, parsed.fragment))
        if target.parent == SCENARIO_AUDIO_ROOT.resolve() and target.name in (SCENARIO_AUDIO_WAVS | {"MANIFEST.json"}):
            relative = os.path.relpath("scenario_audio/" + target.name, Path(current).parent).replace(os.sep, "/")
            return urlunsplit(("", "", relative, parsed.query, parsed.fragment))
        if target.parent == RESPONSE_AUDIO_ROOT.resolve() and target.name in (RESPONSE_AUDIO_WAVS | {"MANIFEST.json"}):
            relative = os.path.relpath("response_audio/" + target.name, Path(current).parent).replace(os.sep, "/")
            return urlunsplit(("", "", relative, parsed.query, parsed.fragment))
        if target.parent == DISTRIBUTED_AUDIO_ROOT.resolve() and target.name in (DISTRIBUTED_AUDIO_WAVS | {"MANIFEST.json"}):
            relative = os.path.relpath("distributed_audio/" + target.name, Path(current).parent).replace(os.sep, "/")
            return urlunsplit(("", "", relative, parsed.query, parsed.fragment))
        if target.parent == IMAGING_AUDIO_ROOT.resolve() and target.name in (IMAGING_AUDIO_WAVS | {"MANIFEST.json"}):
            relative = os.path.relpath("imaging_audio/" + target.name, Path(current).parent).replace(os.sep, "/")
            return urlunsplit(("", "", relative, parsed.query, parsed.fragment))
        if target.parent == WEIGHTED_AUDIO_ROOT.resolve() and target.name in (WEIGHTED_AUDIO_WAVS | {"MANIFEST.json"}):
            relative = os.path.relpath("weighted_audio/" + target.name, Path(current).parent).replace(os.sep, "/")
            return urlunsplit(("", "", relative, parsed.query, parsed.fragment))
        if target.parent == NOISE_AUDIO_ROOT.resolve() and target.name in (NOISE_AUDIO_WAVS | {"MANIFEST.json"}):
            relative = os.path.relpath("noise_audio/" + target.name, Path(current).parent).replace(os.sep, "/")
            return urlunsplit(("", "", relative, parsed.query, parsed.fragment))
        if target.parent == MASK_AUDIO_ROOT.resolve() and target.name in (MASK_AUDIO_WAVS | {"MANIFEST.json"}):
            relative = os.path.relpath("mask_audio/" + target.name, Path(current).parent).replace(os.sep, "/")
            return urlunsplit(("", "", relative, parsed.query, parsed.fragment))
        if target.parent == FOCUS_AUDIO_ROOT.resolve() and target.name in (set(FOCUS_AUDIO_WAVS) | {"MANIFEST.json"}):
            relative = os.path.relpath("focus_audio/" + target.name, Path(current).parent).replace(os.sep, "/")
            return urlunsplit(("", "", relative, parsed.query, parsed.fragment))
        return repository_url(parsed, target)

    html = rewrite_href_targets(html, transform)
    html = rewrite_room_image_src(html, source_path, current)
    # Only generated local WAV links gain controls; no autoplay and no remote media.
    def player(match):
        href, label = unescape(match.group(1)), match.group(2)
        parsed = urlsplit(href)
        # Browsers may reject or downmix 16 channels. Preserve the analysis
        # input as a download link; only the explicit mono derivatives play.
        if parsed.path.endswith("real_audio/demand_nriver_16ch_10s.wav"):
            return match.group(0)
        if parsed.scheme or parsed.query or parsed.fragment or not re.fullmatch(r"(?:\.\./)?(?:audio|real_audio|moving_audio|tracking_audio|gss_audio|binaural_audio|spectral_audio|stft_audio|sweep_audio|baseline_audio|reflection_audio|phase_audio|geometry_audio|focus_audio|derivative_audio|apa_audio|reference_audio|delay_audio|mint_audio|mask_audio|noise_audio|scenario_audio|weighted_audio|response_audio|imaging_audio|distributed_audio)/[a-z0-9_]+\.wav", parsed.path):
            return match.group(0)
        safe_href = escape(href, quote=True)
        safe_label = escape(re.sub(r'<[^>]+>', '', unescape(label)), quote=True)
        return (match.group(0) + f'<audio class="audio-sample" controls preload="none" '
                f'aria-label="{safe_label}" src="{safe_href}">请使用上方 WAV 链接下载。</audio>')
    return re.sub(r'<a href="([^"]+)">(.*?)</a>', player, html, flags=re.S)


def render(md_text, source_path=None):
    """返回 (html, 标题数)；主标识稳定，并保留旧 sec-N 别名。"""
    import markdown
    source_path = Path(source_path) if source_path is not None else SRC / HOME_FNAME
    try:
        from scripts.heading_aliases import historical_aliases, has_historical_sequential_aliases
    except ModuleNotFoundError:
        from heading_aliases import historical_aliases, has_historical_sequential_aliases
    is_research = source_path.resolve().parent == RESEARCH_ROOT.resolve()
    records = heading_records(parse_headings(md_text))
    html = render_markdown(md_text)
    validate_url_schemes(html)
    counter = [0]
    research_slugs = Counter()

    def repl(m):
        counter[0] += 1
        tag, inner = m.group(1), m.group(2)
        _level, _text, primary, legacy = records[counter[0] - 1]
        alias = ("" if primary == legacy or has_historical_sequential_aliases(source_path.name) else
                 f'<span id="{legacy}" class="anchor-alias" aria-hidden="true"></span>')
        alias += "".join(
            f'<span id="{old}" class="anchor-alias" aria-hidden="true"></span>'
            for old in historical_aliases(source_path.name, primary)
            if old != primary
        )
        if is_research:
            # 保留 Markdown/GitHub 风格的研究页深链，同时沿用站点目录标识。
            slug = re.sub(r"\s+", "-", re.sub(r"[^\w\s-]", "", _text.lower()))
            research_slugs[slug] += 1
            if research_slugs[slug] > 1:
                slug += f"-{research_slugs[slug] - 1}"
            alias += f'<span id="{escape(slug, quote=True)}" class="anchor-alias" aria-hidden="true"></span>'
        return f'{alias}<{tag} id="{primary}">{inner}</{tag}>'

    html = re.sub(r"<(h[1-4])>(.*?)</\1>", repl, html, flags=re.S)
    html = re.sub(r'<th(?=[\s>])(?![^>]*\bscope=)([^>]*)>',
                  r'<th scope="col"\1>', html, flags=re.S)
    # These engineering tables need phrases rather than single-character
    # columns on phones. Keep the change local to chapter 6 and screen CSS.
    if source_path.name == "06_aec.md":
        def aec_table(match):
            inner = match.group(1)
            headers = re.findall(r'<th\b[^>]*>(.*?)</th>', inner, flags=re.S)
            labels = {unescape(re.sub(r'<[^>]+>', '', value)).strip() for value in headers}
            readable = bool(labels & {"训练参考", "仍需实测的资源", "它实际解决什么"})
            return '<table'+(' class="aec-readable-table"' if readable else '')+'>'+inner+'</table>'
        html = re.sub(r'<table>(.*?)</table>', aec_table, html, flags=re.S)
    if source_path.name == "07_wpe-dereverberation.md":
        def wpe_table(match):
            inner = match.group(1)
            headers = re.findall(r'<th\b[^>]*>(.*?)</th>', inner, flags=re.S)
            labels = [unescape(re.sub(r'<[^>]+>', '', value)).strip() for value in headers]
            readable = labels == ["实现", "主要源码", "与教学基线的差别", "最小核对实验"]
            return '<table'+(' class="wpe-readable-table"' if readable else '')+'>'+inner+'</table>'
        html = re.sub(r'<table>(.*?)</table>', wpe_table, html, flags=re.S)
    if source_path.name == "08_speech-separation.md":
        def separation_table(match):
            inner = match.group(1)
            headers = re.findall(r'<th\b[^>]*>(.*?)</th>', inner, flags=re.S)
            labels = tuple(unescape(re.sub(r'<[^>]+>', '', value)).strip() for value in headers)
            readable = labels in {
                ("路线", "代表", "思想", "局限"),
                ("结构", "代表", "主要结构", "结果复现要求", "优点与限制"),
                ("固定时轴输出", "解析MSE", "PCM误差整数能量 $E$", "实际PCM MSE", "实际PCM相对平方误差"),
            }
            return '<table'+(' class="separation-readable-table"' if readable else '')+'>'+inner+'</table>'
        html = re.sub(r'<table>(.*?)</table>', separation_table, html, flags=re.S)
    if source_path.name == "09_source-tracking.md":
        def tracking_table(match):
            inner = match.group(1)
            headers = re.findall(r'<th\b[^>]*>(.*?)</th>', inner, flags=re.S)
            labels = tuple(unescape(re.sub(r'<[^>]+>', '', value)).strip() for value in headers)
            readable = labels in {
                ("方法", "原理", "优点", "缺点", "适用"),
                ("方法族", "代表", "思想", "声学表现"),
            }
            return '<table'+(' class="tracking-readable-table"' if readable else '')+'>'+inner+'</table>'
        html = re.sub(r'<table>(.*?)</table>', tracking_table, html, flags=re.S)
    if source_path.name == "10_engineering-practice.md":
        def noise_table(match):
            inner = match.group(1)
            headers = re.findall(r'<th\b[^>]*>(.*?)</th>', inner, flags=re.S)
            labels = tuple(unescape(re.sub(r'<[^>]+>', '', value)).strip() for value in headers)
            readable = labels in {
                ("输出", "阶跃前浮点 MSE", "阶跃前 PCM MSE", "阶跃后浮点 MSE", "阶跃后 PCM MSE"),
                ("支路", "噪声功率来源", "能检查的问题"),
            }
            return '<table'+(' class="noise-readable-table"' if readable else '')+'>'+inner+'</table>'
        html = re.sub(r'<table>(.*?)</table>', noise_table, html, flags=re.S)
    if source_path.name == "11_selection-guide.md":
        def selection_table(match):
            inner = match.group(1)
            headers = re.findall(r'<th\b[^>]*>(.*?)</th>', inner, flags=re.S)
            labels = tuple(unescape(re.sub(r'<[^>]+>', '', value)).strip() for value in headers)
            readable = labels in {
                ("决策对象", "适合先尝试的方案", "何时换方案", "同时检查的风险"),
                ("任务条件", "一路波束/掩码波束", "连续语音分离（CSS）", "目标说话人提取（TSE）"),
                ("条件或证据", "一路掩码波束", "两路 CSS", "注册声纹 TSE", "决定"),
            }
            return '<table'+(' class="selection-readable-table"' if readable else '')+'>'+inner+'</table>'
        html = re.sub(r'<table>(.*?)</table>', selection_table, html, flags=re.S)
    # 保留 table 原生语义；横向滚动由可聚焦的外层区域承担，键盘用户也能操作宽表。
    if source_path == RESEARCH_ROOT / "03_industrial_deployment.md":
        def industrial_table(match):
            inner = match.group(1)
            headers = re.findall(r'<th\b[^>]*>(.*?)</th>', inner, flags=re.S)
            labels = tuple(unescape(re.sub(r'<[^>]+>', '', value)).strip() for value in headers)
            readable = labels in {
                ("对照", "输出总帧数", "六个脉冲相对参考时间的峰值采样点偏移（帧）", "1～9 s 偏移增量（帧）", "部分消费调用", "输入取尽后补出帧数"),
                ("接口", "需要核对的对象", "不能省略的条件"),
                ("阅读位置", "要核对的变量/步骤", "独立判据"),
                ("固定对象", "实际执行与输入", "独立核对结果", "代替项和未执行范围"),
            }
            return '<table'+(' class="industrial-readable-table"' if readable else '')+'>'+inner+'</table>'
        html = re.sub(r'<table>(.*?)</table>', industrial_table, html, flags=re.S)
    if source_path == RESEARCH_ROOT / "04_source_reproduction.md":
        def source_contract_table(match):
            inner = match.group(1)
            headers = re.findall(r'<th\b[^>]*>(.*?)</th>', inner, flags=re.S)
            labels = tuple(unescape(re.sub(r'<[^>]+>', '', value)).strip() for value in headers)
            readable = labels == ("核查对象", "固定源中实际结构", "证据所能支持的范围")
            return '<table'+(' class="source-contract-readable-table"' if readable else '')+'>'+inner+'</table>'
        html = re.sub(r'<table>(.*?)</table>', source_contract_table, html, flags=re.S)
    policies = NARROW_TABLE_POLICIES.get(source_path.resolve(), {})
    if policies:
        def budget_table(match):
            attrs, inner = match.group(1), match.group(2)
            headers = re.findall(r'<th\b[^>]*>(.*?)</th>', inner, flags=re.S)
            labels = tuple(unescape(re.sub(r'<[^>]+>', '', value)).strip() for value in headers)
            policy = policies.get(labels)
            if policy is None:
                return match.group(0)
            width, columns = policy
            def layout_attrs(original, class_name, variable):
                # Raw HTML can already carry class/style; never emit duplicate
                # attributes or discard its existing presentation contract.
                for name, addition, separator in (("class", class_name, " "), ("style", variable, ";")):
                    pattern = rf'\b{name}\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s>]+))'
                    existing = re.search(pattern, original)
                    if existing:
                        value = next(value for value in existing.groups() if value is not None)
                        value = value.rstrip(";") + separator + addition
                        original = (original[:existing.start()] +
                                    f'{name}="{escape(unescape(value), quote=True)}"' + original[existing.end():])
                    else:
                        original += f' {name}="{addition}"'
                return original
            # Preserve native table roles, header scope, text, links and TeX.
            attrs = layout_attrs(attrs, "tutorial-budget-table", f"--tutorial-table-width:{width}em")
            def budget_row(row):
                column = 0
                def budget_cell(cell):
                    nonlocal column
                    column += 1
                    if column not in columns:
                        return cell.group(0)
                    tag, cell_attrs, content = cell.group(1), cell.group(2), cell.group(3)
                    cell_attrs = layout_attrs(cell_attrs, "tutorial-short-column", f"--tutorial-column-width:{columns[column]}em")
                    return f'<{tag}{cell_attrs}>{content}</{tag}>'
                return re.sub(r'<(th|td)(\b[^>]*)>(.*?)</\1>', budget_cell, row.group(0), flags=re.S)
            inner = re.sub(r'<tr\b[^>]*>.*?</tr>', budget_row, inner, flags=re.S)
            return f'<table{attrs}>{inner}</table>'
        html = re.sub(r'<table([^>]*)>(.*?)</table>', budget_table, html, flags=re.S)
    html = re.sub(
        r"<table([^>]*)>(.*?)</table>",
        (r'<div class="table-scroll" tabindex="0" role="region" '
         r'aria-label="数据表，可横向滚动"><table\1>\2</table></div>'),
        html,
        flags=re.S,
    )
    html = rewrite_site_links(html, source_path)

    # 导航链文 guilty .md 后缀 → 篇名（F21）
    def nav_text(m):
        href, text = m.group(1), m.group(2).strip()
        base = href.split("#")[0].split("/")[-1]
        if base == "index.html" and text.endswith(".md"):
            return f'<a href="{href}">🏠 导读与导航</a>'
        for fname, label in LABEL_BY_FNAME.items():
            if base == fname.replace(".md", ".html") and text == fname:
                return f'<a href="{href}">{label}</a>'
        return m.group(0)

    html = re.sub(r'<code>(<a href="[^"]+">[^<>]+</a>)</code>', r'\1', html)
    html = re.sub(r'<a href="([^"]+)">(?:<code>)?([^<>]*\.md)(?:</code>)?</a>', nav_text, html)
    # 图片加载失败占位
    html = re.sub(r'<img ([^>]*?)src="([^"]+)"',
                  r'''<img \1src="\2" onerror="this.alt+='（图缺失）'"''', html)
    # 表格里的裸文件名（01_xxx.md）也改成可点链接
    def bare_link(m):
        pre, fname = m.group(1), m.group(2)
        prefix = "../" if is_research else ""
        if fname == HOME_FNAME:
            return f'{pre}<a href="{prefix}index.html">{fname}</a>'
        if fname in LABEL_BY_FNAME:
            return f'{pre}<a href="{prefix}{fname.replace(".md", ".html")}">{fname}</a>'
        return m.group(0)

    html = map_table_cell_text(html, lambda text: re.sub(
        r'(^|[\s>(])((?:0\d|1\d)_[^<\s)"]+\.md)', bare_link, text))
    return html, counter[0]


def promote_content_headings(html, heads):
    """内容页源文件以 h2 写篇名；站点中提升一级，得到唯一 h1 和连续层级。"""
    for old, new in (("h2", "h1"), ("h3", "h2"), ("h4", "h3")):
        html = re.sub(fr"<{old}([^>]*)>(.*?)</{old}>",
                      fr"<{new}\1>\2</{new}>", html, flags=re.S)
    promoted = [(max(1, level - 1), text) for level, text in heads]
    return html, promoted


def sub_list(html_name, heads, start_idx=1, include_level1=False):
    """当前页的小节目录，返回 (html, 下一起始编号)。编号与 render 同序。"""
    parts = ["<ul>"]
    idx = start_idx
    for lvl, text, primary, _legacy in heading_records(heads):
        if lvl >= 2 or (include_level1 and lvl == 1):
            pad = ("" if lvl <= 2 else
                   ("&nbsp;&nbsp;" if lvl == 3 else "&nbsp;&nbsp;&nbsp;&nbsp;— "))
            parts.append(f'<li>{pad}<a href="{html_name}#{primary}">{text}</a></li>')
        idx += 1
    parts.append("</ul>")
    return "\n".join(parts), idx


def sidebar_with_anchors(current, heads):
    """current: None=首页。首页展开自己的目录（修首页零锚点bug）。"""
    parts = []
    if current is None:
        parts.append('<div class="chap cur" aria-current="page">🏠 导读与导航（首页）</div>')
        lst, _ = sub_list("index.html", heads)
        parts.append(lst)
    else:
        parts.append('<div class="chap"><a href="index.html">🏠 导读与导航（首页）</a></div>')
    for fname, label in CHAPTERS:
        html_name = fname.replace(".md", ".html")
        if fname == current:
            parts.append(f'<div class="chap cur" aria-current="page">{label}</div>')
            # 内容页的源 h2 会提升为 h1，但仍是源导航基线的一部分。
            lst, _ = sub_list(html_name, heads, include_level1=True)
            parts.append(lst)
        else:
            parts.append(f'<div class="chap"><a href="{html_name}">{label}</a></div>')
    parts.append('<div class="chap"><a href="research/index.html">源码研究</a></div>')
    return "\n".join(parts)


def research_sidebar(current, heads):
    parts = ['<div class="chap"><a href="../index.html">🏠 导读与导航（首页）</a></div>']
    for fname, label in RESEARCH:
        html_name = "index.html" if fname == "README.md" else fname.replace(".md", ".html")
        if fname == current:
            parts.append(f'<div class="chap cur" aria-current="page">{label}</div>')
            parts.append(sub_list(html_name, heads)[0])
        else:
            parts.append(f'<div class="chap"><a href="{html_name}">{label}</a></div>')
    for fname, label in CHAPTERS:
        parts.append(f'<div class="chap"><a href="../{fname.replace(".md", ".html")}">{label}</a></div>')
    return "\n".join(parts)


def _validate_publish_targets(targets, boundary=None):
    """Preflight links before backups, replacement or stale-file removal."""
    for target in map(Path, targets):
        base = Path(boundary) if boundary is not None else target.parent
        if not target.parent.is_relative_to(base):
            raise ValueError('发布目标越出指定目录')
        for parent in (target.parent, *target.parent.parents):
            if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                raise ValueError('发布目录必须为普通目录：'+str(parent))
            if parent == base:
                break
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise ValueError('发布目标必须为普通文件：'+str(target))


def _validate_site_output(directory):
    """Reject linked destination directories before creating or scanning them."""
    directory = Path(directory)
    from codes.chapters.ch10.examples.generate_noise_mismatch import validate_asset_directory
    validate_asset_directory(directory / "noise_audio", check=False)
    from codes.chapters.ch11.examples.generate_selection_audio import validate_asset_directory as validate_scenario
    validate_scenario(directory / "scenario_audio", check=False)
    from codes.chapters.appendix_a.examples.generate_weighted_audio import validate_asset_directory as validate_weighted
    validate_weighted(directory / "weighted_audio", check=False)
    from codes.chapters.appendix_b.examples.generate_response_audio import validate_asset_directory as validate_response
    validate_response(directory / "response_audio", check=False)
    from codes.chapters.ch00.io_contracts import validate_asset_directory as validate_imaging
    validate_imaging(directory / "imaging_audio", IMAGING_AUDIO_WAVS | {"MANIFEST.json"}, check=False)
    validate_imaging(directory / "distributed_audio", DISTRIBUTED_AUDIO_WAVS | {"MANIFEST.json"}, check=False)
    validate_imaging(directory / "spectral_audio", SPECTRAL_AUDIO_WAVS | {"MANIFEST.json"}, check=False)
    validate_imaging(directory / 'sweep_audio', SWEEP_AUDIO_WAVS | {'MANIFEST.json'}, check=False)
    validate_imaging(directory / 'baseline_audio', BASELINE_AUDIO_WAVS | {'MANIFEST.json'}, check=False)
    validate_imaging(directory / 'reflection_audio', REFLECTION_AUDIO_WAVS | {'MANIFEST.json'}, check=False)
    validate_imaging(directory / 'phase_audio', PHASE_AUDIO_WAVS | {'MANIFEST.json'}, check=False)
    validate_imaging(directory / 'reference_audio', REFERENCE_AUDIO_WAVS | {'MANIFEST.json'}, check=False)
    validate_imaging(directory / 'delay_audio', DELAY_AUDIO_WAVS | {'MANIFEST.json'}, check=False)
    subdirectories = ('research', 'audio', 'real_audio', 'room_audio', 'moving_audio',
                      'tracking_audio', 'gss_audio', 'binaural_audio', 'spectral_audio', 'stft_audio', 'sweep_audio', 'baseline_audio', 'reflection_audio', 'phase_audio',
                      'geometry_audio', 'focus_audio', 'derivative_audio', 'apa_audio', 'reference_audio', 'delay_audio',
                      'mint_audio', 'mask_audio', 'noise_audio', 'scenario_audio', 'weighted_audio', 'response_audio', 'imaging_audio', 'distributed_audio')
    for folder in (directory, *(directory/name for name in subdirectories)):
        if folder.is_symlink() or (folder.exists() and not folder.is_dir()):
            raise ValueError('站点目标必须为普通目录：'+str(folder))
        if folder.is_dir() and any(p.is_symlink() for p in folder.iterdir()):
            raise ValueError('站点目标不能包含符号链接：'+str(folder))


def publish_files(replacements, removals=(), *, boundary=None):
    """替换失败时恢复整批旧文件；不承诺断电或进程强杀时的原子性。"""
    replacements = [(Path(source), Path(target)) for source, target in replacements]
    targets = [target for _, target in replacements] + [Path(path) for path in removals]
    if not targets:
        return
    if len(set(targets)) != len(targets):
        raise ValueError("发布目标重复")
    _validate_publish_targets(targets, boundary)
    backup_dir = Path(tempfile.mkdtemp(prefix=".publish-backup-", dir=targets[0].parent))
    snapshots = []
    preserve_backup = False
    try:
        for index, target in enumerate(targets):
            backup = backup_dir / str(index) if target.exists() else None
            if backup is not None:
                shutil.copy2(target, backup)
            snapshots.append((target, backup))
        try:
            for source, target in replacements:
                os.replace(source, target)
            for target in removals:
                Path(target).unlink()
        except BaseException:
            failures = []
            for target, backup in snapshots:
                try:
                    if backup is None:
                        target.unlink(missing_ok=True)
                    else:
                        shutil.copy2(backup, target)
                except OSError as error:
                    failures.append(f"{target}: {error}")
            if failures:
                preserve_backup = True
                raise RuntimeError(f"发布恢复失败，备份保留在 {backup_dir}：{failures}")
            raise
    finally:
        if not preserve_backup:
            shutil.rmtree(backup_dir)


def main():
    names = [f for f, _ in CHAPTERS]
    expected = set(source_outputs().values())
    _validate_site_output(OUT)
    with tempfile.TemporaryDirectory(prefix=".site-build-", dir=ROOT) as tmp:
        temp_out = Path(tmp)
        build_digest = source_digest()
        inline_layout_script = INLINE_LAYOUT_PATH.read_text(encoding="utf-8")
        home_md = (SRC / HOME_FNAME).read_text(encoding="utf-8")
        home_heads = parse_headings(home_md)
        home_html, home_n = render(home_md, SRC / HOME_FNAME)
        assert home_n == len(home_heads), f"首页锚点 {home_n} vs 标题 {len(home_heads)}"
        toc, _ = sub_list("index.html", home_heads)
        (temp_out / "index.html").write_text(PAGE.format(
            title="导读与导航", css=CSS, crumb="导读与导航",
            home_href="index.html",
            source_digest=build_digest,
            inline_layout_script=inline_layout_script,
            sidebar=sidebar_with_anchors(None, home_heads), toc=toc,
            body=home_html, pn=""), encoding="utf-8")
        for i, (fname, label) in enumerate(CHAPTERS):
            md = (SRC / fname).read_text(encoding="utf-8")
            heads = parse_headings(md)
            html_name = fname.replace(".md", ".html")
            body, n = render(md, SRC / fname)
            assert n == len(heads), f"{fname}: 锚点 {n} vs 标题 {len(heads)}"
            body, heads = promote_content_headings(body, heads)
            toc, _ = sub_list(html_name, heads, include_level1=True)
            prev = (f'<a href="{names[i-1].replace(".md", ".html")}">← 上一篇</a>'
                    if i > 0 else '<a href="index.html">← 导读</a>')
            nxt = (f'<a href="{names[i+1].replace(".md", ".html")}">下一篇 →</a>'
                   if i < len(names) - 1 else '<a href="index.html">回首页 →</a>')
            pn = f'<div class="pn"><span>{prev}</span><span>{nxt}</span></div>'
            (temp_out / html_name).write_text(PAGE.format(
                title=label, css=CSS, crumb=label,
                home_href="index.html",
                source_digest=build_digest,
                inline_layout_script=inline_layout_script,
                sidebar=sidebar_with_anchors(fname, heads), toc=toc,
                body=body, pn=pn), encoding="utf-8")
        (temp_out / "research").mkdir()
        for fname, label in RESEARCH:
            source = RESEARCH_ROOT / fname
            md = source.read_text(encoding="utf-8")
            heads = parse_headings(md)
            body, n = render(md, source)
            assert n == len(heads), f"{fname}: 锚点 {n} vs 标题 {len(heads)}"
            html_name = "index.html" if fname == "README.md" else fname.replace(".md", ".html")
            (temp_out / "research" / html_name).write_text(PAGE.format(
                title=label, css=CSS, crumb=label, home_href="../index.html",
                source_digest=build_digest, sidebar=research_sidebar(fname, heads),
                inline_layout_script=inline_layout_script,
                toc=sub_list(html_name, heads)[0], body=body, pn=""), encoding="utf-8")
        built = {path.relative_to(temp_out).as_posix() for path in temp_out.rglob("*.html")}
        if built != expected:
            raise SystemExit(f"站点产物集合异常：期望 {sorted(expected)}，实际 {sorted(built)}")
        OUT.mkdir(exist_ok=True)
        (OUT / "research").mkdir(exist_ok=True)
        stale = [path for path in OUT.glob("*.html") if path.name not in expected]
        stale += [path for path in (OUT / "research").glob("*.html")
                  if "research/" + path.name not in expected]
        manifest = json.loads(main_audio_manifest_path(CODE_CHAPTERS).read_text())
        audio_names = [record["file"] for record in manifest["files"]]
        if len(audio_names) != len(set(audio_names)) or any(not re.fullmatch(r"[a-z0-9_]+\.wav", name) for name in audio_names):
            raise ValueError("音频清单含重复或不安全路径")
        (temp_out / "audio").mkdir()
        (OUT / "audio").mkdir(exist_ok=True)
        for record in manifest["files"]:
            chapter = MAIN_AUDIO_GROUP_CHAPTER[record["group"]]
            if record.get("chapter") != chapter:
                raise ValueError(f"音频章节归属不符：{record['file']}")
            source = main_audio_path(CODE_CHAPTERS, record["group"], record["file"])
            if hashlib.sha256(source.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"音频校验失败：{source.name}")
            shutil.copy2(source, temp_out / "audio" / source.name)
        stale += [path for path in (OUT / "audio").glob("*.wav") if path.name not in audio_names]
        real_names = stage_real_audio(REAL_AUDIO_ROOT, temp_out / "real_audio")
        (OUT / "real_audio").mkdir(exist_ok=True)
        stale += [path for path in (OUT / "real_audio").iterdir()
                  if path.is_file() and path.name not in real_names]
        room_names = stage_room_audio(ROOM_AUDIO_ROOT, temp_out / "room_audio")
        (OUT / "room_audio").mkdir(exist_ok=True)
        stale += [path for path in (OUT / "room_audio").iterdir()
                  if path.is_file() and path.name not in room_names]
        moving_names = stage_moving_audio(MOVING_AUDIO_ROOT, temp_out / "moving_audio")
        (OUT / "moving_audio").mkdir(exist_ok=True)
        stale += [path for path in (OUT / "moving_audio").iterdir()
                  if path.is_file() and path.name not in moving_names]
        tracking_names = stage_tracking_audio(TRACKING_AUDIO_ROOT, temp_out / "tracking_audio")
        (OUT / "tracking_audio").mkdir(exist_ok=True)
        stale += [path for path in (OUT / "tracking_audio").iterdir()
                  if path.is_file() and path.name not in tracking_names]
        gss_names = stage_gss_audio(GSS_AUDIO_ROOT, temp_out / "gss_audio")
        (OUT / "gss_audio").mkdir(exist_ok=True)
        stale += [path for path in (OUT / "gss_audio").iterdir()
                  if path.is_file() and path.name not in gss_names]
        binaural_names = stage_binaural_audio(BINAURAL_AUDIO_ROOT, temp_out / "binaural_audio")
        (OUT / "binaural_audio").mkdir(exist_ok=True)
        stale += [path for path in (OUT / "binaural_audio").iterdir()
                  if path.is_file() and path.name not in binaural_names]
        stft_names = stage_stft_audio(STFT_AUDIO_ROOT, temp_out / "stft_audio")
        (OUT / "stft_audio").mkdir(exist_ok=True)
        stale += [path for path in (OUT / "stft_audio").iterdir()
                  if path.is_file() and path.name not in stft_names]
        geometry_names = stage_geometry_audio(GEOMETRY_AUDIO_ROOT, temp_out / "geometry_audio")
        (OUT / "geometry_audio").mkdir(exist_ok=True)
        stale += [path for path in (OUT / "geometry_audio").iterdir()
                  if path.is_file() and path.name not in geometry_names]
        focus_names = stage_focus_audio(FOCUS_AUDIO_ROOT, temp_out / "focus_audio")
        (OUT / "focus_audio").mkdir(exist_ok=True)
        stale += [path for path in (OUT / "focus_audio").iterdir()
                  if path.is_file() and path.name not in focus_names]
        derivative_names = stage_derivative_audio(DERIVATIVE_AUDIO_ROOT, temp_out / "derivative_audio")
        (OUT / "derivative_audio").mkdir(exist_ok=True)
        stale += [path for path in (OUT / "derivative_audio").iterdir()
                  if path.is_file() and path.name not in derivative_names]
        apa_names = stage_apa_audio(APA_AUDIO_ROOT, temp_out / "apa_audio")
        (OUT / "apa_audio").mkdir(exist_ok=True)
        stale += [path for path in (OUT / "apa_audio").iterdir()
                  if path.is_file() and path.name not in apa_names]
        reference_names = stage_reference_audio(REFERENCE_AUDIO_ROOT, temp_out / "reference_audio")
        (OUT / "reference_audio").mkdir(exist_ok=True)
        delay_names = stage_delay_audio(DELAY_AUDIO_ROOT, temp_out / "delay_audio")
        (OUT / "delay_audio").mkdir(exist_ok=True)
        mint_names = stage_mint_audio(MINT_AUDIO_ROOT, temp_out / "mint_audio")
        (OUT / "mint_audio").mkdir(exist_ok=True)
        stale += [path for path in (OUT / "mint_audio").iterdir()
                  if path.is_file() and path.name not in mint_names]
        mask_names = stage_mask_audio(MASK_AUDIO_ROOT, temp_out / "mask_audio")
        (OUT / "mask_audio").mkdir(exist_ok=True)
        stale += [path for path in (OUT / "mask_audio").iterdir()
                  if path.is_file() and path.name not in mask_names]
        noise_names = stage_noise_audio(NOISE_AUDIO_ROOT, temp_out / "noise_audio")
        (OUT / "noise_audio").mkdir(exist_ok=True)
        scenario_names = stage_scenario_audio(SCENARIO_AUDIO_ROOT, temp_out / "scenario_audio")
        (OUT / "scenario_audio").mkdir(exist_ok=True)
        weighted_names = stage_weighted_audio(WEIGHTED_AUDIO_ROOT, temp_out / "weighted_audio")
        (OUT / "weighted_audio").mkdir(exist_ok=True)
        response_names = stage_response_audio(RESPONSE_AUDIO_ROOT, temp_out / "response_audio")
        (OUT / "response_audio").mkdir(exist_ok=True)
        imaging_names = stage_imaging_audio(IMAGING_AUDIO_ROOT, temp_out / "imaging_audio")
        (OUT / "imaging_audio").mkdir(exist_ok=True)
        distributed_names = stage_distributed_audio(DISTRIBUTED_AUDIO_ROOT, temp_out / "distributed_audio")
        (OUT / "distributed_audio").mkdir(exist_ok=True)
        spectral_names = stage_spectral_audio(SPECTRAL_AUDIO_ROOT, temp_out / "spectral_audio")
        (OUT / "spectral_audio").mkdir(exist_ok=True)
        sweep_names = stage_sweep_audio(SWEEP_AUDIO_ROOT, temp_out / 'sweep_audio')
        (OUT / 'sweep_audio').mkdir(exist_ok=True)
        baseline_names = stage_baseline_audio(BASELINE_AUDIO_ROOT, temp_out / 'baseline_audio')
        (OUT / 'baseline_audio').mkdir(exist_ok=True)
        reflection_names = stage_reflection_audio(REFLECTION_AUDIO_ROOT, temp_out / 'reflection_audio')
        (OUT / 'reflection_audio').mkdir(exist_ok=True)
        phase_names = stage_phase_audio(PHASE_AUDIO_ROOT, temp_out / 'phase_audio')
        (OUT / 'phase_audio').mkdir(exist_ok=True)
        publish_files([(temp_out / name, OUT / name) for name in sorted(expected)] +
                      [(temp_out / "audio" / name, OUT / "audio" / name) for name in audio_names] +
                      [(temp_out / "real_audio" / name, OUT / "real_audio" / name)
                       for name in sorted(real_names)] +
                      [(temp_out / "room_audio" / name, OUT / "room_audio" / name)
                       for name in sorted(room_names)] +
                      [(temp_out / "moving_audio" / name, OUT / "moving_audio" / name)
                       for name in sorted(moving_names)] +
                      [(temp_out / "tracking_audio" / name, OUT / "tracking_audio" / name)
                       for name in sorted(tracking_names)] +
                      [(temp_out / "gss_audio" / name, OUT / "gss_audio" / name)
                       for name in sorted(gss_names)] +
                      [(temp_out / "binaural_audio" / name, OUT / "binaural_audio" / name)
                       for name in sorted(binaural_names)] +
                      [(temp_out / "stft_audio" / name, OUT / "stft_audio" / name)
                       for name in sorted(stft_names)] +
                      [(temp_out / "geometry_audio" / name, OUT / "geometry_audio" / name)
                       for name in sorted(geometry_names)] +
                      [(temp_out / "focus_audio" / name, OUT / "focus_audio" / name)
                       for name in sorted(focus_names)] +
                      [(temp_out / "derivative_audio" / name, OUT / "derivative_audio" / name)
                       for name in sorted(derivative_names)] +
                      [(temp_out / "apa_audio" / name, OUT / "apa_audio" / name)
                       for name in sorted(apa_names)] +
                      [(temp_out / "reference_audio" / name, OUT / "reference_audio" / name)
                       for name in sorted(reference_names)] +
                      [(temp_out / "delay_audio" / name, OUT / "delay_audio" / name)
                       for name in sorted(delay_names)] +
                      [(temp_out / "mint_audio" / name, OUT / "mint_audio" / name)
                       for name in sorted(mint_names)] +
                      [(temp_out / "mask_audio" / name, OUT / "mask_audio" / name)
                       for name in sorted(mask_names)] +
                      [(temp_out / "noise_audio" / name, OUT / "noise_audio" / name)
                       for name in sorted(noise_names)] +
                      [(temp_out / "scenario_audio" / name, OUT / "scenario_audio" / name)
                       for name in sorted(scenario_names)] +
                      [(temp_out / "weighted_audio" / name, OUT / "weighted_audio" / name)
                       for name in sorted(weighted_names)] +
                      [(temp_out / "response_audio" / name, OUT / "response_audio" / name)
                       for name in sorted(response_names)] +
                      [(temp_out / "imaging_audio" / name, OUT / "imaging_audio" / name)
                       for name in sorted(imaging_names)] +
                      [(temp_out / "distributed_audio" / name, OUT / "distributed_audio" / name)
                       for name in sorted(distributed_names)] +
                      [(temp_out / "spectral_audio" / name, OUT / "spectral_audio" / name)
                       for name in sorted(spectral_names)] +
                      [(temp_out / 'sweep_audio' / name, OUT / 'sweep_audio' / name)
                       for name in sorted(sweep_names)] +
                      [(temp_out / 'baseline_audio' / name, OUT / 'baseline_audio' / name)
                       for name in sorted(baseline_names)] +
                      [(temp_out / 'reflection_audio' / name, OUT / 'reflection_audio' / name)
                       for name in sorted(reflection_names)] +
                      [(temp_out / 'phase_audio' / name, OUT / 'phase_audio' / name)
                       for name in sorted(phase_names)], stale, boundary=OUT)
    print("DONE", len(expected), "pages")


if __name__ == "__main__":
    main()
