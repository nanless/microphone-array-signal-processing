#!/usr/bin/env python3
"""把 chapters/ 14 篇及 codes/chapters/ch00/research/ 6 篇 Markdown 建成静态站。

用法（报告根目录）：
    .venv/bin/python scripts/build_site.py

产物：site/index.html（首页）+ site/01..13_*.html（13 篇正文），
另有 site/research/index.html 和 5 篇独立研究页、主清单 109 个与独立实验 53 个合成 WAV，
以及 4 个真实录音/派生 WAV；源码按章保存，发布 URL 保持原样。
左侧边栏 = 首页 + 13 篇 + 每篇的二级及以下小节锚点，顶部面包屑，
文末上一篇/下一篇（首页不输出该盒）。图片直接引用 ../figures/（不复制）。
数学公式用 MathJax CDN 渲染（离线时显示源码，页面顶部有提示）。
"""
import os
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
                                                restore_code, validate_url_schemes)
    from scripts.code_layout import MAIN_AUDIO_GROUP_CHAPTER, main_audio_manifest_path, main_audio_path
except ModuleNotFoundError:  # direct ``python scripts/build_site.py``
    from build_markdown_helpers import (ALLOWED_LINK_SCHEMES, protect_code,
                                        restore_code, validate_url_schemes)
    from code_layout import MAIN_AUDIO_GROUP_CHAPTER, main_audio_manifest_path, main_audio_path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "chapters"
OUT = ROOT / "site"
CODE_CHAPTERS = ROOT / "codes" / "chapters"
RESEARCH_ROOT = CODE_CHAPTERS / "ch00" / "research"
REAL_AUDIO_ROOT = CODE_CHAPTERS / "ch02" / "real_audio"
ROOM_AUDIO_ROOT = CODE_CHAPTERS / "appendix_b" / "room_audio"
MOVING_AUDIO_ROOT = CODE_CHAPTERS / "ch09" / "moving_audio"
TRACKING_AUDIO_ROOT = CODE_CHAPTERS / "ch09" / "tracking_audio"
GSS_AUDIO_ROOT = CODE_CHAPTERS / "ch08" / "gss_audio"
BINAURAL_AUDIO_ROOT = CODE_CHAPTERS / "ch01" / "binaural_audio"
STFT_AUDIO_ROOT = CODE_CHAPTERS / "ch02" / "stft_audio"
DERIVATIVE_AUDIO_ROOT = CODE_CHAPTERS / "ch05" / "derivative_audio"
APA_AUDIO_ROOT = CODE_CHAPTERS / "ch06" / "apa_audio"
MINT_AUDIO_ROOT = CODE_CHAPTERS / "ch07" / "mint_audio"
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


def main_audio_sources():
    """Map only declared chapter-owned WAVs to their stable published names."""
    manifest = json.loads(main_audio_manifest_path(CODE_CHAPTERS).read_text(encoding="utf-8"))
    result = {}
    for record in manifest["files"]:
        chapter = MAIN_AUDIO_GROUP_CHAPTER[record["group"]]
        if record.get("chapter") != chapter:
            raise ValueError(f"音频章节归属不符：{record['file']}")
        result[main_audio_path(CODE_CHAPTERS, record["group"], record["file"]).resolve()] = record["file"]
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

CSS = """
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;line-height:1.75;color:#1a1a2e;background:#fafbfc}
.topbar{position:sticky;top:0;z-index:10;background:#1a1a2e;color:#fff;padding:10px 20px;font-size:15px}
.topbar a{color:#9ec5f0;text-decoration:none}.topbar a:hover{text-decoration:underline}
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
mjx-assistive-mml{width:1px!important;height:1px!important}
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
@media(max-width:900px){.side{display:none}.main{padding:20px}.toc-mobile{display:block}.topbar{font-size:14px}mjx-container[jax="CHTML"]:not([display="true"]){display:inline-block;vertical-align:middle}.aec-readable-table th,.aec-readable-table td,.wpe-readable-table th,.wpe-readable-table td{min-width:8em}}
@media print{.topbar,.side,.pn,.topbtn,.toc-mobile{display:none}.main{padding:0}.table-scroll{overflow:visible}table{display:table}a{color:#000;text-decoration:none}pre{white-space:pre-wrap;background:#fff;color:#000;border:1px solid #ccc}}
"""

PAGE = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · 麦克风阵列信号处理教程</title>
<meta name="source-digest" content="{source_digest}"><style>{css}</style>
<script>
window.MathJax = {{tex: {{inlineMath: [['$', '$'], ['\\\\(', '\\\\)']], displayMath: [['$$', '$$']]}}}};
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
    manifest = json.loads((source / "MANIFEST.json").read_text(encoding="utf-8"))
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
    manifest = json.loads((source / "MANIFEST.json").read_text(encoding="utf-8"))
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
    manifest = json.loads((source / "MANIFEST.json").read_text(encoding="utf-8"))
    records = manifest["files"]
    expected = set(MOVING_AUDIO_WAVS) | {"MANIFEST.json"}
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
    destination.mkdir()
    for name in sorted(expected):
        shutil.copy2(source / name, destination / name)
    return expected



def stage_binaural_audio(source, destination):
    """Publish five independent time/level cue fixtures with complete tails."""
    import wave
    expected = BINAURAL_AUDIO_WAVS | {"MANIFEST.json"}
    if (source.is_symlink() or {p.name for p in source.iterdir()} != expected
            or any(p.is_symlink() or not p.is_file() for p in source.iterdir())):
        raise ValueError("双耳线索目录必须恰含五份普通WAV及独立清单")
    manifest = json.loads((source / "MANIFEST.json").read_text(encoding="utf-8"))
    if (set(manifest["files"]) != BINAURAL_AUDIO_WAVS
            or manifest["sample_rate_hz"] != 16000
            or manifest["common_export_gain"] != 1):
        raise ValueError("双耳线索清单集合、采样率或共同增益不符")
    required_sources = {"codes/chapters/ch01/core/binaural_cues.py",
                        "codes/chapters/ch01/examples/generate_binaural_cues.py",
                        "codes/chapters/ch00/core/audio_samples.py"}
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


def stage_stft_audio(source, destination):
    """Publish three independent finite-window convolution fixtures with complete tails."""
    import wave
    expected = STFT_AUDIO_WAVS | {"MANIFEST.json"}
    if (source.is_symlink() or {p.name for p in source.iterdir()} != expected
            or any(p.is_symlink() or not p.is_file() for p in source.iterdir())):
        raise ValueError("STFT卷积目录必须恰含三份普通WAV及独立清单")
    manifest = json.loads((source / "MANIFEST.json").read_text(encoding="utf-8"))
    if (set(manifest["files"]) != STFT_AUDIO_WAVS
            or manifest["sample_rate_hz"] != 16000
            or manifest["common_export_gain"] != 1):
        raise ValueError("STFT卷积清单集合、采样率或共同增益不符")
    required_sources = {"codes/chapters/ch02/core/stft_convolution.py",
                        "codes/chapters/ch02/examples/generate_stft_convolution.py",
                        "codes/chapters/ch02/core/spectral.py",
                        "codes/chapters/ch02/core/conventions.py",
                        "codes/chapters/ch00/core/audio_samples.py"}
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
    import wave
    expected = set(GEOMETRY_AUDIO_WAVS) | {"MANIFEST.json"}
    if (source.is_symlink() or {p.name for p in source.iterdir()} != expected
            or any(p.is_symlink() or not p.is_file() for p in source.iterdir())):
        raise ValueError("多频几何目录必须恰含三份普通WAV及独立清单")
    manifest = json.loads((source / "MANIFEST.json").read_text(encoding="utf-8"))
    if (set(manifest["files"]) != set(GEOMETRY_AUDIO_WAVS)
            or manifest["sample_rate_hz"] != 32000
            or manifest["common_export_gain"] != 1):
        raise ValueError("多频几何清单集合、采样率或共同增益不符")
    required_sources = {"codes/chapters/ch03/core/geometry_audio.py",
                        "codes/chapters/ch03/examples/generate_geometry_audio.py",
                        "codes/chapters/ch03/core/geometry.py",
                        "codes/chapters/ch02/core/conventions.py",
                        "codes/chapters/ch00/core/audio_samples.py"}
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
    import wave
    expected = set(FOCUS_AUDIO_WAVS) | {"MANIFEST.json"}
    if (source.is_symlink() or {p.name for p in source.iterdir()} != expected
            or any(p.is_symlink() or not p.is_file() for p in source.iterdir())):
        raise ValueError("已知酉聚焦目录必须恰含四份普通WAV及独立清单")
    manifest = json.loads((source / "MANIFEST.json").read_text(encoding="utf-8"))
    if (set(manifest["files"]) != set(FOCUS_AUDIO_WAVS)
            or manifest["sample_rate_hz"] != 16000
            or manifest["common_export_gain"] != 1):
        raise ValueError("已知酉聚焦清单集合、采样率或共同增益不符")
    required_sources = {"codes/chapters/ch04/core/focus_audio.py",
                        "codes/chapters/ch04/examples/generate_focus_audio.py",
                        "codes/chapters/ch03/core/geometry.py",
                        "codes/chapters/ch02/core/conventions.py",
                        "codes/chapters/ch00/core/audio_samples.py"}
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
    import wave
    expected = set(DERIVATIVE_AUDIO_WAVS) | {"MANIFEST.json"}
    if (source.is_symlink() or {p.name for p in source.iterdir()} != expected
            or any(p.is_symlink() or not p.is_file() for p in source.iterdir())):
        raise ValueError("导数约束目录必须恰含四份普通WAV及独立清单")
    manifest = json.loads((source / "MANIFEST.json").read_text(encoding="utf-8"))
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
                        "codes/chapters/ch00/core/audio_samples.py"}
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
    import sys
    if str(ROOT.resolve()) not in sys.path:
        sys.path.insert(0, str(ROOT.resolve()))
    from codes.chapters.ch06.examples.generate_apa_audio import check_assets
    check_assets(source, replay=False)
    expected = APA_AUDIO_WAVS | {"MANIFEST.json"}
    destination.mkdir()
    for name in sorted(expected):
        shutil.copy2(source / name, destination / name)
    return expected


def stage_mint_audio(source, destination):
    """Validate independent known-path inverse PCM before publication."""
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


def stage_tracking_audio(source, destination):
    """Validate the independent PCM-to-observation experiment before publishing."""
    import wave
    expected = set(TRACKING_AUDIO_WAVS) | {"MANIFEST.json"}
    if (source.is_symlink() or {p.name for p in source.iterdir()} != expected
            or any(p.is_symlink() or not p.is_file() for p in source.iterdir())):
        raise ValueError("追踪音频目录必须恰有两份普通WAV及清单")
    manifest = json.loads((source/"MANIFEST.json").read_text())
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
    destination.mkdir()
    for name in sorted(expected):
        shutil.copy2(source/name,destination/name)
    return expected

def stage_gss_audio(source, destination):
    """核对受控 GSS 教学链的五路 PCM 与可复算中间状态。"""
    import wave
    import zipfile
    manifest = json.loads((source / "MANIFEST.json").read_text(encoding="utf-8"))
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
                    "codes/chapters/ch07/core/dereverberation.py"}
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
                       TRACKING_AUDIO_ROOT, GSS_AUDIO_ROOT, BINAURAL_AUDIO_ROOT, STFT_AUDIO_ROOT, GEOMETRY_AUDIO_ROOT, FOCUS_AUDIO_ROOT, DERIVATIVE_AUDIO_ROOT, APA_AUDIO_ROOT, MINT_AUDIO_ROOT):
        paths += sorted(asset_root.glob("*"))
    paths += sorted((ROOT / "figures").glob("fig*.png"))
    paths += [Path(__file__), ROOT / "scripts" / "build_markdown_helpers.py",
              ROOT / "scripts" / "heading_aliases.py",
              ROOT / "scripts" / "legacy_sequential_anchors.json",
              ROOT / "scripts" / "code_layout.py",
              ROOT / "scripts" / "make_figures.py",
              ROOT / "scripts" / "make_aec_figures.py", ROOT / "requirements.txt"]
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def parse_headings(md):
    """掃 md 源码标题，跳过围栏代码块，避免幽灵条目。"""
    heads = []
    fence_char = None
    fence_len = 0
    for line in md.splitlines():
        s = line.strip()
        marker = re.match(r"^(`{3,}|~{3,})", s)
        if marker and fence_char is None:
            fence_char = marker.group(1)[0]
            fence_len = len(marker.group(1))
            continue
        if (marker and fence_char is not None
                and marker.group(1)[0] == fence_char
                and len(marker.group(1)) >= fence_len):
            fence_char = None
            fence_len = 0
            continue
        if fence_char is not None:
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            heads.append((len(m.group(1)), clean_label(m.group(2))))
    return heads


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
    """Keep the supplementary room chart inside the published site bundle."""
    def replace(match):
        parsed, target = local_link_target(unescape(match.group(3)), source_path)
        if target == (ROOM_AUDIO_ROOT / "ROOM_RESULTS.png").resolve():
            relative = os.path.relpath("room_audio/ROOM_RESULTS.png", Path(current).parent).replace(os.sep, "/")
            value = urlunsplit(("", "", relative, parsed.query, parsed.fragment))
            return match.group(1) + match.group(2) + escape(value, quote=True) + match.group(2)
        return match.group(0)
    return re.sub(r'(<img\b[^>]*?\bsrc=)(["\x27])(.*?)\2', replace, html, flags=re.S)


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
        if target.parent == STFT_AUDIO_ROOT.resolve() and target.name in (STFT_AUDIO_WAVS | {"MANIFEST.json"}):
            relative = os.path.relpath("stft_audio/" + target.name, Path(current).parent).replace(os.sep, "/")
            return urlunsplit(("", "", relative, parsed.query, parsed.fragment))
        if target.parent == GEOMETRY_AUDIO_ROOT.resolve() and target.name in (set(GEOMETRY_AUDIO_WAVS) | {"MANIFEST.json"}):
            relative = os.path.relpath("geometry_audio/" + target.name, Path(current).parent).replace(os.sep, "/")
            return urlunsplit(("", "", relative, parsed.query, parsed.fragment))
        if target.parent == DERIVATIVE_AUDIO_ROOT.resolve() and target.name in (set(DERIVATIVE_AUDIO_WAVS) | {"MANIFEST.json"}):
            relative = os.path.relpath("derivative_audio/" + target.name, Path(current).parent).replace(os.sep, "/")
            return urlunsplit(("", "", relative, parsed.query, parsed.fragment))
        if target.parent == APA_AUDIO_ROOT.resolve() and target.name in (APA_AUDIO_WAVS | {"MANIFEST.json"}):
            relative = os.path.relpath("apa_audio/" + target.name, Path(current).parent).replace(os.sep, "/")
            return urlunsplit(("", "", relative, parsed.query, parsed.fragment))
        if target.parent == MINT_AUDIO_ROOT.resolve() and target.name in (set(MINT_AUDIO_WAVS) | {"MANIFEST.json"}):
            relative = os.path.relpath("mint_audio/" + target.name, Path(current).parent).replace(os.sep, "/")
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
        if parsed.scheme or parsed.query or parsed.fragment or not re.fullmatch(r"(?:\.\./)?(?:audio|real_audio|moving_audio|tracking_audio|gss_audio|binaural_audio|stft_audio|geometry_audio|focus_audio|derivative_audio|apa_audio|mint_audio)/[a-z0-9_]+\.wav", parsed.path):
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
    # 数学段暂存：防 markdown 吃下划线、防浏览器吞 <，转完再贴回
    repo = []

    def stash(m):
        repo.append(m.group(0).replace("<", r"\lt "))
        return f"@@MATH{len(repo) - 1}@@"

    md_text, code_repo = protect_code(md_text)
    md_text = re.sub(r"\$\$.*?\$\$", stash, md_text, flags=re.S)
    md_text = re.sub(r"\$[^$]+?\$", stash, md_text, flags=re.S)
    md_text = restore_code(md_text, code_repo)
    html = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
    html = re.sub(r"@@MATH(\d+)@@", lambda m: repo[int(m.group(1))], html)
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
    # 保留 table 原生语义；横向滚动由可聚焦的外层区域承担，键盘用户也能操作宽表。
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

    html = re.sub(r'(^|[\s>(])((?:0\d|1\d)_[^<\s)"]+\.md)', bare_link, html)
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


def publish_files(replacements, removals=()):
    """替换失败时恢复整批旧文件；不承诺断电或进程强杀时的原子性。"""
    replacements = [(Path(source), Path(target)) for source, target in replacements]
    targets = [target for _, target in replacements] + [Path(path) for path in removals]
    if not targets:
        return
    if len(set(targets)) != len(targets):
        raise ValueError("发布目标重复")
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
    with tempfile.TemporaryDirectory(prefix=".site-build-", dir=ROOT) as tmp:
        temp_out = Path(tmp)
        build_digest = source_digest()
        home_md = (SRC / HOME_FNAME).read_text(encoding="utf-8")
        home_heads = parse_headings(home_md)
        home_html, home_n = render(home_md, SRC / HOME_FNAME)
        assert home_n == len(home_heads), f"首页锚点 {home_n} vs 标题 {len(home_heads)}"
        toc, _ = sub_list("index.html", home_heads)
        (temp_out / "index.html").write_text(PAGE.format(
            title="导读与导航", css=CSS, crumb="导读与导航",
            home_href="index.html",
            source_digest=build_digest,
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
        mint_names = stage_mint_audio(MINT_AUDIO_ROOT, temp_out / "mint_audio")
        (OUT / "mint_audio").mkdir(exist_ok=True)
        stale += [path for path in (OUT / "mint_audio").iterdir()
                  if path.is_file() and path.name not in mint_names]
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
                      [(temp_out / "mint_audio" / name, OUT / "mint_audio" / name)
                       for name in sorted(mint_names)], stale)
    print("DONE", len(expected), "pages")


if __name__ == "__main__":
    main()
