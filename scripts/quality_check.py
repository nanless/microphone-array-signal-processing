#!/usr/bin/env python3
"""发布前质量门禁：占位符、图片、站内链接、合订 HTML 与 PDF。"""

from __future__ import annotations

import re
import sys
import hashlib
import json
import wave
import math
import unicodedata
from collections import Counter
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse

try:
    from scripts.code_layout import MAIN_AUDIO_GROUP_CHAPTER, main_audio_manifest_path, main_audio_path
    from scripts.build_site import _check_tracking_members, _read_tracking_manifest
    from scripts.build_markdown_helpers import code_block_ranges, parsed_markdown_headings
except ModuleNotFoundError:  # direct ``python scripts/quality_check.py``
    from code_layout import MAIN_AUDIO_GROUP_CHAPTER, main_audio_manifest_path, main_audio_path
    from build_site import _check_tracking_members, _read_tracking_manifest
    from build_markdown_helpers import code_block_ranges, parsed_markdown_headings


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from codes.chapters.ch00.io_contracts import (
    strict_json_loads, validate_asset_directory, validate_parent_chain,
)

CHAPTERS = ROOT / "chapters"
SITE = ROOT / "site"
DIST = ROOT / "dist"
CODE_CHAPTERS = ROOT / "codes" / "chapters"
RESEARCH_ROOT = CODE_CHAPTERS / "ch00" / "research"
REAL_AUDIO_ROOT = CODE_CHAPTERS / "ch02" / "real_audio"
ROOM_AUDIO_ROOT = CODE_CHAPTERS / "appendix_b" / "room_audio"
MOVING_AUDIO_ROOT = CODE_CHAPTERS / "ch09" / "moving_audio"
TRACKING_AUDIO_ROOT = CODE_CHAPTERS / "ch09" / "tracking_audio"
GSS_AUDIO_ROOT = CODE_CHAPTERS / "ch08" / "gss_audio"
BINAURAL_AUDIO_ROOT = CODE_CHAPTERS / "ch01" / "binaural_audio"
SPECTRAL_AUDIO_ROOT = CODE_CHAPTERS / "ch01" / "spectral_audio"
SPECTRAL_AUDIO_CHANNELS = {'flat_source.wav': 1, 'flat_stereo.wav': 2,
                           'tilted_source.wav': 1, 'tilted_stereo.wav': 2}
DERIVATIVE_AUDIO_ROOT = ROOT / "codes/chapters/ch05/derivative_audio"
APA_AUDIO_ROOT = ROOT / "codes/chapters/ch06/apa_audio"
MINT_AUDIO_ROOT = ROOT / "codes/chapters/ch07/mint_audio"
MASK_AUDIO_ROOT = ROOT / "codes/chapters/ch08/mask_audio"
SCENARIO_AUDIO_ROOT = CODE_CHAPTERS / "ch11" / "scenario_audio"
WEIGHTED_AUDIO_ROOT = CODE_CHAPTERS / "appendix_a" / "weighted_audio"
RESPONSE_AUDIO_ROOT = CODE_CHAPTERS / "appendix_b" / "response_audio"
IMAGING_AUDIO_ROOT = CODE_CHAPTERS / "ch14" / "imaging_audio"
DISTRIBUTED_AUDIO_ROOT = CODE_CHAPTERS / "ch15" / "distributed_audio"
DISTRIBUTED_AUDIO_CHANNELS = {
    name+'.wav': (4 if name.startswith('array_') else 1) for name in (
        'reference_node1', 'reference_node2', 'array_white', 'array_correlated',
        'local_node1', 'central_white', 'compressed_white', 'central_correlated',
        'compressed_correlated', 'stale_correlated', 'central_node2_correlated',
        'remote_scalar_white', 'transport_pcm16_white', 'clock_misaligned_white',
        'clock_linear_corrected_white', 'packet_zerofill_white', 'packet_local_fallback_white',
    )
}
DISTRIBUTED_AUDIO_SOURCES = (
    'codes/chapters/ch15/core/distributed.py', 'codes/chapters/ch15/core/distributed_audio.py',
    'codes/chapters/ch15/examples/generate_distributed_audio.py', 'codes/chapters/ch02/core/conventions.py',
    'codes/chapters/ch04/core/covariance.py', 'codes/chapters/ch10/sro_closed_loop_demo.py',
    'codes/chapters/ch10/core/engineering.py', 'codes/chapters/ch00/core/audio_samples.py',
    'codes/chapters/ch00/io_contracts.py',
)
RESPONSE_AUDIO_WAVS = {"response_" + name + ".wav" for name in
                       ("source", "reflection_a", "reflection_b", "full_a", "full_b")}
WEIGHTED_AUDIO_WAVS = {"weighted_" + name + ".wav" for name in
                       ("target", "array", "ols", "gls", "reversed")}
SCENARIO_AUDIO_WAVS = {f"selection_{scene}_{kind}.wav" for scene in ("single", "dual")
                       for kind in ("target", "mixture", "fir3", "fir9")}
NOISE_AUDIO_ROOT = CODE_CHAPTERS / "ch10" / "noise_audio"
NOISE_AUDIO_WAVS = {"noise_" + name + ".wav" for name in
                    ("reference", "component", "mixture", "fixed", "polluted", "known_variance")}
FOCUS_AUDIO_ROOT = ROOT / "codes/chapters/ch04/focus_audio"
GEOMETRY_AUDIO_ROOT = ROOT / "codes/chapters/ch03/geometry_audio"
STFT_AUDIO_ROOT = ROOT / "codes/chapters/ch02/stft_audio"

EDITING_MARKERS = re.compile(
    r"待核实|待补(?:实测|充)?|链接待补|成绩待补|清单#|"
    r"修订说明|编号不动|只调标题层级|新增块一律"
)

EXPECTED_SECTION_COUNTS = {
    "00_overview.md": 10,
    "01_problem-definition.md": 3,
    "02_basics-signal-model.md": 8,
    "03_array-geometry.md": 5,
    "04_doa-estimation.md": 11,
    "05_beamforming.md": 13,
    "06_aec.md": 17,
    "07_wpe-dereverberation.md": 10,
    "08_speech-separation.md": 8,
    "09_source-tracking.md": 6,
    "10_engineering-practice.md": 12,
    "11_selection-guide.md": 7,
    "12_appendix-symbols-math.md": 4,
    "13_appendix-guide.md": 7,
    "14_acoustic-imaging.md": 14,
    "15_distributed-enhancement.md": 16,
}
# 第 1～13 章的源 h4 进入合订目录和 PDF 第三级书签。此表是独立发布
# 基线，不从构建脚本或待检产物反推。
EXPECTED_SUBSECTION_COUNTS = {
    "01_problem-definition.md": 19,
    "02_basics-signal-model.md": 46,  # 两个独立主题：STFT合成与协方差启动权重
    "03_array-geometry.md": 31,
    "04_doa-estimation.md": 45,
    "05_beamforming.md": 41,
    "06_aec.md": 67,
    "07_wpe-dereverberation.md": 53,
    "08_speech-separation.md": 59,
    "09_source-tracking.md": 56,
    "10_engineering-practice.md": 54,
    "11_selection-guide.md": 37,
    "12_appendix-symbols-math.md": 36,
    "13_appendix-guide.md": 29,
    "14_acoustic-imaging.md": 61,
    # 39 separately navigable topics and 24 exercise headings, manually read.
    "15_distributed-enhancement.md": 63,
}
# 上表为独立发布基线，不从待检 HTML 或构建器反推。
EXPECTED_CHAPTERS = [
    ("00_overview.md", "导读与导航"),
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
EXPECTED_CHAPTER_COUNT = 16
EXPECTED_SECTION_COUNT = 151
EXPECTED_SUBSECTION_COUNT = 697
EXPECTED_OUTLINE_ITEM_COUNT = 864
EXPECTED_FIGURE_NUMBERS = set(range(1, 73))
EXPECTED_EXERCISE_COUNT = 333
EXPECTED_EXERCISE_COUNTS = {
    '01_problem-definition.md': 10, '02_basics-signal-model.md': 18,
    '03_array-geometry.md': 17, '04_doa-estimation.md': 23,
    '05_beamforming.md': 22, '06_aec.md': 39, '07_wpe-dereverberation.md': 21,
    '08_speech-separation.md': 29, '09_source-tracking.md': 23,
    '10_engineering-practice.md': 33, '11_selection-guide.md': 25,
    '12_appendix-symbols-math.md': 19, '13_appendix-guide.md': 14,
    '14_acoustic-imaging.md': 16, '15_distributed-enhancement.md': 24,
}
# 研究附站使用独立显式清单，不挤占 16 篇教程或教程 PDF 大纲基线。
# 此清单不能从构建器或待检 HTML 反推。
EXPECTED_RESEARCH_PAGES = (
    ("README.md", "index.html"),
    ("01_spatial_and_tracking.md", "01_spatial_and_tracking.html"),
    ("02_aec_wpe_separation.md", "02_aec_wpe_separation.html"),
    ("03_industrial_deployment.md", "03_industrial_deployment.html"),
    ("04_source_reproduction.md", "04_source_reproduction.html"),
    ("05_exercises_and_audio.md", "05_exercises_and_audio.html"),
)
EXPECTED_RESEARCH_PAGE_COUNT = 6
ALLOWED_LINK_SCHEMES = {"http", "https", "mailto"}
COLLOQUIAL_REVIEW = re.compile(
    r"乱飞|跳格子|翻车|掉链子|猪队友|吃进去|喂给|神经网络接管|记死|照抄"
)
NUMERIC_ASCII_RANGE = re.compile(
    r"(?<![A-Za-z0-9_.])\d+(?:\.\d+)?\s*~\s*\d+(?:\.\d+)?"
    r"(?![A-Za-z0-9_.])"
)


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids: list[str] = []
        self.links: list[str] = []
        self.nav_links: list[str] = []
        self.images: list[tuple[str, str | None]] = []
        self.heading_levels: list[int] = []
        self.h1_count = 0
        self.tags: Counter[str] = Counter()
        self.th_without_scope = 0
        self.source_digest = ""
        self.html_lang = ""
        self.nav_depth = 0

    def handle_starttag(self, tag, attrs):
        data = dict(attrs)
        self.tags[tag] += 1
        if tag == "nav":
            self.nav_depth += 1
        if tag == "h1":
            self.h1_count += 1
        if re.fullmatch(r"h[1-6]", tag):
            self.heading_levels.append(int(tag[1]))
        if tag == "html":
            self.html_lang = data.get("lang", "")
        if tag == "th" and data.get("scope") not in {"col", "row"}:
            self.th_without_scope += 1
        if tag == "meta" and data.get("name") == "source-digest":
            self.source_digest = data.get("content", "")
        if "id" in data:
            self.ids.append(data["id"])
        if tag == "a" and "href" in data:
            self.links.append(data["href"])
            if self.nav_depth:
                self.nav_links.append(data["href"])
        if tag == "img" and "src" in data:
            self.images.append((data["src"], data.get("alt")))

    def handle_endtag(self, tag):
        if tag == "nav" and self.nav_depth:
            self.nav_depth -= 1


def fail(errors: list[str], message: str):
    errors.append(message)


def strip_fenced_code(text: str):
    """Mask native root fences and raw code, preserving source line positions."""
    for start, end in sorted(code_block_ranges(text), reverse=True):
        text = text[:start] + re.sub(r"[^\r\n]", " ", text[start:end]) + text[end:]
    return text


def strip_inline_code(text: str):
    """屏蔽行内代码但保留字符位置和换行，避免把示例语法当正文。"""
    pattern = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)", re.S)
    return pattern.sub(lambda match: " " * len(match.group(0)), text)


def clean_heading_text(text: str):
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[`*_~]", "", text)
    return re.sub(r"\s+", " ", unescape(text)).strip()


def outline_heading_text(text: str):
    """Render heading labels independently while preserving inline code identifiers."""
    marker = "\ue000"
    protected = re.sub(r"`([^`]+)`", lambda match: match.group(1).replace("_", marker), text)
    return clean_heading_text(protected).replace(marker, "_")


def paragraph_review_candidates(text: str, min_chars: int = 280,
                                min_sentences: int = 6):
    """返回需要人工复核的长正文段；长度是线索，不是发布失败条件。"""
    prose = strip_fenced_code(text)
    candidates = []
    offset = 0
    for block in re.split(r"\n\s*\n", prose):
        start = prose.find(block, offset)
        offset = start + len(block)
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        first = lines[0]
        if (re.match(r"^(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|>|\||```|~~~|\$\$|<)", first)
                or all(line.startswith("|") for line in lines)):
            continue
        visible = " ".join(lines)
        visible = strip_inline_code(visible)
        visible = re.sub(r"!\[([^]]*)\]\([^)]*\)", r"\1", visible)
        visible = re.sub(r"\[([^]]*)\]\([^)]*\)", r"\1", visible)
        han_count = len(re.findall(r"[\u3400-\u9fff]", visible))
        word_count = len(re.findall(r"\b[A-Za-z]+(?:[-'][A-Za-z]+)*\b", visible))
        sentence_count = len(re.findall(r"[。！？；]", visible))
        sentence_count += len(re.findall(r"[.!?](?=\s|$)", visible))
        length_value = han_count if han_count >= 20 else word_count
        length_unit = "汉字" if han_count >= 20 else "英文词"
        length_limit = min_chars if han_count >= 20 else 160
        if length_value >= length_limit or sentence_count >= min_sentences:
            candidates.append({
                "line": prose.count("\n", 0, start) + 1,
                "chars": length_value,
                "unit": length_unit,
                "sentences": sentence_count,
                "preview": re.sub(r"\s+", " ", visible)[:72],
            })
    return candidates


def semantic_heading_ids(text: str):
    """按公开锚点规则独立计算源 Markdown 的主标题标识。"""
    seen = Counter()
    result = []
    for index, (level, title) in enumerate(markdown_headings(text), 1):
        label = clean_heading_text(title)
        numbered = re.match(r"^(\d+(?:\.\d+)+)(?=\s|$)", label)
        if numbered:
            base = "sec-" + numbered.group(1).replace(".", "-")
        elif label:
            base = "sec-u-" + hashlib.sha1(label.encode("utf-8")).hexdigest()[:10]
        else:
            base = f"sec-{index}"
        seen[base] += 1
        primary = base if seen[base] == 1 else f"{base}-{seen[base]}"
        result.append((level, label, primary))
    return result


def markdown_headings(text: str):
    return parsed_markdown_headings(text)


def section_count_from_markdown(text: str, overview=False):
    headings = markdown_headings(text)
    if overview:
        return sum(level == 2 for level, _ in headings)
    return sum(level == 3 for level, _ in headings)


def structure_issues(documents: dict[str, str]):
    issues = []
    actual = set(documents)
    expected = set(EXPECTED_SECTION_COUNTS)
    if actual != expected:
        issues.append(f"章节文件集不符合基线：缺失 {sorted(expected - actual)}，多出 {sorted(actual - expected)}")
    total = 0
    for name, expected_count in EXPECTED_SECTION_COUNTS.items():
        if name not in documents:
            continue
        actual_count = section_count_from_markdown(
            documents[name], overview=name == "00_overview.md")
        total += actual_count
        if actual_count != expected_count:
            issues.append(f"节数不符合基线：{name}: {actual_count}，应为 {expected_count}")
    if len(documents) != EXPECTED_CHAPTER_COUNT:
        issues.append(f"源文档数应为 {EXPECTED_CHAPTER_COUNT}，实际 {len(documents)}")
    if total != EXPECTED_SECTION_COUNT:
        issues.append(f"全书节数应为 {EXPECTED_SECTION_COUNT}，实际 {total}")
    subsection_total = 0
    for name, expected_count in EXPECTED_SUBSECTION_COUNTS.items():
        if name not in documents:
            continue
        actual_count = sum(level == 4 for level, _title in markdown_headings(documents[name]))
        subsection_total += actual_count
        if actual_count != expected_count:
            issues.append(f"PDF 子节数不符合基线：{name}: {actual_count}，应为 {expected_count}")
    if subsection_total != EXPECTED_SUBSECTION_COUNT:
        issues.append(
            f"PDF 第三级子节数应为 {EXPECTED_SUBSECTION_COUNT}，实际 {subsection_total}")
    outline_total = len(documents) + total + subsection_total
    if outline_total != EXPECTED_OUTLINE_ITEM_COUNT:
        issues.append(
            f"PDF 大纲项总数应为 {EXPECTED_OUTLINE_ITEM_COUNT}，实际 {outline_total}")
    return issues


def exercise_ids_from_markdown(text):
    """Only actual exercise titles; references, tables and code do not count."""
    result = []
    for level, title in markdown_headings(text):
        if level == 4 and (re.match(r'^E\d{2}-\d{2}\b', title) or re.match(r'^题\s+\d+', title)):
            result.extend(re.findall(r'\bE\d{2}-\d{2}\b', title))
    # Appendix B's first two pre-existing exercises are bold block labels.
    result.extend(re.findall(r'^\s*\*\*(E\d{2}-\d{2})[：\s]', strip_fenced_code(text), re.M))
    return result


def exercise_definition_issues(documents):
    issues, all_ids = [], []
    for filename, count in EXPECTED_EXERCISE_COUNTS.items():
        found = exercise_ids_from_markdown(documents.get(filename, ''))
        chapter = filename[:2]
        expected = {f'E{chapter}-{i:02d}' for i in range(1, count+1)}
        if len(found) != count or set(found) != expected:
            issues.append(f'练习题干集不符合基线：{filename}，缺失 {sorted(expected-set(found))}，多出 {sorted(set(found)-expected)}，题干数 {len(found)}/{count}')
        all_ids.extend(found)
    if len(all_ids) != EXPECTED_EXERCISE_COUNT or len(set(all_ids)) != EXPECTED_EXERCISE_COUNT:
        issues.append(f'可执行练习题干应为 {EXPECTED_EXERCISE_COUNT} 个唯一ID，实际 {len(all_ids)} 个题干/{len(set(all_ids))} 个唯一ID')
    return issues


def formula_semantic_issues(documents: dict[str, str]):
    issues = []
    definitions: dict[str, str] = {}
    references = []
    for name, original in documents.items():
        text = strip_inline_code(strip_fenced_code(original))
        chapter_match = re.match(r"(\d{2})_", name)
        expected_chapter = int(chapter_match.group(1)) if chapter_match else None
        displays = list(re.finditer(r"\$\$(.*?)\$\$", text, flags=re.S))
        display_ranges = [(match.start(), match.end()) for match in displays]
        valid_tags = []
        for match in re.finditer(r"\\tag(?:\{\d+-\d+\}|[^\s$]*)", text):
            exact = re.fullmatch(r"\\tag\{(\d+)-(\d+)\}", match.group(0))
            if not exact:
                issues.append(f"公式标签语法错误：{name}: {match.group(0)!r}")
                continue
            if not any(start <= match.start() < end for start, end in display_ranges):
                issues.append(f"公式标签不在 $$...$$ 公式块内：{name}: {match.group(0)}")
                continue
            valid_tags.append(exact)
        for display in displays:
            count = len(re.findall(r"\\tag\{\d+-\d+\}", display.group(1)))
            if count > 1:
                issues.append(f"单个公式块含多个编号：{name}")
        for display in displays:
            suffix = text[display.end():]
            legacy = re.match(r"[ \t]*\r?\n[ \t]*[（(]\d+-\d+[)）]", suffix)
            if legacy:
                issues.append(f"旧式公式编号位于公式块外：{name}: {legacy.group(0).strip()!r}")
        chapter_numbers = []
        for match in valid_tags:
            tag = f"{int(match.group(1))}-{int(match.group(2))}"
            if tag in definitions:
                issues.append(f"公式编号重复：{tag}（{definitions[tag]} 与 {name}）")
            else:
                definitions[tag] = name
            if expected_chapter is not None:
                if int(match.group(1)) != expected_chapter:
                    issues.append(f"公式章号不匹配：{name} 中 \\tag{{{tag}}}")
                else:
                    chapter_numbers.append(int(match.group(2)))
        if chapter_numbers and sorted(chapter_numbers) != list(range(1, len(chapter_numbers) + 1)):
            issues.append(f"公式编号不连续：{name}: {sorted(chapter_numbers)}")
        references.extend(
            (f"{int(m.group(1))}-{int(m.group(2))}", name)
            for m in re.finditer(r"式\s*[\(（]\s*(\d+)-(\d+)\s*[\)）]", text)
        )
    for tag, name in references:
        if tag not in definitions:
            issues.append(f"公式引用无定义：{name} 中式({tag})")
    return issues


def section_reference_issues(documents: dict[str, str]):
    issues = []
    sections = set()
    sections_by_file = {}
    owners = {}
    for name, text in documents.items():
        local = set()
        chapter_match = re.match(r"(\d{2})_", name)
        expected_chapter = int(chapter_match.group(1)) if chapter_match else None
        for _level, title in markdown_headings(text):
            match = re.match(r"^(\d+(?:\.\d+)+)(?=\s|$)", title)
            if match:
                number = match.group(1)
                local.add(number)
                owners.setdefault(number, []).append(name)
                if expected_chapter is not None and int(number.split(".")[0]) != expected_chapter:
                    issues.append(f"小节章号不匹配：{name}: {number}")
        sections |= local
        sections_by_file[name] = local
    for number, names in owners.items():
        if len(names) > 1:
            issues.append(f"小节编号重复：{number}: {names}")
    for name, original in documents.items():
        text = strip_fenced_code(original)
        # Section numbers inside absolute web citations belong to that source,
        # not to this book. Keep local-link labels and surrounding prose checked.
        internal_text = re.sub(r'\[[^\]]+\]\(https?://[^\s)]+(?:\s+"[^"]*")?\)', '', text)
        for match in re.finditer(r"§\s*(\d+(?:\.\d+)+)", internal_text):
            if match.group(1) not in sections:
                issues.append(f"小节引用不存在：{name}: §{match.group(1)}")
        link_re = re.compile(r"\[([^\]]+)\]\((?:\./)?([^\s)#]+\.md)(#[^\s)]+)?\)")
        for match in link_re.finditer(text):
            label, target_name, fragment = match.groups()
            section_match = (re.search(r"§\s*(\d+(?:\.\d+)+)", label)
                             or re.search(r"第\s*(\d+(?:\.\d+)+)\s*节", label)
                             or re.fullmatch(r"\s*(\d+(?:\.\d+)+)\s*", label))
            if not section_match:
                continue
            number = section_match.group(1)
            expected_fragment = "#sec-" + number.replace(".", "-")
            if fragment != expected_fragment:
                issues.append(
                    f"具体小节链接未指向语义片段：{name}: [{label}] -> "
                    f"{target_name}{fragment or ''}，应为 {expected_fragment}")
            if target_name in sections_by_file and number not in sections_by_file[target_name]:
                issues.append(f"小节链接目标不包含 {number}：{name} -> {target_name}")
    return issues


def collaboration_baseline_issues(text: str):
    """Check the published-outline baseline in AGENTS, not historical logs."""
    pattern = (r"PDF 的\s*(\d+)\s*个章级、\s*(\d+)\s*个节级、\s*"
               r"(\d+)\s*个子节级书签，共\s*(\d+)\s*个大纲项")
    matches = re.findall(pattern, text)
    expected = (EXPECTED_CHAPTER_COUNT, EXPECTED_SECTION_COUNT,
                EXPECTED_SUBSECTION_COUNT, EXPECTED_OUTLINE_ITEM_COUNT)
    if len(matches) != 1:
        return ["AGENTS 的当前 PDF 书签基线必须有且仅有一条完整记录"]
    actual = tuple(map(int, matches[0]))
    return [] if actual == expected else [f"AGENTS 书签基线过期：{actual}，应为 {expected}"]



def source_control_character_issues(text):
    """Catch accidental Python escapes (e.g. BEL from a TeX approx command)."""
    return [f"源文含控制字符 U+{ord(char):04X}：第{line_no}行"
            for line_no, line in enumerate(text.split("\n"),1)
            for char in line if (ord(char)<32 and char not in "\t\r") or ord(char)==127]

def check_sources(errors: list[str], notices: list[str]):
    agents_path = ROOT / "AGENTS.md"
    if agents_path.exists():
        errors.extend(collaboration_baseline_issues(agents_path.read_text(encoding="utf-8")))
    else:
        fail(errors, "缺少 AGENTS.md 当前协作规范")
    paths = sorted(CHAPTERS.glob("*.md"))
    paths += [ROOT / "README.md", ROOT / "README_EN.md", ROOT / "scripts" / "README.md"]
    for path in paths:
        original = path.read_text(encoding="utf-8")
        errors.extend(f"{path.relative_to(ROOT)}: {issue}"
                      for issue in source_control_character_issues(original))
        prose = strip_fenced_code(original)
        for line_no, line in enumerate(original.splitlines(), 1):
            if EDITING_MARKERS.search(line):
                fail(errors, f"编辑占位：{path.relative_to(ROOT)}:{line_no}: {line.strip()[:100]}")
            if re.search(r"\\\(|\\\)|\\\[|\\\]", line):
                fail(
                    errors,
                    f"不兼容的公式定界符：{path.relative_to(ROOT)}:{line_no}: "
                    f"请使用 $...$ 或 $$...$$",
                )
        style_text = re.sub(r"`[^`]*`", "", prose)
        style_text = re.sub(r"https?://[^\s)]+", "", style_text)
        if "本报告" in style_text:
            fail(errors, f"编辑视角残留：{path.relative_to(ROOT)}: 本报告")
        if NUMERIC_ASCII_RANGE.search(style_text):
            fail(errors, f"中文数字范围使用 ASCII ~：{path.relative_to(ROOT)}")
        for match in COLLOQUIAL_REVIEW.finditer(prose):
            line_no = prose.count("\n", 0, match.start()) + 1
            notices.append(
                f"高风险口语需人工复核：{path.relative_to(ROOT)}:{line_no}: {match.group(0)}")
        paragraph_candidates = paragraph_review_candidates(original)
        if paragraph_candidates:
            worst = max(
                paragraph_candidates,
                key=lambda item: (item["chars"], item["sentences"]),
            )
            notices.append(
                f"段落结构需人工复核：{path.relative_to(ROOT)} 共 "
                f"{len(paragraph_candidates)} 处；最长候选在第 {worst['line']} 行，"
                f"约 {worst['chars']} {worst['unit']}、{worst['sentences']} 句。长度只用于定位。"
            )

    documents = {path.name: path.read_text(encoding="utf-8")
                 for path in sorted(CHAPTERS.glob("*.md"))}
    errors.extend(structure_issues(documents))
    errors.extend(exercise_definition_issues(documents))
    errors.extend(formula_semantic_issues(documents))
    errors.extend(section_reference_issues(documents))


def url_scheme_issues(urls):
    issues = []
    for value in urls:
        parsed = urlparse(value.strip())
        scheme = parsed.scheme.lower()
        if parsed.netloc and not scheme:
            issues.append(f"省略协议的外部链接：{value}")
        elif scheme and scheme not in ALLOWED_LINK_SCHEMES:
            issues.append(f"不安全或不支持的链接协议 {scheme}：{value}")
    return issues


def expected_site_content(name: str, source: str):
    ids = {primary for _level, _title, primary in semantic_heading_ids(source)}
    images = Counter(f"../figures/{figure_name}"
                     for _alt, figure_name, _number in extract_figure_references(source))
    if name == "13_appendix-guide.md":
        if not re.search(r"!\[[^\]]+\]\(\.\./codes/chapters/appendix_b/room_audio/ROOM_RESULTS\.png\)", source):
            raise ValueError("附录 B 缺少房间仿真补充图源引用")
        images["room_audio/ROOM_RESULTS.png"] += 1
    return ids, images


def expected_site_nav_fragments(name: str, source: str):
    """返回必须出现在当前页 nav 中的源 h2～h4 标题锚点；源 h1 可省略。"""
    return {
        primary for level, _title, primary in semantic_heading_ids(source)
        if 2 <= level <= 4
    }


def site_nav_fragment_issues(page_name: str, source_name: str, source: str,
                             nav_links: list[str]):
    """正文中的普通链接不参与导航完整性判定，只检查 nav 内当前页片段。"""
    present = set()
    for href in nav_links:
        parsed = urlparse(href)
        target_name = unquote(parsed.path) or page_name
        if not parsed.scheme and target_name == page_name and parsed.fragment:
            present.add(unquote(parsed.fragment))
    missing = sorted(expected_site_nav_fragments(source_name, source) - present)
    return [f"当前页导航缺少源标题片段：{fragment}" for fragment in missing]


def expected_outline(documents=None):
    """仅从显式篇名清单和 Markdown 标题解析 PDF 期望书签。"""
    if documents is None:
        documents = {path.name: path.read_text(encoding="utf-8")
                     for path in CHAPTERS.glob("*.md")}
    result = []
    for name, label in EXPECTED_CHAPTERS:
        headings = semantic_heading_ids(documents[name])
        section_level = 2 if name == "00_overview.md" else 3
        children = []
        current = None
        raw_headings = markdown_headings(documents[name])
        for (level, _title, _primary), (_raw_level, raw_title) in zip(headings, raw_headings):
            title = outline_heading_text(raw_title)
            if level == section_level:
                current = [title, []]
                children.append(current)
            elif (name in EXPECTED_SUBSECTION_COUNTS
                  and level == section_level + 1 and current is not None):
                current[1].append(title)
        result.append((label, children))
    return result


def expected_outline_heading_ids(documents=None):
    """Independent source heading ids, in the published outline's order."""
    if documents is None:
        documents = {path.name: path.read_text(encoding="utf-8")
                     for path in CHAPTERS.glob("*.md")}
    result = []
    for index, (name, _label) in enumerate(EXPECTED_CHAPTERS):
        chapter_id = "ch-" + str(int(name.split("_", 1)[0]))
        ids = [chapter_id]
        section_level = 2 if name == "00_overview.md" else 3
        for level, _title, primary in semantic_heading_ids(documents[name]):
            if (level == section_level or
                    (name in EXPECTED_SUBSECTION_COUNTS and level == section_level + 1)):
                ids.append(f"{chapter_id}-{primary}")
        result.append(ids)
    return result


def bookmark_destination_issues(reader, bookmark, heading_id):
    """Reject title matches on TOC pages and wrong positions on the same page."""
    issues = []
    named = reader.named_destinations
    matches = [named[key] for key in (heading_id, "/" + heading_id) if key in named]
    if not matches:
        return [f"PDF 缺少标题命名目标：{heading_id}"]
    signatures = []
    for destination in matches:
        try:
            page = reader.get_destination_page_number(destination)
        except Exception:
            return [f"PDF 标题命名目标无法解析：{heading_id}"]
        if page is None or not 0 <= page < len(reader.pages):
            return [f"PDF 标题命名目标越界：{heading_id}"]
        signatures.append((page, tuple(destination.dest_array[1:])))
    if any(signature != signatures[0] for signature in signatures[1:]):
        return [f"PDF 标题命名目标歧义：{heading_id}"]
    try:
        page = reader.get_destination_page_number(bookmark)
        signature = (page, tuple(bookmark.dest_array[1:]))
    except Exception:
        return [f"PDF 书签目标无法解析：{heading_id}"]
    if page is None or not 0 <= page < len(reader.pages):
        return [f"PDF 书签目标越界：{heading_id}"]
    if signature != signatures[0]:
        issues.append(f"PDF 书签与正文命名目标不一致：{heading_id} -> p{page + 1}，"
                      f"应为 p{signatures[0][0] + 1} 的指定标题位置")
    return issues


def extract_figure_references(text: str):
    pattern = re.compile(r"!\[([^\]]*)\]\(\.\./figures/(fig(\d{2})_[^\s)\"]+\.png)(?:\s+\"[^\"]*\")?\)")
    return [(match.group(1), match.group(2), int(match.group(3)))
            for match in pattern.finditer(strip_fenced_code(text))]


def figure_inventory_issues(references, png_names):
    issues = []
    refs = Counter(name for _alt, name, _number in references)
    numbers = set()
    names_by_number = {}
    for alt, name, number in references:
        numbers.add(number)
        names_by_number.setdefault(number, set()).add(name)
        alt_match = re.match(r"^\s*图\s*(\d+)(?=\D|$)", alt)
        if not alt_match:
            issues.append(f"图片 alt 未以图号开头：{name}: {alt!r}")
        elif int(alt_match.group(1)) != number:
            issues.append(f"图号与文件名不匹配：alt 图{alt_match.group(1)} -> {name}")
    if numbers != EXPECTED_FIGURE_NUMBERS:
        issues.append(
            f"正文图号应为 1..{max(EXPECTED_FIGURE_NUMBERS)}：缺失 {sorted(EXPECTED_FIGURE_NUMBERS - numbers)}，"
            f"多出 {sorted(numbers - EXPECTED_FIGURE_NUMBERS)}")
    for number, names in names_by_number.items():
        if len(names) > 1:
            issues.append(f"同一图号对应多个文件：图{number}: {sorted(names)}")
    orphans = sorted(set(png_names) - set(refs))
    if orphans:
        issues.append(f"孤立 PNG（正文未引用）：{orphans}")
    missing = sorted(set(refs) - set(png_names))
    if missing:
        issues.append(f"正文引用但图片目录缺失：{missing}")
    return issues


def png_provenance_issues(image_path: Path, script_path: Path):
    from PIL import Image
    with Image.open(image_path) as image:
        metadata = dict(image.info)
    source = metadata.get("SourceScript", "")
    accepted_sources = {script_path.name, script_path.relative_to(ROOT).as_posix()}
    expected_digest = hashlib.sha256(script_path.read_bytes()).hexdigest()
    issues = []
    if source not in accepted_sources:
        issues.append(f"SourceScript={source!r}，应为 {sorted(accepted_sources)} 之一")
    if metadata.get("SourceScriptDigest") != expected_digest:
        issues.append("SourceScriptDigest 与当前绘图脚本的完整 sha256 不一致")
    return issues


def wpe_figure_provenance_issues(metadata, report):
    """图21须同时绑定绘图源、唯一WPE核及其标量验证源。"""
    paths = ("scripts/make_figures.py",
             "codes/chapters/ch07/core/dereverberation.py",
             "codes/chapters/ch02/core/conventions.py")
    expected = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in paths}
    issues = []
    try:
        inputs = json.loads(metadata.get("GeneratorInputs", ""))
    except (TypeError, ValueError):
        inputs = None
    if inputs != expected:
        issues.append("图21 PNG真实生成输入摘要失效")
    if (not isinstance(report, dict) or report.get("generator_inputs") != expected or
            report.get("source_sha256") != expected[paths[0]]):
        issues.append("图21数值报告真实生成输入摘要失效")
    return issues


def check_figures(errors: list[str]):
    references = []
    for path in CHAPTERS.glob("*.md"):
        references.extend(extract_figure_references(path.read_text(encoding="utf-8")))
    png_names = [path.name for path in (ROOT / "figures").glob("fig*.png")]
    errors.extend(figure_inventory_issues(references, png_names))
    for name in sorted(set(item[1] for item in references)):
        path = ROOT / "figures" / name
        if not path.exists() or path.stat().st_size == 0:
            fail(errors, f"图片缺失或为空：figures/{name}")
            continue
        try:
            from PIL import Image
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
            if width < 800 or height < 300:
                fail(errors, f"图片分辨率过低：figures/{name}: {width}×{height}")
            number = int(re.match(r"fig(\d{2})_", name).group(1))
            script_name = ("make_figures.py" if number <= 25 or number in (33, 34, 35, 36, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72)
                           else "make_aec_figures.py")
            script_path = ROOT / "scripts" / script_name
            for issue in png_provenance_issues(path, script_path):
                fail(errors, f"PNG 溯源失效：figures/{name}: {issue}")
            if number == 21:
                report_path = ROOT / "codes/chapters/ch07/reports/figure21_wpe.json"
                with Image.open(path) as image:
                    for issue in wpe_figure_provenance_issues(
                            image.info, json.loads(report_path.read_text(encoding="utf-8"))):
                        fail(errors, issue)
            if number == 44:
                expected = hashlib.sha256((ROOT / "codes/chapters/ch09/tracking_audio/MANIFEST.json").read_bytes()).hexdigest()
                with Image.open(path) as image:
                    if image.info.get("AudioManifestDigest") != expected:
                        fail(errors, "图44独立追踪音频清单摘要失效")
            if number == 58:
                expected = hashlib.sha256((APA_AUDIO_ROOT / "MANIFEST.json").read_bytes()).hexdigest()
                with Image.open(path) as image:
                    if image.info.get("AudioManifestDigest") != expected:
                        fail(errors, "图58独立APA音频清单摘要失效")
            if number == 59:
                expected = hashlib.sha256((MINT_AUDIO_ROOT / "MANIFEST.json").read_bytes()).hexdigest()
                with Image.open(path) as image:
                    if image.info.get("AudioManifestDigest") != expected:
                        fail(errors, "图59独立已知路径逆音频清单摘要失效")
            if number == 60:
                expected = hashlib.sha256((MASK_AUDIO_ROOT / "MANIFEST.json").read_bytes()).hexdigest()
                with Image.open(path) as image:
                    if image.info.get("AudioManifestDigest") != expected:
                        fail(errors, "图60独立FFT掩码音频清单摘要失效")
            if number == 64:
                expected = hashlib.sha256((SCENARIO_AUDIO_ROOT / "MANIFEST.json").read_bytes()).hexdigest()
                with Image.open(path) as image:
                    if image.info.get("AudioManifestDigest") != expected:
                        fail(errors, "图64跨场景清单摘要失效")
                _check_selection_scenarios_report(ROOT / "codes/chapters/ch11/reports/figure64_selection_scenarios.json")
            if number == 65:
                expected = hashlib.sha256((WEIGHTED_AUDIO_ROOT / "MANIFEST.json").read_bytes()).hexdigest()
                with Image.open(path) as image:
                    if image.info.get("AudioManifestDigest") != expected:
                        fail(errors, "图65已知噪声权重清单摘要失效")
                _check_weighted_noise_report(ROOT / "codes/chapters/appendix_a/reports/figure65_weighted_noise.json")
            if number == 66:
                manifest_path = RESPONSE_AUDIO_ROOT / 'MANIFEST.json'
                with Image.open(path) as image:
                    if image.info.get('AudioManifestDigest') != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
                        fail(errors, '图66同DRR频响清单摘要失效')
                _check_equal_drr_report(ROOT/'codes/chapters/appendix_b/reports/figure66_equal_drr_response.json')
            if number == 63:
                expected = hashlib.sha256((NOISE_AUDIO_ROOT / "MANIFEST.json").read_bytes()).hexdigest()
                with Image.open(path) as image:
                    if image.info.get("AudioManifestDigest") != expected:
                        fail(errors, "图63独立噪声失配清单摘要失效")
            if number == 62:
                _check_engineering_limits_report(ROOT / "codes/chapters/ch10/reports/figure62_engineering_limits.json")
            if number == 61:
                _check_tracking_information_report(ROOT / "codes/chapters/ch09/reports/figure61_tracking_information.json")
            if number in (34, 35, 36, 40, 41, 43, 45, 47, 49):
                expected = hashlib.sha256((ROOT / "codes/chapters/ch00/audio/MANIFEST.json").read_bytes()).hexdigest()
                with Image.open(path) as image:
                    if image.info.get("AudioManifestDigest") != expected:
                        fail(errors, f"图 {number} 音频清单摘要失效")
        except Exception as exc:
            fail(errors, f"图片无法解码：figures/{name}: {exc}")


IMAGING_FIGURE_REPORTS = {
    67: 'figure67_imaging_psf.json',
    68: 'figure68_imaging_model_checks.json',
    69: 'figure69_imaging_calibration.json',
}
IMAGING_FIGURE_SOURCES = (
    'scripts/make_figures.py', 'codes/chapters/ch14/core/imaging.py',
    'codes/chapters/ch02/core/conventions.py', 'codes/chapters/ch00/io_contracts.py',
)


def _check_imaging_figure_report(path, number):
    """Read current source-bound reports against independent fixed controls.

    No drawing/teaching function is imported, no report is repaired, and no
    image is generated. The independent controls fix the declared inputs;
    changing the report's inputs cannot manufacture a matching expectation.
    PNG decoding, script provenance and final visual QA remain separate checks.
    """
    import numpy as np
    from fractions import Fraction
    report = strict_json_loads(validate_parent_chain(path).read_bytes())
    if (not isinstance(report, dict)
            or set(report) != {'schema_version', 'scope', 'source_sha256', 'results'}
            or type(report['schema_version']) is not int or report['schema_version'] != 1
            or report['scope'] != 'deterministic teaching controls; no industrial measurement'
            or not isinstance(report['source_sha256'], dict)
            or set(report['source_sha256']) != set(IMAGING_FIGURE_SOURCES)
            or not isinstance(report['results'], dict)):
        raise ValueError(f'图{number}报告结构、范围或四源集合不同')
    for relative in IMAGING_FIGURE_SOURCES:
        actual = hashlib.sha256(validate_parent_chain(ROOT / relative).read_bytes()).hexdigest()
        if report['source_sha256'][relative] != actual:
            raise ValueError(f'图{number}真实源摘要过期：{relative}')

    def reject_bool(value):
        if isinstance(value, list):
            for item in value:
                reject_bool(item)
        elif type(value) not in (int, float):
            raise ValueError(f'图{number}数值字段含非数字或布尔值')

    def close(actual, expected, label):
        reject_bool(actual)
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        if (a.shape != b.shape or not np.isfinite(a).all()
                or not np.allclose(a, b, rtol=2e-12, atol=2e-14)):
            raise ValueError(f'图{number}独立数值不符：{label}')

    def complex_close(actual, expected, label):
        if not isinstance(actual, dict) or set(actual) != {'real', 'imag'}:
            raise ValueError(f'图{number}复数存储不同：{label}')
        close(actual['real'], np.real(expected), label+' real')
        close(actual['imag'], np.imag(expected), label+' imag')

    def exact_int(actual, expected, label):
        if type(actual) is not int or actual != expected:
            raise ValueError(f'图{number}整数条件不同：{label}')

    a = np.array([[1, 1], [1, np.exp(-2j*np.pi/3)]])
    w, q = a/2, np.array([1., .25])
    p = np.array([[1., .25], [.25, 1.]])

    def two_cell(data):
        complex_close(data['A'], a, '两格负相位传播')
        complex_close(data['W'], w, '两格权重')
        close(data['P'], p, '两格PSF')
        close(data['q'], q, '两格边际源量')
        for key, expected_r, expected_b, inverse, relative in (
                ('independent', a@np.diag(q)@a.conj().T, [17/16, .5], q, 0.),
                ('coherent', np.outer(a@np.array([1., .5]), (a@np.array([1., .5])).conj()),
                 [21/16, .75], [1.2, .45], np.sqrt(.15))):
            case = data['cases'][key]
            complex_close(case['R'], expected_r, key+' CSM')
            close(case['b'], expected_b, key+'扫描量')
            close(case['q_inverse'], inverse, key+'逆解')
            close(case['full_csm_residual']['relative_frobenius'], relative, key+'完整CSM残差')
        gamma = np.linspace(-1., 1., 41)
        close(data['coherence_gamma'], gamma, '互相干控制点')
        close(data['coherence_inverse_q'], np.column_stack([1+.2*gamma, .25+.2*gamma]),
              '交叉项改变非相干逆解')

    data = report['results']
    if number in (67, 68):
        two_cell(data['two_cell'])
    if number == 67:
        spatial = data['spherical_scan']
        close(spatial['frequency_hz'], 4000., '频率')
        close(spatial['sound_speed_m_s'], 343., '声速')
        close(spatial['grid_shape'], [21, 21], '网格形状')
        close(spatial['source_grid_indices'], [215, 225], '真实源格')
        close(spatial['reference_position_m'], [0, 0, 0], '参考位置')
        angles = 2*np.pi*np.arange(8)/8
        microphones = np.column_stack([.2*np.cos(angles), .2*np.sin(angles), np.zeros(8)])
        axis = np.linspace(-.3, .3, 21)
        xx, yy = np.meshgrid(axis, axis, indexing='xy')
        grid = np.column_stack([xx.ravel(), yy.ravel(), np.full(441, .6)])
        close(spatial['microphones_m'], microphones, '8麦几何')
        close(spatial['grid_m'], grid, '441格坐标')
        close(spatial['grid_axis_m'], axis, '坐标轴')
        close(spatial['source_positions_m'], grid[[215, 225]], '真实源坐标')
        close(spatial['q'], q, '空间扫描边际源量')
        distances = np.linalg.norm(grid[None, :, :]-microphones[:, None, :], axis=2)
        reference = np.linalg.norm(grid, axis=1)
        transfer = reference[None, :]/distances*np.exp(
            -2j*np.pi*4000/343*(distances-reference[None, :]))
        weights = transfer/np.sum(np.abs(transfer)**2, axis=0)
        csm = transfer[:, [215, 225]]@np.diag(q)@transfer[:, [215, 225]].conj().T
        # Each scalar quadratic is evaluated independently of teaching scan code.
        scan = np.array([np.vdot(col, csm@col).real for col in weights.T])
        columns = np.abs(weights.conj().T@transfer[:, [215, 225]])**2
        complex_close(spatial['A'], transfer, '球面传播')
        complex_close(spatial['W'], weights, '球面权重')
        complex_close(spatial['R'], csm, '已知两源CSM')
        close(spatial['psf_columns'], columns, '两源PSF列')
        close(spatial['b'], scan, '完整二维扫描')
        direction = grid/np.linalg.norm(grid, axis=1)[:, None]
        plane = np.exp(2j*np.pi*4000/343*(microphones@direction.T))/8
        close(spatial['plane_mismatch_b'], [np.vdot(col, csm@col).real for col in plane.T],
              '错误平面波模型')
        close(spatial['dirty_grid_sum'], scan.sum(), '扫描格求和')
        close(spatial['source_power_sum'], 1.25, '源总量')
        if len(spatial['local_peaks']) != 2:
            raise ValueError('图67两处局部峰数量不同')
        for source_index, grid_index in enumerate((215, 225)):
            local = np.flatnonzero(np.linalg.norm(grid[:, :2]-grid[grid_index, :2], axis=1) <= .09+1e-12)
            peak = int(local[np.argmax(scan[local])])
            row = spatial['local_peaks'][source_index]
            exact_int(row['peak_grid_index'], peak, '局部峰格索引')
            close(row['peak_position_m'], grid[peak], '局部峰位置')
            close(row['position_error_m'], np.linalg.norm(grid[peak]-grid[grid_index]), '峰偏移')
    elif number == 68:
        if type(data['log_plot_floor']) is not float or data['log_plot_floor'] != 1e-16:
            raise ValueError('图68对数显示地板不同')
        for name, passes in (('forward', 1), ('forward_backward', 2)):
            result = data[name]
            exact_int(result['iterations'], 8, '外层轮数')
            exact_int(result['passes_per_iteration'], passes, '每轮扫描次数')
            if result['sweep'] != name:
                raise ValueError('图68遍历方向不同')
            close(result['relaxation'], 1., '松弛系数')
            # Exact rational closed form for this two-cell zero-start control.
            # Forward: e1=(1/16)^n, e2=-e1/4. The final reverse pass also
            # updates e1=-e2/4, reducing e1 by 16 without another e2 update.
            history = [[0., 0.]]
            for n in range(1, 9):
                error = Fraction(1, 16**n)
                history.append([float(1+error/(16 if passes == 2 else 1)), float(Fraction(1, 4)-error/4)])
            close(result['history'], history, name+'精确分数逐轮状态')
            close(result['q'], history[-1], name+'末轮状态')
            close(result['scan_residual_history'], np.asarray(history)@p.T-[17/16, .5], name+'扫描残差')
        clean = data['rank_one_clean_sc']
        exact_int(clean['completed_iterations'], 20, 'CLEAN轮数')
        close(clean['damping'], .6, 'CLEAN扣除比例')
        if len(clean['steps']) != 20:
            raise ValueError('图68 CLEAN逐轮记录数量不同')
        for n, step in enumerate(clean['steps'], 1):
            power = 2*.4**(n-1)
            exact_int(step['iteration'], n, 'CLEAN轮序')
            exact_int(step['peak_index'], 0, 'CLEAN唯一峰格')
            close(step['peak_power'], power, 'CLEAN扣除前功率')
            close(step['allocated_power'], .6*power, 'CLEAN本轮分配')
            close(step['residual_scan'], [2*.4**n], 'CLEAN扣除后扫描')
            complex_close(step['h'], np.ones(2), 'CLEAN分量')
            complex_close(step['component_csm'], power*np.ones((2, 2)), 'CLEAN分量CSM')
        cumulative = 2*(1-.4**np.arange(1, 21))
        close(data['rank_one_cumulative'], cumulative, 'CLEAN累计分配')
        close(clean['clean_map'], [cumulative[-1]], 'CLEAN末轮累计图')
        complex_close(clean['residual_csm'], 2*.4**20*np.ones((2, 2)), 'CLEAN末轮残余CSM')
    elif number == 69:
        gains = np.linspace(.4, 2.2, 61)
        close(data['second_channel_gain'], gains, '已知增益控制')
        close(data['uncalibrated_scan'], (1+gains)**2/4, '校准前扫描')
        close(data['calibrated_scan'], np.ones(61), '校准后扫描')
        close(data['unequal_amplitude_diagonal_controls'], [1., 8/25, 16/25, 1.], '不等幅删对角控制')
        close(data['different_csm_objective_optima'], [41/25, 41/21, 87/35], '三种明确拟合目标')
        close(data['intercept_prediction'], -8/5, '独立截距预期')
        close(data['region_db_relative_to_one'],
              [10*np.log10(5/4), 10*np.log10(25/16), -10*np.log10(2)], '线性量汇总后的dB')
        if data['scope'] != 'analytic controls; default-intercept estimate is independent mathematics, not sklearn execution':
            raise ValueError('图69截距执行范围说明不同')
    else:
        raise ValueError('unknown imaging figure identity')


def check_imaging_figures(errors):
    """Strictly read three reports, preserving every failed source/numeric check."""
    for number, filename in IMAGING_FIGURE_REPORTS.items():
        try:
            _check_imaging_figure_report(ROOT/'codes/chapters/ch14/reports'/filename, number)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            fail(errors, f'图{number}声学成像报告：{error}')


def _check_engineering_limits_report(path):
    """Independent integer recurrence and geometric-series checks for figure62."""
    import math
    from fractions import Fraction
    report = _read_tracking_manifest(path)
    if (type(report.get('schema_version')) is not int or report.get('schema_version') != 1 or report.get('script_sha256') !=
            hashlib.sha256((ROOT/'scripts/make_figures.py').read_bytes()).hexdigest()):
        raise ValueError('图62真实绘图源摘要不同')
    update = report['soft_update']
    if (any(type(value) is not int for value in update['power']) or
            type(update['initial_noise']) is not int or
            type(update['initial_probability']) is not int or
            update['power'] != [9,9,9] or update['initial_noise'] != 1 or
            update['initial_probability'] != 0 or update['alpha_p'] != '1/2' or
            update['alpha_d'] != '4/5'):
        raise ValueError('图62噪声递推题设不同')
    expected = [(1,'1/2','9/10','9/5','13/5','1'),
                (1,'3/4','19/20','54/25','97/25','1'),
                (0,'3/8','7/8','603/200','613/125','13/5')]
    rows = update['rows']
    if len(rows) != 3:
        raise ValueError('图62递推次数不同')
    for row, values in zip(rows,expected):
        fields = ['indicator','speech_probability','retention','soft_noise','ungated_noise','hard_noise']
        if type(row['indicator']) is not int or tuple(row[field] for field in fields) != values:
            raise ValueError('图62独立分数递推不符')
    finite = report['finite_tail']
    def close(actual, expected):
        if isinstance(actual,bool) or not isinstance(actual,(int,float)) or not math.isfinite(actual) or not math.isclose(actual,expected,rel_tol=2e-14,abs_tol=2e-14):
            raise ValueError('图62独立逆积分数值不符')
    if finite['sample_rate_hz'] != 10 or not finite['not_rt20']:
        raise ValueError('图62截尾条件缺失')
    if any(len(finite[key]) != 4 for key in ('squared_impulse','reverse_energy','time_s','finite_db','infinite_db')):
        raise ValueError('图62逆积分数组长度不同')
    for n in range(4):
        close(finite['squared_impulse'][n],float(Fraction(1,2**n)))
        energy=sum((Fraction(1,2**k) for k in range(n,4)),Fraction(0))
        close(finite['reverse_energy'][n],float(energy))
        close(finite['time_s'][n],n/10)
        close(finite['finite_db'][n],10*math.log10(float(energy/Fraction(15,8))))
        close(finite['infinite_db'][n],10*math.log10(2**(-n)))
    close(finite['infinite_t60_s'],60/(10*math.log10(2)*10))
    close(finite['four_point_endpoint_extrapolation_s'],60*.3/(10*math.log10(15)))
    tasks = report['nonpreemptive']
    expected_tasks={'B_start_ms':-1,'B_finish_ms':14,'A_release_ms':0,'A_deadline_ms':10,
                    'A_start_ms':14,'A_finish_ms':16,'A_response_ms':16,
                    'blocking_supremum_plus_service_ms':17,'utilization':.35}
    if (tasks != expected_tasks or any(type(tasks[key]) is not type(value)
                                      for key, value in expected_tasks.items())):
        raise ValueError('图62非抢占时间线与上界不同')


def _check_tracking_information_report(path):
    """Recompute figure61 using scalar geometry and averaged noise precision."""
    import math
    report = _read_tracking_manifest(path)
    script = ROOT / 'scripts/make_figures.py'
    if (report['script_sha256'] != hashlib.sha256(script.read_bytes()).hexdigest()
            or report['schema_version'] != 1
            or 'ignoring propagation delay' not in report['scope']
            or report['stationary_observer_xy_m'] != [0, 0]
            or report['initial_position_m'] != [1, 2] or report['velocity_m_s'] != [1, 0]
            or report['comparison_scale'] != 2):
        raise ValueError('图61生成源或瞬时几何模型不符')
    def close(actual, expected):
        if (type(actual) not in (int, float) or not math.isfinite(actual)
                or not math.isclose(actual, expected, rel_tol=2e-13, abs_tol=1e-14)):
            raise ValueError('图61独立解析复算不符')
    times, positions, angles = report['time_s'], report['positions_xy_m'], report['bearing_deg']
    if any(len(values) != 101 for values in (times, positions, angles)):
        raise ValueError('图61几何采样数量不符')
    for i, (time, position, angle) in enumerate(zip(times, positions, angles)):
        close(time, i/50)
        if len(position) != 2:
            raise ValueError('图61位置维度不符')
        close(position[0], 1+time); close(position[1], 2)
        close(angle, math.degrees(math.atan2(1+time, 2)))
    rows = report['jacobian_rows_without_positive_denominators']
    if rows != [[2, -1, 0, 0], [2, -2, 2, -2], [2, -3, 4, -6]] or report['scale_null_direction'] != [1, 2, 1, 0]:
        raise ValueError('图61雅可比/尺度零方向不符')
    state = report['same_scalar_state']
    if (state['prior_mean'] != 0 or state['prior_variance'] != 1 or state['observations'] != [1, 1]
            or state['measurement_variances'] != [1, 1]
            or state['rho_0_9_exact_mean'] != '20/39' or state['rho_0_9_exact_variance'] != '19/39'
            or 'singular batch inverse not performed' not in state['rho_1']):
        raise ValueError('图61相关观测模型或奇异极限说明不符')
    close(state['independent_assumption_variance'], 1/3)
    if any(len(state[key]) != 101 for key in ('rho', 'posterior_mean', 'posterior_variance')):
        raise ValueError('图61相关性采样数量不符')
    for i, (rho, mean, variance) in enumerate(zip(state['rho'], state['posterior_mean'], state['posterior_variance'])):
        close(rho, i/100)
        # The average of two same-state measurements has noise (1+rho)/2;
        # scalar Bayesian precision adds its reciprocal to prior precision 1.
        averaged_noise = (1+rho)/2
        close(mean, 1/(1+averaged_noise))
        close(variance, averaged_noise/(1+averaged_noise))


def check_site(errors: list[str]):
    pages = sorted(SITE.glob("*.html"))
    if len(pages) != EXPECTED_CHAPTER_COUNT:
        fail(errors, f"站点页面数应为 {EXPECTED_CHAPTER_COUNT}，实际 {len(pages)}")
    documents = {path.name: path.read_text(encoding="utf-8")
                 for path in CHAPTERS.glob("*.md")}
    source_by_page = {"index.html": "00_overview.md"}
    source_by_page.update({name.replace(".md", ".html"): name
                           for name, _label in EXPECTED_CHAPTERS[1:]})
    expected_digest = site_source_digest()
    for path in pages:
        page_text = path.read_text(encoding="utf-8")
        parser = PageParser()
        parser.feed(page_text)
        duplicates = [key for key, count in Counter(parser.ids).items() if count > 1]
        if duplicates:
            fail(errors, f"重复 HTML id：{path.name}: {duplicates[:5]}")
        if parser.h1_count != 1:
            fail(errors, f"页面应有且仅有一个 h1：{path.name}: {parser.h1_count}")
        if parser.html_lang != "zh-CN":
            fail(errors, f"页面语言应为 zh-CN：{path.name}: {parser.html_lang or '缺失'}")
        if any(next_level > level + 1 for level, next_level in
               zip(parser.heading_levels, parser.heading_levels[1:])):
            fail(errors, f"页面标题层级跳级：{path.name}: {parser.heading_levels}")
        for landmark in ("header", "main", "footer", "nav"):
            if not parser.tags[landmark]:
                fail(errors, f"页面缺少 {landmark} 语义地标：{path.name}")
        if parser.th_without_scope:
            fail(errors, f"表头缺少 scope：{path.name}: {parser.th_without_scope}")
        if parser.source_digest != expected_digest:
            fail(errors, f"站点页面不是当前源文件生成：{path.name}")
        if "main-content" not in parser.ids or '#main-content' not in parser.links:
            fail(errors, f"页面缺少键盘跳转正文链接：{path.name}")
        current_count = page_text.count('aria-current="page"')
        if current_count != 1:
            fail(errors, f"页面应有且仅有一个当前导航项：{path.name}: {current_count}")
        if ":focus-visible" not in page_text:
            fail(errors, f"页面缺少键盘焦点样式：{path.name}")
        for issue in url_scheme_issues(parser.links + [src for src, _alt in parser.images]):
            fail(errors, f"站点链接协议错误：{path.name}: {issue}")
        source_name = source_by_page.get(path.name)
        if source_name and source_name in documents:
            expected_ids, expected_images = expected_site_content(
                source_name, documents[source_name])
            missing_ids = sorted(expected_ids - set(parser.ids))
            if missing_ids:
                fail(errors, f"站点缺少源标题锚点：{path.name}: {missing_ids[:5]}")
            for issue in site_nav_fragment_issues(
                    path.name, source_name, documents[source_name], parser.nav_links):
                fail(errors, f"站点导航不完整：{path.name}: {issue}")
            for issue in external_markdown_link_issues(documents[source_name], parser.links):
                fail(errors, f"{path.name}: {issue}")
            actual_images = Counter(src for src, _alt in parser.images)
            if actual_images != expected_images:
                fail(errors, f"站点图片与源 Markdown 不一致：{path.name}: "
                     f"实际 {dict(actual_images)}，应为 {dict(expected_images)}")
    # 研究目录与教程目录存在同名 index.html，须按完整目标路径查片段。
    check_site_links(errors)

    ordered = ["index.html"] + [name.replace(".md", ".html")
        for name, _ in EXPECTED_CHAPTERS[1:]]
    for index, name in enumerate(ordered[1:], 1):
        page_text = (SITE / name).read_text(encoding="utf-8") if (SITE / name).exists() else ""
        match = re.search(r'<div class="pn">.*?href="([^"]+)".*?href="([^"]+)".*?</div>',
                          page_text, flags=re.S)
        expected = (ordered[index - 1], ordered[index + 1] if index + 1 < len(ordered)
                    else "index.html")
        if not match or match.groups() != expected:
            actual = match.groups() if match else "缺失"
            fail(errors, f"上一篇/下一篇导航顺序错误：{name}: {actual}，应为 {expected}")


def external_markdown_link_issues(source: str, links: list[str]):
    """外部 Markdown 来源必须仍是原 URL，不得按本地文档规则改为 HTML。"""
    import markdown
    source_parser = PageParser()
    source_parser.feed(markdown.markdown(strip_fenced_code(source), extensions=["tables"]))
    expected = Counter(href for href in source_parser.links
                       if urlparse(href).scheme in {"https", "http"}
                       and unquote(urlparse(href).path).lower().endswith(".md"))
    missing = expected - Counter(links)
    return [f"外部 Markdown 链接被改写或遗漏：{href}" for href in missing]


def check_research_site(errors: list[str]):
    research_root = RESEARCH_ROOT
    output_root = SITE / "research"
    expected_names = {output for _source, output in EXPECTED_RESEARCH_PAGES}
    actual_names = {path.relative_to(output_root).as_posix()
                    for path in output_root.rglob("*.html")}
    if actual_names != expected_names:
        fail(errors, f"研究页面集不符合独立基线：缺失 {sorted(expected_names - actual_names)}，"
             f"多出 {sorted(actual_names - expected_names)}")
    if len(actual_names) != EXPECTED_RESEARCH_PAGE_COUNT:
        fail(errors, f"研究页面数应为 {EXPECTED_RESEARCH_PAGE_COUNT}，实际 {len(actual_names)}")
    expected_digest = site_source_digest()
    for source_name, output_name in EXPECTED_RESEARCH_PAGES:
        source_path = research_root / source_name
        path = output_root / output_name
        if not source_path.is_file():
            fail(errors, f"研究源文档缺失：{source_name}")
            continue
        if not path.is_file():
            continue  # 页面集差异已记录缺页，不用缺页触发文件读取异常。
        source = source_path.read_text(encoding="utf-8")
        text = path.read_text(encoding="utf-8")
        parser = PageParser()
        parser.feed(text)
        prefix = f"research/{output_name}"
        if parser.source_digest != expected_digest:
            fail(errors, f"研究页面不是当前源文件生成：{prefix}")
        if parser.h1_count != 1 or parser.html_lang != "zh-CN":
            fail(errors, f"研究页面须有唯一 h1 和 zh-CN 语言：{prefix}")
        duplicates = [key for key, count in Counter(parser.ids).items() if count > 1]
        if duplicates:
            fail(errors, f"研究页面重复 HTML id：{prefix}: {duplicates[:5]}")
        if any(right > left + 1 for left, right in
               zip(parser.heading_levels, parser.heading_levels[1:])):
            fail(errors, f"研究页面标题层级跳级：{prefix}")
        for landmark in ("header", "main", "footer", "nav"):
            if not parser.tags[landmark]:
                fail(errors, f"研究页面缺少 {landmark} 语义地标：{prefix}")
        if parser.th_without_scope:
            fail(errors, f"研究页面表头缺少 scope：{prefix}")
        if ("main-content" not in parser.ids or "#main-content" not in parser.links
                or ":focus-visible" not in text):
            fail(errors, f"研究页面缺少键盘跳转或焦点样式：{prefix}")
        if text.count('aria-current="page"') != 1:
            fail(errors, f"研究页面须有唯一当前导航项：{prefix}")
        expected_ids = {primary for _level, _title, primary in semantic_heading_ids(source)}
        if expected_ids - set(parser.ids):
            fail(errors, f"研究页面缺少源标题锚点：{prefix}: "
                 f"{sorted(expected_ids - set(parser.ids))[:5]}")
        for issue in site_nav_fragment_issues(output_name, source_name, source, parser.nav_links):
            fail(errors, f"研究页面导航不完整：{prefix}: {issue}")
        for issue in external_markdown_link_issues(source, parser.links):
            fail(errors, f"{prefix}: {issue}")


def check_site_links(errors: list[str]):
    """跨全部教程/研究页面核对 URL、文件和片段，不使用易冲突的 basename。"""
    parsed = {}
    for path in SITE.rglob("*.html"):
        parser = PageParser()
        parser.feed(path.read_text(encoding="utf-8"))
        parsed[path.resolve()] = parser
    for path, parser in parsed.items():
        name = path.relative_to(SITE.resolve()).as_posix()
        for issue in url_scheme_issues(parser.links + [src for src, _alt in parser.images]):
            fail(errors, f"站点链接协议错误：{name}: {issue}")
        for href in parser.links:
            url = urlparse(href)
            if url.scheme or url.netloc:
                continue
            target = ((SITE / unquote(url.path).lstrip("/")) if url.path.startswith("/")
                      else (path.parent / unquote(url.path)) if url.path else path).resolve()
            if not target.is_file():
                fail(errors, f"站内链接目标不存在：{name} -> {href}")
                continue
            if target.suffix.lower() == ".html" and url.fragment:
                if target not in parsed:
                    extra = PageParser()
                    extra.feed(target.read_text(encoding="utf-8"))
                    # 外部目录的本地 HTML 也必须核对，且避免边遍历边改字典。
                    target_ids = extra.ids
                else:
                    target_ids = parsed[target].ids
                if unquote(url.fragment) not in target_ids:
                    fail(errors, f"站内锚点不存在：{name} -> {href}")
        for src, alt in parser.images:
            url = urlparse(src)
            if not url.scheme and not url.netloc:
                target = (SITE / unquote(url.path).lstrip("/") if url.path.startswith("/")
                          else path.parent / unquote(url.path))
                if not target.resolve().is_file():
                    fail(errors, f"站点图片不存在：{name} -> {src}")
            if alt is None or not alt.strip():
                fail(errors, f"站点图片缺少非空替代文本：{name} -> {src}")


def check_combined_html(errors: list[str]):
    path = DIST / "combined.html"
    if not path.exists():
        fail(errors, "缺少 dist/combined.html")
        return
    text = path.read_text(encoding="utf-8")
    if "file://" in text:
        fail(errors, "dist/combined.html 含 file:// 链接")
    if re.search(r'href="(?:\./)?(?:\d{2})_[^"]+\.html', text):
        fail(errors, "dist/combined.html 含分篇 HTML 死链")
    if re.search(r"<blockquote>\s*</blockquote>", text, flags=re.S):
        fail(errors, "dist/combined.html 含空引用块，打印后会留下无文字色条")
    parser = PageParser()
    parser.feed(text)
    ids = set(parser.ids)
    documents = {path.name: path.read_text(encoding="utf-8")
                 for path in CHAPTERS.glob("*.md")}
    expected_ids = set()
    expected_images = Counter()
    for index, (name, _label) in enumerate(EXPECTED_CHAPTERS):
        chapter_id = "ch-" + str(int(name.split("_", 1)[0]))
        expected_ids.add(chapter_id)
        section_level = 2 if name == "00_overview.md" else 3
        for level, _title, primary in semantic_heading_ids(documents[name]):
            if (level == section_level
                    or (name in EXPECTED_SUBSECTION_COUNTS
                        and level == section_level + 1)):
                expected_ids.add(f"{chapter_id}-{primary}")
        expected_images.update(
            f"../figures/{figure_name}"
            for _alt, figure_name, _number in extract_figure_references(documents[name])
        )
        if name == "13_appendix-guide.md":
            expected_images["../codes/chapters/appendix_b/room_audio/ROOM_RESULTS.png"] += 1
    missing_ids = sorted(expected_ids - ids)
    if missing_ids:
        fail(errors, f"dist/combined.html 缺少源章节或小节：{missing_ids[:8]}")
    actual_images = Counter(src for src, _alt in parser.images)
    if actual_images != expected_images:
        fail(errors, "dist/combined.html 图片引用与源 Markdown 不一致")
    if "全书完" not in text:
        fail(errors, "dist/combined.html 缺少固定结束标记“全书完”")
    for issue in url_scheme_issues(parser.links + [src for src, _alt in parser.images]):
        fail(errors, f"dist/combined.html 链接协议错误：{issue}")
    for href in parser.links:
        if href.startswith("#") and href[1:] not in ids:
            fail(errors, f"dist/combined.html 内部锚点不存在：{href}")
    expected = source_digest()
    if f"源文件 sha256 {expected}" not in text:
        fail(errors, f"合订 HTML 不是当前源文件生成：应含 {expected}")


def source_digest():
    digest = hashlib.sha256()
    paths = sorted(CHAPTERS.glob("*.md"))
    paths += [RESEARCH_ROOT / name for name, _ in EXPECTED_RESEARCH_PAGES]
    paths += [ROOT / "scripts" / "build_site.py", ROOT / "scripts" / "build_markdown_helpers.py",
              ROOT / "scripts" / "inline_layout.js",
              ROOT / "scripts" / "heading_aliases.py",
              ROOT / "scripts" / "legacy_sequential_anchors.json"]
    paths += sorted(path for path in (ROOT / "scripts" / "vendor" / "mathjax-3.2.2").rglob("*")
                    if path.is_file())
    paths += sorted((ROOT / "figures").glob("fig*.png"))
    paths += [ROOT / "scripts" / name for name in
              ("build_pdf.py", "make_figures.py", "make_aec_figures.py")]
    paths.append(ROOT / "requirements.txt")
    paths.append(ROOT / "codes/chapters/ch00/io_contracts.py")
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def site_source_digest():
    digest = hashlib.sha256()
    paths = sorted(CHAPTERS.glob("*.md"))
    paths += [RESEARCH_ROOT / name for name, _ in EXPECTED_RESEARCH_PAGES]
    manifest_path = main_audio_manifest_path(CODE_CHAPTERS)
    paths += [manifest_path]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths += sorted(main_audio_path(CODE_CHAPTERS, record["group"], record["file"])
                    for record in manifest["files"])
    for asset_root in (REAL_AUDIO_ROOT, ROOM_AUDIO_ROOT, MOVING_AUDIO_ROOT,
                       TRACKING_AUDIO_ROOT, GSS_AUDIO_ROOT, BINAURAL_AUDIO_ROOT, SPECTRAL_AUDIO_ROOT, STFT_AUDIO_ROOT, GEOMETRY_AUDIO_ROOT, FOCUS_AUDIO_ROOT, DERIVATIVE_AUDIO_ROOT, APA_AUDIO_ROOT, MINT_AUDIO_ROOT, MASK_AUDIO_ROOT, NOISE_AUDIO_ROOT, SCENARIO_AUDIO_ROOT, WEIGHTED_AUDIO_ROOT, RESPONSE_AUDIO_ROOT, IMAGING_AUDIO_ROOT, DISTRIBUTED_AUDIO_ROOT):
        paths += sorted(asset_root.glob("*"))
    paths += sorted((ROOT / "figures").glob("fig*.png"))
    paths += [ROOT / "scripts" / name for name in
              ("build_site.py", "build_markdown_helpers.py", "inline_layout.js", "heading_aliases.py",
               "legacy_sequential_anchors.json", "code_layout.py", "make_figures.py",
               "make_aec_figures.py")]
    paths.append(ROOT / "requirements.txt")
    paths.append(ROOT / "codes/chapters/ch00/io_contracts.py")
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def norm(text: str):
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"[\s·：:，,。；;、“”‘’「」『』（）()—–\-]", "", text)


def bookmark_title_matches_page(title: str, page_text: str) -> bool:
    """接受 PDF 提取保留标题文字、但丢失 MathJax 公式的情况。"""
    page_key = norm(page_text)
    title_key = norm(title)
    if title_key and title_key in page_key:
        return True
    if "$" not in title:
        return False
    mathless = norm(re.sub(r"\$[^$]*\$", "", title))
    return len(mathless) >= 8 and mathless in page_key


def pdf_outline_tree(items):
    """把 pypdf 的交错 destination/list 表示转成 [(destination, children)]。"""
    result = []
    for item in items:
        if isinstance(item, list):
            if result:
                result[-1][1].extend(pdf_outline_tree(item))
        else:
            result.append([item, []])
    return result


def pdf_page_is_empty(extracted_text: str, image_count: int) -> bool:
    """Flag a page with neither searchable text nor raster content."""
    return not extracted_text.strip() and image_count == 0


def pdf_page_is_sparse(extracted_text: str, image_count: int, page_no: int) -> bool:
    """Catch near-empty body pages left by a split closing paragraph or callout."""
    visible = re.sub(r"\s+", "", extracted_text)
    return page_no > 5 and image_count == 0 and 0 < len(visible) < 120


def check_pdf(errors: list[str], notices: list[str]):
    path = DIST / "microphone-array-tutorial.pdf"
    if not path.exists():
        fail(errors, "缺少合订 PDF")
        return
    try:
        from pypdf import PdfReader
    except ImportError:
        fail(errors, "缺少 pypdf，无法检查 PDF")
        return
    reader = PdfReader(str(path))
    if len(reader.pages) < 100:
        fail(errors, f"PDF 页数异常：{len(reader.pages)}")
    metadata = reader.metadata or {}
    if metadata.get("/Title") != "麦克风阵列信号处理教程":
        fail(errors, "PDF 缺少正确的标题元数据")
    if metadata.get("/SourceDigest") != source_digest():
        fail(errors, "PDF 不是当前源文件生成，SourceDigest 不一致")
    if str(reader.trailer["/Root"].get("/Lang", "")) != "zh-CN":
        fail(errors, "PDF 缺少 /Lang zh-CN")
    root = reader.trailer["/Root"]
    mark_info = root.get("/MarkInfo")
    is_marked = bool(mark_info and mark_info.get_object().get("/Marked"))
    if not is_marked or not root.get("/StructTreeRoot"):
        fail(errors, "PDF 缺少结构标签根节点；检查 Chrome 导出与书签写入")
    else:
        structure = root["/StructTreeRoot"].get_object()
        if not structure.get("/K") or not structure.get("/ParentTree"):
            fail(errors, "PDF 结构树缺少内容或父树")
        if not any("/StructParents" in page for page in reader.pages):
            fail(errors, "PDF 页面未连接到结构树")
        notices.append("PDF 已包含结构标签；尚未完成 PDF/UA、公式辅助文本与辅助技术阅读顺序的完整验收")
    width = float(reader.pages[0].mediabox.width)
    height = float(reader.pages[0].mediabox.height)
    if abs(width - 595.28) > 2 or abs(height - 841.89) > 2:
        fail(errors, f"PDF 纸型不是 A4：{width:.1f}×{height:.1f} pt")
    extracted = []
    for page_no, page in enumerate(reader.pages, 1):
        page_text = page.extract_text() or ""
        extracted.append(page_text)
        if not page_text.strip() and pdf_page_is_empty(page_text, len(page.images)):
            fail(errors, f"PDF 第 {page_no} 页没有正文或图片；检查章末装饰线与强制分页")
        if pdf_page_is_sparse(page_text, len(page.images), page_no):
            fail(errors, f"PDF 第 {page_no} 页只有少量正文且无图片；检查孤立转场、末段与强制分页")
        for ref in page.get("/Annots", []):
            obj = ref.get_object()
            action = obj.get("/A")
            uri = str(action.get("/URI")) if action and action.get("/URI") else ""
            if uri.startswith("file:"):
                fail(errors, f"PDF 本地路径链接：p{page_no}: {uri}")
    text = "\n".join(extracted)
    if not extracted or "全书完" not in extracted[-1]:
        fail(errors, "PDF 末页缺少固定结束标记“全书完”")
    elif norm(extracted[-1]) == "全书完":
        fail(errors, "PDF 末页只有结束标记，缺少同页正文")
    radicals = re.findall(r"[\u2e80-\u2eff\u2f00-\u2fdf]", text)
    if radicals:
        fail(errors, f"PDF 文本层含部首类错误码位：{len(radicals)} 个")
    expected = expected_outline()
    expected_ids = expected_outline_heading_ids()
    actual = pdf_outline_tree(reader.outline)
    if [item[0].title for item in actual] != [label for label, _ in expected]:
        fail(errors, "PDF 顶级书签标题或顺序与源 Markdown 不一致")
    else:
        for (parent, children), (label, expected_children), heading_ids in zip(
                actual, expected, expected_ids):
            if [child.title for child, _subchildren in children] != [
                    title for title, _subtitles in expected_children]:
                fail(errors, f"PDF 节书签标题或顺序不一致：{label}")
                continue
            destinations = [parent]
            for (child, subchildren), (_title, expected_subtitles) in zip(
                    children, expected_children):
                actual_subtitles = [subchild.title for subchild, _ in subchildren]
                if actual_subtitles != expected_subtitles:
                    fail(errors, f"PDF 子节书签标题或顺序不一致：{label} / {child.title}")
                destinations.append(child)
                destinations.extend(subchild for subchild, _ in subchildren)
            if len(destinations) != len(heading_ids):
                fail(errors, f"PDF 书签与源标题目标数量不一致：{label}")
            for destination, heading_id in zip(destinations, heading_ids):
                errors.extend(bookmark_destination_issues(reader, destination, heading_id))
                page_no = reader.get_destination_page_number(destination)
                page_text = extracted[page_no] if 0 <= page_no < len(extracted) else ""
                if not bookmark_title_matches_page(destination.title, page_text):
                    fail(errors, f"PDF 书签落页未出现标题：{destination.title} -> p{page_no + 1}")
    import importlib.util
    module_path = ROOT / "scripts" / "build_pdf.py"
    spec = importlib.util.spec_from_file_location("quality_math_detector", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if module.contains_unrendered_math(text):
        fail(errors, "PDF 疑似含未渲染的公式源码")


EXPECTED_AUDIO_STEMS = {
    "math_block_dry", "math_block_linear", "math_block_circular",
    "selection_clean", "selection_mixture", "selection_fir3", "selection_fir9",
    "agc_blocks_input", "agc_blocks_10ms", "agc_blocks_100ms", "agc_blocks_100ms_wrong_alpha",
    "css_overlap_reference", "css_overlap_mixture", "css_overlap_naive", "css_overlap_aligned",
    "wpe_predictable_target", "wpe_predictable_reverberant",
    "wpe_predictable_oracle_inverse", "wpe_predictable_output",
    "aec_dropout_target", "aec_dropout_microphone",
    "aec_dropout_complete_reference_residual", "aec_dropout_missing_reference_residual",
    "gsc_reference", "gsc_array", "gsc_always_adapt", "gsc_gate_frozen",
    "doa_ambiguity_tone", "doa_ambiguity_broadband",
    "dma_calibration_array", "dma_calibration_target", "dma_calibration_mismatch", "dma_calibration_corrected",
    "room_decay_dry", "room_decay_short_drr0", "room_decay_long_drr0", "room_decay_long_drr6",
    "alignment_reference", "alignment_array", "alignment_unaligned", "alignment_aligned",
    "interpolation_ideal_half", "interpolation_linear_half", "interpolation_ideal_one", "interpolation_linear_twice",
    "clock_reference", "clock_array", "clock_index_mean", "clock_oracle_mean",
    "spatial_reference", "spatial_array", "spatial_mic1", "spatial_unaligned", "spatial_aligned",
    "aec_far", "aec_near", "aec_microphone", "aec_frozen", "aec_unfrozen",
    "wpe_dry", "wpe_reverberant", "wpe_output", "separation_source1", "separation_source2",
    "separation_mixture", "separation_recovered1", "separation_recovered2",
    "engineering_reference", "engineering_clipped", "engineering_low_level", "engineering_dropout", "tracking_pan",
    "correlation_reference", "correlation_single", "correlation_independent", "correlation_common",
    "polarity_reference", "polarity_array", "polarity_uncorrected", "polarity_corrected",
    "conditioning_reference", "conditioning_well_input", "conditioning_ill_input",
    "conditioning_well_output", "conditioning_ill_output",
    "nonlinear_reference", "nonlinear_echo", "nonlinear_estimate", "nonlinear_residual",
    "fractional_reference", "fractional_array", "fractional_unaligned", "fractional_aligned",
    "aec_methods_reference", "aec_methods_true_echo", "aec_methods_microphone",
    "aec_methods_nlms_residual",
    "aec_methods_ipnlms_residual", "aec_methods_rls_residual", "aec_methods_kalman_residual",
    "aec_subband_reference", "aec_subband_true_echo", "aec_subband_diagonal_model",
    "aec_subband_missing_cross_terms",
    "spectral_clean", "spectral_noise", "spectral_noisy", "spectral_floor04", "spectral_floor00",
}


EXPECTED_REAL_AUDIO_CHANNELS = {
    "demand_nriver_16ch_10s.wav": 16,
    "demand_nriver_ch01_10s.wav": 1,
    "demand_nriver_mean02_10s.wav": 1,
    "demand_nriver_mean16_10s.wav": 1,
}
EXPECTED_REAL_AUDIO_FILES = set(EXPECTED_REAL_AUDIO_CHANNELS) | {
    "MANIFEST.json", "ATTRIBUTION.txt", "LICENSE.txt", "README.md",
}
EXPECTED_REAL_AUDIO_SOURCE_LINKS = {
    "../core/real_recordings.py": "https://github.com/nanless/microphone-array-signal-processing/blob/main/codes/chapters/ch02/core/real_recordings.py",
    "../examples/prepare_real_recordings.py": "https://github.com/nanless/microphone-array-signal-processing/blob/main/codes/chapters/ch02/examples/prepare_real_recordings.py",
}


def _validate_audio_metadata(value, path="manifest", *, per_channel_rms=False):
    """Check publication metadata types, not the family's signal model.

    This finite field policy is independent of the builder. Boolean controls
    and arbitrary numerical diagnostics retain their own family validators.
    """
    integer_fields = {"schema_version", "sample_rate_hz", "channels", "samples",
                      "samples_per_channel", "frames", "sample_width_bytes"}
    number_fields = {"common_export_gain", "common_gain", "duration_s", "rms",
                     "peak", "quantization_max_abs_error", "quantization_max_abs_error_bound"}

    def numbers(item, location):
        if isinstance(item, list):
            for index, entry in enumerate(item):
                numbers(entry, f"{location}[{index}]")
        else:
            try:
                valid = type(item) in (int, float) and math.isfinite(item)
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError("audio metadata must be a finite non-bool number: " + location)

    if isinstance(value, dict):
        for key, item in value.items():
            location = path + "." + key
            if (key in integer_fields and type(item) is not int
                    and not (key in {"samples", "frames"} and isinstance(item, dict))):
                raise ValueError("audio metadata must be an exact integer: " + location)
            if key in number_fields:
                if key == "rms" and per_channel_rms:
                    if (not isinstance(item, list) or not item
                            or type(value.get("channels")) is not int
                            or len(item) != value["channels"]
                            or any(type(number) not in (int, float) for number in item)):
                        raise ValueError("per-channel RMS must match the WAV channel count: " + location)
                elif type(item) not in (int, float):
                    raise ValueError("audio metadata must be a scalar number: " + location)
                numbers(item, location)
            _validate_audio_metadata(item, location, per_channel_rms=per_channel_rms)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_audio_metadata(item, f"{path}[{index}]", per_channel_rms=per_channel_rms)
    return value


def _read_audio_manifest(path, *, per_channel_rms=False):
    """Read ordinary JSON without duplicate-key or numeric coercion shortcuts."""
    validate_parent_chain(path)
    manifest = strict_json_loads(Path(path).read_bytes())
    if not isinstance(manifest, dict):
        raise ValueError("audio manifest root must be an object")
    return _validate_audio_metadata(manifest, per_channel_rms=per_channel_rms)


def check_real_audio(errors):
    """Independent inventory and PCM/attribution checks for the DEMAND excerpt."""
    import numpy as np
    root = REAL_AUDIO_ROOT
    try:
        for folder in (root, SITE / "real_audio"):
            validate_asset_directory(folder, EXPECTED_REAL_AUDIO_FILES, check=True)
        for name in EXPECTED_REAL_AUDIO_FILES:
            source, published = root / name, SITE / "real_audio" / name
            if (source.is_symlink() or published.is_symlink() or
                    (name != "README.md" and source.read_bytes() != published.read_bytes())):
                fail(errors, f"真实录音站点副本不符：{name}")
        source_readme = (root / "README.md").read_text(encoding="utf-8")
        published_readme = (SITE / "real_audio/README.md").read_text(encoding="utf-8")
        expected_readme = source_readme
        for local, remote in EXPECTED_REAL_AUDIO_SOURCE_LINKS.items():
            token = f"]({local})"
            if source_readme.count(token) != 1:
                fail(errors, f"真实录音源说明缺少唯一源码链接：{local}")
            expected_readme = expected_readme.replace(token, f"]({remote})")
        if published_readme != expected_readme:
            fail(errors, "真实录音发布说明与定向转换后的源文件不符")

        import markdown

        class ReadmeLinks(HTMLParser):
            def __init__(self):
                super().__init__()
                self.hrefs = []

            def handle_starttag(self, tag, attrs):
                if tag == "a":
                    self.hrefs.extend(value for key, value in attrs if key == "href")

        readme_links = ReadmeLinks()
        readme_links.feed(markdown.markdown(published_readme, extensions=["tables"]))
        for href in readme_links.hrefs:
            parsed = urlparse(href)
            if parsed.netloc and not parsed.scheme:
                fail(errors, f"真实录音发布说明含无协议外部链接：{href}")
                continue
            if parsed.scheme:
                if parsed.scheme not in ALLOWED_LINK_SCHEMES:
                    fail(errors, f"真实录音发布说明含不安全链接：{href}")
                continue
            if not parsed.path:
                continue
            target = ((SITE / "real_audio") / unquote(parsed.path)).resolve()
            if (parsed.path.startswith("/") or
                    not target.is_relative_to(SITE.resolve()) or
                    not target.is_file()):
                fail(errors, f"真实录音发布说明含失效的本地链接：{href}")
        attribution = (root / "ATTRIBUTION.txt").read_text(encoding="utf-8")
        for required in ("Joachim Thiemann", "Nobutaka Ito", "Emmanuel Vincent",
                         "1227121", "creativecommons.org/licenses/by-sa/3.0"):
            if required not in attribution:
                fail(errors, f"真实录音署名缺少 {required}")
        if "creativecommons.org/licenses/by-sa/3.0" not in (root / "LICENSE.txt").read_text(encoding="utf-8"):
            fail(errors, "真实录音许可地址缺失")
        manifest = _read_audio_manifest(root / "MANIFEST.json", per_channel_rms=True)
        expected_inputs = {"codes/chapters/ch02/core/real_recordings.py", "codes/chapters/ch02/examples/prepare_real_recordings.py",
                           'codes/chapters/ch00/io_contracts.py'}
        if set(manifest["generator_inputs"]) != expected_inputs:
            fail(errors, "真实录音生成来源清单不符")
        for name in expected_inputs:
            if hashlib.sha256(validate_parent_chain(ROOT / name).read_bytes()).hexdigest() != manifest["generator_inputs"].get(name):
                fail(errors, f"真实录音生成源过期：{name}")
        records = manifest["files"]
        if len(records) != 4 or {r["file"] for r in records} != set(EXPECTED_REAL_AUDIO_CHANNELS):
            raise ValueError("真实录音清单不符")
        for record in records:
            path = root / record["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                fail(errors, f"真实录音摘要不符：{path.name}")
            with wave.open(str(path), "rb") as wav:
                if (wav.getframerate(), wav.getsampwidth(), wav.getcomptype(), wav.getnframes(), wav.getnchannels()) != (
                        16000, 2, "NONE", 160000, EXPECTED_REAL_AUDIO_CHANNELS[path.name]):
                    raise ValueError(f"真实录音 PCM 格式不符：{path.name}")
                pcm = np.frombuffer(wav.readframes(160000), dtype="<i2").astype(float)
                pcm = pcm.reshape(160000, wav.getnchannels()) / 32768
            if (record["sample_rate_hz"], record["samples"], record["channels"], record["duration_s"]) != (
                    16000, 160000, EXPECTED_REAL_AUDIO_CHANNELS[path.name], 10):
                fail(errors, f"真实录音尺寸元数据不符：{path.name}")
            rms = np.asarray(record["rms"])
            if (rms.shape != (pcm.shape[1],) or not np.all(np.isfinite(rms)) or
                    not np.allclose(rms, np.sqrt(np.mean(pcm**2, axis=0)), rtol=0, atol=1e-14)):
                fail(errors, f"真实录音 RMS 不符：{path.name}")
            if not np.isfinite(record["peak"]) or abs(record["peak"] - np.max(np.abs(pcm))) > 1e-15:
                fail(errors, f"真实录音峰值不符：{path.name}")
            if not np.isfinite(record["common_export_gain"]) or record["common_export_gain"] != 1:
                fail(errors, f"真实录音共同增益应为 1：{path.name}")
        parser = VisibleMediaParser()
        parser.feed((SITE / "research/05_exercises_and_audio.html").read_text())
        allowed_audio_roots = ("../audio/", "../real_audio/", "../room_audio/",
                               "../gss_audio/", "../moving_audio/", "../tracking_audio/", "../binaural_audio/", "../spectral_audio/", "../stft_audio/", "../geometry_audio/", "../focus_audio/", "../derivative_audio/", "../apa_audio/", "../mint_audio/", "../mask_audio/", "../noise_audio/", "../scenario_audio/", "../weighted_audio/", "../response_audio/", "../imaging_audio/", "../distributed_audio/")
        if any(not (p.get("src") or "").startswith(allowed_audio_roots)
               for p in parser.items):
            fail(errors, "未知试听控件来源")
        _check_visible_audio(
            SITE / "research/05_exercises_and_audio.html", "../", "real_audio",
            {name for name, channels in EXPECTED_REAL_AUDIO_CHANNELS.items() if channels == 1}, set(),
        )
    except Exception as exc:
        fail(errors, f"真实录音检查失败：{exc}")


def check_room_audio(errors):
    """Independently verify the 18 generated room WAVs and staged site assets."""
    import numpy as np
    root, published = ROOM_AUDIO_ROOT, SITE / "room_audio"
    try:
        from codes.chapters.appendix_b.examples.check_room_assets import check_assets
        check_assets(root)
        check_assets(published)
        manifest = _read_audio_manifest(root / "MANIFEST.json")
        records = manifest["files"]
        names = [record["file"] for record in records]
        cases = {record["case"] for record in records}
        roles = {(record["case"], record["role"]) for record in records}
        expected = set(names) | {"MANIFEST.json", "ROOM_RESULTS.png", "RESULTS.json"}
        if (len(records) != 18 or len(set(names)) != 18 or len(cases) != 6 or
                roles != {(case, role) for case in cases
                          for role in ("source", "direct", "full")} or
                any(not re.fullmatch(r"[a-z0-9_]+\.wav", name) for name in names)):
            raise ValueError("房间合成音频必须是六组三联、共18个安全文件名")
        for folder in (root, published):
            validate_asset_directory(folder, expected, check=True)
        if (manifest["provenance"] != "mathematically synthesized white Gaussian noise, not recorded speech"
                or manifest["pyroomacoustics_version"] != "0.10.0"
                or manifest["sample_rate_hz"] != 16000
                or manifest["max_order"] != 40
                or not 0 < manifest["common_gain"] < 1):
            raise ValueError("房间样本来源、版本或共同增益不符")
        for name in expected:
            source, copy = root / name, published / name
            if source.is_symlink() or copy.is_symlink() or source.read_bytes() != copy.read_bytes():
                raise ValueError(f"房间站点副本不符：{name}")
        png = (root / "ROOM_RESULTS.png").read_bytes()
        if (not png.startswith(b"\x89PNG\r\n\x1a\n") or
                int.from_bytes(png[16:20], "big") < 1200 or
                int.from_bytes(png[20:24], "big") < 1000):
            raise ValueError("房间结果图格式或尺寸不符")
        report = strict_json_loads((root / "RESULTS.json").read_bytes())
        if (report.get("schema_version") != 1 or
                report.get("status") != "pyroomacoustics_simulation_executed" or
                report.get("pyroomacoustics_version_installed") != "0.10.0" or
                report.get("actual_max_order") != 40 or
                report.get("generator", {}).get("path") != "codes/chapters/appendix_b/examples/room_srp_exercise.py" or
                report["generator"].get("sha256") != hashlib.sha256(
                    (ROOT / "codes/chapters/appendix_b/examples/room_srp_exercise.py").read_bytes()).hexdigest() or
                report.get("assets", {}).get("figure", {}).get("sha256") != hashlib.sha256(png).hexdigest() or
                report["assets"].get("audio_manifest", {}).get("sha256") != hashlib.sha256(
                    (root / "MANIFEST.json").read_bytes()).hexdigest()):
            raise ValueError("房间结果报告的版本、源或资产摘要不符")
        published_rows = (
            ("fixed_near_left", 1.000, -30.000, 0.546, 5.755, -29, 1.000),
            ("fixed_near_right", 1.000, 30.000, 0.546, 5.755, 29, 1.000),
            ("fixed_far_left", 2.500, -30.000, 0.553, -2.113, -32, 2.000),
            ("fixed_far_right", 2.500, 30.000, 0.553, -2.113, 32, 2.000),
            ("seeded_1", 1.099, -41.244, 0.555, 4.866, -40, 1.244),
            ("seeded_2", 1.973, 2.396, 0.563, -0.003, 3, 0.604),
        )
        if len(report.get("results", [])) != len(published_rows):
            raise ValueError("房间结果报告必须有六个位置")
        for row, baseline in zip(report["results"], published_rows):
            actual = (row["name"], row["distance_m"], row["true_azimuth_deg"],
                      row["t60_s_median"], row["drr_db_median"],
                      row["estimated_azimuth_deg"], row["absolute_doa_error_deg"])
            if actual[0] != baseline[0] or any(round(value, 3) != reference
                                               for value, reference in zip(actual[1:], baseline[1:])):
                raise ValueError(f"房间结果报告与已发布三位小数表不符：{row['name']}")
        chapter = (CHAPTERS / "13_appendix-guide.md").read_text(encoding="utf-8")
        for label, distance, azimuth, t60, drr, estimate, error in published_rows:
            # The publication table is checked by its six rounded numeric columns,
            # not by position-only labels that might be edited independently.
            formatted = (f"{distance:.3f}", f"{azimuth:+.3f}" if azimuth > 0 else f"{azimuth:.3f}",
                         f"{t60:.3f}", f"{drr:.3f}", f"{estimate:+d}" if estimate > 0 else str(estimate),
                         f"{error:.3f}")
            if not any(all(value in cell.replace('−', '-') for value, cell in zip(formatted, line.split('|')[2:8]))
                       for line in chapter.splitlines() if line.startswith('| ') and len(line.split('|')) >= 9):
                raise ValueError(f"房间正文表与结果报告不符：{label}")
        for record in records:
            path = root / record["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"房间 WAV 摘要不符：{path.name}")
            channels = 1 if record["role"] == "source" else 4
            with wave.open(str(path), "rb") as wav:
                if (wav.getframerate(), wav.getsampwidth(), wav.getcomptype(),
                        wav.getnchannels(), wav.getnframes()) != (
                            16000, 2, "NONE", channels, record["frames"]):
                    raise ValueError(f"房间 WAV 格式不符：{path.name}")
                samples = wav.readframes(wav.getnframes())
            pcm = np.frombuffer(samples, dtype="<i2").astype(np.int32)
            if (record["channels"] != channels or record["frames"] < 16000 or
                    np.max(np.abs(pcm)) > 26215):
                raise ValueError(f"房间 WAV 通道、长度或峰值不符：{path.name}")
        appendix = (SITE / "13_appendix-guide.html").read_text(encoding="utf-8")
        if ('src="room_audio/ROOM_RESULTS.png"' not in appendix or
                'href="room_audio/MANIFEST.json"' not in appendix):
            raise ValueError("附录站点未发布房间图或音频清单")
    except Exception as exc:
        fail(errors, f"房间合成样本检查失败：{exc}")


def check_moving_audio(errors):
    """独立核对移动声源三组 PCM、真值清单和网页副本。"""
    source, published = MOVING_AUDIO_ROOT, SITE / "moving_audio"
    expected_channels = {"source.wav": 1, "static_array.wav": 2, "moving_array.wav": 2}
    expected = set(expected_channels) | {"MANIFEST.json"}
    try:
        for folder in (source, published):
            validate_asset_directory(folder, expected, check=True)
        manifest = _read_audio_manifest(source / "MANIFEST.json")
        if (set(manifest["files"]) != set(expected_channels)
                or manifest["sample_rate_hz"] != 16000
                or "free field" not in manifest["model"]
                or len(manifest["truth"]["time_seconds"]) < 100):
            raise ValueError("清单模型、真值或样本集合不符")
        source_paths = {"codes/chapters/ch09/examples/moving_source_audio.py",
                        "codes/chapters/ch09/core/moving_source.py",
                        "codes/chapters/ch00/core/audio_samples.py",
                        "codes/chapters/ch02/core/conventions.py"}
        if set(manifest.get("source_sha256", {})) != source_paths:
            raise ValueError("移动声源生成源码清单不完整")
        for name, digest in manifest["source_sha256"].items():
            if hashlib.sha256(validate_parent_chain(ROOT / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"移动声源生成源码已变化：{name}")
        for folder in (source, published):
            validate_asset_directory(folder, expected, check=True)
        for name in expected:
            original, copy = source / name, published / name
            if original.is_symlink() or copy.is_symlink() or original.read_bytes() != copy.read_bytes():
                raise ValueError(f"网页副本与源文件不一致：{name}")
        for name, channels in expected_channels.items():
            record = manifest["files"][name]
            path = source / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"WAV 摘要不符：{name}")
            with wave.open(str(path), "rb") as wav:
                if (wav.getframerate(), wav.getsampwidth(), wav.getnchannels(),
                        wav.getnframes(), wav.getcomptype()) != (
                            16000, 2, channels, record["samples_per_channel"], "NONE"):
                    raise ValueError(f"WAV 格式不符：{name}")
        from codes.chapters.ch09.examples.moving_source_audio import generate
        generate(source, check=True)
        for page, prefix in ((SITE / "09_source-tracking.html", ""),
                             (SITE / "research/05_exercises_and_audio.html", "../")):
            _check_visible_audio(page, prefix, "moving_audio", set(expected_channels), {"MANIFEST.json"})
    except Exception as exc:
        fail(errors, f"移动声源合成样本检查失败：{exc}")



def check_binaural_audio(errors):
    """Independently verify cue files, their actual source hashes and players."""
    source, published = BINAURAL_AUDIO_ROOT, SITE / "binaural_audio"
    wav_names = {"reference.wav", "itd_only.wav", "ild_only.wav",
                 "consistent.wav", "conflicting.wav"}
    expected = wav_names | {"MANIFEST.json"}
    required_sources = {"codes/chapters/ch01/core/binaural_cues.py",
                        "codes/chapters/ch01/examples/generate_binaural_cues.py",
                        "codes/chapters/ch00/core/audio_samples.py",
                        'codes/chapters/ch00/io_contracts.py'}
    try:
        for folder in (source, published):
            validate_asset_directory(folder, expected, check=True)
        for name in expected:
            if (source / name).read_bytes() != (published / name).read_bytes():
                raise ValueError(f"双耳线索网页副本不同：{name}")
        manifest = _read_audio_manifest(source / "MANIFEST.json")
        if (set(manifest["files"]) != wav_names or manifest["sample_rate_hz"] != 16000
                or manifest["common_export_gain"] != 1
                or set(manifest["source_sha256"]) != required_sources):
            raise ValueError("双耳线索清单参数或真实源集合不符")
        for name, digest in manifest["source_sha256"].items():
            if hashlib.sha256(validate_parent_chain(ROOT / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"双耳线索生成源码已变化：{name}")
        for name in wav_names:
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
        for page, prefix in ((SITE / "01_problem-definition.html", ""),
                             (SITE / "research/05_exercises_and_audio.html", "../")):
            _check_visible_audio(page, prefix, "binaural_audio", wav_names, {"MANIFEST.json"})
    except Exception as exc:
        fail(errors, f"双耳线索实验检查失败：{exc}")


def check_stft_audio(errors):
    """Independently verify convolution files, their actual source hashes and players."""
    source, published = STFT_AUDIO_ROOT, SITE / "stft_audio"
    wav_names = {"stft_roundtrip.wav", "full_convolution.wav", "framewise_mtf.wav"}
    expected = wav_names | {"MANIFEST.json"}
    required_sources = {"codes/chapters/ch02/core/stft_convolution.py",
                        "codes/chapters/ch02/examples/generate_stft_convolution.py",
                        "codes/chapters/ch02/core/spectral.py",
                        "codes/chapters/ch02/core/conventions.py",
                        "codes/chapters/ch00/core/audio_samples.py",
                        'codes/chapters/ch00/io_contracts.py'}
    try:
        for folder in (source, published):
            validate_asset_directory(folder, expected, check=True)
        for name in expected:
            if (source / name).read_bytes() != (published / name).read_bytes():
                raise ValueError(f"STFT卷积网页副本不同：{name}")
        manifest = _read_audio_manifest(source / "MANIFEST.json")
        if (set(manifest["files"]) != wav_names or manifest["sample_rate_hz"] != 16000
                or manifest["common_export_gain"] != 1
                or set(manifest["source_sha256"]) != required_sources):
            raise ValueError("STFT卷积清单参数或真实源集合不符")
        for name, digest in manifest["source_sha256"].items():
            if hashlib.sha256(validate_parent_chain(ROOT / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"STFT卷积生成源码已变化：{name}")
        for name in wav_names:
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
        for page, prefix in ((SITE / "02_basics-signal-model.html", ""),
                             (SITE / "research/05_exercises_and_audio.html", "../")):
            _check_visible_audio(page, prefix, "stft_audio", wav_names, {"MANIFEST.json"})
    except Exception as exc:
        fail(errors, f"STFT卷积实验检查失败：{exc}")


def check_geometry_audio(errors):
    """Independently verify geometry files, their actual source hashes and players."""
    source, published = GEOMETRY_AUDIO_ROOT, SITE / "geometry_audio"
    wav_names = {"geometry_reference.wav", "geometry_u.wav", "geometry_v.wav"}
    expected = wav_names | {"MANIFEST.json"}
    required_sources = {"codes/chapters/ch03/core/geometry_audio.py",
                        "codes/chapters/ch03/examples/generate_geometry_audio.py",
                        "codes/chapters/ch03/core/geometry.py",
                        "codes/chapters/ch02/core/conventions.py",
                        "codes/chapters/ch00/core/audio_samples.py",
                        'codes/chapters/ch00/io_contracts.py'}
    try:
        for folder in (source, published):
            validate_asset_directory(folder, expected, check=True)
        for name in expected:
            if (source / name).read_bytes() != (published / name).read_bytes():
                raise ValueError(f"多频几何网页副本不同：{name}")
        manifest = _read_audio_manifest(source / "MANIFEST.json")
        if (set(manifest["files"]) != wav_names or manifest["sample_rate_hz"] != 32000
                or manifest["common_export_gain"] != 1
                or set(manifest["source_sha256"]) != required_sources):
            raise ValueError("多频几何清单参数或真实源集合不符")
        for name, digest in manifest["source_sha256"].items():
            if hashlib.sha256(validate_parent_chain(ROOT / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"多频几何生成源码已变化：{name}")
        for name in wav_names:
            path, record = source / name, manifest["files"][name]
            if (record["channels"] != (1 if name == "geometry_reference.wav" else 6) or record["samples_per_channel"] != 64000
                    or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]):
                raise ValueError(f"多频几何摘要或参数不符：{name}")
            with wave.open(str(path), "rb") as wav:
                if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(),
                        wav.getnframes(), wav.getcomptype()) != (32000, 1 if name == "geometry_reference.wav" else 6, 2, 64000, "NONE"):
                    raise ValueError(f"多频几何PCM格式不符：{name}")
                if len(wav.readframes(64000)) != 64000 * (1 if name == "geometry_reference.wav" else 6) * 2:
                    raise ValueError(f"多频几何PCM数据截断：{name}")
        for page, prefix in ((SITE / "03_array-geometry.html", ""),
                             (SITE / "research/05_exercises_and_audio.html", "../")):
            _check_visible_audio(page, prefix, "geometry_audio", wav_names, {"MANIFEST.json"})
    except Exception as exc:
        fail(errors, f"多频几何实验检查失败：{exc}")


def check_focus_audio(errors):
    """Independently verify known-unitary focusing files, their actual source hashes and players."""
    source, published = FOCUS_AUDIO_ROOT, SITE / "focus_audio"
    wav_names = {"focus_reference.wav", "focus_delayed_source.wav", "focus_array.wav", "focus_known_focused.wav"}
    expected = wav_names | {"MANIFEST.json"}
    required_sources = {"codes/chapters/ch04/core/focus_audio.py",
                        "codes/chapters/ch04/examples/generate_focus_audio.py",
                        "codes/chapters/ch03/core/geometry.py",
                        "codes/chapters/ch02/core/conventions.py",
                        "codes/chapters/ch00/core/audio_samples.py",
                        'codes/chapters/ch00/io_contracts.py'}
    try:
        for folder in (source, published):
            validate_asset_directory(folder, expected, check=True)
        for name in expected:
            if (source / name).read_bytes() != (published / name).read_bytes():
                raise ValueError(f"已知酉聚焦网页副本不同：{name}")
        manifest = _read_audio_manifest(source / "MANIFEST.json")
        if (set(manifest["files"]) != wav_names or manifest["sample_rate_hz"] != 16000
                or manifest["common_export_gain"] != 1
                or set(manifest["source_sha256"]) != required_sources):
            raise ValueError("已知酉聚焦清单参数或真实源集合不符")
        for name, digest in manifest["source_sha256"].items():
            if hashlib.sha256(validate_parent_chain(ROOT / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"已知酉聚焦生成源码已变化：{name}")
        for name in wav_names:
            path, record = source / name, manifest["files"][name]
            if (record["channels"] != (1 if name in {"focus_reference.wav", "focus_delayed_source.wav"} else 4) or record["samples_per_channel"] != 32024
                    or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]):
                raise ValueError(f"已知酉聚焦摘要或参数不符：{name}")
            with wave.open(str(path), "rb") as wav:
                if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(),
                        wav.getnframes(), wav.getcomptype()) != (16000, 1 if name in {"focus_reference.wav", "focus_delayed_source.wav"} else 4, 2, 32024, "NONE"):
                    raise ValueError(f"已知酉聚焦PCM格式不符：{name}")
                if len(wav.readframes(32024)) != 32024 * (1 if name in {"focus_reference.wav", "focus_delayed_source.wav"} else 4) * 2:
                    raise ValueError(f"已知酉聚焦PCM数据截断：{name}")
        for page, prefix in ((SITE / "04_doa-estimation.html", ""),
                             (SITE / "research/05_exercises_and_audio.html", "../")):
            _check_visible_audio(page, prefix, "focus_audio", wav_names, {"MANIFEST.json"})
    except Exception as exc:
        fail(errors, f"已知酉聚焦实验检查失败：{exc}")


def check_derivative_audio(errors):
    """Independently verify derivative-constraint files, their actual source hashes and players."""
    source, published = DERIVATIVE_AUDIO_ROOT, SITE / "derivative_audio"
    wav_names = {"derivative_reference.wav", "derivative_single.wav", "derivative_array.wav", "derivative_constrained.wav"}
    expected = wav_names | {"MANIFEST.json"}
    required_sources = {"codes/chapters/ch05/core/derivative_audio.py",
                        "codes/chapters/ch05/examples/generate_derivative_audio.py",
                        "codes/chapters/ch05/core/beamforming.py",
                        "codes/chapters/ch04/core/covariance.py",
                        "codes/chapters/ch03/core/geometry.py",
                        "codes/chapters/ch02/core/conventions.py",
                        "codes/chapters/ch00/core/audio_samples.py",
                        'codes/chapters/ch00/io_contracts.py'}
    try:
        for folder in (source, published):
            validate_asset_directory(folder, expected, check=True)
        for name in expected:
            if (source / name).read_bytes() != (published / name).read_bytes():
                raise ValueError(f"导数约束网页副本不同：{name}")
        manifest = _read_audio_manifest(source / "MANIFEST.json")
        if (set(manifest["files"]) != wav_names or manifest["sample_rate_hz"] != 16000 or manifest["samples_per_channel"] != 32002
                or manifest["common_export_gain"] != 1
                or set(manifest["source_sha256"]) != required_sources):
            raise ValueError("导数约束清单参数或真实源集合不符")
        for name, digest in manifest["source_sha256"].items():
            if hashlib.sha256(validate_parent_chain(ROOT / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"导数约束生成源码已变化：{name}")
        for name in wav_names:
            path, record = source / name, manifest["files"][name]
            if (record["sample_rate_hz"] != 16000 or record["channels"] != (3 if name == "derivative_array.wav" else 1) or record["samples_per_channel"] != 32002
                    or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]):
                raise ValueError(f"导数约束摘要或参数不符：{name}")
            with wave.open(str(path), "rb") as wav:
                if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(),
                        wav.getnframes(), wav.getcomptype()) != (16000, 3 if name == "derivative_array.wav" else 1, 2, 32002, "NONE"):
                    raise ValueError(f"导数约束PCM格式不符：{name}")
                if len(wav.readframes(32002)) != 32002 * (3 if name == "derivative_array.wav" else 1) * 2:
                    raise ValueError(f"导数约束PCM数据截断：{name}")
        for page, prefix in ((SITE / "05_beamforming.html", ""),
                             (SITE / "research/05_exercises_and_audio.html", "../")):
            _check_visible_audio(page, prefix, "derivative_audio", wav_names, {"MANIFEST.json"})
    except Exception as exc:
        fail(errors, f"导数约束实验检查失败：{exc}")


def check_apa_audio(errors):
    """Independent PCM integer scoring and visible controls for E06-39."""
    import math
    import struct
    source, published = APA_AUDIO_ROOT, SITE / "apa_audio"
    wav_names = {"apa_"+key+".wav" for key in ("reference", "true_echo", "microphone", "nlms_residual", "apa2_residual", "apa4_residual")}
    expected_files = wav_names | {"MANIFEST.json"}
    required_sources = {"codes/chapters/ch06/core/apa_audio.py",
                        "codes/chapters/ch06/examples/generate_apa_audio.py",
                        "codes/chapters/ch06/core/aec.py",
                        "codes/chapters/ch06/core/aec_numeric.py",
                        "codes/chapters/ch06/core/aec_affine_projection.py",
                        "codes/chapters/ch02/core/conventions.py",
                        "codes/chapters/ch00/core/audio_samples.py",
                        'codes/chapters/ch00/io_contracts.py'}
    try:
        for folder in (source, published):
            try:
                validate_asset_directory(folder, expected_files, check=True)
            except ValueError as exc:
                raise ValueError("APA目录须恰含六普通WAV和清单：" + str(exc)) from exc
        for name in expected_files:
            if (source/name).read_bytes() != (published/name).read_bytes():
                raise ValueError("APA网页副本不同："+name)
        manifest = _read_audio_manifest(source / "MANIFEST.json")
        if (set(manifest['files']) != wav_names or manifest['sample_rate_hz'] != 16000
                or manifest['samples_per_channel'] != 32013 or manifest['common_export_gain'] != 1
                or set(manifest['source_sha256']) != required_sources):
            raise ValueError("APA清单参数或真实源集合不同")
        for name, digest in manifest['source_sha256'].items():
            if hashlib.sha256(validate_parent_chain(ROOT/name).read_bytes()).hexdigest() != digest:
                raise ValueError("APA真实源码摘要过期："+name)
        measures = manifest['pcm_measurements']
        if measures['holdout_interval_samples'] != [24000, 32000] or measures['pcm_decode_divisor'] != 32768:
            raise ValueError("APA评分窗或PCM解码分母不同")
        actual_sums = {}
        for name in wav_names:
            path, record = source/name, manifest['files'][name]
            if (record['sample_rate_hz'], record['channels'], record['samples_per_channel']) != (16000, 1, 32013) or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
                raise ValueError("APA文件参数或摘要不同："+name)
            with wave.open(str(path), 'rb') as reader:
                if (reader.getframerate(), reader.getnchannels(), reader.getsampwidth(), reader.getnframes(), reader.getcomptype()) != (16000, 1, 2, 32013, 'NONE'):
                    raise ValueError("APA实际PCM格式不同："+name)
                raw = reader.readframes(32013)
            if len(raw) != 64026:
                raise ValueError("APA实际PCM截断："+name)
            integers = struct.unpack('<32013h', raw)
            squared = sum(v*v for v in integers[24000:32000])
            key = name[4:-4]
            actual_sums[key] = squared
            score = measures['powers'][key]
            if (score['integer_squared_sum'] != squared or score['sample_denominator'] != 8000
                    or score['mean_square'] != squared/(32768**2*8000)):
                raise ValueError("APA实际整数评分不符："+name)
        if actual_sums['microphone'] <= 0:
            raise ValueError("APA麦克风评分分母须严格正")
        for key in ('nlms_residual', 'apa2_residual', 'apa4_residual'):
            ratio = measures['ratios'][key]
            denominator = actual_sums[key]
            if ratio['integer_numerator'] != actual_sums['microphone'] or ratio['integer_denominator'] != denominator or ratio['zero_output'] != (denominator == 0):
                raise ValueError("APA实际功率比分母不符")
            expected_db = 10*math.log10(actual_sums['microphone']/denominator) if denominator else None
            reported = ratio['microphone_to_residual_total_power_ratio_db']
            if ((expected_db is None and reported is not None)
                    or (expected_db is not None and
                        (isinstance(reported, bool) or not isinstance(reported, (int, float))
                         or not math.isfinite(reported) or abs(expected_db-reported) > 1e-12))):
                raise ValueError("APA实际总功率比不同")
        for page, prefix in ((SITE / "06_aec.html", ""),
                             (SITE / "research/05_exercises_and_audio.html", "../")):
            _check_visible_audio(page, prefix, "apa_audio", wav_names, {"MANIFEST.json"})
    except Exception as exc:
        fail(errors, f"APA实验检查失败：{exc}")


def check_mint_audio(errors):
    """Independent PCM integer scores, source hashes and usable media controls."""
    import math
    import struct
    source, published = MINT_AUDIO_ROOT, SITE / 'mint_audio'
    names = {'reference': ('mint_reference.wav', 1), 'well_array': ('mint_well_array.wav', 2),
             'near_array': ('mint_near_array.wav', 2), 'well_exact': ('mint_well_exact.wav', 1),
             'near_exact': ('mint_near_exact.wav', 1), 'near_regularized': ('mint_near_regularized.wav', 1)}
    wav_names = {name for name, _ in names.values()}
    expected = wav_names | {'MANIFEST.json'}
    sources = {'codes/chapters/ch07/core/mint_teaching.py',
               'codes/chapters/ch07/examples/mint_teaching_demo.py',
               'codes/chapters/ch02/core/conventions.py', 'codes/chapters/ch00/core/audio_samples.py',
               'codes/chapters/ch00/io_contracts.py'}
    try:
        for folder in (source, published):
            validate_asset_directory(folder, expected, check=True)
        manifest = _read_audio_manifest(source / 'MANIFEST.json')
        if (set(manifest['source_sha256']) != sources or set(manifest['files']) != wav_names or
                set(manifest['samples']) != set(names) or type(manifest['sample_rate_hz']) is not int or
                manifest['sample_rate_hz'] != 16000 or type(manifest['samples_per_channel']) is not int or
                manifest['samples_per_channel'] != 32512 or type(manifest['common_export_gain']) not in (int, float) or
                manifest['common_export_gain'] != 1):
            raise ValueError('MINT来源/文件集合或共同格式不符')
        for name in sources:
            if hashlib.sha256(validate_parent_chain(ROOT/name).read_bytes()).hexdigest() != manifest['source_sha256'][name]:
                raise ValueError('MINT真实源摘要过期：'+name)
        from codes.chapters.ch07.examples.mint_teaching_demo import check_assets
        # This read-only replay binds every float field to its source; the
        # independent integer scoring below checks the actual stored PCM.
        check_assets(source, replay=True)
        for name in expected:
            if (source/name).read_bytes() != (published/name).read_bytes():
                raise ValueError('MINT发布副本不同：'+name)
        decoded = {}
        for key, (filename, channels) in names.items():
            path = source/filename
            record = manifest['files'][filename]
            if (record['sample_rate_hz'], record['channels'], record['samples_per_channel']) != (16000, channels, 32512):
                raise ValueError('MINT清单尺寸不同：'+filename)
            if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
                raise ValueError('MINT真实WAV摘要不同：'+filename)
            with wave.open(str(path), 'rb') as wav:
                if (wav.getframerate(), wav.getnchannels(), wav.getnframes(), wav.getsampwidth(), wav.getcomptype()) != (16000, channels, 32512, 2, 'NONE'):
                    raise ValueError('MINT实际PCM格式不同：'+filename)
                raw = wav.readframes(32512)
            values = struct.unpack('<'+'h'*(32512*channels), raw)
            decoded[key] = [values[channel::channels] for channel in range(channels)]
        truth = decoded['reference'][0][2400:29600]
        denominator = sum(v*v for v in truth)
        if denominator <= 0:
            raise ValueError('MINT参考评分能量必须为正')
        for key, (filename, channels) in names.items():
            sample = manifest['samples'][key]
            scores = sample['pcm_measurements']
            error_sums = [sum((v-r)**2 for v, r in zip(channel[2400:29600], truth)) for channel in decoded[key]]
            cross = [sum(v*r for v, r in zip(channel[2400:29600], truth)) for channel in decoded[key]]
            tails = [sum(v*v for v in channel[32000:32512]) for channel in decoded[key]]
            expected_ints = {'integer_reference_squared_sum': denominator,
                             'sample_denominator_per_channel': 27200, 'tail_samples_per_channel': 512,
                             'pcm_decode_divisor': 32768}
            if any(type(scores[k]) is not int or scores[k] != v for k, v in expected_ints.items()):
                raise ValueError('MINT整数分母不同：'+filename)
            for field, actual in (('integer_error_squared_sum_per_channel', error_sums),
                                  ('integer_output_reference_cross_sum_per_channel', cross),
                                  ('integer_tail_squared_sum_per_channel', tails)):
                if (not isinstance(scores[field], list) or len(scores[field]) != channels or
                        any(type(v) is not int for v in scores[field]) or scores[field] != actual):
                    raise ValueError('MINT真实PCM整数统计不同：'+filename)
            for field, actual in (
                    ('total_reference_mse_per_channel', [v/(32768**2*27200) for v in error_sums]),
                    ('relative_squared_reference_error_per_channel', [v/denominator for v in error_sums]),
                    ('projection_gain_per_channel', [v/denominator for v in cross]),
                    ('tail_mean_square_per_channel', [v/(32768**2*512) for v in tails])):
                recorded = scores[field]
                if (not isinstance(recorded, list) or len(recorded) != channels or
                        any(type(v) not in (int, float) or not math.isfinite(v) or
                            not math.isclose(v, a, rel_tol=1e-13, abs_tol=0) for v, a in zip(recorded, actual))):
                    raise ValueError('MINT实际PCM浮点评分不同：'+filename)
            error = sample['quantization_max_abs_error']
            if (sample['file'] != filename or scores['scoring_interval_samples'] != [2400, 29600] or
                    scores['tail_interval_samples'] != [32000, 32512] or
                    type(error) not in (int, float) or not math.isfinite(error) or not 0 <= error <= 1/65536):
                raise ValueError('MINT评分区域或量化边界不同：'+filename)
        for page, prefix in ((SITE / "07_wpe-dereverberation.html", ""),
                             (SITE / "research/05_exercises_and_audio.html", "../")):
            _check_visible_audio(page, prefix, "mint_audio", wav_names, {"MANIFEST.json"})
    except Exception as exc:
        fail(errors, f'MINT实验检查失败：{exc}')


def check_tracking_audio(errors):
    """Verify published PCM, regeneration provenance and real observation controls."""
    source, published = TRACKING_AUDIO_ROOT, SITE/"tracking_audio"
    expected = {"source.wav","array_noisy.wav","MANIFEST.json"}
    try:
        for folder in (source, published):
            validate_asset_directory(folder, expected, check=True)
        for name in expected:
            if (source/name).read_bytes() != (published/name).read_bytes():
                raise ValueError(f"追踪音频网页副本不同：{name}")
        manifest = _read_audio_manifest(source / "MANIFEST.json")
        required_sources = {
            'codes/chapters/ch09/examples/chapter09_tracking_audio.py',
            'codes/chapters/ch09/core/tracking_audio.py', 'codes/chapters/ch09/core/moving_source.py',
            'codes/chapters/ch09/core/tracking.py', 'codes/chapters/ch04/core/doa.py',
            'codes/chapters/ch04/core/covariance.py', 'codes/chapters/ch02/core/conventions.py',
            'codes/chapters/ch00/core/audio_samples.py'}
        if set(manifest.get('source_sha256', {})) != required_sources:
            raise ValueError('追踪音频真实生成源集合不完整')
        for name,digest in manifest['source_sha256'].items():
            if hashlib.sha256(validate_parent_chain(ROOT/name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"追踪音频生成源码已变化：{name}")
        for name,channels in {"source.wav":1,"array_noisy.wav":2}.items():
            path = source/name
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['files'][name]['sha256']:
                raise ValueError(f"追踪音频摘要不符：{name}")
            with wave.open(str(path),'rb') as wav:
                if (wav.getframerate(),wav.getnchannels(),wav.getsampwidth(),
                        wav.getnframes(),wav.getcomptype()) != (16000,channels,2,32000,'NONE'):
                    raise ValueError(f"追踪音频PCM格式不符：{name}")
        from codes.chapters.ch09.examples.chapter09_tracking_audio import generate
        generate(source, check=True)
        _check_tracking_frame_arithmetic(manifest, source/'array_noisy.wav')
        for page, prefix in ((SITE/'09_source-tracking.html', ''),
                             (SITE/'research/05_exercises_and_audio.html', '../')):
            _check_visible_audio(page, prefix, 'tracking_audio', expected-{'MANIFEST.json'}, {'MANIFEST.json'})
    except Exception as exc:
        fail(errors,f"PCM观测追踪实验检查失败：{exc}")


def _check_tracking_frame_arithmetic(manifest, array_path):
    """Scalar, standard-library check independent of the KF/GCC array kernels."""
    import math
    import struct
    with wave.open(str(array_path), 'rb') as wav:
        pcm = struct.unpack('<64000h', wav.readframes(32000))

    def close(actual, expected, label, *, tolerance=2e-12):
        if (type(actual) not in (int, float) or not math.isfinite(actual)
                or not math.isclose(actual, expected, rel_tol=2e-13, abs_tol=tolerance)):
            raise ValueError('追踪音频独立标量复算不符：'+label)

    for chain in ('float_analysis', 'pcm_analysis'):
        frames, scores = manifest[chain]['frames'], manifest[chain]['scores']
        if any(not isinstance(v, list) or len(v) != 197 for v in frames.values()):
            raise ValueError('追踪音频逐帧结果长度不符')
        valid, initialized, last = [], [], None
        for i in range(197):
            start, center = 160*i, (160*i+255.5)/16000
            if type(frames['start_sample'][i]) is not int or frames['start_sample'][i] != start:
                raise ValueError('追踪音频帧起点必须为真实整数样本编号')
            close(frames['state_time_s'][i], center, '状态时刻')
            close(frames['available_time_s'][i], (start+512)/16000, '可用时刻')
            # Retarded-time quadratic solved for propagation delay; no Newton kernel.
            px, py, vx, speed = -.8+.8*center, 1.5, .8, 343.
            a, b, r2 = speed**2-vx**2, 2*px*vx, px*px+py*py
            delay = 2*r2/(b+math.sqrt(b*b+4*a*r2))
            emission = center-delay
            source_x = -.8+.8*emission
            close(frames['center_emission_time_s'][i], emission, '发射时刻')
            close(frames['truth_angle_deg'][i], math.degrees(math.atan2(source_x, py)), '方向真值')
            tau = (math.hypot(source_x-.05, py)-math.hypot(source_x+.05, py))/speed*16000
            close(frames['truth_tau10_samples'][i], tau, '同发射事件时差')
            flag = frames['observation_valid'][i]
            if type(flag) is not bool:
                raise ValueError('追踪音频观测标记必须为bool')
            valid.append(flag)
            if flag:
                close(frames['measurement_time_s'][i], center, '观测时刻')
                last = center
                close(frames['observation_angle_deg'][i], math.degrees(math.asin(
                    -343*frames['observation_tau10_samples'][i]/16000/.1)), '时差换算方向')
            elif any(frames[key][i] is not None for key in
                     ('measurement_time_s', 'observation_angle_deg', 'observation_tau10_samples')):
                raise ValueError('缺测观测字段必须为null')
            if last is None:
                if frames['last_valid_measurement_time_s'][i] is not None:
                    raise ValueError('初始化之前不能有最近观测时刻')
            else:
                close(frames['last_valid_measurement_time_s'][i], last, '最近有效观测时刻')
            has_state = frames['filtered_angle_deg'][i] is not None
            initialized.append(has_state)
            expected_phase = ('uninitialized' if not has_state else
                              'initialized_from_observation' if sum(initialized) == 1 else
                              'posterior' if flag else 'prediction_only')
            if frames['state_phase'][i] != expected_phase:
                raise ValueError('追踪音频初始化/后验/预测状态混淆')
            if has_state and frames['angle_variance_deg2'][i] < 0:
                raise ValueError('追踪音频状态方差为负')
            if chain == 'pcm_analysis':
                codes = pcm[2*start:2*(start+512)]
                rms = math.sqrt(math.fsum(v*v for v in codes)/1024)/32768/manifest['common_export_gain']
                close(frames['rms_before_export'][i], rms, '实际PCM均方根')
        counts = {'frame_count': 197, 'valid_observation_count': sum(valid),
                  'missing_observation_count': sum(not v for v in valid),
                  'initialized_missing_count': sum(s and not v for s, v in zip(initialized, valid))}
        for field, value in counts.items():
            if type(scores[field]) is not int or scores[field] != value:
                raise ValueError('追踪音频评分整数分母不符：'+field)
        for field, values, mask in (
            ('raw_valid_rmse_deg', frames['observation_angle_deg'], valid),
            ('filtered_valid_rmse_deg', frames['filtered_angle_deg'], [v and s for v, s in zip(valid, initialized)]),
            ('filtered_missing_rmse_deg', frames['filtered_angle_deg'], [not v and s for v, s in zip(valid, initialized)])):
            errors = [(x-t)**2 for x, t, selected in zip(values, frames['truth_angle_deg'], mask) if selected]
            if not errors:
                if scores[field] is not None:
                    raise ValueError('空评分集合不能报告数值')
            else:
                close(scores[field], math.sqrt(math.fsum(errors)/len(errors)), field)

class VisibleMediaParser(HTMLParser):
    """Collect usable media and links outside hidden or inert subtrees."""
    def __init__(self):
        super().__init__()
        self.stack, self.items, self.links = [], [], set()

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        hidden = (tag in ('template', 'noscript') or (tag == 'dialog' and 'open' not in values) or
                  'hidden' in values or 'inert' in values or
                  values.get('aria-hidden', '').lower() == 'true' or
                  bool(re.search(r'display\s*:\s*none|visibility\s*:\s*hidden', values.get('style', ''), re.I)))
        visible_summary = (tag == 'summary' and bool(self.stack)
                           and self.stack[-1][0] == 'details' and not self.stack[-1][4])
        if visible_summary:
            parent = self.stack[-1]
            self.stack[-1] = (*parent[:4], True)
        if tag not in ('area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'):
            self.stack.append((tag, hidden, tag == 'details' and 'open' not in values,
                               visible_summary, False))
        collapsed = any(item[2] and not (index+1 < len(self.stack)
                        and self.stack[index+1][0] == 'summary' and self.stack[index+1][3])
                        for index, item in enumerate(self.stack))
        if not hidden and not collapsed and not any(item[1] for item in self.stack):
            if tag == 'audio':
                self.items.append(values)
            if tag == 'a':
                self.links.add(values.get('href'))

    def handle_endtag(self, tag):
        for index in range(len(self.stack)-1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break


def _check_visible_audio(page, prefix, directory, names, links):
    parser = VisibleMediaParser()
    parser.feed(page.read_text(encoding='utf-8'))
    base = prefix+directory+'/'
    label = {"binaural_audio": "双耳线索", "stft_audio": "STFT卷积",
             "geometry_audio": "多频几何", "focus_audio": "已知酉聚焦",
             "derivative_audio": "导数约束"}.get(directory, directory)
    players = [p for p in parser.items if (p.get('src') or '').startswith(base)]
    if len(players) != len(names) or {p.get('src') for p in players} != {base+n for n in names}:
        raise ValueError('缺少'+label+'播放器或可见播放器缺失或重复（集合不符）：'+page.name)
    if any('autoplay' in p or 'controls' not in p or p.get('preload') != 'none' or
           not (p.get('aria-label') or '').strip() for p in players):
        raise ValueError(label+'播放器标签或控件不同（控制/加载属性不符）：'+page.name)
    if not {base+n for n in links} <= parser.links:
        raise ValueError('缺少'+label+'独立清单：可见独立资产链接缺失（独立清单链接缺失）：'+page.name)


def _selection_scenario_pcm(source):
    """Independent standard-library PCM integer scoring, never repaired."""
    import struct
    pcm = {}
    for scene in ('single','dual'):
        for kind in ('target','mixture','fir3','fir9'):
            filename = f'selection_{scene}_{kind}.wav'
            with wave.open(str(source/filename),'rb') as w:
                if (w.getframerate(),w.getnchannels(),w.getsampwidth(),w.getnframes(),w.getcomptype()) != (16000,1,2,32008,'NONE'):
                    raise ValueError('selection scenario actual PCM format differs')
                data = w.readframes(32008)
            pcm[filename] = struct.unpack('<32008h',data)
    rates, rows = {}, {}
    for scene in ('single','dual'):
        reference = pcm[f'selection_{scene}_target.wav'][1600:30400]
        denominator = sum(v*v for v in reference)
        if denominator <= 0:
            raise ValueError('selection scenario reference has no energy')
        rates[scene], rows[scene] = {}, {}
        for length in (3,9):
            kind,delay = f'fir{length}',(length-1)//2
            output = pcm[f'selection_{scene}_{kind}.wav'][1600+delay:30400+delay]
            numerator = sum((a-b)**2 for a,b in zip(output,reference))
            rates[scene][kind] = numerator/denominator
            rows[scene][kind] = {'source_window':[1600,30400],
                'output_window':[1600+delay,30400+delay], 'scored_samples':28800,
                'integer_error_squared_sum':numerator,'integer_reference_squared_sum':denominator}
    return rates, rows


def _selection_scenario_analytic():
    """Sine-ratio control, independent of the plot's finite cosine sum."""
    import math
    result = {}
    for scene in ('single','dual'):
        result[scene] = {}
        for length in (3,9):
            def signed(f):
                x=math.pi*f/16000
                return math.sin(length*x)/(length*math.sin(x))
            norm=abs(signed(500))
            target_error=(signed(1500)/norm-1)**2 if scene=='dual' else 0.
            result[scene][f'fir{length}']=(target_error+(signed(3500)/norm)**2)/(2 if scene=='dual' else 1)
    return result


def _selection_scenario_decision(table):
    values = {kind:[(table['single'][kind]+3*table['dual'][kind])/4,
                   (3*table['single'][kind]+table['dual'][kind])/4]
              for kind in ('fir3','fir9')}
    difference=table['dual']['fir9']-table['dual']['fir3']
    slope=table['single']['fir3']-table['dual']['fir3']-table['single']['fir9']+table['dual']['fir9']
    return {'q_single_weight':[.25,.75], 'endpoint_costs':values,
            'interval_worst':{k:max(v) for k,v in values.items()},
            'scene_worst':{k:max(table['single'][k],table['dual'][k]) for k in ('fir3','fir9')},
            'crossing_q_single':difference/slope}


def _compare_selection_report(actual,expected,where='report'):
    import math
    if type(actual) is not type(expected):
        raise ValueError('selection report type differs: '+where)
    if isinstance(expected,dict):
        if actual.keys()!=expected.keys():
            raise ValueError('selection report fields differ: '+where)
        for key,value in expected.items():
            _compare_selection_report(actual[key],value,where+'.'+key)
    elif isinstance(expected,list):
        if len(actual)!=len(expected):
            raise ValueError('selection report list differs: '+where)
        for index,(left,right) in enumerate(zip(actual,expected)):
            _compare_selection_report(left,right,where+f'[{index}]')
    elif isinstance(expected,float):
        if not math.isfinite(actual) or not math.isclose(actual,expected,rel_tol=1e-13,abs_tol=1e-16):
            raise ValueError('selection report numeric value differs: '+where)
    elif actual!=expected:
        raise ValueError('selection report value differs: '+where)


def _check_selection_scenarios_report(path):
    from codes.chapters.ch11.examples.generate_selection_audio import validate_asset_directory
    folder=ROOT/'codes/chapters/ch11/scenario_audio'
    validate_asset_directory(folder,check=True)
    pcm,integers=_selection_scenario_pcm(folder)
    analytic=_selection_scenario_analytic()
    expected={'schema_version':1,
        'script_sha256':hashlib.sha256((ROOT/'scripts/make_figures.py').read_bytes()).hexdigest(),
        'audio_manifest_sha256':hashlib.sha256((folder/'MANIFEST.json').read_bytes()).hexdigest(),
        'scope':'mathematical tones; scenario-weighted NMSE, not pooled reference energy or mean dB',
        'integer_pcm':integers,'pcm_nmse':pcm,'analytic_nmse':analytic,
        'pcm_decision':_selection_scenario_decision(pcm),
        'analytic_decision':_selection_scenario_decision(analytic)}
    _compare_selection_report(_read_tracking_manifest(path),expected)


def check_scenario_audio(errors):
    """Bind all eight actual WAVs, integer denominators and visible controls."""
    import math
    import struct
    try:
        from codes.chapters.ch11.examples.generate_selection_audio import check_assets,validate_asset_directory
        source,published=SCENARIO_AUDIO_ROOT,SITE/'scenario_audio'
        validate_asset_directory(source,check=True)
        validate_asset_directory(published,check=True)
        manifest = _validate_audio_metadata(check_assets(source))
        expected_sources={'codes/chapters/ch11/core/selection_audio.py',
            'codes/chapters/ch11/examples/generate_selection_audio.py',
            'codes/chapters/ch00/core/audio_samples.py',
                          'codes/chapters/ch00/io_contracts.py'}
        if set(manifest['source_sha256'])!=expected_sources:
            raise ValueError('scenario source set differs')
        for path in expected_sources:
            if manifest['source_sha256'][path]!=hashlib.sha256(validate_parent_chain(ROOT/path).read_bytes()).hexdigest():
                raise ValueError('scenario source SHA is stale: '+path)
        if set(manifest['files'])!=SCENARIO_AUDIO_WAVS or type(manifest['common_export_gain']) is not float or manifest['common_export_gain']!=.8:
            raise ValueError('scenario eight files or common gain differs')
        for filename in sorted(SCENARIO_AUDIO_WAVS|{'MANIFEST.json'}):
            if (source/filename).read_bytes()!=(published/filename).read_bytes():
                raise ValueError('scenario published bytes differ: '+filename)
        rates,rows=_selection_scenario_pcm(published)
        for scene in ('single','dual'):
            for kind in ('fir3','fir9'):
                filename=f'selection_{scene}_{kind}.wav'
                measured=manifest['pcm_analysis']['candidates'][filename]
                row=rows[scene][kind]
                for field in ('integer_error_squared_sum','integer_reference_squared_sum','scored_samples'):
                    if type(measured[field]) is not int or measured[field]!=row[field]:
                        raise ValueError('scenario integer score differs: '+filename+'/'+field)
                if not math.isclose(measured['aligned_total_nmse'],rates[scene][kind],rel_tol=1e-14,abs_tol=0):
                    raise ValueError('scenario actual PCM NMSE differs: '+filename)
        for page,prefix in ((SITE/'11_selection-guide.html',''),
                            (SITE/'research/05_exercises_and_audio.html','../')):
            _check_visible_audio(page,prefix,'scenario_audio',SCENARIO_AUDIO_WAVS,{'MANIFEST.json'})
    except (OSError,ValueError,KeyError,TypeError,struct.error,wave.Error) as exc:
        fail(errors,'独立两场景选型音频：'+str(exc))


def _weighted_integer_pcm(directory):
    """Measure the five actual WAVs independently of teaching/plot analyses."""
    import struct
    integers = {}
    for filename in sorted(WEIGHTED_AUDIO_WAVS):
        channels = 2 if filename == 'weighted_array.wav' else 1
        with wave.open(str(Path(directory) / filename), 'rb') as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(),
                    wav.getnframes(), wav.getcomptype()) != (16000, channels, 2, 32000, 'NONE'):
                raise ValueError('weighted PCM format differs: ' + filename)
            raw = wav.readframes(32000)
        if len(raw) != channels * 64000:
            raise ValueError('weighted PCM is truncated: ' + filename)
        integers[filename] = struct.unpack('<' + str(channels * 32000) + 'h', raw)
    lo, hi = 1600, 30400
    reference = integers['weighted_target.wav'][lo:hi]
    denominator = sum(int(v) ** 2 for v in reference)
    if denominator == 0:
        raise ValueError('weighted reference energy is zero')
    result = {}
    for method in ('ols', 'gls', 'reversed'):
        filename = 'weighted_' + method + '.wav'
        numerator = sum((int(y) - int(r)) ** 2 for y, r in
                        zip(integers[filename][lo:hi], reference))
        result[filename] = {
            'integer_error_squared_sum': numerator,
            'integer_reference_squared_sum': denominator,
            'scored_samples': hi - lo, 'mse': numerator / ((hi - lo) * 32768**2),
            'nmse': numerator / denominator,
        }
    return result


def _check_weighted_noise_report(path):
    from codes.chapters.appendix_a.examples.generate_weighted_audio import validate_asset_directory
    directory = ROOT / 'codes/chapters/appendix_a/weighted_audio'
    validate_asset_directory(directory, check=True)
    expected = {
        'schema_version': 1,
        'script_sha256': hashlib.sha256((ROOT / 'scripts/make_figures.py').read_bytes()).hexdigest(),
        'audio_manifest_sha256': hashlib.sha256((directory / 'MANIFEST.json').read_bytes()).hexdigest(),
        'scope': 'known deterministic orthogonal tones; no covariance estimation or listening test',
        'score_window': [1600, 30400],
        'weights': {'ols': [.5, .5], 'gls': [.8, .2], 'reversed': [.2, .8]},
        'analytic_mse': {'ols': 9 / 16000, 'gls': 9 / 25000, 'reversed': 117 / 100000},
        'integer_pcm': _weighted_integer_pcm(directory),
    }
    _compare_selection_report(_read_tracking_manifest(path), expected)


def check_weighted_audio(errors):
    import struct
    try:
        from codes.chapters.appendix_a.examples.generate_weighted_audio import (
            check_assets, validate_asset_directory,
        )
        source, published = WEIGHTED_AUDIO_ROOT, SITE / 'weighted_audio'
        validate_asset_directory(source, check=True)
        validate_asset_directory(published, check=True)
        manifest = _validate_audio_metadata(check_assets(source))
        expected_sources = {
            'codes/chapters/appendix_a/core/weighted_audio.py',
            'codes/chapters/appendix_a/examples/generate_weighted_audio.py',
            'codes/chapters/ch00/core/audio_samples.py',

            'codes/chapters/ch00/io_contracts.py'}
        if set(manifest['source_sha256']) != expected_sources:
            raise ValueError('weighted real source set differs')
        for path in expected_sources:
            if manifest['source_sha256'][path] != hashlib.sha256(validate_parent_chain(ROOT / path).read_bytes()).hexdigest():
                raise ValueError('weighted source SHA is stale: ' + path)
        if (set(manifest['files']) != WEIGHTED_AUDIO_WAVS
                or type(manifest['common_export_gain']) is not float
                or manifest['common_export_gain'] != 1.):
            raise ValueError('weighted five files/common gain differs')
        for filename in WEIGHTED_AUDIO_WAVS | {'MANIFEST.json'}:
            if (source / filename).read_bytes() != (published / filename).read_bytes():
                raise ValueError('weighted published bytes differ: ' + filename)
        for filename, measured in _weighted_integer_pcm(published).items():
            row = manifest['pcm_analysis']['candidates'][filename]
            for field, expected in measured.items():
                _compare_selection_report(row[field], expected, 'weighted PCM/' + filename + '/' + field)
        for page, prefix in ((SITE / '12_appendix-symbols-math.html', ''),
                             (SITE / 'research/05_exercises_and_audio.html', '../')):
            _check_visible_audio(page, prefix, 'weighted_audio', WEIGHTED_AUDIO_WAVS, {'MANIFEST.json'})
    except (OSError, ValueError, KeyError, TypeError, struct.error, wave.Error) as exc:
        fail(errors, '独立已知噪声加权音频：' + str(exc))


def _response_integer_pcm(directory):
    """Independent ordinary WAV/int scoring; no teaching analyzer is used."""
    import struct
    integers = {}
    for name in ('source', 'full_a', 'full_b'):
        with wave.open(str(directory / ('response_' + name + '.wav')), 'rb') as stream:
            if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth(),
                    stream.getnframes(), stream.getcomptype()) != (16000, 1, 2, 32002, 'NONE'):
                raise ValueError('response PCM format differs')
            data = stream.readframes(32002)
        if len(data) != 64004:
            raise ValueError('response PCM is truncated')
        integers[name] = struct.unpack('<32002h', data)
    reference = integers['source'][1600:30400]
    denominator = sum(x*x for x in reference)
    if denominator == 0:
        raise ValueError('response reference has zero energy')
    result = {}
    for name in ('full_a', 'full_b'):
        numerator = sum((x-y)**2 for x, y in zip(integers[name][1600:30400], reference))
        result['response_' + name + '.wav'] = {
            'integer_error_squared_sum': numerator,
            'integer_reference_squared_sum': denominator, 'scored_samples': 28800,
            'mse': numerator/(28800*32768**2), 'nmse': numerator/denominator,
        }
    return result


def _check_equal_drr_report(path):
    import math
    from codes.chapters.appendix_b.examples.generate_response_audio import validate_asset_directory
    directory = ROOT / 'codes/chapters/appendix_b/response_audio'
    validate_asset_directory(directory, check=True)
    gains = {'a': [(2+math.sqrt(2))/4, .5], 'b': [(2-math.sqrt(2))/4, .5]}
    expected = {
        'schema_version': 1,
        'script_sha256': hashlib.sha256((ROOT/'scripts/make_figures.py').read_bytes()).hexdigest(),
        'audio_manifest_sha256': hashlib.sha256((directory/'MANIFEST.json').read_bytes()).hexdigest(),
        'scope': 'known short FIRs and equal-power tones; no real room, blind estimation or listening test',
        'score_window': [1600, 30400], 'frequencies_hz': [2000, 4000],
        'rir_drr_db': 10*math.log10(2), 'reflection_squared_gain': gains,
        'analytic_output_reflection_nmse': {name: sum(values)/2 for name, values in gains.items()},
        'integer_pcm': _response_integer_pcm(directory),
    }
    _compare_selection_report(_read_tracking_manifest(path), expected)


def check_response_audio(errors):
    import struct
    try:
        from codes.chapters.appendix_b.examples.generate_response_audio import check_assets, validate_asset_directory
        source, published = RESPONSE_AUDIO_ROOT, SITE/'response_audio'
        manifest = _validate_audio_metadata(check_assets(source))
        validate_asset_directory(published, check=True)
        expected_sources = {
            'codes/chapters/appendix_b/core/response_audio.py',
            'codes/chapters/appendix_b/examples/generate_response_audio.py',
            'codes/chapters/ch00/core/audio_samples.py',

            'codes/chapters/ch00/io_contracts.py'}
        if set(manifest['source_sha256']) != expected_sources:
            raise ValueError('response real source set differs')
        for name in expected_sources:
            if manifest['source_sha256'][name] != hashlib.sha256(validate_parent_chain(ROOT/name).read_bytes()).hexdigest():
                raise ValueError('response source SHA is stale')
        if (set(manifest['files']) != RESPONSE_AUDIO_WAVS
                or type(manifest['common_export_gain']) is not float
                or manifest['common_export_gain'] != 1.):
            raise ValueError('response five files/common gain differs')
        for name in RESPONSE_AUDIO_WAVS | {'MANIFEST.json'}:
            if (source/name).read_bytes() != (published/name).read_bytes():
                raise ValueError('response site bytes differ: '+name)
        for name, measured in _response_integer_pcm(published).items():
            for field, expected in measured.items():
                _compare_selection_report(manifest['pcm_analysis']['candidates'][name][field], expected,
                                          'response PCM/'+name+'/'+field)
        for page, prefix in ((SITE/'13_appendix-guide.html', ''),
                             (SITE/'research/05_exercises_and_audio.html', '../')):
            _check_visible_audio(page, prefix, 'response_audio', RESPONSE_AUDIO_WAVS, {'MANIFEST.json'})
    except (OSError, ValueError, KeyError, TypeError, struct.error, wave.Error) as error:
        fail(errors, '独立同DRR频响音频：'+str(error))


def _distributed_integer_pcm(directory):
    """Read actual PCM with stdlib integers; never use the generating scorer."""
    import struct
    members = set(DISTRIBUTED_AUDIO_CHANNELS) | {'MANIFEST.json'}
    validate_asset_directory(directory, members, check=True)
    channels = {}
    for name, count in DISTRIBUTED_AUDIO_CHANNELS.items():
        with wave.open(str(directory / name), 'rb') as reader:
            if (reader.getnchannels(), reader.getsampwidth(), reader.getframerate(),
                    reader.getnframes(), reader.getcomptype()) != (count, 2, 16000, 32000, 'NONE'):
                raise ValueError('distributed actual PCM format differs: ' + name)
            raw = reader.readframes(32000)
        if len(raw) != 64000 * count:
            raise ValueError('distributed PCM length differs: ' + name)
        flat = struct.unpack('<' + str(32000 * count) + 'h', raw)
        channels[name[:-4]] = [flat[c::count] for c in range(count)]
    measurements = {}
    non_outputs = {'reference_node1', 'reference_node2', 'array_white', 'array_correlated', 'remote_scalar_white'}
    for key, data in channels.items():
        reference = None if key in non_outputs else ('reference_node2' if key == 'central_node2_correlated' else 'reference_node1')
        windows = {}
        for label, (start, stop) in {'steady': (1600, 30400), 'packet': (16000, 16800)}.items():
            n = stop - start
            powers = [sum(v*v for v in ch[start:stop]) for ch in data]
            row = {'samples': n, 'integer_squared_sum_by_channel': powers,
                   'mean_square_by_channel': [v / (n * 32768**2) for v in powers]}
            if reference is not None:
                target = channels[reference][0][start:stop]
                E = sum((v-r)**2 for v, r in zip(data[0][start:stop], target))
                D = sum(r*r for r in target)
                row.update(reference_energy=D/32768**2, total_mse=E/(n*32768**2),
                           normalized_mse=E/D, integer_error_squared_sum_E=E,
                           integer_reference_squared_sum_D=D, integer_sample_denominator=n,
                           integer_nmse_E_over_D=E/D)
            windows[label] = row
        measurements[key] = {'reference_key': reference, 'windows': windows}
    return measurements


def check_spectral_audio(errors):
    """Current-source replay plus independent header, byte and integer-power checks."""
    try:
        import numpy as np
        from codes.chapters.ch01.examples.generate_spectral_cues import check_assets
        source, published = SPECTRAL_AUDIO_ROOT, SITE / 'spectral_audio'
        manifest = check_assets(source)
        members = set(SPECTRAL_AUDIO_CHANNELS) | {'MANIFEST.json'}
        validate_asset_directory(published, members, check=True)
        required_sources = {'codes/chapters/ch01/core/spectral_cues.py',
                            'codes/chapters/ch01/examples/generate_spectral_cues.py',
                            'codes/chapters/ch00/core/audio_samples.py',
                            'codes/chapters/ch00/io_contracts.py'}
        if set(manifest['source_sha256']) != required_sources:
            raise ValueError('spectral generating source set differs')
        for name in members:
            if (source/name).read_bytes() != (published/name).read_bytes():
                raise ValueError('spectral published bytes differ: ' + name)
        for name, channels in SPECTRAL_AUDIO_CHANNELS.items():
            frames = 32000 if channels == 1 else 32001
            with wave.open(str(published/name), 'rb') as stream:
                if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth(),
                        stream.getnframes(), stream.getcomptype()) != (16000, channels, 2, frames, 'NONE'):
                    raise ValueError('spectral independent PCM format differs: ' + name)
                pcm = np.frombuffer(stream.readframes(frames), dtype='<i2').reshape(-1, channels)
            if channels == 2:
                # Python integer squares provide a separate route from the
                # generator's NumPy reduction and decoded-float scoring.
                energy = [sum(int(value)**2 for value in pcm[1600:30400, c]) for c in range(2)]
                expected = {'left_squared_sum_E': energy[0], 'right_squared_sum_E': energy[1],
                            'integer_denominator_D': 28800*32768**2, 'samples': 28800,
                            'left_mean_square': energy[0]/(28800*32768**2),
                            'right_mean_square': energy[1]/(28800*32768**2),
                            'ild_right_minus_left_db': 10*math.log10(energy[1]/energy[0])}
                actual = manifest['samples'][name.split('_')[0]]['pcm_integer_measurements']
                if actual != expected:
                    raise ValueError('spectral independent integer scoring differs: ' + name)
        for page, prefix in ((SITE/'01_problem-definition.html', ''),
                             (SITE/'research/05_exercises_and_audio.html', '../')):
            _check_visible_audio(page, prefix, 'spectral_audio', set(SPECTRAL_AUDIO_CHANNELS), {'MANIFEST.json'})
    except (OSError, ValueError, KeyError, TypeError, wave.Error) as error:
        fail(errors, '独立方向谱形音频：' + str(error))


def check_distributed_audio(errors):
    """Exact member/copy checks, current-source replay and independent integer E/D."""
    try:
        from codes.chapters.ch15.examples.generate_distributed_audio import check_assets
        source, published = DISTRIBUTED_AUDIO_ROOT, SITE / 'distributed_audio'
        manifest = check_assets(source)
        members = set(DISTRIBUTED_AUDIO_CHANNELS) | {'MANIFEST.json'}
        validate_asset_directory(published, members, check=True)
        if set(manifest['source_sha256']) != set(DISTRIBUTED_AUDIO_SOURCES):
            raise ValueError('distributed generating source set differs')
        for path in DISTRIBUTED_AUDIO_SOURCES:
            if manifest['source_sha256'][path] != hashlib.sha256(validate_parent_chain(ROOT/path).read_bytes()).hexdigest():
                raise ValueError('distributed current source SHA differs: ' + path)
        for name in members:
            if (source/name).read_bytes() != (published/name).read_bytes():
                raise ValueError('distributed published bytes differ: ' + name)
        for key, expected in _distributed_integer_pcm(published).items():
            declared = manifest['samples'][key]['pcm_measurements']
            if declared['reference_key'] != expected['reference_key']:
                raise ValueError('distributed scoring reference differs: ' + key)
            for label, row in expected['windows'].items():
                for field, value in row.items():
                    _compare_selection_report(declared['windows'][label][field], value,
                                              'distributed PCM/' + key + '/' + label + '/' + field)
        for page, prefix in ((SITE/'15_distributed-enhancement.html', ''),
                             (SITE/'research/05_exercises_and_audio.html', '../')):
            _check_visible_audio(page, prefix, 'distributed_audio', set(DISTRIBUTED_AUDIO_CHANNELS), {'MANIFEST.json'})
    except (OSError, ValueError, KeyError, TypeError, wave.Error) as error:
        fail(errors, '独立分布式协同音频：' + str(error))


def _check_distributed_figure_report(path, number):
    """Reconstruct costs and update chronology from equations, independently."""
    import numpy as np
    report = _read_tracking_manifest(path)
    fields={'schema_version','scope','source_sha256','results'}
    if number in (70,72):fields.add('audio_manifest_sha256')
    if (set(report)!=fields or report['scope'] !=
            'known-statistics finite teaching controls; not blind DANSE or industrial performance'):
        raise ValueError('distributed figure fields/execution scope differ')
    expected_sources = set(DISTRIBUTED_AUDIO_SOURCES) | {'scripts/make_figures.py'}
    if number == 71:
        expected_sources = {'scripts/make_figures.py', 'codes/chapters/ch15/core/distributed.py',
                            'codes/chapters/ch02/core/conventions.py', 'codes/chapters/ch04/core/covariance.py',
                            'codes/chapters/ch00/io_contracts.py'}
    if type(report['schema_version']) is not int or report['schema_version'] != 1 or set(report['source_sha256']) != expected_sources:
        raise ValueError('distributed figure source/schema differs')
    for name in expected_sources:
        if report['source_sha256'][name] != hashlib.sha256(validate_parent_chain(ROOT/name).read_bytes()).hexdigest():
            raise ValueError('distributed figure current source SHA differs: ' + name)
    def close(actual, expected, label):
        if not np.allclose(actual, expected, rtol=2e-11, atol=2e-14):
            raise ValueError('distributed figure independent value differs: ' + label)
    def z(value):
        return np.asarray(value['real']) + 1j*np.asarray(value['imag'])
    def component(row):
        fields = ('target_distortion_power', 'noise_power', 'other_error_power',
                  'target_noise_cross_power', 'target_other_cross_power', 'noise_other_cross_power')
        close(row['total_mse'], sum(row[f] for f in fields), 'signed component sum')
        close(row['normalized_mse'], row['total_mse']/row['reference_power'], 'float reference denominator')
    data = report['results']
    if number in (70, 72):
        manifest_path = DISTRIBUTED_AUDIO_ROOT/'MANIFEST.json'
        if report['audio_manifest_sha256'] != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
            raise ValueError('distributed figure audio manifest SHA differs')
        manifest = _read_tracking_manifest(manifest_path)
        pcm = _distributed_integer_pcm(DISTRIBUTED_AUDIO_ROOT)
        def score(key, label, declared):
            row = pcm[key]['windows'][label]
            expected = {'E': row['integer_error_squared_sum_E'], 'D': row['integer_reference_squared_sum_D'],
                        'samples': row['samples'], 'nmse': row['integer_nmse_E_over_D']}
            _compare_selection_report(declared, expected, f'figure{number}/{key}/{label}')
        if number == 70:
            expected = {'local_node1': 4/9, 'central_white': 2/13, 'compressed_white': 2/13,
                        'stale_correlated': 2/13, 'central_correlated': 8/69, 'compressed_correlated': 8/69}
            if set(data['rows']) != set(expected) or data['window'] != [1600,30400]:
                raise ValueError('figure70 cases/scoring window differ')
            for key, target in expected.items():
                row = data['rows'][key]
                close(row['analytic']['normalized_mse'], target, 'fixed rational cost')
                close(row['analytic']['total_mse'], .0068*target, 'absolute model cost')
                a = np.array([1.,.5,2.,-.5]); noise=np.eye(4)
                if key.endswith('correlated'):
                    noise[0,2]=noise[2,0]=.2;noise[0,3]=noise[3,0]=.8
                w = np.array([4,2,0,0])/9 if key=='local_node1' else (
                    np.array([25,4,11,-24])/69 if key in ('central_correlated','compressed_correlated') else
                    np.array([2,1,4,-1])/13)
                close([row['analytic']['target_distortion'],row['analytic']['noise_power'],
                       row['analytic']['reference_target_power']],
                      [.0068*(w@a-1)**2,.0068*(w@noise@w),.0068], 'independent signal/noise quadratic forms')
                component(row['float'])
                _compare_selection_report(row['float'], manifest['samples'][key]['float_components']['steady'], 'figure70 float')
                score(key, 'steady', row['pcm'])
        else:
            expected = {'compressed_white', 'clock_misaligned_white', 'clock_linear_corrected_white',
                        'packet_zerofill_white', 'packet_local_fallback_white'}
            if set(data['rows']) != expected:
                raise ValueError('figure72 case set differs')
            for key, row in data['rows'].items():
                labels = ('steady', 'packet') if key.startswith('packet_') else ('steady',)
                for label in labels:
                    floating = row['float_components'][label] if len(labels)>1 else row['float_components']
                    component(floating)
                    _compare_selection_report(floating, manifest['samples'][key]['float_components'][label], 'figure72 float')
                    score(key, label, row['pcm'][label] if len(labels)>1 else row['pcm'])
            for key, expected in (('packet_zerofill_white',461/676), ('packet_local_fallback_white',4/9)):
                close(data['rows'][key]['float_components']['packet']['normalized_mse'], expected, 'gap rational control')
            cross = [[data['rows'][key]['float_components']['other_error_power']/ .0068,
                      data['rows'][key]['float_components']['target_noise_cross_power']/ .0068]
                     for key in ('compressed_white', 'clock_misaligned_white', 'clock_linear_corrected_white')]
            close(data['cross_terms'], cross, 'signed cross panel')
            _compare_selection_report(data['clock'], manifest['float_clock_control'], 'figure72 known-rate SRC')
            if data['parameters']['scoring_windows_samples'] != {'steady':[1600,30400], 'packet':[16000,16800]}:
                raise ValueError('figure72 gap/global windows differ')
        return
    if number != 71:
        raise ValueError('unknown distributed figure')
    a = np.array([1., .5, 2., -.5]); Rs = np.outer(a,a); Rn = np.eye(4)
    Rn[0,2] = Rn[2,0] = .2; Rn[0,3] = Rn[3,0] = .8; Rx = Rs+Rn
    close(data['model']['Rs'], Rs, 'source covariance'); close(data['model']['correlated'], Rn, 'noise covariance')
    close(data['model']['a'],a,'steering vector');close(data['model']['white'],np.eye(4),'white covariance')
    if data['model']['nodes']!=[[0,1],[2,3]] or data['model']['references']!=[0,2]:
        raise ValueError('figure71 node/reference model differs')
    if set(data['trajectories']) != {'round_robin','simultaneous','simultaneous_half'}:
        raise ValueError('figure71 trajectory set differs')
    for key,scale in (('central_node1',1),('central_node2',4)):
        close([data[key]['total_mse'],data[key]['reference_target_power'],data[key]['normalized_mse']],
              [scale*8/69,scale,8/69],'centralized reference-specific anchor')
    nodes, refs = ([0,1],[2,3]), (0,2)
    for name, (schedule, alpha, budget, solved) in {'round_robin': ('round_robin',1.,40,14),
            'simultaneous': ('simultaneous',1.,100,26), 'simultaneous_half': ('simultaneous',.5,100,76)}.items():
        run = data['trajectories'][name]
        if (run['schedule'],run['relaxation'],run['max_update_solves'],run['update_solve_count'],
                run['initial_local_solve_count'],run['total_solve_count']) != (schedule,alpha,budget,solved,2,solved+2):
            raise ValueError('figure71 actual solve budget differs: '+name)
        before = [a[list(ids)].astype(complex) for ids in nodes]
        last = [np.zeros(4,complex) for _ in nodes]; last_versions=[0,0]
        coefficients=[];peer_maps=[[],[]]
        for k, ids in enumerate(nodes):
            last[k][list(ids)] = np.linalg.solve(Rx[np.ix_(ids,ids)], Rs[list(ids),refs[k]])
            coefficients.append(last[k][list(ids)].copy())
        count=0
        for epoch, h in enumerate(run['history'],1):
            active = [(epoch-1)%2] if schedule=='round_robin' else [0,1]
            count += len(active)
            if (h['epoch'] != epoch or h['active_nodes'] != active or h['update_solve_count'] != count
                    or h['total_solve_count'] != count+2 or len(h['solutions']) != len(active)):
                raise ValueError('figure71 chronology/solve count differs')
            close([z(v) for v in h['compressions_before']], before, 'old broadcast snapshot')
            after = [v.copy() for v in before]
            for sol, k in zip(h['solutions'],active):
                peer=1-k
                close([z(v) for v in sol['incoming_compressions']],before,'shared old snapshot')
                T = np.zeros((3,4),complex); T[:2,list(nodes[k])] = np.eye(2); T[2,list(nodes[peer])] = before[peer].conj()
                u=np.linalg.solve(T@Rx@T.conj().T,T@Rs[:,refs[k]])
                close(z(sol['raw_projection']),T,'solve projection');close(z(sol['raw_compressed_weights']),u,'raw solve coefficients')
                close(z(sol['weights']),T.conj().T@u,'old solve proposal')
                after[k]=(1-alpha)*before[k]+alpha*u[:2]
                close(z(sol['new_compression']),after[k],'relaxed local broadcast')
                last[k]=T.conj().T@u;last_versions[k]=count
                coefficients[k]=u;peer_maps[k]=[peer]
            close([z(v) for v in h['compressions_after']],after,'new broadcasts')
            for k in range(2):
                peer_map=h['receiver_peer_maps'][k]
                if peer_map!=peer_maps[k]:raise ValueError('figure71 stale receiver peer map differs')
                close(z(h['receiver_coefficients_raw_coordinates'][k]),coefficients[k],'frozen receiver coefficients')
                T=np.zeros((2+len(peer_map),4),complex)
                T[:2,list(nodes[k])]=np.eye(2)
                for j,p in enumerate(peer_map):T[2+j,list(nodes[p])]=after[p].conj()
                u=z(h['receiver_coefficients_raw_coordinates'][k]); w=T.conj().T@u
                close(z(h['current_receiver_projections'][k]),T,'current receive projection')
                close(z(h['cached_outputs'][k]),w,'current effective output')
                close(z(h['outputs_at_last_solve'][k]),last[k],'preserved proposal snapshot')
                e=np.eye(4)[:,refs[k]];dist=float(np.real((w-e).conj()@Rs@(w-e)));noise=float(np.real(w.conj()@Rn@w))
                c=h['cached_components'][k]
                close([c['target_distortion'],c['noise_power'],c['total_mse'],c['normalized_mse']],
                      [dist,noise,dist+noise,(dist+noise)/Rs[refs[k],refs[k]]],'arbitrary effective weight full cost')
                close([c['reference_target_power'],c['target_noise_cross_term']],
                      [Rs[refs[k],refs[k]],0.],'model target denominator/uncorrelated cross')
                residual=np.linalg.norm(Rx@w-Rs[:,refs[k]])/np.linalg.norm(Rs[:,refs[k]])
                close(h['normal_equation_relative_residuals'][k],residual,'current residual')
            if h['output_last_update'] != last_versions or h['output_age_in_update_solves'] != [count-v for v in last_versions]:
                raise ValueError('figure71 receiver coefficient versions differ')
            for sol in h['solutions']:
                k=sol['node'];w=z(sol['weights']);e=np.eye(4)[:,refs[k]]
                dist=float(np.real((w-e).conj()@Rs@(w-e)));noise=float(np.real(w.conj()@Rn@w))
                close([sol['components']['target_distortion'],sol['components']['noise_power'],sol['components']['total_mse']],
                      [dist,noise,dist+noise],'old proposal cost')
                close(z(sol['effective_weights_after_broadcast']),z(h['cached_outputs'][k]),'active output after simultaneous receive')
            before=after
        if count!=solved or max(run['history'][-1]['normal_equation_relative_residuals'])>1e-8:
            raise ValueError('figure71 final stopping condition fails')
        if (run['unused_update_budget']!=budget-solved or run['tolerance']!=1e-8
                or run['status']!='known_covariance_residual_reached'):
            raise ValueError('figure71 final stop/budget metadata differs')
        for field in ('compressions', 'cached_outputs', 'receiver_coefficients_raw_coordinates'):
            history_field='compressions_after' if field=='compressions' else field
            close([z(v) for v in run[field]],[z(v) for v in run['history'][-1][history_field]],'final '+field)
        if run['receiver_peer_maps'] != peer_maps:raise ValueError('figure71 final receiver mapping differs')
        if len(run['history'])>1 and max(run['history'][-2]['normal_equation_relative_residuals'])<=1e-8:
            raise ValueError('figure71 unnecessary post-stop solves')


def check_distributed_figures(errors):
    for number, name in {70:'compression',71:'updates',72:'transport'}.items():
        try:
            _check_distributed_figure_report(ROOT/f'codes/chapters/ch15/reports/figure{number}_distributed_{name}.json',number)
        except (OSError, ValueError, KeyError, TypeError, IndexError, wave.Error) as error:
            fail(errors, f'图{number}分布式协同报告：{error}')


def check_imaging_audio(errors):
    """Check published bytes and independently recover single-tone PCM CSMs."""
    try:
        import numpy as np
        from codes.chapters.ch14.examples.generate_imaging_audio import check_assets
        names = {'source_1.wav': 1, 'source_2_phase_code.wav': 1,
                 'source_2_coherent.wav': 1, 'array_phase_code.wav': 2,
                 'array_coherent.wav': 2}
        source, published = IMAGING_AUDIO_ROOT, SITE / 'imaging_audio'
        check_assets(source)
        members = set(names) | {'MANIFEST.json'}
        validate_asset_directory(published, members, check=True)
        pcm = {}
        for name in sorted(members):
            if (source / name).read_bytes() != (published / name).read_bytes():
                raise ValueError('imaging published bytes differ: ' + name)
            if name.endswith('.wav'):
                with wave.open(str(published / name), 'rb') as reader:
                    if (reader.getnchannels(), reader.getsampwidth(), reader.getframerate(),
                            reader.getnframes(), reader.getcomptype()) != (names[name], 2, 24000, 48004, 'NONE'):
                        raise ValueError('imaging actual PCM format differs: ' + name)
                    data = reader.readframes(48004)
                pcm[name] = np.frombuffer(data, dtype='<i2').reshape(48004, names[name]).T / 32768.
        def csm(signal):
            phasors = []
            for block in range(20):
                indices = np.arange(2400 * block + 252, 2400 * block + 2160)
                phasors.append(2 / 1908 * (signal[:, indices] @
                    np.exp(-2j * np.pi * 2000 * indices / 24000)))
            z = np.stack(phasors, axis=1)
            return (z @ z.conj().T) / 40 / .02
        a = np.array([[1, 1], [1, np.exp(-2j * np.pi / 3)]])
        expected = {
            'array_phase_code.wav': (a * np.array([1, .25])) @ a.conj().T,
            'array_coherent.wav': np.outer(a @ np.array([1, .5]), (a @ np.array([1, .5])).conj()),
        }
        # Derived from half-PCM-step phasor error, 0.3 peak and fixed 0.02 scale.
        error_bound = (2 * .3 / 32768 + 1 / 32768**2) / .02
        for name, reference in expected.items():
            actual = csm(pcm[name])
            if np.max(np.abs(actual - reference)) > error_bound:
                raise ValueError('imaging actual PCM CSM exceeds quantization bound: ' + name)
        for page, prefix in ((SITE / '14_acoustic-imaging.html', ''),
                             (SITE / 'research/05_exercises_and_audio.html', '../')):
            _check_visible_audio(page, prefix, 'imaging_audio', set(names), {'MANIFEST.json'})
    except (OSError, ValueError, KeyError, TypeError, wave.Error) as error:
        fail(errors, '独立声学成像音频：' + str(error))


def check_noise_audio(errors):
    """Independent actual integer scoring plus current-source replay and visibility."""
    import math
    import struct
    try:
        from codes.chapters.ch10.examples.generate_noise_mismatch import check_assets, validate_asset_directory
        source, published = NOISE_AUDIO_ROOT, SITE/'noise_audio'
        validate_asset_directory(source, check=True)
        validate_asset_directory(published, check=True)
        manifest = _validate_audio_metadata(check_assets(source))
        source_paths = {
            'codes/chapters/ch10/core/noise_mismatch.py',
            'codes/chapters/ch10/examples/generate_noise_mismatch.py',
            'codes/chapters/ch10/core/noise_suppression.py',
            'codes/chapters/ch02/core/spectral.py',
            'codes/chapters/ch02/core/conventions.py',
            'codes/chapters/ch00/core/audio_samples.py',
            'codes/chapters/ch00/io_contracts.py'}
        if set(manifest['source_sha256']) != source_paths:
            raise ValueError('noise_audio真实源集合不同')
        for path in source_paths:
            if manifest['source_sha256'][path] != hashlib.sha256(validate_parent_chain(ROOT/path).read_bytes()).hexdigest():
                raise ValueError('noise_audio真实源摘要过期：'+path)
        if set(manifest['files']) != NOISE_AUDIO_WAVS or manifest['common_export_gain'] != 1:
            raise ValueError('noise_audio六文件与共同增益不同')
        pcm = {}
        for name in sorted(NOISE_AUDIO_WAVS | {'MANIFEST.json'}):
            if (source/name).read_bytes() != (published/name).read_bytes():
                raise ValueError('noise_audio发布副本不同：'+name)
            if name.endswith('.wav'):
                with wave.open(str(published/name),'rb') as reader:
                    if (reader.getnchannels(),reader.getsampwidth(),reader.getframerate(),reader.getnframes(),reader.getcomptype()) != (1,2,16000,32000,'NONE'):
                        raise ValueError('noise_audio实际PCM尺寸不同')
                    data = reader.readframes(32000)
                pcm[name] = struct.unpack('<32000h',data)
        for key,(start,stop) in {'before_step':(9600,16000),'after_step':(22400,28800)}.items():
            row = manifest['pcm_analysis']['score_windows'][key]
            reference = pcm['noise_reference.wav'][start:stop]
            denominator = sum(value*value for value in reference)
            if row['sample_interval'] != [start,stop] or row['samples'] != 6400 or row['reference_squared_sum_pcm_integer'] != denominator or row['pcm_amplitude_denominator'] != 32768:
                raise ValueError('noise_audio真实整数分母不同')
            for filename in ('noise_mixture.wav','noise_fixed.wav','noise_polluted.wav','noise_known_variance.wav'):
                numerator = sum((value-ref)**2 for value,ref in zip(pcm[filename][start:stop],reference))
                score = row['scores'][filename]
                if type(score['error_squared_sum_pcm_integer']) is not int or score['error_squared_sum_pcm_integer'] != numerator or not math.isclose(score['nmse'],numerator/denominator,rel_tol=1e-14,abs_tol=0) or not math.isclose(score['mse'],numerator/(6400*32768**2),rel_tol=1e-14,abs_tol=0):
                    raise ValueError('noise_audio实际整数误差不同：'+filename)
                if filename != 'noise_mixture.wav':
                    floating = manifest['floating_point']['score_windows'][key]['scores'][filename]
                    total = floating['target_distortion_mse']+floating['residual_noise_mse']+floating['twice_cross_term']
                    if not math.isclose(total,floating['mse'],rel_tol=1e-12,abs_tol=1e-17):
                        raise ValueError('noise_audio浮点分量与总体误差不符')
        for page,prefix in ((SITE/'10_engineering-practice.html',''),
                            (SITE/'research/05_exercises_and_audio.html','../')):
            _check_visible_audio(page,prefix,'noise_audio',NOISE_AUDIO_WAVS,{'MANIFEST.json'})
    except (OSError,ValueError,KeyError,TypeError,struct.error,wave.Error) as exc:
        fail(errors,'独立噪声估计失配音频：'+str(exc))


def check_mask_audio(errors):
    """Independently recompute integer errors and phase from actual PCM."""
    import math
    import struct
    names = {key: 'mask_'+key+'.wav' for key in
             ('target', 'other', 'mixture', 'bounded_real', 'unbounded_real', 'complex_oracle')}
    sources = {'codes/chapters/ch08/core/mask_representation.py',
               'codes/chapters/ch08/examples/mask_representation_demo.py',
               'codes/chapters/ch02/core/conventions.py', 'codes/chapters/ch00/core/audio_samples.py',
               'codes/chapters/ch00/io_contracts.py'}
    source, published = MASK_AUDIO_ROOT, SITE/'mask_audio'
    try:
        expected = set(names.values()) | {'MANIFEST.json'}
        for folder in (source, published):
            validate_asset_directory(folder, expected, check=True)
        manifest = _read_audio_manifest(source / "MANIFEST.json")
        if set(manifest['source_sha256']) != sources:
            raise ValueError('mask_audio真实源集合不同')
        for path in sources:
            if hashlib.sha256(validate_parent_chain(ROOT/path).read_bytes()).hexdigest() != manifest['source_sha256'][path]:
                raise ValueError('mask_audio真实源摘要过期：'+path)
        from codes.chapters.ch08.examples.mask_representation_demo import check_assets
        check_assets(source, replay=True)
        for name in expected:
            if (source/name).read_bytes() != (published/name).read_bytes():
                raise ValueError('mask_audio发布副本不同：'+name)
        decoded = {}
        for key, filename in names.items():
            with wave.open(str(source/filename), 'rb') as wav:
                if (wav.getframerate(), wav.getnchannels(), wav.getnframes(), wav.getsampwidth(), wav.getcomptype()) != (16000, 1, 32000, 2, 'NONE'):
                    raise ValueError('mask_audio实际PCM尺寸不同')
                raw = wav.readframes(32000)
            decoded[key] = struct.unpack('<32000h', raw)[2400:29600]
        truth = decoded['target']
        denominator = sum(v*v for v in truth)
        # Complete integer periods make the four analytic columns orthogonal.
        # This scalar route is independent of the generator's array LS solver.
        design = [[fn(2*math.pi*f*n/16000) for n in range(2400, 29600)]
                  for f in (500, 1250) for fn in (math.cos, math.sin)]
        for key, filename in names.items():
            output = decoded[key]
            numerator = sum((a-b)**2 for a, b in zip(output, truth))
            score = manifest['samples'][key]['pcm_measurements']
            for field, value in {'integer_reference_squared_sum': denominator,
                                 'integer_error_squared_sum': numerator,
                                 'integer_mse_denominator': 27200*32768**2,
                                 'sample_denominator': 27200, 'pcm_decode_divisor': 32768}.items():
                if type(score[field]) is not int or score[field] != value:
                    raise ValueError('mask_audio真实整数统计不同：'+filename)
            for field, value in {'total_reference_mse': numerator/(27200*32768**2),
                                 'relative_squared_reference_error': numerator/denominator}.items():
                if (type(score[field]) not in (int, float) or not math.isfinite(score[field]) or
                        not math.isclose(score[field], value, rel_tol=1e-13, abs_tol=0)):
                    raise ValueError('mask_audio真实PCM误差不同：'+filename)
            coefficients = [2*math.fsum(v*x for v, x in zip(output, column))/(27200*32768)
                            for column in design]
            phase = score['phase_ls']
            if phase['gain_or_time_compensation'] is not False or type(phase['rank']) is not int or phase['rank'] != 4:
                raise ValueError('mask_audio相位诊断范围不同')
            actual_coefficients = phase['coefficients']
            if (len(actual_coefficients) != 4 or any(type(a) not in (int, float) or not math.isfinite(a) or
                    not math.isclose(a, b, rel_tol=1e-11, abs_tol=2e-13)
                    for a, b in zip(actual_coefficients, coefficients))):
                raise ValueError('mask_audio独立相位系数不同：'+filename)
            residual = math.fsum((v/32768-math.fsum(c*d[n] for c, d in zip(coefficients, design)))**2
                                  for n, v in enumerate(output))
            if not math.isclose(phase['residual_squared_sum'], residual, rel_tol=1e-7, abs_tol=1e-16):
                raise ValueError('mask_audio独立相位残差不同：'+filename)
        for page, prefix in ((SITE/'08_speech-separation.html', ''),
                             (SITE/'research/05_exercises_and_audio.html', '../')):
            _check_visible_audio(page, prefix, 'mask_audio', set(names.values()), {'MANIFEST.json'})
    except Exception as exc:
        fail(errors, f'掩码表示实验检查失败：{exc}')


def check_gss_audio(errors):
    """独立核对教学 GSS 的五路 PCM、状态文件和站点副本。"""
    import zipfile
    import math
    import struct
    source, published = GSS_AUDIO_ROOT, SITE / "gss_audio"
    channels = {"source_1.wav": 1, "source_2.wav": 1, "mixture.wav": 2,
                "enhanced_correct.wav": 1, "enhanced_missed.wav": 1}
    expected = set(channels) | {"MANIFEST.json", "STATE.npz"}
    try:
        for folder in (source, published):
            validate_asset_directory(folder, expected, check=True)
        from codes.chapters.ch08.examples.gss_teaching_demo import generate
        generate(source, check=True)
        manifest = _read_audio_manifest(source / "MANIFEST.json")
        if (set(manifest["files"]) != expected - {"MANIFEST.json"}
                or manifest["sample_rate_hz"] != 16000):
            raise ValueError("GSS 清单集合或采样率不符")
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
            if hashlib.sha256(validate_parent_chain(ROOT / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"GSS 生成源码已变化：{name}")
        for name in expected:
            original, copy = source / name, published / name
            if original.is_symlink() or copy.is_symlink() or original.read_bytes() != copy.read_bytes():
                raise ValueError(f"GSS 站点副本不符：{name}")
            if name != "MANIFEST.json" and hashlib.sha256(original.read_bytes()).hexdigest() != manifest["files"][name]["sha256"]:
                raise ValueError(f"GSS 资产摘要不符：{name}")
        if not zipfile.is_zipfile(source / "STATE.npz"):
            raise ValueError("GSS 中间状态不是 NPZ")
        decoded = {}
        for name, n_channels in channels.items():
            record = manifest["files"][name]
            with wave.open(str(source / name), "rb") as wav:
                if (wav.getframerate(), wav.getsampwidth(), wav.getnchannels(),
                        wav.getnframes(), wav.getcomptype()) != (
                            16000, 2, n_channels, record["samples_per_channel"], "NONE"):
                    raise ValueError(f"GSS PCM 格式不符：{name}")
                raw = wav.readframes(record['samples_per_channel'])
            count = record['samples_per_channel']*n_channels
            decoded[name] = struct.unpack('<'+'h'*count, raw)[::n_channels][3200:23200]
        reference = decoded['source_1.wav']
        count = len(reference)
        reference_total = sum(reference)
        centered_ref = [v*count-reference_total for v in reference]
        ref_energy = sum(v*v for v in centered_ref)
        for key in ('mixture', 'enhanced_correct', 'enhanced_missed'):
            values = decoded[key+'.wav']
            total = sum(values)
            centered = [v*count-total for v in values]
            cross = sum(a*b for a, b in zip(centered, centered_ref))
            energy = sum(v*v for v in centered)
            residual_numerator = energy*ref_energy-cross*cross
            if residual_numerator <= 0 or cross == 0:
                raise ValueError('GSS独立评分须有非零投影及残差')
            actual = 10*math.log10(cross*cross/residual_numerator)
            recorded = manifest['pcm_si_sdr_db'][key]
            if (type(recorded) not in (int, float) or not math.isfinite(recorded) or
                    not math.isclose(recorded, actual, rel_tol=1e-12, abs_tol=1e-10)):
                raise ValueError('GSS实际PCM中心化SI-SDR不同：'+key)
        for page, prefix in ((SITE/'08_speech-separation.html', ''),
                             (SITE/'research/05_exercises_and_audio.html', '../')):
            _check_visible_audio(page, prefix, 'gss_audio', set(channels), {'MANIFEST.json', 'STATE.npz'})
    except Exception as exc:
        fail(errors, f"GSS 教学样本检查失败：{exc}")


def check_audio(errors):
    """Independent published PCM inventory, provenance, format and player checks."""
    root = CODE_CHAPTERS
    try:
        manifest = _read_audio_manifest(main_audio_manifest_path(root))
        records = manifest["files"]
        names = {stem + ".wav" for stem in EXPECTED_AUDIO_STEMS}
        if manifest.get("schema_version") != 2 or len(records) != 109 or {r["file"] for r in records} != names:
            fail(errors, "音频清单必须包含独立基线的 109 个 WAV")
        expected_by_chapter = {chapter: set() for chapter in set(MAIN_AUDIO_GROUP_CHAPTER.values())}
        for record in records:
            chapter = MAIN_AUDIO_GROUP_CHAPTER.get(record["group"])
            if chapter is None or record.get("chapter") != chapter:
                fail(errors, f"音频章节归属不符：{record['file']}")
                continue
            expected_by_chapter[chapter].add(record["file"])
        for chapter, expected_files in expected_by_chapter.items():
            members = expected_files | ({"MANIFEST.json"} if chapter == "ch00" else set())
            try:
                validate_asset_directory(root / chapter / "audio", members, check=True)
            except ValueError as exc:
                raise ValueError(f"{chapter} 源音频文件集合不符（或父链/普通文件类型不符）：{exc}") from exc
        validate_asset_directory(root / "ch00/audio", {"MANIFEST.json"}, check=True)
        try:
            validate_asset_directory(SITE / "audio", names, check=True)
        except ValueError as exc:
            raise ValueError(f"站点音频文件集合不符（或父链/普通文件类型不符）：{exc}") from exc
        site_audio_entries = list((SITE / "audio").iterdir())
        if ({p.name for p in site_audio_entries} != names or
                any(not p.is_file() or p.is_symlink() for p in site_audio_entries)):
            fail(errors, "站点音频文件集合不符")
        if set(manifest["groups"]) != {"spatial", "aec", "aec_methods", "aec_subband", "wpe", "separation", "engineering", "tracking",
                                      "correlation", "polarity", "conditioning", "nonlinear", "fractional_array",
                                      "spectral_subtraction", "clock_drift", "interpolation", "alignment_error", "room_decay", "dma_calibration", "doa_ambiguity", "gsc_gate", "aec_dropout", "wpe_predictable", "css_overlap", "agc_blocks", "selection_tradeoff", "math_block"}:
            fail(errors, "音频实验组不符")
        expected_inputs = {"scripts/code_layout.py", "codes/chapters/ch00/examples/generate_audio_samples.py", "codes/chapters/ch00/core/audio_samples.py",
                           "codes/chapters/ch10/core/engineering.py",
                           "codes/chapters/ch06/core/aec.py", "codes/chapters/ch06/core/aec_numeric.py", "codes/chapters/ch06/core/aec_ipnlms.py",
                           "codes/chapters/ch06/core/aec_rls.py", "codes/chapters/ch06/core/aec_kalman_matrix.py",
                           "codes/chapters/ch06/core/aec_subband.py",
                           "codes/chapters/ch07/core/dereverberation.py",
                           "codes/chapters/ch02/core/spectral.py", "codes/chapters/ch02/core/conventions.py",
                           "codes/chapters/ch03/core/geometry.py",
                           "codes/chapters/ch10/core/noise_suppression.py", "codes/chapters/appendix_a/core/math_foundations.py", "codes/chapters/ch05/core/gsc.py",
                           "codes/chapters/ch08/core/css.py", "codes/chapters/ch08/core/separation.py",
                           'codes/chapters/ch00/io_contracts.py'}
        if set(manifest["generator_inputs"]) != expected_inputs:
            fail(errors, "音频生成来源清单不完整")
        for name, expected in manifest["generator_inputs"].items():
            if name not in expected_inputs:
                continue
            if hashlib.sha256(validate_parent_chain(ROOT / name).read_bytes()).hexdigest() != expected:
                fail(errors, f"音频生成源过期：{name}")
        import numpy as np
        for record in records:
            name = record["file"]
            if name not in names:
                continue
            path = main_audio_path(root, record["group"], name)
            blob = path.read_bytes()
            if hashlib.sha256(blob).hexdigest() != record["sha256"] or (SITE / "audio" / name).read_bytes() != blob:
                fail(errors, f"音频摘要或站点副本不符：{name}")
            with wave.open(str(path), "rb") as wav:
                frames, channels = wav.getnframes(), wav.getnchannels()
                if (wav.getframerate(), wav.getsampwidth(), wav.getcomptype()) != (16000, 2, "NONE"):
                    fail(errors, f"音频格式不符：{name}")
                raw = wav.readframes(frames)
            if record["sample_rate_hz"] != 16000 or record["duration_s"] != frames / 16000:
                fail(errors, f"音频清单采样率或时长不符：{name}")
            expected_group = ("math_block" if name.startswith("math_block_") else
                              "selection_tradeoff" if name.startswith("selection_") else
                              "agc_blocks" if name.startswith("agc_blocks_") else
                              "css_overlap" if name.startswith("css_overlap_") else
                              "wpe_predictable" if name.startswith("wpe_predictable_") else
                              "aec_dropout" if name.startswith("aec_dropout_") else
                              "gsc_gate" if name.startswith("gsc_") else
                              "doa_ambiguity" if name.startswith("doa_ambiguity_") else
                              "dma_calibration" if name.startswith("dma_calibration_") else
                              "room_decay" if name.startswith("room_decay_") else
                              "alignment_error" if name.startswith("alignment_") else
                              "spectral_subtraction" if name.startswith("spectral_") else
                              "clock_drift" if name.startswith("clock_") else
                              "fractional_array" if name.startswith("fractional_") else
                              "aec_methods" if name.startswith("aec_methods_") else
                              "aec_subband" if name.startswith("aec_subband_") else name.split("_", 1)[0])
            if record["group"] != expected_group:
                fail(errors, f"音频分组归属不符：{name}")
            if len(raw) != frames * channels * 2 or frames != record["samples"] or channels != record["channels"]:
                fail(errors, f"音频尺寸不符：{name}")
            samples = np.frombuffer(raw, dtype="<i2").astype(float) / 32768
            measured_rms = float(np.sqrt(np.mean(samples**2)))
            if not np.isfinite(record["rms"]) or abs(measured_rms - record["rms"]) > 1e-15:
                fail(errors, f"音频 RMS 不符：{name}")
            if (not np.isfinite(record["peak"]) or np.max(np.abs(samples)) > .80002
                    or abs(float(np.max(np.abs(samples))) - record["peak"]) > 1e-15):
                fail(errors, f"音频峰值不符：{name}")
            if not np.isfinite(record["common_export_gain"]) or not 0 < record["common_export_gain"] <= 1:
                fail(errors, f"音频比较组增益非法：{name}")
            if record["common_export_gain"] != manifest["groups"][record["group"]]["common_export_gain"]:
                fail(errors, f"音频比较组增益不一致：{name}")
            if not 0 <= record["quantization_max_abs_error"] <= .5/32768 + 1e-15:
                fail(errors, f"音频量化误差超限：{name}")
        parser = VisibleMediaParser()
        parser.feed((SITE / "research/05_exercises_and_audio.html").read_text())
        synthetic_players = [p for p in parser.items if (p.get("src") or "").startswith("../audio/")]
        expected_players = Counter({"../audio/" + n: 1 for n in names})
        # The moving-source comparison repeats the existing panning sample.
        expected_players["../audio/tracking_pan.wav"] += 1
        if Counter(p.get("src") for p in synthetic_players) != expected_players:
            fail(errors, "试听控件集合不符")
        for player in parser.items:
            if "autoplay" in player or "controls" not in player or player.get("preload") != "none" or not (player.get("aria-label") or "").strip():
                fail(errors, "试听控件必须有标签和控制、不自动播放或预加载")
    except Exception as exc:
        fail(errors, f"音频检查失败：{exc}")


def main():
    errors: list[str] = []
    notices: list[str] = []
    check_sources(errors, notices)
    check_figures(errors)
    check_imaging_figures(errors)
    check_distributed_figures(errors)
    check_site(errors)
    check_research_site(errors)
    check_audio(errors)
    check_real_audio(errors)
    check_room_audio(errors)
    check_moving_audio(errors)
    check_tracking_audio(errors)
    check_gss_audio(errors)
    check_binaural_audio(errors)
    check_spectral_audio(errors)
    check_stft_audio(errors)
    check_geometry_audio(errors)
    check_focus_audio(errors)
    check_derivative_audio(errors)
    check_apa_audio(errors)
    check_mint_audio(errors)
    check_mask_audio(errors)
    check_noise_audio(errors)
    check_weighted_audio(errors)
    check_response_audio(errors)
    check_imaging_audio(errors)
    check_distributed_audio(errors)
    check_scenario_audio(errors)
    check_combined_html(errors)
    check_pdf(errors, notices)
    for item in notices:
        print(f"QUALITY NOTICE: {item}")
    if errors:
        print("QUALITY CHECK FAILED")
        print("\n".join(f"- {item}" for item in errors))
        raise SystemExit(1)
    print("QUALITY CHECK PASSED")


if __name__ == "__main__":
    main()
