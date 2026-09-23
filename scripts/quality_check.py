#!/usr/bin/env python3
"""发布前质量门禁：占位符、图片、站内链接、合订 HTML 与 PDF。"""

from __future__ import annotations

import re
import sys
import hashlib
import json
import wave
import unicodedata
from collections import Counter
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parent.parent
CHAPTERS = ROOT / "chapters"
SITE = ROOT / "site"
DIST = ROOT / "dist"

EDITING_MARKERS = re.compile(
    r"待核实|待补(?:实测|充)?|链接待补|成绩待补|清单#|"
    r"修订说明|编号不动|只调标题层级|新增块一律"
)

EXPECTED_SECTION_COUNTS = {
    "00_overview.md": 10,
    "01_problem-definition.md": 2,
    "02_basics-signal-model.md": 8,
    "03_array-geometry.md": 5,
    "04_doa-estimation.md": 11,
    "05_beamforming.md": 11,
    "06_aec.md": 1,
    "07_wpe-dereverberation.md": 1,
    "08_speech-separation.md": 2,
    "09_source-tracking.md": 5,
    "10_engineering-practice.md": 12,
    "11_selection-guide.md": 7,
    "12_appendix-symbols-math.md": 4,
    "13_appendix-guide.md": 7,
}
# 第 6、7 章的源 h4 单独进入合订目录和 PDF 第三级书签。此表是独立发布
# 基线，不从构建脚本或待检产物反推。
EXPECTED_SUBSECTION_COUNTS = {
    "06_aec.md": 18,
    "07_wpe-dereverberation.md": 3,
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
    ("12_appendix-symbols-math.md", "附录 A · 符号术语数学"),
    ("13_appendix-guide.md", "附录 B · 路径地图与练习"),
]
EXPECTED_CHAPTER_COUNT = 14
EXPECTED_SECTION_COUNT = 86
EXPECTED_SUBSECTION_COUNT = 21
EXPECTED_OUTLINE_ITEM_COUNT = 121
EXPECTED_FIGURE_NUMBERS = set(range(1, 37))
# 研究附站使用独立显式清单，不挤占 14 篇教程或 121 项 PDF 大纲基线。
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
    lines = []
    fence_char = None
    fence_len = 0
    for line in text.splitlines():
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker and fence_char is None:
            fence_char = marker.group(1)[0]
            fence_len = len(marker.group(1))
            lines.append("")
            continue
        if (marker and fence_char is not None and marker.group(1)[0] == fence_char
                and len(marker.group(1)) >= fence_len):
            fence_char = None
            fence_len = 0
            lines.append("")
            continue
        if fence_char is None:
            lines.append(line)
        else:
            lines.append("")
    return "\n".join(lines)


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
    headings = []
    for line in strip_fenced_code(text).splitlines():
        match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if match:
            headings.append((len(match.group(1)), match.group(2).strip()))
    return headings


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


def check_sources(errors: list[str], notices: list[str]):
    paths = sorted(CHAPTERS.glob("*.md"))
    paths += [ROOT / "README.md", ROOT / "README_EN.md", ROOT / "scripts" / "README.md"]
    for path in paths:
        original = path.read_text(encoding="utf-8")
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
        for level, title, _primary in headings:
            if level == section_level:
                current = [title, []]
                children.append(current)
            elif (name in EXPECTED_SUBSECTION_COUNTS
                  and level == section_level + 1 and current is not None):
                current[1].append(title)
        result.append((label, children))
    return result


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
            f"正文图号应为 1..36：缺失 {sorted(EXPECTED_FIGURE_NUMBERS - numbers)}，"
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
            script_name = ("make_figures.py" if number <= 25 or number in (33, 34, 35, 36)
                           else "make_aec_figures.py")
            script_path = ROOT / "scripts" / script_name
            for issue in png_provenance_issues(path, script_path):
                fail(errors, f"PNG 溯源失效：figures/{name}: {issue}")
            if number in (34, 35, 36):
                expected = hashlib.sha256((ROOT / "codes/audio/MANIFEST.json").read_bytes()).hexdigest()
                with Image.open(path) as image:
                    if image.info.get("AudioManifestDigest") != expected:
                        fail(errors, f"图 {number} 音频清单摘要失效")
        except Exception as exc:
            fail(errors, f"图片无法解码：figures/{name}: {exc}")


def check_site(errors: list[str]):
    pages = sorted(SITE.glob("*.html"))
    if len(pages) != 14:
        fail(errors, f"站点页面数应为 14，实际 {len(pages)}")
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

    ordered = ["index.html"] + [
        f"{index:02d}_{slug}.html" for index, slug in (
            (1, "problem-definition"), (2, "basics-signal-model"),
            (3, "array-geometry"), (4, "doa-estimation"),
            (5, "beamforming"), (6, "aec"), (7, "wpe-dereverberation"),
            (8, "speech-separation"), (9, "source-tracking"),
            (10, "engineering-practice"), (11, "selection-guide"),
            (12, "appendix-symbols-math"), (13, "appendix-guide"),
        )
    ]
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
    research_root = ROOT / "codes" / "research"
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
    if re.search(r'href="(?:\./)?(?:0\d|1[0-3])_[^"]+\.html', text):
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
        expected_ids.add(f"ch-{index}")
        section_level = 2 if name == "00_overview.md" else 3
        for level, _title, primary in semantic_heading_ids(documents[name]):
            if (level == section_level
                    or (name in EXPECTED_SUBSECTION_COUNTS
                        and level == section_level + 1)):
                expected_ids.add(f"ch-{index}-{primary}")
        expected_images.update(
            f"../figures/{figure_name}"
            for _alt, figure_name, _number in extract_figure_references(documents[name])
        )
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
    paths += [ROOT / "codes" / "research" / name for name, _ in EXPECTED_RESEARCH_PAGES]
    paths += [ROOT / "scripts" / "build_site.py"]
    paths += sorted(path for path in (ROOT / "scripts" / "vendor" / "mathjax-3.2.2").rglob("*")
                    if path.is_file())
    paths += sorted((ROOT / "figures").glob("fig*.png"))
    paths += [ROOT / "scripts" / name for name in
              ("build_pdf.py", "make_figures.py", "make_aec_figures.py")]
    paths.append(ROOT / "requirements.txt")
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def site_source_digest():
    digest = hashlib.sha256()
    paths = sorted(CHAPTERS.glob("*.md"))
    paths += [ROOT / "codes" / "research" / name for name, _ in EXPECTED_RESEARCH_PAGES]
    paths += sorted((ROOT / "codes" / "audio").glob("*.wav"))
    paths += [ROOT / "codes" / "audio" / "MANIFEST.json"]
    paths += sorted((ROOT / "codes" / "real_audio").glob("*"))
    paths += sorted((ROOT / "figures").glob("fig*.png"))
    paths += [ROOT / "scripts" / name for name in
              ("build_site.py", "make_figures.py", "make_aec_figures.py")]
    paths.append(ROOT / "requirements.txt")
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
        notices.append("PDF 未包含完整结构标签；当前 Chrome 打印链只验收语言、文本层和书签，不能据此声称阅读顺序已通过")
    width = float(reader.pages[0].mediabox.width)
    height = float(reader.pages[0].mediabox.height)
    if abs(width - 595.28) > 2 or abs(height - 841.89) > 2:
        fail(errors, f"PDF 纸型不是 A4：{width:.1f}×{height:.1f} pt")
    extracted = []
    for page_no, page in enumerate(reader.pages, 1):
        extracted.append(page.extract_text() or "")
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
    actual = pdf_outline_tree(reader.outline)
    if [item[0].title for item in actual] != [label for label, _ in expected]:
        fail(errors, "PDF 顶级书签标题或顺序与源 Markdown 不一致")
    else:
        for (parent, children), (label, expected_children) in zip(actual, expected):
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
            for destination in destinations:
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


def check_real_audio(errors):
    """Independent inventory and PCM/attribution checks for the DEMAND excerpt."""
    import numpy as np
    root = ROOT / "codes/real_audio"
    try:
        for folder in (root, SITE / "real_audio"):
            if {p.name for p in folder.iterdir() if p.is_file()} != EXPECTED_REAL_AUDIO_FILES:
                fail(errors, "真实录音文件或许可集合不符")
        for name in EXPECTED_REAL_AUDIO_FILES:
            source, published = root / name, SITE / "real_audio" / name
            if source.is_symlink() or published.is_symlink() or source.read_bytes() != published.read_bytes():
                fail(errors, f"真实录音站点副本不符：{name}")
        attribution = (root / "ATTRIBUTION.txt").read_text(encoding="utf-8")
        for required in ("Joachim Thiemann", "Nobutaka Ito", "Emmanuel Vincent",
                         "1227121", "creativecommons.org/licenses/by-sa/3.0"):
            if required not in attribution:
                fail(errors, f"真实录音署名缺少 {required}")
        if "creativecommons.org/licenses/by-sa/3.0" not in (root / "LICENSE.txt").read_text(encoding="utf-8"):
            fail(errors, "真实录音许可地址缺失")
        manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
        expected_inputs = {"codes/array_tutorial/real_recordings.py", "codes/examples/prepare_real_recordings.py"}
        if set(manifest["generator_inputs"]) != expected_inputs:
            fail(errors, "真实录音生成来源清单不符")
        for name in expected_inputs:
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != manifest["generator_inputs"].get(name):
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
        class Players(HTMLParser):
            def __init__(self):
                super().__init__()
                self.items = []
            def handle_starttag(self, tag, attrs):
                if tag == "audio":
                    self.items.append(dict(attrs))
        parser = Players()
        parser.feed((SITE / "research/05_exercises_and_audio.html").read_text())
        real = [p for p in parser.items if not (p.get("src") or "").startswith("../audio/")]
        if len(real) != 3 or {p.get("src") for p in real} != {
                "../real_audio/" + name for name, channels in EXPECTED_REAL_AUDIO_CHANNELS.items() if channels == 1}:
            fail(errors, "真实录音试听控件集合不符")
        if any("autoplay" in p or "controls" not in p or p.get("preload") != "none"
               or not p.get("aria-label") for p in real):
            fail(errors, "真实录音试听控件必须有标签和控制且不自动播放")
    except Exception as exc:
        fail(errors, f"真实录音检查失败：{exc}")


def check_audio(errors):
    """Independent published PCM inventory, provenance, format and player checks."""
    root = ROOT / "codes/audio"
    try:
        manifest = json.loads((root / "MANIFEST.json").read_text())
        records = manifest["files"]
        names = {stem + ".wav" for stem in EXPECTED_AUDIO_STEMS}
        if len(records) != 44 or {r["file"] for r in records} != names:
            fail(errors, "音频清单必须包含独立基线的 44 个 WAV")
        if {p.name for p in root.glob("*.wav")} != names or {p.name for p in (SITE / "audio").glob("*.wav")} != names:
            fail(errors, "源音频或站点音频文件集合不符")
        if set(manifest["groups"]) != {"spatial", "aec", "wpe", "separation", "engineering", "tracking",
                                      "correlation", "polarity", "conditioning", "nonlinear", "fractional_array"}:
            fail(errors, "音频实验组不符")
        expected_inputs = {"codes/examples/generate_audio_samples.py", "codes/array_tutorial/audio_samples.py",
                           "codes/array_tutorial/aec.py", "codes/array_tutorial/dereverberation.py",
                           "codes/array_tutorial/spectral.py", "codes/array_tutorial/conventions.py",
                           "codes/array_tutorial/geometry.py"}
        if set(manifest["generator_inputs"]) != expected_inputs:
            fail(errors, "音频生成来源清单不完整")
        for name, expected in manifest["generator_inputs"].items():
            if name not in expected_inputs:
                continue
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                fail(errors, f"音频生成源过期：{name}")
        import numpy as np
        for record in records:
            name = record["file"]
            if name not in names:
                continue
            path = root / name
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
            expected_group = "fractional_array" if name.startswith("fractional_") else name.split("_", 1)[0]
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
        class AudioParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.players = []
            def handle_starttag(self, tag, attrs):
                if tag == "audio":
                    self.players.append(dict(attrs))
        parser = AudioParser()
        parser.feed((SITE / "research/05_exercises_and_audio.html").read_text())
        synthetic_players = [p for p in parser.players if (p.get("src") or "").startswith("../audio/")]
        if len(synthetic_players) != 44 or {p.get("src") for p in synthetic_players} != {"../audio/" + n for n in names}:
            fail(errors, "试听控件集合不符")
        for player in parser.players:
            if "autoplay" in player or "controls" not in player or player.get("preload") != "none" or not player.get("aria-label"):
                fail(errors, "试听控件必须有标签和控制、不自动播放或预加载")
    except Exception as exc:
        fail(errors, f"音频检查失败：{exc}")


def main():
    errors: list[str] = []
    notices: list[str] = []
    check_sources(errors, notices)
    check_figures(errors)
    check_site(errors)
    check_research_site(errors)
    check_audio(errors)
    check_real_audio(errors)
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
