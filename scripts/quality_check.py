#!/usr/bin/env python3
"""发布前质量门禁：占位符、图片、站内链接、合订 HTML 与 PDF。"""

from __future__ import annotations

import re
import sys
import hashlib
import unicodedata
from collections import Counter
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


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids: list[str] = []
        self.links: list[str] = []
        self.images: list[str] = []
        self.h1_count = 0
        self.tags: Counter[str] = Counter()
        self.th_without_scope = 0
        self.source_digest = ""

    def handle_starttag(self, tag, attrs):
        data = dict(attrs)
        self.tags[tag] += 1
        if tag == "h1":
            self.h1_count += 1
        if tag == "th" and data.get("scope") not in {"col", "row"}:
            self.th_without_scope += 1
        if tag == "meta" and data.get("name") == "source-digest":
            self.source_digest = data.get("content", "")
        if "id" in data:
            self.ids.append(data["id"])
        if tag == "a" and "href" in data:
            self.links.append(data["href"])
        if tag == "img" and "src" in data:
            self.images.append(data["src"])


def fail(errors: list[str], message: str):
    errors.append(message)


def check_sources(errors: list[str]):
    for path in sorted(CHAPTERS.glob("*.md")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if EDITING_MARKERS.search(line):
                fail(errors, f"编辑占位：{path.relative_to(ROOT)}:{line_no}: {line.strip()[:100]}")
            if re.search(r"\\\(|\\\)|\\\[|\\\]", line):
                fail(
                    errors,
                    f"不兼容的公式定界符：{path.relative_to(ROOT)}:{line_no}: "
                    f"请使用 $...$ 或 $$...$$",
                )


def check_figures(errors: list[str]):
    refs: Counter[str] = Counter()
    for path in CHAPTERS.glob("*.md"):
        refs.update(re.findall(r"\.\./figures/([^\s)\"]+)", path.read_text(encoding="utf-8")))
    if len(refs) != 33:
        fail(errors, f"正文唯一图片数应为 33，实际 {len(refs)}")
    for name in refs:
        path = ROOT / "figures" / name
        if not path.exists() or path.stat().st_size == 0:
            fail(errors, f"图片缺失或为空：figures/{name}")


def check_site(errors: list[str]):
    pages = sorted(SITE.glob("*.html"))
    if len(pages) != 14:
        fail(errors, f"站点页面数应为 14，实际 {len(pages)}")
    parsed = {}
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
        for landmark in ("header", "main", "footer", "nav"):
            if not parser.tags[landmark]:
                fail(errors, f"页面缺少 {landmark} 语义地标：{path.name}")
        if parser.th_without_scope:
            fail(errors, f"表头缺少 scope：{path.name}: {parser.th_without_scope}")
        if parser.source_digest != expected_digest:
            fail(errors, f"站点页面不是当前源文件生成：{path.name}")
        parsed[path.name] = parser
    for name, parser in parsed.items():
        for href in parser.links:
            url = urlparse(href)
            if url.scheme or href.startswith("mailto:"):
                continue
            target_name = unquote(url.path) or name
            target = SITE / target_name
            if not target.exists():
                fail(errors, f"站内链接目标不存在：{name} -> {href}")
                continue
            if url.fragment and target.suffix == ".html":
                target_parser = parsed.get(target.name)
                if target_parser and url.fragment not in target_parser.ids:
                    fail(errors, f"站内锚点不存在：{name} -> {href}")
        for src in parser.images:
            url = urlparse(src)
            if not url.scheme and not (SITE / unquote(url.path)).resolve().exists():
                fail(errors, f"站点图片不存在：{name} -> {src}")


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
    expected = source_digest()
    if f"源文件 sha256 {expected}" not in text:
        fail(errors, f"合订 HTML 不是当前源文件生成：应含 {expected}")


def source_digest():
    digest = hashlib.sha256()
    paths = sorted(CHAPTERS.glob("*.md"))
    paths += sorted((ROOT / "figures").glob("fig*.png"))
    paths += [ROOT / "scripts" / name for name in
              ("build_pdf.py", "make_figures.py", "make_aec_figures.py")]
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def site_source_digest():
    digest = hashlib.sha256()
    paths = sorted(CHAPTERS.glob("*.md"))
    paths += sorted((ROOT / "figures").glob("fig*.png"))
    paths += [ROOT / "scripts" / name for name in
              ("build_site.py", "make_figures.py", "make_aec_figures.py")]
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def norm(text: str):
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"[\s·：:，,。；;、“”‘’「」『』（）()—–\-]", "", text)


def expected_outline():
    # 期望值必须来自 Markdown 源文件，不能从待检的 combined.html 自证正确。
    import importlib.util
    module_path = ROOT / "scripts" / "build_pdf.py"
    spec = importlib.util.spec_from_file_location("quality_build_pdf", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    _page, outline = module.build_html()
    return [(label, [title for title, _sid in children])
            for label, _cid, children in outline]


def check_pdf(errors: list[str]):
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
    radicals = re.findall(r"[\u2e80-\u2eff\u2f00-\u2fdf]", text)
    if radicals:
        fail(errors, f"PDF 文本层含部首类错误码位：{len(radicals)} 个")
    expected = expected_outline()
    actual = []
    for item in reader.outline:
        if isinstance(item, list):
            if actual:
                actual[-1][1].extend(item)
        else:
            actual.append([item, []])
    if [item[0].title for item in actual] != [label for label, _ in expected]:
        fail(errors, "PDF 顶级书签标题或顺序与合订 HTML 不一致")
    else:
        for (parent, children), (label, expected_children) in zip(actual, expected):
            if [child.title for child in children] != expected_children:
                fail(errors, f"PDF 节书签标题或顺序不一致：{label}")
                continue
            for destination in [parent, *children]:
                page_no = reader.get_destination_page_number(destination)
                title_key = norm(destination.title)
                page_key = norm(extracted[page_no]) if 0 <= page_no < len(extracted) else ""
                if title_key and title_key not in page_key:
                    fail(errors, f"PDF 书签落页未出现标题：{destination.title} -> p{page_no + 1}")
    if "$" in text or "MathJax" in text or re.search(r"\\(?:tag|text|qquad|frac|varepsilon)\b", text):
        fail(errors, "PDF 疑似含未渲染的公式源码")


def main():
    errors: list[str] = []
    check_sources(errors)
    check_figures(errors)
    check_site(errors)
    check_combined_html(errors)
    check_pdf(errors)
    if errors:
        print("QUALITY CHECK FAILED")
        print("\n".join(f"- {item}" for item in errors))
        raise SystemExit(1)
    print("QUALITY CHECK PASSED")


if __name__ == "__main__":
    main()
