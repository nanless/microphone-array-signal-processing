#!/usr/bin/env python3
"""发布前质量门禁：占位符、图片、站内链接、合订 HTML 与 PDF。"""

from __future__ import annotations

import re
import sys
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

    def handle_starttag(self, tag, attrs):
        data = dict(attrs)
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
    for path in pages:
        page_text = path.read_text(encoding="utf-8")
        parser = PageParser()
        parser.feed(page_text)
        duplicates = [key for key, count in Counter(parser.ids).items() if count > 1]
        if duplicates:
            fail(errors, f"重复 HTML id：{path.name}: {duplicates[:5]}")
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
    if (reader.metadata or {}).get("/Title") != "麦克风阵列信号处理教程":
        fail(errors, "PDF 缺少正确的标题元数据")
    width = float(reader.pages[0].mediabox.width)
    height = float(reader.pages[0].mediabox.height)
    if abs(width - 595.28) > 2 or abs(height - 841.89) > 2:
        fail(errors, f"PDF 纸型不是 A4：{width:.1f}×{height:.1f} pt")
    top = [item for item in reader.outline if not isinstance(item, list)]
    if len(top) != 14:
        fail(errors, f"PDF 顶级书签应为 14，实际 {len(top)}")
    child_count = sum(len(item) for item in reader.outline if isinstance(item, list))
    if child_count < 50:
        fail(errors, f"PDF 节级书签过少：{child_count}")
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
