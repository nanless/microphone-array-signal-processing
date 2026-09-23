#!/usr/bin/env python3
"""合订本构建脚本：14 篇文档 → 单页 HTML → Chrome 无头打印 A4 PDF。

做的事情（按顺序）：
  1. 读 chapters/ 14 篇 Markdown，用 markdown 库转 HTML（数学段先 shield 再贴回，
     与 build_site.py 同逻辑，保证两端渲染一致）。
  2. 每篇包进 <div class="chap">，篇标题记 id="ch-{i}"；统一标题层级后，
     编号小节使用 ch-{i}-sec-x-y 稳定标识，并保留旧顺序别名。
  3. 跨篇 .md 链改成合订本内部锚点；分章导航块和页脚行删除；图片
     ../figures/ 原样透传（combined.html 与 figures/ 同处仓库根的
     相邻目录，相对关系成立；单发 HTML 给别人会缺图，要分发请发 PDF）。
  4. 在 dist/ 中写临时 HTML 并打印临时 PDF，检查页数、末页、公式文本和打印尺度，
     再写三级书签并校验链接。全部通过后成对替换正式 HTML/PDF；可捕获的发布异常
     会恢复旧文件，不承诺断电时的原子性。

用法（仓库根目录）：
    .venv/bin/python scripts/build_pdf.py                 # 全量：HTML + PDF + 书签
    .venv/bin/python scripts/build_pdf.py --html-only     # 只合 HTML，不调 Chrome
    .venv/bin/python scripts/build_pdf.py --pdf-only      # 只打印（复用现有 HTML）
    .venv/bin/python scripts/build_pdf.py --no-bookmarks  # 跳过书签（调排版时省时间）
    CHROME_BIN=/path/to/chrome .venv/bin/python scripts/build_pdf.py  # 非 macOS

依赖：markdown、pypdf（根 README 依赖行）；Chrome（macOS 默认路径，余者走环境变量
或 which 回退）。PDF 公式使用仓库内固定版本 MathJax 3.2.2 与 WOFF 字体，构建不联网。

已知边界（诚实写在前面）：
  - PDF 书签与合订本 TOC 包含篇/节两级；第 6、7 章再收入源 h4 作为第三级。
  - 页眉页脚关闭（--no-pdf-header-footer），PDF 内无页码——Chrome 无头打印不支持
    CSS 生成页码，要页码得换 WeasyPrint/Prince 链路。
  - 节书签定位靠"节标题文本首次出现页"逐章顺序搜索，标题串进正文会指偏，
    偏差一般不超过 1 页；章书签用同样方法，目录页自动排除。
  - Chrome 153+ 实测写完 PDF 进程常不退出：本脚本看文件大小（稳定 15 秒即收工，
    主动 kill），不傻等进程结束；超时/页数（<100 页）判失败。
"""
import argparse
import datetime
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

try:
    from scripts import build_site
except ModuleNotFoundError:  # 直接执行脚本时使用同目录模块。
    import build_site

ROOT = Path(__file__).parent.parent
SRC = ROOT / "chapters"
OUT = ROOT / "dist"
MATHJAX_DIR = ROOT / "scripts" / "vendor" / "mathjax-3.2.2"
MATHJAX_SCRIPT = MATHJAX_DIR / "tex-mml-chtml.js"
MATHJAX_FONT_DIR = MATHJAX_DIR / "output" / "chtml" / "fonts" / "woff-v2"
MATHJAX_SCRIPT_SHA256 = "300480069078b5892d2363a2b65e2dfbbf30fe5c80f83edbfecf4610fd093862"
MATHJAX_BOLDSYMBOL = MATHJAX_DIR / "input" / "tex" / "extensions" / "boldsymbol.js"
MATHJAX_BOLDSYMBOL_SHA256 = "d6771fee0772db2657796c8d0e20e1878bb3237f6d3ed1e828e1834a4ff743ca"


def check_mathjax_assets():
    """Require the exact local script and complete CHTML font family."""

    if not MATHJAX_SCRIPT.is_file():
        raise SystemExit(f"缺少本地 MathJax：{MATHJAX_SCRIPT}")
    if hashlib.sha256(MATHJAX_SCRIPT.read_bytes()).hexdigest() != MATHJAX_SCRIPT_SHA256:
        raise SystemExit("本地 MathJax 脚本摘要不符；不能打印 PDF")
    if (not MATHJAX_BOLDSYMBOL.is_file() or
            hashlib.sha256(MATHJAX_BOLDSYMBOL.read_bytes()).hexdigest() != MATHJAX_BOLDSYMBOL_SHA256):
        raise SystemExit("本地 MathJax boldsymbol 扩展缺失或摘要不符；不能打印 PDF")
    fonts = sorted(MATHJAX_FONT_DIR.glob("MathJax_*.woff"))
    if len(fonts) != 23 or any(path.stat().st_size == 0 for path in fonts):
        raise SystemExit("本地 MathJax WOFF 字体不完整；不能打印 PDF")
    if not (MATHJAX_DIR / "LICENSE").is_file():
        raise SystemExit("本地 MathJax 许可文本缺失；不能打印 PDF")

CHAPTERS = [
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

# 第 6、7 章的源 h4 是较长算法章中不可省略的导航层。其余章节仍停在篇/节两级，
# 避免把算例和局部说明无差别塞入印刷目录与 PDF 书签。
PDF_THIRD_LEVEL_FILES = {"06_aec.md", "07_wpe-dereverberation.md"}
PDF_THIRD_LEVEL_CHAPTER_IDS = {
    f"ch-{index}" for index, (name, _label) in enumerate(CHAPTERS)
    if name in PDF_THIRD_LEVEL_FILES
}

CSS = """
@page{size:A4;margin:16mm 15mm 18mm}
body{font-family:"STHeiti","Hiragino Sans GB","Microsoft YaHei",sans-serif;font-size:16px;line-height:1.75;color:#1a1a2e;max-width:860px;margin:0 auto;padding:24px}
p{margin:0 0 .85em;break-inside:avoid;orphans:2;widows:2}li>p{margin:.3em 0}
img{max-width:100%;height:auto;display:block;margin:12px auto}
table{border-collapse:collapse;margin:12px 0;display:block;overflow-x:visible;max-width:100%}
th,td{border:1px solid #dfe3ea;padding:5px 9px;font-size:13.5px;text-align:left}
th{background:#f0f4f9}code{font-family:"STHeiti",monospace;background:#f0f3f7;padding:1px 5px;border-radius:4px;font-size:13px}
pre{font-family:"STHeiti",monospace;background:#f4f6f9;padding:12px;border-radius:6px;overflow-x:visible;white-space:pre-wrap}
pre code{background:none;padding:0}
blockquote{border-left:3px solid #2f6db3;margin:12px 0;padding:6px 12px;background:#f2f7fd}
a{color:#2f6db3;text-decoration:none}
h1{font-size:24px;border-bottom:2px solid #1a1a2e;padding-bottom:6px}
h2{font-size:20px;margin-top:30px;border-bottom:1px solid #e5e8ee;padding-bottom:5px}
h3{font-size:16.5px}h4{font-size:15px}
.cover{text-align:center;padding:120px 0 60px;break-after:page;page-break-after:always}
.cover h1{font-size:34px;border:none}
.cover .sub{font-size:17px;color:#444;margin-top:18px}
.cover .meta{font-size:13.5px;color:#777;margin-top:40px}
.chap{page-break-before:always}
.toc-page>h1{font-size:28px;margin-bottom:18px}
.toc{columns:2;column-gap:28px;padding-left:20px}
.toc>li{margin:5px 0;break-inside:avoid-column}
.toc .sec{font-size:12.5px;color:#333;padding-left:18px}
.toc .subsec{font-size:11.5px;color:#555;padding-left:18px}
.anchor-alias{display:none}
.book-end{text-align:center;color:#777;margin:36px 0 8px;font-size:13px}
.book-ending{break-inside:avoid;page-break-inside:avoid}
/* MathJax 的 serif 中文回退在部分 macOS 字体中会生成部首码位的 ToUnicode。 */
mjx-mtext>mjx-utext{font-family:MJXZERO,"STHeiti","Hiragino Sans GB","Microsoft YaHei",sans-serif!important}
@media print{
body{max-width:none;margin:0;padding:0}
.chap{page-break-before:always}
table{display:table;width:100%;table-layout:fixed}
thead{display:table-header-group}
tr{break-inside:avoid}
td,th{word-break:break-word;overflow-wrap:anywhere}
/* MathJax 的行内长式与展示式均需留在 A4 正文宽度内。 */
mjx-container{font-size:80%!important;max-width:100%}
h1,h2,h3,h4{break-after:avoid}
.chap>h1{margin:0 0 3mm;line-height:1.3}
.chap>h2:first-of-type{margin-top:3mm;margin-bottom:2mm;line-height:1.3}
blockquote,pre{break-inside:avoid}
/* A4 可用高度为 263 mm；紧凑的章/首节标题和图外间距共占约 33 mm。 */
img{max-height:225mm;object-fit:contain}
th,code,pre,blockquote{-webkit-print-color-adjust:exact;print-color-adjust:exact}
}
"""


INLINE_CODE_RE = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)", re.S)
ALLOWED_LINK_SCHEMES = {"http", "https", "mailto"}


def protect_code(md):
    """暂存围栏和行内代码，使数学正则不会改写代码中的 `$...$`。"""
    repo = []

    def stash(value):
        repo.append(value)
        return f"@@CODETOKEN{len(repo) - 1}@@"

    lines = md.splitlines(keepends=True)
    protected = []
    index = 0
    while index < len(lines):
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", lines[index])
        if not marker:
            protected.append(lines[index])
            index += 1
            continue
        fence_char = marker.group(1)[0]
        fence_len = len(marker.group(1))
        block = [lines[index]]
        index += 1
        while index < len(lines):
            block.append(lines[index])
            closing = re.match(r"^\s{0,3}(`{3,}|~{3,})\s*$", lines[index].rstrip("\r\n"))
            index += 1
            if (closing and closing.group(1)[0] == fence_char
                    and len(closing.group(1)) >= fence_len):
                break
        protected.append(stash("".join(block)))

    text = "".join(protected)
    text = INLINE_CODE_RE.sub(lambda match: stash(match.group(0)), text)
    return text, repo


def restore_code(text, repo):
    return re.sub(r"@@CODETOKEN(\d+)@@", lambda m: repo[int(m.group(1))], text)


def validate_url_schemes(html):
    """只允许文档内相对链接以及 http、https、mailto 外链。"""
    class TargetParser(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.targets = []

        def handle_starttag(self, _tag, attrs):
            self.targets.extend((key, value) for key, value in attrs
                                if key.lower() in {"href", "src"} and value is not None)

    parser = TargetParser()
    parser.feed(html)
    for attribute, value in parser.targets:
        parsed = urlparse(value.strip())
        scheme = parsed.scheme.lower()
        if parsed.netloc and not scheme:
            raise ValueError(f"不允许省略协议的外部 {attribute}：{value}")
        if scheme and scheme not in ALLOWED_LINK_SCHEMES:
            raise ValueError(f"不安全或不支持的 {attribute} 协议：{scheme}")


def shield_math(md):
    """数学段暂存：防 markdown 吃下划线（_x_→斜体），防浏览器吞 <（i<j）。
    转完 markdown 再原样贴回。"""
    repo = []

    def stash(m):
        repo.append(m.group(0).replace("<", r"\lt "))
        return f"@@MATH{len(repo) - 1}@@"

    md, code_repo = protect_code(md)
    md = re.sub(r"\$\$.*?\$\$", stash, md, flags=re.S)
    md = re.sub(r"\$[^$]+?\$", stash, md, flags=re.S)
    md = restore_code(md, code_repo)
    return md, repo


def unshield_math(html, repo):
    def back(m):
        return repo[int(m.group(1))]
    return re.sub(r"@@MATH(\d+)@@", back, html)


def plain_text(html):
    """去标签取纯文本（书签标题用）。"""
    t = re.sub(r"<[^>]+>", "", html)
    return re.sub(r"\s+", " ", t).strip()


def source_digest():
    """发布输入的稳定摘要；避免把生成物自身所在提交写回生成物造成循环漂移。"""
    digest = hashlib.sha256()
    paths = sorted(SRC.glob("*.md"))
    paths += [ROOT / "codes" / "research" / name for name, _ in build_site.RESEARCH]
    paths += [ROOT / "scripts" / "build_site.py"]
    paths += sorted(path for path in MATHJAX_DIR.rglob("*") if path.is_file())
    paths += sorted((ROOT / "figures").glob("fig*.png"))
    paths += [Path(__file__), ROOT / "scripts" / "make_figures.py",
              ROOT / "scripts" / "make_aec_figures.py", ROOT / "requirements.txt"]
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:12]


HTML_TO_CH = {fname.replace(".md", ".html"): i for i, (fname, _) in enumerate(CHAPTERS)}
REPOSITORY_BLOB_BASE = "https://github.com/nanless/microphone-array-signal-processing/blob/main/"


def heading_anchor(text, fallback_index):
    """与分页站一致：编号标题用 sec-x-y，其余用稳定内容摘要。"""
    label = plain_text(text)
    numbered = re.match(r"^(\d+(?:\.\d+)+)(?=\s|$)", label)
    if numbered:
        return "sec-" + numbered.group(1).replace(".", "-")
    if label:
        return "sec-u-" + hashlib.sha1(label.encode("utf-8")).hexdigest()[:10]
    return f"sec-{fallback_index}"


def rewrite_book_links(html):
    """把分篇链接改成合订本内部链接，兼容新旧标识。"""
    def repl(m):
        fname, fragment, text = m.group(1), m.group(2), m.group(3)
        idx = HTML_TO_CH.get(fname)
        if idx is None:
            return m.group(0)
        if not fragment:
            target = f"ch-{idx}"
        elif re.fullmatch(r"sec-(?:\d+(?:-\d+)*|u-[0-9a-f]{10}(?:-\d+)?)", fragment):
            # sec-1 是分篇页标题；合订时该重复标题被外层 ch-N 标题替代。
            target = f"ch-{idx}" if fragment == "sec-1" else f"ch-{idx}-{fragment}"
        else:
            raise ValueError(f"合订本无法映射章节锚点：{fname}#{fragment}")
        return f'<a href="#{target}">{text}</a>'

    return re.sub(
        r'<a href="(?:\./)?([^"#/]+\.html)(?:#([^"]+))?">(.*?)</a>',
        repl, html, flags=re.S)


def rewrite_repository_links(html, source_path=None):
    """把源码相对链接改为可移植的仓库链接，避免 PDF 泄露构建机路径。"""
    source_path = source_path or SRC / "00_overview.md"
    chapters = {(SRC / name).resolve(): i for i, (name, _) in enumerate(CHAPTERS)}

    def transform(href):
        parsed, target = build_site.local_link_target(href, source_path)
        if (not parsed.scheme and not parsed.netloc and not parsed.path
                and not parsed.query and parsed.fragment):
            # A chapter-local link must receive the same chapter prefix as
            # its heading when many standalone pages become one document.
            target = Path(source_path).resolve()
        if target is None:
            return href
        if target in chapters and not parsed.query:
            idx = chapters[target]
            fragment = parsed.fragment
            if not fragment or fragment == "sec-1":
                return f"#ch-{idx}"
            if re.fullmatch(r"sec-(?:\d+(?:-\d+)*|u-[0-9a-f]{10}(?:-\d+)?)", fragment):
                return f"#ch-{idx}-{fragment}"
            raise ValueError(f"合订本无法映射章节锚点：{target.name}#{fragment}")
        return build_site.repository_url(parsed, target)

    return build_site.rewrite_href_targets(html, transform)


def remove_page_info(html):
    """删除分篇页脚信息及其引用块，避免打印出空色条。"""
    html = re.sub(
        r"<blockquote>\s*<p>📄.*?</p>\s*</blockquote>",
        "", html, flags=re.S)
    return re.sub(r"<p>📄 本篇信息.*?</p>", "", html, flags=re.S)


def append_book_end(html):
    """Keep the final paragraph and end marker together, not on a marker-only page."""
    pattern = r'(<p>(?:(?!<p>).)*?</p>\s*(?:<hr\s*/?>\s*)?)$'
    result, count = re.subn(pattern, r'<div class="book-ending">\1<div class="book-end">全书完</div></div>', html, flags=re.S)
    if count != 1:
        raise ValueError("末章必须以正文段落收尾，才能保持结束标记与正文同页")
    return result


def resolve_build_date(explicit=None):
    """返回可复现的封面日期；显式参数优先，其次 SOURCE_DATE_EPOCH。"""
    if explicit:
        try:
            return datetime.date.fromisoformat(explicit).isoformat()
        except ValueError as exc:
            raise ValueError("--build-date 必须是 YYYY-MM-DD") from exc
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch is not None:
        try:
            return datetime.datetime.fromtimestamp(int(epoch), datetime.timezone.utc).date().isoformat()
        except (ValueError, OverflowError, OSError) as exc:
            raise ValueError("SOURCE_DATE_EPOCH 必须是有效的 Unix 秒数") from exc
    return datetime.date.today().isoformat()


def build_html(build_date=None):
    check_mathjax_assets()
    """合 14 篇为单页 HTML。返回 (page, outline)，outline 为
    [(章label, 章id, [(节title, 节id, [(子节title, 子节id), ...]), ...]), ...]，
    供印刷目录和书签定位用。"""
    import markdown
    body_parts = []
    outline = []  # 章级
    n_imgs = 0
    for i, (fname, label) in enumerate(CHAPTERS):
        md, repo = shield_math((SRC / fname).read_text(encoding="utf-8"))
        html = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
        html = unshield_math(html, repo)
        validate_url_schemes(html)
        # 与分页站一致：编号标题用语义标识，旧 sec-N 作别名。
        source_heading = [0]
        used_heading_ids = {}

        def tag_source_heading(m):
            source_heading[0] += 1
            base = heading_anchor(m.group(2), source_heading[0])
            used_heading_ids[base] = used_heading_ids.get(base, 0) + 1
            primary = (base if used_heading_ids[base] == 1 else
                       f"{base}-{used_heading_ids[base]}")
            legacy = f"sec-{source_heading[0]}"
            alias = ("" if primary == legacy else
                     f'<span id="ch-{i}-{legacy}" class="anchor-alias"></span>')
            return (f'{alias}<{m.group(1)} id="ch-{i}-{primary}">'
                    f'{m.group(2)}</{m.group(1)}>')

        html = re.sub(r"<(h[1-4])>(.*?)</\1>", tag_source_heading, html, flags=re.S)
        html = rewrite_repository_links(html, SRC / fname)
        html = rewrite_book_links(html)
        # 每篇开头的引用块都是分篇导航。用位置边界删除，不依赖某一种中文句式。
        html = re.sub(
            r"^\s*(?:<blockquote>.*?</blockquote>\s*)+(?:<hr\s*/?>\s*)?",
            "", html, flags=re.S)
        # 篇末“本篇信息”位于引用块中；整块删除，避免只删段落后留下空色条。
        html = remove_page_info(html)
        # 外层已经提供篇标题，删掉源文重复标题。导读以 h2 为节；其余篇章
        # 原文以 h2 作篇标题、h3 作节标题，因此在合订本里提升一级。
        # 外层已提供篇标题；保留原标题 id 作隐藏别名，
        # 避免分页站点中指向篇标题的合法深链在合订本中失效。
        html = re.sub(
            r'<h[12][^>]*\bid="([^"]+)"[^>]*>.*?</h[12]>',
            r'<span id="\1" class="anchor-alias"></span>',
            html, count=1, flags=re.S)
        if i > 0:
            html = re.sub(
                r"<h([34])([^>]*)>(.*?)</h\1>",
                lambda m: (f"<h{int(m.group(1)) - 1}{m.group(2)}>{m.group(3)}"
                           f"</h{int(m.group(1)) - 1}>"),
                html, flags=re.S)

        # h2 进入目录和 PDF 书签；第 6、7 章的 h3 作为其前一 h2 的子节。
        secs = []

        def tag_outline_heading(m):
            level, attrs, inner = int(m.group(1)), m.group(2), m.group(3)
            match = re.search(r'\bid="([^"]+)"', attrs)
            if level == 2:
                fallback = f"ch-{i}-s{len(secs)}"
            else:
                parent_index = max(0, len(secs) - 1)
                child_index = len(secs[-1][2]) if secs else 0
                fallback = f"ch-{i}-s{parent_index}-{child_index}"
            section_id = match.group(1) if match else fallback
            if level == 2:
                secs.append((plain_text(inner), section_id, []))
            elif fname in PDF_THIRD_LEVEL_FILES and secs:
                secs[-1][2].append((plain_text(inner), section_id))
            if match:
                return f"<h{level}{attrs}>{inner}</h{level}>"
            return f'<h{level} id="{section_id}"{attrs}>{inner}</h{level}>'

        html = re.sub(r"<h([23])([^>]*)>(.*?)</h\1>", tag_outline_heading,
                      html, flags=re.S)
        n_imgs += len(re.findall(r"<img ", html))
        if i == len(CHAPTERS) - 1:
            html = append_book_end(html)
        body_parts.append(f'<div class="chap" id="ch-{i}"><h1>{label}</h1>{html}</div>')
        outline.append((label, f"ch-{i}", secs))
    # 篇/节目录；第 6、7 章再显示第三级子节。
    toc = ['<section class="toc-page"><h1>目录</h1><ul class="toc">']
    for label, cid, secs in outline:
        toc.append(f"<li><a href=\"#{cid}\">{label}</a>")
        if secs:
            toc.append('<ul class="sec">')
            for title, sid, subsecs in secs:
                toc.append(f'<li><a href="#{sid}">{title}</a>')
                if subsecs:
                    toc.append('<ul class="subsec">')
                    toc.extend(f'<li><a href="#{subid}">{subtitle}</a></li>'
                               for subtitle, subid in subsecs)
                    toc.append('</ul>')
                toc.append('</li>')
            toc.append("</ul>")
        toc.append("</li>")
    toc.append("</ul></section>")
    date_s = resolve_build_date(build_date)
    cover = (f'<div class="cover"><h1>麦克风阵列信号处理教程</h1>'
             f'<div class="sub">深入浅出 · 从阵列摆位到工程选型（合订本）</div>'
             f'<div class="meta">构建日期 {date_s} · 源文件 sha256 {source_digest()} · '
             f'共 14 篇：导读、11 章正文、2 篇附录</div></div>')
    page = ("<!DOCTYPE html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
            f"<title>麦克风阵列信号处理教程（合订本）</title><style>{CSS}</style>"
            "<script>\nwindow.MathJax = {tex: {inlineMath: [['$', '$'], ['\\\\(', '\\\\)']], displayMath: [['$$', '$$']]}, chtml: {fontURL: '../scripts/vendor/mathjax-3.2.2/output/chtml/fonts/woff-v2'}};\n</script>"
            "<script defer src=\"../scripts/vendor/mathjax-3.2.2/tex-mml-chtml.js\"></script>"
            "</head><body>" + cover + "\n".join(toc) + "\n".join(body_parts)
            + '</body></html>')
    n_secs = sum(len(s) for _, _, s in outline)
    n_subsecs = sum(len(subsecs) for _, _, secs in outline
                    for _title, _sid, subsecs in secs)
    unique_imgs = len(set(re.findall(r'<img [^>]*src="([^"]+)"', page)))
    print(f"合订：{len(CHAPTERS)} 篇 / {n_secs} 节 / {n_subsecs} 子节 / "
          f"{unique_imgs} 张唯一图片（{n_imgs} 次引用）")
    return page, outline


def norm(s):
    # 两步归一：① NFKC 折叠大部分兼容汉字（如双⽿→双耳）；
    # ② CJK 部首扩展区（U+2E80–U+2EFF，如⻅⻨⻛⻚⻆）没有兼容分解，
    #    NFKC 折不动，实测出现一个补一个（Chrome 按页子集化字体所致）。
    import unicodedata
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(str.maketrans({"⻅": "见", "⻨": "麦", "⻛": "风",
                                   "⻚": "页", "⻆": "角"}))
    s = s.replace(" ", "").replace("\n", "").replace("·", "")
    # PDF 提取会把全角/半角标点、弯引号和破折号互换；书签定位只关心
    # 标题的字母、数字与汉字序列，因此统一忽略这些排版标点。
    return re.sub(r'[：:，,。；;、“”‘’「」『』（）()—–\-]', '', s)


def locate(texts, key, start, skip):
    """在 texts[start:] 里找 key 所在页（跳过 skip 集）。
    两轮：先找"标题独占一整行"的页（h2 块级元素换行，正文串文是行内，
    不会被误命中）；找不到再退回子串首次出现。找不到返回 None。"""
    nkey = norm(key)
    for i in range(start, len(texts)):
        if i in skip:
            continue
        for line in texts[i].split("\n"):
            if norm(line) == nkey or norm(line).startswith(nkey):
                return i
    for i in range(start, len(texts)):
        if i in skip:
            continue
        if key in texts[i] or (nkey and nkey in norm(texts[i])):
            return i
    # Chrome 打印后，PDF 文本提取有时会完全丢掉 MathJax 公式，只留下公式
    # 两侧的文字。例如标题 ``7.1.1 $\Delta$ 与 $K$ 的参数换算`` 会变成
    # ``7.1.1  与  的参数换算``。此时用“删去行内公式后的标题”定位；节号与
    # 其余中文仍可避免误命中正文中的普通句子。
    if "$" in key:
        mathless = norm(re.sub(r"\$[^$]*\$", "", key))
        if len(mathless) >= 8:
            for i in range(start, len(texts)):
                if i in skip:
                    continue
                for line in texts[i].split("\n"):
                    nline = norm(line)
                    if nline == mathless or nline.startswith(mathless):
                        return i
    # Chrome/PDF 字体子集化有时会替换破折号或引号，长标题也可能被换行拆开。
    # 节号加标题前缀在同一章内足够唯一；只在完整匹配失败后使用，避免正文
    # 偶然出现标题后半句时抢先命中。
    prefix = nkey[:10]
    if len(prefix) >= 8:
        for i in range(start, len(texts)):
            if i not in skip and prefix in norm(texts[i]):
                return i
    return None


def add_bookmarks(pdf_path, outline):
    """按 outline 写三级大纲（篇 + 节 + 指定子节），校验后原子替换 PDF。"""
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError as e:
        raise SystemExit("缺少 pypdf，无法写入并校验书签") from e
    reader = PdfReader(str(pdf_path))
    texts = []
    for p in reader.pages:
        try:
            texts.append(p.extract_text() or "")
        except Exception:
            texts.append("")
    ch_labels = [label for label, _, _ in outline]
    toc_pages = {i for i, t in enumerate(texts)
                 if sum(1 for label in ch_labels if norm(label) in norm(t)) >= 3}
    writer = PdfWriter()
    writer.append(reader)
    writer.add_metadata({
        "/Title": "麦克风阵列信号处理教程",
        "/Subject": "从阵列摆位到工程选型",
        "/Creator": "scripts/build_pdf.py",
        "/SourceDigest": source_digest(),
    })
    from pypdf.generic import NameObject, TextStringObject
    writer._root_object.update({NameObject("/Lang"): TextStringObject("zh-CN")})
    prev, n_ch, n_sec, n_subsec = -1, 0, 0, 0
    for label, _cid, secs in outline:
        page_no = locate(texts, label, prev + 1, toc_pages)
        if page_no is None:
            print(f"章书签跳过（找不到起始页）：{label}")
            continue
        parent = writer.add_outline_item(label, page_no)
        prev, n_ch = page_no, n_ch + 1
        sprev = page_no
        for title, _sid, subsecs in secs:
            # 从本节往后找标题出现处；注意用 sp 而不是 sp+1——一页可能挤多个节，
            # +1 会跳过同页后面的节。标题串进正文时会指偏（一般 ≤1 页）。
            sp = locate(texts, title, sprev, toc_pages)
            if sp is None:
                continue
            section_parent = writer.add_outline_item(title, sp, parent=parent)
            sprev, n_sec = sp, n_sec + 1
            subprev = sp
            for subtitle, _subid in subsecs:
                subpage = locate(texts, subtitle, subprev, toc_pages)
                if subpage is None:
                    continue
                writer.add_outline_item(subtitle, subpage, parent=section_parent)
                subprev, n_subsec = subpage, n_subsec + 1
    expected_secs = sum(len(secs) for _, _, secs in outline)
    expected_subsecs = sum(len(subsecs) for _, _, secs in outline
                           for _title, _sid, subsecs in secs)
    if (n_ch != len(outline) or n_sec != expected_secs
            or n_subsec != expected_subsecs):
        raise SystemExit(
            f"书签不完整：篇 {n_ch}/{len(outline)}，节 {n_sec}/{expected_secs}，"
            f"子节 {n_subsec}/{expected_subsecs}")
    fd, tmp_name = tempfile.mkstemp(prefix="bookmarked-", suffix=".pdf", dir=pdf_path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with open(tmp_path, "wb") as f:
            writer.write(f)
        PdfReader(str(tmp_path))
        os.replace(tmp_path, pdf_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    print(f"书签写入：{n_ch}/{len(outline)} 篇，{n_sec} 节，{n_subsec} 子节")
    return n_ch, n_sec, n_subsec


def find_chrome():
    default = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    chrome = os.environ.get("CHROME_BIN", default)
    if not Path(chrome).exists():
        found = (shutil.which("google-chrome") or shutil.which("chromium")
                 or shutil.which("chrome"))
        if found:
            chrome = found
    return chrome


def contains_unrendered_math(text):
    """检测 PDF 文本层里常见的未渲染 TeX 痕迹。"""
    return bool(
        "$" in text
        or "MathJax" in text
        or re.search(
            r"\\(?:begin|end|left|right|tag|text|quad|qquad|frac|dfrac|tfrac|"
            r"sqrt|mathbf|mathrm|mathbb|mathcal|operatorname|sum|prod|int|"
            r"cdot|times|leq|geq|alpha|beta|gamma|theta|lambda|mu|sigma|omega|"
            r"varepsilon)(?![A-Za-z])",
            text,
        )
    )


def validate_pdf_text_codepoints(text):
    """与发布门禁相同地拒绝 CJK 部首错误映射，不修改提取文本或 PDF。"""
    radicals = re.findall(r"[\u2e80-\u2eff\u2f00-\u2fdf]", text)
    if radicals:
        codes = sorted({f"U+{ord(character):04X}" for character in radicals})
        raise SystemExit(f"PDF 文本层含部首类错误码位：{codes}；请检查字体回退与 ToUnicode")


def validate_pdf_math_example(text, type3_glyph_counts):
    """Reject a print with missing MathJax glyph fonts on the AEC hand example.

    Chrome embeds CHTML WOFF glyphs as Type3 fonts without usable ToUnicode,
    so successful formulas may *not* appear in extracted text. The canary page
    must instead contain both populated math font subsets. We also separately
    reject raw TeX and inspect the rendered page during release QA.
    """

    start = text.find("不提前舍入时，上例的精确分数")
    end = text.find("分区块频域卡尔曼滤波", start)
    if start < 0 or end < 0:
        raise SystemExit("PDF 数学字形探针的正文边界缺失；不能验收公式")
    if len(type3_glyph_counts) < 2 or sum(type3_glyph_counts) < 150:
        raise SystemExit(
            f"PDF 数学字形探针的字形子集不完整：{type3_glyph_counts}；"
            "疑似 MathJax 字体尚未就绪，保留原发布件"
        )


def validate_pdf_body_scale(reader):
    """拒绝 Chrome 因过宽内容而缩小整书；CSS 正文固定为 16 px。"""
    samples = []
    for page_number, page in enumerate(reader.pages, 1):
        def inspect(text, cm, tm, _font, font_size):
            # 长中文正文排除数学上下标、图像、旋转标签及其他字号。
            if (abs(font_size - 16) < 0.01 and
                    len(re.findall(r"[\u4e00-\u9fff]", text)) >= 8 and
                    abs(cm[1]) < 1e-6 and abs(tm[1]) < 1e-6):
                samples.append((page_number, abs(cm[0] * tm[0])))
        page.extract_text(visitor_text=inspect)
    if not samples:
        raise SystemExit("PDF 缺少可检查缩放比例的 16 px 中文正文，不能确认打印尺度")
    # CSS 96 px/in 到 PDF 72 pt/in 正常为 0.75；仅为浮点取整留少量余地。
    too_small = [(page, round(scale, 4)) for page, scale in samples if scale < 0.74]
    if too_small:
        raise SystemExit(f"PDF 正文被额外缩小（正常 px→pt 比例为 0.75）：{too_small[:5]}；请检查过宽公式或表格")


def print_pdf(combined, pdf, timeout_min_pages=100):
    chrome = find_chrome()
    if not Path(chrome).exists():
        raise SystemExit(f"找不到 Chrome：{chrome}")
    fd, tmp_name = tempfile.mkstemp(prefix="printed-", suffix=".pdf", dir=pdf.parent)
    os.close(fd)
    tmp_pdf = Path(tmp_name)
    tmp_pdf.unlink()
    completed_by_stability = False
    log_text = ""
    try:
        with tempfile.TemporaryDirectory() as udd, tempfile.TemporaryFile(mode="w+") as log:
            proc = subprocess.Popen(
                [chrome, "--headless", "--disable-gpu", "--no-sandbox",
                 "--allow-file-access-from-files",
                 f"--user-data-dir={udd}", "--timeout=180000",
                 "--virtual-time-budget=30000",
                 f"--print-to-pdf={tmp_pdf}", "--no-pdf-header-footer",
                 combined.as_uri()], stdout=log, stderr=subprocess.STDOUT, text=True)
            deadline, last_size, stable_since = time.time() + 600, -1, None
            while time.time() < deadline:
                if proc.poll() is not None:
                    break
                if tmp_pdf.exists():
                    sz = tmp_pdf.stat().st_size
                    now = time.time()
                    if sz == last_size and sz > 64 * 1024:
                        if stable_since is not None and now - stable_since > 15:
                            completed_by_stability = True
                            break
                    else:
                        last_size, stable_since = sz, now
                time.sleep(5)
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            log.seek(0)
            log_text = log.read()[-1000:]
            if proc.returncode not in (0, None) and not completed_by_stability:
                raise SystemExit(f"Chrome 打印失败（退出码 {proc.returncode}）\n{log_text}")
        if not tmp_pdf.exists():
            raise SystemExit(f"PDF 生成失败\n{log_text}")
        print(log_text)
        print("generated", tmp_pdf, tmp_pdf.stat().st_size // 1024 // 1024, "MB")
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(tmp_pdf))
            npages = len(reader.pages)
            page_texts = [page.extract_text() or "" for page in reader.pages]
            last_text = page_texts[-1]
        except Exception as e:
            raise SystemExit(f"PDF 校验失败（可能被截断）：{e}")
        print("pages:", npages)
        if npages < timeout_min_pages:
            raise SystemExit(f"PDF 页数异常（{npages} 页），疑似截断")
        if "全书完" not in last_text:
            raise SystemExit("PDF 末页未检测到固定结束标记，疑似截断")
        if contains_unrendered_math("\n".join(page_texts)):
            raise SystemExit("PDF 文本层含未渲染的公式源码；保留原发布件")
        math_page = next((i for i, page_text in enumerate(page_texts)
                          if "不提前舍入时，上例的精确分数" in page_text), None)
        if math_page is None:
            raise SystemExit("PDF 数学字形探针的正文边界缺失；不能验收公式")
        fonts = reader.pages[math_page]["/Resources"].get("/Font", {})
        type3_glyph_counts = [len(font.get_object().get("/CharProcs", {}))
                              for font in fonts.values()
                              if font.get_object().get("/Subtype") == "/Type3"]
        validate_pdf_math_example("\n".join(page_texts), type3_glyph_counts)
        validate_pdf_text_codepoints("\n".join(page_texts))
        validate_pdf_body_scale(reader)
        os.replace(tmp_pdf, pdf)
    finally:
        tmp_pdf.unlink(missing_ok=True)
    print("saved", pdf, pdf.stat().st_size // 1024 // 1024, "MB")
    return npages


def check_figures():
    """合订前检查：正文引用的图必须存在、非空，并覆盖 39 个唯一文件。"""
    missing = []
    refs = set()
    for fname, _ in CHAPTERS:
        md = (SRC / fname).read_text(encoding="utf-8")
        for m in re.finditer(r"\.\./figures/([^\")\s]+)", md):
            refs.add(m.group(1))
            path = ROOT / "figures" / m.group(1)
            if not path.exists() or path.stat().st_size == 0:
                missing.append(f"{fname}: {m.group(1)}")
    if missing:
        raise SystemExit("缺图，中止：\n" + "\n".join(missing))
    if len(refs) != 39:
        raise SystemExit(f"唯一图片数异常：期望 39，实际 {len(refs)}")
    print(f"图片检查通过（{len(CHAPTERS)} 篇、{len(refs)} 张唯一图片）")


def outline_from_html(html):
    """从合订 HTML 还原篇、节与指定子节，供 --pdf-only 使用。"""
    outline = []
    blocks = re.findall(
        r'<div class="chap" id="(ch-\d+)"><h1>(.*?)</h1>(.*?)(?=<div class="chap"|</body>)',
        html, flags=re.S)
    for cid, label, body in blocks:
        secs = []
        for match in re.finditer(r'<h([23])([^>]*)>(.*?)</h\1>', body, flags=re.S):
            level, attrs, title = int(match.group(1)), match.group(2), match.group(3)
            id_match = re.search(r'\bid="([^"]+)"', attrs)
            if not id_match:
                continue
            heading_id = id_match.group(1)
            if level == 2:
                secs.append((plain_text(title), heading_id, []))
            elif cid in PDF_THIRD_LEVEL_CHAPTER_IDS and secs:
                secs[-1][2].append((plain_text(title), heading_id))
        outline.append((plain_text(label), cid, secs))
    return outline


def digest_from_html(html):
    match = re.search(r"源文件 sha256 ([0-9a-f]{12})", html)
    return match.group(1) if match else None


def validate_pdf_links(pdf_path):
    """发布件不得含构建机本地路径。"""
    from pypdf import PdfReader
    bad = []
    for page_no, page in enumerate(PdfReader(str(pdf_path)).pages, 1):
        for ref in page.get("/Annots", []):
            obj = ref.get_object()
            action = obj.get("/A")
            uri = str(action.get("/URI")) if action and action.get("/URI") else ""
            if uri.startswith("file:"):
                bad.append((page_no, uri))
    if bad:
        sample = "\n".join(f"p{p}: {u}" for p, u in bad[:5])
        raise SystemExit(f"PDF 含 {len(bad)} 个本地 file:// 链接：\n{sample}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="合订本构建：chapters/ → dist/combined.html → dist PDF")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--html-only", action="store_true", help="只合 HTML，不调 Chrome")
    mode.add_argument("--pdf-only", action="store_true", help="只打印（复用现有 HTML）+ 写书签")
    ap.add_argument("--no-bookmarks", action="store_true", help="跳过书签写入")
    ap.add_argument("--min-pages", type=int, default=100, help="PDF 页数下限（默认 100）")
    ap.add_argument("--build-date", help="封面构建日期 YYYY-MM-DD；也可设置 SOURCE_DATE_EPOCH")
    args = ap.parse_args(argv)
    check_mathjax_assets()
    OUT.mkdir(exist_ok=True)
    combined = OUT / "combined.html"
    pdf = OUT / "microphone-array-tutorial.pdf"
    outline = None
    tmp_html = None
    tmp_pdf = None
    try:
        if not args.pdf_only:
            check_figures()
            page, outline = build_html(args.build_date)
            fd, tmp_name = tempfile.mkstemp(prefix="combined-", suffix=".html", dir=OUT)
            os.close(fd)
            tmp_html = Path(tmp_name)
            tmp_html.write_text(page, encoding="utf-8")
        if args.html_only:
            build_site.publish_files([(tmp_html, combined)])
            print("saved", combined, "html only, skip printing")
            return
        if args.pdf_only:
            html = combined.read_text(encoding="utf-8")
            embedded_digest = digest_from_html(html)
            current_digest = source_digest()
            if embedded_digest != current_digest:
                raise SystemExit(
                    "--pdf-only 拒绝使用陈旧 combined.html："
                    f"内嵌摘要 {embedded_digest or '缺失'}，当前源文件 {current_digest}。"
                    "请先运行完整构建或 --html-only。")
            outline = outline_from_html(html)
            subsec_count = sum(len(subsecs) for _, _, secs in outline
                               for _title, _sid, subsecs in secs)
            print(f"--pdf-only：从 HTML 反推 {len(outline)} 篇、"
                  f"{sum(len(s) for _, _, s in outline)} 节、{subsec_count} 子节")
        fd, tmp_name = tempfile.mkstemp(prefix="validated-", suffix=".pdf", dir=OUT)
        os.close(fd)
        tmp_pdf = Path(tmp_name)
        # 临时 HTML 与正式 HTML 同目录，图片相对路径保持一致。
        print_pdf(tmp_html or combined, tmp_pdf, args.min_pages)
        if not args.no_bookmarks:
            add_bookmarks(tmp_pdf, outline)
        validate_pdf_links(tmp_pdf)
        replacements = [] if tmp_html is None else [(tmp_html, combined)]
        replacements.append((tmp_pdf, pdf))
        build_site.publish_files(replacements)
        print("saved", pdf)
    finally:
        if tmp_html is not None:
            tmp_html.unlink(missing_ok=True)
        if tmp_pdf is not None:
            tmp_pdf.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
