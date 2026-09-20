#!/usr/bin/env python3
"""把 chapters/ 14 篇 Markdown 建成多级页面静态站，输出到 site/。

用法（报告根目录）：
    .venv/bin/python scripts/build_site.py

产物：site/index.html（首页）+ site/01..13_*.html（13 篇正文），
左侧边栏 = 首页 + 13 篇 + 每篇的二级及以下小节锚点，顶部面包屑，
文末上一篇/下一篇（首页不输出该盒）。图片直接引用 ../figures/（不复制）。
数学公式用 MathJax CDN 渲染（离线时显示源码，页面顶部有提示）。
"""
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "chapters"
OUT = ROOT / "site"

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
    ("10_engineering-practice.md", "第 10 章 · 工程实现与产业实践"),
    ("11_selection-guide.md", "第 11 章 · 总结与选型指南"),
    ("12_appendix-symbols-math.md", "附录 A · 符号术语数学"),
    ("13_appendix-guide.md", "附录 B · 路径地图与练习"),
]
HOME_FNAME = "00_overview.md"
LABEL_BY_FNAME = {f: l for f, l in CHAPTERS}
LABEL_BY_FNAME[HOME_FNAME] = "🏠 导读与导航（首页）"

CSS = """
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;line-height:1.75;color:#1a1a2e;background:#fafbfc}
.topbar{position:sticky;top:0;z-index:10;background:#1a1a2e;color:#fff;padding:10px 20px;font-size:15px}
.topbar a{color:#9ec5f0;text-decoration:none}.topbar a:hover{text-decoration:underline}
.wrap{display:flex;max-width:1280px;margin:0 auto}
.side{width:300px;flex-shrink:0;padding:20px 14px;position:sticky;top:47px;height:calc(100vh - 47px);overflow-y:auto;background:#fff;border-right:1px solid #e5e8ee;font-size:13.5px}
.side a{color:#2f6db3;text-decoration:none}.side a:hover{text-decoration:underline}
.side .chap{margin:10px 0 2px;font-weight:700}.side .chap.cur{color:#c0392b}
.side ul{margin:2px 0 6px;padding-left:16px;color:#666}.side li{margin:2px 0}
.main{flex:1;min-width:0;padding:28px 36px;background:#fff}
.main img{max-width:100%;height:auto;display:block;margin:14px auto;border:1px solid #eee;min-height:40px;background:#f6f8fb}
table{border-collapse:collapse;margin:14px 0;display:block;overflow-x:auto;max-width:100%}
th,td{border:1px solid #dfe3ea;padding:6px 10px;font-size:14px;text-align:left}
th{background:#f0f4f9}code{background:#f0f3f7;padding:1px 5px;border-radius:4px;font-size:13.5px}
pre{background:#1a1a2e;color:#e8ecf3;padding:14px;border-radius:8px;overflow-x:auto}
pre code{background:none;color:inherit;padding:0}
blockquote{border-left:3px solid #2f6db3;margin:14px 0;padding:8px 14px;background:#f2f7fd;color:#333}
mjx-container[jax="CHTML"]{overflow-x:auto;overflow-y:hidden;max-width:100%}
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
.topbtn{position:fixed;bottom:20px;right:20px;background:#1a1a2e;color:#fff;border-radius:50%;width:42px;height:42px;text-align:center;line-height:42px;text-decoration:none;font-size:18px;opacity:.75}
.offline-note{display:none;background:#fff7e6;border:1px solid #e6c87a;color:#7a5b00;padding:8px 14px;font-size:13.5px}
@media(max-width:900px){.side{display:none}.main{padding:20px}.toc-mobile{display:block}.topbar{font-size:14px}}
@media print{.topbar,.side,.pn,.topbtn,.toc-mobile{display:none}.main{padding:0}table{display:table}a{color:#000;text-decoration:none}pre{white-space:pre-wrap;background:#fff;color:#000;border:1px solid #ccc}}
"""

PAGE = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · 麦克风阵列信号处理教程</title><style>{css}</style>
<script>
window.MathJax = {{tex: {{inlineMath: [['$', '$'], ['\\(', '\\)']], displayMath: [['$$', '$$']]}}}};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head><body id="top">
<div class="topbar"><a href="index.html">🏠 首页</a> &nbsp;/&nbsp; {crumb}</div>
<div class="wrap"><nav class="side">{sidebar}</nav>
<main class="main"><div class="offline-note" id="offnote">当前离线：公式显示为源码，正文讲解不受影响。</div>
<details class="toc-mobile"><summary>本页目录</summary>{toc}</details>
{body}{pn}
<div class="foot">麦克风阵列信号处理教程 · 静态站由 scripts/build_site.py 生成</div>
</main></div><a class="topbtn" href="#top" title="回顶部">↑</a>
<script>if(!navigator.onLine)document.getElementById('offnote').style.display='block';</script>
</body></html>
"""


def clean_label(text):
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"[`*_~]", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    return text.strip()


def parse_headings(md):
    """掃 md 源码标题，跳过围栏代码块，避免幽灵条目。"""
    heads = []
    in_fence = False
    for line in md.splitlines():
        s = line.strip()
        if s.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            heads.append((len(m.group(1)), clean_label(m.group(2))))
    return heads


def render(md_text):
    """返回 (html, 锚点数)。锚点按真实 h 标签顺序编号，是唯一的真相源。"""
    import markdown
    # 数学段暂存：防 markdown 吃下划线、防浏览器吞 <，转完再贴回
    repo = []

    def stash(m):
        repo.append(m.group(0).replace("<", r"\lt "))
        return f"@@MATH{len(repo) - 1}@@"

    md_text = re.sub(r"\$\$.*?\$\$", stash, md_text, flags=re.S)
    md_text = re.sub(r"\$[^$]+?\$", stash, md_text, flags=re.S)
    html = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
    html = re.sub(r"@@MATH(\d+)@@", lambda m: repo[int(m.group(1))], html)
    counter = [0]

    def repl(m):
        counter[0] += 1
        tag, inner = m.group(1), m.group(2)
        return f"<{tag} id=\"sec-{counter[0]}\">{inner}</{tag}>"

    html = re.sub(r"<(h[1-4])>(.*?)</\1>", repl, html, flags=re.S)
    # md 内链 .md → .html；00 首页 → index.html
    html = re.sub(r"\.md((?:#[^\"')\s]*)?)([\"')])",
                  lambda m: ".html" + m.group(1) + m.group(2), html)
    html = html.replace(HOME_FNAME.replace(".md", ".html"), "index.html")

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
        if fname == HOME_FNAME:
            return f'{pre}<a href="index.html">{fname}</a>'
        if fname in LABEL_BY_FNAME:
            return f'{pre}<a href="{fname.replace(".md", ".html")}">{fname}</a>'
        return m.group(0)

    html = re.sub(r'(^|[\s>(])((?:0\d|1\d)_[^<\s)"]+\.md)', bare_link, html)
    return html, counter[0]


def sub_list(html_name, heads, start_idx=1):
    """当前页的小节目录，返回 (html, 下一起始编号)。编号与 render 同序。"""
    parts = ["<ul>"]
    idx = start_idx
    for lvl, text in heads:
        if lvl >= 2:
            pad = "" if lvl == 2 else ("&nbsp;&nbsp;" if lvl == 3 else "&nbsp;&nbsp;&nbsp;&nbsp;— ")
            parts.append(f'<li>{pad}<a href="{html_name}#sec-{idx}">{text}</a></li>')
        idx += 1
    parts.append("</ul>")
    return "\n".join(parts), idx


def sidebar_with_anchors(current, heads):
    """current: None=首页。首页展开自己的目录（修首页零锚点bug）。"""
    parts = []
    if current is None:
        parts.append('<div class="chap cur">🏠 导读与导航（首页）</div>')
        lst, _ = sub_list("index.html", heads)
        parts.append(lst)
    else:
        parts.append('<div class="chap"><a href="index.html">🏠 导读与导航（首页）</a></div>')
    for fname, label in CHAPTERS:
        html_name = fname.replace(".md", ".html")
        if fname == current:
            parts.append(f'<div class="chap cur">{label}</div>')
            lst, _ = sub_list(html_name, heads)
            parts.append(lst)
        else:
            parts.append(f'<div class="chap"><a href="{html_name}">{label}</a></div>')
    return "\n".join(parts)


def main():
    OUT.mkdir(exist_ok=True)
    home_md = (SRC / HOME_FNAME).read_text(encoding="utf-8")
    home_heads = parse_headings(home_md)
    home_html, home_n = render(home_md)
    assert home_n == len(home_heads), f"首页锚点 {home_n} vs 标题 {len(home_heads)}"
    toc, _ = sub_list("index.html", home_heads)
    (OUT / "index.html").write_text(PAGE.format(
        title="导读与导航", css=CSS, crumb="导读与导航",
        sidebar=sidebar_with_anchors(None, home_heads), toc=toc,
        body=home_html, pn=""), encoding="utf-8")
    print("saved index.html")
    names = [f for f, _ in CHAPTERS]
    for i, (fname, label) in enumerate(CHAPTERS):
        md = (SRC / fname).read_text(encoding="utf-8")
        heads = parse_headings(md)
        html_name = fname.replace(".md", ".html")
        body, n = render(md)
        assert n == len(heads), f"{fname}: 锚点 {n} vs 标题 {len(heads)}"
        toc, _ = sub_list(html_name, heads)
        if i > 0:
            prev = f'<a href="{names[i-1].replace(".md", ".html")}">← 上一篇</a>'
        else:
            prev = '<span class="off">← 上一篇</span>'
        if i < len(names) - 1:
            nxt = f'<a href="{names[i+1].replace(".md", ".html")}">下一篇 →</a>'
        else:
            nxt = '<a href="index.html">回首页 →</a>'
        pn = f'<div class="pn"><span>{prev}</span><span>{nxt}</span></div>'
        (OUT / html_name).write_text(PAGE.format(
            title=label, css=CSS, crumb=label,
            sidebar=sidebar_with_anchors(fname, heads), toc=toc,
            body=body, pn=pn), encoding="utf-8")
        print("saved", html_name)
    print("DONE", len(names) + 1, "pages")


if __name__ == "__main__":
    main()
