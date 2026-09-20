#!/usr/bin/env python3
"""把 拆分版/ 12 篇合成一个带目录超链接的单页 HTML，再调 Chrome 无头打印成 PDF。

用法（报告根目录）：
    .venv/bin/python Scripts/build_pdf.py
产物：site/教程合订本.html → site/麦克风阵列信号处理教程.pdf
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "拆分版"
OUT = ROOT / "site"

CHAPTERS = [
    ("00_首页_导读与导航.md", "导读与导航"),
    ("01_问题定义与双耳启示.md", "第 1 章 · 问题定义与双耳启示"),
    ("02_基础_声音到达阵列时发生了什么.md", "第 2 章 · 声音到达阵列时发生了什么"),
    ("03_阵列几何形态.md", "第 3 章 · 阵列几何形态"),
    ("04_声源定位DOA估计.md", "第 4 章 · 声源定位（DOA 估计）"),
    ("05_波束形成Beamforming.md", "第 5 章 · 波束形成"),
    ("06_声学回声消除AEC.md", "第 6 章 · 声学回声消除（AEC）"),
    ("07_去混响WPE.md", "第 7 章 · 去混响（WPE）"),
    ("08_语音分离.md", "第 8 章 · 语音分离"),
    ("09_声源追踪SourceTracking.md", "第 9 章 · 声源追踪"),
    ("10_工程实现评测与产业实践.md", "第 10 章 · 工程实现与产业实践"),
    ("11_总结与选型指南.md", "第 11 章 · 总结与选型指南"),
    ("12_附录A_符号术语数学.md", "附录 A · 符号术语数学"),
    ("13_附录B_路径地图前沿踩坑练习复现.md", "附录 B · 路径地图与练习"),
]

CSS = """
body{font-family:"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;line-height:1.75;color:#1a1a2e;max-width:860px;margin:0 auto;padding:24px}
img{max-width:100%;height:auto;display:block;margin:12px auto}
table{border-collapse:collapse;margin:12px 0;display:block;overflow-x:visible;max-width:100%}
th,td{border:1px solid #dfe3ea;padding:5px 9px;font-size:13.5px;text-align:left}
th{background:#f0f4f9}code{background:#f0f3f7;padding:1px 5px;border-radius:4px;font-size:13px}
pre{background:#f4f6f9;padding:12px;border-radius:6px;overflow-x:visible;white-space:pre-wrap}
pre code{background:none;padding:0}
blockquote{border-left:3px solid #2f6db3;margin:12px 0;padding:6px 12px;background:#f2f7fd}
a{color:#2f6db3;text-decoration:none}
h1{font-size:24px;border-bottom:2px solid #1a1a2e;padding-bottom:6px}
h2{font-size:20px;margin-top:30px;border-bottom:1px solid #e5e8ee;padding-bottom:5px}
h3{font-size:16.5px}h4{font-size:15px}
.chap{page-break-before:always}
.toc li{margin:3px 0}
@media print{
.chap{page-break-before:always}
table{display:table;width:100%}
thead{display:table-header-group}
tr{break-inside:avoid}
td,th{word-break:break-word}
h2,h3,h4{break-after:avoid}
blockquote,pre{break-inside:avoid}
img{max-height:92vh;object-fit:contain}
th,code,pre,blockquote{-webkit-print-color-adjust:exact;print-color-adjust:exact}
}
"""


def shield_math(md):
    """数学段暂存：防 markdown 吃下划线（_x_→斜体），防浏览器吞 <（i<j）。
    转完 markdown 再原样贴回。"""
    repo = []

    def stash(m):
        repo.append(m.group(0).replace("<", r"\lt "))
        return f"@@MATH{len(repo) - 1}@@"

    md = re.sub(r"\$\$.*?\$\$", stash, md, flags=re.S)
    md = re.sub(r"\$[^$]+?\$", stash, md, flags=re.S)
    return md, repo


def unshield_math(html, repo):
    def back(m):
        return repo[int(m.group(1))]
    return re.sub(r"@@MATH(\d+)@@", back, html)


def main():
    import markdown
    import json
    body_parts = []
    toc = ['<h1>麦克风阵列信号处理教程</h1><ul class="toc">']
    # 书签侧车文件：[(章标题, [(sec_id, 小节标题)])]，供加书签脚本用
    sidecar = []
    counter = [0]

    def number_headings(html):
        def repl(m):
            counter[0] += 1
            tag, inner = m.group(1), m.group(2)
            return f"<{tag} id=\"sec-{counter[0]}\">{inner}</{tag}>"
        return re.sub(r"<(h[1-4])>(.*?)</\1>", repl, html, flags=re.S)

    def plain_text(html):
        t = re.sub(r"<[^>]+>", "", html)
        return re.sub(r"\s+", " ", t).strip()
    for i, (fname, label) in enumerate(CHAPTERS):
        md, repo = shield_math((SRC / fname).read_text(encoding="utf-8"))
        html = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
        html = unshield_math(html, repo)
        html = re.sub(r"\.md((?:#[^\"')\s]*)?)([\"')])",
                      lambda m: ".html" + m.group(1) + m.group(2), html)
        html = html.replace("00_首页_导读与导航.html", "合订本")
        # 篇内跨篇 .html 链在单文件里无意义，转成文字
        html = re.sub(r'<a href="(?:0\d|1\d)_[^"]+\.html">([^<]+)</a>', r"\1", html)
        # 去掉分章导航块与页脚行：它们的 ./xx.md 链在单文件里会变成 file:// 死链
        html = re.sub(r"<blockquote>\s*<p>⚠️ 本篇是系列教程.*?</blockquote>", "", html, flags=re.S)
        html = re.sub(r"<blockquote>\s*<p>🏠 首页导读.*?</blockquote>", "", html, flags=re.S)
        html = re.sub(r"<p>📄 本篇信息.*?</p>", "", html, flags=re.S)
        body_parts.append(f'<div class="chap" id="ch-{i}"><h1>{label}</h1>{html}</div>')
        toc.append(f"<li><a href=\"#ch-{i}\">{label}</a></li>")
    toc.append("</ul>")
    page = ("<!DOCTYPE html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
            f"<title>麦克风阵列信号处理教程（合订本）</title><style>{CSS}</style>"
            "<script>\nwindow.MathJax = {tex: {inlineMath: [['$', '$'], ['\\\\(', '\\\\)']], displayMath: [['$$', '$$']]}};\n</script>"
            "<script async src=\"https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js\"></script>"
            "</head><body>" + "\n".join(toc) + "\n".join(body_parts) + "</body></html>")
    combined = OUT / "教程合订本.html"
    combined.write_text(page, encoding="utf-8")
    print("saved", combined, combined.stat().st_size // 1024, "KB")
    if "--html-only" in sys.argv:
        print("html only, skip printing")
        return
    pdf = OUT / "麦克风阵列信号处理教程.pdf"
    import tempfile
    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    with tempfile.TemporaryDirectory() as udd:
        r = subprocess.run([chrome, "--headless", "--disable-gpu", "--no-sandbox",
                            f"--user-data-dir={udd}", "--virtual-time-budget=120000",
                            f"--print-to-pdf={pdf}", "--no-pdf-header-footer",
                            combined.as_uri()], capture_output=True, text=True, timeout=180)
    print(r.stdout[-500:] if r.stdout else "", r.stderr[-500:] if r.stderr else "")
    if pdf.exists():
        print("saved", pdf, pdf.stat().st_size // 1024 // 1024, "MB")
    else:
        raise SystemExit("PDF 生成失败")


if __name__ == "__main__":
    main()
