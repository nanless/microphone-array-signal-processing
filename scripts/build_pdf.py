#!/usr/bin/env python3
"""把 chapters/ 14 篇合成一个带目录超链接的单页 HTML，再调 Chrome 无头打印成 PDF.

书签口径：仅一级章节书签（节级锚点不写入 PDF，见 scripts/README.md 备注）。

用法（报告根目录）：
    .venv/bin/python scripts/build_pdf.py
产物：dist/combined.html → dist/microphone-array-tutorial.pdf
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "chapters"
OUT = ROOT / "dist"

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
    ("10_engineering-practice.md", "第 10 章 · 工程实现与产业实践"),
    ("11_selection-guide.md", "第 11 章 · 总结与选型指南"),
    ("12_appendix-symbols-math.md", "附录 A · 符号术语数学"),
    ("13_appendix-guide.md", "附录 B · 路径地图与练习"),
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


def add_bookmarks(pdf_path):
    """按 CHAPTERS 给 PDF 写一级大纲（章节书签）。

    做法：逐页抽文本，每章定位其 <h1>label 首次出现的非常规页。
    目录页（含 3 个以上章节标题的页）排除在外，避免指到目录。
    pypdf 缺失或定位失败都不让构建失败，只打印警告。
    """
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        print("pypdf 未安装，跳过书签写入（.venv/bin/pip install pypdf）")
        return
    reader = PdfReader(str(pdf_path))
    texts = []
    for p in reader.pages:
        try:
            texts.append(p.extract_text() or "")
        except Exception:
            texts.append("")
    labels = [label for _, label in CHAPTERS]
    toc_pages = {i for i, t in enumerate(texts)
                 if sum(1 for label in labels if label in t) >= 3}

    def norm(s):
        # NFKC 把 PDF 抽词时的兼容汉字（如双⽿）折叠回普通字（如双耳）
        import unicodedata
        s = unicodedata.normalize("NFKC", s)
        return s.replace(" ", "").replace("\n", "").replace("·", "")

    writer = PdfWriter()
    writer.append(reader)
    prev, added = -1, 0
    for _, label in CHAPTERS:
        page_no = None
        for i in range(prev + 1, len(texts)):
            if i in toc_pages:
                continue
            if label in texts[i] or norm(label) in norm(texts[i]):
                page_no = i
                break
        if page_no is None:
            print(f"书签跳过（找不到起始页）：{label}")
            continue
        writer.add_outline_item(label, page_no)
        prev, added = page_no, added + 1
    with open(pdf_path, "wb") as f:
        writer.write(f)
    print(f"书签写入 {added}/{len(CHAPTERS)}")


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
        html = html.replace("00_overview.html", "合订本")
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
    OUT.mkdir(exist_ok=True)
    combined = OUT / "combined.html"
    combined.write_text(page, encoding="utf-8")
    print("saved", combined, combined.stat().st_size // 1024, "KB")
    if "--html-only" in sys.argv:
        print("html only, skip printing")
        return
    pdf = OUT / "microphone-array-tutorial.pdf"
    import tempfile
    default_chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    chrome = os.environ.get("CHROME_BIN", default_chrome)
    if not Path(chrome).exists():
        found = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chrome")
        if found:
            chrome = found
    # 说明：不用 --virtual-time-budget（新版 Chrome 下虚时间与网络 fetch 叠加会 hang 住
    # 打印进程）。--timeout 给页面真实时间做 MathJax 渲染，时间到即落版。
    # 另：实测 Chrome 153 写完 PDF 后进程常常不退出，所以这里不傻等进程结束，
    # 而是轮询 PDF 文件大小——稳定 15 秒即视为写完，主动 kill，避免无限 hang。
    import time
    with tempfile.TemporaryDirectory() as udd:
        proc = subprocess.Popen(
            [chrome, "--headless", "--disable-gpu", "--no-sandbox",
             f"--user-data-dir={udd}", "--timeout=180000",
             f"--print-to-pdf={pdf}", "--no-pdf-header-footer",
             combined.as_uri()],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if pdf.exists():
            pdf.unlink()  # 删掉旧文件，避免把上一版当成这次的
        deadline, last_size, stable_since = time.time() + 600, -1, None
        while time.time() < deadline:
            if proc.poll() is not None:
                break  # 自己退出了
            if pdf.exists():
                sz = pdf.stat().st_size
                now = time.time()
                if sz == last_size and sz > 1024 * 1024:
                    if stable_since is not None and now - stable_since > 15:
                        break  # 15 秒没长个，写完了
                else:
                    last_size, stable_since = sz, now
            time.sleep(5)
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    print((proc.stdout.read() or "")[-500:] if proc.stdout else "")
    if pdf.exists():
        print("saved", pdf, pdf.stat().st_size // 1024 // 1024, "MB")
        try:
            from pypdf import PdfReader
            npages = len(PdfReader(str(pdf)).pages)
        except Exception as e:
            raise SystemExit(f"PDF 校验失败（可能被截断）：{e}")
        print("pages:", npages)
        if npages < 100:
            raise SystemExit(f"PDF 页数异常（{npages} 页），疑似截断")
        add_bookmarks(pdf)
    else:
        raise SystemExit("PDF 生成失败")


if __name__ == "__main__":
    main()
