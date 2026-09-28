"""Markdown code shielding and link-scheme checks shared by both publishers."""

import re
from html.parser import HTMLParser
from urllib.parse import urlparse


INLINE_CODE_RE = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)", re.S)
ALLOWED_LINK_SCHEMES = {"http", "https", "mailto"}


def protect_code(md_text):
    """暂存围栏和行内代码，使数学正则不会改写代码中的 `$...$`。"""
    repo = []

    def stash(value):
        repo.append(value)
        return f"@@CODETOKEN{len(repo) - 1}@@"

    lines = md_text.splitlines(keepends=True)
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
    """只允许相对链接以及 http、https、mailto 外链。"""
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
