"""Markdown syntax boundaries shared by both publishers, without publishing policy.

The pinned Markdown parser owns fence, indentation, list and heading grammar.
Math must be stashed before its table parser sees TeX ``|`` characters. Restore
math in parsed code as literal code, rather than guessing whether four spaces
mean a code block or a paragraph inside a list.
"""

import re
from html.parser import HTMLParser
from urllib.parse import urlparse

import markdown
from markdown.extensions import Extension
from markdown.extensions.fenced_code import FencedBlockPreprocessor
from markdown.inlinepatterns import BACKTICK_RE
from markdown.treeprocessors import Treeprocessor


INLINE_CODE_RE = re.compile(BACKTICK_RE, re.S)
ALLOWED_LINK_SCHEMES = {"http", "https", "mailto"}
MARKDOWN_EXTENSIONS = ("tables", "fenced_code", "sane_lists")
# Tokenize HTML tags, not arbitrary '<...>' in TeX. Quoted attributes may contain
# '<' and '>'; neither their values nor raw code contents are mathematical text.
HTML_TAG_RE = re.compile(
    r"<!--.*?-->|</?[A-Za-z][A-Za-z0-9:-]*(?=[\s/>])"
    r'''(?:[^"'<>]|"[^"]*"|'[^']*')*>''', re.S)


class _Stash(list):
    def __init__(self, source, kind):
        super().__init__()
        self.prefix = "@@" + kind
        while self.prefix in source:
            self.prefix += "X"
        self.literal = set()

    def stash(self, value, literal=False):
        index = len(self)
        self.append(value)
        if literal:
            self.literal.add(index)
        return f"{self.prefix}{index}@@"

    def substitute(self, text, transform=lambda value, _index: value):
        return re.sub(re.escape(self.prefix) + r"(\d+)@@",
                      lambda m: transform(self[int(m[1])], int(m[1])), text)


def _raw_code_ranges(text):
    stack, ranges = [], []
    for match in HTML_TAG_RE.finditer(text):
        tag_match = re.match(r"</?([A-Za-z][A-Za-z0-9:-]*)", match[0])
        if not tag_match or tag_match[1].lower() not in {"pre", "code"}:
            continue
        tag = tag_match[1].lower()
        if match[0].startswith("</"):
            if stack and stack[-1][0] == tag:
                _, start = stack.pop()
                ranges.append((start, match.end()))
        elif not match[0].endswith("/>"):
            stack.append((tag, match.start()))
    ranges.extend((start, len(text)) for _, start in stack)
    return ranges


def code_block_ranges(text):
    """Root fences use the configured parser's exact fence grammar.

    This helper deliberately does not classify indentation; the actual block
    parser does that when math is restored, including nested lists.
    """
    ranges = _raw_code_ranges(text)
    # Normalize line endings exactly as Markdown does, retaining source offsets
    # so protected code and subsequent line-number diagnostics stay unchanged.
    normalized, offsets, index = [], [0], 0
    while index < len(text):
        char = text[index]
        if char == "\r":
            index += 2 if text[index:index + 2] == "\r\n" else 1
            normalized.append("\n")
        else:
            index += 1
            normalized.append(char)
        offsets.append(index)
    normalized = "".join(normalized)
    ranges.extend((offsets[m.start()], offsets[min(m.end(), len(normalized))])
                  for m in FencedBlockPreprocessor.FENCED_BLOCK_RE.finditer(normalized + "\n"))
    return ranges


def _math_ranges(text):
    """Paired dollar delimiters, respecting backslash parity and paragraph bounds.

    Blank paragraphs terminate an unmatched delimiter. This prevents a dollar
    in one code/prose paragraph from consuming a later unrelated equation;
    multiline equations within one paragraph remain supported.
    """
    delimiters = list(re.finditer(r"\$+", text))
    result, index = [], 0
    while index < len(delimiters):
        opening = delimiters[index]
        if len(opening[0]) not in (1, 2) or _backslashes_before(text, opening.start()) % 2:
            index += 1
            continue
        closing_index = index + 1
        while closing_index < len(delimiters):
            closing = delimiters[closing_index]
            if re.search(r"\n[ \t]*\n", text[opening.end():closing.start()]):
                break
            if (closing[0] == opening[0]
                    and not _backslashes_before(text, closing.start()) % 2):
                result.append((opening.start(), closing.end()))
                index = closing_index + 1
                break
            closing_index += 1
        else:
            index += 1
            continue
        if closing_index == len(delimiters) or not result or result[-1][0] != opening.start():
            index += 1
    return result


def _backslashes_before(text, position):
    start = position
    while start and text[start - 1] == "\\":
        start -= 1
    return position - start


def protect_code(md_text):
    """Protect native fences/backticks and raw HTML boundaries before math."""
    repo = _Stash(md_text, "CODETOKEN")
    ranges = code_block_ranges(md_text)
    math_ranges = _math_ranges(md_text)
    ranges.extend((m.start(), m.end()) for m in HTML_TAG_RE.finditer(md_text)
                  if not any(a <= m.start() and m.end() <= b for a, b in math_ranges))
    ranges.extend((m.start(), m.end()) for m in INLINE_CODE_RE.finditer(md_text)
                  if m.group(2))
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    text = md_text
    for start, end in reversed(merged):
        text = text[:start] + repo.stash(text[start:end]) + text[end:]
    return text, repo


def restore_code(text, repo):
    return repo.substitute(text)


def shield_math(text):
    """Stash math before table parsing; retain original text for literal code."""
    source = text
    text, code_repo = protect_code(text)
    repo = _Stash(source, "MATH")
    replacements = [(a - _backslashes_before(text, a), b, False)
                    for a, b in _math_ranges(text)]
    # Keep escape parity through Markdown's own backslash processing.
    for match in re.finditer(r"\\+\$", text):
        if len(match[0][:-1]) % 2 and not any(a <= match.start() < b for a, b, _ in replacements):
            replacements.append((match.start(), match.end(), True))
    for start, end, literal in sorted(replacements, reverse=True):
        text = text[:start] + repo.stash(text[start:end], literal) + text[end:]
    return restore_code(text, code_repo), repo


def unshield_math(html, repo):
    """Restore TeX in prose, but original escaped text in parsed code nodes."""
    from html import escape
    code_ranges = _raw_code_ranges(html)
    def restore(match):
        index = int(match[1])
        value = repo[index]
        if any(start <= match.start() < end for start, end in code_ranges):
            return escape(value, quote=True)
        return value if index in repo.literal else value.replace("<", r"\lt ")
    return re.sub(re.escape(repo.prefix) + r"(\d+)@@", restore, html)


def render_markdown(text):
    protected, repo = shield_math(text)
    html = markdown.markdown(protected, extensions=MARKDOWN_EXTENSIONS)
    return unshield_math(html, repo)


def parsed_markdown_headings(text):
    """Return raw heading labels selected by Markdown, not a second fence scanner.

    Only syntax recognition is shared. Each publisher/validator still owns its
    labels, expected quantities, historical aliases and anchor calculations.
    """
    protected, repo = shield_math(text)
    records = []
    class Collect(Treeprocessor):
        def run(self, root):
            records.extend((int(node.tag[1]), repo.substitute(node.text or "").strip())
                           for node in root.iter()
                           if re.fullmatch(r"h[1-6]", node.tag))
    class HeadingExtension(Extension):
        def extendMarkdown(self, md):
            md.treeprocessors.register(Collect(md), "tutorial_heading_source", 25)
    markdown.markdown(protected, extensions=[*MARKDOWN_EXTENSIONS, HeadingExtension()])
    return records


def map_table_cell_text(html, transform):
    """Map table label text only, excluding attributes, pre and existing links."""
    starts = [0] + [m.end() for m in re.finditer("\n", html)]
    edits = []
    class CellText(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=False)
            self.stack = []

        def handle_starttag(self, tag, attrs):
            if tag in {"table", "th", "td", "pre", "a"}:
                self.stack.append(tag)

        def handle_endtag(self, tag):
            if tag in self.stack:
                # A closed parent also closes any tracked children.
                position = len(self.stack) - 1 - self.stack[::-1].index(tag)
                del self.stack[position:]

        def handle_data(self, data):
            if ("table" in self.stack and ({"th", "td"} & set(self.stack))
                    and not {"pre", "a"} & set(self.stack)):
                line, column = self.getpos()
                start = starts[line - 1] + column
                edits.append((start, start + len(data), transform(data)))
    parser = CellText()
    parser.feed(html)
    parser.close()
    for start, end, replacement in reversed(edits):
        html = html[:start] + replacement + html[end:]
    return html


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
