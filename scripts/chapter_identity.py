"""Published topic identities survive the 12--15 chapter-number permutation.

Display numbers and code directories follow the current book. Hidden PDF names
retain their original topic, and the four original HTML routes remain aliases.
The fixed snapshot records real previous headings, not ordinal guesses.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

NUMBERING_PATH = Path(__file__).with_name("chapter_numbering.json")
NUMBERING = json.loads(NUMBERING_PATH.read_text(encoding="utf-8"))
TOPICS = {row["source"]: row for row in NUMBERING["chapters"]}
ORIGINAL_TO_CURRENT = {row["original_source"]: row["source"] for row in TOPICS.values()}
LEGACY_HTML_ROUTES = {old.replace(".md", ".html"): new.replace(".md", ".html")
                      for old, new in ORIGINAL_TO_CURRENT.items()}


def canonical_source(name: str) -> str:
    return ORIGINAL_TO_CURRENT.get(name, name)


def historical_source(name: str) -> str:
    row = TOPICS.get(canonical_source(name))
    return row["original_source"] if row else name


def pdf_topic_number(name: str) -> int:
    row = TOPICS.get(canonical_source(name))
    if row:
        return row["pdf_identity"]
    match = re.fullmatch(r"(\d{2})_[^/]+\.md", name)
    if match is None:
        raise ValueError("chapter source requires a two-digit identity")
    return int(match.group(1))


def numbering_heading_aliases(name: str) -> dict[str, str]:
    return NUMBERING["heading_aliases"].get(canonical_source(name), {})


def exercise_alias_records(name: str) -> dict[str, dict[str, str]]:
    return NUMBERING["exercise_aliases"].get(canonical_source(name), {})


def canonical_exercise(name: str, fragment: str) -> str:
    for current, row in exercise_alias_records(name).items():
        if fragment == row["original_id"]:
            return current
    return fragment


def inject_exercise_aliases(html: str, name: str, prefix: str = "") -> str:
    """Attach old topic-local IDs without adding exercises to Markdown counts."""
    records = exercise_alias_records(name)

    def alias(match):
        current = match.group(2)
        row = records.get(current)
        if not row:
            return match.group(0)
        old = prefix + row["original_id"]
        return (f'<span id="{old}" class="anchor-alias exercise-alias" '
                f'aria-hidden="true"></span>' + match.group(0))

    return re.sub(r'(<a\s+id=["\']' + re.escape(prefix)
                  + r')(e\d{2}-\d{2})(["\']\s*>)', alias, html)
