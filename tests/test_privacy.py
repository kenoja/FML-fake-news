"""Privacy scan of public outputs.

Why: the repo is public. An e-mail address would expose a person, and a long quote would copy
article text that must stay in the private data repo (AGENTS.md: excerpts <= 50 words, <= 6 per section).
The scanners are plain functions so the tests can prove they fail on an injected violation.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import pytest

csv.field_size_limit(sys.maxsize)

REPO = Path(__file__).resolve().parents[1]
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ALLOWED_EMAIL = re.compile(r"^noreply@anthropic\.com$|@example\.(com|org|net)$", re.I)
MAX_WORDS = 50
MAX_PER_SECTION = 6
QUOTED_RE = re.compile(r"[\"“]([^\"“”\n]{20,}?)[\"”]")


def _words(text: str) -> int:
    return len(text.split())


def find_emails(text: str) -> list[str]:
    return [m for m in EMAIL_RE.findall(text) if not ALLOWED_EMAIL.search(m)]


def blockquotes(text: str) -> list[str]:
    """Consecutive '> ' lines form one quote block."""
    blocks, cur = [], []
    for line in text.splitlines():
        if line.startswith(">"):
            cur.append(line.lstrip("> "))
        elif cur:
            blocks.append(" ".join(cur))
            cur = []
    if cur:
        blocks.append(" ".join(cur))
    return blocks


def excerpt_blocks(text: str) -> list[str]:
    """Quote blocks that could hold article text.

    Our own bold call-out boxes ('> **Read this first**') and made-up teaching examples
    ('(invented)') are not article excerpts, so they are not counted.
    """
    return [b for b in blockquotes(text) if not b.startswith("**") and "(invented)" not in b]


def quote_violations(text: str) -> list[str]:
    """Quote blocks and inline "quoted" spans over the word limit, or too many blocks per file."""
    blocks = excerpt_blocks(text)
    out = [f"quote block of {_words(b)} words" for b in blocks if _words(b) > MAX_WORDS]
    out += [f"inline quote of {_words(q)} words" for q in QUOTED_RE.findall(text)
            if _words(q) > MAX_WORDS and "`" not in q]
    if len(blocks) > MAX_PER_SECTION:
        out.append(f"{len(blocks)} quote blocks (max {MAX_PER_SECTION})")
    return out


def long_cells(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as f:
        return [f"{path.name}: cell with {_words(v)} words in '{k}'"
                for row in csv.DictReader(f) for k, v in row.items() if v and _words(v) > MAX_WORDS]


def _public_files() -> list[Path]:
    files = sorted((REPO / "results" / "tables").glob("*.csv"))
    files += sorted((REPO / "content" / "sections").glob("*.md"))
    report = REPO / "reports" / "report.md"
    return files + ([report] if report.is_file() else [])


def scan_files(files: list[Path]) -> list[str]:
    fails = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        fails += [f"{f.name}: e-mail address {e[:3]}..." for e in find_emails(text)]
        if f.suffix == ".csv":
            fails += long_cells(f)
        else:
            fails += [f"{f.name}: {v}" for v in quote_violations(text)]
    return fails


OWNER_RE = re.compile(r"kenoja|jahn|\bkeno\b|github\.com/[\w.-]+/FML-|\b(?!<owner>)[\w.-]+/FML-(?:fake-news|data)\b",
                      re.I)


def owner_hits(text: str) -> list[str]:
    """GitHub usernames / owner paths. Why: a username identifies a team member in a public repo."""
    return [m.group(0) for m in OWNER_RE.finditer(text)]


def _doc_files() -> list[Path]:
    files = [REPO / "README.md", REPO / ".env.example", REPO / ".gitignore"]
    for sub in ("docs", "content/sections", "reports"):
        files += sorted((REPO / sub).rglob("*.md"))
    return [f for f in files if f.is_file()]


def test_docs_have_no_github_owner_names():
    hits = {f.name: owner_hits(f.read_text(encoding="utf-8", errors="ignore")) for f in _doc_files()}
    assert {k: v for k, v in hits.items() if v} == {}


def test_owner_scanner_catches_injected_username():
    assert owner_hits("see " + "keno" + "ja/FML-data") and owner_hits("by " + "Jah" + "n")
    assert owner_hits("<owner>/<repo> and <owner>/FML-data") == []


def test_public_outputs_have_no_emails_or_long_quotes():
    assert scan_files(_public_files()) == []


def test_scanner_catches_injected_email(tmp_path):
    bad = tmp_path / "x.md"
    bad.write_text("Contact: someone" + "@" + "university-mail.nl for details")
    assert scan_files([bad])


def test_scanner_catches_long_quote_block_and_csv_cell(tmp_path):
    long = " ".join(["word"] * 60)
    md = tmp_path / "x.md"
    md.write_text(f"Intro\n\n> {long}\n")
    assert scan_files([md])
    table = tmp_path / "t.csv"
    table.write_text(f"doc_id,justification\nC1,{long}\n")
    assert scan_files([table])


def test_scanner_allows_short_quote(tmp_path):
    md = tmp_path / "x.md"
    md.write_text("> " + " ".join(["word"] * 40) + "\n")
    assert scan_files([md]) == []
