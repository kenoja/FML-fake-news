"""The team reads the site to learn the words. A term that is used but never linked, or a jargon word
used before it is explained, sends a reader without technical background searching. These tests build
the site and check the links and the plain-words rules."""
import re
from pathlib import Path

import pytest

from sitegen import build as sitegen

REPO = Path(__file__).resolve().parents[1]
SECTIONS = REPO / "content" / "sections"
TEACHING = ["04-word-counts", "05-tf-idf", "06-feature-engineering", "07-llm", "08-results-and-why-they-fail","12-how-we-used-ai"]
JARGON = ["standardised", "threshold", "epistemic"]   # must not appear in 'Explained simply'
MUST_HAVE_ENTRY = ["threshold", "epistemic marker", "standardise", "logarithm", "sample", "n-gram", "weight",
                   "regularisation", "overfitting", "bootstrap", "confusion matrix", "zero-shot", "token"]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("site")
    return sitegen.build(2, out=out, report_out=None).sections


def part(slug: str, heading: str) -> str:
    text = (SECTIONS / f"{slug}.md").read_text(encoding="utf-8")
    return text.split(f"## {heading}")[1].split("\n## ")[0]


def link_events(html_text: str, terms) -> dict[str, list[str]]:
    """For each glossary entry, the order of 'link' and 'plain' uses in the prose of one section."""
    events: dict[str, list[str]] = {}
    skip = 0
    html_text = re.sub(r"<figure class=\"cell.*?</figure>", "", html_text, flags=re.S)  # code and outputs are not prose
    html_text = re.sub(r'<a class="gl" href="#([^"]+)"[^>]*>(.*?)</a>', "\x01\\1\x01", html_text)
    for token in re.split(r"(<[^>]+>)", html_text):
        tag = re.match(r"<(/?)([a-zA-Z0-9]+)", token)
        if tag:
            if tag.group(2).lower() in sitegen.NO_LINK_TAGS and not token.endswith("/>"):
                skip += -1 if tag.group(1) else 1
            continue
        if skip > 0:
            continue
        linked_here = set(re.findall("\x01([^\x01]+)\x01", token))
        for anchor in linked_here:
            events.setdefault(anchor, []).append("link")
        plain = re.sub("\x01[^\x01]+\x01", "", token)
        for term, anchor, _ in terms:
            # one alias is linked per stretch of text, so a second alias in the same stretch is no miss
            if anchor not in linked_here and re.search(rf"(?<![\w-]){re.escape(term)}(?![\w-])", plain, re.I):
                events.setdefault(anchor, []).append("plain")
    return events


def test_first_use_of_every_glossary_term_is_linked(built):
    # A term whose first use is plain text would not lead the reader to its explanation.
    terms = sitegen.glossary_terms()
    bad = []
    for s in built:
        if s.slug == sitegen.GLOSSARY:
            continue
        for anchor, seq in link_events(s.html, terms).items():
            if seq[0] != "link":
                bad.append(f"{s.slug}: {anchor}")
    assert not bad, bad


def test_glossary_has_the_terms_the_sections_rely_on():
    names = " ".join(t for t, _, _ in sitegen.glossary_terms()).lower()
    missing = [t for t in MUST_HAVE_ENTRY if t not in names]
    assert not missing, missing


def test_glossary_entries_are_short():
    text = (SECTIONS / "99-glossary.md").read_text(encoding="utf-8")
    for name, body in re.findall(r"^### +(.+?)\n(.*?)(?=^### |\Z)", text, re.M | re.S):
        definition = re.sub(r"^Also:.*$", "", body, flags=re.M)
        assert len(definition.split()) <= 80, name


@pytest.mark.parametrize("slug", TEACHING)
def test_explained_simply_avoids_unexplained_jargon(slug):
    text = part(slug, "Explained simply").lower()
    assert not [w for w in JARGON if w in text]
    assert len(text.split()) <= 450


@pytest.mark.parametrize("slug", TEACHING)
def test_check_yourself_has_two_pairs(slug):
    assert len(re.findall(r"^Q:", part(slug, "Check yourself"), re.M)) >= 2


def test_website_bootstrap_is_gone_from_the_site():
    for path in SECTIONS.glob("*.md"):
        assert "website bootstrap" not in path.read_text(encoding="utf-8").lower()
