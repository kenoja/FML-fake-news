"""Numbers quoted in the result sections (05, 08, 10) must come from results/ or config.

Why: sections are read like the report. A rerun that changes a result must not leave the old value in the
prose. Same rule and same matcher as `test_report_numbers.py`, applied to the sections that quote results.
"""
from __future__ import annotations

import re

import pytest

from tests.test_report_numbers import REPO, _version_numbers, source_numbers, unmatched

SECTIONS = ["02-data-and-cleaning", "03-split-by-website", "09-limitations", "10-conclusion"]


def _prose(slug: str) -> str:
    text = (REPO / "content" / "sections" / f"{slug}.md").read_text(encoding="utf-8")
    text = re.sub(r"\A---.*?---\n", " ", text, flags=re.S)  # front matter
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)  # cell/output markers (file names, line ranges)
    for v in _version_numbers():
        text = text.replace(v, " ")
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\b[0-9a-f]{16}\b", " ", text)  # config hashes
    text = re.sub(r"0-149 words|150-249 words|>= 250 words|shorter than 250 words|under 250 words", " ", text)  # thresholds
    text = re.sub(r"\b[Ss]ections? \d+(\.\d+)?|^#+ \d+(\.\d+)?", " ", text, flags=re.M)
    return text


@pytest.mark.parametrize("slug", SECTIONS)
def test_numbers_in_section_come_from_results(slug):
    bad = unmatched(_prose(slug), source_numbers())
    assert not bad, f"numbers in {slug} that are not in results/ or config: {bad}"


def test_section_checker_catches_a_wrong_number():
    assert unmatched("F1 0.6843 on test.", source_numbers()) == ["0.6843"]
