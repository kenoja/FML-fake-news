"""Every number in the report must come from results/ (or config), never from memory.

Why: the report is the document people read. If a rerun changes a result and the prose keeps the old
value, readers are misled without any error. This test fails when a decimal, a percentage or a large
integer in the report body cannot be found (after rounding to the digits shown) in metrics.json,
the CSV tables or the project config.
"""
from __future__ import annotations

import csv
import importlib.metadata as md
import json
import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "reports" / "report.md"
SECTIONS = ["1. What we wanted", "2. Data, models", "3. Approach", "4. Evaluation", "5. Findings",
            "6. Interpretation", "7. Use of AI"]

DECIMAL = re.compile(r"(?<![\d.])(\d+\.\d+)(?!\d|\.\d)(%?)")
INTEGER = re.compile(r"(?<![\d.,])(\d{1,3}(?:,\d{3})+|\d{3,})(?![\d,]*\.\d)(?!\d)")
PERCENT_INT = re.compile(r"(?<![\d.])(\d+)\s?(?:%|percent)")


def _walk(obj, out: set[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.add(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _walk(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk(v, out)


def source_numbers() -> set[float]:
    nums: set[float] = set()
    _walk(json.loads((REPO / "results" / "metrics.json").read_text(encoding="utf-8")), nums)
    _walk(yaml.safe_load((REPO / "config" / "project.yaml").read_text(encoding="utf-8")), nums)
    for f in (REPO / "results" / "tables").glob("*.csv"):
        with f.open(encoding="utf-8", newline="") as fh:
            for row in csv.reader(fh):
                for cell in row:
                    try:
                        nums.add(float(cell))
                    except ValueError:
                        pass
    return nums


def _body() -> str:
    text = REPORT.read_text(encoding="utf-8")
    return re.split(r"\n#+\s*References\b", text, flags=re.I)[0]


def _version_numbers() -> set[str]:
    """Library versions are quoted in the data section; they come from the installed packages."""
    return {md.version(p) for p in ("scikit-learn", "pandas", "numpy", "scipy", "matplotlib", "anthropic")}


def _found(x: float, digits: int, nums: set[float]) -> bool:
    return any(round(abs(v), digits) == round(x, digits) for v in nums)  # the report writes "-0.0244" as a sign + number


def unmatched(body: str, nums: set[float]) -> list[str]:
    missing = []
    for m in DECIMAL.finditer(body):
        value, pct = float(m.group(1)), m.group(2)
        digits = len(m.group(1).split(".")[1])
        ok = _found(value, digits, nums) or (pct and _found(value / 100, digits + 2, nums))
        if not ok:
            missing.append(m.group(0))
    for m in INTEGER.finditer(body):
        value = float(m.group(1).replace(",", ""))
        if 1900 <= value <= 2100:  # years in citations and dates
            continue
        if not _found(value, 0, nums):
            missing.append(m.group(0))
    return missing


def test_report_exists_with_seven_parts():
    body = _body()
    for title in SECTIONS:
        assert f"## {title}" in body, f"report lacks part '{title}'"


@pytest.mark.parametrize("path", sorted(set(re.findall(r"\]\((\.\./[^)]+)\)", REPORT.read_text(encoding="utf-8")))))
def test_linked_files_exist(path):
    assert (REPORT.parent / path).resolve().is_file(), f"report links {path}, which does not exist"


def test_every_number_in_report_comes_from_results():
    body = _body()
    versions = _version_numbers()
    for v in versions:  # versions like 1.9.1 contain dots; blank them so they are not read as decimals
        body = body.replace(v, " ")
    body = re.sub(r"Python \d+\.\d+(\.\d+)?", "Python", body)
    body = re.sub(r"https?://\S+", " ", body)
    body = re.sub(r"\b[Ss]ections? \d+(\.\d+)?|^#+ \d+(\.\d+)?", " ", body, flags=re.M)  # cross-references
    body = re.sub(r"0-149 words|150-249 words|>= 250 words|shorter than 250 words", " ", body)  # length thresholds, not results
    bad = unmatched(body, source_numbers())
    assert not bad, f"numbers in the report that are not in results/ or config: {bad}"


def test_checker_catches_a_wrong_number():
    """The check must fail when a number is typed from memory (otherwise the test proves nothing)."""
    nums = source_numbers()
    assert unmatched("The LLM reaches F1 0.6843 on test.", nums) == ["0.6843"]
    assert unmatched("The LLM reaches F1 0.6842 on test.", nums) == []
    assert unmatched("We used 4,242 documents.", nums) == ["4,242"]
