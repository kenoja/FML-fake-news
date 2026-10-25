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
SECTIONS = ["Question and hypotheses", "Data and cleaning", "Splitting by website", "Word counts", "TF-IDF",
            "TF-IDF plus extra features", "The LLM classifier", "Results and why they fail", "Limitations",
            "Conclusion", "How we used AI"]
DIGITS = 3   # one rounding rule: every precision, recall, F1 and p-value is written with three decimals

DECIMAL = re.compile(r"(?<![\d.])(\d+\.\d+)(?!\d|\.\d)(%?)")
INTEGER = re.compile(r"(?<![\d.,])(\d{1,3}(?:,\d{3})+|\d{3,})(?![\d,]*\.\d)(?!\d)")
PERCENT_INT = re.compile(r"(?<![\d.])(\d+)\s?(?:%|percent)")


def _walk(obj, out: set[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.add(float(obj))
    elif isinstance(obj, dict):
        if {"tp", "fn", "fp", "tn"} <= obj.keys():  # the prose also quotes class sizes (tp + fn, fp + tn)
            out.update({float(obj["tp"] + obj["fn"]), float(obj["fp"] + obj["tn"])})
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


def test_report_has_every_part():
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
    body = re.sub(r"\b[CM]\d{3,5}[a-z]{0,2}\b|\b[0-9a-f]{16}\b", " ", body)  # article ids and prompt hashes
    body = re.sub(r"\b[Ss]ections? \d+(\.\d+)?|^#+ \d+(\.\d+)?", " ", body, flags=re.M)  # cross-references
    body = re.sub(r"0-149 words|150-249 words|>= 250 words|shorter than 250 words", " ", body)  # length thresholds, not results
    bad = unmatched(body, source_numbers())
    assert not bad, f"numbers in the report that are not in results/ or config: {bad}"


def test_checker_catches_a_wrong_number():
    """The check must fail when a number is typed from memory (otherwise the test proves nothing)."""
    nums = source_numbers()
    assert unmatched("The LLM reaches F1 0.796 on test.", nums) == ["0.796"]
    assert unmatched("The LLM reaches F1 0.795 on test.", nums) == []
    assert unmatched("We used 4,242 documents.", nums) == ["4,242"]


def test_scores_use_one_rounding_rule():
    """0.68 in one place and 0.684 in another looks like two different results; the report uses one rule."""
    text = re.sub(r"https?://\S+", " ", _body())
    scores = re.findall(r"(?:precision|recall|F1(?: score)?|p) (?:of |=|<|reached |is )?\s?(\d\.\d+)", text)
    wrong = [x for x in scores if len(x.split(".")[1]) > DIGITS and x != "0.0001"]
    assert not wrong, f"scores written with more than {DIGITS} decimals (except p < 0.0001): {wrong}"
    metrics = json.loads((REPO / "results" / "metrics.json").read_text(encoding="utf-8"))
    best = max(r["f1"] for r in metrics["comparison"]["table"])
    assert f"{best:.{DIGITS}f}" in text, "the best test F1 is not written with the standard rounding"


def test_report_length_is_within_the_limit():
    """The assignment asks for 3,000-5,000 words of report text; tables and figure links are not prose."""
    prose = [line for line in _body().splitlines() if not line.startswith(("|", "!["))]
    n = len(re.findall(r"\b\w[\w'-]*\b", " ".join(prose)))
    print(f"report body: {n} words")
    assert 3000 <= n <= 5000, f"report body has {n} words"


ABLATION_WORDS = {  # words that name an ablation row in prose -> model_id in ablation.csv
    "n-gram": "lr_plus_ngrams", "certainty": "lr_plus_epistemic", "epistemic": "lr_plus_epistemic",
    "style": "lr_plus_style", "all three groups": "lr_extended",
}
ABLATION_TEXTS = ["content/sections/06-feature-engineering.md", "reports/report.md", "docs/decisions.md"]


def _ablation_rows() -> dict[str, tuple[float, float]]:
    with (REPO / "results" / "tables" / "ablation.csv").open(encoding="utf-8", newline="") as fh:
        return {r["model_id"]: (float(r["f1"]), float(r["delta_f1"])) for r in csv.DictReader(fh)}


def ablation_mismatches(text: str, rows: dict[str, tuple[float, float]]) -> list[str]:
    """Three-decimal numbers in a sentence that names ablation rows must belong to those rows."""
    bad = []
    for sentence in re.split(r"(?<=[.;])\s+(?=[A-Z])|\n", text):
        low = sentence.lower()
        named = {mid for word, mid in ABLATION_WORDS.items() if word in low}
        if not named:
            continue
        allowed = {round(rows["lr_words_v2"][0], DIGITS)}  # plain TF-IDF is the reference in every comparison
        for mid in named:
            allowed |= {round(rows[mid][0], DIGITS), round(abs(rows[mid][1]), DIGITS)}
        bad += [n for n in re.findall(r"(?<![\d.])\d\.\d{3}(?!\d)", sentence) if round(float(n), DIGITS) not in allowed]
    return bad


@pytest.mark.parametrize("rel", ABLATION_TEXTS)
def test_ablation_numbers_in_prose_match_the_table(rel):
    """A stale F1 or delta for a feature family misleads the verdict, so prose must match ablation.csv."""
    bad = ablation_mismatches((REPO / rel).read_text(encoding="utf-8"), _ablation_rows())
    assert not bad, f"{rel}: ablation numbers that disagree with results/tables/ablation.csv: {bad}"


def test_ablation_checker_catches_a_changed_number():
    rows = _ablation_rows()
    assert ablation_mismatches("Style features gave 0.909, a change of -0.009 in validation F1.", rows) == []
    assert ablation_mismatches("Style features gave 0.913, a change of -0.001 in validation F1.", rows) == ["0.913", "0.001"]


SECTION_NUMBER = re.compile(r"(?<![\d.])\d\.\d{3}(?!\d)")


def section_mismatches(text: str, nums: set[float]) -> list[str]:
    """Three-decimal numbers in a section that appear nowhere in results/."""
    return [n for n in SECTION_NUMBER.findall(text) if not _found(float(n), DIGITS, nums)]


def test_every_three_decimal_number_in_sections_comes_from_results():
    """The site is read before the report; a stale 0.795 in a section misleads just as much."""
    nums = source_numbers()
    bad = {}
    for f in sorted((REPO / "content" / "sections").glob("*.md")):
        found = section_mismatches(re.sub(r"https?://\S+", " ", f.read_text(encoding="utf-8")), nums)
        if found:
            bad[f.name] = found
    assert not bad, f"three-decimal numbers in sections that are not in results/: {bad}"


def test_section_checker_catches_a_wrong_number():
    nums = source_numbers()
    assert section_mismatches("Precision is 0.985.", nums) == []
    assert section_mismatches("Precision is 0.214.", nums) == ["0.214"]


def test_worked_example_matches_metrics():
    """Section 08 explains scores with the LLM's test counts; they must be the real counts."""
    metrics = json.loads((REPO / "results" / "metrics.json").read_text(encoding="utf-8"))
    llm = next(r for r in metrics["comparison"]["table"] if r["model"] == "final_llm")
    text = (REPO / "content" / "sections" / "08-results-and-why-they-fail.md").read_text(encoding="utf-8").split("## The code")[0]
    for count in (llm["tp"], llm["tp"] + llm["fn"], llm["fp"], llm["fp"] + llm["tn"]):
        assert re.search(rf"\b{count}\b", text), f"worked example lacks the count {count}"
    for score in ("precision", "recall", "f1"):
        assert f"{llm[score]:.{DIGITS}f}" in text, f"worked example lacks {score}"
    assert "optimistic" in text
