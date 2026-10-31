"""The PDF build must show every figure of the report and must not change any number while restyling it."""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "reports" / "latex"))

import build_pdf  # noqa: E402

REPORT = (REPO / "reports" / "report.md").read_text(encoding="utf-8")


def test_every_report_figure_has_a_pdf_version():
    """A figure without a restyled version would make the build fail or drop the figure silently."""
    used = {Path(p).name for p in re.findall(r"\]\((\.\./results/figures/[^)]+)\)", REPORT)}
    assert used, "the report shows no figures"
    assert used <= set(build_pdf.FIGURES), f"no PDF version for {sorted(used - set(build_pdf.FIGURES))}"


def test_restyling_keeps_every_number():
    """Captions, widths and dashes change the layout only: the numbers must be the same before and after."""
    _, _, summary, body = build_pdf.split_report(REPORT)
    restyled = build_pdf.column_widths(build_pdf.interval_dashes(build_pdf.table_captions(body)))
    numbers = lambda text: sorted(re.findall(r"\d+(?:[.,]\d+)*", text))  # noqa: E731
    assert numbers(restyled) == numbers(body)
    assert "Executive summary" not in body and summary


def test_interval_dashes_only_touch_intervals():
    assert build_pdf.interval_dashes("0.607 (0.500 to 0.714)") == "0.607 (0.500–0.714)"
    assert build_pdf.interval_dashes("from 3,146 to 2,608") == "from 3,146 to 2,608"


def test_apa_numbers_drop_the_leading_zero_only_for_scores():
    """APA 7 writes values that cannot exceed 1 without a leading zero; C can exceed 1 and money is not a score."""
    text = "F1 0.721 (0.614–0.824), p = 0.029, p < 0.001, recall 1.000, C = 0.01 from {0.01, 0.1, 1}, $0.26, 10.1162/x"
    assert build_pdf.apa_numbers(text) == (
        "F1 .721 (.614–.824), p = .029, p < .001, recall 1.000, C = 0.01 from {0.01, 0.1, 1}, $0.26, 10.1162/x")


def test_short_entries_and_headers_get_enough_width():
    """A narrow yes/no column next to long cells used to wrap its header one word per line."""
    header = ["Method", "Trained on our data", "Settings"]
    body = [["1. Word counts + logistic regression", "yes", "C from [0.01, 0.1, 1, 10, 100], class weight balanced"]]
    trained = build_pdf.widths(header, body)[1]
    assert trained >= len("Trained on"), "the header needs room for two lines"
    assert build_pdf.widths(["Website", "Recall: TF-IDF + features"], [["Website 1", "0.684"]])[0] >= len("Website 1")
