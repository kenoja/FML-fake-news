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
