"""Build the PDF report from reports/report.md: restyled figures, a title page and one colour scheme.

Usage:  PYTHONPATH=src python reports/latex/build_pdf.py
Names are never stored in this public repository: set FML_REPORT_AUTHORS to print them on the title page and
FML_REPORT_OUT to write that copy outside the repository.
"""
from __future__ import annotations

import datetime as dt
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BUILD = HERE / "build"
sys.path.insert(0, str(HERE))

import figures  # noqa: E402

# the report's figures and the restyled version used in the PDF (with its width on the page)
FIGURES = {
    "word_counts_validation_confusion.png": ("cm_word_counts.pdf", "45%"),
    "tfidf_validation_confusion.png": ("cm_tfidf.pdf", "45%"),
    "llm_validation_confusion.png": ("cm_llm.pdf", "45%"),
    "ablation_f1.png": ("ablation.pdf", "88%"),
    "comparison_confusion.png": ("cm_test.pdf", "100%"),
    "f1_validation_test.png": ("f1_drop.pdf", "88%"),
}


def website_sizes():
    """Articles per website, set and label (counts only), for the title visual; None without the data repo."""
    try:
        from fml.pipeline import analysis_frame
        from fml.split import load_split
    except ImportError:
        return None
    try:
        frame = analysis_frame()[["doc_id", "website", "label"]].merge(load_split()[["doc_id", "split"]], on="doc_id")
    except Exception:  # the private data repo is missing: build without the title visual
        return None
    sizes = frame.groupby(["split", "label", "website"]).size().rename("articles").reset_index()
    return sizes.drop(columns="website")                       # no website names leave this function


def split_report(text: str) -> tuple[str, str, str, str]:
    """Title, subtitle, executive summary and body of the assembled report."""
    first, rest = text.split("\n", 1)
    title, _, subtitle = first.lstrip("# ").partition("? ")
    summary = re.search(r"### Executive summary\n(.*?)(?=\n### )", rest, flags=re.S).group(1).strip()
    body = rest.replace(f"### Executive summary\n\n{summary}\n", "", 1)
    return title + "?", subtitle, summary, body


def table_captions(md: str) -> str:
    """Turn the italic line under a table into a pandoc table caption."""
    lines, out = md.split("\n"), []
    last = ""
    for line in lines:
        match = re.fullmatch(r"\*([^*].*[^*])\*", line.strip())
        if match and last.startswith("|"):
            out.append(f"Table: {match.group(1)}")
        else:
            out.append(line)
        if line.strip():
            last = line.strip()
    return "\n".join(out)


def interval_dashes(md: str) -> str:
    """Write intervals as (0.500–0.714) so that table cells stay on one line."""
    return re.sub(r"\((\d\.\d{3}) to (\d\.\d{3})\)", "(\\1–\\2)", md)


def column_widths(md: str) -> str:
    """Give each table column a width that fits its content (pandoc reads widths from the dash line)."""
    lines = md.split("\n")
    i = 0
    while i < len(lines):
        if lines[i].startswith("|") and i + 1 < len(lines) and re.fullmatch(r"\|(\s*:?-+:?\s*\|)+", lines[i + 1].strip()):
            j = i + 2
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for k, line in enumerate(lines[i:j]) if k != 1]
            body = rows[1:] or rows
            longest = [max(len(row[c]) for row in body if c < len(row)) for c in range(len(rows[0]))]
            header = [max(len(word) for word in cell.split()) if cell else 0 for cell in rows[0]]  # headers wrap at spaces
            weights = [min(max(n, h, 6), 40) for n, h in zip(longest, header)]
            lines[i + 1] = "|" + "|".join("-" * w for w in weights) + "|"
            lines[i] = lines[i] + " " * max(0, sum(weights) + len(weights) + 1 - len(lines[i]) + 80)  # force relative widths
            i = j
        else:
            i += 1
    return "\n".join(lines)


def figure_links(md: str) -> str:
    """Point every report figure to its restyled PDF version."""
    def swap(match):
        caption, name = match.group(1), Path(match.group(2)).name
        pdf, width = FIGURES[name]
        return f"![{caption}](figures/{pdf}){{width={width}}}"
    return re.sub(r"!\[([^\]]*)\]\((\.\./results/figures/[^)]+)\)", swap, md)


def pandoc(md: str) -> str:
    result = subprocess.run(["pandoc", "-f", "markdown", "-t", "latex", "--shift-heading-level-by=-1", "--wrap=none"],
                            input=md, capture_output=True, text=True, check=True)
    return result.stdout


def polish(tex: str) -> str:
    """Run-in labels in navy, unnumbered references with hanging indent, figures kept in place."""
    tex = re.sub(r"^\\textbf\{([^{}]{1,80}?\.)\}", r"\\runin{\1}", tex, flags=re.M)
    tex = tex.replace("\\section{References}\\label{references}",
                      "\\clearpage\\section*{References}\\addcontentsline{toc}{section}{References}\\markboth{References}{}")
    head, sep, refs = tex.partition("\\section*{References}")
    refs = refs.replace("\\begin{itemize}", "\\begin{itemize}[label={}, leftmargin=1.4em, itemindent=-1.4em, itemsep=3pt]\n\\small", 1)
    tex = head + sep + refs
    tex = re.sub(r"\\begin\{longtable\}.*?\\end\{longtable\}", keep_together, tex, flags=re.S)
    return tex.replace("\\begin{figure}", "\\begin{figure}[H]")


def keep_together(match) -> str:
    """Start a table on a new page if it would not fit on this one (tables here are short)."""
    rows = match.group(0).count("\\\\")
    return f"\\needspace{{{2 * rows + 4}\\baselineskip}}\n" + match.group(0)


def title_numbers() -> dict:
    """Website 4's share of the conspiracy test articles, read from the results (never typed)."""
    import json
    metrics = json.loads((REPO / "results" / "metrics.json").read_text(encoding="utf-8"))["comparison"]
    largest = max(metrics["per_website"], key=lambda row: row["test_articles"])
    return {"w4-articles": str(largest["test_articles"]), "test-conspiracy": str(metrics["test_conspiracy"])}


def fill(template: str, values: dict) -> str:
    for key, value in values.items():
        template = template.replace(f"${key}$", value)
    return template


def main() -> Path:
    BUILD.mkdir(exist_ok=True)
    figures.build_all(REPO, BUILD / "figures", website_sizes())
    title, subtitle, summary, body = split_report((REPO / "reports" / "report.md").read_text(encoding="utf-8"))
    body_tex = polish(pandoc(figure_links(column_widths(interval_dashes(table_captions(body))))))
    visual = "figures/title_visual.pdf" if (BUILD / "figures" / "title_visual.pdf").is_file() else ""
    tex = fill((HERE / "template.tex").read_text(encoding="utf-8"), {
        "title": title, "subtitle": subtitle, "short-title": title,
        "kicker": "Machine-learning project report",
        "author": os.environ.get("FML_REPORT_AUTHORS", "Members A, B, C, D and E"),
        "date": dt.date.today().strftime("%-d %B %Y"),
        "title-visual": visual, "summary": pandoc(summary), "body": body_tex,
        **title_numbers(),
    })
    if not visual:
        tex = tex.replace("\\includegraphics[width=\\textwidth]{}", "")
    (BUILD / "report.tex").write_text(tex, encoding="utf-8")
    subprocess.run(["latexmk", "-xelatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
                   cwd=BUILD, check=True, capture_output=True)
    pdf = Path(os.environ.get("FML_REPORT_OUT", REPO / "reports" / "report.pdf"))   # a copy with names goes elsewhere
    shutil.copy(BUILD / "report.pdf", pdf)
    return pdf


if __name__ == "__main__":
    print(main())
