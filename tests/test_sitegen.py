"""The walkthrough shows the code that actually ran: cells are read from the repo at build time,
and a cell that cannot be built is shown as broken and fails a strict build."""
import html
import json
import re

import pytest

from sitegen import build as sitegen


def write_section(content, body):
    (content / "sections").mkdir(parents=True, exist_ok=True)
    (content / "sections" / "01-demo.md").write_text(
        "---\ntitle: Demo\npart: Evaluation\nhypothesis: It works.\n---\n" + body)


def test_code_and_output_cells_are_rendered_from_real_files(tmp_path, monkeypatch):
    results = sitegen.REPO / "results"
    created = not results.exists()
    results.mkdir(exist_ok=True)
    probe = results / "_sitegen_probe.json"
    probe.write_text(json.dumps({"baseline": {"f1": 0.71234, "n": 3}}))
    try:
        content = tmp_path / "content"
        write_section(content, "## Goal and hypothesis\nx\n<!-- cell: src/fml/data.py#load_corpus -->\n"
                               "<!-- output: results/_sitegen_probe.json#baseline | F1 on validation -->\n")
        res = sitegen.build(cycle=1, strict=True, out=tmp_path / "site", content=content)
        page = (tmp_path / "site" / "index.html").read_text()
        text = html.unescape(re.sub(r"<[^>]+>", "", page))               # drop syntax-highlighting tags
        assert "In&nbsp;[1]:" in page and "def load_corpus() -> pd.DataFrame:" in text  # real source
        assert "0.712" in page and "F1 on validation" in page
        assert 'data-term="cycle-01/01-demo"' in page                  # feedback thread per section
        assert res.warnings == []
    finally:
        probe.unlink()
        if created:
            results.rmdir()


def test_a_missing_reference_fails_the_strict_build(tmp_path):
    content = tmp_path / "content"
    write_section(content, "<!-- cell: src/fml/does_not_exist.py#nothing -->\n")
    with pytest.raises(SystemExit):
        sitegen.build(cycle=1, strict=True, out=tmp_path / "site", content=content)
    page = (tmp_path / "site" / "index.html").read_text()
    assert "could not be built" in page


def four_part_demo(content):
    (content / "sections").mkdir(parents=True, exist_ok=True)
    (content / "sections" / "99-glossary.md").write_text(
        "---\ntitle: Glossary\n---\n### F1 score\nAlso: F1\nOne number that combines precision and recall. More.\n\n"
        "### Precision\nOf the articles called conspiracy, the share that really are.\n")
    write_section(content, "## Explained simply\nWe use the F1 score here, and F1 again.\n`F1` in code stays.\n"
                           "## The code\n<!-- cell: src/fml/data.py#load_corpus -->\n"
                           "## Report\nOur F1 score was good.\n"
                           "## Check yourself\nQ: What does precision measure?\nA: The share of correct alarms.\n")
    (content / "references.md").write_text("Miani, A. et al. (2022). LOCO.")


def test_report_is_assembled_from_report_parts_only(tmp_path):
    # Why: the report must say exactly what the site's report boxes say, and nothing from the
    # teaching text, so teammates review the real report while they learn.
    content = tmp_path / "content"
    four_part_demo(content)
    report = tmp_path / "report.md"
    sitegen.build(cycle=2, strict=True, out=tmp_path / "site", content=content, report_out=report)
    text = report.read_text()
    assert "## Demo\n\nOur F1 score was good." in text and "## References" in text
    assert "We use the F1 score" not in text and "precision measure" not in text and "load_corpus" not in text
    assert sitegen.report_text(content).split("\n", 1)[1] in text  # gate sees the same words


def test_glossary_terms_are_linked_once_and_never_inside_code(tmp_path):
    # Why: non-technical readers must be able to jump to a definition, but links inside code would break it.
    content = tmp_path / "content"
    four_part_demo(content)
    sitegen.build(cycle=2, strict=True, out=tmp_path / "site", content=content, report_out=tmp_path / "r.md")
    page = (tmp_path / "site" / "index.html").read_text()
    demo = page.split('id="01-demo"')[1].split('id="99-glossary"')[0]
    explained = demo.split("part-explained")[1].split("part-code")[0]
    assert explained.count('class="gl" href="#g-f1-score"') == 1          # first use only
    assert '<code>F1</code>' in explained                                   # code untouched
    assert 'title="One number that combines precision and recall."' in explained
    assert 'id="g-f1-score"' in page and 'id="g-precision"' in page


def test_check_yourself_answers_are_hidden_until_clicked(tmp_path):
    # Why: the questions are for self-assessment; showing the answer next to the question defeats them.
    content = tmp_path / "content"
    four_part_demo(content)
    sitegen.build(cycle=2, strict=True, out=tmp_path / "site", content=content, report_out=tmp_path / "r.md")
    page = (tmp_path / "site" / "index.html").read_text()
    assert re.search(r'<details class="check"><summary>What does <a [^>]+>precision</a> measure\?</summary>'
                     r'<p>The share of correct alarms.</p></details>', page)


def test_a_single_number_can_be_quoted_in_the_report_text(tmp_path):
    # Why: report prose often cites one value (e.g. an agreement share); it must build, or CI fails.
    results = sitegen.REPO / "results"
    probe = results / "_sitegen_scalar.json"
    results.mkdir(exist_ok=True)
    probe.write_text(json.dumps({"llm": {"agreement": 0.96123}}))
    try:
        content = tmp_path / "content"
        write_section(content, "## Report\n<!-- output: results/_sitegen_scalar.json#llm.agreement | Agreement -->\n")
        res = sitegen.build(cycle=2, strict=True, out=tmp_path / "site", content=content, report_out=tmp_path / "r.md")
        assert res.warnings == [] and "**Agreement:** 0.961" in (tmp_path / "r.md").read_text()
    finally:
        probe.unlink()


def test_comments_and_code_links_get_the_repository_from_github(monkeypatch):
    # Why: giscus shows nothing without the repository name, but the name contains the owner's
    # username, which must not be stored in the repo. GitHub supplies it during the Pages build.
    monkeypatch.setenv("GITHUB_REPOSITORY", "someone/some-repo")
    settings = sitegen.site_settings()
    assert settings["giscus"]["repo"] == "someone/some-repo"
    assert settings["repo_url"] == "https://github.com/someone/some-repo"
