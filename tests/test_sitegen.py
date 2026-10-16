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
