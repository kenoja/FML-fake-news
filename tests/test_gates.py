"""The hard gates protect the validity of the comparison and the team's privacy. Each test shows
the failure a gate exists to catch."""
import json

from agent import tools


def test_scoring_one_model_on_test_with_two_configurations_fails(tmp_path, monkeypatch):
    # Tuning on the test set and reporting the best run inflates the score: only one config per model.
    monkeypatch.setattr(tools, "REPO", tmp_path)
    log = tmp_path / "results" / "test_access_log.jsonl"
    log.parent.mkdir()
    log.write_text(json.dumps({"model_id": "lr_words", "config_hash": "a"}) + "\n")
    assert tools.gate_test_access() == []
    with log.open("a") as f:
        f.write(json.dumps({"model_id": "lr_words", "config_hash": "a"}) + "\n")  # exact rerun is fine
    assert tools.gate_test_access() == []
    with log.open("a") as f:
        f.write(json.dumps({"model_id": "lr_words", "config_hash": "b"}) + "\n")
    assert tools.gate_test_access()


def test_changing_the_split_after_it_was_frozen_fails(tmp_path, monkeypatch):
    # Every comparison relies on the same split; re-splitting after seeing results is cherry-picking.
    monkeypatch.setattr(tools, "REPO", tmp_path)
    split = tmp_path / "data" / "splits" / "split_v1.csv"
    split.parent.mkdir(parents=True)
    split.write_text("doc_id,split\nC1,train\n")
    assert tools.gate_split_frozen() == []          # first sight: frozen
    assert tools.gate_split_frozen() == []          # unchanged: fine
    split.write_text("doc_id,split\nC1,test\n")
    assert tools.gate_split_frozen()


def test_private_terms_and_emails_are_caught(tmp_path, monkeypatch):
    # The repo is public; names and e-mail addresses must never be published.
    monkeypatch.setattr(tools, "REPO", tmp_path)
    monkeypatch.setenv("FML_PRIVACY_TERMS", "Jane Example,Example University")
    (tmp_path / "content").mkdir()
    page = tmp_path / "content" / "x.md"
    page.write_text("Written for the course.")
    assert tools.gate_privacy() == []
    page.write_text("Written by Jane Example.")
    assert tools.gate_privacy()
    page.write_text("See github.com/janeexampleuser")  # a term inside a longer word is not a match
    assert tools.gate_privacy() == []
    page.write_text("Contact: someone" + "@" + "university-mail.nl")  # split so this file passes the scan itself
    assert tools.gate_privacy()


def test_sections_without_required_parts_fail(tmp_path, monkeypatch):
    # Teammates rely on every section having the same four parts, short explanations and self-checks.
    monkeypatch.setattr(tools, "REPO", tmp_path)
    sec = tmp_path / "content" / "sections"
    sec.mkdir(parents=True)
    good = ("## Explained simply\nshort\n## The code\n<!-- cell: a.py#f -->\n## Report\ntext\n"
            "## Check yourself\nQ: one?\nA: yes\nQ: two?\nA: yes\n")
    (sec / "03-split-by-website.md").write_text(good)
    (sec / "99-glossary.md").write_text("### Term\nmeaning\n")
    assert tools.gate_sections() == []
    (sec / "03-split-by-website.md").write_text("## Explained simply\n" + "word " * 500 + "\n## Report\nx\n"
                                               "## Check yourself\nQ: only one?\n")
    fails = " | ".join(tools.gate_sections())
    assert "missing headings: ## The code" in fails and "no code cell" in fails
    assert "500 words" in fails and "1 question" in fails


def test_report_length_counts_only_the_report_parts(tmp_path, monkeypatch):
    # The course limit applies to the report; teaching text on the site must not count toward it.
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "project.yaml").write_text((tools.REPO / "config" / "project.yaml").read_text())
    monkeypatch.setattr(tools, "REPO", tmp_path)
    sec = tmp_path / "content" / "sections"
    sec.mkdir(parents=True)
    (sec / "01-a.md").write_text("---\ntitle: A\n---\n## Explained simply\n" + "teach " * 4000 +
                                 "\n## Report\n" + "report " * 3500 + "\n")
    assert tools.gate_report_length() == []
    (sec / "02-b.md").write_text("---\ntitle: B\n---\n## Report\n" + "more " * 2000 + "\n")
    assert "words" in tools.gate_report_length()[0]


def test_copied_article_text_is_detected(tmp_path, monkeypatch):
    # Full texts are not ours to publish; a 40-word copy must be caught wherever it lands.
    import gzip
    data = tmp_path / "data"
    (data / "loco_911").mkdir(parents=True)
    words = [f"w{i}" for i in range(200)]
    with gzip.open(data / "loco_911" / "loco_911.jsonl.gz", "wt") as f:
        f.write(json.dumps({"txt": " ".join(words)}) + "\n")
    repo = tmp_path / "repo"
    (repo / "content").mkdir(parents=True)
    monkeypatch.setattr(tools, "REPO", repo)
    monkeypatch.setenv("FML_DATA_DIR", str(data))
    (repo / "content" / "a.md").write_text("short quote: " + " ".join(words[5:20]))
    assert tools.gate_no_article_text() == []
    (repo / "content" / "a.md").write_text("long copy: " + " ".join(words[3:48]))
    assert tools.gate_no_article_text()


# ---- public CSV cells: justifications and excerpts stay short (privacy: no article text in public tables)

def _long_cells(folder, max_cell_words, max_just_words):
    import pandas as pd

    bad = []
    for f in sorted(folder.glob("*.csv")):
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        for col in df.columns:
            limit = max_just_words if "justification" in col or col == "llm_reason" else max_cell_words
            for i, cell in enumerate(df[col]):
                if len(cell.split()) > limit:
                    bad.append(f"{f.name}:{col}:row{i} has {len(cell.split())} words")
    return bad


def _limits():
    from fml.paths import project_config

    return project_config()["excerpts"]["max_words"]


def test_public_csv_cells_obey_excerpt_limits():
    from pathlib import Path

    tables = Path(__file__).resolve().parents[1] / "results" / "tables"
    limit = _limits()
    assert _long_cells(tables, limit, limit) == []
    assert _long_cells(tables, limit, 25) == [], "justifications must be at most 25 words"


def test_gate_catches_an_injected_60_word_cell(tmp_path):
    # A long justification is usually copied article text; the gate must see it.
    (tmp_path / "x.csv").write_text("doc_id,claude_justification\nA1," + " ".join(["w"] * 60) + "\n")
    assert _long_cells(tmp_path, _limits(), _limits())
