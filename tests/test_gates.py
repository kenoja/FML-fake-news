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
    # Teammates rely on every section having the same parts (code, explanation, pitfalls, infra).
    monkeypatch.setattr(tools, "REPO", tmp_path)
    sec = tmp_path / "content" / "sections"
    sec.mkdir(parents=True)
    (sec / "03-split-by-website.md").write_text("## Goal and hypothesis\n## Explanation\n")
    fails = tools.gate_sections()
    assert any("missing headings" in f for f in fails) and any("no code cell" in f for f in fails)


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
