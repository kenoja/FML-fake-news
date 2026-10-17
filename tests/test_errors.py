"""Error analysis: disagreement sets, the label-vs-model rule, honest human-check reporting, excerpt limits.

Why it matters: H6 ("many errors are label noise") is only credible if the sets are counted correctly, the rule
is explicit, kappa is never invented, and no public file quotes more than the allowed excerpt.
"""
from __future__ import annotations

import json
import re

import pandas as pd
import pytest

from fml import errors
from fml.paths import REPO_ROOT, project_config

LR = pd.DataFrame({"doc_id": list("abcdef"), "label": [True, True, True, False, False, False],
                   "pred": [True, False, False, True, False, False]})
LLM = pd.DataFrame({"doc_id": list("abcdef"), "label": LR["label"],
                    "pred": [False, True, False, True, True, False]})


def test_disagreement_sets_by_hand():
    s = errors.disagreement_sets(LR, LLM).set_index("doc_id")
    assert s.loc["a", "set"] == "llm_only_wrong"          # LR right, LLM wrong
    assert s.loc["b", "set"] == "lr_only_wrong"
    assert s.loc["c", "set"] == "both_wrong" and s.loc["c", "error_type"] == "false negative"
    assert s.loc["d", "set"] == "both_wrong" and s.loc["d", "error_type"] == "false positive"
    assert s.loc["e", "set"] == "llm_only_wrong" and s.loc["e", "error_type"] == "false positive"
    assert "f" not in s.index                              # both right: not a disagreement document


def test_mismatched_documents_fail_loudly():
    with pytest.raises(RuntimeError):
        errors.disagreement_sets(LR, LLM.iloc[:5])


@pytest.mark.parametrize("true_label,stance,verdict", [
    (True, "endorses", errors.MODEL_WRONG), (True, "rejects_or_debunks", errors.LABEL_PROBLEM),
    (True, "reports_neutrally", errors.LABEL_PROBLEM), (True, "no_claim", errors.LABEL_PROBLEM),
    (False, "endorses", errors.LABEL_PROBLEM), (False, "rejects_or_debunks", errors.MODEL_WRONG),
    (False, "reports_neutrally", errors.MODEL_WRONG), (False, "no_claim", errors.MODEL_WRONG),
    (True, "parse_error", errors.UNCODED)])
def test_error_rule(true_label, stance, verdict):
    assert errors.classify_error(true_label, stance) == verdict


def test_both_wrong_always_coded_and_others_capped_with_seed():
    sets = pd.DataFrame({"doc_id": [f"d{i:03d}" for i in range(100)],
                         "set": ["both_wrong"] * 30 + ["lr_only_wrong"] * 60 + ["llm_only_wrong"] * 10})
    a, b = errors.select_for_coding(sets, 20, 1), errors.select_for_coding(sets, 20, 1)
    assert a.equals(b)
    assert (a["set"] == "both_wrong").sum() == 30 and (a["set"] == "lr_only_wrong").sum() == 20
    assert (a["set"] == "llm_only_wrong").sum() == 10


def test_stratified_sample_size_and_every_set_present():
    df = pd.DataFrame({"doc_id": range(76), "set": ["both_wrong"] * 41 + ["lr_only_wrong"] * 28 + ["llm_only_wrong"] * 7})
    s = errors.stratified_sample(df, 40, 5)
    assert len(s) == 40 and s["doc_id"].is_unique
    assert set(s["set"]) == {"both_wrong", "lr_only_wrong", "llm_only_wrong"}
    assert errors.stratified_sample(df, 40, 5).equals(s)


def test_parse_stance_never_guesses():
    assert errors.parse_stance('{"stance": "Endorses", "justification": "x"}')["stance"] == "endorses"
    assert errors.parse_stance("probably endorses")["stance"] == errors.PARSE_ERROR
    assert errors.parse_stance('{"stance": "maybe"}')["stance"] == errors.PARSE_ERROR
    long = errors.parse_stance(json.dumps({"stance": "no_claim", "justification": " ".join(["w"] * 60)}))
    assert len(long["justification"].split()) == errors.JUSTIFICATION_MAX_WORDS


def test_missing_key_and_cache_fail_loudly(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_CLASSIFIER_API_KEY", raising=False)
    df = pd.DataFrame({"doc_id": ["x"], "txt": ["some text"]})
    with pytest.raises(RuntimeError, match="no API key"):
        errors.code_stances(df, cache_file=tmp_path / "stance.jsonl")


def test_cache_holds_only_id_hash_answer_and_is_reused(tmp_path):
    class Fake:
        calls = 0

        def complete(self, prompt, article):
            Fake.calls += 1
            from fml.llm import Completion
            return Completion('{"stance": "no_claim", "justification": "plain news"}', 1, 1)

    df = pd.DataFrame({"doc_id": ["x"], "txt": ["secret article words"]})
    path = tmp_path / "stance.jsonl"
    errors.code_stances(df, client=Fake(), cache_file=path)
    errors.code_stances(df, client=None, cache_file=path)   # second run: cache only, no client needed
    assert Fake.calls == 1
    rec = json.loads(path.read_text().splitlines()[0])
    assert set(rec) == {"doc_id", "prompt_hash", "answer"}
    assert "secret" not in path.read_text()


def _sample(tmp_path, human):
    p = tmp_path / "s.csv"
    pd.DataFrame({"doc_id": list("abcd"), "claude_stance": ["endorses", "no_claim", "endorses", "reports_neutrally"],
                  "human_stance": human, "human_note": [""] * 4}).to_csv(p, index=False)
    return p


def test_kappa_not_yet_verified_when_empty(tmp_path):
    r = errors.human_agreement(_sample(tmp_path, [""] * 4))
    assert r["status"] == "not yet verified" and r["kappa"] is None


def test_kappa_matches_hand_value_and_rejects_bad_labels(tmp_path):
    # human: endorses, no_claim, no_claim, reports_neutrally vs claude above: 3/4 agree.
    # chance agreement: claude (.5,.25,.25 over endorses/no_claim/neutral), human (.25,.5,.25) -> .125+.125+.0625=.3125
    r = errors.human_agreement(_sample(tmp_path, ["endorses", "no_claim", "no_claim", "reports_neutrally"]))
    assert r["kappa"] == pytest.approx((0.75 - 0.3125) / (1 - 0.3125))
    with pytest.raises(ValueError):
        errors.human_agreement(_sample(tmp_path, ["yes", "", "", ""]))


def test_rerun_never_overwrites_human_work(tmp_path):
    p = _sample(tmp_path, ["endorses", "", "", ""])
    sample = pd.DataFrame({c: ["x"] for c in errors.CHECK_COLUMNS if c != "human_stance"})
    assert errors.write_check_sample(sample.assign(true_label=True, lr_pred=True, llm_pred=False), p) is False
    assert "endorses" in p.read_text()


# ---------------------------------------------------------------- excerpt discipline (AGENTS.md hard rule)

def _excerpts(md: str) -> list[str]:
    """Excerpts are blockquote paragraphs (lines starting with '>'; an empty '>' line ends one)."""
    blocks, cur = [], []
    for line in md.splitlines():
        if line.startswith(">") and line.lstrip("> ").strip():
            cur.append(line.lstrip("> ").strip())
        elif cur:
            blocks.append(" ".join(cur))
            cur = []
    if cur:
        blocks.append(" ".join(cur))
    return [re.sub(r"\s+", " ", b).strip() for b in blocks if b.strip()]


def test_every_excerpt_in_content_is_short_and_few():
    cfg = project_config()["excerpts"]
    for md in (REPO_ROOT / "content").rglob("*.md"):
        ex = _excerpts(md.read_text(encoding="utf-8"))
        assert len(ex) <= cfg["max_per_section"], f"{md.name}: {len(ex)} excerpts"
        for e in ex:
            assert len(e.split()) <= cfg["max_words"], f"{md.name}: excerpt of {len(e.split())} words"


def test_excerpt_checker_catches_a_long_quote():
    assert len(_excerpts("text\n\n> " + " ".join(["w"] * 60) + "\n\nmore")[0].split()) == 60


def test_parse_stance_tolerates_reasoning_before_complete_json_only():
    # The coder model sometimes reasons first; a complete JSON object after the prose is fine, a cut-off one is not.
    ok = errors.parse_stance('It is a news item.\n\n{"stance":"no_claim","justification":"plain news"}')
    assert ok["stance"] == "no_claim"
    cut = errors.parse_stance('It is news.\n\n{"stance":"no_claim","justification":"cut off')
    assert cut["stance"] == errors.PARSE_ERROR
