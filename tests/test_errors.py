"""Error analysis: disagreement sets, the label-vs-model rule, the human-check file, excerpt limits.

Why it matters: H6 ("many errors are label noise") is only credible if the sets are counted correctly, the rule
is explicit, the human-check file is never overwritten, and no public file quotes more than the allowed excerpt.
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
    ])
def test_error_rule(true_label, stance, verdict):
    assert errors.classify_error(true_label, stance) == verdict


def test_unreadable_stance_is_not_judged():
    with pytest.raises(ValueError):
        errors.classify_error(True, errors.UNREADABLE)


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
    assert errors.parse_stance("probably endorses")["stance"] == errors.UNREADABLE
    assert errors.parse_stance('{"stance": "maybe"}')["stance"] == errors.UNREADABLE
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
            return Completion('{"stance": "no_claim", "justification": "plain news"}', 1)

    df = pd.DataFrame({"doc_id": ["x"], "txt": ["secret article words"]})
    path = tmp_path / "stance.jsonl"
    errors.code_stances(df, client=Fake(), cache_file=path)
    errors.code_stances(df, client=None, cache_file=path)   # second run: cache only, no client needed
    assert Fake.calls == 1
    rec = json.loads(path.read_text().splitlines()[0])
    assert set(rec) == {"doc_id", "prompt_hash", "answer"}
    assert "secret" not in path.read_text()


def test_classification_table_rows_sum_to_errors():
    """Every error must appear in exactly one column, or a reader cannot reconcile the table with the error count."""
    sets = pd.DataFrame({"doc_id": list("abcdef"), "set": ["both_wrong"] * 4 + ["lr_only_wrong"] * 2,
                         "error_type": ["false negative"] * 3 + ["false positive"] * 3})
    coded = sets.iloc[[0, 1, 3, 4]].copy()          # the other errors have no usable stance
    coded["claude_stance"] = ["endorses", "no_claim", "endorses", "rejects_or_debunks"]
    coded["true_label"] = [True, True, False, False]
    coded["verdict"] = [errors.classify_error(t, s) for t, s in zip(coded["true_label"], coded["claude_stance"])]
    table = errors.error_classification_table(coded, sets)
    assert "unparsed" not in table.columns
    parts = [errors.MODEL_WRONG, errors.LABEL_PROBLEM, "not_sampled"]
    assert (table[parts].sum(axis=1) == table["errors_total"]).all()
    assert table.loc[(table["set"] == "all") & (table["error_type"] == "all"), "not_sampled"].item() == 2


def _sample(tmp_path, human):
    p = tmp_path / "s.csv"
    pd.DataFrame({"doc_id": list("abcd"), "claude_stance": ["endorses", "no_claim", "endorses", "reports_neutrally"],
                  "human_stance": human}).to_csv(p, index=False)
    return p


def test_justification_is_shortened_and_names_are_stripped():
    text = "The article quotes Jane Doe and Scott Wolter about the WTC " + " ".join(["word"] * 40)
    out = errors.scrub_justification(text)
    assert len(out.split()) <= 25
    assert "Jane" not in out and "Wolter" not in out and "[name]" in out and "WTC" in out


def test_check_sheet_is_blind_and_claude_answers_are_separate(tmp_path):
    """The team must not see Claude's stance while coding, or their check is not independent."""
    p = tmp_path / "new.csv"
    sample = pd.DataFrame({"doc_id": list("ab"), "website": ["x.com", "y.org"], "claude_stance": ["no_claim", "endorses"],
                           "claude_justification": ["j1", "j2"], "set": ["x", "y"]})
    assert errors.write_check_sample(sample, p) is True
    out = pd.read_csv(p, dtype=str, keep_default_na=False)
    assert list(out.columns) == ["doc_id", "website", "human_stance"]
    assert (out["human_stance"] == "").all()
    errors.write_claude_answers(sample, tmp_path / "answers.csv")
    assert list(pd.read_csv(tmp_path / "answers.csv").columns) == ["doc_id", "claude_stance", "claude_justification"]


def test_check_sample_caps_rows_per_website_and_balances_classes():
    """One prolific site must not fill the human check, or it would tell us about that site only."""
    sites = ["big.com"] * 20 + [f"s{i}.org" for i in range(10)] + ["c1.net"] * 15 + [f"c{i}.net" for i in range(2, 8)]
    df = pd.DataFrame({"doc_id": [f"d{i}" for i in range(len(sites))], "website": sites,
                       "label": [False] * 30 + [True] * 21})
    sample = errors.balanced_website_sample(df, 20, cap=4, seed=1)
    assert len(sample) == 20 and sample["label"].sum() == 10
    assert sample["website"].value_counts().max() <= 4


def test_examples_cover_every_error_cell_and_a_correct_contrast():
    sets = errors.disagreement_sets(LR, LLM)
    correct = pd.DataFrame({"doc_id": ["f", "z"], "true_label": [False, True]})
    ex = errors.pick_examples(sets, correct)
    assert set(ex["cell"]) == {"lr_only_wrong", "llm_only_wrong", "both_wrong", "correct_mainstream", "correct_conspiracy"}
    assert ex["doc_id"].is_unique


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
    assert cut["stance"] == errors.UNREADABLE
