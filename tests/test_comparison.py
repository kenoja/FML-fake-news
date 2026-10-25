"""Why: the verdict "two models differ" rests on the discordant pairs and the intervals, so a wrong 2x2 table,
a one-sided p-value or a table with the wrong columns would silently change the conclusions in section 08."""
from __future__ import annotations

import json
from math import comb

import numpy as np
import pytest

from fml.comparison import mcnemar_pair, model_row
from fml.results import results_dir

# 12 documents, 6 per class. A is wrong on docs 5 and 10, B on docs 0, 1 and 11.
Y = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
A = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0], dtype=bool)
B = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1], dtype=bool)


def hand_mcnemar_p(k: int, n: int) -> float:
    """Two-sided exact binomial(n, 0.5) p-value: 2 * P(X <= k), capped at 1."""
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def test_discordant_counts_and_p_value_match_hand_computation():
    r = mcnemar_pair(Y, A, B, "a", "b")
    # A is right where B is wrong on docs {0, 1, 11}; B is right where A is wrong on docs {5, 10}
    assert (r["a_only_correct"], r["b_only_correct"]) == (3, 2)
    assert r["both_correct"] == 7 and r["both_wrong"] == 0
    assert r["p_value"] == pytest.approx(hand_mcnemar_p(2, 5))


def test_clear_difference_gives_small_p():
    y = np.ones(30, dtype=bool)
    r = mcnemar_pair(y, np.ones(30, dtype=bool), np.zeros(30, dtype=bool), "a", "b")
    assert (r["a_only_correct"], r["b_only_correct"]) == (30, 0)
    assert r["p_value"] == pytest.approx(hand_mcnemar_p(0, 30)) and r["p_value"] < 1e-6


def test_model_row_has_counts_scores_and_intervals_only():
    row = model_row("m", Y, A)
    assert (row["tp"], row["fn"], row["fp"], row["tn"]) == (5, 1, 1, 5)
    assert row["precision"] == pytest.approx(5 / 6) and row["recall"] == pytest.approx(5 / 6)
    for score in ("precision", "recall", "f1"):
        assert row[f"{score}_ci_low"] <= row[score] <= row[f"{score}_ci_high"]
    assert "macro_f1" not in row           # only the statistics listed in AGENTS.md


def test_bootstrap_is_reproducible():
    assert model_row("m", Y, A) == model_row("m", Y, A)


def test_always_mainstream_baseline_scores_zero_but_looks_accurate_on_accuracy():
    """Why: shows that accuracy hides the failure on the rarer conspiracy class, so F1 is used instead."""
    row = model_row("b", Y, np.zeros(len(Y), dtype=bool))
    assert (row["precision"], row["recall"], row["f1"]) == (0, 0, 0)
    assert (row["tp"], row["fp"]) == (0, 0) and row["tn"] == 6


def test_website_recall_counts_only_conspiracy_articles_of_each_site():
    from fml.pipeline import conspiracy_website_recall
    y = np.array([1, 1, 1, 0, 0], dtype=bool)
    sites = np.array(["a", "a", "b", "a", "c"])
    table = conspiracy_website_recall(y, {"m": np.array([1, 0, 1, 1, 0], dtype=bool)}, sites)
    assert list(table["website"]) == ["a", "b"]              # mainstream-only site c is not listed
    assert list(table["test_articles"]) == [2, 1] and list(table["recall_m"]) == [0.5, 1.0]


def test_stage_output_has_four_models_plus_baseline_and_three_tests():
    """Real results: the stored McNemar counts must equal a re-count from the saved predictions."""
    path = results_dir() / "metrics.json"
    comp = json.loads(path.read_text())["comparison"] if path.exists() else {}
    if not comp.get("mcnemar"):
        pytest.skip("comparison stage has not run")
    from fml.pipeline import read_final_predictions
    models = [r["model"] for r in comp["table"]]
    assert models == ["final2_word_counts", "final2_tfidf", "final2_tfidf_features", "final_llm", "always_mainstream"]
    assert comp["baseline"]["accuracy"] == pytest.approx(1 - comp["test_conspiracy"] / comp["test_documents"], abs=1e-3)
    assert (results_dir() / "tables" / "test_website_recall.csv").is_file()
    assert set(comp["mcnemar"]) == {"words_vs_tfidf", "tfidf_vs_features", "best_lr_vs_llm"}
    lr, llm = read_final_predictions()
    llm = llm[llm["pass"] == 1].set_index("doc_id")
    best = lr[lr["model_id"] == comp["lr_best"]].set_index("doc_id").loc[llm.index]
    y = llm["label"].to_numpy(dtype=bool)
    a_ok, b_ok = best["pred"].to_numpy(dtype=bool) == y, llm["pred"].to_numpy(dtype=bool) == y
    m = comp["mcnemar"]["best_lr_vs_llm"]
    assert (m["a_only_correct"], m["b_only_correct"]) == (int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum()))
    n = m["a_only_correct"] + m["b_only_correct"]
    assert m["p_value"] == pytest.approx(hand_mcnemar_p(min(m["a_only_correct"], m["b_only_correct"]), n), abs=1e-4)
