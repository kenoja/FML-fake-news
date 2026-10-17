"""Head-to-head statistics: McNemar counts and p-values must match a hand re-computation.

Why it matters: the verdict "LR and LLM differ" rests on the discordant pairs, so a wrong 2x2 table
or a one-sided p-value would silently change the conclusion of section 10.
"""
from __future__ import annotations

import json
from math import comb

import numpy as np
import pytest

from fml.comparison import length_bucket_f1, model_row, pairwise, per_website_table
from fml.results import results_dir

# 12 documents, 6 per class. A is wrong on docs 5 and 10, B on docs 0, 1 and 11.
Y = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
A = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0], dtype=bool)   # wrong on docs 5 and 10
B = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1], dtype=bool)   # wrong on docs 0, 1 and 11
SITES = np.array(list("aabbccddeeff"))


def hand_mcnemar_p(k: int, n: int) -> float:
    """Two-sided exact binomial(n, 0.5) p-value: 2 * P(X <= k), capped at 1."""
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def test_discordant_counts_and_p_value_match_hand_computation():
    r = pairwise(Y, A, B, SITES, "a", "b")
    # A-only right = {0, 1, 11} (B wrong, A right); B-only right = {5, 10} (A wrong, B right).
    assert (r["a_only_correct"], r["b_only_correct"]) == (3, 2)
    assert r["both_correct"] + r["both_wrong"] + 5 == len(Y)
    assert r["both_wrong"] == 0 and r["both_correct"] == 7
    assert r["mcnemar_p"] == pytest.approx(hand_mcnemar_p(2, 5))


def test_clear_difference_gives_small_p():
    y = np.ones(30, dtype=bool)
    a, b = np.ones(30, dtype=bool), np.zeros(30, dtype=bool)
    r = pairwise(y, a, b, np.arange(30) % 5, "a", "b")
    assert (r["a_only_correct"], r["b_only_correct"]) == (30, 0)
    assert r["mcnemar_p"] == pytest.approx(hand_mcnemar_p(0, 30))
    assert r["mcnemar_p"] < 1e-6


def test_delta_f1_sign_and_estimate():
    r = pairwise(Y, A, B, SITES, "a", "b")
    f1 = lambda p: 2 * (Y & p).sum() / (2 * (Y & p).sum() + (~Y & p).sum() + (Y & ~p).sum())  # noqa: E731
    assert r["delta_f1_b_minus_a"]["documents"]["estimate"] == pytest.approx(f1(B) - f1(A))
    lo, hi = r["delta_f1_b_minus_a"]["websites"]["ci_low"], r["delta_f1_b_minus_a"]["websites"]["ci_high"]
    assert lo <= r["delta_f1_b_minus_a"]["websites"]["estimate"] <= hi


def test_model_row_counts_and_both_cis_present():
    row = model_row("m", Y, A, SITES)
    assert (row["tp"], row["fn"], row["fp"], row["tn"]) == (5, 1, 1, 5)
    assert row["precision"] == pytest.approx(5 / 6) and row["recall"] == pytest.approx(5 / 6)
    for key in ("f1_ci_doc_low", "f1_ci_site_high", "macro_f1_ci_site_low", "balanced_accuracy_ci_doc_high"):
        assert key in row


def test_per_website_table_has_n_docs_and_accuracy():
    t = per_website_table(SITES, Y, {"a": A})
    assert t["documents"].sum() == len(Y)
    # site "f" holds docs 10 (label 0, A says 1 -> wrong) and 11 (label 0, A says 0 -> right)
    assert t.loc[t["website"] == "f", "accuracy_a"].iloc[0] == pytest.approx(0.5)


def test_length_bucket_is_nan_without_conspiracy_documents():
    t = length_bucket_f1(np.array([10, 10, 200, 200, 400, 400]), np.array([1, 0, 0, 0, 1, 0], dtype=bool),
                         {"a": np.array([1, 0, 0, 0, 1, 1], dtype=bool)})
    assert t["conspiracy"].tolist() == [1, 0, 1] and np.isnan(t["f1_a"].iloc[1])


def test_stage_output_matches_prediction_files():
    """Real files: the stored McNemar counts must equal a re-count from the saved predictions."""
    import pandas as pd
    path = results_dir() / "metrics.json"
    if not path.exists() or not json.loads(path.read_text())["comparison"]:
        pytest.skip("comparison stage has not run")
    comp = json.loads(path.read_text())["comparison"]
    lr = pd.read_csv(results_dir() / "tables" / "predictions_lr.csv")
    llm = pd.read_csv(results_dir() / "tables" / "predictions_llm.csv")
    llm = llm[llm["pass"] == 1].set_index("doc_id")
    best = lr[lr["model_id"] == comp["lr_best"]].set_index("doc_id").loc[llm.index]
    y = llm["label"].to_numpy(dtype=bool)
    a_ok, b_ok = best["pred"].to_numpy(dtype=bool) == y, llm["pred"].to_numpy(dtype=bool) == y
    m = comp["mcnemar"]["lr_best_vs_llm"]
    assert (m["a_only_correct"], m["b_only_correct"]) == (int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum()))
    n = m["a_only_correct"] + m["b_only_correct"]
    assert m["mcnemar_p"] == pytest.approx(hand_mcnemar_p(min(m["a_only_correct"], m["b_only_correct"]), n), abs=1e-4)
