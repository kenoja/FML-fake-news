"""Why: the test split is the one honest estimate we have. If selection code could read it, or one model
could be scored on test under several configurations, the final numbers would be tuned to the test set."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest

import fml.models as models
import fml.pipeline as pipeline
from fml.evaluate import (bootstrap_ci, classification_metrics, majority_baseline, mcnemar_exact,
                          stratified_random_baseline)


def synthetic() -> tuple[pd.DataFrame, pd.DataFrame]:
    """40 websites x 6 documents; conspiracy sites use 'cons' words. Returns (corpus, split file)."""
    rng = np.random.default_rng(1)
    rows = []
    for site in range(40):
        label = site % 3 == 0
        for k in range(6):
            words = list(rng.choice(["news", "report", "city", "state", "said"], 8)) + (["hidden", "truth"] if label and k % 2 == 0 else ["agency"])
            rows.append({"doc_id": f"d{site}-{k}", "website": f"site{site}", "txt": " ".join(words), "label": label})
    df = pd.DataFrame(rows)
    parts = np.array(["train"] * 24 + ["validation"] * 8 + ["test"] * 8)
    split = pd.DataFrame({"doc_id": df["doc_id"], "split": df["website"].str[4:].astype(int).map(lambda i: parts[i])})
    return df, split


@pytest.fixture
def frames(monkeypatch):
    df, split = synthetic()
    monkeypatch.setattr(models, "load_split", lambda *a, **k: split)
    monkeypatch.setattr(pipeline, "load_split", lambda *a, **k: split)
    return df, split


def test_split_frames_refuses_test(frames):
    df, _ = frames
    with pytest.raises(PermissionError):
        models.split_frames(df, ("train", "test"))


def test_selection_never_reads_test_rows(frames):
    """Poison the test rows; model selection must give exactly the same grid and vocabulary."""
    df, split = frames
    poisoned = df.copy()
    is_test = poisoned["doc_id"].isin(split.loc[split["split"] == "test", "doc_id"])
    poisoned.loc[is_test, "txt"] = "poisonword " * 8
    poisoned.loc[is_test, "label"] = ~poisoned.loc[is_test, "label"]

    def run(frame):
        f = models.split_frames(frame, ("train", "validation"))
        assert not set(f["train"]["doc_id"]) & set(f["validation"]["doc_id"])
        pipe, grid = models.tune_on_validation(f["train"], f["validation"], {"min_df": 1}, c_grid=(0.1, 1.0, 10.0))
        return pipe, grid

    (pipe_a, grid_a), (pipe_b, grid_b) = run(df), run(poisoned)
    pd.testing.assert_frame_equal(grid_a, grid_b)
    vec = pipe_b.named_steps["features"].named_transformers_["tfidf"]
    assert "poisonword" not in vec.vocabulary_
    assert len(grid_a) == 6 and grid_a["selected"].sum() == 1


def builders(C: float = 1.0):
    cfg = {"model_id": "toy", "C": C, "features": {"min_df": 1}}

    def build():
        def fit_predict(train, test):
            return models.build_pipeline(cfg["features"], C=C).fit(train, train["label"]).predict(test)
        return pipeline.ModelSpec(cfg, fit_predict)
    return {"toy": build}


def test_evaluate_final_logs_and_is_idempotent(frames, tmp_path):
    df, split = frames
    log = tmp_path / "log.jsonl"
    out = pipeline.evaluate_final("toy", builders(), df, log)
    again = pipeline.evaluate_final("toy", builders(), df, log)
    entries = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(entries) == 1 and set(entries[0]) == {"model_id", "config_hash", "at", "n_docs"}
    assert entries[0]["n_docs"] == out["n_docs"] == int((split["split"] == "test").sum())
    assert out["confusion"] == again["confusion"]


def test_evaluate_final_rejects_changed_config(frames, tmp_path):
    df, _ = frames
    log = tmp_path / "log.jsonl"
    pipeline.evaluate_final("toy", builders(1.0), df, log)
    with pytest.raises(PermissionError):
        pipeline.evaluate_final("toy", builders(10.0), df, log)
    assert len(log.read_text().splitlines()) == 1


def test_evaluate_final_unknown_model(frames, tmp_path):
    df, _ = frames
    with pytest.raises(KeyError):
        pipeline.evaluate_final("nope", builders(), df, tmp_path / "log.jsonl")


def test_metrics_on_hand_table():
    y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    p = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]          # tp 3, fn 1, fp 1, tn 5
    m = classification_metrics(y, p)
    assert m["precision"] == 0.75 and m["recall"] == 0.75 and m["f1"] == 0.75
    assert m["balanced_accuracy"] == pytest.approx((0.75 + 5 / 6) / 2)
    assert m["macro_f1"] == pytest.approx((0.75 + 10 / 12) / 2)


def test_baselines():
    y_train = [True] * 3 + [False] * 7
    y = np.array([True] * 3 + [False] * 7)
    maj = majority_baseline(y_train, y)
    assert maj["recall"] == 0 and maj["balanced_accuracy"] == 0.5
    rnd = stratified_random_baseline(y_train, y, seed=1)
    assert 0.2 < rnd["recall"] < 0.4 and rnd["balanced_accuracy"] == pytest.approx(0.5, abs=0.05)


def test_bootstrap_ci_contains_point_estimate_and_is_reproducible():
    rng = np.random.default_rng(3)
    y = rng.random(300) < 0.3
    p = np.where(rng.random(300) < 0.85, y, ~y)
    sites = np.repeat(np.arange(30), 10)
    for groups in (None, sites):
        ci = bootstrap_ci(y, p, groups=groups, n=300)
        for name, v in ci.items():
            assert v["ci_low"] <= v["estimate"] <= v["ci_high"], name
        assert ci == bootstrap_ci(y, p, groups=groups, n=300)
    wide, narrow = bootstrap_ci(y, p, groups=sites, n=300), bootstrap_ci(y, p, n=300)
    assert wide["f1"]["ci_high"] - wide["f1"]["ci_low"] > 0.8 * (narrow["f1"]["ci_high"] - narrow["f1"]["ci_low"])


def test_mcnemar_hand_computed():
    # A alone right on 2 documents, B alone right on 8, both right on 5, both wrong on 5.
    y = np.ones(20, dtype=bool)
    a = np.array([1] * 2 + [0] * 8 + [1] * 5 + [0] * 5, dtype=bool)
    b = np.array([0] * 2 + [1] * 8 + [1] * 5 + [0] * 5, dtype=bool)
    r = mcnemar_exact(y, a, b)
    assert (r["a_only_correct"], r["b_only_correct"], r["both_correct"], r["both_wrong"]) == (2, 8, 5, 5)
    assert r["p_value"] == pytest.approx(112 / 1024) == pytest.approx(binomtest(2, 10, 0.5).pvalue)
    assert mcnemar_exact(y, a, a)["p_value"] == 1.0
