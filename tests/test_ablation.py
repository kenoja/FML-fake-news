"""Why: a feature comparison is only worth reading if it can tell a useful feature from a useless one, if it
gives the same numbers every time, if it shows validation numbers only, and if scoring the two LR models on
test leaves exactly one log line each."""
from __future__ import annotations

import numpy as np
import pandas as pd

import fml.models as models
import fml.pipeline as pipeline
from fml.ablation import THRESHOLD, choose_best, run_ablation, verdict, verdict_sentence

C_GRID = (0.1, 1.0, 10.0)


def planted(seed: int = 0, informative: bool = True) -> pd.DataFrame:
    """240 websites x 5 documents. Text is pure noise; column `signal` separates the classes (if informative)."""
    rng = np.random.default_rng(seed)
    rows = []
    for site in range(240):
        label = site % 3 == 0
        for k in range(5):
            shift = 2.0 if (label and informative) else 0.0
            rows.append({"doc_id": f"d{site}-{k}", "website": f"site{site}", "label": label,
                         "txt": " ".join(rng.choice(["news", "report", "city", "state", "said", "year"], 12)),
                         # an uninformative feature is constant, so the result does not depend on luck
                         "signal": rng.normal(shift, 1.0) if informative else 0.0})
    return pd.DataFrame(rows)


def variants() -> dict:
    base = {"min_df": 1}
    return {"words": {"label": "words", "features": base},
            "plus_signal": {"label": "+signal", "features": {**base, "numeric_columns": ["signal"]}}}


def run(df: pd.DataFrame) -> pd.DataFrame:
    site = df["website"].str[4:].astype(int)
    train, valid = df[site < 160].reset_index(drop=True), df[site >= 160].reset_index(drop=True)
    return run_ablation(train, valid, variants(), reference="words", c_grid=C_GRID)


def test_planted_informative_feature_helps():
    row = run(planted()).set_index("model_id").loc["plus_signal"]
    assert row["delta_f1"] > 0.15
    assert row["verdict"] == "above threshold"


def test_uninformative_feature_is_not_claimed_to_help():
    row = run(planted(informative=False)).set_index("model_id").loc["plus_signal"]
    assert row["verdict"] != "above threshold"


def test_table_has_validation_f1_only():
    """Test-set numbers must never appear in the comparison table."""
    table = run(planted())
    assert list(table.columns) == ["model_id", "label", "C", "class_weight", "f1", "delta_f1", "verdict"]
    assert table.loc[table["model_id"] == "words", "verdict"].iloc[0] == "reference"


def test_ablation_is_reproducible():
    pd.testing.assert_frame_equal(run(planted()), run(planted()))


def test_verdict_needs_a_clear_margin():
    assert verdict(THRESHOLD) == "above threshold"
    assert verdict(-THRESHOLD) == "hurts"
    assert verdict(THRESHOLD / 2) == "no clear gain"


def test_choose_best_keeps_plain_tfidf_unless_a_group_clearly_helps():
    assert choose_best(run(planted(informative=False)), "words")["chosen"] == "words"
    assert choose_best(run(planted()), "words")["chosen"] == "plus_signal"


def test_verdict_sentence_says_no_when_nothing_helps():
    assert "do not help" in verdict_sentence(run(planted(informative=False)), "words")
    assert "+signal" in verdict_sentence(run(planted()), "words")


def test_test_log_has_one_entry_per_model(tmp_path, monkeypatch):
    df = planted()
    site = df["website"].str[4:].astype(int)
    split = pd.DataFrame({"doc_id": df["doc_id"],
                          "split": np.where(site < 160, "train", np.where(site < 200, "validation", "test"))})
    monkeypatch.setattr(pipeline, "load_split", lambda *a, **k: split)

    def spec(model_id):
        def fit(train, test):
            blocks = models.fit_features(train, {"min_df": 1})
            model = models.fit_logistic(models.feature_matrix(blocks, train), train["label"], 1.0, None)
            X_test = models.feature_matrix(blocks, test)
            return model.predict(X_test), model.predict_proba(X_test)[:, 1]
        return pipeline.ModelSpec({"model_id": model_id}, lambda tr, te: fit(tr, te)[0], fit)

    builders = {m: (lambda m=m: spec(m)) for m in ("lr_words", "lr_extended")}
    log, frozen = tmp_path / "log.jsonl", tmp_path / "frozen.json"
    for m in ("lr_words", "lr_extended", "lr_words"):
        pipeline.freeze_model(m, builders, frozen, log)
        out = pipeline.evaluate_final(m, builders=builders, frame=df, log_path=log, frozen_path=frozen)
        assert list(out["predictions"].columns) == ["model_id", "doc_id", "label", "pred", "proba"]
        assert set(out["predictions"]["doc_id"]) <= set(split.loc[split["split"] == "test", "doc_id"])
    assert [e["model_id"] for e in pipeline.read_access_log(log)] == ["lr_words", "lr_extended"]
