"""Why: an ablation is only worth reading if it can tell a useful feature group from a useless one, if it
gives the same numbers every time, and if scoring the two LR models on test leaves exactly one log line each."""
from __future__ import annotations

import numpy as np
import pandas as pd

import fml.pipeline as pipeline
from fml.ablation import choose_between, run_ablation, verdict

C_GRID = (0.1, 1.0, 10.0)


def planted(seed: int = 0, informative: bool = True) -> pd.DataFrame:
    """60 websites x 5 documents. Text is pure noise; column `signal` separates the classes (if informative)."""
    rng = np.random.default_rng(seed)
    rows = []
    for site in range(60):
        label = site % 3 == 0
        for k in range(5):
            shift = 2.0 if (label and informative) else 0.0
            rows.append({"doc_id": f"d{site}-{k}", "website": f"site{site}", "label": label,
                         "txt": " ".join(rng.choice(["news", "report", "city", "state", "said", "year"], 12)),
                         "signal": rng.normal(shift, 1.0)})
    return pd.DataFrame(rows)


def variants() -> dict:
    base = {"min_df": 1}
    return {"words": {"label": "words", "kind": "reference", "features": base},
            "plus_signal": {"label": "+signal", "kind": "add_one", "features": {**base, "numeric_columns": ["signal"]}}}


def run(df: pd.DataFrame) -> pd.DataFrame:
    site = df["website"].str[4:].astype(int)
    train, valid = df[site < 40].reset_index(drop=True), df[site >= 40].reset_index(drop=True)
    return run_ablation(train, valid, variants(), reference="words", c_grid=C_GRID, n_boot=200)[0]


def test_planted_informative_feature_helps():
    row = run(planted()).set_index("model_id").loc["plus_signal"]
    assert row["delta_f1_vs_m1"] > 0.15
    assert row["delta_f1_vs_m1_verdict"] == "helps"
    assert row["mcnemar_p_m1"] < 0.05


def test_uninformative_feature_is_not_claimed_to_help():
    row = run(planted(informative=False)).set_index("model_id").loc["plus_signal"]
    assert row["delta_f1_vs_m1_verdict"] != "helps"


def test_ablation_is_reproducible():
    pd.testing.assert_frame_equal(run(planted()), run(planted()))


def test_verdict_requires_whole_ci_on_one_side():
    assert verdict(0.01, 0.1) == "helps"
    assert verdict(-0.1, -0.01) == "hurts"
    assert verdict(-0.01, 0.1) == "no clear gain"


def test_choose_between_keeps_m1_inside_noise():
    t = run(planted(informative=False))
    t.loc[t["model_id"] == "plus_signal", "delta_f1_vs_m1_verdict"] = "no clear gain"
    assert choose_between(t, "words", "plus_signal")["chosen"] == "words"
    t.loc[t["model_id"] == "plus_signal", "delta_f1_vs_m1_verdict"] = "helps"
    assert choose_between(t, "words", "plus_signal")["chosen"] == "plus_signal"


def test_test_log_has_one_entry_per_model(tmp_path, monkeypatch):
    df = planted()
    site = df["website"].str[4:].astype(int)
    split = pd.DataFrame({"doc_id": df["doc_id"],
                          "split": np.where(site < 40, "train", np.where(site < 50, "validation", "test"))})
    monkeypatch.setattr(pipeline, "load_split", lambda *a, **k: split)

    def spec(model_id):
        def fit(train, test):
            pipe = pipeline.build_pipeline({"min_df": 1}, C=1.0).fit(train, train["label"])
            return pipe.predict(test), pipe.predict_proba(test)[:, 1]
        return pipeline.ModelSpec({"model_id": model_id}, lambda tr, te: fit(tr, te)[0], fit)

    builders = {m: (lambda m=m: spec(m)) for m in ("lr_words", "lr_extended")}
    log, frozen = tmp_path / "log.jsonl", tmp_path / "frozen.json"
    for m in ("lr_words", "lr_extended", "lr_words"):
        pipeline.freeze_model(m, builders, frozen, log)
        out = pipeline.evaluate_final(m, builders=builders, frame=df, log_path=log, frozen_path=frozen)
        assert list(out["predictions"].columns) == ["model_id", "doc_id", "label", "pred", "proba"]
        assert set(out["predictions"]["doc_id"]) <= set(split.loc[split["split"] == "test", "doc_id"])
    assert [e["model_id"] for e in pipeline.read_access_log(log)] == ["lr_words", "lr_extended"]
