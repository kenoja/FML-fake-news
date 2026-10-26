"""Why these tests exist: the feature groups are only fair comparisons if they are length-normalised,
safe on odd input, honest about their known weaknesses, and described from training rows only."""
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from fml.features import EpistemicMarkers, StyleFeatures, build_ngram_vectorizer, scaled
from fml.features.epistemic import COLUMNS as EPI_COLUMNS, LEXICON, count_markers
from fml.features.style import COLUMNS as STYLE_COLUMNS, allcaps_share, mattr
from fml.features._text import per_n, words
from fml.features.summary import feature_summary
from fml.paths import project_config

TEXT = ("WAKE up! Was it an inside job? Officials said the report was debunked. "
        "Perhaps, but clearly the cover-up goes on. \"No comment,\" he said.")


def test_doubling_a_document_leaves_rates_unchanged():
    """Per-1,000-word values must not grow with length, or long articles would look 'more conspiratorial'."""
    doubled = TEXT + " " + TEXT
    for t in (EpistemicMarkers(), StyleFeatures()):
        one, two = t.transform([TEXT])[0], t.transform([doubled])[0]
        # sentence length and type-token ratio are not counts; only rates and shares are length-free by design
        keep = [i for i, n in enumerate(t.get_feature_names_out()) if n != "style_mattr"]
        assert np.allclose(one[keep], two[keep], rtol=0, atol=1e-9)


def test_lexicon_coverage_counts_list_size_and_train_shares():
    """Coverage reports the exact list size and the per-class share of documents with a hit."""
    from fml.features.epistemic import lexicon_coverage
    df = pd.DataFrame({"txt": ["The inside job and a cover-up.", "No evidence at all, never a cover-up.", "Plain weather report.",
                               "Plain sports report."], "label": [1, 0, 0, 1]})
    coverage = lexicon_coverage(df).set_index(["category", "class"])
    for category, terms in LEXICON.items():
        assert coverage.loc[(category, "conspiracy"), "list_size"] == len(terms)
    assert coverage.loc[("hidden_truth", "conspiracy"), "share_with_hit"] == 0.5
    assert coverage.loc[("hidden_truth", "mainstream"), "share_with_hit"] == 0.5


def test_rates_read_per_n_words_from_config():
    """The normalisation length comes from config (features.per_n_words), not from the code."""
    n = len(words(TEXT))
    hidden = EpistemicMarkers().transform([TEXT])[0][EPI_COLUMNS.index("epistemic_hidden_truth")]
    assert hidden == pytest.approx(count_markers(TEXT)["hidden_truth"] / n * per_n())
    assert per_n() == project_config()["features"]["per_n_words"]


def test_allcaps_ratio_on_hand_made_string():
    """'A' and 'I' are single letters and must not count; WAKE and FBI do: 2 of 5 eligible words."""
    toks = words("WAKE up A I see the FBI")
    assert [t for t in toks if t.isupper()] == ["WAKE", "A", "I", "FBI"]
    assert allcaps_share(toks) == pytest.approx(2 / 5)


def test_punctuation_and_quote_rates():
    row = StyleFeatures().transform(["Stop! Why? Now! \"Hi\" there."])[0]
    n = 5
    assert row[STYLE_COLUMNS.index("style_exclaim_per_n")] == pytest.approx(2 / n * per_n())
    assert row[STYLE_COLUMNS.index("style_question_per_n")] == pytest.approx(1 / n * per_n())
    assert row[STYLE_COLUMNS.index("style_quotes_per_n")] == pytest.approx(2 / n * per_n())


def test_mattr_window_and_repetition():
    """Repetition lowers MATTR; long texts are averaged over windows so length alone does not change it."""
    assert mattr(["a", "b", "c", "d"], window=100) == 1.0
    assert mattr(["a", "a", "a", "a"], window=2) == pytest.approx(0.5)
    assert mattr(["a", "b", "a", "b", "a", "b"], window=2) == pytest.approx(1.0)
    assert mattr(["a", "b", "a", "b"], window=3) == pytest.approx(2 / 3)


def test_lexicon_counts_on_hand_made_sentence():
    sentence = "Perhaps it was an inside job, clearly a cover-up, and the mainstream media call it a debunked myth."
    c = count_markers(sentence)
    assert c == {"hedges": 1, "boosters": 1, "hidden_truth": 3, "debunking": 2}


def test_lexicon_matches_whole_words_only():
    """'mythology' must not count as 'myth'; curly apostrophes must not hide a phrase."""
    assert count_markers("Greek mythology is old.")["debunking"] == 0
    assert count_markers("They don’t want you to know.")["hidden_truth"] == 1


def test_lexicon_documented_and_four_categories():
    assert set(LEXICON) == {"hedges", "boosters", "hidden_truth", "debunking"}
    assert len(EPI_COLUMNS) == 4


def test_negation_caveat_known_weakness():
    """KNOWN WEAKNESS (Chen 2026): word lists ignore negation, so a text that DENIES a cover-up still
    scores as 'hidden truth' vocabulary. This test documents it; if it ever fails, update section 06."""
    assert count_markers("There is no evidence of cover-up.")["hidden_truth"] == 1
    assert count_markers("There is no evidence of cover-up.")["debunking"] == 1


@pytest.mark.parametrize("empty", ["", "   \n", None, float("nan")])
def test_empty_text_is_safe(empty):
    """An empty or missing text must give zeros, not a division by zero or NaN."""
    for t in (EpistemicMarkers(), StyleFeatures()):
        out = t.transform([empty])
        assert out.shape == (1, len(t.get_feature_names_out()))
        assert np.isfinite(out).all() and (out == 0).all()


def test_scaler_is_fitted_on_train_only():
    """Standardising with all data would leak the test rows' mean and spread into training."""
    train = ["calm text here today", "more calm words go here"]
    test = ["WAKE UP!!! WAKE UP!!!"]
    pipe = scaled("style").fit(train)
    assert pipe.named_steps["scale"].n_samples_seen_ == len(train)
    raw = StyleFeatures().transform(train)
    assert np.allclose(pipe.named_steps["scale"].mean_, raw.mean(axis=0))
    assert pipe.transform(test).shape == (1, len(STYLE_COLUMNS))


def test_ngram_vectorizer_reads_config_and_learns_from_fit_only():
    cfg = project_config()["features"]
    vec = build_ngram_vectorizer()
    assert vec.ngram_range == tuple(cfg["ngram_range"]) and vec.min_df == cfg["ngram_min_df"]
    assert vec.max_features == cfg["ngram_max_features"]
    train = ["the inside job claim"] * 3 + ["a calm report"] * 3
    vec.fit(train)
    assert "inside job" in vec.vocabulary_
    assert "zebra" not in vec.vocabulary_  # unseen words are not learned by transform
    vec.transform(["zebra zebra"])
    assert "zebra" not in vec.vocabulary_


def test_transformers_work_in_a_sklearn_pipeline():
    from sklearn.linear_model import LogisticRegression
    X = ["WAKE UP!!! inside job cover-up"] * 4 + ["The report found no evidence. It was debunked."] * 4
    y = [1] * 4 + [0] * 4
    for g in ("epistemic", "style"):
        pipe = Pipeline([("f", scaled(g)), ("clf", LogisticRegression())]).fit(X, y)
        assert list(pipe.predict(X)) == y


def test_feature_summary_uses_only_the_rows_it_is_given():
    """The summary describes whatever frame it receives, so the stage must pass train rows only."""
    train = pd.DataFrame({"txt": ["Stop! Now!", "Calm words. Plain text.", "WOW! Look at this.", "Quiet day. Nice."],
                          "label": [True, False, True, False]})
    s = feature_summary(train)
    assert (s["n_conspiracy"] == 2).all() and (s["n_mainstream"] == 2).all()
    row = s.set_index("feature").loc["style_exclaim_per_n"]
    assert row["mean_conspiracy"] > row["mean_mainstream"] == 0


@pytest.mark.needs_data
def test_feature_summary_file_is_train_rows_only():
    from fml.models import split_frames
    from fml.pipeline import analysis_frame
    from fml.results import results_dir
    path = results_dir() / "tables" / "feature_summary.csv"
    if not path.is_file():
        pytest.skip("run `fml run-all` or features_stage() first")
    table = pd.read_csv(path)
    train = split_frames(analysis_frame(), ("train",))["train"]
    assert (table["n_train_docs"] == len(train)).all()
    assert table["n_conspiracy"].iloc[0] == int(train["label"].sum())
    assert table["n_mainstream"].iloc[0] == int((~train["label"]).sum())
