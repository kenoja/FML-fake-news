"""Analyses 1 and 2 choose C on validation only and learn their word list from training articles only."""
import numpy as np

from fml.models import choose_c, tfidf_vectorizer, word_counts_vectorizer


def test_choose_c_uses_validation_scores_and_the_configured_grid():
    # A model picked by training score alone would overfit; the grid must be scored on validation rows.
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(60, 5))
    y_train = X_train[:, 0] > 0
    X_valid = rng.normal(size=(40, 5))
    y_valid = X_valid[:, 0] > 0
    model, grid = choose_c(X_train, y_train, X_valid, y_valid)
    assert list(grid["C"]) == [0.1, 1, 10]
    assert grid["chosen"].sum() == 1
    assert model.C == grid.loc[grid["chosen"], "C"].iloc[0]
    assert grid.loc[grid["chosen"], "f1"].iloc[0] == grid["f1"].max()


def test_vocabulary_comes_from_training_texts_only():
    # Words that only occur in validation texts must not enter the vocabulary (leakage).
    train = ["alpha beta", "alpha beta gamma", "beta gamma alpha"]
    for make in (word_counts_vectorizer, tfidf_vectorizer):
        vectorizer = make().fit(train)
        assert "delta" not in vectorizer.vocabulary_
        assert vectorizer.transform(["delta delta alpha"]).shape[1] == len(vectorizer.vocabulary_)
