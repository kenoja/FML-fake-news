"""Why: anything learned from validation/test text (vocabulary, IDF weights) before training leaks
information about the data we claim not to have seen. These tests must FAIL for a leaky pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from fml.models import build_pipeline

TRAIN_WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
HELD_OUT_WORDS = ["zulu", "yankee", "xray"]          # appear only outside the training rows


def make_frames(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    def docs(n: int, words: list[str]) -> list[str]:
        return [" ".join(rng.choice(words, 12)) for _ in range(n)]

    train = pd.DataFrame({"txt": docs(30, TRAIN_WORDS), "label": [i % 2 == 0 for i in range(30)]})
    valid = pd.DataFrame({"txt": docs(10, TRAIN_WORDS + HELD_OUT_WORDS), "label": [i % 2 == 0 for i in range(10)]})
    valid.loc[0, "txt"] = "zulu yankee xray alpha"
    return train, valid


def train_tokens(train: pd.DataFrame) -> set[str]:
    return {t for txt in train["txt"] for t in txt.split()}


def leaky_vectorizer(train: pd.DataFrame, valid: pd.DataFrame) -> TfidfVectorizer:
    """The mistake we guard against: vocabulary and IDF learned on train AND validation text."""
    return TfidfVectorizer(min_df=1).fit(pd.concat([train["txt"], valid["txt"]]))


def assert_no_vocabulary_leak(vectorizer: TfidfVectorizer, train: pd.DataFrame) -> None:
    assert set(vectorizer.vocabulary_) <= train_tokens(train), "vocabulary contains tokens not in train"


def test_pipeline_vocabulary_comes_from_train_only():
    train, valid = make_frames()
    pipe = build_pipeline({"min_df": 1}).fit(train, train["label"])
    pipe.predict(valid)                                   # scoring validation must not change anything
    vec = pipe.named_steps["features"].named_transformers_["tfidf"]
    assert_no_vocabulary_leak(vec, train)
    assert not set(vec.vocabulary_) & set(HELD_OUT_WORDS)


def test_leaky_fit_is_detected():
    """If this test fails, the leakage check above has become blind."""
    train, valid = make_frames()
    with pytest.raises(AssertionError):
        assert_no_vocabulary_leak(leaky_vectorizer(train, valid), train)


def test_idf_weights_do_not_depend_on_validation_text():
    train, valid = make_frames()
    a = build_pipeline({"min_df": 1}).fit(train, train["label"]).named_steps["features"].named_transformers_["tfidf"]
    changed = valid.assign(txt="completely different words")
    _ = build_pipeline({"min_df": 1}).fit(train, train["label"]).predict(changed)
    b = build_pipeline({"min_df": 1}).fit(train, train["label"]).named_steps["features"].named_transformers_["tfidf"]
    assert np.allclose(a.idf_, b.idf_)
