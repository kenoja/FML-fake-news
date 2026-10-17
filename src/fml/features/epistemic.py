"""Epistemic markers: how certain a text sounds, and whether it claims a hidden truth.

Where the categories come from (published authors only):
- Hedges (doubt words) and boosters (certainty words): Islam et al. (2020).
- Hidden-truth and debunking words: Marini & Jezek (2024). Evidence on certainty is mixed (Fong et al. 2021).
Known weakness: matching is by word list, so "no evidence of a cover-up" still counts "cover-up".
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ._text import as_texts, normalise_quotes, per_n_words, words

LEXICON: dict[str, tuple[str, ...]] = {
    # hedges: the writer signals doubt or distance
    "hedges": ("perhaps", "allegedly", "possibly", "reportedly", "apparently", "supposedly", "maybe",
               "might", "could be", "seems to", "appears to", "it is claimed", "so-called"),
    # boosters: the writer signals certainty
    "boosters": ("undeniable", "undeniably", "clearly", "obviously", "undoubtedly", "certainly",
                 "without a doubt", "beyond doubt", "of course", "proof", "proves", "indisputable"),
    # hidden truth: claims of concealment
    "hidden_truth": ("cover-up", "cover up", "coverup", "they don't want you to know", "wake up",
                     "inside job", "mainstream media", "false flag", "the truth about", "hidden truth",
                     "controlled demolition"),
    # debunking: the text argues against a claim
    "debunking": ("debunked", "debunk", "no evidence", "conspiracy theory", "conspiracy theories",
                  "conspiracy theorists", "myth", "myths", "discredited", "baseless", "hoax"),
}


def compile_terms(terms: tuple[str, ...]) -> re.Pattern:
    """One pattern per category; the edges stop "myth" from matching inside "mythology"."""
    parts = sorted((re.escape(t) for t in terms), key=len, reverse=True)
    return re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(parts) + r")(?![A-Za-z0-9])")


PATTERNS = {category: compile_terms(terms) for category, terms in LEXICON.items()}
COLUMNS = [f"epistemic_{category}" for category in LEXICON]


def count_markers(text: str) -> dict[str, int]:
    """Number of list words found per category (capital letters ignored)."""
    lowered = normalise_quotes(text).lower()
    return {category: len(pattern.findall(lowered)) for category, pattern in PATTERNS.items()}


def lexicon_coverage(train: pd.DataFrame) -> pd.DataFrame:
    """Per category and class: list size and share of training articles with at least one hit."""
    counts = [count_markers(text) for text in as_texts(train["txt"])]
    labels = train["label"].to_numpy()
    rows = []
    for category, terms in LEXICON.items():
        hits = np.array([c[category] for c in counts])
        for name, in_class in (("conspiracy", labels == 1), ("mainstream", labels == 0)):
            share = float((hits[in_class] > 0).mean()) if in_class.any() else 0.0
            rows.append({"category": category, "list_size": len(terms), "class": name,
                         "train_documents": int(in_class.sum()), "share_with_hit": share})
    return pd.DataFrame(rows)


class EpistemicMarkers(BaseEstimator, TransformerMixin):
    """Text -> 4 columns of list hits per 1,000 words. Nothing is learned."""

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        rows = []
        for text in as_texts(X):
            n_words = len(words(text))
            counts = count_markers(text)
            rows.append([per_n_words(counts[category], n_words) for category in LEXICON])
        return np.asarray(rows, dtype=float).reshape(-1, len(LEXICON))

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(COLUMNS)
