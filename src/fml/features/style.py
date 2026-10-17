"""Style features: capitals, punctuation, quotes, sentence length and repetition.

Why: Horne & Adali (2017) found more all-caps words, fewer quotes and more repetition in fake-news
bodies; Perez-Rosas et al. (2018) found such features weak across domains.
"""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ..paths import project_config
from ._text import as_texts, normalise_quotes, per_n_words, words

COLUMNS = ["style_allcaps_share", "style_exclaim_per_n", "style_question_per_n", "style_quotes_per_n",
           "style_mean_sentence_len", "style_mattr"]
SENTENCE_END = re.compile(r"[.!?]+(?:\s+|$)|\n+")
QUOTE_CHARS = ('"', "“", "”")


def allcaps_share(tokens: list[str]) -> float:
    """Share of words (2+ letters) written in capitals, like "FBI" or "WAKE"."""
    n_cand = n_upper = 0
    for t, k in Counter(tokens).items():
        kind = _token_kind(t)
        if kind:
            n_cand += k
            n_upper += k * (kind == 2)
    return n_upper / n_cand if n_cand else 0.0


@lru_cache(maxsize=None)
def _token_kind(token: str) -> int:
    """0 = fewer than 2 letters (skipped), 1 = normal word, 2 = word in capitals (cached: it is slow)."""
    if sum(c.isalpha() for c in token) < 2:
        return 0
    return 2 if token.isupper() else 1


def mean_sentence_length(text: str, n_words: int) -> float:
    """Words per sentence (a sentence ends at . ! ? or a line break)."""
    sentences = [s for s in SENTENCE_END.split(normalise_quotes(text)) if words(s)]
    return n_words / len(sentences) if sentences else 0.0


def previous_positions(ids: np.ndarray) -> np.ndarray:
    """For each word, the position of its previous occurrence (-1 if it is new)."""
    order = np.argsort(ids, kind="stable")
    prev = np.full(len(ids), -1)
    same = ids[order][1:] == ids[order][:-1]
    prev[order[1:][same]] = order[:-1][same]
    return prev


def mattr(tokens: list[str], window: int) -> float:
    """Share of distinct words, averaged over every `window`-word span (so length does not matter)."""
    words_lower = [t.lower() for t in tokens]
    n = len(words_lower)
    if n == 0:
        return 0.0
    if n <= window:
        return len(set(words_lower)) / n
    ids = np.unique(np.asarray(words_lower), return_inverse=True)[1].ravel()
    prev = previous_positions(ids)
    # a word counts once per window if its previous occurrence lies before the window starts
    pos = np.arange(n)
    first = np.maximum(np.maximum(pos - window + 1, prev + 1), 0)
    last = np.minimum(pos, n - window)
    distinct_sum = int(np.clip(last - first + 1, 0, None).sum())
    return distinct_sum / ((n - window + 1) * window)


def count_quotes(text: str) -> int:
    """Number of quotation marks in the text."""
    text = normalise_quotes(text)
    return sum(text.count(q) for q in QUOTE_CHARS)


def style_row(text: str, window: int) -> list[float]:
    """The six style numbers for one article."""
    tokens = words(text)
    n_words = len(tokens)
    return [allcaps_share(tokens),
            per_n_words(text.count("!"), n_words),
            per_n_words(text.count("?"), n_words),
            per_n_words(count_quotes(text), n_words),
            mean_sentence_length(text, n_words),
            mattr(tokens, window)]


def style_matrix(texts, window: int) -> np.ndarray:
    """One row of six style numbers per article."""
    rows = [style_row(text, window) for text in as_texts(texts)]
    return np.asarray(rows, dtype=float).reshape(-1, len(COLUMNS))


class StyleFeatures(BaseEstimator, TransformerMixin):
    """Thin scikit-learn wrapper around style_matrix. Nothing is learned."""

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        return style_matrix(X, int(project_config()["features"]["mattr_window"]))

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(COLUMNS)
