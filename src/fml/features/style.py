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


def mattr(tokens: list[str], window: int) -> float:
    """Share of different words, averaged over every window of `window` words (so length does not matter)."""
    words_lower = [t.lower() for t in tokens]
    n = len(words_lower)
    if n == 0:
        return 0.0
    if n <= window:                              # a short text is one single window
        return len(set(words_lower)) / n
    counts = Counter(words_lower[:window])       # how often each word occurs in the current window
    total = len(counts)                          # different words in the first window
    for start in range(1, n - window + 1):       # slide the window one word to the right
        leaving = words_lower[start - 1]
        counts[leaving] -= 1
        if counts[leaving] == 0:
            del counts[leaving]                  # that word is no longer in the window
        counts[words_lower[start + window - 1]] += 1
        total += len(counts)
    windows = n - window + 1
    return total / (windows * window)


def count_quotes(text: str) -> int:
    """Number of quotation marks in the text."""
    text = normalise_quotes(text)
    return sum(text.count(q) for q in QUOTE_CHARS)


def style_row(text: str, window: int) -> list[float]:
    """The six style numbers for one article."""
    tokens = words(text)
    n_words = len(tokens)
    exclamations = per_n_words(text.count("!"), n_words)
    questions = per_n_words(text.count("?"), n_words)
    quotes = per_n_words(count_quotes(text), n_words)
    return [allcaps_share(tokens), exclamations, questions, quotes,
            mean_sentence_length(text, n_words), mattr(tokens, window)]


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
