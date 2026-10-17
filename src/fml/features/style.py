"""Group 3: style features computed from the cleaned text only.

Why: Horne & Adali (2017) found more all-caps words, fewer quotes and more repetition in fake-news
bodies; Perez-Rosas et al. (2018) found punctuation and readability features strong within a domain
but weak across domains. Hence the website-held-out split matters for judging them.
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
    """Share of words with at least 2 letters that are written entirely in capitals ("FBI", "WAKE")."""
    n_cand = n_upper = 0
    for t, k in Counter(tokens).items():
        kind = _token_kind(t)
        if kind:
            n_cand += k
            n_upper += k * (kind == 2)
    return n_upper / n_cand if n_cand else 0.0


@lru_cache(maxsize=None)
def _token_kind(token: str) -> int:
    """0 = fewer than 2 letters (not counted), 1 = counted word, 2 = counted word written in capitals.

    Memoised: the same words recur across thousands of articles, and the per-character check is slow.
    """
    if sum(c.isalpha() for c in token) < 2:
        return 0
    return 2 if token.isupper() else 1


def mean_sentence_length(text: str, n_words: int) -> float:
    """Words per sentence; sentences end at . ! ? or a line break. 0.0 for an empty text."""
    sentences = [s for s in SENTENCE_END.split(normalise_quotes(text)) if words(s)]
    return n_words / len(sentences) if sentences else 0.0


def mattr(tokens: list[str], window: int) -> float:
    """Moving-average type-token ratio: mean share of distinct words over every `window`-word span.

    Plain type-token ratio falls as texts get longer, so it would mostly measure length. Averaging over
    fixed-size windows removes that effect. Texts not longer than the window use the whole text.
    """
    toks = [t.lower() for t in tokens]
    if not toks:
        return 0.0
    if len(toks) <= window:
        return len(set(toks)) / len(toks)
    # Vectorised: a token at position i is the first of its word in window [s, s+window) exactly when its
    # previous occurrence is before s, so summing those windows over all i gives the sum of distinct counts.
    n = len(toks)
    ids = np.unique(np.asarray(toks), return_inverse=True)[1].ravel()
    order = np.argsort(ids, kind="stable")
    prev = np.full(n, -1)
    same = ids[order][1:] == ids[order][:-1]
    prev[order[1:][same]] = order[:-1][same]
    pos = np.arange(n)
    first = np.maximum(np.maximum(pos - window + 1, prev + 1), 0)
    last = np.minimum(pos, n - window)
    distinct_sum = int(np.clip(last - first + 1, 0, None).sum())
    return distinct_sum / ((n - window + 1) * window)


class StyleFeatures(BaseEstimator, TransformerMixin):
    """Text -> 6 style columns. Stateless; rates use `features.per_n_words`, the window `features.mattr_window`."""

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        window = int(project_config()["features"]["mattr_window"])
        rows = []
        for text in as_texts(X):
            toks = words(text)
            n = len(toks)
            quotes = sum(normalise_quotes(text).count(q) for q in QUOTE_CHARS)
            rows.append([allcaps_share(toks), per_n_words(text.count("!"), n), per_n_words(text.count("?"), n),
                         per_n_words(quotes, n), mean_sentence_length(text, n), mattr(toks, window)])
        return np.asarray(rows, dtype=float).reshape(-1, len(COLUMNS))

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(COLUMNS)
