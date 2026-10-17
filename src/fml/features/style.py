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
    """Share of distinct words, averaged over every `window`-word span (so length does not matter)."""
    toks = [t.lower() for t in tokens]
    if not toks:
        return 0.0
    if len(toks) <= window:
        return len(set(toks)) / len(toks)
    # a word counts once per window if its previous occurrence lies before the window starts
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
    """Text -> 6 style columns. Nothing is learned."""

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
