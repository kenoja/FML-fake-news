"""Style features: capitals, punctuation, quotes, sentence length and repetition."""
# Why: Horne & Adali (2017) found more all-caps words, fewer quotes and more repetition in fake-news
# bodies; Perez-Rosas et al. (2018) found such features weak across domains.
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from ._text import as_texts, normalise_quotes, per_n_words, words

COLUMNS = ["style_allcaps_share", "style_exclaim_per_n", "style_question_per_n", "style_quotes_per_n",
           "style_mean_sentence_len", "style_mattr"]
SENTENCE_END = re.compile(r"[.!?]+(?:\s+|$)|\n+")
QUOTE_CHARS = ('"', "“", "”")


def allcaps_share(tokens: list[str]) -> float:
    """Share of words (2+ letters) written in capitals, like "FBI" or "WAKE"."""
    n_candidates = 0
    n_capitals = 0
    for token in tokens:
        letters = sum(c.isalpha() for c in token)
        if letters < 2:                          # skip one-letter words and numbers
            continue
        n_candidates += 1
        if token.isupper():
            n_capitals += 1
    return n_capitals / n_candidates if n_candidates else 0.0


def mean_sentence_length(text: str, n_words: int) -> float:
    """Words per sentence (a sentence ends at . ! ? or a line break)."""
    sentences = [s for s in SENTENCE_END.split(normalise_quotes(text)) if words(s)]
    return n_words / len(sentences) if sentences else 0.0


def mattr(tokens: list[str], window: int) -> float:
    """Share of different words, averaged over every window of `window` words (so length does not matter)."""
    lower = [t.lower() for t in tokens]
    n = len(lower)
    if n == 0:
        return 0.0
    if n <= window:                              # a short text is one single window
        return len(set(lower)) / n
    counts = Counter(lower[:window])             # word counts inside the current window
    total = len(counts)                          # different words, summed over all windows
    for start in range(1, n - window + 1):       # slide the window one word to the right
        counts[lower[start - 1]] -= 1            # this word leaves the window
        if counts[lower[start - 1]] == 0:
            del counts[lower[start - 1]]
        counts[lower[start + window - 1]] += 1   # this word enters the window
        total += len(counts)
    n_windows = n - window + 1
    return total / (n_windows * window)


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
