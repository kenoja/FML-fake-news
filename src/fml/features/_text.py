"""Tiny text helpers shared by the numeric feature groups."""
from __future__ import annotations

import re

from ..paths import project_config

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)*")


def normalise_quotes(text: str) -> str:
    """Curly apostrophes to straight ones, so a curly and a straight "don't" count as the same word."""
    return text.replace("’", "'").replace("‘", "'")


def words(text: str) -> list[str]:
    return WORD_RE.findall(normalise_quotes(text))


def per_n() -> int:
    """Normalisation length (words) from config, e.g. 1000: counts become 'per 1,000 words'."""
    return int(project_config()["features"]["per_n_words"])


def per_n_words(count: float, n_words: int) -> float:
    """count / n_words * per_n, and 0.0 for an empty text instead of a division by zero."""
    if n_words == 0:
        return 0.0
    return float(count) / n_words * per_n()


def as_texts(X) -> list[str]:
    """Accept a list, Series, 1-D array or one-column frame of strings; missing values count as empty."""
    if hasattr(X, "iloc") and getattr(X, "ndim", 1) == 2:
        X = X.iloc[:, 0]
    return ["" if (t is None or t != t) else str(t) for t in X]
