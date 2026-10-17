"""Small text helpers shared by the feature groups."""
from __future__ import annotations

import re

from ..paths import project_config

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)*")


def normalise_quotes(text: str) -> str:
    """Turn curly apostrophes into straight ones."""
    return text.replace("’", "'").replace("‘", "'")


def words(text: str) -> list[str]:
    """Split a text into words."""
    return WORD_RE.findall(normalise_quotes(text))


def per_n() -> int:
    """Words per rate (1,000), from config."""
    return int(project_config()["features"]["per_n_words"])


def per_n_words(count: float, n_words: int) -> float:
    """Count per 1,000 words; 0.0 for an empty text."""
    if n_words == 0:
        return 0.0
    return float(count) / n_words * per_n()


def as_texts(X) -> list[str]:
    """Turn a list, Series or one-column frame into a list of strings (missing = empty)."""
    if hasattr(X, "iloc") and getattr(X, "ndim", 1) == 2:
        X = X.iloc[:, 0]
    return ["" if (t is None or t != t) else str(t) for t in X]
