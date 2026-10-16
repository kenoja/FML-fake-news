"""Feature groups beyond plain words: n-grams, epistemic markers and style.

Every transformer takes only the cleaned article text (a column of strings), so a feature can never
use the website name or the label. Numeric groups are followed by a StandardScaler inside the sklearn
pipeline, so the scaler's means and spreads are learned on the training rows only.
"""
from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .epistemic import EpistemicMarkers
from .ngrams import build_ngram_vectorizer
from .style import StyleFeatures

NUMERIC_GROUPS = {"epistemic": EpistemicMarkers, "style": StyleFeatures}


def scaled(group: str) -> Pipeline:
    """A numeric feature group followed by a StandardScaler (fitted on train when the pipeline is)."""
    return Pipeline([("raw", NUMERIC_GROUPS[group]()), ("scale", StandardScaler())])


__all__ = ["EpistemicMarkers", "StyleFeatures", "build_ngram_vectorizer", "scaled", "NUMERIC_GROUPS"]
