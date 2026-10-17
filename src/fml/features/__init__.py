"""Feature groups beyond plain words: n-grams, epistemic markers and style.

Every transformer takes only the cleaned article text (a column of strings), so a feature can never
use the website name or the label. Numeric groups are followed by a StandardScaler inside the sklearn
pipeline, so the scaler's means and spreads are learned on the training rows only.
"""
from __future__ import annotations

from functools import partial

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .epistemic import EpistemicMarkers
from .ngrams import build_ngram_vectorizer
from .style import StyleFeatures

NUMERIC_GROUPS = {"epistemic": EpistemicMarkers, "style": StyleFeatures}


# Variants for the negation check only. Kept out of NUMERIC_GROUPS, which the feature summary iterates
# (it would list the epistemic columns twice).
VARIANT_GROUPS = {"epistemic_negation": partial(EpistemicMarkers, negation_aware=True)}


def scaled(group: str) -> Pipeline:
    """A numeric feature group followed by a StandardScaler (fitted on train when the pipeline is)."""
    make = NUMERIC_GROUPS.get(group) or VARIANT_GROUPS[group]
    return Pipeline([("raw", make()), ("scale", StandardScaler())])


__all__ = ["EpistemicMarkers", "StyleFeatures", "build_ngram_vectorizer", "scaled", "NUMERIC_GROUPS"]
