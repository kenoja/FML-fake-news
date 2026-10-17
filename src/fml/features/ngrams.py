"""N-grams: TF-IDF over runs of 1 to 3 words, so phrases like "inside job" or "no evidence of" count.

Why: Wang & Manning (2012) found word and bigram features strong for text classification.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

from ..paths import project_config


def build_ngram_vectorizer() -> TfidfVectorizer:
    """Unfitted n-gram TF-IDF vectoriser (settings in config); fit it on training articles only."""
    cfg = project_config()["features"]
    return TfidfVectorizer(ngram_range=tuple(cfg["ngram_range"]), min_df=cfg["ngram_min_df"],
                           max_features=cfg["ngram_max_features"], sublinear_tf=True, lowercase=True)
