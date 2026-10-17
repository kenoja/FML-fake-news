"""Group 1: TF-IDF over word 1-, 2- and 3-grams.

TF-IDF weights a word by how often it occurs in a document (term frequency) and down-weights words
that occur in many documents (inverse document frequency). N-grams of two or three words capture
phrases such as "inside job" or "no evidence of" that single words cannot.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

from ..paths import project_config


def build_ngram_vectorizer() -> TfidfVectorizer:
    """Unfitted vectoriser with settings from `config/project.yaml: features`.

    It must be fitted inside the sklearn pipeline on training rows only: the vocabulary and the IDF
    weights are learned quantities, and fitting them on validation or test text would leak those
    documents' words into training.
    """
    cfg = project_config()["features"]
    return TfidfVectorizer(ngram_range=tuple(cfg["ngram_range"]), min_df=cfg["ngram_min_df"],
                           max_features=cfg["ngram_max_features"], sublinear_tf=True, lowercase=True)
