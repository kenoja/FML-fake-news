"""Group 2: epistemic markers (how certain a text sounds and whether it claims a hidden truth).

The lexicon below is a small, documented word list. Rates are counts per `features.per_n_words`
words, so a long article does not get a higher value just because it is long.

Where the categories come from (published authors only):
- Hedges and boosters follow the hedge/booster distinction used by Islam et al. (2020) for
  separating reliable from unreliable news; our words are a short hand-picked subset, not their list.
- "Hidden truth" and debunking vocabulary follows the finding of Marini & Jezek (2024) that
  conspiracy discourse is marked by evidentiality/certainty markers and by claims of concealed truth,
  while debunking language marks texts that argue *against* such claims.
- Evidence on certainty is mixed (Fong et al. 2021), so these are hypotheses to test, not facts.

Known weakness (Chen 2026): matching is by word list, so negation and scope are ignored.
"no evidence of a cover-up" still counts "cover-up" as hidden-truth vocabulary. N-grams sit next
to this group in the model partly for that reason.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ._text import as_texts, normalise_quotes, per_n_words, words

LEXICON: dict[str, tuple[str, ...]] = {
    # Hedges: the writer signals doubt or distance (Islam et al. 2020).
    "hedges": ("perhaps", "allegedly", "possibly", "reportedly", "apparently", "supposedly", "maybe",
               "might", "could be", "seems to", "appears to", "it is claimed", "so-called"),
    # Boosters: the writer signals certainty (Islam et al. 2020; Marini & Jezek 2024).
    "boosters": ("undeniable", "undeniably", "clearly", "obviously", "undoubtedly", "certainly",
                 "without a doubt", "beyond doubt", "of course", "proof", "proves", "indisputable"),
    # Hidden-truth phrases: claims of concealment (Marini & Jezek 2024).
    "hidden_truth": ("cover-up", "cover up", "coverup", "they don't want you to know", "wake up",
                     "inside job", "mainstream media", "false flag", "the truth about", "hidden truth",
                     "controlled demolition"),
    # Debunking vocabulary: the text argues against a claim (typical of mainstream explainers).
    "debunking": ("debunked", "debunk", "no evidence", "conspiracy theory", "conspiracy theories",
                  "conspiracy theorists", "myth", "myths", "discredited", "baseless", "hoax"),
}


def _compile(terms: tuple[str, ...]) -> re.Pattern:
    """One regex per category; lookarounds stop "myth" from matching inside "mythology"."""
    parts = sorted((re.escape(t) for t in terms), key=len, reverse=True)
    return re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(parts) + r")(?![A-Za-z0-9])")


_PATTERNS = {cat: _compile(terms) for cat, terms in LEXICON.items()}
COLUMNS = [f"epistemic_{cat}" for cat in LEXICON]


def count_markers(text: str) -> dict[str, int]:
    """Raw number of lexicon hits per category (case-insensitive, no negation handling)."""
    low = normalise_quotes(text).lower()
    return {cat: len(p.findall(low)) for cat, p in _PATTERNS.items()}


class EpistemicMarkers(BaseEstimator, TransformerMixin):
    """Text -> 4 columns of lexicon hits per `features.per_n_words` words. Stateless: nothing is learned."""

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        rows = []
        for text in as_texts(X):
            n = len(words(text))
            counts = count_markers(text)
            rows.append([per_n_words(counts[cat], n) for cat in LEXICON])
        return np.asarray(rows, dtype=float).reshape(-1, len(LEXICON))

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(COLUMNS)
