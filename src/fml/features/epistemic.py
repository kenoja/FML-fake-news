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
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ._text import WORD_RE, as_texts, normalise_quotes, per_n_words, words

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


NEGATION_WINDOW = 3
NEGATORS = frozenset({"not", "no", "never"})


def _is_negator(token: str) -> bool:
    return token in NEGATORS or token.endswith("n't")


def count_markers_negation(text: str, lexicon: dict[str, tuple[str, ...]] | None = None
                           ) -> dict[str, dict[str, int]]:
    """Lexicon hits per category, split into plain hits and negated hits.

    A hit is *negated* when not/no/never or a word ending in n't stands within `NEGATION_WINDOW` (3)
    tokens before its first word: "not proven" and "never clearly" are negated. The window is a crude
    scope rule, chosen for being easy to explain; it ignores punctuation and clause boundaries. Nothing
    is learned: this only counts. `lexicon` defaults to `LEXICON` (a different one is for tests).
    """
    patterns = _PATTERNS if lexicon is None else {c: _compile(t) for c, t in lexicon.items()}
    low = normalise_quotes(text).lower()
    spans = [(m.start(), m.group()) for m in WORD_RE.finditer(low)]
    starts = [s for s, _ in spans]
    out = {}
    for cat, p in patterns.items():
        hits = negated = 0
        for m in p.finditer(low):
            i = int(np.searchsorted(starts, m.start()))      # index of the hit's first token
            if any(_is_negator(spans[j][1]) for j in range(max(0, i - NEGATION_WINDOW), i)):
                negated += 1
            else:
                hits += 1
        out[cat] = {"hits": hits, "negated": negated}
    return out


def lexicon_coverage(df_train: pd.DataFrame) -> pd.DataFrame:
    """Per category and class: lexicon size and the share of TRAIN documents with at least one hit.

    Columns: `list_size` (exact number of terms), `share_plain` (>= 1 plain hit, as the model counts),
    `share_negation_aware` (>= 1 hit that is not negated), `share_any_negated` (>= 1 negated hit) and
    `negated_hit_share` (negated hits / all hits). Needs columns `txt` and `label` (1 = conspiracy);
    pass training rows only so that coverage is described without looking at validation or test.
    """
    counted = [count_markers_negation(t) for t in as_texts(df_train["txt"])]
    labels = df_train["label"].to_numpy()
    rows = []
    for cat, terms in LEXICON.items():
        hits = np.array([c[cat]["hits"] for c in counted])
        neg = np.array([c[cat]["negated"] for c in counted])
        for name, mask in (("conspiracy", labels == 1), ("mainstream", labels == 0)):
            n = int(mask.sum())
            total = int(hits[mask].sum() + neg[mask].sum())
            rows.append({"category": cat, "list_size": len(terms), "class": name, "train_documents": n,
                         "share_plain": float(((hits + neg)[mask] > 0).mean()) if n else 0.0,
                         "share_negation_aware": float((hits[mask] > 0).mean()) if n else 0.0,
                         "share_any_negated": float((neg[mask] > 0).mean()) if n else 0.0,
                         "negated_hit_share": float(neg[mask].sum() / total) if total else 0.0})
    return pd.DataFrame(rows)


class EpistemicMarkers(BaseEstimator, TransformerMixin):
    """Text -> 4 columns of lexicon hits per `features.per_n_words` words. Stateless: nothing is learned.

    With `negation_aware=True` negated hits (see `count_markers_negation`) are left out of the counts.
    """

    def __init__(self, negation_aware: bool = False):
        self.negation_aware = negation_aware

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        rows = []
        for text in as_texts(X):
            n = len(words(text))
            if self.negation_aware:
                counts = {c: v["hits"] for c, v in count_markers_negation(text).items()}
            else:
                counts = count_markers(text)
            rows.append([per_n_words(counts[cat], n) for cat in LEXICON])
        return np.asarray(rows, dtype=float).reshape(-1, len(LEXICON))

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(COLUMNS)
