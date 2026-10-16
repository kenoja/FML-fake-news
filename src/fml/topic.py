"""Diagnostics for the off-topic filter: which pattern term admitted a document, and who is removed.

The pattern itself lives only in config/project.yaml and is applied by `data.on_topic_mask`. Here it is
split into its terms (top-level alternatives) so we can see which term let each document through.
"""
from __future__ import annotations

import re

import pandas as pd

from .paths import project_config

MAX_EXCERPT_WORDS = 50


def pattern_terms() -> list[str]:
    """The alternatives of the configured pattern, e.g. "pentagon", "ground zero", "\\bwtc\\b"."""
    return project_config()["data"]["topic_filter_pattern"].split("|")


def term_label(term: str) -> str:
    """Readable name of a regex term ("\\bwtc\\b" -> "wtc")."""
    return re.sub(r"\\b|\\", "", term)


def term_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean frame, one column per pattern term: does title or text match that term alone?

    The OR over all columns equals `on_topic_mask`; a test keeps the two in step.
    """
    text = df["title"].fillna("") + " " + df["txt"].fillna("")
    out = {}
    for term in pattern_terms():
        rx = re.compile(term, re.IGNORECASE)
        out[term_label(term)] = text.map(lambda t, rx=rx: bool(rx.search(t))).to_numpy()
    return pd.DataFrame(out, index=df.index)


def mention_counts(df: pd.DataFrame, term: str) -> pd.Series:
    """How often `term` occurs in title + text (one mention suggests a passing reference)."""
    rx = re.compile(term, re.IGNORECASE)
    text = df["title"].fillna("") + " " + df["txt"].fillna("")
    return text.map(lambda t: len(rx.findall(t)))


def short_title(title: str, max_words: int = MAX_EXCERPT_WORDS) -> str:
    """Title cut to the public excerpt limit (a title is article text too)."""
    words = (title or "").split()
    return " ".join(words[:max_words]) if words else "(no title)"
