"""Cleaning: duplicates first, then everything that would let a model recognise the website.

Cleaned texts exist only in memory (never written to the public repo); only counts are logged to
results/tables/cleaning_log.csv.
"""
from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache

import numpy as np
import pandas as pd

from .paths import data_dir, project_config

NEAR_DUPLICATE_THRESHOLD = 0.9
BOILERPLATE_MIN_DOCS = 3          # a line repeated in this many documents of one website is boilerplate
MIN_STEM_LENGTH = 6               # shorter site-name stems ("time", "cnn") are too often ordinary words

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w.%+-]+@[\w-]+(?:\.[\w-]+)+")


def normalise(text: str) -> str:
    """Lower-case, letters/digits only, single spaces: two texts differing only in layout match."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def text_hash(text: str) -> str:
    return hashlib.sha1(normalise(text).encode("utf-8")).hexdigest()


def _count(df: pd.DataFrame) -> dict:
    return {"documents": int(len(df)), "words": int(df["txt"].str.split().map(len).sum())}


def _log_row(step: str, before: dict, after: dict, note: str = "") -> dict:
    return {"step": step, "documents_before": before["documents"], "documents_after": after["documents"],
            "words_before": before["words"], "words_after": after["words"], "note": note}


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the first of each group of documents with identical normalised text."""
    return df.loc[~df["txt"].map(text_hash).duplicated(keep="first")]


def find_near_duplicates(df: pd.DataFrame, threshold: float = NEAR_DUPLICATE_THRESHOLD) -> pd.DataFrame:
    """Pairs (doc_id_a, doc_id_b, cosine, same_website) of documents with TF-IDF cosine >= threshold.

    The TF-IDF here is fitted on the whole corpus. That is not leakage: it is used only to ask
    "are these two texts copies of each other?", never as a feature of any model and never with
    labels. No learned quantity reaches a classifier, so test information cannot influence training.
    Models fit their own TF-IDF on the training split only.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    X = TfidfVectorizer(sublinear_tf=True, min_df=2).fit_transform(df["txt"])  # rows are L2-normalised
    ids = df["doc_id"].to_numpy()
    sites = df["website"].to_numpy()
    rows = []
    for start in range(0, X.shape[0], 500):
        sims = (X[start:start + 500] @ X.T).toarray()
        for i in range(sims.shape[0]):
            sims[i, : start + i + 1] = 0.0            # upper triangle only: each pair once
        for i, j in zip(*np.nonzero(sims >= threshold)):
            rows.append((ids[start + i], ids[j], float(sims[i, j]), bool(sites[start + i] == sites[j])))
    return pd.DataFrame(rows, columns=["doc_id_a", "doc_id_b", "cosine", "same_website"])


def website_names(df: pd.DataFrame, metadata: list[dict] | None = None) -> list[str]:
    """Domains from the `website` field and the metadata file, plus their long-enough name stems.

    Longest first, so "news.example.com" is removed before its stem would be.
    """
    if metadata is None:
        path = data_dir() / project_config()["data"]["website_metadata_file"]
        metadata = json.loads(path.read_text(encoding="utf-8"))
    domains = set(df["website"].dropna()) | {m["website"] for m in metadata if m.get("website")}
    names = set()
    for d in domains:
        d = d.lower().removeprefix("www.")
        names.add(d)
        parts = d.split(".")
        stem = parts[-2] if len(parts) >= 2 else parts[0]
        if len(stem) >= MIN_STEM_LENGTH:
            names.add(stem)
    return sorted(names, key=len, reverse=True)


def strip_urls_and_emails(text: str) -> str:
    return EMAIL_RE.sub(" ", URL_RE.sub(" ", text))


@lru_cache(maxsize=4)
def _name_pattern(names: tuple[str, ...]) -> re.Pattern:
    """Compiled once per name list: recompiling it for every document dominated the cleaning time."""
    return re.compile(r"(?<![\w])(?:" + "|".join(re.escape(n) for n in names) + r")(?![\w])", re.IGNORECASE)


def strip_website_names(text: str, names: list[str]) -> str:
    if not names:
        return text
    return _name_pattern(tuple(names)).sub(" ", text)


def boilerplate_lines(df: pd.DataFrame, min_docs: int = BOILERPLATE_MIN_DOCS) -> dict[str, set[str]]:
    """Per website, the normalised lines that occur in >= min_docs different documents of it.

    Menus, cookie notices, "share this" and author footers repeat on one site; a model could use
    them to recognise the site instead of reading the article.
    """
    out: dict[str, set[str]] = {}
    for site, group in df.groupby("website"):
        counts: dict[str, int] = {}
        for txt in group["txt"]:
            for line in {normalise(l) for l in txt.splitlines()}:
                if line:
                    counts[line] = counts.get(line, 0) + 1
        out[site] = {l for l, c in counts.items() if c >= min_docs}
    return out


def strip_boilerplate(df: pd.DataFrame, lines: dict[str, set[str]]) -> pd.Series:
    def clean(row) -> str:
        bad = lines.get(row["website"], set())
        return "\n".join(l for l in row["txt"].splitlines() if normalise(l) not in bad)
    return df.apply(clean, axis=1)


def clean_corpus(df: pd.DataFrame, metadata: list[dict] | None = None,
                 threshold: float = NEAR_DUPLICATE_THRESHOLD,
                 drop_exact: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (cleaned frame, step log, near-duplicate pairs). The input frame is not changed.

    Exact duplicates are dropped. Near-duplicates are only detected and logged: dropping them
    would change the 2,608 documents the team fixed for the off-topic filter, so that needs a team
    decision (docs/decisions.md). Cleaned text is in column `txt`; the original is not kept.
    """
    log = []
    cur = df.copy()
    log.append(_log_row("start", _count(cur), _count(cur)))

    before = _count(cur)
    if drop_exact:
        cur = drop_exact_duplicates(cur)
    log.append(_log_row("exact duplicates (normalised-text hash)", before, _count(cur)))

    pairs = find_near_duplicates(cur, threshold)
    drop_b = pairs["doc_id_b"].nunique() if len(pairs) else 0
    log.append(_log_row(f"near-duplicates (TF-IDF cosine >= {threshold}): detected, not removed",
                        _count(cur), _count(cur),
                        f"{len(pairs)} pairs ({int(pairs['same_website'].sum()) if len(pairs) else 0} within one website); "
                        f"{drop_b} documents would go"))

    names = website_names(cur, metadata)
    for step, fn in (
        ("URLs and e-mail-like strings removed", strip_urls_and_emails),
        ("website names removed", lambda t: strip_website_names(t, names)),
    ):
        before = _count(cur)
        cur["txt"] = cur["txt"].map(fn)
        log.append(_log_row(step, before, _count(cur)))

    before = _count(cur)
    cur["txt"] = strip_boilerplate(cur, boilerplate_lines(cur))
    log.append(_log_row(f"lines repeated in >= {BOILERPLATE_MIN_DOCS} documents of one website removed",
                        before, _count(cur)))
    return cur, pd.DataFrame(log), pairs
