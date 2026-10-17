"""The one frozen train/validation/test split, grouped by website.

Why grouped: the label belongs to the website, and articles from one site share style, layout and
topics. If one site were in both train and test, a model could score well by recognising the site.
Whole websites therefore go to exactly one split. The split is created once and never edited.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .paths import REPO_ROOT, project_config

SPLITS = ("train", "validation", "test")


class SplitExists(RuntimeError):
    """Raised instead of overwriting the frozen split file."""


def split_path():
    return REPO_ROOT / project_config()["split"]["file"]


def assign_websites(sizes: pd.Series, ratios: dict[str, float], seed: int) -> dict[str, str]:
    """Greedy assignment of whole websites to splits for ONE class.

    Biggest websites first (ties broken by a seeded shuffle); each goes to the split whose share of
    its target would be lowest after adding it, so documents end up close to the ratios.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sizes))
    shuffled = sizes.iloc[order]
    ranked = shuffled.sort_values(ascending=False, kind="stable")
    total = float(sizes.sum())
    target = {s: ratios[s] * total for s in SPLITS}
    filled = {s: 0.0 for s in SPLITS}
    out = {}
    for site, n in ranked.items():
        best = min(SPLITS, key=lambda s: (filled[s] + n) / target[s])
        out[site] = best
        filled[best] += n
    return out


def make_split(df: pd.DataFrame, path=None) -> pd.DataFrame:
    """Create split_v1.csv (doc_id, split) for ALL documents, grouped by website, stratified by class.

    Refuses to overwrite an existing file: re-splitting after seeing results would invalidate every
    comparison. Websites are assigned separately inside each class, so the class mix of every split
    stays close to the corpus mix.
    """
    cfg = project_config()["split"]
    path = path or split_path()
    if path.exists():
        raise SplitExists(f"{path} already exists and is frozen; refusing to overwrite it.")
    if (df.groupby(cfg["group_by"])[cfg["stratify_by"]].nunique() > 1).any():
        raise ValueError("a website has more than one class; cannot group by website")
    site_split: dict[str, str] = {}
    for k, (_, part) in enumerate(df.groupby(cfg["stratify_by"], sort=True)):
        sizes = part[cfg["group_by"]].value_counts().sort_index()
        site_split.update(assign_websites(sizes, cfg["ratios"], cfg["seed"] + k))
    out = pd.DataFrame({"doc_id": df["doc_id"].to_numpy(), "split": df[cfg["group_by"]].map(site_split).to_numpy()})
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def load_split(path=None) -> pd.DataFrame:
    """Read the frozen split (doc_id, split). Fails loudly if it was never created."""
    path = path or split_path()
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run make_split first")
    return pd.read_csv(path)


def split_summary(df: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    """Websites and documents per split and class (one row per split x class)."""
    merged = df[["doc_id", "website", "subcorpus"]].merge(split, on="doc_id", how="left")
    if merged["split"].isna().any():
        raise ValueError("some documents have no split")
    rows = []
    for s in SPLITS:
        for c, g in merged[merged["split"] == s].groupby("subcorpus"):
            rows.append({"split": s, "class": c, "websites": int(g["website"].nunique()), "documents": int(len(g))})
    t = pd.DataFrame(rows)
    t["share_of_documents"] = t["documents"] / t["documents"].sum()
    return t


def check_conspiracy_websites(summary: pd.DataFrame) -> list[str]:
    """Splits with fewer conspiracy websites than the configured minimum (loud warning if any)."""
    minimum = project_config()["split"]["min_positive_groups_per_split"]
    pos = project_config()["data"]["positive_label"]
    low = []
    for s in SPLITS:
        n = int(summary[(summary["split"] == s) & (summary["class"] == pos)]["websites"].sum())
        if n < minimum:
            low.append(f"{s} ({n} conspiracy websites)")
    if low:
        warnings.warn("FEW CONSPIRACY WEBSITES: " + ", ".join(low) + f" < {minimum}. Results are fragile; "
                      "see results/tables/per_website_counts.csv", stacklevel=2)
    return low
