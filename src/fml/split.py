"""The one frozen train/validation/test split. Whole websites go to exactly one split."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .paths import REPO_ROOT, project_config

SPLITS = ("train", "validation", "test")


class SplitExists(RuntimeError):
    """Raised instead of overwriting the frozen split file."""


def split_path():
    """Where the frozen split file lives."""
    return REPO_ROOT / project_config()["split"]["file"]


def assign_websites(sizes: pd.Series, ratios: dict[str, float], seed: int) -> dict[str, str]:
    """Give each website (biggest first) to the split that is furthest below its target size."""
    rng = np.random.default_rng(seed)
    shuffled = sizes.iloc[rng.permutation(len(sizes))]          # shuffle first, so equal sizes are mixed
    biggest_first = shuffled.sort_values(ascending=False, kind="stable")
    total = float(sizes.sum())
    target = {name: ratios[name] * total for name in SPLITS}
    filled = {name: 0.0 for name in SPLITS}
    site_split = {}
    for site, n_docs in biggest_first.items():
        # pick the split that would be least full (as a share of its target) after adding this website
        best, best_share = None, None
        for name in SPLITS:
            share = (filled[name] + n_docs) / target[name]
            if best is None or share < best_share:
                best, best_share = name, share
        site_split[site] = best
        filled[best] += n_docs
    return site_split


def make_split(df: pd.DataFrame, path=None) -> pd.DataFrame:
    """Create the split file once (grouped by website, per class); never overwrite it."""
    cfg = project_config()["split"]
    path = path or split_path()
    if path.exists():
        raise SplitExists(f"{path} already exists and is frozen; refusing to overwrite it.")
    classes_per_site = df.groupby(cfg["group_by"])[cfg["stratify_by"]].nunique()
    if (classes_per_site > 1).any():
        raise ValueError("a website has more than one class; cannot group by website")
    site_split = {}
    for k, (_, one_class) in enumerate(df.groupby(cfg["stratify_by"], sort=True)):
        sizes = one_class[cfg["group_by"]].value_counts().sort_index()
        site_split.update(assign_websites(sizes, cfg["ratios"], cfg["seed"] + k))
    split_of_doc = df[cfg["group_by"]].map(site_split)
    out = pd.DataFrame({"doc_id": df["doc_id"].to_numpy(), "split": split_of_doc.to_numpy()})
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def load_split(path=None) -> pd.DataFrame:
    """Read the frozen split; fail loudly if it was never created."""
    path = path or split_path()
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run make_split first")
    return pd.read_csv(path)


def split_summary(df: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    """Websites and documents per split and class."""
    merged = df[["doc_id", "website", "subcorpus"]].merge(split, on="doc_id", how="left")
    if merged["split"].isna().any():
        raise ValueError("some documents have no split")
    rows = []
    for name in SPLITS:
        in_split = merged[merged["split"] == name]
        for label, part in in_split.groupby("subcorpus"):
            rows.append({"split": name, "class": label, "websites": int(part["website"].nunique()),
                         "documents": int(len(part))})
    table = pd.DataFrame(rows)
    table["share_of_documents"] = table["documents"] / table["documents"].sum()
    return table


def websites_per_split(summary: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Websites per split and class, read from a split summary."""
    return {name: {row["class"]: int(row["websites"]) for _, row in summary[summary["split"] == name].iterrows()}
            for name in SPLITS}


def check_conspiracy_websites(summary: pd.DataFrame) -> list[str]:
    """Warn about splits with too few conspiracy websites."""
    minimum = project_config()["split"]["min_positive_groups_per_split"]
    positive = project_config()["data"]["positive_label"]
    too_few = []
    for name in SPLITS:
        rows = summary[(summary["split"] == name) & (summary["class"] == positive)]
        n_websites = int(rows["websites"].sum())
        if n_websites < minimum:
            too_few.append(f"{name} ({n_websites} conspiracy websites)")
    if too_few:
        warnings.warn("FEW CONSPIRACY WEBSITES: " + ", ".join(too_few) + f" < {minimum}. Results are fragile.",
                      stacklevel=2)
    return too_few
