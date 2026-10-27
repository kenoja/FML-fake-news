"""Data stage: length distribution, the frozen split and their tables, figures, metrics."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data import load_corpus, summary  # noqa: E402
from .paths import project_config  # noqa: E402
from .results import CONSPIRACY_COLOUR, MAINSTREAM_COLOUR, savefig, write_metrics, write_table  # noqa: E402
from .split import SPLITS, check_conspiracy_websites, load_split, make_split, split_path, split_summary  # noqa: E402

QUANTILES = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
# Rule of thumb, stated up front: under ~50 words a text is a headline or a snippet, with too few
# words for a bag-of-words model or a reader to judge. Our shortest article is compared with it.
MIN_USEFUL_WORDS = 50
SHORT_BUCKETS = (150, 250)
COLOURS = {"conspiracy": CONSPIRACY_COLOUR, "mainstream": MAINSTREAM_COLOUR}


def length_table(df: pd.DataFrame) -> pd.DataFrame:
    """Words per document (whitespace-separated, raw text): quantiles per class and overall."""
    words = df["txt"].str.split().map(len)
    rows = []
    for name, w in (("all", words), *((c, words[df["subcorpus"] == c]) for c in ("conspiracy", "mainstream"))):
        rows.append({"class": name, "documents": int(len(w)), **{f"q{int(q * 100)}": float(w.quantile(q)) for q in QUANTILES},
                     "mean": float(w.mean())})
    return pd.DataFrame(rows)


def length_statement(df: pd.DataFrame) -> dict:
    words = df["txt"].str.split().map(len)
    shortest = int(words.min())
    out = {
        "shortest_words": shortest,
        "median_words": float(words.median()),
        "minimum_length_needed": bool(shortest < MIN_USEFUL_WORDS),
        "threshold_words": MIN_USEFUL_WORDS,
        **{f"documents_under_{b}_words": int((words < b).sum()) for b in SHORT_BUCKETS},
    }
    out["statement"] = (
        f"The shortest article has {shortest} words, above the {MIN_USEFUL_WORDS}-word level below which a text "
        f"is a snippet rather than an article, so no minimum length is applied. {out[f'documents_under_{SHORT_BUCKETS[1]}_words']} "
        f"of {len(df)} documents are under {SHORT_BUCKETS[1]} words."
        if not out["minimum_length_needed"] else
        f"The shortest article has {shortest} words, below {MIN_USEFUL_WORDS}: a minimum length is needed."
    )
    return out


def plot_length(df: pd.DataFrame):
    words = df["txt"].str.split().map(len)
    bins = np.logspace(np.log10(words.min()), np.log10(words.max()), 40)
    fig, ax = plt.subplots(figsize=(7, 4))
    for c in ("mainstream", "conspiracy"):
        ax.hist(words[df["subcorpus"] == c], bins=bins, alpha=0.7, color=COLOURS[c], label=c)
    ax.set_xscale("log")
    ax.set_xlabel("Words per article (log scale)")
    ax.set_ylabel("Documents")
    ax.set_title("Length distribution by class")
    ax.legend()
    return fig


def plot_split_sizes(summ: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.38
    x = np.arange(len(SPLITS))
    for k, c in enumerate(("mainstream", "conspiracy")):
        vals = [int(summ[(summ["split"] == s) & (summ["class"] == c)]["documents"].sum()) for s in SPLITS]
        webs = [int(summ[(summ["split"] == s) & (summ["class"] == c)]["websites"].sum()) for s in SPLITS]
        bars = ax.bar(x + (k - 0.5) * width, vals, width, color=COLOURS[c], label=c)
        for b, v, w in zip(bars, vals, webs):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v}\n({w} sites)", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, SPLITS)
    ax.set_ylabel("Documents")
    ax.set_title("Documents (and websites) per split and class")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.legend()
    return fig


def per_website_counts(df: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    m = df[["doc_id", "website", "subcorpus"]].merge(split, on="doc_id")
    return (m.groupby(["split", "subcorpus", "website"]).size().rename("documents").reset_index()
            .sort_values(["split", "subcorpus", "documents"], ascending=[True, True, False]))


def site_concentration(df: pd.DataFrame, split: pd.DataFrame, top: int = 3) -> tuple[pd.DataFrame, dict]:
    """Conspiracy articles of the largest sites: how many in train, and their share of all conspiracy articles."""
    con = df[df["subcorpus"] == "conspiracy"][["doc_id", "website"]].merge(split, on="doc_id")
    counts = con["website"].value_counts().sort_index().sort_values(ascending=False, kind="stable")   # ties: by name
    rows = []
    for rank, site in enumerate(counts.index[:top], start=1):
        mine = con[con["website"] == site]
        rows.append({"Website": f"Website {rank}", "Articles": int(len(mine)),
                     "Share of conspiracy articles": float(len(mine) / len(con)),
                     "Articles in train": int((mine["split"] == "train").sum())})
    table = pd.DataFrame(rows)
    metrics = {"top3_sites_share": float(table["Articles"].sum() / len(con)),
               "top3_train_articles": int(table["Articles in train"].sum())}
    return table, metrics


def run_data_stage() -> dict:
    """Describe lengths, create the split if it does not exist (else read it), write results."""
    df = load_corpus()
    ltab = length_table(df)
    write_table("length_quantiles", ltab)
    savefig(plot_length(df), "length_distribution")
    lstat = length_statement(df)

    if not split_path().exists():
        make_split(df)
    split = load_split()
    if set(split["doc_id"]) != set(df["doc_id"]):
        raise ValueError("split file and corpus have different doc_ids")
    summ = split_summary(df, split)

    low = check_conspiracy_websites(summ)
    if low:
        print("WARNING: few conspiracy websites in " + ", ".join(low))
        write_table("per_website_counts", per_website_counts(df, split))
    site_split = df[["doc_id", "website"]].merge(split, on="doc_id").groupby("website")["split"].nunique()

    write_metrics("data", {
        **summary(df),
        "length": lstat,
    })
    write_metrics("split", {
        "file": project_config()["split"]["file"],
        "seed": project_config()["split"]["seed"],
        "documents": int(len(split)),
        "websites_in_more_than_one_split": int((site_split > 1).sum()),
        "per_split": {s: {c: {"websites": int(r["websites"]), "documents": int(r["documents"])}
                          for _, r in summ[summ["split"] == s].iterrows() for c in [r["class"]]} for s in SPLITS},
        "few_conspiracy_websites_flag": bool(low),
        "splits_with_few_conspiracy_websites": low,
        "min_positive_groups_per_split": project_config()["split"]["min_positive_groups_per_split"],
    })
    return {"split_summary": summ, "length": lstat}
