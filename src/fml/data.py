"""Load the 9/11 subset of LOCO and apply the documented off-topic filter."""
from __future__ import annotations

import gzip
import json
import re

import pandas as pd

from .paths import data_dir, project_config


def load_corpus() -> pd.DataFrame:
    """All 3,146 documents of the 9/11 seed, one row per document, with a boolean `label` column.

    `label` is True for documents from conspiracy-oriented websites. It describes the WEBSITE,
    not what the article itself argues — keep that in mind in every interpretation.
    """
    cfg = project_config()["data"]
    path = data_dir() / cfg["subset_file"]
    with gzip.open(path, "rt", encoding="utf-8") as f:
        df = pd.DataFrame(json.loads(line) for line in f)
    df["label"] = df[cfg["label_field"]] == cfg["positive_label"]
    return df


def on_topic_mask(df: pd.DataFrame) -> pd.Series:
    """True where title or text mentions 9/11 according to the configured pattern."""
    pattern = re.compile(project_config()["data"]["topic_filter_pattern"], re.IGNORECASE)
    text = df["title"].fillna("") + " " + df["txt"].fillna("")
    return text.map(lambda t: bool(pattern.search(t)))


def summary(df: pd.DataFrame) -> dict:
    """Counts used in the data section and as a smoke test that the data repo is reachable."""
    on_topic = on_topic_mask(df)
    return {
        "documents": int(len(df)),
        "conspiracy": int(df["label"].sum()),
        "mainstream": int((~df["label"]).sum()),
        "websites_conspiracy": int(df.loc[df["label"], "website"].nunique()),
        "websites_mainstream": int(df.loc[~df["label"], "website"].nunique()),
        "on_topic_documents": int(on_topic.sum()),
        "on_topic_conspiracy": int((on_topic & df["label"]).sum()),
        "on_topic_mainstream": int((on_topic & ~df["label"]).sum()),
    }
